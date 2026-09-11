"""Nearest-neighbour inspection of the joint cross-survey embedding space.

Embeds ``--views-per-star`` light-curve *segments* per star per instrument
(same seeded, un-augmented 500-1500-day eval windows as ``downstream.py``),
pools every instrument's segments into ONE joint space, and for each segment
retrieves its ``--k`` nearest neighbours by cosine similarity — always
excluding segments of the same (star, instrument), so overlapping windows of
the same light curve can't produce trivial matches and every hit is either a
different star or the same star seen through a *different* survey.

Outputs (next to the checkpoint, and into the run's wandb if --wandb-id):

- ``knn_report.txt``  — human-readable: random query segments with their k
  neighbours (gaia id, instrument, class, period, cosine, same-star flag).
- ``knn_metrics.json`` — full-dataset metrics:
    * same-star cross-instrument hit@1/@k (the star-level alignment number)
    * class purity@k and k-NN majority accuracy (vs the class marginal)
    * period coherence |dlog10 P| between query and neighbours
    * neighbour-instrument mixing matrix (is the space actually joint?)
- ``knn_topk.npz``     — segment table + top-k indices/sims for reuse.

Usage::

    python analysis/knn_inspect.py --ckpt runs/Cross-Survey-LC/cs3079146-lejepa/checkpoints/epoch=699-step=59500.ckpt \
        --wandb Cross-Survey-LC --wandb-id cs3079146-lejepa
"""

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # benchmarks/lightcurves (main script, downstream.py, runs/)


def load_modules():
    spec = importlib.util.spec_from_file_location("downstream",
                                                  ROOT / "downstream.py")
    ds_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds_mod)
    return ds_mod, ds_mod.ladder


class SegDataset(torch.utils.data.Dataset):
    """One item = one seeded eval segment; k encodes (object, replica)."""

    def __init__(self, records, indices, bands, cfg, n_views, seed):
        self.records, self.indices = records, indices
        self.bands, self.cfg = tuple(bands), cfg
        self.n_views, self.seed = n_views, seed

    def __len__(self):
        return len(self.indices) * self.n_views

    def __getitem__(self, k):
        import ladder  # registered by downstream.py

        obj, rep = divmod(k, self.n_views)
        rec = self.records[self.indices[obj]]
        rng = np.random.default_rng((self.seed, self.indices[obj], rep))
        return k, ladder.make_eval_view(rec, self.cfg, rng, bands=self.bands,
                                        augment=False)


def build_records(ladder, args):
    """load_records + kept gaia ids (mirrors the ladder's loop 1:1)."""
    from datasets import load_dataset

    tok = ladder.resolve_hf_token(args.hf_token)
    ds = load_dataset(ladder.DATASET, split="train", token=tok)
    if args.max_objects > 0:
        ds = ds.select(range(min(args.max_objects, len(ds))))
    raw = list(ds)
    classes = sorted({ex["class_str"] for ex in raw})
    label_to_idx = {c: i for i, c in enumerate(classes)}
    records, gaia = [], []
    for ex in raw:
        rec = ladder._to_record(ex, label_to_idx)
        if ladder._has_two_instruments(rec, args.min_obs):
            records.append(rec)
            gaia.append(int(ex["gaia_dr3_source_id"]))
    g = np.float32(np.median([r["norm"][3] for r in records]))
    for r in records:
        c0, s0, centers, _ = r["norm"]
        r["norm"] = (c0, s0, centers, g)
    print(f"{len(records)} records; band-global scale {g:.5f} mag")
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    return records, np.array(gaia), idx_to_label


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--views-per-star", type=int, default=4)
    ap.add_argument("--n-report", type=int, default=4,
                    help="random report queries per instrument")
    ap.add_argument("--min-obs", type=int, default=8)
    ap.add_argument("--max-objects", type=int, default=0,
                    help="first N stars only (smoke tests)")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--wandb-id", default=None)
    args = ap.parse_args()

    ds_mod, ladder = load_modules()
    ladder.configure_instruments("hibb/tess-ztf-atlas-asassn-isect",
                                 holdout="ATLAS", exclude=("TESS",))
    cfg = ladder.ViewConfig(window_days=500.0, window_days_max=1500.0,
                            min_window_obs=200, over_budget="tail",
                            eval_tokens=512, min_tokens=args.min_obs,
                            norm="band-global")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Reuse downstream's checkpoint loader (needs width/depth/... on args).
    args.width, args.depth = 256, 4
    args.projector, args.n_slices = "identity", 128
    backbone = ds_mod.load_backbone(args, device)

    records, gaia, idx_to_label = build_records(ladder, args)
    labels = np.array([r["label"] for r in records])
    logp = np.array([np.log10(r["period"]) if r["period"] else np.nan
                     for r in records])
    _, val_idx = ladder.stratified_split(records, 0.1, 0)
    is_val_star = np.zeros(len(records), bool)
    is_val_star[val_idx] = True

    # ---- embed segments, all instruments into one table ----
    nv = args.views_per_star
    emb, star, inst = [], [], []
    for ii, name in enumerate(ladder.INSTRUMENTS):
        idxs = [i for i, r in enumerate(records)
                if ladder._has_bands(r, ladder.INST_BANDS[name])]
        ds = SegDataset(records, idxs, ladder.INST_BANDS[name], cfg, nv,
                        args.seed)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=lambda b: (torch.tensor([x[0] for x in b]),
                                  ladder._tokenize_batch([x[1] for x in b])))
        out = torch.zeros(len(ds), backbone.embed_dim)
        for ks, (v, p, m) in dl:
            out[ks] = backbone(v.to(device), p.to(device), m.to(device)).float().cpu()
        emb.append(out)
        star.append(np.repeat(np.array(idxs), nv))
        inst.append(np.full(len(ds), ii))
        print(f"{name}: {len(idxs)} stars -> {len(ds)} segments")
    emb = torch.nn.functional.normalize(torch.cat(emb), dim=1).to(device)
    star = np.concatenate(star)
    inst = np.concatenate(inst)
    M = len(star)

    # ---- chunked cosine top-k, same (star, instrument) masked out ----
    star_t = torch.as_tensor(star, device=device)
    inst_t = torch.as_tensor(inst, device=device)
    nn_idx = torch.zeros(M, args.k, dtype=torch.long)
    nn_sim = torch.zeros(M, args.k)
    for lo in range(0, M, 4096):
        hi = min(lo + 4096, M)
        sims = emb[lo:hi] @ emb.T
        same = (star_t[None, :] == star_t[lo:hi, None]) & \
               (inst_t[None, :] == inst_t[lo:hi, None])
        sims.masked_fill_(same, -2.0)
        s, i = sims.topk(args.k, dim=1)
        nn_idx[lo:hi], nn_sim[lo:hi] = i.cpu(), s.cpu()
    nn_idx, nn_sim = nn_idx.numpy(), nn_sim.numpy()

    # ---- metrics ----
    q_star, n_star = star, star[nn_idx]            # [M], [M, k]
    same_star = n_star == q_star[:, None]           # cross-inst by construction
    q_cls, n_cls = labels[star], labels[star[nn_idx]]
    same_cls = n_cls == q_cls[:, None]
    maj = np.array([np.bincount(row).argmax() for row in n_cls])
    dlogp = np.abs(logp[star][:, None] - logp[star[nn_idx]])
    fin = np.isfinite(dlogp)
    val_q = is_val_star[star]

    insts = list(ladder.INSTRUMENTS)
    mix = np.zeros((len(insts), len(insts)))
    for a in range(len(insts)):
        m = inst == a
        for b in range(len(insts)):
            mix[a, b] = (inst[nn_idx[m]] == b).mean()

    def block(mask, tag):
        sup = int(mask.sum())
        return {
            f"{tag}/n_queries": sup,
            f"{tag}/same_star_hit@1": float(same_star[mask, 0].mean()),
            f"{tag}/same_star_hit@k": float(same_star[mask].any(1).mean()),
            f"{tag}/class_purity@k": float(same_cls[mask].mean()),
            f"{tag}/knn_majority_top1": float((maj[mask] == q_cls[mask]).mean()),
            f"{tag}/med_abs_dlogP": float(np.median(dlogp[mask][fin[mask]])),
            f"{tag}/frac_dlogP<0.05": float((dlogp[mask][fin[mask]] < 0.05).mean()),
        }

    metrics = {"k": args.k, "views_per_star": nv, "n_segments": M,
               "class_marginal_top1": float(
                   np.mean(labels[star] == np.bincount(labels[star]).argmax())),
               "chance_same_star_hit@k": float(
                   args.k * (len(insts) - 1) * nv / M),
               "neighbor_instrument_mix":
                   {insts[a]: {insts[b]: round(float(mix[a, b]), 4)
                               for b in range(len(insts))}
                    for a in range(len(insts))}}
    metrics.update(block(np.ones(M, bool), "all"))
    metrics.update(block(val_q, "val_stars"))
    for a, nm in enumerate(insts):
        metrics.update(block(inst == a, f"query_{nm}"))
    percls = {}
    for c in np.unique(q_cls):
        m = q_cls == c
        if m.sum() >= 50:
            percls[idx_to_label[int(c)]] = {
                "n": int(m.sum()),
                "purity@k": round(float(same_cls[m].mean()), 3),
                "same_star_hit@k": round(float(same_star[m].any(1).mean()), 3),
            }
    metrics["per_class"] = percls

    # ---- human-readable report ----
    rng = np.random.default_rng(args.seed)
    lines = [f"joint-space {args.k}-NN report (segments; same star+instrument "
             "excluded from candidates)\n"]
    for a, nm in enumerate(insts):
        pool = np.where((inst == a) & val_q)[0]
        for q in rng.choice(pool, min(args.n_report, len(pool)), replace=False):
            s = star[q]
            lines.append(
                f"query  {nm:>7} gaia {gaia[s]}  {idx_to_label[labels[s]]:<12} "
                f"P={records[s]['period'] or float('nan'):.4g} d")
            for r, (j, sim) in enumerate(zip(nn_idx[q], nn_sim[q])):
                t = star[j]
                flag = "  <-- SAME STAR" if t == s else ""
                lines.append(
                    f"  nn{r + 1} {insts[inst[j]]:>7} gaia {gaia[t]}  "
                    f"{idx_to_label[labels[t]]:<12} "
                    f"P={records[t]['period'] or float('nan'):.4g} d  "
                    f"cos={sim:.3f}{flag}")
            lines.append("")
    report = "\n".join(lines)
    print(report)
    print(json.dumps({k: v for k, v in metrics.items()
                      if not isinstance(v, dict)}, indent=2))

    out = Path(args.ckpt).parent
    (out / "knn_report.txt").write_text(report)
    with open(out / "knn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(out / "knn_topk.npz", nn_idx=nn_idx, nn_sim=nn_sim,
                        star=star, inst=inst, gaia=gaia, labels=labels,
                        logp=logp, is_val_star=is_val_star)
    print(f"wrote {out}/knn_report.txt, knn_metrics.json, knn_topk.npz")

    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb, id=args.wandb_id, dir=str(ROOT / "runs"),
                         resume="allow" if args.wandb_id else None,
                         name=None if args.wandb_id else "knn-inspect")
        run.summary.update({f"knn/{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        cols = ["query_inst", "query_gaia", "query_class", "query_P",
                "rank", "nn_inst", "nn_gaia", "nn_class", "nn_P", "cos",
                "same_star"]
        rows = []
        for a, nm in enumerate(insts):
            pool = np.where((inst == a) & val_q)[0]
            for q in np.random.default_rng(args.seed).choice(
                    pool, min(args.n_report, len(pool)), replace=False):
                s = star[q]
                for r, (j, sim) in enumerate(zip(nn_idx[q], nn_sim[q])):
                    t = star[j]
                    rows.append([nm, str(gaia[s]), idx_to_label[labels[s]],
                                 records[s]["period"], r + 1,
                                 insts[inst[j]], str(gaia[t]),
                                 idx_to_label[labels[t]],
                                 records[t]["period"], float(sim), t == s])
        run.log({"knn/report": wandb.Table(columns=cols, data=rows),
                 "knn/instrument_mix": wandb.Table(
                     columns=["query \\ nn"] + insts,
                     data=[[insts[a]] + [float(mix[a, b])
                                         for b in range(len(insts))]
                           for a in range(len(insts))])})
        run.finish()


if __name__ == "__main__":
    main()
