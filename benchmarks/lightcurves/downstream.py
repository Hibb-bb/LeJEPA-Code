"""Offline downstream evaluation of a pretrained LeJEPA light-curve encoder.

Loads a Lightning checkpoint produced by ``pretrain.py``, embeds
every object once per instrument (mean of ``--n-eval-views`` un-augmented
capped-span eval views, i.e. the same 500-1500-day windows the encoder was
pretrained on) and runs two frozen-encoder downstream tasks:

1. **Variable-star classification** — multinomial logistic regression on the
   embeddings, target ``class_str``. The label is a property of the *star*
   (this is an intersection dataset: the same objects appear in every
   survey), so the class taxonomy is identical across surveys by
   construction; only per-survey coverage differs.
2. **Period regression** — ridge regression on ``log10(period)`` (objects
   with a finite catalog period only).

Both tasks are evaluated as a full cross-survey transfer matrix: for each
source instrument A the probe is fit on the embeddings of the *pretraining
train-split* stars seen through A, then evaluated on the *val-split* stars
seen through every instrument B (same star split as pretraining, so the
encoder never saw a test star and the probe never saw a test star through
any survey). ``A == B`` rows are the in-survey reference; ``B == ATLAS``
(the pretraining holdout) is the zero-shot instrument transfer number.

Usage (matches the debug run's geometry)::

    python downstream.py \
        --ckpt runs/Cross-Survey-LC/cs3079146-lejepa/checkpoints/epoch=699-step=59500.ckpt \
        --dataset hibb/tess-ztf-atlas-asassn-isect \
        --exclude-instrument TESS --holdout-instrument ATLAS \
        --width 256 --depth 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# ``pretrain.py`` lives next to this script (sys.path[0]); import it so the
# view pipeline, record loader and model builder stay single-sourced.
import pretrain as ladder  # noqa: E402


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
class EvalViewDataset(torch.utils.data.Dataset):
    """One item = one seeded, un-augmented eval view of one object.

    ``indices`` are positions into ``records`` (only objects that actually
    have the instrument); each object contributes ``n_views`` items whose
    windows differ only through the seeded rng, so the per-object embedding
    (the mean over its views) is deterministic.
    """

    def __init__(self, records, indices, bands, cfg, n_views, seed):
        self.records = records
        self.indices = indices
        self.bands = tuple(bands)
        self.cfg = cfg
        self.n_views = n_views
        self.seed = seed

    def __len__(self):
        return len(self.indices) * self.n_views

    def __getitem__(self, k):
        obj, rep = divmod(k, self.n_views)
        rec = self.records[self.indices[obj]]
        rng = np.random.default_rng((self.seed, self.indices[obj], rep))
        view = ladder.make_eval_view(rec, self.cfg, rng, bands=self.bands,
                                     augment=False)
        return obj, view


def _collate_views(batch):
    objs = torch.tensor([b[0] for b in batch], dtype=torch.long)
    return objs, ladder._tokenize_batch([b[1] for b in batch])


@torch.no_grad()
def embed_instrument(backbone, records, indices, inst, cfg, args, device):
    """[len(indices), width] mean-of-views embedding for one instrument."""
    ds = EvalViewDataset(records, indices, ladder.INST_BANDS[inst], cfg,
                         args.n_eval_views, args.seed)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=_collate_views,
    )
    out = torch.zeros(len(indices), backbone.embed_dim)
    for objs, (v, p, m) in dl:
        z = backbone(v.to(device), p.to(device), m.to(device)).float().cpu()
        out.index_add_(0, objs, z)
    return (out / args.n_eval_views).numpy()


def load_backbone(args, device):
    """Rebuild the pretrained rung and load the checkpoint's encoder."""
    model = ladder.build_model(
        args.width, args.width, lamb=0.0, n_slices=args.n_slices,
        depth=args.depth, mode="lejepa", projector=args.projector,
    )
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items()
          if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # sigreg buffers may differ across versions; the backbone must be exact.
    bad = [k for k in missing + unexpected if k.startswith("backbone.")]
    if bad:
        raise RuntimeError(f"backbone keys did not load: {bad}")
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch')}); "
          f"non-backbone unmatched keys: {missing + unexpected or 'none'}")
    return model.backbone.to(device).eval()


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def fit_eval_classification(emb, results):
    """Logistic regression per source instrument, evaluated on every target."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    for src in ladder.INSTRUMENTS:
        Xtr, ytr = emb[src]["train"], emb[src]["train_label"]
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(
            scaler.transform(Xtr), ytr
        )
        for tgt in ladder.INSTRUMENTS:
            Xte, yte = emb[tgt]["val"], emb[tgt]["val_label"]
            pred = clf.predict(scaler.transform(Xte))
            results[f"cls/{src}_to_{tgt}"] = {
                "n_train": len(ytr), "n_test": len(yte),
                "top1": float(accuracy_score(yte, pred)),
                "balanced": float(balanced_accuracy_score(yte, pred)),
                "macro_f1": float(f1_score(yte, pred, average="macro")),
            }


def fit_eval_period(emb, results):
    """Ridge on log10(period) per source instrument, all targets.

    Returns ``{f"{src}_to_{tgt}": (y_true, y_pred)}`` on the val split for
    the scatter figure.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    preds = {}
    for src in ladder.INSTRUMENTS:
        ok = np.isfinite(emb[src]["train_logp"])
        Xtr, ytr = emb[src]["train"][ok], emb[src]["train_logp"][ok]
        scaler = StandardScaler().fit(Xtr)
        reg = Ridge(alpha=1.0).fit(scaler.transform(Xtr), ytr)
        for tgt in ladder.INSTRUMENTS:
            ok = np.isfinite(emb[tgt]["val_logp"])
            Xte, yte = emb[tgt]["val"][ok], emb[tgt]["val_logp"][ok]
            pred = reg.predict(scaler.transform(Xte))
            err = np.abs(pred - yte)
            preds[f"{src}_to_{tgt}"] = (yte, pred)
            results[f"per/{src}_to_{tgt}"] = {
                "n_train": int(len(ytr)), "n_test": int(len(yte)),
                "r2": float(r2_score(yte, pred)),
                "rmse_dex": float(np.sqrt(np.mean((pred - yte) ** 2))),
                "med_abs_dex": float(np.median(err)),
                "frac_within_0.1dex": float((err < 0.1).mean()),
            }
    return preds


def period_scatter_figure(preds, results):
    """Pred-vs-true log10(P) grid: one panel per (fit-on, tested-on) pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    insts = ladder.INSTRUMENTS
    n = len(insts)
    fig, axes = plt.subplots(n, n, figsize=(3.6 * n, 3.6 * n),
                             sharex=True, sharey=True)
    lo = min(y.min() for y, _ in preds.values())
    hi = max(y.max() for y, _ in preds.values())
    pad = 0.05 * (hi - lo)
    for i, src in enumerate(insts):
        for j, tgt in enumerate(insts):
            ax = axes[i, j]
            y, p = preds[f"{src}_to_{tgt}"]
            r = results[f"per/{src}_to_{tgt}"]
            ax.scatter(y, p, s=4, alpha=0.3, linewidths=0)
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_title(f"fit {src} → test {tgt}\n"
                         f"R²={r['r2']:.2f}  med|Δ|={r['med_abs_dex']:.2f} dex",
                         fontsize=9)
            if i == n - 1:
                ax.set_xlabel("true log10 P [d]")
            if j == 0:
                ax.set_ylabel("pred log10 P [d]")
    for ax in axes.ravel():
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Period regression on frozen embeddings (val split)")
    fig.tight_layout()
    return fig


# StarEmbed (arXiv:2510.06200) benchmarks seven CSPVS classes on ZTF. Map our
# 16-class VSX-style taxonomy onto them for the class-highlight UMAP figure
# (classes with no analogue — Cepheids, DSCT, ELL, PCEB — stay gray).
# "RS CVn" has no exact match; ROT (spotted rotational variables) is the
# closest VSX analogue.
STAREMBED_CLASSES = {
    "EW": ("EW/EB", "EW/EB-OC"),
    "EA": ("EA",),
    "RRab": ("RRAB", "RRab-Blazhko"),
    "RRc": ("RRC", "RRc-Blazhko"),
    "RRd": ("RRD",),
    "RS CVn~ROT": ("ROT",),
    "LPV": ("LPV",),
}


def starembed_umap_figure(emb, idx_to_label, seed=0):
    """Per-instrument UMAP with one panel per StarEmbed class.

    Rows = instruments (each with its own UMAP fit on that instrument's
    train+val embeddings, cosine metric); columns = the seven StarEmbed
    classes. Every panel shows all of the instrument's objects in gray and
    colours only the one class, so per-class structure is readable even for
    the rare classes.

    Args:
        emb: the per-instrument dict built in ``main`` (``train``/``val``
            embeddings and labels).
        idx_to_label: class index -> ``class_str`` name.
        seed: UMAP/PCA random state.

    Returns:
        The matplotlib figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    insts = list(ladder.INSTRUMENTS)
    classes = list(STAREMBED_CLASSES)
    fig, axes = plt.subplots(
        len(insts), len(classes),
        figsize=(3.1 * len(classes), 3.3 * len(insts)), squeeze=False,
    )
    cmap = plt.get_cmap("tab10")
    method = "UMAP"
    for r, inst in enumerate(insts):
        X = np.concatenate([emb[inst]["train"], emb[inst]["val"]])
        y = np.concatenate([emb[inst]["train_label"], emb[inst]["val_label"]])
        names = np.array([idx_to_label[int(c)] for c in y])
        try:
            import umap
            z = umap.UMAP(n_components=2, random_state=seed,
                          metric="cosine").fit_transform(X)
        except ImportError:
            from sklearn.decomposition import PCA
            z = PCA(2, random_state=seed).fit_transform(X)
            method = "PCA"
        for c, (se_name, ours) in enumerate(STAREMBED_CLASSES.items()):
            ax = axes[r, c]
            ax.scatter(*z.T, c="lightgray", s=4, alpha=0.4, linewidths=0)
            m = np.isin(names, ours)
            ax.scatter(*z[m].T, c=[cmap(c)], s=6, alpha=0.8, linewidths=0)
            ax.set_title(f"{inst}: {se_name} (n={int(m.sum())})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{method} per instrument, frozen embeddings "
                 "(gray = all objects, colour = one StarEmbed class)")
    fig.tight_layout()
    return fig


def print_matrix(results, task, metric, title):
    insts = ladder.INSTRUMENTS
    print(f"\n{title} ({metric}; rows = probe fit on, cols = tested on val split)")
    print(f"{'':>8}" + "".join(f"{t:>10}" for t in insts))
    for src in insts:
        row = "".join(
            f"{results[f'{task}/{src}_to_{t}'][metric]:>10.3f}" for t in insts
        )
        print(f"{src:>8}" + row)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Lightning .ckpt from the ladder script")
    # --- model geometry (must match the checkpoint) ---
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--projector", choices=["identity", "mlp"], default="identity")
    ap.add_argument("--n-slices", type=int, default=128)
    # --- data (defaults = the debug run) ---
    ap.add_argument("--dataset", default="hibb/tess-ztf-atlas-asassn-isect",
                    choices=sorted(ladder.INSTRUMENT_REGISTRY))
    ap.add_argument("--exclude-instrument", action="append", default=[])
    ap.add_argument("--holdout-instrument", default=None)
    ap.add_argument("--max-objects", type=int, default=0)
    ap.add_argument("--min-obs", type=int, default=8)
    # Must match pretraining so load_records yields the identical record
    # list (and therefore the identical stratified split).
    ap.add_argument("--min-train-instruments", type=int, default=2)
    ap.add_argument("--exclude-stars-file", type=str, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--hf-token", type=str, default=None,
                    help="overrides HF_TOKEN env var / cached `hf auth login`")
    # --- eval views (same 500-1500 day windows as pretraining) ---
    ap.add_argument("--window-days", type=float, default=500.0)
    ap.add_argument("--window-days-max", type=float, default=1500.0)
    ap.add_argument("--min-window-obs", type=int, default=200)
    ap.add_argument("--over-budget", choices=["tail", "random"], default="tail")
    ap.add_argument("--eval-tokens", type=int, default=512)
    ap.add_argument("--norm", choices=["band", "band-global", "object", "view"],
                    default="band")
    ap.add_argument("--n-eval-views", type=int, default=4,
                    help="windows averaged into each object's embedding")
    # --- runtime ---
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output prefix (default: <ckpt dir>/downstream)")
    ap.add_argument("--wandb", default=None,
                    help="wandb project to log tables/plots to")
    ap.add_argument("--wandb-id", default=None,
                    help="resume this wandb run id (log the downstream "
                         "results into the pretraining run)")
    args = ap.parse_args()

    args.hf_token = ladder.resolve_hf_token(args.hf_token)

    ladder.configure_instruments(args.dataset, args.holdout_instrument,
                                 tuple(args.exclude_instrument))
    cfg = ladder.ViewConfig(
        window_days=args.window_days, window_days_max=args.window_days_max,
        min_window_obs=args.min_window_obs, over_budget=args.over_budget,
        eval_tokens=args.eval_tokens, min_tokens=args.min_obs, norm=args.norm,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_backbone(args, device)

    records, label_to_idx = ladder.load_records(args)
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    train_idx, val_idx = ladder.stratified_split(records, args.val_frac,
                                                 args.split_seed)
    labels = np.array([r["label"] for r in records])
    logp = np.array([np.log10(r["period"]) if r["period"] else np.nan
                     for r in records], dtype=np.float64)
    print(f"period coverage: {np.isfinite(logp).sum()}/{len(records)} objects")

    # Per-instrument object availability + class histogram (train/val split
    # is shared with pretraining: probes are fit on train stars, tested on
    # val stars the encoder never saw).
    emb = {}
    for inst in ladder.INSTRUMENTS:
        has = np.array([ladder._has_bands(r, ladder.INST_BANDS[inst])
                        for r in records])
        tr = [i for i in train_idx if has[i]]
        va = [i for i in val_idx if has[i]]
        hist = {idx_to_label[c]: int(n) for c, n in
                zip(*np.unique(labels[has], return_counts=True))}
        print(f"{inst}: {has.sum()} objects ({len(tr)} train / {len(va)} val); "
              f"classes {hist}")
        emb[inst] = {
            "train": embed_instrument(backbone, records, tr, inst, cfg, args, device),
            "val": embed_instrument(backbone, records, va, inst, cfg, args, device),
            "train_label": labels[tr], "val_label": labels[va],
            "train_logp": logp[tr], "val_logp": logp[va],
            "train_idx": np.array(tr), "val_idx": np.array(va),
        }
        print(f"{inst}: embedded")

    results = {}
    fit_eval_classification(emb, results)
    preds = fit_eval_period(emb, results)

    print_matrix(results, "cls", "macro_f1", "Variable-star classification")
    print_matrix(results, "cls", "top1", "Variable-star classification")
    print_matrix(results, "per", "r2", "Period regression (log10 days)")
    print_matrix(results, "per", "med_abs_dex", "Period regression (log10 days)")

    out = Path(args.out or Path(args.ckpt).parent / "downstream")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.json", "w") as f:
        json.dump({"args": vars(args), "classes": idx_to_label,
                   "results": results}, f, indent=2)
    np.savez_compressed(
        f"{out}_embeddings.npz",
        **{f"{inst}_{k}": v for inst, d in emb.items() for k, v in d.items()},
    )
    fig = period_scatter_figure(preds, results)
    fig.savefig(f"{out}_period_scatter.png", dpi=120)
    se_fig = starembed_umap_figure(emb, idx_to_label, seed=args.seed)
    se_fig.savefig(f"{out}_starembed_umap.png", dpi=120)
    print(f"\nwrote {out}.json, {out}_embeddings.npz, "
          f"{out}_period_scatter.png and {out}_starembed_umap.png")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb, id=args.wandb_id, dir=str(ladder.RUNS_DIR),
            resume="allow" if args.wandb_id else None,
            name=None if args.wandb_id else f"downstream-{Path(args.ckpt).stem}",
            config={f"downstream/{k}": v for k, v in vars(args).items()},
        )
        insts = list(ladder.INSTRUMENTS)
        for task, metric in [("cls", "macro_f1"), ("cls", "top1"),
                             ("per", "r2"), ("per", "med_abs_dex")]:
            rows = [[src] + [results[f"{task}/{src}_to_{t}"][metric]
                             for t in insts] for src in insts]
            run.log({f"downstream/{task}_{metric}":
                     wandb.Table(columns=["fit_on \\ test_on"] + insts,
                                 data=rows)})
        run.summary.update({
            f"downstream/{cell}/{k}": v
            for cell, m in results.items() for k, v in m.items()
        })
        run.log({"downstream/period_scatter": wandb.Image(fig),
                 "downstream/starembed_umap": wandb.Image(se_fig)})
        run.finish()


if __name__ == "__main__":
    main()
