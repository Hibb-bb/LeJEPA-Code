"""UMAP of frozen cs3079146 embeddings over ZTF / ASAS-SN / ATLAS / CSS.

Uses the single-GPU band-global checkpoint (which never saw CSS; ATLAS was
the training holdout — both are zero-shot instruments here), the merged
isect+CSS dataset restricted to stars cross-matched in ALL four surveys,
and the downstream protocol (mean of 4 seeded eval views per star per
instrument, training normalisation constant). One joint cosine UMAP over
every instrument's embeddings, then:

  <out>_surveys.png  joint map coloured by survey + one panel per survey
  <out>_classes.png  one panel per variable-star class (gray = all points)
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # benchmarks/lightcurves (main script, downstream.py, runs/)
TRAIN_SCALE = 0.13494884967803955  # band-global scale of the cs3079146 runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/Cross-Survey-LC/cs3079146-lejepa/"
                                      "checkpoints/epoch=699-step=59500.ckpt")
    ap.add_argument("--max-stars", type=int, default=4000,
                    help="0 = all cross-matched stars")
    ap.add_argument("--n-eval-views", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output prefix (default: <ckpt dir>/umap_css)")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("downstream",
                                                  ROOT / "downstream.py")
    ds_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds_mod)
    ladder = ds_mod.ladder
    ladder.configure_instruments("hibb/tess-ztf-atlas-asassn-isect+css",
                                 holdout="ATLAS", exclude=("TESS",))
    insts = list(ladder.INSTRUMENTS)  # ZTF, ATLAS, ASASSN, CSS

    largs = argparse.Namespace(
        max_objects=0, min_obs=8, norm="band", min_train_instruments=1,
        exclude_stars_file=None,
        hf_token=ladder.resolve_hf_token(),
    )
    records, label_to_idx = ladder.load_records(largs)
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    # The checkpoint's normalisation is a train-time constant: per-band
    # centres from the star itself, ONE fixed global scale.
    for r in records:
        c0, s0, centers, _ = r["norm"]
        r["norm"] = (c0, s0, centers, np.float32(TRAIN_SCALE))

    cross = [i for i, r in enumerate(records)
             if all(ladder._has_bands(r, ladder.INST_BANDS[x]) for x in insts)]
    rng = np.random.default_rng(args.seed)
    n_take = len(cross) if args.max_stars <= 0 else min(args.max_stars,
                                                        len(cross))
    sel = sorted(rng.choice(cross, n_take, replace=False).tolist())
    print(f"{len(records)} records, {len(cross)} cross-matched in all four "
          f"surveys, {len(sel)} sampled")

    cfg = ladder.ViewConfig(window_days=500.0, window_days_max=1500.0,
                            min_window_obs=200, over_budget="tail",
                            eval_tokens=512, min_tokens=8, norm="band-global")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.width, args.depth = 256, 4
    args.projector, args.n_slices = "identity", 128
    backbone = ds_mod.load_backbone(args, device)

    X, inst_id = [], []
    for k, inst in enumerate(insts):
        X.append(ds_mod.embed_instrument(backbone, records, sel, inst, cfg,
                                         args, device))
        inst_id.append(np.full(len(sel), k))
        print(f"{inst}: embedded {len(sel)} stars")
    X = np.concatenate(X)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    inst_id = np.concatenate(inst_id)
    labels = np.array([records[i]["label"] for i in sel])
    cls = np.tile(labels, len(insts))
    logp = np.log10([records[i]["period"] if records[i]["period"]
                     else np.nan for i in sel])
    logp_all = np.tile(logp, len(insts))

    # Pairwise instrument separability (training-callback definition:
    # 3-fold CV accuracy of a logistic probe telling survey A from B;
    # 0.5 = the clouds fully overlap).
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    print("\npairwise sep_acc (0.5 = indistinguishable):")
    for a in range(len(insts)):
        for b in range(a + 1, len(insts)):
            xa, xb = X[inst_id == a], X[inst_id == b]
            x = np.concatenate([xa, xb])
            t = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
            s = cross_val_score(LogisticRegression(max_iter=500), x, t,
                                cv=3).mean()
            print(f"  {insts[a]:>7} vs {insts[b]:<7} sep_acc={s:.3f}")

    import umap
    z = umap.UMAP(n_components=2, random_state=args.seed,
                  metric="cosine").fit_transform(X)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    out = Path(args.out or Path(args.ckpt).parent / "umap_css")
    kw = dict(s=2.5, alpha=0.5, linewidths=0)
    inst_col = {"ZTF": "#0072B2", "ASASSN": "#E69F00", "ATLAS": "#D55E00",
                "CSS": "#009E73"}
    tag = {"ATLAS": " (held out)", "CSS": " (never trained)"}

    # --- survey figure ----------------------------------------------------
    fig, axes = plt.subplots(1, 1 + len(insts),
                             figsize=(4.2 * (1 + len(insts)), 4.2))
    ax = axes[0]
    for k, inst in enumerate(insts):
        m = inst_id == k
        ax.scatter(*z[m].T, c=inst_col[inst], label=inst, **kw)
    ax.legend(loc="best", frameon=False, markerscale=6, fontsize=11)
    ax.set_title("all surveys")
    for k, inst in enumerate(insts):
        ax = axes[1 + k]
        ax.scatter(*z.T, c="lightgray", **kw)
        ax.scatter(*z[inst_id == k].T, c=inst_col[inst], **kw)
        ax.set_title(inst + tag.get(inst, ""))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{out}_surveys.png", dpi=200)

    # --- class figure -----------------------------------------------------
    counts = {c: int((labels == c).sum()) for c in np.unique(labels)}
    show = [c for c, n in sorted(counts.items(), key=lambda t: -t[1])
            if n >= 40][:8]
    cmap = plt.get_cmap("tab10")
    ncol = 4
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.0 * nrow),
                             squeeze=False)
    for j, c in enumerate(show):
        ax = axes[j // ncol, j % ncol]
        ax.scatter(*z.T, c="lightgray", **kw)
        m = cls == c
        ax.scatter(*z[m].T, c=[cmap(j)], s=3.5, alpha=0.75, linewidths=0)
        ax.set_title(f"{idx_to_label[int(c)]}  (n={counts[c]} stars)",
                     fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(show), nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(f"{out}_classes.png", dpi=200)

    # --- big grid: every class x every survey ----------------------------
    all_cls = [c for c, n in sorted(counts.items(), key=lambda t: -t[1])]
    cmap20 = plt.get_cmap("tab20")
    nrows, ncols = len(insts), len(all_cls)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.1 * ncols, 2.3 * nrows),
                             squeeze=False)
    for r, inst in enumerate(insts):
        for j, c in enumerate(all_cls):
            ax = axes[r, j]
            ax.scatter(*z.T, c="lightgray", s=1.0, alpha=0.3, linewidths=0)
            m = (cls == c) & (inst_id == r)
            ax.scatter(*z[m].T, c=[cmap20(j % 20)], s=3.0, alpha=0.85,
                       linewidths=0)
            if r == 0:
                ax.set_title(f"{idx_to_label[int(c)]}\nn={counts[c]}",
                             fontsize=10)
            if j == 0:
                ax.set_ylabel(inst + tag.get(inst, ""), fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{out}_classes_by_survey.png", dpi=140)

    # --- period heat map: is the manifold organised by period? ------------
    order = rng.permutation(len(z))  # shuffled draw order, no overplot bias
    vmin, vmax = np.nanpercentile(logp_all, [2, 98])
    fig, axes = plt.subplots(1, 1 + len(insts),
                             figsize=(4.9 + 4.2 * len(insts), 4.4))
    sc = axes[0].scatter(*z[order].T, c=logp_all[order], cmap="viridis",
                         vmin=vmin, vmax=vmax, s=2.5, alpha=0.8,
                         linewidths=0)
    axes[0].set_title("all surveys")
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.05, pad=0.02)
    cb.set_label(r"log$_{10}$ P [d]")
    for k, inst in enumerate(insts):
        ax = axes[1 + k]
        ax.scatter(*z.T, c="0.92", s=1.2, alpha=0.5, linewidths=0)
        m = order[inst_id[order] == k]
        ax.scatter(*z[m].T, c=logp_all[m], cmap="viridis", vmin=vmin,
                   vmax=vmax, s=2.5, alpha=0.85, linewidths=0)
        ax.set_title(inst + tag.get(inst, ""))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{out}_period.png", dpi=200)

    np.savez_compressed(f"{out}.npz", z=z, inst_id=inst_id, cls=cls,
                        star=np.tile(np.array(sel), len(insts)),
                        logp=logp_all,
                        gaia=np.tile(np.array([records[i]["gaia"] or 0
                                               for i in sel]), len(insts)),
                        instruments=np.array(insts))
    print(f"wrote {out}_surveys.png, {out}_classes.png, "
          f"{out}_classes_by_survey.png, {out}_period.png, {out}.npz")


if __name__ == "__main__":
    main()
