"""3-encoder x 4-survey UMAP grid: LeJEPA vs supervised vs random init.

Embeds the stars cross-matched in all four surveys (ZTF / ATLAS / ASASSN /
CSS) with three encoders of identical architecture — the LeJEPA checkpoint,
the supervised-baseline checkpoint, and a randomly initialised network —
using the downstream protocol (mean of 4 seeded eval views, the training
band-global normalisation constant). Each row gets its own cosine UMAP fit;
columns show each survey over the row's full map in gray.

Writes ``runs/Cross-Survey-LC/umap_compare.png`` (+ ``.npz`` with each row's
2-D coordinates).
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # benchmarks/lightcurves (main script, downstream.py, runs/)
TRAIN_SCALE = 0.13494884967803955

ENCODERS = [
    ("LeJEPA",
     "runs/Cross-Survey-LC/cs3079146-lejepa/checkpoints/epoch=699-step=59500.ckpt"),
    ("Supervised",
     "runs/Cross-Survey-LC/cs3079146-supervised/checkpoints/epoch=699-step=59500.ckpt"),
    ("Random init", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stars", type=int, default=0, help="0 = all")
    ap.add_argument("--n-eval-views", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/Cross-Survey-LC/umap_compare")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("downstream",
                                                  ROOT / "downstream.py")
    ds_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds_mod)
    ladder = ds_mod.ladder
    ladder.configure_instruments("hibb/tess-ztf-atlas-asassn-isect+css",
                                 holdout="ATLAS", exclude=("TESS",))
    insts = list(ladder.INSTRUMENTS)

    largs = argparse.Namespace(
        max_objects=0, min_obs=8, norm="band", min_train_instruments=1,
        exclude_stars_file=None,
        hf_token=ladder.resolve_hf_token(),
    )
    records, _ = ladder.load_records(largs)
    for r in records:
        c0, s0, centers, _ = r["norm"]
        r["norm"] = (c0, s0, centers, np.float32(TRAIN_SCALE))
    cross = [i for i, r in enumerate(records)
             if all(ladder._has_bands(r, ladder.INST_BANDS[x]) for x in insts)]
    rng = np.random.default_rng(args.seed)
    n_take = len(cross) if args.max_stars <= 0 else min(args.max_stars,
                                                        len(cross))
    sel = sorted(rng.choice(cross, n_take, replace=False).tolist())
    print(f"{len(cross)} cross-matched stars, {len(sel)} sampled")

    cfg = ladder.ViewConfig(window_days=500.0, window_days_max=1500.0,
                            min_window_obs=200, over_budget="tail",
                            eval_tokens=512, min_tokens=8, norm="band-global")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.width, args.depth = 256, 4
    ladder.HEAD_DIM = 32  # pre-2026-09 checkpoints: 8 heads x 32
    args.projector, args.n_slices = "identity", 128

    import umap
    zs = {}
    for name, ckpt in ENCODERS:
        if ckpt is not None:
            args.ckpt = ckpt
            backbone = ds_mod.load_backbone(args, device)
        else:
            torch.manual_seed(args.seed)
            model = ladder.build_model(args.width, args.width, 0.0,
                                       args.n_slices, args.depth,
                                       mode="lejepa", projector="identity")
            backbone = model.backbone.to(device).eval()
        X, inst_id = [], []
        for k, inst in enumerate(insts):
            X.append(ds_mod.embed_instrument(backbone, records, sel, inst,
                                             cfg, args, device))
            inst_id.append(np.full(len(sel), k))
        X = np.concatenate(X)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        inst_id = np.concatenate(inst_id)
        zs[name] = umap.UMAP(n_components=2, random_state=args.seed,
                             metric="cosine").fit_transform(X)
        print(f"{name}: embedded + UMAP done")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    inst_col = {"ZTF": "#0072B2", "ASASSN": "#E69F00", "ATLAS": "#D55E00",
                "CSS": "#009E73"}
    tag = {"ATLAS": " (held out)", "CSS": " (held out)"}
    show = ["ZTF", "ASASSN", "ATLAS", "CSS"]  # display column order
    kw = dict(s=2.0, alpha=0.5, linewidths=0)
    nrow, ncol = len(ENCODERS), len(show)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.2 * nrow),
                             squeeze=False)
    for r, (name, _) in enumerate(ENCODERS):
        z = zs[name]
        for k, inst in enumerate(show):
            ax = axes[r, k]
            ax.scatter(*z.T, c="lightgray", **kw)
            ax.scatter(*z[inst_id == insts.index(inst)].T,
                       c=inst_col[inst], **kw)
            if r == 0:
                ax.set_title(inst + tag.get(inst, ""), fontsize=14)
            if k == 0:
                ax.set_ylabel(name, fontsize=15)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=170)
    np.savez_compressed(f"{args.out}.npz", inst_id=inst_id,
                        star=np.tile(np.array(sel), len(insts)),
                        instruments=np.array(insts),
                        **{f"z_{n.replace(' ', '_')}": z
                           for n, z in zs.items()})
    print(f"wrote {args.out}.png, {args.out}.npz")


if __name__ == "__main__":
    main()
