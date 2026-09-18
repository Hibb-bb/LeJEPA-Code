"""Report figures from saved downstream results (no re-encoding).

Produces three figures next to ``--out-prefix``:

1. ``<prefix>_transfer.png`` - the cross-survey transfer matrices (rows =
   probe fit on, cols = tested on; test split) for classification macro-F1
   and period R², one heatmap per method, read from ``downstream.json``.
2. ``<prefix>_confusion.png`` - row-normalised confusion matrices of the
   in-survey classification probe on the test split, rows = method, cols =
   survey. Probes are refit here from the saved per-survey embeddings /
   features (``downstream.py`` stores metrics, not predictions), with the
   same probe types as the reported numbers (MLP for encoders, random forest
   for handcrafted features).
3. ``<prefix>_perclass.png`` - per-class test F1 of those in-survey probes,
   classes ordered by training-set frequency (annotated), one panel per
   survey - the class-imbalance view.

Usage::

    python analysis/report_figs.py --out-prefix runs/Pair-ZTF-ASASSN/report_100ep \
        "Handcrafted + RF=rf=runs/Pair-ZTF-ASASSN/handcrafted3163304/features_rf.json" \
        "Contrastive=mlp=runs/.../pair3163303-contrastive/checkpoints/downstream.json" \
        "LeJEPA λ=0.02=mlp=runs/.../pair3163526-lejepa/checkpoints/downstream.json" ...

Each positional is ``label=probe=downstream.json``; the embeddings file is
found next to the json (``<stem>_embeddings.npz``). ``--confusion-runs``
selects which labels get refit for figures 2-3 (default: all).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import downstream as ds  # noqa: E402

INSTS = ("ZTF", "ATLAS", "ASASSN")
# Sequential blue ramp (dataviz reference palette, steps 100 -> 700).
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
METHOD_COLORS = {"handcrafted": "#1baf7a", "contrastive": "#eb6834",
                 "lejepa": ["#9ec3ee", "#2a78d6", "#123c6e"]}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def parse(spec):
    label, probe, path = spec.rsplit("=", 2)  # label may itself contain '='

    path = Path(path)
    meta = json.load(open(path))
    emb_path = path.with_name(path.stem + "_embeddings.npz")
    return dict(label=label, probe=probe, meta=meta, emb_path=emb_path)


def method_color(labels):
    out, k = {}, 0
    for lab in labels:
        low = lab.lower()
        if "hand" in low:
            out[lab] = METHOD_COLORS["handcrafted"]
        elif "contrast" in low:
            out[lab] = METHOD_COLORS["contrastive"]
        else:
            out[lab] = METHOD_COLORS["lejepa"][min(k, 2)]; k += 1
    return out


def blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blues_ref", BLUES)


# --------------------------------------------------------------------------
def fig_transfer(runs, out):
    import matplotlib.pyplot as plt
    tasks = [("cls", "macro_f1", "Classification macro-F1", 0.0, 0.7),
             ("per", "r2", "Period R²", 0.0, 0.85)]
    n = len(runs)
    fig, axes = plt.subplots(len(tasks), n, figsize=(2.9 * n, 2.9 * len(tasks) + 0.6),
                             squeeze=False)
    cmap = blue_cmap()
    for r, (task, metric, title, vmin, vmax) in enumerate(tasks):
        for c, run in enumerate(runs):
            ax = axes[r, c]
            res = run["meta"]["results"]
            M = np.array([[res.get(f"{task}/{s}_to_{t}", {}).get(metric, np.nan)
                           for t in INSTS] for s in INSTS], dtype=float)
            shown = np.where(np.isnan(M), np.nan, np.clip(M, vmin, vmax))
            ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
            for i in range(3):
                for j in range(3):
                    v = M[i, j]
                    if np.isnan(v):
                        ax.text(j, i, "—", ha="center", va="center", color=INK2, fontsize=9)
                        continue
                    dark = (v - vmin) / (vmax - vmin) > 0.55
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if dark else INK)
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(INSTS, fontsize=8); ax.set_yticklabels(INSTS, fontsize=8)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(run["label"], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{title}\nprobe fit on", fontsize=9)
            if r == len(tasks) - 1:
                ax.set_xlabel("tested on", fontsize=9)
    fig.suptitle("Cross-survey transfer, test split (100-epoch pretraining; "
                 "handcrafted baseline evaluated in-survey only)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
def refit_in_survey(run, seed=0):
    """Refit the in-survey classification probe per survey; return
    ``{inst: (y_true, y_pred)}`` on the test split plus train counts."""
    emb = np.load(run["emb_path"])
    meta = run["meta"]
    classes = {int(k): v for k, v in meta["classes"].items()}
    anom = set(meta.get("anomaly_classes", []))
    kept = [c for i, c in sorted(classes.items()) if c not in anom]
    label_map = np.full(len(classes), -1, dtype=np.int64)
    for j, c in enumerate(kept):
        label_map[[i for i, n in classes.items() if n == c][0]] = j
    ds.PROBE = run["probe"]
    out, counts = {}, {}
    for inst in INSTS:
        def rows(sp):
            y = label_map[emb[f"{inst}_{sp}_label"]]
            k = y >= 0
            return emb[f"{inst}_{sp}"][k], y[k]
        Xtr, ytr = rows("train"); Xva, yva = rows("val"); Xte, yte = rows("test")
        sc = ds._Imputer(Xtr)
        predict = ds._fit_probe("cls", sc.transform(Xtr), ytr, sc.transform(Xva), yva,
                                len(kept), "cuda" if _cuda() else "cpu", seed)
        out[inst] = (yte, predict(sc.transform(Xte)))
        counts[inst] = np.bincount(ytr, minlength=len(kept))
        print(f"  refit {run['label']} / {inst}: {len(ytr)} train, {len(yte)} test", flush=True)
    return kept, out, counts


def _cuda():
    import torch
    return torch.cuda.is_available()


def fig_confusion(fits, out):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    labels = list(fits)
    kept = fits[labels[0]][0]
    # Order classes by pooled training frequency (most common first).
    pooled = sum(sum(fits[l][2].values()) for l in labels)
    order = np.argsort(-pooled)
    names = [kept[i] for i in order]
    n = len(kept)
    fig, axes = plt.subplots(len(labels), 3, figsize=(3.6 * 3 + 0.8, 3.5 * len(labels) + 0.8),
                             squeeze=False)
    cmap = blue_cmap()
    for r, lab in enumerate(labels):
        _, preds, _ = fits[lab]
        for c, inst in enumerate(INSTS):
            ax = axes[r, c]
            y, p = preds[inst]
            cm = confusion_matrix(y, p, labels=order)
            support = cm.sum(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                cmn = np.where(support > 0, cm / support, np.nan)
            ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
            for i in range(n):
                for j in range(n):
                    if support[i, 0] == 0:
                        continue
                    v = cmn[i, j]
                    if v >= 0.05:
                        ax.text(j, i, f"{v:.2f}" if v < 0.995 else "1.0", ha="center",
                                va="center", fontsize=5.5,
                                color="white" if v > 0.55 else INK)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(names, rotation=90, fontsize=6.5)
            ax.set_yticklabels([f"{nm} ({int(s)})" for nm, s in zip(names, support[:, 0])],
                               fontsize=6.5)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            from sklearn.metrics import f1_score
            f1 = f1_score(y, p, average="macro")
            ax.set_title(f"{lab} — {inst}   macro-F1 {f1:.2f}", fontsize=9, loc="left")
            if c == 0:
                ax.set_ylabel("true class (test support)", fontsize=8)
            if r == len(labels) - 1:
                ax.set_xlabel("predicted class", fontsize=8)
    fig.suptitle("In-survey classification, test split: row-normalised confusion "
                 "(recall on the diagonal)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig_perclass(fits, out):
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    labels = list(fits)
    kept = fits[labels[0]][0]
    colors = method_color(labels)
    n = len(kept)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=False)
    w = 0.8 / len(labels)
    for r, inst in enumerate(INSTS):
        ax = axes[r]
        cnt = fits[labels[0]][2][inst]
        order = np.argsort(-cnt)
        for m, lab in enumerate(labels):
            y, p = fits[lab][1][inst]
            f1 = f1_score(y, p, labels=order, average=None, zero_division=0)
            x = np.arange(n) + (m - (len(labels) - 1) / 2) * w
            ax.bar(x, f1, w * 0.92, color=colors[lab], label=lab, linewidth=0)
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{kept[i]}\n(n={int(cnt[i])})" for i in order], fontsize=7.5)
        ax.set_ylim(0, 1.02); ax.set_ylabel("test F1")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{inst}: per-class F1 of the in-survey probe (classes ordered by "
                     f"training count)", fontsize=10, loc="left")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=probe=downstream.json")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--confusion-runs", default=None,
                    help="comma-separated labels to refit for figures 2-3 (default all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")

    runs = [parse(s) for s in args.runs]
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    fig_transfer(runs, f"{args.out_prefix}_transfer.png")

    want = (set(args.confusion_runs.split(",")) if args.confusion_runs
            else {r["label"] for r in runs})
    fits = {}
    for run in runs:
        if run["label"] in want:
            fits[run["label"]] = refit_in_survey(run, args.seed)
    preds = {f"{lab}|{inst}|{k}": v for lab, (_, pr, _) in fits.items()
             for inst, (yt, yp) in pr.items() for k, v in (("true", yt), ("pred", yp))}
    np.savez_compressed(f"{args.out_prefix}_predictions.npz", **preds)
    fig_confusion(fits, f"{args.out_prefix}_confusion.png")
    fig_perclass(fits, f"{args.out_prefix}_perclass.png")


if __name__ == "__main__":
    main()
