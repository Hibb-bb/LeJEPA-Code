"""Bar plot of the val-selected probe's test macro-F1 for every train -> test
survey setting (no averaging over the out-of-survey cells).

Reads ``probe_sweep.py``'s JSON: for each encoder and each *training* survey
the probe config was selected on that survey's val macro-F1; this plots the
selected probe's test macro-F1 on each *test* survey. One panel per training
survey, one group per test survey, one bar per method. A handcrafted
random-forest baseline (in-survey only, no HPO) can be added from its
``downstream.json``.

Usage::

    python analysis/plot_sweep_selected.py runs/Pair-ZTF-ASASSN/probe_sweep_100ep.json \
        --handcrafted runs/Pair-ZTF-ASASSN/handcrafted3163304/features_rf.json \
        --out runs/Pair-ZTF-ASASSN/selected_f1_100ep.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

INSTS = ("ZTF", "ATLAS", "ASASSN")
ORDER = ["Contrastive", "LeJEPA λ=0.02", "LeJEPA λ=0.2", "LeJEPA λ=2.0"]
COLORS = {"Handcrafted + RF": "#1baf7a", "Contrastive": "#eb6834",
          "LeJEPA λ=0.02": "#9ec3ee", "LeJEPA λ=0.2": "#2a78d6",
          "LeJEPA λ=2.0": "#123c6e",
          "LeJEPA λ=0.02 + ASAS-SN predictor": "#4a3aa7"}
INK2, GRID = "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--handcrafted", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="f1", choices=["f1", "bal"],
                    help="f1 = macro F1, bal = balanced accuracy")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = json.load(open(args.sweep))["runs"]
    methods = [m for m in ORDER if m in sweep] + [m for m in sweep if m not in ORDER]
    vals = {m: {(s, t): sweep[m]["sources"][s]["best_by_val"][f"test_{args.metric}_{t}"]
                for s in INSTS for t in INSTS} for m in methods}
    cfgs = {m: {s: sweep[m]["sources"][s]["best_by_val"] for s in INSTS} for m in methods}
    if args.handcrafted:
        res = json.load(open(args.handcrafted))["results"]
        key = "macro_f1" if args.metric == "f1" else "balanced"
        vals = {"Handcrafted + RF": {(s, s): res[f"cls/{s}_to_{s}"][key] for s in INSTS},
                **vals}
        methods = ["Handcrafted + RF"] + methods

    n_m = len(methods)
    w = 0.82 / n_m
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, src in zip(axes, INSTS):
        for k, m in enumerate(methods):
            y = [vals[m].get((src, t), np.nan) for t in INSTS]
            x = np.arange(3) + (k - (n_m - 1) / 2) * w
            ax.bar(x, y, w * 0.92, color=COLORS.get(m, "#888888"), label=m, linewidth=0)
            for xi, v in zip(x, y):
                if np.isfinite(v):
                    ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.5, color=INK2, rotation=90)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"test: {t}" + ("\n(in-survey)" if t == src else "")
                            for t in INSTS], fontsize=9)
        ax.set_title(f"probe trained on {src}", fontsize=10.5, loc="left")
        ax.set_ylim(0, 0.78)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("test macro F1" if args.metric == "f1" else "test balanced accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=n_m, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Classification (13 classes), test split: MLP probe selected on the "
                 "training survey's val macro-F1 (100-epoch pretraining; handcrafted "
                 "RF baseline in-survey only)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)

    md = Path(args.out).with_suffix(".md")
    with open(md, "w") as f:
        f.write("| method | " + " | ".join(f"{s}→{t}" for s in INSTS for t in INSTS) + " |\n")
        f.write("|---" * 10 + "|\n")
        for m in methods:
            f.write(f"| {m} | " + " | ".join(
                "—" if (s, t) not in vals[m] else f"{vals[m][(s, t)]:.3f}"
                for s in INSTS for t in INSTS) + " |\n")
        f.write("\nSelected probe config (class weight, lr, batch) per training survey:\n\n")
        for m in methods:
            if m in cfgs:
                f.write(f"- {m}: " + "; ".join(
                    f"{s} ({cfgs[m][s].get('class_weight', 'none')}, {cfgs[m][s]['lr']}, {cfgs[m][s]['batch_size']})"
                    for s in INSTS) + "\n")
    print("wrote", md)


if __name__ == "__main__":
    main()
