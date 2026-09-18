"""Bar-chart comparison of downstream results across methods.

Reads the ``downstream.json`` files written by ``downstream.py`` and draws,
per task, the test-split metric on each survey: the *in-survey* number (probe
fit and tested on the same survey) and the *cross-survey* number (mean over
probes fit on the other surveys). Methods evaluated in-survey only (e.g. the
handcrafted random-forest baseline) simply have no cross-survey bar.

Usage::

    python analysis/compare_pair.py --out runs/Pair-ZTF-ASASSN/compare_100ep.png \
        "Handcrafted + RF=runs/Pair-ZTF-ASASSN/handcrafted3163304/features_rf.json" \
        "Contrastive=runs/Pair-ZTF-ASASSN/pair3163303-contrastive/checkpoints/downstream.json" \
        "LeJEPA λ=0.02=runs/.../pair3163526-lejepa/checkpoints/downstream.json" ...

Each positional is ``label=path``; the first '=' splits label from path only
at the last '=' so labels may contain '='.
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Categorical slots from the dataviz reference palette (validated adjacent
# pairs); the three LeJEPA lambdas share one hue stepped light -> dark since
# lambda is ordered.
COLORS = {
    "handcrafted": "#1baf7a",
    "contrastive": "#eb6834",
    "lejepa": ["#9ec3ee", "#2a78d6", "#123c6e"],
}
PANELS = [
    ("cls", "macro_f1", "Classification (13 classes)", "macro F1", (0, 0.75)),
    ("per", "r2", "Period regression", "R² on log10 P", (-0.5, 1.0)),
]


def load(spec):
    label, path = spec.rsplit("=", 1)
    d = json.load(open(path))
    return label, d["results"], list(d["args"].get("exclude_instrument", []))


def cell(results, task, metric, src, tgt):
    v = results.get(f"{task}/{src}_to_{tgt}", {}).get(metric, float("nan"))
    return float(v)


def color_for(labels):
    out, k = {}, 0
    for lab in labels:
        low = lab.lower()
        if "hand" in low or "rf" in low:
            out[lab] = COLORS["handcrafted"]
        elif "contrast" in low:
            out[lab] = COLORS["contrastive"]
        else:
            out[lab] = COLORS["lejepa"][min(k, 2)]
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=downstream.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--insts", default="ZTF,ATLAS,ASASSN")
    ap.add_argument("--title", default="Frozen-encoder downstream, test split "
                    "(ZTF + ASAS-SN pretraining, ATLAS held out, 100 epochs)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    insts = args.insts.split(",")
    runs = [load(s) for s in args.runs]
    labels = [r[0] for r in runs]
    colors = color_for(labels)

    n_m = len(runs)
    width = 0.8 / n_m
    fig, axes = plt.subplots(len(PANELS), 2, figsize=(11, 3.4 * len(PANELS)),
                             sharex="col")
    for r, (task, metric, title, ylab, ylim) in enumerate(PANELS):
        for c, mode in enumerate(("in-survey", "cross-survey")):
            ax = axes[r, c]
            for m, (lab, res, _) in enumerate(runs):
                vals = []
                for tgt in insts:
                    if mode == "in-survey":
                        vals.append(cell(res, task, metric, tgt, tgt))
                    else:
                        v = [cell(res, task, metric, s, tgt) for s in insts if s != tgt]
                        vals.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
                x = np.arange(len(insts)) + (m - (n_m - 1) / 2) * width
                ax.bar(x, vals, width * 0.92, color=colors[lab], label=lab,
                       linewidth=0)
                # Values below the axis floor are clipped: print them.
                for xi, v in zip(x, vals):
                    if np.isfinite(v) and v < ylim[0]:
                        ax.text(xi, ylim[0] + 0.02 * (ylim[1] - ylim[0]),
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=7, rotation=90, color="#52514e")
            ax.axhline(0, color="#52514e", lw=0.8)
            ax.set_xticks(np.arange(len(insts)))
            ax.set_xticklabels(insts)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", color="#e6e5e1", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if c == 0:
                ax.set_ylabel(ylab)
            sub = ("probe fit on the same survey" if mode == "in-survey"
                   else "probe fit on another survey (mean of the other two)")
            ax.set_title(f"{title} — {mode}\n{sub}", fontsize=10, loc="left")
            if r == len(PANELS) - 1:
                ax.set_xlabel("test survey")
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=min(n_m, 5),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(args.title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    # Companion table (markdown) with the same numbers.
    md = Path(args.out).with_suffix(".md")
    with open(md, "w") as f:
        for task, metric, title, ylab, _ in PANELS:
            f.write(f"\n### {title}: {ylab} (test split)\n\n")
            f.write("| method | " + " | ".join(f"{t} in" for t in insts)
                    + " | " + " | ".join(f"{t} cross" for t in insts) + " |\n")
            f.write("|---" * (1 + 2 * len(insts)) + "|\n")
            for lab, res, _ in runs:
                ins = [cell(res, task, metric, t, t) for t in insts]
                cross = []
                for t in insts:
                    v = [cell(res, task, metric, s, t) for s in insts if s != t]
                    cross.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
                fmt = lambda v: "—" if not np.isfinite(v) else f"{v:.3f}"  # noqa: E731
                f.write(f"| {lab} | " + " | ".join(map(fmt, ins)) + " | "
                        + " | ".join(map(fmt, cross)) + " |\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
