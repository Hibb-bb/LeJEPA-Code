"""Build the canonical star-level train/val/test split for a dataset.

Writes ``data/splits/<dataset basename>_seed<seed>.json`` with three lists
of ``gaia_dr3_source_id`` (as strings), stratified by ``class_str`` over
EVERY row of the HF dataset (not just the rows a particular run keeps), so
each pretraining / downstream run — whatever its ``--min-obs``, holdout or
exclusion filters — sees the identical held-out stars. Pretraining trains
on ``train`` and monitors ``val``; ``test`` is reported once by
``downstream.py``. Rows without a gaia id are not listed (runs put them in
train). Commit the resulting file.

Usage::

    python data/make_splits.py                       # default dataset, seed 0
    python data/make_splits.py --dataset hibb/TESS-ZTF-isect --seed 1
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import get_token

HERE = Path(__file__).resolve().parent  # data/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hibb/tess-ztf-atlas-asassn-isect")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = os.environ.get("HF_TOKEN") or get_token()
    ds = load_dataset(args.dataset, split="train", token=tok,
                      columns=["gaia_dr3_source_id", "class_str"])
    gids = ds["gaia_dr3_source_id"]
    classes = ds["class_str"]

    by_class, seen, n_dup, n_nogaia = {}, set(), 0, 0
    for g, c in zip(gids, classes):
        if not g:
            n_nogaia += 1
            continue
        g = str(g)
        if g in seen:  # same star twice -> first row wins (as load_records)
            n_dup += 1
            continue
        seen.add(g)
        by_class.setdefault(c, []).append(g)

    rng = np.random.default_rng(args.seed)
    split = {"train": [], "val": [], "test": []}
    for c in sorted(by_class):
        ids = np.array(by_class[c])
        rng.shuffle(ids)
        n_val = int(round(len(ids) * args.val_frac))
        n_test = int(round(len(ids) * args.test_frac))
        split["val"] += ids[:n_val].tolist()
        split["test"] += ids[n_val:n_val + n_test].tolist()
        split["train"] += ids[n_val + n_test:].tolist()
    for k in split:
        split[k].sort()

    out = Path(args.out or HERE / "splits" /
               f"{args.dataset.split('/')[-1]}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": args.dataset, "seed": args.seed,
        "val_frac": args.val_frac, "test_frac": args.test_frac,
        "stratified_by": "class_str", "n_rows": len(gids),
        "n_no_gaia": n_nogaia, "n_duplicate_gaia": n_dup,
        "classes": {c: len(v) for c, v in sorted(by_class.items())},
        "sizes": {k: len(v) for k, v in split.items()},
    }
    out.write_text(json.dumps({**meta, **split}, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
