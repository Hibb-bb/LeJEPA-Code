"""Small hyper-parameter sweep of the downstream MLP classification probe.

Re-uses the per-survey embeddings saved by ``downstream.py``
(``downstream_embeddings.npz`` next to a checkpoint, or a handcrafted
``features.npz``) so no re-encoding is needed. For every (hidden, lr, batch)
in the grid one probe is fit per source survey on its train stars; the
config is selected on that survey's **val** macro-F1 (never on test or on
another survey), and the selected probe is then scored on the test split of
every survey, the same protocol as ``downstream.py``.

Usage (interactive GPU node)::

    python analysis/probe_sweep.py \
        "LeJEPA λ=0.02=runs/Pair-ZTF-ASASSN/pair3163526-lejepa/checkpoints/downstream_embeddings.npz" \
        "Contrastive=runs/Pair-ZTF-ASASSN/pair3163303-contrastive/checkpoints/downstream_embeddings.npz" \
        --out runs/Pair-ZTF-ASASSN/probe_sweep_100ep.json

The anomaly classes / class list are read from the ``downstream.json`` that
sits next to each embeddings file.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import downstream as ds  # noqa: E402

INSTS = ("ZTF", "ATLAS", "ASASSN")
# hidden size had no effect in the first sweep (0.439 / 0.439 / 0.435 mean test
# F1), so that axis is replaced by the class weighting of the loss. batch 256
# keeps downstream.py's default config (512, 1e-3, 256, balanced) in the grid.
GRID = dict(hidden=(512,), lr=(3e-4, 1e-3, 3e-3), batch_size=(128, 256, 512),
            class_weight=("none", "sqrt", "balanced"))


def load_run(spec):
    label, path = spec.rsplit("=", 1)
    path = Path(path)
    emb = np.load(path)
    meta_path = path.parent / "downstream.json"
    if not meta_path.exists():  # handcrafted: features_*.json next to it
        cands = sorted(path.parent.glob("*.json"))
        meta_path = next(c for c in cands if "cls_classes" in c.read_text())
    meta = json.load(open(meta_path))
    classes = {int(k): v for k, v in meta["classes"].items()}
    anom = set(meta.get("anomaly_classes", []))
    kept = [c for i, c in sorted(classes.items()) if c not in anom]
    label_map = np.full(len(classes), -1, dtype=np.int64)
    for j, c in enumerate(kept):
        label_map[[i for i, n in classes.items() if n == c][0]] = j
    data = {}
    for inst in INSTS:
        data[inst] = {}
        for sp in ("train", "val", "test"):
            y = label_map[emb[f"{inst}_{sp}_label"]]
            keep = y >= 0
            data[inst][sp] = (emb[f"{inst}_{sp}"][keep], y[keep])
    return label, data, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=embeddings.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from sklearn.metrics import f1_score, balanced_accuracy_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    out = {"grid": GRID, "runs": {}}
    preds_out = {}  # "<label>|<src>|<tgt>|true/pred" of each val-selected probe
    t0 = time.time()
    for spec in args.runs:
        label, data, kept = load_run(spec)
        n_cls = len(kept)
        res = {"n_classes": n_cls, "sources": {}}
        print(f"\n=== {label}: {n_cls} classes, {len(configs)} configs x {len(INSTS)} sources")
        for src in INSTS:
            Xtr, ytr = data[src]["train"]
            Xva, yva = data[src]["val"]
            sc = ds._Imputer(Xtr)
            Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
            rows = []
            best = None
            for cfg in configs:
                m = ds.MLPProbe("cls", n_cls, device, args.seed, **cfg)
                m.fit(Xtr_s, ytr, Xva_s, yva)
                val_f1 = float(m.best_val)
                row = {**cfg, "val_f1": val_f1, "epochs": m.epochs_run}
                for tgt in INSTS:
                    Xte, yte = data[tgt]["test"]
                    p = m.predict(sc.transform(Xte))
                    row[f"test_f1_{tgt}"] = float(f1_score(yte, p, average="macro"))
                    row[f"test_bal_{tgt}"] = float(balanced_accuracy_score(yte, p))
                rows.append(row)
                if best is None or val_f1 > best["val_f1"]:
                    best = row
                    for tgt in INSTS:
                        Xte, yte = data[tgt]["test"]
                        preds_out[f"{label}|{src}|{tgt}|true"] = yte
                        preds_out[f"{label}|{src}|{tgt}|pred"] = m.predict(sc.transform(Xte))
                print(f"  {src:<7} cw={cfg['class_weight']:<9} lr={cfg['lr']:<7} bs={cfg['batch_size']:<4} "
                      f"val F1 {val_f1:.3f}  test " +
                      " ".join(f"{t}:{row[f'test_f1_{t}']:.3f}" for t in INSTS) +
                      f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
            # Default-config reference (downstream.py's MLP_CFG).
            is_ref = lambda r, cw: (r["hidden"] == 512 and r["lr"] == 1e-3  # noqa: E731
                                    and r["batch_size"] == 256 and r["class_weight"] == cw)
            ref = next((r for r in rows if is_ref(r, "balanced")), None)
            ref_unw = next((r for r in rows if is_ref(r, "none")), None)
            res["sources"][src] = {"rows": rows, "best_by_val": best, "default": ref,
                                   "default_unweighted": ref_unw}
            print(f"  -> {src}: best by val F1 = cw {best['class_weight']} lr{best['lr']} "
                  f"bs{best['batch_size']} (val {best['val_f1']:.3f}); test "
                  + " ".join(f"{t}:{best[f'test_f1_{t}']:.3f}" for t in INSTS))
        res["classes"] = kept
        res["train_counts"] = {s: np.bincount(data[s]["train"][1], minlength=n_cls).tolist()
                               for s in INSTS}
        out["runs"][label] = res
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        np.savez_compressed(str(Path(args.out).with_suffix("")) + "_predictions.npz",
                            **preds_out)
    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
