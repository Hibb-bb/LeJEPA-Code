"""Handcrafted light-curve features: the non-learned baseline for downstream.py.

Reproduces the StarEmbed feature set
(``src/model/handcrafted_features/extract_feats.py`` in
https://github.com/skai-institute/StarEmbed): per band, the 34 FATS features
(Lomb-Scargle period + fit, Psi_CS/eta, autocorrelation length, Con,
PairSlopeTrend, the 3x4 Fourier harmonic amplitudes and relative phases and
the CAR sigma/tau/mean) concatenated with the 33 ``light_curve`` (Rust)
features (amplitude, Anderson-Darling, beyond-n-std, cusum, eta, IPR,
kurtosis, linear fit/trend, magnitude percentage ratio, max slope, mean,
median, MAD, median buffer range, Otsu split, percent amplitude, reduced
chi2, skew, std, Stetson K, weighted mean).

Fairness with the learned embeddings: features are computed on exactly the
observations the encoder sees in ``downstream.py`` -- the same
``--window-days-max``-capped eval span of each star, drawn with the same seed
(``(seed, record index, view)``), per instrument, on raw magnitudes (no
normalisation). So a star seen through ZTF, ATLAS or ASAS-SN gets one feature
vector per instrument (bands of that instrument concatenated; a band the
star lacks in that window is filled with NaN, later median-imputed by the
probe), and the train/val/test split is the dataset's own.

Per star and instrument the per-band vectors are **concatenated** in
blue -> red band order (StarEmbed convention; ZTF g + r with
``--exclude-band i_ZTF``, ATLAS c + o, ASAS-SN g + V -> 2 x 67 = 134 dims).
Cross-survey transfer needs one feature space for every survey, so every
instrument must keep the same number of bands (checked); ``--band-agg
mean`` averages over bands instead (67 dims, band-count agnostic). The output
``.npz`` has the same layout as ``downstream.py``'s embedding dump and is
consumed with ``downstream.py --features <npz>``.

Usage::

    python handcrafted_features.py --dataset pc/ZTF-ATLAS-ASASSN-isect \
        --exclude-instrument TESS --holdout-instrument ATLAS \
        --out runs/handcrafted/ztf-atlas-asassn --num-workers 64
    python downstream.py --features runs/handcrafted/ztf-atlas-asassn.npz \
        --dataset pc/ZTF-ATLAS-ASASSN-isect --exclude-instrument TESS \
        --holdout-instrument ATLAS --probe mlp

``--n-eval-views 1`` (default) computes features on one window per star;
``k > 1`` averages the feature vectors over ``k`` seeded windows (the
embedding side averages embeddings the same way).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pretrain as ladder  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Identical to StarEmbed's list (its two duplicates, Con and
# PairSlopeTrend, dropped: FATS returns one value per unique name).
FATS_FEATURE_NAMES = [
    "PeriodLS", "Period_fit", "Psi_CS", "Psi_eta", "Autocor_length", "Con",
    "PairSlopeTrend",
    "Freq1_harmonics_amplitude_0", "Freq1_harmonics_amplitude_1",
    "Freq1_harmonics_amplitude_2", "Freq1_harmonics_amplitude_3",
    "Freq2_harmonics_amplitude_0", "Freq2_harmonics_amplitude_1",
    "Freq2_harmonics_amplitude_2", "Freq2_harmonics_amplitude_3",
    "Freq3_harmonics_amplitude_0", "Freq3_harmonics_amplitude_1",
    "Freq3_harmonics_amplitude_2", "Freq3_harmonics_amplitude_3",
    "Freq1_harmonics_rel_phase_0", "Freq1_harmonics_rel_phase_1",
    "Freq1_harmonics_rel_phase_2", "Freq1_harmonics_rel_phase_3",
    "Freq2_harmonics_rel_phase_0", "Freq2_harmonics_rel_phase_1",
    "Freq2_harmonics_rel_phase_2", "Freq2_harmonics_rel_phase_3",
    "Freq3_harmonics_rel_phase_0", "Freq3_harmonics_rel_phase_1",
    "Freq3_harmonics_rel_phase_2", "Freq3_harmonics_rel_phase_3",
    "CAR_sigma", "CAR_tau", "CAR_mean",
]

_LC_EXTRACTOR = None


def lc_extractor():
    """The StarEmbed ``light_curve.Extractor`` (built lazily per process)."""
    global _LC_EXTRACTOR
    if _LC_EXTRACTOR is None:
        import light_curve as lc
        _LC_EXTRACTOR = lc.Extractor(
            lc.Amplitude(), lc.AndersonDarlingNormal(),
            lc.BeyondNStd(nstd=1), lc.BeyondNStd(nstd=2), lc.BeyondNStd(nstd=3),
            lc.Cusum(), lc.Eta(), lc.EtaE(),
            lc.InterPercentileRange(0.25), lc.InterPercentileRange(0.1),
            lc.Kurtosis(), lc.LinearFit(), lc.LinearTrend(),
            lc.MagnitudePercentageRatio(), lc.MaximumSlope(), lc.Mean(),
            lc.Median(), lc.MedianAbsoluteDeviation(),
            lc.MedianBufferRangePercentage(), lc.OtsuSplit(),
            lc.PercentAmplitude(), lc.ReducedChi2(), lc.Skew(),
            lc.StandardDeviation(), lc.StetsonK(), lc.WeightedMean(),
        )
    return _LC_EXTRACTOR


def lc_feature_names():
    return list(lc_extractor().names)


def fats_features(t, m, e):
    """34 FATS features of one band (NaN vector on failure / too few points)."""
    n = len(FATS_FEATURE_NAMES)
    if t.size < 5:
        return np.full(n, np.nan)
    import FATS
    try:
        fs = FATS.FeatureSpace(Data=["magnitude", "time", "error"],
                               featureList=FATS_FEATURE_NAMES)
        res = fs.calculateFeature(np.array([m, t, e])).result(method="dict")
        return np.array([float(res[k]) for k in FATS_FEATURE_NAMES], dtype=np.float64)
    except Exception:  # noqa: BLE001 - FATS raises assorted numeric errors
        return np.full(n, np.nan)


def lc_features(t, m, e):
    """33 ``light_curve`` features of one band (NaN vector if < 5 points)."""
    ext = lc_extractor()
    if t.size < 5:
        return np.full(len(ext.names), np.nan)
    try:
        return np.asarray(ext(t, m, e, sorted=True, check=False), dtype=np.float64)
    except Exception:  # noqa: BLE001
        return np.full(len(ext.names), np.nan)


def band_window(record, band, lo, hi):
    """Sorted raw ``(t, mag, mag_unc)`` of one band inside ``[lo, hi)``."""
    arr = record["bands"].get(band)
    if arr is None:
        return None
    t, v, u = arr
    m = (t >= lo) & (t < hi)
    if not m.any():
        return None
    t, v, u = t[m].astype(np.float64), v[m].astype(np.float64), u[m].astype(np.float64)
    o = np.argsort(t, kind="stable")
    t, v, u = t[o], v[o], u[o]
    # Strictly increasing times for the Rust extractor; jitter exact ties.
    same = np.diff(t) <= 0
    if same.any():
        t = t + np.arange(t.size) * 1e-6
    u = np.where(u > 0, u, 1e-3)
    return t, v, u


def eval_window(record, cfg, rng, bands):
    """The span :func:`pretrain.make_eval_view` would use (same rng draws)."""
    all_t = np.concatenate([record["bands"][b][0] for b in record["bands"]])
    lo, hi = float(all_t.min()), float(all_t.max()) + 1.0
    if cfg.window_days_max > 0 and hi - lo > cfg.window_days_max:
        lo, hi = ladder._capped_span(record, cfg, rng, bands)
    return lo, hi


# Worker state (set once per process by the pool initializer; the records
# list is inherited through fork, not pickled per task).
_W = {}


def _init_worker(records, cfg, seed, n_views, use_fats):
    _W.update(records=records, cfg=cfg, seed=seed, n_views=n_views,
              use_fats=use_fats)
    lc_extractor()


def _one(task):
    """Feature vector of one (record index, instrument): mean over views."""
    i, inst = task
    rec, cfg = _W["records"][i], _W["cfg"]
    bands = ladder.INST_BANDS[inst]
    n_fats = len(FATS_FEATURE_NAMES) if _W["use_fats"] else 0
    n_lc = len(lc_extractor().names)
    per_band = n_fats + n_lc
    vecs = []
    for rep in range(_W["n_views"]):
        rng = np.random.default_rng((_W["seed"], i, rep))
        lo, hi = eval_window(rec, cfg, rng, bands)
        out = np.full(len(bands) * per_band, np.nan)
        for k, b in enumerate(bands):
            w = band_window(rec, b, lo, hi)
            if w is None:
                continue
            t, m, e = w
            parts = []
            if _W["use_fats"]:
                parts.append(fats_features(t, m, e))
            parts.append(lc_features(t, m, e))
            out[k * per_band:(k + 1) * per_band] = np.concatenate(parts)
        vecs.append(out)
    vec = np.nanmean(np.stack(vecs), axis=0) if len(vecs) > 1 else vecs[0]
    return i, inst, vec


def per_band_names(use_fats=True):
    return (list(FATS_FEATURE_NAMES) if use_fats else []) + lc_feature_names()


def feature_names(inst, use_fats=True):
    """Per-band concatenated names (``<inst>_<split>_perband`` columns)."""
    return [f"{ladder.BAND_NAMES[b]}_{f}" for b in ladder.INST_BANDS[inst]
            for f in per_band_names(use_fats)]


def band_average(X, n_bands):
    """``[n, n_bands * d] -> [n, d]`` nanmean over bands (NaN if no band)."""
    if X.shape[0] == 0:
        return X.reshape(0, X.shape[1] // max(n_bands, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(X.reshape(X.shape[0], n_bands, -1), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pc/ZTF-ATLAS-ASASSN-isect",
                    choices=sorted(ladder.INSTRUMENT_REGISTRY))
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--exclude-instrument", action="append", default=[])
    ap.add_argument("--exclude-band", action="append", default=[],
                    help="drop a band column, e.g. i_ZTF (match pretraining)")
    ap.add_argument("--band-agg", choices=["concat", "mean"], default="concat",
                    help="concat: per-band vectors concatenated blue->red "
                         "(all instruments need the same band count); mean: "
                         "nanmean over bands")
    ap.add_argument("--holdout-instrument", default=None)
    ap.add_argument("--max-objects", type=int, default=0)
    ap.add_argument("--min-obs", type=int, default=8)
    ap.add_argument("--min-train-instruments", type=int, default=2)
    ap.add_argument("--exclude-stars-file", type=str, default=None)
    ap.add_argument("--split-file", type=str, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--hf-token", type=str, default=None)
    # Same eval-window geometry as downstream.py.
    ap.add_argument("--window-days", type=float, default=500.0)
    ap.add_argument("--window-days-max", type=float, default=1500.0)
    ap.add_argument("--min-window-obs", type=int, default=200)
    ap.add_argument("--n-eval-views", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fats", action="store_true",
                    help="light_curve features only (FATS is ~4 s per band)")
    ap.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--out", required=True,
                    help="output prefix; writes <out>.npz and <out>_meta.json")
    args = ap.parse_args()

    ladder.configure_instruments(args.dataset, args.holdout_instrument,
                                 tuple(args.exclude_instrument),
                                 tuple(args.exclude_band))
    n_bands = {inst: len(b) for inst, b in ladder.INST_BANDS.items()}
    if args.band_agg == "concat" and len(set(n_bands.values())) != 1:
        raise SystemExit(f"--band-agg concat needs equal band counts per "
                         f"instrument, got {n_bands}; drop bands with "
                         f"--exclude-band or use --band-agg mean")
    cfg = ladder.ViewConfig(window_days=args.window_days,
                            window_days_max=args.window_days_max,
                            min_window_obs=args.min_window_obs,
                            min_tokens=args.min_obs)
    records, label_to_idx = ladder.load_records(args)
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    train_idx, val_idx, test_idx = ladder.split_records(records, args)
    labels = np.array([r["label"] for r in records])
    logp = np.array([np.log10(r["period"]) if r["period"] else np.nan
                     for r in records])
    use_fats = not args.no_fats

    tasks, membership = [], {}
    for inst in ladder.INSTRUMENTS:
        has = np.array([ladder._has_bands(r, ladder.INST_BANDS[inst]) for r in records])
        membership[inst] = {
            "train": [i for i in train_idx if has[i]],
            "val": [i for i in val_idx if has[i]],
            "test": [i for i in test_idx if has[i]],
        }
        tasks += [(i, inst) for i in np.flatnonzero(has)]
        print(f"{inst}: {int(has.sum())} objects "
              f"({', '.join(f'{len(v)} {k}' for k, v in membership[inst].items())}), "
              f"{len(feature_names(inst, use_fats)) if args.band_agg == 'concat' else len(per_band_names(use_fats))} "
              f"features ({args.band_agg} over bands "
              f"{[ladder.BAND_NAMES[b] for b in ladder.INST_BANDS[inst]]})")
    print(f"{len(tasks)} (star, instrument) feature vectors with "
          f"{args.num_workers} workers (fats={use_fats})")

    t0 = time.time()
    feats = {inst: {} for inst in ladder.INSTRUMENTS}
    ctx = mp.get_context("fork")
    with ctx.Pool(args.num_workers, initializer=_init_worker,
                  initargs=(records, cfg, args.seed, args.n_eval_views,
                            use_fats)) as pool:
        for n, (i, inst, vec) in enumerate(
                pool.imap_unordered(_one, tasks, chunksize=8), 1):
            feats[inst][i] = vec
            if n % 2000 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"  {n}/{len(tasks)} done, {el/60:.1f} min elapsed, "
                      f"eta {el/n*(len(tasks)-n)/60:.1f} min", flush=True)

    out = {}
    for inst in ladder.INSTRUMENTS:
        for name, idx in membership[inst].items():
            X = np.stack([feats[inst][i] for i in idx]) if idx else \
                np.zeros((0, len(feature_names(inst, use_fats))))
            if args.band_agg == "mean":
                X = band_average(X, len(ladder.INST_BANDS[inst]))
            out[f"{inst}_{name}"] = X.astype(np.float32)
            out[f"{inst}_{name}_label"] = labels[idx]
            out[f"{inst}_{name}_logp"] = logp[idx]
            out[f"{inst}_{name}_idx"] = np.array(idx)
        nan_frac = float(np.isnan(out[f"{inst}_train"]).mean()) if membership[inst]["train"] else 0.0
        print(f"{inst}: NaN fraction in train features {nan_frac:.3f}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{outp}.npz", **out)
    with open(f"{outp}_meta.json", "w") as f:
        json.dump({"args": vars(args), "classes": idx_to_label,
                   "band_agg": args.band_agg,
                   "feature_names": ({inst: feature_names(inst, use_fats)
                                      for inst in ladder.INSTRUMENTS}
                                     if args.band_agg == "concat"
                                     else per_band_names(use_fats)),
                   "instruments": list(ladder.INSTRUMENTS),
                   "minutes": (time.time() - t0) / 60}, f, indent=2)
    print(f"wrote {outp}.npz and {outp}_meta.json in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
