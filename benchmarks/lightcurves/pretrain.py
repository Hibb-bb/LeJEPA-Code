"""
Typical use::

    # Local PC_matches data (default --dataset pc/ZTF-ATLAS-ASASSN-isect,
    # read from $PC_MATCHES_ROOT or /projects/bfrf/data/PC_matches) needs no
    # token; the legacy hibb/ HF Hub datasets need
    export HF_TOKEN=hf_...    # or `hf auth login`

    # CPU smoke test (tiny, ~1 min): proves the pipeline end to end
    python pretrain.py --width 32 --depth 2 --epochs 1 \
        --max-objects 96 --batch-size 8 --n-slices 64 \
        --global-tokens 96 --local-tokens 48 --num-workers 0 --precision 32
"""

import argparse
import collections
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def resolve_hf_token(explicit=None):
    """Hugging Face token for the private datasets.

    Precedence: ``--hf-token`` flag > ``HF_TOKEN`` env var > the token cached
    by ``hf auth login`` (``~/.cache/huggingface/token``). Never hardcode a
    token in this file: the repository is public.
    """
    if explicit:
        return explicit
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    from huggingface_hub import get_token

    return get_token()

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from loguru import logger

import stable_pretraining as spt
from stable_pretraining.backbone import (
    MLP,
    RoMAELightCurveBackbone,
    tokenize_lightcurves,
)
from stable_pretraining.callbacks import WitnessCallback
from stable_pretraining.methods import LeJEPALightCurve
from stable_pretraining.methods.lejepa import LeJEPA, LeJEPAOutput
from stable_pretraining.optim import apply_mup, mup_param_groups

# Datasets. ``hibb/...`` ids are private HF Hub repos (need HF_TOKEN);
# ``pc/<name>`` ids are the local PC_matches ``DatasetDict``s under
# --data-root (default PC_MATCHES_ROOT / /projects/bfrf/data/PC_matches),
# which ship their own unified train/validation/test split (a ``split``
# column, disjoint by gaia id) and the corrected ASAS-SN photometry.
PC_ROOT = Path(os.environ.get("PC_MATCHES_ROOT", "/projects/bfrf/data/PC_matches"))
LOCAL_PREFIX = "pc/"
INSTRUMENT_REGISTRY = {
    "pc/ZTF-ATLAS-ASASSN-isect": {
        "ZTF": {"g_ZTF": 472.0, "r_ZTF": 634.0, "i_ZTF": 789.0},
        "ATLAS": {"c_ATLAS": 530.0, "o_ATLAS": 680.0},
        "ASASSN": {"V_ASASSN": 550.0, "g_ASASSN": 480.0},
    },
    "pc/ZTF-ATLAS-ASASSN-CSS-LINEAR-PTF-isect": {
        "ZTF": {"g_ZTF": 472.0, "r_ZTF": 634.0, "i_ZTF": 789.0},
        "ATLAS": {"c_ATLAS": 530.0, "o_ATLAS": 680.0},
        "ASASSN": {"V_ASASSN": 550.0, "g_ASASSN": 480.0},
        "CSS": {"V_CSS": 550.0},
        # LINEAR is unfiltered (clear); PTF g / Mould R.
        "LINEAR": {"r_LINEAR": 600.0},
        "PTF": {"g_PTF": 480.0, "R_PTF": 658.0},
    },
    "hibb/TESS-ZTF-isect": {
        "TESS": {"TESS": 786.0},
        "ZTF": {"g_ZTF": 472.0, "r_ZTF": 634.0, "i_ZTF": 789.0},
    },
    "hibb/tess-ztf-atlas-asassn-isect": {
        "TESS": {"TESS": 786.0},
        "ZTF": {"g_ZTF": 472.0, "r_ZTF": 634.0, "i_ZTF": 789.0},
        "ATLAS": {"c_ATLAS": 530.0, "o_ATLAS": 680.0},
        "ASASSN": {"V_ASASSN": 550.0, "g_ASASSN": 480.0},
    },
    "hibb/tess-ztf-atlas-asassn-isect+css": {
        "TESS": {"TESS": 786.0},
        "ZTF": {"g_ZTF": 472.0, "r_ZTF": 634.0, "i_ZTF": 789.0},
        "ATLAS": {"c_ATLAS": 530.0, "o_ATLAS": 680.0},
        "ASASSN": {"V_ASASSN": 550.0, "g_ASASSN": 480.0},
        "CSS": {"V_CSS": 550.0},
    },
}
DATASET_SOURCES = {
    "hibb/tess-ztf-atlas-asassn-isect+css":
        ["hibb/tess-ztf-atlas-asassn-isect", "hibb/CSSxPC"],
}
DENSE_INSTRUMENTS = ("TESS",)  # minutes cadence: density-randomised views
_EXT_PER_EBV = {}  # r_v -> {band: A_lambda/E(B-V)}, lazily built

# Filter transmission percentiles (nm): band column -> {10: .., 50: .., 90: ..}
# from data/filter_percentiles.json (SVO profiles; "p% of the integrated
# transmission lies bluewards of lambda_p"). Keys there use survey-native
# names ("g_ASAS-SN"); ours drop the hyphen, so match on that.
FILTER_PERCENTILES_FILE = Path(__file__).resolve().parent / "data" / "filter_percentiles.json"
WAVE_PERCENTILES = (10, 50, 90)


def load_filter_percentiles(path=FILTER_PERCENTILES_FILE):
    """Return ``{band column: {percentile(int): wavelength nm}}``."""
    import json
    raw = json.loads(Path(path).read_text())["filters"]
    out = {}
    for name, d in raw.items():
        key = name.replace("-", "")
        out[key] = {int(p): float(a) / 10.0 for p, a in d["angstrom"].items()}
    return out


# Filled by configure_instruments(); module globals so every view function
# (which runs in DataLoader workers via fork) sees the same layout.
DATASET = "hibb/TESS-ZTF-isect"
BAND_NAMES = {}          # band id -> column name
BAND_WAVELENGTHS = {}    # band id -> effective wavelength nm (extinction + default position)
BAND_PERCENTILES = {}    # band id -> {10: nm, 50: nm, 90: nm}; missing bands are absent
INST_BANDS = {}          # instrument -> tuple of band ids
INSTRUMENTS = ()         # all instruments (eval / probes)
TRAIN_INSTRUMENTS = ()   # instruments usable for LeJEPA views
HOLDOUT = None
ALL_BANDS = ()
DENSE_BANDS = ()


def configure_instruments(dataset: str, holdout: str | None = None,
                          exclude: tuple = (), exclude_bands: tuple = ()):
    """Set the module-level band layout for ``dataset`` (see registry).

    ``exclude`` drops instruments entirely (their bands are never loaded);
    ``exclude_bands`` drops single band columns (e.g. ``i_ZTF``, the
    StarEmbed convention: ZTF g + r only); ``holdout`` keeps one instrument
    out of the LeJEPA views but probes/evaluates it. Bands within an
    instrument are ordered blue -> red so band position k means the same
    thing across surveys (handcrafted-feature concatenation relies on it).
    """
    global DATASET, BAND_NAMES, BAND_WAVELENGTHS, BAND_PERCENTILES, INST_BANDS
    global INSTRUMENTS, TRAIN_INSTRUMENTS, HOLDOUT, ALL_BANDS, DENSE_BANDS
    global BAND_WAVELENGTHS_AA
    reg = {k: dict(sorted(((c, w) for c, w in v.items() if c not in exclude_bands),
                          key=lambda cw: cw[1]))
           for k, v in INSTRUMENT_REGISTRY[dataset].items() if k not in exclude}
    unknown = set(exclude_bands) - {c for v in INSTRUMENT_REGISTRY[dataset].values() for c in v}
    if unknown:
        raise ValueError(f"--exclude-band {sorted(unknown)} not in {dataset}")
    reg = {k: v for k, v in reg.items() if v}
    if len(reg) < 2:
        raise ValueError(f"need >= 2 instruments after excluding {exclude}")
    DATASET = dataset
    BAND_NAMES, BAND_WAVELENGTHS, INST_BANDS = {}, {}, {}
    i = 0
    for inst, bands in reg.items():
        ids = []
        for col, wl in bands.items():
            BAND_NAMES[i] = col; BAND_WAVELENGTHS[i] = wl; ids.append(i); i += 1
        INST_BANDS[inst] = tuple(ids)
    INSTRUMENTS = tuple(reg)
    if holdout is not None and holdout not in reg:
        raise ValueError(f"holdout {holdout!r} not in {INSTRUMENTS}")
    HOLDOUT = holdout
    TRAIN_INSTRUMENTS = tuple(x for x in INSTRUMENTS if x != holdout)
    ALL_BANDS = tuple(BAND_NAMES)
    DENSE_BANDS = tuple(b for x in DENSE_INSTRUMENTS if x in INST_BANDS
                        for b in INST_BANDS[x])
    BAND_WAVELENGTHS_AA = {b: w * 10.0 for b, w in BAND_WAVELENGTHS.items()}
    pct = load_filter_percentiles()
    BAND_PERCENTILES = {b: pct[n] for b, n in BAND_NAMES.items() if n in pct}
    missing = [n for n in BAND_NAMES.values() if n not in pct]
    if missing:
        logger.warning(f"no filter percentiles for {missing}; percentile "
                       f"wavelength positions unavailable for those bands")
    _EXT_PER_EBV.clear()


# ---------------------------------------------------------------------------
# Wavelength encoding ablation (--wave-pos / --wave-ctx / --pool / ...).
# configure_wave() fills these from the CLI after configure_instruments():
#   BAND_POS  band id -> tuple of extra rotary position coords (after time)
#   BAND_FEAT band id -> tuple of per-band content features (extra value
#             channels, linearly embedded by the tubelet projection) or None
#   ROPE_BLOCKS head-dim layout for BlockRope, or None (= NDPRope equal split)
# ---------------------------------------------------------------------------
WAVE_POS_CHOICES = ("eff", "pct", "pct-nd", "width", "index", "none")
WAVE_CTX_CHOICES = ("none", "pct", "onehot")
WAVE = dict(pos="eff", ctx="none", pool="cls", scale=1.0, shuffle=False,
            time_frac=0.5, p_rope=0.75)
BAND_POS = {}
BAND_FEAT = None
ROPE_BLOCKS = None
N_POS_DIMS = 2
N_CHANNELS = 1


def add_wave_args(ap):
    """CLI flags shared by pretrain.py and downstream.py (must match the
    checkpoint)."""
    ap.add_argument("--wave-pos", choices=WAVE_POS_CHOICES, default="eff",
                    help="wavelength as rotary POSITION: eff = log effective "
                         "wavelength (1 axis); pct = log 10/50/90%% "
                         "transmission percentiles (3 axial axes); pct-nd = "
                         "same 3 coords encoded jointly with nD-RoPE "
                         "(simplex block); width = log lambda50 + log "
                         "(lambda90/lambda10) (2 axes); index = raw band id; "
                         "none = time only")
    ap.add_argument("--wave-ctx", choices=WAVE_CTX_CHOICES, default="none",
                    help="wavelength as token CONTENT (extra input channels): "
                         "pct = log percentile triple; onehot = band id "
                         "(non-transferable ceiling)")
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls",
                    help="cls: CLS sits at rotary position 0, so absolute "
                         "wavelength leaks in via CLS attention; mean: only "
                         "relative wavelength is visible")
    ap.add_argument("--wavelength-scale", type=float, default=1.0,
                    help="log-wavelength coords are divided by this")
    ap.add_argument("--shuffle-wave", action="store_true",
                    help="sanity control: permute the band -> wavelength "
                         "descriptor mapping (positions and features) with "
                         "a fixed seed; extinction physics unaffected")
    ap.add_argument("--time-frac", type=float, default=0.5,
                    help="fraction of head_dim rotated by time; the rest is "
                         "shared by the wavelength axes")
    ap.add_argument("--p-rope", type=float, default=0.75)
    ap.add_argument("--head-dim", type=int, default=HEAD_DIM)


def _even_split(total, k):
    """Split ``total`` (even) into ``k`` even parts, as equal as possible."""
    if k == 0:
        return []
    pairs = total // 2
    base, extra = divmod(pairs, k)
    return [2 * (base + (1 if i < extra else 0)) for i in range(k)]


def configure_wave(args):
    """Derive BAND_POS / BAND_FEAT / ROPE_BLOCKS from the CLI (after
    :func:`configure_instruments`)."""
    global BAND_POS, BAND_FEAT, ROPE_BLOCKS, N_POS_DIMS, N_CHANNELS, HEAD_DIM
    WAVE.update(pos=args.wave_pos, ctx=args.wave_ctx, pool=args.pool,
                scale=args.wavelength_scale, shuffle=args.shuffle_wave,
                time_frac=args.time_frac, p_rope=args.p_rope)
    HEAD_DIM = args.head_dim
    if HEAD_DIM % 2:
        raise ValueError(f"--head-dim must be even, got {HEAD_DIM}")
    bands = list(ALL_BANDS)
    eff = {b: BAND_WAVELENGTHS[b] for b in bands}
    pct = {b: BAND_PERCENTILES.get(b) for b in bands}
    if WAVE["shuffle"]:
        perm = np.random.default_rng(1000 + args.split_seed).permutation(len(bands))
        eff = {b: eff[bands[j]] for b, j in zip(bands, perm)}
        pct = {b: pct[bands[j]] for b, j in zip(bands, perm)}
        logger.warning(f"--shuffle-wave: band->descriptor permutation {perm.tolist()}")
    need_pct = WAVE["pos"] in ("pct", "pct-nd", "width") or WAVE["ctx"] == "pct"
    if need_pct and any(v is None for v in pct.values()):
        raise ValueError(f"filter percentiles missing for "
                         f"{[BAND_NAMES[b] for b, v in pct.items() if v is None]}")
    ref = float(np.exp(np.mean([np.log(w) for w in eff.values()])))
    sc = WAVE["scale"]
    lg = lambda lam: float(np.log(lam / ref) / sc)  # noqa: E731
    if WAVE["pos"] == "eff":
        BAND_POS = {b: (lg(eff[b]),) for b in bands}
    elif WAVE["pos"] in ("pct", "pct-nd"):
        BAND_POS = {b: tuple(lg(pct[b][q]) for q in WAVE_PERCENTILES) for b in bands}
    elif WAVE["pos"] == "width":
        BAND_POS = {b: (lg(pct[b][50]), float(np.log(pct[b][90] / pct[b][10]) / sc))
                    for b in bands}
    elif WAVE["pos"] == "index":
        BAND_POS = {b: (float(b),) for b in bands}
    else:
        BAND_POS = {b: () for b in bands}
    if WAVE["ctx"] == "pct":
        BAND_FEAT = {b: tuple(lg(pct[b][q]) for q in WAVE_PERCENTILES) for b in bands}
    elif WAVE["ctx"] == "onehot":
        BAND_FEAT = {b: tuple(1.0 if j == i else 0.0 for j in range(len(bands)))
                     for i, b in enumerate(bands)}
    else:
        BAND_FEAT = None
    n_wave = len(next(iter(BAND_POS.values())))
    N_POS_DIMS = 1 + n_wave
    N_CHANNELS = 1 + (len(next(iter(BAND_FEAT.values()))) if BAND_FEAT else 0)

    # Rotary layout: time keeps ~time_frac of the head channels, the
    # wavelength axes share the rest (equal axial slices, or one simplex
    # block whose size must be a multiple of 2*(n+1)).
    p = WAVE["p_rope"]
    if n_wave == 0:
        wave_dim = 0
    elif WAVE["pos"] == "pct-nd":
        unit = 2 * (n_wave + 1)
        wave_dim = max(unit, (int(round(HEAD_DIM * (1 - WAVE["time_frac"]))) // unit) * unit)
    else:
        wave_dim = 2 * (int(round(HEAD_DIM * (1 - WAVE["time_frac"]))) // 2)
        wave_dim = max(2 * n_wave, wave_dim)
    time_dim = HEAD_DIM - wave_dim
    if time_dim < 2:
        raise ValueError(f"head_dim {HEAD_DIM} too small for {n_wave} wavelength axes")
    blocks = [dict(kind="axial", axes=[0], dim=time_dim, p=p)]
    if WAVE["pos"] == "pct-nd":
        blocks.append(dict(kind="simplex", axes=list(range(1, N_POS_DIMS)),
                           dim=wave_dim, p=p, seed=args.seed))
    else:
        for i, d in enumerate(_even_split(wave_dim, n_wave)):
            blocks.append(dict(kind="axial", axes=[1 + i], dim=d, p=p))
    # Equal axial split == plain NDPRope (keeps old checkpoint keys valid).
    if all(b["kind"] == "axial" for b in blocks) and len({b["dim"] for b in blocks}) == 1:
        ROPE_BLOCKS = None
    else:
        ROPE_BLOCKS = blocks
    logger.info(f"wave encoding: pos={WAVE['pos']} ctx={WAVE['ctx']} "
                f"pool={WAVE['pool']} n_pos_dims={N_POS_DIMS} "
                f"n_channels={N_CHANNELS} head_dim={HEAD_DIM} rope="
                f"{[(b['kind'], b['axes'], b['dim']) for b in blocks]}")


def _inst_times(record, inst):
    return np.concatenate(
        [record["bands"][b][0] for b in INST_BANDS[inst] if b in record["bands"]]
        or [np.empty(0)]
    )


def _record_instruments(record, min_obs=1, train_only=False):
    """Instruments this object has with >= ``min_obs`` observations."""
    pool = TRAIN_INSTRUMENTS if train_only else INSTRUMENTS
    return [x for x in pool if _inst_times(record, x).size >= min_obs]

# ---------------------------------------------------------------------------
# Reference rung: the (width, proj_dim, batch_size) at which lambda_ref and
# base_lr were tuned (CLI-overridable via --ref-*).
# ---------------------------------------------------------------------------
REF_WIDTH = 128
REF_PROJ_DIM = 32
REF_BATCH_SIZE = 128
DEFAULT_WIDTH = 360   # 6 heads x 60; default depth 6 (RoMAE-base 720x12 OOMs at batch 128)

BASE_FANIN = 256  # mu-P normalization: fan_in at which lr mult == 1
# RoMAE head_dim: fixed across the ladder (mu-P grows the head COUNT).
# 60 = 720 / 12 heads (RoMAE-base). Must be even; the rotary layout over the
# position axes is derived from it in configure_wave() (--head-dim overrides,
# e.g. 32 to load pre-2026-09 checkpoints).
HEAD_DIM = 60


def master_lambda(
    lamb_ref: float,
    width: int,
    proj_dim: int,
    batch_size: int,
    ref_width: int = REF_WIDTH,
    ref_proj_dim: int = REF_PROJ_DIM,
    ref_batch_size: int = REF_BATCH_SIZE,
) -> float:
    """lambda(n, d, B) = lambda_ref * sqrt((B*d/n) / (B*d/n)_ref)."""
    ratio = (batch_size * proj_dim / width) / (
        ref_batch_size * ref_proj_dim / ref_width
    )
    return lamb_ref * math.sqrt(ratio)


# ---------------------------------------------------------------------------
# View generation (the light-curve-specific extension point)
# ---------------------------------------------------------------------------
@dataclass
class ViewConfig:
    """Knobs for :func:`make_views`. All time quantities are in days (MJD).

    Augmentations (each applied independently per view with probability
    ``p_*``; all operate in *magnitude* space before normalisation):

    - ``resample``: uncertainty resampling ``mag + eps * mag_unc``.
    - ``global_mode``: how the two global views are formed.
      ``"instrument"`` = TESS-only vs ZTF-only over a shared window;
      ``"concat"`` = the two *chimeras* ``TESS[lo,m) + ZTF[m,hi)`` and
      ``ZTF[lo,m) + TESS[m,hi)`` at a random split ``m`` (same star, same
      span, each half from a different telescope — must embed identically);
      ``"both"`` = all four globals.
    - distance mimic: a single offset ``dm ~ U(-distance_mag, distance_mag)``
      added to every band (a pure distance change is achromatic in mag).
    - extinction jitter: ``dE(B-V) ~ N(0, ebv_jitter)`` (either sign — extra
      reddening or de-reddening) converted per band to ``A_lambda`` with the
      CCM89 law (``extinction`` package), so it is *chromatic*: blue bands
      shift more than red ones.
    - period shift: with the catalog ``period`` P, the view's window is moved
      by ``k * P`` for a non-zero integer ``k`` in ``[-max_period_shift,
      max_period_shift]`` (falls back to ``k = 0`` when the shifted window is
      empty or off the data). A periodic star is in the same phase there, so
      the model sees a *different cycle* of the same star.
    - window length: the shared global window is ``window_days`` long, or,
      with ``window_days_max > 0``, Gaussian around the midpoint of
      ``[window_days, window_days_max]`` with std a quarter of the span,
      clipped to that interval (N(1000, 250) days for the default 500-1500),
      so most windows sit at the scale of the downstream probe view (capped
      at ``window_days_max`` too, keeping train and eval aligned). A view
      holding more observations than its token budget is cut at the tail
      (max input length) with ``over_budget="tail"``, or cadence-thinned by
      random subsampling with ``"random"``.
    - sparsity filter: a sampled window must hold >= ``min_window_obs``
      observations per participating instrument (globals) or of the view's
      bands (probe span). Too-sparse draws are rejected and redrawn (with a
      fresh length, so sparse cadences are pushed toward longer windows);
      after ``window_tries`` failures the densest candidate seen is used, so
      no object is ever dropped.
    - TESS density: ``tess_density="random"`` draws every TESS-only view's
      token count log-uniformly in ``[min_tokens, budget]`` so a dense 2-min
      cadence is no longer a free "this is TESS" label for the encoder.
    - observation dropout: with prob. ``p_drop`` a fraction ``f ~ U(0,
      max_drop_frac)`` of the view's observations is removed at random;
      additionally ``n_gaps`` contiguous windows of ``gap_frac`` of the span
      are blanked (the model has to bridge missing cycles). Random dropout is
      close to a no-op when the view exceeds its token budget (it is already
      subsampled), so gaps are the one that actually removes information.

    Normalisation uses per-object statistics *fixed at load time* (see
    ``_to_record``) so the brightness augmentations above survive it.
    ``norm="band"`` (default) subtracts each band's own median (removes the
    TESS/ZTF zero-point mismatch and the star's colour) and divides by one
    pooled scale (keeps cross-band amplitude ratios); ``"band-global"``
    centres the same way but divides by a single *dataset-wide* scale
    (the median of the per-object scales, pooled in :func:`load_records`),
    so the star's absolute variability amplitude — one of the most
    class-discriminative quantities — survives normalisation instead of
    being divided out per object; ``"object"`` uses a single pooled centre;
    ``"view"`` standardises each view by its own mean/std, which silently
    cancels the distance and (most of the) extinction augmentations —
    ablation only.
    """

    window_days: float = 500.0      # min shared cross-instrument window length
    window_days_max: float = 0.0    # >0: length ~ N(mid, span/4), clipped
    window_tries: int = 16          # attempts to find a both-populated window
    min_window_obs: int = 200       # reject sparser sampled windows (see below)
    tess_density: str = "full"      # "full" | "random": TESS token count ~ log-U
    local_window_frac: float = 1.0  # local window = frac of the global window
    n_local: int = 4                # number of local (multi-crop) views
    global_tokens: int = 512        # token budget per global view
    local_tokens: int = 256         # token budget per local view
    eval_tokens: int = 512          # token budget for the canonical eval view
    over_budget: str = "tail"       # "tail" = cut in time | "random" subsample
    min_tokens: int = 8             # below this a view falls back to full range
    time_scale: float = 1.0         # positions = (t - origin) / time_scale
    resample: bool = True           # uncertainty resampling on/off
    global_mode: str = "instrument"  # "instrument" | "concat" | "both"
    local_mode: str = "all"         # "all" | "instrument" | "mixed"
    p_distance: float = 0.0         # prob. of a distance (achromatic) offset
    distance_mag: float = 1.0       # |dm| upper bound in mag
    p_extinction: float = 0.0       # prob. of an extinction jitter
    ebv_jitter: float = 0.1         # std of dE(B-V) in mag
    r_v: float = 3.1                # CCM89 R_V
    p_period_shift: float = 0.0     # prob. of shifting a window by k periods
    max_period_shift: int = 2       # |k| <= this
    p_drop: float = 0.0             # prob. of randomly dropping observations
    max_drop_frac: float = 0.5      # drop fraction ~ U(0, max_drop_frac)
    n_gaps: int = 0                 # contiguous time gaps blanked per view
    gap_frac: float = 0.1           # each gap = frac of the view's time span
    norm: str = "band"              # "band" | "object" | "view"

    @property
    def n_global(self) -> int:
        return 4 if self.global_mode == "both" else 2


# extinction.ccm89 wants Angstrom; BAND_WAVELENGTHS is in nm (set by
# configure_instruments).
BAND_WAVELENGTHS_AA = {}


def _band_extinction_per_ebv(band_ids, r_v: float) -> dict:
    """``A_lambda / E(B-V)`` for each band under CCM89 (``A_V = R_V E(B-V)``)."""
    import extinction  # optional dependency, only needed for the jitter

    waves = np.array([BAND_WAVELENGTHS_AA[b] for b in band_ids], dtype=np.float64)
    a_lambda = extinction.ccm89(waves, a_v=r_v, r_v=r_v)  # A_lambda at E(B-V)=1
    return dict(zip(band_ids, a_lambda.astype(np.float32)))




def _gather(record: dict, band_ids, lo: float, hi: float):
    """Concatenate ``(t, mag, mag_unc, band)`` from ``band_ids`` in ``[lo, hi)``."""
    ts, vs, us, bs = [], [], [], []
    for b in band_ids:
        arr = record["bands"].get(b)
        if arr is None:
            continue
        t, v, u = arr
        m = (t >= lo) & (t < hi)
        if m.any():
            ts.append(t[m]); vs.append(v[m]); us.append(u[m])
            bs.append(np.full(int(m.sum()), b, dtype=np.int64))
    if not ts:
        return (np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
    return (
        np.concatenate(ts), np.concatenate(vs),
        np.concatenate(us), np.concatenate(bs),
    )


def _gather_segments(record: dict, segments):
    """Union of :func:`_gather` over ``[(band_ids, lo, hi), ...]``."""
    parts = [_gather(record, b, lo, hi) for (b, lo, hi) in segments]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def _count(record: dict, band_ids, lo: float, hi: float) -> int:
    n = 0
    for b in band_ids:
        arr = record["bands"].get(b)
        if arr is not None:
            n += int(((arr[0] >= lo) & (arr[0] < hi)).sum())
    return n


def _shift_by_periods(record, band_ids, lo, hi, cfg, rng):
    """Move ``[lo, hi)`` by ``k * period`` (random non-zero integer ``k``).

    Returns the original window when the object has no usable period, the
    draw fails ``p_period_shift``, or no candidate ``k`` leaves at least
    ``min_tokens`` observations of ``band_ids`` inside the shifted window.
    """
    period = record.get("period")
    if not period or cfg.p_period_shift <= 0 or rng.random() >= cfg.p_period_shift:
        return lo, hi
    ks = [k for k in range(-cfg.max_period_shift, cfg.max_period_shift + 1) if k]
    rng.shuffle(ks)
    for k in ks:
        s, e = lo + k * period, hi + k * period
        if _count(record, band_ids, s, e) >= cfg.min_tokens:
            return s, e
    return lo, hi


def _sample_shared_window(record: dict, cfg: ViewConfig, rng: np.random.Generator):
    """Pick a window where two (random) training instruments both observe.

    Cross-instrument invariance is only meaningful over a shared time span,
    but instruments overlap only partially (e.g. TESS sectors vs multi-year
    ZTF). Instrument pairs are tried in random order; inside each pair's
    time-overlap we look for a window holding >= ``min_window_obs`` of
    each; sparser draws are rejected and redrawn with a fresh length (so
    sparse cadences drift toward longer windows). After ``window_tries``
    failures the densest candidate seen is used. Pairs with NO usable
    co-temporal overlap (e.g. CSS 2005-13 vs ATLAS 2015+) become cross-era
    pairs: each instrument gets a density-checked window in its own span
    and the second is phase-anchored to the first (see ``_phase_anchor``).
    Returns ``(lo, hi, pair, wins)``: the pooled span, the chosen pair and
    a per-instrument window dict ``{inst: (lo_i, hi_i)}``.
    """
    insts = _record_instruments(record, cfg.min_tokens, train_only=True)
    if len(insts) == 1:
        # Single-survey star (--min-train-instruments 1): both globals come
        # from the same instrument; _global_instrument_views draws the
        # second one its own window so the pair is different-epoch.
        lo, hi = _sample_instrument_window(record, insts[0], cfg, rng)
        x = insts[0]
        return lo, hi, (x, x), {x: (lo, hi)}
    pairs = [(a, b) for i, a in enumerate(insts) for b in insts[i + 1:]]
    rng.shuffle(pairs)
    need = max(cfg.min_tokens, cfg.min_window_obs)
    best, best_n = None, -1
    for a, b in pairs:
        ta, tb = _inst_times(record, a), _inst_times(record, b)
        lo = max(ta.min(), tb.min())
        hi = min(ta.max(), tb.max())
        if hi - lo < cfg.window_days:
            continue
        for _ in range(cfg.window_tries):
            length = _sample_window_length(cfg, hi - lo, rng)
            start = rng.uniform(lo, hi - length)
            end = start + length
            n_a = int(((ta >= start) & (ta < end)).sum())
            n_b = int(((tb >= start) & (tb < end)).sum())
            n = min(n_a, n_b)
            if n >= need:
                return start, end, (a, b), {a: (start, end), b: (start, end)}
            if n > best_n:
                best, best_n = (start, end, (a, b)), n
    if best is not None:
        start, end, (a, b) = best
        return start, end, (a, b), {a: (start, end), b: (start, end)}
    # No co-temporal overlap at all: cross-era pair. Statistically valid
    # because views only need to be conditionally independent given the
    # star's latent state; for (cyclo)stationary sources, windows from
    # different eras are exchangeable once phase-aligned modulo the period.
    a, b = pairs[0]
    wa = _sample_instrument_window(record, a, cfg, rng)
    wb = _phase_anchor(record, wa[0], *_sample_instrument_window(record, b, cfg, rng))
    return min(wa[0], wb[0]), max(wa[1], wb[1]), (a, b), {a: wa, b: wb}


def _phase_anchor(record, t_ref, lo, hi):
    """Snap window start ``lo`` onto the phase of ``t_ref`` modulo the period.

    Cross-era positive pairs are exchangeable under a cyclostationary model
    only up to phase; shifting by ``round((lo - t_ref)/P) * P`` aligns them.
    The residual phase slip is ``dt * sigma_P / P^2`` cycles — negligible at
    catalog period precision even across a decade. No-op without a period
    (the pair then remains a plain "same star, different epoch" positive).
    """
    p = record.get("period")
    if not p:
        return lo, hi
    lo2 = t_ref + round((lo - t_ref) / p) * p
    return lo2, lo2 + (hi - lo)


def _sample_instrument_window(record, inst, cfg, rng):
    """Density-checked ``window_days``-scale window on ONE instrument's span.

    Same acceptance rule as the pair sampler (>= ``min_window_obs`` of the
    instrument, densest candidate after ``window_tries`` failures, full span
    when it is shorter than ``window_days``).
    """
    ta = _inst_times(record, inst)
    lo, hi = float(ta.min()), float(ta.max())
    if hi - lo < cfg.window_days:
        return lo, hi + 1.0
    need = max(cfg.min_tokens, cfg.min_window_obs)
    best, best_n = (lo, hi + 1.0), -1
    for _ in range(cfg.window_tries):
        length = _sample_window_length(cfg, hi - lo, rng)
        start = rng.uniform(lo, hi - length)
        n = int(((ta >= start) & (ta < start + length)).sum())
        if n >= need:
            return start, start + length
        if n > best_n:
            best, best_n = (start, start + length), n
    return best


def _sample_window_length(cfg, available, rng):
    """Fixed ``window_days``, or truncated-Gaussian in ``[window_days, max]``.

    Mean = the interval midpoint (1000 d for the default 500-1500), std = a
    quarter of the span, clipped to the interval and to the pair's overlap.
    """
    hi = min(cfg.window_days_max, available) if cfg.window_days_max > 0 else 0
    if hi <= cfg.window_days:
        return cfg.window_days
    mean = 0.5 * (cfg.window_days + cfg.window_days_max)
    std = 0.25 * (cfg.window_days_max - cfg.window_days)
    return float(np.clip(rng.normal(mean, std), cfg.window_days, hi))


def _augment_mags(v, u, b, cfg, rng):
    """Magnitude-space augmentations: resampling, distance offset, extinction."""
    if cfg.resample:
        v = v + rng.standard_normal(v.shape).astype(v.dtype) * u
    if cfg.p_distance > 0 and rng.random() < cfg.p_distance:
        v = v + np.float32(rng.uniform(-cfg.distance_mag, cfg.distance_mag))
    if cfg.p_extinction > 0 and rng.random() < cfg.p_extinction:
        table = _EXT_PER_EBV.get(cfg.r_v)
        if table is None:
            table = _EXT_PER_EBV[cfg.r_v] = _band_extinction_per_ebv(ALL_BANDS, cfg.r_v)
        d_ebv = np.float32(rng.normal(0.0, cfg.ebv_jitter))
        a_lambda = np.array([table[int(x)] for x in b], dtype=np.float32)
        v = v + d_ebv * a_lambda
    return v


def _drop_observations(t, v, b, cfg, rng):
    """Observation dropout: random per-point drop + contiguous time gaps.

    Never drops below ``cfg.min_tokens`` observations.
    """
    keep = np.ones(t.size, dtype=bool)
    if cfg.p_drop > 0 and rng.random() < cfg.p_drop:
        frac = rng.uniform(0.0, cfg.max_drop_frac)
        keep &= rng.random(t.size) >= frac
    if cfg.n_gaps > 0 and t.size > 1:
        lo, hi = float(t.min()), float(t.max())
        gap = cfg.gap_frac * (hi - lo)
        for _ in range(cfg.n_gaps):
            s = rng.uniform(lo, max(lo, hi - gap))
            keep &= ~((t >= s) & (t < s + gap))
    if keep.sum() < cfg.min_tokens:
        return t, v, b
    return t[keep], v[keep], b[keep]


def _build_view(record, segments, origin, budget, cfg, rng, augment=True):
    """One view: gather -> augment (mag space) -> subsample -> standardize.

    ``segments`` is ``[(band_ids, lo, hi), ...]``; a plain view has one
    segment, a concat (chimera) view has two with different band groups.
    Returns ``(t, v, b)`` as 1-D float/long tensors with time re-based to
    ``origin`` (stable rotary positions). Falls back to the union of the
    segments' band groups over the full range when the window is too sparse,
    so the returned view is never empty as long as those bands have any
    observations for this object.
    """
    t, v, u, b = _gather_segments(record, segments)
    if t.size < cfg.min_tokens:
        all_bands = tuple(sorted({x for (bands, _, _) in segments for x in bands}))
        t, v, u, b = _gather(record, all_bands, -np.inf, np.inf)
    if t.size == 0:
        # Should not happen after the load-time filter; guard defensively.
        return (torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.long))

    if augment:
        v = _augment_mags(v, u, b, cfg, rng)
        t, v, b = _drop_observations(t, v, b, cfg, rng)

    # TESS-only view: randomise the token count so cadence/density is not an
    # instrument tag (applied to eval views too, deterministically via rng).
    if cfg.tess_density == "random" and DENSE_BANDS and \
            set(np.unique(b).tolist()) <= set(DENSE_BANDS):
        budget = int(np.exp(rng.uniform(np.log(cfg.min_tokens), np.log(budget))))
        budget = max(cfg.min_tokens, budget)

    if t.size > budget:
        if cfg.over_budget == "tail":
            # Max input length: keep the earliest ``budget`` observations.
            idx = np.sort(np.argsort(t, kind="stable")[:budget])
        else:  # "random": thin the cadence, keep the full span
            idx = np.sort(rng.choice(t.size, size=budget, replace=False))
        t, v, b = t[idx], v[idx], b[idx]

    if cfg.norm in ("band", "band-global"):
        _, _, centers, scale = record["norm"]
        center = np.array([centers.get(int(x), 0.0) for x in b], dtype=v.dtype)
        v = (v - center) / scale
    elif cfg.norm == "object":
        center, scale, _, _ = record["norm"]
        v = (v - center) / scale
    else:
        v = (v - v.mean()) / (v.std() + 1e-8)
    t = (t - origin) / cfg.time_scale
    return (
        torch.from_numpy(np.ascontiguousarray(t)).float(),
        torch.from_numpy(np.ascontiguousarray(v)).float(),
        torch.from_numpy(np.ascontiguousarray(b)).long(),
    )


def _global_instrument_views(record, wins, cfg, rng, pair):
    """Same star, two observatories: ``a``-only and ``b``-only globals.

    ``wins`` maps each instrument to its window — one shared window for a
    co-temporal pair, per-era (phase-anchored) windows for a cross-era
    pair. For a same-instrument pair (single-survey star) the second global
    gets its own independently placed window, so the positive pair is
    "same star, different epoch" instead of two copies of one window.
    """
    views = []
    for k, name in enumerate(pair):
        lo, hi = wins[name]
        if k == 1 and pair[0] == pair[1]:
            lo, hi = _sample_instrument_window(record, name, cfg, rng)
        bands = INST_BANDS[name]
        s, e = _shift_by_periods(record, bands, lo, hi, cfg, rng)
        views.append(_build_view(record, [(bands, s, e)], s, cfg.global_tokens, cfg, rng))
    return views


def _global_concat_views(record, lo, hi, cfg, rng, pair):
    """Light-curve concat: ``A[lo,m) + B[m,hi)`` and ``B[lo,m) + A[m,hi)``.

    ``(A, B)`` = ``pair``, split ``m`` drawn uniformly from the middle
    half of the window so both halves are non-trivial. Both chimeras cover
    the same star over the same span with each half from a different
    telescope, so their embeddings should coincide.
    """
    m = rng.uniform(lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo))
    views = []
    A, B = INST_BANDS[pair[0]], INST_BANDS[pair[1]]
    for first, second in ((A, B), (B, A)):
        s, e = _shift_by_periods(record, ALL_BANDS, lo, hi, cfg, rng)
        d = s - lo
        segs = [(first, s, m + d), (second, m + d, e)]
        views.append(_build_view(record, segs, s, cfg.global_tokens, cfg, rng))
    return views


def make_views(record: dict, cfg: ViewConfig, rng: np.random.Generator):
    """Generate the LeJEPA view set for one object.

    Returns ``(views, inst_ids)``: ``views = [global_0, ...,
    global_{n_global-1}, local_0, ...]`` with each entry a ``(t, v, b)``
    triple, and ``inst_ids`` each view's ``TRAIN_INSTRUMENTS`` index (-1 for
    mixed/all-band views), consumed by the ``--adv-weight`` GRL head. Globals are the cross-instrument pair
    (``global_mode="instrument"``), the concat/chimera pair (``"concat"``) or
    both (four globals); locals are shorter all-band crops. Every view is
    independently uncertainty-resampled and (with the configured
    probabilities) distance-shifted, extinction-jittered and period-shifted —
    see :class:`ViewConfig`.
    """
    lo, hi, pair, wins = _sample_shared_window(record, cfg, rng)
    views, inst_ids = [], []
    if cfg.global_mode in ("instrument", "both"):
        views += _global_instrument_views(record, wins, cfg, rng, pair)
        inst_ids += [_inst_idx(pair[0]), _inst_idx(pair[1])]
    if cfg.global_mode in ("concat", "both"):
        views += _global_concat_views(record, lo, hi, cfg, rng, pair)
        inst_ids += [-1, -1]  # chimeras mix both telescopes
    if not views:
        raise ValueError(f"unknown global_mode {cfg.global_mode!r}")

    # Which telescope(s) the locals see, decided once per view set:
    # "all" = all bands; "instrument" = half TESS / half ZTF (shuffled);
    # "mixed" = a 50/50 coin between those two, so systematics are also
    # removed at the local scale rather than only through the two globals.
    single = cfg.local_mode == "instrument" or (
        cfg.local_mode == "mixed" and rng.random() < 0.5
    )
    if single and cfg.n_local > 0:
        half = cfg.n_local // 2
        srcs = [pair[0]] * half + [pair[1]] * (cfg.n_local - half)
        rng.shuffle(srcs)
        local_bands = [(INST_BANDS[x], _inst_idx(x), x) for x in srcs]
    else:
        local_bands = [(ALL_BANDS, -1, None)] * cfg.n_local
    for bands, ii, src in local_bands:
        # Single-instrument locals crop inside their OWN instrument's global
        # window (matters for cross-era pairs, where the pooled span covers
        # a gap that neither instrument observed).
        wlo, whi = wins.get(src, (lo, hi)) if src else (lo, hi)
        sub_i = max(cfg.local_window_frac * (whi - wlo), 1e-3)
        s = rng.uniform(wlo, max(wlo, whi - sub_i))
        s, e = _shift_by_periods(record, bands, s, s + sub_i, cfg, rng)
        views.append(
            _build_view(record, [(bands, s, e)], s, cfg.local_tokens, cfg, rng)
        )
        inst_ids.append(ii)
    return views, inst_ids


def _inst_idx(name: str) -> int:
    """Index of ``name`` in ``TRAIN_INSTRUMENTS`` (GRL head label)."""
    return TRAIN_INSTRUMENTS.index(name)


def make_eval_view(record: dict, cfg: ViewConfig, rng: np.random.Generator,
                   bands=ALL_BANDS, augment=False):
    """Canonical view for probing: full span over ``bands``.

    Un-augmented (and index-seeded) on the val set; on the train set it is
    augmented like any other view so the probes / supervised baseline see the
    same nuisances as the LeJEPA views.
    """
    all_t = np.concatenate([record["bands"][b][0] for b in record["bands"]])
    lo, hi = float(all_t.min()), float(all_t.max()) + 1.0
    if cfg.window_days_max > 0 and hi - lo > cfg.window_days_max:
        # Cap the probe view to the longest training window so the
        # downstream/eval views live on the same scale as the LeJEPA views.
        # Anchor it on the cross-instrument overlap when there is one.
        lo, hi = _capped_span(record, cfg, rng, bands)
    return _build_view(
        record, [(bands, lo, hi)], lo, cfg.eval_tokens, cfg, rng, augment=augment
    )


def _capped_span(record, cfg, rng, bands=None):
    """``window_days_max``-long span on the instruments' common overlap.

    Placements are drawn until one holds >= ``min_window_obs`` observations
    of ``bands`` (default: all of the object's bands); after
    ``window_tries`` sparse draws the densest candidate is used, so the
    probe/eval views are filtered away from empty corners of the light
    curve just like the training windows.
    """
    L = cfg.window_days_max
    all_t = np.concatenate([record["bands"][b][0] for b in record["bands"]])
    # Time-overlap of every instrument this object has (all of them, so the
    # held-out instrument's probe view shares the span with the others).
    ts = [_inst_times(record, x) for x in _record_instruments(record)]
    lo = max(t.min() for t in ts); hi = min(t.max() for t in ts)
    if hi <= lo:  # no common overlap: use the pooled range
        lo, hi = float(all_t.min()), float(all_t.max())
    if hi - lo <= L:
        mid = 0.5 * (lo + hi)  # overlap fits: centre the span on it
        return float(mid - L / 2), float(mid + L / 2)
    count_bands = tuple(bands) if bands else tuple(record["bands"])
    best, best_n = lo, -1
    for _ in range(cfg.window_tries):
        start = rng.uniform(lo, hi - L)
        n = _count(record, count_bands, start, start + L)
        if n >= cfg.min_window_obs:
            return float(start), float(start + L)
        if n > best_n:
            best, best_n = start, n
    return float(best), float(best + L)


def make_probe_views(record, cfg, rng, augment):
    """``{"joint": view, <inst>: view, ...}, valid`` probe views.

    ``valid`` is a bool per instrument (in ``INSTRUMENTS`` order). Views of
    missing instruments are placeholders (never used: masked by ``valid``).
    """
    views = {"joint": make_eval_view(record, cfg, rng, ALL_BANDS, augment)}
    valid = []
    for inst in INSTRUMENTS:
        has = _has_bands(record, INST_BANDS[inst])
        valid.append(has)
        views[inst] = (make_eval_view(record, cfg, rng, INST_BANDS[inst], augment)
                       if has else views["joint"])
    return views, tuple(valid)


def _has_bands(record: dict, bands) -> bool:
    return any(b in record["bands"] for b in bands)


# ---------------------------------------------------------------------------
# Dataset + collation
# ---------------------------------------------------------------------------
class LightCurveDataset(torch.utils.data.Dataset):
    """Yields ``(((probe_views, valid), lejepa_views), label)`` per object.

    Train items use fresh entropy and augmented probe views; val items are
    seeded by index (same views every epoch) and un-augmented probe views.
    ``lejepa_views`` is ``None`` when ``with_views=False`` (baseline modes
    that never run the LeJEPA loss).
    """

    def __init__(self, records, cfg: ViewConfig, train: bool, seed: int = 0,
                 with_views: bool = True):
        self.records = records
        self.cfg = cfg
        self.train = train
        self.seed = seed
        self.with_views = with_views

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        # Train: fresh entropy per sample (stochastic views are the
        # augmentation). Val: seeded by index so losses/metrics are stable.
        rng = np.random.default_rng() if self.train else \
            np.random.default_rng(self.seed + i)
        probe = make_probe_views(rec, self.cfg, rng, augment=self.train)
        views, view_inst = (make_views(rec, self.cfg, rng)
                            if self.with_views else (None, None))
        med = rec.get("inst_med_mag", {})
        faint = tuple(
            x in FAINT_MAG and med.get(x, -np.inf) > FAINT_MAG[x]
            for x in INSTRUMENTS
        )
        return (probe, views, view_inst, faint, rec.get("uid", i)), rec["label"]


def _tokenize_batch(view_list):
    """Tokenize one view across a batch -> (values, positions, pad_mask)."""
    times = [t for (t, _, _) in view_list]
    values = [v for (_, v, _) in view_list]
    bands = [b for (_, _, b) in view_list]
    if not BAND_POS:  # configure_wave() not called (legacy analysis scripts)
        return tokenize_lightcurves(times, values, bands,
                                    band_wavelengths=BAND_WAVELENGTHS)
    return tokenize_lightcurves(
        times, values, bands, band_positions=BAND_POS, band_features=BAND_FEAT
    )


def collate(batch, n_global: int = 2):
    """List[((probe, views), label)] -> dict of tokenized view triples.

    Keys: ``probe_joint`` / ``probe_<inst>`` (capped-span views),
    ``inst_valid`` [N, n_inst], ``global_i`` / ``local_i`` (LeJEPA views, if
    present) and ``label``.
    """
    items, labels = zip(*batch)
    probes = [it[0][0] for it in items]
    out = {
        f"probe_{k}": _tokenize_batch([p[k] for p in probes]) for k in probes[0]
    }
    out["inst_valid"] = torch.tensor([it[0][1] for it in items], dtype=torch.bool)
    out["inst_faint"] = torch.tensor([it[3] for it in items], dtype=torch.bool)
    out["label"] = torch.tensor(labels, dtype=torch.long)
    # Dataset-level star index: with --oversample-paired the same star can
    # sit twice in a batch, and a contrastive loss must treat those rows as
    # positives (see pretrain_contrastive.py), not as false negatives.
    out["star_id"] = torch.tensor([it[4] for it in items], dtype=torch.long)
    if items[0][1] is not None:
        n_views = len(items[0][1])
        for j in range(n_views):
            key = f"global_{j}" if j < n_global else f"local_{j - n_global}"
            out[key] = _tokenize_batch([it[1][j] for it in items])
        # [N, n_views] TRAIN_INSTRUMENTS index per view (-1 = mixed bands),
        # consumed by the --adv-weight gradient-reversal head.
        out["view_inst"] = torch.tensor([it[2] for it in items],
                                        dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _to_record(ex, label_to_idx):
    """Convert a raw HF example into a compact per-object record."""
    bands = {}
    for idx, name in BAND_NAMES.items():
        band = ex["lightcurve"].get(name)
        if band is None or band.get("length", 0) == 0:
            continue
        t = np.asarray(band["mjd"], dtype=np.float64)
        v = np.asarray(band["mag"], dtype=np.float32)
        u = np.asarray(band["mag_unc"], dtype=np.float32)
        # Vega -> AB where the dataset provides the offset (ASAS-SN V).
        if band.get("mag_sys") == "Vega" and band.get("Vega-to-AB") is not None:
            v = v + np.float32(band["Vega-to-AB"])
        # Drop non-finite rows (missing mag / unc) and flagged epochs.
        good = np.isfinite(t) & np.isfinite(v) & np.isfinite(u)
        if band.get("clean") is not None and len(band["clean"]) == t.size:
            good &= np.asarray(band["clean"], dtype=bool)
        if good.sum() == 0:
            continue
        bands[idx] = (t[good], v[good], np.nan_to_num(u[good], nan=0.0))
    period = ex.get("period")
    period = float(period) if period is not None and np.isfinite(period) and period > 0 else None
    # Per-object normalisation fixed at load time so that view-level
    # brightness augmentations (distance, extinction) are not cancelled by a
    # per-view standardisation (see ViewConfig.norm).
    # norm = (pooled_center, pooled_scale, {band: center}, band_scale) where
    # band_scale is the pooled std of the per-band-centred residuals, so
    # --norm band keeps the physical amplitude ratios between bands.
    if bands:
        all_v = np.concatenate([b[1] for b in bands.values()])
        centers = {b: np.float32(np.median(v)) for b, (_, v, _) in bands.items()}
        resid = np.concatenate([v - centers[b] for b, (_, v, _) in bands.items()])
        norm = (
            np.float32(np.median(all_v)), np.float32(max(all_v.std(), 1e-2)),
            centers, np.float32(max(resid.std(), 1e-2)),
        )
    else:
        norm = (np.float32(0.0), np.float32(1.0), {}, np.float32(1.0))
    # Median observed magnitude per instrument (fainter = noisier; used by
    # the UMAP faint-star viz filter, see FAINT_MAG).
    med_mag = {}
    for inst, ids in INST_BANDS.items():
        v = [bands[b][1] for b in ids if b in bands]
        if v:
            med_mag[inst] = float(np.median(np.concatenate(v)))
    return {
        "label": label_to_idx[ex["class_str"]], "bands": bands,
        "period": period, "norm": norm, "inst_med_mag": med_mag,
        "gaia": ex.get("gaia_dr3_source_id"),
        # Local PC_matches datasets carry their own unified split (train /
        # validation / test) and a coarser superclass label.
        "split": ex.get("split"), "superclass": ex.get("superclass_str"),
    }


def _has_two_instruments(rec, min_obs, min_insts=2):
    """At least ``min_insts`` *training* instruments with >= ``min_obs`` obs."""
    return len(_record_instruments(rec, min_obs, train_only=True)) >= min_insts


def is_local_dataset(name: str = None) -> bool:
    """``pc/<name>`` = local PC_matches DatasetDict (own split column)."""
    return (name or DATASET).startswith(LOCAL_PREFIX)


def _iter_source(src, args, token):
    """Yield raw examples of one source: HF Hub ``train`` split, or every
    split of a local PC_matches DatasetDict (``ex["split"]`` says which).
    ``--max-objects`` caps the rows per (source, split) for smoke tests."""
    from datasets import load_dataset, load_from_disk

    if is_local_dataset(src):
        root = Path(getattr(args, "data_root", None) or PC_ROOT)
        path = root / src[len(LOCAL_PREFIX):]
        if not path.exists():
            raise FileNotFoundError(f"local dataset {src!r} not found at {path} "
                                    f"(set --data-root / PC_MATCHES_ROOT)")
        dd = load_from_disk(str(path))
        splits = [dd[k] for k in ("train", "validation", "test") if k in dd]
    else:
        if token is None:
            raise RuntimeError(
                "Set HF_TOKEN, run `hf auth login`, or pass --hf-token with "
                f"read access to {src} — it is a private dataset."
            )
        splits = [load_dataset(src, split="train", token=token)]
    for ds in splits:
        if args.max_objects > 0:
            ds = ds.select(range(min(args.max_objects, len(ds))))
        yield from ds


def load_records(args):
    """Load ``args.dataset`` into per-object records + a label map.

    HF Hub datasets need ``HF_TOKEN`` (private repos); local ``pc/`` datasets
    are read from --data-root. ``--max-objects`` keeps only the first N rows
    per split (smoke tests).
    """
    token = resolve_hf_token(args.hf_token) if not is_local_dataset() else None

    # Optional star blacklist (e.g. StarEmbed val/test/anom crossmatches):
    # one gaia_dr3_source_id per line; matching rows never become records.
    excl = set()
    excl_file = getattr(args, "exclude_stars_file", None)
    if excl_file:
        excl = {int(x) for x in Path(excl_file).read_text().split() if x.strip()}
        logger.info(f"star exclusion list: {len(excl)} gaia ids from {excl_file}")

    # A DATASET may be a union of several HF repos (see DATASET_SOURCES):
    # rows are merged by gaia_dr3_source_id, first source wins on metadata
    # (class_str/period), light-curve band columns are unioned.
    sources = DATASET_SOURCES.get(DATASET, [DATASET])
    merged, order, n_excl = {}, [], 0
    for src in sources:
        for j, ex in enumerate(_iter_source(src, args, token)):
            gid = ex.get("gaia_dr3_source_id")
            if gid in excl:
                n_excl += 1
                continue
            key = gid if gid else (src, j)
            cur = merged.get(key)
            if cur is None:
                merged[key] = dict(ex)
                order.append(key)
            else:
                cur["lightcurve"] = {
                    **cur["lightcurve"],
                    **{k: v for k, v in ex["lightcurve"].items() if v},
                }
    raw = [merged[k] for k in order]
    if excl:
        logger.info(f"excluded {n_excl} source rows via the star blacklist")

    classes = sorted({ex["class_str"] for ex in raw})
    label_to_idx = {c: i for i, c in enumerate(classes)}

    min_insts = getattr(args, "min_train_instruments", 2)
    records = []
    for ex in raw:
        rec = _to_record(ex, label_to_idx)
        if _has_two_instruments(rec, args.min_obs, min_insts):
            rec["uid"] = len(records)  # dataset-level star index (see collate)
            records.append(rec)

    # --norm band-global: one dataset-wide scale (median of the per-object
    # pooled residual stds) replaces each record's own, so absolute
    # variability amplitude survives normalisation.
    if getattr(args, "norm", "band") == "band-global" and records:
        g = np.float32(np.median([r["norm"][3] for r in records]))
        for r in records:
            c0, s0, centers, _ = r["norm"]
            r["norm"] = (c0, s0, centers, g)
        logger.info(f"band-global norm: dataset scale = {g:.4f} mag")

    logger.info(
        f"Loaded {len(records)}/{len(raw)} objects with >=2 training "
        f"instruments (>={args.min_obs} obs each; train={TRAIN_INSTRUMENTS}, "
        f"holdout={HOLDOUT}); "
        f"{len(classes)} classes: {classes}"
    )
    return records, label_to_idx


def stratified_split(records, val_frac, seed, test_frac=0.0):
    """Deterministic per-class stratified train/val/test index split.

    Fallback for when no split file is given (see :func:`split_records`).
    Returns ``(train_idx, val_idx, test_idx)``; ``test_idx`` is empty when
    ``test_frac == 0``.
    """
    rng = np.random.default_rng(seed)
    by_class = {}
    for i, r in enumerate(records):
        by_class.setdefault(r["label"], []).append(i)
    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in sorted(by_class.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_frac))
        n_test = int(round(len(idxs) * test_frac))
        val_idx.extend(idxs[:n_val].tolist())
        test_idx.extend(idxs[n_val:n_val + n_test].tolist())
        train_idx.extend(idxs[n_val + n_test:].tolist())
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def default_split_file(dataset, seed=0):
    """``data/splits/<dataset basename>_seed<seed>.json`` (built by
    ``data/make_splits.py``)."""
    return (Path(__file__).resolve().parent / "data" / "splits"
            / f"{dataset.split('/')[-1]}_seed{seed}.json")


def split_records(records, args):
    """Star-level train/val/test split of ``records`` -> index lists.

    The canonical split is a file of ``gaia_dr3_source_id`` lists written
    once by ``data/make_splits.py`` over the *whole* HF dataset (stratified
    by ``class_str``), so every run — whatever its ``--min-obs``,
    ``--holdout-instrument`` or exclusion filters — holds out the identical
    stars. ``--split-file`` selects it; the default path is
    :func:`default_split_file` for ``--dataset``/``--split-seed``. Records
    absent from the file (no gaia id, or a star the file predates) go to
    train and are counted in the log. With ``--split-file none`` the old
    on-the-fly stratified split is used (``--val-frac``/``--test-frac``).
    """
    if is_local_dataset():
        # The dataset's own split column is the canonical one; --split-file
        # is ignored for pc/ datasets.
        names = {"train": "train", "validation": "val", "val": "val",
                 "test": "test"}
        out = {"train": [], "val": [], "test": []}
        for i, r in enumerate(records):
            out[names[r["split"]]].append(i)
        logger.info(f"dataset split column: {len(out['train'])} train / "
                    f"{len(out['val'])} val / {len(out['test'])} test objects")
        return out["train"], out["val"], out["test"]
    sf = getattr(args, "split_file", None)
    if sf is None:
        sf = default_split_file(DATASET, args.split_seed)
    if str(sf).lower() != "none" and Path(sf).exists():
        import json
        d = json.loads(Path(sf).read_text())
        where = {}
        for name in ("train", "val", "test"):
            for g in d[name]:
                where[str(g)] = name
        out = {"train": [], "val": [], "test": []}
        n_unlisted = 0
        for i, r in enumerate(records):
            g = r.get("gaia")
            name = where.get(str(g)) if g else None
            if name is None:
                n_unlisted += 1
                name = "train"
            out[name].append(i)
        logger.info(f"split file {sf}: {len(out['train'])} train / "
                    f"{len(out['val'])} val / {len(out['test'])} test objects "
                    f"({n_unlisted} unlisted -> train)")
        return out["train"], out["val"], out["test"]
    if str(sf).lower() != "none":
        logger.warning(f"split file {sf} not found; falling back to an "
                       f"on-the-fly stratified split (run data/make_splits.py)")
    tr, va, te = stratified_split(records, args.val_frac, args.split_seed,
                                  getattr(args, "test_frac", 0.0))
    logger.info(f"stratified split: {len(tr)} train / {len(va)} val / "
                f"{len(te)} test objects")
    return tr, va, te


def build_data(args, records, cfg):
    # Test stars are never touched here: the encoder trains on train, its
    # online probes watch val; downstream.py reports val and test.
    train_idx, val_idx, _ = split_records(records, args)
    train_recs = [records[i] for i in train_idx]
    val_recs = [records[i] for i in val_idx]
    k = getattr(args, "oversample_paired", 1)
    if k > 1:
        paired = [r for r in train_recs
                  if _has_two_instruments(r, args.min_obs)]
        train_recs = train_recs + paired * (k - 1)
        logger.info(f"oversampled {len(paired)} multi-instrument stars x{k}: "
                    f"train epoch now {len(train_recs)} items")

    with_views = args.mode in VIEW_MODES
    train_ds = LightCurveDataset(train_recs, cfg, train=True, with_views=with_views)
    val_ds = LightCurveDataset(val_recs, cfg, train=False, seed=args.seed,
                               with_views=with_views)

    collate_fn = partial(collate, n_global=cfg.n_global)
    # View construction is CPU-bound; deep prefetch + pinned host memory keep
    # the GPU fed (prefetch_factor requires num_workers > 0).
    perf = dict(pin_memory=True, persistent_workers=args.num_workers > 0)
    if args.num_workers > 0:
        perf["prefetch_factor"] = 4
    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
            drop_last=True, shuffle=True, collate_fn=collate_fn, **perf,
        ),
        val=torch.utils.data.DataLoader(
            val_ds, batch_size=min(256, max(1, len(val_recs))),
            num_workers=args.num_workers, collate_fn=collate_fn, **perf,
        ),
    )


# ---------------------------------------------------------------------------
# Model + forward
# ---------------------------------------------------------------------------
# Modes whose train batches carry LeJEPA-style global/local views. Sibling
# scripts (pretrain_contrastive.py) register their mode here, in FORWARDS
# and in MODEL_BUILDERS (mode -> fn(args, n_classes) -> model).
VIEW_MODES = {"lejepa"}
FORWARDS = {}
MODEL_BUILDERS = {}


def build_model(width, proj_dim, lamb, n_slices, depth, mode="lejepa",
                n_classes=None, projector="identity"):
    """One ladder rung: RoMAE encoder at ``width`` + SIGReg projector.

    ``projector="identity"`` (default) drops the MLP head entirely: the
    prediction loss and SIGReg act *directly on the backbone embeddings*,
    so the effective projection dim is ``width`` and invariance can no
    longer hide in a throwaway head. ``"mlp"`` restores the BN+ReLU
    2048-2048-``proj_dim`` projector.

    ``mode="random"`` / ``"supervised"`` return the same encoder wrapped with
    a linear head instead (frozen in ``random``).
    """
    assert width % HEAD_DIM == 0, f"width must be a multiple of head_dim={HEAD_DIM}"
    nhead = width // HEAD_DIM

    backbone = RoMAELightCurveBackbone(
        encoder_kwargs=dict(d_model=width, nhead=nhead, depth=depth),
        tubelet_size=(1, 1, 1), n_channels=N_CHANNELS, n_pos_dims=N_POS_DIMS,
        p_rope_val=WAVE["p_rope"], rope_blocks=ROPE_BLOCKS, pool=WAVE["pool"],
    )
    if mode != "lejepa":
        model = apply_mup(
            SupervisedLightCurve(backbone, width, n_classes), base_fanin=BASE_FANIN
        )
        if mode == "random":
            model.backbone.requires_grad_(False)
        return model
    if projector == "identity":
        proj = nn.Identity()
    else:
        proj = MLP(
            in_channels=width,
            hidden_channels=[2048, 2048, proj_dim],
            norm_layer="batch_norm",
            activation_layer=nn.ReLU,
            inplace=True,
            dropout=0.0,
        )
    predictor = pred_inst = None
    if PREDICTOR_INST is not None:
        # Encoder -> predictor (this instrument's views only) -> projector.
        pred_inst = TRAIN_INSTRUMENTS.index(PREDICTOR_INST)
        predictor = MLP(
            in_channels=width, hidden_channels=[PREDICTOR_HIDDEN, width],
            norm_layer="batch_norm", activation_layer=nn.ReLU,
            inplace=True, dropout=0.0,
        )
    model = LeJEPALightCurve(
        backbone, projector=proj, lamb=lamb, n_slices=n_slices,
        predictor=predictor, predictor_inst=pred_inst,
    )
    if ADV_WEIGHT > 0:
        # Linear telescope discriminator fed through the gradient-reversal
        # layer (see _adversarial_loss); trains with the same optimizer.
        model.inst_head = nn.Linear(width, len(TRAIN_INSTRUMENTS))
    return apply_mup(model, base_fanin=BASE_FANIN)


def _val_losses(model, global_views, local_views, view_inst=None):
    """LeJEPA loss on a view set with the model in eval mode.

    ``LeJEPALightCurve.forward`` only computes the loss when ``self.training``,
    so validation calls its building block directly (BatchNorm in the
    projector/predictor uses running stats, which is what we want here).
    """
    return model.loss_on_views(global_views, local_views, view_inst)[:3]


TRANSFER_PROBES = True


def probe_names():
    """Linear-probe inputs: joint + one per instrument (+ ``<holdout>_fit``).

    For a held-out instrument ``emb_<holdout>`` is, at *train* time, the
    embedding of a random training instrument's view, so its probe never sees
    a held-out token and its eval score is *probe transfer*; ``<holdout>_fit``
    is the ordinary probe trained on the held-out embeddings.

    With ``TRANSFER_PROBES`` there is additionally one probe per ordered
    instrument pair, ``<A>_to_<B>``: trained on ``A`` embeddings, evaluated on
    ``B`` embeddings. ``eval/probe_<A>_to_<B>_f1`` vs ``eval/probe_<B>_f1``
    is the operational measure of cross-instrument invariance.
    """
    names = ["joint", *INSTRUMENTS]
    if HOLDOUT is not None:
        names.append(f"{HOLDOUT}_fit")
    if TRANSFER_PROBES:
        names += [f"{a}_to_{b}" for a in INSTRUMENTS for b in INSTRUMENTS if a != b]
    return names


@torch.no_grad()
def _probe_embeddings(backbone, batch, out, stage):
    """Detached capped-span embeddings for the linear probes + inst eval."""
    for name in ["joint", *INSTRUMENTS]:
        v, p, m = batch[f"probe_{name}"]
        out[f"emb_{name}"] = backbone(v, p, m).detach()
    if TRANSFER_PROBES:
        for a in INSTRUMENTS:
            for b in INSTRUMENTS:
                if a != b:  # train on A, evaluate on B
                    out[f"emb_{a}_to_{b}"] = out[f"emb_{a}" if stage == "fit" else f"emb_{b}"]
    if HOLDOUT is not None:
        out[f"emb_{HOLDOUT}_fit"] = out[f"emb_{HOLDOUT}"]
        if stage == "fit":
            # Probe transfer: train on a training instrument's embeddings.
            src = TRAIN_INSTRUMENTS[int(torch.randint(len(TRAIN_INSTRUMENTS), ()))]
            out[f"emb_{HOLDOUT}"] = out[f"emb_{src}"]
    out["inst_valid"] = batch["inst_valid"]
    out["inst_faint"] = batch["inst_faint"]
    out["label"] = batch["label"].long()


PREDICTOR_INST = None    # --predictor-instrument: its views get a predictor
PREDICTOR_HIDDEN = 1024  # --predictor-hidden
ADV_WEIGHT = 0.0   # GRL strength for the instrument-confusion loss (CLI)
ADV_RAMP = 0.1     # fraction of training over which the GRL strength ramps
FAINT_MAG = {}     # instrument -> median-mag threshold: stars fainter than
                   # this in that instrument are dropped from the UMAP figure
                   # (viz only; metrics/losses unaffected). Set via
                   # --asassn-faint-mag.


class _GradReverse(torch.autograd.Function):
    """Identity forward; gradients scaled by ``-lamb`` on the way back."""

    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lamb * grad, None


def _adversarial_loss(self, output, batch):
    """DANN-style instrument-confusion loss on the view embeddings.

    A linear head learns to predict the telescope from each view's
    (undetached) backbone embedding while the reversed gradient pushes the
    encoder to make telescopes indistinguishable — this attacks the additive
    offset ``z = f(star) + g(instrument)`` directly in embedding space,
    where the prediction loss only penalises it quadratically and SIGReg
    not at all. The GRL strength ramps 0 -> ``ADV_WEIGHT`` over the first
    ``ADV_RAMP`` of training so the head is a competent adversary before
    the encoder starts fighting it (the head's own CE is never scaled).
    Returns ``(ce, acc, n)``.
    """
    vi = batch["view_inst"].to(output.features.device)  # [N, n_views]
    labels = vi.T.reshape(-1)  # view-major, matches features layout
    mask = labels >= 0
    if not bool(mask.any()):
        zero = output.features.sum() * 0.0
        return zero, zero, 0
    ramp = min(1.0, float(self.current_epoch)
               / max(1.0, ADV_RAMP * (self.trainer.max_epochs or 1)))
    z = _GradReverse.apply(output.features[mask], ADV_WEIGHT * ramp)
    logits = self.model.inst_head(z)
    ce = torch.nn.functional.cross_entropy(logits, labels[mask])
    acc = (logits.argmax(1) == labels[mask]).float().mean()
    return ce, acc, int(mask.sum())


def lejepa_forward(self, batch, stage):
    """``--mode lejepa``: LeJEPA loss + detached probe embeddings."""
    out = {}
    global_views = [batch[k] for k in sorted(batch) if k.startswith("global")]
    local_views = [batch[k] for k in sorted(batch) if k.startswith("local")]
    if stage == "fit":
        output: LeJEPAOutput = self.model.forward(
            global_views=global_views, local_views=local_views,
            view_inst=batch["view_inst"],
        )
        loss, pred_loss, sigreg_loss = (
            output.loss, output.inv_loss, output.sigreg_loss
        )
        if ADV_WEIGHT > 0:
            adv_ce, adv_acc, n_adv = _adversarial_loss(self, output, batch)
            if n_adv:
                loss = loss + adv_ce
                self.log("train/adv_ce", adv_ce, on_step=True,
                         on_epoch=True, sync_dist=True)
                self.log("train/adv_acc", adv_acc, on_step=True,
                         on_epoch=True, sync_dist=True)
        tag = "train"
    else:
        # Canonical view -> embedding for the witness ...
        output: LeJEPAOutput = self.model.forward(
            values=batch["probe_joint"][0],
            positions=batch["probe_joint"][1],
            pad_mask=batch["probe_joint"][2],
        )
        # ... and the deterministic view set -> real validation losses.
        loss, pred_loss, sigreg_loss = _val_losses(
            self.model, global_views, local_views, batch["view_inst"]
        )
        tag = "val"

    out["loss"] = loss
    out["embedding"] = output.embedding
    out["projection"] = output.projection
    _probe_embeddings(self.model.backbone, batch, out, stage)
    if stage != "fit":
        # Projector-space copies for InstrumentEmbeddingEval: the LeJEPA
        # loss acts there, so this is where invariance shows up first.
        with torch.no_grad():
            for name in INSTRUMENTS:
                out[f"proj_{name}"] = self.model.projector(out[f"emb_{name}"])
        if ADV_WEIGHT > 0:
            # Head accuracy on the canonical probe views: chance level
            # (1/n_train_inst) means the embedding no longer encodes the
            # telescope linearly.
            with torch.no_grad():
                ls, ys = [], []
                for i, inst in enumerate(TRAIN_INSTRUMENTS):
                    m = batch["inst_valid"][:, INSTRUMENTS.index(inst)]
                    if bool(m.any()):
                        ls.append(self.model.inst_head(out[f"emb_{inst}"][m]))
                        ys.append(torch.full((int(m.sum()),), i,
                                             device=m.device))
                if ls:
                    lg, y = torch.cat(ls), torch.cat(ys)
                    self.log("val/adv_acc",
                             (lg.argmax(1) == y).float().mean(),
                             on_epoch=True, sync_dist=True)

    # pred = invariance (prediction) term, sigreg = SIGReg term,
    # loss = pred + lamb * sigreg (see LeJEPA._compute_loss).
    log_kw = dict(on_step=stage == "fit", on_epoch=True, sync_dist=True)
    self.log(f"{tag}/sigreg_loss", sigreg_loss, **log_kw)
    self.log(f"{tag}/pred_loss", pred_loss, **log_kw)
    self.log(f"{tag}/loss", loss, **log_kw)
    return out


def random_forward(self, batch, stage):
    """``--mode random``: frozen random-init encoder; only the probes train."""
    out = {}
    _probe_embeddings(self.model.backbone, batch, out, stage)
    # A zero loss with a graph so Lightning's backward is a no-op.
    out["loss"] = self.model.head.weight.sum() * 0.0
    return out


def supervised_forward(self, batch, stage):
    """``--mode supervised``: encoder + linear head, CE on the joint view."""
    out = {}
    v, p, m = batch["probe_joint"]
    logits = self.model.head(self.model.backbone(v, p, m))
    loss = torch.nn.functional.cross_entropy(logits, batch["label"].long())
    out["loss"] = loss
    _probe_embeddings(self.model.backbone, batch, out, stage)
    tag = "train" if stage == "fit" else "val"
    self.log(f"{tag}/ce", loss, on_step=stage == "fit", on_epoch=True,
             sync_dist=True)
    return out


FORWARDS.update(lejepa=lejepa_forward, random=random_forward,
                supervised=supervised_forward)


class SupervisedLightCurve(nn.Module):
    """Backbone + linear head (baseline modes). Same encoder as LeJEPA."""

    def __init__(self, backbone, embed_dim, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim


class InstrumentEmbeddingEval(pl.Callback):
    """Cross-instrument embedding quality on the val set, logged to wandb.

    For every val star one capped-span embedding per instrument is produced
    (``emb_<inst>``). At the end of each validation epoch, for every
    instrument pair ``(a, b)`` with enough stars observed by both:

    - ``inst/<a>-<b>/pair_cos``: mean cosine similarity of the two embeddings
      of the *same* star; ``rand_cos``: same for random star pairs;
      ``cos_gap`` = the difference (higher = better).
    - ``inst/<a>-<b>/recall@1`` / ``recall@5``: nearest-neighbour retrieval of
      the same star across the pair (cosine), both directions averaged.
    - ``inst/<a>-<b>/sep_acc``: held-out accuracy of a logistic regression
      that tries to tell ``a`` embeddings from ``b`` ones. 0.5 = the two
      clouds fully overlap (invariant), 1.0 = trivially separable.
    - ``inst/mean_*``: the same metrics averaged over pairs (``inst/*`` keeps
      the 2-instrument names for backward-compatible dashboards).
    - ``instproj/...``: the same metrics in projector space (LeJEPA mode),
      where the invariance loss acts — the first place invariance appears.
    - Every ``every`` epochs: a UMAP figure of all instruments' embeddings:
      (a) coloured by instrument with ``n_marked`` stars' embeddings joined
      (one colour per star; this pairs panel draws the *training*
      instruments only — the held-out one is untrained and just clutters
      the pair lines), (b) one panel per instrument over the rest in gray
      (held-out included), (c) coloured by class.
    """

    def __init__(self, idx_to_label=None, every=5, max_points=3000,
                 n_marked=8, seed=0, min_pairs=20):
        self.idx_to_label = idx_to_label or {}
        self.every = every
        self.max_points = max_points
        self.n_marked = n_marked
        self.seed = seed
        self.min_pairs = min_pairs
        self._buf = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self._buf = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        if "inst_valid" not in outputs:
            return
        self._buf.append((
            [outputs[f"emb_{x}"].float().cpu() for x in INSTRUMENTS],
            outputs["inst_valid"].cpu(), outputs["label"].cpu(),
            [outputs[f"proj_{x}"].float().cpu() for x in INSTRUMENTS]
            if f"proj_{INSTRUMENTS[0]}" in outputs else None,
            outputs["inst_faint"].cpu(),
        ))

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._buf or trainer.sanity_checking:
            return
        embs = [torch.nn.functional.normalize(torch.cat([b[0][i] for b in self._buf]), dim=1)
                for i in range(len(INSTRUMENTS))]
        valid = torch.cat([b[1] for b in self._buf])
        y = torch.cat([b[2] for b in self._buf])
        faint = torch.cat([b[4] for b in self._buf])
        projs = None
        if self._buf[0][3] is not None:
            projs = [torch.nn.functional.normalize(torch.cat([b[3][i] for b in self._buf]), dim=1)
                     for i in range(len(INSTRUMENTS))]
        if trainer.world_size > 1:
            # Under DDP each rank buffers only its val shard; gather so the
            # pair metrics and the UMAP are computed on the full val set and
            # stay comparable with single-device runs (tensors are tiny).
            def g(t):
                out = pl_module.all_gather(t.to(pl_module.device))
                return out.reshape(-1, *t.shape[1:]).cpu()
            embs = [g(e) for e in embs]
            valid = g(valid.to(torch.uint8)).bool()
            y = g(y)
            faint = g(faint.to(torch.uint8)).bool()
            if projs is not None:
                projs = [g(p) for p in projs]
        metrics = self._pair_metrics(embs, valid, "inst")
        if projs is not None:
            metrics.update(self._pair_metrics(projs, valid, "instproj"))
        if not metrics:
            return
        pl_module.log_dict(metrics, on_epoch=True, rank_zero_only=True)
        if trainer.current_epoch % self.every == 0 and trainer.is_global_zero:
            # Viz-only faint filter (FAINT_MAG): drop noisy points from the
            # UMAP figure without touching the pair metrics above.
            self._plot(trainer, [e.numpy() for e in embs],
                       (valid & ~faint).numpy(), y.numpy())

    def _pair_metrics(self, embs, valid, prefix):
        """Pairwise cross-instrument metrics on unit-normalised ``embs``.

        Under DDP the caller has already all-gathered the full val set.
        """
        metrics, agg = {}, collections.defaultdict(list)
        for i, a_name in enumerate(INSTRUMENTS):
            for j in range(i + 1, len(INSTRUMENTS)):
                b_name = INSTRUMENTS[j]
                keep = valid[:, i] & valid[:, j]
                n = int(keep.sum())
                if n < self.min_pairs:
                    continue
                a, b = embs[i][keep], embs[j][keep]
                sim = a @ b.T
                pair = sim.diag()
                off = sim[~torch.eye(n, dtype=torch.bool)]
                rank_ab = (sim > pair[:, None]).sum(1)
                rank_ba = (sim > pair[None, :]).sum(0)
                m = {
                    "pair_cos": pair.mean().item(),
                    "rand_cos": off.mean().item(),
                    "cos_gap": (pair.mean() - off.mean()).item(),
                    "recall@1": ((rank_ab == 0).float().mean()
                                 + (rank_ba == 0).float().mean()).item() / 2,
                    "recall@5": ((rank_ab < 5).float().mean()
                                 + (rank_ba < 5).float().mean()).item() / 2,
                    "sep_acc": self._separability(a, b),
                    "n_pairs": float(n),
                }
                for k, v in m.items():
                    metrics[f"{prefix}/{a_name}-{b_name}/{k}"] = v
                    agg[k].append(v)
        for k, vs in agg.items():
            metrics[f"{prefix}/mean_{k}"] = float(np.mean(vs))
            if len(INSTRUMENTS) == 2:
                metrics[f"{prefix}/{k}"] = float(np.mean(vs))
        return metrics

    @staticmethod
    def _separability(a, b):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        x = torch.cat([a, b]).numpy()
        t = np.r_[np.zeros(len(a)), np.ones(len(b))]
        clf = LogisticRegression(max_iter=500)
        return float(cross_val_score(clf, x, t, cv=3).mean())

    def _plot(self, trainer, embs, valid, y):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(self.seed)
        n_all = len(y)
        sel = (rng.choice(n_all, self.max_points, replace=False)
               if n_all > self.max_points else np.arange(n_all))
        n = len(sel)
        # The joint UMAP fit and the per-instrument/class panels keep every
        # instrument (held-out included); only the pairs panel (leftmost) is
        # restricted to the training instruments — the held-out one is
        # untrained by construction and just clutters the pair lines.
        show = [i for i, nm in enumerate(INSTRUMENTS)
                if nm in TRAIN_INSTRUMENTS] or list(range(len(INSTRUMENTS)))
        # Stack [inst0 rows..., inst1 rows..., ...] keeping only valid ones.
        xs, inst_id, star_id, cls = [], [], [], []
        for i in range(len(INSTRUMENTS)):
            ok = valid[sel, i]
            xs.append(embs[i][sel][ok]); inst_id.append(np.full(ok.sum(), i))
            star_id.append(np.arange(n)[ok]); cls.append(y[sel][ok])
        x = np.concatenate(xs); inst_id = np.concatenate(inst_id)
        star_id = np.concatenate(star_id); cls = np.concatenate(cls)
        try:
            import umap
            z = umap.UMAP(n_components=2, random_state=self.seed,
                          metric="cosine").fit_transform(x)
            method = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA
            z = PCA(2, random_state=self.seed).fit_transform(x)
            method = "PCA"

        n_panels = 2 + len(INSTRUMENTS)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
        kw = dict(s=5, alpha=0.5, linewidths=0)
        cmap = plt.get_cmap("tab10")
        inst_col = [cmap(i) for i in range(len(INSTRUMENTS))]
        markers = "o^sDvP*X"
        ax = axes[0]
        for i in show:
            ax.scatter(*z[inst_id == i].T, c=[inst_col[i]],
                       label=INSTRUMENTS[i], **kw)
        # Mark a few stars: same colour = same star, marker = instrument.
        full = np.where(valid[sel][:, show].all(1))[0]
        marked = rng.choice(full, min(self.n_marked, len(full)), replace=False)
        for j, sid in enumerate(marked):
            c = cmap(j % 10)
            msk = (star_id == sid) & np.isin(inst_id, show)
            pts = z[msk]; order = inst_id[msk]
            ax.plot(pts[:, 0], pts[:, 1], c=c, lw=1.0, alpha=0.8)
            for pt, ii in zip(pts, order):
                ax.scatter(*pt, marker=markers[ii % len(markers)], s=90, c=[c],
                           edgecolors="k", zorder=5)
        mk = ", ".join(f"{markers[i % len(markers)]}={INSTRUMENTS[i]}"
                       for i in show)
        ax.set_title(f"joint by instrument ({mk}; same colour = same star)", fontsize=9)
        ax.legend(loc="best", markerscale=4, fontsize=8)

        for i in range(len(INSTRUMENTS)):
            name = INSTRUMENTS[i]
            ax = axes[1 + i]
            ax.scatter(*z.T, c="lightgray", **kw)
            ax.scatter(*z[inst_id == i].T, c=[inst_col[i]], **kw)
            ax.set_title(f"{name} only (gray = all)"
                         + (" [held out]" if name == HOLDOUT else ""))

        ax = axes[-1]
        top = [c for c, _ in
               sorted(zip(*np.unique(cls, return_counts=True)),
                      key=lambda t: -t[1])[:10]]
        ax.scatter(*z.T, c="lightgray", **kw)
        for j, c in enumerate(top):
            m = cls == c
            ax.scatter(*z[m].T, c=[cmap(j)], label=self.idx_to_label.get(int(c), c), **kw)
        ax.set_title("by class (top 10)")
        ax.legend(loc="best", fontsize=7, markerscale=4)

        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(z[:, 0].min(), z[:, 0].max())
            ax.set_ylim(z[:, 1].min(), z[:, 1].max())
        fig.suptitle(f"{method} of val embeddings, epoch {trainer.current_epoch}")
        fig.tight_layout()

        logger_ = trainer.logger
        if logger_ is not None and hasattr(logger_, "experiment") and \
                logger_.__class__.__name__ == "WandbLogger":
            import wandb
            logger_.experiment.log({"inst/umap": wandb.Image(fig),
                                    "trainer/global_step": trainer.global_step})
        elif trainer.log_dir:
            out = Path(trainer.log_dir) / f"umap_epoch{trainer.current_epoch:04d}.png"
            fig.savefig(out, dpi=110)
            logger.info(f"saved {out}")
        plt.close(fig)


def resolve_resume(args):
    """Checkpoint path for ``--resume``, or ``None`` to start fresh.

    ``auto`` looks for ``last.ckpt`` in the directory Lightning writes this
    run's checkpoints to under WandbLogger (``RUNS_DIR/<project>/<run id>``),
    so a resubmitted SLURM job with the same ``WANDB_RUN_ID`` picks up where
    the previous one stopped, and the first submission starts fresh.
    """
    if not args.resume:
        return None
    if args.resume != "auto":
        if not Path(args.resume).exists():
            raise FileNotFoundError(f"--resume {args.resume} does not exist")
        return args.resume
    run_id = os.environ.get("WANDB_RUN_ID")
    if not (args.wandb and run_id):
        raise ValueError("--resume auto needs wandb on and WANDB_RUN_ID set "
                         "(it locates runs/<project>/<run id>/checkpoints); "
                         "pass an explicit checkpoint path otherwise")
    last = RUNS_DIR / args.wandb / run_id / "checkpoints" / "last.ckpt"
    if last.exists():
        logger.warning(f"resuming from {last}")
        return str(last)
    logger.warning(f"--resume auto: no {last} yet, starting fresh")
    return None


def train_once(args, lamb, records, n_classes, cfg, sweep_mode=False,
               idx_to_label=None):
    """Build everything fresh and train one run at the given lambda.

    Returns ``(witness_proj, witness_emb)`` so their ``.history`` can be read
    for certification (proj space is where SIGReg acts). In ``sweep_mode``
    checkpointing and loggers are disabled.
    """
    pl.seed_everything(args.seed, workers=True)
    data = build_data(args, records, cfg)
    if args.mode in MODEL_BUILDERS:
        model = MODEL_BUILDERS[args.mode](args, n_classes)
    else:
        model = build_model(args.width, args.proj_dim, lamb, args.n_slices,
                            args.depth, mode=args.mode, n_classes=n_classes,
                            projector=args.projector)
    forward = FORWARDS[args.mode]

    module = spt.Module(
        model=model,
        forward=forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": args.base_lr,
                "weight_decay": 0.05,
                "betas": (0.9, 0.999),
                # mu-P per-layer learning rates (see cifar10 script).
                "params": mup_param_groups(
                    model, base_lr=args.base_lr,
                    weight_decay=0.05, base_fanin=BASE_FANIN,
                ),
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": min(
                    min(10, max(1, args.epochs // 10))
                    * (len(data.train) // args.devices),
                    (len(data.train) // args.devices) * args.epochs - 1,
                ),
                "start_factor": 0.01,
                "end_lr": args.base_lr / 1000,
                "total_steps": (len(data.train) // args.devices) * args.epochs,
            },
            "interval": "step",
        },
    )

    witnesses = []
    witness_proj = witness_emb = None
    if args.mode == "lejepa":
        witness_proj = WitnessCallback(
            name="witness_proj", target="projection",
            queue_length=2048, target_shape=args.proj_dim,
        )
        witness_emb = WitnessCallback(
            name="witness_emb", target="embedding",
            queue_length=2048, target_shape=model.embed_dim,
        )
        witnesses = [witness_proj, witness_emb]
    # One linear probe per downstream input: joint (TESS+ZTF), TESS-only,
    # ZTF-only. Metrics: eval/probe_<input>_{top1,balanced,f1}.
    probes = []
    if not args.no_probe:
        for name in probe_names():
            probes.append(
                spt.callbacks.OnlineProbe(
                    module,
                    name=f"probe_{name}",
                    input=f"emb_{name}",
                    target="label",
                    probe=nn.Linear(model.embed_dim, n_classes),
                    loss=nn.CrossEntropyLoss(),
                    metrics={
                        "top1": torchmetrics.classification.MulticlassAccuracy(
                            n_classes, average="micro"
                        ),
                        "balanced": torchmetrics.classification.MulticlassAccuracy(
                            n_classes, average="macro"
                        ),
                        "f1": torchmetrics.classification.MulticlassF1Score(
                            n_classes, average="macro"
                        ),
                    },
                    optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 1e-6},
                )
            )
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger

        run_logger = WandbLogger(
            project=args.wandb,
            save_dir=str(RUNS_DIR),
            name=f"{args.mode}-{DATASET.split('/')[-1].replace('-isect', '')}"
                 + "".join(f"-no{x}" for x in args.exclude_instrument)
                 + "".join(f"-no{x}" for x in args.exclude_band)
                 + (f"-hold{HOLDOUT}" if HOLDOUT else "")
                 + f"-w{args.width}-d{args.proj_dim}-B{args.batch_size}"
                 + (f"-lam{lamb:.4g}" if args.mode == "lejepa" else "")
                 + (f"-adv{args.adv_weight:g}" if args.adv_weight > 0 else "")
                 + ("-idproj" if args.projector == "identity" else "")
                 + (f"-pred{PREDICTOR_INST}" if PREDICTOR_INST else ""),
            config={**vars(args), "lamb": lamb, "sweep_mode": sweep_mode},
            # Continue the same wandb run when a job is resubmitted with
            # the same WANDB_RUN_ID (--resume auto).
            resume="allow",
        )
    else:
        run_logger = not sweep_mode

    # last.ckpt every epoch (the resume point; a wall-time kill loses at most
    # one epoch) + a kept snapshot every --ckpt-every epochs. dirpath=None
    # keeps Lightning's layout: runs/<project>/<run id>/checkpoints/.
    # Lightning only writes last.ckpt in an epoch where it also saved a
    # regular checkpoint, hence the rolling save_top_k=1 file next to it.
    checkpoints = []
    if not sweep_mode:
        checkpoints.append(ModelCheckpoint(save_last=True, save_top_k=1,
                                           every_n_epochs=1,
                                           filename="rolling-{epoch}"))
        # every_n_epochs is part of the callback's state key, so a second
        # callback with every_n_epochs=1 would collide with the one above.
        if args.ckpt_every > 1:
            checkpoints.append(ModelCheckpoint(save_top_k=-1,
                                               every_n_epochs=args.ckpt_every))

    trainer = pl.Trainer(
        default_root_dir=str(RUNS_DIR),
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        num_sanity_val_steps=0,
        enable_checkpointing=not sweep_mode,
        logger=run_logger,
        callbacks=[
            *probes,
            InstrumentEmbeddingEval(idx_to_label, every=args.umap_every),
            *witnesses,
            # GPU util / memory, CPU, RAM as hardware/* metrics (any logger).
            spt.callbacks.HardwareMonitor(interval_seconds=10),
            *checkpoints,
        ],
        precision=args.precision,
    )
    trainer.fit(module, datamodule=data,
                ckpt_path=None if sweep_mode else resolve_resume(args))
    return witness_proj, witness_emb


def build_parser():
    """The CLI; sibling scripts extend it (see pretrain_contrastive.py)."""
    ap = argparse.ArgumentParser()
    # --- ladder / mu-P (mirror the cifar10 script) ---
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                    help="encoder d_model (one rung of the ladder)")
    ap.add_argument("--depth", type=int, default=6,
                    help="encoder depth")
    ap.add_argument("--proj-dim", type=int, default=REF_PROJ_DIM,
                    help="SIGReg dimension = projector output dim")
    ap.add_argument("--projector", choices=["identity", "mlp"],
                    default="mlp",
                    help="mlp (default): BN+ReLU 2048-2048-proj_dim head; the "
                         "prediction + SIGReg losses act on its output while "
                         "probes/downstream read the encoder embedding; "
                         "identity: losses act directly on the embeddings "
                         "(proj dim = width)")
    ap.add_argument("--predictor-instrument", default=None,
                    help="LeJEPA: views of this (lower-quality) training "
                         "instrument, e.g. ASASSN, pass through an MLP "
                         "predictor between encoder and projector")
    ap.add_argument("--predictor-hidden", type=int, default=PREDICTOR_HIDDEN,
                    help="hidden width of the predictor MLP")
    ap.add_argument("--lamb-ref", type=float, default=0.02,
                    help="additive lambda tuned at the REFERENCE rung; "
                         "rescaled here by the master rule")
    ap.add_argument("--sweep-lamb", type=str, default=None,
                    help="comma-separated lambda grid, e.g. "
                         "'0.005,0.02,0.08,0.3'. Runs the Step-0 sweep at THIS "
                         "config and reports the recommended lambda_ref.")
    ap.add_argument("--base-lr", type=float, default=4e-4,
                    help="mu-P base lr (transfers as-is across widths)")
    ap.add_argument("--batch-size", type=int, default=REF_BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=700)
    ap.add_argument("--n-slices", type=int, default=128,
                    help="M; keep FIXED between Step 0 and the ladder")
    ap.add_argument("--ref-width", type=int, default=REF_WIDTH)
    ap.add_argument("--ref-proj-dim", type=int, default=REF_PROJ_DIM)
    ap.add_argument("--ref-batch-size", type=int, default=REF_BATCH_SIZE)
    # --- data ---
    ap.add_argument("--dataset", default="pc/ZTF-ATLAS-ASASSN-isect",
                    choices=sorted(INSTRUMENT_REGISTRY),
                    help="dataset id (instrument layout from the registry): "
                         "pc/<name> = local PC_matches DatasetDict under "
                         "--data-root with its own train/validation/test "
                         "split; hibb/<name> = private HF Hub repo")
    ap.add_argument("--data-root", type=str, default=None,
                    help=f"PC_matches directory for pc/ datasets (default "
                         f"$PC_MATCHES_ROOT or {PC_ROOT})")
    ap.add_argument("--exclude-instrument", action="append", default=[],
                    help="drop this instrument entirely (repeatable), e.g. TESS")
    ap.add_argument("--exclude-band", action="append", default=[],
                    help="drop this band column (repeatable), e.g. i_ZTF "
                         "(StarEmbed uses ZTF g + r only)")
    ap.add_argument("--no-transfer-probes", action="store_true",
                    help="skip the <A>_to_<B> cross-instrument transfer probes")
    ap.add_argument("--holdout-instrument", default=None,
                    help="instrument excluded from all LeJEPA views; probed "
                         "and evaluated zero-shot (see probe_names)")
    ap.add_argument("--max-objects", type=int, default=0,
                    help="cap objects (streams only this many); 0 = full split")
    ap.add_argument("--min-obs", type=int, default=8,
                    help="min observations per instrument to keep an object")
    ap.add_argument("--min-train-instruments", type=int, default=2,
                    help="min training instruments per object; 1 admits "
                         "single-survey stars (their positive pairs are "
                         "different-epoch windows of the same instrument)")
    ap.add_argument("--exclude-stars-file", type=str, default=None,
                    help="file of gaia_dr3_source_ids (one per line) to drop "
                         "entirely, e.g. StarEmbed val/test/anom crossmatches")
    ap.add_argument("--oversample-paired", type=int, default=1,
                    help="repeat multi-instrument stars this many times in "
                         "the train epoch so cross-instrument pairs are not "
                         "drowned out by single-survey stars")
    add_wave_args(ap)
    ap.add_argument("--split-file", type=str, default=None,
                    help="(hibb/ datasets only; pc/ datasets use their own "
                         "split column) gaia-id train/val/test split json from "
                         "data/make_splits.py (default: data/splits/"
                         "<dataset>_seed<split-seed>.json; 'none' = "
                         "on-the-fly stratified split)")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="only for the on-the-fly fallback split")
    ap.add_argument("--test-frac", type=float, default=0.1,
                    help="only for the on-the-fly fallback split")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--hf-token", type=str, default=None,
                    help="overrides HF_TOKEN env var / cached `hf auth login`")
    # --- view generation ---
    ap.add_argument("--window-days", type=float, default=500.0,
                    help="min shared cross-instrument window length")
    ap.add_argument("--window-days-max", type=float, default=1500.0,
                    help="max window length; length ~ N(mid, span/4) clipped "
                         "to [min, max] (N(1000, 250) by default; also caps "
                         "the probe/eval views); 0 = fixed --window-days")
    ap.add_argument("--min-window-obs", type=int, default=200,
                    help="reject sampled windows holding fewer observations "
                         "(per instrument for globals, of the view's bands "
                         "for the probe span) and redraw; densest candidate "
                         "after window_tries failures")
    ap.add_argument("--over-budget", choices=["tail", "random"], default="tail",
                    help="view over its token budget: cut the tail (max "
                         "input length) or random-subsample the cadence")
    ap.add_argument("--adv-weight", type=float, default=0.0,
                    help=">0: DANN gradient-reversal instrument-confusion "
                         "loss on the view embeddings, with this GRL "
                         "strength (ramped over the first --adv-ramp of "
                         "training). Logs train/adv_{ce,acc}, val/adv_acc.")
    ap.add_argument("--adv-ramp", type=float, default=0.1,
                    help="fraction of epochs over which the GRL strength ramps")
    ap.add_argument("--tess-density", choices=["full", "random"], default="random",
                    help="random: TESS view token count ~ log-U(min_obs, budget)")
    ap.add_argument("--n-local", type=int, default=4)
    ap.add_argument("--global-tokens", type=int, default=512)
    ap.add_argument("--local-tokens", type=int, default=256)
    ap.add_argument("--eval-tokens", type=int, default=512)
    ap.add_argument("--no-resample", action="store_true",
                    help="disable uncertainty resampling (ablation)")
    ap.add_argument("--global-mode", choices=["instrument", "concat", "both"],
                    default="instrument",
                    help="globals: TESS-vs-ZTF ('instrument'), the two "
                         "half-A+half-B chimeras ('concat'), or all four")
    ap.add_argument("--local-mode", choices=["all", "instrument", "mixed"],
                    default="all",
                    help="bands of each local view: all | one instrument "
                         "(50/50 TESS/ZTF) | mixed (coin flip per local)")
    ap.add_argument("--p-distance", type=float, default=0.0,
                    help="prob. of an achromatic brightness offset per view")
    ap.add_argument("--distance-mag", type=float, default=1.0,
                    help="offset ~ U(-x, x) mag")
    ap.add_argument("--p-extinction", type=float, default=0.15,
                    help="prob. of a CCM89 extinction jitter per view")
    ap.add_argument("--ebv-jitter", type=float, default=0.2,
                    help="dE(B-V) ~ N(0, x) mag (either sign)")
    ap.add_argument("--p-period-shift", type=float, default=0.0,
                    help="prob. of moving a view's window by k*period")
    ap.add_argument("--max-period-shift", type=int, default=2, help="|k| <= x")
    ap.add_argument("--norm", choices=["band", "band-global", "object", "view"],
                    default="band",
                    help="fixed per-object stats: per-band centre + pooled "
                         "scale ('band'), pooled centre ('object'); 'view' "
                         "per-view standardisation cancels the brightness "
                         "augmentations (ablation only)")
    # --- trainer ---
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--accelerator", type=str, default="auto")
    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--wandb", type=str, default="lejepa-lightcurves",
                    help="wandb project name (on by default); one named run "
                         "per candidate during --sweep-lamb. Logs args as "
                         "config and hardware/* GPU-util metrics.")
    ap.add_argument("--no-wandb", action="store_true",
                    help="disable wandb (falls back to Lightning's CSV/TB logger)")
    ap.add_argument("--p-drop", type=float, default=0.0,
                    help="prob. of random observation dropout per view")
    ap.add_argument("--max-drop-frac", type=float, default=0.5,
                    help="drop fraction ~ U(0, x)")
    ap.add_argument("--n-gaps", type=int, default=0,
                    help="contiguous time gaps blanked per view (0 = off)")
    ap.add_argument("--gap-frac", type=float, default=0.1,
                    help="each gap = x of the view's time span")
    ap.add_argument("--umap-every", type=int, default=5,
                    help="log the UMAP embedding figure every n epochs")
    ap.add_argument("--asassn-faint-mag", type=float, default=16.0,
                    help="drop stars whose ASAS-SN median mag is fainter "
                         "than this from the UMAP figure (ASAS-SN photometry "
                         "is noise-dominated near its ~17 mag limit); viz "
                         "only, metrics unaffected; 0 = off")
    ap.add_argument("--mode", choices=["lejepa", "random", "supervised"],
                    default="lejepa",
                    help="lejepa = SSL pretraining; random = frozen random-init "
                         "encoder (probe-only floor); supervised = same encoder "
                         "+ linear head trained with CE on the joint view")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the online linear probe (pretraining only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    # --- checkpointing / resume ---
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="also keep a snapshot every n epochs (epoch=<n-1>-*.ckpt, "
                         "never overwritten; 0 or 1 = off). last.ckpt is refreshed "
                         "every epoch regardless")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint to resume from (weights, optimizer, "
                         "scheduler, epoch), or 'auto' = this run's last.ckpt "
                         "(runs/<wandb project>/$WANDB_RUN_ID/checkpoints/), "
                         "starting fresh if it does not exist yet. Keep "
                         "--epochs identical: the LR schedule is built from it")
    return ap


def apply_args(args):
    """Set the module-level layout/config globals from parsed ``args``."""
    if args.no_wandb:
        args.wandb = None
    configure_instruments(args.dataset, args.holdout_instrument,
                          tuple(args.exclude_instrument), tuple(args.exclude_band))
    configure_wave(args)
    global TRANSFER_PROBES, ADV_WEIGHT, ADV_RAMP, FAINT_MAG
    global PREDICTOR_INST, PREDICTOR_HIDDEN
    TRANSFER_PROBES = not args.no_transfer_probes
    PREDICTOR_INST = getattr(args, "predictor_instrument", None) or None
    PREDICTOR_HIDDEN = getattr(args, "predictor_hidden", PREDICTOR_HIDDEN)
    if PREDICTOR_INST is not None and PREDICTOR_INST not in TRAIN_INSTRUMENTS:
        raise ValueError(f"--predictor-instrument {PREDICTOR_INST!r} not a "
                         f"training instrument {TRAIN_INSTRUMENTS}")
    ADV_WEIGHT, ADV_RAMP = args.adv_weight, args.adv_ramp
    if args.asassn_faint_mag > 0 and "ASASSN" in INSTRUMENTS:
        FAINT_MAG = {"ASASSN": args.asassn_faint_mag}
    if args.projector == "identity":
        # SIGReg acts on the embeddings: the effective projection dim is the
        # width, on BOTH sides of the master-lambda ratio, so lambda_ref is
        # still read off unchanged at the reference width/batch.
        args.proj_dim = args.width
        args.ref_proj_dim = args.ref_width


def make_view_config(args) -> ViewConfig:
    return ViewConfig(
        window_days=args.window_days,
        window_days_max=args.window_days_max,
        min_window_obs=args.min_window_obs,
        over_budget=args.over_budget,
        tess_density=args.tess_density,
        n_local=args.n_local,
        global_tokens=args.global_tokens,
        local_tokens=args.local_tokens,
        eval_tokens=args.eval_tokens,
        min_tokens=args.min_obs,
        resample=not args.no_resample,
        global_mode=args.global_mode,
        local_mode=args.local_mode,
        p_distance=args.p_distance,
        distance_mag=args.distance_mag,
        p_extinction=args.p_extinction,
        ebv_jitter=args.ebv_jitter,
        p_period_shift=args.p_period_shift,
        max_period_shift=args.max_period_shift,
        p_drop=args.p_drop,
        max_drop_frac=args.max_drop_frac,
        n_gaps=args.n_gaps,
        gap_frac=args.gap_frac,
        norm=args.norm,
    )


def main():
    args = build_parser().parse_args()
    apply_args(args)
    cfg = make_view_config(args)
    records, label_to_idx = load_records(args)
    n_classes = len(label_to_idx)

    if args.sweep_lamb is not None:
        from stable_pretraining.lambda_sweep import run_lambda_sweep

        assert args.devices == 1, "--sweep-lamb requires --devices 1"
        assert args.mode == "lejepa", "--sweep-lamb requires --mode lejepa"
        grid = [float(x) for x in args.sweep_lamb.split(",") if x.strip()]
        print(
            f"Step-0 lambda sweep at (width={args.width}, "
            f"proj_dim={args.proj_dim}, B={args.batch_size}): grid {grid}"
        )
        result = run_lambda_sweep(
            lambda lam: train_once(
                args, lam, records, n_classes, cfg, sweep_mode=True
            )[0].history,
            grid,
        )
        rec = result["selection"]["recommended"]
        if rec is not None:
            print(
                "\nLadder command for each rung (--ref-* pin the anchor to "
                "THIS sweep's config):\n"
                f"  python {Path(__file__).name} --width <n> "
                f"--lamb-ref {rec:g} --ref-width {args.width} "
                f"--ref-proj-dim {args.proj_dim} "
                f"--ref-batch-size {args.batch_size}"
            )
        return

    lamb = master_lambda(
        args.lamb_ref, args.width, args.proj_dim, args.batch_size,
        ref_width=args.ref_width, ref_proj_dim=args.ref_proj_dim,
        ref_batch_size=args.ref_batch_size,
    )
    print(
        f"rung: width={args.width} proj_dim={args.proj_dim} B={args.batch_size} "
        f"-> lambda={lamb:.5f} (ref {args.lamb_ref} at "
        f"({args.ref_width}, {args.ref_proj_dim}, {args.ref_batch_size})), "
        f"base_lr={args.base_lr}"
    )
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    train_once(args, lamb, records, n_classes, cfg, idx_to_label=idx_to_label)


if __name__ == "__main__":
    sys.exit(main())
