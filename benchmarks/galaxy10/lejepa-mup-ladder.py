"""LeJEPA width-transfer ladder on Galaxy10 SDSS: mu-P + the lambda master rule.

Galaxy-scale sibling of ``benchmarks/cifar10/lejepa-mup-ladder.py`` (see the
ImageNet-10 script's docstring for the full protocol). One invocation either
trains ONE rung of the width ladder, or (with ``--sweep-lamb``) runs the
automated Step-0 lambda sweep at one configuration:

- **mu-P**: ``apply_mup`` + ``mup_param_groups`` — ``--base-lr`` is tuned
  once at the reference width and reused verbatim at every rung.
- **Master rule**: ``lambda(n, d, B) = lambda_ref * sqrt((B*d/n)/(B*d/n)_ref)``
  with ``n`` the encoder width, ``d`` the SIGReg dimension (``--proj-dim``,
  the projector output), ``B`` the batch size. ``--ref-*`` pin the anchor's
  config.
- **Certification**: ``--sweep-lamb`` trains each candidate briefly and
  certifies it from the proj-space witnesses (stable AND bounded), then
  recommends the smallest certified lambda with one grid step of margin.

Galaxy10 adaptations vs the CIFAR-10 script:

- **Data**: Galaxy10 SDSS (astroNN) — one HDF5 file of 21,785 galaxy cutouts,
  69x69x3 uint8 (g, r, i bands) under key ``'images'``, integer labels 0-9
  under key ``'ans'``. Downloaded automatically (~200 MB) on first use.
- **Split**: Galaxy10 has NO official train/val split. ``--val-frac`` and
  ``--split-seed`` define a deterministic per-class stratified split
  (``split_indices``) so the split can be aligned exactly with prior runs.
  Per-channel normalization stats are computed over the TRAIN split at first
  load and cached to a JSON next to the h5, keyed by (val_frac, split_seed).
- **Backbone tiling**: ``img_size=64, patch_size=8`` -> 8x8 tokens from the
  69px frame (globals RandomResizedCrop to 64px); local crops 32x32 -> 4x4
  tokens via ``dynamic_img_size``.
- **Augmentations**: galaxy-appropriate — horizontal AND vertical flips
  (no preferred orientation on the sky), mild brightness/contrast jitter
  only (galaxy color is physics, not nuisance — no hue/saturation jitter,
  no grayscale), no blur/solarize. Each choice is documented at the
  transform definitions below.

NOTE on the lambda anchor: a prior notebook-wiring Galaxy10 anchor (additive
lambda ~ 20 at d=128, with SIGReg applied directly on a 64-d projector and
WITHOUT this codebase's SIGReg statistic) does NOT transfer to this script's
wiring. Run ``--sweep-lamb`` first to establish the anchor here.

Typical use::

    # Step 0 at the reference config
    python lejepa-mup-ladder.py --sweep-lamb "0.005,0.02,0.08,0.3" \
        --width 128 --epochs 50

    # Ladder rungs: same base lr, lambda rescaled from the anchor
    python lejepa-mup-ladder.py --width 256 --lamb-ref <anchor> \
        --ref-width 128 --ref-proj-dim 32 --ref-batch-size 256

NOTE on fused attention: timm's fused SDPA path ignores ``attn.scale``, so
``set_fused_attn(False)`` below is REQUIRED for the mu-P 1/head_dim patch to
take effect.
"""

import argparse
import json
import math
import os
import time
import sys
import urllib.request
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from loguru import logger
from PIL import Image
from timm.layers import set_fused_attn
from torchvision.transforms import v2

import stable_pretraining as spt
from stable_pretraining.backbone import MLP
from stable_pretraining.callbacks import WitnessCallback
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import LeJEPA, LeJEPAOutput
from stable_pretraining.optim import apply_mup, mup_param_groups

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402

# mu-P attention needs the unfused path (see module docstring).
set_fused_attn(False)

# ---------------------------------------------------------------------------
# Reference rung: the (width, proj_dim, batch_size) at which lambda_ref and
# base_lr were tuned (CLI-overridable via --ref-*).
# ---------------------------------------------------------------------------
REF_WIDTH = 128
REF_PROJ_DIM = 32
REF_BATCH_SIZE = 256

BASE_FANIN = 256  # mu-P normalization: fan_in at which lr mult == 1

GLOBAL_SIZE = 64  # global crops from the 69px Galaxy10 frame (8x8 tokens)
LOCAL_SIZE = 32   # multi-crop local views (multiple of patch_size=8)

NUM_CLASSES = 10

# Canonical Galaxy10 SDSS release (astroNN docs,
# https://astronn.readthedocs.io/en/latest/galaxy10sdss.html):
# 21,785 images, 69x69x3 uint8 under 'images', labels 0-9 under 'ans'.
GALAXY10_URL = "https://zenodo.org/records/10844811/files/Galaxy10.h5"
GALAXY10_SIZE = 210_234_548  # bytes; verified against the astroNN docs


def download_galaxy10(data_dir: Path) -> Path:
    """Download Galaxy10.h5 into ``data_dir`` if absent; return its path.

    Robust to multi-process launches (torchrun/srun): non-zero LOCAL_RANK
    processes wait for rank 0's download instead of racing it; the temp file
    is per-pid and moved into place atomically; an existing file is trusted
    only if it has the expected size (a truncated file is re-downloaded).

    Args:
        data_dir: Directory the h5 file lives in (``get_data_dir("galaxy10")``).

    Returns:
        Path to the (possibly just downloaded) ``Galaxy10.h5``.
    """
    h5_path = data_dir / "Galaxy10.h5"

    def _valid() -> bool:
        return h5_path.exists() and h5_path.stat().st_size == GALAXY10_SIZE

    if h5_path.exists() and not _valid():
        logger.warning(
            f"{h5_path} has size {h5_path.stat().st_size}, expected "
            f"{GALAXY10_SIZE} — treating as corrupt and re-downloading"
        )
        h5_path.unlink()
    if _valid():
        return h5_path

    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        logger.info("LOCAL_RANK != 0: waiting for rank 0 to download Galaxy10")
        for _ in range(1800):  # up to 30 min
            if _valid():
                return h5_path
            time.sleep(1)
        raise TimeoutError(f"timed out waiting for rank 0 to produce {h5_path}")

    logger.info(f"Downloading Galaxy10 SDSS (~200 MB) from {GALAXY10_URL}")
    tmp_path = data_dir / f"Galaxy10.h5.part.{os.getpid()}"
    try:
        urllib.request.urlretrieve(GALAXY10_URL, tmp_path)
        if tmp_path.stat().st_size != GALAXY10_SIZE:
            raise OSError(
                f"downloaded {tmp_path.stat().st_size} bytes, expected "
                f"{GALAXY10_SIZE}"
            )
        os.replace(tmp_path, h5_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info(
        f"Downloaded {h5_path} ({h5_path.stat().st_size / 1e6:.1f} MB)"
    )
    return h5_path


class Galaxy10SDSS(torch.utils.data.Dataset):
    """Galaxy10 SDSS as an in-memory torch dataset of (PIL image, label).

    Loads the whole h5 into memory once as uint8 numpy (~200 MB) — the
    dataset is small enough that this beats per-item h5 reads under
    multi-worker loading.

    Args:
        h5_path: Path to ``Galaxy10.h5``.
    """

    def __init__(self, h5_path: Path):
        try:
            import h5py
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "Galaxy10 SDSS is stored as HDF5; reading it requires h5py "
                "('pip install h5py')."
            ) from e
        with h5py.File(h5_path, "r") as f:
            self.images: np.ndarray = np.asarray(f["images"], dtype=np.uint8)
            self.labels: np.ndarray = np.asarray(f["ans"], dtype=np.int64)
        assert self.images.shape[0] == self.labels.shape[0]
        logger.info(
            f"Loaded Galaxy10 SDSS: {self.images.shape[0]} images of shape "
            f"{self.images.shape[1:]}"
        )

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return Image.fromarray(self.images[idx]), int(self.labels[idx])


def split_indices(
    labels: np.ndarray, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stratified train/val split.

    Galaxy10 has no official split; this takes ``val_frac`` of EACH class
    (per-class shuffle with ``numpy.random.default_rng(seed)``) so class
    balance is preserved and the split is reproducible from (val_frac, seed).

    Args:
        labels: Integer class labels, shape (N,).
        val_frac: Fraction of each class assigned to validation.
        seed: Seed for the per-class shuffles.

    Returns:
        (train_idx, val_idx): Sorted, disjoint, exhaustive index arrays.
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for c in np.unique(labels):
        cls_idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = int(round(val_frac * len(cls_idx)))
        val_parts.append(cls_idx[:n_val])
        train_parts.append(cls_idx[n_val:])
    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    return train_idx, val_idx


def load_or_compute_norm_stats(
    dataset: Galaxy10SDSS,
    train_idx: np.ndarray,
    h5_path: Path,
    val_frac: float,
    split_seed: int,
) -> tuple[list[float], list[float]]:
    """Per-channel mean/std over the TRAIN split, cached next to the h5.

    The cache JSON is keyed by (val_frac, split_seed) so different splits do
    not share (subtly wrong) statistics.

    Args:
        dataset: Loaded Galaxy10SDSS (uint8 images in memory).
        train_idx: Train-split indices from :func:`split_indices`.
        h5_path: Path to the h5 (the cache lives next to it).
        val_frac: Split parameter (cache key).
        split_seed: Split parameter (cache key).

    Returns:
        (mean, std): Per-channel lists in [0, 1] units (matching ToImage's
        uint8 -> [0, 1] scaling).
    """
    cache_path = h5_path.parent / "galaxy10_norm_stats.json"
    key = f"val_frac={val_frac:g},split_seed={split_seed}"
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            logger.warning(f"corrupt norm-stats cache {cache_path}; recomputing")
            cache = {}
        if key in cache:
            return cache[key]["mean"], cache[key]["std"]
    logger.info("Computing per-channel norm stats over the train split")
    imgs = dataset.images[train_idx]
    mean, std = [], []
    for c in range(imgs.shape[-1]):
        channel = imgs[..., c].astype(np.float32) / 255.0
        mean.append(float(channel.mean(dtype=np.float64)))
        std.append(float(channel.std(dtype=np.float64)))
    cache[key] = {"mean": mean, "std": std}
    # Atomic write: concurrent runs (e.g. DDP ranks) must never observe a
    # half-written JSON; a corrupt cache is recomputed rather than fatal.
    tmp = cache_path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(cache, indent=2))
    os.replace(tmp, cache_path)
    logger.info(f"Train-split norm stats mean={mean} std={std} -> {cache_path}")
    return mean, std


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


def _photometric_transforms():
    # Galaxy-appropriate photometric recipe — each op is a statement about
    # what is nuisance vs physics in an SDSS cutout:
    # - Horizontal AND vertical flips: galaxies have no preferred orientation
    #   on the sky, so BOTH reflections are exact nuisances (unlike natural
    #   images, where "up" is semantic and vertical flips are avoided).
    # - Mild brightness/contrast jitter ONLY: overall flux scale and sky
    #   background vary with observing conditions (airmass, seeing, moon), so
    #   they are nuisances — but kept mild because surface brightness itself
    #   correlates with morphology class.
    # - NO hue/saturation jitter and NO grayscale: galaxy color is physics —
    #   the g/r/i band ratios encode stellar population (age, metallicity,
    #   star formation), and e.g. red ellipticals vs blue spirals is exactly
    #   the signal a probe should find. Never destroy it.
    # - No blur (the cutouts are already PSF/seeing-limited at 69px) and no
    #   solarize (it inverts the flux ordering, which is physical).
    return [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # This repo's ColorJitter requires all four factors (its param-log
        # tensor cannot hold None), so wrap the raw torchvision v2 op for a
        # brightness/contrast-only jitter at p=0.8.
        transforms.WrapTorchTransform(
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8
            )
        ),
    ]


def _global_transform(mean, std):
    # Global views: the galaxy sits centered in the 69px cutout, so crops
    # keeping >= 50% of the area still contain the object (bulge + most of
    # the disk) rather than mostly sky — a global view must see the galaxy.
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop(
            (GLOBAL_SIZE, GLOBAL_SIZE), scale=(0.5, 1.0)
        ),
        *_photometric_transforms(),
        transforms.ToImage(mean=mean, std=std),
    )


def _local_transform(mean, std):
    # Local views: small crops (10-40% of the frame) sample sub-structure —
    # spiral arms, bars, bulges, tidal features, companions — the multi-crop
    # signal that local structure should predict the global object.
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop(
            (LOCAL_SIZE, LOCAL_SIZE), scale=(0.1, 0.4)
        ),
        *_photometric_transforms(),
        transforms.ToImage(mean=mean, std=std),
    )


def lejepa_forward(self, batch, stage):
    out = {}
    images = batch.get("image")
    if stage == "fit":
        global_views = [batch[k]["image"] for k in batch if k.startswith("global")]
        local_views = [batch[k]["image"] for k in batch if k.startswith("local")]
        labels = next(
            batch[k]["label"]
            for k in batch
            if k.startswith("global") or k.startswith("local")
        )
        output: LeJEPAOutput = self.model.forward(
            global_views=global_views, local_views=local_views, images=images
        )
        out["label"] = labels.repeat(len(global_views))
    else:
        output: LeJEPAOutput = self.model.forward(images=images)
        out["label"] = batch["label"].long()

    out["loss"] = output.loss
    out["embedding"] = output.embedding
    out["projection"] = output.projection

    self.log(f"{stage}/sigreg", output.sigreg_loss, on_step=True, on_epoch=True,
             sync_dist=True)
    self.log(f"{stage}/inv", output.inv_loss, on_step=True, on_epoch=True,
             sync_dist=True)
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True,
             sync_dist=True)
    return out


def build_model(width: int, proj_dim: int, lamb: float, n_slices: int) -> LeJEPA:
    """One ladder rung: the timm ViT family re-tiled for Galaxy10 at ``width``.

    ``embed_dim``/``num_heads`` are overridden per rung (head_dim fixed at
    64, the mu-P convention of growing the head count); ``img_size=64,
    patch_size=8`` re-tile the same architecture for the 64px global crops
    (locals 32px -> 4x4 tokens via ``dynamic_img_size``). The projector
    downsamples to ``proj_dim`` — the SIGReg dimension.
    """
    assert width % 64 == 0, "width must be a multiple of head_dim=64"
    projector = MLP(
        in_channels=width,
        hidden_channels=[2048, 2048, proj_dim],
        norm_layer="batch_norm",
        activation_layer=nn.ReLU,
        inplace=True,
        dropout=0.0,
    )
    model = LeJEPA(
        "vit_small_patch16_224",
        projector=projector,
        lamb=lamb,
        n_slices=n_slices,
        embed_dim=width,
        num_heads=width // 64,
        img_size=GLOBAL_SIZE,
        patch_size=8,
    )
    return apply_mup(model, base_fanin=BASE_FANIN)


def build_data(args, data_dir):
    h5_path = download_galaxy10(data_dir)
    base = Galaxy10SDSS(h5_path)
    train_idx, val_idx = split_indices(base.labels, args.val_frac, args.split_seed)
    mean, std = load_or_compute_norm_stats(
        base, train_idx, h5_path, args.val_frac, args.split_seed
    )

    n_global, n_views = 2, 8
    train_transform = transforms.MultiViewTransform(
        {
            **{f"global_{i}": _global_transform(mean, std) for i in range(n_global)},
            **{
                f"local_{i}": _local_transform(mean, std)
                for i in range(n_views - n_global)
            },
        }
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((GLOBAL_SIZE, GLOBAL_SIZE)),
        transforms.ToImage(mean=mean, std=std),
    )

    train_subset = torch.utils.data.Subset(base, train_idx.tolist())
    val_subset = torch.utils.data.Subset(base, val_idx.tolist())
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                train_subset, names=["image", "label"], transform=train_transform
            ),
            batch_size=args.batch_size, num_workers=args.num_workers,
            drop_last=True, persistent_workers=args.num_workers > 0, shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                val_subset, names=["image", "label"], transform=val_transform
            ),
            batch_size=256, num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        ),
    )
    return data


def train_once(args, lamb, data_dir, sweep_mode=False):
    """Build everything fresh and train one run at the given lambda.

    Returns the two witness callbacks so their ``.history`` can be read
    (proj space is the certification space — where SIGReg acts). In
    ``sweep_mode`` checkpointing and loggers are disabled.
    """
    pl.seed_everything(args.seed, workers=True)
    data = build_data(args, data_dir)
    model = build_model(args.width, args.proj_dim, lamb, args.n_slices)

    module = spt.Module(
        model=model,
        forward=lejepa_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": args.base_lr,
                "weight_decay": 0.05,
                "betas": (0.9, 0.999),
                # mu-P per-layer learning rates (schedulers preserve the
                # per-group ratios). Incompatible with FSDP2/DeepSpeed — see
                # mup_param_groups docstring.
                "params": mup_param_groups(
                    model, base_lr=args.base_lr,
                    weight_decay=0.05, base_fanin=BASE_FANIN,
                ),
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                # Absolute step counts, sharded by device count (a fraction
                # is misread as an absolute step when epochs <= 10).
                "peak_step": min(
                    min(10, max(1, args.epochs // 10))
                    * (len(data.train) // args.devices),
                    # cosine phase needs T_max >= 1 (guards --epochs 1)
                    (len(data.train) // args.devices) * args.epochs - 1,
                ),
                "start_factor": 0.01,
                "end_lr": args.base_lr / 1000,
                "total_steps": (len(data.train) // args.devices) * args.epochs,
            },
            "interval": "step",
        },
    )

    # The certificate, in both spaces. Judge lambda on the PROJ-space
    # witnesses (where SIGReg acts); emb-space r_eff is the quantity
    # expected to track probe accuracy.
    witness_proj = WitnessCallback(
        name="witness_proj", target="projection",
        queue_length=2048, target_shape=args.proj_dim,
    )
    witness_emb = WitnessCallback(
        name="witness_emb", target="embedding",
        queue_length=2048, target_shape=model.embed_dim,
    )
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger

        pl_logger = WandbLogger(
            project=args.wandb,
            name=f"galaxy10-lejepa-mup-w{args.width}-d{args.proj_dim}"
                 f"-B{args.batch_size}-lam{lamb:.4g}",
        )
    else:
        # Default logger for single runs; none during a sweep (an
        # N-candidate sweep should not scatter N anonymous log dirs).
        pl_logger = not sweep_mode
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        num_sanity_val_steps=0,
        enable_checkpointing=not sweep_mode,
        logger=pl_logger,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(model.embed_dim, NUM_CLASSES),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(
                        NUM_CLASSES
                    ),
                },
                optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 1e-6},
            ),
            witness_proj,
            witness_emb,
        ],
        precision="16-mixed",
    )
    trainer.fit(module, datamodule=data)
    return witness_proj, witness_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=REF_WIDTH,
                    help="encoder embed_dim (one rung of the ladder)")
    ap.add_argument("--proj-dim", type=int, default=REF_PROJ_DIM,
                    help="SIGReg dimension = projector output dim")
    ap.add_argument("--lamb-ref", type=float, default=0.02,
                    help="additive lambda tuned at the REFERENCE rung; "
                         "rescaled here by the master rule")
    ap.add_argument("--sweep-lamb", type=str, default=None,
                    help="comma-separated lambda grid, e.g. "
                         "'0.005,0.02,0.08,0.3'. Runs the Step-0 sweep AT "
                         "THIS config (raw additive lambdas, not rescaled) "
                         "and reports the recommended lambda_ref for this "
                         "(width, proj_dim, batch_size) as the reference "
                         "rung.")
    ap.add_argument("--base-lr", type=float, default=4e-4,
                    help="mu-P base lr (transfers as-is across widths)")
    ap.add_argument("--batch-size", type=int, default=REF_BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=100,
                    help="per run; for --sweep-lamb keep it long enough "
                         "that the witnesses equilibrate (the certifier "
                         "rejects drifting rows as unstable)")
    ap.add_argument("--n-slices", type=int, default=1024,
                    help="M; keep FIXED between Step 0 and the ladder")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of EACH class held out for validation; "
                         "Galaxy10 has no official split — keep (val_frac, "
                         "split_seed) fixed to align with prior runs")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed of the deterministic stratified split")
    ap.add_argument("--ref-width", type=int, default=REF_WIDTH,
                    help="reference-rung width the anchor was tuned at")
    ap.add_argument("--ref-proj-dim", type=int, default=REF_PROJ_DIM,
                    help="reference-rung proj_dim the anchor was tuned at")
    ap.add_argument("--ref-batch-size", type=int, default=REF_BATCH_SIZE,
                    help="reference-rung batch size the anchor was tuned at")
    ap.add_argument("--devices", type=int, default=1,
                    help="GPUs; the sweep requires 1 (witness history is "
                         "rank-0-local and DDP re-executes the script)")
    ap.add_argument("--wandb", type=str, default=None,
                    help="wandb project name; enables WandbLogger (also "
                         "during --sweep-lamb: one named run per candidate, "
                         "so witness trajectories are inspectable). All "
                         "witness/probe/loss metrics flow to the logger via "
                         "pl_module.log.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    data_dir = get_data_dir("galaxy10")

    if args.sweep_lamb is not None:
        from stable_pretraining.lambda_sweep import run_lambda_sweep

        assert args.devices == 1, "--sweep-lamb requires --devices 1"
        grid = [float(x) for x in args.sweep_lamb.split(",") if x.strip()]
        print(
            f"Step-0 lambda sweep at (width={args.width}, "
            f"proj_dim={args.proj_dim}, B={args.batch_size}): grid {grid}"
        )
        result = run_lambda_sweep(
            lambda lam: train_once(args, lam, data_dir, sweep_mode=True)[
                0
            ].history,
            grid,
        )
        rec = result["selection"]["recommended"]
        if rec is not None:
            print(
                "\nLadder command for each rung (the --ref-* values pin the "
                "anchor to THIS sweep's config):\n"
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
    train_once(args, lamb, data_dir)


if __name__ == "__main__":
    main()
