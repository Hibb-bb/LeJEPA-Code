"""LeJEPA width-transfer ladder on CIFAR-10: mu-P + the lambda master rule.

CIFAR-scale sibling of ``benchmarks/imagenet10/lejepa-mup-ladder.py`` (see
that script's docstring for the full protocol). One invocation either trains
ONE rung of the width ladder, or (with ``--sweep-lamb``) runs the automated
Step-0 lambda sweep at one configuration:

- **mu-P**: ``apply_mup`` + ``mup_param_groups`` — ``--base-lr`` is tuned
  once at the reference width and reused verbatim at every rung.
- **Master rule**: ``lambda(n, d, B) = lambda_ref * sqrt((B*d/n)/(B*d/n)_ref)``
  with ``n`` the encoder width, ``d`` the SIGReg dimension (``--proj-dim``,
  the projector output), ``B`` the batch size. ``--ref-*`` pin the anchor's
  config.
- **Certification**: ``--sweep-lamb`` trains each candidate briefly and
  certifies it from the proj-space witnesses (stable AND bounded), then
  recommends the smallest certified lambda with one grid step of margin.

CIFAR adaptations vs the ImageNet-10 script: the backbone is the same timm
ViT family re-tiled for 32x32 inputs (``img_size=32, patch_size=4`` -> 8x8
tokens; local crops 16x16 -> 4x4 tokens via ``dynamic_img_size``), the
augmentations follow this directory's SSL benchmarks (crop / flip / jitter /
grayscale — no blur or solarize at 32px), and data comes from torchvision
CIFAR-10 through ``FromTorchDataset`` like the other ``benchmarks/cifar10``
scripts.

Typical use::

    # Step 0 at the reference config
    python lejepa-mup-ladder.py --sweep-lamb "0.005,0.02,0.08,0.3" \
        --width 192 --epochs 50

    # Ladder rungs: same base lr, lambda rescaled from the anchor
    python lejepa-mup-ladder.py --width 384 --lamb-ref <anchor> \
        --ref-width 192 --ref-proj-dim 32 --ref-batch-size 256

NOTE on fused attention: timm's fused SDPA path ignores ``attn.scale``, so
``set_fused_attn(False)`` below is REQUIRED for the mu-P 1/head_dim patch to
take effect.
"""

import argparse
import math
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from timm.layers import set_fused_attn

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
REF_WIDTH = 192
REF_PROJ_DIM = 32
REF_BATCH_SIZE = 256

BASE_FANIN = 256  # mu-P normalization: fan_in at which lr mult == 1

GLOBAL_SIZE = 32  # full CIFAR frame
LOCAL_SIZE = 16   # multi-crop local views (multiple of patch_size=4)


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
    # CIFAR-scale recipe (see the SSL benchmarks in this directory):
    # no blur / solarize at 32x32.
    return [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
    ]


def _global_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((GLOBAL_SIZE, GLOBAL_SIZE), scale=(0.3, 1.0)),
        *_photometric_transforms(),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )


def _local_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((LOCAL_SIZE, LOCAL_SIZE), scale=(0.05, 0.3)),
        *_photometric_transforms(),
        transforms.ToImage(**spt.data.static.CIFAR10),
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
    """One ladder rung: the timm ViT family re-tiled for CIFAR at ``width``.

    ``embed_dim``/``num_heads`` are overridden per rung (head_dim fixed at
    64, the mu-P convention of growing the head count); ``img_size=32,
    patch_size=4`` re-tile the same architecture for CIFAR frames. The
    projector downsamples to ``proj_dim`` — the SIGReg dimension.
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
        patch_size=4,
    )
    return apply_mup(model, base_fanin=BASE_FANIN)


def build_data(args, data_dir):
    n_global, n_views = 2, 8
    train_transform = transforms.MultiViewTransform(
        {
            **{f"global_{i}": _global_transform() for i in range(n_global)},
            **{f"local_{i}": _local_transform() for i in range(n_views - n_global)},
        }
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((GLOBAL_SIZE, GLOBAL_SIZE)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )

    cifar_train = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True
    )
    cifar_val = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True
    )
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                cifar_train, names=["image", "label"], transform=train_transform
            ),
            batch_size=args.batch_size, num_workers=args.num_workers,
            drop_last=True, persistent_workers=args.num_workers > 0, shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                cifar_val, names=["image", "label"], transform=val_transform
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

        logger = WandbLogger(
            project=args.wandb,
            name=f"lejepa-mup-w{args.width}-d{args.proj_dim}"
                 f"-B{args.batch_size}-lam{lamb:.4g}",
        )
    else:
        # Default logger for single runs; none during a sweep (an
        # N-candidate sweep should not scatter N anonymous log dirs).
        logger = not sweep_mode
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        num_sanity_val_steps=0,
        enable_checkpointing=not sweep_mode,
        logger=logger,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(model.embed_dim, 10),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(10),
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

    data_dir = get_data_dir("cifar10")

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
