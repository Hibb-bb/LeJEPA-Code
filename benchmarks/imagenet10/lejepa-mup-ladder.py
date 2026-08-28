"""LeJEPA width-transfer ladder on ImageNet-10: mu-P + the lambda master rule.

The reference example for scaling experiments. One invocation either trains
ONE rung of the width ladder, or (with ``--sweep-lamb``) runs the automated
Step-0 lambda sweep at one configuration. The two scaling rules that make
hyperparameters transfer across rungs are applied automatically:

1. **mu-P (learning rate transfers as-is).** ``apply_mup`` re-initializes the
   encoder (spectral init, 1/head_dim attention logits) and
   ``mup_param_groups`` trains every weight at ``BASE_LR * BASE_FANIN /
   fan_in`` — so the SAME ``--base-lr`` (tuned once, at the proxy width) is
   used at every rung.

2. **The lambda master rule (lambda scales with the config).** The SIGReg
   weight is NOT transferred as-is: for the additive loss
   ``inv + lamb * sigreg`` used in this repo,

       lambda(n, d, B) = lambda_ref * sqrt( (B*d/n) / (B*d/n)_ref )

   where ``n`` is the encoder width, ``d`` the SIGReg dimension (the
   PROJECTOR OUTPUT dim — see below), and ``B`` the per-view batch size.
   ``lambda_ref`` is tuned once at the reference rung (Step 0: smallest
   lambda whose witness certificate passes, with one grid step of margin),
   and ``master_lambda`` rescales it for every other rung. Keep ``n_slices``
   (M) fixed between the Step-0 sweep and the ladder.

Where SIGReg lives — the projector output. The upstream default projector
ends at 512-d; the official minimal recipe downsamples to a SMALL proj_dim
(16). The theory's ``d`` is THIS dimension (it sets the equilibration clock
~ d^2, the sketch density M/d, and the sqrt(d) factor of the master rule),
so this script makes it an explicit knob (``--proj-dim``) and builds the
projector accordingly. For proportional-regime experiments set
``--proj-dim`` to ``width // gamma`` per rung instead of a constant.

Typical use (automated Step-0 sweep at the reference rung, then the ladder)::

    # Step 0: sweep lambda at the reference config; each candidate is
    # trained and certified from the proj-space witnesses (stable AND
    # bounded); the smallest certified lambda with one grid step of margin
    # is reported as the anchor.
    python lejepa-mup-ladder.py --sweep-lamb "0.005,0.02,0.08,0.3" \
        --width 384 --epochs 50

    # Ladder: same base lr at every rung (mu-P), lambda rescaled by the
    # master rule from the anchor. Pass the SAME --ref-* values as the
    # sweep config (they default to the module constants).
    python lejepa-mup-ladder.py --width 192 --lamb-ref <anchor>
    python lejepa-mup-ladder.py --width 768 --lamb-ref <anchor>

NOTE on fused attention: timm's fused SDPA path ignores ``attn.scale``, so
the mu-P 1/head_dim patch only takes effect with fused attention disabled —
``set_fused_attn(False)`` below is REQUIRED for mu-P-faithful runs.
"""

import argparse
import math
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from timm.layers import set_fused_attn

import stable_pretraining as spt
from stable_pretraining.backbone import MLP
from stable_pretraining.callbacks import WitnessCallback
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import LeJEPA, LeJEPAOutput
from stable_pretraining.optim import apply_mup, mup_param_groups

# mu-P attention needs the unfused path (see module docstring).
set_fused_attn(False)

# ---------------------------------------------------------------------------
# Reference rung: the (width, proj_dim, batch_size) at which lambda_ref and
# base_lr were tuned. Change these ONLY when you re-run Step 0.
# ---------------------------------------------------------------------------
REF_WIDTH = 384
REF_PROJ_DIM = 64
REF_BATCH_SIZE = 256

BASE_FANIN = 256  # mu-P normalization: fan_in at which lr mult == 1


def master_lambda(
    lamb_ref: float,
    width: int,
    proj_dim: int,
    batch_size: int,
    ref_width: int = REF_WIDTH,
    ref_proj_dim: int = REF_PROJ_DIM,
    ref_batch_size: int = REF_BATCH_SIZE,
) -> float:
    """lambda(n, d, B) = lambda_ref * sqrt((B*d/n) / (B*d/n)_ref).

    Specializations worth knowing:
    - fixed d and B, width ladder:      lambda ~ 1/sqrt(n)  (the classic rule)
    - proportional regime d = n/gamma:  lambda flat in scale
    - batch-size change at fixed (n,d): lambda ~ sqrt(B)
    """
    ratio = (batch_size * proj_dim / width) / (
        ref_batch_size * ref_proj_dim / ref_width
    )
    return lamb_ref * math.sqrt(ratio)


def _photometric_transforms():
    return [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
        transforms.RandomSolarize(threshold=128, p=0.2),
    ]


def _global_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
        *_photometric_transforms(),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def _local_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.3)),
        *_photometric_transforms(),
        transforms.ToImage(**spt.data.static.ImageNet),
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
    """One ladder rung: ViT at ``width`` with a projector ending at ``proj_dim``.

    The named config supplies patch size etc.; ``embed_dim``/``num_heads``
    are overridden per rung (head_dim fixed at 64, the mu-P convention of
    growing the head count). The projector downsamples to ``proj_dim`` — the
    SIGReg dimension, matching the official minimal recipe's shape (the
    upstream default would leave SIGReg in 512-d).
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
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette", split="train",
                revision="refs/convert/parquet", cache_dir=data_dir,
                transform=train_transform,
            ),
            batch_size=args.batch_size, num_workers=args.num_workers,
            drop_last=True, persistent_workers=args.num_workers > 0, shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette", split="validation",
                revision="refs/convert/parquet", cache_dir=data_dir,
                transform=val_transform,
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
    ``sweep_mode`` checkpointing and loggers are disabled (an N-candidate
    sweep should not leave N checkpoints and log dirs behind).
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
                # Absolute step counts, sharded by device count. A fraction
                # here would be misread as an absolute step when
                # 10/epochs >= 1 (epochs <= 10), silently collapsing warmup
                # to one step on exactly the short runs a sweep uses.
                "peak_step": min(10, max(1, args.epochs // 10))
                * (len(data.train) // args.devices),
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
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        num_sanity_val_steps=0,
        enable_checkpointing=not sweep_mode,
        logger=not sweep_mode,
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
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

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
                         "THIS config (values are raw additive lambdas here, "
                         "not rescaled), certifies each candidate from the "
                         "proj-space witnesses, and reports the recommended "
                         "lambda_ref for this (width, proj_dim, batch_size) "
                         "as the reference rung.")
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=16)
    args = ap.parse_args()

    data_dir = str(get_data_dir("imagenet10"))

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
