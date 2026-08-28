"""LeJEPA ViT-S/16 on ImageNet-10 (Imagenette) with mu-P and witness probes.

Variant of ``lejepa-vit-small-quick.py`` that

1. re-parametrizes the model with mu-P (``apply_mup``: spectral init on all
   Linear weights, 1/head_dim attention logits) so that ``BASE_LR`` tuned
   here transfers across encoder width,
2. trains with per-layer mu-P learning rates (``mup_param_groups`` wired
   through the optimizer config's ``"params"`` key), and
3. monitors distribution health in BOTH spaces with ``WitnessCallback``
   (embedding space, dim ``model.embed_dim``; projection space, dim 512).
"""

import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.callbacks import WitnessCallback
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import LeJEPA, LeJEPAOutput
from stable_pretraining.optim import apply_mup, mup_param_groups

# SIGReg weight. NOTE: the per-view SIGReg fix (statistic computed per view
# over the N batch samples and averaged over views, instead of pooled over
# all V*N correlated rows) changed the scale of the SIGReg term — any lambda
# tuned against the old pooled statistic is stale. Re-tune lambda when
# adapting this recipe; 0.02 is a starting point, not a transferred value.
LAMB = 0.02

# Learning rate at the mu-P base width (fan_in == BASE_FANIN). With
# mup_param_groups, every >=2-D weight trains at BASE_LR * BASE_FANIN / fan_in
# and 1-D/bias/norm/token parameters at BASE_LR.
BASE_LR = 4e-4
BASE_FANIN = 256


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
    # Detached projector outputs, consumed by the projection-space witness.
    out["projection"] = output.projection
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return out


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    num_gpus = torch.cuda.device_count() or 1
    batch_size = 128
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))
    global_views = 2
    all_views = 8

    data_dir = str(get_data_dir("imagenet10"))

    train_transform = transforms.MultiViewTransform(
        {
            **{f"global_{i}": _global_transform() for i in range(global_views)},
            **{
                f"local_{i}": _local_transform()
                for i in range(all_views - global_views)
            },
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
                "frgfm/imagenette",
                split="train",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=train_transform,
            ),
            batch_size=batch_size,
            num_workers=8,
            drop_last=True,
            persistent_workers=True,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="validation",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=val_transform,
            ),
            batch_size=batch_size,
            num_workers=8,
            persistent_workers=True,
        ),
    )

    model = LeJEPA(
        encoder_name="vit_small_patch16_224",
        lamb=LAMB,
        n_slices=1024,
        n_points=17,
    )
    # mu-P: spectral init on Linear weights + 1/head_dim attention logits.
    apply_mup(model, base_fanin=BASE_FANIN)

    module = spt.Module(
        model=model,
        forward=lejepa_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                # Pre-built mu-P parameter groups: create_optimizer uses the
                # "params" key instead of the module's flat parameter list.
                # Each group carries its own lr (BASE_LR * BASE_FANIN/fan_in)
                # and weight_decay; 1-D/bias/norm/token params run at BASE_LR
                # with wd=0.
                "params": mup_param_groups(
                    model,
                    base_lr=BASE_LR,
                    weight_decay=0.05,
                    base_fanin=BASE_FANIN,
                ),
                "betas": (0.9, 0.999),
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": 2 / max_epochs,
                "start_factor": 0.01,
                "end_lr": BASE_LR / 100,
                "total_steps": (len(data.train) // num_gpus) * max_epochs,
            },
            "interval": "step",
        },
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
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
                    "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
                },
                optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 1e-6},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10000,
                metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
                input_dim=model.embed_dim,
                k=20,
            ),
            # Distribution-health witnesses in both spaces.
            WitnessCallback(
                name="witness_emb",
                target="embedding",
                queue_length=2048,
                target_shape=model.embed_dim,
            ),
            WitnessCallback(
                name="witness_proj",
                target="projection",
                queue_length=2048,
                target_shape=512,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.CSVLogger(
            save_dir=str(Path(__file__).parent / "logs"),
            name="lejepa-mup-witness-inet10",
        ),
        precision="16-mixed",
        enable_checkpointing=False,
        devices=num_gpus,
        accelerator="gpu",
    )

    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
