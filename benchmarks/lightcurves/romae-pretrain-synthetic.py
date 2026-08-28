"""Minimal CPU-runnable RoMAE pretraining demo on synthetic async light curves.

Intent: a dry run of the pipeline we will later point at ELAsTiCC-scale
photometric time series. Each object is an **asynchronous** multiband flux
sequence — every band has its own independent epochs and epoch count, with
no alignment or imputation. Each observation becomes one token via
:func:`stable_pretraining.backbone.romae.tokenize_lightcurves`, which emits
values ``[B, N, 1, 1, 1]`` (BTCHW for ``(1, 1, 1)`` tubelets), positions in
the model's ``[B, n_pos_dims, N]`` layout with axis 0 = time and axis 1 =
**log effective wavelength** (default; ``--band-encoding index`` for the
integer-index ablation), and a pad mask for the ragged lengths. The
n-dimensional continuous RoPE then attends jointly over time and wavelength,
so cross-band attention distance reflects spectral distance and the position
space is survey-transferable.

The RoMAE ``Encoder`` is also usable as a LeJEPA backbone, with two caveats:
its forward returns the full token sequence ``[B, N, d_model]`` — pool it
yourself (masked mean over non-padding tokens, or prepend a learned CLS
token via :class:`RoMAEBase`-style wiring at position 0) before applying
SIGReg — and the shared rotary module caches sin/cos per forward, so bare-
``Encoder`` use requires ``rope.reset_cache()`` between forwards (the full
model classes handle this internally). Monitor embedding health with
``stable_pretraining.callbacks.WitnessCallback`` as in the ladder scripts.

Synthetic data, per object and band: ``n_b ~ U{10..40}`` epochs at
independent uniform-random times over [0, 1]; flux a per-object
random-frequency sinusoid with a wavelength-dependent offset plus noise,
standardized per object.

Example:
    python benchmarks/lightcurves/romae-pretrain-synthetic.py --steps 50
"""

import argparse
import math

import torch
from loguru import logger

from stable_pretraining.backbone.romae import (
    LSST_BAND_WAVELENGTHS,
    RoMAEForPreTraining,
    gen_mask,
    tokenize_lightcurves,
)

N_BANDS = 6
N_OBJECTS = 128
BATCH_SIZE = 16
MIN_EPOCHS_PER_BAND, MAX_EPOCHS_PER_BAND = 10, 40
NOISE_STD = 0.1
LR = 3e-3


def make_synthetic_lightcurves(
    n_objects: int, generator: torch.Generator, band_encoding: str = "wavelength"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate padded ASYNC synthetic multiband light curves.

    Each band draws its own independent epoch count and epoch times — the
    bands share no time grid, exercising the paper's asynchronous
    multivariate claim rather than a synchronized special case.

    Args:
        n_objects: Number of objects (rows) to generate.
        generator: CPU RNG for reproducibility.
        band_encoding: ``"wavelength"`` (log effective wavelength positions,
            LSST ugrizy) or ``"index"`` (integer band index; ablation).

    Returns:
        ``(values [B, N_max, 1, 1, 1], positions [B, 2, N_max],
        pad_mask [B, N_max])`` from :func:`tokenize_lightcurves`.
    """
    times, values, bands = [], [], []
    for _ in range(n_objects):
        freq = 1.0 + 7.0 * torch.rand(1, generator=generator)
        phase = 2.0 * math.pi * torch.rand(1, generator=generator)
        t_parts, v_parts, b_parts = [], [], []
        for band in range(N_BANDS):
            n_b = int(
                torch.randint(
                    MIN_EPOCHS_PER_BAND, MAX_EPOCHS_PER_BAND + 1, (1,),
                    generator=generator,
                )
            )
            t_b = torch.rand(n_b, generator=generator)  # independent per band
            # Wavelength-dependent offset so the band position axis carries
            # physically-flavored signal.
            lam = LSST_BAND_WAVELENGTHS[band]
            offset = 0.3 * math.log(lam / 622.0)
            flux = (
                torch.sin(2.0 * math.pi * freq * t_b + phase)
                + offset
                + NOISE_STD * torch.randn(n_b, generator=generator)
            )
            t_parts.append(t_b)
            v_parts.append(flux)
            b_parts.append(torch.full((n_b,), band, dtype=torch.long))
        t = torch.cat(t_parts)
        v = torch.cat(v_parts)
        v = (v - v.mean()) / (v.std() + 1e-8)
        times.append(t)
        values.append(v)
        bands.append(torch.cat(b_parts))

    wavelengths = LSST_BAND_WAVELENGTHS if band_encoding == "wavelength" else None
    return tokenize_lightcurves(times, values, bands, band_wavelengths=wavelengths)


def build_model(width: int, depth: int) -> RoMAEForPreTraining:
    """Build a small RoMAEForPreTraining for (time, wavelength) positions."""
    nhead = max(2, width // 16)
    if (width // nhead) % 4 != 0:
        raise ValueError(
            f"width={width} with nhead={nhead} gives head_dim={width // nhead}; "
            "need head_dim divisible by 4 (2 position axes x 2 rotation halves)."
        )
    return RoMAEForPreTraining(
        encoder_kwargs=dict(d_model=width, nhead=nhead, depth=depth),
        decoder_kwargs=dict(d_model=max(16, width // 2), nhead=2, depth=2),
        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=2,
        p_rope_val=0.75,
    )


def main(args: argparse.Namespace) -> list[float]:
    """Run the pretraining loop and return the per-step losses."""
    if not 0.0 < args.mask_ratio <= 1.0:
        raise ValueError(
            f"--mask-ratio must be in (0, 1], got {args.mask_ratio} "
            "(0 masks nothing, so there is no reconstruction loss)"
        )
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    values, positions, pad_mask = make_synthetic_lightcurves(
        N_OBJECTS, generator, band_encoding=args.band_encoding
    )
    logger.info(
        f"Synthetic dataset: {N_OBJECTS} objects, padded length "
        f"{values.shape[1]}, band encoding: {args.band_encoding}"
    )

    model = build_model(args.width, args.depth)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"RoMAEForPreTraining: width={args.width} depth={args.depth} "
                f"({n_params / 1e3:.0f}K params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    losses: list[float] = []
    for step in range(args.steps):
        idx = torch.randint(0, N_OBJECTS, (BATCH_SIZE,), generator=generator)
        batch_pad = pad_mask[idx]
        mask = gen_mask(args.mask_ratio, batch_pad, generator=generator)
        _, loss = model(values[idx], mask, positions[idx], pad_mask=batch_pad)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 5 == 0 or step == args.steps - 1:
            logger.info(f"step {step:4d} | loss {losses[-1]:.4f}")
    return losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RoMAE pretraining on synthetic async multiband light curves"
    )
    parser.add_argument("--width", type=int, default=64, help="encoder d_model")
    parser.add_argument("--depth", type=int, default=2, help="encoder depth")
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--band-encoding", choices=["wavelength", "index"],
                        default="wavelength",
                        help="band position axis: log effective wavelength "
                             "(physical, survey-transferable) or integer "
                             "index (ablation)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
