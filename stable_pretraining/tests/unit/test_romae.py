"""Unit tests for the vendored RoMAE backbone (stable_pretraining.backbone.romae)."""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.backbone.romae import (
    LSST_BAND_WAVELENGTHS,
    Encoder,
    NDPRope,
    RoMAEForClassification,
    RoMAEForPreTraining,
    _get_attn_mask,
    gen_mask,
    tokenize_lightcurves,
)

pytestmark = pytest.mark.unit


def _sorted_times(b: int, n: int, seed: int) -> torch.Tensor:
    """Random sorted float times in [0, 1], distinct per batch row: [b, n]."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, n, generator=g).sort(dim=1).values


def _band_positions(b: int, n: int, seed: int) -> torch.Tensor:
    """Random band indices 0..5 as floats: [b, n]."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 6, (b, n), generator=g).float()


def test_encoder_irregular_continuous_positions():
    """Encoder forward on irregular 1-d continuous positions: shape + finiteness."""
    torch.manual_seed(0)
    b, n, d_model, nhead = 2, 17, 64, 4
    encoder = Encoder(d_model=d_model, nhead=nhead, depth=2).eval()
    rope = NDPRope(head_dim=d_model // nhead, n_dims=1, p=0.75)

    values = torch.randn(b, n, 1)  # [B, N, C=1]
    x = nn.Linear(1, d_model)(values)  # token embeddings [B, N, d_model]
    # Times as [B, N, 1], permuted to the model's [B, n_pos_dims, N] layout.
    positions = _sorted_times(b, n, seed=1).unsqueeze(-1).permute(0, 2, 1)

    out = encoder(x, positions, rope)
    assert out.shape == (b, n, d_model)
    assert torch.isfinite(out).all()


def test_encoder_two_dim_positions():
    """Positions of dim 2 (time, band) exercise the n-dimensional RoPE path."""
    torch.manual_seed(0)
    b, n, d_model, nhead = 2, 17, 64, 4
    encoder = Encoder(d_model=d_model, nhead=nhead, depth=2).eval()
    rope = NDPRope(head_dim=d_model // nhead, n_dims=2, p=0.75)

    x = torch.randn(b, n, d_model)
    times = _sorted_times(b, n, seed=1)
    bands = _band_positions(b, n, seed=2)
    positions = torch.stack([times, bands], dim=1)  # [B, 2, N]

    out = encoder(x, positions, rope)
    assert out.shape == (b, n, d_model)
    assert torch.isfinite(out).all()

    # The second positional axis must actually matter: permuting the bands
    # (while keeping times fixed) has to change the output.
    rope.reset_cache()
    positions_flipped = torch.stack([times, bands.flip(dims=[1])], dim=1)
    out_flipped = encoder(x, positions_flipped, rope)
    assert not torch.allclose(out, out_flipped)


def _pretrain_model() -> RoMAEForPreTraining:
    return RoMAEForPreTraining(
        encoder_kwargs=dict(d_model=32, nhead=4, depth=2),
        decoder_kwargs=dict(d_model=16, nhead=2, depth=1),
        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=2,
    )


def _pretrain_inputs(b: int = 2, n: int = 24):
    values = torch.randn(b, n, 1, 1, 1)  # BTCHW with (1, 1, 1) tubelets
    times = _sorted_times(b, n, seed=3)
    bands = _band_positions(b, n, seed=4)
    positions = torch.stack([times, bands], dim=1)  # [B, 2, N]
    pad_mask = torch.zeros(b, n, dtype=torch.bool)
    return values, positions, pad_mask


def test_pretraining_loss_backprop_and_masking():
    """Pretraining loss is finite, reaches encoder params, and depends on the mask."""
    torch.manual_seed(0)
    model = _pretrain_model().eval()
    values, positions, pad_mask = _pretrain_inputs()

    g = torch.Generator().manual_seed(0)
    mask_25 = gen_mask(0.25, pad_mask, generator=g)
    g = torch.Generator().manual_seed(0)
    mask_75 = gen_mask(0.75, pad_mask, generator=g)
    assert mask_75.sum() > mask_25.sum()

    logits, loss_25 = model(values, mask_25, positions, pad_mask=pad_mask)
    assert loss_25 is not None
    assert torch.isfinite(loss_25)
    assert logits.shape[1] == int(mask_25.sum(dim=1)[0])

    loss_25.backward()
    wq = model.encoder.layers[0].attention.wq.weight
    assert wq.grad is not None
    assert torch.isfinite(wq.grad).all()
    assert wq.grad.abs().sum() > 0

    # Masking actually masks: a different mask ratio changes the loss
    # (fixed seed, same model and inputs).
    model.zero_grad()
    _, loss_75 = model(values, mask_75, positions, pad_mask=pad_mask)
    assert torch.isfinite(loss_75)
    assert not torch.allclose(loss_25, loss_75)


def test_classification_output_shape():
    """RoMAEForClassification returns [B, num_classes] logits; loss needs labels."""
    torch.manual_seed(0)
    b, num_classes = 2, 5
    model = RoMAEForClassification(
        dim_output=num_classes,
        encoder_kwargs=dict(d_model=32, nhead=4, depth=2),
        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=2,
    ).eval()
    values, positions, pad_mask = _pretrain_inputs(b=b)

    logits, loss = model(values, positions, pad_mask=pad_mask)
    assert logits.shape == (b, num_classes)
    assert torch.isfinite(logits).all()
    assert loss is None

    labels = torch.randint(0, num_classes, (b,))
    _, loss = model(values, positions, pad_mask=pad_mask, label=labels)
    assert torch.isfinite(loss)


def test_global_position_shift_invariance():
    """Encoder output is (near-)invariant to a global position shift.

    RoPE only enters attention through q/k inner products, where the rotations
    combine into R(p_j - p_i) — purely relative. The p-RoPE truncation does
    not break this: truncated channels get an infinite timescale, i.e. angle
    zero (identity rotation) for EVERY position, so they are unaffected by any
    shift. Exact invariance therefore holds in exact arithmetic; the tolerance
    below only absorbs float32 error from evaluating sin/cos at shifted angles.
    """
    torch.manual_seed(0)
    b, n, d_model, nhead = 2, 17, 32, 4
    encoder = Encoder(d_model=d_model, nhead=nhead, depth=2).eval()
    rope = NDPRope(head_dim=d_model // nhead, n_dims=1, p=0.75)

    x = torch.randn(b, n, d_model)
    positions = _sorted_times(b, n, seed=5)[:, None, :]  # [B, 1, N]

    out = encoder(x, positions, rope)
    rope.reset_cache()  # sin/cos cache is per-forward; must reset before reuse
    out_shifted = encoder(x, positions + 10.0, rope)

    assert torch.allclose(out, out_shifted, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Asynchronous multivariate support: tokenizer, padding, loss masking
# ---------------------------------------------------------------------------


def _ragged_objects(seed: int = 0):
    """Three objects with per-band-independent epochs and unequal lengths."""
    g = torch.Generator().manual_seed(seed)
    times, values, bands = [], [], []
    for lengths in ([5, 3, 7], [11, 2], [4, 4, 4, 4]):
        t_parts, b_parts = [], []
        for band, n_b in enumerate(lengths):
            t_parts.append(torch.rand(n_b, generator=g))
            b_parts.append(torch.full((n_b,), band, dtype=torch.long))
        t = torch.cat(t_parts)
        times.append(t)
        values.append(torch.randn(t.numel(), generator=g))
        bands.append(torch.cat(b_parts))
    return times, values, bands


def test_tokenize_lightcurves_async_ragged():
    """Ragged async objects: shapes, sorting, wavelength math, determinism."""
    times, values, bands = _ragged_objects()
    lengths = [t.numel() for t in times]
    n_max = max(lengths)

    vals, pos, pad = tokenize_lightcurves(
        times, values, bands, band_wavelengths=LSST_BAND_WAVELENGTHS
    )
    assert vals.shape == (3, n_max, 1, 1, 1)
    assert pos.shape == (3, 2, n_max)
    assert pad.shape == (3, n_max)
    # Pad mask marks exactly the trailing padding of each row.
    for i, n in enumerate(lengths):
        assert (~pad[i]).sum() == n
        assert not pad[i, :n].any() and pad[i, n:].all()
        # Time axis sorted within the valid prefix (positions carry +1).
        t_axis = pos[i, 0, :n]
        assert (t_axis[1:] >= t_axis[:-1]).all()
        # Padding positions are exactly zero (CLS offset not applied there).
        assert (pos[i, :, n:] == 0).all()

    # Wavelength math: log(lambda / geometric mean) + 1 for a known token.
    lams = torch.tensor(list(LSST_BAND_WAVELENGTHS.values()))
    ref = lams.log().mean().exp()
    i = 1  # object 1 observes only bands 0 and 1
    order = torch.argsort(times[i])
    b_sorted = bands[i][order]
    expected = torch.log(
        torch.tensor([LSST_BAND_WAVELENGTHS[int(b)] for b in b_sorted]) / ref
    ) + 1.0
    torch.testing.assert_close(pos[i, 1, : lengths[i]], expected)

    # Determinism.
    vals2, pos2, pad2 = tokenize_lightcurves(
        times, values, bands, band_wavelengths=LSST_BAND_WAVELENGTHS
    )
    torch.testing.assert_close(vals, vals2)
    torch.testing.assert_close(pos, pos2)
    assert (pad == pad2).all()

    # Index-mode ablation: axis 1 is the raw band id (+1 CLS offset).
    _, pos_idx, _ = tokenize_lightcurves(times, values, bands)
    torch.testing.assert_close(
        pos_idx[i, 1, : lengths[i]], b_sorted.float() + 1.0
    )

    # Missing wavelength raises with the band named.
    with pytest.raises(KeyError):
        tokenize_lightcurves(times, values, bands, band_wavelengths={0: 500.0})


def test_padding_equivalence_encoder():
    """A padded row attends identically to its unpadded computation.

    Runs the bare Encoder on (a) object A alone at its true length and
    (b) object A padded alongside a longer object B, with the additive
    attention mask built from the pad mask. The valid-prefix outputs must
    match — padding must neither attend nor be attended to.
    """
    torch.manual_seed(3)
    d_model, nhead = 64, 4
    n_a, n_b = 9, 15
    encoder = Encoder(d_model=d_model, nhead=nhead, depth=2).eval()
    embed = nn.Linear(1, d_model)

    va = torch.randn(1, n_a, 1)
    vb = torch.randn(1, n_b, 1)
    ta = _sorted_times(1, n_a, seed=4)
    tb = _sorted_times(1, n_b, seed=5)

    # (a) unpadded forward of A.
    rope = NDPRope(head_dim=d_model // nhead, n_dims=1, p=0.75)
    out_a = encoder(embed(va), ta.unsqueeze(1), rope)

    # (b) A padded to len(B) in a 2-row batch, additive mask from pad mask.
    va_pad = torch.cat([va, torch.zeros(1, n_b - n_a, 1)], dim=1)
    ta_pad = torch.cat([ta, torch.zeros(1, n_b - n_a)], dim=1)
    x = embed(torch.cat([va_pad, vb], dim=0))
    positions = torch.stack([torch.cat([ta_pad, tb], dim=0)], dim=1)
    pad_mask = torch.zeros(2, n_b, dtype=torch.bool)
    pad_mask[0, n_a:] = True
    attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)
    rope.reset_cache()
    out_batch = encoder(x, positions, rope, attn_mask=attn_mask)

    torch.testing.assert_close(
        out_batch[0, :n_a], out_a[0], rtol=1e-4, atol=1e-5
    )


def test_pretraining_loss_ignores_padding_values():
    """Mutating padded value entries must not change the pretraining loss."""
    torch.manual_seed(6)
    times, values, bands = _ragged_objects(seed=7)
    vals, pos, pad = tokenize_lightcurves(
        times, values, bands, band_wavelengths=LSST_BAND_WAVELENGTHS
    )
    model = RoMAEForPreTraining(
        encoder_kwargs=dict(d_model=32, nhead=2, depth=1),
        decoder_kwargs=dict(d_model=16, nhead=2, depth=1),
        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=2,
    ).eval()
    g = torch.Generator().manual_seed(8)
    mask = gen_mask(0.5, pad, generator=g)

    _, loss1 = model(vals, mask, pos, pad_mask=pad)
    vals_mutated = vals.clone()
    vals_mutated[pad[:, :, None, None, None].expand_as(vals)] = 1e6
    _, loss2 = model(vals_mutated, mask, pos, pad_mask=pad)

    torch.testing.assert_close(loss1, loss2)
