"""
- All pydantic / pydantic-settings config classes (``EncoderConfig``,
  ``RoMAEBaseConfig``, ``RoMAEForClassificationConfig``,
  ``RoMAEForPreTrainingConfig``) are replaced by plain constructor keyword
  arguments carrying the same defaults; the nested encoder/decoder configs
  become ``encoder_kwargs`` / ``decoder_kwargs`` dicts merged over
  ``ENCODER_DEFAULTS`` / ``DECODER_DEFAULTS``.
- No einops: ``patchify`` is rewritten with reshape/permute (the original
  einops pattern is quoted in a comment at the rewrite site).
- No safetensors/accelerate/wandb/nvidia-ml-py imports: the checkpoint
  helpers (``load_weights``, ``from_pretrained``, ``load_from_checkpoint``)
  are dropped — use ``torch.load`` / ``load_state_dict`` directly.
- ``gen_mask`` samples index subsets with ``torch.randperm`` (optionally
  seeded through a ``torch.Generator``) instead of Python's
  ``random.sample``; the distribution (uniform subsets without replacement)
  and the count-equalization logic are unchanged.
- Devices are generalized: masks, caches, and buffers follow the input
  tensor's device (upstream already threads a ``device`` argument; no CUDA
  hardcode remains).

The math is otherwise preserved exactly: rotary angle computation, p-RoPE
truncation (``inf`` timescales for NoPE channels), pretraining masking and
mask-count equalization, encoder/decoder wiring, and loss normalization
(per-tubelet target normalization, zeroed padding entries included in the
mean).
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default hyperparameters of the upstream ``EncoderConfig``.
ENCODER_DEFAULTS: dict = {
    "d_model": 342,
    "nhead": 8,
    "layer_norm_eps": 1e-12,
    "depth": 6,
    # To manually set the dimension of the MLP, change dim_feedforward.
    # If None it is chosen by multiplying d_model by mlp_ratio.
    "dim_feedforward": None,
    "mlp_ratio": 4.0,
    # Stochastic depth value.
    "drop_path_rate": 0.0,
    # Dropout to be applied throughout the Transformer.
    "hidden_drop_rate": 0.0,
    "attn_proj_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "pos_drop_rate": 0.0,
}


def get_encoder_size(size: str) -> dict:
    """Get the parameters of a specific RoMAE model encoder size.

    Args:
        size: One of ``RoMAE-tiny-shallow``, ``RoMAE-tiny``, ``RoMAE-small``,
            ``RoMAE-base``, ``RoMAE-large``.

    Returns:
        Dict of ``d_model``/``nhead``/``depth`` overrides.
    """
    sizes = {
        "RoMAE-tiny-shallow": {"d_model": 180, "nhead": 3, "depth": 2},
        "RoMAE-tiny": {"d_model": 180, "nhead": 3, "depth": 12},
        "RoMAE-small": {"d_model": 432, "nhead": 6, "depth": 12},
        "RoMAE-base": {"d_model": 720, "nhead": 12, "depth": 12},
        "RoMAE-large": {"d_model": 960, "nhead": 16, "depth": 24},
    }
    if size not in sizes:
        raise ValueError(f"Unknown encoder size: {size}")
    return sizes[size]


# Default decoder used by RoMAEForPreTraining upstream.
DECODER_DEFAULTS: dict = {**ENCODER_DEFAULTS, **get_encoder_size("RoMAE-tiny-shallow")}


def patchify(tubelet_size: tuple[int, int, int], x: torch.Tensor) -> torch.Tensor:
    """Convert a BTCHW input into a sequence of tubelets.

    Expected input format: BTCHW, e.g. a regular video could be
    ``(1, 32, 3, 244, 244)``.

    Args:
        tubelet_size: ``(p1, p2, p3)`` tubelet extent along T, H, W.
        x: Input tensor of shape ``[B, T, C, H, W]``.

    Returns:
        Tensor of shape ``[B, (T/p1)*(H/p2)*(W/p3), p1*p2*p3*C]``.
    """
    # einops: rearrange(x, 'b (t p1) c (h p2) (w p3) -> b (t h w) (p1 p2 p3 c)')
    b, T, c, H, W = x.shape
    p1, p2, p3 = tubelet_size
    t, h, w = T // p1, H // p2, W // p3
    x = x.reshape(b, t, p1, c, h, p2, w, p3)
    x = x.permute(0, 1, 4, 6, 2, 5, 7, 3)
    return x.reshape(b, t * h * w, p1 * p2 * p3 * c)


def prepare_positions(
    b: int, positions: tuple[Optional[torch.Tensor], ...]
) -> torch.Tensor:
    """Stack per-axis position tensors, reserving position zero for CLS.

    Take in a bunch of potentially None positions and replace None's with
    position zero (equating to no position in RoPE). This ensures that all
    position tensors have the same length (corresponding to the number of
    tokens). Additionally, move all positions forward by 1 and give the CLS
    token position zero.

    Args:
        b: Batch size.
        positions: Tuple of per-axis position tensors ``[B, N]`` (or None).

    Returns:
        Tensor of shape ``[B, n_axes, N]``.
    """
    if all([i is None for i in positions]):
        raise AttributeError(
            "All position tensors cannot be None, set at least one to a valid value!"
        )
    n_positions = 0
    device = None
    for i in positions:
        if i is not None:
            n_positions = i.shape[1]
            device = i.device
            break
    pos = []
    for i, p in enumerate(positions):
        if p is None:
            pos.append(torch.zeros((b, n_positions), device=device))
        else:
            pos.append(p + 1)
    return torch.stack(pos).permute(1, 0, 2)


def gen_mask(
    mask_ratio: float,
    pad_mask: torch.Tensor,
    single: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate a token mask for pre-training. True marks masked values.

    Non-padding tokens are assumed to sit at the front of each row. After
    uniformly sampling ``ceil(n_i * mask_ratio)`` maskable indices per row,
    trailing (padding) positions are additionally marked so that every row
    has the same total number of masked entries — required so the model's
    boolean-index reshapes stay rectangular.

    Args:
        mask_ratio: Fraction of (non-padding) tokens to mask out, in [0, 1].
        pad_mask: Bool tensor ``[B, N]``; True where the input is padding.
        single: If True, equalize against ``ceil(N * mask_ratio)`` instead of
            the per-batch max.
        generator: Optional CPU ``torch.Generator`` for reproducible sampling.

    Returns:
        Bool tensor ``[B, N]``; True where the token is masked.
    """
    if mask_ratio < 0 or mask_ratio > 1:
        raise ValueError(
            f"Mask ratio must be between 0 and 1, but was given {mask_ratio}"
        )

    ratio = mask_ratio
    per_sample_n = (~pad_mask).sum(dim=1)
    n_masked_per_sample = (per_sample_n * ratio).ceil().int()
    mask = torch.zeros(pad_mask.shape, dtype=torch.bool, device=pad_mask.device)
    for i in range(pad_mask.shape[0]):
        # Upstream: random.sample(range(per_sample_n[i]), n_masked_per_sample[i])
        idxs = torch.randperm(per_sample_n[i].item(), generator=generator)[
            : n_masked_per_sample[i].item()
        ]
        mask[i, idxs] = True
    if single:
        max_masked = torch.tensor(pad_mask.shape[1] * ratio).ceil().int()
    else:
        max_masked = n_masked_per_sample.max()
    diff_from_max = n_masked_per_sample - max_masked
    for i in range(diff_from_max.shape[0]):
        for j in range(pad_mask.shape[1] + diff_from_max[i], pad_mask.shape[1]):
            mask[i, j] = True

    return mask


# Implementation of stochastic depth from:
# Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
# Originally taken from timm (huggingface/pytorch-image-models).
def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample in residual main paths."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


def get_drop_path(drop_path_rate: float, layer_id: int, depth: int) -> nn.Module:
    """Calculate stochastic depth probability and return the initialized layer."""
    if drop_path_rate > 0.0:
        if layer_id == 0:
            return DropPath(0.0)
        return DropPath((layer_id / (depth - 1)) * drop_path_rate)
    return nn.Identity()


class DummyPosEmbedding(nn.Module):
    """To be used as a "None" positional embedding."""

    @staticmethod
    def forward(x, *_, **__):
        return x

    def reset_cache(self) -> None:
        """No-op; kept for interface parity with NDPRope."""


class AbsoluteSinCosine(nn.Module):
    """Standard absolute sin/cos positional encodings.

    Based on the original encodings used in "Attention is All You Need";
    see the PyTorch transformer tutorial.
    """

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def reset_cache(self) -> None:
        """No-op; kept for interface parity with NDPRope."""

    def forward(
        self,
        x: torch.Tensor,
        positions,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add absolute encodings to ``x`` ([B, N, D]), gathered through ``mask``."""
        if mask is None:
            return x + self.pe[None, : x.shape[1]].expand(x.shape[0], -1, -1)
        if mask[0].sum() != x.shape[1]:
            # Upstream prepends a CLS slot to the mask here (its intermediate
            # expression is computed and discarded there; the extended mask
            # feeds the return below — semantics preserved).
            mask = torch.cat(
                [torch.ones(x.shape[0], 1, device=mask.device, dtype=torch.bool), mask],
                dim=1,
            )
        return x + self.pe[None, : mask.shape[1], :].expand(x.shape[0], -1, -1)[
            mask
        ].reshape(x.shape[0], x.shape[1], -1)


class NDPRope(nn.Module):
    """N-dimensional continuous p-RoPE.

    The initial p-RoPE code was converted from the JAX implementation of
    "Round and Round We Go! What Makes Rotary Positional Encodings Useful?"
    (https://openreview.net/forum?id=GtvuNrk58a). The head dimension is split
    evenly across ``n_dims`` positional axes; within each axis, a fraction
    ``p`` of the rotation angles are active and the rest get an infinite
    timescale (angle 0 — i.e. NoPE channels).

    The sin/cos tables are cached on the first forward of a pass (queries)
    and reused for keys and deeper layers; call :meth:`reset_cache` after
    each full model forward (the RoMAE models do this automatically).
    """

    def __init__(
        self,
        head_dim: int,
        base: int = 10000,
        p: float = 1,
        n_dims: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if head_dim % n_dims != 0:
            raise AttributeError(
                f"The head dimension ({head_dim}) is not divisible by the "
                f"number of positional axis ({n_dims})!"
            )
        if 0 > p or p > 1:
            raise AttributeError(f"Provided p value ({p}) is not between 0 and 1!")

        self.axis_dim = head_dim // n_dims
        self.n_dims = n_dims

        rope_angles = int(p * self.axis_dim // 2)
        nope_angles = self.axis_dim // 2 - rope_angles

        fraction = 2.0 * torch.arange(0, rope_angles) / self.axis_dim
        self.register_buffer(
            "timescale",
            F.pad(base**fraction, (0, nope_angles), mode="constant", value=torch.inf),
        )

        self.cache: Optional[list] = None

    def reset_cache(self) -> None:
        """Clear the cached per-axis sin/cos tables."""
        self.cache = None

    def get_sin_cos(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sin/cos rotation tables for one positional axis ([B, N])."""
        sinusoid_inp = positions[..., None] / self.timescale[None, None, :]
        sinusoid_inp = sinusoid_inp[..., None, :]
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        return sin, cos

    def apply_ndprope(
        self, x: torch.Tensor, angles: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Rotate one axis-slice of the head dimension by the given angles."""
        sin, cos = angles
        first_half, second_half = torch.tensor_split(x, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        out = torch.concatenate([first_part, second_part], dim=-1)
        return out.to(x.dtype)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply the rotary encoding.

        Args:
            x: Tensor of shape ``[batch_size, seq_len, nhead, head_dim]``.
            positions: Tensor of shape ``[batch_size, n_dims, seq_len]``.
                For 3D positions this would be ``[batch_size, 3, seq_len]``.

        Returns:
            Rotated tensor with the same shape as ``x``.
        """
        B, seq_len, nhead, head_dim = x.shape

        if self.cache is None:
            self.cache = []
            for i in range(self.n_dims):
                self.cache.append(self.get_sin_cos(positions[:, i].reshape(B, -1)))

        views = []
        for i in range(self.n_dims):
            views.append(
                self.apply_ndprope(
                    x[..., self.axis_dim * i : self.axis_dim * (i + 1)],
                    self.cache[i],
                )
            )
        x = torch.cat(views, dim=-1)
        return x


def simplex_directions(n: int) -> torch.Tensor:
    """Unit vectors from the centroid to the ``n + 1`` vertices of a regular
    simplex in ``R^n`` (``[[1.0]]`` for ``n == 1``), as in nD-RoPE."""
    if n == 1:
        return torch.tensor([[1.0]])
    pts = torch.eye(n + 1)
    pts = pts - pts.mean(dim=0, keepdim=True)
    u, _, _ = torch.linalg.svd(pts.T, full_matrices=False)
    red = pts @ u[:, :-1]  # (n+1, n)
    return red / red.norm(dim=1, keepdim=True)


class SimplexRope(nn.Module):
    """nD-RoPE (Li et al. 2026, arXiv:2606.12146) over a group of position axes.

    Instead of one axis per channel block (axial RoPE), every rotation angle
    is the inner product ``omega^T x`` between the *whole* ``n``-dimensional
    position vector of the group and a wave vector ``omega``. The wave
    vectors are the ``M = n + 1`` centroid-to-vertex directions of a regular
    simplex (isotropic, full-rank coverage), replicated over ``S`` geometric
    scales ``theta^(-s/S)``, so the block has ``dim = 2 * M * S`` channels.
    Each head optionally gets its own random rotation of the simplex
    (``rotate=True``, as in the reference implementation), drawn once at
    construction from ``seed``.

    ``p`` mirrors p-RoPE: only the ``round(p * S)`` highest-frequency scales
    rotate, the remaining scales get magnitude 0 (NoPE channels).

    Args:
        dim: Channels of this block (must be a multiple of ``2 * (n + 1)``).
        n_axes: Number of position axes in the group.
        nhead: Number of attention heads (for per-head rotations).
        theta: Base of the geometric scale ladder.
        p: Fraction of scales that rotate.
        rotate: Random per-head rotation of the simplex.
        seed: RNG seed for the rotations.
    """

    def __init__(
        self,
        dim: int,
        n_axes: int,
        nhead: int,
        theta: float = 100.0,
        p: float = 1.0,
        rotate: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        m = n_axes + 1 if n_axes > 1 else 1
        if dim % (2 * m) != 0:
            raise ValueError(
                f"SimplexRope dim {dim} must be a multiple of 2*(n_axes+1)={2 * m}"
            )
        self.dim, self.n_axes, self.nhead, self.m = dim, n_axes, nhead, m
        self.n_scales = dim // (2 * m)
        base = simplex_directions(n_axes)  # (M, n)
        g = torch.Generator().manual_seed(seed)
        freqs = []
        for _ in range(nhead):
            if rotate and n_axes > 1:
                q, _ = torch.linalg.qr(torch.randn(n_axes, n_axes, generator=g))
                if torch.linalg.det(q) < 0:
                    q[:, 0] = -q[:, 0]
                freqs.append(base @ q.T)
            else:
                freqs.append(base)
        self.register_buffer("freqs", torch.stack(freqs))  # (H, M, n)
        s = torch.arange(self.n_scales, dtype=torch.float32)
        mag = theta ** (-s / max(self.n_scales, 1))
        n_active = int(round(p * self.n_scales))
        mag[n_active:] = 0.0
        self.register_buffer("mag", mag)  # (S,)
        self.cache: Optional[tuple] = None

    def reset_cache(self) -> None:
        self.cache = None

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate ``x [B, N, H, dim]`` by ``positions [B, n_axes, N]``."""
        b, n, h, d = x.shape
        if self.cache is None:
            pos = positions.transpose(1, 2).float()  # (B, N, n)
            proj = torch.einsum("bnd,hmd->bnhm", pos, self.freqs)  # (B,N,H,M)
            ang = proj[..., None] * self.mag  # (B,N,H,M,S)
            ang = ang.reshape(b, n, h, self.m * self.n_scales)
            self.cache = (torch.sin(ang), torch.cos(ang))
        sin, cos = self.cache
        first, second = torch.tensor_split(x, 2, dim=-1)
        out = torch.cat(
            [first * cos - second * sin, second * cos + first * sin], dim=-1
        )
        return out.to(x.dtype)


class BlockRope(nn.Module):
    """Rotary encoding with an explicit head-dimension layout.

    The head dimension is cut into contiguous channel blocks, each rotated
    by its own subset of the position axes and its own scheme:

    - ``kind="axial"`` — one axis, standard (p-)RoPE on that slice
      (:class:`NDPRope` with ``n_dims=1``). Several axial blocks of unequal
      width are allowed, which :class:`NDPRope` alone (equal split) is not.
    - ``kind="simplex"`` — a group of axes encoded jointly with
      :class:`SimplexRope` (nD-RoPE).

    ``blocks`` is a list of dicts ``{"kind", "axes", "dim", ...}`` whose
    ``dim`` values sum to ``head_dim``; extra keys are passed to the block
    constructor (``p``, ``base``, ``theta``, ``rotate``, ``seed``). Example
    for ``(time, l10, l50, l90)`` positions at ``head_dim=60``::

        [{"kind": "axial", "axes": [0], "dim": 36, "p": 0.75},
         {"kind": "simplex", "axes": [1, 2, 3], "dim": 24, "p": 0.75}]
    """

    def __init__(self, head_dim: int, nhead: int, blocks: list[dict]):
        super().__init__()
        self.head_dim = head_dim
        self.axes, self.dims, mods = [], [], []
        for spec in blocks:
            spec = dict(spec)
            kind, axes, dim = spec.pop("kind"), list(spec.pop("axes")), spec.pop("dim")
            if kind == "axial":
                if len(axes) != 1:
                    raise ValueError("axial block takes exactly one axis")
                mods.append(NDPRope(head_dim=dim, n_dims=1, **spec))
            elif kind == "simplex":
                mods.append(SimplexRope(dim, len(axes), nhead, **spec))
            else:
                raise ValueError(f"unknown rope block kind {kind!r}")
            self.axes.append(axes)
            self.dims.append(dim)
        if sum(self.dims) != head_dim:
            raise ValueError(
                f"rope block dims {self.dims} must sum to head_dim={head_dim}"
            )
        self.blocks = nn.ModuleList(mods)

    def reset_cache(self) -> None:
        for m in self.blocks:
            m.reset_cache()

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``x [B, N, H, head_dim]``, ``positions [B, n_pos_dims, N]``."""
        outs, lo = [], 0
        for mod, axes, dim in zip(self.blocks, self.axes, self.dims):
            outs.append(mod(x[..., lo:lo + dim], positions[:, axes]))
            lo += dim
        return torch.cat(outs, dim=-1)


def _get_inpt_pos_embedding(pos_encoding: str, d_model: int, max_len: int) -> nn.Module:
    """Return the positional encoding applied at the input, per config."""
    if pos_encoding == "absolute":
        return AbsoluteSinCosine(d_model=d_model, max_len=max_len)
    return DummyPosEmbedding()


def _get_attn_pos_embedding(
    pos_encoding: str,
    d_model: int,
    nhead: int,
    n_dims: int,
    p: float,
    rope_blocks: Optional[list] = None,
) -> nn.Module:
    """Return the positional encoding applied at each attention block."""
    if pos_encoding == "ropend":
        if rope_blocks is not None:
            return BlockRope(d_model // nhead, nhead, rope_blocks)
        return NDPRope(n_dims=n_dims, head_dim=d_model // nhead, p=p)
    return DummyPosEmbedding()


def _init_weights(m: nn.Module) -> None:
    """Initialize linear and RMSNorm weights (via ``self.apply(_init_weights)``)."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.RMSNorm):
        nn.init.constant_(m.weight, 1.0)


def _get_attn_mask(
    x_shape: tuple[int, ...], device, pad_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Generate an additive attention mask from a pad mask (zeros if None)."""
    if pad_mask is not None:
        attn_mask = torch.full(
            (x_shape[0], x_shape[1], x_shape[1]), float("-inf"), device=device
        )
        # The attention mask only needs to be applied to the columns (keys)
        attn_mask[~pad_mask] = 0
        attn_mask = attn_mask.permute([0, 2, 1])[:, None, ...]
    else:
        attn_mask = torch.zeros((1, x_shape[1], x_shape[1]), device=device)
    return attn_mask


class Attention(nn.Module):
    """Multi-head attention module (Llama-style, bias-free projections)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        attn_drop_rate: float = 0.0,
        attn_proj_drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.n_kv_heads = nhead
        self.head_dim = d_model // nhead
        # Explicit logit scale, passed to F.scaled_dot_product_attention below.
        # The default reproduces SDPA's standard 1/sqrt(head_dim); exposing it
        # as a float attribute (alongside the int ``head_dim``) lets
        # ``stable_pretraining.optim.apply_mup`` retune it to 1/head_dim for
        # mu-P width transfer. SDPA honours an explicit ``scale`` even on its
        # fused path, so no eager fallback is needed.
        self.scale = self.head_dim**-0.5
        self.attn_dropout_val = attn_drop_rate
        self.proj_dropout = nn.Dropout(attn_proj_drop_rate)
        self.pos_dropout = nn.Dropout(pos_drop_rate)
        self.wq = nn.Linear(d_model, nhead * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(nhead * self.head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        pos_emb: Optional[nn.Module] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention forward pass with optional rotary encoding of q/k."""
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if pos_emb is not None:
            xq = self.pos_dropout(pos_emb(xq, positions))
            xk = self.pos_dropout(pos_emb(xk, positions))

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            dropout_p=self.attn_dropout_val if self.training else 0.0,
            scale=self.scale,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.proj_dropout(self.wo(output))


class FeedForward(nn.Module):
    """SiLU feed-forward block (bias-free)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with rotary attention and stochastic depth."""

    def __init__(
        self,
        layer_id: int,
        d_model: int = ENCODER_DEFAULTS["d_model"],
        nhead: int = ENCODER_DEFAULTS["nhead"],
        depth: int = ENCODER_DEFAULTS["depth"],
        layer_norm_eps: float = ENCODER_DEFAULTS["layer_norm_eps"],
        dim_feedforward: Optional[int] = ENCODER_DEFAULTS["dim_feedforward"],
        mlp_ratio: float = ENCODER_DEFAULTS["mlp_ratio"],
        drop_path_rate: float = ENCODER_DEFAULTS["drop_path_rate"],
        hidden_drop_rate: float = ENCODER_DEFAULTS["hidden_drop_rate"],
        attn_drop_rate: float = ENCODER_DEFAULTS["attn_drop_rate"],
        attn_proj_drop_rate: float = ENCODER_DEFAULTS["attn_proj_drop_rate"],
        pos_drop_rate: float = ENCODER_DEFAULTS["pos_drop_rate"],
    ):
        super().__init__()
        self.n_heads = nhead
        self.dim = d_model
        self.head_dim = d_model // nhead
        self.attention = Attention(
            d_model=d_model,
            nhead=nhead,
            attn_drop_rate=attn_drop_rate,
            attn_proj_drop_rate=attn_proj_drop_rate,
            pos_drop_rate=pos_drop_rate,
        )
        self.drop_path = get_drop_path(drop_path_rate, layer_id, depth)

        hidden_dim = (
            dim_feedforward
            if dim_feedforward is not None
            else round(mlp_ratio * d_model)
        )
        self.feed_forward = FeedForward(
            dim=d_model, hidden_dim=hidden_dim, dropout=hidden_drop_rate
        )
        self.layer_id = layer_id
        self.attention_norm = nn.RMSNorm(d_model, eps=layer_norm_eps)
        self.ffn_norm = nn.RMSNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        pos_embed: Optional[nn.Module] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention and feed-forward with residuals."""
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), positions, pos_embed, mask)
        )
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Encoder(nn.Module):
    """RoMAE transformer encoder.

    Args:
        d_model: Embedding dimension.
        nhead: Number of attention heads.
        layer_norm_eps: RMSNorm epsilon.
        depth: Number of transformer blocks.
        dim_feedforward: Explicit MLP hidden size (overrides ``mlp_ratio``).
        mlp_ratio: MLP hidden size multiplier when ``dim_feedforward`` is None.
        drop_path_rate: Stochastic-depth rate.
        hidden_drop_rate: Dropout inside the feed-forward blocks.
        attn_proj_drop_rate: Dropout after the attention output projection.
        attn_drop_rate: Attention-probability dropout.
        pos_drop_rate: Dropout applied to rotary-encoded q/k.
    """

    def __init__(
        self,
        d_model: int = ENCODER_DEFAULTS["d_model"],
        nhead: int = ENCODER_DEFAULTS["nhead"],
        layer_norm_eps: float = ENCODER_DEFAULTS["layer_norm_eps"],
        depth: int = ENCODER_DEFAULTS["depth"],
        dim_feedforward: Optional[int] = ENCODER_DEFAULTS["dim_feedforward"],
        mlp_ratio: float = ENCODER_DEFAULTS["mlp_ratio"],
        drop_path_rate: float = ENCODER_DEFAULTS["drop_path_rate"],
        hidden_drop_rate: float = ENCODER_DEFAULTS["hidden_drop_rate"],
        attn_proj_drop_rate: float = ENCODER_DEFAULTS["attn_proj_drop_rate"],
        attn_drop_rate: float = ENCODER_DEFAULTS["attn_drop_rate"],
        pos_drop_rate: float = ENCODER_DEFAULTS["pos_drop_rate"],
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = depth

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id,
                    d_model=d_model,
                    nhead=nhead,
                    depth=depth,
                    layer_norm_eps=layer_norm_eps,
                    dim_feedforward=dim_feedforward,
                    mlp_ratio=mlp_ratio,
                    drop_path_rate=drop_path_rate,
                    hidden_drop_rate=hidden_drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    attn_proj_drop_rate=attn_proj_drop_rate,
                    pos_drop_rate=pos_drop_rate,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        pos_encoding: nn.Module,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run all transformer blocks.

        Args:
            x: Embeddings ``[B, N, d_model]``.
            positions: Continuous positions ``[B, n_pos_dims, N]``.
            pos_encoding: Shared rotary module (e.g. :class:`NDPRope`); its
                sin/cos cache persists across layers — reset it between
                forward passes with different positions.
            attn_mask: Additive attention mask (see :func:`_get_attn_mask`).
        """
        for layer in self.layers:
            x = layer(x, positions, pos_encoding, attn_mask)
        return x


class CLSClassifierHead(nn.Module):
    """Simple default head utilizing the CLS token for classification."""

    def __init__(
        self,
        d_model: int,
        d_output: int,
        layer_norm_eps: float,
        head_drop_rate: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate),
            nn.RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output),
        )

    def forward(self, x, *_, **__):
        # Take out the CLS token which is at position zero
        x = x[:, 0, :]
        return self.head(x)


class MeanClassifierHead(nn.Module):
    """Simple default head utilizing the mean of all tokens for classification."""

    def __init__(
        self,
        d_model: int,
        d_output: int,
        layer_norm_eps: float,
        head_drop_rate: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate),
            nn.RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output),
        )

    def forward(self, x, pad_mask: Optional[torch.Tensor] = None, *_, **__):
        # We only calculate the mean across non-padding values.
        if pad_mask is not None:
            n_unmasked = (~pad_mask).sum(dim=1)
            x[pad_mask] = 0
            x = x.sum(dim=1) / n_unmasked[:, None]
        else:
            x = x.mean(dim=1)
        return self.head(x)


class InterpolationHead(nn.Module):
    """Interpolation head predicting original tubelet values from MASK tokens."""

    def __init__(
        self,
        d_model: int,
        d_output: int,
        layer_norm_eps: float,
        head_drop_rate: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate),
            nn.RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output),
        )

    def forward(self, x, *_, **__):
        return self.head(x)


class RoMAEBase(nn.Module):
    """Base RoMAE model class with layers shared between all RoMAE models.

    Args:
        encoder_kwargs: Overrides merged over :data:`ENCODER_DEFAULTS`.
        use_cls: Whether to insert a learned CLS token at the start of the
            sequence.
        pos_encoding: ``"ropend"`` (continuous rotary) or ``"absolute"``.
        max_len: Maximum input length for precomputed absolute encodings.
        tubelet_size: ``(p1, p2, p3)`` tubelet extent along T, H, W.
        n_channels: Number of input channels.
        head_drop_rate: Dropout inside the prediction head.
        n_pos_dims: Number of positional axes (rows of ``positions``).
        p_rope_val: p-RoPE truncation fraction in [0, 1].
        rope_blocks: Optional explicit head-dim layout (see
            :class:`BlockRope`); ``None`` = equal axial split over
            ``n_pos_dims`` axes (:class:`NDPRope`).
    """

    def __init__(
        self,
        encoder_kwargs: Optional[dict] = None,
        use_cls: bool = True,
        pos_encoding: str = "ropend",
        max_len: int = 1500,
        tubelet_size: tuple[int, int, int] = (1, 1, 16),
        n_channels: int = 1,
        head_drop_rate: float = 0.0,
        n_pos_dims: int = 3,
        p_rope_val: float = 0.75,
        rope_blocks: Optional[list] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder_kwargs = {**ENCODER_DEFAULTS, **(encoder_kwargs or {})}
        self.use_cls = use_cls
        if pos_encoding not in ("ropend", "absolute"):
            # Upstream validated this via a pydantic Literal; without the
            # guard a typo silently selects DummyPosEmbedding (no positional
            # information at all).
            raise ValueError(
                f"pos_encoding must be 'ropend' or 'absolute', got "
                f"{pos_encoding!r}"
            )
        self.pos_encoding = pos_encoding
        self.max_len = max_len
        self.tubelet_size = tuple(tubelet_size)
        self.n_channels = n_channels
        self.head_drop_rate = head_drop_rate
        self.n_pos_dims = n_pos_dims
        self.p_rope_val = p_rope_val
        self.rope_blocks = rope_blocks

        self.loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
        self.loss_fn = None
        self.head = None

        self.inpt_pos_dropout = nn.Dropout(self.encoder_kwargs["pos_drop_rate"])

        # All models use the exact same encoder
        self.encoder: nn.Module = Encoder(**self.encoder_kwargs)
        # Projection from tubelets to embedding dimension
        proj_input_dim = (
            self.tubelet_size[0]
            * self.tubelet_size[1]
            * self.tubelet_size[2]
            * n_channels
        )
        self.projection = nn.Linear(proj_input_dim, self.encoder_kwargs["d_model"])

        # Classification token
        if use_cls:
            self.register_parameter(
                "cls", nn.Parameter(torch.zeros(self.encoder_kwargs["d_model"]))
            )
            nn.init.trunc_normal_(self.cls, std=0.02)
        else:
            self.cls = None

        # A useful zero buffer
        self.register_buffer("zeros", torch.zeros(1))

    def apply_head_loss(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor],
        pad_mask: Optional[torch.Tensor] = None,
    ):
        """Apply head and calculate loss."""
        logits = self.head(x, pad_mask=pad_mask)
        loss = self.get_loss(logits, label)
        return logits, loss

    def get_pos_embs(self, nhead: int, d_model: int) -> tuple[nn.Module, nn.Module]:
        """Build input and attention positional embeddings from stored settings."""
        inpt_pos_embedding = _get_inpt_pos_embedding(
            pos_encoding=self.pos_encoding, d_model=d_model, max_len=self.max_len
        )
        attn_pos_embedding = _get_attn_pos_embedding(
            pos_encoding=self.pos_encoding,
            d_model=d_model,
            nhead=nhead,
            n_dims=self.n_pos_dims,
            p=self.p_rope_val,
            rope_blocks=self.rope_blocks,
        )
        return inpt_pos_embedding, attn_pos_embedding

    def set_loss_fn(self, loss_fn) -> None:
        """Set the model loss function (accepts prediction and target tensors)."""
        self.loss_fn = loss_fn

    def set_head(self, head) -> None:
        """Set the model head (maps final embeddings to loss-function inputs)."""
        self.head = head

    def get_loss(
        self, logits: torch.Tensor, label: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Call the loss function if the label is not None."""
        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)
        return loss

    def add_cls(self, x, positions, pad_mask=None):
        """Prepend the CLS token (position zero, never padding) if enabled."""
        if self.cls is not None:
            # Add classification token to the beginning of all relevant tensors
            x = torch.cat((self.cls.expand(x.shape[0], 1, -1), x), dim=1)
            positions = torch.cat(
                [self.zeros.expand(x.shape[0], positions.shape[1], -1), positions],
                dim=2,
            )
            if pad_mask is not None:
                pad_mask = torch.cat(
                    [(self.zeros > 0.5).expand(x.shape[0], -1), pad_mask], dim=1
                )
        return x, positions, pad_mask


class RoMAEForPreTraining(RoMAEBase):
    """RoMAE masked-autoencoding pretraining model (encoder + light decoder).

    Args:
        encoder_kwargs: Encoder overrides merged over :data:`ENCODER_DEFAULTS`.
        decoder_kwargs: Decoder overrides merged over :data:`DECODER_DEFAULTS`
            (upstream default: ``RoMAE-tiny-shallow``).
        normalize_targets: Whether to normalize the target tubelet values.
            Normalization is done per-tubelet, therefore this should not be
            True when the tubelet size is very small like ``(1, 1, 1)``.
        **base_kwargs: See :class:`RoMAEBase`.
    """

    def __init__(
        self,
        encoder_kwargs: Optional[dict] = None,
        decoder_kwargs: Optional[dict] = None,
        normalize_targets: bool = False,
        **base_kwargs,
    ):
        super().__init__(encoder_kwargs=encoder_kwargs, **base_kwargs)
        self._normalize_targets = normalize_targets
        self.decoder_kwargs = {**DECODER_DEFAULTS, **(decoder_kwargs or {})}
        self.decoder = Encoder(**self.decoder_kwargs)
        # Projection from encoder embedding dimension to decoder
        # embedding dimension
        self.encoder_decoder_proj = nn.Linear(
            self.encoder_kwargs["d_model"], self.decoder_kwargs["d_model"]
        )

        self.encoder_inpt_pos_embedding, self.encoder_attn_pos_embedding = (
            self.get_pos_embs(
                nhead=self.encoder_kwargs["nhead"],
                d_model=self.encoder_kwargs["d_model"],
            )
        )
        self.decoder_inpt_pos_embedding, self.decoder_attn_pos_embedding = (
            self.get_pos_embs(
                nhead=self.decoder_kwargs["nhead"],
                d_model=self.decoder_kwargs["d_model"],
            )
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.decoder_kwargs["d_model"])
        )
        self.set_head(
            InterpolationHead(
                d_model=self.decoder_kwargs["d_model"],
                d_output=math.prod(self.tubelet_size) * self.n_channels,
                layer_norm_eps=self.decoder_kwargs["layer_norm_eps"],
                head_drop_rate=self.head_drop_rate,
            )
        )
        self.set_loss_fn(nn.MSELoss(reduction="none"))

    def reset_pos_cache(self) -> None:
        """Clear all rotary sin/cos caches (called after every forward)."""
        self.encoder_inpt_pos_embedding.reset_cache()
        self.encoder_attn_pos_embedding.reset_cache()
        self.decoder_inpt_pos_embedding.reset_cache()
        self.decoder_attn_pos_embedding.reset_cache()

    def normalize_targets(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize target values per tubelet (invalid for (1, 1, 1) tubelets)."""
        if self._normalize_targets:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.0e-6) ** 0.5
        return x

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        label=None,
        *_,
        **__,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Masked-autoencoding forward pass.

        Args:
            values: Input tensor in BTCHW layout ``[B, T, C, H, W]``.
            mask: Bool token mask ``[B, N]``; True marks tokens to reconstruct.
                Every row must contain the same number of True entries (see
                :func:`gen_mask`).
            positions: Continuous positions ``[B, n_pos_dims, N]``.
            pad_mask: Optional bool ``[B, N]``; True marks padding tokens.
            label: Unused (kept for interface parity).

        Returns:
            ``(logits, loss)`` — reconstruction of masked tubelets and the
            mean MSE over them (padding entries zeroed); both None when no
            token is masked.
        """
        b = values.shape[0]
        npd = self.n_pos_dims
        # Convert input to a sequence of tubelets
        x = patchify(self.tubelet_size, values)

        # Extract all the values that are being masked out
        m_x = x[mask].reshape(b, -1, x.shape[-1])
        m_positions = positions[mask[:, None].expand(-1, npd, -1)].reshape(b, npd, -1)
        m_pad_mask = None
        if pad_mask is not None:
            m_pad_mask = pad_mask[mask].reshape(b, -1)

        # Now get all the values that are not masked out
        x = x[~mask].reshape(b, -1, x.shape[-1])
        positions = positions[~mask[:, None, ...].expand(-1, npd, -1)].reshape(
            b, npd, -1
        )
        if pad_mask is not None:
            pad_mask = pad_mask[~mask].reshape(b, -1)

        # Project into embeddings
        x = self.projection(x)
        # Add classification token to the beginning of all relevant tensors
        x, positions, pad_mask = self.add_cls(x, positions, pad_mask)

        x = self.inpt_pos_dropout(self.encoder_inpt_pos_embedding(x, positions, ~mask))

        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)

        # Encoder forward pass
        x = self.encoder(
            x,
            positions=positions,
            pos_encoding=self.encoder_attn_pos_embedding,
            attn_mask=attn_mask,
        )
        # Project tokens from the encoder dimension to decoder dimension
        x = self.encoder_decoder_proj(x)

        mask_tokens = self.mask_token.expand(b, m_x.shape[1], -1)

        # Apply input positional encodings to our MASK tokens.
        mask_tokens = self.inpt_pos_dropout(
            self.decoder_inpt_pos_embedding(mask_tokens, m_positions, mask)
        )

        # Append MASK token and positional information
        x = torch.cat([x, mask_tokens], dim=1)
        positions = torch.cat([positions, m_positions], dim=2)
        if pad_mask is not None:
            pad_mask = torch.cat([pad_mask, m_pad_mask], dim=1)

        # Get our new attention and padding masks
        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)

        # Decoder forward pass
        x = self.decoder(
            x,
            positions=positions,
            pos_encoding=self.decoder_attn_pos_embedding,
            attn_mask=attn_mask,
        )
        m_x = self.normalize_targets(m_x)
        x = x[:, -m_x.shape[-2] :]

        logits, loss = None, None
        if m_x.shape[1] != 0:
            logits, loss = self.apply_head_loss(x, m_x)
            if m_pad_mask is not None:
                # Remove loss from padding:
                loss[m_pad_mask] = 0
            loss = loss.mean()

        # We reset the positional embedding caches to avoid
        # inter-loop dependencies in the Trainer, which break torch compile.
        self.reset_pos_cache()

        return logits, loss


class RoMAEForClassification(RoMAEBase):
    """RoMAE Encoder with an MLP head for classification/regression.

    Usually you want to initialize this using pre-trained weights from
    :class:`RoMAEForPreTraining` (copy ``encoder``/``cls``/``projection``
    weights via ``load_state_dict``). Note: upstream's ``from_pretrained``
    (dropped in this port) copied only ``encoder`` and ``cls``, leaving
    ``projection`` freshly initialized — copy ``projection`` too only when
    you want full input-pipeline transfer; skip it to reproduce upstream's
    fine-tuning recipe exactly.

    Args:
        dim_output: Number of output classes (head output dimension).
        encoder_kwargs: Encoder overrides merged over :data:`ENCODER_DEFAULTS`.
        **base_kwargs: See :class:`RoMAEBase`.
    """

    def __init__(
        self,
        dim_output: int,
        encoder_kwargs: Optional[dict] = None,
        **base_kwargs,
    ):
        super().__init__(encoder_kwargs=encoder_kwargs, **base_kwargs)
        self.dim_output = dim_output
        if self.use_cls:
            self.set_head(
                CLSClassifierHead(
                    d_model=self.encoder_kwargs["d_model"],
                    d_output=dim_output,
                    layer_norm_eps=self.encoder_kwargs["layer_norm_eps"],
                    head_drop_rate=self.head_drop_rate,
                )
            )
        else:
            self.set_head(
                MeanClassifierHead(
                    d_model=self.encoder_kwargs["d_model"],
                    d_output=dim_output,
                    layer_norm_eps=self.encoder_kwargs["layer_norm_eps"],
                    head_drop_rate=self.head_drop_rate,
                )
            )
        self.set_loss_fn(nn.CrossEntropyLoss())
        self.inpt_pos_embedding, self.attn_pos_embedding = self.get_pos_embs(
            nhead=self.encoder_kwargs["nhead"],
            d_model=self.encoder_kwargs["d_model"],
        )
        self.apply(_init_weights)

    def reset_pos_cache(self) -> None:
        """Clear all rotary sin/cos caches (called after every forward)."""
        self.inpt_pos_embedding.reset_cache()
        self.attn_pos_embedding.reset_cache()

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        *_,
        **__,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Classification forward pass.

        Args:
            values: Input tensor in BTCHW layout ``[B, T, C, H, W]``.
            positions: Continuous positions ``[B, n_pos_dims, N]``.
            pad_mask: Optional bool ``[B, N]``; True marks padding tokens.
            label: Optional class labels ``[B]``; loss is None without them.

        Returns:
            ``(logits, loss)`` with logits of shape ``[B, dim_output]``.
        """
        x = patchify(self.tubelet_size, values)
        x = self.projection(x)
        x, positions, pad_mask = self.add_cls(x, positions, pad_mask)
        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)
        x = self.inpt_pos_embedding(x, positions)
        x = self.encoder(
            x,
            positions=positions,
            pos_encoding=self.attn_pos_embedding,
            attn_mask=attn_mask,
        )
        logits, loss = self.apply_head_loss(x, label, pad_mask)

        self.reset_pos_cache()
        return logits, loss


class RoMAELightCurveBackbone(RoMAEBase):
    """RoMAE encoder as a pooled feature extractor for JEPA-style SSL.

    Wraps the shared :class:`RoMAEBase` machinery (tubelet projection, CLS
    token, continuous n-D rotary encoding, transformer encoder) but returns a
    single pooled embedding ``[B, d_model]`` instead of a task head — the shape
    a method like :class:`~stable_pretraining.methods.LeJEPALightCurve`
    expects from its backbone. It is the encoder-only, no-decoder counterpart
    of :class:`RoMAEForClassification` (same forward up to the head).

    The forward takes the ``(values, positions, pad_mask)`` triple emitted by
    :func:`tokenize_lightcurves` and resets the rotary sin/cos cache after each
    pass (bare ``Encoder`` use otherwise leaks positions across forwards).

    Args:
        encoder_kwargs: Encoder overrides merged over :data:`ENCODER_DEFAULTS`
            (e.g. ``d_model``/``nhead``/``depth`` for a width ladder).
        pool: ``"cls"`` returns the CLS token; ``"mean"`` returns the
            masked mean over non-padding tokens (CLS excluded).
        **base_kwargs: See :class:`RoMAEBase` (``tubelet_size``,
            ``n_channels``, ``n_pos_dims``, ``p_rope_val``, ...). For
            light-curve tokens use ``tubelet_size=(1, 1, 1)``, ``n_channels=1``
            and ``n_pos_dims=2`` (time, wavelength).
    """

    def __init__(
        self,
        encoder_kwargs: Optional[dict] = None,
        pool: str = "cls",
        **base_kwargs,
    ):
        super().__init__(encoder_kwargs=encoder_kwargs, use_cls=True, **base_kwargs)
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")
        self.pool = pool
        self.inpt_pos_embedding, self.attn_pos_embedding = self.get_pos_embs(
            nhead=self.encoder_kwargs["nhead"],
            d_model=self.encoder_kwargs["d_model"],
        )
        # Exposed for downstream heads/probes/witnesses (parity with timm's
        # ``backbone.embed_dim``).
        self.embed_dim = self.encoder_kwargs["d_model"]
        self.apply(_init_weights)

    def reset_pos_cache(self) -> None:
        """Clear the rotary sin/cos caches (called after every forward)."""
        self.inpt_pos_embedding.reset_cache()
        self.attn_pos_embedding.reset_cache()

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of tokenized light curves to pooled embeddings.

        Args:
            values: BTCHW tubelet input ``[B, N, 1, 1, 1]``.
            positions: Continuous positions ``[B, n_pos_dims, N]``.
            pad_mask: Optional bool ``[B, N]``; True marks padding tokens.

        Returns:
            Pooled embeddings ``[B, d_model]``.
        """
        x = patchify(self.tubelet_size, values)
        x = self.projection(x)
        x, positions, pad_mask = self.add_cls(x, positions, pad_mask)
        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)
        x = self.inpt_pos_embedding(x, positions)
        x = self.encoder(
            x,
            positions=positions,
            pos_encoding=self.attn_pos_embedding,
            attn_mask=attn_mask,
        )
        if self.pool == "cls":
            out = x[:, 0, :]
        else:
            # Masked mean over the real (non-CLS, non-padding) tokens.
            tokens = x[:, 1:, :]
            if pad_mask is not None:
                real = ~pad_mask[:, 1:]
                denom = real.sum(dim=1, keepdim=True).clamp(min=1)
                out = (tokens * real[..., None]).sum(dim=1) / denom
            else:
                out = tokens.mean(dim=1)
        self.reset_pos_cache()
        return out


# ---------------------------------------------------------------------------
# Asynchronous multivariate time-series tokenization
# ---------------------------------------------------------------------------

#: Approximate effective wavelengths (nm) of the LSST ugrizy filters, for use
#: as the ``band_wavelengths`` argument of :func:`tokenize_lightcurves`.
LSST_BAND_WAVELENGTHS = {0: 368.0, 1: 480.0, 2: 622.0, 3: 754.0, 4: 869.0, 5: 971.0}

#: ZTF g/r/i effective wavelengths (nm).
ZTF_BAND_WAVELENGTHS = {0: 472.0, 1: 634.0, 2: 789.0}

#: Effective wavelengths (nm) for the ``hibb/TESS-ZTF-isect`` band layout:
#: 0 = TESS (broad red optical, ~786 nm pivot), 1/2/3 = ZTF g/r/i. Pass this as
#: the ``band_wavelengths`` argument of :func:`tokenize_lightcurves` so the
#: wavelength position axis is shared and consistent across single-instrument
#: views (TESS-only vs ZTF-only), which is what makes cross-instrument LeJEPA
#: views comparable on the same rotary axis.
TESS_ZTF_BAND_WAVELENGTHS = {0: 786.0, 1: 472.0, 2: 634.0, 3: 789.0}


def tokenize_lightcurves(
    times: list,
    values: list,
    bands: list,
    band_wavelengths: Optional[dict] = None,
    time_scale: float = 1.0,
    wavelength_scale: float = 1.0,
    ref_wavelength: Optional[float] = None,
    cls_offset: bool = True,
    band_positions: Optional[dict] = None,
    band_features: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize asynchronous multiband light curves for RoMAE.

    Each *observation* becomes one token; bands may be sampled at completely
    different epochs and in different numbers (asynchronous multivariate
    series) — there is no alignment or imputation. The n-dimensional
    continuous RoPE then attends jointly over the time axis and the band
    axis.

    The band axis is encoded as a **physical wavelength coordinate** when
    ``band_wavelengths`` is given: position ``log(lambda / lambda_ref) /
    wavelength_scale``. This makes cross-band attention distance reflect
    spectral distance (u-g close, u-y far; SEDs and the extinction law are
    smooth in log-wavelength) and makes positions **survey-transferable**: a
    different filter set (ZTF, ATLAS, Roman) is just new points on the same
    wavelength axis, with no band-vocabulary surgery. Passing
    ``band_wavelengths=None`` falls back to the raw integer band index
    (ablation mode).

    The relative scale between the axes is the one honest hyperparameter:
    ``time_scale`` and ``wavelength_scale`` jointly set "how much time
    separation equals one unit of log-wavelength separation" via the
    per-axis rotary frequencies.

    Args:
        times: Length-``B`` list of 1-D tensors — observation epochs per
            object, in any order (sorted internally).
        values: Length-``B`` list of 1-D tensors — fluxes/magnitudes,
            aligned with ``times``.
        bands: Length-``B`` list of 1-D integer tensors — band id per
            observation, aligned with ``times``.
        band_wavelengths: Mapping band id -> effective wavelength (any
            consistent unit), e.g. :data:`LSST_BAND_WAVELENGTHS`. ``None``
            uses the integer index as the coordinate.
        time_scale: Time positions are ``t / time_scale``.
        wavelength_scale: Wavelength positions are
            ``log(lambda / lambda_ref) / wavelength_scale``.
        ref_wavelength: Center of the log-wavelength axis. Default: the
            geometric mean of the provided wavelengths (or 0-centering for
            index mode, where the raw index is used unchanged).
        cls_offset: Shift all positions by +1 so the CLS token alone owns
            position 0 (mirroring :func:`prepare_positions`); padding
            positions stay 0.
        band_positions: Mapping band id -> sequence of ``k`` extra position
            coordinates (already in final units, e.g. log-wavelength
            percentiles). When given it *replaces* the ``band_wavelengths``
            axis: positions become ``[time, *coords]`` (``k`` may be 0 for
            a time-only encoding).
        band_features: Mapping band id -> sequence of ``F`` per-band
            content features appended to the value channel, so values
            become ``[B, N, 1 + F, 1, 1]`` (the tubelet projection then
            adds a linear embedding of the filter descriptor to each
            token). Padding tokens get zeros.

    Returns:
        Tuple of ``values [B, N_max, C, 1, 1]`` (BTCHW layout for
        ``(1, 1, 1)`` tubelets; ``C = 1 + F``), ``positions
        [B, n_pos_dims, N_max]`` with axis 0 = time and the remaining axes
        = wavelength (or band index, or ``band_positions``), and boolean
        ``pad_mask [B, N_max]`` (True where padded). Tokens are sorted by
        time within each object.
    """
    if not (len(times) == len(values) == len(bands)):
        raise ValueError(
            f"times/values/bands must have equal length, got "
            f"{len(times)}/{len(values)}/{len(bands)}"
        )
    n_objects = len(times)
    lengths = [t.numel() for t in times]
    n_max = max(lengths) if lengths else 0

    if band_wavelengths is not None:
        lut_size = int(max(band_wavelengths)) + 1
        lut = torch.full((lut_size,), float("nan"))
        for k, lam in band_wavelengths.items():
            lut[int(k)] = float(lam)
        if ref_wavelength is None:
            lams = torch.tensor([float(v) for v in band_wavelengths.values()])
            ref_wavelength = float(lams.log().mean().exp())  # geometric mean

    def _lut(table, what):
        size = int(max(table)) + 1 if table else 0
        width = len(next(iter(table.values()))) if table else 0
        lut = torch.full((size, width), float("nan"))
        for k, row in table.items():
            if len(row) != width:
                raise ValueError(f"{what}: band {k} has {len(row)} entries, expected {width}")
            lut[int(k)] = torch.tensor([float(x) for x in row])
        return lut

    def _lookup(lut, b, i, what):
        bad = b >= lut.shape[0]
        rows = lut[b.clamp(max=max(lut.shape[0] - 1, 0))]
        bad = bad | torch.isnan(rows).any(dim=-1)
        if bad.any():
            raise KeyError(
                f"object {i}: band ids {sorted(set(b[bad].tolist()))} missing from {what}"
            )
        return rows

    pos_lut = _lut(band_positions, "band_positions") if band_positions is not None else None
    feat_lut = _lut(band_features, "band_features") if band_features is not None else None
    n_extra = pos_lut.shape[1] if pos_lut is not None else 1
    n_feat = feat_lut.shape[1] if feat_lut is not None else 0

    out_values = torch.zeros(n_objects, n_max, 1 + n_feat)
    out_time = torch.zeros(n_objects, n_max)
    out_wave = torch.zeros(n_objects, n_extra, n_max)
    pad_mask = torch.ones(n_objects, n_max, dtype=torch.bool)

    for i, (t, v, b) in enumerate(zip(times, values, bands)):
        if not (t.numel() == v.numel() == b.numel()):
            raise ValueError(
                f"object {i}: times/values/bands lengths differ "
                f"({t.numel()}/{v.numel()}/{b.numel()})"
            )
        n = t.numel()
        order = torch.argsort(t.float())
        t, v, b = t.float()[order], v.float()[order], b.long()[order]
        if pos_lut is not None:
            wave = _lookup(pos_lut, b, i, "band_positions").T  # (k, n)
        elif band_wavelengths is not None:
            out_of_range = b >= lut.numel()
            in_range_lam = lut[b.clamp(max=lut.numel() - 1)]
            bad = out_of_range | torch.isnan(in_range_lam)
            if bad.any():
                missing = sorted(set(b[bad].tolist()))
                raise KeyError(
                    f"object {i}: band ids {missing} missing from "
                    "band_wavelengths"
                )
            lam = in_range_lam
            wave = torch.log(lam / ref_wavelength) / wavelength_scale
        else:
            wave = b.float()
        out_values[i, :n, 0] = v
        if feat_lut is not None:
            out_values[i, :n, 1:] = _lookup(feat_lut, b, i, "band_features")
        out_time[i, :n] = t / time_scale
        out_wave[i, :, :n] = wave
        pad_mask[i, :n] = False

    if cls_offset:
        out_time = out_time + 1.0
        out_wave = out_wave + 1.0
    positions = torch.cat([out_time[:, None, :], out_wave], dim=1)
    positions = positions * (~pad_mask)[:, None, :]
    return out_values[:, :, :, None, None], positions, pad_mask
