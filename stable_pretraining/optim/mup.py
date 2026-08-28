"""Maximal-update parametrization (mu-P) helpers for AdamW-style training.

mu-P (Yang & Hu, Tensor Programs V) keeps feature-learning dynamics
width-independent so that hyperparameters tuned at a small base width
transfer to wider models. For AdamW this reduces to three ingredients:

1. **Per-layer learning rates**: every matrix-like (>=2-D) weight trains at
   ``base_lr * base_fanin / fan_in`` while vector-like (1-D / bias / norm)
   parameters stay at ``base_lr`` — see :func:`mup_param_groups`.
2. **Spectral initialization**: ``nn.Linear`` weights are drawn from
   ``N(0, std^2)`` with ``std = sqrt(fan_out / fan_in) / (sqrt(fan_in) +
   sqrt(fan_out))`` — see :func:`apply_mup`.
3. **1/d attention**: attention logits are scaled by ``1 / head_dim``
   instead of the standard ``1 / sqrt(head_dim)`` — see :func:`apply_mup`.

Typical usage with :class:`stable_pretraining.Module`::

    model = LeJEPA("vit_small_patch16_224")
    apply_mup(model, base_fanin=256)
    module = spt.Module(
        model=model,
        forward=my_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "params": mup_param_groups(
                    model, base_lr=4e-4, weight_decay=0.05
                ),
                "betas": (0.9, 0.999),
            },
            "scheduler": {...},
        },
    )

The ``"params"`` key is honored by
:func:`stable_pretraining.optim.utils.create_optimizer`: pre-built groups
replace the module's flat parameter list and disable the
``exclude_bias_norm`` re-grouping (each group already carries its own
``weight_decay``).
"""

from __future__ import annotations

import math
from typing import List

import torch.nn as nn
from loguru import logger as logging

from .utils import is_bias_or_norm_param

__all__ = ["apply_mup", "mup_param_groups"]


def _spectral_std(fan_in: int, fan_out: int) -> float:
    """Spectral initialization std: ``sqrt(fan_out/fan_in) / (sqrt(fan_in) + sqrt(fan_out))``."""
    return math.sqrt(fan_out / fan_in) / (math.sqrt(fan_in) + math.sqrt(fan_out))


def apply_mup(
    model: nn.Module, base_fanin: int = 256, zero_init_query: bool = False
) -> nn.Module:
    """Apply mu-P initialization and attention scaling to ``model`` in-place.

    Three modifications are made:

    1. Every ``nn.Linear`` weight is re-initialized from
       ``N(0, std^2)`` with the spectral std
       ``sqrt(fan_out/fan_in) / (sqrt(fan_in) + sqrt(fan_out))``.
       Biases are left untouched.
    2. Every timm-style attention block (any module exposing both a
       ``scale`` attribute and a ``head_dim`` attribute, e.g.
       ``timm.layers.attention.Attention``) has its logit scale set to
       ``1 / head_dim`` (mu-P attention) instead of the standard
       ``1 / sqrt(head_dim)``. Because timm's fused-SDPA path calls
       ``F.scaled_dot_product_attention`` without a ``scale`` argument
       (so ``self.scale`` would be silently ignored), ``fused_attn`` is
       set to ``False`` on patched blocks so the eager path — which does
       use ``self.scale`` — runs.
    3. Optionally (``zero_init_query=True``), for attention blocks with a
       fused ``qkv`` projection the query third of ``qkv.weight`` (rows
       ``0:num_heads*head_dim``) is zeroed, so attention starts as a
       uniform average (a common mu-P/transfer stabilization).

    Args:
        model: Model to modify in-place (e.g., a timm ViT or a wrapper
            containing one).
        base_fanin: Accepted for call-site symmetry with
            :func:`mup_param_groups` (one config dict can drive both);
            the spectral std depends only on each layer's own
            fan-in/fan-out, so this value does not affect initialization.
        zero_init_query: Zero the query rows of every fused ``qkv``
            projection.

    Returns:
        The same ``model`` instance, modified in-place.
    """
    # Pass 1: spectral re-initialization of Linear weights. Done before the
    # attention pass so ``zero_init_query`` is not clobbered by the re-init
    # of the ``qkv`` Linear (a child of the attention module).
    n_linear = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            std = _spectral_std(module.in_features, module.out_features)
            module.weight.data.normal_(mean=0.0, std=std)
            n_linear += 1

    # Pass 2: patch attention blocks. timm's Attention exposes float
    # ``scale`` and int ``head_dim`` attributes (verified against
    # timm.layers.attention.Attention); guard with hasattr + type checks so
    # unrelated modules with a ``scale`` sub-module/parameter are skipped.
    n_attn = 0
    n_q_zeroed = 0
    for module in model.modules():
        if not (hasattr(module, "scale") and hasattr(module, "head_dim")):
            continue
        if not isinstance(module.scale, (int, float)) or not isinstance(
            module.head_dim, int
        ):
            continue
        module.scale = 1.0 / module.head_dim
        if getattr(module, "fused_attn", False):
            # The fused path ignores ``self.scale`` — force the eager path.
            module.fused_attn = False
        n_attn += 1

        if zero_init_query:
            qkv = getattr(module, "qkv", None)
            if isinstance(qkv, nn.Linear) and qkv.out_features % 3 == 0:
                q_rows = qkv.out_features // 3
                qkv.weight.data[:q_rows].zero_()
                n_q_zeroed += 1

    logging.info(
        f"apply_mup: re-initialized {n_linear} Linear weights (spectral std), "
        f"patched {n_attn} attention blocks to scale=1/head_dim"
        + (f", zeroed q in {n_q_zeroed} qkv projections" if zero_init_query else "")
    )
    return model


def mup_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.0,
    base_fanin: int = 256,
) -> List[dict]:
    """Build mu-P (AdamW) parameter groups with per-layer learning rates.

    Every trainable >=2-D weight is grouped by its fan-in and assigned
    ``lr = base_lr * base_fanin / fan_in`` with the given ``weight_decay``.
    Fan-in resolution:

    - ``nn.Linear`` weight: the module's ``in_features``;
    - conv weight ``[out, in, *kernel]``: ``in * prod(kernel)``
      (i.e., ``weight.shape[1:].numel()``);
    - other 2-D parameters: ``param.shape[1]``.

    .. warning::
        The returned groups capture ``Parameter`` objects by identity at call
        time. Sharding strategies that re-parametrize the module between
        ``configure_model`` and ``configure_optimizers`` (Lightning
        ``strategy="fsdp2"``, DeepSpeed) replace those objects with new
        (D)Tensors, so a pre-built optimizer would train dangling parameters
        and silently do nothing. Use these groups with single-device or DDP
        strategies, or rebuild them inside ``configure_optimizers`` after
        sharding.

    All 1-D / bias / norm parameters (classified via
    :func:`stable_pretraining.optim.utils.is_bias_or_norm_param`) go into a
    single group at ``base_lr`` with ``weight_decay=0.0``. Broadcast token
    parameters with a leading dimension of 1 that are not owned by a
    Linear/conv module (``cls_token``, ``pos_embed``, register tokens)
    also go into that base group: they are embedding-like, and mu-P trains
    embedding-like parameters at the base (width-independent) Adam rate —
    grouping ``pos_embed`` by ``shape[1:].numel()`` would divide its lr by
    the sequence length.

    Wiring through ``spt.Module``: the single-optimizer config accepts the
    pre-built groups under a ``"params"`` key —
    ``optim={"optimizer": {"type": "AdamW", "params": mup_param_groups(...),
    "betas": ...}, ...}`` — which
    :func:`~stable_pretraining.optim.utils.create_optimizer` uses instead
    of the module's flat parameter list (``exclude_bias_norm`` splitting is
    skipped; the groups already carry per-group ``weight_decay``).
    Warmup/annealing schedulers built from ``torch.optim.lr_scheduler``
    primitives scale each group's lr multiplicatively, so the mu-P ratios
    are preserved through scheduling.

    Args:
        model: Model whose ``named_parameters()`` are grouped.
        base_lr: Learning rate at the base width (fan-in ``base_fanin``).
        weight_decay: Weight decay applied to the >=2-D weight groups.
        base_fanin: Reference fan-in; a layer with ``fan_in == base_fanin``
            trains at exactly ``base_lr``.

    Returns:
        List of parameter-group dicts, one per distinct fan-in — each
        ``{"params": [...], "lr": base_lr * base_fanin / fan_in,
        "weight_decay": weight_decay}`` — plus one base group
        ``{"params": [...], "lr": base_lr, "weight_decay": 0.0}`` for the
        1-D/bias/norm (and token) parameters.
    """
    # Map each parameter name to its owning module so fan-in resolves from
    # module metadata (Linear.in_features; conv kernel shape) rather than
    # from the raw tensor shape alone.
    owner: dict[str, nn.Module] = {}
    for mod_name, module in model.named_modules():
        for p_name, _ in module.named_parameters(recurse=False):
            full_name = f"{mod_name}.{p_name}" if mod_name else p_name
            owner[full_name] = module

    matrix_by_fanin: dict[int, list] = {}
    base_group_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if is_bias_or_norm_param(name, param):
            base_group_params.append(param)
            continue

        module = owner.get(name)

        # Embedding-like broadcast tokens (cls_token [1,1,D], pos_embed
        # [1,L,D], register tokens): base Adam rate, no decay.
        if (
            param.shape[0] == 1
            and not isinstance(module, (nn.Linear, nn.modules.conv._ConvNd))
        ):
            base_group_params.append(param)
            continue

        if isinstance(module, nn.Linear):
            fan_in = module.in_features
        elif param.dim() == 2:
            fan_in = param.shape[1]
        else:
            # Conv-style [out, in, *kernel]: fan_in = in * prod(kernel).
            fan_in = param.shape[1:].numel()

        matrix_by_fanin.setdefault(int(fan_in), []).append(param)

    param_groups = [
        {
            "params": params,
            "lr": base_lr * base_fanin / fan_in,
            "weight_decay": weight_decay,
        }
        for fan_in, params in sorted(matrix_by_fanin.items())
    ]
    if base_group_params:
        param_groups.append(
            {"params": base_group_params, "lr": base_lr, "weight_decay": 0.0}
        )

    n_matrix = sum(len(g) for g in matrix_by_fanin.values())
    logging.info(
        f"mup_param_groups: {len(matrix_by_fanin)} fan-in groups "
        f"({n_matrix} matrix params, fan-ins {sorted(matrix_by_fanin)}), "
        f"{len(base_group_params)} params at base lr {base_lr} (wd=0)"
    )
    return param_groups
