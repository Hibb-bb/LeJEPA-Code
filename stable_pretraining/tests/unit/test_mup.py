"""Unit tests for mu-P helpers (stable_pretraining.optim.mup)."""

import math

import pytest
import torch
import torch.nn as nn

timm = pytest.importorskip("timm")
# Importing the package pulls its hard deps (loguru/omegaconf/hydra); skip
# cleanly in stripped environments rather than error at collection.
spt_optim = pytest.importorskip("stable_pretraining.optim")

apply_mup = spt_optim.apply_mup
mup_param_groups = spt_optim.mup_param_groups

BASE_FANIN = 256
BASE_LR = 1e-3


class ToyMLP(nn.Module):
    """Two Linears with different fan-ins plus a norm layer (no forward needed)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.norm = nn.LayerNorm(512)
        self.fc2 = nn.Linear(1024, 512)


def _group_of(groups, param):
    matches = [g for g in groups if any(p is param for p in g["params"])]
    assert len(matches) == 1, "param must appear in exactly one group"
    return matches[0]


@pytest.mark.unit
def test_mup_param_groups_toy_mlp():
    model = ToyMLP()
    groups = mup_param_groups(
        model, base_lr=BASE_LR, weight_decay=0.05, base_fanin=BASE_FANIN
    )

    # fan_in 256 -> multiplier 1.0
    g_fc1 = _group_of(groups, model.fc1.weight)
    assert g_fc1["lr"] == pytest.approx(BASE_LR * 1.0)
    assert g_fc1["weight_decay"] == pytest.approx(0.05)

    # fan_in 1024 -> multiplier 256/1024 = 0.25
    g_fc2 = _group_of(groups, model.fc2.weight)
    assert g_fc2["lr"] == pytest.approx(BASE_LR * 0.25)
    assert g_fc2["weight_decay"] == pytest.approx(0.05)

    # Biases and norm parameters: base lr, weight_decay 0.
    for param in [
        model.fc1.bias,
        model.fc2.bias,
        model.norm.weight,
        model.norm.bias,
    ]:
        g = _group_of(groups, param)
        assert g["lr"] == pytest.approx(BASE_LR)
        assert g["weight_decay"] == 0.0

    # Union of all group params == all model params, no duplicates.
    grouped_ids = [id(p) for g in groups for p in g["params"]]
    model_ids = {id(p) for p in model.parameters()}
    assert len(grouped_ids) == len(set(grouped_ids)), "duplicate params across groups"
    assert set(grouped_ids) == model_ids, "groups must cover exactly all params"


@pytest.mark.unit
def test_apply_mup_vit_tiny():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    returned = apply_mup(model, base_fanin=BASE_FANIN)
    assert returned is model  # in-place

    # Every attention block uses the mu-P 1/head_dim logit scale (and the
    # eager attention path, which is the one that reads ``scale``).
    for block in model.blocks:
        attn = block.attn
        assert attn.scale == pytest.approx(1.0 / attn.head_dim)
        assert attn.fused_attn is False

    # Sampled Linear weights match the spectral std within 3x.
    for weight in [
        model.blocks[0].mlp.fc1.weight,
        model.blocks[5].attn.qkv.weight,
        model.blocks[11].attn.proj.weight,
    ]:
        fan_out, fan_in = weight.shape
        expected = math.sqrt(fan_out / fan_in) / (
            math.sqrt(fan_in) + math.sqrt(fan_out)
        )
        assert expected / 3 < weight.std().item() < expected * 3

    # One 2-image forward to prove nothing broke.
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, model.num_features)
    assert torch.isfinite(out).all()


@pytest.mark.unit
def test_apply_mup_zero_init_query():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    apply_mup(model, base_fanin=BASE_FANIN, zero_init_query=True)
    for block in model.blocks:
        attn = block.attn
        q_rows = attn.num_heads * attn.head_dim
        assert attn.qkv.weight[:q_rows].abs().sum() == 0
        # k/v thirds keep their spectral init.
        assert attn.qkv.weight[q_rows:].abs().sum() > 0


@pytest.mark.unit
def test_adamw_constructs_and_steps():
    model = ToyMLP()
    groups = mup_param_groups(
        model, base_lr=BASE_LR, weight_decay=0.01, base_fanin=BASE_FANIN
    )
    optimizer = torch.optim.AdamW(groups)

    before = model.fc1.weight.detach().clone()
    loss = (
        model.fc1(torch.randn(4, 256)).pow(2).mean()
        + model.fc2(torch.randn(4, 1024)).pow(2).mean()
        + model.norm(torch.randn(4, 512)).pow(2).mean()
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert not torch.equal(model.fc1.weight, before), "step must update weights"


@pytest.mark.unit
class TestMupParamGroupsViT:
    """Group assignment on a real timm ViT: tokens, conv patch embed, coverage."""

    def test_vit_tiny_group_assignment(self):
        timm = pytest.importorskip("timm")
        from stable_pretraining.optim.mup import mup_param_groups

        model = timm.create_model("vit_tiny_patch16_224", num_classes=0)
        base_lr = 1e-3
        groups = mup_param_groups(model, base_lr=base_lr, base_fanin=256)

        # Exact coverage, no duplicates.
        grouped = [p for g in groups for p in g["params"]]
        assert len(grouped) == len(set(map(id, grouped)))
        assert set(map(id, grouped)) == set(
            id(p) for p in model.parameters() if p.requires_grad
        )

        lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
        wd_of = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
        named = dict(model.named_parameters())

        # Broadcast tokens land in the base-lr, wd=0 group.
        for name in ("cls_token", "pos_embed"):
            assert lr_of[id(named[name])] == pytest.approx(base_lr)
            assert wd_of[id(named[name])] == 0.0

        # Conv patch embed: fan_in = in_chans * k * k = 3 * 16 * 16 = 768.
        pe = named["patch_embed.proj.weight"]
        assert lr_of[id(pe)] == pytest.approx(base_lr * 256 / 768)

        # A block Linear: qkv has in_features = embed_dim = 192.
        qkv = named["blocks.0.attn.qkv.weight"]
        assert lr_of[id(qkv)] == pytest.approx(base_lr * 256 / 192)
