"""Contrastive (multi-positive NT-Xent) pretraining on multi-survey light curves.

Drop-in sibling of ``pretrain.py``: same data pipeline, view generation,
RoMAE encoder, mu-P parametrisation, probes and evaluation callbacks, but the
``LeJEPA`` invariance + SIGReg objective is replaced by
:class:`~stable_pretraining.methods.ContrastiveLightCurve`, i.e. the
``nt_xent_multi`` loss: every view of a star (globals and locals) is an
anchor whose positives are all *other* views of the same star in the batch
and whose negatives are the views of every other star (negatives span all
GPUs under DDP).

Everything is reused from ``pretrain.py`` by importing it as a module and
registering ``--mode contrastive`` in its ``VIEW_MODES`` / ``FORWARDS`` /
``MODEL_BUILDERS`` hooks, so any flag of ``pretrain.py`` (dataset layout,
holdout instrument, augmentations, wavelength encoding, ...) works here
unchanged. Additional flags:

--temperature        NT-Xent temperature (default 0.2)
--no-anchor-locals   locals are positives of the globals but never anchors
--projector          defaults to ``mlp`` here (contrastive losses need a
                     throwaway head; ``identity`` contrasts the embeddings)

Typical use::

    # CPU smoke test (tiny): proves the pipeline end to end
    python pretrain_contrastive.py --width 32 --depth 2 --epochs 1 \
        --max-objects 96 --batch-size 8 --global-tokens 96 --local-tokens 48 \
        --num-workers 0 --precision 32

    # one GPU, default rung
    python pretrain_contrastive.py --dataset hibb/tess-ztf-atlas-asassn-isect \
        --exclude-instrument TESS --temperature 0.2 --epochs 100

``downstream.py`` consumes the resulting checkpoints exactly like LeJEPA ones
(same backbone class and ``--wave-*`` flags).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pretrain as P  # noqa: E402

from stable_pretraining.backbone import MLP, RoMAELightCurveBackbone  # noqa: E402
from stable_pretraining.methods import ContrastiveLightCurve  # noqa: E402
from stable_pretraining.optim import apply_mup  # noqa: E402

MODE = "contrastive"


def build_contrastive_model(args, n_classes=None):
    """RoMAE encoder at ``args.width`` + NT-Xent projector (mu-P applied)."""
    assert args.width % P.HEAD_DIM == 0, \
        f"width must be a multiple of head_dim={P.HEAD_DIM}"
    backbone = RoMAELightCurveBackbone(
        encoder_kwargs=dict(d_model=args.width, nhead=args.width // P.HEAD_DIM,
                            depth=args.depth),
        tubelet_size=(1, 1, 1), n_channels=P.N_CHANNELS, n_pos_dims=P.N_POS_DIMS,
        p_rope_val=P.WAVE["p_rope"], rope_blocks=P.ROPE_BLOCKS, pool=P.WAVE["pool"],
    )
    if args.projector == "identity":
        proj = nn.Identity()
    else:
        proj = MLP(
            in_channels=args.width,
            hidden_channels=[2048, 2048, args.proj_dim],
            norm_layer="batch_norm",
            activation_layer=nn.ReLU,
            inplace=True,
            dropout=0.0,
        )
    model = ContrastiveLightCurve(
        backbone, projector=proj, temperature=args.temperature,
        anchor_locals=not args.no_anchor_locals,
    )
    if P.ADV_WEIGHT > 0:
        model.inst_head = nn.Linear(args.width, len(P.TRAIN_INSTRUMENTS))
    return apply_mup(model, base_fanin=P.BASE_FANIN)


def _val_loss(model, global_views, local_views, star_id):
    """Contrastive loss on a deterministic val view set, model in eval mode
    (``ContrastiveLightCurve.forward`` only computes it when training)."""
    all_views = list(global_views) + list(local_views)
    feats = [model._encode(v) for v in all_views]
    proj = model.projector(torch.cat(feats))
    bs = global_views[0][0].shape[0]
    proj = proj.view(len(all_views), bs, -1)
    return model.compute_loss(proj, len(global_views), star_id.to(proj.device))


def contrastive_forward(self, batch, stage):
    """``--mode contrastive``: NT-Xent loss + detached probe embeddings.

    Mirrors :func:`pretrain.lejepa_forward`; the logged ``pred_loss`` /
    ``sigreg_loss`` pair is replaced by ``pos_sim`` / ``neg_sim`` (mean
    cosine similarity of positive and negative pairs in projector space).
    """
    out = {}
    global_views = [batch[k] for k in sorted(batch) if k.startswith("global")]
    local_views = [batch[k] for k in sorted(batch) if k.startswith("local")]
    star_id = batch["star_id"]
    if stage == "fit":
        output = self.model.forward(
            global_views=global_views, local_views=local_views, star_id=star_id
        )
        loss, pos_sim, neg_sim = output.loss, output.pos_sim, output.neg_sim
        if P.ADV_WEIGHT > 0:
            adv_ce, adv_acc, n_adv = P._adversarial_loss(self, output, batch)
            if n_adv:
                loss = loss + adv_ce
                self.log("train/adv_ce", adv_ce, on_step=True,
                         on_epoch=True, sync_dist=True)
                self.log("train/adv_acc", adv_acc, on_step=True,
                         on_epoch=True, sync_dist=True)
        tag = "train"
    else:
        output = self.model.forward(
            values=batch["probe_joint"][0],
            positions=batch["probe_joint"][1],
            pad_mask=batch["probe_joint"][2],
        )
        loss, pos_sim, neg_sim = _val_loss(
            self.model, global_views, local_views, star_id
        )
        tag = "val"

    out["loss"] = loss
    out["embedding"] = output.embedding
    out["projection"] = output.projection
    P._probe_embeddings(self.model.backbone, batch, out, stage)
    if stage != "fit":
        with torch.no_grad():
            for name in P.INSTRUMENTS:
                out[f"proj_{name}"] = self.model.projector(out[f"emb_{name}"])

    log_kw = dict(on_step=stage == "fit", on_epoch=True, sync_dist=True)
    self.log(f"{tag}/pos_sim", pos_sim, **log_kw)
    self.log(f"{tag}/neg_sim", neg_sim, **log_kw)
    self.log(f"{tag}/loss", loss, **log_kw)
    return out


# Register the mode with pretrain.py's hooks.
P.VIEW_MODES.add(MODE)
P.FORWARDS[MODE] = contrastive_forward
P.MODEL_BUILDERS[MODE] = build_contrastive_model


def build_parser():
    ap = P.build_parser()
    ap._option_string_actions["--mode"].choices.append(MODE)
    ap.set_defaults(mode=MODE, projector="mlp", proj_dim=128,
                    wandb="contrastive-lightcurves")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="NT-Xent temperature")
    ap.add_argument("--no-anchor-locals", action="store_true",
                    help="local views are positives of their star's globals "
                         "(and negatives of other stars) but not anchors")
    return ap


def main():
    args = build_parser().parse_args()
    if args.sweep_lamb is not None:
        raise SystemExit("--sweep-lamb is a LeJEPA (SIGReg) tool; not "
                         "applicable to the contrastive objective")
    P.apply_args(args)
    cfg = P.make_view_config(args)
    records, label_to_idx = P.load_records(args)
    n_classes = len(label_to_idx)
    print(
        f"rung: width={args.width} proj_dim={args.proj_dim} B={args.batch_size} "
        f"temperature={args.temperature} anchor_locals={not args.no_anchor_locals} "
        f"base_lr={args.base_lr}"
    )
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    # lamb is unused by the contrastive mode (only the run name reads it).
    P.train_once(args, 0.0, records, n_classes, cfg, idx_to_label=idx_to_label)


if __name__ == "__main__":
    sys.exit(main())
