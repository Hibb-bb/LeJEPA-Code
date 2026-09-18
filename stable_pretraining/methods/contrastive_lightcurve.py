"""Multi-positive contrastive learning for asynchronous multivariate light curves.

Contrastive sibling of :class:`~stable_pretraining.methods.LeJEPALightCurve`:
the same RoMAE backbone and ``(values, positions, pad_mask)`` view triples,
but the loss is the multi-positive NT-Xent
(:class:`~stable_pretraining.losses.MultiPositiveNTXEntLoss`, "nt_xent_multi")
instead of invariance-to-center + SIGReg. Every view of a star (globals and
locals alike) is an anchor whose positives are *all* other views of the same
star in the (cross-GPU) batch; views of every other star are negatives.

Each view is encoded independently (views differ in token count), projected,
and the ``[V*N, K]`` projections are contrasted with star ids repeated per
view. The ids should be dataset-level object indices so a star that is
oversampled into the same batch is a positive of itself rather than a false
negative.

Example::

    from stable_pretraining.backbone import RoMAELightCurveBackbone
    from stable_pretraining.methods import ContrastiveLightCurve

    backbone = RoMAELightCurveBackbone(
        encoder_kwargs=dict(d_model=64, nhead=2, depth=2),
        tubelet_size=(1, 1, 1), n_channels=1, n_pos_dims=2,
    )
    model = ContrastiveLightCurve(backbone, temperature=0.2)

    model.train()
    out = model(global_views=[view_tess, view_ztf], local_views=[view_a],
                star_id=torch.arange(N))
    out.loss.backward()

    model.eval()
    features = model(values=v, positions=p, pad_mask=m).embedding  # [N, D]
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from stable_pretraining.backbone import MLP
from stable_pretraining.losses import MultiPositiveNTXEntLoss
from transformers.utils import ModelOutput

View = tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]


@dataclass
class ContrastiveOutput(ModelOutput):
    """Output of :class:`ContrastiveLightCurve`.

    :ivar loss: Multi-positive NT-Xent loss (0 in eval mode).
    :ivar embedding: Backbone embeddings of the global views
        [n_global*N, D] (train) or [N, D] (eval).
    :ivar projection: Detached projector outputs of the global views
        [n_global*N, K] (train) or [N, K] (eval).
    :ivar features: Undetached backbone features of all views, view-major
        [n_views*N, D] (train only), for auxiliary embedding-space losses.
    :ivar pos_sim: Mean cosine similarity between positive pairs (monitor).
    :ivar neg_sim: Mean cosine similarity between negative pairs (monitor).
    """

    loss: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None
    projection: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    pos_sim: Optional[torch.Tensor] = None
    neg_sim: Optional[torch.Tensor] = None


class ContrastiveLightCurve(nn.Module):
    """Multi-positive NT-Xent over light-curve views.

    :param backbone: A pooled light-curve encoder returning ``[N, D]`` from a
        ``(values, positions, pad_mask)`` view, e.g.
        :class:`~stable_pretraining.backbone.RoMAELightCurveBackbone`. Must
        expose ``embed_dim``.
    :param projector: Optional projection head. When ``None``, the SimCLR
        style ``MLP(embed_dim -> 2048 -> 2048 -> proj_dim)`` with BN+ReLU is
        created.
    :param proj_dim: Output dim of the default projector.
    :param temperature: NT-Xent temperature.
    :param anchor_locals: If ``False``, local views are only *positives* for
        the global anchors (they are still encoded and projected) and never
        anchors themselves; mirrors LeJEPA where locals pull towards the
        global center. Default ``True``: every view is an anchor.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: Optional[nn.Module] = None,
        proj_dim: int = 128,
        temperature: float = 0.2,
        anchor_locals: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim
        if projector is None:
            projector = MLP(
                in_channels=embed_dim,
                hidden_channels=[2048, 2048, proj_dim],
                norm_layer="batch_norm",
                activation_layer=nn.ReLU,
                inplace=True,
                dropout=0.0,
            )
        self.projector = projector
        self.loss_fn = MultiPositiveNTXEntLoss(temperature=temperature)
        self.temperature = temperature
        self.anchor_locals = anchor_locals
        self.embed_dim = embed_dim

    def _encode(self, view: View) -> torch.Tensor:
        values, positions, pad_mask = view
        return self.backbone(values, positions, pad_mask)

    @staticmethod
    @torch.no_grad()
    def _pair_stats(z: torch.Tensor, ids: torch.Tensor):
        """Mean cosine similarity of positive and of negative pairs."""
        z = nn.functional.normalize(z, dim=-1)
        sim = z @ z.T
        eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        pos = (ids[:, None] == ids[None, :]) & ~eye
        neg = ~pos & ~eye
        pos_sim = sim[pos].mean() if bool(pos.any()) else sim.new_tensor(0.0)
        neg_sim = sim[neg].mean() if bool(neg.any()) else sim.new_tensor(0.0)
        return pos_sim, neg_sim

    def compute_loss(
        self,
        all_projected: torch.Tensor,
        n_global: int,
        star_id: torch.Tensor,
    ):
        """NT-Xent over ``all_projected`` ``[V, N, K]`` with ids ``[N]``.

        Returns ``(loss, pos_sim, neg_sim)``. With ``anchor_locals=False``
        the local rows stay in the candidate set (positives of their star's
        globals, negatives of every other star) but are not anchors.
        """
        n_views, bs, k = all_projected.shape
        z = all_projected.reshape(n_views * bs, k)
        ids = star_id.repeat(n_views)
        anchor_mask = None
        if not self.anchor_locals:
            anchor_mask = torch.zeros(n_views * bs, dtype=torch.bool, device=z.device)
            anchor_mask[: n_global * bs] = True
        loss = self.loss_fn(z, ids, anchor_mask)
        pos_sim, neg_sim = self._pair_stats(z, ids)
        return loss, pos_sim, neg_sim

    def forward(
        self,
        global_views: Optional[list[View]] = None,
        local_views: Optional[list[View]] = None,
        star_id: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> ContrastiveOutput:
        if self.training:
            assert global_views is not None, (
                "global_views must be provided in training mode"
            )
            local_views = local_views or []
            all_views = list(global_views) + list(local_views)
            n_global = len(global_views)
            bs = global_views[0][0].shape[0]
            if star_id is None:
                star_id = torch.arange(bs, device=global_views[0][0].device)

            feats = [self._encode(v) for v in all_views]  # each [N, D]
            all_features = torch.cat(feats)  # [V*N, D]
            all_projected = self.projector(all_features)
            all_projected = all_projected.view(len(all_views), bs, -1)

            loss, pos_sim, neg_sim = self.compute_loss(
                all_projected, n_global, star_id.to(all_projected.device)
            )
            return ContrastiveOutput(
                loss=loss,
                embedding=torch.cat(feats[:n_global]).detach(),
                projection=all_projected[:n_global]
                .reshape(-1, all_projected.size(-1))
                .detach(),
                features=all_features,
                pos_sim=pos_sim,
                neg_sim=neg_sim,
            )
        assert values is not None, "values must be provided in eval mode"
        embedding = self.backbone(values, positions, pad_mask)
        projection = self.projector(embedding).detach()
        zero = torch.tensor(0.0, device=embedding.device)
        return ContrastiveOutput(
            loss=zero, embedding=embedding, projection=projection,
            pos_sim=zero, neg_sim=zero,
        )
