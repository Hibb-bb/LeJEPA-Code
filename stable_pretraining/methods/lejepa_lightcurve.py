"""LeJEPA for asynchronous multivariate light curves (RoMAE backbone).

Light-curve sibling of :class:`~stable_pretraining.methods.LeJEPA`. The image
method builds a timm ViT and consumes crop tensors ``[N, C, H, W]``; here the
backbone is a :class:`~stable_pretraining.backbone.RoMAELightCurveBackbone`
(CLS-pooled RoMAE encoder) and each *view* is the
``(values, positions, pad_mask)`` triple emitted by
:func:`~stable_pretraining.backbone.tokenize_lightcurves`.

Everything downstream of the backbone is shared with the image method — the
sliced Epps-Pulley SIGReg statistic (:class:`SlicedEppsPulley`), the
multi-view invariance-to-center loss (:meth:`LeJEPA._compute_loss`), and the
:class:`LeJEPAOutput` container are imported from ``lejepa`` rather than
reimplemented, so the two methods stay in lockstep.

Because light-curve views have *different* token counts (a TESS-only view and
a ZTF-only view of the same star span different numbers of observations),
views cannot be concatenated into one padded backbone call the way image crops
can; each view is encoded independently and the pooled ``[N, D]`` embeddings
(which no longer carry a token axis) are stacked for the loss.

Example::

    from stable_pretraining.backbone import RoMAELightCurveBackbone
    from stable_pretraining.methods import LeJEPALightCurve

    backbone = RoMAELightCurveBackbone(
        encoder_kwargs=dict(d_model=64, nhead=2, depth=2),
        tubelet_size=(1, 1, 1), n_channels=1, n_pos_dims=2,
    )
    model = LeJEPALightCurve(backbone)

    # each view is (values [N, T, 1, 1, 1], positions [N, 2, T], pad_mask [N, T])
    model.train()
    out = model(global_views=[view_tess, view_ztf], local_views=[view_a, view_b])
    out.loss.backward()

    model.eval()
    out = model(values=v, positions=p, pad_mask=m)
    features = out.embedding  # [N, D]
"""

from typing import Optional

import torch
import torch.nn as nn

from stable_pretraining.backbone import MLP

from .lejepa import LeJEPA, LeJEPAOutput, SlicedEppsPulley

View = tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]


class LeJEPALightCurve(nn.Module):
    """Multi-view invariance + sliced Epps-Pulley SIGReg over light curves.

    :param backbone: A pooled light-curve encoder returning ``[N, D]`` from a
        ``(values, positions, pad_mask)`` view, e.g.
        :class:`~stable_pretraining.backbone.RoMAELightCurveBackbone`. Must
        expose ``embed_dim``.
    :param projector: Optional projection head. When ``None``, a
        ``Linear(embed_dim, 512) -> BN+ReLU MLP(512->2048->2048->512)`` is
        created (mirrors :class:`LeJEPA`).
    :param n_slices: Random projection directions for the goodness-of-fit test.
    :param t_max: EP integration upper bound.
    :param n_points: EP quadrature nodes.
    :param lamb: SIGReg weight lambda (additive convention).
    :param predictor: Optional ``[N, D] -> [N, D]`` head applied *after the
        encoder and before the projector* to the views of one instrument
        (``predictor_inst``), e.g. the lowest-quality survey: the encoder
        embeds every view, the predictor maps the noisy survey's embedding
        into the shared space the loss lives in, and downstream consumers
        still read the raw encoder embedding. Gated per view with the
        ``view_inst`` tensor passed to :meth:`forward`.
    :param predictor_inst: The instrument id (as in ``view_inst``) whose
        views go through ``predictor``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: Optional[nn.Module] = None,
        n_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
        lamb: float = 0.02,
        predictor: Optional[nn.Module] = None,
        predictor_inst: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor
        self.predictor_inst = predictor_inst
        embed_dim = backbone.embed_dim

        if projector is None:
            projector = nn.Sequential(
                nn.Linear(embed_dim, 512, bias=True),
                MLP(
                    in_channels=512,
                    hidden_channels=[2048, 2048, 512],
                    norm_layer="batch_norm",
                    activation_layer=nn.ReLU,
                    inplace=True,
                    dropout=0.0,
                ),
            )
        self.projector = projector

        self.sigreg = SlicedEppsPulley(
            num_slices=n_slices, t_max=t_max, n_points=n_points
        )
        self.lamb = lamb
        self.embed_dim = embed_dim

    def _encode(self, view: View) -> torch.Tensor:
        """Backbone-encode one view triple to pooled embeddings ``[N, D]``."""
        values, positions, pad_mask = view
        return self.backbone(values, positions, pad_mask)

    def _predict(self, feats: list[torch.Tensor],
                 view_inst: Optional[torch.Tensor]) -> list[torch.Tensor]:
        """Route the ``predictor_inst`` rows of each view through the
        predictor (``view_inst`` is ``[N, n_views]``; unchanged if no
        predictor is configured)."""
        if self.predictor is None or self.predictor_inst is None:
            return feats
        assert view_inst is not None, "view_inst is required with a predictor"
        out = []
        for j, f in enumerate(feats):
            m = (view_inst[:, j] == self.predictor_inst).to(f.device)
            if bool(m.any()):
                # Only the selected rows go through the predictor (so its
                # BatchNorm statistics are that instrument's alone).
                f = f.clone()
                f[m] = self.predictor(f[m]).to(f.dtype)  # autocast may emit half
            out.append(f)
        return out

    def loss_on_views(self, global_views, local_views, view_inst=None):
        """Encode -> (predictor) -> projector -> LeJEPA loss; the pieces the
        training branch of :meth:`forward` is built from, exposed so callers
        can compute the loss in eval mode. Returns
        ``(loss, inv_loss, sigreg_loss, feats, all_projected)`` with
        ``feats`` the raw encoder embeddings per view and ``all_projected``
        ``[V, N, K]``."""
        local_views = local_views or []
        all_views = list(global_views) + list(local_views)
        n_global = len(global_views)
        feats = [self._encode(v) for v in all_views]  # each [N, D]
        pred = self._predict(feats, view_inst)
        all_projected = self.projector(torch.cat(pred))
        bs = global_views[0][0].shape[0]
        all_projected = all_projected.view(len(all_views), bs, -1)
        loss, inv_loss, sigreg_loss = LeJEPA._compute_loss(
            all_projected, n_global, self.sigreg, self.lamb
        )
        return loss, inv_loss, sigreg_loss, feats, all_projected

    def forward(
        self,
        global_views: Optional[list[View]] = None,
        local_views: Optional[list[View]] = None,
        view_inst: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> LeJEPAOutput:
        if self.training:
            assert global_views is not None, (
                "global_views must be provided in training mode"
            )
            n_global = len(global_views)
            # Per-view encode: views differ in token count, so unlike the image
            # method we cannot batch them into a single padded backbone call.
            loss, inv_loss, sigreg_loss, feats, all_projected = self.loss_on_views(
                global_views, local_views, view_inst
            )
            all_features = torch.cat(feats)  # [V*N, D], raw encoder output

            embedding = torch.cat(feats[:n_global]).detach()
            projection = (
                all_projected[:n_global]
                .reshape(-1, all_projected.size(-1))
                .detach()
            )
            return LeJEPAOutput(
                loss=loss,
                embedding=embedding,
                inv_loss=inv_loss,
                sigreg_loss=sigreg_loss,
                projection=projection,
                # Undetached, all views (globals + locals), view-major: for
                # auxiliary embedding-space losses (e.g. a DANN GRL head).
                features=all_features,
            )
        else:
            assert values is not None, "values must be provided in eval mode"
            embedding = self.backbone(values, positions, pad_mask)
            projection = self.projector(embedding).detach()
            zero = torch.tensor(0.0, device=embedding.device)
            return LeJEPAOutput(
                loss=zero,
                embedding=embedding,
                inv_loss=zero,
                sigreg_loss=zero,
                projection=projection,
            )
