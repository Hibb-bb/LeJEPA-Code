"""Unit tests for per-view SIGReg in LeJEPA.

The Epps-Pulley statistic must be computed PER VIEW over the N batch samples
and averaged over views (matching the official LeJEPA reference), never by
pooling the V*N correlated rows into a single sample set — pooling tests the
view mixture instead of the per-view embedding distribution and inflates the
effective sample count (and hence effective lambda) by ~V.

These tests exercise the pure-torch ``EppsPulley``/``SlicedEppsPulley`` math
in a single process, where the DDP all-reduce is a no-op.
"""

import pytest
import torch

# stable_pretraining.methods.lejepa imports the package Module, which pulls
# in lightning at import time.
pytest.importorskip("lightning")

from stable_pretraining.methods.lejepa import EppsPulley, SlicedEppsPulley  # noqa: E402


@pytest.mark.unit
class TestEppsPulleyPerView:
    """Per-view Epps-Pulley statistic (single process)."""

    def test_duplicated_view_invariance(self):
        """V identical copies of a batch give the same statistic as the batch.

        The old pooled code violated this by a factor of ~V because it scaled
        the statistic by the inflated sample count V*N.
        """
        torch.manual_seed(0)
        V, N, D = 4, 32, 16
        x = torch.randn(N, D)
        sigreg = SlicedEppsPulley(num_slices=64)

        single = sigreg(x)
        # Rewind the step counter so the stacked call draws the same slices.
        sigreg.global_step.zero_()
        stacked = sigreg(x.unsqueeze(0).expand(V, N, D))

        torch.testing.assert_close(stacked, single, rtol=1e-5, atol=1e-7)

    def test_2d_backward_compat(self):
        """[N, S] input reproduces the original dim-0 computation exactly."""
        torch.manual_seed(1)
        N, S = 64, 8
        x = torch.randn(N, S)
        ep = EppsPulley()

        out = ep(x)

        # Hand-rolled reference of the old 2-D-only implementation.
        x_t = x.unsqueeze(-1) * ep.t
        cos_mean = x_t.cos().mean(0)
        sin_mean = x_t.sin().mean(0)
        err = (cos_mean - ep.phi).square() + sin_mean.square()
        ref = (err @ ep.weights) * N

        assert out.shape == (S,)
        torch.testing.assert_close(out, ref)

    def test_stacked_equals_mean_of_single_view_statistics(self):
        """[V, N, S] statistic equals per-view statistics when views differ.

        Uses pre-projected slices so the same (implicit) slice directions
        apply to every call, avoiding the random projection matrix.
        """
        torch.manual_seed(2)
        V, N, S = 3, 32, 8
        proj = torch.randn(V, N, S)
        ep = EppsPulley()

        stacked = ep(proj)  # [V, S]
        per_view = torch.stack([ep(proj[v]) for v in range(V)])  # [V, S]

        assert stacked.shape == (V, S)
        torch.testing.assert_close(stacked, per_view)
        # The SlicedEppsPulley scalar is the mean over views and slices.
        torch.testing.assert_close(stacked.mean(), per_view.mean())

    def test_gradient_flows(self):
        """Backprop through the multi-view statistic yields finite grads."""
        torch.manual_seed(3)
        V, N, D = 4, 32, 16
        x = torch.randn(V, N, D, requires_grad=True)
        sigreg = SlicedEppsPulley(num_slices=32)

        stat = sigreg(x)
        stat.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0


@pytest.mark.unit
class TestComputeLossPerView:
    """The bug site itself: LeJEPA._compute_loss must feed sigreg per-view."""

    def test_compute_loss_duplicated_views(self):
        """V duplicated views give the same sigreg_loss as a single view.

        This pins the fix at the actual bug site: the old
        ``sigreg(all_projected.reshape(-1, K))`` pooled the V*N correlated
        rows and returned ~V times the single-view statistic.
        """
        from stable_pretraining.methods.lejepa import LeJEPA

        torch.manual_seed(4)
        V, N, K, lamb = 4, 32, 16, 0.02
        x = torch.randn(N, K)
        all_projected = x.unsqueeze(0).expand(V, N, K)

        sigreg = SlicedEppsPulley(num_slices=64)
        single = sigreg(x)

        sigreg.global_step.zero_()  # replay identical slice directions
        loss, inv_loss, sigreg_loss = LeJEPA._compute_loss(
            all_projected, n_global=2, sigreg=sigreg, lamb=lamb
        )

        torch.testing.assert_close(sigreg_loss, single, rtol=1e-5, atol=1e-7)
        # Identical views: the invariance term is exactly zero.
        torch.testing.assert_close(inv_loss, torch.zeros(()), atol=1e-7, rtol=0)
        torch.testing.assert_close(loss, inv_loss + lamb * sigreg_loss)

    def test_compute_loss_matches_per_view_mean(self):
        """With distinct views, sigreg_loss is the mean of per-view statistics."""
        from stable_pretraining.methods.lejepa import LeJEPA

        torch.manual_seed(5)
        V, N, K = 3, 32, 16
        all_projected = torch.randn(V, N, K)

        sigreg = SlicedEppsPulley(num_slices=64)
        _, _, sigreg_loss = LeJEPA._compute_loss(
            all_projected, n_global=2, sigreg=sigreg, lamb=0.02
        )

        per_view = []
        for v in range(V):
            sigreg.global_step.zero_()
            per_view.append(sigreg(all_projected[v]))
        ref = torch.stack(per_view).mean()

        torch.testing.assert_close(sigreg_loss, ref, rtol=1e-5, atol=1e-7)
