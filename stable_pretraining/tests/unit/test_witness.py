"""Unit tests for the WitnessCallback spectral health certificate."""

import math

import pytest
import torch

pytest.importorskip("lightning")

from stable_pretraining.callbacks.witness import (  # noqa: E402
    WitnessCallback,
    compute_witnesses,
)


def _mp_edges(n: int, d: int, sigma2: float = 1.0) -> tuple:
    """Marchenko-Pastur null bulk edges for n samples in dimension d."""
    edge = math.sqrt(d / n)
    return sigma2 * (1 - edge) ** 2, sigma2 * (1 + edge) ** 2


@pytest.mark.unit
class TestComputeWitnesses:
    """Test the pure spectrum->metrics computation on synthetic data."""

    N = 8192
    D = 64

    def _white(self, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(self.N, self.D, generator=g)

    def test_white_noise_clauses(self):
        """z ~ N(0, I_d) with n >> d passes the scale and two-sided clauses."""
        z = self._white()
        m = compute_witnesses(z, sigma2=1.0, band=(1 / 3, 0.55))
        lo, hi = _mp_edges(self.N, self.D)

        assert m["lo"] == pytest.approx(lo)
        assert m["hi"] == pytest.approx(hi)
        # Clause 1 (scale): normalized trace near 1.
        assert 0.9 <= m["tr"] <= 1.1
        # Clause 2 (two-sided): spectrum inside the slightly slackened MP bulk.
        assert m["lmin"] >= 0.9 * lo
        assert m["lmax"] <= 1.1 * hi
        # Shape: participation ratio near its n >> d expectation d / (1 + d/n).
        assert m["rfrac"] > 0.8
        # NOTE: `healthy` itself may be 0.0 here — white noise sits above
        # band[1] — so the individual clauses are asserted instead.

    def test_rank_collapse(self):
        """Rank-1 z gives lmin ~ 0 and rfrac ~ 1/d, and is not healthy."""
        g = torch.Generator().manual_seed(1)
        direction = torch.randn(self.D, generator=g)
        direction = direction / direction.norm()
        coeffs = torch.randn(self.N, 1, generator=g)
        z = coeffs @ direction[None, :]

        m = compute_witnesses(z)

        assert m["lmin"] < 1e-4
        assert m["reff"] < 1.5
        assert m["rfrac"] < 2.0 / self.D
        assert m["healthy"] == 0.0

    def test_scale_violation(self):
        """3x-scaled white noise gives tr ~ 9 and fails clause 1."""
        z = 3.0 * self._white(seed=2)
        m = compute_witnesses(z)

        assert 8.5 <= m["tr"] <= 9.5
        assert not (0.8 <= m["tr"] <= 1.25)
        assert m["healthy"] == 0.0

    def test_spiked_direction_two_sided_clause(self):
        """One spiked column trips the two-sided clause by an order of magnitude.

        The spike moves the participation ratio a lot (reff drops ~80%) yet
        leaves it far from the collapse value 1 — while lmax overshoots the
        clause-2 threshold 4*hi by ~5x, giving an unambiguous, localized
        detection. At this spike size clauses 1 and 3 fire as well; the point
        of clause 2 is that it names the failure mode (a spiked direction)
        rather than diluting it into an aggregate statistic.
        """
        z = self._white(seed=3)
        z[:, 0] = 5.0 * z[:, 0]
        m = compute_witnesses(z)
        _, hi = _mp_edges(self.N, self.D)

        # Clause 2 (threshold lmax <= 4*hi) catches the spike decisively.
        assert m["lmax"] > 4.0 * hi
        assert m["lmax"] > 20.0
        # The effective rank moves but stays far from collapse (reff ~ 1),
        # and drops below the band, so clause 3 fires too.
        assert m["reff"] > 10.0
        assert m["rfrac"] < 1 / 3
        assert m["healthy"] == 0.0


@pytest.mark.unit
class TestWitnessCallbackConstruction:
    """Test constructor behavior that needs no Trainer."""

    def test_state_key_and_shape_flattening(self):
        cb = WitnessCallback(
            name="witness",
            target="embedding",
            queue_length=256,
            target_shape=(16, 4),
        )
        assert cb.state_key == "WitnessCallback[name=witness]"
        assert cb.target_shape == 64
        assert cb.sigma2 == 1.0
        assert cb.band == (1 / 3, 0.55)
