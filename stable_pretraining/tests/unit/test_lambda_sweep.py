"""Unit tests for the Step-0 lambda-sweep certifier (pure python, no deps)."""

import pytest

from stable_pretraining.lambda_sweep import (
    certify,
    format_report,
    is_stable,
    run_lambda_sweep,
    select_lambda,
)


def _record(tr=1.0, lmin=0.5, lmax=2.0, reff=30.0, rfrac=0.45, healthy=1.0,
            epoch=0):
    return dict(tr=tr, lmin=lmin, lmax=lmax, reff=reff, rfrac=rfrac,
                healthy=healthy, epoch=epoch)


@pytest.mark.unit
class TestCertify:
    def test_stable_and_bounded_certifies(self):
        history = [_record(epoch=e) for e in range(3)]
        c = certify(history)
        assert c["stable"] and c["bounded"] and c["certified"]

    def test_drifting_reff_fails_stability(self):
        """20% drift in reff between the last two epochs -> unstable.

        An unequilibrated run measures speed, not the equilibrium — it must
        not certify even when the final snapshot happens to look healthy.
        """
        history = [_record(reff=24.0, epoch=0), _record(reff=30.0, epoch=1)]
        c = certify(history)
        assert not c["stable"]
        assert c["bounded"]  # final snapshot IS in bounds...
        assert not c["certified"]  # ...but stability gates it.

    def test_stably_collapsed_fails_boundedness(self):
        """A stably dead state (healthy=0 every epoch) must not certify."""
        history = [
            _record(lmin=1e-6, reff=2.0, rfrac=0.03, healthy=0.0, epoch=e)
            for e in range(3)
        ]
        c = certify(history)
        assert c["stable"]
        assert not c["bounded"]
        assert not c["certified"]

    def test_short_history_is_unstable(self):
        assert not certify([_record()])["certified"]
        assert not certify([])["certified"]
        assert certify([])["last"] is None

    def test_is_stable_tolerance(self):
        a, b = _record(tr=1.00), _record(tr=1.04)
        assert is_stable([a, b], tail=2, rel_tol=0.05)
        assert not is_stable([_record(tr=1.00), _record(tr=1.10)], tail=2,
                             rel_tol=0.05)


@pytest.mark.unit
class TestSelectLambda:
    def _certs(self, outcomes):
        """outcomes: {lam: certified_bool} -> minimal cert dicts."""
        return {
            lam: {
                "certified": ok,
                "stable": ok,
                "bounded": ok,
                "last": _record(healthy=float(ok)),
                "n_records": 2,
            }
            for lam, ok in outcomes.items()
        }

    def test_floor_located_recommends_margin(self):
        """Failure below the smallest pass -> recommend one grid step up."""
        sel = select_lambda(
            self._certs({0.01: False, 0.05: True, 0.1: True, 0.5: True})
        )
        assert sel["smallest_certified"] == 0.05
        assert sel["floor_located"]
        assert sel["recommended"] == 0.1

    def test_floor_unlocated_warns(self):
        """Bottom of the grid certifies -> no margin claimable."""
        sel = select_lambda(self._certs({0.01: True, 0.05: True}))
        assert sel["smallest_certified"] == 0.01
        assert not sel["floor_located"]
        assert sel["recommended"] == 0.01
        assert "downward" in sel["reason"]

    def test_non_contiguous_certified_set_denies_margin(self):
        """pass, fail, pass above the floor -> no margin claim, warn."""
        sel = select_lambda(
            self._certs({0.01: False, 0.05: True, 0.1: False, 0.5: True})
        )
        assert sel["smallest_certified"] == 0.05
        assert sel["floor_located"]
        assert sel["recommended"] == 0.05
        assert "non-contiguous" in sel["reason"]

    def test_nothing_certifies(self):
        sel = select_lambda(self._certs({0.01: False, 0.05: False}))
        assert sel["smallest_certified"] is None
        assert sel["recommended"] is None
        assert "upward" in sel["reason"]

    def test_floor_located_but_no_headroom(self):
        """Only the topmost grid point certifies -> ask for a larger grid."""
        sel = select_lambda(self._certs({0.01: False, 0.05: True}))
        assert sel["smallest_certified"] == 0.05
        assert sel["floor_located"]
        assert sel["recommended"] == 0.05
        assert "upward" in sel["reason"]


@pytest.mark.unit
class TestRunLambdaSweep:
    def test_end_to_end(self, capsys):
        """train_fn returning canned histories reproduces the selection."""

        def train_fn(lam):
            if lam < 0.05:  # collapsed
                return [
                    _record(lmin=0.0, reff=1.5, rfrac=0.02, healthy=0.0,
                            epoch=e)
                    for e in range(2)
                ]
            return [_record(epoch=e) for e in range(2)]

        result = run_lambda_sweep(train_fn, [0.5, 0.01, 0.1, 0.05],
                                  verbose=True)
        assert result["selection"]["smallest_certified"] == 0.05
        assert result["selection"]["recommended"] == 0.1
        out = capsys.readouterr().out
        assert "recommended lambda_ref: 0.1" in out
        # The report table renders every grid point.
        for lam in ("0.01", "0.05", "0.1", "0.5"):
            assert lam in out

    def test_format_report_handles_empty_history(self):
        certs = {0.02: certify([])}
        text = format_report(certs, select_lambda(certs))
        assert "NONE" in text


@pytest.mark.unit
class TestGuards:
    def test_tail_below_two_raises(self):
        with pytest.raises(ValueError):
            is_stable([_record(), _record()], tail=1)

    def test_collapsed_lmin_fluctuation_is_stable(self):
        """Near-zero lmin noise must not read as instability.

        A stably collapsed run should fail on boundedness, so the sweep
        report points at the state, not at run length.
        """
        a = _record(lmin=1e-6, reff=2.0, rfrac=0.03, healthy=0.0)
        b = _record(lmin=1e-8, reff=2.0, rfrac=0.03, healthy=0.0)
        c = certify([a, b])
        assert c["stable"] and not c["bounded"] and not c["certified"]
