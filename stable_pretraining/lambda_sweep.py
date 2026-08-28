"""Automatic SIGReg-lambda selection by spectral health certification.

Implements the Step-0 protocol: sweep a grid of lambda values, train each
candidate briefly, and certify each run from the per-epoch history of the
three spectral health witnesses (:func:`compute_witnesses` in
:mod:`stable_pretraining.callbacks.witness`):

- clause 1 (scale): normalized trace ``tr = trace(S) / (d * sigma2)``;
- clause 2 (two-sided): extreme eigenvalues ``lmin`` / ``lmax`` of the
  uncentered second-moment matrix, against O(1) Loewner constants widened
  by the Marchenko-Pastur sampling factors;
- clause 3 (shape): participation ratio ``rfrac = reff / d`` inside a band.

A lambda is **certified** iff its witnesses are simultaneously

- **bounded** — the final epoch's ``healthy`` flag is 1.0 (all three
  clauses hold), and
- **stable** — every witness agrees across the last ``tail`` recorded
  epochs to ``rel_tol`` relative tolerance.

Stability alone would accept stably-collapsed runs; boundedness alone would
accept transients passing through the band. The selection rule is *smallest
certified lambda with one grid step of margin*: the loss cannot rank lambda
values (each lambda defines a different objective, and collapse attains a
perfect prediction loss), so the certificate replaces it on this axis.

This module is dependency-free (no lightning import); it consumes plain
lists of per-epoch witness dicts. ``WitnessCallback.history`` provides them
in trainer-based runs, and any notebook loop can build them directly with
``compute_witnesses``.

Example (with a Lightning trainer; see
``benchmarks/imagenet10/lejepa-mup-ladder.py --sweep-lamb`` for a complete
CLI)::

    from stable_pretraining.lambda_sweep import run_lambda_sweep

    def train_fn(lamb):
        witness = WitnessCallback(name="w", target="projection",
                                  queue_length=2048, target_shape=proj_dim)
        ...build module and trainer with [witness], fit...
        return witness.history

    result = run_lambda_sweep(train_fn, [0.005, 0.02, 0.08, 0.3])
    anchor = result["selection"]["recommended"]
"""

from typing import Callable, Dict, Iterable, List, Sequence

#: Witness keys whose record-to-record agreement defines stability.
WITNESS_KEYS: Sequence[str] = ("tr", "lmin", "lmax", "reff")


def is_stable(
    history: List[dict], tail: int = 2, rel_tol: float = 0.05,
    atol: float = 1e-3
) -> bool:
    """True iff every witness agrees across the last ``tail`` records.

    Agreement is relative with an absolute floor: ``|h[k] - last[k]| <=
    rel_tol * max(|last[k]|, atol)`` for every ``k`` in
    :data:`WITNESS_KEYS` and every record ``h`` of the last ``tail``. The
    ``atol`` floor keeps near-zero witnesses (e.g. ``lmin`` of a collapsed
    run) from registering as unstable on meaningless relative fluctuation —
    a stably collapsed run should fail on *boundedness*, so the diagnosis
    points at the state, not the run length. Fewer than ``tail`` records is
    unstable by definition (an unequilibrated run measures speed, not the
    equilibrium).

    ``tail`` counts *records*, not epochs: with one validation pass per
    epoch (the default) they coincide, but under sub-epoch validation
    schedules the stability window shrinks accordingly. ``tail`` must be at
    least 2 — ``tail=1`` would reduce certification to boundedness alone.
    """
    if tail < 2:
        raise ValueError(f"tail must be >= 2, got {tail}")
    if len(history) < tail:
        return False
    last = history[-1]
    for h in history[-tail:-1]:
        for k in WITNESS_KEYS:
            if abs(h[k] - last[k]) > rel_tol * max(abs(last[k]), atol):
                return False
    return True


def certify(
    history: List[dict], tail: int = 2, rel_tol: float = 0.05
) -> dict:
    """Certify one lambda's run from its witness-record history.

    Args:
        history: Per-record witness dicts (one per validation pass, each
            carrying at least :data:`WITNESS_KEYS`, ``rfrac`` and
            ``healthy``), oldest first.
        tail: Number of trailing records that must agree for stability
            (must be >= 2).
        rel_tol: Relative tolerance of that agreement.

    Returns:
        Dict with ``stable``, ``bounded`` (final ``healthy`` flag),
        ``certified`` (= stable and bounded), ``last`` (the final record or
        None), and ``n_records``.
    """
    last = history[-1] if history else None
    stable = is_stable(history, tail=tail, rel_tol=rel_tol)
    bounded = bool(last is not None and last.get("healthy", 0.0) >= 1.0)
    return {
        "stable": stable,
        "bounded": bounded,
        "certified": stable and bounded,
        "last": last,
        "n_records": len(history),
    }


def select_lambda(certs: Dict[float, dict]) -> dict:
    """Smallest certified lambda, with one grid step of margin.

    The floor (smallest admissible lambda) is *located* when the grid point
    immediately below the smallest certified one exists and fails; the
    recommended anchor is then one grid step above the floor edge, so the
    transferred lambda at other rungs keeps clearance. When the bottom of
    the grid itself certifies, the floor is below the grid and no margin
    can be claimed — the smallest certified value is returned with a
    warning to extend the grid downward.

    Returns:
        Dict with ``smallest_certified``, ``recommended`` (both None when
        nothing certifies), ``floor_located`` and a human-readable
        ``reason``.
    """
    lams = sorted(certs)
    passing = [lam for lam in lams if certs[lam]["certified"]]
    if not passing:
        unstable = [lam for lam in lams if not certs[lam]["stable"]]
        reason = (
            "no lambda certified — extend the grid upward"
            + (
                f"; {len(unstable)} row(s) failed on stability, so also "
                "consider training longer (an unequilibrated sweep measures "
                "speed, not the equilibrium)"
                if unstable
                else ""
            )
        )
        return {
            "smallest_certified": None,
            "recommended": None,
            "floor_located": False,
            "reason": reason,
        }

    smallest = passing[0]
    idx = lams.index(smallest)
    floor_located = idx > 0 and not certs[lams[idx - 1]]["certified"]
    # Margin can only be claimed off a CONTIGUOUS certified run: the next
    # grid point above the floor edge must itself certify. A gap in the
    # certified set (pass, fail, pass) signals noise or a second failure
    # mode at larger lambda and must not be glossed as margin.
    next_up = lams[idx + 1] if idx + 1 < len(lams) else None
    contiguous = next_up is not None and certs[next_up]["certified"]

    if floor_located and contiguous:
        recommended = next_up
        reason = (
            f"floor located between {lams[idx - 1]:g} and {smallest:g}; "
            f"recommending {recommended:g} = smallest certified plus one "
            "grid step of margin"
        )
    elif floor_located and next_up is not None:
        recommended = smallest
        reason = (
            f"floor located below {smallest:g}, but the next grid point "
            f"{next_up:g} does NOT certify — the certified set is "
            "non-contiguous (noise near the floor, or a second failure "
            "mode at larger lambda); investigate before anchoring"
        )
    elif floor_located:
        recommended = smallest
        reason = (
            f"floor located below {smallest:g}, but no certified value "
            "above it for margin — extend the grid upward before anchoring"
        )
    else:
        recommended = smallest
        reason = (
            f"the bottom of the grid ({smallest:g}) already certifies, so "
            "the floor is unlocated — extend the grid downward to claim a "
            "margin"
        )

    return {
        "smallest_certified": smallest,
        "recommended": recommended,
        "floor_located": floor_located,
        "reason": reason,
    }


def format_report(certs: Dict[float, dict], selection: dict) -> str:
    """Plain-text sweep report: one row per lambda, then the selection."""
    header = (
        f"{'lambda':>10} {'tr':>7} {'lmin':>8} {'lmax':>8} {'rfrac':>7} "
        f"{'stable':>7} {'bounded':>8} {'certified':>10}"
    )
    lines = [header, "-" * len(header)]
    for lam in sorted(certs):
        c = certs[lam]
        last = c["last"] or {}
        lines.append(
            f"{lam:>10g} "
            f"{last.get('tr', float('nan')):>7.3f} "
            f"{last.get('lmin', float('nan')):>8.3f} "
            f"{last.get('lmax', float('nan')):>8.3f} "
            f"{last.get('rfrac', float('nan')):>7.3f} "
            f"{str(c['stable']):>7} {str(c['bounded']):>8} "
            f"{str(c['certified']):>10}"
        )
    lines.append("")
    rec = selection["recommended"]
    lines.append(
        f"recommended lambda_ref: {rec:g}" if rec is not None
        else "recommended lambda_ref: NONE"
    )
    lines.append(f"  ({selection['reason']})")
    return "\n".join(lines)


def run_lambda_sweep(
    train_fn: Callable[[float], List[dict]],
    lambdas: Iterable[float],
    tail: int = 2,
    rel_tol: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Sweep lambda, certify each run, and select the anchor.

    Args:
        train_fn: Called once per candidate with the lambda value; must
            train (briefly) at that lambda and return the per-epoch witness
            history of the space SIGReg acts on (e.g.
            ``WitnessCallback.history``).
        lambdas: Grid of candidate lambda values (additive convention).
        tail: Trailing witness records that must agree for stability
            (records = validation passes; must be >= 2).
        rel_tol: Relative tolerance of that agreement.
        verbose: Print the report table.

    Returns:
        ``{"certs": {lambda: cert}, "selection": {...}}`` — see
        :func:`certify` and :func:`select_lambda`.
    """
    certs: Dict[float, dict] = {}
    for lam in sorted(set(float(x) for x in lambdas)):
        certs[lam] = certify(train_fn(lam), tail=tail, rel_tol=rel_tol)
    selection = select_lambda(certs)
    if verbose:
        print(format_report(certs, selection))
    return {"certs": certs, "selection": selection}
