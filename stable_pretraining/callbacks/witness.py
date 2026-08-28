import math
from typing import Iterable, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .registry import log as _spt_log

from .queue import find_or_create_queue_callback


def compute_witnesses(
    z: torch.Tensor, sigma2: float = 1.0, band: tuple = (1 / 3, 0.55)
) -> dict:
    """Compute the three-clause spectral health witnesses of queued embeddings.

    Given ``n`` embedding vectors ``z`` (one per row, dimension ``d``), forms
    the *uncentered* second-moment matrix ``S = z.T @ z / n`` in float32 on CPU
    and derives five scalars from its eigenvalues, plus a binary health
    certificate. The null model is ``z ~ N(0, sigma2 * I_d)``, whose sample
    spectrum concentrates on the Marchenko-Pastur bulk
    ``[lo, hi] = sigma2 * (1 -/+ sqrt(d / n)) ** 2`` computed from the actual
    number of rows ``n``.

    Args:
        z: Tensor of shape ``(n, d)`` holding the queued embeddings. Any
            floating dtype and device are accepted; the computation moves the
            data to CPU float32 (fp16 queues, and MPS lacks ``eigvalsh``).
        sigma2: Per-coordinate variance of the null model.
        band: Inclusive ``(low, high)`` bounds on ``rfrac = reff / d`` for the
            shape clause.

    Returns:
        Dict of Python floats with keys ``tr`` (clause 1, scale:
        ``ev.sum() / (d * sigma2)``), ``lmin`` / ``lmax`` (clause 2, two-sided
        extreme eigenvalues), ``reff`` (clause 3, shape: participation ratio
        ``ev.sum()**2 / (ev**2).sum()``), ``rfrac`` (``reff / d``), ``healthy``
        (1.0 iff all three clauses hold, else 0.0), and the Marchenko-Pastur
        null edges ``lo`` / ``hi``. Clause 2 passes iff
        ``0.15 * lo <= lmin`` and ``lmax <= 4.0 * hi`` — O(1) Loewner
        constants widened by the MP sampling factors, calibrated so the full
        ``rfrac`` band remains jointly attainable with clause 2 (see the
        inline note at the ``healthy`` computation).

    Note:
        When ``n < d`` the second-moment matrix is rank-deficient, ``lmin``
        is exactly 0, and the certificate is 0.0 by construction — a
        rank-deficient sample cannot certify health. Choose ``queue_length``
        comfortably above ``d``.
    """
    z = z.detach().float().cpu()
    n, d = z.shape

    S = z.T @ z / n
    ev = torch.linalg.eigvalsh(S)

    total = ev.sum()
    tr = (total / (d * sigma2)).item()
    lmin = ev[0].item()
    lmax = ev[-1].item()
    reff = (total**2 / (ev**2).sum().clamp(min=1e-12)).item()
    rfrac = reff / d

    edge = math.sqrt(d / n)
    lo = sigma2 * (1 - edge) ** 2
    hi = sigma2 * (1 + edge) ** 2

    # Clause-2 thresholds: O(1) Loewner constants (c, C) = (0.15, 4.0) widened
    # by the MP sampling factors. The healthy state carries Theta(1)
    # *population* anisotropy, so the admissible spread must not shrink to the
    # MP bulk as n grows: a spectrum with mean sigma2 supported in
    # [c, C] * sigma2 can reach eigenvalue variance (1 - c)(C - 1) = 2.55
    # (two-point bound), covering the whole default rfrac band down to
    # 1 / (1 + 2) = 1/3. Tighter multiples of the bare MP edges (e.g.
    # [0.5 lo, 2 hi]) make clauses 2 and 3 jointly unsatisfiable at n >> d.
    healthy = float(
        (0.8 <= tr <= 1.25)
        and (lmin >= 0.15 * lo)
        and (lmax <= 4.0 * hi)
        and (band[0] <= rfrac <= band[1])
    )

    return {
        "tr": tr,
        "lmin": lmin,
        "lmax": lmax,
        "reff": reff,
        "rfrac": rfrac,
        "healthy": healthy,
        "lo": lo,
        "hi": hi,
    }


class WitnessCallback(Callback):
    """Three-clause spectral health certificate monitor using queue discovery.

    Logs a health certificate of the uncentered embedding second-moment
    spectrum against the Marchenko-Pastur null ``z ~ N(0, sigma2 * I_d)``:
    clause 1 checks the scale (normalized trace ``tr``), clause 2 checks both
    extreme eigenvalues (``lmin`` / ``lmax``) against the null bulk edges, and
    clause 3 checks the shape (participation ratio ``reff`` relative to ``d``).
    The two-sided clause catches spiked or dead directions that a rank
    statistic alone under-reports. ``queue_length`` must comfortably exceed
    the embedding dimension: with fewer queued rows than dimensions the
    sample spectrum is rank-deficient and the certificate is 0.0 by
    construction.

    Args:
        name: Unique name for this callback instance. Used for logging and
            metric keys (``{name}/tr``, ``{name}/lmin``, ``{name}/lmax``,
            ``{name}/reff``, ``{name}/rfrac``, ``{name}/healthy``).
        target: Key in the batch dict containing the feature embeddings to
            monitor.
        queue_length: Size of the circular buffer for caching embeddings across
            validation batches. Larger values give a more representative
            estimate.
        target_shape: Shape of the target embeddings — either a single int
            (e.g., ``768``) or a sequence whose product is used (e.g.,
            ``(16, 48)``).
        sigma2: Per-coordinate variance of the null model.
        band: Inclusive ``(low, high)`` bounds on ``rfrac = reff / d`` for the
            shape clause.
        verbose: If ``True``, also log the Marchenko-Pastur null edges
            (``{name}/mp_lo``, ``{name}/mp_hi``). ``None`` inherits the global
            ``spt`` verbosity setting.
    """

    def __init__(
        self,
        name: str,
        target: str,
        queue_length: int,
        target_shape: Union[int, Iterable[int]],
        sigma2: float = 1.0,
        band: tuple = (1 / 3, 0.55),
        verbose: bool = None,
    ) -> None:
        super().__init__()

        if isinstance(target_shape, (list, tuple)):
            if len(target_shape) == 1:
                target_shape = target_shape[0]
            else:
                target_shape = int(torch.prod(torch.tensor(target_shape)))

        self.name = name
        self.target = target
        self.queue_length = queue_length
        self.target_shape = target_shape
        self.sigma2 = sigma2
        self.band = band
        from .utils import resolve_verbose

        self.verbose = resolve_verbose(verbose)

        self._target_queue = None

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"WitnessCallback[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find or create the queue callback for target features."""
        if self._target_queue is None:
            self._target_queue = find_or_create_queue_callback(
                trainer,
                self.target,
                self.queue_length,
                self.target_shape,
                torch.float32,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"  target queue: {self.target}")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute the witness metrics on the first validation batch only."""
        if batch_idx > 0:
            return

        logging.info("  computing witnesses on first validation batch")

        embeddings = self._target_queue.data

        if embeddings is None:
            logging.warning(
                f"! {self.name}: queue data not available (not in validation?)"
            )
            return

        if embeddings.numel() == 0:
            logging.warning(
                f"! {self.name}: queue data is empty, skipping witness computation"
            )
            return

        if trainer.global_rank == 0:
            with torch.no_grad():
                metrics = compute_witnesses(embeddings, self.sigma2, self.band)

                for key in ("tr", "lmin", "lmax", "reff", "rfrac", "healthy"):
                    pl_module.log(f"{self.name}/{key}", metrics[key])
                if self.verbose:
                    _spt_log(f"{self.name}/mp_lo", metrics["lo"])
                    _spt_log(f"{self.name}/mp_hi", metrics["hi"])
