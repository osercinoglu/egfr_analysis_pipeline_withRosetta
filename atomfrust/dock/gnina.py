"""GNINA — CNN-rescored docking, driven as a subprocess.

GNINA 1.3 is smina plus a convolutional rescoring network, on the accuracy–cost Pareto
frontier and the one comparator in methods §3.7 that reports DUD-E screening directly. It
inherits smina's command line, so :class:`GninaBackend` is :class:`SminaBackend` with a
different binary, two extra flags and a different score property; the shared invocation is
not restated.

**Subprocess, emphatically.** GNINA links torch and, in its GPU build, CUDA. Importing that
into every worker process of a spawn pool — which is what a Python binding would mean —
would put a multi-hundred-megabyte, driver-version-sensitive dependency in the import path of
a package whose main job is to score Rosetta energies. :meth:`available` is
``shutil.which("gnina")`` and the absence of GNINA costs exactly this backend.

**The CNN score is the point, so it is what ``score`` carries.** ``--cnn_scoring rescore``
docks with the Vina-family scoring function and rescores the resulting poses with the
network, which is the cheap mode; ``refinement`` and ``all`` use the network inside the
search and cost far more. ``CNNaffinity`` is a predicted pK (higher is better) — the opposite
sign convention to smina's ``minimizedAffinity`` in kcal/mol, so the two must never be pooled
into one column without conversion.

**GNINA is precisely the kind of generator S4.2 exists for.** ML-derived scoring can rank a
physically invalid pose highly; the gate in :mod:`atomfrust.dock.posebusters` is what keeps
such a pose out of the frustration analysis, and a per-generator pass rate below 90% is
grounds for excluding the generator or reporting it separately with the failure mode named.
"""

from __future__ import annotations

from atomfrust.dock.smina import SminaBackend

__all__ = ["GninaBackend"]


class GninaBackend(SminaBackend):
    """Dock with GNINA. Same interface, same box logic, CNN score.

    ``cnn_scoring`` is exposed because it changes what the pose *is*, not just how fast it
    arrives: under ``rescore`` the geometry is Vina's and only the ranking is the network's,
    while under ``refinement`` the network moves atoms. A pass-rate comparison against smina
    is only interpretable when this is recorded, so it goes into every pose's metadata via
    the command line.
    """

    name = "gnina"
    binary = "gnina"
    score_property = "CNNaffinity"

    def __init__(self, binary: str | None = None, *, cnn_scoring: str = "rescore", **kwargs) -> None:
        super().__init__(binary=binary, **kwargs)
        self.cnn_scoring = cnn_scoring

    @property
    def extra_args(self) -> tuple[str, ...]:  # type: ignore[override]
        return ("--cnn_scoring", self.cnn_scoring)
