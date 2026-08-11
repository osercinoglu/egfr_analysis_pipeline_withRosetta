"""Metrics and inference — one implementation, shared by every consumer.

Methods §3.3 and S0.3 require a single implementation of every metric: the CLI, the report
layer, the notebooks and the tests all import from here, so a number quoted in one place
cannot disagree with the same number quoted in another. Pure NumPy/SciPy — no PyRosetta,
no run directory, no I/O.

Everything public returns an :class:`Estimate`, never a bare float, except the two
multiplicity utilities, whose outputs are per-configuration tables rather than single
estimates.
"""

from atomfrust.metrics.inference import (
    Estimate,
    benjamini_hochberg,
    bootstrap_ci,
    maxT_permutation,
    paired_delta,
    pearson,
    spearman,
)
from atomfrust.metrics.screening import adjusted_logauc, auroc, bedroc, enrichment_factor

__all__ = [
    "Estimate",
    "auroc",
    "bedroc",
    "adjusted_logauc",
    "enrichment_factor",
    "pearson",
    "spearman",
    "paired_delta",
    "maxT_permutation",
    "benjamini_hochberg",
    "bootstrap_ci",
]
