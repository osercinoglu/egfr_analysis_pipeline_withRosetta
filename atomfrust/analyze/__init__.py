"""Analysis over stored decoy energies — pure arrays, no PyRosetta anywhere.

Everything here is a *query* over a finished run directory. That is the point of storing
direct pair energies rather than ``E_ij``: the many-body formula, the contact definition,
the shell, the index function and the thresholds are all chosen here, after the expensive
part is over, so re-analysing under different settings costs no Rosetta time.
"""

from atomfrust.analyze.classify import (
    CLASSES,
    class_counts,
    classify_index,
    quantile_thresholds,
)
from atomfrust.analyze.converge import (
    DEFAULT_GRID,
    convergence_curve,
    n_star,
    sigma_standard_error,
)
from atomfrust.analyze.strata import (
    REDUNDANCY_THRESHOLD,
    STRATUM_AXES,
    assign_strata,
    axis_redundancy,
    pocket_descriptors,
    sigma_by_stratum,
)
from atomfrust.analyze.zscore import (
    INDEX_FUNCTIONS,
    compute_all_indices,
    compute_index,
    decoy_summary,
    normality_diagnostics,
    rank_percentile,
    robust_z,
    zscore,
)

__all__ = [
    # zscore (D1)
    "INDEX_FUNCTIONS",
    "compute_all_indices",
    "compute_index",
    "decoy_summary",
    "normality_diagnostics",
    "rank_percentile",
    "robust_z",
    "zscore",
    # classify (D2)
    "CLASSES",
    "class_counts",
    "classify_index",
    "quantile_thresholds",
    # converge (D4)
    "DEFAULT_GRID",
    "convergence_curve",
    "n_star",
    "sigma_standard_error",
    # strata (D5)
    "REDUNDANCY_THRESHOLD",
    "STRATUM_AXES",
    "assign_strata",
    "axis_redundancy",
    "pocket_descriptors",
    "sigma_by_stratum",
]
