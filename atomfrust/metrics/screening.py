"""Virtual-screening metrics: AUROC, BEDROC, adjusted logAUC, EF.

Methods §3.9 reports all four because they answer different questions: AUROC is a global
ranking statistic that a method can win while enriching nothing at the top, and BEDROC,
adjusted logAUC and EF weight the early part of the ranking, where a screen is actually
used. Reporting one of them alone hides the divergence, so this module exists to make the
whole set one import.

Every function returns an :class:`~atomfrust.metrics.inference.Estimate` and every one
refuses to bootstrap pooled molecules: a CI on a screening metric requires ``groups``
naming the target each molecule was screened against, because molecules within a target
share the receptor and the chemical series and are not independent (§3.9, R33/R34). The
error is raised rather than warned — a silently over-narrow CI is exactly the failure this
package is meant to prevent.

Score convention: **higher is better** throughout. A frustration index where a low value
means a better binder must be negated by the caller, once, at the call site.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from atomfrust.metrics.inference import Estimate, bootstrap_ci

__all__ = ["auroc", "bedroc", "adjusted_logauc", "enrichment_factor"]


def _validate(
    scores: Sequence[float] | np.ndarray, labels: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Return an ``(n, 2)`` array of ``[score, label]`` rows, or raise.

    Pairs travel as rows from here on so that a bootstrap resamples molecules, not scores
    and labels independently.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel()
    if s.size != y.size:
        raise ValueError(f"scores and labels have lengths {s.size} and {y.size}")
    if s.size == 0:
        raise ValueError("scores is empty")
    if not np.isfinite(s).all():
        raise ValueError("scores contains non-finite values")

    y = np.asarray(y, dtype=np.float64)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be binary 0/1 (inactive/active)")
    n_active = int(y.sum())
    if n_active == 0 or n_active == y.size:
        raise ValueError(
            f"labels must contain both classes; got {n_active} actives of {y.size}"
        )
    return np.column_stack([s, y])


def _require_groups(groups: object, n_boot: int, metric: str) -> None:
    if n_boot > 0 and groups is None:
        raise ValueError(
            f"{metric}: groups is required when n_boot > 0. Molecules within a target share "
            "the receptor and the chemical series, so a molecule-level bootstrap understates "
            "the interval (methods §3.9: paired tests across targets, never pooled "
            "molecule-level). Pass the target label of each molecule as `groups`."
        )


def _estimate(
    rows: np.ndarray,
    statistic,
    groups,
    n_boot: int,
    seed: int,
    method: str,
) -> Estimate:
    """Point value with no bootstrap, or the clustered bootstrap interval around it."""
    if n_boot <= 0:
        value = float(statistic(rows))
        n_groups = None if groups is None else int(np.unique(np.asarray(groups)).size)
        return Estimate(
            value=value, n=rows.shape[0], n_groups=n_groups, method=method, seed=seed
        )
    est = bootstrap_ci(rows, statistic, groups=groups, n_boot=n_boot, seed=seed)
    return Estimate(
        value=est.value,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
        n=est.n,
        n_groups=est.n_groups,
        method=f"{method}+{est.method}",
        n_boot=est.n_boot,
        seed=seed,
    )


# ------------------------------------------------------------------------------ AUROC


def _auroc_value(rows: np.ndarray) -> float:
    """AUROC as the normalised Mann-Whitney U, exact under ties via mid-ranks.

    ``U / (n_pos * n_neg)`` is the definition, not an approximation of the trapezoidal
    area: the two agree exactly, and the rank form needs no curve and handles tied scores
    by mid-ranking rather than by an arbitrary sort order.
    """
    s, y = rows[:, 0], rows[:, 1]
    n_pos = np.count_nonzero(y)
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(s)
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auroc(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    groups: Sequence[object] | np.ndarray | None = None,
    n_boot: int = 0,
    seed: int = 0,
) -> Estimate:
    """Area under the ROC curve. 1.0 perfect, 0.5 random, 0.0 perfectly inverted."""
    _require_groups(groups, n_boot, "auroc")
    rows = _validate(scores, labels)
    return _estimate(rows, _auroc_value, groups, n_boot, seed, "auroc")


# ----------------------------------------------------------------------------- BEDROC


def _bedroc_value(rows: np.ndarray, alpha: float) -> float:
    """BEDROC of Truchon & Bajorath (2007), evaluated in a cancellation-free form.

    Two rewrites keep the small-alpha limit computable in double precision, which the
    printed formula is not:

    - ``cosh(a) - cosh(b) = 2 sinh((a+b)/2) sinh((a-b)/2)``. As alpha -> 0 both cosh terms
      approach 1 and their difference loses every significant digit; the product form has
      no subtraction at all.
    - ``1 - exp(u)`` and ``exp(u) - 1`` become ``-expm1(u)`` and ``expm1(u)``.

    With these, alpha = 1e-6 reproduces AUROC to ~1e-10 rather than to two digits, which is
    what makes the alpha -> 0 limit testable. Ties are mid-ranked, matching :func:`auroc`.
    """
    s, y = rows[:, 0], rows[:, 1]
    n = int(y.sum())
    total = y.size
    if n == 0 or n == total:
        return np.nan
    if alpha <= 0:
        raise ValueError("bedroc alpha must be positive")

    ra = n / total
    # Descending rank: 1 is the best-scoring molecule.
    ranks = stats.rankdata(-s)[y == 1]
    x = ranks / total

    rie_num = np.exp(-alpha * x).mean()
    rie_den = (-np.expm1(-alpha)) / (total * np.expm1(alpha / total))
    rie = rie_num / rie_den

    scale = ra * np.sinh(alpha / 2) / (2 * np.sinh(alpha * (1 - ra) / 2) * np.sinh(alpha * ra / 2))
    offset = -1.0 / np.expm1(alpha * (1 - ra))
    return float(rie * scale + offset)


def bedroc(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    alpha: float = 80.5,
    groups: Sequence[object] | np.ndarray | None = None,
    n_boot: int = 0,
    seed: int = 0,
) -> Estimate:
    """Boltzmann-enhanced discrimination of ROC, early-recognition weighted by ``alpha``.

    Default alpha = 80.5 is the value the methods document adopts from Chaput et al. 2016;
    it puts ~80% of the weight in the top 2% of the ranking. As alpha -> 0 the weighting
    flattens and BEDROC converges to AUROC exactly.
    """
    _require_groups(groups, n_boot, "bedroc")
    rows = _validate(scores, labels)
    return _estimate(
        rows,
        lambda r: _bedroc_value(r, alpha),
        groups,
        n_boot,
        seed,
        f"bedroc(alpha={alpha})",
    )


# -------------------------------------------------------------------- adjusted logAUC


def _roc_points(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC curve as ``(fpr, tpr)`` starting at ``(0, 0)``, one vertex per distinct score."""
    s, y = rows[:, 0], rows[:, 1]
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    # One vertex per tie block: within equal scores the ordering is arbitrary, so only the
    # block boundaries are curve points.
    s_sorted = s[order]
    keep = np.r_[np.diff(s_sorted) != 0, True]
    tpr = np.r_[0.0, tps[keep] / tps[-1]]
    fpr = np.r_[0.0, fps[keep] / fps[-1]]
    return fpr, tpr


def _adjusted_logauc_value(rows: np.ndarray, lam: float) -> float:
    """Semi-log AUC over ``[lam, 1]``, minus the random-classifier value.

    The log abscissa is what gives early enrichment its weight: the first decade of false
    positive rate occupies a third of the axis at lam = 0.001. The random baseline
    ``(1 - lam) / ln(1 / lam)`` is subtracted so 0 means "no better than random" on the
    same scale for any lam, which is the "adjusted" in the name (Mysinger & Shoichet 2010).
    """
    if not 0 < lam < 1:
        raise ValueError("lam must lie strictly between 0 and 1")
    if np.count_nonzero(rows[:, 1]) in (0, rows.shape[0]):
        return np.nan

    fpr, tpr = _roc_points(rows)
    tpr_at_lam = float(np.interp(lam, fpr, tpr))
    keep = fpr > lam
    x = np.r_[lam, fpr[keep]]
    y = np.r_[tpr_at_lam, tpr[keep]]
    if x[-1] < 1.0:
        x = np.r_[x, 1.0]
        y = np.r_[y, y[-1]]

    log_x = np.log10(x)
    area = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(log_x)))
    logauc = area / np.log10(1.0 / lam)
    random_logauc = (1.0 - lam) / np.log(1.0 / lam)
    return logauc - random_logauc


def adjusted_logauc(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    lam: float = 0.001,
    groups: Sequence[object] | np.ndarray | None = None,
    n_boot: int = 0,
    seed: int = 0,
) -> Estimate:
    """Adjusted logAUC over false positive rates in ``[lam, 1]``. 0 means random."""
    _require_groups(groups, n_boot, "adjusted_logauc")
    rows = _validate(scores, labels)
    return _estimate(
        rows,
        lambda r: _adjusted_logauc_value(r, lam),
        groups,
        n_boot,
        seed,
        f"adjusted_logauc(lam={lam})",
    )


# --------------------------------------------------------------------------------- EF


def _ef_value(rows: np.ndarray, fraction: float) -> float:
    """Enrichment factor at the top ``fraction``, ties broken against the method.

    Sorting is by score descending with actives placed *after* inactives inside a tie
    block. A tie at the cut is not evidence, and the alternative — inheriting the input
    order — makes the number depend on how the caller happened to build the array.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    s, y = rows[:, 0], rows[:, 1]
    total = y.size
    n_active = int(y.sum())
    if n_active == 0:
        return np.nan

    n_top = max(1, int(np.ceil(fraction * total)))
    order = np.lexsort((y, -s))
    hits = int(y[order][:n_top].sum())
    return float((hits / n_top) / (n_active / total))


def enrichment_factor(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    fraction: float = 0.01,
    groups: Sequence[object] | np.ndarray | None = None,
    n_boot: int = 0,
    seed: int = 0,
) -> Estimate:
    """Enrichment factor at the top ``fraction`` of the ranking (default EF1%).

    ``n_top = ceil(fraction * N)``, never fewer than one molecule. The ceiling is stated
    because EF is not scale-free: its maximum is ``1 / fraction`` capped by the number of
    actives, so EF values are comparable only at equal library composition.
    """
    _require_groups(groups, n_boot, "enrichment_factor")
    rows = _validate(scores, labels)
    return _estimate(
        rows,
        lambda r: _ef_value(r, fraction),
        groups,
        n_boot,
        seed,
        f"enrichment_factor(fraction={fraction})",
    )
