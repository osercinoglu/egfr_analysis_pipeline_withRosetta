"""Estimation and testing: the :class:`Estimate` type, resampling, and multiplicity.

Three decisions are baked into this module because getting them wrong is what makes a
swept-parameter tool report noise as a finding.

**Every estimate is an interval, not a number.** :class:`Estimate` carries the point value,
its CI, the sample sizes it came from and the seed that produced it. A bare float loses the
information needed to judge it, and once one function returns a float the convention is
gone — so nothing here returns one.

**Resampling is clustered.** Molecules within a target are not independent: they share the
receptor, the pocket, the docking protocol and usually a chemical series. The methods
document (§3.9) requires "paired tests across targets, never pooled molecule-level tests".
:func:`bootstrap_ci` therefore resamples *targets* with replacement and then molecules
within each drawn target, which is the two-stage cluster bootstrap. Passing ``groups=None``
gives the i.i.d. bootstrap, which is correct only when the rows really are exchangeable —
the screening metrics refuse it outright.

**Multiplicity is a maximum, not a family.** Benjamini-Hochberg controls the false discovery
rate over a *fixed, pre-specified* family of tests. This tool makes contact definition,
cutoff, shell, many-body mode, index function, thresholds, descriptor and decoy axis all
selectable, so a reported best-case correlation is a maximum over >10^4 configurations, and
BH does not price a maximum. :func:`maxT_permutation` builds the null distribution of the
*largest* |r| over the whole grid, which is what a reported maximum requires (plan §2.3).
:func:`benjamini_hochberg` is provided for the genuinely pre-specified families of §3.9, and
for no other use. For scale: at n = 19 the 5% critical value for a *single* pre-specified
test is already r = 0.456.

All randomness comes from ``np.random.default_rng(seed)``. There is no global RNG state
here, so a result is reproducible from the seed recorded on the estimate that carries it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "Estimate",
    "pearson",
    "spearman",
    "paired_delta",
    "maxT_permutation",
    "benjamini_hochberg",
    "bootstrap_ci",
]


@dataclass(frozen=True)
class Estimate:
    """A point estimate with everything needed to judge it.

    ``ci_low``/``ci_high`` are ``None`` when no bootstrap was requested — absent, not zero.
    ``n`` counts observations (molecules, systems, pairs), ``n_groups`` counts the clusters
    they were resampled in, and ``method`` names both the statistic and the interval
    procedure so a stored table can explain itself without the calling code.
    """

    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = 0
    n_groups: int | None = None
    method: str = ""
    n_boot: int = 0
    seed: int | None = None
    p_value: float | None = None


# ------------------------------------------------------------------------- resampling


def _as_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _group_codes(groups: Sequence[object] | np.ndarray | None, n: int) -> np.ndarray | None:
    """Dense integer codes for group labels of any dtype, or ``None`` for ungrouped data."""
    if groups is None:
        return None
    labels = np.asarray(groups).ravel()
    if labels.size != n:
        raise ValueError(f"groups has length {labels.size}, expected {n}")
    codes, _ = pd.factorize(pd.Series(labels))
    return np.asarray(codes, dtype=np.int64)


def _resample_indices(
    rng: np.random.Generator, n: int, codes: np.ndarray | None, n_boot: int
) -> list[np.ndarray]:
    """Bootstrap index sets: i.i.d. rows, or targets-then-molecules-within-target.

    The two-stage draw keeps the number of clusters fixed at the observed count and lets
    cluster sizes vary, which is what propagates between-target variance into the interval.
    A single-stage draw over pooled rows would understate it by exactly the amount that
    matters here — the between-target component.
    """
    if codes is None:
        return [rng.integers(0, n, size=n) for _ in range(n_boot)]

    members = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    n_groups = len(members)
    out = []
    for _ in range(n_boot):
        drawn = rng.integers(0, n_groups, size=n_groups)
        out.append(
            np.concatenate(
                [members[g][rng.integers(0, members[g].size, size=members[g].size)] for g in drawn]
            )
        )
    return out


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    groups: Sequence[object] | np.ndarray | None = None,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Estimate:
    """Percentile bootstrap CI for ``statistic`` evaluated on the rows of ``values``.

    ``values`` is resampled along its first axis, so a paired statistic is bootstrapped by
    passing the pairs as columns (``np.column_stack([scores, labels])``) rather than by
    resampling the two vectors separately, which would break the pairing.

    Replicates that come out non-finite are dropped rather than propagated: a resample can
    legitimately contain one label class only, for which a ranking metric is undefined. The
    count that survived is not silently lost — it is the ``n_boot`` on the returned estimate.
    """
    values = np.asarray(values)
    if values.shape[0] == 0:
        raise ValueError("values is empty")

    point = float(statistic(values))
    codes = _group_codes(groups, values.shape[0])
    n_groups = None if codes is None else int(codes.max() + 1)
    method = "bootstrap_percentile" if codes is None else "cluster_bootstrap_percentile"

    if n_boot <= 0:
        return Estimate(
            value=point, n=values.shape[0], n_groups=n_groups, method="point", seed=seed
        )

    rng = np.random.default_rng(seed)
    replicates = np.array(
        [statistic(values[idx]) for idx in _resample_indices(rng, values.shape[0], codes, n_boot)],
        dtype=np.float64,
    )
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        raise ValueError("every bootstrap replicate was non-finite; the statistic is degenerate")

    low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(
        value=point,
        ci_low=float(low),
        ci_high=float(high),
        n=values.shape[0],
        n_groups=n_groups,
        method=method,
        n_boot=int(finite.size),
        seed=seed,
    )


# ------------------------------------------------------------------------ correlations


def _correlation(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    groups: Sequence[object] | np.ndarray | None,
    n_boot: int,
    seed: int,
    kind: str,
) -> Estimate:
    x = _as_1d(x, "x")
    y = _as_1d(y, "y")
    if x.size != y.size:
        raise ValueError(f"x and y have lengths {x.size} and {y.size}")
    if x.size < 3:
        raise ValueError(f"{kind} needs at least 3 observations, got {x.size}")

    fn = stats.pearsonr if kind == "pearson" else stats.spearmanr
    result = fn(x, y)
    value, p_value = float(result[0]), float(result[1])

    pairs = np.column_stack([x, y])

    def statistic(rows: np.ndarray) -> float:
        if np.ptp(rows[:, 0]) == 0 or np.ptp(rows[:, 1]) == 0:
            return np.nan
        return float(fn(rows[:, 0], rows[:, 1])[0])

    if n_boot <= 0:
        codes = _group_codes(groups, x.size)
        return Estimate(
            value=value,
            n=x.size,
            n_groups=None if codes is None else int(codes.max() + 1),
            method=kind,
            seed=seed,
            p_value=p_value,
        )

    est = bootstrap_ci(pairs, statistic, groups=groups, n_boot=n_boot, seed=seed)
    # The parametric p is kept because it is what the methods document reports, but it
    # assumes independent observations; when groups were supplied the CI is the honest part.
    return Estimate(
        value=value,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
        n=est.n,
        n_groups=est.n_groups,
        method=f"{kind}+{est.method}",
        n_boot=est.n_boot,
        seed=seed,
        p_value=p_value,
    )


def pearson(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    groups: Sequence[object] | np.ndarray | None = None,
    n_boot: int = 0,
    seed: int = 0,
) -> Estimate:
    """Pearson r — the affinity-relationship statistic of methods §3.9."""
    return _correlation(x, y, groups, n_boot, seed, "pearson")


def spearman(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    groups: Sequence[object] | np.ndarray | None = None,
    n_boot: int = 0,
    seed: int = 0,
) -> Estimate:
    """Spearman rho — the statistic for ranking claims, per methods §3.9."""
    return _correlation(x, y, groups, n_boot, seed, "spearman")


# ----------------------------------------------------------------------- paired tests


def paired_delta(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> Estimate:
    """Mean per-target difference ``a - b``, with a bootstrap CI and a sign-flip p.

    Method comparison happens at the target level and nowhere else (R34). Observations are
    collapsed to one difference per group first, so a target contributing 500 molecules and
    a target contributing 12 weigh the same and no within-target correlation is mistaken for
    evidence. ``groups`` is mandatory for exactly that reason.

    The p-value flips the signs of the per-target differences, which is the permutation null
    for a paired design under exchangeability of the sign of the effect. It is randomised
    rather than exhaustive, so it carries the usual ``(count + 1) / (n_boot + 1)`` floor.
    """
    a = _as_1d(a, "a")
    b = _as_1d(b, "b")
    if a.size != b.size:
        raise ValueError(f"a and b have lengths {a.size} and {b.size}")
    if groups is None:
        raise ValueError("paired_delta requires groups: the comparison unit is the target")
    codes = _group_codes(groups, a.size)
    assert codes is not None  # narrowed by the check above
    n_groups = int(codes.max() + 1)
    if n_groups < 2:
        raise ValueError("paired_delta needs at least 2 groups; comparison is across targets")

    per_group = np.array([np.mean(a[codes == g] - b[codes == g]) for g in range(n_groups)])
    value = float(per_group.mean())

    if n_boot <= 0:
        return Estimate(
            value=value, n=a.size, n_groups=n_groups, method="paired_delta", seed=seed
        )

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    replicates = per_group[draws].mean(axis=1)
    low, high = np.percentile(replicates, [2.5, 97.5])

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, n_groups))
    null = (signs * per_group).mean(axis=1)
    p_value = (np.count_nonzero(np.abs(null) >= abs(value)) + 1) / (n_boot + 1)

    return Estimate(
        value=value,
        ci_low=float(low),
        ci_high=float(high),
        n=a.size,
        n_groups=n_groups,
        method="paired_delta+group_bootstrap+sign_flip",
        n_boot=n_boot,
        seed=seed,
        p_value=float(p_value),
    )


# ---------------------------------------------------------------------- multiplicity


def maxT_permutation(
    grid: dict[str, np.ndarray],
    y: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Max-T permutation adjustment across a whole configuration grid.

    ``grid`` maps a configuration label to its descriptor vector; every vector and ``y``
    share one observation order. The statistic is |Pearson r|; a Spearman version is the
    same call on rank-transformed inputs, which is why there is no mode switch.

    On each permutation the outcome is shuffled *once* and the largest |r| over the entire
    grid is recorded. Comparing an observed |r| against that distribution asks "how often
    does a grid this size throw up a correlation this large by chance", which is the
    question a reported maximum actually poses. Per-configuration p-values are returned
    beside it as ``p_raw`` so the price of the sweep is visible as the gap between the two.

    Permuting the outcome, not the descriptors, keeps the between-configuration correlation
    structure intact — configurations differing in one setting are near-duplicates, and that
    redundancy makes the max-T null far less conservative than a Bonferroni over 10^4 tests.

    Returns one row per configuration in grid order, with columns ``config``, ``statistic``
    (signed r), ``abs_statistic``, ``p_raw``, ``p_adjusted``, ``n``, ``n_perm``, ``seed``.
    """
    if not grid:
        raise ValueError("grid is empty")
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1")

    names = list(grid)
    y = _as_1d(y, "y")
    n = y.size

    X = np.empty((len(names), n), dtype=np.float64)
    for row, name in enumerate(names):
        vector = _as_1d(grid[name], f"grid[{name!r}]")
        if vector.size != n:
            raise ValueError(f"grid[{name!r}] has length {vector.size}, expected {n}")
        X[row] = vector
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("grid and y must be finite; drop or impute missing values first")

    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1)
    # A constant descriptor has no correlation with anything; r = 0 lets it stay in the
    # grid (it is a swept configuration like any other) and take p_adjusted = 1.
    constant = norms == 0
    Xn = Xc / np.where(constant, 1.0, norms)[:, None]

    yc = y - y.mean()
    y_norm = np.linalg.norm(yc)
    if y_norm == 0:
        raise ValueError("y is constant; no correlation is defined")
    yn = yc / y_norm

    r_obs = Xn @ yn
    r_obs[constant] = 0.0
    abs_obs = np.abs(r_obs)

    rng = np.random.default_rng(seed)
    ge_max = np.zeros(len(names), dtype=np.int64)
    ge_own = np.zeros(len(names), dtype=np.int64)
    block = max(1, 2_000_000 // max(n, len(names)))
    done = 0
    while done < n_perm:
        size = min(block, n_perm - done)
        permuted = rng.permuted(np.broadcast_to(yn, (size, n)), axis=1)
        r_null = np.abs(permuted @ Xn.T)  # (size, n_configs)
        ge_max += (r_null.max(axis=1)[:, None] >= abs_obs[None, :]).sum(axis=0)
        ge_own += (r_null >= abs_obs[None, :]).sum(axis=0)
        done += size

    return pd.DataFrame(
        {
            "config": names,
            "statistic": r_obs,
            "abs_statistic": abs_obs,
            "p_raw": (ge_own + 1) / (n_perm + 1),
            "p_adjusted": (ge_max + 1) / (n_perm + 1),
            "n": n,
            "n_perm": n_perm,
            "seed": seed,
        }
    )


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values), in the input order.

    Correct for the pre-specified descriptor and axis families of methods §3.9, and wrong
    for anything swept — see the module docstring and :func:`maxT_permutation`. The
    step-up enforces monotonicity from the largest p downwards, so a q-value is never
    smaller than that of a more significant test.
    """
    p = _as_1d(p_values, "p_values")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie in [0, 1]")

    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m, dtype=np.float64)
    q[order] = np.minimum(q_sorted, 1.0)
    return q
