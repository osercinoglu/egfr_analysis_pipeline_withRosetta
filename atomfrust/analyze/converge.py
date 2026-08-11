"""Convergence of the frustration index in the decoy count N (R29, methods experiment D1).

**This is subsampling over one stored ensemble, not eight runs.** Decoy *i* is seeded
``base_seed + i`` — ``frustration.py:441`` in the parallel path and ``frustration.py:666``
in the sequential one — and §4 of the plan makes that verbatim seeding part of the
run-directory contract. The first N rows of a 2000-decoy run therefore *are* the ensemble
an N-decoy run would have produced, exactly and not approximately: same seeds, same
mutations, same repacks, same energies. An eight-point sweep over
``N ∈ {10 … 2000}`` costs one generation run, and the N = 50 prefix of a stored 1000-decoy
run reproduces a separately executed 50-decoy run bit for bit.

The prefix must be a genuine prefix in ``decoy_id`` order. ``decoys`` here is expected in
ascending ``decoy_id``; a shuffled ensemble still yields a valid convergence curve but no
longer corresponds to any executable run, which is the whole point of the exercise.

**The caveat, because it is easy to invert.** ``CLAUDE.md`` warns that mixing ``n_decoys``
across *structures* makes them non-comparable: the index divides by the decoy σ, so a
different N shifts the F scale and a correlation computed over structures at mixed N mixes
scales. That warning is about comparing *different structures at different N*. Subsampling
compares *one structure against itself* at several N, where the shift is exactly the effect
being measured — the curve asks how far the N-decoy scale has settled toward the reference
scale. It is the intended use and is unaffected. Nothing here licenses pooling structures
run at different N; the sweep tells you which single N to run them all at.

Two quantities are reported per grid point, and they answer different questions:

``spearman_rho``
    Rank agreement of the per-pair index at N against the reference N. Rank-based, so it is
    insensitive to the very scale shift described above — it asks whether the *ordering* of
    contacts has converged. N\\* is the smallest N reaching ρ ≥ 0.95 (:func:`n_star`).

``sigma_relative_error``
    ``1/√(2(N−1))``, the relative standard error of a sample SD (methods parameter #27).
    Ranks can agree while every value still carries ~10% noise from σ alone at N = 50,
    falling to ~4.5% at N = 250 and ~2.2% at N = 1000. Reporting it next to ρ keeps the
    statistical cost of a small N visible; ρ alone would hide it.

Pure NumPy/SciPy/pandas: no PyRosetta, no I/O, and all randomness from a single
``np.random.default_rng(seed)``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

__all__ = [
    "DEFAULT_GRID",
    "convergence_curve",
    "n_star",
    "sigma_standard_error",
]

DEFAULT_GRID: tuple[int, ...] = (10, 25, 50, 100, 250, 500, 1000, 2000)
"""The eight-point sweep of methods experiment D1."""

_CI_PERCENTILES = (2.5, 97.5)
_BOOT_CHUNK = 256  # draws per vectorised batch; fixes the RNG stream, so keep it constant


def sigma_standard_error(n: int) -> float:
    """``1/√(2(n−1))`` — relative standard error of a sample SD from ``n`` observations.

    The index divides by the decoy σ, so this is the noise floor the decoy count alone
    imposes on every value: 10.1% at N = 50, 4.5% at N = 250, 2.2% at N = 1000
    (methods parameter #27). The large-``n`` normal approximation; exact enough well below
    the N where anyone stops adding decoys.
    """
    if n < 2:
        raise ValueError(f"a sample SD needs at least 2 observations, got n={n}")
    return 1.0 / np.sqrt(2.0 * (n - 1))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ as Pearson r of midranks. NaN when either vector is constant."""
    return _row_correlation(rankdata(x)[None, :], rankdata(y)[None, :])[0]


def _row_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between corresponding rows of two ``(B, n)`` matrices."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (a * b).sum(axis=1) / denominator, np.nan)


def _bootstrap_rho_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile CI for ρ, resampling **pairs** with replacement.

    Pairs are the unit of resampling because the curve is a statement about the per-pair
    index vector: the uncertainty asked about is "would another pocket of this size rank
    the same way", not "would another decoy draw". Resampling decoys instead would destroy
    the prefix property the whole module rests on.

    Draws are batched so ranking is vectorised, which makes the batch size part of the RNG
    stream — it is a module constant for that reason.
    """
    if n_boot <= 0:
        return (np.nan, np.nan)
    n = x.size
    rhos = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        size = min(_BOOT_CHUNK, n_boot - done)
        idx = rng.integers(0, n, size=(size, n))
        rhos[done : done + size] = _row_correlation(
            rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
        )
        done += size
    finite = rhos[np.isfinite(rhos)]
    if finite.size == 0:
        return (np.nan, np.nan)
    low, high = np.percentile(finite, _CI_PERCENTILES)
    return (float(low), float(high))


def convergence_curve(
    native: np.ndarray,
    decoys: np.ndarray,
    grid: Sequence[int] = DEFAULT_GRID,
    index: str = "zscore",
    reference_n: int | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Rank agreement of the index at each N against the reference N.

    Parameters
    ----------
    native
        ``(n_pairs,)`` native energies.
    decoys
        ``(n_decoys, n_pairs)`` decoy energies **in ascending decoy_id order** — row N is
        the decoy seeded ``base_seed + N``, so ``decoys[:N]`` is the N-decoy run.
    grid
        Candidate decoy counts. Points exceeding ``len(decoys)`` are skipped rather than
        raising, and listed in ``result.attrs["skipped_grid_points"]``; a stored run is
        routinely shorter than the full sweep and that is not an error.
    reference_n
        The N every other N is compared against. Defaults to the largest available grid
        point, i.e. the whole stored ensemble rounded down to the sweep. An explicit value
        is always evaluated, even if it is not in ``grid``.

    Returns
    -------
    One row per evaluated grid point, ascending in ``n_decoys``, with columns
    ``n_decoys, spearman_rho, rho_ci_low, rho_ci_high, sigma_relative_error, n_pairs,
    is_reference``. The reference row has ρ = 1.0 by construction. ``attrs`` carries
    ``index``, ``reference_n``, ``n_decoys_available``, ``n_boot``, ``seed`` and
    ``skipped_grid_points``.

    ρ is NaN where the index is constant across pairs at either N — an undefined rank
    correlation, reported rather than silently zeroed, and :func:`n_star` treats it as
    not-reached.
    """
    from atomfrust.analyze.zscore import compute_index

    native = np.asarray(native, dtype=np.float64)
    decoys = np.asarray(decoys, dtype=np.float64)
    if decoys.ndim != 2:
        raise ValueError(f"decoys must be 2-D (n_decoys, n_pairs), got shape {decoys.shape}")
    if decoys.shape[1] != native.shape[0]:
        raise ValueError(
            f"native has {native.shape[0]} pairs but decoys has {decoys.shape[1]}"
        )

    n_available = decoys.shape[0]
    usable = sorted({int(n) for n in grid if 2 <= int(n) <= n_available})
    skipped = sorted({int(n) for n in grid if int(n) > n_available})

    if reference_n is None:
        if not usable:
            raise ValueError(
                f"no grid point fits {n_available} stored decoys; grid={tuple(grid)}"
            )
        reference_n = max(usable)
    reference_n = int(reference_n)
    if reference_n > n_available:
        raise ValueError(
            f"reference_n={reference_n} exceeds the {n_available} stored decoys"
        )
    if reference_n < 2:
        raise ValueError(f"reference_n={reference_n} is too small for a sample SD")

    points = sorted(set(usable) | {reference_n})
    # One index vector per point; the reference is one of them, never recomputed.
    values = {n: compute_index(native, decoys[:n], index=index) for n in points}
    reference_values = values[reference_n]

    rows = []
    for n in points:
        rng = np.random.default_rng(seed)  # same resamples at every N, so CIs are comparable
        rho = _spearman(values[n], reference_values)
        ci_low, ci_high = _bootstrap_rho_ci(values[n], reference_values, n_boot, rng)
        rows.append(
            {
                "n_decoys": n,
                "spearman_rho": rho,
                "rho_ci_low": ci_low,
                "rho_ci_high": ci_high,
                "sigma_relative_error": sigma_standard_error(n),
                "n_pairs": int(native.size),
                "is_reference": n == reference_n,
            }
        )

    curve = pd.DataFrame(rows)
    curve.attrs.update(
        {
            "index": index,
            "reference_n": reference_n,
            "n_decoys_available": n_available,
            "n_boot": n_boot,
            "seed": seed,
            "skipped_grid_points": skipped,
        }
    )
    return curve


def n_star(curve: pd.DataFrame, threshold: float = 0.95) -> int | None:
    """Smallest ``n_decoys`` whose ρ against the reference reaches ``threshold``.

    ``None`` when no evaluated N reaches it — a real answer on a short run ("this ensemble
    does not show convergence"), not a failure. NaN ρ counts as not-reached. The reference
    row itself has ρ = 1.0, so a returned N\\* equal to ``reference_n`` means only the
    trivial self-comparison passed.
    """
    reached = curve.loc[curve["spearman_rho"] >= threshold, "n_decoys"]
    return int(reached.min()) if len(reached) else None
