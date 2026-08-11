"""Frustration indices and the diagnostics that decide which one to trust.

The prototype computed exactly one index, inline, in the survey loop
(``frustration.py:688-701``): a Z-score over the decoy energies, per pair, in Python. Two
defects follow from that.

**The index was not a choice.** A Z-score is a distance in units of the decoy standard
deviation. It orders contacts sensibly whatever the decoy distribution looks like, but the
thresholds applied to it (0.78 / −1.0) are quantiles of an assumed Gaussian null. If the
decoy energies for a pair are skewed or heavy-tailed — one clashing rotamer in the ensemble
is enough — the same numeric cutoff selects a different tail probability, and does so
inconsistently across pairs. :data:`INDEX_FUNCTIONS` therefore offers two alternatives that
do not assume a shape: ``rank_percentile``, which is the empirical tail fraction itself and
is invariant to any monotone transform of the energies, and ``robust_z``, which replaces
mean and standard deviation with median and scaled MAD so that a handful of outliers cannot
inflate the denominator. This is R26.

**The assumption was never checked.** :func:`normality_diagnostics` reports Shapiro–Wilk,
skewness and excess kurtosis per pair, so "the Gaussian index is fine here" becomes a
measurement rather than a hope. This is R27.

Everything here is array arithmetic over decoy energies already on disk — no pose, no
Rosetta, no I/O — so the index is re-specifiable against a finished run, which is the whole
point of storing direct energies rather than derived indices (see §4 of the plan).

**Sign convention.** Rosetta energies are negative-is-favourable, and every index below is
written as *decoy minus native*. A native contact more favourable than its decoys therefore
yields a **positive** index, and "minimally frustrated" is the positive tail. This matches
``frustration.py:700`` and is why the classification thresholds are ``> 0.78`` and
``< −1.0`` rather than the reverse.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "INDEX_FUNCTIONS",
    "zscore",
    "rank_percentile",
    "robust_z",
    "compute_index",
    "compute_all_indices",
    "normality_diagnostics",
    "decoy_summary",
]

#: Scale below which a spread estimate is treated as exactly zero. Matches the prototype's
#: ``if sigma < 1e-9`` guard (``frustration.py:697``); a degenerate decoy distribution has
#: no scale to measure against, so the index is 0.0 rather than ±inf.
_EPS = 1e-9

#: Consistency factor making the MAD an unbiased estimator of σ for Gaussian data, so
#: ``robust_z`` is on the same scale as ``zscore`` when the decoys really are Gaussian.
_MAD_TO_SIGMA = 1.4826


def _as_pair_arrays(native: np.ndarray, decoys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and upcast. ``native`` is ``(n_pairs,)``, ``decoys`` is ``(n_decoys, n_pairs)``."""
    native = np.asarray(native, dtype=np.float64)
    decoys = np.asarray(decoys, dtype=np.float64)
    if native.ndim != 1:
        raise ValueError(f"native must be 1-D (n_pairs,), got shape {native.shape}")
    if decoys.ndim != 2:
        raise ValueError(f"decoys must be 2-D (n_decoys, n_pairs), got shape {decoys.shape}")
    if decoys.shape[1] != native.shape[0]:
        raise ValueError(
            f"decoys has {decoys.shape[1]} pairs but native has {native.shape[0]}; "
            "the pair axis must be shared and identically ordered (join on pair_id first)"
        )
    if decoys.shape[0] < 2:
        raise ValueError(
            f"need >= 2 decoys to estimate a spread, got {decoys.shape[0]}; the prototype "
            "silently dropped such pairs (frustration.py:693-694), which is worse"
        )
    return native, decoys


def _safe_ratio(numerator: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``numerator / scale``, with 0.0 wherever the scale is degenerate."""
    out = np.zeros_like(numerator)
    ok = scale >= _EPS
    out[ok] = numerator[ok] / scale[ok]
    return out


def zscore(native: np.ndarray, decoys: np.ndarray) -> np.ndarray:
    """``(mean(decoys) − native) / std(decoys, ddof=1)``; 0.0 where σ < 1e-9.

    The published index. ``ddof=1`` because the decoys are a sample drawn from the decoy
    ensemble, not the ensemble itself.
    """
    native, decoys = _as_pair_arrays(native, decoys)
    return _safe_ratio(decoys.mean(axis=0) - native, decoys.std(axis=0, ddof=1))


def rank_percentile(native: np.ndarray, decoys: np.ndarray) -> np.ndarray:
    """``2·((#decoys with E > native) + ½·ties)/N − 1``, in [−1, +1].

    The empirical tail fraction, rescaled so the sign convention and the direction of
    "favourable" match :func:`zscore`: +1 when the native is more favourable than every
    decoy, −1 when it is worse than every one, 0 at the median. Half-weighting ties keeps
    the index antisymmetric, so a decoy set identical to the native gives exactly 0 instead
    of ±1.

    Distribution-free: it depends only on the ordering of the energies, so no assumption
    about the decoy distribution can be violated. The price is granularity — with N decoys
    it takes only 2N+1 distinct values, and it saturates at ±1 for any native outside the
    ensemble's range, where a Z-score still discriminates.
    """
    native, decoys = _as_pair_arrays(native, decoys)
    n = decoys.shape[0]
    greater = (decoys > native).sum(axis=0)
    ties = (decoys == native).sum(axis=0)
    return 2.0 * (greater + 0.5 * ties) / n - 1.0


def robust_z(native: np.ndarray, decoys: np.ndarray) -> np.ndarray:
    """``(median(decoys) − native) / (1.4826·MAD)``; 0.0 where the MAD < 1e-9.

    Same units as :func:`zscore` under Gaussian decoys, but with a breakdown point of 50%
    in both location and scale. A decoy ensemble is generated by mutation and repacking, so
    a small number of members can land in a clashing rotamer and carry an energy orders of
    magnitude off the bulk; those inflate the mean and, worse, the standard deviation,
    driving the Z-score toward zero for exactly the contacts most affected.

    The MAD is degenerate whenever more than half the decoys share one energy value even if
    the rest do not — a stricter guard than the standard deviation's, and deliberately so.
    """
    native, decoys = _as_pair_arrays(native, decoys)
    median = np.median(decoys, axis=0)
    mad = np.median(np.abs(decoys - median), axis=0)
    return _safe_ratio(median - native, _MAD_TO_SIGMA * mad)


#: The registry the CLI and the analysis layer select from by name.
INDEX_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "zscore": zscore,
    "rank_percentile": rank_percentile,
    "robust_z": robust_z,
}


def compute_index(
    native: np.ndarray, decoys: np.ndarray, index: str = "zscore"
) -> np.ndarray:
    """One index by name. ``native`` ``(n_pairs,)``, ``decoys`` ``(n_decoys, n_pairs)`` → ``(n_pairs,)``."""
    try:
        function = INDEX_FUNCTIONS[index]
    except KeyError:
        raise ValueError(
            f"unknown index {index!r}; available: {sorted(INDEX_FUNCTIONS)}"
        ) from None
    return function(native, decoys)


def compute_all_indices(native: np.ndarray, decoys: np.ndarray) -> pd.DataFrame:
    """All three indices from one pass over the same decoy vector.

    Computing them together is not an optimisation, it is a correctness property: the three
    columns are then guaranteed to describe the same ensemble, so disagreement between them
    is evidence about the decoy distribution rather than about which subset of decoys each
    index happened to see.
    """
    native, decoys = _as_pair_arrays(native, decoys)
    return pd.DataFrame(
        {name: function(native, decoys) for name, function in INDEX_FUNCTIONS.items()}
    )


def decoy_summary(decoys: np.ndarray) -> pd.DataFrame:
    """Location and scale per pair: ``decoy_mean, decoy_std, decoy_median, decoy_mad, n_decoys``.

    ``decoy_std`` is ``ddof=1``. ``decoy_mad`` is the **raw** median absolute deviation,
    unscaled — :func:`robust_z` applies the 1.4826 factor itself, and storing the raw value
    keeps the column a plain descriptive statistic rather than a Gaussian-conditioned one.
    """
    decoys = np.asarray(decoys, dtype=np.float64)
    if decoys.ndim != 2:
        raise ValueError(f"decoys must be 2-D (n_decoys, n_pairs), got shape {decoys.shape}")
    n_decoys, n_pairs = decoys.shape
    median = np.median(decoys, axis=0)
    return pd.DataFrame(
        {
            "decoy_mean": decoys.mean(axis=0),
            "decoy_std": decoys.std(axis=0, ddof=1) if n_decoys > 1 else np.full(n_pairs, np.nan),
            "decoy_median": median,
            "decoy_mad": np.median(np.abs(decoys - median), axis=0),
            "n_decoys": np.full(n_pairs, n_decoys, dtype=np.int32),
        }
    )


def normality_diagnostics(decoys: np.ndarray) -> pd.DataFrame:
    """Per-pair ``shapiro_W, shapiro_p, skew, excess_kurtosis``.

    These are the evidence for or against reading a Z-score as a tail probability. A small
    ``shapiro_p`` says the Gaussian null the 0.78 / −1.0 thresholds were quantiles of does
    not hold for that contact; the sign of ``skew`` says in which direction, and
    ``excess_kurtosis`` says whether the tail is heavy enough for the mean and σ to be the
    wrong summaries. Report them, do not filter on them — with thousands of pairs, some
    ``p`` values are small by construction.

    Shapiro–Wilk is the one place a Python loop over pairs is unavoidable: SciPy's
    implementation takes a single sample. Degenerate (zero-variance) pairs get NaN rather
    than SciPy's warning-and-nan, and pairs with fewer than 3 decoys get NaN because the
    statistic is undefined there. Above ~5000 decoys SciPy's ``p`` is an approximation.
    """
    decoys = np.asarray(decoys, dtype=np.float64)
    if decoys.ndim != 2:
        raise ValueError(f"decoys must be 2-D (n_decoys, n_pairs), got shape {decoys.shape}")
    n_decoys, n_pairs = decoys.shape

    # A constant column has no shape to test; SciPy would return nan after a catastrophic-
    # cancellation warning, so exclude it up front and write the nan ourselves.
    degenerate = decoys.std(axis=0) < _EPS

    skew = np.full(n_pairs, np.nan)
    excess_kurtosis = np.full(n_pairs, np.nan)
    live = ~degenerate
    if live.any():
        skew[live] = stats.skew(decoys[:, live], axis=0)
        excess_kurtosis[live] = stats.kurtosis(decoys[:, live], axis=0, fisher=True)

    shapiro_w = np.full(n_pairs, np.nan)
    shapiro_p = np.full(n_pairs, np.nan)
    if n_decoys >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # large-N p-value approximation notice
            for k in np.flatnonzero(live):
                result = stats.shapiro(decoys[:, k])
                shapiro_w[k] = result.statistic
                shapiro_p[k] = result.pvalue

    return pd.DataFrame(
        {
            "shapiro_W": shapiro_w,
            "shapiro_p": shapiro_p,
            "skew": skew,
            "excess_kurtosis": excess_kurtosis,
        }
    )
