"""D1 tests for atomfrust.analyze.zscore — indices and normality diagnostics.

Every assertion below is against a closed form worked out by hand, or against numbers the
prototype already wrote to disk. Self-consistency between two implementations of the same
formula would prove nothing about whether the formula is the published one.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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

pytestmark = pytest.mark.unit

PARQUET = Path("results/5GMP_F62_frustration.parquet")


# ------------------------------------------------------------------ zscore


def test_zscore_matches_a_hand_computation():
    # decoys [-2,-1,0,1,2]: mean 0, sample variance 10/4 = 2.5, so sigma = sqrt(2.5).
    # native -5 -> (0 - (-5))/sqrt(2.5) = 5/sqrt(2.5) = sqrt(10).
    decoys = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    native = np.array([-5.0])
    assert zscore(native, decoys) == pytest.approx([sqrt(10.0)], abs=1e-12)


def test_zscore_sign_convention_favourable_native_is_positive():
    decoys = np.array([[0.0, 0.0], [2.0, 2.0]])
    native = np.array([-3.0, 5.0])  # more favourable, then less favourable
    result = zscore(native, decoys)
    assert result[0] > 0
    assert result[1] < 0


def test_zscore_is_zero_when_the_decoy_spread_is_degenerate():
    # A constant decoy column has no scale; the prototype's guard (frustration.py:697)
    # returns 0.0 rather than +/-inf, and so must this.
    decoys = np.array([[-4.0, 1.0], [-4.0, 3.0], [-4.0, 5.0]])
    native = np.array([-40.0, 0.0])
    result = zscore(native, decoys)
    assert result[0] == 0.0
    assert np.isfinite(result[1]) and result[1] != 0.0


def test_zscore_uses_ddof_one():
    # decoys [0, 2]: ddof=1 sigma is sqrt(2) = 1.4142, ddof=0 would be 1.0.
    decoys = np.array([[0.0], [2.0]])
    native = np.array([0.0])
    assert zscore(native, decoys) == pytest.approx([1.0 / sqrt(2.0)], abs=1e-12)


def test_zscore_is_vectorised_over_pairs():
    rng = np.random.default_rng(0)
    decoys = rng.normal(size=(40, 250))
    native = rng.normal(size=250)
    result = zscore(native, decoys)
    assert result.shape == (250,)
    for k in (0, 17, 249):
        expected = (decoys[:, k].mean() - native[k]) / decoys[:, k].std(ddof=1)
        assert result[k] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------- rank_percentile


def test_rank_percentile_saturates_at_both_extremes():
    decoys = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    native = np.array([-10.0, 10.0])  # better than all decoys, then worse than all
    assert rank_percentile(native, decoys) == pytest.approx([1.0, -1.0])


def test_rank_percentile_handles_ties_with_half_weight():
    # decoys [-1,-1,0,1] vs native -1: 2 strictly greater, 2 tied
    # -> 2*(2 + 0.5*2)/4 - 1 = 0.5
    decoys = np.array([[-1.0], [-1.0], [0.0], [1.0]])
    native = np.array([-1.0])
    assert rank_percentile(native, decoys) == pytest.approx([0.5])


def test_rank_percentile_is_zero_at_the_ensemble_median_and_for_an_all_tie_set():
    decoys = np.array([[-2.0, 7.0], [-1.0, 7.0], [1.0, 7.0], [2.0, 7.0]])
    native = np.array([0.0, 7.0])
    assert rank_percentile(native, decoys) == pytest.approx([0.0, 0.0])


def test_rank_percentile_is_invariant_to_monotone_rescaling():
    # It depends only on order, so an affine (increasing) transform of every energy leaves
    # it unchanged -- the property a Z-score does not have.
    rng = np.random.default_rng(3)
    decoys = rng.normal(size=(30, 12))
    native = rng.normal(size=12)
    plain = rank_percentile(native, decoys)
    rescaled = rank_percentile(3.5 * native + 11.0, 3.5 * decoys + 11.0)
    assert np.array_equal(plain, rescaled)


# ---------------------------------------------------------------- robust_z


def test_robust_z_matches_a_hand_computation():
    # decoys [0,1,2,3,4,1000]: median 2.5, |x-median| = [2.5,1.5,0.5,0.5,1.5,997.5],
    # MAD = median of that = (1.5+1.5)/2 = 1.5. native -1 -> 3.5/(1.4826*1.5).
    decoys = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [1000.0]])
    native = np.array([-1.0])
    assert robust_z(native, decoys) == pytest.approx([3.5 / (1.4826 * 1.5)], rel=1e-12)


def test_robust_z_resists_an_outlier_that_flattens_the_zscore():
    decoys = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [1000.0]])
    native = np.array([-1.0])
    z = zscore(native, decoys)[0]
    rz = robust_z(native, decoys)[0]
    # The single outlier inflates both mean and sigma; sigma dominates, so z collapses.
    assert z < 0.5
    assert rz > 1.5
    assert rz > 3.0 * z


def test_robust_z_agrees_with_zscore_on_a_clean_symmetric_sample():
    # Both estimate the same quantity when the decoys really are Gaussian.
    rng = np.random.default_rng(11)
    decoys = rng.normal(loc=-5.0, scale=2.0, size=(4000, 1))
    native = np.array([-9.0])
    assert robust_z(native, decoys)[0] == pytest.approx(zscore(native, decoys)[0], rel=0.05)


def test_robust_z_is_zero_when_the_mad_is_degenerate():
    # More than half the decoys share one value: MAD = 0 even though the std is not.
    decoys = np.array([[1.0], [1.0], [1.0], [1.0], [50.0]])
    native = np.array([-100.0])
    assert robust_z(native, decoys) == pytest.approx([0.0])
    assert zscore(native, decoys)[0] > 0.0


# ----------------------------------------------------------- registry / API


def test_index_functions_registry_holds_exactly_the_three_names():
    assert set(INDEX_FUNCTIONS) == {"zscore", "rank_percentile", "robust_z"}


def test_compute_index_dispatches_to_each_registered_function():
    rng = np.random.default_rng(5)
    decoys = rng.normal(size=(25, 8))
    native = rng.normal(size=8)
    for name, function in INDEX_FUNCTIONS.items():
        assert np.array_equal(compute_index(native, decoys, name), function(native, decoys))
    assert np.array_equal(compute_index(native, decoys), zscore(native, decoys))


def test_compute_index_rejects_an_unknown_name():
    decoys = np.zeros((3, 2))
    with pytest.raises(ValueError, match="unknown index"):
        compute_index(np.zeros(2), decoys, "percentile")


def test_compute_all_indices_returns_the_three_columns_unchanged():
    rng = np.random.default_rng(7)
    decoys = rng.normal(size=(60, 15))
    native = rng.normal(size=15)
    table = compute_all_indices(native, decoys)
    assert list(table.columns) == ["zscore", "rank_percentile", "robust_z"]
    assert len(table) == 15
    for name, function in INDEX_FUNCTIONS.items():
        assert np.allclose(table[name].to_numpy(), function(native, decoys))


@pytest.mark.parametrize(
    "native, decoys, message",
    [
        (np.zeros(3), np.zeros((5, 4)), "pair axis must be shared"),
        (np.zeros((3, 1)), np.zeros((5, 3)), "native must be 1-D"),
        (np.zeros(3), np.zeros(3), "decoys must be 2-D"),
        (np.zeros(3), np.zeros((1, 3)), "need >= 2 decoys"),
    ],
)
def test_malformed_input_raises_rather_than_returning_nan(native, decoys, message):
    with pytest.raises(ValueError, match=message):
        zscore(native, decoys)


# -------------------------------------------------------------- summaries


def test_decoy_summary_matches_hand_computed_moments():
    # column 0: [1,2,3,4] -> mean 2.5, ddof=1 std sqrt(5/3), median 2.5,
    #           |x-2.5| = [1.5,0.5,0.5,1.5] -> MAD 1.0
    # column 1: constant 7 -> zero spread throughout
    decoys = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0], [4.0, 7.0]])
    table = decoy_summary(decoys)
    assert list(table.columns) == [
        "decoy_mean",
        "decoy_std",
        "decoy_median",
        "decoy_mad",
        "n_decoys",
    ]
    assert table["decoy_mean"].tolist() == pytest.approx([2.5, 7.0])
    assert table["decoy_std"].tolist() == pytest.approx([sqrt(5.0 / 3.0), 0.0])
    assert table["decoy_median"].tolist() == pytest.approx([2.5, 7.0])
    assert table["decoy_mad"].tolist() == pytest.approx([1.0, 0.0])
    assert table["n_decoys"].tolist() == [4, 4]


def test_normality_diagnostics_matches_hand_computed_shape_statistics():
    # x = [0,0,0,4]: mean 1, m2 = 3, m3 = 6, m4 = 21
    # skew = 6/3^1.5 = 1.15470..., excess kurtosis = 21/9 - 3 = -2/3
    decoys = np.array([[0.0], [0.0], [0.0], [4.0]])
    table = normality_diagnostics(decoys)
    assert list(table.columns) == ["shapiro_W", "shapiro_p", "skew", "excess_kurtosis"]
    assert table["skew"][0] == pytest.approx(6.0 / 3.0**1.5, rel=1e-10)
    assert table["excess_kurtosis"][0] == pytest.approx(-2.0 / 3.0, rel=1e-10)


def test_normality_diagnostics_separates_a_gaussian_sample_from_a_skewed_one():
    rng = np.random.default_rng(19)
    gaussian = rng.normal(size=(200, 1))
    exponential = rng.exponential(size=(200, 1))
    table = normality_diagnostics(np.hstack([gaussian, exponential]))
    assert table["shapiro_p"][0] > 0.05
    assert table["shapiro_p"][1] < 1e-6
    assert abs(table["skew"][0]) < 0.4
    assert table["skew"][1] > 1.0  # exponential skew is 2 in the limit


def test_normality_diagnostics_reports_nan_for_a_degenerate_pair():
    decoys = np.array([[3.0, 0.0], [3.0, 1.0], [3.0, 2.0], [3.0, 9.0]])
    table = normality_diagnostics(decoys)
    assert table.iloc[0].isna().all()
    assert table.iloc[1].notna().all()


# ------------------------------------------------- cross-validation on disk


@pytest.mark.skipif(not PARQUET.exists(), reason="results/ not present (dvc pull)")
def test_zscore_reproduces_the_stored_5gmp_f_index():
    """The strongest available check: the prototype's own numbers, from its own run.

    ``results/5GMP_F62_frustration.parquet`` stores only the moments, not the decoy
    energies, so the ensemble is reconstructed as the unique symmetric two-point set with
    the stored ``decoy_mean`` and ``decoy_std`` (ddof=1). That exercises the real
    :func:`compute_index` path — mean, ddof=1 std, sigma guard and sign — against numbers
    produced by an entirely separate implementation.
    """
    stored = pd.read_parquet(PARQUET)
    native = stored["E_native"].to_numpy(dtype=np.float64)
    mean = stored["decoy_mean"].to_numpy(dtype=np.float64)
    std = stored["decoy_std"].to_numpy(dtype=np.float64)
    expected = stored["F_index"].to_numpy(dtype=np.float64)

    # {mean + s/sqrt(2), mean - s/sqrt(2)} has mean `mean` and ddof=1 std exactly `s`.
    offset = std / sqrt(2.0)
    decoys = np.vstack([mean + offset, mean - offset])

    result = compute_index(native, decoys, "zscore")
    deviation = np.abs(result - expected).max()
    assert deviation < np.finfo(np.float32).eps, f"max deviation {deviation:.3e}"

    # And the closed form straight off the stored columns, with no reconstruction at all.
    direct = np.where(std < 1e-9, 0.0, (mean - native) / np.where(std < 1e-9, 1.0, std))
    assert np.abs(direct - expected).max() < np.finfo(np.float32).eps
