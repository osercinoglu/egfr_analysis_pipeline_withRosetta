"""D4 tests for atomfrust.analyze.converge — the N-as-a-prefix convergence sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atomfrust.analyze.converge import (
    DEFAULT_GRID,
    convergence_curve,
    n_star,
    sigma_standard_error,
)

# --------------------------------------------------------------------- fallback

# converge.py imports compute_index lazily so it loads without D1 in place. These tests
# need *something* to call, so if analyze/zscore.py has not landed yet, register a minimal
# stand-in under its import name. It reproduces frustration.py:688-701 exactly — mean minus
# native over the sample SD, sigma < 1e-9 collapsing to 0 — which is all this module
# depends on. The fallback lives here and never in the module under test.
try:  # pragma: no cover - depends on whether D1 has landed
    from atomfrust.analyze.zscore import compute_index  # noqa: F401

    USING_FALLBACK = False
except ImportError:  # pragma: no cover
    import sys
    import types

    def compute_index(native, decoys, index: str = "zscore"):
        if index != "zscore":
            raise ValueError(f"fallback only implements zscore, got {index!r}")
        native = np.asarray(native, dtype=np.float64)
        decoys = np.asarray(decoys, dtype=np.float64)
        mu = decoys.mean(axis=0)
        sigma = decoys.std(axis=0, ddof=1)
        return np.where(sigma < 1e-9, 0.0, (mu - native) / np.where(sigma < 1e-9, 1.0, sigma))

    _stub = types.ModuleType("atomfrust.analyze.zscore")
    _stub.compute_index = compute_index
    sys.modules["atomfrust.analyze.zscore"] = _stub
    USING_FALLBACK = True


# ---------------------------------------------------------------------- fixtures


def noisy_ensemble(n_pairs: int = 200, n_decoys: int = 250, seed: int = 7):
    """Per-pair signal buried in decoy noise: the index converges as N grows."""
    rng = np.random.default_rng(seed)
    native = np.zeros(n_pairs)
    mu = rng.normal(0.0, 2.0, size=n_pairs)
    decoys = mu[None, :] + rng.normal(0.0, 3.0, size=(n_decoys, n_pairs))
    return native, decoys


def rank_stable_ensemble(n_pairs: int = 60, n_decoys: int = 300, seed: int = 3):
    """An ensemble whose index ordering is *exactly* N-independent, by construction.

    Every decoy shifts all pairs by the same amount, so ``mean`` picks up a pair-independent
    offset and ``std`` is a pair-independent positive scale. The index is then an affine,
    order-preserving map of ``mu - native`` at every N, so rho against any reference is
    exactly 1.0 and N* is the smallest grid point. Known answer, no tolerance needed.
    """
    rng = np.random.default_rng(seed)
    native = rng.normal(0.0, 1.0, size=n_pairs)
    mu = native + rng.normal(0.0, 1.0, size=n_pairs)
    shifts = rng.normal(0.0, 1.0, size=n_decoys)
    return native, mu[None, :] + shifts[:, None]


# ------------------------------------------------------------- sigma_standard_error


@pytest.mark.unit
@pytest.mark.parametrize(
    "n, published", [(50, 0.101), (250, 0.045), (1000, 0.022)]
)
def test_sigma_standard_error_matches_the_published_values(n, published):
    """Methods parameter #27: 10.1% at N=50, 4.5% at N=250, 2.2% at N=1000, to 2 s.f."""
    assert sigma_standard_error(n) == pytest.approx(published, abs=0.0005)


@pytest.mark.unit
def test_sigma_standard_error_is_the_closed_form_and_decreases_with_n():
    assert sigma_standard_error(10) == pytest.approx(1.0 / np.sqrt(18.0))
    values = [sigma_standard_error(n) for n in DEFAULT_GRID]
    assert all(b < a for a, b in zip(values, values[1:]))


@pytest.mark.unit
def test_sigma_standard_error_rejects_a_sample_too_small_for_an_sd():
    with pytest.raises(ValueError, match="at least 2"):
        sigma_standard_error(1)


# ------------------------------------------------------------------ curve shape


@pytest.mark.unit
def test_curve_has_the_contracted_columns_and_one_row_per_usable_grid_point():
    native, decoys = noisy_ensemble()
    curve = convergence_curve(native, decoys, grid=(10, 25, 50, 100, 250), n_boot=25)

    assert list(curve.columns) == [
        "n_decoys",
        "spearman_rho",
        "rho_ci_low",
        "rho_ci_high",
        "sigma_relative_error",
        "n_pairs",
        "is_reference",
    ]
    assert list(curve.n_decoys) == [10, 25, 50, 100, 250]
    assert (curve.n_pairs == native.size).all()
    assert curve.sigma_relative_error.iloc[2] == pytest.approx(0.101, abs=0.0005)


@pytest.mark.unit
def test_grid_points_beyond_the_stored_ensemble_are_skipped_and_recorded():
    """A stored run is routinely shorter than the full sweep — not an error."""
    native, decoys = noisy_ensemble(n_decoys=120)
    curve = convergence_curve(native, decoys, grid=DEFAULT_GRID, n_boot=25)

    assert list(curve.n_decoys) == [10, 25, 50, 100]
    assert curve.attrs["skipped_grid_points"] == [250, 500, 1000, 2000]
    assert curve.attrs["n_decoys_available"] == 120
    assert curve.attrs["reference_n"] == 100


@pytest.mark.unit
def test_the_reference_defaults_to_the_largest_available_n_and_scores_exactly_one():
    native, decoys = noisy_ensemble()
    curve = convergence_curve(native, decoys, grid=(10, 50, 250), n_boot=25)

    reference = curve[curve.is_reference]
    assert len(reference) == 1
    assert int(reference.n_decoys.iloc[0]) == 250 == curve.attrs["reference_n"]
    assert reference.spearman_rho.iloc[0] == pytest.approx(1.0)
    assert reference.rho_ci_low.iloc[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_an_explicit_reference_is_evaluated_even_when_absent_from_the_grid():
    native, decoys = noisy_ensemble()
    curve = convergence_curve(
        native, decoys, grid=(10, 50), reference_n=200, n_boot=25
    )
    assert list(curve.n_decoys) == [10, 50, 200]
    assert curve.loc[curve.n_decoys == 200, "spearman_rho"].iloc[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_rho_climbs_toward_the_reference_as_n_grows():
    native, decoys = noisy_ensemble()
    curve = convergence_curve(native, decoys, grid=(10, 25, 50, 100, 250), n_boot=25)
    rho = curve.spearman_rho.to_numpy()

    assert rho[0] < rho[-1]
    assert rho[-1] == pytest.approx(1.0)
    # Monotone-ish, not monotone: sampling noise can invert adjacent points, but the trend
    # must be unambiguous.
    assert np.corrcoef(np.arange(len(rho)), rho)[0, 1] > 0.8
    assert (np.diff(rho) > -0.05).all()


# ------------------------------------------------------------- N as a prefix


@pytest.mark.unit
def test_a_prefix_of_a_long_run_equals_a_short_run():
    """The claim the whole module rests on: subsampling to N is not an approximation of an
    N-decoy run, it is that run. Truncating the ensemble to 100 decoys must give exactly the
    curve obtained by referencing N=100 inside the 250-decoy ensemble."""
    native, decoys = noisy_ensemble(n_decoys=250)
    grid = (10, 25, 50, 100)

    from_long = convergence_curve(native, decoys, grid=grid, reference_n=100, n_boot=25)
    from_short = convergence_curve(native, decoys[:100], grid=grid, n_boot=25)

    pd.testing.assert_frame_equal(from_long, from_short)


# ------------------------------------------------------------------- bootstrap


@pytest.mark.unit
def test_bootstrap_ci_brackets_the_point_estimate_and_is_seed_reproducible():
    native, decoys = noisy_ensemble()
    first = convergence_curve(native, decoys, grid=(10, 50, 250), n_boot=60, seed=0)
    again = convergence_curve(native, decoys, grid=(10, 50, 250), n_boot=60, seed=0)
    other = convergence_curve(native, decoys, grid=(10, 50, 250), n_boot=60, seed=1)

    pd.testing.assert_frame_equal(first, again)
    assert (first.rho_ci_low <= first.spearman_rho + 1e-9).all()
    assert (first.rho_ci_high >= first.spearman_rho - 1e-9).all()
    assert (first.rho_ci_low <= first.rho_ci_high).all()
    # Only the CI is stochastic; the point estimate is not.
    assert np.allclose(first.spearman_rho, other.spearman_rho)
    assert not np.allclose(first.rho_ci_low, other.rho_ci_low)


@pytest.mark.unit
def test_the_interval_narrows_as_pairs_are_added():
    """Pairs are the resampling unit, so more pairs must mean a tighter interval."""
    small = convergence_curve(*noisy_ensemble(n_pairs=40), grid=(10, 250), n_boot=60)
    large = convergence_curve(*noisy_ensemble(n_pairs=400), grid=(10, 250), n_boot=60)

    width = lambda c: float(c.rho_ci_high.iloc[0] - c.rho_ci_low.iloc[0])  # noqa: E731
    assert width(large) < width(small)


# ---------------------------------------------------------------------- n_star


@pytest.mark.unit
def test_n_star_on_a_rank_stable_ensemble_is_the_smallest_grid_point():
    """Known answer: the ordering is N-independent by construction, so rho is 1.0
    everywhere and the sweep converges at its first point."""
    native, decoys = rank_stable_ensemble()
    curve = convergence_curve(native, decoys, grid=(10, 25, 50, 100, 250), n_boot=25)

    assert np.allclose(curve.spearman_rho, 1.0)
    assert n_star(curve) == 10


@pytest.mark.unit
def test_n_star_picks_the_smallest_n_at_or_above_the_threshold():
    curve = pd.DataFrame(
        {
            "n_decoys": [10, 25, 50, 100],
            "spearman_rho": [0.80, 0.94, 0.96, 1.00],
        }
    )
    assert n_star(curve, threshold=0.95) == 50
    assert n_star(curve, threshold=0.94) == 25
    assert n_star(curve, threshold=0.80) == 10


@pytest.mark.unit
def test_n_star_is_none_when_the_threshold_is_never_reached():
    curve = pd.DataFrame(
        {"n_decoys": [10, 25, 50], "spearman_rho": [0.10, 0.42, 0.71]}
    )
    assert n_star(curve, threshold=0.95) is None


@pytest.mark.unit
def test_n_star_treats_an_undefined_rho_as_not_reached():
    curve = pd.DataFrame({"n_decoys": [10, 25], "spearman_rho": [np.nan, np.nan]})
    assert n_star(curve) is None


@pytest.mark.unit
def test_n_star_on_a_real_noisy_curve_is_a_grid_point_below_the_reference():
    native, decoys = noisy_ensemble()
    curve = convergence_curve(native, decoys, grid=(10, 25, 50, 100, 250), n_boot=25)
    star = n_star(curve, threshold=0.95)

    assert star in set(curve.n_decoys)
    assert star is not None and star <= curve.attrs["reference_n"]


# ------------------------------------------------------------------- bad input


@pytest.mark.unit
def test_a_pair_count_mismatch_is_an_error():
    with pytest.raises(ValueError, match="pairs"):
        convergence_curve(np.zeros(5), np.zeros((10, 6)), grid=(10,), n_boot=5)


@pytest.mark.unit
def test_a_one_dimensional_decoy_array_is_an_error():
    with pytest.raises(ValueError, match="2-D"):
        convergence_curve(np.zeros(5), np.zeros(5), grid=(10,), n_boot=5)


@pytest.mark.unit
def test_a_reference_beyond_the_stored_ensemble_is_an_error():
    native, decoys = noisy_ensemble(n_decoys=50)
    with pytest.raises(ValueError, match="exceeds"):
        convergence_curve(native, decoys, grid=(10,), reference_n=500, n_boot=5)


@pytest.mark.unit
def test_an_ensemble_shorter_than_every_grid_point_is_an_error():
    native, decoys = noisy_ensemble(n_decoys=5)
    with pytest.raises(ValueError, match="no grid point fits"):
        convergence_curve(native, decoys, grid=DEFAULT_GRID, n_boot=5)
