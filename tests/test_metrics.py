"""D6 tests for atomfrust.metrics — closed-form fixtures, not self-consistency.

Every check here pins a metric to something computed independently of this package: the
Mann-Whitney U from SciPy, a hand-worked BH table, an EF ranking whose answer is arithmetic,
or an analytic limit. A test that only proves the code agrees with itself would pass on a
consistently wrong implementation, which is the failure mode that matters for a module whose
whole purpose is to be the single source of every reported number.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from atomfrust.metrics import (
    Estimate,
    adjusted_logauc,
    auroc,
    bedroc,
    benjamini_hochberg,
    bootstrap_ci,
    enrichment_factor,
    maxT_permutation,
    paired_delta,
    pearson,
    spearman,
)
from atomfrust.metrics import inference, screening

pytestmark = pytest.mark.unit


def _screen(n=200, n_active=40, seed=0):
    """A screen with real but imperfect signal: actives shifted up by one sigma."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(n)
    labels[:n_active] = 1.0
    scores = rng.normal(size=n) + labels
    groups = np.array([f"T{i % 5}" for i in range(n)])
    return scores, labels, groups


# ---------------------------------------------------------------------------- AUROC


def test_auroc_equals_mann_whitney_u_normalisation():
    """AUROC is U / (n_pos * n_neg) by definition; SciPy computes U independently."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        labels = rng.integers(0, 2, size=120).astype(float)
        scores = rng.normal(size=120)
        u = stats.mannwhitneyu(
            scores[labels == 1], scores[labels == 0], alternative="two-sided"
        ).statistic
        expected = u / (labels.sum() * (labels.size - labels.sum()))
        assert auroc(scores, labels).value == pytest.approx(expected, abs=1e-12)


def test_auroc_extremes():
    labels = np.r_[np.ones(10), np.zeros(10)]
    perfect = np.r_[np.linspace(1, 2, 10), np.linspace(-2, -1, 10)]
    assert auroc(perfect, labels).value == pytest.approx(1.0)
    assert auroc(-perfect, labels).value == pytest.approx(0.0)
    assert auroc(np.full(20, 3.0), labels).value == pytest.approx(0.5)


def test_auroc_ties_are_mid_ranked():
    """One tied pair between the classes contributes exactly half a concordance."""
    labels = np.array([1.0, 0.0])
    assert auroc(np.array([1.0, 1.0]), labels).value == pytest.approx(0.5)


def test_single_class_labels_raise():
    with pytest.raises(ValueError, match="both classes"):
        auroc(np.arange(5.0), np.ones(5))


# --------------------------------------------------------------------------- BEDROC


def test_bedroc_converges_to_auroc_as_alpha_goes_to_zero():
    """The alpha -> 0 limit of BEDROC is exactly AUROC; 1e-6 should already be within 1e-4."""
    rng = np.random.default_rng(3)
    scores = rng.normal(size=150)  # distinct by construction, so no tie convention enters
    labels = np.zeros(150)
    labels[rng.choice(150, size=30, replace=False)] = 1.0
    assert bedroc(scores, labels, alpha=1e-6).value == pytest.approx(
        auroc(scores, labels).value, abs=1e-4
    )


def test_bedroc_default_alpha_is_chaput():
    assert inspect.signature(bedroc).parameters["alpha"].default == 80.5


def test_bedroc_and_ef_match_rdkit():
    """External reference. RDKit is not a dependency, so this skips where it is absent."""
    scoring = pytest.importorskip("rdkit.ML.Scoring.Scoring")
    rng = np.random.default_rng(0)
    scores = rng.normal(size=300)
    labels = np.zeros(300)
    labels[rng.choice(300, size=50, replace=False)] = 1.0
    ranked = [[int(labels[i])] for i in np.argsort(-scores)]

    assert bedroc(scores, labels).value == pytest.approx(
        scoring.CalcBEDROC(ranked, 0, 80.5), abs=1e-12
    )
    assert enrichment_factor(scores, labels).value == pytest.approx(
        scoring.CalcEnrichment(ranked, 0, [0.01])[0], abs=1e-12
    )


def test_bedroc_rewards_early_recognition_over_auroc():
    """Actives packed at the very top: BEDROC(80.5) near 1 while AUROC is unremarkable."""
    n, n_active = 1000, 100
    labels = np.zeros(n)
    labels[:10] = 1.0  # 10 actives at the top
    labels[500:590] = 1.0  # the other 90 buried mid-list
    scores = np.linspace(1.0, 0.0, n)
    assert labels.sum() == n_active
    assert bedroc(scores, labels).value > auroc(scores, labels).value


# ------------------------------------------------------------------- adjusted logAUC


def test_adjusted_logauc_is_zero_for_a_random_ranking():
    """A ranking whose ROC is the diagonal must land on the subtracted baseline."""
    n = 2000
    labels = np.tile([1.0, 0.0], n // 2)
    scores = np.linspace(1.0, 0.0, n)  # actives and inactives perfectly interleaved
    assert adjusted_logauc(scores, labels).value == pytest.approx(0.0, abs=2e-3)


def test_adjusted_logauc_is_positive_for_early_enrichment():
    n = 2000
    labels = np.zeros(n)
    labels[:20] = 1.0
    scores = np.linspace(1.0, 0.0, n)
    assert adjusted_logauc(scores, labels).value > 0.5


# ------------------------------------------------------------------------------- EF


def test_ef1_percent_on_a_hand_constructed_ranking():
    """100 molecules, 10 actives, the single top-1% slot is an active: EF = 1/(10/100) = 10."""
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[0] = 1.0  # rank 1
    labels[[1, 2, 3, 4, 50, 60, 70, 80, 90]] = 1.0
    assert labels.sum() == 10
    assert enrichment_factor(scores, labels, fraction=0.01).value == pytest.approx(10.0)
    # Top 10% holds 5 of the 10 actives: (5/10) / (10/100) = 5.
    assert enrichment_factor(scores, labels, fraction=0.10).value == pytest.approx(5.0)
    # The whole library can only ever be enriched 1-fold.
    assert enrichment_factor(scores, labels, fraction=1.0).value == pytest.approx(1.0)


def test_ef_breaks_ties_against_the_method():
    """An all-tied ranking must not score above random by accident of input order."""
    labels = np.r_[np.ones(10), np.zeros(90)]
    assert enrichment_factor(np.zeros(100), labels, fraction=0.01).value == pytest.approx(0.0)


# ---------------------------------------------------------- grouped bootstrap (R33)


@pytest.mark.parametrize("metric", [auroc, bedroc, adjusted_logauc, enrichment_factor])
def test_screening_metric_refuses_a_pooled_molecule_level_bootstrap(metric):
    scores, labels, _ = _screen()
    with pytest.raises(ValueError, match="groups is required"):
        metric(scores, labels, n_boot=50)


@pytest.mark.parametrize("metric", [auroc, bedroc, adjusted_logauc, enrichment_factor])
def test_screening_metric_with_groups_reports_an_interval(metric):
    scores, labels, groups = _screen()
    est = metric(scores, labels, groups=groups, n_boot=200, seed=1)
    assert est.ci_low is not None and est.ci_high is not None
    assert est.ci_low <= est.value <= est.ci_high
    assert est.n_groups == 5
    assert est.seed == 1


def test_cluster_bootstrap_resamples_targets_then_molecules():
    """The drawn indices must be a union of whole targets, redrawn within each."""
    codes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    rng = np.random.default_rng(0)
    resamples = inference._resample_indices(rng, codes.size, codes, 50)
    for idx in resamples:
        assert idx.size == codes.size
        drawn = codes[idx].reshape(3, 3)
        # Three targets drawn, each contributing a block of three of its own molecules.
        assert (drawn == drawn[:, :1]).all()
    # Targets are drawn with replacement, so a repeated target must occur across 50 draws.
    assert any(len(set(codes[idx].reshape(3, 3)[:, 0])) < 3 for idx in resamples)


def test_cluster_bootstrap_is_wider_than_the_pooled_one():
    """Between-target variance is what clustering restores; ignoring it narrows the CI."""
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(6), 40)
    offsets = rng.normal(scale=1.5, size=6)  # targets differ systematically
    values = offsets[groups] + rng.normal(scale=0.2, size=groups.size)
    rows = values[:, None]

    pooled = bootstrap_ci(rows, lambda r: float(r.mean()), n_boot=400, seed=0)
    clustered = bootstrap_ci(rows, lambda r: float(r.mean()), groups=groups, n_boot=400, seed=0)
    assert (clustered.ci_high - clustered.ci_low) > 3 * (pooled.ci_high - pooled.ci_low)
    assert clustered.method == "cluster_bootstrap_percentile"


def test_bootstrap_is_reproducible_from_the_seed_alone():
    scores, labels, groups = _screen()
    a = auroc(scores, labels, groups=groups, n_boot=100, seed=5)
    b = auroc(scores, labels, groups=groups, n_boot=100, seed=5)
    c = auroc(scores, labels, groups=groups, n_boot=100, seed=6)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)


# ------------------------------------------------------------------------ inference


def test_pearson_and_spearman_match_scipy():
    rng = np.random.default_rng(2)
    x = rng.normal(size=40)
    y = 0.6 * x + rng.normal(size=40)
    assert pearson(x, y).value == pytest.approx(stats.pearsonr(x, y).statistic)
    assert pearson(x, y).p_value == pytest.approx(stats.pearsonr(x, y).pvalue)
    assert spearman(x, y).value == pytest.approx(stats.spearmanr(x, y).statistic)


def test_paired_delta_collapses_to_one_difference_per_target():
    """A constant +2 offset in every target, however many molecules each contributes."""
    groups = np.array(["A"] * 50 + ["B"] * 3 + ["C"] * 7)
    b = np.arange(60, dtype=float)
    a = b + 2.0
    est = paired_delta(a, b, groups, n_boot=500, seed=0)
    assert est.value == pytest.approx(2.0)
    assert est.n_groups == 3
    assert est.n == 60
    assert est.ci_low == pytest.approx(2.0) and est.ci_high == pytest.approx(2.0)
    # The sign-flip null over 3 equal differences puts mass 2/2^3 on |mean| >= |observed|,
    # so 3 targets cannot reach p < 0.05 however large the effect. That floor is the point
    # of testing at the target level, and it should be visible rather than smoothed away.
    assert est.p_value == pytest.approx(0.25, abs=0.05)


def test_paired_delta_needs_more_than_one_target():
    with pytest.raises(ValueError, match="at least 2 groups"):
        paired_delta([1.0, 2.0], [0.0, 1.0], ["A", "A"])


# --------------------------------------------------------------------- multiplicity


def test_benjamini_hochberg_against_a_worked_example():
    """Hand-computed p*m/i with the monotone step-up applied from the largest p down."""
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205])
    expected = np.array([0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.080, 0.074 * 8 / 7, 0.205])
    assert benjamini_hochberg(p) == pytest.approx(expected)


def test_benjamini_hochberg_preserves_input_order_and_caps_at_one():
    p = np.array([0.9, 0.001, 0.5])
    q = benjamini_hochberg(p)
    assert np.argmin(q) == 1
    assert q.max() <= 1.0


def test_maxT_is_more_conservative_than_the_per_configuration_p():
    rng = np.random.default_rng(4)
    y = rng.normal(size=30)
    grid = {f"cfg{i}": rng.normal(size=30) for i in range(20)}
    grid["cfg_signal"] = y + rng.normal(scale=0.5, size=30)
    table = maxT_permutation(grid, y, n_perm=500, seed=0)

    assert isinstance(table, pd.DataFrame)
    assert list(table["config"]) == list(grid)
    assert (table["p_adjusted"] >= table["p_raw"]).all()
    signal = table.set_index("config").loc["cfg_signal"]
    assert signal["p_adjusted"] < 0.05  # real signal survives the price of the sweep


def test_maxT_controls_the_family_wise_error_rate_on_a_null_grid():
    """FWER, not FDR: over replicates with no signal anywhere, rejections stay near 5%.

    Each replicate sweeps a 12-configuration grid against an unrelated outcome, exactly the
    situation of plan §2.3 where a maximum over a grid is reported. Rejecting means *any*
    configuration reaches p_adjusted < 0.05. The empirical rate is asserted below 0.10:
    the target is 0.05 and the binomial standard error at 200 replicates is 0.015, so this
    is a ~3-sigma bound that still fails loudly if the adjustment is dropped.
    """
    n_replicates, n_config, n_obs, n_perm = 200, 12, 19, 199
    rng = np.random.default_rng(1234)
    rejections = 0
    for _ in range(n_replicates):
        y = rng.normal(size=n_obs)
        grid = {f"cfg{i}": rng.normal(size=n_obs) for i in range(n_config)}
        table = maxT_permutation(grid, y, n_perm=n_perm, seed=int(rng.integers(1 << 30)))
        rejections += bool((table["p_adjusted"] < 0.05).any())

    rate = rejections / n_replicates
    assert rate < 0.10, f"empirical FWER {rate:.3f} exceeds the bound"


def test_maxT_constant_descriptor_is_never_significant():
    rng = np.random.default_rng(0)
    y = rng.normal(size=25)
    table = maxT_permutation({"flat": np.ones(25), "real": y}, y, n_perm=100, seed=0)
    flat = table.set_index("config").loc["flat"]
    assert flat["statistic"] == 0.0
    assert flat["p_adjusted"] == pytest.approx(1.0)


def test_maxT_is_reproducible_from_the_seed():
    rng = np.random.default_rng(9)
    y = rng.normal(size=20)
    grid = {f"cfg{i}": rng.normal(size=20) for i in range(5)}
    first = maxT_permutation(grid, y, n_perm=200, seed=3)
    second = maxT_permutation(grid, y, n_perm=200, seed=3)
    pd.testing.assert_frame_equal(first, second)


# ------------------------------------------------------ the Estimate contract (R36)


def _estimate_returning_calls():
    """Every public function that must return an Estimate, with arguments to call it."""
    scores, labels, groups = _screen(n=60, n_active=20, seed=1)
    x = np.linspace(0, 1, 12)
    y = x + np.array([0.1, -0.1] * 6)
    paired_groups = np.array(list("ABCDEF") * 2)
    return {
        "auroc": lambda: auroc(scores, labels, groups=groups, n_boot=20, seed=0),
        "bedroc": lambda: bedroc(scores, labels, groups=groups, n_boot=20, seed=0),
        "adjusted_logauc": lambda: adjusted_logauc(
            scores, labels, groups=groups, n_boot=20, seed=0
        ),
        "enrichment_factor": lambda: enrichment_factor(
            scores, labels, groups=groups, n_boot=20, seed=0
        ),
        "pearson": lambda: pearson(x, y, n_boot=20, seed=0),
        "spearman": lambda: spearman(x, y, n_boot=20, seed=0),
        "paired_delta": lambda: paired_delta(y, x, paired_groups, n_boot=20, seed=0),
        "bootstrap_ci": lambda: bootstrap_ci(
            x[:, None], lambda r: float(r.mean()), n_boot=20, seed=0
        ),
    }


def test_every_public_function_returns_an_estimate():
    """Asserted over the modules' exports, so a new metric cannot quietly return a float."""
    calls = _estimate_returning_calls()
    table_returning = {"maxT_permutation", "benjamini_hochberg"}
    exported = set(screening.__all__) | (set(inference.__all__) - {"Estimate"})

    assert exported - table_returning == set(calls), "an export is not covered by this test"
    for name, call in calls.items():
        result = call()
        assert isinstance(result, Estimate), f"{name} returned {type(result).__name__}"
        assert result.method, f"{name} left method empty"
        assert result.n > 0


def test_estimate_is_frozen():
    est = auroc(*_screen()[:2])
    with pytest.raises(dataclasses.FrozenInstanceError):
        est.value = 0.0  # type: ignore[misc]
