"""D5 tests for atomfrust.analyze.strata — σ across strata (S2.5) and axis redundancy (S2.6).

Everything here is synthetic: the module is a query over stored tables, so a table built in
the test is exactly the input it sees in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atomfrust.analyze.strata import (
    AXIS_SOURCE_COLUMN,
    POLAR_RESIDUES,
    REDUNDANCY_THRESHOLD,
    SIDE_CHAIN_VOLUME_A3,
    STRATUM_AXES,
    assign_strata,
    axis_redundancy,
    pocket_descriptors,
    sigma_by_stratum,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------- fixtures


def make_nodes(resnames, rel_sasa=None, kinds=None, n_heavy=None) -> pd.DataFrame:
    n = len(resnames)
    return pd.DataFrame(
        {
            "node_id": [f"A:{10 + k}" for k in range(n)],
            "kind": kinds if kinds is not None else ["protein"] * n,
            "resname": resnames,
            "n_heavy": n_heavy if n_heavy is not None else [5] * n,
            "rel_sasa": (
                rel_sasa if rel_sasa is not None else [np.nan] * n
            ),
        }
    )


def make_pairs(nodes: pd.DataFrame, edges) -> pd.DataFrame:
    ids = list(nodes["node_id"])
    return pd.DataFrame(
        {
            "pair_id": np.arange(len(edges), dtype=np.int32),
            "node_i": [ids[i] for i, _ in edges],
            "node_j": [ids[j] for _, j in edges],
        }
    )


def make_descriptors(n_pockets: int, seed: int = 0) -> pd.DataFrame:
    """A cohort of pockets with descriptors spread over each axis."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "pocket_id": [f"P{k:02d}" for k in range(n_pockets)],
            "mean_burial": np.linspace(0.1, 0.9, n_pockets),
            "frac_polar": np.linspace(0.05, 0.95, n_pockets),
            "mean_volume": np.linspace(40.0, 130.0, n_pockets),
            "median_sigma": rng.uniform(0.5, 1.5, n_pockets),
        }
    )


# ------------------------------------------------------------ pocket_descriptors


def test_descriptors_use_rel_sasa_when_present():
    """Burial is 1 - rel_sasa, so a buried pocket scores high, and the source is recorded."""
    nodes = make_nodes(["ALA", "LEU", "SER"], rel_sasa=[0.0, 0.2, 0.4])
    pairs = make_pairs(nodes, [(0, 1), (1, 2)])

    out = pocket_descriptors(nodes, pairs, np.array([True, True, True]))

    assert out["burial_source"] == "rel_sasa"
    assert out["mean_burial"] == pytest.approx(1.0 - (0.0 + 0.2 + 0.4) / 3)
    assert out["n_pocket_residues"] == 3
    assert out["mean_residue_degree"] == pytest.approx((1 + 2 + 1) / 3)


def test_burial_falls_back_to_degree_when_rel_sasa_is_absent_or_nan():
    """The documented fallback path: no usable rel_sasa -> mean incident-pair count."""
    resnames = ["ALA", "LEU", "SER"]
    edges = [(0, 1), (1, 2)]

    all_nan = make_nodes(resnames, rel_sasa=[np.nan] * 3)
    out_nan = pocket_descriptors(all_nan, make_pairs(all_nan, edges), np.ones(3, dtype=bool))

    no_column = make_nodes(resnames).drop(columns=["rel_sasa"])
    out_missing = pocket_descriptors(
        no_column, make_pairs(no_column, edges), np.ones(3, dtype=bool)
    )

    for out in (out_nan, out_missing):
        assert out["burial_source"] == "degree"
        assert out["mean_burial"] == pytest.approx((1 + 2 + 1) / 3)


def test_partially_finite_rel_sasa_still_uses_rel_sasa():
    """One usable value is enough; the fallback exists for the total absence of the signal."""
    nodes = make_nodes(["ALA", "LEU", "SER"], rel_sasa=[np.nan, 0.5, np.nan])
    out = pocket_descriptors(nodes, make_pairs(nodes, [(0, 1)]), np.ones(3, dtype=bool))

    assert out["burial_source"] == "rel_sasa"
    assert out["mean_burial"] == pytest.approx(0.5)


def test_non_residue_nodes_are_pocket_nodes_but_not_residues():
    """A ligand has heavy atoms and a degree, but no polarity or side-chain volume."""
    nodes = make_nodes(
        ["ALA", "LEU", "STI"],
        rel_sasa=[0.1, 0.1, np.nan],
        kinds=["protein", "protein", "ligand"],
        n_heavy=[5, 8, 37],
    )
    pairs = make_pairs(nodes, [(0, 2), (1, 2)])

    out = pocket_descriptors(nodes, pairs, np.ones(3, dtype=bool))

    assert out["n_pocket_nodes"] == 3
    assert out["n_pocket_residues"] == 2
    assert out["pocket_heavy_atoms"] == 50
    assert out["frac_polar"] == pytest.approx(0.0)  # ALA and LEU are hydrophobic
    assert out["mean_volume"] == pytest.approx(
        (SIDE_CHAIN_VOLUME_A3["ALA"] + SIDE_CHAIN_VOLUME_A3["LEU"]) / 2
    )


def test_frac_polar_matches_the_property_table():
    resnames = ["ARG", "GLU", "ALA", "VAL", "SER"]
    nodes = make_nodes(resnames, rel_sasa=[0.3] * 5)
    out = pocket_descriptors(nodes, make_pairs(nodes, [(0, 1)]), np.ones(5, dtype=bool))

    expected = sum(name in POLAR_RESIDUES for name in resnames) / len(resnames)
    assert out["frac_polar"] == pytest.approx(expected)


def test_non_standard_residues_are_excluded_not_guessed():
    nodes = make_nodes(["ALA", "MSE"], rel_sasa=[0.2, 0.2])
    out = pocket_descriptors(nodes, make_pairs(nodes, [(0, 1)]), np.ones(2, dtype=bool))

    assert out["n_pocket_residues"] == 2
    assert out["n_typed_residues"] == 1
    assert out["mean_volume"] == pytest.approx(SIDE_CHAIN_VOLUME_A3["ALA"])


def test_positional_indices_are_accepted_as_a_mask():
    nodes = make_nodes(["ALA", "LEU", "SER"], rel_sasa=[0.0, 0.2, 0.4])
    pairs = make_pairs(nodes, [(0, 1), (1, 2)])

    by_index = pocket_descriptors(nodes, pairs, np.array([0, 2]))
    by_bool = pocket_descriptors(nodes, pairs, np.array([True, False, True]))

    assert by_index == by_bool


def test_empty_pocket_does_not_crash():
    nodes = make_nodes(["ALA", "LEU"], rel_sasa=[0.2, 0.3])
    out = pocket_descriptors(nodes, make_pairs(nodes, [(0, 1)]), np.zeros(2, dtype=bool))

    assert out["n_pocket_residues"] == 0
    assert np.isnan(out["mean_burial"])
    assert np.isnan(out["mean_volume"])


def test_mask_length_is_validated():
    nodes = make_nodes(["ALA", "LEU"], rel_sasa=[0.2, 0.3])
    with pytest.raises(ValueError, match="pocket_mask"):
        pocket_descriptors(nodes, make_pairs(nodes, [(0, 1)]), np.ones(5, dtype=bool))


# ---------------------------------------------------------------- assign_strata


def test_assign_strata_adds_one_column_per_axis_with_the_requested_bin_count():
    out = assign_strata(make_descriptors(30), n_strata=3)

    for axis in STRATUM_AXES:
        column = f"stratum_{axis}"
        assert column in out.columns
        assert set(out[column].cat.categories) == {"Q1", "Q2", "Q3"}
        assert out[column].value_counts().to_dict() == {"Q1": 10, "Q2": 10, "Q3": 10}


def test_assign_strata_is_deterministic_across_calls():
    descriptors = make_descriptors(23, seed=7)
    first = assign_strata(descriptors, n_strata=4)
    second = assign_strata(descriptors, n_strata=4)

    # A stratum must be a property of the pocket, not of where it sat in the table.
    order = np.random.default_rng(99).permutation(len(descriptors))
    shuffled = assign_strata(descriptors.iloc[order], n_strata=4)

    for axis in STRATUM_AXES:
        column = f"stratum_{axis}"
        pd.testing.assert_series_equal(first[column], second[column])
        pd.testing.assert_series_equal(
            first.set_index("pocket_id")[column],
            shuffled.set_index("pocket_id")[column].reindex(first["pocket_id"]),
        )


def test_ordering_is_monotone_in_the_source_column():
    out = assign_strata(make_descriptors(12), n_strata=3)
    codes = out["stratum_burial"].cat.codes.to_numpy()
    assert np.all(np.diff(codes) >= 0)  # mean_burial was constructed increasing


def test_fewer_pockets_than_strata_yields_fewer_bins_not_a_crash():
    out = assign_strata(make_descriptors(2), n_strata=5)
    for axis in STRATUM_AXES:
        assert len(out[f"stratum_{axis}"].cat.categories) <= 2
    assert out["stratum_burial"].notna().all()


def test_a_constant_descriptor_collapses_to_one_stratum():
    descriptors = make_descriptors(6)
    descriptors["frac_polar"] = 0.5
    out = assign_strata(descriptors, n_strata=3)

    assert list(out["stratum_polarity"].cat.categories) == ["Q1"]
    assert (out["stratum_polarity"] == "Q1").all()


def test_missing_descriptor_gives_a_missing_stratum():
    descriptors = make_descriptors(9)
    descriptors.loc[0, "mean_volume"] = np.nan
    out = assign_strata(descriptors, n_strata=3)

    assert pd.isna(out.loc[0, "stratum_volume"])
    assert out["stratum_volume"].notna().sum() == 8


def test_missing_source_column_is_a_clear_error():
    descriptors = make_descriptors(5).drop(columns=[AXIS_SOURCE_COLUMN["volume"]])
    with pytest.raises(KeyError, match="mean_volume"):
        assign_strata(descriptors)


# ------------------------------------------------------------- sigma_by_stratum


def test_cv_is_zero_for_identical_sigmas_and_grows_with_spread():
    """S2.5 in miniature: CV is the scale-free statement of how much sigma moves."""
    base = make_descriptors(9)
    cvs = []
    for spread in (0.0, 0.05, 0.2):
        descriptors = base.copy()
        # Within each of the three strata (3 pockets each), sigma is 1 +/- spread.
        descriptors["median_sigma"] = np.tile([1.0 - spread, 1.0, 1.0 + spread], 3)
        table = sigma_by_stratum(assign_strata(descriptors, n_strata=3))
        burial = table[table["axis"] == "burial"]
        assert (burial["n"] == 3).all()
        cvs.append(float(burial["cv_sigma"].mean()))

    assert cvs[0] == pytest.approx(0.0)
    assert cvs[0] < cvs[1] < cvs[2]


def test_cv_across_strata_detects_an_architecture_dependent_sigma():
    """The failure mode a Z-score hides: tight within a stratum, very different between."""
    descriptors = make_descriptors(9)
    descriptors["median_sigma"] = np.repeat([0.5, 1.0, 2.0], 3) + np.tile(
        [-1e-3, 0.0, 1e-3], 3
    )
    table = sigma_by_stratum(assign_strata(descriptors, n_strata=3))
    burial = table[table["axis"] == "burial"]

    assert (burial["cv_sigma"] < 0.01).all()
    assert float(burial["cv_across_strata"].iloc[0]) > 0.5
    assert burial["cv_across_strata"].nunique() == 1  # one value, broadcast per axis


def test_moments_match_a_hand_computation():
    descriptors = make_descriptors(6)
    descriptors["median_sigma"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    table = sigma_by_stratum(assign_strata(descriptors, n_strata=3))
    row = table[(table["axis"] == "burial") & (table["stratum"] == "Q1")].iloc[0]

    assert row["n"] == 2
    assert row["mean_sigma"] == pytest.approx(1.5)
    assert row["std_sigma"] == pytest.approx(np.std([1.0, 2.0], ddof=1))
    assert row["cv_sigma"] == pytest.approx(np.std([1.0, 2.0], ddof=1) / 1.5)


def test_single_pocket_stratum_reports_n_without_a_std():
    descriptors = make_descriptors(3)
    descriptors["median_sigma"] = [1.0, 2.0, 3.0]
    table = sigma_by_stratum(assign_strata(descriptors, n_strata=3))
    burial = table[table["axis"] == "burial"]

    assert (burial["n"] == 1).all()
    assert burial["std_sigma"].isna().all()
    assert burial["cv_sigma"].isna().all()


def test_every_axis_appears_in_the_fixed_order():
    table = sigma_by_stratum(assign_strata(make_descriptors(12)))
    assert list(dict.fromkeys(table["axis"])) == list(STRATUM_AXES)


def test_an_alternative_sigma_column_is_honoured():
    descriptors = assign_strata(make_descriptors(6))
    descriptors["mean_sigma_axisD"] = np.arange(6.0)
    table = sigma_by_stratum(descriptors, sigma_column="mean_sigma_axisD")

    assert table[table["axis"] == "burial"]["n"].sum() == 6
    with pytest.raises(KeyError, match="nope"):
        sigma_by_stratum(descriptors, sigma_column="nope")


def test_strata_columns_are_required():
    with pytest.raises(KeyError, match="assign_strata"):
        sigma_by_stratum(make_descriptors(4))


# -------------------------------------------------------------- axis_redundancy


def test_an_axis_against_itself_is_exactly_one():
    rng = np.random.default_rng(11)
    values = rng.normal(size=200)
    table = axis_redundancy({"identity": values, "chemotype": rng.normal(size=200)})
    self_rows = table[table["is_self"]]

    assert len(self_rows) == 2
    assert self_rows["pearson_r"].to_numpy() == pytest.approx(1.0)
    assert self_rows["spearman_rho"].to_numpy() == pytest.approx(1.0)
    assert self_rows["exceeds_threshold"].isna().all()


def test_uncorrelated_axes_fall_below_the_s2_6_threshold():
    """The S2.6 verdict: an axis that carries independent information."""
    rng = np.random.default_rng(3)
    n = 4000
    identity = rng.normal(size=n)
    chemotype = rng.normal(size=n)  # constructed independent

    table = axis_redundancy({"identity": identity, "chemotype": chemotype})
    row = table[
        (table["axis_a"] == "identity") & (table["axis_b"] == "chemotype")
    ].iloc[0]

    assert abs(row["pearson_r"]) < 0.1
    assert row["max_abs_correlation"] < REDUNDANCY_THRESHOLD
    assert bool(row["exceeds_threshold"]) is False


def test_a_redundant_axis_is_flagged():
    rng = np.random.default_rng(5)
    identity = rng.normal(size=500)
    chemotype = identity + 0.05 * rng.normal(size=500)

    table = axis_redundancy({"identity": identity, "chemotype": chemotype})
    row = table[table["is_self"] == False].iloc[0]  # noqa: E712

    assert row["max_abs_correlation"] > REDUNDANCY_THRESHOLD
    assert bool(row["exceeds_threshold"]) is True


def test_spearman_sees_a_monotone_relation_that_pearson_understates():
    x = np.linspace(0.1, 5.0, 300)
    table = axis_redundancy({"a": x, "b": np.exp(3 * x)})
    row = table[table["is_self"] == False].iloc[0]  # noqa: E712

    assert row["spearman_rho"] == pytest.approx(1.0)
    assert row["pearson_r"] < row["spearman_rho"]


def test_all_unordered_pairs_are_present_once():
    rng = np.random.default_rng(1)
    axes = {name: rng.normal(size=50) for name in ("A", "B", "C", "D")}
    table = axis_redundancy(axes)

    assert len(table) == 10  # 4 self + 6 off-diagonal
    keys = {frozenset((a, b)) for a, b in zip(table["axis_a"], table["axis_b"])}
    assert len(keys) == 10


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="pair ordering"):
        axis_redundancy({"a": np.zeros(10), "b": np.zeros(9)})


def test_non_finite_entries_are_dropped_pairwise():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    row = axis_redundancy({"a": a, "b": b}).iloc[1]

    assert row["n"] == 3
    assert row["pearson_r"] == pytest.approx(1.0)


def test_a_constant_axis_yields_nan_not_an_exception():
    table = axis_redundancy({"a": np.ones(20), "b": np.arange(20.0)})
    row = table[table["is_self"] == False].iloc[0]  # noqa: E712

    assert np.isnan(row["pearson_r"])
    assert pd.isna(row["exceeds_threshold"])


def test_empty_input_returns_the_empty_table():
    table = axis_redundancy({})
    assert table.empty
    assert "max_abs_correlation" in table.columns
