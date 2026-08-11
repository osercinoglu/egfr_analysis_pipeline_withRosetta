"""G6 tests — the chemotype axis (axis D).

The unit tier is where the estimand is pinned, and it is pinned on **synthetic energies**
rather than on a pose, because the argument the plan makes is an argument about a
distribution, not about Rosetta. Three things are asserted there and nowhere else:

1. a member that never reaches a residue contributes an **exact zero**, and that zero is data
   (it is counted) rather than missingness (it is not dropped);
2. a naive residue-anchored per-contact Z over a *changing* molecule is a function of
   **occupancy and size** even when the underlying interaction is identical at every residue
   by construction — which is why the estimand is a ligand-node scalar instead;
3. removing the energy-on-heavy-atom-count slope changes which member wins, so the slope is
   load-bearing rather than cosmetic.

The integration tier loads 5GMP and scores at most four library members (three real ones from
the vendored ``tests/data/libraries/local`` fixture plus one deliberately unplaceable
molecule). Each real member costs a placement, a parametrisation subprocess, a pose load and
a pocket repack, so the ensemble is shared across the whole module by a session fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROCESSED = Path("data/processed")
PARAMS = Path("data/ligands/params")
FIXTURES = Path("tests/data/libraries/local")


# ============================================================ synthetic scaffolding


def _synthetic_pairs(residues: list[int], ligand_resnum: int = 100) -> pd.DataFrame:
    """A frozen pair table: every residue paired with the ligand, plus one protein pair.

    Only the columns :func:`ligand_site_energies` reads are present — ``pair_id``, the node
    ids and the two pose residue numbers — which is the point: the estimand is defined over
    a table, not over a pose.
    """
    rows = []
    for r in residues:
        rows.append(
            {"node_i": f"A:{r}", "node_j": f"A:{ligand_resnum}", "i": r, "j": ligand_resnum}
        )
    rows.append(
        {
            "node_i": f"A:{residues[0]}",
            "node_j": f"A:{residues[-1]}",
            "i": residues[0],
            "j": residues[-1],
        }
    )
    frame = pd.DataFrame(rows)
    frame.insert(0, "pair_id", np.arange(len(frame), dtype=np.int32))
    return frame


def _member_energies(pairs: pd.DataFrame, per_residue: dict[int, float], ligand_resnum: int = 100):
    """``(e_direct, e_fa_rep)`` over a pair table, from a residue -> energy mapping.

    A residue absent from the mapping gets exactly ``(0.0, 0.0)`` — Rosetta's answer for a
    pair with no energy-graph edge, and the point mass this axis has to reason about.
    """
    e_direct = np.zeros(len(pairs), dtype=np.float64)
    i = pairs["i"].to_numpy()
    j = pairs["j"].to_numpy()
    partner = np.where(i == ligand_resnum, j, i)
    incident = (i == ligand_resnum) | (j == ligand_resnum)
    for k in range(len(pairs)):
        if incident[k]:
            e_direct[k] = per_residue.get(int(partner[k]), 0.0)
    return e_direct, np.zeros_like(e_direct)


# ============================================================ unit: the estimand


@pytest.mark.unit
def test_a_member_that_never_reaches_a_residue_is_an_exact_zero_and_is_kept():
    """The point mass at zero is data. Counted as non-contacting, never dropped."""
    from atomfrust.decoys.chemotype import ligand_site_energies

    residues = [10, 11, 12, 13]
    pairs = _synthetic_pairs(residues)

    reaches_all, fa = _member_energies(pairs, {r: -2.0 for r in residues})
    reaches_two, _ = _member_energies(pairs, {10: -2.0, 11: -2.0})

    big = ligand_site_energies(
        pairs, reaches_all, fa, ligand_resnum=100, shell_resnums=residues, mode="pair_only"
    )
    small = ligand_site_energies(
        pairs, reaches_two, fa, ligand_resnum=100, shell_resnums=residues, mode="pair_only"
    )

    assert big.n_keys == small.n_keys == 4, "the key set is frozen, not per member"
    assert big.n_contacting == 4 and small.n_contacting == 2
    # Not dropped: the two unreached keys are present, and their energy is exactly 0.0.
    unreached = small.per_key.loc[~small.per_key["contacts"]]
    assert len(unreached) == 2
    assert (unreached["e_direct"].to_numpy() == 0.0).all()
    assert small.total == pytest.approx(-4.0) and big.total == pytest.approx(-8.0)


@pytest.mark.unit
def test_the_shell_is_frozen_so_energy_outside_the_native_pocket_is_not_counted():
    """Recomputing the shell per decoy was rejected: the estimand would vary per sample.

    A bigger molecule that reaches a residue the native never touched must not thereby be
    scored over a bigger site — it would be measuring a different quantity from the one every
    other member was measured on.
    """
    from atomfrust.decoys.chemotype import ligand_site_energies

    residues = [10, 11, 12, 99]
    pairs = _synthetic_pairs(residues)
    native_shell = [10, 11, 12]  # residue 99 is outside the native pocket

    reaches_further, fa = _member_energies(pairs, {10: -1.0, 11: -1.0, 12: -1.0, 99: -50.0})
    site = ligand_site_energies(
        pairs, reaches_further, fa, ligand_resnum=100, shell_resnums=native_shell,
        mode="pair_only",
    )
    assert site.n_keys == 3
    assert 99 not in set(site.per_key["pose_resnum"])
    assert site.total == pytest.approx(-3.0), "the -50 outside the frozen shell is not counted"


@pytest.mark.unit
def test_a_naive_per_contact_z_measures_occupancy_not_chemistry():
    """**The reason axis D is a ligand-node scalar.**

    Every member here interacts with every residue it reaches at exactly the same strength,
    so there is no chemistry to discriminate — the residues differ only in how many members
    reach them. A residue-anchored Z nevertheless returns a different number per residue,
    because it is taken over a mixture of a point mass at zero and a continuous part whose
    weight is the occupancy. Its answer is an artefact of reach.
    """
    from atomfrust.analyze.zscore import zscore

    n_members = 10
    occupancies = {"full": 10, "most": 8, "half": 5, "few": 2}
    strength = -3.0

    decoys = np.zeros((n_members, len(occupancies)))
    for column, count in enumerate(occupancies.values()):
        decoys[:count, column] = strength
    native = np.full(len(occupancies), strength)

    z = zscore(native, decoys)
    by_name = dict(zip(occupancies, z))

    # Identical physics at every residue, yet the statistic ranges over more than a full
    # sigma and is monotone in occupancy.
    assert by_name["full"] == 0.0, "a degenerate (fully occupied) column has no scale at all"
    assert by_name["half"] > 0.9
    assert by_name["few"] > by_name["most"] > 0.0
    spread = float(np.ptp(z))
    assert spread > 1.0, f"identical chemistry produced a Z spread of {spread:.2f}"
    occupancy_fraction = np.array(list(occupancies.values())) / n_members
    rho = float(pd.Series(z).corr(pd.Series(occupancy_fraction), method="spearman"))
    assert rho == pytest.approx(-1.0), "the naive Z is a monotone function of occupancy alone"


@pytest.mark.unit
def test_removing_the_size_slope_changes_which_member_wins():
    """The ligand-node scalar is size-confounded until the regression slope is removed.

    Five members whose site energy is *nothing but* their heavy-atom count, plus one
    mid-sized member with a genuinely stronger interaction. Raw, the biggest molecule wins
    and the specific one does not. Residualised on heavy-atom count, the specific one does.
    """
    from atomfrust.decoys.chemotype import rank_percentile_scores, size_regression

    sizes = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 30.0])
    energy = -1.0 * sizes
    energy[-1] -= 15.0  # the specific binder: same size, better interaction

    raw_rank = rank_percentile_scores(energy)
    assert raw_rank[4] == pytest.approx(1.0), "raw, the largest molecule wins"
    assert raw_rank[-1] < 1.0, "raw, the specific binder is beaten by sheer size"

    fit = size_regression(energy, sizes)
    assert fit.n == 6
    assert fit.slope < -0.5, f"energy falls with size: slope {fit.slope:.3f} REU/heavy atom"
    residual = fit.residuals(energy, sizes)
    resid_rank = rank_percentile_scores(residual)
    assert resid_rank[-1] == pytest.approx(1.0), "residualised, the specific binder wins"
    assert resid_rank[4] < 1.0


@pytest.mark.unit
def test_size_regression_refuses_to_fit_what_it_cannot():
    """Fewer than three members, or one constant size, is a refusal — not a fit to noise."""
    from atomfrust.decoys.chemotype import size_regression

    tiny = size_regression(np.array([-1.0, -2.0]), np.array([10.0, 20.0]))
    assert tiny.slope == 0.0 and tiny.intercept == pytest.approx(-1.5) and tiny.n == 2

    flat = size_regression(np.array([-1.0, -2.0, -3.0]), np.array([20.0, 20.0, 20.0]))
    assert flat.slope == 0.0 and flat.intercept == pytest.approx(-2.0)

    with_nan = size_regression(
        np.array([-1.0, np.nan, -3.0, -4.0]), np.array([10.0, 20.0, 30.0, 40.0])
    )
    assert with_nan.n == 3, "a member that failed to score takes no part in the fit"

    empty = size_regression(np.array([np.nan]), np.array([10.0]))
    assert empty.n == 0 and empty.slope == 0.0


@pytest.mark.unit
def test_the_member_ranks_are_the_packages_own_rank_percentile():
    """No second implementation of the index: `analyze.zscore.rank_percentile` is the one."""
    from atomfrust.analyze.zscore import rank_percentile
    from atomfrust.decoys.chemotype import rank_percentile_scores, rank_within

    values = np.array([-5.0, -1.0, -3.0, -3.0])
    scores = rank_percentile_scores(values)
    # Leave-one-out, exactly as the native is ranked against the members.
    expected = [
        float(rank_percentile(np.array([v]), np.delete(values, k).reshape(-1, 1))[0])
        for k, v in enumerate(values)
    ]
    assert scores == pytest.approx(expected)
    assert scores[0] == pytest.approx(1.0), "the most favourable member scores +1"
    assert scores[1] == pytest.approx(-1.0)
    assert scores[2] == pytest.approx(scores[3]), "ties rank identically"

    # A member that failed to score is NaN, and takes no part in anyone else's rank.
    with_failure = rank_percentile_scores(np.array([-5.0, np.nan, -1.0, -3.0]))
    assert np.isnan(with_failure[1])
    assert with_failure[0] == pytest.approx(1.0) and with_failure[2] == pytest.approx(-1.0)
    assert np.isnan(rank_percentile_scores(np.array([-5.0, -1.0]))).all(), (
        "leaving one out of two leaves an ensemble of one, which is not a rank"
    )
    assert np.isnan(rank_within(-1.0, np.array([np.nan])))


@pytest.mark.unit
def test_the_auroc_is_the_affine_image_of_the_rank_and_ties_count_half():
    from atomfrust.decoys.chemotype import auroc_against, rank_within

    ensemble = np.array([-1.0, -2.0, -3.0, -4.0])
    assert auroc_against(-5.0, ensemble) == pytest.approx(1.0)
    assert auroc_against(0.0, ensemble) == pytest.approx(0.0)
    assert auroc_against(-2.5, ensemble) == pytest.approx(0.5)
    # -2.0 beats only -1.0 outright and ties with -2.0: (1 + 0.5)/4.
    assert auroc_against(-2.0, ensemble) == pytest.approx(0.375), "the tie counts a half"
    for value in (-5.0, -2.5, -2.0, 0.0):
        assert auroc_against(value, ensemble) == pytest.approx(
            (rank_within(value, ensemble) + 1.0) / 2.0
        )
    assert np.isnan(auroc_against(np.nan, ensemble))


@pytest.mark.unit
def test_the_per_residue_decomposition_is_occupancy_filtered_and_labelled():
    """Descriptive only, and the table says so on every row."""
    from atomfrust.decoys.chemotype import per_residue_frame

    keys = pd.DataFrame(
        {"pair_id": [0, 1, 2], "pose_resnum": [10, 11, 12], "node_id": ["A:10", "A:11", "A:12"]}
    )
    contacts = np.array(
        [[True, True, False], [True, True, False], [True, False, False], [True, False, True]]
    )
    energies = np.where(contacts, -2.0, 0.0)
    native = np.array([-2.5, -2.5, -2.5])

    frame = per_residue_frame(keys, energies, contacts, native, min_occupancy=0.8)
    assert list(frame["pose_resnum"]) == [10], "only the 100%-occupied key survives 0.8"
    assert frame.attrs["n_keys_before_occupancy_filter"] == 3
    assert frame.attrs["descriptive_only"] is True
    assert set(frame["interpretation"]) == {"descriptive_only"}
    assert int(frame["n_contacting_members"].iloc[0]) == 4
    assert frame["native_rank_percentile"].iloc[0] == pytest.approx(1.0), (
        "-2.5 is more favourable than every member's -2.0, and favourable is the + tail"
    )

    loose = per_residue_frame(keys, energies, contacts, native, min_occupancy=0.4)
    assert list(loose["pose_resnum"]) == [10, 11]
    assert list(loose["n_contacting_members"]) == [4, 2]

    with pytest.raises(ValueError, match="shape mismatch"):
        per_residue_frame(keys, energies[:, :2], contacts, native)


# ============================================================ unit: the gate


class _StubAxis:
    """A generator with its scores pre-computed, so the gate can be tested without a pose."""

    def __init__(self, frame: pd.DataFrame, gate: float = 0.75) -> None:
        from atomfrust.decoys.chemotype import ChemotypeDecoyGenerator

        self.axis = "chemotype"
        self.gate_auroc = gate
        self._frame = frame
        self.member_scores = lambda: frame
        self.cross_axis_redundancy = ChemotypeDecoyGenerator.cross_axis_redundancy.__get__(self)
        self.native_rank_auroc = ChemotypeDecoyGenerator.native_rank_auroc.__get__(self)
        self.gate_passed = ChemotypeDecoyGenerator.gate_passed.__get__(self)


def _scores_frame(auroc: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "source_ref": ["lib:a", "lib:b", "lib:c", "lib:d", "native:X"],
            "is_native": [False, False, False, False, True],
            "rank_percentile": [1.0, 0.5, -0.5, -1.0, 1.0],
            "rank_percentile_resid": [1.0, 0.5, -0.5, -1.0, 1.0],
        }
    )
    frame.attrs["native_rank_auroc"] = auroc
    frame.attrs["native_rank_auroc_resid"] = auroc
    return frame


@pytest.mark.unit
def test_the_redundancy_output_refuses_to_emit_until_the_positive_control_passes():
    """Without the gate, a degenerate axis-D score is uncorrelated with axis A *by
    construction*, and S2.6's redundancy test would pass by noise."""
    from atomfrust.decoys.chemotype import PositiveControlFailed

    failing = _StubAxis(_scores_frame(0.60))
    assert failing.native_rank_auroc() == pytest.approx(0.60)
    assert failing.gate_passed() is False
    with pytest.raises(PositiveControlFailed, match="positive control"):
        failing.cross_axis_redundancy({"lib:a": 1.0, "lib:b": 0.2, "lib:c": -0.3, "lib:d": -1.0})

    undefined = _StubAxis(_scores_frame(float("nan")))
    with pytest.raises(PositiveControlFailed):
        undefined.cross_axis_redundancy([1.0, 0.2, -0.3, -1.0])


@pytest.mark.unit
def test_a_passing_gate_emits_a_redundancy_row_with_its_evidence_attached():
    passing = _StubAxis(_scores_frame(0.95))
    assert passing.gate_passed() is True

    by_ref = passing.cross_axis_redundancy(
        {"lib:a": 1.0, "lib:b": 0.5, "lib:c": -0.5, "lib:d": -1.0}, other_axis="identity"
    )
    assert len(by_ref) == 1
    row = by_ref.iloc[0]
    assert row["n"] == 4
    assert row["spearman_rho"] == pytest.approx(1.0)
    assert row["native_rank_auroc"] == pytest.approx(0.95) and row["gate_passed"]
    assert row["other_axis"] == "identity"
    assert row["score_column"] == "rank_percentile_resid", "size is removed by default"

    positional = passing.cross_axis_redundancy([1.0, 0.5, -0.5, -1.0])
    assert positional.iloc[0]["spearman_rho"] == pytest.approx(1.0)

    # A member the other axis has no score for is dropped from n, not imputed.
    partial = passing.cross_axis_redundancy({"lib:a": 1.0, "lib:b": 0.5, "lib:c": -0.5})
    assert partial.iloc[0]["n"] == 3

    with pytest.raises(ValueError, match="positional"):
        passing.cross_axis_redundancy([1.0, 0.5])


@pytest.mark.unit
def test_the_docking_free_route_is_the_live_one_on_this_machine():
    """Documents the environment the MCS ablation is the *only* route in.

    If this fails because a docking binary appeared, the module docstring's claim that the
    docking-free route is the live one has expired — and the ablation becomes a comparison
    (run both) rather than the single available placement.
    """
    from atomfrust.dock import available_backends

    available = available_backends()
    assert "mcs_align" in available
    assert "smina" not in available and "gnina" not in available, (
        f"a docking backend is now available ({available}) — run the chemotype axis under "
        "both placements and report them side by side, per G6's acceptance criterion"
    )


# ============================================================ integration fixtures


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")


@pytest.fixture(scope="module")
def context_5gmp():
    """A DecoyContext over 5GMP — same shape as tests/test_decoys_pose.py."""
    from atomfrust.decoys.base import DecoyContext
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex
    from atomfrust.regions import resolve_regions
    from atomfrust.settings import Settings
    from atomfrust.spec import LigandSpec, SystemSpec

    pdb = PROCESSED / "5GMP_clean.pdb"
    params = PARAMS / "F62.params"
    _require(pdb, params)

    spec = SystemSpec.from_pdb(pdb, system_id="5GMP")
    spec = spec.model_copy(
        update={"ligands": (LigandSpec(selector=spec.ligands[0].selector, params=params),)}
    )
    loaded = load_complex(spec)
    settings = Settings()
    _, pairs = build_graph(loaded.nodes, loaded.geometry, settings)
    return DecoyContext(
        pose=loaded.pose,
        nodes=loaded.nodes,
        pairs=pairs,
        regions=resolve_regions(loaded.nodes, loaded.geometry),
        settings=settings,
    )


@pytest.fixture(scope="module")
def members():
    """Four members: two actives, one property decoy, one molecule that cannot be placed.

    From the vendored fixture — nothing is downloaded. Methane is added by hand as the
    unplaceable case: its maximum common substructure with a 39-heavy-atom kinase inhibitor
    is one atom, below :class:`~atomfrust.dock.mcs_align.MCSAlignBackend`'s floor.
    """
    from atomfrust.chem.libraries import LocalSDFAdapter, MolRecord

    _require(FIXTURES)
    records = list(LocalSDFAdapter(FIXTURES).records())
    by_id = {r.source_id: r for r in records}
    return [
        by_id["GEFITINIB"],
        by_id["ERLOTINIB"],
        by_id["DeepCoy_1"],
        MolRecord(
            smiles="C",
            inchikey=None,
            source="handmade",
            source_id="METHANE",
            role="property_decoy",
        ),
    ]


@pytest.fixture(scope="module")
def axis(context_5gmp, members):
    from atomfrust.decoys.chemotype import ChemotypeDecoyGenerator

    return ChemotypeDecoyGenerator(context_5gmp, members, base_seed=42)


@pytest.fixture(scope="module")
def scores(axis):
    """The primary output, computed once — every member costs a pose."""
    return axis.member_scores()


# ============================================================ integration: the axis


@pytest.mark.integration
def test_the_frozen_shell_is_the_native_pocket_and_nothing_else(axis, context_5gmp):
    from atomfrust.decoys.chemotype import native_pocket_shell

    shell = axis.shell_resnums
    assert shell == sorted(shell) and len(shell) > 10
    assert axis.pose_resnum not in shell, "the ligand is not in its own pocket shell"
    # Frozen: asking twice returns the same object, and it never consults a member.
    assert axis.shell_resnums is shell
    assert shell == native_pocket_shell(
        context_5gmp.pose, context_5gmp.nodes, axis.pose_resnum, axis.shell_A
    )
    print(f"\nG6 5GMP frozen pocket shell at {axis.shell_A} A: {len(shell)} protein residues")


@pytest.mark.integration
def test_every_member_carries_covariates_and_a_status(scores, members):
    """The covariate table the plan requires: present for every member, failures included."""
    from atomfrust.decoys.chemotype import MEMBER_COVARIATES

    assert len(scores) == len(members) + 1, "one row per member plus the native"
    assert scores["is_native"].sum() == 1

    library = scores.loc[~scores["is_native"].astype(bool)]
    for column in MEMBER_COVARIATES:
        assert column in scores.columns
        assert library[column].notna().all(), f"{column} missing for a member"
    assert set(library["status"]) >= {"ok"}
    assert (library["estimand"] == "ligand_node_scalar").all()
    assert not library["per_contact_index_valid"].any(), (
        "the per-pair energies are raw material, not a vector to Z-score"
    )
    print(
        "\nG6 member table:\n"
        + library[
            ["source_id", "role", "status", "hac", "mw", "e_site", "n_contacting_keys", "n_keys"]
        ].to_string(index=False)
    )


@pytest.mark.integration
def test_a_molecule_that_cannot_be_placed_is_a_row_not_an_exception(scores):
    """A screening library's unusable members are a result, so they are counted, not raised."""
    methane = scores.loc[scores["source_id"] == "METHANE"].iloc[0]
    assert methane["status"].startswith("placement_failed")
    assert np.isnan(methane["e_site"])
    assert np.isnan(methane["rank_percentile"]), "a member with no score takes no rank"
    assert np.isfinite(methane["hac"]), "its covariates are still known"


@pytest.mark.integration
def test_the_estimand_is_a_ligand_node_scalar_over_a_shared_key_set(axis, scores):
    """Every scored member is measured over the *same* keys — that is the correspondence rule."""
    scored = [k for k in range(len(axis.members)) if axis._score_member(k).ok]
    assert len(scored) >= 2
    reference = axis._score_member(scored[0]).site.per_key
    for k in scored[1:]:
        other = axis._score_member(k).site.per_key
        assert list(other["pair_id"]) == list(reference["pair_id"])
        assert list(other["pose_resnum"]) == list(reference["pose_resnum"])

    library = scores.loc[~scores["is_native"].astype(bool)]
    ok = library.loc[library["status"] == "ok"]
    # Reach differs between molecules — which is exactly why a per-contact Z would be a
    # mixture and the scalar is not.
    assert ok["n_contacting_keys"].min() >= 1
    assert (ok["n_contacting_keys"] <= ok["n_keys"]).all()


@pytest.mark.integration
def test_the_size_slope_is_measured_and_removed(scores):
    """The confound is reported as a number, and the residualised score exists beside the raw."""
    attrs = scores.attrs
    assert {"size_slope", "size_intercept", "size_r", "size_n"} <= set(attrs)
    assert "e_site_resid" in scores.columns and "rank_percentile_resid" in scores.columns
    print(
        f"\nG6 energy-on-heavy-atom-count regression (n={int(attrs['size_n'])}): "
        f"slope {attrs['size_slope']:.3f} REU/heavy atom, intercept "
        f"{attrs['size_intercept']:.2f}, r {attrs['size_r']:.3f}"
    )
    library = scores.loc[~scores["is_native"].astype(bool)]
    ok = library.loc[library["status"] == "ok"]
    fitted = attrs["size_intercept"] + attrs["size_slope"] * ok["hac"]
    assert np.allclose(ok["e_site"] - fitted, ok["e_site_resid"])


@pytest.mark.integration
def test_the_native_ranks_within_its_own_ensemble(axis, scores):
    """**The positive-control gate.** Reported whether it passes or not, and it must be
    consistent with the score column it claims to summarise."""
    auroc = axis.native_rank_auroc()
    native = scores.loc[scores["is_native"].astype(bool)].iloc[0]
    ensemble = scores.loc[~scores["is_native"].astype(bool), "e_site"].to_numpy(dtype=float)
    ensemble = ensemble[np.isfinite(ensemble)]

    print(
        f"\nG6 positive control on 5GMP (F62 vs {ensemble.size} scored members): "
        f"native e_site {native['e_site']:.2f}, members "
        f"{np.array2string(np.sort(ensemble), precision=2)}, AUROC {auroc:.3f} "
        f"(residualised {axis.native_rank_auroc(residualised=True):.3f}), "
        f"gate {axis.gate_auroc:.2f} -> {'PASS' if scores.attrs['gate_passed'] else 'FAIL'}"
    )
    assert 0.0 <= auroc <= 1.0
    assert auroc == pytest.approx((float(native["rank_percentile"]) + 1.0) / 2.0)
    assert scores.attrs["gate_passed"] is (auroc >= axis.gate_auroc)

    # Deliberately NOT asserting `auroc >= 0.75`. That is a scientific outcome, not a
    # property of this code, and at the moment it does not hold: on 5GMP against the three
    # vendored fixture molecules the native F62 scores -446 REU while two members score
    # -722 and -671, giving AUROC 0.333 (0.000 after residualising on size). Asserting the
    # gate passes would turn an unestablished result into a green test.
    #
    # What must hold is that the machinery is honest about it, and that is asserted above
    # and in `test_the_redundancy_output_is_gated_on_the_real_ensemble`: when the gate
    # fails, no cross-axis redundancy number is emitted.
    #
    # Three fixture molecules is not evidence the axis is broken either — they are tiny
    # hand-written stand-ins, not a property-matched decoy set, and MCS alignment places
    # them without any pose search. Establishing the gate needs a real library (plan G3's
    # DUD-E/DEKOIS adapters) against a real target.


@pytest.mark.integration
def test_the_redundancy_output_is_gated_on_the_real_ensemble(axis, scores):
    """The gate is wired to the real numbers, not only to the stub in the unit tier."""
    from atomfrust.decoys.chemotype import PositiveControlFailed

    original = axis.gate_auroc
    try:
        axis.gate_auroc = 1.5  # unreachable
        with pytest.raises(PositiveControlFailed, match="no override"):
            axis.cross_axis_redundancy([0.0] * len(axis.members))
    finally:
        axis.gate_auroc = original

    if scores.attrs["gate_passed"]:
        result = axis.cross_axis_redundancy(
            list(np.arange(len(axis.members), dtype=float)), other_axis="identity"
        )
        assert len(result) == 1 and result.iloc[0]["gate_passed"]


@pytest.mark.integration
def test_the_per_residue_decomposition_is_descriptive_and_occupancy_filtered(axis):
    frame = axis.per_residue_decomposition()
    assert frame.attrs["descriptive_only"] is True
    assert "descriptive" in frame.attrs["note"]
    if len(frame):
        assert set(frame["interpretation"]) == {"descriptive_only"}
        assert (frame["occupancy"] >= 0.8).all()
        assert (frame["n_contacting_members"] > 0).all()
    loose = axis.per_residue_decomposition(min_occupancy=0.0)
    assert len(loose) >= len(frame)
    print(
        f"\nG6 per-residue decomposition (descriptive): {len(frame)} of "
        f"{frame.attrs['n_keys_before_occupancy_filter']} keys at >=80% member occupancy"
    )


@pytest.mark.integration
def test_generate_returns_a_decoy_result_that_says_what_it_is(axis):
    """The axis satisfies the DecoyGenerator protocol, and its result is self-describing."""
    from atomfrust.decoys.base import DecoyGenerator

    assert isinstance(axis, DecoyGenerator)
    result = axis.generate(0)
    assert result.decoy_id == 0
    assert result.e_direct.shape[0] == len(axis.context.pairs)
    assert result.index_row["axis"] == "chemotype"
    assert result.index_row["generator"] == "chemotype/mcs_align"
    assert result.index_row["per_contact_index_valid"] is False
    assert result.index_row["source_ref"]

    with pytest.raises(IndexError, match="library"):
        axis.generate(len(axis.members))


@pytest.mark.integration
def test_the_native_is_relaxed_by_the_members_protocol(axis, context_5gmp):
    """Relaxing decoys but not the native would rig the gate in the decoys' favour."""
    from atomfrust.decoys.identity import assert_backbone_identical

    native = axis.prepare_native()
    assert native is not context_5gmp.pose, "the crystal pose itself must not be mutated"
    assert assert_backbone_identical(context_5gmp.pose, native, tol=1e-6) <= 1e-6
