"""Tests for the F2/F3/F6 validation anchors.

Two tiers. The ``unit`` tier covers the pure logic — masks, the reference-table reader, the
bootstrap, the pinned sets, the registry hook — and touches neither PyRosetta nor ``data/``.
The ``integration`` tier runs the cases themselves at the smallest configuration that still
exercises every code path, and skips cleanly when PyRosetta or the DVC-tracked data is
absent.

**No scientific value is asserted that was not measured first.** The two numbers pinned
below — 22 ligand-incident contacts for 5EM8 at a 6 A heavy-atom cutoff, and 5GMP's pocket
mutate set being a strict subset of the whole protein — are geometry, deterministic and
decoy-independent, and were read off a run before being written here. The frustration counts
themselves are *not* pinned: at smoke scale they are not a reproduction claim (see the
module docstring of ``atomfrust.validation.anchors``), and pinning them would convert a
measurement into an expectation nobody validated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomfrust.validation import anchors
from atomfrust.validation.anchors import (
    CASES,
    PINNED_INTERFACES,
    PINNED_REFERENCE_STRUCTURES,
    CaseResult,
    bootstrap_fraction_ci,
    case_interface_fraction,
    case_pocket_repack_equivalence,
    case_reference_counts,
    find_interface_structure,
    interface_mask,
    ligand_incident_mask,
    paper_reference_counts,
    register_all,
    write_protein_only_pdb,
)

PROCESSED = anchors.PROCESSED_DIR
PARAMS = anchors.PARAMS_DIR


def _pairs(rows: list[tuple[str, str, str, str, bool]]) -> pd.DataFrame:
    """``(node_i, node_j, kind_i, kind_j, same_chain)`` as the graph would emit them."""
    return pd.DataFrame(
        rows, columns=["node_i", "node_j", "kind_i", "kind_j", "same_chain"]
    )


# --------------------------------------------------------------------------- unit tier


@pytest.mark.unit
def test_ligand_incident_mask_selects_exactly_one_nonprotein_endpoint():
    pairs = _pairs(
        [
            ("A:745", "A:790", "protein", "protein", True),
            ("A:745", "A:9001", "protein", "ligand", True),
            ("A:9001", "A:9002", "ligand", "metal", True),
            ("A:797", "A:9002", "noncanonical", "metal", True),
        ]
    )
    assert ligand_incident_mask(pairs).tolist() == [False, True, False, True]


@pytest.mark.unit
def test_interface_mask_needs_two_protein_endpoints_in_different_chains():
    pairs = _pairs(
        [
            ("A:10", "B:20", "protein", "protein", False),
            ("A:10", "A:20", "protein", "protein", True),
            ("A:10", "B:9001", "protein", "ligand", False),
        ]
    )
    assert interface_mask(pairs).tolist() == [True, False, False]


@pytest.mark.unit
def test_paper_reference_counts_normalises_the_id_case(tmp_path):
    csv = tmp_path / "ref.csv"
    csv.write_text(
        "pdb_id,paper_minimally_frustrated_contacts,"
        "paper_highly_frustrated_contacts,affinity_pM\n1xkk,21,0,3\n"
    )
    table = paper_reference_counts(csv)
    assert table.loc["1XKK", "paper_minimally_frustrated_contacts"] == 21


@pytest.mark.unit
def test_paper_reference_counts_names_the_missing_column(tmp_path):
    csv = tmp_path / "ref.csv"
    csv.write_text("pdb_id,affinity_pM\n1xkk,3\n")
    with pytest.raises(ValueError, match="paper_minimally_frustrated_contacts"):
        paper_reference_counts(csv)


@pytest.mark.unit
def test_real_reference_table_spans_the_published_range():
    if not anchors.REFERENCE_TABLE.exists():  # pragma: no cover - always present in-repo
        pytest.skip(f"missing {anchors.REFERENCE_TABLE}")
    counts = paper_reference_counts()["paper_minimally_frustrated_contacts"]
    assert (counts.min(), counts.max()) == anchors.PAPER_COUNT_RANGE
    assert len(counts) == anchors.FULL_SET_STRUCTURES


@pytest.mark.unit
def test_bootstrap_ci_brackets_the_point_estimate():
    successes = np.array([1] * 30 + [0] * 70)
    low, high = bootstrap_fraction_ci(successes, n_boot=500, seed=1)
    assert low < successes.mean() < high


@pytest.mark.unit
def test_bootstrap_ci_of_nothing_is_nan():
    low, high = bootstrap_fraction_ci(np.array([]))
    assert np.isnan(low) and np.isnan(high)


@pytest.mark.unit
def test_case_result_rejects_a_status_outside_the_three():
    with pytest.raises(ValueError, match="PASS/FAIL/SKIP"):
        CaseResult("x", "OK")


@pytest.mark.unit
def test_case_result_dict_carries_the_full_shape():
    keys = set(CaseResult("x", "PASS", 1.0, 2.0, 0.5, "why").as_dict())
    assert keys == {"name", "status", "measured", "expected", "tolerance", "detail", "extra"}


@pytest.mark.unit
def test_write_protein_only_pdb_drops_heteroatoms_and_other_chains(tmp_path):
    source = tmp_path / "in.pdb"
    source.write_text(
        "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
        "ATOM      2  N   ALA B   1      11.104   6.134  -6.504  1.00  0.00           N\n"
        "ATOM      3  N  BALA A   2      11.104   6.134  -6.504  0.50  0.00           N\n"
        "ATOM      4  N   ALA C   1      11.104   6.134  -6.504  1.00  0.00           N\n"
        "HETATM    5  O   HOH A 900      11.104   6.134  -6.504  1.00  0.00           O\n"
        "END\n"
    )
    out = write_protein_only_pdb(source, ("A", "B"), tmp_path / "out.pdb")
    lines = [l for l in out.read_text().splitlines() if l.startswith("ATOM")]
    assert len(lines) == 2
    assert {l[21] for l in lines} == {"A", "B"}


@pytest.mark.unit
def test_write_protein_only_pdb_refuses_a_chain_that_is_not_there(tmp_path):
    source = tmp_path / "in.pdb"
    source.write_text(
        "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
    )
    with pytest.raises(ValueError, match="no ATOM records"):
        write_protein_only_pdb(source, ("Z",), tmp_path / "out.pdb")


@pytest.mark.unit
def test_find_interface_structure_is_case_insensitive_and_never_fetches(tmp_path):
    (tmp_path / "1brs.pdb").write_text("ATOM\n")
    assert find_interface_structure("1BRS", [tmp_path]).name == "1brs.pdb"
    assert find_interface_structure("9ZZZ", [tmp_path]) is None
    assert find_interface_structure("1BRS", [tmp_path / "nope"]) is None


@pytest.mark.unit
def test_pinned_sets_are_what_was_pinned_before_any_run():
    # F2 asks for the cases to be chosen and pinned *before* running, so that the +/-3 pp
    # tolerance means something. This test is the pin: changing either tuple has to be a
    # deliberate, reviewed edit rather than a quiet post-hoc reselection.
    assert tuple(c.pdb_id for c in PINNED_INTERFACES) == ("1BRS", "1AY7", "1JTG")
    assert all(len(c.chains) == 2 and c.reason for c in PINNED_INTERFACES)
    assert PINNED_REFERENCE_STRUCTURES == ("5EM8", "5GMP", "1XKK")


@pytest.mark.unit
def test_registry_names_match_the_plans_cli_surface():
    assert set(CASES) == {
        "interface-fraction",
        "reference-counts",
        "pocket-repack-equivalence",
    }
    assert all(callable(f) for f in CASES.values())


@pytest.mark.unit
def test_register_all_accepts_a_mapping_or_a_callable():
    mapping: dict = {}
    assert register_all(mapping) == sorted(CASES)
    assert set(mapping) == set(CASES)

    seen: list[str] = []
    assert register_all(lambda name, fn: seen.append(name)) == sorted(CASES)
    assert set(seen) == set(CASES)


@pytest.mark.unit
def test_register_all_is_inert_when_the_registry_has_no_usable_shape():
    assert register_all(object()) == []


@pytest.mark.unit
def test_the_three_cases_reach_the_orchestrator_registry():
    # Importing anchors publishes them; the sibling module owns F1/F4/F5 and knows nothing
    # about this one, so `atomfrust validate --case F3` depends on this registration.
    cases = pytest.importorskip("atomfrust.validation.cases")
    assert {"F2", "F3", "F6"} <= set(cases.CASES)
    assert register_all() == []  # idempotent: a second call must not raise on duplicates
    for name in ("F2", "F3", "F6"):
        case = cases.get_case(name)
        assert case.summary and case.description


@pytest.mark.unit
def test_orchestrator_expectations_are_the_pinned_ones():
    cases = pytest.importorskip("atomfrust.validation.cases")
    f2 = cases.get_case("F2")
    assert f2.expected == {"frac_minimally": anchors.PUBLISHED_INTERFACE_FRACTION}
    assert f2.tolerance == {
        "frac_minimally": anchors.PUBLISHED_INTERFACE_TOLERANCE_PP / 100.0
    }
    # F3 gates on the scale (does the computed range overlap the published 4-23?), not on
    # per-structure agreement, which smoke scale could not support.
    assert cases.get_case("F3").expected == {"range_overlaps_paper": True}
    # F6 gates on nothing: plan step F6 says a low rho is a finding, not a failure.
    assert dict(cases.get_case("F6").expected) == {}


# -------------------------------------------------------------------- integration tier


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")


@pytest.fixture(scope="module")
def pyrosetta_available():
    pytest.importorskip("pyrosetta")


@pytest.mark.integration
def test_interface_fraction_skips_and_says_what_it_needs(pyrosetta_available, tmp_path):
    result = case_interface_fraction(search=[tmp_path])
    assert result.status == "SKIP"
    assert result.expected == anchors.PUBLISHED_INTERFACE_FRACTION
    assert result.extra["pinned"] == ["1BRS", "1AY7", "1JTG"]
    for pdb_id in ("1BRS", "1AY7", "1JTG"):
        assert pdb_id in result.detail
    assert "downloaded" in result.detail


@pytest.mark.integration
def test_interface_fraction_skips_when_a_found_file_lacks_the_pinned_chains(
    pyrosetta_available, tmp_path
):
    # A single-chain file under the right id is missing data, not a failed measurement.
    for case in anchors.PINNED_INTERFACES:
        (tmp_path / f"{case.pdb_id}.pdb").write_text(
            "ATOM      1  N   ALA "
            + case.chains[0]
            + "   1      11.104   6.134  -6.504  1.00  0.00           N\n"
        )
    result = case_interface_fraction(search=[tmp_path])
    assert result.status == "SKIP"
    assert len(result.extra["unusable"]) == len(anchors.PINNED_INTERFACES)


@pytest.mark.integration
def test_reference_counts_returns_a_well_formed_smoke_scale_result(pyrosetta_available):
    _require(PROCESSED / "5EM8_clean.pdb", PARAMS / "5Q4.params", anchors.REFERENCE_TABLE)
    result = case_reference_counts(pdb_ids=("5EM8",), n_decoys=2)

    assert result.status in ("PASS", "FAIL")
    assert set(result.measured) == {"counts", "range", "pearson_r", "pearson_p"}
    assert result.expected["counts"] == {"5EM8": 4}
    assert result.extra["smoke_scale"] is True
    assert "SMOKE SCALE" in result.detail
    assert result.extra["full_run_core_hours"] > 0

    (row,) = result.extra["per_structure"]
    # Measured, not assumed: 5EM8's ligand touches 22 protein residues at a 6 A heavy-atom
    # minimum distance. Pure geometry of the deposited pose, so it is decoy-independent and
    # a change here means the graph or the cutoff moved.
    assert row["n_ligand_contacts"] == 22
    assert 0 <= row["computed_minimally"] <= row["n_ligand_contacts"]
    # The whole point of A4: this is the paper's scale (4-23), not the prototype's 266-407.
    assert row["n_ligand_contacts"] < row["n_contacts_total"]


@pytest.mark.integration
def test_pocket_repack_equivalence_pairs_two_arms_and_reports_rho(pyrosetta_available):
    _require(PROCESSED / "5GMP_clean.pdb", PARAMS / "F62.params")
    result = case_pocket_repack_equivalence(n_decoys=2, shell_radii=(8.0,))

    assert result.status == "PASS"
    (row,) = result.measured
    assert row["shell_A"] == 8.0
    # A restricted arm must actually restrict something, or the comparison is vacuous.
    assert 0 < row["repack_residues"] < row["baseline_repack_residues"]
    assert 0 < row["mutate_residues"] <= row["repack_residues"]
    for key in ("rho_pocket", "rho_all"):
        assert np.isnan(row[key]) or -1.0 <= row[key] <= 1.0
    assert row["speedup"] > 0
    assert "rho_pocket" in result.detail
