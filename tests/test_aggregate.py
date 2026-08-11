"""D3 tests for atomfrust.analyze.aggregate — selectors, mandatory covariates, R28.

The last section is the acceptance test: every class count stored in
``results/egfr_frustration_summary.csv`` is recomputed from the corresponding per-contact
parquet through this module's registry and asserted equal exactly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomfrust.analyze.aggregate import (
    COVARIATES,
    DESCRIPTORS,
    DescriptorInput,
    covariates,
    pocket_mask,
    shell_nodes,
    summarize,
    summarize_many,
)
from atomfrust.analyze.classify import classify_index

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
SUMMARY_CSV = REPO / "results" / "egfr_frustration_summary.csv"
RESULTS_DIR = REPO / "results"
PROCESSED_DIR = REPO / "data" / "processed"


# ------------------------------------------------------------------- small fixtures


def toy_system():
    """Four protein residues in two chains plus one ligand; six pairs.

    Distances are chosen so that a 6 Å heavy-min shell keeps a strict subset of an 8 Å one,
    which is what the shell-sensitivity test needs.
    """
    nodes = pd.DataFrame(
        {
            "node_id": ["A:1", "A:2", "A:3", "B:1", "L:LIG:1"],
            "pose_resnum": [1, 2, 3, 4, 5],
            "kind": ["protein", "protein", "protein", "protein", "ligand"],
            "chain": ["A", "A", "A", "B", "L"],
            "resseq": [1, 2, 3, 1, 1],
            "icode": ["", "", "", "", ""],
            "resname": ["ALA", "TRP", "GLY", "LEU", "LIG"],
            "name1": ["A", "W", "G", "L", "X"],
            "n_heavy": [5, 14, 4, 8, 30],
            "rel_sasa": [0.5, 0.25, 1.0, 0.1, np.nan],
            "mutable": [True, True, True, True, False],
        }
    )
    pairs = pd.DataFrame(
        {
            "pair_id": np.arange(6, dtype=np.int32),
            "node_i": ["A:1", "A:1", "A:2", "A:1", "A:2", "A:3"],
            "node_j": ["A:2", "A:3", "A:3", "B:1", "B:1", "L:LIG:1"],
            "i": [1, 1, 2, 1, 2, 3],
            "j": [2, 3, 3, 4, 4, 5],
            "kind_i": ["protein"] * 6,
            "kind_j": ["protein", "protein", "protein", "protein", "protein", "ligand"],
            "same_chain": [True, True, True, False, False, False],
            "seq_sep": [1, 2, 1, -1, -1, -1],
            "seq_sep_pose": [1, 2, 1, -1, -1, -1],
            "d_ca": [5.0, 7.0, 9.0, 6.0, 11.0, np.nan],
            "d_cb": [4.5, 6.5, 8.5, 5.5, 10.5, np.nan],
            "d_heavy_min": [3.0, 5.0, 7.0, 4.0, 9.0, 3.5],
            "is_bonded": [False] * 6,
            "E_native": [-2.0, -1.0, -0.5, -3.0, -0.25, -4.0],
        }
    )
    F = np.array([2.0, 0.9, 0.0, -2.0, 0.5, 3.0])
    return pairs, nodes, F


def toy_labels(F):
    return classify_index(F)


# ---------------------------------------------------------------------- descriptors


def test_registry_holds_exactly_the_specified_descriptors():
    assert set(DESCRIPTORS) == {
        "count_minimal",
        "count_highly",
        "count_neutral",
        "frac_minimal",
        "frac_highly",
        "frac_neutral",
        "energy_weighted_sum",
        "mean_Z",
        "median_Z",
        "mean_Z_top_decile",
        "count_minimal_per_pocket_residue",
        "net_frustration",
    }


def test_counts_and_continuous_descriptors_against_closed_form():
    F = np.array([3.0, 2.0, 0.0, -2.0])
    d = DescriptorInput(
        F=F, labels=classify_index(F), n_pocket_residues=4, energy=np.array([-1.0] * 4)
    )
    assert DESCRIPTORS["count_minimal"](d) == 2.0
    assert DESCRIPTORS["count_neutral"](d) == 1.0
    assert DESCRIPTORS["count_highly"](d) == 1.0
    assert DESCRIPTORS["net_frustration"](d) == 1.0
    assert DESCRIPTORS["mean_Z"](d) == pytest.approx(0.75)
    assert DESCRIPTORS["median_Z"](d) == pytest.approx(1.0)
    # ceil(0.1 * 4) = 1 contact, so the top decile is the single largest F.
    assert DESCRIPTORS["mean_Z_top_decile"](d) == pytest.approx(3.0)
    assert DESCRIPTORS["energy_weighted_sum"](d) == pytest.approx(3.0)
    assert DESCRIPTORS["count_minimal_per_pocket_residue"](d) == pytest.approx(0.5)


def test_energy_weighted_sum_is_nan_without_energies_not_a_silent_fallback():
    F = np.array([1.0, -2.0])
    d = DescriptorInput(F=F, labels=classify_index(F), n_pocket_residues=2, energy=None)
    assert math.isnan(DESCRIPTORS["energy_weighted_sum"](d))


def test_fractions_are_in_unit_interval_and_sum_to_one():
    pairs, nodes, F = toy_system()
    mask = pocket_mask(pairs, nodes, "all")
    row = summarize(pairs, nodes, F, toy_labels(F), mask)
    fractions = [
        row[f"desc__frac_{name}__zscore__default"]
        for name in ("minimal", "highly", "neutral")
    ]
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert sum(fractions) == pytest.approx(1.0)


# ------------------------------------------------------------------------ selectors


def test_selector_all_takes_every_pair():
    pairs, nodes, F = toy_system()
    assert pocket_mask(pairs, nodes, "all").all()


def test_incident_to_ligand_selects_the_protein_ligand_interface_only():
    pairs, nodes, F = toy_system()
    mask = pocket_mask(pairs, nodes, "incident_to_ligand")
    assert list(pairs.loc[mask, "pair_id"]) == [5]


def test_incident_to_uses_the_either_rule_unless_told_otherwise():
    """summarize_ligand_frustration's rule: one partner in the set is enough."""
    pairs, nodes, F = toy_system()
    either = pocket_mask(pairs, nodes, "incident_to", node_ids=["A:1"])
    both = pocket_mask(pairs, nodes, "incident_to", node_ids=["A:1", "A:2"], require_both=True)
    assert list(pairs.loc[either, "pair_id"]) == [0, 1, 3]
    assert list(pairs.loc[both, "pair_id"]) == [0]


def test_inter_chain_selects_cross_chain_pairs():
    pairs, nodes, F = toy_system()
    mask = pocket_mask(pairs, nodes, "inter_chain")
    assert list(pairs.loc[mask, "pair_id"]) == [3, 4, 5]


def test_within_shell_uses_stored_distances_and_excludes_nan():
    pairs, nodes, F = toy_system()
    on_ca = pocket_mask(pairs, nodes, "within_shell", shell_A=7.0, reference="ca")
    # pair 5 has NaN d_ca (a ligand has no Calpha) and must not slip in.
    assert list(pairs.loc[on_ca, "pair_id"]) == [0, 1, 3]
    on_heavy = pocket_mask(pairs, nodes, "within_shell", shell_A=5.0, reference="heavy_min")
    assert list(pairs.loc[on_heavy, "pair_id"]) == [0, 1, 3, 5]


def test_shell_A_changes_the_count_without_touching_decoy_data():
    """R24/U5: the shell is an analyze-time parameter over stored distances.

    F and the labels are computed once, from one decoy ensemble, and reused verbatim; only
    the radius moves. If the count moves with it, the shell is genuinely post-hoc.
    """
    pairs, nodes, F = toy_system()
    labels = toy_labels(F)

    rows = {}
    for shell_A in (4.0, 6.0, 10.0):
        mask = pocket_mask(pairs, nodes, "within_shell", shell_A=shell_A, reference="heavy_min")
        rows[shell_A] = summarize(
            pairs, nodes, F, labels, mask, shell_label=f"heavy{shell_A:g}"
        )

    counts = [rows[s][f"desc__count_minimal__zscore__heavy{s:g}"] for s in (4.0, 6.0, 10.0)]
    totals = [rows[s]["n_contacts_total"] for s in (4.0, 6.0, 10.0)]
    assert totals == [3, 4, 6]
    assert counts[0] < counts[-1]
    # The shell label travels with the number, so two radii cannot be joined as one column.
    assert "desc__count_minimal__zscore__heavy4" in rows[4.0]
    assert "desc__count_minimal__zscore__heavy4" not in rows[10.0]


def test_shell_nodes_returns_partners_of_the_ligand_not_the_ligand():
    pairs, nodes, F = toy_system()
    assert shell_nodes(pairs, nodes, shell_A=4.0, reference="heavy_min") == ["A:3"]
    assert shell_nodes(pairs, nodes, shell_A=1.0, reference="heavy_min") == []


def test_unknown_selector_and_stray_keyword_both_raise():
    pairs, nodes, F = toy_system()
    with pytest.raises(ValueError, match="unknown selector"):
        pocket_mask(pairs, nodes, "near_the_ligand_ish")
    with pytest.raises(TypeError, match="unexpected keyword"):
        pocket_mask(pairs, nodes, "within_shell", shell_A=6.0, shell_ref="heavy_min")
    with pytest.raises(ValueError, match="requires shell_A"):
        pocket_mask(pairs, nodes, "within_shell")


# ----------------------------------------------------------------------- covariates


def test_every_row_carries_all_seven_covariates():
    pairs, nodes, F = toy_system()
    labels = toy_labels(F)
    for selector, kw in (
        ("all", {}),
        ("incident_to_ligand", {}),
        ("inter_chain", {}),
        ("within_shell", {"shell_A": 4.0}),
        ("incident_to", {"node_ids": ["A:1"]}),
    ):
        mask = pocket_mask(pairs, nodes, selector, **kw)
        row = summarize(pairs, nodes, F, labels, mask)
        assert set(COVARIATES) <= set(row), selector


def test_covariates_are_the_graph_statistics_they_claim_to_be():
    pairs, nodes, F = toy_system()
    row = covariates(pairs, nodes, pocket_mask(pairs, nodes, "all"))
    assert row["n_contacts_total"] == 6
    assert row["n_pocket_residues"] == 5
    assert row["mean_residue_degree"] == pytest.approx(2 * 6 / 5)
    assert row["ligand_heavy_atoms"] == 30
    assert row["n_protein_residues"] == 4
    assert row["n_resolved_residues"] == 4
    # rel_sasa x Tien et al. max ASA for ALA/TRP/GLY/LEU; the ligand's NaN drops out.
    expected = 0.5 * 129.0 + 0.25 * 285.0 + 1.0 * 104.0 + 0.1 * 201.0
    assert row["pocket_sasa_A2"] == pytest.approx(expected)


def test_a_count_cannot_be_obtained_without_its_size():
    """R28: there is no argument that suppresses the covariates."""
    pairs, nodes, F = toy_system()
    mask = pocket_mask(pairs, nodes, "all")
    row = summarize(pairs, nodes, F, toy_labels(F), mask)
    assert row["n_contacts_total"] == int(mask.sum())
    assert row["desc__count_minimal__zscore__default"] <= row["n_contacts_total"]


# --------------------------------------------------------------------- degenerate case


def test_empty_mask_gives_zero_counts_and_nan_fractions_not_a_crash():
    pairs, nodes, F = toy_system()
    mask = np.zeros(len(pairs), dtype=bool)
    row = summarize(pairs, nodes, F, toy_labels(F), mask)

    for name in ("count_minimal", "count_highly", "count_neutral", "net_frustration"):
        assert row[f"desc__{name}__zscore__default"] == 0.0
    for name in ("frac_minimal", "frac_highly", "frac_neutral", "mean_Z", "median_Z",
                 "mean_Z_top_decile", "count_minimal_per_pocket_residue"):
        assert math.isnan(row[f"desc__{name}__zscore__default"])
    assert row["n_contacts_total"] == 0
    assert row["n_pocket_residues"] == 0
    assert math.isnan(row["mean_residue_degree"])
    assert math.isnan(row["pocket_sasa_A2"])
    # The whole-system covariates survive an empty pocket; they describe the system.
    assert row["n_protein_residues"] == 4
    assert row["ligand_heavy_atoms"] == 30


def test_mismatched_array_length_raises():
    pairs, nodes, F = toy_system()
    with pytest.raises(ValueError, match="expected"):
        summarize(pairs, nodes, F[:-1], toy_labels(F)[:-1], np.ones(len(pairs), bool))


# -------------------------------------------------------------------- summarize_many


def test_summarize_many_one_row_per_system_with_shared_columns():
    pairs, nodes, F = toy_system()
    frame = summarize_many(
        {
            "sysA": {"pairs": pairs, "nodes": nodes, "F": F},
            "sysB": {"pairs": pairs, "nodes": nodes, "F": F * -1.0},
        },
        selector="all",
        shell_label="all",
    )
    assert list(frame["system_id"]) == ["sysA", "sysB"]
    assert set(COVARIATES) <= set(frame.columns)
    assert frame["desc__count_minimal__zscore__all"].iloc[0] > (
        frame["desc__count_minimal__zscore__all"].iloc[1]
    )
    # Both rows came from one call, so a correlation over this frame cannot be computed
    # against a column that exists for only some systems.
    assert not frame.columns.duplicated().any()
    assert len(frame.columns) == 1 + len(DESCRIPTORS) + len(COVARIATES)


def test_summarize_many_classifies_from_F_when_labels_are_absent():
    pairs, nodes, F = toy_system()
    supplied = summarize_many(
        {"s": {"pairs": pairs, "nodes": nodes, "F": F, "labels": classify_index(F)}},
        selector="all",
    )
    derived = summarize_many({"s": {"pairs": pairs, "nodes": nodes, "F": F}}, selector="all")
    pd.testing.assert_frame_equal(supplied, derived)


def test_summarize_many_with_no_systems_still_has_the_covariate_columns():
    frame = summarize_many({})
    assert frame.empty
    assert set(COVARIATES) <= set(frame.columns)


# =========================================================== acceptance: stored counts


def _parse_clean_pdb(path: Path):
    """Cα coordinates per protein residue in file order, heavy-atom counts, ligand atoms.

    Pose residue k (1-based, as stored in the parquet's ``resi``/``resj``) is the k-th
    protein residue in the cleaned PDB. That identification is not assumed here — it is
    checked by exact contact-set equality below, the same check A2 applied to all 50
    structures it swept (``analysis/diagnose_counts.py:26-28``).
    """
    ca: list[np.ndarray] = []
    n_heavy: list[int] = []
    ligand: list[list[float]] = []
    current = None
    for line in path.read_text().splitlines():
        record, name = line[:6], line[12:16].strip()
        element = line[76:78].strip()
        is_hydrogen = element == "H" or (not element and name[:1] == "H")
        if record == "ATOM  ":
            key = (line[21], line[22:27])
            if key != current:
                current = key
                ca.append(np.full(3, np.nan))
                n_heavy.append(0)
            if not is_hydrogen:
                n_heavy[-1] += 1
            if name == "CA":
                ca[-1] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif record == "HETATM" and line[17:20].strip() != "HOH" and not is_hydrogen:
            ligand.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(ca), np.array(n_heavy), np.array(ligand, dtype=float)


def _prototype_contacts(ca: np.ndarray, cutoff: float = 10.0, seq_sep_min: int = 4):
    """``get_protein_contacts`` (frustration.py:70-108) as array arithmetic."""
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    index = np.arange(len(ca))
    ok = (d <= cutoff) & (np.abs(index[:, None] - index[None, :]) >= seq_sep_min) & np.isfinite(d)
    i, j = np.where(np.triu(ok, k=1))
    return set(zip((i + 1).tolist(), (j + 1).tolist()))


def _prototype_pocket(ca: np.ndarray, ligand: np.ndarray, cutoff: float = 10.0) -> set[int]:
    """``get_ligand_contacts`` (frustration.py:111-158): ligand heavy atom to protein Cα."""
    if ligand.size == 0:
        return set()
    d = np.linalg.norm(ca[:, None, :] - ligand[None, :, :], axis=-1)
    # A residue whose Calpha is missing has no distance at all; the prototype skips it
    # (it requires has("CA")), so it is out of the pocket rather than at distance NaN.
    nearest = np.full(len(ca), np.inf)
    resolved = ~np.isnan(d).all(axis=1)
    nearest[resolved] = np.nanmin(d[resolved], axis=1)
    return set((np.where(nearest <= cutoff)[0] + 1).tolist())


def _as_graph_tables(df: pd.DataFrame, n_heavy: np.ndarray, n_ligand_atoms: int):
    """The prototype's per-contact parquet expressed in the D3 tables.

    Only the columns the aggregation actually reads are filled: node ids, kinds and the
    native energy. Distances are left NaN because the parquet stores none — which is
    precisely why the pocket below has to come from the structure file.
    """
    pairs = pd.DataFrame(
        {
            "pair_id": np.arange(len(df), dtype=np.int32),
            "node_i": [f"A:{int(r)}" for r in df["resi"]],
            "node_j": [f"A:{int(r)}" for r in df["resj"]],
            "i": df["resi"].to_numpy(),
            "j": df["resj"].to_numpy(),
            "kind_i": "protein",
            "kind_j": "protein",
            "same_chain": True,
            "seq_sep": (df["resj"] - df["resi"]).to_numpy(),
            "d_ca": np.nan,
            "d_cb": np.nan,
            "d_heavy_min": np.nan,
            "is_bonded": False,
            "E_native": df["E_native"].to_numpy(),
        }
    )
    protein = pd.DataFrame(
        {
            "node_id": [f"A:{k + 1}" for k in range(len(n_heavy))],
            "kind": "protein",
            "name1": "X",
            "n_heavy": n_heavy,
            "rel_sasa": np.nan,
        }
    )
    ligand = pd.DataFrame(
        {
            "node_id": ["L:LIG:1"],
            "kind": ["ligand"],
            "name1": ["X"],
            "n_heavy": [n_ligand_atoms],
            "rel_sasa": [np.nan],
        }
    )
    return pairs, pd.concat([protein, ligand], ignore_index=True)


def _stored_systems():
    summary = pd.read_csv(SUMMARY_CSV)
    out = []
    for row in summary.itertuples():
        parquet = RESULTS_DIR / f"{row.pdb_id}_{row.ligand_comp_id}_frustration.parquet"
        clean = PROCESSED_DIR / f"{row.pdb_id}_clean.pdb"
        if parquet.exists() and clean.exists():
            out.append((row.pdb_id, row, parquet, clean))
    return out


needs_run_artifacts = pytest.mark.skipif(
    not SUMMARY_CSV.exists() or not PROCESSED_DIR.exists(),
    reason="prototype run artifacts are DVC-tracked; run `dvc pull` to enable",
)


@needs_run_artifacts
def test_stored_summary_counts_are_reproduced_exactly():
    """Every class count in ``egfr_frustration_summary.csv``, recomputed through D3.

    The parquet alone is not enough: it holds the *whole* protein–protein contact set and
    stores no pocket membership, while the summary counts are over the ligand-pocket
    subset. The pocket residue list is recomputed here from ``data/processed/{PDB}_clean.pdb``
    under the prototype's rule, and the pose numbering that links the two is verified by
    exact contact-set equality rather than assumed.

    Given that pocket, ``pocket_mask(..., 'incident_to', node_ids=pocket)`` is the selector
    that reproduces ``summarize_ligand_frustration``: at least one partner in the shell.
    """
    systems = _stored_systems()
    assert systems, "no structure had both a parquet and a cleaned PDB"

    failures = []
    for pdb_id, stored, parquet, clean in systems:
        df = pd.read_parquet(parquet)
        ca, n_heavy, ligand = _parse_clean_pdb(clean)

        if set(zip(df["resi"], df["resj"])) != _prototype_contacts(ca):
            failures.append(f"{pdb_id}: pose-numbering mapping did not verify")
            continue

        pocket = _prototype_pocket(ca, ligand)
        pairs, nodes = _as_graph_tables(df, n_heavy, len(ligand))
        F = df["F_index"].to_numpy()
        labels = classify_index(F)
        mask = pocket_mask(
            pairs, nodes, "incident_to", node_ids=[f"A:{r}" for r in sorted(pocket)]
        )
        row = summarize(pairs, nodes, F, labels, mask, shell_label="ca10")

        got = {
            "n_contacts_total": row["n_contacts_total"],
            "n_minimally_frustrated": int(row["desc__count_minimal__zscore__ca10"]),
            "n_neutral": int(row["desc__count_neutral__zscore__ca10"]),
            "n_highly_frustrated": int(row["desc__count_highly__zscore__ca10"]),
        }
        want = {
            "n_contacts_total": int(stored.n_contacts_total),
            "n_minimally_frustrated": int(stored.n_minimally_frustrated),
            "n_neutral": int(stored.n_neutral),
            "n_highly_frustrated": int(stored.n_highly_frustrated),
        }
        if got != want:
            failures.append(f"{pdb_id}: got {got}, stored {want}")

    assert not failures, "\n".join(failures)


@needs_run_artifacts
def test_stored_frac_minimally_is_reproduced_and_the_labels_agree():
    """``frac_minimally`` and the stored per-contact class, both through D3.

    The stored labels are recomputed from the stored ``F_index`` with
    :func:`classify_index`, which is the D2 refactor's own claim — if this fails, the
    threshold move changed a classification.
    """
    for pdb_id, stored, parquet, clean in _stored_systems():
        df = pd.read_parquet(parquet)
        F = df["F_index"].to_numpy()
        labels = classify_index(F)
        assert list(labels) == list(df["frustration_class"]), pdb_id

        ca, n_heavy, ligand = _parse_clean_pdb(clean)
        pairs, nodes = _as_graph_tables(df, n_heavy, len(ligand))
        pocket = _prototype_pocket(ca, ligand)
        mask = pocket_mask(
            pairs, nodes, "incident_to", node_ids=[f"A:{r}" for r in sorted(pocket)]
        )
        row = summarize(pairs, nodes, F, labels, mask, shell_label="ca10")
        assert row["desc__frac_minimal__zscore__ca10"] == pytest.approx(
            float(stored.frac_minimally), abs=1e-9
        ), pdb_id


@needs_run_artifacts
def test_whole_contact_set_counts_are_reproduced_without_any_structure_file():
    """The parquet-only claim, for the record.

    This one needs nothing but the parquet: over *all* contacts the counts are recomputable
    exactly, and their sum is the row count. The pocket subset is not — the parquet stores
    no pocket membership and no distance to the ligand — which is why the tests above read
    the cleaned PDB.
    """
    for pdb_id, stored, parquet, clean in _stored_systems():
        df = pd.read_parquet(parquet)
        pairs, nodes = _as_graph_tables(df, np.zeros(0, dtype=int), 0)
        F = df["F_index"].to_numpy()
        row = summarize(
            pairs, nodes, F, classify_index(F), pocket_mask(pairs, nodes, "all"),
            shell_label="all",
        )
        counted = sum(
            row[f"desc__count_{name}__zscore__all"] for name in ("minimal", "neutral", "highly")
        )
        assert counted == len(df), pdb_id
        assert row["n_contacts_total"] == len(df), pdb_id
        assert row["desc__count_minimal__zscore__all"] == float(
            (df["frustration_class"] == "minimally_frustrated").sum()
        ), pdb_id
