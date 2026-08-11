"""G7 tests for atomfrust.covalent — PyRosetta-free: text fixtures and synthetic nodes.

The one non-synthetic test reads ``results/preparation_summary.csv`` and asserts how many of
the 61 EGFR complexes are flagged covalent. That number is the reason this module exists, so
it is pinned here rather than left in prose; the test skips when the DVC-tracked file is not
pulled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomfrust.covalent import (
    anchors_from_preparation_summary,
    anchors_from_struct_conn,
    apply_covalent_constraints,
    bonds_for_graph,
    covalent_stratum,
)
from atomfrust.decoys.base import DecoyContext
from atomfrust.decoys.identity import IdentityDecoyGenerator
from atomfrust.graph import Geometry, Node, add_contact_definition, build_superset
from atomfrust.regions import resolve_regions
from atomfrust.settings import ContactSettings, SupersetSettings
from atomfrust.spec import CovalentAnchor, LigandSpec, Receptor, ResidueSelector, SystemSpec

pytestmark = pytest.mark.unit

#: Stage 4 flags 15 of the 61 complexes covalent (all CYS797-SG). A change in this number
#: means the structure set or the _struct_conn parsing moved, and every covalent-stratum
#: statement downstream is about a different cohort.
EXPECTED_COVALENT = 15

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_SUMMARY = REPO_ROOT / "results" / "preparation_summary.csv"


# ------------------------------------------------------------------ fixtures

_CIF_HEADER = """data_TEST
#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_comp_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.pdbx_ptnr1_PDB_ins_code
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.pdbx_ptnr2_PDB_ins_code
_struct_conn.pdbx_dist_value
"""

#: The 5GMP case: an acrylamide C bonded to CYS797 SG, alongside a disulfide that must not
#: be mistaken for a ligand anchor.
CIF_COVALENT = _CIF_HEADER + (
    "disulf1 disulf  A CYS 240 SG ? A CYS 248 SG  ? 2.031\n"
    "covale1 covale A CYS 797 SG ? A F62 1101 CAR ? 1.811\n"
    "#\n"
)

#: A non-covalent complex: the file still has a _struct_conn category, just no covale row.
CIF_NON_COVALENT = _CIF_HEADER + "disulf1 disulf A CYS 240 SG ? A CYS 248 SG ? 2.031\n#\n"

#: The ligand named first rather than second, in another chain, with an insertion code on
#: the receptor side — the orientation and the fields the parser must not assume away.
CIF_REVERSED = _CIF_HEADER + (
    "covale1 covale B 634 1101 C21 ? B CYS 797 SG A 1.795\n#\n"
)


def write_cif(tmp_path: Path, text: str, name: str = "test.cif") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def protein_node(pose_resnum: int, resseq: int, chain: str = "A", **kwargs) -> Node:
    return Node(
        node_id=f"{chain}:{resseq}",
        pose_resnum=pose_resnum,
        kind="protein",
        chain=chain,
        resseq=resseq,
        icode="",
        resname="CYS",
        name1="C",
        **kwargs,
    )


def ligand_node(pose_resnum: int, resseq: int = 1101, chain: str = "A") -> Node:
    return Node(
        node_id=f"{chain}:{resseq}",
        pose_resnum=pose_resnum,
        kind="ligand",
        chain=chain,
        resseq=resseq,
        icode="",
        resname="F62",
        ccd_id="F62",
        rosetta_name="F62",
        mutable=False,
    )


def anchor_spec(anchor: CovalentAnchor | None = None) -> SystemSpec:
    """A 5GMP-shaped spec: one ligand at A:1101, optionally anchored to A:797."""
    return SystemSpec(
        system_id="5GMP_F62",
        receptor=Receptor(pdb_id="5GMP", chains=("A",)),
        ligands=(
            LigandSpec(
                selector=ResidueSelector(chain="A", resseq=1101, comp_id="F62"),
                covalent_anchor=anchor,
            ),
        ),
    )


ANCHOR = CovalentAnchor(chain="A", resseq=797, atom="SG", ligand_atom="CAR")


# ------------------------------------------------------------------- parsing


def test_struct_conn_yields_the_anchor(tmp_path):
    (anchor,) = anchors_from_struct_conn(write_cif(tmp_path, CIF_COVALENT))
    assert (anchor.chain, anchor.resseq, anchor.icode) == ("A", 797, "")
    assert (anchor.atom, anchor.ligand_atom) == ("SG", "CAR")


def test_struct_conn_without_a_covale_row_yields_none(tmp_path):
    assert anchors_from_struct_conn(write_cif(tmp_path, CIF_NON_COVALENT)) == []


def test_struct_conn_without_the_category_yields_none(tmp_path):
    assert anchors_from_struct_conn(write_cif(tmp_path, "data_TEST\n#\n_cell.length_a 1.0\n")) == []


def test_struct_conn_reads_the_ligand_on_either_side(tmp_path):
    """Partner order is a deposition detail; the receptor side is the standard residue."""
    (anchor,) = anchors_from_struct_conn(write_cif(tmp_path, CIF_REVERSED))
    assert (anchor.chain, anchor.resseq, anchor.icode) == ("B", 797, "A")
    assert (anchor.atom, anchor.ligand_atom) == ("SG", "C21")


def test_struct_conn_honours_an_explicit_ligand_comp_id(tmp_path):
    path = write_cif(tmp_path, CIF_COVALENT)
    assert len(anchors_from_struct_conn(path, ligand_comp_ids=["F62"])) == 1
    # A comp id that is not in the file selects nothing, rather than falling back to a guess.
    assert anchors_from_struct_conn(path, ligand_comp_ids=["03P"]) == []


# ------------------------------------------------------- Stage-4 CSV recovery


def summary_csv(tmp_path: Path, **overrides) -> Path:
    row = {
        "pdb_id": "5GMP",
        "status": "OK",
        "egfr_chain": "A",
        "ligand_comp_id": "F62",
        "is_covalent": True,
        "covalent_protein_residue": "CYS797",
        "covalent_protein_atom": "SG",
        "covalent_ligand_atom": "CAR",
    }
    row.update(overrides)
    path = tmp_path / "preparation_summary.csv"
    pd.DataFrame([row, {**row, "pdb_id": "1XKK", "is_covalent": False,
                        "covalent_protein_residue": "",
                        "covalent_protein_atom": "", "covalent_ligand_atom": ""}]).to_csv(
        path, index=False
    )
    return path


def test_preparation_summary_recovers_the_anchor(tmp_path):
    (anchor,) = anchors_from_preparation_summary(summary_csv(tmp_path), "5GMP")
    # The chain is egfr_chain: Stage 4 does not write the linkage's own chain.
    assert (anchor.chain, anchor.resseq, anchor.atom, anchor.ligand_atom) == (
        "A", 797, "SG", "CAR",
    )


def test_preparation_summary_non_covalent_row_yields_none(tmp_path):
    assert anchors_from_preparation_summary(summary_csv(tmp_path), "1XKK") == []


def test_preparation_summary_unknown_pdb_id_raises(tmp_path):
    with pytest.raises(KeyError, match="9ZZZ"):
        anchors_from_preparation_summary(summary_csv(tmp_path), "9ZZZ")


def test_preparation_summary_unreadable_residue_label_raises(tmp_path):
    path = summary_csv(tmp_path, covalent_protein_residue="cysteine")
    with pytest.raises(ValueError, match="covalent_protein_residue"):
        anchors_from_preparation_summary(path, "5GMP")


@pytest.mark.skipif(
    not PREPARATION_SUMMARY.exists(), reason="results/ is DVC-tracked; run `dvc pull`"
)
def test_real_preparation_summary_covalent_count():
    table = pd.read_csv(PREPARATION_SUMMARY)
    assert len(table) == 61
    assert int(table["is_covalent"].sum()) == EXPECTED_COVALENT

    recovered = [
        anchor
        for pdb_id in table["pdb_id"]
        for anchor in anchors_from_preparation_summary(PREPARATION_SUMMARY, pdb_id)
    ]
    assert len(recovered) == EXPECTED_COVALENT
    # Every one is the same warhead chemistry: CYS797 SG. If that stops holding, the
    # "covalent" stratum has stopped being one thing.
    assert {(a.resseq, a.atom) for a in recovered} == {(797, "SG")}


# --------------------------------------------------------------- constraints


def test_apply_covalent_constraints_marks_exactly_the_anchor():
    nodes = [protein_node(1, 796), protein_node(2, 797), protein_node(3, 798), ligand_node(4)]
    out = apply_covalent_constraints(nodes, [ANCHOR])

    frozen = [n for n in out if n.frozen_reason == "covalent_anchor"]
    assert [n.node_id for n in frozen] == ["A:797"]
    assert frozen[0].mutable is False
    assert all(n.mutable for n in out if n.kind == "protein" and n.node_id != "A:797")
    assert all(n.frozen_reason is None for n in out if n.node_id != "A:797")
    # Node is frozen; the caller's list must be untouched.
    assert all(n.frozen_reason is None for n in nodes)


def test_apply_covalent_constraints_re_enforces_an_existing_marker():
    """load_complex tags the anchor itself; a tagged-but-mutable node is the silent failure."""
    nodes = [protein_node(1, 797, mutable=True, frozen_reason="covalent_anchor")]
    (out,) = apply_covalent_constraints(nodes)
    assert out.mutable is False
    assert out.frozen_reason == "covalent_anchor"


def test_apply_covalent_constraints_is_idempotent():
    nodes = [protein_node(1, 797), protein_node(2, 798)]
    once = apply_covalent_constraints(nodes, [ANCHOR])
    assert apply_covalent_constraints(once, [ANCHOR]) == once


def fake_pose(nodes):
    """Enough pose for native_aa_frequency: it reads is_protein() and name1() only."""

    class _Residue:
        def __init__(self, node):
            self._node = node

        def is_protein(self):
            return self._node.kind == "protein"

        def name1(self):
            return self._node.name1

    class _Pose:
        def total_residue(self):
            return len(nodes)

        def residue(self, i):
            return _Residue(nodes[i - 1])

    return _Pose()


def test_identity_generator_never_targets_the_anchor():
    nodes = apply_covalent_constraints(
        [protein_node(1, 796), protein_node(2, 797), protein_node(3, 798), ligand_node(4)],
        [ANCHOR],
    )
    # Regions come from the real resolver, whose mutable_only intersection is the path that
    # carries the freeze into a decoy protocol.
    geom = Geometry(
        heavy_xyz=np.arange(12, dtype=float).reshape(4, 3),
        heavy_node=np.arange(4),
        ca_xyz=np.arange(12, dtype=float).reshape(4, 3),
        cb_xyz=np.full((4, 3), np.nan),
    )
    regions = resolve_regions(nodes, geom)
    context = DecoyContext(
        pose=fake_pose(nodes),
        nodes=nodes,
        pairs=pd.DataFrame({"pair_id": np.array([], dtype=np.int32)}),
        regions=regions,
    )

    for scope in ("whole_protein", "contact_shell"):
        positions = IdentityDecoyGenerator(context=context, scope=scope).target_positions()
        assert positions == [1, 3], scope  # pose resnums; 2 is the anchor, 4 the ligand


# --------------------------------------------------------------------- graph


def geometry_two_nodes(separation: float) -> Geometry:
    """Two nodes ``separation`` apart — far beyond any cutoff when it is large."""
    ca = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    return Geometry(
        heavy_xyz=ca.copy(),
        heavy_node=np.array([0, 1]),
        ca_xyz=np.array([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]]),
        cb_xyz=np.full((2, 3), np.nan),
    )


def test_bonds_for_graph_pairs_the_anchor_with_its_ligand():
    nodes = [protein_node(1, 797), ligand_node(2)]
    assert bonds_for_graph(anchor_spec(ANCHOR), nodes) == (("A:797", "A:1101"),)


def test_bonds_for_graph_is_empty_without_an_anchor():
    nodes = [protein_node(1, 797), ligand_node(2)]
    assert bonds_for_graph(anchor_spec(None), nodes) == ()


def test_bonds_for_graph_names_a_missing_anchor_node():
    nodes = [protein_node(1, 796), ligand_node(2)]
    with pytest.raises(ValueError, match="A:797"):
        bonds_for_graph(anchor_spec(ANCHOR), nodes)


def test_forced_bond_survives_distance_and_sequence_separation():
    """The whole point of forcing: the edge is chemistry, not a consequence of the cutoff."""
    nodes = [protein_node(1, 797), ligand_node(2)]
    bonds = bonds_for_graph(anchor_spec(ANCHOR), nodes)

    # 40 A apart, and a superset seq-sep filter that admits nothing adjacent.
    pairs = build_superset(
        nodes, geometry_two_nodes(40.0), SupersetSettings(seq_sep_min=4), bonds
    )
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert (row["node_i"], row["node_j"]) == ("A:797", "A:1101")
    assert bool(row["is_bonded"]) is True
    assert row["d_heavy_min"] == pytest.approx(40.0)

    # And it is a contact under a definition whose cutoffs it violates by 34 A.
    marked = add_contact_definition(
        pairs,
        "strict",
        ContactSettings(definition="ca_ca", cutoff_A=6.0, ligand_cutoff_A=6.0, seq_sep_min=8),
        SupersetSettings(seq_sep_min=4),
    )
    assert bool(marked["in__strict"].iloc[0]) is True


def test_forced_bond_exempts_a_protein_pair_from_sequence_separation():
    """The seq-sep exemption in build_superset, on a pair where seq_sep is defined at all.

    A protein-ligand anchor carries ``seq_sep == -1`` (separation is undefined across kinds),
    so the exemption above is never exercised by the anchor itself. It is the mechanism
    bonds_for_graph relies on, so it is checked where it can be seen.
    """
    nodes = [protein_node(1, 797), protein_node(2, 798)]
    geom = geometry_two_nodes(3.0)
    geom = Geometry(
        heavy_xyz=geom.heavy_xyz,
        heavy_node=geom.heavy_node,
        ca_xyz=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        cb_xyz=geom.cb_xyz,
    )
    unforced = build_superset(nodes, geom, SupersetSettings(seq_sep_min=4))
    assert len(unforced) == 0

    forced = build_superset(
        nodes, geom, SupersetSettings(seq_sep_min=4), (("A:797", "A:798"),)
    )
    assert len(forced) == 1
    assert bool(forced["is_bonded"].iloc[0]) is True


# ----------------------------------------------------------------- reporting


def test_covalent_stratum_splits_on_the_spec():
    table = pd.DataFrame({"system_id": ["5GMP_F62", "1XKK_FMM"], "frac_minimal": [0.5, 0.4]})
    spec_by_id = {"5GMP_F62": anchor_spec(ANCHOR), "1XKK_FMM": anchor_spec(None)}
    stratum = covalent_stratum(table, spec_by_id)
    assert list(stratum) == [True, False]
    assert stratum.name == "is_covalent"
    assert stratum.index.equals(table.index)


def test_covalent_stratum_falls_back_to_labels_then_to_the_table():
    labelled = anchor_spec(None).model_copy(update={"labels": {"is_covalent": "true"}})
    table = pd.DataFrame(
        {"system_id": ["labelled", "from_table", "unknown"], "is_covalent": [False, "True", ""]}
    )
    stratum = covalent_stratum(table, {"labelled": labelled})
    # The spec wins over the table's False; the unspecced rows fall back to the column.
    assert list(stratum) == [True, True, False]


def test_covalent_stratum_requires_system_id():
    with pytest.raises(KeyError, match="system_id"):
        covalent_stratum(pd.DataFrame({"pdb_id": ["5GMP"]}), {})
