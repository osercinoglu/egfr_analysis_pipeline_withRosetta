"""B7 tests for atomfrust.graph — PyRosetta-free, synthetic geometry."""

from __future__ import annotations

import numpy as np
import pytest

from atomfrust.graph import (
    Geometry,
    Node,
    _candidate_pairs_bruteforce,
    _candidate_pairs_kdtree,
    add_contact_definition,
    build_graph,
    build_superset,
    nodes_to_frame,
)
from atomfrust.settings import ContactSettings, Settings, SupersetSettings

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ builders


def protein_node(index, chain="A", resseq=None, kind="protein"):
    resseq = index if resseq is None else resseq
    return Node(
        node_id=f"{chain}:{resseq}",
        pose_resnum=index,
        kind=kind,
        chain=chain,
        resseq=resseq,
        icode="",
        resname="ALA",
        name1="A",
    )


def ligand_node(index, chain="A", resseq=1101, resname="LIG", kind="ligand"):
    return Node(
        node_id=f"{chain}:{resseq}",
        pose_resnum=index,
        kind=kind,
        chain=chain,
        resseq=resseq,
        icode="",
        resname=resname,
        ccd_id=resname,
        rosetta_name=resname,
        mutable=False,
    )


def geometry_from(ca_list, heavy_map=None, cb_list=None):
    """ca_list: per node Cα (or None). heavy_map: node index -> array of heavy coords."""
    n = len(ca_list)
    ca = np.array([c if c is not None else [np.nan] * 3 for c in ca_list], dtype=float)
    cb = (
        np.array([c if c is not None else [np.nan] * 3 for c in cb_list], dtype=float)
        if cb_list is not None
        else np.full((n, 3), np.nan)
    )
    xyz, owner = [], []
    for k in range(n):
        atoms = (heavy_map or {}).get(k)
        if atoms is None:
            atoms = [ca_list[k]] if ca_list[k] is not None else []
        for a in atoms:
            xyz.append(a)
            owner.append(k)
    return Geometry(
        heavy_xyz=np.array(xyz, dtype=float).reshape(-1, 3),
        heavy_node=np.array(owner, dtype=np.int64),
        ca_xyz=ca,
        cb_xyz=cb,
    )


# ------------------------------------------ acceptance: KD-tree == brute force


@pytest.mark.parametrize("seed", range(6))
def test_kdtree_candidates_match_bruteforce_exactly(seed):
    rng = np.random.default_rng(seed)
    n = 40
    ca = rng.uniform(0, 25, size=(n, 3))
    # A quarter of the nodes are ligand-like: no Ca, several heavy atoms.
    ca_list = [None if k % 4 == 3 else ca[k] for k in range(n)]
    heavy = {
        k: ca[k] + rng.uniform(-1.5, 1.5, size=(rng.integers(1, 6), 3)) for k in range(n)
    }
    geom = geometry_from(ca_list, heavy)

    for ca_cut, heavy_cut in ((12.0, 8.0), (6.0, 4.0), (20.0, 15.0)):
        assert _candidate_pairs_kdtree(geom, ca_cut, heavy_cut) == \
            _candidate_pairs_bruteforce(geom, ca_cut, heavy_cut)


def test_kdtree_handles_degenerate_inputs():
    empty = Geometry(
        heavy_xyz=np.zeros((0, 3)),
        heavy_node=np.zeros((0,), dtype=np.int64),
        ca_xyz=np.zeros((0, 3)),
        cb_xyz=np.zeros((0, 3)),
    )
    assert _candidate_pairs_kdtree(empty, 12.0, 8.0) == set()

    single = geometry_from([[0.0, 0.0, 0.0]])
    assert _candidate_pairs_kdtree(single, 12.0, 8.0) == set()


# ------------------------ acceptance: cross-chain pairs are not sequence-separated


def test_sequence_separation_is_per_chain():
    """The prototype separated by pose index (frustration.py:98), which is meaningless
    across a chain boundary. Residues A:10 and B:12 are not sequence neighbours."""
    nodes = [
        protein_node(1, chain="A", resseq=10),
        protein_node(2, chain="B", resseq=12),
        protein_node(3, chain="A", resseq=12),
    ]
    geom = geometry_from([[0, 0, 0], [2, 0, 0], [4, 0, 0]])

    pairs = build_superset(nodes, geom, SupersetSettings(seq_sep_min=4))
    got = {(r.node_i, r.node_j) for r in pairs.itertuples()}

    assert ("A:10", "B:12") in got, "cross-chain pair was wrongly sequence-separated"
    assert ("A:10", "A:12") not in got, "same-chain |i-j|=2 should be excluded at seq_sep_min=4"

    cross = pairs[(pairs.node_i == "A:10") & (pairs.node_j == "B:12")].iloc[0]
    assert cross.seq_sep == -1 and not cross.same_chain


def test_seq_sep_min_one_admits_everything():
    nodes = [protein_node(1, resseq=10), protein_node(2, resseq=11)]
    geom = geometry_from([[0, 0, 0], [2, 0, 0]])
    assert len(build_superset(nodes, geom, SupersetSettings(seq_sep_min=1))) == 1


# ------------------- acceptance: ligand beyond the contact cutoff, inside the superset


def test_ligand_outside_contact_cutoff_still_appears_in_the_superset():
    """A ligand 6.1 A away is in the superset (heavy cutoff 8) but is not a contact under
    a 6.0 A ligand rule. That headroom is what makes a wider shell re-selectable later."""
    nodes = [protein_node(1, resseq=10), ligand_node(2)]
    geom = geometry_from([[0.0, 0.0, 0.0], None], {0: [[0, 0, 0]], 1: [[6.1, 0, 0]]})

    pairs = build_superset(nodes, geom, SupersetSettings(heavy_cutoff_A=8.0))
    assert len(pairs) == 1
    assert pairs.iloc[0].d_heavy_min == pytest.approx(6.1, abs=1e-4)

    tight = add_contact_definition(
        pairs, "ca_ca", ContactSettings(ligand_cutoff_A=6.0), SupersetSettings()
    )
    assert not tight["in__ca_ca"].iloc[0]

    wide = add_contact_definition(
        pairs, "ca_ca", ContactSettings(ligand_cutoff_A=7.0), SupersetSettings()
    )
    assert wide["in__ca_ca"].iloc[0]


# ------------------------ THE A4 correction: a Ca rule must not exclude the ligand


def test_ca_ca_definition_still_includes_ligand_contacts():
    """A ligand has no Ca. If 'ca_ca' judged it by a Ca rule it would be excluded by
    construction, silently restoring the protein-only behaviour A4 identified."""
    nodes = [protein_node(1, resseq=10), protein_node(2, resseq=20), ligand_node(3)]
    geom = geometry_from(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], None],
        {0: [[0, 0, 0]], 1: [[5, 0, 0]], 2: [[2.5, 1.0, 0.0]]},
    )
    _, pairs = build_graph(nodes, geom, Settings())

    ligand_pairs = pairs[(pairs.kind_i == "ligand") | (pairs.kind_j == "ligand")]
    assert len(ligand_pairs) == 2
    assert ligand_pairs["in__ca_ca"].all(), "ligand contacts dropped by a Ca-Ca definition"


def test_ligand_pairs_are_judged_by_heavy_distance_not_nan_ca():
    nodes = [protein_node(1, resseq=10), ligand_node(2)]
    geom = geometry_from([[0.0, 0.0, 0.0], None], {0: [[0, 0, 0]], 1: [[3.0, 0, 0]]})
    pairs = build_superset(nodes, geom)
    assert np.isnan(pairs.iloc[0].d_ca)  # no Ca on a ligand
    out = add_contact_definition(pairs, "ca_ca", ContactSettings())
    assert out["in__ca_ca"].iloc[0]


def test_sequence_separation_never_filters_ligand_pairs():
    """seq_sep is -1 for a ligand pair; a seq-sep filter must not silently drop it."""
    nodes = [protein_node(1, resseq=10), ligand_node(2, resseq=11)]
    geom = geometry_from([[0.0, 0.0, 0.0], None], {0: [[0, 0, 0]], 1: [[3.0, 0, 0]]})
    pairs = build_superset(nodes, geom, SupersetSettings(seq_sep_min=4))
    assert len(pairs) == 1
    out = add_contact_definition(pairs, "d", ContactSettings(seq_sep_min=4))
    assert out["in__d"].iloc[0]


# --------------------------------------- acceptance: narrowing a cutoff is a filter


def test_narrowing_a_cutoff_is_a_pure_column_filter():
    rng = np.random.default_rng(0)
    nodes = [protein_node(k + 1, resseq=k + 1) for k in range(30)]
    ca = rng.uniform(0, 20, size=(30, 3))
    geom = geometry_from([c for c in ca])

    pairs = build_superset(nodes, geom, SupersetSettings(ca_cutoff_A=12.0))
    wide = add_contact_definition(pairs, "w", ContactSettings(cutoff_A=10.0))
    narrow = add_contact_definition(pairs, "n", ContactSettings(cutoff_A=8.0))

    # Same rows, same order, same distances — only the boolean column differs.
    assert wide.pair_id.tolist() == narrow.pair_id.tolist()
    assert np.allclose(wide.d_ca, narrow.d_ca, equal_nan=True)
    assert narrow["in__n"].sum() < wide["in__w"].sum()
    assert (narrow["in__n"] <= wide["in__w"]).all(), "narrowing must only remove contacts"


def test_cutoff_beyond_the_superset_is_a_hard_error():
    nodes = [protein_node(1, resseq=1), protein_node(2, resseq=9)]
    geom = geometry_from([[0, 0, 0], [5, 0, 0]])
    pairs = build_superset(nodes, geom, SupersetSettings(ca_cutoff_A=12.0))

    with pytest.raises(ValueError, match="requires regeneration"):
        add_contact_definition(pairs, "x", ContactSettings(cutoff_A=15.0))
    with pytest.raises(ValueError, match="requires regeneration"):
        add_contact_definition(pairs, "x", ContactSettings(ligand_cutoff_A=20.0))


# ---------------------------------------------------------------------- bonds


def test_forced_bond_is_a_contact_regardless_of_distance():
    nodes = [protein_node(1, resseq=797), ligand_node(2, resseq=1101)]
    geom = geometry_from([[0.0, 0.0, 0.0], None], {0: [[0, 0, 0]], 1: [[50.0, 0, 0]]})

    pairs = build_superset(nodes, geom, bonds=(("A:797", "A:1101"),))
    assert len(pairs) == 1 and pairs.iloc[0].is_bonded

    out = add_contact_definition(pairs, "d", ContactSettings())
    assert out["in__d"].iloc[0], "a covalent bond must be a contact at any distance"


def test_bond_to_an_unknown_node_raises():
    nodes = [protein_node(1, resseq=1)]
    geom = geometry_from([[0, 0, 0]])
    with pytest.raises(KeyError, match="unknown node"):
        build_superset(nodes, geom, bonds=(("A:1", "Z:999"),))


def test_bond_survives_a_sequence_separation_filter():
    nodes = [protein_node(1, resseq=10), protein_node(2, resseq=11)]
    geom = geometry_from([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pairs = build_superset(nodes, geom, SupersetSettings(seq_sep_min=4),
                           bonds=(("A:10", "A:11"),))
    assert len(pairs) == 1 and pairs.iloc[0].is_bonded


# --------------------------------------------------------------------- tables


def test_pair_ids_are_dense_sorted_and_stable():
    rng = np.random.default_rng(3)
    nodes = [protein_node(k + 1, resseq=k + 1) for k in range(15)]
    geom = geometry_from([c for c in rng.uniform(0, 15, size=(15, 3))])
    a = build_superset(nodes, geom)
    b = build_superset(nodes, geom)
    assert a.pair_id.tolist() == list(range(len(a)))
    assert a.equals(b), "graph construction must be deterministic"


def test_node_frame_schema_and_empty_case():
    df = nodes_to_frame([protein_node(1), ligand_node(2)])
    assert list(df.node_id) == ["A:1", "A:1101"]
    assert df.kind.tolist() == ["protein", "ligand"]
    assert not df.mutable.iloc[1], "a ligand is not identity-randomisable"
    assert nodes_to_frame([]).empty


def test_geometry_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="atom count"):
        Geometry(
            heavy_xyz=np.zeros((3, 3)),
            heavy_node=np.zeros((2,), dtype=np.int64),
            ca_xyz=np.zeros((1, 3)),
            cb_xyz=np.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="node count"):
        Geometry(
            heavy_xyz=np.zeros((1, 3)),
            heavy_node=np.zeros((1,), dtype=np.int64),
            ca_xyz=np.zeros((2, 3)),
            cb_xyz=np.zeros((1, 3)),
        )


def test_node_count_mismatch_is_caught():
    with pytest.raises(ValueError, match="node list"):
        build_superset([protein_node(1)], geometry_from([[0, 0, 0], [1, 0, 0]]))


def test_cb_cb_definition_skips_residues_without_cb():
    """Glycine has no Cβ. NaN must read as 'not a contact', never as a crash."""
    nodes = [protein_node(1, resseq=1), protein_node(2, resseq=9)]
    geom = geometry_from(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], cb_list=[[0.5, 0, 0], None]
    )
    pairs = build_superset(nodes, geom)
    out = add_contact_definition(pairs, "cb", ContactSettings(definition="cb_cb"))
    assert not out["in__cb"].iloc[0]
