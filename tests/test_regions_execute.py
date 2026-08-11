"""B9/B10 tests — selector language and execution. No PyRosetta in the unit tier."""

from __future__ import annotations

import numpy as np
import pytest

from atomfrust.execute import ShardSpec, WorkUnit, execute, plan_units, run_serial
from atomfrust.graph import Geometry, Node
from atomfrust.regions import SelectorError, ResolvedRegions, resolve_regions, select

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ fixtures


def make_nodes_and_geometry():
    """Ten protein residues on a line in chain A, three in chain B, one ligand and a zinc."""
    nodes, ca, heavy, owner = [], [], [], []

    def add(node, ca_xyz, atoms):
        index = len(nodes)
        nodes.append(node)
        ca.append(ca_xyz if ca_xyz is not None else [np.nan] * 3)
        for atom in atoms:
            heavy.append(atom)
            owner.append(index)

    for k in range(10):
        add(
            Node(f"A:{k + 1}", k + 1, "protein", "A", k + 1, "", "ALA", "A"),
            [k * 3.0, 0.0, 0.0],
            [[k * 3.0, 0.0, 0.0], [k * 3.0, 1.0, 0.0]],
        )
    for k in range(3):
        add(
            Node(f"B:{k + 1}", 11 + k, "protein", "B", k + 1, "", "GLY", "G"),
            [k * 3.0, 30.0, 0.0],
            [[k * 3.0, 30.0, 0.0]],
        )
    add(
        Node("A:900", 14, "ligand", "A", 900, "", "LIG", "X", mutable=False),
        None,
        [[12.0, 2.0, 0.0], [13.0, 2.0, 0.0]],
    )
    add(
        Node("A:901", 15, "metal", "A", 901, "", "ZN", "X", mutable=False),
        None,
        [[27.0, 0.0, 0.0]],
    )

    geom = Geometry(
        heavy_xyz=np.array(heavy, dtype=float),
        heavy_node=np.array(owner, dtype=np.int64),
        ca_xyz=np.array(ca, dtype=float),
        cb_xyz=np.full((len(nodes), 3), np.nan),
    )
    return nodes, geom


@pytest.fixture
def system():
    return make_nodes_and_geometry()


def ids(nodes, mask):
    return [n.node_id for n, keep in zip(nodes, mask) if keep]


# ============================================================ B9: selectors


def test_basic_kind_selectors(system):
    nodes, geom = system
    assert select("all", nodes, geom).sum() == len(nodes)
    assert select("none", nodes, geom).sum() == 0
    assert select("protein", nodes, geom).sum() == 13
    # "ligand" covers every non-protein component, metals included.
    assert ids(nodes, select("ligand", nodes, geom)) == ["A:900", "A:901"]


def test_chain_and_resn_and_resi(system):
    nodes, geom = system
    assert len(ids(nodes, select("chain B", nodes, geom))) == 3
    assert ids(nodes, select("resn GLY", nodes, geom)) == ["B:1", "B:2", "B:3"]
    assert ids(nodes, select("chain A and resi 2-4", nodes, geom)) == ["A:2", "A:3", "A:4"]
    assert ids(nodes, select("chain A and resi 1,3,5", nodes, geom)) == ["A:1", "A:3", "A:5"]
    assert len(ids(nodes, select("chain A,B and resi 1", nodes, geom))) == 2


def test_boolean_composition_and_precedence(system):
    nodes, geom = system
    # `and` binds tighter than `or`.
    left = select("chain B or chain A and resi 1", nodes, geom)
    right = select("chain B or (chain A and resi 1)", nodes, geom)
    assert np.array_equal(left, right)
    assert ids(nodes, left) == ["A:1", "B:1", "B:2", "B:3"]

    assert np.array_equal(
        select("not protein", nodes, geom), select("ligand", nodes, geom)
    )
    assert select("protein and not chain B", nodes, geom).sum() == 10


def test_within_selects_a_real_shell(system):
    nodes, geom = system
    # The ligand sits at x = 12-13, y = 2; A:5 (x=12) and A:4 (x=9) are the near residues.
    shell = ids(nodes, select("protein and within(3.0, ligand)", nodes, geom))
    assert "A:5" in shell
    assert "B:1" not in shell

    wide = select("protein and within(30.0, ligand)", nodes, geom).sum()
    narrow = select("protein and within(3.0, ligand)", nodes, geom).sum()
    assert wide > narrow > 0


def test_within_is_monotone_in_radius(system):
    nodes, geom = system
    previous = -1
    for radius in (1.0, 3.0, 6.0, 12.0, 40.0):
        count = int(select(f"within({radius}, ligand)", nodes, geom).sum())
        assert count >= previous
        previous = count


def test_within_an_empty_set_is_empty(system):
    nodes, geom = system
    assert select("within(10.0, none)", nodes, geom).sum() == 0


def test_layer_partitions_the_protein(system):
    nodes, geom = system
    core = select("layer(core)", nodes, geom)
    boundary = select("layer(boundary)", nodes, geom)
    surface = select("layer(surface)", nodes, geom)
    # Every node lands in exactly one layer or none (ligands have no centre).
    assert not (core & boundary).any()
    assert not (core & surface).any()
    assert not (boundary & surface).any()


@pytest.mark.parametrize(
    "expression",
    ["", "   ", "chain", "resi", "nonsense", "chain A and", "within(3.0 ligand)",
     "layer(middle)", "resi 20-10", "protein)", "resi 1.5"],
)
def test_malformed_expressions_raise(system, expression):
    nodes, geom = system
    with pytest.raises(SelectorError):
        select(expression, nodes, geom)


def test_xml_escape_hatch_reports_that_it_is_unimplemented(system):
    nodes, geom = system
    with pytest.raises(SelectorError, match="not implemented"):
        select("xml:file#name", nodes, geom)


# ====================================================== B9: resolved regions


def test_regions_resolve_and_frozen_is_the_complement(system):
    nodes, geom = system
    regions = resolve_regions(
        nodes, geom,
        mutate="protein and within(6.0, ligand)",
        repack="protein and within(12.0, ligand)",
        minimize="protein and within(6.0, ligand)",
    )
    counts = regions.counts()
    assert 0 < counts["mutate"] <= counts["repack"]
    assert counts["frozen"] == len(nodes) - counts["repack"]
    assert np.array_equal(regions.frozen, ~(regions.repack | regions.minimize))


def test_mutate_not_a_subset_of_repack_raises(system):
    """A mutated residue that is not repacked keeps the old identity's rotamer."""
    nodes, geom = system
    with pytest.raises(SelectorError, match="mutate is not a subset of repack"):
        resolve_regions(nodes, geom, mutate="protein", repack="chain B", minimize="none")


def test_minimize_not_a_subset_of_repack_raises(system):
    nodes, geom = system
    with pytest.raises(SelectorError, match="minimize is not a subset of repack"):
        resolve_regions(nodes, geom, mutate="none", repack="chain B", minimize="protein")


def test_an_empty_mutate_set_is_legal(system):
    """A repack-only decoy is a meaningful control: same sequence, new side-chain packing."""
    nodes, geom = system
    regions = resolve_regions(nodes, geom, mutate="none", repack="protein", minimize="none")
    assert regions.counts()["mutate"] == 0
    assert regions.counts()["repack"] == 13


def test_non_mutable_nodes_are_never_mutated(system):
    """A ligand, a metal or a covalent anchor must not be identity-randomised even when an
    expression names it. `protein` would not, but `all` would."""
    nodes, geom = system
    regions = resolve_regions(nodes, geom, mutate="all", repack="all", minimize="none")
    mutated = ids(nodes, regions.mutate)
    assert "A:900" not in mutated and "A:901" not in mutated
    assert len(mutated) == 13


def test_pose_resnums_round_trip(system):
    nodes, geom = system
    regions = resolve_regions(nodes, geom, mutate="chain B", repack="protein", minimize="none")
    assert regions.pose_resnums(nodes, "mutate") == [11, 12, 13]
    assert len(regions.pose_resnums(nodes, "frozen")) == regions.counts()["frozen"]


# ============================================================ B10: sharding


def test_shard_parse_and_validation():
    assert ShardSpec.parse(None) == ShardSpec(0, 1)
    assert ShardSpec.parse("2/4") == ShardSpec(2, 4)
    for bad in ("1", "a/b", "4/4", "-1/4", "1/0"):
        with pytest.raises(ValueError):
            ShardSpec.parse(bad)


def test_shards_partition_the_work_exactly():
    """Disjoint and complete, with no coordination between shards."""
    full = plan_units(["S"], n_decoys=100)
    assert len(full) == 100

    pieces = [plan_units(["S"], 100, shard=ShardSpec(k, 4)) for k in range(4)]
    assert sum(len(p) for p in pieces) == 100
    assert sorted(u.decoy_id for piece in pieces for u in piece) == list(range(100))
    for a in range(4):
        for b in range(a + 1, 4):
            assert not set(pieces[a]) & set(pieces[b])


def test_planning_is_deterministic_and_ordered():
    a = plan_units(["S1", "S2"], 5, axes=("identity", "chemotype"))
    b = plan_units(["S1", "S2"], 5, axes=("identity", "chemotype"))
    assert a == b
    assert len(a) == 2 * 2 * 5
    assert a[0] == WorkUnit("S1", "identity", 0)


def test_resume_subtracts_completed_work():
    completed = {("S", "identity"): {0, 1, 2, 7}}
    units = plan_units(["S"], 10, completed=completed)
    assert [u.decoy_id for u in units] == [3, 4, 5, 6, 8, 9]


def test_a_larger_n_decoys_extends_rather_than_restarting():
    """The prototype short-circuited on an existing parquet, so raising --n_decoys did
    nothing (run_pipeline.py:241). Here it plans exactly the new decoys."""
    completed = {("S", "identity"): set(range(50))}
    units = plan_units(["S"], 200, completed=completed)
    assert [u.decoy_id for u in units] == list(range(50, 200))


def test_resume_composes_with_sharding():
    completed = {("S", "identity"): {0, 4, 8}}
    pieces = [plan_units(["S"], 12, completed=completed, shard=ShardSpec(k, 4)) for k in range(4)]
    got = sorted(u.decoy_id for piece in pieces for u in piece)
    assert got == [1, 2, 3, 5, 6, 7, 9, 10, 11]


def test_seed_is_base_plus_decoy_id():
    assert WorkUnit("S", "identity", 7).seed(42) == 49


# ======================================================== B10: execution


def _double_factory(offset: int):
    return lambda unit: unit.decoy_id * 2 + offset


def test_serial_and_parallel_agree_exactly():
    """Results must not depend on worker count — the property that lets a run be sharded
    across machines and still be one ensemble."""
    units = plan_units(["S"], 24)
    serial = dict(run_serial(units, _double_factory(3)))

    for workers in (1, 2, 4, 8):
        collected: dict[WorkUnit, int] = {}
        completed = execute(
            units,
            _double_factory,
            task_args=(3,),
            workers=workers,
            on_result=lambda unit, value: collected.__setitem__(unit, value),
        )
        assert completed == len(units)
        assert collected == serial


def test_sharded_execution_reassembles_into_the_unsharded_result():
    units = plan_units(["S"], 20)
    reference = dict(run_serial(units, _double_factory(0)))

    merged: dict[WorkUnit, int] = {}
    for k in range(3):
        piece = plan_units(["S"], 20, shard=ShardSpec(k, 3))
        execute(
            piece,
            _double_factory,
            task_args=(0,),
            workers=2,
            on_result=lambda unit, value: merged.__setitem__(unit, value),
        )
    assert merged == reference


def test_executing_nothing_is_not_an_error():
    assert execute([], _double_factory, task_args=(0,), workers=4) == 0


# ============================ B9 acceptance: agreement with the prototype


@pytest.mark.integration
def test_within_ca_reproduces_the_prototype_ligand_contacts():
    """`within_ca(10.0, ligand)` must return exactly the residue set the prototype's
    get_ligand_contacts (frustration.py:111) returns at the same cutoff, and the
    heavy-atom `within` must be a strict superset of it."""
    from pathlib import Path

    import sys

    processed = Path("data/processed/5GMP_clean.pdb")
    params = Path("data/ligands/params/F62.params")
    if not (processed.exists() and params.exists()):
        pytest.skip("data/ not present (dvc pull)")

    sys.path.insert(0, "src")
    import frustration as legacy

    from atomfrust.pose import load_complex
    from atomfrust.spec import LigandSpec, SystemSpec

    spec = SystemSpec.from_pdb(processed, system_id="5GMP")
    spec = spec.model_copy(
        update={"ligands": (LigandSpec(selector=spec.ligands[0].selector, params=params),)}
    )
    loaded = load_complex(spec)

    ligand_resnum = loaded.components[0].pose_resnum
    expected = set(legacy.get_ligand_contacts(loaded.pose, ligand_resnum, cutoff=10.0))

    mask = select("protein and within_ca(10.0, ligand)", loaded.nodes, loaded.geometry)
    got = {n.pose_resnum for n, keep in zip(loaded.nodes, mask) if keep}
    assert got == expected, (
        f"differs from the prototype by {len(got ^ expected)} residues"
    )

    heavy = select("protein and within(10.0, ligand)", loaded.nodes, loaded.geometry)
    heavy_set = {n.pose_resnum for n, keep in zip(loaded.nodes, heavy) if keep}
    assert expected < heavy_set, "heavy-atom within must strictly contain the Ca rule"
