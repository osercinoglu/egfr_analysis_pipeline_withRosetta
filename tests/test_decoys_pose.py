"""G5 tests — the pose axis (axis B).

The properties that make this axis the *easy* ligand axis, each asserted rather than assumed:
the molecule is unchanged so RMSD to native is well defined and thresholdable; the protein
backbone is untouched; a seed reproduces a placement exactly; the energy that moves is the
energy the ligand touches; the S4.2 validity verdict is on the record; and running out of
attempts is a status, not a hang.

Kept to <=3 decoys throughout — a decoy costs ~0.7 s on 5GMP and none of these properties
needs an ensemble to show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROCESSED = Path("data/processed")
PARAMS = Path("data/ligands/params")


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")


def _context(pdb_stem: str, params_name: str):
    """A DecoyContext over the prototype's contact definition — same shape as test_decoys.py."""
    from atomfrust.decoys.base import DecoyContext
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex
    from atomfrust.regions import resolve_regions
    from atomfrust.settings import ContactSettings, Settings
    from atomfrust.spec import LigandSpec, SystemSpec

    pdb = PROCESSED / f"{pdb_stem}_clean.pdb"
    params = PARAMS / f"{params_name}.params"
    _require(pdb, params)

    spec = SystemSpec.from_pdb(pdb, system_id=pdb_stem)
    spec = spec.model_copy(
        update={"ligands": (LigandSpec(selector=spec.ligands[0].selector, params=params),)}
    )
    loaded = load_complex(spec)
    settings = Settings()
    _, pairs = build_graph(
        loaded.nodes,
        loaded.geometry,
        settings,
        definitions={
            "proto": ContactSettings(cutoff_A=10.0, seq_sep_min=4, seq_sep_basis="pose_index")
        },
    )
    return (
        DecoyContext(
            pose=loaded.pose,
            nodes=loaded.nodes,
            pairs=pairs,
            regions=resolve_regions(loaded.nodes, loaded.geometry),
            settings=settings,
        ),
        loaded,
    )


@pytest.fixture(scope="module")
def context_5gmp():
    return _context("5GMP", "F62")


@pytest.fixture(scope="module")
def decoys_5gmp(context_5gmp):
    """Three decoys, generated once and shared. The budget for the whole module."""
    from atomfrust.decoys.pose import PoseDecoyGenerator

    context, _ = context_5gmp
    generator = PoseDecoyGenerator(context, base_seed=42)
    return generator, [generator.generate(i) for i in range(3)]


# ============================================================ unit: pure logic


@pytest.mark.unit
def test_ligand_node_refuses_to_guess_between_several_components():
    """Which molecule was randomised *is* the identity of the experiment, so a system with
    two poseable components must be told which one rather than getting the first."""
    from atomfrust.decoys.pose import ligand_node

    class FakeNode:
        def __init__(self, kind: str, resnum: int) -> None:
            self.kind = kind
            self.pose_resnum = resnum
            self.node_id = f"A:{resnum}"

    class FakeContext:
        def __init__(self, nodes) -> None:
            self.nodes = nodes

    two = FakeContext([FakeNode("protein", 1), FakeNode("ligand", 2), FakeNode("ligand", 3)])
    with pytest.raises(ValueError, match="several poseable components"):
        ligand_node(two)
    assert ligand_node(two, 3).pose_resnum == 3
    with pytest.raises(ValueError, match="no poseable component at pose residue 9"):
        ligand_node(two, 9)

    with pytest.raises(ValueError, match="needs a ligand or cofactor"):
        ligand_node(FakeContext([FakeNode("protein", 1)]))


@pytest.mark.unit
def test_the_docking_backends_this_axis_would_prefer_are_absent_here():
    """Documents the environment the perturbation route exists for.

    If this ever fails because smina or gnina appeared on PATH, the module docstring's claim
    that a docking ensemble is unavailable has expired and `backend=` should become the
    default route rather than an option.
    """
    from atomfrust.dock import available_backends

    available = available_backends()
    assert "smina" not in available and "gnina" not in available, (
        f"a docking backend is now available ({available}) — revisit PoseDecoyGenerator's "
        "default route, which is a perturbation ensemble precisely because none was"
    )


# ================================================= integration: the pose axis


@pytest.mark.integration
def test_generated_poses_clear_the_min_rmsd_threshold(decoys_5gmp):
    """G5's own criterion: a 'decoy' pose within repacking noise of the native tests nothing."""
    generator, results = decoys_5gmp
    from atomfrust.decoys.pose import ligand_rmsd

    rmsds = [r.index_row["rmsd_to_native"] for r in results]
    print(f"\nG5 ligand RMSD to native (5GMP, F62, 3 decoys): "
          f"{', '.join(f'{r:.2f}' for r in rmsds)} A "
          f"(min {min(rmsds):.2f}, median {float(np.median(rmsds)):.2f}, max {max(rmsds):.2f})")

    for result in results:
        assert result.index_row["status"] == "ok"
        assert result.index_row["rmsd_to_native"] >= generator.min_rmsd_A
        # The recorded number must be the pose that is actually scored, not the proposal:
        # relaxation could have pulled the ligand back if its chis were in the movemap.
        measured = ligand_rmsd(generator.context.pose, result.pose, generator.pose_resnum)
        assert measured == pytest.approx(result.index_row["rmsd_to_native"], abs=1e-9)


@pytest.mark.integration
def test_the_ligand_moves_through_its_own_fold_tree_jump(context_5gmp):
    """Rigid-body sampling goes through the kinematics layer, not through set_xyz, so the
    ligand's internal geometry is preserved by construction rather than by care."""
    from atomfrust.decoys.pose import PoseDecoyGenerator, find_ligand_jump

    context, _ = context_5gmp
    generator = PoseDecoyGenerator(context)
    assert generator.jump is not None, "5GMP's ligand should be its jump's downstream partner"
    assert find_ligand_jump(context.pose, generator.pose_resnum) == generator.jump
    assert find_ligand_jump(context.pose, 1) is None, "residue 1 is the fold-tree root"


@pytest.mark.integration
def test_the_protein_backbone_is_bit_identical(decoys_5gmp):
    """Axis A's invariant, inherited: native and decoy must differ only in the ligand's
    placement and the pocket's rotamers, at fixed backbone geometry."""
    from atomfrust.decoys.identity import assert_backbone_identical

    generator, results = decoys_5gmp
    for result in results:
        assert result.index_row["backbone_rmsd"] <= 1e-6
        assert assert_backbone_identical(generator.context.pose, result.pose, tol=1e-6) <= 1e-6


@pytest.mark.integration
def test_the_same_seed_reproduces_the_same_pose(context_5gmp):
    """Both halves of the proposal must be seeded: the rigid-body move draws from Rosetta's
    RNG and the torsion sampling from Python's, so seeding one alone reproduces half a pose."""
    from atomfrust.decoys.pose import PoseDecoyGenerator, ligand_rmsd

    context, _ = context_5gmp
    generator = PoseDecoyGenerator(context, base_seed=42)
    first = generator.generate(0)
    again = generator.generate(0)
    other = generator.generate(1)

    assert ligand_rmsd(first.pose, again.pose, generator.pose_resnum) == 0.0
    assert np.array_equal(first.e_direct, again.e_direct)
    assert first.index_row["seed"] == 42 and other.index_row["seed"] == 43
    assert ligand_rmsd(first.pose, other.pose, generator.pose_resnum) > 0.5
    assert not np.array_equal(first.e_direct, other.e_direct)


@pytest.mark.integration
def test_energy_moves_where_the_ligand_is_and_nowhere_else(context_5gmp, decoys_5gmp):
    """The axis must be *local*: moving the ligand may not perturb a pair 15 A away.

    A non-zero delta out there would mean the repack shell had leaked, or the backbone had
    moved, or the score function was being re-initialised differently — all of which would
    contaminate the Z-score with something that is not the ligand.
    """
    from atomfrust.decoys.base import extract_energies
    from atomfrust.decoys.pose import ligand_heavy_coords

    context, loaded = context_5gmp
    generator, results = decoys_5gmp
    _, native_direct, _ = extract_energies(context.pose, context.pairs, context.score_function)

    ligand_xyz = ligand_heavy_coords(context.pose, generator.pose_resnum)
    distance: dict[str, float] = {}
    for node in loaded.nodes:
        if node.kind != "protein":
            continue
        residue = context.pose.residue(node.pose_resnum)
        points = np.asarray(
            [
                (residue.xyz(i).x, residue.xyz(i).y, residue.xyz(i).z)
                for i in range(1, residue.nheavyatoms() + 1)
            ],
            dtype=float,
        )
        distance[node.node_id] = float(
            np.sqrt(((points[:, None, :] - ligand_xyz[None, :, :]) ** 2).sum(axis=2)).min()
        )

    pairs = context.pairs
    ligand_incident = ((pairs.kind_i != "protein") | (pairs.kind_j != "protein")).to_numpy()
    d_i = pairs.node_i.map(distance).to_numpy(dtype=float)
    d_j = pairs.node_j.map(distance).to_numpy(dtype=float)
    distant = ~ligand_incident & (np.nan_to_num(d_i, nan=0.0) > 15.0) & (
        np.nan_to_num(d_j, nan=0.0) > 15.0
    )
    assert ligand_incident.sum() > 0 and distant.sum() > 100

    for result in results:
        near = np.abs(result.e_direct[ligand_incident] - native_direct[ligand_incident])
        far = np.abs(result.e_direct[distant] - native_direct[distant])
        print(
            f"\nG5 decoy {result.decoy_id} (RMSD {result.index_row['rmsd_to_native']:.2f} A): "
            f"ligand-incident |delta e_direct| mean {near.mean():.3f} max {near.max():.1f} "
            f"over {ligand_incident.sum()} pairs | distant protein-protein (>15 A) mean "
            f"{far.mean():.2e} max {far.max():.2e} over {distant.sum()} pairs"
        )
        assert near.max() > 1.0, "moving the ligand must change what the ligand touches"
        assert far.max() < 1e-3, "a pair 15 A from the ligand must not notice it moved"


@pytest.mark.integration
def test_the_validity_verdict_is_recorded_for_every_decoy(decoys_5gmp):
    """S4.2's gate is wired even though only the built-in subset can run on this machine.

    The `pose_checker` column is the point: a run that reports a pass rate must say which
    implementation produced it, because the subset is a floor rather than the criterion.
    """
    from atomfrust.dock import posebusters_available

    _, results = decoys_5gmp
    expected = "posebusters" if posebusters_available() else "builtin_subset"
    for result in results:
        row = result.index_row
        assert set(row) >= {"pose_valid", "pose_checker", "pose_checks_run", "pose_failed_checks"}
        assert row["pose_checker"] == expected
        assert row["pose_checks_run"] > 0, "a verdict of 'nothing ran' is not a verdict"
        assert isinstance(row["pose_valid"], bool)
        assert row["pose_valid"] == (row["pose_failed_checks"] == "")


@pytest.mark.integration
def test_the_native_pose_is_gated_by_the_same_checker(context_5gmp):
    """Characterises the gate rather than asserting it passes.

    5GMP's native ligand fails `protein_ligand_clash` at 1.81 A, and that is correct: 5GMP is
    covalent (`preparation_summary.csv`: CYS797 SG to ligand atom CAR) and a covalent bond is
    a sub-van-der-Waals distance. The built-in subset has no concept of one, so a covalent
    system's pass rate is floored at zero — which is why `require_valid` defaults to False.
    """
    from atomfrust.decoys.pose import PoseDecoyGenerator
    from atomfrust.dock import check_poses, pose_pass_rate

    context, _ = context_5gmp
    generator = PoseDecoyGenerator(context)
    native = generator.native_pose3d()
    assert native.rmsd_to_reference == 0.0

    qc = check_poses([native], generator.receptor_pdb(context.pose), checker="builtin_subset")
    ran = qc[~qc["skipped"].astype(bool)]
    failed = sorted(ran.loc[~ran["passed"].astype(bool), "check"].tolist())
    print(
        f"\nG5 native 5GMP under the built-in subset: {len(ran)} checks ran, "
        f"failed {failed or ['none']} (pass rate {pose_pass_rate(qc):.2f})"
    )
    assert set(qc["checker"]) == {"builtin_subset"}
    assert failed == ["protein_ligand_clash"], (
        "expected exactly the covalent CYS797 linkage to read as a clash; a different "
        "failure means the receptor written for the gate is wrong"
    )


@pytest.mark.integration
def test_exhausting_max_attempts_is_a_status_not_a_hang(context_5gmp):
    """An unreachable threshold must terminate with the best proposal and say why."""
    from atomfrust.decoys.pose import PoseDecoyGenerator

    context, _ = context_5gmp
    generator = PoseDecoyGenerator(context, base_seed=7, min_rmsd_A=500.0, max_attempts=2)
    result = generator.generate(0)

    assert result.index_row["status"] == "rmsd_below_min"
    assert result.index_row["n_attempts"] == 2
    assert 0.0 < result.index_row["rmsd_to_native"] < 500.0
    assert result.e_direct.shape[0] == len(context.pairs), (
        "an exhausted decoy is still a decoy: it carries energies and a status, it is not "
        "dropped and it is not an exception"
    )


@pytest.mark.integration
def test_an_unavailable_backend_says_so_instead_of_falling_back(context_5gmp):
    """`backend=` is the seam a docking program plugs into. Silently degrading to the
    perturbation route would relabel the experiment without saying so."""
    from atomfrust.decoys.pose import PoseDecoyGenerator
    from atomfrust.dock import BackendUnavailable, SminaBackend

    context, _ = context_5gmp
    generator = PoseDecoyGenerator(context, backend=SminaBackend())
    assert generator.generator_name == "pose/smina"
    with pytest.raises(BackendUnavailable, match="not available on this machine"):
        generator.generate(0)


@pytest.mark.integration
def test_the_ligand_molecule_carries_rosettas_bond_orders(context_5gmp):
    """Bond orders come from the ResidueType, not from proximity perception.

    The gate's bond-length and bond-angle checks are run against a distance-geometry bounds
    matrix built from these bond orders, so perceived single bonds where the molecule has
    double bonds would fail the native for a defect that exists only in the perception step.
    """
    from rdkit import Chem

    from atomfrust.decoys.pose import PoseDecoyGenerator, ligand_mol_from_pose

    context, _ = context_5gmp
    generator = PoseDecoyGenerator(context)
    mol = ligand_mol_from_pose(context.pose, generator.pose_resnum)

    assert mol.GetNumConformers() == 1
    assert mol.GetNumAtoms() == 39, "F62 is a 39-heavy-atom ligand"
    orders = {bond.GetBondType() for bond in mol.GetBonds()}
    assert Chem.BondType.DOUBLE in orders, "a perceived-bond molecule would be all single"
    assert Chem.MolToSmiles(mol), "the molecule must sanitise into a SMILES"
