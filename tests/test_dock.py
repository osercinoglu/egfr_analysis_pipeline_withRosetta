"""G4 tests for atomfrust.dock — backend probing, MCS placement, and the validity gate.

Everything here is the ``unit`` tier and **nothing requires a docking binary**. That is the
property under test as much as it is a convenience: a machine without smina or GNINA must
still be able to prove that the package degrades correctly, that MCS alignment places a
molecule, and that the gate rejects an invalid pose. The binaries themselves are exercised
only where their *absence* is the subject.

RDKit is required for the placement and gate tests and is skipped for cleanly when missing,
matching ``tests/test_chem.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomfrust.dock import (
    POSEBUSTERS_CHECKS,
    QC_COLUMNS,
    S4_2_MIN_PASS_RATE,
    BackendUnavailable,
    GninaBackend,
    MCSAlignBackend,
    Pose3D,
    PoseBackend,
    PrePosedBackend,
    SminaBackend,
    available_backends,
    check_poses,
    filter_valid,
    get_backend,
    list_backends,
    meets_s4_2,
    passing_pose_indices,
    pose_pass_rate,
    posebusters_available,
)
from atomfrust.dock.base import rdkit_available

needs_rdkit = pytest.mark.skipif(not rdkit_available(), reason="RDKit not installed")

#: A binary name no machine has. Used so the "absent binary" tests test absence rather than
#: the local machine's software inventory.
ABSENT_BINARY = "atomfrust-no-such-docking-binary"


# ------------------------------------------------------------------ fixtures


def _embed(smiles: str, seed: int = 0xC0FFEE):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return mol


@pytest.fixture
def reference_molblock() -> str:
    """A tiny native ligand: benzamide, embedded and minimised. No PDB needed."""
    from rdkit import Chem

    return Chem.MolToMolBlock(_embed("c1ccccc1C(=O)N"))


@pytest.fixture
def clean_pose() -> Pose3D:
    """A physically sensible pose — octane, ETKDG-embedded and MMFF-minimised."""
    from rdkit import Chem

    return Pose3D(molblock=Chem.MolToMolBlock(_embed("CCCCCCCC")), source="fixture")


@pytest.fixture
def clashing_pose() -> Pose3D:
    """The same molecule with one terminal carbon moved onto the other terminus.

    A deliberate, unambiguous violation: the two atoms are seven bonds apart, so no torsion
    or ring closure can explain the contact.
    """
    from rdkit import Chem

    mol = _embed("CCCCCCCC")
    conformer = mol.GetConformer()
    conformer.SetAtomPosition(7, conformer.GetAtomPosition(0))
    return Pose3D(molblock=Chem.MolToMolBlock(mol), source="fixture")


def _write_receptor(path: Path, points: np.ndarray, element: str = "C") -> Path:
    """A minimal well-formed PDB: one heavy atom per point, element in columns 77-78."""
    lines = []
    for serial, (x, y, z) in enumerate(np.atleast_2d(points), start=1):
        lines.append(
            f"ATOM  {serial:5d}  {element:<3s}ALA A{serial:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _coords(pose: Pose3D) -> np.ndarray:
    return np.asarray(pose.to_mol().GetConformer().GetPositions(), dtype=float)


def _verdict(qc: pd.DataFrame, check: str, pose_index: int = 0) -> pd.Series:
    row = qc[(qc["pose_index"] == pose_index) & (qc["check"] == check)]
    assert len(row) == 1, f"expected exactly one {check!r} row, got {len(row)}"
    return row.iloc[0]


# ------------------------------------------------------------------ registry and probing


@pytest.mark.unit
def test_known_backends_are_registered():
    assert set(list_backends()) == {"mcs_align", "smina", "gnina", "preposed"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("mcs_align", MCSAlignBackend),
        ("smina", SminaBackend),
        ("gnina", GninaBackend),
        ("preposed", PrePosedBackend),
    ],
)
def test_get_backend_constructs_by_name(name, cls):
    backend = get_backend(name)
    assert isinstance(backend, cls)
    assert backend.name == name
    assert isinstance(backend, PoseBackend)  # the Protocol is runtime-checkable


@pytest.mark.unit
def test_get_backend_accepts_documented_aliases():
    assert isinstance(get_backend("MCS"), MCSAlignBackend)
    assert isinstance(get_backend("pre-posed"), PrePosedBackend)


@pytest.mark.unit
def test_get_backend_rejects_unknown_names_and_lists_the_known_ones():
    with pytest.raises(ValueError) as excinfo:
        get_backend("vina")
    message = str(excinfo.value)
    assert "vina" in message
    for name in list_backends():
        assert name in message


@pytest.mark.unit
@pytest.mark.parametrize("cls", [SminaBackend, GninaBackend])
def test_absent_binary_reports_unavailable_without_raising(cls):
    """The whole degradation story in one assertion: a question, not an exception."""
    backend = cls(binary=ABSENT_BINARY)
    assert backend.available() is False
    assert backend.version() is None


@pytest.mark.unit
def test_available_is_a_bool_for_every_backend_on_this_machine():
    """Whatever is or is not installed here, no backend may raise when asked."""
    for name in list_backends():
        assert isinstance(get_backend(name).available(), bool)
    assert set(available_backends()) <= set(list_backends())
    assert "preposed" in available_backends()


@pytest.mark.unit
@pytest.mark.parametrize("cls", [SminaBackend, GninaBackend])
def test_posing_with_an_absent_binary_raises_backend_unavailable(cls, tmp_path):
    backend = cls(binary=ABSENT_BINARY)
    with pytest.raises(BackendUnavailable) as excinfo:
        backend.pose("CCO", tmp_path / "receptor.pdb")
    assert ABSENT_BINARY in str(excinfo.value)


@pytest.mark.unit
def test_gnina_inherits_the_smina_invocation_and_adds_cnn_flags():
    backend = GninaBackend(binary=ABSENT_BINARY, cnn_scoring="refinement")
    assert backend.extra_args == ("--cnn_scoring", "refinement")
    assert backend.score_property == "CNNaffinity"
    assert SminaBackend.score_property == "minimizedAffinity"


@needs_rdkit
@pytest.mark.unit
def test_smina_invocation_and_output_parsing_against_a_stub(tmp_path, monkeypatch, reference_molblock):
    """Drive the subprocess path with a stub that impersonates smina's contract.

    Not a substitute for smina, and it does not dock anything: it checks the two things this
    package owns and smina does not — that the command names the receptor, the autobox
    ligand and the seed, and that ``minimizedAffinity`` comes back as
    :attr:`~atomfrust.dock.base.Pose3D.score`. Without it the whole subprocess branch is
    unexecuted on every machine that lacks the binary, which is every machine here.
    """
    stub = tmp_path / "smina"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "argv = sys.argv[1:]\n"
        "out = argv[argv.index('--out') + 1]\n"
        "lig = argv[argv.index('--ligand') + 1]\n"
        "open(out, 'w').write(\n"
        "    open(lig).read().rstrip('\\n').replace('$$$$', '')\n"
        "    + '\\n>  <minimizedAffinity>\\n-7.5\\n\\n$$$$\\n')\n"
        "sys.stderr.write(' '.join(argv))\n"
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    receptor = _write_receptor(tmp_path / "receptor.pdb", np.zeros((1, 3)))
    backend = SminaBackend()
    assert backend.available() is True

    poses = backend.pose("CCO", receptor, reference_ligand=reference_molblock, seed=42)
    assert len(poses) == 1
    pose = poses[0]
    assert pose.source == "smina"
    assert pose.score == pytest.approx(-7.5)
    assert pose.metadata["blind"] is False
    assert pose.metadata["seed"] == 42
    command = pose.metadata["command"]
    assert "--receptor" in command and str(receptor) in command
    assert "--autobox_ligand" in command and "--seed 42" in command
    assert pose.to_mol().GetNumAtoms() > 0


# ------------------------------------------------------------------ MCS alignment


@needs_rdkit
@pytest.mark.unit
def test_mcs_align_places_the_shared_scaffold_on_the_reference(reference_molblock):
    """The pose must sit *where the reference sat*, atom for atom over the common core."""
    from rdkit import Chem

    backend = MCSAlignBackend()
    assert backend.available() is True

    poses = backend.pose("c1ccccc1C(=O)NC", receptor_pdb=None, reference_ligand=reference_molblock)
    assert len(poses) == 1
    pose = poses[0]
    assert pose.source == "mcs_align"
    assert pose.score is None  # no scoring function; a fabricated score would be averaged

    # Benzamide's nine heavy atoms are the common substructure with N-methylbenzamide.
    assert pose.metadata["n_mcs_atoms"] >= 9
    assert pose.metadata["receptor_used"] is False
    assert pose.rmsd_to_reference is not None and pose.rmsd_to_reference < 0.5

    # Independent of the backend's own bookkeeping: re-derive the correspondence and compare
    # coordinates directly.
    reference = Chem.MolFromMolBlock(reference_molblock, removeHs=False)
    pattern = Chem.MolFromSmarts(pose.metadata["mcs_smarts"])
    posed = pose.to_mol()
    posed_match = posed.GetSubstructMatch(pattern)
    ref_match = reference.GetSubstructMatch(pattern)
    assert posed_match and ref_match

    posed_xyz = np.array([list(posed.GetConformer().GetAtomPosition(i)) for i in posed_match])
    ref_xyz = np.array([list(reference.GetConformer().GetAtomPosition(i)) for i in ref_match])
    assert np.sqrt(((posed_xyz - ref_xyz) ** 2).sum(axis=1)).max() < 1.0


@needs_rdkit
@pytest.mark.unit
def test_mcs_align_needs_no_receptor_which_is_the_ablation(reference_molblock, tmp_path):
    """Passing a receptor changes nothing — that is the point, not an oversight.

    The chemotype axis run through this backend cannot be explained by a docking program's
    behaviour, because no receptor ever enters the placement.
    """
    backend = MCSAlignBackend()
    without = backend.pose("c1ccccc1C(=O)NC", None, reference_molblock, seed=3)[0]
    with_receptor = backend.pose(
        "c1ccccc1C(=O)NC", tmp_path / "nonexistent.pdb", reference_molblock, seed=3
    )[0]
    assert without.molblock == with_receptor.molblock


@needs_rdkit
@pytest.mark.unit
def test_mcs_align_emits_distinct_poses_for_distinct_seeds(reference_molblock):
    poses = MCSAlignBackend().pose(
        "c1ccccc1C(=O)NCCCC", None, reference_molblock, n_poses=3, seed=11
    )
    assert len(poses) == 3
    # Seeded seed+k, but returned best-MCS-RMSD first, so the seeds are a set not a sequence.
    assert sorted(p.metadata["seed"] for p in poses) == [11, 12, 13]
    assert len({p.molblock for p in poses}) > 1  # the tail samples different conformations
    rmsds = [p.rmsd_to_reference for p in poses]
    assert rmsds == sorted(rmsds)


@needs_rdkit
@pytest.mark.unit
def test_mcs_align_without_a_reference_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="reference_ligand"):
        MCSAlignBackend().pose("CCO", tmp_path / "receptor.pdb")


@needs_rdkit
@pytest.mark.unit
def test_mcs_align_refuses_a_molecule_that_shares_too_little(reference_molblock):
    """No MCS means no placement. Silently posing it anyway would put a molecule in the
    pocket by accident and let it into a chemotype comparison."""
    with pytest.raises(ValueError):
        MCSAlignBackend(min_mcs_atoms=6).pose("[Na+].[Cl-]", None, reference_molblock)


# ------------------------------------------------------------------ pre-posed pass-through


@needs_rdkit
@pytest.mark.unit
def test_preposed_round_trips_a_molblock_byte_for_byte(reference_molblock):
    backend = PrePosedBackend()
    assert backend.available() is True
    poses = backend.pose(reference_molblock, receptor_pdb=None)
    assert len(poses) == 1
    assert poses[0].molblock == reference_molblock
    assert poses[0].source == "preposed"
    assert poses[0].metadata["passthrough"] is True


@needs_rdkit
@pytest.mark.unit
def test_preposed_reads_multi_record_sdf_files(reference_molblock, tmp_path):
    from rdkit import Chem

    second = Chem.MolToMolBlock(_embed("CCO"))
    sdf = tmp_path / "poses.sdf"
    sdf.write_text(f"{reference_molblock}$$$$\n{second}$$$$\n")

    poses = PrePosedBackend().pose(sdf, receptor_pdb=None, n_poses=5)
    assert len(poses) == 2
    assert poses[0].molblock == reference_molblock
    assert poses[1].molblock == second


@pytest.mark.unit
def test_preposed_refuses_a_smiles():
    """A SMILES has no coordinates; embedding one would invent the pose it claims to relay."""
    with pytest.raises(ValueError, match="no coordinates"):
        PrePosedBackend().pose("c1ccccc1C(=O)N", receptor_pdb=None)


@needs_rdkit
@pytest.mark.unit
def test_preposed_computes_rmsd_against_a_reference(reference_molblock):
    pose = PrePosedBackend().pose(
        reference_molblock, None, reference_ligand=reference_molblock
    )[0]
    assert pose.rmsd_to_reference == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ the validity gate


@needs_rdkit
@pytest.mark.unit
def test_gate_passes_a_clean_pose_and_names_the_implementation(clean_pose):
    qc = check_poses([clean_pose], checker="builtin_subset")

    assert list(qc.columns) == list(QC_COLUMNS)
    assert set(qc["checker"]) == {"builtin_subset"}
    assert set(qc["check"]) == set(POSEBUSTERS_CHECKS)
    executed = qc[~qc["skipped"]]
    assert executed["passed"].all(), executed[~executed["passed"]][["check", "detail"]].to_dict()
    assert pose_pass_rate(qc) == 1.0


@needs_rdkit
@pytest.mark.unit
def test_gate_flags_an_internal_clash_and_names_the_failing_check(clashing_pose):
    qc = check_poses([clashing_pose], checker="builtin_subset")

    clash = _verdict(qc, "internal_steric_clash")
    assert clash["passed"] is np.False_ or clash["passed"] is False
    assert "clashing pair" in clash["detail"]
    assert pose_pass_rate(qc) == 0.0
    assert passing_pose_indices(qc) == []
    assert filter_valid([clashing_pose], qc) == []


@needs_rdkit
@pytest.mark.unit
def test_gate_reports_which_implementation_ran_under_auto(clean_pose):
    """``auto`` follows the environment, and says so in every row — this is the assertion
    that stops a built-in-subset run being reported as a PoseBusters run."""
    qc = check_poses([clean_pose], checker="auto")
    expected = "posebusters" if posebusters_available() else "builtin_subset"
    assert set(qc["checker"]) == {expected}


@pytest.mark.unit
@pytest.mark.skipif(posebusters_available(), reason="posebusters is installed here")
def test_requesting_real_posebusters_when_absent_raises_rather_than_substituting(clean_pose):
    with pytest.raises(BackendUnavailable, match="posebusters"):
        check_poses([clean_pose], checker="posebusters")


@needs_rdkit
@pytest.mark.unit
def test_protein_aware_checks_are_skipped_not_faked_without_a_receptor(clean_pose):
    qc = check_poses([clean_pose], receptor_pdb=None, checker="builtin_subset")
    for check in ("protein_ligand_clash", "inside_receptor_bbox"):
        assert bool(_verdict(qc, check)["skipped"]) is True


@needs_rdkit
@pytest.mark.unit
def test_gate_flags_a_protein_ligand_clash(clean_pose, tmp_path):
    coords = _coords(clean_pose)
    cage = coords.mean(axis=0) + np.array(
        [[dx, dy, dz] for dx in (-14, 14) for dy in (-14, 14) for dz in (-14, 14)], dtype=float
    )

    clean_receptor = _write_receptor(tmp_path / "clean.pdb", cage)
    qc = check_poses([clean_pose], clean_receptor, checker="builtin_subset")
    assert bool(_verdict(qc, "protein_ligand_clash")["passed"]) is True
    assert bool(_verdict(qc, "inside_receptor_bbox")["passed"]) is True

    # One receptor atom placed exactly on a ligand atom, everything else unchanged.
    clashing_receptor = _write_receptor(
        tmp_path / "clash.pdb", np.vstack([cage, coords[0]])
    )
    qc = check_poses([clean_pose], clashing_receptor, checker="builtin_subset")
    verdict = _verdict(qc, "protein_ligand_clash")
    assert bool(verdict["passed"]) is False
    # PDB coordinates carry three decimals, so "exactly on top of" is 5e-4 A of rounding.
    assert verdict["value"] == pytest.approx(0.0, abs=1e-2)
    assert "clashing atom pair" in verdict["detail"]


@needs_rdkit
@pytest.mark.unit
def test_gate_flags_a_pose_outside_the_receptor(clean_pose, tmp_path):
    from rdkit import Chem

    coords = _coords(clean_pose)
    receptor = _write_receptor(tmp_path / "receptor.pdb", coords.mean(axis=0).reshape(1, 3))

    mol = clean_pose.to_mol()
    conformer = mol.GetConformer()
    for index in range(mol.GetNumAtoms()):
        position = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, (position.x + 100.0, position.y, position.z))
    far_away = Pose3D(molblock=Chem.MolToMolBlock(mol), source="fixture")

    qc = check_poses([far_away], receptor, checker="builtin_subset")
    verdict = _verdict(qc, "inside_receptor_bbox")
    assert bool(verdict["passed"]) is False
    assert verdict["value"] > 90.0


@needs_rdkit
@pytest.mark.unit
def test_gate_flags_an_unparseable_pose_without_raising():
    qc = check_poses([Pose3D(molblock="not a molblock", source="fixture")],
                     checker="builtin_subset")
    assert bool(_verdict(qc, "sanitization")["passed"]) is False
    assert set(qc["check"]) == set(POSEBUSTERS_CHECKS)
    assert pose_pass_rate(qc) == 0.0


@needs_rdkit
@pytest.mark.unit
def test_identity_checks_use_the_metadata_reference(clean_pose):
    from rdkit import Chem

    honest = Pose3D(
        molblock=clean_pose.molblock,
        source="fixture",
        metadata={"smiles": Chem.CanonSmiles("CCCCCCCC")},
    )
    qc = check_poses([honest], checker="builtin_subset")
    assert bool(_verdict(qc, "stereochemistry_preserved")["passed"]) is True
    assert bool(_verdict(qc, "formal_charge_preserved")["passed"]) is True

    wrong = Pose3D(molblock=clean_pose.molblock, source="fixture",
                   metadata={"reference_smiles": "CC(=O)[O-]"})
    qc = check_poses([wrong], checker="builtin_subset")
    assert bool(_verdict(qc, "stereochemistry_preserved")["passed"]) is False
    assert bool(_verdict(qc, "formal_charge_preserved")["passed"]) is False


@needs_rdkit
@pytest.mark.unit
def test_gate_keeps_the_clean_pose_and_drops_the_clashing_one(clean_pose, clashing_pose):
    poses = [clean_pose, clashing_pose]
    qc = check_poses(poses, checker="builtin_subset")
    assert passing_pose_indices(qc) == [0]
    assert filter_valid(poses, qc) == [clean_pose]
    assert pose_pass_rate(qc) == pytest.approx(0.5)


@pytest.mark.unit
def test_empty_input_returns_the_schema_not_an_error():
    qc = check_poses([])
    assert qc.empty
    assert list(qc.columns) == list(QC_COLUMNS)
    assert np.isnan(pose_pass_rate(qc))
    assert meets_s4_2(qc) is False  # nothing measured is not a pass


@pytest.mark.unit
def test_check_poses_rejects_an_unknown_checker(clean_pose=None):
    with pytest.raises(ValueError, match="builtin_subset"):
        check_poses([Pose3D(molblock="", source="x")], checker="posebuster")


# ------------------------------------------------------------------ S4.2 pass rate


def _synthetic_qc(n_poses: int, n_failing: int, *, checker: str = "builtin_subset") -> pd.DataFrame:
    """One row per pose per check, with ``n_failing`` poses failing one check each."""
    rows = []
    for index in range(n_poses):
        for position, check in enumerate(POSEBUSTERS_CHECKS):
            rows.append(
                {
                    "pose_index": index,
                    "pose_id": f"p{index}",
                    "source": "fixture",
                    "checker": checker,
                    "check": check,
                    "passed": not (index < n_failing and position == 3),
                    "skipped": False,
                    "value": None,
                    "detail": "",
                }
            )
    return pd.DataFrame(rows, columns=list(QC_COLUMNS))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n_poses", "n_failing", "expected", "meets"),
    [(10, 0, 1.00, True), (10, 1, 0.90, True), (10, 2, 0.80, False), (100, 9, 0.91, True)],
)
def test_pass_rate_is_the_s4_2_fraction(n_poses, n_failing, expected, meets):
    """S4.2: >=90% of poses entering analysis pass *all* checks — poses, not checks."""
    qc = _synthetic_qc(n_poses, n_failing)
    assert pose_pass_rate(qc) == pytest.approx(expected)
    assert meets_s4_2(qc) is meets
    assert S4_2_MIN_PASS_RATE == 0.90


@pytest.mark.unit
def test_a_pose_fails_the_rate_if_any_single_check_fails():
    """One failure is a failed pose: the criterion is "pass all checks", not "mostly pass"."""
    qc = _synthetic_qc(4, 1)
    assert pose_pass_rate(qc) == pytest.approx(0.75)
    assert passing_pose_indices(qc) == [1, 2, 3]


@pytest.mark.unit
def test_skipped_checks_do_not_count_as_passes():
    qc = _synthetic_qc(2, 0)
    qc.loc[qc["check"] == "protein_ligand_clash", ["passed", "skipped"]] = [False, True]
    assert pose_pass_rate(qc) == 1.0  # skipped rows are excluded from the verdict

    qc.loc[qc["check"] == "bond_lengths", "passed"] = False
    assert pose_pass_rate(qc) == 0.0
