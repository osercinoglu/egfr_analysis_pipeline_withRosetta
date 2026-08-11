"""Axis B — the ligand-pose decoy: the protein is fixed and the *placement* is randomised.

Axis A (:mod:`atomfrust.decoys.identity`) asks "what if this pocket held a different
sequence?". Axis B asks the complementary question: "what if this molecule sat somewhere
else?". It is the **easy ligand axis** and lands before the chemotype axis (G6) for one
reason — the molecule is unchanged, so every per-contact key stays well defined. The pair
table is frozen on the native pose exactly as before, node ``i`` is still residue ``i``, and
the ligand node is still the same ligand. Nothing about correspondence has to be invented,
which is precisely what G6 cannot say.

**This is a perturbation ensemble, not a docking ensemble.** The plan's G5 entry says
"re-dock the ligand". On this machine that is not achievable: ``smina`` and ``gnina`` are
absent from ``PATH`` (``shutil.which`` returns ``None`` for both), so
:func:`atomfrust.dock.available_backends` reports only ``mcs_align`` and ``preposed`` — and
``mcs_align`` aligns onto the native ligand, which for the *same* molecule reproduces the
native pose rather than sampling alternatives to it. What is implemented instead is a
rigid-body perturbation of the native ligand (translation + rotation about its own centre,
applied through the ligand's fold-tree jump) combined with torsion sampling over the
rotatable bonds the ``.params`` file declares, followed by Rosetta repacking of the pocket
side chains around the displaced ligand.

The two are not the same experiment and the difference is recorded rather than glossed:
a perturbation ensemble samples the neighbourhood of the crystal pose, a docking ensemble
samples the poses a scoring function *prefers*. ``index_row["generator"]`` says which one
produced a decoy (``pose/perturb`` or ``pose/<backend name>``). :meth:`_propose_pose` is the
single seam: pass ``backend=SminaBackend()`` (or any :class:`~atomfrust.dock.base.PoseBackend`)
once a binary exists and the rest of the module is unchanged.

**The validity gate is wired, not deferred.** Every proposed pose — perturbed or docked —
goes through :func:`atomfrust.dock.check_poses` against the receptor as it will actually be
scored, and the verdict lands in ``index_row`` (``pose_valid``, ``pose_checker``,
``pose_failed_checks``). ``posebusters`` is not installed here either, so ``pose_checker``
reads ``builtin_subset``; that column exists so a subset result can never be presented as
S4.2 itself. Invalid poses are **recorded, not silently dropped**: S4.2 is a *pass rate* over
the poses a generator produces, and a generator that discarded its own failures would make
its rate unmeasurable. ``require_valid=True`` opts into rejection-sampling instead.

**Two measured facts about the gate on this data, both worth knowing before reading a pass
rate.** First, on 5GMP the *native* pose fails ``protein_ligand_clash`` — one contact at
1.81 A, well inside the 0.75x-vdW floor. That is not a defect in the pose: 5GMP is one of the
15 covalent complexes (``preparation_summary.csv``: ``CYS797 SG`` to ligand atom ``CAR``), and
a covalent bond is a sub-van-der-Waals distance by definition. The built-in subset has no
concept of a covalent ligand, so a covalent system's pass rate is floored at zero and
``require_valid=True`` would reject every proposal forever. This is why the gate defaults to
*recording* rather than rejecting. Second, displacing a ligand inside an ATP pocket that fits
it snugly produces genuine steric clashes (measured ``fa_rep``-dominated pair energies up to
~2.3e3 REU); that is what a pose decoy *is*, and it is the reason the analysis layer's
``exclude_fa_rep`` switch exists.

**Invariants inherited from axis A.** The protein backbone must stay bit-identical — the
jump moves only the ligand (it is the downstream partner of its own jump, and the fold tree
here is ``EDGE 1 305 -1 / EDGE 1 306 1``), and minimisation runs with ``set_bb(False)`` and
``set_jump(False)``. :func:`~atomfrust.decoys.identity.assert_backbone_identical` is applied
as a post-condition on every decoy, not merely tested.

**The ligand is frozen during relaxation.** Its chi angles are excluded from the movemap and
it is prevented from repacking, so the ligand coordinates that are scored are exactly the
ones that were proposed and checked. ``rmsd_to_native`` is therefore an exact property of
the proposal rather than a number the minimiser has since invalidated, and the pose that
passed the gate is the pose that entered the energy.
"""

from __future__ import annotations

import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atomfrust.decoys.base import DecoyContext, DecoyResult, extract_energies
from atomfrust.decoys.identity import assert_backbone_identical

__all__ = [
    "PoseDecoyGenerator",
    "ligand_node",
    "ligand_heavy_coords",
    "ligand_rmsd",
    "find_ligand_jump",
    "ligand_mol_from_pose",
]

#: Kinds that can be *moved* by this axis. A metal is a node and an energy partner, but a
#: lone ion has no pose to randomise — displacing it samples nothing a docking program would
#: ever propose.
POSEABLE_KINDS = frozenset({"ligand", "cofactor"})


# ------------------------------------------------------------------ ligand geometry


def ligand_node(context: DecoyContext, pose_resnum: int | None = None) -> Any:
    """The node this axis moves.

    With ``pose_resnum`` given, that node; otherwise the single poseable component. Two
    poseable components with no choice made is an error rather than a silent "first one":
    which molecule was randomised is the identity of the experiment.
    """
    candidates = [n for n in context.nodes if n.kind in POSEABLE_KINDS]
    if pose_resnum is not None:
        for node in candidates:
            if node.pose_resnum == pose_resnum:
                return node
        raise ValueError(
            f"no poseable component at pose residue {pose_resnum}; poseable components are "
            + (", ".join(f"{n.node_id} ({n.pose_resnum})" for n in candidates) or "(none)")
        )
    if not candidates:
        raise ValueError(
            "the pose axis needs a ligand or cofactor to move, and this system has none. "
            f"Node kinds present: {sorted({n.kind for n in context.nodes})}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "several poseable components — "
            + ", ".join(f"{n.node_id} ({n.pose_resnum})" for n in candidates)
            + " — pass ligand_resnum to say which one this axis randomises"
        )
    return candidates[0]


def ligand_heavy_coords(pose: Any, resnum: int) -> np.ndarray:
    """``(n_heavy, 3)`` coordinates of one residue, in pose atom order."""
    residue = pose.residue(resnum)
    rows = []
    for index in range(1, residue.natoms() + 1):
        if residue.is_virtual(index):
            continue
        if residue.atom_type(index).element().strip() == "H":
            continue
        xyz = residue.xyz(index)
        rows.append((xyz.x, xyz.y, xyz.z))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def ligand_rmsd(native: Any, decoy: Any, resnum: int) -> float:
    """In-place heavy-atom RMSD of one residue between two poses.

    No superposition, deliberately: both poses live in the receptor's frame and the
    displacement *in that frame* is the quantity G5 thresholds on. Superimposing first would
    report conformer similarity, which is a different (and much smaller) number. Atom
    correspondence is by index, which is exact here because the two poses hold the same
    residue type — the whole point of this axis.
    """
    a = ligand_heavy_coords(native, resnum)
    b = ligand_heavy_coords(decoy, resnum)
    if a.shape != b.shape:
        raise ValueError(f"ligand atom counts differ: {a.shape[0]} vs {b.shape[0]}")
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def find_ligand_jump(pose: Any, resnum: int) -> int | None:
    """The fold-tree jump whose downstream partner is this residue, or ``None``.

    Moving a ligand through its jump is the route Rosetta itself uses for rigid-body
    sampling: the transform is applied by the kinematics layer, so the ligand's internal
    geometry is untouched by construction and no atom is repositioned by hand. ``None`` means
    the ligand is not a jump's direct downstream partner — a pose built with an unusual fold
    tree — and the caller falls back to transforming coordinates directly.
    """
    tree = pose.fold_tree()
    for jump in range(1, pose.num_jump() + 1):
        if int(tree.downstream_jump_residue(jump)) == int(resnum):
            return jump
    return None


def ligand_mol_from_pose(pose: Any, resnum: int) -> Any:
    """An RDKit ``Mol`` for one residue, built from Rosetta's own chemistry.

    Elements, **bond orders** and formal charges are read from the ``ResidueType`` — which is
    the ``.params`` file, i.e. the same chemistry the energy function is using — rather than
    perceived from coordinates. Proximity bonding off a PDB block would make every aromatic
    ring a set of single bonds, and the validity gate's bond-length and bond-angle checks are
    run against a distance-geometry bounds matrix that depends on exactly those bond orders.
    A perceived-bond molecule would therefore fail the gate for a defect that exists only in
    the perception step.

    Hydrogens are included when the residue type has them; the ligands parametrised by this
    pipeline (``molfile_to_params --keep-names``) are heavy-atom-only Kekulé structures, so
    RDKit supplies the implicit hydrogen count from valence.
    """
    from rdkit import Chem
    from rdkit.Chem import rdchem
    from rdkit.Geometry import Point3D

    residue = pose.residue(resnum)
    rtype = residue.type()

    order = {1: rdchem.BondType.SINGLE, 2: rdchem.BondType.DOUBLE,
             3: rdchem.BondType.TRIPLE, 4: rdchem.BondType.AROMATIC}

    mol = Chem.RWMol()
    index_of: dict[int, int] = {}
    positions: list[tuple[float, float, float]] = []
    for atom_index in range(1, residue.natoms() + 1):
        if residue.is_virtual(atom_index):
            continue
        element = residue.atom_type(atom_index).element().strip().capitalize()
        atom = Chem.Atom(element)
        atom.SetFormalCharge(int(rtype.formal_charge(atom_index)))
        index_of[atom_index] = mol.AddAtom(atom)
        xyz = residue.xyz(atom_index)
        positions.append((xyz.x, xyz.y, xyz.z))

    for atom_index in index_of:
        for neighbour in rtype.bonded_neighbor(atom_index):
            neighbour = int(neighbour)
            if neighbour <= atom_index or neighbour not in index_of:
                continue
            bond = int(rtype.bond_type(atom_index, neighbour))
            mol.AddBond(
                index_of[atom_index], index_of[neighbour], order.get(bond, rdchem.BondType.SINGLE)
            )

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for position, (x, y, z) in enumerate(positions):
        conformer.SetAtomPosition(position, Point3D(x, y, z))
    mol = mol.GetMol()
    mol.AddConformer(conformer, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # A residue type whose bond orders do not add up is worth reporting through the gate
        # (which will fail `sanitization` and say so) rather than raising here and losing the
        # decoy entirely.
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    return mol


# ------------------------------------------------------------------ the generator


@dataclass
class PoseDecoyGenerator:
    """Axis B generator: randomise where the ligand sits, keep everything else.

    ``min_rmsd_A`` is G5's own threshold — a "decoy" pose that is the native pose to within
    the coordinate error of a repack tests nothing, so proposals below it are rejected and
    re-drawn with escalated magnitudes. Exhausting ``max_attempts`` is a **status**, not an
    exception and not a loop: the best proposal is returned with
    ``status="rmsd_below_min"``, so a system whose pocket is too tight to displace the ligand
    2 A is visible in the index rather than hanging the run.
    """

    context: DecoyContext
    base_seed: int = 42
    axis: str = "pose"
    min_rmsd_A: float = 2.0        # keep poses at least this far from native
    max_attempts: int = 20

    #: Which component moves. ``None`` when the system has exactly one.
    ligand_resnum: int | None = None

    #: Perturbation magnitudes for the first attempt. Rosetta's ``RigidBodyPerturbMover``
    #: treats these as Gaussian widths (degrees, angstrom). Measured on 5GMP over 8 decoys:
    #: (20 deg, 1.5 A) gives ligand RMSD 2.73-5.30 A, median 4.05, one attempt each;
    #: (30, 2.0) gives 3.37-7.01 (poses that leave the pocket entirely) and (15, 1.0) gives
    #: 2.29-5.22 but starts needing retries. These defaults sit just above ``min_rmsd_A``
    #: without throwing the ligand into solvent.
    rotation_deg: float = 20.0
    translation_A: float = 1.5
    #: Multiplier applied to both magnitudes on each rejected attempt.
    escalate: float = 1.2
    #: Probability that a given rotatable torsion is resampled, and the Gaussian width of the
    #: resampling. Full uniform resampling produces internally clashing conformers that the
    #: validity gate then rejects, which wastes attempts rather than sampling anything.
    torsion_fraction: float = 0.5
    torsion_sigma_deg: float = 45.0

    #: Pocket shell repacked around the displaced ligand, and whether it is then minimised.
    repack_radius_A: float = 8.0
    minimise: bool = True

    #: Optional docking backend. ``None`` uses the perturbation route described in the module
    #: docstring. Anything satisfying :class:`~atomfrust.dock.base.PoseBackend` is accepted,
    #: so ``backend=SminaBackend()`` needs no change here once the binary exists.
    backend: Any | None = None
    backend_poses: int = 8

    #: Validity gate. ``checker`` is passed straight to
    #: :func:`~atomfrust.dock.posebusters.check_poses`; ``require_valid`` turns the gate into
    #: rejection sampling instead of a recorded verdict.
    checker: str = "auto"
    require_valid: bool = False

    strict_backbone: bool = True

    def __post_init__(self) -> None:
        self._node = ligand_node(self.context, self.ligand_resnum)
        self._resnum = int(self._node.pose_resnum)
        self._jump = find_ligand_jump(self.context.pose, self._resnum)
        self._native_mol: Any | None = None
        self._tempdir: Any | None = None
        self._backend_poses_cache: dict[int, list[Any]] = {}

    # ------------------------------------------------------------------ public

    @property
    def pose_resnum(self) -> int:
        """Pose residue number of the component this generator moves."""
        return self._resnum

    @property
    def jump(self) -> int | None:
        """Fold-tree jump used to move the ligand, or ``None`` for the coordinate route."""
        return self._jump

    @property
    def generator_name(self) -> str:
        """What ``index_row["generator"]`` will say — ``pose/perturb`` or ``pose/<backend>``.

        Readable before any decoy exists, so a run manifest can record which route it is
        about to take rather than inferring it afterwards from the rows it produced.
        """
        return f"pose/{self._route_name()}"

    def generate(self, decoy_id: int) -> DecoyResult:
        started = time.perf_counter()
        seed = self.base_seed + decoy_id

        decoy, rmsd, attempts, status = self._accepted_proposal(seed)
        self._relax_pocket(decoy)
        # The verdict is re-taken *after* relaxation so it describes the pose that is about
        # to be scored: the ligand has not moved (its chis are out of the movemap and its
        # jump is frozen), but the pocket side chains have, and they are half of the
        # protein-ligand clash check. The provisional verdict inside the attempt loop exists
        # only to serve `require_valid`, where re-packing every rejected proposal would make
        # rejection sampling cost as much as acceptance.
        qc = self._verdict(decoy)
        drift = assert_backbone_identical(
            self.context.pose, decoy, tol=1e-6 if self.strict_backbone else 0.05
        )
        pair_id, e_direct, e_fa_rep = extract_energies(
            decoy, self.context.pairs, self.context.score_function
        )
        return DecoyResult(
            decoy_id=decoy_id,
            pair_id=pair_id,
            e_direct=e_direct,
            e_fa_rep=e_fa_rep,
            pose=decoy,
            index_row={
                "decoy_id": decoy_id,
                "axis": self.axis,
                "generator": self.generator_name,
                "seed": seed,
                "ligand_node": self._node.node_id,
                "ligand_resnum": self._resnum,
                "jump": self._jump,
                "rmsd_to_native": rmsd,
                "min_rmsd_A": self.min_rmsd_A,
                "n_attempts": attempts,
                "backbone_rmsd": drift,
                "repack_residues": len(self._pocket_shell(decoy)),
                "wall_s": time.perf_counter() - started,
                "status": status,
                **qc,
            },
        )

    def prepare_native(self) -> Any:
        """The native reference for this axis: the crystal pose, unmoved.

        Axis A repacks the native so native and decoys share a protocol
        (:meth:`~atomfrust.decoys.identity.IdentityDecoyGenerator.prepare_native`). Here the
        symmetric statement is weaker and is left to the caller: the decoys differ from the
        native by a displacement, and "the same protocol without the displacement" is the
        native pose itself. Repacking it is axis A's question, not this one's.
        """
        return self.context.pose

    def native_pose3d(self) -> Any:
        """The native ligand as a :class:`~atomfrust.dock.base.Pose3D`.

        Useful as the reference member of the ensemble: G5's acceptance is that the native
        ranks above the decoys, which needs the native scored as one of them.
        """
        return self._as_pose3d(self.context.pose, source="native")

    # ------------------------------------------------------------------ proposals

    def _route_name(self) -> str:
        return "perturb" if self.backend is None else getattr(self.backend, "name", "backend")

    def _accepted_proposal(self, seed: int) -> tuple[Any, float, int, str]:
        """Draw until the proposal is far enough from native (and valid, if required).

        Rejection happens **before** repacking, which is what keeps ``max_attempts`` cheap:
        a proposal is a clone plus a jump move plus a few ``set_chi`` calls, while the
        pocket repack that follows acceptance is the expensive step and runs once.

        Exhaustion returns the furthest proposal with a status naming which condition was
        never met, so a pocket too tight to displace its ligand ``min_rmsd_A`` shows up as a
        row in the index rather than as a hang.
        """
        best: tuple[Any, float] | None = None
        best_valid = False
        for attempt in range(1, max(1, self.max_attempts) + 1):
            decoy = self._propose_pose(seed, attempt)
            rmsd = ligand_rmsd(self.context.pose, decoy, self._resnum)
            valid = not self.require_valid or bool(self._verdict(decoy)["pose_valid"])
            if best is None or rmsd > best[1]:
                best, best_valid = (decoy, rmsd), valid
            if rmsd >= self.min_rmsd_A and valid:
                return decoy, rmsd, attempt, "ok"
        assert best is not None
        status = "rmsd_below_min" if best[1] < self.min_rmsd_A else "invalid_pose"
        if best_valid and best[1] >= self.min_rmsd_A:  # pragma: no cover - loop would have returned
            status = "ok"
        return best[0], best[1], max(1, self.max_attempts), status

    def _propose_pose(self, seed: int, attempt: int) -> Any:
        """One candidate placement. **The seam a docking backend replaces.**

        The perturbation route is documented in the module docstring; the backend route asks
        a :class:`~atomfrust.dock.base.PoseBackend` for poses of the *same* molecule and
        stamps pose ``attempt-1`` onto the ligand's coordinates.
        """
        self._seed_everything(seed * 1000 + attempt)
        if self.backend is not None:
            return self._propose_from_backend(seed, attempt)
        return self._propose_by_perturbation(attempt)

    def _propose_by_perturbation(self, attempt: int) -> Any:
        """Rigid-body move through the ligand's jump, plus torsion sampling.

        The rigid-body part goes through the fold tree rather than through ``set_xyz``:
        ``RigidBodyPerturbMover`` applies the transform in the kinematics layer, so the
        ligand's bond lengths and angles are preserved by construction. Only when the ligand
        is not a jump's downstream partner does the coordinate route run — and that route
        applies one rotation about the ligand centroid plus a translation, which is rigid for
        the same reason.
        """
        from pyrosetta.rosetta.protocols.rigid import (
            RigidBodyPerturbMover,
            partner_downstream,
        )

        decoy = self.context.pose.clone()
        scale = self.escalate ** (attempt - 1)
        rotation = self.rotation_deg * scale
        translation = self.translation_A * scale

        if self._jump is not None:
            RigidBodyPerturbMover(
                self._jump, rotation, translation, partner_downstream
            ).apply(decoy)
        else:
            self._rigid_move_by_coordinates(decoy, rotation, translation)

        residue = decoy.residue(self._resnum)
        for chi in range(1, residue.nchi() + 1):
            if random.random() >= self.torsion_fraction:
                continue
            shifted = decoy.chi(chi, self._resnum) + random.gauss(0.0, self.torsion_sigma_deg)
            decoy.set_chi(chi, self._resnum, shifted)
        return decoy

    def _rigid_move_by_coordinates(self, decoy: Any, rotation_deg: float, translation_A: float) -> None:
        """Fallback when the ligand has no jump of its own: rotate about its centroid, translate.

        Kept deliberately simple — a random axis, a Gaussian angle and a Gaussian
        displacement — because it exists only for fold trees the jump route cannot address.
        """
        from pyrosetta.rosetta.core.id import AtomID
        from pyrosetta.rosetta.numeric import xyzVector_double_t

        residue = decoy.residue(self._resnum)
        coords = ligand_heavy_coords(decoy, self._resnum)
        centre = coords.mean(axis=0)

        axis = np.array([random.gauss(0, 1) for _ in range(3)])
        axis /= np.linalg.norm(axis) or 1.0
        angle = np.deg2rad(random.gauss(0.0, rotation_deg))
        cross = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        matrix = (
            np.eye(3) * np.cos(angle)
            + np.sin(angle) * cross
            + (1 - np.cos(angle)) * np.outer(axis, axis)
        )
        shift = np.array([random.gauss(0.0, translation_A) for _ in range(3)])

        for atom_index in range(1, residue.natoms() + 1):
            if residue.is_virtual(atom_index):
                continue
            xyz = residue.xyz(atom_index)
            point = np.array([xyz.x, xyz.y, xyz.z])
            moved = matrix @ (point - centre) + centre + shift
            decoy.set_xyz(
                AtomID(atom_index, self._resnum),
                xyzVector_double_t(float(moved[0]), float(moved[1]), float(moved[2])),
            )

    def _propose_from_backend(self, seed: int, attempt: int) -> Any:
        """Stamp a backend-produced pose of the same molecule onto the ligand residue.

        Poses are requested once per seed and consumed one per attempt, because a docking run
        is expensive and returns a ranked list — asking again per attempt would repeat the
        search rather than walk down its results.
        """
        from atomfrust.dock.base import BackendUnavailable

        if not self.backend.available():
            raise BackendUnavailable(
                f"backend {self._route_name()!r} was requested but is not available on this "
                "machine; leave backend=None to use the perturbation route"
            )
        poses = self._backend_poses_cache.get(seed)
        if poses is None:
            native = self._native_ligand_mol()
            poses = list(
                self.backend.pose(
                    native,
                    self.receptor_pdb(self.context.pose),
                    reference_ligand=native,
                    n_poses=max(self.backend_poses, self.max_attempts),
                    seed=seed,
                )
            )
            self._backend_poses_cache[seed] = poses
        if not poses:
            raise ValueError(f"backend {self._route_name()!r} returned no poses for seed {seed}")

        chosen = poses[(attempt - 1) % len(poses)]
        decoy = self.context.pose.clone()
        self._apply_molblock(decoy, chosen.molblock)
        return decoy

    def _apply_molblock(self, decoy: Any, molblock: str) -> None:
        """Copy heavy-atom coordinates from a backend pose onto the ligand residue.

        Correspondence is by substructure match against the native molecule, not by atom
        order: a docking program may reorder atoms or add hydrogens, and matching by index
        would then scramble the molecule silently.
        """
        from rdkit import Chem
        from pyrosetta.rosetta.core.id import AtomID
        from pyrosetta.rosetta.numeric import xyzVector_double_t

        native = Chem.RemoveHs(Chem.Mol(self._native_ligand_mol()))
        probe = Chem.RemoveHs(Chem.MolFromMolBlock(molblock, removeHs=False))
        match = probe.GetSubstructMatch(native)
        if len(match) != native.GetNumAtoms():
            raise ValueError(
                "backend pose does not match the native molecule atom for atom; the pose "
                "axis requires the same molecule (a different molecule is axis D, "
                "decoys/chemotype.py)"
            )
        conformer = probe.GetConformer()

        residue = decoy.residue(self._resnum)
        heavy = [
            i
            for i in range(1, residue.natoms() + 1)
            if not residue.is_virtual(i) and residue.atom_type(i).element().strip() != "H"
        ]
        if len(heavy) != native.GetNumAtoms():
            raise ValueError(
                f"ligand has {len(heavy)} heavy atoms but the native molecule has "
                f"{native.GetNumAtoms()}"
            )
        for position, atom_index in enumerate(heavy):
            point = conformer.GetAtomPosition(match[position])
            decoy.set_xyz(
                AtomID(atom_index, self._resnum),
                xyzVector_double_t(float(point.x), float(point.y), float(point.z)),
            )
        if len(heavy) != residue.natoms():
            from pyrosetta.rosetta.core.conformation import idealize_hydrogens

            idealize_hydrogens(decoy.residue(self._resnum), decoy.conformation())

    # ------------------------------------------------------------------ relaxation

    def _pocket_shell(self, decoy: Any) -> list[int]:
        """Protein residues near the ligand in *either* pose, intersected with ``repack``.

        The union matters: the residues that must move are those the ligand has vacated as
        well as those it has arrived at, and a shell drawn around only one of the two would
        leave half of them holding rotamers packed against empty space.
        """
        native_xyz = ligand_heavy_coords(self.context.pose, self._resnum)
        decoy_xyz = ligand_heavy_coords(decoy, self._resnum)
        ligand_xyz = np.vstack([native_xyz, decoy_xyz])

        allowed = {
            node.pose_resnum
            for node, keep in zip(self.context.nodes, self.context.regions.repack)
            if keep and node.kind == "protein"
        }
        shell: list[int] = []
        for resnum in sorted(allowed):
            residue = decoy.residue(resnum)
            points = np.asarray(
                [
                    (residue.xyz(i).x, residue.xyz(i).y, residue.xyz(i).z)
                    for i in range(1, residue.nheavyatoms() + 1)
                ],
                dtype=float,
            ).reshape(-1, 3)
            if points.size == 0:
                continue
            deltas = points[:, None, :] - ligand_xyz[None, :, :]
            if np.sqrt((deltas**2).sum(axis=2)).min() <= self.repack_radius_A:
                shell.append(resnum)
        return shell

    def _relax_pocket(self, decoy: Any) -> None:
        """Repack the shell around the displaced ligand, then minimise its chi angles.

        The ligand is prevented from repacking and excluded from the movemap, so the
        coordinates that were proposed, RMSD'd and gated are the coordinates that get scored.
        Backbone and jumps are frozen, which is what keeps the axis-A invariant intact here.
        """
        import pyrosetta
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            OperateOnResidueSubset,
            PreventRepackingRLT,
            RestrictToRepacking,
        )
        from pyrosetta.rosetta.protocols.minimization_packing import (
            MinMover,
            PackRotamersMover,
        )
        from pyrosetta.rosetta.utility import vector1_bool

        shell = set(self._pocket_shell(decoy))
        if not shell:
            return

        total = decoy.total_residue()
        outside = vector1_bool(total)
        for k in range(1, total + 1):
            outside[k] = k not in shell

        scorefxn = pyrosetta.create_score_function(self.context.score_function)
        task_factory = TaskFactory()
        task_factory.push_back(RestrictToRepacking())
        task_factory.push_back(OperateOnResidueSubset(PreventRepackingRLT(), outside))
        packer = PackRotamersMover(scorefxn)
        packer.task_factory(task_factory)
        packer.apply(decoy)

        if not self.minimise:
            return
        movemap = MoveMap()
        movemap.set_bb(False)
        movemap.set_chi(False)
        movemap.set_jump(False)
        for resnum in sorted(shell):
            movemap.set_chi(resnum, True)
        mover = MinMover()
        mover.movemap(movemap)
        mover.score_function(scorefxn)
        mover.min_type("lbfgs_armijo_nonmonotone")
        mover.tolerance(0.01)
        mover.apply(decoy)

    def _seed_everything(self, seed: int) -> None:
        """Seed Python, NumPy and Rosetta from one number.

        ``RigidBodyPerturbMover`` draws from Rosetta's RNG and the torsion sampling from
        Python's, so both must be seeded or the decoy is reproducible in only one of its two
        halves. The decoy-level seed stays ``base_seed + decoy_id``; the attempt index enters
        as ``seed * 1000 + attempt`` so retries explore new poses instead of redrawing the
        rejected one. That composition is injective for ``max_attempts < 1000``, which is the
        only regime it is used in — beyond it two different decoys could share a stream.
        """
        import pyrosetta

        random.seed(seed)
        np.random.seed(seed % (2**32))
        pyrosetta.rosetta.numeric.random.rg().set_seed(int(seed))

    # ------------------------------------------------------------------ validity gate

    def _native_ligand_mol(self) -> Any:
        if self._native_mol is None:
            self._native_mol = ligand_mol_from_pose(self.context.pose, self._resnum)
        return self._native_mol

    def receptor_pdb(self, pose: Any) -> Path:
        """The receptor as the gate will see it: this pose, minus the ligand being moved.

        Written per call rather than cached because the pocket side chains differ between
        the native pose, a raw proposal and a relaxed decoy — a clash check against a stale
        receptor is a check against a structure that is not being scored. Only the moving
        component is removed; a second ligand or a cofactor stays in the receptor, where it
        is a legitimate clash partner.
        """
        if self._tempdir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="atomfrust_pose_axis_")
        path = Path(self._tempdir.name) / "receptor.pdb"
        full = Path(self._tempdir.name) / "complex.pdb"
        pose.dump_pdb(str(full))
        info = pose.pdb_info()
        chain = info.chain(self._resnum).strip()
        number = str(int(info.number(self._resnum)))
        kept = [
            line
            for line in full.read_text().splitlines()
            if not (
                line.startswith("HETATM")
                and line[21:22].strip() == chain
                and line[22:26].strip() == number
            )
        ]
        path.write_text("\n".join(kept) + "\n")
        return path

    def _as_pose3d(self, pose: Any, source: str) -> Any:
        from rdkit import Chem

        from atomfrust.dock.base import Pose3D

        native = self._native_ligand_mol()
        mol = ligand_mol_from_pose(pose, self._resnum)
        return Pose3D(
            molblock=Chem.MolToMolBlock(mol),
            source=source,
            rmsd_to_reference=ligand_rmsd(self.context.pose, pose, self._resnum),
            metadata={
                "pose_id": f"{self._node.node_id}/{source}",
                "smiles": Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(native))),
                "generator": f"pose/{self._route_name()}",
            },
        )

    def _verdict(self, decoy: Any) -> dict[str, Any]:
        """Run the S4.2 gate on one proposal and flatten the verdict into index columns.

        A gate that cannot run is reported as ``pose_checker="unavailable"`` with
        ``pose_valid=None`` — never as a pass. "We did not check" and "it passed" are
        different facts and only one of them may be counted towards S4.2.
        """
        from atomfrust.dock import check_poses

        try:
            pose3d = self._as_pose3d(decoy, source=f"pose/{self._route_name()}")
            qc = check_poses(
                [pose3d], self.receptor_pdb(decoy), checker=self.checker
            )
        except Exception as exc:  # RDKit or posebusters unavailable, unparseable residue type
            return {
                "pose_valid": None,
                "pose_checker": "unavailable",
                "pose_checks_run": 0,
                "pose_failed_checks": f"{type(exc).__name__}: {exc}",
            }

        ran = qc[~qc["skipped"].astype(bool)]
        failed = sorted(ran.loc[~ran["passed"].astype(bool), "check"].tolist())
        return {
            "pose_valid": bool(len(ran) > 0 and not failed),
            "pose_checker": str(qc["checker"].iloc[0]) if len(qc) else "unavailable",
            "pose_checks_run": int(len(ran)),
            "pose_failed_checks": ",".join(failed),
        }
