"""Axis A — the protein-side decoy: sequence randomisation on a fixed backbone.

Three orthogonal switches replace the prototype's single hardcoded protocol
(``frustration.py:298-380``):

============  ==================================================  ==========================
switch        values                                              controls
============  ==================================================  ==========================
``scope``     ``whole_protein`` · ``contact_shell`` · ``region``   *which* residues change
``identity``  ``native`` · ``composition`` · ``uniform20``         *what* identities exist
``placement`` ``permute`` · ``inplace``                            *where* they sit
============  ==================================================  ==========================

Step A4 established what the paper actually does: *"we randomly shuffle the protein
sequence and then repack the resulting sequence onto the backbone"* — a **permutation**,
which preserves the native amino-acid multiset exactly. That is
``identity=native, placement=permute``, the default here.

The prototype instead drew i.i.d. from the native composition
(``np.random.choice(aa_letters, p=aa_probs)``, ``frustration.py:337``), which lets the
composition of each decoy fluctuate around the native one. That is
``identity=composition, placement=inplace`` — available, but no longer the default.

``protocol="legacy"`` reproduces the prototype exactly, including its RNG consumption order
and its sequential ``MutateResidue`` application, so stored results remain checkable. It is
kept for that purpose only.

**`mutation` is not a speed switch.** The plan justified a single ``PackerTask`` as replacing
~300 sequential ``MutateResidue`` calls. Measured on 5GMP that buys **nothing**: 4.4 s per
decoy either way, because packing and minimisation dominate and the mutations do not. The two
paths also do not agree exactly — they design the identical sequence (305/305 positions) but
land in different rotamer minima, so a handful of pairs differ by enough to matter
(``E_ij`` max |Δ| ≈ 9-11 with ``fa_rep`` excluded, r ≈ 0.998, median |Δ| = 0).
``sequential`` is therefore the default. ``packer_task`` is kept because it is the **only**
way to express a restricted repack or frozen region, which is what region-focused decoys
(C7 / user request 4) require — not because it is faster.

The backbone invariant is enforced as a runtime post-condition, not merely tested:
``MutateResidue`` rebuilds the carbonyl O from idealised geometry and drifts ~0.5-0.8 A from
the crystallographic position, so N/CA/C/O are restored from the native pose after packing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from atomfrust.decoys.base import DecoyContext, DecoyResult, extract_energies

__all__ = [
    "IdentityDecoyGenerator",
    "native_aa_frequency",
    "assert_backbone_identical",
    "BACKBONE_ATOMS",
]

BACKBONE_ATOMS = ("N", "CA", "C", "O")

Scope = Literal["whole_protein", "contact_shell", "region"]
Identity = Literal["native", "composition", "uniform20"]
Placement = Literal["permute", "inplace"]

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def native_aa_frequency(pose: Any) -> dict[str, float]:
    """Position-independent amino-acid frequency of this protein.

    Insertion order is first-appearance order in the pose. That is not cosmetic: the
    prototype builds its sampling alphabet as ``list(aa_freq.keys())``
    (``frustration.py:329``), so the dict's order determines which letter each
    ``np.random.choice`` draw maps to. Reproducing legacy decoys requires reproducing this
    order exactly.
    """
    counts: dict[str, int] = {}
    total = 0
    for i in range(1, pose.total_residue() + 1):
        residue = pose.residue(i)
        if not residue.is_protein():
            continue
        letter = residue.name1()
        counts[letter] = counts.get(letter, 0) + 1
        total += 1
    return {aa: n / total for aa, n in counts.items()} if total else {}


def assert_backbone_identical(native: Any, decoy: Any, tol: float = 1e-6) -> float:
    """Raise if any backbone atom moved. Returns the maximum deviation.

    The prototype guarded this with a 0.05 A test
    (``test_frustration.py::test_decoy_backbone_unchanged``). Here it is a post-condition on
    every decoy, because a drifting backbone silently changes what the Z-score measures:
    the whole construction assumes native and decoy differ in sequence *at fixed geometry*.
    """
    worst = 0.0
    for i in range(1, native.total_residue() + 1):
        native_residue = native.residue(i)
        if not native_residue.is_protein():
            continue
        decoy_residue = decoy.residue(i)
        for atom in BACKBONE_ATOMS:
            if not (native_residue.has(atom) and decoy_residue.has(atom)):
                continue
            a = native_residue.xyz(atom)
            b = decoy_residue.xyz(atom)
            worst = max(worst, float((a - b).norm()))
    if worst > tol:
        raise AssertionError(
            f"decoy backbone moved by {worst:.4f} A (tolerance {tol:g}). "
            "The Z-score assumes native and decoy differ in sequence at fixed geometry."
        )
    return worst


@dataclass
class IdentityDecoyGenerator:
    """Axis A generator."""

    context: DecoyContext
    scope: Scope = "whole_protein"
    identity: Identity = "native"
    placement: Placement = "permute"
    base_seed: int = 42
    protocol: Literal["modern", "legacy"] = "modern"
    mutation: Literal["packer_task", "sequential"] = "sequential"
    seeding: Literal["substream", "stream"] = "substream"
    relax: Literal["min", "mc", "none"] = "min"
    mc_cycles: int = 10
    mc_kT: float = 0.8
    native_repack: bool = True
    strict_backbone: bool = True
    axis: str = "identity"

    def __post_init__(self) -> None:
        self._native_frequency = native_aa_frequency(self.context.pose)
        if self.mutation == "sequential" and self._regions_are_restricted():
            raise ValueError(
                "mutation='sequential' repacks the whole pose and cannot honour a "
                "restricted repack/frozen region — the prototype had no per-residue packer "
                "operation at all (frustration.py:347). Use mutation='packer_task' for "
                "region-focused decoys."
            )

    def _regions_are_restricted(self) -> bool:
        """True when some protein residue is excluded from repacking."""
        regions = self.context.regions
        protein = np.array([n.kind == "protein" for n in self.context.nodes])
        return bool((protein & ~regions.repack).any())

    # ------------------------------------------------------------ positions

    def target_positions(self) -> list[int]:
        """Pose residue numbers whose identity may change.

        ``whole_protein`` is the prototype's behaviour and mutates every protein residue.
        Parameter #7 of the methods document says "contacting residues", which is
        ``contact_shell`` — the plan records `scope` as the real deviation from the paper,
        not `identity`.
        """
        nodes = self.context.nodes
        mutable = self.context.regions.mutate
        if self.scope == "whole_protein":
            return [n.pose_resnum for n in nodes if n.kind == "protein" and n.mutable]
        if self.scope in ("contact_shell", "region"):
            return [n.pose_resnum for n, keep in zip(nodes, mutable) if keep]
        raise ValueError(f"unknown scope {self.scope!r}")

    # ------------------------------------------------------------ sequences

    def _position_rng(self, seed: int, decoy_id: int, position: int) -> np.random.Generator:
        """An independent stream for one position of one decoy.

        Without this, identities come from a single stream consumed in iteration order, so
        the identity at a position depends on how many residues precede it in the mutate
        set. Comparing `scope=whole_protein` against `scope=contact_shell` would then differ
        in *sequence* as well as in scope, and the comparison would be unpaired.

        The decoy-level seed stays ``base_seed + decoy_id`` regardless — that is what makes
        an N-decoy prefix exact and results independent of worker count.
        """
        return np.random.default_rng(
            np.random.SeedSequence(entropy=seed, spawn_key=(decoy_id, position))
        )

    def draw_sequence(
        self,
        positions: list[int],
        rng: np.random.Generator,
        decoy_id: int = 0,
        seed: int | None = None,
    ) -> list[str]:
        """One-letter identities for ``positions``, per the identity/placement switches."""
        if (
            self.seeding == "substream"
            and self.placement == "inplace"
            and self.identity in ("composition", "uniform20")
        ):
            base = self.base_seed if seed is None else seed
            return [
                self._draw_one(self._position_rng(base, decoy_id, position))
                for position in positions
            ]
        if self.identity == "native":
            native = [self.context.pose.residue(i).name1() for i in positions]
            if self.placement == "permute":
                order = rng.permutation(len(native))
                return [native[k] for k in order]
            # inplace with the native multiset and no permutation is the native sequence;
            # legal, and exactly the repack-only control.
            return native

        if self.identity == "composition":
            letters = list(self._native_frequency.keys())
            probabilities = list(self._native_frequency.values())
            drawn = rng.choice(letters, size=len(positions), p=probabilities)
            return list(drawn)

        if self.identity == "uniform20":
            return list(rng.choice(list(AA_ALPHABET), size=len(positions)))

        raise ValueError(f"unknown identity mode {self.identity!r}")

    def _draw_one(self, rng: np.random.Generator) -> str:
        """A single identity from an independent stream. Only meaningful for the
        position-independent modes: a permutation is a property of the whole set, so
        `placement="permute"` cannot be decomposed this way and keeps a single stream."""
        if self.identity == "composition":
            return str(
                rng.choice(
                    list(self._native_frequency.keys()),
                    p=list(self._native_frequency.values()),
                )
            )
        return str(rng.choice(list(AA_ALPHABET)))

    # ------------------------------------------------------------- generate

    def generate(self, decoy_id: int) -> DecoyResult:
        import time

        started = time.perf_counter()
        seed = self.base_seed + decoy_id
        if self.protocol == "legacy":
            decoy, n_mutated = self._generate_legacy(seed)
        else:
            decoy, n_mutated = self._generate_modern(seed)

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
                "generator": f"identity/{self.protocol}",
                "scope": self.scope,
                "identity": self.identity,
                "placement": self.placement,
                "seed": seed,
                "n_mutated": n_mutated,
                "backbone_rmsd": drift,
                "relax": self.relax,
                "repack_residues": int(self.context.regions.repack.sum()),
                "wall_s": time.perf_counter() - started,
                "status": "ok",
            },
        )

    def prepare_native(self) -> Any:
        """The pose the native energies should be read from.

        Chen et al. state the native contact energies are *"obtained in a similar fashion by
        omitting the shuffling step"* (step A4) — i.e. the native is repacked and relaxed
        under the same protocol as the decoys, only without the sequence shuffle. The
        prototype instead scored the crystal pose as deposited
        (``frustration.py:612-621``), so native and decoys were treated asymmetrically and
        every repacking gain landed entirely in the native-decoy gap.

        With ``native_repack=False`` this returns the crystal pose unchanged, which is what
        the prototype did and what reproducing it requires.
        """
        if not self.native_repack:
            return self.context.pose

        import pyrosetta

        self._seed_everything(self.base_seed)
        native = self.context.pose.clone()
        scorefxn = pyrosetta.create_score_function(self.context.score_function)
        positions = self.target_positions()
        sequence = [native.residue(i).name1() for i in positions]  # unchanged: no shuffle
        if self.mutation == "packer_task":
            self._design_in_one_pass(native, positions, sequence, scorefxn)
        else:
            from pyrosetta.rosetta.core.pack.task import TaskFactory
            from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
            from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

            task_factory = TaskFactory()
            task_factory.push_back(RestrictToRepacking())
            packer = PackRotamersMover(scorefxn)
            packer.task_factory(task_factory)
            packer.apply(native)
        self._relax(native, scorefxn)
        self._restore_backbone(native)
        return native

    # ----------------------------------------------------------- protocols

    def _seed_everything(self, seed: int) -> None:
        import pyrosetta

        random.seed(seed)
        np.random.seed(seed)
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

    def _generate_modern(self, seed: int) -> tuple[Any, int]:
        """Draw identities, apply them, repack, relax, restore the backbone."""
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
        from pyrosetta.rosetta.protocols.minimization_packing import (
            PackRotamersMover,
        )
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
        import pyrosetta

        self._seed_everything(seed)
        rng = np.random.default_rng(seed)

        decoy = self.context.pose.clone()
        positions = self.target_positions()
        sequence = self.draw_sequence(
            positions, rng, decoy_id=seed - self.base_seed, seed=self.base_seed
        )
        n_mutated = sum(
            1
            for position, letter in zip(positions, sequence)
            if decoy.residue(position).name1() != letter
        )

        scorefxn = pyrosetta.create_score_function(self.context.score_function)
        if self.mutation == "packer_task":
            self._design_in_one_pass(decoy, positions, sequence, scorefxn)
        else:
            self._mutate_sequentially(decoy, positions, sequence)
            task_factory = TaskFactory()
            task_factory.push_back(RestrictToRepacking())
            packer = PackRotamersMover(scorefxn)
            packer.task_factory(task_factory)
            packer.apply(decoy)

        self._relax(decoy, scorefxn)
        self._restore_backbone(decoy)
        return decoy, n_mutated

    def _generate_legacy(self, seed: int) -> tuple[Any, int]:
        """Bit-for-bit reproduction of ``frustration.generate_decoy``.

        Kept so stored results stay checkable. Every detail matters: the alphabet comes from
        ``list(aa_freq.keys())`` in first-appearance order, ``np.random.choice`` is called
        once **per residue** (not once for a vector), and each mutation is applied to the
        pose immediately so later draws see earlier mutations.
        """
        import pyrosetta
        from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code, name_from_aa
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

        self._seed_everything(seed)

        decoy = self.context.pose.clone()
        aa_letters = list(self._native_frequency.keys())
        aa_probs = list(self._native_frequency.values())

        n_mutated = 0
        for i in range(1, decoy.total_residue() + 1):
            if not decoy.residue(i).is_protein():
                continue
            letter = np.random.choice(aa_letters, p=aa_probs)
            MutateResidue(i, name_from_aa(aa_from_oneletter_code(letter))).apply(decoy)
            n_mutated += 1


        scorefxn = pyrosetta.create_score_function(self.context.score_function)
        task_factory = TaskFactory()
        task_factory.push_back(RestrictToRepacking())
        packer = PackRotamersMover(scorefxn)
        packer.task_factory(task_factory)
        packer.apply(decoy)

        # The prototype always did exactly one chi-only MinMover pass, whatever `relax` says.
        self._minimise(decoy, scorefxn)
        self._restore_backbone(decoy)
        return decoy, n_mutated

    def _mutate_sequentially(self, decoy: Any, positions: list[int], sequence: list[str]) -> None:
        """One ``MutateResidue`` per position — the prototype's mechanism."""
        from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code, name_from_aa
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

        for position, letter in zip(positions, sequence):
            if decoy.residue(position).name1() == letter:
                continue
            MutateResidue(position, name_from_aa(aa_from_oneletter_code(letter))).apply(decoy)

    def _design_in_one_pass(
        self, decoy: Any, positions: list[int], sequence: list[str], scorefxn: Any
    ) -> None:
        """Design and repack in a single ``PackRotamersMover`` call.

        Replaces ~300 sequential ``MutateResidue`` applications plus a separate whole-pose
        repack (``frustration.py:342-350``). Positions are grouped by their target identity,
        so the task carries at most 20 operations rather than one per residue.

        This also makes the regions real: only ``repack`` is repacked and ``frozen`` is
        prevented outright, which the prototype could not express — it pushed a single
        whole-pose ``RestrictToRepacking()`` with no per-residue operation anywhere.
        """
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            OperateOnResidueSubset,
            PreventRepackingRLT,
            RestrictAbsentCanonicalAASRLT,
            RestrictToRepackingRLT,
        )
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
        from pyrosetta.rosetta.utility import vector1_bool

        total = decoy.total_residue()

        def subset(resnums) -> Any:
            mask = vector1_bool(total)
            for k in range(1, total + 1):
                mask[k] = False
            for resnum in resnums:
                mask[int(resnum)] = True
            return mask

        task_factory = TaskFactory()

        by_letter: dict[str, list[int]] = {}
        for position, letter in zip(positions, sequence):
            by_letter.setdefault(letter, []).append(position)
        for letter, group in sorted(by_letter.items()):
            restrict = RestrictAbsentCanonicalAASRLT()
            restrict.aas_to_keep(letter)
            task_factory.push_back(OperateOnResidueSubset(restrict, subset(group)))

        nodes = self.context.nodes
        regions = self.context.regions
        mutating = set(positions)
        repack_only = [
            n.pose_resnum
            for n, keep in zip(nodes, regions.repack)
            if keep and n.pose_resnum not in mutating
        ]
        if repack_only:
            task_factory.push_back(
                OperateOnResidueSubset(RestrictToRepackingRLT(), subset(repack_only))
            )
        frozen = [n.pose_resnum for n, keep in zip(nodes, regions.frozen) if keep]
        if frozen:
            task_factory.push_back(
                OperateOnResidueSubset(PreventRepackingRLT(), subset(frozen))
            )

        packer = PackRotamersMover(scorefxn)
        packer.task_factory(task_factory)
        packer.apply(decoy)

    def _relax(self, pose: Any, scorefxn: Any) -> None:
        if self.relax == "min":
            self._minimise(pose, scorefxn)
        elif self.relax == "mc":
            self._monte_carlo(pose, scorefxn)
        elif self.relax != "none":
            raise ValueError(f"unknown relax mode {self.relax!r}")

    def _monte_carlo(self, pose: Any, scorefxn: Any) -> None:
        """Short side-chain Monte-Carlo at fixed backbone.

        Chen et al. say only *"A short Monte-Carlo relaxation is then performed to better
        eliminate many of the possible side-chain clashes with the backbone fixed"* — no
        cycle count, no move set, no temperature. This implements the literal reading with
        the pieces Rosetta provides: each cycle re-anneals the rotamers (``PackRotamersMover``
        is itself a simulated annealing over rotamers, drawing fresh from the RNG) and then
        minimises chi, with a Metropolis test on the result; the lowest-energy pose is
        recovered at the end.

        The interpretation is recorded here rather than buried, because the paper does not
        specify it and any comparison against published numbers inherits the choice.
        """
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
        from pyrosetta.rosetta.protocols.moves import MonteCarlo

        task_factory = TaskFactory()
        task_factory.push_back(RestrictToRepacking())
        packer = PackRotamersMover(scorefxn)
        packer.task_factory(task_factory)

        monte_carlo = MonteCarlo(pose, scorefxn, self.mc_kT)
        for _ in range(self.mc_cycles):
            packer.apply(pose)
            self._minimise(pose, scorefxn)
            monte_carlo.boltzmann(pose)
        monte_carlo.recover_low(pose)

    def _minimise(self, decoy: Any, scorefxn: Any) -> None:
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.minimization_packing import MinMover

        movemap = MoveMap()
        movemap.set_bb(False)
        movemap.set_chi(True)
        movemap.set_jump(False)
        mover = MinMover()
        mover.movemap(movemap)
        mover.score_function(scorefxn)
        mover.min_type("lbfgs_armijo_nonmonotone")
        mover.tolerance(0.01)
        mover.apply(decoy)

    def _restore_backbone(self, decoy: Any) -> None:
        """Copy native N/CA/C/O verbatim.

        ``MutateResidue`` preserves N/CA/C but rebuilds the carbonyl O from idealised bond
        geometry, drifting ~0.5-0.8 A from the crystallographic position. This is not a
        minimiser artifact and no movemap setting prevents it.
        """
        from pyrosetta.rosetta.core.id import AtomID

        native = self.context.pose
        for i in range(1, native.total_residue() + 1):
            native_residue = native.residue(i)
            if not native_residue.is_protein():
                continue
            decoy_residue = decoy.residue(i)
            for atom in BACKBONE_ATOMS:
                if native_residue.has(atom) and decoy_residue.has(atom):
                    decoy.set_xyz(
                        AtomID(decoy_residue.atom_index(atom), i), native_residue.xyz(atom)
                    )
