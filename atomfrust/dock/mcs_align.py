"""MCS alignment onto the native ligand — the docking-free ablation, not a consolation prize.

Two jobs, and the second is the reason this module exists.

**It is the zero-binary backend.** It needs only RDKit, so ``atomfrust`` can place a
candidate molecule in a pocket on a machine with no smina, no GNINA and no GPU. That makes
the chemotype axis (G6) runnable everywhere.

**It is the control that tells us what the chemotype axis is measuring.** G6 asks whether
frustration separates real binders from decoy chemotypes. If the decoys are *docked*, a
positive result has two explanations that the experiment alone cannot distinguish: the
frustration index discriminates chemotypes, or the docking program does — it refuses to find
a good pose for a non-binder, and frustration merely reports the resulting bad geometry.
Aligning every candidate onto the native ligand by maximum common substructure removes the
docking program from the loop entirely: every molecule is placed by the same deterministic
geometric rule, with no scoring function and no search. If the chemotype signal survives
that, it is a property of the molecules and the energy function. If it collapses, the axis
was measuring the docking program. Reporting the chemotype axis under only one placement
method leaves that ambiguity in the result, so this backend is run *alongside* the docking
backends, not instead of them.

``receptor_pdb`` is accepted and ignored. That is the ablation stated in the signature: the
placement never consults the protein, so the pose can be sterically wrong — which is exactly
why :mod:`atomfrust.dock.posebusters` gates it like every other backend rather than trusting
it because it is "just an alignment".

Method: :func:`rdkit.Chem.rdFMCS.FindMCS` finds the common substructure; the candidate is
embedded with the matched atoms restrained to the reference's coordinates (a coordinate map
into ETKDG, then a force field with strong distance restraints to fixed points, then a rigid
alignment on the matched pairs). The result is a candidate whose shared scaffold sits where
the native scaffold sat and whose remaining atoms are in a low-energy conformation around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atomfrust.dock.base import (
    AllChem,
    Chem,
    Pose3D,
    as_mol_with_conformer,
    molblock_from_mol,
    rdkit_available,
    require_rdkit,
)

__all__ = ["MCSAlignBackend"]


class MCSAlignBackend:
    """Place a molecule by aligning its maximum common substructure onto a reference ligand.

    ``min_mcs_atoms`` is a floor, not a preference: an "alignment" onto two shared atoms
    fixes a bond direction and nothing else, so the resulting pose carries no information
    about the pocket and would silently pollute a chemotype comparison. Below the floor the
    molecule is reported as unposeable (``ValueError``) rather than posed badly.

    ``ring_matches_ring_only`` defaults to True so a ring atom never maps onto a chain atom.
    Without it the MCS happily matches an aliphatic chain into an aromatic ring, and the
    restrained embedding then has to bend the candidate into a shape the force field
    disagrees with, which shows up downstream as a bond-angle failure in the validity gate
    rather than as the bad correspondence it actually is.
    """

    name = "mcs_align"

    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        ring_matches_ring_only: bool = True,
        complete_rings_only: bool = False,
        min_mcs_atoms: int = 3,
        restrain: bool = True,
        max_embed_attempts: int = 4,
    ) -> None:
        self.timeout_s = timeout_s
        self.ring_matches_ring_only = ring_matches_ring_only
        self.complete_rings_only = complete_rings_only
        self.min_mcs_atoms = min_mcs_atoms
        self.restrain = restrain
        self.max_embed_attempts = max_embed_attempts

    def available(self) -> bool:
        """True iff RDKit imports. There is no binary to find, so this is the only question."""
        return rdkit_available()

    def pose(
        self,
        smiles_or_mol: str | Any,
        receptor_pdb: Path | str | None = None,
        reference_ligand: str | Path | Any | None = None,
        n_poses: int = 1,
        seed: int = 0,
    ) -> list[Pose3D]:
        """Align ``smiles_or_mol`` onto ``reference_ligand``. ``receptor_pdb`` is ignored.

        Returns up to ``n_poses`` poses, seeded ``seed + k``, ordered by MCS RMSD (the only
        quality signal available without a scoring function). ``score`` is ``None`` on every
        pose: this backend does not score, and a fabricated number would be averaged with
        smina's by some later report.
        """
        require_rdkit()
        if reference_ligand is None:
            raise ValueError(
                "MCSAlignBackend needs reference_ligand: it aligns onto the native ligand "
                "and has no other way to know where the pocket is"
            )
        reference = as_mol_with_conformer(reference_ligand)
        candidate = _sanitised(smiles_or_mol)

        mcs_smarts, cand_match, ref_match = self._match(candidate, reference)
        if len(cand_match) < self.min_mcs_atoms:
            raise ValueError(
                f"maximum common substructure has {len(cand_match)} atom(s), below "
                f"min_mcs_atoms={self.min_mcs_atoms}; this molecule shares too little with "
                "the reference for an alignment to place it meaningfully"
            )

        poses: list[Pose3D] = []
        for k in range(max(1, n_poses)):
            posed, rmsd, method = _restrained_embed(
                candidate,
                reference,
                list(zip(cand_match, ref_match)),
                seed=seed + k,
                restrain=self.restrain,
                max_attempts=self.max_embed_attempts,
            )
            if posed is None:
                continue
            poses.append(
                Pose3D(
                    molblock=molblock_from_mol(posed),
                    source=self.name,
                    score=None,
                    rmsd_to_reference=rmsd,
                    metadata={
                        "mcs_smarts": mcs_smarts,
                        "n_mcs_atoms": len(cand_match),
                        "mcs_rmsd": rmsd,
                        "placement": method,
                        "seed": seed + k,
                        "receptor_used": False,
                        "smiles": Chem.MolToSmiles(candidate),
                    },
                )
            )
        if not poses:
            raise ValueError(
                "restrained embedding failed for every attempt; the MCS correspondence is "
                "probably geometrically inconsistent with the candidate"
            )
        poses.sort(key=lambda p: (p.rmsd_to_reference is None, p.rmsd_to_reference))
        return poses

    def _match(self, candidate: Any, reference: Any) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
        from rdkit.Chem import rdFMCS

        result = rdFMCS.FindMCS(
            [Chem.RemoveHs(Chem.Mol(candidate)), Chem.RemoveHs(Chem.Mol(reference))],
            timeout=int(self.timeout_s),
            ringMatchesRingOnly=self.ring_matches_ring_only,
            completeRingsOnly=self.complete_rings_only,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )
        if not result.smartsString:
            raise ValueError("no common substructure between candidate and reference ligand")
        pattern = Chem.MolFromSmarts(result.smartsString)
        cand_match = candidate.GetSubstructMatch(pattern)
        ref_match = reference.GetSubstructMatch(pattern)
        if not cand_match or not ref_match:
            # FindMCS reports a SMARTS its own matcher then fails to re-find; rare, and a
            # silent empty tuple here would become an unrestrained embed placed nowhere near
            # the pocket.
            raise ValueError(
                f"MCS SMARTS {result.smartsString!r} did not re-match both molecules"
            )
        return result.smartsString, cand_match, ref_match


def _sanitised(smiles_or_mol: str | Any) -> Any:
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES {smiles_or_mol!r}")
        return mol
    mol = Chem.Mol(smiles_or_mol)
    Chem.SanitizeMol(mol)
    return mol


def _restrained_embed(
    candidate: Any,
    reference: Any,
    pairs: list[tuple[int, int]],
    *,
    seed: int,
    restrain: bool,
    max_attempts: int,
) -> tuple[Any | None, float | None, str]:
    """Embed ``candidate`` with the matched atoms held at the reference's coordinates.

    Three stages, each recoverable from the failure of the previous one:

    1. ETKDG with a ``coordMap`` — distance geometry is told about the target positions, so
       the embedding starts close instead of being dragged there afterwards.
    2. A force field with the matched atoms restrained to *fixed extra points* at the
       reference coordinates. This repairs the bond lengths and angles that the coordinate
       map distorts, which is what keeps the pose inside the validity gate's tolerances.
    3. A rigid alignment on the matched pairs, which removes any residual drift of the
       scaffold as a whole while leaving the internal geometry the force field produced.

    Stage 1 failing falls back to a free embedding plus stage 3 (``placement="align_only"``);
    the pose is then only rigidly superimposed and its RMSD reports how badly that fits.
    """
    mol_h = Chem.AddHs(Chem.Mol(candidate))  # AddHs appends, so heavy indices survive
    ref_conf = reference.GetConformer()
    coord_map = {cand_idx: ref_conf.GetAtomPosition(ref_idx) for cand_idx, ref_idx in pairs}

    conf_id = -1
    for attempt in range(max(1, max_attempts)):
        try:
            conf_id = AllChem.EmbedMolecule(
                mol_h,
                coordMap=coord_map,
                randomSeed=seed + attempt,
                useRandomCoords=attempt > 0,
                ignoreSmoothingFailures=True,
            )
        except Exception:
            conf_id = -1
        if conf_id >= 0:
            break

    method = "coord_map"
    if conf_id < 0:
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol_h, params)
        except Exception:
            conf_id = -1
        if conf_id < 0:
            return None, None, "failed"
        method = "align_only"

    if restrain and method == "coord_map":
        _restrained_minimise(mol_h, ref_conf, pairs)

    try:
        from rdkit.Chem import rdMolAlign

        rdMolAlign.AlignMol(mol_h, reference, atomMap=pairs)
    except Exception:
        pass

    return mol_h, _mcs_rmsd(mol_h, reference, pairs), method


def _restrained_minimise(mol_h: Any, ref_conf: Any, pairs: list[tuple[int, int]]) -> None:
    """MMFF (UFF fallback) with each matched atom tied to a fixed point at its target.

    Extra points rather than position constraints because the restraint has to be to a
    coordinate in the *reference's* frame, which is not a property of any atom in this
    molecule. A failure here is not fatal: the coordinate-mapped embedding is already
    approximately right, and the validity gate is what decides whether it is right enough.
    """
    try:
        properties = AllChem.MMFFGetMoleculeProperties(mol_h)
        field = (
            AllChem.MMFFGetMoleculeForceField(mol_h, properties)
            if properties is not None
            else AllChem.UFFGetMoleculeForceField(mol_h)
        )
        if field is None:
            return
        for cand_idx, ref_idx in pairs:
            position = ref_conf.GetAtomPosition(ref_idx)
            point = field.AddExtraPoint(position.x, position.y, position.z, fixed=True) - 1
            field.AddDistanceConstraint(point, cand_idx, 0.0, 0.0, 100.0)
        field.Initialize()
        field.Minimize(maxIts=400, energyTol=1e-6, forceTol=1e-4)
    except Exception:
        return


def _mcs_rmsd(posed: Any, reference: Any, pairs: list[tuple[int, int]]) -> float | None:
    """RMSD over the matched atoms only — the quantity the alignment actually controls.

    A whole-molecule RMSD would be undefined (the molecules differ) and a whole-molecule
    best-fit RMSD would hide a scaffold that landed in the wrong place.
    """
    try:
        posed_conf = posed.GetConformer()
        ref_conf = reference.GetConformer()
    except Exception:
        return None
    total = 0.0
    for cand_idx, ref_idx in pairs:
        a = posed_conf.GetAtomPosition(cand_idx)
        b = ref_conf.GetAtomPosition(ref_idx)
        total += (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
    return float((total / len(pairs)) ** 0.5) if pairs else None
