"""Poses that already exist — crystal ligands, a co-folding output, a colleague's SDF.

The backend for input that was posed by something outside this package. It generates nothing:
it hands the coordinates through unchanged and lets them be scored, RMSD'd and — this is why
it is a backend rather than a special case in the caller — **gated by PoseBusters like every
other source**. A crystal pose is not exempt from the validity gate, and an AlphaFold3 or
Boltz-2 complex is exactly the kind of input the gate exists to catch (methods §3.7).

**Pass-through is byte-exact.** A molblock in is the same molblock out, character for
character. Parsing and re-serialising would silently normalise formatting, charge blocks and
coordinate precision, so a comparison of "what we analysed" against "what the generator
produced" could no longer be made with a diff. Validation therefore parses a *copy*.

**Available everywhere.** There is no binary and no search, so :meth:`available` is always
True. RDKit is needed only to validate and to compute an RMSD; without it the pass-through
still works and those fields are simply ``None``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atomfrust.dock.base import (
    Chem,
    Pose3D,
    as_mol_with_conformer,
    iter_sdf_records,
    molblock_from_mol,
    rdkit_available,
    symmetry_rmsd,
)

__all__ = ["PrePosedBackend"]


class PrePosedBackend:
    """Wrap already-posed coordinates as :class:`~atomfrust.dock.base.Pose3D`.

    Accepts a molblock string, a path to a ``.mol``/``.sdf`` (several records give several
    poses), or an RDKit ``Mol`` that carries a conformer. A SMILES string is refused: it has
    no coordinates, and silently embedding one would turn "this is the pose that was given to
    me" into "this is a pose I invented", which is the one thing this backend must not do.
    """

    name = "preposed"

    def __init__(self, *, validate: bool = True) -> None:
        self.validate = validate

    def available(self) -> bool:
        """Always True — no binary, no library, nothing to probe."""
        return True

    def pose(
        self,
        smiles_or_mol: str | Any,
        receptor_pdb: Path | str | None = None,
        reference_ligand: str | Path | Any | None = None,
        n_poses: int = 1,
        seed: int = 0,
    ) -> list[Pose3D]:
        """Return the supplied pose(s) unchanged. ``receptor_pdb`` and ``seed`` are ignored.

        They stay in the signature because the caller is a loop over backends that cannot
        know which one it holds; ``metadata["passthrough"]`` is True so a report can tell a
        generated pose from a given one.
        """
        molblocks = _molblocks_of(smiles_or_mol)
        if not molblocks:
            raise ValueError("PrePosedBackend received nothing that contains 3D coordinates")

        reference = (
            as_mol_with_conformer(reference_ligand)
            if reference_ligand is not None and rdkit_available()
            else None
        )

        poses: list[Pose3D] = []
        for index, molblock in enumerate(molblocks[: max(1, n_poses)]):
            rmsd = None
            if self.validate and rdkit_available():
                mol = Chem.MolFromMolBlock(molblock, removeHs=False)
                if mol is None:
                    raise ValueError(
                        f"record {index} is not a parseable molblock; refusing to pass "
                        "unparseable coordinates into the validity gate"
                    )
                if mol.GetNumConformers() == 0:
                    raise ValueError(f"record {index} has no conformer")
                if reference is not None:
                    rmsd = symmetry_rmsd(mol, reference)
            poses.append(
                Pose3D(
                    molblock=molblock,
                    source=self.name,
                    score=None,
                    rmsd_to_reference=rmsd,
                    metadata={"passthrough": True, "record": index},
                )
            )
        return poses


def _molblocks_of(source: str | Path | Any) -> list[str]:
    """Molblock text for each input record, without a parse-and-reserialise round trip."""
    if source is None:
        return []

    if isinstance(source, (str, Path)):
        text = str(source)
        if "M  END" in text:
            return list(iter_sdf_records(text)) or [text]
        path = Path(text)
        if path.exists():
            content = path.read_text()
            if "M  END" not in content:
                raise ValueError(f"{path.name} holds no MDL records")
            return list(iter_sdf_records(content)) or [content]
        raise ValueError(
            f"PrePosedBackend cannot pose {text!r}: it is not a molblock and not an existing "
            "file. A SMILES string has no coordinates — use a docking backend or "
            "MCSAlignBackend to create them."
        )

    if not rdkit_available():
        raise ValueError("an RDKit Mol was supplied but RDKit is not importable")
    mol = Chem.Mol(source)
    if mol.GetNumConformers() == 0:
        raise ValueError("the supplied Mol carries no conformer, so it is not a pose")
    return [molblock_from_mol(mol)]
