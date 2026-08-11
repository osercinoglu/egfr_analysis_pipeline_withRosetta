"""MUV — the second experimental-negative resource, and the reason it is not a decoy set.

17 targets, ~30 actives and ~15,000 **experimentally measured inactives** each (methods
document #21). MUV was built from PubChem BioAssay confirmatory screens, so a MUV "decoy" is
a compound that was put in front of the target and did not bind. That is a categorically
different object from a DUD-E property-matched decoy, and this adapter is the reason the
distinction survives: everything in ``cmp_list_MUV_<aid>_decoys.dat`` is emitted as
``measured_inactive``, never as ``property_decoy``.

Why that matters here: S3.4 and S5.8 exist to answer the circularity objection — that a
computational specificity measure validated against computational negatives proves little. A
MUV inactive is admissible evidence for that claim and a DUD-E decoy is not, so pooling them
would quietly convert the strongest experiment in the plan into the weakest.

**Layout.** MUV distributes two whitespace-delimited tables per target, named by PubChem
assay id::

    <root>/cmp_list_MUV_<aid>_actives.dat    PUBCHEM_SID  PUBCHEM_CID  smiles
    <root>/cmp_list_MUV_<aid>_decoys.dat     PUBCHEM_SID  PUBCHEM_CID  smiles

The target is the assay id (``466``, ``548``, ...), kept as the string MUV uses rather than
mapped to a protein name: the assay is the unit the labels belong to, and two assays against
the same protein are not interchangeable. A header row is skipped by
:func:`~atomfrust.chem.libraries.base.read_smiles_table`.

``source_id`` is the PubChem SID, the identifier MUV keys on; the CID is not retained,
because one field per record is what ``decoys/index.parquet`` stores as ``source_ref`` and a
second identifier would have to be reconstructed from the file anyway.

SMILES only — MUV ships no coordinates, so ``has_3d`` is always ``False`` and a docking
backend must generate conformers.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from atomfrust.chem.libraries.base import FileLibraryAdapter, MolRecord, Role, read_smiles_table

__all__ = ["MUVAdapter"]

_FILENAME = re.compile(r"cmp_list_MUV_(?P<aid>\w+)_(?P<kind>actives|decoys)\.dat$", re.IGNORECASE)

#: MUV's "decoys" are assayed non-binders. The mapping is one line and it is the whole point
#: of this module, so it is spelled out rather than inferred from the word "decoy".
_KIND_TO_ROLE: dict[str, Role] = {"actives": "active", "decoys": "measured_inactive"}


class MUVAdapter(FileLibraryAdapter):
    """Adapter over an unpacked MUV cache. Emits ``measured_inactive``, never ``property_decoy``."""

    name = "muv"

    def targets(self) -> list[str]:
        return sorted({aid for aid, _, _ in self._files()})

    def _files(self) -> list[tuple[str, Path, Role]]:
        if not self.root.is_dir():
            return []
        found: list[tuple[str, Path, Role]] = []
        for path in sorted(self.root.rglob("cmp_list_MUV_*.dat")):
            match = _FILENAME.search(path.name)
            if match is None or not path.is_file():
                continue
            found.append(
                (match.group("aid"), path, _KIND_TO_ROLE[match.group("kind").lower()])
            )
        return found

    def _iter_target_records(self, target: str) -> Iterator[MolRecord]:
        for aid, path, role in self._files():
            if aid != target:
                continue
            # PUBCHEM_SID, PUBCHEM_CID, smiles: the SMILES is the third column, not the first.
            for index, (smiles, _cid, fields) in enumerate(
                read_smiles_table(path, smiles_column=2, id_column=1)
            ):
                yield MolRecord(
                    smiles=smiles,
                    inchikey=self._inchikey(smiles),
                    source=self.name,
                    source_id=fields[0] if fields else f"MUV{target}_{index}",
                    role=role,
                    has_3d=False,
                    target=target,
                )
