"""DEKOIS 2.0 — the external hold-out, deliberately built by someone else's rules.

81 targets, 40 actives and 1,200 decoys each (methods document #20). Its value to this
project is not that it is better than DUD-E but that its decoy construction is *independent*:
DEKOIS matches on a different property set with a different dissimilarity criterion, so a
result that survives both is not an artefact of one construction. Tier 5 uses it once, at the
end of WP5 (S5.8), which is a discipline this adapter cannot enforce and the run manifest can.

Every DEKOIS molecule is a ``property_decoy`` or an ``active``. **There are no measured
inactives here** — for those, see :mod:`.muv` and DUD-E's experimental set.

**Layout is matched by name, tolerantly, because DEKOIS's own naming has moved.** Releases
have shipped ``<TARGET>_Celling-v1.12_decoyset.sdf``, ``<TARGET>_decoys.sdf`` and
``<TARGET>.sdf`` for the actives, sometimes one directory per target and sometimes flat::

    <root>/<TARGET>*decoy*.sdf     -> property_decoy
    <root>/<TARGET>*active*.sdf    -> active
    <root>/<TARGET>.sdf            -> active   (the bare-stem convention)
    <root>/<TARGET>/<same files>   -> same, per-target directory layout

The target name is the filename stem up to the first ``_``, or the directory name when there
is one. Matching is case-insensitive on the role token and case-*preserving* on the target,
since DEKOIS names targets in upper case (``COX1``) where DUD-E uses lower (``pgh1``) and
silently normalising them would fuse two different targets from two different libraries into
one row of a summary table.

DEKOIS ships SDF with coordinates, so ``has_3d`` is usually ``True`` — read from the z column
rather than the dimension tag, and still only a conformer, not a pose (see :mod:`.base`).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from atomfrust.chem.libraries.base import FileLibraryAdapter, MolRecord, Role, read_sdf

__all__ = ["DEKOIS2Adapter"]

_DECOY_TOKEN = re.compile(r"decoy", re.IGNORECASE)
_ACTIVE_TOKEN = re.compile(r"active|ligand", re.IGNORECASE)


class DEKOIS2Adapter(FileLibraryAdapter):
    """Adapter over an unpacked DEKOIS 2.0 cache. See the module docstring for the layout."""

    name = "dekois2"

    def targets(self) -> list[str]:
        return sorted({target for target, _, _ in self._files()})

    def _files(self) -> list[tuple[str, Path, Role]]:
        """``(target, path, role)`` for every SDF under the root, flat or per-target."""
        if not self.root.is_dir():
            return []
        found: list[tuple[str, Path, Role]] = []
        for path in sorted(self.root.rglob("*.sdf*")):
            if not path.is_file() or path.suffix not in (".sdf", ".gz"):
                continue
            if path.suffix == ".gz" and not path.name.lower().endswith(".sdf.gz"):
                continue
            found.append((self._target_of(path), path, self._role_of(path)))
        return found

    def _target_of(self, path: Path) -> str:
        parent = path.parent
        if parent != self.root and parent.is_dir():
            return parent.name
        stem = path.name.split(".")[0]
        return stem.split("_")[0]

    @staticmethod
    def _role_of(path: Path) -> Role:
        """Decoy wins over active: ``COX1_actives_decoyset.sdf`` is a decoy file whose name
        mentions the actives it was matched to, and reading it as actives would invert the
        labels of a whole target."""
        if _DECOY_TOKEN.search(path.name):
            return "property_decoy"
        if _ACTIVE_TOKEN.search(path.name):
            return "active"
        return "active"

    def _iter_target_records(self, target: str) -> Iterator[MolRecord]:
        for name, path, role in self._files():
            if name != target:
                continue
            for index, (title, smiles, has_3d, data) in enumerate(read_sdf(path)):
                yield MolRecord(
                    smiles=smiles,
                    inchikey=self._inchikey(smiles),
                    source=self.name,
                    source_id=title or f"{target}_{role}_{index}",
                    role=role,
                    has_3d=has_3d,
                    target=target,
                    properties=data,
                )
