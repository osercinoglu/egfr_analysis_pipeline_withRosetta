"""DUD-E — the field-standard property-matched decoy set, and its experimental negatives.

102 targets, 22,886 clustered actives, 50 property-matched decoys per active (methods
document #11, #13). Decoys are drawn to match six physicochemical properties of the actives
while being topologically dissimilar under ECFP4, retaining the most dissimilar 25% (#15).

**Two roles come out of one directory, and conflating them would break S3.4.** The
``*_final.ism`` decoys are *synthetic*: matched by construction, assayed by nobody. DUD-E
separately carries 9,219 compounds with **measured** absence of affinity up to 30 µM, of
which 1,070 belong to COX-1/PGH1 — the largest such set and the one WP3 ranks actives
against (#14). This adapter emits the first as ``property_decoy`` and the second as
``measured_inactive``, and :meth:`FileLibraryAdapter.records` will not mix them unless asked.

**Layout.** DUD-E's distribution is one directory per target, either directly under the
cache root or under an ``all/`` subdirectory depending on whether the per-target tarballs or
the combined ``all.tar.gz`` was unpacked; both are accepted::

    <root>/<target>/actives_final.ism        SMILES  ChEMBL-id  [ligand-id]   -> active
    <root>/<target>/decoys_final.ism         SMILES  ZINC-id                  -> property_decoy
    <root>/<target>/actives_final.sdf[.gz]   conformers                       -> active, has_3d
    <root>/<target>/decoys_final.sdf[.gz]    conformers                       -> property_decoy, has_3d
    <root>/all/<target>/...                  same files, combined-tarball layout

**The SDFs are conformers, not poses.** ``decoys_final.sdf.gz`` holds generated 3D
structures of free molecules; nothing in them refers to a receptor. They are a legitimate
starting geometry for a docking backend (G4) and are not a binding mode. ``has_3d=True``
says only that coordinates exist. Conformers are read only when ``prefer_sdf=True``, because
for everything except docking the SMILES file is the smaller, faster, equally complete
source — and the two describe the same molecules, so reading both would double-count.

**The experimental negatives have no canonical filename.** DUD-E's web distribution exposes
them through the per-target pages rather than as a fixed per-target file inside the tarball,
so this adapter accepts any of ``inactives_final.ism``, ``experimental_decoys.ism`` or any
``*inactive*.ism`` / ``*experimental*.ism`` under the target directory, and
``scripts/fetch_decoy_libraries.py`` documents where they come from. A filename the adapter
does not recognise yields nothing rather than the wrong role: silently importing measured
non-binders as synthetic decoys is the exact failure S3.4 is designed to detect, so the
recognition rule is deliberately explicit and its patterns are a class attribute a caller can
extend.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from atomfrust.chem.libraries.base import (
    FileLibraryAdapter,
    MolRecord,
    Role,
    read_sdf,
    read_smiles_table,
)

__all__ = ["DUDEAdapter"]


class DUDEAdapter(FileLibraryAdapter):
    """Adapter over an unpacked DUD-E cache. See the module docstring for the layout."""

    name = "dude"

    #: ``(glob, role)`` in precedence order. Explicit rather than inferred — see module doc.
    SMILES_PATTERNS: tuple[tuple[str, Role], ...] = (
        ("actives_final.ism", "active"),
        ("decoys_final.ism", "property_decoy"),
        ("inactives_final.ism", "measured_inactive"),
        ("experimental_decoys.ism", "measured_inactive"),
        ("*inactive*.ism", "measured_inactive"),
        ("*experimental*.ism", "measured_inactive"),
    )

    SDF_PATTERNS: tuple[tuple[str, Role], ...] = (
        ("actives_final.sdf*", "active"),
        ("decoys_final.sdf*", "property_decoy"),
    )

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        compute_inchikey: bool = False,
        prefer_sdf: bool = False,
    ) -> None:
        super().__init__(root, compute_inchikey=compute_inchikey)
        self.prefer_sdf = prefer_sdf

    def targets(self) -> list[str]:
        """Target names, lower-cased as DUD-E writes them (``pgh1``, ``egfr``, ``cxcr4``)."""
        found = set()
        for base in self._search_bases():
            if not base.is_dir():
                continue
            for entry in base.iterdir():
                if entry.is_dir() and self._target_files(entry):
                    found.add(entry.name)
        return sorted(found)

    def _search_bases(self) -> list[Path]:
        return [self.root, self.root / "all"]

    def _target_dir(self, target: str) -> Path | None:
        for base in self._search_bases():
            candidate = base / target
            if candidate.is_dir() and self._target_files(candidate):
                return candidate
        return None

    def _target_files(self, directory: Path) -> list[tuple[Path, Role, bool]]:
        """``(path, role, is_sdf)`` for one target directory, deduplicated by path.

        A file matched by an earlier pattern is not re-matched by a later one, which is what
        keeps ``inactives_final.ism`` out of the generic ``*inactive*.ism`` bucket twice.
        """
        patterns = self.SDF_PATTERNS if self.prefer_sdf else self.SMILES_PATTERNS
        seen: dict[Path, tuple[Role, bool]] = {}
        for pattern, role in patterns:
            for path in sorted(directory.glob(pattern)):
                if path.is_file() and path not in seen:
                    seen[path] = (role, self.prefer_sdf)
        if self.prefer_sdf:
            # Measured inactives are SMILES-only in every distribution seen, so they would
            # vanish from an SDF-preferring read. Add them back rather than lose the role.
            for pattern, role in self.SMILES_PATTERNS:
                if role != "measured_inactive":
                    continue
                for path in sorted(directory.glob(pattern)):
                    if path.is_file() and path not in seen:
                        seen[path] = (role, False)
        return [(path, role, is_sdf) for path, (role, is_sdf) in sorted(seen.items())]

    def _iter_target_records(self, target: str) -> Iterator[MolRecord]:
        directory = self._target_dir(target)
        if directory is None:
            return
        for path, role, is_sdf in self._target_files(directory):
            if is_sdf:
                yield from self._sdf_records(path, target, role)
            else:
                yield from self._ism_records(path, target, role)

    def _ism_records(self, path: Path, target: str, role: Role) -> Iterator[MolRecord]:
        for index, (smiles, identifier, _fields) in enumerate(read_smiles_table(path)):
            yield MolRecord(
                smiles=smiles,
                inchikey=self._inchikey(smiles),
                source=self.name,
                source_id=identifier or f"{target}_{role}_{index}",
                role=role,
                has_3d=False,
                target=target,
            )

    def _sdf_records(self, path: Path, target: str, role: Role) -> Iterator[MolRecord]:
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
