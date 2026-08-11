"""Any SDF or SMILES file the user points at — including DeepCoy's published decoy sets.

**This is how DeepCoy enters the project.** DeepCoy is the strongest published alternative to
the DUD-E protocol (DOE 0.166 → 0.032 on DUD-E, 0.109 → 0.038 on DEKOIS 2.0, while making
decoys *harder* to dock: Vina AUROC 0.70 → 0.63 — methods document #18, #19), and its authors
released the generated decoy sets as files. Consuming those files through this adapter keeps
the DeepCoy **model** out of the dependency graph: no TensorFlow, no GPU, no generation step
whose output would differ from the published sets and therefore not be the thing #18 measured.
Regenerating decoys is a different experiment from reproducing one.

It is also the escape hatch for anything else with a file: an in-house series, a supplier
catalogue subset, a DeepCoy set the user generated themselves.

**Role assignment is explicit-first, filename-second, and never silent.** ``role_map`` pins a
file (or a glob) to a role. Otherwise the filename is read: ``*inactive*`` or ``*measured*``
→ ``measured_inactive``, ``*decoy*`` → ``property_decoy``, ``*active*`` or ``*ligand*`` →
``active``. A file matching none of these gets ``default_role``, which is ``property_decoy``
— the safe default, because it keeps an unlabelled file out of the ``measured_inactive``
bucket that S3.4's claim rests on. ``strict=True`` turns the fallback into an error for
callers who would rather be told.

Both formats are read: ``.sdf``/``.sdf.gz`` (coordinates → honest ``has_3d``) and
``.smi``/``.ism``/``.txt`` (SMILES, ``has_3d=False``). ``targets()`` reports subdirectory
names, so a per-target DeepCoy set unpacks and works; flat files land under the target
``""``.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterator, Mapping
from pathlib import Path

from atomfrust.chem.libraries.base import (
    ROLES,
    FileLibraryAdapter,
    MolRecord,
    Role,
    read_sdf,
    read_smiles_table,
)

__all__ = ["LocalSDFAdapter"]

_SDF_SUFFIXES = (".sdf", ".sdf.gz")
_SMILES_SUFFIXES = (".smi", ".ism", ".txt")

#: ``(filename substring, role)`` in precedence order. ``inactive`` is tested before ``active``
#: because it contains it.
_NAME_RULES: tuple[tuple[str, Role], ...] = (
    ("inactive", "measured_inactive"),
    ("measured", "measured_inactive"),
    ("decoy", "property_decoy"),
    ("active", "active"),
    ("ligand", "active"),
)


class LocalSDFAdapter(FileLibraryAdapter):
    """Adapter over a directory of local SDF/SMILES files. See the module docstring."""

    name = "local"

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        compute_inchikey: bool = False,
        role_map: Mapping[str, str] | None = None,
        default_role: str = "property_decoy",
        strict: bool = False,
        source: str | None = None,
    ) -> None:
        super().__init__(root, compute_inchikey=compute_inchikey)
        if default_role not in ROLES:
            raise ValueError(f"default_role must be one of {list(ROLES)}, got {default_role!r}")
        for pattern, role in (role_map or {}).items():
            if role not in ROLES:
                raise ValueError(f"role_map[{pattern!r}] = {role!r} is not one of {list(ROLES)}")
        self.role_map = dict(role_map or {})
        self.default_role = default_role
        self.strict = strict
        #: Overridable so a curated set keeps its provenance (``source="deepcoy"``) instead of
        #: everything hand-supplied collapsing into one ``local`` bucket in a summary table.
        self.source = source or self.name

    def targets(self) -> list[str]:
        return sorted({target for target, _, _ in self._files()})

    def _files(self) -> list[tuple[str, Path, Role]]:
        if not self.root.is_dir():
            return []
        found: list[tuple[str, Path, Role]] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or self._kind(path) is None:
                continue
            relative = path.parent.relative_to(self.root)
            target = relative.parts[0] if relative.parts else ""
            found.append((target, path, self._role_of(path)))
        return found

    @staticmethod
    def _kind(path: Path) -> str | None:
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in _SDF_SUFFIXES):
            return "sdf"
        if any(name.endswith(suffix) for suffix in _SMILES_SUFFIXES):
            return "smiles"
        return None

    def _role_of(self, path: Path) -> Role:
        for pattern, role in self.role_map.items():
            if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern):
                return role  # type: ignore[return-value]
        name = path.name.lower()
        for token, role in _NAME_RULES:
            if token in name:
                return role
        if self.strict:
            raise ValueError(
                f"cannot infer a role for {path.name!r}: name it *_actives / *_decoys / "
                f"*_inactives, or pass role_map={{'{path.name}': '<role>'}}"
            )
        return self.default_role  # type: ignore[return-value]

    def _iter_target_records(self, target: str) -> Iterator[MolRecord]:
        for name, path, role in self._files():
            if name != target:
                continue
            if self._kind(path) == "sdf":
                yield from self._sdf_records(path, target, role)
            else:
                yield from self._smiles_records(path, target, role)

    def _sdf_records(self, path: Path, target: str, role: Role) -> Iterator[MolRecord]:
        for index, (title, smiles, has_3d, data) in enumerate(read_sdf(path)):
            yield MolRecord(
                smiles=smiles,
                inchikey=self._inchikey(smiles),
                source=self.source,
                source_id=title or f"{path.stem}_{index}",
                role=role,
                has_3d=has_3d,
                target=target or None,
                properties=data,
            )

    def _smiles_records(self, path: Path, target: str, role: Role) -> Iterator[MolRecord]:
        for index, (smiles, identifier, _fields) in enumerate(read_smiles_table(path)):
            yield MolRecord(
                smiles=smiles,
                inchikey=self._inchikey(smiles),
                source=self.source,
                source_id=identifier or f"{path.stem}_{index}",
                role=role,
                has_3d=False,
                target=target or None,
            )
