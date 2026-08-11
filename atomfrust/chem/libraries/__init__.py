"""External molecule libraries, behind one interface.

The chemotype decoy axis (plan step G6) asks "is *this molecule* specifically favoured at
this site?", and answering it needs molecules that are not the native ligand. Four published
sources supply them and each ships a different file format, a different identifier scheme and
— the part that matters — a different *kind* of negative:

============ ================================================= =====================
adapter      what it is                                        role(s) emitted
============ ================================================= =====================
``dude``     102 targets, 50 property-matched decoys per active ``active``,
             plus 9,219 measured non-binders (#11, #14, #15)    ``property_decoy``,
                                                                ``measured_inactive``
``dekois2``  81 targets, 40 actives + 1,200 decoys (#20);       ``active``,
             independent construction, Tier-5 hold-out          ``property_decoy``
``muv``      17 targets, ~15,000 *assayed* inactives each (#21) ``active``,
                                                                ``measured_inactive``
``zinc_random`` unmatched random sample — the **control**       ``property_decoy``
``local``    any local SDF/SMILES, incl. DeepCoy's published    caller-declared
             sets (#18, #19)
============ ================================================= =====================

Two things this package deliberately does not do:

**It does not dock.** Every record is a 2D molecule, or at most a conformer. DUD-E's
``decoys_final.sdf.gz`` and DEKOIS's SDFs carry coordinates of the *free* molecule; a
conformer is not a pose and nothing here knows what a receptor is. Placing a molecule in a
pocket, and gating the result through PoseBusters, is G4 (``atomfrust.dock``).

**It does not download.** The DUD-E, DEKOIS 2.0 and MUV licences permit use but not
redistribution, so nothing is vendored and nothing is fetched on import or on read.
``scripts/fetch_decoy_libraries.py`` populates a gitignored cache
(:func:`~atomfrust.chem.libraries.base.default_cache_root`), and an adapter over a missing
cache reports ``available() is False`` instead of raising — a sweep on a machine that has
DUD-E but not MUV should lose one library, not fall over.
"""

from __future__ import annotations

from pathlib import Path

from atomfrust.chem.libraries.base import (
    DOE_PROPERTIES,
    MEASURED_ROLES,
    PROPERTY_COLUMNS,
    ROLES,
    SYNTHETIC_ROLES,
    DecoyLibraryAdapter,
    FileLibraryAdapter,
    MolRecord,
    Role,
    default_cache_root,
    doe_score,
    property_summary,
    rdkit_available,
)
from atomfrust.chem.libraries.dekois import DEKOIS2Adapter
from atomfrust.chem.libraries.dude import DUDEAdapter
from atomfrust.chem.libraries.local import LocalSDFAdapter
from atomfrust.chem.libraries.muv import MUVAdapter
from atomfrust.chem.libraries.zinc import ZINCRandomAdapter

__all__ = [
    "MolRecord",
    "Role",
    "ROLES",
    "SYNTHETIC_ROLES",
    "MEASURED_ROLES",
    "DecoyLibraryAdapter",
    "FileLibraryAdapter",
    "DUDEAdapter",
    "DEKOIS2Adapter",
    "MUVAdapter",
    "ZINCRandomAdapter",
    "LocalSDFAdapter",
    "ADAPTERS",
    "get_adapter",
    "default_cache_root",
    "property_summary",
    "PROPERTY_COLUMNS",
    "doe_score",
    "DOE_PROPERTIES",
    "rdkit_available",
]

#: Name → class. The key is also the cache subdirectory and the value written to
#: ``MolRecord.source``, so a ``source_ref`` in a result table names the adapter that can read
#: it back.
ADAPTERS: dict[str, type[FileLibraryAdapter]] = {
    DUDEAdapter.name: DUDEAdapter,
    DEKOIS2Adapter.name: DEKOIS2Adapter,
    MUVAdapter.name: MUVAdapter,
    ZINCRandomAdapter.name: ZINCRandomAdapter,
    LocalSDFAdapter.name: LocalSDFAdapter,
}


def get_adapter(name: str, root: Path | None = None, **kwargs) -> DecoyLibraryAdapter:
    """Adapter for ``name``, rooted at ``root/<name>``.

    ``root`` is the **cache root** holding all libraries, not one library's directory — the
    layout the fetch script writes and :func:`default_cache_root` returns. Construct the class
    directly (``DUDEAdapter(some_dir)``) to point at a single library elsewhere.

    An unknown name raises ``KeyError`` listing the known ones, because the caller is almost
    always a CLI flag and a typo should say what was meant, not return ``None``.
    """
    try:
        cls = ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown decoy library {name!r}; known libraries are {sorted(ADAPTERS)}"
        ) from None
    base = Path(root) if root is not None else default_cache_root()
    return cls(base / name, **kwargs)
