"""Chemistry: turning a molecule into something Rosetta can score.

``scripts/05_prepare_ligand.py`` parametrised 51 crystal ligands one at a time, keyed on CCD
codes, gated on matching PDB atom names, with no cache and no failure taxonomy. This package
is the same backend — the vendored ``src/molfile_to_params.py`` — reached through an interface
that survives a screening library:

* :func:`~atomfrust.chem.paramize.paramize` returns a :class:`~atomfrust.chem.paramize.ParamRecord`
  instead of raising, so every rejected molecule carries a :class:`~atomfrust.chem.paramize.ParamFailure`
  member and "all failures categorised" is a ``groupby``;
* :class:`~atomfrust.chem.cache.ParamCache` keys on ``inchikey|protonation|conformer``, so a
  changed protonation model is a miss rather than a silent reuse;
* :class:`~atomfrust.chem.codes.CodeAllocator` hands out 3-character Rosetta names
  deterministically, generalising the hand-written ``634 -> Z34`` override.

Nothing here imports PyRosetta at module scope. RDKit is imported lazily inside
:mod:`~atomfrust.chem.paramize` and its absence is reported per molecule, so a machine that
only runs crystal structures can still import this package.
"""

from __future__ import annotations

from atomfrust.chem.cache import ParamCache
from atomfrust.chem.codes import DEFAULT_RESERVED_CODES, CodeAllocator
from atomfrust.chem.paramize import (
    SUPPORTED_ELEMENTS,
    ParamFailure,
    ParamRecord,
    paramize,
    rdkit_available,
)

# A different protonation state is a different molecule: PROTONATION_VERSION feeds both the
# cache key and the regeneration key, so changing the model is a miss rather than a silent
# reuse of a stored analysis.
from atomfrust.chem.protonation import (
    PROTONATION_VERSION,
    Protomer,
    ProtomerSet,
    canonical_state,
    enumerate_states,
    sensitivity_table,
)

__all__ = [
    # paramize / cache / codes (G1)
    "ParamCache",
    "CodeAllocator",
    "DEFAULT_RESERVED_CODES",
    "ParamFailure",
    "ParamRecord",
    "SUPPORTED_ELEMENTS",
    "paramize",
    "rdkit_available",
    # protonation (G2)
    "PROTONATION_VERSION",
    "Protomer",
    "ProtomerSet",
    "canonical_state",
    "enumerate_states",
    "sensitivity_table",
]
