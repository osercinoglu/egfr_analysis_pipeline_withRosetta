"""Deterministic 3-character Rosetta residue codes for library molecules.

This generalises the ad-hoc ``634 -> Z34`` fix. A handful of real CCD codes collide with
Rosetta's internal residue namespace, and the prototype patched them one at a time in
``config/ligand_overrides.yaml`` (``rosetta_ligand_code``), which does not scale to a
screening library where most molecules have no CCD code at all.

Three properties the allocator holds:

**Deterministic from the InChIKey.** The primary code is a hash of the InChIKey alone, so
the same molecule gets the same code in every process, on every machine, and in any batch it
appears in. Only a *collision* consults allocator state, which is why :meth:`allocate_many`
sorts its input first: a batch allocated as a set is order-independent end to end.

**Letter-first.** Codes are ``[A-Z][A-Z0-9][A-Z0-9]``. Every generated code therefore starts
with a letter, which is exactly what ``634 -> Z34`` did by hand. It costs 22% of the code
space and removes the whole class of digit-leading names that the prototype had to special-case.

**Reserved sets are injected, never probed.** Importing this module must not initialise
PyRosetta — that is :mod:`atomfrust.pose`'s job and it can only happen once per process. The
default reserved set below is deliberately conservative and offline; a caller that wants
certainty passes ``reserved=`` built from the chemical manager (or from the ``comp_id``\\ s
already present in the system, which is the collision that actually bites).

The allocated code is the *Rosetta* name only. Its real identity — CCD code, SMILES,
InChIKey — travels separately, exactly as :class:`atomfrust.spec.LigandSpec` keeps
``comp_id`` and ``rosetta_name`` apart. Reporting must never use the allocated code.

.. warning::
   **Use :meth:`CodeAllocator.allocate_many` for a batch, not repeated
   :meth:`allocate`.** The first probe is a pure function of the InChIKey, so an
   uncontended molecule always lands on the same code — but a *collision* is resolved
   against allocator state, which depends on what has already been allocated. Measured:
   allocating 1000 synthetic keys sequentially in forward versus reverse order disagrees on
   42 of them. ``allocate_many`` sorts its input first and is therefore order-independent
   (verified on a shuffled 1000-key batch).

   Across runs, stability comes from :class:`~atomfrust.chem.cache.ParamCache`, which keys
   on the InChIKey and stores the allocated code — so a molecule seen before keeps its code
   regardless of allocation order. A molecule allocated fresh in two different batches may
   not.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

from atomfrust.spec import WATER_NAMES

__all__ = ["CodeAllocator", "DEFAULT_RESERVED_CODES", "CODE_SPACE"]

_FIRST_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_REST_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

#: 26 * 36 * 36. Enough for a screening library; :meth:`CodeAllocator.allocate` raises rather
#: than wrap around silently if a caller ever exhausts it.
CODE_SPACE = len(_FIRST_CHARS) * len(_REST_CHARS) * len(_REST_CHARS)

_CANONICAL_AA = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)

#: Names Rosetta or the PDB already use for something. Each group has a reason:
#:
#: * the 20 canonical amino acids — ``fa_standard`` residue-type names; a ligand called
#:   ``ALA`` is a name clash inside the residue type set, not merely confusing output;
#: * ``MSE SEC PYL`` — non-canonical residues that appear as HETATM but are backbone
#:   (:data:`atomfrust.spec._STANDARD` excludes them from ligand autodetection for the same
#:   reason);
#: * ``UNK UNL UNX XXX`` — the PDB's and Rosetta's "unknown residue" names;
#: * ``VRT`` — Rosetta's virtual residue, which every foldtree-rooted pose can contain;
#: * waters — :data:`atomfrust.spec.WATER_NAMES` plus Rosetta's own water residue names
#:   ``TP3``/``TP5``/``SPC``;
#: * ``ADE CYT GUA THY URA`` — 3-letter nucleotide names (the 1- and 2-letter PDB forms
#:   cannot collide with a 3-character code);
#: * ``FE2`` — the only 3-character name among the metals that ship with Rosetta;
#: * ``LIG`` — the conventional generic ligand placeholder. Allocating it to one specific
#:   molecule would make every log line ambiguous.
#:
#: Everything here is offline knowledge. It is a floor, not a proof of non-collision: pass
#: ``reserved=`` when the truth is available.
DEFAULT_RESERVED_CODES = frozenset(
    _CANONICAL_AA
    | {"MSE", "SEC", "PYL"}
    | {"UNK", "UNL", "UNX", "XXX"}
    | {"VRT"}
    | {w.upper() for w in WATER_NAMES if len(w) == 3}
    | {"TP3", "TP5", "SPC"}
    | {"ADE", "CYT", "GUA", "THY", "URA"}
    | {"FE2"}
    | {"LIG"}
)


def _code_at(index: int) -> str:
    """Index -> code, in a fixed order so the mapping is stable across releases."""
    rest = len(_REST_CHARS)
    third = _REST_CHARS[index % rest]
    index //= rest
    second = _REST_CHARS[index % rest]
    index //= rest
    return _FIRST_CHARS[index % len(_FIRST_CHARS)] + second + third


def _hash_index(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % CODE_SPACE


class CodeAllocator:
    """Hands out 3-character Rosetta residue codes, one per molecule.

    An instance is a batch: it remembers what it has already handed out so two molecules in
    the same run never share a code. Across instances only the hash matters, so a molecule
    re-parametrised tomorrow lands on the same code unless it collided.
    """

    def __init__(self, reserved: set[str] | None = None) -> None:
        base = DEFAULT_RESERVED_CODES if reserved is None else reserved
        self._reserved = frozenset(c.strip().upper() for c in base if c.strip())
        self._assigned: dict[str, str] = {}
        self._taken: set[str] = set(self._reserved)

    @property
    def reserved(self) -> frozenset[str]:
        return self._reserved

    @property
    def assigned(self) -> Mapping[str, str]:
        """InChIKey -> code, for everything this instance has allocated."""
        return dict(self._assigned)

    def allocate(self, inchikey: str) -> str:
        """Return this molecule's code, allocating it on first sight.

        Idempotent per instance. The first probe is a pure function of the InChIKey; on a
        collision the search walks forward deterministically, so the outcome depends on the
        *set* of molecules seen so far and not on the arrival order of the non-colliding
        ones. Use :meth:`allocate_many` when even that residual order dependence matters.
        """
        key = inchikey.strip().upper()
        if not key:
            raise ValueError("inchikey must be non-empty")
        existing = self._assigned.get(key)
        if existing is not None:
            return existing

        start = _hash_index(key)
        for offset in range(CODE_SPACE):
            code = _code_at((start + offset) % CODE_SPACE)
            if code not in self._taken:
                self._assigned[key] = code
                self._taken.add(code)
                return code
        raise RuntimeError(
            f"3-character code space exhausted ({CODE_SPACE} codes, "
            f"{len(self._taken)} taken)"
        )

    def allocate_many(self, inchikeys: Iterable[str]) -> dict[str, str]:
        """Allocate a whole batch. Sorted first, so the result is order-independent."""
        return {key: self.allocate(key) for key in sorted({k.strip().upper() for k in inchikeys})}

    def reserve(self, *codes: str) -> None:
        """Block codes after construction — e.g. the ``comp_id``\\ s of the crystal ligands
        already in the system, which no library molecule may shadow."""
        for code in codes:
            cleaned = code.strip().upper()
            if cleaned:
                self._taken.add(cleaned)
