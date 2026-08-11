"""An unmatched random ZINC sample — the negative control, not a decoy source.

**What this is for.** Every property-matched decoy set (DUD-E, DEKOIS, DeepCoy) is built to
remove the trivial signal: if actives and decoys differ in molecular weight, any method that
notices molecular weight scores well while measuring nothing. A random ZINC sample keeps that
signal, so its performance is the **ceiling attributable to trivial property discrimination**
— the number every matched-set result must be read against. A method that scores the same on
matched decoys as on random ZINC has learned properties; one that scores lower on matched
decoys and above chance is doing something else. That comparison is the point of running this
adapter at all (methods document §3.5, "an unmatched random ZINC sample as negative control
establishing the ceiling attributable to trivial property discrimination"; S2.3).

**It must never be used as a decoy set.** Reporting enrichment against random ZINC as though
it were a screening result inflates every number and is precisely the bias Chaput et al.
demonstrated on DUD-E, where removing biased targets collapsed four commercial programs from
30/27/14/11 successes to 5/4/2/2 (#16, #17). The record's ``source`` is ``zinc_random``, which
is how a report tells the control apart from a decoy set — the ``role`` field cannot, since
both are synthetic negatives with nothing measured behind them, and ``property_decoy`` is the
honest label for both.

**Layout.** ZINC tranche files, or any SMILES table the fetch script wrote, under the root::

    <root>/*.smi | *.ism | *.txt      smiles  zinc_id    (a leading header row is skipped)
    <root>/<tranche>/*.smi            same, tranche-per-directory

``targets()`` returns ``[]``. A random sample is not target-specific and pretending otherwise
would invite ``records(target=...)`` calls that quietly return a per-target subset of an
unmatched sample. ``records()`` with no target reads everything under the root; the "target"
of a control is the tranche it was drawn from, which is recorded per file as the sample name.

Sampling is *not* done here. The fetch script draws the sample with a recorded seed and
writes it down, so the composition of the control is a fixed, citable artefact rather than
something re-randomised on every read.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from atomfrust.chem.libraries.base import FileLibraryAdapter, MolRecord, read_smiles_table

__all__ = ["ZINCRandomAdapter"]

_SUFFIXES = (".smi", ".ism", ".txt")


class ZINCRandomAdapter(FileLibraryAdapter):
    """Unmatched random ZINC molecules. Read the module docstring before using the output."""

    name = "zinc_random"

    def targets(self) -> list[str]:
        """Always empty — see the module docstring. ``available()`` is overridden to match."""
        return []

    def available(self) -> bool:
        return bool(self._files())

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in _SUFFIXES
        )

    def records(
        self,
        target: str | None = None,
        limit: int | None = None,
        *,
        roles: Sequence[str] | None = None,
    ) -> Iterator[MolRecord]:
        """``target`` is accepted for protocol conformance and must be ``None``.

        Raising rather than ignoring it: a caller passing a target here believes it is getting
        a target-specific set, and silently returning an unmatched sample instead would make
        the control look like a decoy set in exactly the situation it is meant to guard.
        """
        if target is not None:
            raise ValueError(
                "ZINCRandomAdapter has no targets: an unmatched random sample is a control, "
                "not a per-target decoy set"
            )
        wanted = self._resolve_roles(roles)
        if "property_decoy" not in wanted:
            return
        emitted = 0
        for path in self._files():
            sample = path.stem
            for index, (smiles, identifier, _fields) in enumerate(read_smiles_table(path)):
                yield MolRecord(
                    smiles=smiles,
                    inchikey=self._inchikey(smiles),
                    source=self.name,
                    source_id=identifier or f"{sample}_{index}",
                    role="property_decoy",
                    has_3d=False,
                    target=None,
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
