"""The record type, the adapter protocol, and the two library-level measurements.

**What a record is, and what it is not.** :class:`MolRecord` is a *molecule*: a SMILES
string, an identity, a provenance pair, and a role. It is not a pose and not a complex.
DUD-E ships ``decoys_final.sdf.gz`` conformers and DEKOIS 2.0 ships SDF with coordinates,
and neither is a pose *in a receptor* — a conformer is a shape, a pose is a placement.
``has_3d`` therefore reports only "this file carried coordinates", which is useful to a
docking backend as a starting geometry and useless as a binding mode. Putting the molecule
in the pocket is G4's job (``atomfrust.dock``); nothing in this package docks, scores, or
knows what a receptor is.

**Why ``role`` is a field and not a directory convention.** The three roles are not three
flavours of the same thing:

``active``
    Measured binder.
``property_decoy``
    A *synthetic* negative. Presumed inactive because it was drawn to match the actives'
    physicochemical properties while being topologically dissimilar (DUD-E: ECFP4, most
    dissimilar 25% retained — methods document #15). Nobody assayed it.
``measured_inactive``
    Assayed and found not to bind. DUD-E's experimental-decoy set is 9,219 such compounds,
    of which 1,070 belong to COX-1/PGH1 (#14); MUV is ~15,000 per target across 17 targets
    (#21).

Success criterion S3.4 exists precisely because a computational specificity measure
validated against computational negatives proves little. If a ``property_decoy`` ever
reaches an S3.4 tally it silently turns the experiment back into the circular one it was
designed to replace, and no downstream check would catch it, because the two look identical
in every other column. Hence: the role travels on every record, ``records()`` filters on it,
and the two are never pooled by default — a caller that wants both must ask for both.

An unmatched random ZINC sample is also ``property_decoy``: it is a synthetic negative. What
makes it a *control* rather than a decoy set is that it was not property-matched, and that
fact lives in ``source == "zinc_random"``, not in the role. See :mod:`.zinc`.

**Two measurements live here.** :func:`property_summary` (the descriptor table every
generator comparison starts from) and :func:`doe_score` (S2.3's headline number). They are
in this module rather than a separate one because both are defined over *records*, which is
the only vocabulary this package has.

RDKit is guarded exactly as :mod:`atomfrust.chem.paramize` guards it: a machine without
RDKit can still import this package, list targets, and read SMILES out of a file — it just
cannot compute descriptors or InChIKeys, and says so rather than returning zeros.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "Role",
    "ROLES",
    "SYNTHETIC_ROLES",
    "MEASURED_ROLES",
    "MolRecord",
    "DecoyLibraryAdapter",
    "FileLibraryAdapter",
    "default_cache_root",
    "property_summary",
    "PROPERTY_COLUMNS",
    "DOE_PROPERTIES",
    "doe_score",
    "rdkit_available",
    "RDKIT_IMPORT_ERROR",
    "read_smiles_table",
    "read_sdf",
]

Role = Literal["active", "property_decoy", "measured_inactive"]

#: Every role, in the order a report should list them.
ROLES: tuple[Role, ...] = ("active", "property_decoy", "measured_inactive")

#: Negatives nobody assayed. Never pool these with :data:`MEASURED_ROLES` (see module doc).
SYNTHETIC_ROLES: frozenset[str] = frozenset({"property_decoy"})

#: Negatives with a measurement behind them — the S3.4 / S5.8 resource.
MEASURED_ROLES: frozenset[str] = frozenset({"measured_inactive"})


try:  # pragma: no cover - exercised by the environment, not the tests
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    RDLogger.DisableLog("rdApp.*")
    RDKIT_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    Chem = Crippen = Descriptors = Lipinski = None  # type: ignore[assignment]
    RDKIT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def rdkit_available() -> bool:
    return RDKIT_IMPORT_ERROR is None


@dataclass(frozen=True)
class MolRecord:
    """One molecule out of one library.

    ``source`` + ``source_id`` is the provenance pair that G6 writes into
    ``decoys/index.parquet`` as ``source_ref``, so a decoy in a result table can be traced
    back to the file and line it came from. Keep ``source_id`` as the library's *own*
    identifier (ZINC code, ChEMBL id, PubChem SID, SDF title) — a synthesised index is a
    fallback, not a default, because it does not survive a re-download.

    ``inchikey`` is ``None`` when it was not computed, never ``""``: absent and empty are
    different claims and the adapters do not compute it unless asked (one RDKit parse per
    molecule over a 15,000-compound MUV target is not free, and
    :func:`atomfrust.chem.paramize.paramize` computes it downstream anyway).

    ``properties`` carries numeric fields the source file happened to supply (SDF data
    fields, ``.dat`` columns). It is provenance, not computation:
    :func:`property_summary` never reads it, because a mixture of file-supplied and
    RDKit-computed descriptors in one column is not a descriptor.
    """

    smiles: str
    inchikey: str | None
    source: str
    source_id: str
    role: Role
    has_3d: bool = False
    target: str | None = None
    properties: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "source": self.source,
            "source_id": self.source_id,
            "role": self.role,
            "has_3d": self.has_3d,
            "target": self.target,
            "source_ref": self.source_ref,
        }

    @property
    def source_ref(self) -> str:
        """``"{source}:{source_id}"`` — the single string G6 stores per decoy."""
        return f"{self.source}:{self.source_id}"


@runtime_checkable
class DecoyLibraryAdapter(Protocol):
    """What every library looks like once the file-format differences are gone.

    ``available()`` must answer without raising and without network access: a missing cache
    is the normal state on a fresh clone (licences bar vendoring, so nothing is downloaded
    by default), and it has to degrade to "this library is not here" rather than to a
    traceback in the middle of a sweep.

    ``records()`` is an iterator because MUV is ~15,000 molecules per target and DUD-E is 50
    decoys per active over 22,886 clustered actives (#11, #21); materialising a library to
    count it is a mistake at this scale.
    """

    name: str

    def available(self) -> bool: ...

    def targets(self) -> list[str]: ...

    def records(
        self,
        target: str | None = None,
        limit: int | None = None,
        *,
        roles: Sequence[str] | None = None,
    ) -> Iterator[MolRecord]: ...


def default_cache_root() -> Path:
    """Where downloaded libraries live: ``$ATOMFRUST_LIBRARY_CACHE`` or ``<repo>/.cache/libraries``.

    Repo-local and gitignored rather than under ``$HOME``: the DUD-E, DEKOIS 2.0 and MUV
    licences permit use but not redistribution, so the fetch script is versioned and its
    output is not. Keeping the cache beside the code makes "which libraries does this clone
    have" a ``ls``, and makes a stale cache deletable without hunting through ``~/.cache``.
    """
    import os

    override = os.environ.get("ATOMFRUST_LIBRARY_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[3] / ".cache" / "libraries"


class FileLibraryAdapter:
    """Shared machinery: root resolution, availability, target discovery, filtering.

    Subclasses implement :meth:`_iter_target_records` and :meth:`targets`; everything a
    caller sees — the role filter, the limit, the "not available" contract — is here so the
    five adapters cannot drift apart on it.

    ``root`` is the *library's own directory* (``<cache>/dude``), not the cache root;
    :func:`atomfrust.chem.libraries.get_adapter` composes the two. That split is what lets a
    test point an adapter straight at a fixture directory.
    """

    name: str = "base"

    def __init__(self, root: Path | str | None = None, *, compute_inchikey: bool = False) -> None:
        self.root = Path(root) if root is not None else default_cache_root() / self.name
        self.compute_inchikey = compute_inchikey

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(root={str(self.root)!r})"

    def available(self) -> bool:
        """True when this root holds at least one readable target. Never raises."""
        try:
            return bool(self.targets())
        except OSError:
            return False

    def targets(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def records(
        self,
        target: str | None = None,
        limit: int | None = None,
        *,
        roles: Sequence[str] | None = None,
    ) -> Iterator[MolRecord]:
        """Yield records, optionally restricted to one target, some roles, and a count.

        ``roles=None`` means every role *present in the files*, which is not the same as
        "every role": an adapter reading a directory that holds only ``decoys_final.ism``
        yields only ``property_decoy``. Ask explicitly
        (``roles=["measured_inactive"]``) whenever the distinction matters — S3.4 does.
        """
        wanted = self._resolve_roles(roles)
        targets = [target] if target is not None else self.targets()
        emitted = 0
        for name in targets:
            for record in self._iter_target_records(name):
                if record.role not in wanted:
                    continue
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    @staticmethod
    def _resolve_roles(roles: Sequence[str] | None) -> frozenset[str]:
        if roles is None:
            return frozenset(ROLES)
        unknown = sorted(set(roles) - set(ROLES))
        if unknown:
            raise ValueError(f"unknown role(s) {unknown}; known roles are {list(ROLES)}")
        return frozenset(roles)

    def _iter_target_records(self, target: str) -> Iterator[MolRecord]:  # pragma: no cover
        raise NotImplementedError

    # -------------------------------------------------------------- shared record helpers

    def _inchikey(self, smiles: str) -> str | None:
        """InChIKey when asked for and computable, else ``None`` — never a guess."""
        if not (self.compute_inchikey and rdkit_available() and smiles):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToInchiKey(mol) or None
        except Exception:
            return None


# ------------------------------------------------------------------------ file readers
#
# DUD-E ships its conformer SDFs gzipped and its SMILES plain, and a user who gunzips half a
# cache should not get a parse error, so compression is decided by suffix in one place.
#
# Two readers cover all five adapters, because every one of these libraries ships either a
# whitespace-delimited SMILES table or an SDF. Keeping them here rather than in each adapter
# means "is has_3d honest" is answered once.


def _open_text(path: Path):
    if path.suffix == ".gz":
        import gzip

        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")


def read_smiles_table(
    path: Path,
    *,
    smiles_column: int = 0,
    id_column: int | None = 1,
    delimiter: str | None = None,
) -> Iterator[tuple[str, str, list[str]]]:
    """Yield ``(smiles, identifier, all_fields)`` from a ``.ism`` / ``.smi`` / ``.dat`` file.

    Blank lines and ``#`` comments are skipped, as is a leading header row — ZINC tranche
    files start with ``smiles zinc_id`` and MUV's ``.dat`` files with a PubChem column
    header, and a header parsed as a molecule is a silent one-record error rather than a
    loud one.

    ``identifier`` falls back to the empty string when the file has no id column; the caller
    synthesises one and records that it did.
    """
    first = True
    with _open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(delimiter) if delimiter else line.split()
            if not fields:
                continue
            if first:
                first = False
                if _looks_like_header(fields):
                    continue
            smiles = fields[smiles_column] if len(fields) > smiles_column else ""
            if not smiles:
                continue
            identifier = ""
            if id_column is not None and len(fields) > id_column:
                identifier = fields[id_column]
            yield smiles, identifier, fields


_HEADER_TOKENS = frozenset({"smiles", "smi", "canonical_smiles", "structure"})


def _looks_like_header(fields: Sequence[str]) -> bool:
    return any(f.strip().lower() in _HEADER_TOKENS for f in fields)


def read_sdf(path: Path, *, with_smiles: bool = True) -> Iterator[tuple[str, str, bool, dict]]:
    """Yield ``(title, smiles, has_3d, numeric_data_fields)`` per SDF record.

    ``has_3d`` is read from the **coordinates**, not from the molblock's 2D/3D dimension tag:
    the tag is a claim by whatever wrote the file and several of these libraries were
    re-exported by tools that got it wrong, whereas a z column of zeros is a fact. The rule
    is conservative in one known way — a genuinely planar 3D molecule (benzene alone) reads
    as 2D — which is the right direction to be wrong in, since ``has_3d=True`` is what makes
    a downstream backend skip conformer generation.

    SMILES needs RDKit; without it the SMILES is empty and the rest of the record is still
    correct, so target listing and role accounting work on a machine that has no RDKit.
    """
    with _open_text(path) as handle:
        text = handle.read()
    for block in text.split("$$$$"):
        block = block.lstrip("\n")
        if not block.strip():
            continue
        lines = block.splitlines()
        if len(lines) < 4:
            continue
        title = lines[0].strip()
        has_3d = _molblock_has_3d(lines)
        data = _molblock_numeric_data(lines)
        smiles = ""
        if with_smiles and rdkit_available():
            try:
                mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
            except Exception:
                smiles = ""
        yield title, smiles, has_3d, data


def _molblock_has_3d(lines: Sequence[str]) -> bool:
    """Any atom with a non-zero z coordinate. V2000 fixed columns; V3000 falls back to split."""
    counts = lines[3]
    try:
        n_atoms = int(counts[0:3])
    except (ValueError, IndexError):
        return _v3000_has_3d(lines)
    for line in lines[4 : 4 + n_atoms]:
        try:
            z = float(line[20:30])
        except (ValueError, IndexError):
            continue
        if abs(z) > 1e-4:
            return True
    return False


def _v3000_has_3d(lines: Sequence[str]) -> bool:
    for line in lines:
        if not line.startswith("M  V30 ") or "BEGIN" in line or "END" in line:
            continue
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            z = float(fields[6])
        except ValueError:
            continue
        if abs(z) > 1e-4:
            return True
    return False


def _molblock_numeric_data(lines: Sequence[str]) -> dict[str, float]:
    """SDF ``> <NAME>`` fields whose value parses as a float. Non-numeric fields are dropped
    rather than stored as strings, because ``properties`` is typed ``dict[str, float]`` and a
    stringly-typed escape hatch there would be used."""
    data: dict[str, float] = {}
    key: str | None = None
    for line in lines:
        if line.startswith("> "):
            start, end = line.find("<"), line.rfind(">")
            key = line[start + 1 : end] if 0 <= start < end else None
        elif key is not None:
            stripped = line.strip()
            if stripped:
                try:
                    data[key] = float(stripped)
                except ValueError:
                    pass
            key = None
    return data


# ------------------------------------------------------------------------ measurements

#: Columns of :func:`property_summary`. The first five are the ones G3 is specified on; the
#: last two are here because :data:`DOE_PROPERTIES` needs them and computing them twice would
#: be the only alternative.
PROPERTY_COLUMNS = (
    "mw", "hac", "logp", "formal_charge", "rotatable_bonds", "hba", "hbd",
)

#: The DOE property set. These six are DUD-E's own matching properties (Mysinger et al. 2012,
#: methods document #11/#15) — molecular weight, cLogP, rotatable bonds, hydrogen-bond
#: acceptors, hydrogen-bond donors, net charge — so a DOE computed here is computed over the
#: axes the decoys were matched on, which is what makes the reference values in
#: :func:`doe_score` comparable. Heavy-atom count is deliberately excluded: it is ~0.95
#: correlated with molecular weight and would double-weight size.
DOE_PROPERTIES = ("mw", "logp", "rotatable_bonds", "hba", "hbd", "formal_charge")


def property_summary(records: Iterable[MolRecord]) -> pd.DataFrame:
    """Descriptor table: one row per record, RDKit-computed, NaN where the SMILES is unusable.

    Columns are the identity fields (``source``, ``source_id``, ``role``, ``target``,
    ``smiles``) plus :data:`PROPERTY_COLUMNS`: molecular weight, heavy-atom count, cLogP,
    formal charge, rotatable bonds, H-bond acceptors and donors.

    A molecule RDKit cannot parse becomes a row of NaNs rather than a dropped row, so the
    table's length always matches the library's and "how many were unusable" is
    ``df["mw"].isna().sum()`` rather than a diff against a count taken elsewhere.
    """
    if not rdkit_available():
        raise RuntimeError(
            f"property_summary needs RDKit (import failed: {RDKIT_IMPORT_ERROR})"
        )
    rows = []
    for record in records:
        row: dict[str, Any] = {
            "source": record.source,
            "source_id": record.source_id,
            "role": record.role,
            "target": record.target,
            "smiles": record.smiles,
        }
        row.update(_descriptors(record.smiles))
        rows.append(row)
    columns = ["source", "source_id", "role", "target", "smiles", *PROPERTY_COLUMNS]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def _descriptors(smiles: str) -> dict[str, float]:
    nan = {name: math.nan for name in PROPERTY_COLUMNS}
    if not smiles:
        return nan
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return nan
    if mol is None:
        return nan
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "hac": float(mol.GetNumHeavyAtoms()),
        "logp": float(Crippen.MolLogP(mol)),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
    }


def doe_score(
    actives: Iterable[MolRecord] | pd.DataFrame,
    decoys: Iterable[MolRecord] | pd.DataFrame,
    *,
    properties: Sequence[str] = DOE_PROPERTIES,
) -> float:
    """Deviation from optimal embedding (Imrie et al. 2021; Vogel et al. 2011). Lower is better.

    **What it measures.** Whether a machine can tell actives from decoys using nothing but
    physicochemical properties. Perfectly embedded decoys make the two indistinguishable in
    property space and score 0; a set whose properties separate cleanly scores 0.5, the
    maximum.

    **How it is computed here.** Each property in ``properties`` is min-max normalised over
    the pooled set, so all six contribute on a comparable scale. Then, for each active *a*:

    1. Euclidean distances in that normalised space from *a* to every **other active** and to
       every decoy.
    2. Sweeping a distance threshold *t* gives two cumulative fractions — ``Sa(t)`` of the
       other actives within *t*, ``Sd(t)`` of the decoys within *t*. Under optimal embedding
       the two grow together, so the curve ``Sa`` vs ``Sd`` is the diagonal.
    3. ``DOE_a = ∫₀¹ |Sa − Sd| dSd``, by trapezoid over the realised curve.

    The score is the mean of ``DOE_a`` over actives. This is the pair of empirical CDFs of
    property-space distance — the decoys' compared against the actives' own — integrated,
    which is why it is bounded by 0.5 and why an unmatched set scores high.

    **Reference values**, for calibrating an implementation against the literature: DUD-E
    0.166, DeepCoy on the same targets 0.032 (−81%); DEKOIS 2.0 0.109, DeepCoy 0.038 (−66%)
    — methods document #18. These come from the six DUD-E matching properties, which is why
    :data:`DOE_PROPERTIES` is that set and not a wider one; a different property set is a
    different number and must not be compared against these.

    **There is a finite-sample floor**, so 0 is not reachable in practice: decoys drawn from
    the actives' own distribution score ~0.05 at 100 actives against 1,000 decoys, and more at
    smaller n, because each active's two CDFs are estimated from finite samples. Compare sets
    at comparable sizes, and read a difference of 0.01 between two sets of different size as
    noise. This is also why DeepCoy's 0.032 is near the achievable minimum rather than
    obviously improvable.

    Both arguments accept records or an already-computed :func:`property_summary` frame. The
    frame path is what makes this testable and RDKit-free.

    Raises ``ValueError`` for fewer than two actives (step 1 needs another active) or no
    decoys, and for a property column that is missing or entirely non-finite.
    """
    a_frame = _property_frame(actives, properties)
    d_frame = _property_frame(decoys, properties)
    if len(a_frame) < 2:
        raise ValueError(f"doe_score needs >= 2 actives with finite properties, got {len(a_frame)}")
    if len(d_frame) < 1:
        raise ValueError("doe_score needs >= 1 decoy with finite properties, got 0")

    pooled = np.vstack([a_frame.to_numpy(float), d_frame.to_numpy(float)])
    lo, hi = pooled.min(axis=0), pooled.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)  # a constant property is uninformative, not fatal
    act = (a_frame.to_numpy(float) - lo) / span
    dec = (d_frame.to_numpy(float) - lo) / span

    scores = []
    for i in range(len(act)):
        d_act = np.linalg.norm(np.delete(act, i, axis=0) - act[i], axis=1)
        d_dec = np.linalg.norm(dec - act[i], axis=1)
        scores.append(_embedding_deviation(np.sort(d_act), np.sort(d_dec)))
    return float(np.mean(scores))


def _embedding_deviation(sorted_actives: np.ndarray, sorted_decoys: np.ndarray) -> float:
    """``∫|Sa − Sd| dSd`` for one active's distance CDFs, both already sorted ascending."""
    thresholds = np.unique(np.concatenate([sorted_actives, sorted_decoys]))
    s_act = np.searchsorted(sorted_actives, thresholds, side="right") / len(sorted_actives)
    s_dec = np.searchsorted(sorted_decoys, thresholds, side="right") / len(sorted_decoys)
    # Anchor at the origin so the curve is integrated over the whole unit interval even when
    # the nearest neighbour of `a` is a decoy (Sd > 0 at the first threshold).
    s_act = np.concatenate([[0.0], s_act])
    s_dec = np.concatenate([[0.0], s_dec])
    return float(_trapezoid(np.abs(s_act - s_dec), s_dec))


#: ``np.trapz`` was renamed in NumPy 2 and deprecated in 1.26; bound once rather than warning
#: once per active.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _property_frame(
    source: Iterable[MolRecord] | pd.DataFrame, properties: Sequence[str]
) -> pd.DataFrame:
    frame = source if isinstance(source, pd.DataFrame) else property_summary(source)
    missing = [p for p in properties if p not in frame.columns]
    if missing:
        raise ValueError(f"property column(s) {missing} absent; have {list(frame.columns)}")
    subset = frame[list(properties)].apply(pd.to_numeric, errors="coerce")
    return subset.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
