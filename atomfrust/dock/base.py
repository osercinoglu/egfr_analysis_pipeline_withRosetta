"""The pose-backend contract, and the rule that keeps every backend optional.

**Backends are subprocess-only and probed at runtime.** No module here ever writes
``import smina`` or ``import gnina``. A backend is a binary on ``PATH``, located with
:func:`shutil.which` inside :meth:`SubprocessBackend.available`, and invoked with
:mod:`subprocess`. Two consequences, both deliberate:

* :meth:`available` **returns False, it does not raise**, when the binary is absent. A
  machine without GNINA loses the GNINA backend and nothing else — ``import atomfrust.dock``
  still works, the MCS backend still poses, the validity gate still runs. Any caller can
  therefore ask "what can this machine do?" without a try/except ladder.
* The binary's own version string, not a Python package version, is what provenance records.
  An in-process import would additionally drag a CUDA-linked torch into every worker that
  merely wanted to *read* a pose.

:class:`Pose3D` is the only currency between backends and everything downstream: an MDL
molblock (coordinates *and* bond orders, unlike PDB HETATM records), the backend's own score
when it has one, RMSD to the reference ligand when a reference was given, and a free-form
``metadata`` dict carrying the command line that produced it. ``score`` is ``None`` rather
than ``0.0`` for scoreless backends, because "no score" and "score of zero" are different
facts and only one of them may be averaged.

Nothing in this module may be used to skip :mod:`atomfrust.dock.posebusters`. A backend
produces candidate poses; the gate decides which of them are physically admissible (R23,
S4.2). The two are separate on purpose so a generator's pass rate is measurable rather than
assumed.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Pose3D",
    "PoseBackend",
    "SubprocessBackend",
    "BackendUnavailable",
    "BACKEND_NAMES",
    "get_backend",
    "list_backends",
    "available_backends",
    "rdkit_available",
    "require_rdkit",
    "RDKIT_IMPORT_ERROR",
    "mol_from_molblock",
    "molblock_from_mol",
    "as_mol_with_conformer",
    "embedded_mol",
    "symmetry_rmsd",
]


try:  # pragma: no cover - exercised by the environment, not the tests
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolAlign

    RDLogger.DisableLog("rdApp.*")
    RDKIT_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    Chem = AllChem = rdMolAlign = None  # type: ignore[assignment]
    RDKIT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class BackendUnavailable(RuntimeError):
    """A backend was *asked to work* while its binary or library is missing.

    Distinct from :meth:`PoseBackend.available` returning False, which is a question and
    must never raise. This is the answer to actually calling :meth:`PoseBackend.pose`.
    """


def rdkit_available() -> bool:
    return RDKIT_IMPORT_ERROR is None


def require_rdkit() -> None:
    if RDKIT_IMPORT_ERROR is not None:
        raise BackendUnavailable(f"RDKit is required here but not importable: {RDKIT_IMPORT_ERROR}")


@dataclass(frozen=True)
class Pose3D:
    """One posed copy of one molecule.

    ``molblock`` is MDL, not PDB: a pose that loses its bond orders cannot be re-parametrised
    and cannot be stereochemistry-checked, and PDB HETATM records carry neither.

    ``score`` is the backend's *own* number in the backend's *own* units — smina's affinity
    in kcal/mol, GNINA's CNN affinity in pK units, ``None`` for backends that do not score.
    It is a comparator (methods §3.7), never the frustration index, and scores from different
    backends are not comparable to each other.

    ``rmsd_to_reference`` is symmetry-corrected heavy-atom RMSD to the reference ligand,
    computed *without* re-alignment: the receptor frame is the whole point, so aligning the
    two molecules before measuring would erase the quantity being measured. ``None`` when no
    reference was supplied or when the reference is a different molecule.

    ``metadata`` records how the pose was made — binary path, full command line, seed, mode
    index. It is what makes a pose reproducible; downstream code may add to it but should
    not depend on any key beyond those a given backend documents.
    """

    molblock: str
    source: str
    score: float | None = None
    rmsd_to_reference: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mol(self) -> Any:
        """Parse back to an RDKit ``Mol``, hydrogens kept. Raises on an unparseable pose."""
        return mol_from_molblock(self.molblock)


@runtime_checkable
class PoseBackend(Protocol):
    """What every backend implements. Structural, not inherited — a caller may supply its own.

    ``pose`` takes the molecule to place (SMILES string or RDKit ``Mol``) and the receptor it
    is placed into, and returns up to ``n_poses`` poses ordered best-first by whatever the
    backend considers best. It raises :class:`BackendUnavailable` when the backend cannot
    run; ``available()`` is the way to ask first.

    ``reference_ligand`` is the native ligand — a molblock string, a path to a molfile/SDF,
    or an RDKit ``Mol`` with a conformer. Docking backends use it to place the search box;
    :class:`~atomfrust.dock.mcs_align.MCSAlignBackend` uses it as the template it aligns onto;
    all of them use it as the RMSD reference. Without it a docking backend must search the
    whole receptor, which is a different (blind-docking) experiment.

    ``seed`` is passed through to the backend's own RNG so a run is reproducible; where a
    backend generates several poses from one call, pose *k* is seeded ``seed + k``, matching
    the decoy-seeding convention used elsewhere in the package.
    """

    name: str

    def available(self) -> bool: ...

    def pose(
        self,
        smiles_or_mol: str | Any,
        receptor_pdb: Path | str,
        reference_ligand: str | Path | Any | None = None,
        n_poses: int = 1,
        seed: int = 0,
    ) -> list[Pose3D]: ...


class SubprocessBackend:
    """Shared machinery for backends that are an external binary.

    Subclasses set ``name``, ``binary`` and (optionally) ``version_args``. They inherit the
    contract that matters: :meth:`available` answers by looking on ``PATH`` and swallows even
    an exception from ``shutil.which`` — a broken ``PATH`` entry degrades one backend rather
    than breaking a survey over all of them.
    """

    name: str = "subprocess"
    binary: str = ""
    version_args: tuple[str, ...] = ("--version",)

    def __init__(self, binary: str | None = None, timeout_s: float = 900.0) -> None:
        if binary:
            self.binary = binary
        self.timeout_s = timeout_s

    def resolve_binary(self) -> str | None:
        try:
            return shutil.which(self.binary)
        except Exception:  # pragma: no cover - only a pathological PATH reaches this
            return None

    def available(self) -> bool:
        """True iff the binary is on ``PATH``. Never raises — see the module docstring."""
        return self.resolve_binary() is not None

    def version(self) -> str | None:
        """The binary's self-reported version, or ``None`` if it is absent or mute.

        Recorded in pose metadata: an ML docking tool's pass rate under the S4.2 gate is a
        property of a specific build, so a run that cannot name the build cannot be compared
        to another run.
        """
        path = self.resolve_binary()
        if path is None:
            return None
        try:
            result = subprocess.run(
                [path, *self.version_args], capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            return None
        text = (result.stdout or result.stderr).strip()
        return text.splitlines()[0] if text else None

    def _require_binary(self) -> str:
        path = self.resolve_binary()
        if path is None:
            raise BackendUnavailable(
                f"{self.name}: no {self.binary!r} on PATH. Install it and re-run, or pick "
                f"another backend — {', '.join(list_backends())}. "
                f"Use {type(self).__name__}().available() to test before calling."
            )
        return path

    def _run(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(args), capture_output=True, text=True, timeout=self.timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailable(
                f"{self.name} timed out after {self.timeout_s}s: {' '.join(args[:3])}..."
            ) from exc


#: Registry: backend name -> ``module:class``. Imported lazily so that naming a backend
#: costs nothing and so that :mod:`atomfrust.dock.base` stays importable on its own.
BACKEND_NAMES: dict[str, str] = {
    "mcs_align": "atomfrust.dock.mcs_align:MCSAlignBackend",
    "smina": "atomfrust.dock.smina:SminaBackend",
    "gnina": "atomfrust.dock.gnina:GninaBackend",
    "preposed": "atomfrust.dock.preposed:PrePosedBackend",
}

#: Spellings a user is likely to type. Kept tiny and explicit: a fuzzy resolver would let a
#: typo silently select a *different* generator, which is a silent change of experiment.
_ALIASES = {"mcs": "mcs_align", "mcs-align": "mcs_align", "pre_posed": "preposed",
            "pre-posed": "preposed", "none": "preposed"}


def list_backends() -> tuple[str, ...]:
    """Every registered backend name, whether or not this machine can run it."""
    return tuple(sorted(BACKEND_NAMES))


def get_backend(name: str, **kwargs: Any) -> PoseBackend:
    """Construct a backend by name. Unknown names raise with the known ones listed.

    Constructing a backend does **not** check that it can run: instantiation is free and
    ``available()`` is the question. That split lets a report tabulate "requested / available
    / used" instead of crashing on the first machine that lacks GNINA.
    """
    key = _ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    target = BACKEND_NAMES.get(key)
    if target is None:
        raise ValueError(
            f"unknown pose backend {name!r}. Known backends: {', '.join(list_backends())}"
            + (f" (aliases: {', '.join(sorted(_ALIASES))})" if _ALIASES else "")
        )
    module_name, class_name = target.split(":")
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)(**kwargs)


def available_backends() -> tuple[str, ...]:
    """The subset of :func:`list_backends` this machine can actually run, right now.

    A backend whose constructor raises is reported as unavailable rather than propagating:
    the question "what works here?" must be answerable on a machine where something is
    broken, since that is exactly when it is asked.
    """
    usable = []
    for name in list_backends():
        try:
            if get_backend(name).available():
                usable.append(name)
        except Exception:
            continue
    return tuple(usable)


# ------------------------------------------------------------------ molecule plumbing


def mol_from_molblock(molblock: str, *, sanitize: bool = True, remove_hs: bool = False) -> Any:
    """Parse an MDL molblock. Hydrogens are kept by default — a pose's Hs are data."""
    require_rdkit()
    mol = Chem.MolFromMolBlock(molblock, sanitize=sanitize, removeHs=remove_hs)
    if mol is None:
        raise ValueError("RDKit could not parse the molblock")
    return mol


def molblock_from_mol(mol: Any, conf_id: int = -1) -> str:
    require_rdkit()
    return Chem.MolToMolBlock(mol, confId=conf_id)


def as_mol_with_conformer(source: str | Path | Any) -> Any:
    """Coerce a molblock string, a molfile/SDF path, or a ``Mol`` into a ``Mol`` with 3D coords.

    Accepting all three is not convenience: the native ligand arrives as a file from the prep
    stage, as a molblock from another backend's :class:`Pose3D`, and as a ``Mol`` from a test.
    Refusing a molecule that carries no conformer is the point — a reference without
    coordinates cannot place a box or template an alignment.
    """
    require_rdkit()
    if source is None:
        raise ValueError("no reference ligand supplied")

    if isinstance(source, (str, Path)):
        text = str(source)
        looks_like_molblock = "M  END" in text or "V2000" in text or "V3000" in text
        if looks_like_molblock:
            mol = Chem.MolFromMolBlock(text, removeHs=False)
        else:
            path = Path(text)
            if not path.exists():
                raise ValueError(
                    f"reference ligand {text!r} is neither a molblock nor an existing file"
                )
            mol = _mol_from_file(path)
    else:
        mol = Chem.Mol(source)

    if mol is None:
        raise ValueError("RDKit could not parse the reference ligand")
    if mol.GetNumConformers() == 0:
        raise ValueError("reference ligand has no conformer; 3D coordinates are required")
    return mol


def _mol_from_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".sd"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        return next((m for m in supplier if m is not None), None)
    if suffix in {".mol", ".mdl"}:
        return Chem.MolFromMolFile(str(path), removeHs=False)
    if suffix == ".mol2":
        return Chem.MolFromMol2File(str(path), removeHs=False)
    if suffix in {".pdb", ".ent"}:
        return Chem.MolFromPDBFile(str(path), removeHs=False)
    raise ValueError(f"unsupported reference-ligand file type: {path.name}")


def embedded_mol(smiles_or_mol: str | Any, seed: int = 0) -> Any:
    """A 3D ``Mol`` for a docking backend to move around.

    An input that *already* has a conformer keeps it — re-embedding would throw away a
    library- or crystal-supplied geometry, the same rule
    :func:`atomfrust.chem.paramize.paramize` follows.
    """
    require_rdkit()
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES {smiles_or_mol!r}")
    else:
        mol = Chem.Mol(smiles_or_mol)
        Chem.SanitizeMol(mol)

    if mol.GetNumConformers() > 0:
        return Chem.AddHs(mol, addCoords=True)

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            raise ValueError("ETKDGv3 could not embed a starting conformer")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        # A failed minimisation still leaves a valid embedding, which is all a docking
        # program needs as a starting point — it rebuilds torsions itself.
        pass
    return mol_h


def symmetry_rmsd(probe: Any, reference: Any) -> float | None:
    """In-place, symmetry-corrected heavy-atom RMSD, or ``None`` if the two do not match.

    ``CalcRMS`` does not superimpose, which is required here: both molecules live in the
    receptor's frame and the distance between them *in that frame* is the quantity of
    interest (G5 keeps docked decoys at RMSD >= 2.0 A from native). Superimposing first would
    report conformer similarity instead.
    """
    require_rdkit()
    try:
        probe_heavy = Chem.RemoveHs(Chem.Mol(probe))
        ref_heavy = Chem.RemoveHs(Chem.Mol(reference))
        return float(rdMolAlign.CalcRMS(probe_heavy, ref_heavy))
    except Exception:
        # Different molecules (the chemotype axis) have no correspondence and therefore no
        # RMSD. That is a fact about the pair, not an error.
        return None


def iter_sdf_records(text: str) -> Iterable[str]:
    """Split SDF text into records, dropping the ``$$$$`` terminators and nothing else.

    Text in, text out: a pose that is parsed only to be re-serialised acquires RDKit's
    formatting, which makes a byte-level round-trip impossible and hides whether the original
    file was ever modified.

    Only the single newline that ended each ``$$$$`` line is removed from the following
    record. Stripping leading whitespace generally would eat the *first* line of the next
    molblock — the title line, which is legitimately empty in RDKit's output — and shift the
    whole record up by one, silently turning the program line into the title.
    """
    for position, chunk in enumerate(text.split("$$$$")):
        if "M  END" not in chunk:
            continue
        if position > 0:
            if chunk.startswith("\r\n"):
                chunk = chunk[2:]
            elif chunk.startswith("\n"):
                chunk = chunk[1:]
        yield chunk
