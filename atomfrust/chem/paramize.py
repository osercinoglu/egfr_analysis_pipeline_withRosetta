"""Ligand parametrisation at library scale.

``scripts/05_prepare_ligand.py`` parametrises *one crystal ligand at a time*: it downloads a
CCD entry, converts it, runs ``molfile_to_params``, and validates the result against the PDB
files that contain that ligand. A screening library has no CCD entry and no PDB file, so two
things have to change and nothing else does.

**The atom-name gate stops being universal.** ``scripts/05:111-145`` (``verify_atom_names``)
collects the ``ATOM`` names out of the ``.params`` and requires that, for *every* processed
PDB using that ligand, no params atom is absent from the HETATM records and no heavy PDB atom
is absent from the params (extra hydrogens in the PDB are tolerated, because Rosetta rebuilds
them). That check is load-bearing for a crystal ligand — ``fill_missing_atoms`` fails at pose
load if the names disagree, which is why the CCD-CIF-first cascade exists at all. For a
library molecule there is no HETATM record to match, so the gate has nothing to compare
against. It is therefore a parameter, ``require_atom_name_match``, defaulting to **False**;
crystal callers pass ``True`` plus ``reference_pdbs``.

**Codes are allocated, not looked up.** See :mod:`atomfrust.chem.codes`.

**Failures are data.** :func:`paramize` never raises for a bad molecule. It returns a
:class:`ParamRecord` whose ``failure`` is a member of the closed :class:`ParamFailure`
taxonomy, so the methods document's "all failures categorised" (S1.1) is a ``groupby``
over :meth:`atomfrust.chem.cache.ParamCache.failures` rather than a log grep. Exceptions are
still raised for *programming* errors — a missing output directory, a malformed code — which
are not properties of the molecule.

The backend is the vendored ``src/molfile_to_params.py`` invoked through
``prepare_structures.run_molfile_to_params``: it sets ``PYTHONPATH`` to ``src/`` (so the
vendored ``src/rosetta_py/`` package, which PyRosetta does not ship, is importable) and runs
with ``cwd=params_dir`` (the script writes ``NAME.params`` and ``NAME_0001.pdb`` relative to
the working directory). Both are reproduced by importing the function rather than
reimplementing the call, because CLAUDE.md records that neither can be "cleaned up".

No PyRosetta import at module scope. The optional Rosetta validation runs in a subprocess for
the same reason ``scripts/05:163-190`` does: ``pyrosetta.init`` takes ``-extra_res_fa`` once
per process, and a library has one params file per molecule.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atomfrust.chem.codes import CodeAllocator
from atomfrust.provenance import sha256_file

if TYPE_CHECKING:  # pragma: no cover
    from atomfrust.chem.cache import ParamCache

__all__ = [
    "ParamFailure",
    "ParamRecord",
    "paramize",
    "SUPPORTED_ELEMENTS",
    "RDKIT_IMPORT_ERROR",
    "rdkit_available",
]


class ParamFailure(str, Enum):
    """The closed failure taxonomy. Every unusable molecule is exactly one of these.

    Closed on purpose: a new member is a schema change to ``failures.parquet`` and to every
    downstream tally, so it is a deliberate edit rather than a new string appearing in a log.
    """

    SANITIZE_FAIL = "SANITIZE_FAIL"
    UNSUPPORTED_ELEMENT = "UNSUPPORTED_ELEMENT"
    CHARGE_UNBALANCED = "CHARGE_UNBALANCED"
    CONFORMER_FAIL = "CONFORMER_FAIL"
    ATOMTYPE_UNASSIGNED = "ATOMTYPE_UNASSIGNED"
    PARAMS_EMPTY = "PARAMS_EMPTY"
    ROSETTA_LOAD_FAIL = "ROSETTA_LOAD_FAIL"
    ROSETTA_SCORE_NONFINITE = "ROSETTA_SCORE_NONFINITE"


@dataclass(frozen=True)
class ParamRecord:
    """One molecule's parametrisation outcome — success and failure in the same shape.

    ``rosetta_code`` is the *pose* name (:mod:`atomfrust.chem.codes`), never the reporting
    identity; ``inchikey`` is the identity. ``sha256`` digests the params file so a run
    manifest can prove which parametrisation produced its numbers.
    """

    inchikey: str
    smiles: str
    rosetta_code: str
    params_path: Path | None
    sha256: str | None
    failure: ParamFailure | None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.failure is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inchikey": self.inchikey,
            "smiles": self.smiles,
            "rosetta_code": self.rosetta_code,
            "params_path": str(self.params_path) if self.params_path is not None else None,
            "sha256": self.sha256,
            "failure": self.failure.value if self.failure is not None else None,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParamRecord:
        params_path = data.get("params_path")
        failure = data.get("failure")
        return cls(
            inchikey=data["inchikey"],
            smiles=data.get("smiles", ""),
            rosetta_code=data.get("rosetta_code", ""),
            params_path=Path(params_path) if params_path else None,
            sha256=data.get("sha256"),
            failure=ParamFailure(failure) if failure else None,
            message=data.get("message", "") or "",
        )


#: Elements ``molfile_to_params`` maps onto a real Rosetta atom type by an explicit rule
#: (``src/molfile_to_params.py:241-322``). Anything else either raises "Unknown element" in
#: the script or falls through to its ``ATOMS`` table (``:53-61``), which emits an
#: element-symbol type such as ``Se`` that ``fa_standard`` cannot score — a params file that
#: is written but unusable. Rejecting up front turns that into one categorised failure
#: instead of a downstream pose-load crash.
SUPPORTED_ELEMENTS = frozenset(
    "H C N O S P F Cl Br I B Na K Mg Fe Ca Zn Co Cu".split()
)

try:  # pragma: no cover - exercised only by the environment, not the tests
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    RDKIT_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    Chem = AllChem = None  # type: ignore[assignment]
    RDKIT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def rdkit_available() -> bool:
    return RDKIT_IMPORT_ERROR is None


class RDKitUnavailable(RuntimeError):
    """Raised only by callers that opt into hard failure; :func:`paramize` categorises instead.

    RDKit is a hard dependency of the library path (sanitisation, InChIKey, conformers) but
    not of the crystal path, which reads coordinates from the PDB. Rather than making
    ``import atomfrust.chem`` fail on a machine that only runs crystal structures, absence is
    reported per molecule as ``SANITIZE_FAIL`` with the import error in ``message``.
    """


def paramize(
    smiles_or_mol: str | Any,
    *,
    out_dir: Path | str | None = None,
    cache: ParamCache | None = None,
    allocator: CodeAllocator | None = None,
    rosetta_code: str | None = None,
    require_atom_name_match: bool = False,
    reference_pdbs: Sequence[Path | str] = (),
    reference_comp_id: str | None = None,
    validate_in_rosetta: bool = False,
    conformer_seed: int = 0xC0FFEE,
    charge_tolerance: float = 0.5,
    refresh: bool = False,
    timeout_s: float = 600.0,
) -> ParamRecord:
    """Parametrise one molecule. Returns a :class:`ParamRecord`; does not raise for bad input.

    Pipeline: RDKit sanitise -> element gate -> ETKDGv3 embedding + MMFF (UFF fallback)
    -> SDF -> vendored ``molfile_to_params`` -> params sanity checks.

    ``smiles_or_mol`` is a SMILES string or an RDKit ``Mol``. A ``Mol`` that already carries
    a conformer keeps it — that is how a docked or library-supplied 3D pose survives, since
    re-embedding would throw the geometry away.

    ``out_dir`` is where the SDF and the ``.params`` land; it defaults to the cache's params
    directory. One of ``out_dir`` or ``cache`` is required.

    ``cache`` short-circuits on a hit, **including a cached failure** — re-running a library
    must not re-run RDKit and a subprocess for every molecule that was already categorised.
    Pass ``refresh=True`` to recompute and overwrite.

    ``require_atom_name_match`` is the crystal-ligand gate described in the module docstring.
    It needs ``reference_pdbs`` and ``reference_comp_id`` (the name in the HETATM records,
    i.e. the *Rosetta* code when Stage 4 rewrote it). A mismatch is reported as
    :attr:`ParamFailure.ROSETTA_LOAD_FAIL`: no member of the closed taxonomy names atom-name
    disagreement, and load failure is precisely what it causes (``fill_missing_atoms``).

    ``validate_in_rosetta`` runs the params through a real pose in a subprocess, which is the
    only way to reach :attr:`ParamFailure.ROSETTA_LOAD_FAIL` and
    :attr:`ParamFailure.ROSETTA_SCORE_NONFINITE` honestly. Off by default: it costs a
    PyRosetta init per molecule.
    """
    smiles_in = smiles_or_mol if isinstance(smiles_or_mol, str) else ""
    # Before the RDKit guard: "nowhere to write" is a caller bug and must raise identically
    # whether or not RDKit happens to be installed.
    params_dir = _resolve_out_dir(out_dir, cache)

    if RDKIT_IMPORT_ERROR is not None:
        return ParamRecord(
            inchikey="",
            smiles=smiles_in,
            rosetta_code=rosetta_code or "",
            params_path=None,
            sha256=None,
            failure=ParamFailure.SANITIZE_FAIL,
            message=f"RDKitUnavailable: {RDKIT_IMPORT_ERROR}",
        )

    mol, smiles, problem = _sanitised_mol(smiles_or_mol)
    if problem is not None:
        return ParamRecord(
            inchikey="", smiles=smiles or smiles_in, rosetta_code=rosetta_code or "",
            params_path=None, sha256=None,
            failure=ParamFailure.SANITIZE_FAIL, message=problem,
        )

    try:
        inchikey = Chem.MolToInchiKey(mol) or ""
    except Exception as exc:
        inchikey = ""
        problem = f"InChIKey generation failed: {type(exc).__name__}: {exc}"
    if not inchikey:
        return ParamRecord(
            inchikey="", smiles=smiles, rosetta_code=rosetta_code or "",
            params_path=None, sha256=None, failure=ParamFailure.SANITIZE_FAIL,
            message=problem or "InChIKey generation returned an empty string",
        )

    if cache is not None and not refresh:
        hit = cache.get(inchikey)
        if hit is not None:
            return hit

    record = _paramize_sanitised(
        mol=mol,
        smiles=smiles,
        inchikey=inchikey,
        params_dir=params_dir,
        allocator=allocator,
        rosetta_code=rosetta_code,
        require_atom_name_match=require_atom_name_match,
        reference_pdbs=reference_pdbs,
        reference_comp_id=reference_comp_id,
        validate_in_rosetta=validate_in_rosetta,
        conformer_seed=conformer_seed,
        charge_tolerance=charge_tolerance,
        timeout_s=timeout_s,
    )
    if cache is not None:
        cache.put(record)
    return record


# --------------------------------------------------------------------------- stages


def _resolve_out_dir(out_dir: Path | str | None, cache: ParamCache | None) -> Path:
    if out_dir is not None:
        path = Path(out_dir)
    elif cache is not None:
        path = cache.params_dir
    else:
        raise ValueError("paramize needs out_dir or cache to know where to write .params")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitised_mol(smiles_or_mol: str | Any) -> tuple[Any, str, str | None]:
    """Return ``(mol, canonical_smiles, problem)``. ``problem`` non-None means SANITIZE_FAIL."""
    if isinstance(smiles_or_mol, str):
        text = smiles_or_mol.strip()
        if not text:
            return None, "", "empty SMILES"
        mol = Chem.MolFromSmiles(text)  # sanitises; returns None on failure
        if mol is None:
            return None, text, f"RDKit could not parse SMILES {text!r}"
    elif smiles_or_mol is None:
        return None, "", "molecule is None"
    else:
        mol = Chem.Mol(smiles_or_mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            return None, "", f"RDKit sanitisation failed: {type(exc).__name__}: {exc}"

    try:
        canonical = Chem.MolToSmiles(mol)
    except Exception as exc:
        return None, "", f"canonical SMILES failed: {type(exc).__name__}: {exc}"
    if mol.GetNumAtoms() == 0:
        return None, canonical, "molecule has no atoms"
    return mol, canonical, None


def _paramize_sanitised(
    *,
    mol: Any,
    smiles: str,
    inchikey: str,
    params_dir: Path,
    allocator: CodeAllocator | None,
    rosetta_code: str | None,
    require_atom_name_match: bool,
    reference_pdbs: Sequence[Path | str],
    reference_comp_id: str | None,
    validate_in_rosetta: bool,
    conformer_seed: int,
    charge_tolerance: float,
    timeout_s: float,
) -> ParamRecord:
    code = rosetta_code or (allocator or CodeAllocator()).allocate(inchikey)
    code = code.strip().upper()
    if not (1 <= len(code) <= 3):
        raise ValueError(f"rosetta_code must be 1-3 characters, got {code!r}")

    def fail(failure: ParamFailure, message: str) -> ParamRecord:
        return ParamRecord(inchikey, smiles, code, None, None, failure, message)

    unsupported = sorted(
        {a.GetSymbol() for a in mol.GetAtoms()} - set(SUPPORTED_ELEMENTS)
    )
    if unsupported:
        return fail(
            ParamFailure.UNSUPPORTED_ELEMENT,
            f"no Rosetta atom type for: {', '.join(unsupported)}",
        )

    mol3d, conformer_problem = _embed(mol, conformer_seed)
    if conformer_problem is not None:
        return fail(ParamFailure.CONFORMER_FAIL, conformer_problem)

    sdf_path = params_dir / f"{code}.sdf"
    try:
        mol3d.SetProp("_Name", code)
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol3d)
        writer.close()
    except Exception as exc:
        return fail(ParamFailure.CONFORMER_FAIL, f"SDF write failed: {type(exc).__name__}: {exc}")

    params_path, backend_problem = _run_backend(sdf_path, params_dir, code)
    if backend_problem is not None:
        return fail(ParamFailure.PARAMS_EMPTY, backend_problem)

    atoms = _params_atoms(params_path)
    if not atoms:
        return fail(ParamFailure.PARAMS_EMPTY, f"{params_path.name} contains no ATOM lines")

    untyped = sorted(name for name, ros_type, _ in atoms if ros_type in ("", "X", "XXX"))
    if untyped:
        return fail(
            ParamFailure.ATOMTYPE_UNASSIGNED,
            f"{len(untyped)} atom(s) without a Rosetta type: {', '.join(untyped[:8])}",
        )

    net = sum(charge for _, _, charge in atoms)
    formal = Chem.GetFormalCharge(mol)
    if abs(net - formal) > charge_tolerance:
        return fail(
            ParamFailure.CHARGE_UNBALANCED,
            f"params partial charges sum to {net:.2f}, formal charge is {formal} "
            f"(tolerance {charge_tolerance})",
        )

    if require_atom_name_match:
        problem = _atom_name_mismatch(
            {name for name, _, _ in atoms}, reference_pdbs, reference_comp_id or code
        )
        if problem is not None:
            return fail(ParamFailure.ROSETTA_LOAD_FAIL, problem)

    note = f"{len(atoms)} atoms"
    if validate_in_rosetta:
        failure, message = _validate_in_rosetta(
            params_path, params_dir / f"{code}_0001.pdb", timeout_s
        )
        if failure is not None:
            return fail(failure, message)
        # Keep the score in the record: it is the evidence that validation actually ran.
        note = f"{note}, {message}"

    return ParamRecord(
        inchikey=inchikey,
        smiles=smiles,
        rosetta_code=code,
        params_path=params_path,
        sha256=sha256_file(params_path),
        failure=None,
        message=note,
    )


def _embed(mol: Any, seed: int) -> tuple[Any, str | None]:
    """ETKDGv3 + MMFF, UFF fallback. An existing conformer is kept, not overwritten."""
    if mol.GetNumConformers() > 0:
        return Chem.AddHs(mol, addCoords=True), None

    mol_h = Chem.AddHs(mol)
    settings = AllChem.ETKDGv3()
    settings.randomSeed = seed
    try:
        status = AllChem.EmbedMolecule(mol_h, settings)
        if status != 0:
            # Macrocycles and heavily constrained rings routinely need this second pass.
            settings.useRandomCoords = True
            status = AllChem.EmbedMolecule(mol_h, settings)
    except Exception as exc:
        return mol_h, f"embedding raised {type(exc).__name__}: {exc}"
    if status != 0:
        return mol_h, "ETKDGv3 could not embed a conformer (with and without random coords)"

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        # A failed minimisation is not a failed conformer: the embedded geometry is still
        # valid input for atom typing, which is all molfile_to_params reads.
        pass
    return mol_h, None


def _vendored_src_dir() -> Path | None:
    """Locate the repo's ``src/`` — it holds ``molfile_to_params.py`` and ``rosetta_py/``."""
    candidates = [Path(__file__).resolve().parents[2] / "src", Path("src").resolve()]
    for candidate in candidates:
        if (candidate / "molfile_to_params.py").exists() and (candidate / "rosetta_py").is_dir():
            return candidate
    return None


def _run_backend(sdf_path: Path, params_dir: Path, code: str) -> tuple[Path, str | None]:
    """Call the vendored backend through ``prepare_structures.run_molfile_to_params``.

    Imported rather than reimplemented: that function is the one place that gets
    ``PYTHONPATH=src`` and ``cwd=params_dir`` right, and both are load-bearing.
    """
    src_dir = _vendored_src_dir()
    if src_dir is None:
        return params_dir / f"{code}.params", (
            "vendored src/molfile_to_params.py + src/rosetta_py/ not found; "
            "run from the repo root"
        )
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from prepare_structures import run_molfile_to_params  # noqa: PLC0415

    params_path = run_molfile_to_params(sdf_path, params_dir, code, src_dir / "molfile_to_params.py")
    if not params_path.exists():
        return params_path, f"molfile_to_params produced no {params_path.name}"
    return params_path, None


def _params_atoms(params_path: Path) -> list[tuple[str, str, float]]:
    """``(name, rosetta_type, partial_charge)`` per ATOM line.

    The line is fixed-width — ``"ATOM %-4s %-4s %-4s %.2f"`` at
    ``src/molfile_to_params.py:1053`` — so the name is read by column, exactly as
    ``scripts/05:121`` does. Atom names may contain no spaces, but a *type* field can be
    blank in files from other backends, which is what makes ATOMTYPE_UNASSIGNED reachable.
    """
    atoms: list[tuple[str, str, float]] = []
    for line in params_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[5:9].strip()
        ros_type = line[10:14].strip()
        try:
            charge = float(line[20:].strip() or line.split()[-1])
        except (ValueError, IndexError):
            charge = 0.0
        atoms.append((name, ros_type, charge))
    return atoms


def _is_hydrogen_name(atom_name: str) -> bool:
    """PDB hydrogen naming: leading ``H``, or a digit then ``H`` (``1HB``). Same rule as
    ``scripts/05:104-108``."""
    stripped = atom_name.strip()
    return stripped.startswith("H") or (
        len(stripped) > 1 and stripped[1] == "H" and stripped[0].isdigit()
    )


def _atom_name_mismatch(
    params_atoms: set[str], reference_pdbs: Sequence[Path | str], comp_id: str
) -> str | None:
    """The crystal-ligand gate. ``None`` means every reference PDB agrees.

    Mirrors ``verify_atom_names`` (``scripts/05:111-145``): every params atom must be present
    in the HETATM records, and every *heavy* HETATM atom must be present in the params. Extra
    hydrogens in the PDB are not a mismatch — Rosetta rebuilds hydrogens from ideal geometry.
    """
    paths = [Path(p) for p in reference_pdbs]
    if not paths:
        return "require_atom_name_match=True but no reference_pdbs were given"

    for path in paths:
        if not path.exists():
            return f"reference PDB not found: {path}"
        pdb_atoms = {
            line[12:16].strip()
            for line in path.read_text().splitlines()
            if line.startswith("HETATM") and line[17:20].strip() == comp_id
        }
        if not pdb_atoms:
            return f"no HETATM {comp_id} records in {path.name}"
        only_in_params = params_atoms - pdb_atoms
        only_in_pdb_heavy = {a for a in pdb_atoms if not _is_hydrogen_name(a)} - params_atoms
        if only_in_params or only_in_pdb_heavy:
            return (
                f"atom names disagree with {path.name}: "
                f"params-only {sorted(only_in_params)}, pdb-only {sorted(only_in_pdb_heavy)}"
            )
    return None


_ROSETTA_CHECK = """
import math, sys
import pyrosetta
pyrosetta.init("-extra_res_fa {params} -mute all -in:file:load_PDB_components false", silent=True)
pose = pyrosetta.pose_from_file("{conformer}")
if pose.total_residue() == 0:
    print("RESULT loaded=0 score=nan")
    sys.exit(0)
score = pyrosetta.create_score_function("ref2015")(pose)
print("RESULT loaded=1 score=%r" % score)
"""


def _validate_in_rosetta(
    params_path: Path, conformer_pdb: Path, timeout_s: float
) -> tuple[ParamFailure | None, str]:
    """Load and score the params in a fresh process.

    A subprocess, not this one: ``pyrosetta.init`` takes ``-extra_res_fa`` once per process
    and a library needs one per molecule. ``scripts/05:163-190`` does the same thing for the
    same reason.
    """
    if not conformer_pdb.exists():
        return ParamFailure.ROSETTA_LOAD_FAIL, f"no conformer PDB at {conformer_pdb}"
    code = _ROSETTA_CHECK.format(
        params=params_path.resolve(), conformer=conformer_pdb.resolve()
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ParamFailure.ROSETTA_LOAD_FAIL, f"pose load timed out after {timeout_s}s"

    line = next((l for l in result.stdout.splitlines() if l.startswith("RESULT")), None)
    if line is None:
        return ParamFailure.ROSETTA_LOAD_FAIL, (result.stderr[-400:] or "no RESULT line")
    fields = dict(kv.split("=", 1) for kv in line.split()[1:])
    if fields.get("loaded") != "1":
        return ParamFailure.ROSETTA_LOAD_FAIL, "pose loaded zero residues"
    try:
        score = float(fields["score"])
    except (KeyError, ValueError):
        return ParamFailure.ROSETTA_SCORE_NONFINITE, f"unparseable score in {line!r}"
    if not math.isfinite(score):
        return ParamFailure.ROSETTA_SCORE_NONFINITE, f"REF2015 score is {score}"
    return None, f"REF2015 {score:.2f}"
