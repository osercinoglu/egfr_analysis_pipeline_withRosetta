"""The validity gate: no pose enters frustration analysis until it has passed physical checks.

R23 and success criterion S4.2 (">=90% of poses entering analysis pass all PoseBusters
checks") exist because ML docking and co-folding methods are documented to produce poses that
score well and are physically impossible — stretched bonds, atoms inside each other, ligands
outside the protein. A frustration index computed on such a pose is a number without a
referent, and it will not look wrong: the energy function will happily report a large
``fa_rep``, which the Z-score then normalises away.

**Two implementations, always named.** :func:`check_poses` runs the real `posebusters`
package when it is importable. When it is not, it runs a **built-in subset** — fewer checks,
looser, written here — and the ``checker`` column of every returned row says which one ran
(``"posebusters"`` or ``"builtin_subset"``). The column is not decoration: a report that
claims S4.2 while the subset ran is claiming something weaker than the criterion states, and
the only way to prevent that is to make the distinction impossible to lose. ``check_poses``
never upgrades a subset result to the real name, and passing ``checker="posebusters"``
explicitly raises when the package is absent rather than quietly substituting.

**What the built-in subset does and does not do.** It implements, from the pose alone:

* ``sanitization`` — the molblock parses and sanitises at all;
* ``bond_lengths`` and ``bond_angles`` — every bond, and every 1-3 distance (which is how
  distance geometry encodes an angle), lies within a slack factor of the bounds RDKit's
  ETKDG machinery would have used to embed this molecule
  (:func:`rdkit.Chem.rdDistGeom.GetMoleculeBoundsMatrix`);
* ``internal_steric_clash`` — no pair of atoms separated by four or more bonds is closer than
  the same bounds matrix's lower bound allows;
* ``protein_ligand_clash`` — no ligand heavy atom is closer to a receptor heavy atom than
  ``protein_vdw_scale`` times the sum of their van der Waals radii;
* ``inside_receptor_bbox`` — every ligand atom lies inside the receptor's bounding box plus a
  margin, a crude guard against a pose placed in empty space;
* ``stereochemistry_preserved`` — stereochemistry read back from the 3D coordinates matches
  the intended molecule (from ``metadata["reference_smiles"]`` or ``metadata["smiles"]``);
* ``formal_charge_preserved`` — same, for total formal charge.

Real PoseBusters additionally checks, among others, that the predicted molecule loads and
sanitises against an explicit *true* molecule, that all atoms are connected, molecular formula
and bond identity, tetrahedral chirality and double-bond stereochemistry as *separate*
verdicts, aromatic-ring and double-bond flatness, an internal-energy ceiling relative to an
ETKDG ensemble, distance to organic and inorganic cofactors and to waters as distinct checks,
volume overlap with the protein (not merely a minimum distance), and RMSD to a reference
ligand. The subset conflates constitution with stereochemistry in one verdict, approximates
volume overlap with a minimum distance, and has no internal-energy or flatness check at all.
It is a floor, not a substitute: it will catch a mangled pose and will pass poses PoseBusters
would reject.

Output is long, one row per pose per check, so a failure rate is a ``groupby`` and the
failure *mode* is named — which is what S4.2 requires when a generator falls below 90%.
"""

from __future__ import annotations

import importlib.util
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atomfrust.dock.base import BackendUnavailable, Chem, Pose3D, rdkit_available

__all__ = [
    "POSEBUSTERS_CHECKS",
    "QC_COLUMNS",
    "S4_2_MIN_PASS_RATE",
    "CheckTolerances",
    "check_poses",
    "posebusters_available",
    "pose_pass_rate",
    "meets_s4_2",
    "passing_pose_indices",
    "filter_valid",
]


#: The checks the built-in subset guarantees. Real PoseBusters emits its own, larger set of
#: names; the ``check`` column carries whatever actually ran, and this tuple documents the
#: floor a caller can rely on being present under either implementation's ``checker`` value.
POSEBUSTERS_CHECKS: tuple[str, ...] = (
    "sanitization",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "protein_ligand_clash",
    "inside_receptor_bbox",
    "stereochemistry_preserved",
    "formal_charge_preserved",
)

#: Schema of the frame :func:`check_poses` returns — the ``pose_qc.parquet`` of the run
#: directory contract.
QC_COLUMNS: tuple[str, ...] = (
    "pose_index",
    "pose_id",
    "source",
    "checker",
    "check",
    "passed",
    "skipped",
    "value",
    "detail",
)

#: S4.2. Not a tunable: it is the number in the methods document.
S4_2_MIN_PASS_RATE = 0.90


@dataclass(frozen=True)
class CheckTolerances:
    """Slack for the built-in subset, all of it deliberate rather than tuned.

    ``bounds_slack`` widens the distance-geometry bounds by +-25% before a bond length or
    angle is called a violation. The bounds are the *embedding* limits; a docked pose has
    been through someone else's force field and will not reproduce them exactly, so a tight
    comparison would fail well-formed poses and make the gate useless.

    ``min_topological_separation`` is 4 bonds: 1-2 is a bond, 1-3 is an angle, 1-4 is a
    torsion and can legitimately be short in a syn conformation. Only atoms further apart
    than that have a genuinely non-bonded distance.

    ``protein_vdw_scale`` of 0.75 means a contact closer than three quarters of the vdW
    contact distance is a clash. Hydrogen bonds sit at roughly 0.8 and must survive.
    """

    bounds_slack: float = 0.25
    clash_slack: float = 0.25
    min_topological_separation: int = 4
    protein_vdw_scale: float = 0.75
    bbox_margin: float = 5.0
    skip_waters: bool = True


def posebusters_available() -> bool:
    """Whether the real package can be imported. Never imports it — that costs seconds."""
    try:
        return importlib.util.find_spec("posebusters") is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


def check_poses(
    poses: Sequence[Pose3D],
    receptor_pdb: Path | str | None = None,
    *,
    checker: str = "auto",
    tolerances: CheckTolerances | None = None,
    config: str | None = None,
) -> pd.DataFrame:
    """Run the validity gate. One row per pose per check; columns are :data:`QC_COLUMNS`.

    ``checker`` is ``"auto"`` (real PoseBusters if importable, otherwise the built-in
    subset), ``"posebusters"`` (raise if absent — for a run that must not silently weaken its
    own criterion) or ``"builtin_subset"`` (force the subset, which is how the subset is
    tested on a machine that has the real thing).

    ``receptor_pdb`` is optional and its absence is *recorded*, not worked around: the two
    protein-aware checks come back with ``skipped=True`` rather than a fabricated pass, so a
    pass rate computed over ligand-only checks cannot be mistaken for one that included the
    protein.

    An empty ``poses`` returns an empty frame with the right columns, so a caller can
    concatenate results across systems without special-casing the system that produced
    nothing.
    """
    if checker not in {"auto", "posebusters", "builtin_subset"}:
        raise ValueError(
            f"checker must be 'auto', 'posebusters' or 'builtin_subset', got {checker!r}"
        )
    if not poses:
        return _empty_frame()

    use_real = checker == "posebusters" or (checker == "auto" and posebusters_available())
    if checker == "posebusters" and not posebusters_available():
        raise BackendUnavailable(
            "checker='posebusters' was requested but the posebusters package is not "
            "importable. Install it (`pip install posebusters`) or pass "
            "checker='builtin_subset' and accept that the result is weaker than S4.2."
        )
    if use_real:
        return _check_with_posebusters(poses, receptor_pdb, config=config)

    if not rdkit_available():
        raise BackendUnavailable(
            "the built-in check subset needs RDKit, which is not importable"
        )
    return _check_builtin(poses, receptor_pdb, tolerances or CheckTolerances())


# ------------------------------------------------------------------ real PoseBusters


def _check_with_posebusters(
    poses: Sequence[Pose3D], receptor_pdb: Path | str | None, *, config: str | None
) -> pd.DataFrame:
    """Delegate to the real package and reshape its wide frame into the long schema.

    PoseBusters is given files rather than in-memory molecules: its file readers are the
    supported entry point and they carry the molecule name into the result index, which is
    how each row is mapped back to a pose.

    ``config`` defaults to ``"dock"`` when a receptor is available and ``"mol"`` when it is
    not, which is PoseBusters' own distinction between the protein-aware and ligand-only
    check sets — the same distinction the built-in subset makes by skipping two checks.
    """
    from posebusters import PoseBusters  # noqa: PLC0415 - deliberately not a module import

    mode = config or ("dock" if receptor_pdb is not None else "mol")
    with tempfile.TemporaryDirectory(prefix="posebusters_") as tmp:
        tmpdir = Path(tmp)
        paths = []
        for index, pose in enumerate(poses):
            path = tmpdir / f"pose_{index:04d}.sdf"
            path.write_text(pose.molblock.rstrip("\n") + "\n$$$$\n")
            paths.append(path)
        buster = PoseBusters(config=mode)
        wide = buster.bust(
            mol_pred=paths,
            mol_true=None,
            mol_cond=Path(receptor_pdb) if receptor_pdb is not None else None,
            full_report=False,
        )

    rows: list[dict[str, Any]] = []
    for position, (_, record) in enumerate(wide.iterrows()):
        index = min(position, len(poses) - 1)
        pose = poses[index]
        for name, value in record.items():
            if not isinstance(value, (bool, np.bool_)):
                # Non-boolean columns are diagnostics (distances, counts), not verdicts.
                continue
            rows.append(
                _row(
                    index,
                    pose,
                    checker="posebusters",
                    check=str(name),
                    passed=bool(value),
                )
            )
    return _frame(rows)


# ------------------------------------------------------------------ built-in subset


def _check_builtin(
    poses: Sequence[Pose3D], receptor_pdb: Path | str | None, tol: CheckTolerances
) -> pd.DataFrame:
    receptor_xyz, receptor_radii = (
        _receptor_heavy_atoms(Path(receptor_pdb), tol) if receptor_pdb is not None else (None, None)
    )
    rows: list[dict[str, Any]] = []
    for index, pose in enumerate(poses):
        rows.extend(_check_one(index, pose, receptor_xyz, receptor_radii, tol))
    return _frame(rows)


def _check_one(
    index: int,
    pose: Pose3D,
    receptor_xyz: np.ndarray | None,
    receptor_radii: np.ndarray | None,
    tol: CheckTolerances,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, *, skipped: bool = False, value: float | None = None,
            detail: str = "") -> None:
        rows.append(
            _row(index, pose, checker="builtin_subset", check=check, passed=passed,
                 skipped=skipped, value=value, detail=detail)
        )

    mol = None
    parse_error = ""
    try:
        mol = Chem.MolFromMolBlock(pose.molblock, removeHs=False, sanitize=True)
    except Exception as exc:  # RDKit raises on some malformed blocks rather than returning None
        parse_error = f"{type(exc).__name__}: {exc}"
    if mol is None or mol.GetNumConformers() == 0:
        add("sanitization", False, detail=parse_error or "molblock did not parse into a 3D molecule")
        for name in POSEBUSTERS_CHECKS[1:]:
            add(name, True, skipped=True, detail="pose did not parse")
        return rows
    add("sanitization", True)

    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    distances = _pairwise(coords)

    bounds, bounds_error = _bounds_matrix(mol)
    if bounds is None:
        for name in ("bond_lengths", "bond_angles", "internal_steric_clash"):
            add(name, True, skipped=True, detail=bounds_error)
    else:
        topology = Chem.GetDistanceMatrix(mol)
        _add_geometry_checks(add, mol, distances, bounds, topology, tol)

    if receptor_xyz is None or receptor_xyz.size == 0:
        add("protein_ligand_clash", True, skipped=True, detail="no receptor supplied")
        add("inside_receptor_bbox", True, skipped=True, detail="no receptor supplied")
    else:
        _add_receptor_checks(add, mol, coords, receptor_xyz, receptor_radii, tol)

    _add_identity_checks(add, mol, pose)
    return rows


def _add_geometry_checks(add, mol, distances, bounds, topology, tol: CheckTolerances) -> None:
    """Bonds, angles and internal clashes, all against the distance-geometry bounds matrix.

    One matrix, three checks, because that is what the matrix is: ``bounds[i][j]`` (i<j) is
    the largest distance ETKDG would allow between atoms i and j, ``bounds[j][i]`` the
    smallest. A bond is a 1-2 entry, an angle is a 1-3 entry — fixing the two bond lengths
    fixes the angle iff the 1-3 distance is right — and a steric clash is any further pair
    below its lower bound.
    """
    n = mol.GetNumAtoms()
    upper = np.triu(bounds) + np.triu(bounds, 1).T
    lower = np.tril(bounds) + np.tril(bounds, -1).T

    bonded = np.zeros((n, n), dtype=bool)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded[i, j] = bonded[j, i] = True

    angle = (topology == 2)
    # Heavy atoms only for the clash check. Hydrogen positions in a docked pose are usually
    # rebuilt from idealised geometry rather than searched, so an H...H contact reports on
    # whoever added the hydrogens, not on the pose. Bonds and angles keep their hydrogens:
    # there a wrong H position is a real geometric defect.
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()], dtype=bool)
    far = (topology >= tol.min_topological_separation) & heavy[:, None] & heavy[None, :]

    _add_bounds_check(add, "bond_lengths", mol, distances, lower, upper, bonded, tol.bounds_slack)
    _add_bounds_check(add, "bond_angles", mol, distances, lower, upper, angle, tol.bounds_slack)
    _add_lower_bound_check(add, mol, distances, lower, far, tol.clash_slack)


def _add_bounds_check(add, name, mol, distances, lower, upper, mask, slack) -> None:
    pairs = np.argwhere(np.triu(mask, 1))
    if pairs.size == 0:
        add(name, True, skipped=True, detail="no such pairs in this molecule")
        return
    i, j = pairs[:, 0], pairs[:, 1]
    observed = distances[i, j]
    low = lower[i, j] * (1.0 - slack)
    high = upper[i, j] * (1.0 + slack)
    violation = np.maximum(low - observed, observed - high)
    worst = int(np.argmax(violation))
    passed = bool(violation[worst] <= 0)
    add(
        name,
        passed,
        value=float(violation[worst]),
        detail=(
            ""
            if passed
            else (
                f"{int((violation > 0).sum())} of {len(observed)} outside bounds; worst "
                f"{_atom_label(mol, i[worst])}-{_atom_label(mol, j[worst])} "
                f"{observed[worst]:.2f} A, allowed [{low[worst]:.2f}, {high[worst]:.2f}]"
            )
        ),
    )


def _add_lower_bound_check(add, mol, distances, lower, mask, slack) -> None:
    pairs = np.argwhere(np.triu(mask, 1))
    if pairs.size == 0:
        add("internal_steric_clash", True, skipped=True,
            detail="molecule too small to have non-bonded pairs")
        return
    i, j = pairs[:, 0], pairs[:, 1]
    observed = distances[i, j]
    floor = lower[i, j] * (1.0 - slack)
    violation = floor - observed
    worst = int(np.argmax(violation))
    passed = bool(violation[worst] <= 0)
    add(
        "internal_steric_clash",
        passed,
        value=float(observed[worst]),
        detail=(
            ""
            if passed
            else (
                f"{int((violation > 0).sum())} clashing pair(s); worst "
                f"{_atom_label(mol, i[worst])}-{_atom_label(mol, j[worst])} "
                f"{observed[worst]:.2f} A, minimum {floor[worst]:.2f} A"
            )
        ),
    )


def _add_receptor_checks(add, mol, coords, receptor_xyz, receptor_radii, tol) -> None:
    heavy = np.array(
        [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1], dtype=int
    )
    if heavy.size == 0:
        add("protein_ligand_clash", True, skipped=True, detail="pose has no heavy atoms")
        add("inside_receptor_bbox", True, skipped=True, detail="pose has no heavy atoms")
        return
    ligand_xyz = coords[heavy]
    ligand_radii = np.array(
        [_vdw(mol.GetAtomWithIdx(int(k)).GetSymbol()) for k in heavy], dtype=float
    )

    deltas = ligand_xyz[:, None, :] - receptor_xyz[None, :, :]
    separation = np.sqrt((deltas**2).sum(axis=2))
    contact = tol.protein_vdw_scale * (ligand_radii[:, None] + receptor_radii[None, :])
    overlap = contact - separation
    flat = int(np.argmax(overlap))
    lig_i, rec_j = np.unravel_index(flat, overlap.shape)
    passed = bool(overlap[lig_i, rec_j] <= 0)
    add(
        "protein_ligand_clash",
        passed,
        value=float(separation[lig_i, rec_j]),
        detail=(
            ""
            if passed
            else (
                f"{int((overlap > 0).sum())} clashing atom pair(s); closest "
                f"{_atom_label(mol, int(heavy[lig_i]))} at {separation[lig_i, rec_j]:.2f} A, "
                f"minimum {contact[lig_i, rec_j]:.2f} A"
            )
        ),
    )

    low = receptor_xyz.min(axis=0) - tol.bbox_margin
    high = receptor_xyz.max(axis=0) + tol.bbox_margin
    outside = np.maximum(low - ligand_xyz, ligand_xyz - high)
    excursion = float(outside.max())
    inside = excursion <= 0
    add(
        "inside_receptor_bbox",
        bool(inside),
        value=excursion,
        detail=(
            ""
            if inside
            else f"{int((outside.max(axis=1) > 0).sum())} atom(s) up to {excursion:.2f} A "
                 f"outside the receptor box (margin {tol.bbox_margin:g} A)"
        ),
    )


def _add_identity_checks(add, mol, pose: Pose3D) -> None:
    """Did the pose stay the molecule it was supposed to be?

    The reference identity comes from the pose's own metadata, which is where every backend
    in this package puts it. Without one there is nothing to compare against and both checks
    are skipped rather than guessed: a pose read off disk with no provenance genuinely cannot
    answer "was the stereochemistry preserved?".
    """
    reference_smiles = pose.metadata.get("reference_smiles") or pose.metadata.get("smiles")
    if not reference_smiles:
        add("stereochemistry_preserved", True, skipped=True,
            detail="no reference_smiles/smiles in pose metadata")
        add("formal_charge_preserved", True, skipped=True,
            detail="no reference_smiles/smiles in pose metadata")
        return

    reference = Chem.MolFromSmiles(str(reference_smiles))
    if reference is None:
        add("stereochemistry_preserved", True, skipped=True,
            detail=f"unparseable reference SMILES {reference_smiles!r}")
        add("formal_charge_preserved", True, skipped=True,
            detail=f"unparseable reference SMILES {reference_smiles!r}")
        return

    probe = Chem.Mol(mol)
    try:
        Chem.AssignStereochemistryFrom3D(probe)
    except Exception as exc:
        add("stereochemistry_preserved", False, detail=f"stereo perception failed: {exc}")
    else:
        probe_flat = Chem.RemoveHs(probe)
        # A reference drawn without stereochemistry cannot be used to judge it: perceiving
        # stereo from 3D would then always "disagree" with a SMILES that never specified any.
        if _has_stereo(reference):
            observed = Chem.MolToSmiles(probe_flat)
            expected = Chem.MolToSmiles(reference)
            level = "isomeric"
        else:
            observed = Chem.MolToSmiles(probe_flat, isomericSmiles=False)
            expected = Chem.MolToSmiles(reference, isomericSmiles=False)
            level = "constitution only (reference carries no stereochemistry)"
        matched = observed == expected
        add(
            "stereochemistry_preserved",
            matched,
            detail="" if matched else f"{level}: pose {observed} != reference {expected}",
        )

    observed_charge = Chem.GetFormalCharge(mol)
    expected_charge = Chem.GetFormalCharge(reference)
    matched_charge = observed_charge == expected_charge
    add(
        "formal_charge_preserved",
        matched_charge,
        value=float(observed_charge),
        detail="" if matched_charge else f"pose {observed_charge:+d} != reference {expected_charge:+d}",
    )


def _has_stereo(mol: Any) -> bool:
    return Chem.MolToSmiles(mol) != Chem.MolToSmiles(mol, isomericSmiles=False)


def _bounds_matrix(mol: Any) -> tuple[np.ndarray | None, str]:
    from rdkit.Chem import rdDistGeom

    try:
        return np.asarray(rdDistGeom.GetMoleculeBoundsMatrix(mol), dtype=float), ""
    except Exception as exc:
        return None, f"bounds matrix unavailable: {type(exc).__name__}: {exc}"


def _pairwise(coords: np.ndarray) -> np.ndarray:
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((deltas**2).sum(axis=2))


def _atom_label(mol: Any, index: int) -> str:
    return f"{mol.GetAtomWithIdx(int(index)).GetSymbol()}{index}"


_DEFAULT_VDW = 1.7


def _vdw(symbol: str) -> float:
    try:
        return float(Chem.GetPeriodicTable().GetRvdw(symbol))
    except Exception:
        return _DEFAULT_VDW


_WATER_NAMES = {"HOH", "WAT", "TP3", "DOD"}


def _receptor_heavy_atoms(path: Path, tol: CheckTolerances) -> tuple[np.ndarray, np.ndarray]:
    """Heavy-atom coordinates and vdW radii from a PDB, by column, without RDKit.

    A plain text parse, not a Mol: the receptor may be several thousand atoms with
    non-standard residues and no CONECT records, and every RDKit route to it either
    sanitises (and fails) or discards the element column. Only positions and elements are
    needed here.

    Waters are dropped by default because :func:`atomfrust.pose.load_complex` drops them too,
    so a clash against one would flag a pose against a receptor that will never be scored.
    """
    xyz: list[tuple[float, float, float]] = []
    radii: list[float] = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if tol.skip_waters and line[17:20].strip().upper() in _WATER_NAMES:
            continue
        element = line[76:78].strip() or _element_from_name(line[12:16])
        if element.upper() in {"H", "D"}:
            continue
        try:
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
        radii.append(_vdw(element.capitalize()))
    return (
        np.asarray(xyz, dtype=float).reshape(-1, 3),
        np.asarray(radii, dtype=float),
    )


def _element_from_name(atom_name: str) -> str:
    """Fall back to the PDB atom-name convention when columns 77-78 are blank.

    Columns 13-14 hold the element right-justified for single-letter elements, which is why
    a leading space means "the element is the next character" and a leading digit (``1HB``)
    means hydrogen.
    """
    name = atom_name.rstrip()
    if not name:
        return ""
    if name[0].isdigit():
        return "H"
    if atom_name.startswith(" "):
        return name.strip()[:1]
    return name.strip()[:2]


# ------------------------------------------------------------------ frames and pass rate


def _row(
    index: int,
    pose: Pose3D,
    *,
    checker: str,
    check: str,
    passed: bool,
    skipped: bool = False,
    value: float | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "pose_index": index,
        "pose_id": pose.metadata.get("pose_id"),
        "source": pose.source,
        "checker": checker,
        "check": check,
        "passed": bool(passed),
        "skipped": bool(skipped),
        "value": value,
        "detail": detail,
    }


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=object) for name in QC_COLUMNS})


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_frame()
    frame = pd.DataFrame(rows, columns=list(QC_COLUMNS))
    frame["passed"] = frame["passed"].astype(bool)
    frame["skipped"] = frame["skipped"].astype(bool)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def pose_pass_rate(qc: pd.DataFrame) -> float:
    """The S4.2 fraction: poses for which *every check that ran* passed.

    Skipped checks are excluded from the verdict, not counted as passes — a pose whose
    protein-aware checks were skipped for want of a receptor is judged on the checks that
    actually happened, and a pose with no executed checks at all is not in the denominator,
    because it never entered the gate.

    Returns ``nan`` for an empty frame. Zero is a *result* (everything failed) and must not
    be produced by having measured nothing.
    """
    if qc.empty:
        return float("nan")
    ran = qc[~qc["skipped"].astype(bool)]
    if ran.empty:
        return float("nan")
    return float(ran.groupby("pose_index")["passed"].all().mean())


def passing_pose_indices(qc: pd.DataFrame) -> list[int]:
    """``pose_index`` values that passed every check that ran, in ascending order."""
    if qc.empty:
        return []
    ran = qc[~qc["skipped"].astype(bool)]
    if ran.empty:
        return []
    per_pose = ran.groupby("pose_index")["passed"].all()
    return sorted(int(i) for i, ok in per_pose.items() if bool(ok))


def filter_valid(poses: Sequence[Pose3D], qc: pd.DataFrame) -> list[Pose3D]:
    """The poses that may enter analysis. This is the gate, applied.

    Ordering is preserved so a "best pose" stays the best of what survived.
    """
    keep = set(passing_pose_indices(qc))
    return [pose for index, pose in enumerate(poses) if index in keep]


def meets_s4_2(qc: pd.DataFrame, minimum: float = S4_2_MIN_PASS_RATE) -> bool:
    """Whether this generator clears S4.2 on this sample.

    False is not a reason to drop the poses silently: the criterion says a generator below
    the threshold is excluded *or reported separately with the failure mode named*, and the
    failure mode is a ``groupby("check")`` over the same frame.
    """
    rate = pose_pass_rate(qc)
    return bool(rate == rate and rate >= minimum)  # nan-safe: nan != nan
