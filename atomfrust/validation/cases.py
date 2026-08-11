"""The case registry and the cases themselves (plan steps F1, F4, F5).

A case is a small object, not a script:

``name``
    What ``atomfrust validate --case NAME`` takes. The plan's identifiers (``F1``, ``F4``,
    ``F5``) are the canonical names; lookup is case-insensitive.
``__doc__``
    What the case *proves*. Read it before believing a PASS — several of these cases prove
    something narrower than their title suggests, and F5 in particular is explicit about
    which of its two routes is genuinely independent and which is not.
``expected`` / ``tolerance``
    The stored expectation. Without it a case is a print-out; with it, a change in any
    module the case touches turns into a FAIL with the old and new numbers side by side.
``run()``
    Returns a :class:`CaseResult`. It never raises for a missing input — that is a ``SKIP``
    — and never raises for a disagreeing number — that is a ``FAIL``.

**The expectations here were measured on 2026-08-11 from the artefacts on disk at that
date**, and every one of them is conditional on the ensemble that produced those artefacts.
Both F1 and F4 read per-contact indices from 50-decoy prototype runs; a larger decoy count
tightens the decoy σ, moves the whole F scale, and so moves every fraction and correlation
below. Re-measure and re-pin when the underlying runs are regenerated — do not widen the
tolerance to make a differently-generated ensemble fit.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypeVar

import numpy as np
import pandas as pd

__all__ = [
    "Status",
    "CaseResult",
    "ValidationCase",
    "CASES",
    "register",
    "get_case",
    "all_cases",
    "run_case",
    "LysozymeLayerCase",
    "ApoControlCase",
    "PairEnergyReferenceCase",
]

Status = Literal["PASS", "FAIL", "SKIP"]

#: Seed for every resampling estimate below. A validation case whose verdict moved between
#: runs of the same inputs would be useless as a regression test, so nothing here is allowed
#: to depend on an unseeded RNG.
BOOTSTRAP_SEED = 20260811

#: Bootstrap replicates for F1's confidence interval on the core−surface difference.
#: Part of the stored expectation: changing it changes the interval.
BOOTSTRAP_REPLICATES = 2000


# --------------------------------------------------------------------------- result


@dataclass(frozen=True)
class CaseResult:
    """One case's verdict, in the form the CLI prints and ``--json`` serialises.

    ``measured``, ``expected`` and ``tolerance`` are parallel mappings: a key present in
    ``expected`` is a key that was checked. Keys in ``measured`` that are absent from
    ``expected`` are diagnostics — reported, never asserted — which is how a case can carry
    context (group sizes, which split it used, how many structures it found) without that
    context becoming a brittle assertion.
    """

    name: str
    status: Status
    measured: Mapping[str, Any] | None = None
    expected: Mapping[str, Any] | None = None
    tolerance: Mapping[str, Any] | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "measured": _jsonable(self.measured),
            "expected": _jsonable(self.expected),
            "tolerance": _jsonable(self.tolerance),
            "detail": self.detail,
        }


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else str(value)
    return obj


# ----------------------------------------------------------------------------- case


@dataclass(frozen=True)
class ValidationCase:
    """Base class. Subclasses set the class attributes and implement :meth:`measure`.

    :meth:`run` is deliberately final-ish: it wraps :meth:`measure`, applies the stored
    expectation and turns an exception into a FAIL with a one-line message. A case that
    crashed is a case that failed — it must not take the whole command down with a
    traceback, because the point of the command is to report on several cases at once.
    """

    #: Canonical name, as typed after ``--case``.
    name: str = ""
    #: One line for ``--list``.
    summary: str = ""
    #: Checked keys and their stored values.
    expected: Mapping[str, Any] = field(default_factory=dict)
    #: Absolute tolerance per key. A key with no entry is compared for exact equality.
    tolerance: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ hooks

    def measure(self, root: Path, **options: Any) -> CaseResult:
        """Do the work. Return a SKIP/FAIL directly, or a PASS-shaped result to be checked.

        A result carrying ``measured`` and status ``PASS`` is passed through
        :meth:`check`; any other status is returned untouched.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------- api

    def run(self, root: Path | str = ".", **options: Any) -> CaseResult:
        root = Path(root)
        try:
            result = self.measure(root, **options)
        except Exception as exc:  # noqa: BLE001 - a crashed case is a failed case
            return self.fail(f"the case raised {type(exc).__name__}: {_one_line(exc)}")
        if result.status != "PASS":
            return result
        return self.check(result)

    def check(self, result: CaseResult) -> CaseResult:
        """Apply the stored expectation to a measured result."""
        problems = _deviations(result.measured or {}, self.expected, self.tolerance)
        if problems:
            return CaseResult(
                name=self.name,
                status="FAIL",
                measured=result.measured,
                expected=self.expected,
                tolerance=self.tolerance,
                detail=(result.detail + " " if result.detail else "")
                + "Departure from the stored expectation: "
                + "; ".join(problems),
            )
        return CaseResult(
            name=self.name,
            status="PASS",
            measured=result.measured,
            expected=self.expected,
            tolerance=self.tolerance,
            detail=result.detail,
        )

    # -------------------------------------------------------------- shorthands

    def skip(self, detail: str) -> CaseResult:
        return CaseResult(self.name, "SKIP", None, self.expected, self.tolerance, detail)

    def fail(self, detail: str, measured: Mapping[str, Any] | None = None) -> CaseResult:
        return CaseResult(self.name, "FAIL", measured, self.expected, self.tolerance, detail)

    def measured(self, measured: Mapping[str, Any], detail: str = "") -> CaseResult:
        """A measurement awaiting :meth:`check`."""
        return CaseResult(self.name, "PASS", measured, self.expected, self.tolerance, detail)

    # ------------------------------------------------------------- description

    @property
    def description(self) -> str:
        """The class docstring — what the case proves."""
        return (type(self).__doc__ or "").strip()


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())


def _deviations(
    measured: Mapping[str, Any],
    expected: Mapping[str, Any],
    tolerance: Mapping[str, Any],
) -> list[str]:
    """Every expected key that the measurement missed, as human-readable sentences."""
    problems: list[str] = []
    for key, want in expected.items():
        if key not in measured:
            problems.append(f"{key} was not measured")
            continue
        got = measured[key]
        if isinstance(want, bool):
            if bool(got) != want:
                problems.append(f"{key} = {bool(got)}, expected {want}")
            continue
        tol = tolerance.get(key)
        if tol is None:
            if got != want:
                problems.append(f"{key} = {got!r}, expected {want!r} exactly")
            continue
        value = float(got)
        if not math.isfinite(value):
            problems.append(f"{key} = {value}, expected {float(want):.6g} +/- {float(tol):.6g}")
        elif abs(value - float(want)) > float(tol):
            problems.append(
                f"{key} = {value:.6g}, expected {float(want):.6g} +/- {float(tol):.6g} "
                f"(off by {abs(value - float(want)):.3g})"
            )
    return problems


# ------------------------------------------------------------------------ registry

#: Name -> case. Module-level and mutable so a test can substitute a fake registry; every
#: lookup below reads it at call time rather than capturing it.
CASES: dict[str, ValidationCase] = {}


_Registrable = TypeVar("_Registrable", ValidationCase, type)


def register(case: _Registrable) -> _Registrable:
    """Add a case, as a class decorator or with an instance.

    A class is instantiated with its stored defaults and the *class* is handed back, so the
    decorated name still refers to the class and the registry holds the object. Refuses to
    overwrite: two cases under one name is a bug, not a configuration choice.
    """
    instance: ValidationCase = case() if isinstance(case, type) else case
    key = instance.name.upper()
    if not key:
        raise ValueError("a validation case needs a name")
    if key in CASES:
        raise ValueError(f"validation case {instance.name!r} is already registered")
    CASES[key] = instance
    return case


def get_case(name: str) -> ValidationCase:
    try:
        return CASES[name.strip().upper()]
    except KeyError:
        known = ", ".join(c.name for c in all_cases()) or "(none registered)"
        raise KeyError(f"unknown validation case {name!r}; known cases: {known}") from None


def all_cases() -> list[ValidationCase]:
    """Every registered case, in name order so output is stable."""
    return [CASES[k] for k in sorted(CASES)]


def run_case(name: str, root: Path | str = ".", **options: Any) -> CaseResult:
    return get_case(name).run(root, **options)


# --------------------------------------------------------------- shared utilities


def _pyrosetta_available() -> bool:
    return importlib.util.find_spec("pyrosetta") is not None


def _missing(paths: Mapping[str, Path]) -> list[str]:
    return [f"{label} ({path})" for label, path in paths.items() if not path.exists()]


def _bootstrap_difference(
    a: np.ndarray, b: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES
) -> tuple[float, float]:
    """Percentile 95% CI on ``mean(a) − mean(b)`` by independent resampling of each group.

    The two groups are disjoint sets of contacts, so they are resampled independently; a
    paired bootstrap would be wrong here because no contact appears in both. The interval is
    percentile rather than BCa deliberately: with ~200 and ~230 members the bias correction
    buys little, and a simpler estimator is a better regression target.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(replicates, dtype=float)
    for k in range(replicates):
        draws[k] = rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _protein_and_ligand_geometry(pdb_path: Path):
    """``(ca, heavy_per_residue, ligand_heavy)`` from a PDB, without PyRosetta.

    Ported from ``analysis/diagnose_counts.py`` (``load_structure``, lines 76-98) so F4 can
    run from stored parquets alone. Protein residues are taken in file order, which is the
    order Rosetta assigns pose numbers to, so index ``k`` here is pose residue ``k+1``. That
    identification is not assumed — F4 re-derives the contact set from these coordinates and
    refuses to compare anything if it does not reproduce the stored pairs exactly.
    """
    from Bio.PDB import PDBParser

    model = PDBParser(QUIET=True).get_structure("s", str(pdb_path))[0]
    residues = list(model.get_residues())
    protein = [r for r in residues if r.id[0] == " "]
    het = [r for r in residues if r.id[0] != " " and r.get_resname() != "HOH"]

    ca = np.full((len(protein), 3), np.nan)
    for k, residue in enumerate(protein):
        if "CA" in residue:
            ca[k] = residue["CA"].get_coord()
    heavy = [
        np.array([a.get_coord() for a in r if a.element != "H"], dtype=float) for r in protein
    ]
    ligand = np.array(
        [a.get_coord() for r in het for a in r if a.element != "H"], dtype=float
    ).reshape(-1, 3)
    return ca, heavy, ligand


def _ca_contacts(ca: np.ndarray, cutoff: float = 10.0, seq_sep_min: int = 4) -> set[tuple[int, int]]:
    """The prototype's contact rule, in 1-based pose numbering.

    ``analysis/diagnose_counts.py:101-109``; the rule itself is ``get_protein_contacts``
    (``src/frustration.py:70``): Cα–Cα within ``cutoff`` and pose-index separation at least
    ``seq_sep_min``.
    """
    from scipy.spatial.distance import cdist

    distance = cdist(ca, ca)
    n = len(ca)
    index = np.arange(n)
    separation = np.abs(index[:, None] - index[None, :])
    ok = (distance <= cutoff) & (separation >= seq_sep_min) & np.isfinite(distance)
    i, j = np.where(np.triu(ok, k=1))
    return set(zip((i + 1).tolist(), (j + 1).tolist()))


# ============================================================================== F1


@register
@dataclass(frozen=True)
class LysozymeLayerCase(ValidationCase):
    """F1 — buried lysozyme contacts are more often minimally frustrated than exposed ones.

    *Proves:* that the frustration index carries the qualitative signature every
    frustration method is expected to reproduce — a protein's core is minimally frustrated
    and its surface is not — on a protein with no ligand, no affinity label and no
    connection to the EGFR set. It is the only case here whose subject is the *physics*
    rather than the plumbing.

    *Does not prove:* anything quantitative about EGFR, and nothing at all about ligands.
    One protein, one decoy ensemble.

    **Inputs are read, never generated.** The per-contact index comes from
    ``checkpoints/1LYZ_frustration.parquet``, the prototype's 50-decoy validation run, and
    the geometry from ``data/raw_pdb/1LYZ.pdb``. **The stored expectation is therefore
    decoy-count dependent**: a re-run at a different ``n_decoys`` rescales σ and moves every
    fraction below. Nothing is downloaded; a missing file is a SKIP.

    **The surface group is not ``layer(surface)``.** ``atomfrust.regions`` classifies burial
    by side-chain neighbour count (``LAYER_NEIGHBOUR_CUTOFFS``, ``regions.py:52``), and on
    1LYZ those cutoffs put 42 residues in the core, 85 in the boundary and **2** on the
    surface — so ``layer(surface)`` yields zero contacts with both endpoints exposed and the
    comparison has no denominator. That is a finding about the layer cutoffs, not about
    frustration, so the case does not fail on it: it falls back to ``not layer(core)`` for
    the exposed group, records which split it used in ``split`` and reports the strict
    counts as diagnostics. The fallback is deterministic, so this is still a regression
    test; ``layer(surface)`` is used unchanged the moment it selects enough contacts.

    A contact belongs to a group when **both** endpoints do. Requiring both makes the two
    groups disjoint and the difference interpretable; an "either endpoint" rule would put
    core–surface contacts in both groups and blunt exactly the contrast being measured.
    """

    name: str = "F1"
    summary: str = "lysozyme core contacts are more minimally frustrated than exposed ones"
    expected: Mapping[str, Any] = field(
        default_factory=lambda: {
            # Measured 2026-08-11 from checkpoints/1LYZ_frustration.parquet (50 decoys).
            "core_fraction": 0.5550,
            "exposed_fraction": 0.2532,
            "difference": 0.3018,
            # The plan's acceptance criterion: a bootstrap CI on the difference that
            # excludes zero. Expressed as a boolean so it fails loudly rather than being
            # buried in an interval nobody reads.
            "ci_excludes_zero": True,
            "core_exceeds_exposed": True,
            "split": "layer(core) vs not layer(core)",
        }
    )
    tolerance: Mapping[str, Any] = field(
        default_factory=lambda: {
            "core_fraction": 0.01,
            "exposed_fraction": 0.01,
            "difference": 0.02,
        }
    )

    #: Below this a group is too small to compare; F1 falls back (surface) or skips (core).
    minimum_group: int = 30

    def measure(self, root: Path, **options: Any) -> CaseResult:
        pdb = root / "data" / "raw_pdb" / "1LYZ.pdb"
        index_path = root / "checkpoints" / "1LYZ_frustration.parquet"

        missing = _missing({"the 1LYZ structure": pdb, "the 1LYZ index": index_path})
        if missing:
            return self.skip(
                "missing " + ", ".join(missing) + ". Nothing is downloaded by a validation "
                "case; obtain them with `dvc pull` or regenerate the index with "
                "`python src/run_pipeline.py --mode validate --n_decoys 50`."
            )
        if not _pyrosetta_available():
            return self.skip(
                "PyRosetta is not installed, and the core/surface split needs the pose "
                "geometry that atomfrust.pose.load_complex builds"
            )

        from atomfrust.analyze.classify import MINIMALLY_FRUSTRATED, classify_index
        from atomfrust.pose import load_complex
        from atomfrust.regions import select
        from atomfrust.spec import SystemSpec

        spec = SystemSpec.from_pdb(pdb, autodetect=False, system_id="1LYZ")
        loaded = load_complex(spec)
        core = select("layer(core)", loaded.nodes, loaded.geometry)
        strict_surface = select("layer(surface)", loaded.nodes, loaded.geometry)
        not_core = select("not layer(core)", loaded.nodes, loaded.geometry)

        frame = pd.read_parquet(index_path)
        for column in ("resi", "resj", "F_index"):
            if column not in frame.columns:
                return self.fail(
                    f"{index_path} has no {column!r} column; columns present: "
                    f"{', '.join(map(str, frame.columns))}"
                )

        position = {node.pose_resnum: k for k, node in enumerate(loaded.nodes)}
        unknown = {int(r) for r in frame["resi"]} | {int(r) for r in frame["resj"]}
        unknown -= set(position)
        if unknown:
            return self.fail(
                f"{index_path} references pose residues {sorted(unknown)[:5]} that "
                f"{pdb.name} does not contain ({len(loaded.nodes)} nodes loaded). The index "
                "and the structure describe different molecules."
            )

        i = np.array([position[int(r)] for r in frame["resi"]])
        j = np.array([position[int(r)] for r in frame["resj"]])
        minimal = classify_index(frame["F_index"].to_numpy()) == MINIMALLY_FRUSTRATED

        core_pairs = core[i] & core[j]
        strict_pairs = strict_surface[i] & strict_surface[j]
        if int(strict_pairs.sum()) >= self.minimum_group:
            exposed_pairs, split = strict_pairs, "layer(core) vs layer(surface)"
        else:
            exposed_pairs, split = not_core[i] & not_core[j], "layer(core) vs not layer(core)"

        if int(core_pairs.sum()) < self.minimum_group or int(exposed_pairs.sum()) < self.minimum_group:
            return self.skip(
                f"too few contacts to compare: {int(core_pairs.sum())} core, "
                f"{int(exposed_pairs.sum())} exposed, minimum {self.minimum_group} each"
            )

        core_values = minimal[core_pairs]
        exposed_values = minimal[exposed_pairs]
        difference = float(core_values.mean() - exposed_values.mean())
        low, high = _bootstrap_difference(core_values, exposed_values)

        measured = {
            "core_fraction": float(core_values.mean()),
            "exposed_fraction": float(exposed_values.mean()),
            "difference": difference,
            "ci_low": low,
            "ci_high": high,
            "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
            "core_exceeds_exposed": bool(difference > 0.0),
            "split": split,
            "n_core_contacts": int(core_pairs.sum()),
            "n_exposed_contacts": int(exposed_pairs.sum()),
            "n_contacts": int(len(frame)),
            "n_core_residues": int(core.sum()),
            "n_strict_surface_residues": int(strict_surface.sum()),
            "n_strict_surface_contacts": int(strict_pairs.sum()),
        }
        detail = (
            f"{measured['n_core_contacts']} core contacts {measured['core_fraction']:.1%} "
            f"minimally frustrated vs {measured['n_exposed_contacts']} exposed at "
            f"{measured['exposed_fraction']:.1%}; difference {difference:+.4f}, "
            f"95% CI [{low:+.4f}, {high:+.4f}]; split = {split}. "
            f"layer(surface) selects {measured['n_strict_surface_residues']} of "
            f"{len(loaded.nodes)} residues here."
        )
        return self.measured(measured, detail)


# ============================================================================== F4


@register
@dataclass(frozen=True)
class ApoControlCase(ValidationCase):
    """F4 — deleting the ligand barely changes the per-contact index (it is ligand-blind).

    *Proves:* that under ``many_body.mode = chen_literal`` the published index is, on the
    pinned structure, very nearly a protein-only quantity. Holo and apo runs at the same
    seed give pocket-contact Pearson r = 0.99 and a **bit-identical** ``E_native`` — the
    native reference provably cannot see the ligand, because partner lists come from a
    protein-only contact rule (``src/frustration.py:70``) and the ligand is therefore never
    a member of any of them. The ligand's whole influence arrives through the decoys, via
    side chains that repack differently when the pocket is occupied.

    *Does not prove:* that this is desirable. It is the defect the plan's §2.1 describes,
    and the case exists so that the moment someone fixes it — by making the ligand a graph
    node (step A4/B7) or changing the many-body mode — the change is **detected** rather
    than assumed. A FAIL here is therefore ambiguous and the detail says so: it may mean a
    regression, or it may mean the index finally became ligand-aware. Read the numbers.

    *Ported from* ``analysis/apo_control.py`` (step A3), which produced these values and
    ``analysis/apo_control_report.md``. This case does not re-run the apo survey — it reads
    the two stored parquets, so it needs no PyRosetta and finishes in under a second. If the
    apo parquet is absent the case SKIPs and names the command that produces it. **The
    stored expectation is decoy-count dependent** (both runs are 50 decoys, seed 42).

    The pairing is verified, not assumed: the case re-derives the Cα contact set from the
    holo coordinates and refuses to compare anything unless it reproduces the stored pairs
    exactly, which is also what pins the parquet's ``resi``/``resj`` to file-order residue
    positions.
    """

    name: str = "F4"
    summary: str = "holo vs apo per-contact index on 5GMP — is the index ligand-blind?"
    expected: Mapping[str, Any] = field(
        default_factory=lambda: {
            # Measured 2026-08-11 from results/5GMP_F62_frustration.parquet and
            # analysis/apo/5GMP_apo_frustration.parquet (50 decoys, seed 42), matching
            # analysis/apo_control_report.md.
            "pocket_pearson_r": 0.9904,
            "all_pearson_r": 0.9965,
            "pocket_class_agreement": 0.9660,
            # The verdict itself, at A3's threshold. Stored as a boolean so a case that
            # merely drifts within tolerance still asserts the qualitative claim.
            "ligand_blind": True,
            # Structural, not statistical: E_ij's partner lists are protein-only, so the
            # native reference cannot depend on the ligand. Exact equality, no tolerance.
            "e_native_n_differing": 0,
            "e_native_max_abs_delta": 0.0,
        }
    )
    tolerance: Mapping[str, Any] = field(
        default_factory=lambda: {
            "pocket_pearson_r": 0.01,
            "all_pearson_r": 0.01,
            "pocket_class_agreement": 0.02,
        }
    )

    #: A3's verdict threshold: above this the descriptor is called ligand-blind.
    ligand_blind_r: float = 0.95
    #: Pocket shell: protein residues with a heavy atom within this of a ligand heavy atom
    #: (``heavy5`` in ``analysis/diagnose_counts.py``).
    pocket_cutoff_A: float = 5.0

    def measure(self, root: Path, **options: Any) -> CaseResult:
        from scipy.spatial.distance import cdist
        from scipy.stats import pearsonr, spearmanr

        holo_path = root / "results" / "5GMP_F62_frustration.parquet"
        apo_path = root / "analysis" / "apo" / "5GMP_apo_frustration.parquet"
        pdb = root / "data" / "processed" / "5GMP_clean.pdb"

        missing = _missing(
            {"the holo index": holo_path, "the 5GMP structure": pdb}
        )
        if missing:
            return self.skip(
                "missing " + ", ".join(missing) + "; `dvc pull` restores both"
            )
        if not apo_path.exists():
            return self.skip(
                f"missing the apo index ({apo_path}). Produce it with "
                "`OMP_NUM_THREADS=1 python analysis/apo_control.py --n-jobs 16`, which "
                "runs 50 decoys on a ligand-stripped copy of the same structure."
            )

        holo = pd.read_parquet(holo_path)
        apo = pd.read_parquet(apo_path)
        ca, heavy, ligand = _protein_and_ligand_geometry(pdb)
        if ligand.size == 0:
            return self.fail(f"{pdb} has no non-water HETATM, so there is no ligand to delete")

        stored = set(zip(holo["resi"].tolist(), holo["resj"].tolist()))
        recomputed = _ca_contacts(ca)
        if stored != recomputed:
            return self.fail(
                f"the contact set recomputed from {pdb.name} ({len(recomputed)} pairs) "
                f"differs from the one stored in {holo_path.name} ({len(stored)} pairs, "
                f"{len(stored & recomputed)} shared). The parquet no longer describes this "
                "structure, so pocket membership cannot be assigned to its rows."
            )

        merged = holo.merge(apo, on=["resi", "resj"], suffixes=("_holo", "_apo"))
        if len(merged) != len(holo):
            return self.fail(
                f"the holo and apo indices do not cover the same contacts "
                f"({len(merged)} joined of {len(holo)} holo and {len(apo)} apo rows); the "
                "comparison would not be paired"
            )

        distance = np.array(
            [cdist(h, ligand).min() if len(h) else np.inf for h in heavy]
        )
        pocket = set((np.flatnonzero(distance <= self.pocket_cutoff_A) + 1).tolist())
        in_pocket = np.array(
            [(int(i) in pocket) or (int(j) in pocket) for i, j in zip(merged["resi"], merged["resj"])]
        )

        pocket_rows = merged[in_pocket]
        if len(pocket_rows) < 3:
            return self.fail(
                f"only {len(pocket_rows)} pocket contacts within "
                f"{self.pocket_cutoff_A} A of the ligand; nothing to correlate"
            )

        pocket_r = float(pearsonr(pocket_rows["F_index_holo"], pocket_rows["F_index_apo"])[0])
        all_r = float(pearsonr(merged["F_index_holo"], merged["F_index_apo"])[0])
        pocket_rho = float(spearmanr(pocket_rows["F_index_holo"], pocket_rows["F_index_apo"])[0])
        delta = (pocket_rows["F_index_holo"] - pocket_rows["F_index_apo"]).abs()
        agreement = float(
            (pocket_rows["frustration_class_holo"] == pocket_rows["frustration_class_apo"]).mean()
        )

        measured: dict[str, Any] = {
            "pocket_pearson_r": pocket_r,
            "all_pearson_r": all_r,
            "pocket_spearman_rho": pocket_rho,
            "pocket_class_agreement": agreement,
            "ligand_blind": bool(pocket_r > self.ligand_blind_r),
            "pocket_median_abs_dF": float(delta.median()),
            "pocket_max_abs_dF": float(delta.max()),
            "n_contacts": int(len(merged)),
            "n_pocket_contacts": int(len(pocket_rows)),
            "n_pocket_residues": int(len(pocket)),
            "n_ligand_heavy_atoms": int(len(ligand)),
        }
        for column in ("E_native", "decoy_mean", "decoy_std"):
            difference = (merged[f"{column}_holo"] - merged[f"{column}_apo"]).abs()
            measured[f"{column.lower()}_max_abs_delta"] = float(difference.max())
            measured[f"{column.lower()}_n_differing"] = int((difference > 1e-12).sum())

        detail = (
            f"pocket-contact Pearson r = {pocket_r:.4f} over {len(pocket_rows)} contacts "
            f"({measured['n_pocket_residues']} pocket residues within "
            f"{self.pocket_cutoff_A:.0f} A), class agreement {agreement:.1%}; E_native "
            f"max|delta| = {measured['e_native_max_abs_delta']:.3e} over {len(merged)} "
            f"contacts. A FAIL here is not automatically a regression: r falling below "
            f"{self.ligand_blind_r} would mean the index has become ligand-aware, which is "
            "the plan's intent for step B7 — check which direction it moved."
        )
        return self.measured(measured, detail)


# ============================================================================== F5


@register
@dataclass(frozen=True)
class PairEnergyReferenceCase(ValidationCase):
    """F5 — the stored per-pair energies match independent recomputations to within 1%.

    Two routes, and they are **not equally independent**. The distinction is the whole
    value of this case, so it is stated in the result as well as here.

    **Route A — residue-pair API** (``max_rel_dev_pair_api``). ``EnergyEvaluator``
    (``atomfrust/energy.py:93``) reads the cached REF2015 ``EnergyGraph`` edge and dots it
    with the score-function weights. Route A instead calls
    ``ScoreFunction.eval_ci_2b`` + ``eval_cd_2b`` on the same residue pair, which
    re-invokes the short-range two-body energy methods and fills a fresh ``EMapVector``.
    This is the second route the plan asks for, and it is a real check of the *storage and
    lookup* layer — edge lookup by residue number, the weight dot product, the ``fa_rep``
    extraction, symmetry in i↔j, and the claim that a pair with no edge is exactly 0.0.
    It is **not** an independent check of REF2015: it re-enters the same energy methods, so
    agreement is expected to be exact up to the float32 rounding of the stored column, and
    a bug inside an energy method would be invisible to it. Presenting it alone as
    "validated against an independent reference" would be a tautology.

    **Route B — atom-pair etable** (``max_rel_dev_etable_atomic``). This one is genuine:
    ``fa_atr``, ``fa_rep`` and ``fa_sol`` are recomputed from scratch by looping over every
    atom pair of the two residues through ``AnalyticEtableEvaluator.atom_pair_energy`` and
    summing, then compared against the corresponding components of the graph edge. Rosetta
    reaches the same number through a residue-level trie with neighbour-list pruning and
    count-pair logic; this reaches it by brute force over N×M atoms. Agreement is evidence.
    Restrictions, both honest limits rather than conveniences: only **chemically unbonded**
    residue pairs are used (a bonded pair needs count-pair weights, which this route does
    not model), and only the three etable terms are covered — they carry 78-98% of
    ``|e_direct|`` on the structures measured, with the remainder (``fa_elec``, the hbond
    terms, ``lk_ball_wtd``, ``fa_dun`` and friends) checked only by route A.

    **The edgeless branch is tested on purpose.** Every pair the superset proposes turns out
    to have an energy edge, so the plan's "pairs without an edge asserted to be exactly 0.0"
    would be vacuous if it were checked only over the graph. The case therefore also takes
    the most distant Cα pairs in each pose — far outside any interaction cutoff — and
    requires both ``EnergyEvaluator.pair`` and route A to return exactly 0.0 there. A run
    that finds no edgeless pair at all FAILs rather than passing on an empty assertion.

    *Proves:* that the number stored in ``e_direct`` is the weighted REF2015 short-range
    two-body energy of that residue pair, that the dominant part of it can be rebuilt atom
    by atom, and that an absent edge means zero rather than a missing value.

    *Does not prove:* that REF2015 is the right energy function, nor anything about the
    many-body formula built on top of these pairs (that is F3's and A1's territory).

    Needs no decoys and no stored run — it scores native poses only, so it is the one case
    here whose expectation is not decoy-count dependent.
    """

    name: str = "F5"
    summary: str = "stored pair energies vs two recomputations (one genuinely independent)"
    expected: Mapping[str, Any] = field(
        default_factory=lambda: {
            # Both are maximum *relative* deviations, so the expectation is zero and the
            # tolerance is the plan's 1%. Route A lands at ~6e-8 (float32 storage of
            # e_direct) and route B at ~1e-15 (float64 throughout), measured 2026-08-11.
            "max_rel_dev_pair_api": 0.0,
            "max_rel_dev_fa_rep": 0.0,
            "max_rel_dev_etable_atomic": 0.0,
            # A pair beyond Rosetta's interaction cutoff has no edge; both routes must call
            # that exactly 0.0, not merely small.
            "max_abs_edgeless_energy": 0.0,
            # Presence flag, not a claim about the numbers: it goes missing the moment the
            # atom-pair route is removed, which turns "validated against an independent
            # reference" back into the tautology this case exists to avoid.
            "has_independent_etable_route": True,
        }
    )
    tolerance: Mapping[str, Any] = field(
        default_factory=lambda: {
            "max_rel_dev_pair_api": 0.01,
            "max_rel_dev_fa_rep": 0.01,
            "max_rel_dev_etable_atomic": 0.01,
            "max_abs_edgeless_energy": 0.0,
        }
    )

    #: Complexes to cover. The plan asks for >= 20; a test passes a smaller number.
    n_structures: int = 20
    #: Residue pairs per structure sent through the O(N x M) atom-pair route.
    atomic_pairs_per_structure: int = 300
    #: Far-apart residue pairs per structure used to exercise the "no edge => exactly 0.0"
    #: branch, which the superset itself never reaches.
    edgeless_pairs: int = 200
    #: Energies below this magnitude use it as the relative-deviation denominator, so a
    #: 1e-18 reference cannot manufacture a 100% deviation out of float noise.
    relative_floor: float = 1e-6

    def measure(self, root: Path, **options: Any) -> CaseResult:
        n_structures = int(options.get("n_structures", self.n_structures))
        if not _pyrosetta_available():
            return self.skip("PyRosetta is not installed; every route here scores a pose")

        specs = _discover_specs(root, n_structures)
        if not specs:
            return self.skip(
                f"no scorable structure under {root}: expected "
                "data/processed/*_clean.pdb with matching data/ligands/params/*.params, "
                "or data/raw_pdb/1LYZ.pdb. `dvc pull` restores them."
            )

        from pyrosetta.rosetta.core.scoring import (
            EMapVector,
            ScoringManager,
            etable,
            fa_atr,
            fa_rep,
            fa_sol,
        )

        from atomfrust.energy import EnergyEvaluator
        from atomfrust.graph import build_graph
        from atomfrust.pose import load_complex
        from atomfrust.settings import Settings

        settings = Settings()
        worst_api = worst_fa_rep = worst_atomic = worst_edgeless = 0.0
        n_pairs = n_edges = n_atomic = n_bonded_skipped = 0
        n_edgeless = n_distant_with_edge = 0
        etable_share_numerator = etable_share_denominator = 0.0
        covered: list[str] = []

        for spec in specs:
            loaded = load_complex(spec)
            _, pairs = build_graph(loaded.nodes, loaded.geometry, settings)
            evaluator = EnergyEvaluator(loaded.pose, settings.energy.score_function)
            energies = evaluator.pairs(pairs)

            pose = loaded.pose
            scorefxn = evaluator.scorefxn
            weights = scorefxn.weights()
            fa_rep_weight = scorefxn.get_weight(fa_rep)
            graph = pose.energies().energy_graph()

            # ---- route A: residue-pair API vs the stored EnergyGraph energies -------
            for i, j, e_direct, e_fa_rep, has_edge in zip(
                pairs["i"], pairs["j"], energies["e_direct"], energies["e_fa_rep"],
                energies["has_edge"],
            ):
                emap = EMapVector()
                scorefxn.eval_ci_2b(pose.residue(int(i)), pose.residue(int(j)), pose, emap)
                scorefxn.eval_cd_2b(pose.residue(int(i)), pose.residue(int(j)), pose, emap)
                reference = float(emap.dot(weights))
                n_pairs += 1
                if not has_edge:
                    n_edgeless += 1
                    worst_edgeless = max(worst_edgeless, abs(reference), abs(float(e_direct)))
                    continue
                n_edges += 1
                worst_api = max(worst_api, self._relative(float(e_direct), reference))
                worst_fa_rep = max(
                    worst_fa_rep,
                    self._relative(float(e_fa_rep), float(emap.get(fa_rep)) * fa_rep_weight),
                )

            # ---- the edgeless contract, on pairs the superset never proposes --------
            # Every pair in the superset turns out to have an edge, so checking "no edge
            # implies exactly 0.0" only over the superset would be vacuous. These pairs are
            # chosen for being far apart precisely so the branch is exercised.
            for i, j in _distant_pairs(loaded.nodes, loaded.geometry, self.edgeless_pairs):
                pair_energy = evaluator.pair(i, j)
                if pair_energy.has_edge:
                    n_distant_with_edge += 1
                    continue
                emap = EMapVector()
                scorefxn.eval_ci_2b(pose.residue(i), pose.residue(j), pose, emap)
                scorefxn.eval_cd_2b(pose.residue(i), pose.residue(j), pose, emap)
                n_edgeless += 1
                worst_edgeless = max(
                    worst_edgeless, abs(float(emap.dot(weights))), abs(pair_energy.e_direct)
                )

            # ---- route B: atom-pair etable vs the graph's etable components ---------
            options_etable = scorefxn.energy_method_options().etable_options()
            evaluator_etable = etable.AnalyticEtableEvaluator(
                ScoringManager.get_instance().etable(options_etable).lock()
            )
            unit_weights = EMapVector()
            for term in (fa_atr, fa_rep, fa_sol):
                unit_weights.set(term, 1.0)
            evaluator_etable.set_weights(unit_weights)

            for k in _sample_indices(
                np.flatnonzero(energies["has_edge"].to_numpy()),
                self.atomic_pairs_per_structure,
            ):
                i = int(pairs["i"].iloc[k])
                j = int(pairs["j"].iloc[k])
                residue_i, residue_j = pose.residue(i), pose.residue(j)
                if residue_i.is_bonded(j):
                    n_bonded_skipped += 1
                    continue
                emap = EMapVector()
                for a in range(1, residue_i.natoms() + 1):
                    atom = residue_i.atom(a)
                    for b in range(1, residue_j.natoms() + 1):
                        evaluator_etable.atom_pair_energy(atom, residue_j.atom(b), 1.0, emap, 0.0)
                edge = graph.find_energy_edge(i, j)
                n_atomic += 1
                for term in (fa_atr, fa_rep, fa_sol):
                    worst_atomic = max(
                        worst_atomic, self._relative(float(edge[term]), float(emap.get(term)))
                    )
                etable_share_denominator += abs(float(edge.dot(weights)))
                etable_share_numerator += abs(
                    sum(float(edge[t]) * scorefxn.get_weight(t) for t in (fa_atr, fa_rep, fa_sol))
                )

            covered.append(spec.system_id)

        if n_edges == 0 or n_atomic == 0:
            return self.fail(
                f"no comparable pair found across {len(covered)} structure(s): "
                f"{n_edges} pairs with an edge, {n_atomic} atom-pair comparisons"
            )

        if n_edgeless == 0:
            return self.fail(
                "no edgeless pair was found on any structure, so the 'a pair beyond "
                "Rosetta's interaction cutoff is exactly 0.0' branch of "
                "EnergyEvaluator.pair was never exercised and asserting it would be vacuous"
            )

        measured = {
            "max_rel_dev_pair_api": worst_api,
            "max_rel_dev_fa_rep": worst_fa_rep,
            "max_rel_dev_etable_atomic": worst_atomic,
            "max_abs_edgeless_energy": worst_edgeless,
            "has_independent_etable_route": True,
            "etable_share_of_pair_energy": (
                etable_share_numerator / etable_share_denominator
                if etable_share_denominator > 0
                else float("nan")
            ),
            "n_structures": len(covered),
            "n_pairs": n_pairs,
            "n_pairs_with_edge": n_edges,
            "n_edgeless_pairs_checked": n_edgeless,
            "n_distant_pairs_with_edge": n_distant_with_edge,
            "n_atom_pair_comparisons": n_atomic,
            "n_bonded_pairs_skipped": n_bonded_skipped,
            "structures": ",".join(covered),
        }
        detail = (
            f"{len(covered)} structure(s), {n_edges} of {n_pairs} pairs with an edge. "
            f"Route A (residue-pair API, NOT independent of REF2015 — same energy methods, "
            f"different call path): max relative deviation {worst_api:.3e}. "
            f"Route B (atom-pair etable, genuinely independent recomputation of fa_atr, "
            f"fa_rep, fa_sol over {n_atomic} unbonded pairs, "
            f"{measured['etable_share_of_pair_energy']:.0%} of |e_direct|): "
            f"{worst_atomic:.3e}. {n_edgeless} pair(s) with no edge give exactly "
            f"{worst_edgeless:.3e} on both routes."
        )
        return self.measured(measured, detail)

    def _relative(self, stored: float, reference: float) -> float:
        scale = max(abs(reference), abs(stored), self.relative_floor)
        return abs(stored - reference) / scale


def _distant_pairs(nodes: Sequence[Any], geom: Any, limit: int) -> list[tuple[int, int]]:
    """Pose-residue pairs whose Cα are far enough apart that no energy edge can exist.

    Chosen by taking the largest Cα–Cα separations rather than at random, so the set is
    deterministic and every member is unambiguously outside Rosetta's interaction cutoff.
    A pair that nonetheless carries an edge is counted, not asserted away — see the caller.
    """
    ca = np.asarray(geom.ca_xyz, dtype=float)
    finite = np.flatnonzero(np.isfinite(ca).all(axis=1))
    if finite.size < 2 or limit <= 0:
        return []
    # A full distance matrix is fine here: these poses are a few hundred residues.
    coords = ca[finite]
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    iu = np.triu_indices(len(coords), k=1)
    order = np.argsort(distance[iu])[::-1][:limit]
    return [
        (nodes[finite[iu[0][k]]].pose_resnum, nodes[finite[iu[1][k]]].pose_resnum)
        for k in order
    ]


def _sample_indices(candidates: np.ndarray, limit: int) -> np.ndarray:
    """A seeded subsample, or everything when there is little enough to take."""
    if candidates.size <= limit:
        return candidates
    return np.random.default_rng(BOOTSTRAP_SEED).choice(candidates, limit, replace=False)


def _discover_specs(root: Path, limit: int) -> list[Any]:
    """Scorable systems on disk, at most one per distinct ligand.

    Deduplicating by ligand code is not cosmetic: two specs naming the same ``.params``
    would register the same residue type twice in one process and Rosetta raises
    ``residue type 'X' already exists in the cache``. It also spends the structure budget on
    chemical diversity rather than on near-duplicates of the same complex.

    Protein-only 1LYZ goes first when present, so a run with ``n_structures=1`` still
    covers the no-ligand path.
    """
    from atomfrust.spec import LigandSpec, SystemSpec

    specs: list[Any] = []
    lysozyme = root / "data" / "raw_pdb" / "1LYZ.pdb"
    if lysozyme.exists():
        specs.append(SystemSpec.from_pdb(lysozyme, autodetect=False, system_id="1LYZ"))

    seen: set[str] = set()
    params_dir = root / "data" / "ligands" / "params"
    for pdb in sorted((root / "data" / "processed").glob("*_clean.pdb")):
        if len(specs) >= limit:
            break
        try:
            spec = SystemSpec.from_pdb(pdb, system_id=pdb.stem)
        except Exception:  # noqa: BLE001 - an unreadable file is simply not a candidate
            continue
        ligands = []
        codes = set()
        for ligand in spec.ligands:
            params = params_dir / f"{ligand.selector.comp_id}.params"
            if not params.exists():
                ligands = []
                break
            ligands.append(LigandSpec(selector=ligand.selector, params=params))
            codes.add(str(ligand.selector.comp_id))
        if not ligands or codes & seen:
            continue
        seen |= codes
        specs.append(spec.model_copy(update={"ligands": tuple(ligands)}))
    return specs[:limit]
