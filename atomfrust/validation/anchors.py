"""Validation anchors F2, F3 and F6 — the three numerical gates that are not F1/F4/F5.

Each case is an executable function returning a :class:`CaseResult` with a measured value, a
stored expectation, a tolerance and a prose detail. Nothing here asserts; the caller decides
what a FAIL means.

**F3 (`reference-counts`) is the decisive one.** ``config/pdb_reference_table.csv`` carries
Chen et al.'s own per-structure minimally-frustrated counts (4-23, mean 12.7). Step A4
established what those counts are — *ligand-residue* contacts, the ligand being a node in
the published contact graph — whereas the prototype counted protein-protein pairs lying in a
10 A shell around the ligand (266-407, uncorrelated with the paper at r = 0.163, p = 0.51).
See ``project_status/2026-08-11_0130_a4-paper-check-ligand-is-a-node.md``. This case counts
**ligand-incident** contacts on the new graph: pairs where exactly one endpoint is
non-protein. That is the object the paper reports, and its natural scale — the number of
residues a drug-sized ligand touches — is the paper's 4-23.

**Scale, not certification.** A full reproduction is 61 structures at the paper's 1000
decoys. At the measured per-decoy cost that is thousands of core-hours, so the case is
parameterised by structure list and decoy count and defaults to a **smoke-scale** 3
structures x 10 decoys. The result detail says so, and records the extrapolated cost of the
full run. A Pearson r over three points is reported because the plan asks for it, not
because three points support an inference.

**F2 (`interface-fraction`) needs data that is not in this repository.** Every structure
under ``data/`` is single-chain: ``data/processed/*_clean.pdb`` is one EGFR chain plus one
ligand copy by construction (Stage 4), and ``data/raw_pdb/`` holds only 1LYZ. The case
therefore SKIPs and names the three complexes it is pinned to. They are pinned *here*,
before any of them has been run, which is what F2 asks for; picking them after seeing
fractions would make the +/-3 pp tolerance meaningless.

**F6 (`pocket-repack-equivalence`) is only meaningful because of step C4.**
``IdentityDecoyGenerator._position_rng`` gives every (decoy, position) its own substream, so
the identity drawn at a position does not depend on how many residues precede it in the
mutate set — that is what lets a whole-protein repack and a shell-restricted repack be
compared decoy-by-decoy rather than as two unpaired ensembles. The case honours that
contract: ``identity="composition"``, ``placement="inplace"``, ``seeding="substream"``.
``placement="permute"`` cannot be paired this way at all, because a permutation is a
property of the whole position set (see ``_draw_one``'s docstring). A restricted repack also
requires ``mutation="packer_task"``: ``IdentityDecoyGenerator.__post_init__`` raises for
``mutation="sequential"`` whenever some protein residue is excluded from repacking, since
the prototype had no per-residue packer operation at all.

Per plan step F6, **a Spearman rho below 0.95 is a finding about the shell approximation,
not a test failure** — it is reported, never tuned away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from atomfrust.graph import PROTEIN_KINDS

__all__ = [
    "CaseResult",
    "InterfaceCase",
    "PINNED_INTERFACES",
    "PINNED_REFERENCE_STRUCTURES",
    "PUBLISHED_INTERFACE_FRACTION",
    "PUBLISHED_INTERFACE_TOLERANCE_PP",
    "PAPER_COUNT_RANGE",
    "FULL_SET_STRUCTURES",
    "FULL_SET_DECOYS",
    "CASES",
    "case_interface_fraction",
    "case_reference_counts",
    "case_pocket_repack_equivalence",
    "ligand_incident_mask",
    "interface_mask",
    "paper_reference_counts",
    "bootstrap_fraction_ci",
    "find_interface_structure",
    "register_all",
]

# --------------------------------------------------------------------------- constants

#: Repo root, so a case can be called from anywhere rather than only from the repo root the
#: prototype's ``config.yaml`` paths assumed.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PARAMS_DIR = REPO_ROOT / "data" / "ligands" / "params"
RAW_PDB_DIR = REPO_ROOT / "data" / "raw_pdb"
INTERFACE_DIR = REPO_ROOT / "data" / "interfaces"
REFERENCE_TABLE = REPO_ROOT / "config" / "pdb_reference_table.csv"

#: Chen et al. 2020, Suppl. Fig. 8, via methods-doc claim #8 / S0.2: the atomistic method
#: calls 14.2% of protein-protein interface contacts minimally frustrated (AWSEM: 26.0%).
PUBLISHED_INTERFACE_FRACTION = 0.142
#: S0.2's gate, in percentage points.
PUBLISHED_INTERFACE_TOLERANCE_PP = 3.0

#: The published per-structure minimally-frustrated counts span this range (mean 12.7).
PAPER_COUNT_RANGE = (4, 23)

#: What a full F3 reproduction would be, for the cost extrapolation only.
FULL_SET_STRUCTURES = 61
FULL_SET_DECOYS = 1000


@dataclass(frozen=True)
class InterfaceCase:
    """One pinned protein-protein interface: a PDB id, its two sides, and why it was chosen."""

    pdb_id: str
    chains: tuple[str, str]
    reason: str


#: **Pinned before any of them was run.** All three are two-chain, high-resolution,
#: single-domain complexes from the standard protein-protein docking benchmark with no metal
#: ion, cofactor or nucleic acid that would need parametrisation before a pose will load —
#: which matters because ``atomfrust.pose.DEFAULT_INIT_FLAGS`` sets
#: ``-in:file:load_PDB_components false``, so an unparametrised component fails at load
#: rather than being silently typed from the bundled CCD.
#:
#: 1BRS is the plan's own ``chain_interface`` worked example (§5,
#: ``atomfrust run --spec specs/1BRS.yaml``); the other two add an RNase and a
#: beta-lactamase so the set is not three variations on one fold, and each pairs an enzyme
#: with a proteinaceous inhibitor, which is the interface class Chen et al.'s Suppl. Fig. 8
#: is drawn from.
PINNED_INTERFACES: tuple[InterfaceCase, ...] = (
    InterfaceCase(
        "1BRS", ("A", "D"),
        "barnase-barstar; the plan's own chain_interface worked example, 2.0 A, no cofactors",
    ),
    InterfaceCase(
        "1AY7", ("A", "B"),
        "RNase Sa - barstar, 1.7 A, two small single-domain chains, no metals",
    ),
    InterfaceCase(
        "1JTG", ("A", "B"),
        "TEM-1 beta-lactamase - BLIP, a class A beta-lactamase so no catalytic Zn",
    ),
)

#: **Pinned for F3.** The two endpoints A4 cross-checked against the paper's prose — 5GMP
#: (paper count 16, "more than ten minimally frustrating interactions", 0.8 pM) and 5EM8
#: (paper count 4, "only three", 1090 pM) — plus 1XKK, the paper's largest count (21) and
#: the structure the prototype was developed against. Together they span 4-21 of the
#: published 4-23, so the range-overlap criterion is actually informative on three points.
PINNED_REFERENCE_STRUCTURES: tuple[str, ...] = ("5EM8", "5GMP", "1XKK")


# ------------------------------------------------------------------------ case result


@dataclass(frozen=True)
class CaseResult:
    """One validation case's verdict.

    ``measured`` and ``expected`` are free-form (a float, a dict of per-structure numbers,
    a list of per-radius rows) because the six Stage F cases do not measure commensurable
    things; the *shape* is what is fixed. ``status`` is one of ``PASS`` / ``FAIL`` / ``SKIP``.
    """

    name: str
    status: str
    measured: Any = None
    expected: Any = None
    tolerance: Any = None
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ("PASS", "FAIL", "SKIP"):
            raise ValueError(f"status must be PASS/FAIL/SKIP, got {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "measured": self.measured,
            "expected": self.expected,
            "tolerance": self.tolerance,
            "detail": self.detail,
            "extra": self.extra,
        }


# ------------------------------------------------------------------------ pure helpers


def ligand_incident_mask(pairs: pd.DataFrame) -> np.ndarray:
    """Pairs with **exactly one** non-protein endpoint — the paper's ligand-residue contacts.

    Deliberately stricter than ``aggregate.pocket_mask(..., 'incident_to_ligand')``, which is
    an *at least one* rule and so also admits ligand-ligand and ligand-metal pairs. Those are
    not residue contacts and have no counterpart in the published counts.
    """
    kind_i = pairs["kind_i"].astype(str).to_numpy()
    kind_j = pairs["kind_j"].astype(str).to_numpy()
    protein = list(PROTEIN_KINDS)
    return np.isin(kind_i, protein) ^ np.isin(kind_j, protein)


def interface_mask(pairs: pd.DataFrame) -> np.ndarray:
    """Protein-protein pairs whose endpoints sit in different chains.

    ``same_chain`` is the ``inter_chain`` selector of :func:`aggregate.pocket_mask`; the
    protein-protein restriction is added here so a ligand parked in a second chain cannot
    enter an interface count.
    """
    kind_i = pairs["kind_i"].astype(str).to_numpy()
    kind_j = pairs["kind_j"].astype(str).to_numpy()
    protein = list(PROTEIN_KINDS)
    both_protein = np.isin(kind_i, protein) & np.isin(kind_j, protein)
    return both_protein & ~pairs["same_chain"].to_numpy(dtype=bool)


def paper_reference_counts(path: Path | str | None = None) -> pd.DataFrame:
    """Chen et al.'s per-structure counts, keyed by upper-case PDB id.

    Columns: ``paper_minimally_frustrated_contacts``, ``paper_highly_frustrated_contacts``,
    ``affinity_pM``. The file stores ids lower-case; everything else in this project uses
    upper-case, so the index is normalised here rather than at three call sites.
    """
    table = pd.read_csv(Path(path) if path is not None else REFERENCE_TABLE)
    required = {
        "pdb_id",
        "paper_minimally_frustrated_contacts",
        "paper_highly_frustrated_contacts",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{path or REFERENCE_TABLE} is missing column(s) {sorted(missing)}; "
            f"present: {list(table.columns)}"
        )
    table = table.copy()
    table["pdb_id"] = table["pdb_id"].astype(str).str.upper()
    return table.set_index("pdb_id")


def bootstrap_fraction_ci(
    successes: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a boolean vector.

    Contacts are resampled, not structures: within one structure the contacts are the
    sampling unit for "what fraction of interface contacts is minimally frustrated". A CI
    across *structures* is a different quantity and is reported separately by the case.
    """
    values = np.asarray(successes, dtype=float).ravel()
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def find_interface_structure(
    pdb_id: str, search: Sequence[Path] | None = None
) -> Path | None:
    """Locate a multi-chain coordinate file for ``pdb_id``. Never downloads anything.

    Looked for, in order: ``data/interfaces/<ID>.pdb`` (where a curated interface set would
    go), ``data/raw_pdb/<ID>.pdb`` (Stage 2's output directory), ``data/processed/``.
    Case-insensitive on the stem, since Stage 2 writes upper-case and hand-fetched files
    are usually lower-case.
    """
    directories = list(search) if search is not None else [
        INTERFACE_DIR, RAW_PDB_DIR, PROCESSED_DIR
    ]
    stems = {pdb_id.upper(), pdb_id.lower()}
    for directory in directories:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.iterdir()):
            if candidate.suffix.lower() not in (".pdb", ".ent", ".cif"):
                continue
            stem = candidate.stem.split("_")[0]
            if stem in stems:
                return candidate
    return None


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _pearson(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return (float("nan"), float("nan"))
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def write_protein_only_pdb(source: Path, chains: Iterable[str], destination: Path) -> Path:
    """Copy ATOM records of the named chains, dropping every heteroatom.

    Needed because ``load_complex`` hands the *whole* file to Rosetta and
    ``-in:file:load_PDB_components false`` makes an unparametrised HETATM a load-time error.
    A protein-only interface case has no components to parametrise, so the cheapest correct
    preparation is to remove them. Alternate locations other than blank/``A`` are dropped —
    Rosetta keeps the first conformer anyway, and leaving both would double-count atoms in
    the heavy-atom geometry the graph is built from.
    """
    wanted = {c.strip() for c in chains}
    kept: list[str] = []
    seen: set[str] = set()
    for line in source.read_text(errors="ignore").splitlines(keepends=True):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  "):
            continue
        chain = line[21].strip()
        if chain not in wanted:
            continue
        if line[16] not in (" ", "A"):
            continue
        seen.add(chain)
        kept.append(line)
    # Every requested chain must be present: one side of an interface is not an interface,
    # and letting a partial file through would surface much later as a spec/coordinate
    # mismatch inside `load_complex` rather than here, where the cause is obvious.
    if missing := sorted(wanted - seen):
        raise ValueError(
            f"{source.name} has no ATOM records for chain(s) {missing}; "
            "the pinned chain ids do not match this file"
        )
    destination.write_text("".join(kept) + "TER\nEND\n")
    return destination


# ------------------------------------------------------------------------- the engine


@dataclass(frozen=True)
class FrustrationRun:
    """One system carried from coordinates to per-contact classification, in this process.

    The composition mirrors ``atomfrust.cli.analyze._compute`` exactly — contact selection,
    ``effective_energy``, ``many_body_energies`` per member, index, classify — so a case and
    the CLI cannot drift into computing two different objects. It is done in-process rather
    than through a run directory because a validation case must be runnable without one.
    """

    system_id: str
    nodes: pd.DataFrame
    pairs: pd.DataFrame
    F: np.ndarray
    labels: np.ndarray
    n_decoys: int
    decoy_wall_s: float
    total_wall_s: float
    region_counts: dict[str, int]
    #: ``node_id`` of every node the ``mutate`` selector resolved to, so a case can restrict
    #: a comparison to the contacts the randomised region actually touches.
    mutated_node_ids: tuple[str, ...] = ()

    def incident_to_mutated(self) -> np.ndarray:
        """Pairs with at least one endpoint in the mutate region."""
        wanted = set(self.mutated_node_ids)
        in_i = self.pairs["node_i"].astype(str).isin(wanted).to_numpy()
        in_j = self.pairs["node_j"].astype(str).isin(wanted).to_numpy()
        return in_i | in_j


def _node_codes(pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Dense integer endpoint codes, as ``cli.analyze._node_codes`` does."""
    codes, _ = pd.factorize(
        pd.concat(
            [pairs["node_i"].astype(str), pairs["node_j"].astype(str)], ignore_index=True
        )
    )
    half = len(pairs)
    return codes[:half], codes[half:]


def run_frustration(
    spec: Any,
    settings: Any,
    n_decoys: int,
    generator_kwargs: Mapping[str, Any] | None = None,
    regions: tuple[str, str, str] = ("protein", "protein", "protein"),
) -> FrustrationRun:
    """Coordinates -> per-contact frustration class, serially, with no run directory.

    ``regions`` is ``(mutate, repack, minimize)`` as selector expressions. Serial by design:
    the cases are small and a spawn pool would re-initialise PyRosetta per worker for a
    handful of decoys.
    """
    from atomfrust.analyze.classify import classify_index
    from atomfrust.analyze.zscore import compute_index
    from atomfrust.decoys.base import DecoyContext, extract_energies
    from atomfrust.decoys.identity import IdentityDecoyGenerator
    from atomfrust.energy import effective_energy, many_body_energies
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex
    from atomfrust.regions import resolve_regions

    if n_decoys < 2:
        raise ValueError(f"need >= 2 decoys to estimate a spread, got {n_decoys}")

    started = time.perf_counter()
    loaded = load_complex(spec)
    _, superset = build_graph(loaded.nodes, loaded.geometry, settings)
    nodes_df = _nodes_frame(loaded.nodes)

    column = f"in__{settings.contacts.definition}"
    pairs = superset[superset[column].to_numpy()].reset_index(drop=True)
    if pairs.empty:
        raise ValueError(
            f"{spec.system_id}: no pair of {len(superset)} in the superset satisfies "
            f"{settings.contacts.definition} <= {settings.contacts.cutoff_A} A "
            f"(ligand {settings.contacts.ligand_cutoff_A} A)"
        )

    resolved = resolve_regions(
        loaded.nodes, loaded.geometry, mutate=regions[0], repack=regions[1],
        minimize=regions[2],
    )
    context = DecoyContext(
        pose=loaded.pose,
        nodes=loaded.nodes,
        pairs=pairs,
        regions=resolved,
        settings=settings,
    )
    generator = IdentityDecoyGenerator(
        context=context,
        base_seed=settings.decoys.base_seed,
        **dict(generator_kwargs or {}),
    )

    native_pose = generator.prepare_native()
    _, native_direct, native_fa_rep = extract_energies(
        native_pose, pairs, settings.energy.score_function
    )

    decoy_direct = np.empty((n_decoys, len(pairs)), dtype=np.float64)
    decoy_fa_rep = np.empty_like(decoy_direct)
    decoy_started = time.perf_counter()
    for decoy_id in range(n_decoys):
        result = generator.generate(decoy_id)
        decoy_direct[decoy_id] = result.e_direct
        decoy_fa_rep[decoy_id] = result.e_fa_rep
    decoy_wall = (time.perf_counter() - decoy_started) / n_decoys

    exclude = settings.energy.exclude_fa_rep
    mode = settings.manybody.mode
    code_i, code_j = _node_codes(pairs)

    E_native = many_body_energies(
        code_i, code_j, effective_energy(native_direct, native_fa_rep, exclude), mode
    )
    effective_decoys = effective_energy(decoy_direct, decoy_fa_rep, exclude)
    E_decoys = np.vstack(
        [many_body_energies(code_i, code_j, row, mode) for row in effective_decoys]
    )

    F = compute_index(E_native, E_decoys, settings.analysis.index)
    labels = classify_index(F, settings.analysis.classify)

    return FrustrationRun(
        system_id=spec.system_id,
        nodes=nodes_df,
        pairs=pairs,
        F=F,
        labels=np.asarray(labels, dtype=object),
        n_decoys=n_decoys,
        decoy_wall_s=decoy_wall,
        total_wall_s=time.perf_counter() - started,
        region_counts=resolved.counts(),
        mutated_node_ids=tuple(
            node.node_id for node, keep in zip(loaded.nodes, resolved.mutate) if keep
        ),
    )


def _nodes_frame(nodes: Sequence[Any]) -> pd.DataFrame:
    from atomfrust.graph import nodes_to_frame

    return nodes_to_frame(list(nodes))


def _settings(**overrides: Any) -> Any:
    """A :class:`~atomfrust.settings.Settings` with nested overrides applied by dotted path."""
    from atomfrust.settings import Settings

    base = Settings().model_dump(mode="json")
    for path, value in overrides.items():
        parts = path.split("__")
        node = base
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return Settings.model_validate(base)


def _egfr_spec(pdb_id: str, processed_dir: Path, params_dir: Path) -> Any:
    """Build a spec from a Stage-4 ``{PDB}_clean.pdb`` plus its ``.params``.

    The processed file's HETATM ``resName`` is **already the Rosetta name** (5HG8 stores
    ``Z34``, not its CCD code ``634``), so autodetection yields the pose-facing name and the
    params file is ``{that name}.params``. Reporting in these cases is keyed by PDB id, never
    by ligand code, so the CCD/Rosetta split does not need resolving here.
    """
    from atomfrust.spec import LigandSpec, SystemSpec

    pdb = processed_dir / f"{pdb_id.upper()}_clean.pdb"
    if not pdb.exists():
        raise FileNotFoundError(pdb)
    spec = SystemSpec.from_pdb(pdb, system_id=pdb_id.upper())
    if not spec.ligands:
        raise ValueError(f"{pdb.name} has no non-water heteroatom component to score")
    ligands = []
    for lig in spec.ligands:
        params = params_dir / f"{lig.selector.comp_id}.params"
        if not params.exists():
            raise FileNotFoundError(params)
        ligands.append(LigandSpec(selector=lig.selector, params=params))
    return spec.model_copy(update={"ligands": tuple(ligands)})


def _missing_inputs(pdb_ids: Iterable[str], processed_dir: Path, params_dir: Path) -> list[str]:
    """Which pinned structures cannot be built, and why. Empty means all are ready."""
    from atomfrust.spec import SystemSpec

    problems: list[str] = []
    for pdb_id in pdb_ids:
        pdb = processed_dir / f"{pdb_id.upper()}_clean.pdb"
        if not pdb.exists():
            problems.append(f"{pdb} (dvc pull)")
            continue
        try:
            spec = SystemSpec.from_pdb(pdb, system_id=pdb_id.upper())
        except Exception as exc:  # pragma: no cover - malformed input only
            problems.append(f"{pdb}: {exc}")
            continue
        for lig in spec.ligands:
            params = params_dir / f"{lig.selector.comp_id}.params"
            if not params.exists():
                problems.append(f"{params} (dvc pull)")
    return problems


def _pyrosetta_missing() -> str | None:
    try:
        import pyrosetta  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the machine
        return f"PyRosetta is not importable: {exc}"
    return None


# ------------------------------------------------------------------------------ F2


def case_interface_fraction(
    cases: Sequence[InterfaceCase] = PINNED_INTERFACES,
    n_decoys: int = 25,
    seq_sep_min: int = 1,
    relax: str = "min",
    search: Sequence[Path] | None = None,
) -> CaseResult:
    """F2 — fraction of minimally frustrated protein-protein **interface** contacts.

    The only ligand-independent numerical gate in the plan: 14.2% +/- 3 pp
    (Chen et al. 2020, Suppl. Fig. 8, via methods-doc S0.2). It also exercises the per-chain
    sequence-separation fix (R02) that a single-chain prototype could never test — an
    inter-chain pair carries ``seq_sep = -1`` in ``build_superset`` and is exempt from the
    ``|i-j|`` filter by construction, where the prototype's ``range(i + seq_sep_min, n)``
    over *pose index* silently discarded cross-boundary neighbours.

    ``seq_sep_min`` defaults to 1 (no filter), which is what step A4 found the paper states.
    """
    problem = _pyrosetta_missing()
    if problem:
        return CaseResult("interface-fraction", "SKIP", detail=problem)

    resolved = [(case, find_interface_structure(case.pdb_id, search)) for case in cases]
    available = [(case, path) for case, path in resolved if path is not None]
    if not available:
        wanted = "; ".join(f"{c.pdb_id} chains {'/'.join(c.chains)} — {c.reason}" for c in cases)
        return CaseResult(
            name="interface-fraction",
            status="SKIP",
            expected=PUBLISHED_INTERFACE_FRACTION,
            tolerance=PUBLISHED_INTERFACE_TOLERANCE_PP / 100.0,
            detail=(
                "no multi-chain structure is available. Everything under data/ is "
                "single-chain: data/processed/*_clean.pdb is one EGFR chain plus one ligand "
                "copy by construction (Stage 4), and data/raw_pdb/ holds only 1LYZ. "
                f"This case needs one coordinate file per pinned complex — {wanted} — in "
                f"{INTERFACE_DIR} or {RAW_PDB_DIR}. Nothing is downloaded here by design."
            ),
            extra={"pinned": [c.pdb_id for c in cases]},
        )

    settings = _settings(
        contacts__seq_sep_min=seq_sep_min,
        decoys__n_decoys=n_decoys,
    )
    import tempfile

    rows: list[dict[str, Any]] = []
    unusable: list[str] = []
    with tempfile.TemporaryDirectory(prefix="atomfrust-f2-") as workdir:
        from atomfrust.spec import PocketSpec, Receptor, SystemSpec

        for case, path in available:
            try:
                prepared = write_protein_only_pdb(
                    path, case.chains, Path(workdir) / f"{case.pdb_id}.pdb"
                )
            except ValueError as exc:
                # A file was found under that id but does not carry both pinned chains —
                # a single-chain processed copy, say. That is missing data, not a failed
                # measurement, so it drops out of the set rather than crashing the case.
                unusable.append(f"{case.pdb_id} ({path.name}): {exc}")
                continue
            spec = SystemSpec(
                system_id=f"{case.pdb_id}_{''.join(case.chains)}",
                receptor=Receptor(path=prepared, chains=tuple(case.chains)),
                ligands=(),
                pocket=PocketSpec(mode="chain_interface", chains=tuple(case.chains)),
            )
            run = run_frustration(
                spec,
                settings,
                n_decoys,
                generator_kwargs={"relax": relax},
            )
            mask = interface_mask(run.pairs)
            minimal = run.labels[mask] == "minimally_frustrated"
            low, high = bootstrap_fraction_ci(minimal)
            rows.append(
                {
                    "pdb_id": case.pdb_id,
                    "chains": "/".join(case.chains),
                    "n_interface_contacts": int(mask.sum()),
                    "frac_minimally": float(minimal.mean()) if mask.sum() else float("nan"),
                    "ci_low": low,
                    "ci_high": high,
                    "n_decoys": n_decoys,
                    "decoy_wall_s": run.decoy_wall_s,
                }
            )

    if not rows:
        return CaseResult(
            name="interface-fraction",
            status="SKIP",
            expected=PUBLISHED_INTERFACE_FRACTION,
            tolerance=PUBLISHED_INTERFACE_TOLERANCE_PP / 100.0,
            detail=(
                "a coordinate file was found for every pinned complex, but none of them "
                "carries both pinned chains: " + "; ".join(unusable)
            ),
            extra={"pinned": [c.pdb_id for c in cases], "unusable": unusable},
        )

    fractions = np.array([r["frac_minimally"] for r in rows], dtype=float)
    pooled = float(np.nanmean(fractions))
    deviation_pp = abs(pooled - PUBLISHED_INTERFACE_FRACTION) * 100.0
    status = "PASS" if deviation_pp <= PUBLISHED_INTERFACE_TOLERANCE_PP else "FAIL"
    per_case = ", ".join(
        f"{r['pdb_id']} {r['frac_minimally']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
        f"over {r['n_interface_contacts']} contacts"
        for r in rows
    )
    return CaseResult(
        name="interface-fraction",
        status=status,
        measured=pooled,
        expected=PUBLISHED_INTERFACE_FRACTION,
        tolerance=PUBLISHED_INTERFACE_TOLERANCE_PP / 100.0,
        detail=(
            f"mean minimally-frustrated interface fraction {pooled:.3f} over "
            f"{len(rows)} pinned complex(es) at {n_decoys} decoys "
            f"(published 14.2%, |delta| = {deviation_pp:.1f} pp). Per case: {per_case}. "
            f"seq_sep_min={seq_sep_min}; inter-chain pairs carry seq_sep = -1 and are exempt "
            "from the filter, which is the per-chain fix under test."
        ),
        extra={
            "per_case": rows,
            "skipped": [c.pdb_id for c, p in resolved if p is None],
            "unusable": unusable,
        },
    )


# ------------------------------------------------------------------------------ F3


def case_reference_counts(
    pdb_ids: Sequence[str] = PINNED_REFERENCE_STRUCTURES,
    n_decoys: int = 10,
    ligand_cutoff_A: float = 6.0,
    relax: str = "min",
    processed_dir: Path = PROCESSED_DIR,
    params_dir: Path = PARAMS_DIR,
    reference_table: Path | None = None,
) -> CaseResult:
    """F3 — per-structure reproduction of Chen et al.'s minimally-frustrated counts.

    Counts **ligand-incident** contacts (exactly one non-protein endpoint) classified
    minimally frustrated, which step A4 established is the published object; the prototype's
    266-407 were protein-protein pairs in a 10 A shell and are a different quantity, not a
    mis-scaled one.

    Defaults are **smoke scale** — 3 structures, 10 decoys. The full reproduction is 61
    structures at the paper's 1000 decoys; the result records what that would cost from the
    per-decoy time measured here. No number produced by this function is a full-set result.
    """
    problem = _pyrosetta_missing()
    if problem:
        return CaseResult("reference-counts", "SKIP", detail=problem)

    table_path = reference_table or REFERENCE_TABLE
    if not table_path.exists():
        return CaseResult(
            "reference-counts", "SKIP",
            detail=f"missing the published counts: {table_path}",
        )
    missing = _missing_inputs(pdb_ids, processed_dir, params_dir)
    if missing:
        return CaseResult(
            "reference-counts", "SKIP",
            detail="missing inputs: " + "; ".join(missing),
        )

    paper = paper_reference_counts(table_path)
    unknown = [p for p in pdb_ids if p.upper() not in paper.index]
    if unknown:
        return CaseResult(
            "reference-counts", "SKIP",
            detail=f"{unknown} are not in {table_path.name}",
        )

    settings = _settings(
        contacts__ligand_cutoff_A=ligand_cutoff_A,
        decoys__n_decoys=n_decoys,
    )

    rows: list[dict[str, Any]] = []
    for pdb_id in pdb_ids:
        spec = _egfr_spec(pdb_id, processed_dir, params_dir)
        run = run_frustration(
            spec, settings, n_decoys, generator_kwargs={"relax": relax}
        )
        mask = ligand_incident_mask(run.pairs)
        labels = run.labels[mask]
        rows.append(
            {
                "pdb_id": pdb_id.upper(),
                "n_ligand_contacts": int(mask.sum()),
                "computed_minimally": int((labels == "minimally_frustrated").sum()),
                "computed_highly": int((labels == "highly_frustrated").sum()),
                "paper_minimally": int(
                    paper.loc[pdb_id.upper(), "paper_minimally_frustrated_contacts"]
                ),
                "paper_highly": int(
                    paper.loc[pdb_id.upper(), "paper_highly_frustrated_contacts"]
                ),
                "n_contacts_total": int(len(run.pairs)),
                "decoy_wall_s": run.decoy_wall_s,
            }
        )

    computed = [r["computed_minimally"] for r in rows]
    published = [r["paper_minimally"] for r in rows]
    r_value, p_value = _pearson(computed, published)
    low, high = min(computed), max(computed)
    overlaps = low <= PAPER_COUNT_RANGE[1] and high >= PAPER_COUNT_RANGE[0]

    mean_decoy_s = float(np.mean([r["decoy_wall_s"] for r in rows]))
    full_core_hours = FULL_SET_STRUCTURES * FULL_SET_DECOYS * mean_decoy_s / 3600.0

    per_structure = ", ".join(
        f"{r['pdb_id']} {r['computed_minimally']}/{r['n_ligand_contacts']} vs paper "
        f"{r['paper_minimally']}"
        for r in rows
    )
    return CaseResult(
        name="reference-counts",
        status="PASS" if overlaps else "FAIL",
        measured={"counts": {r["pdb_id"]: r["computed_minimally"] for r in rows},
                  "range": [low, high], "pearson_r": r_value, "pearson_p": p_value},
        expected={"counts": {r["pdb_id"]: r["paper_minimally"] for r in rows},
                  "range": list(PAPER_COUNT_RANGE)},
        tolerance="count range must overlap the published 4-23",
        detail=(
            f"SMOKE SCALE, NOT THE FULL REPRODUCTION: {len(rows)} of {FULL_SET_STRUCTURES} "
            f"structures at {n_decoys} of the paper's {FULL_SET_DECOYS} decoys, relax="
            f"{relax}. Ligand-incident contacts (exactly one non-protein endpoint, "
            f"heavy-atom min <= {ligand_cutoff_A} A). {per_structure}. Computed range "
            f"{low}-{high} vs published {PAPER_COUNT_RANGE[0]}-{PAPER_COUNT_RANGE[1]}; "
            f"Pearson r = {r_value:.3f} (p = {p_value:.3f}) on n = {len(rows)}, which is "
            "reported because the plan asks for it and supports no inference at this n. "
            f"Measured {mean_decoy_s:.1f} s/decoy serially, so the full "
            f"{FULL_SET_STRUCTURES} x {FULL_SET_DECOYS} run is ~{full_core_hours:,.0f} "
            "core-hours before any Monte-Carlo relaxation."
        ),
        extra={
            "per_structure": rows,
            "smoke_scale": True,
            "full_run_core_hours": full_core_hours,
            "mean_decoy_wall_s": mean_decoy_s,
        },
    )


# ------------------------------------------------------------------------------ F6


def case_pocket_repack_equivalence(
    pdb_id: str = "5GMP",
    n_decoys: int = 6,
    shell_radii: Sequence[float] = (8.0, 12.0),
    pocket_shell_A: float = 6.0,
    relax: str = "min",
    processed_dir: Path = PROCESSED_DIR,
    params_dir: Path = PARAMS_DIR,
) -> CaseResult:
    """F6 — whole-protein repack vs shell-restricted repack, paired by ``decoy_id``.

    Both arms mutate the **same** pocket set (``protein and within(pocket_shell_A, ligand)``)
    and differ only in how far the repack and minimisation extend, so the measurement
    isolates the repack region rather than confounding it with a different mutate set. The
    pairing is exact because of step C4: ``_position_rng`` seeds every (decoy, position)
    substream from ``(base_seed, decoy_id, position)``, so a position draws the same identity
    whatever else is in the mutate set. This requires ``identity="composition"``,
    ``placement="inplace"``, ``seeding="substream"`` — the substream branch of
    ``draw_sequence`` covers no other combination, and ``placement="permute"`` cannot be
    decomposed per position at all.

    ``mutation="packer_task"`` in both arms, not for speed (measured: it buys nothing) but
    because it is the only path that can express a restricted repack;
    ``IdentityDecoyGenerator.__post_init__`` raises for ``mutation="sequential"`` when any
    protein residue is excluded from repacking.

    Per plan F6, a rho below 0.95 is a finding about the shell approximation and is reported
    as such — the case does not fail on it.
    """
    problem = _pyrosetta_missing()
    if problem:
        return CaseResult("pocket-repack-equivalence", "SKIP", detail=problem)
    missing = _missing_inputs([pdb_id], processed_dir, params_dir)
    if missing:
        return CaseResult(
            "pocket-repack-equivalence", "SKIP",
            detail="missing inputs: " + "; ".join(missing),
        )

    settings = _settings(decoys__n_decoys=n_decoys)
    generator_kwargs = {
        "scope": "region",
        "identity": "composition",
        "placement": "inplace",
        "seeding": "substream",
        "mutation": "packer_task",
        "relax": relax,
    }
    mutate = f"protein and within({pocket_shell_A}, ligand)"
    spec = _egfr_spec(pdb_id, processed_dir, params_dir)

    baseline = run_frustration(
        spec, settings, n_decoys, generator_kwargs,
        regions=(mutate, "protein", "protein"),
    )
    # The shell approximation is meant to preserve the contacts around the pocket. Contacts
    # far outside it are near-constant between native and decoy under a restricted repack —
    # nothing there is repacked — so pooling them lets a mass of F = 0 dominate a rank
    # correlation. Both are reported; this is the primary.
    pocket_pairs = baseline.incident_to_mutated()

    rows: list[dict[str, Any]] = []
    for radius in shell_radii:
        shell = f"protein and within({radius}, ligand)"
        restricted = run_frustration(
            spec, settings, n_decoys, generator_kwargs,
            regions=(mutate, shell, shell),
        )
        if not restricted.pairs["pair_id"].equals(baseline.pairs["pair_id"]):
            raise AssertionError(
                "the two arms selected different contact sets; they cannot be paired"
            )
        rows.append(
            {
                "shell_A": float(radius),
                "rho_pocket": _spearman(baseline.F[pocket_pairs], restricted.F[pocket_pairs]),
                "rho_all": _spearman(baseline.F, restricted.F),
                "n_pocket_contacts": int(pocket_pairs.sum()),
                "n_contacts": int(len(baseline.pairs)),
                "repack_residues": restricted.region_counts["repack"],
                "baseline_repack_residues": baseline.region_counts["repack"],
                "mutate_residues": restricted.region_counts["mutate"],
                "decoy_wall_s": restricted.decoy_wall_s,
                "baseline_decoy_wall_s": baseline.decoy_wall_s,
                "speedup": baseline.decoy_wall_s / restricted.decoy_wall_s,
            }
        )

    summary = "; ".join(
        f"{r['shell_A']:.0f} A: rho_pocket = {r['rho_pocket']:.3f}, rho_all = "
        f"{r['rho_all']:.3f}, {r['speedup']:.2f}x ({r['repack_residues']} of "
        f"{r['baseline_repack_residues']} residues repacked)"
        for r in rows
    )
    below = [r for r in rows if not (r["rho_pocket"] >= 0.95)]
    return CaseResult(
        name="pocket-repack-equivalence",
        status="PASS",
        measured=rows,
        expected={"rho_reference": 0.95},
        tolerance="none — plan F6 records rho and speed-up rather than gating on them",
        detail=(
            f"{pdb_id}, {n_decoys} paired decoys, mutate set = {mutate} "
            f"({rows[0]['mutate_residues']} residues), identity=composition/inplace with "
            f"substream seeding so identities are position-pinned across arms. {summary}. "
            + (
                f"{len(below)} radius/radii below the 0.95 reference: that is a finding "
                "about the shell approximation, not a failure (plan F6)."
                if below
                else "every radius at or above the 0.95 reference."
            )
        ),
        extra={"per_radius": rows, "pdb_id": pdb_id, "n_decoys": n_decoys},
    )


# --------------------------------------------------------------------------- registry

#: The standalone surface, keyed by the plan's CLI names (§5). Each value is a plain
#: function returning the :class:`CaseResult` defined *in this module*, so the cases are
#: usable with no orchestrator at all.
CASES: dict[str, Callable[..., CaseResult]] = {
    "interface-fraction": case_interface_fraction,
    "reference-counts": case_reference_counts,
    "pocket-repack-equivalence": case_pocket_repack_equivalence,
}

#: Short names under which the cases enter ``atomfrust.validation.cases.CASES``, matching
#: the F-numbering its own cases already use (``F1``, ``F4``, ``F5``).
ORCHESTRATOR_NAMES = {
    "interface-fraction": "F2",
    "reference-counts": "F3",
    "pocket-repack-equivalence": "F6",
}


def _root_paths(root: Path | str) -> tuple[Path, Path, Path]:
    """``(processed, params, reference_table)`` under ``root``, or this repo's own.

    ``atomfrust validate --root`` points a case at a tree; when that tree has no ``data/``
    (the documented way to watch every case SKIP cleanly) the module defaults are used, and
    the case then reports its own missing inputs rather than silently measuring nothing.
    """
    root = Path(root)
    processed = root / "data" / "processed"
    params = root / "data" / "ligands" / "params"
    table = root / "config" / "pdb_reference_table.csv"
    if processed.is_dir() and params.is_dir():
        return processed, params, table
    return PROCESSED_DIR, PARAMS_DIR, REFERENCE_TABLE


def _orchestrator_cases(module: Any) -> list[Any]:
    """Adapters exposing these three cases as ``ValidationCase`` instances.

    Built lazily inside a function because ``atomfrust/validation/cases.py`` belongs to a
    different work stream: this module must import, and its functions must run, whether or
    not that module is present or keeps its current shape. The expectations live here, in
    the adapters, so they are pinned in source exactly as the sibling cases' are.
    """
    from dataclasses import dataclass as _dataclass
    from dataclasses import field as _field

    base = module.ValidationCase

    @_dataclass(frozen=True)
    class InterfaceFractionCase(base):  # type: ignore[valid-type,misc]
        """F2 — minimally frustrated fraction of protein-protein interface contacts.

        *Proves:* the one ligand-independent numerical claim available — Chen et al.'s
        atomistic 14.2% of interface contacts (Suppl. Fig. 8; methods-doc S0.2) — and with
        it the per-chain sequence-separation fix (R02), since an inter-chain pair carries
        ``seq_sep = -1`` and is exempt from the ``|i-j|`` filter by construction, where the
        prototype's ``range(i + seq_sep_min, n)`` over pose index discarded cross-boundary
        neighbours outright.

        *Does not prove:* anything about ligands, and nothing about which interfaces Chen
        et al. averaged over — the published figure is a set average and the three complexes
        pinned here are ours, chosen before any of them was run.

        Currently SKIPs: every structure under ``data/`` is single-chain. Nothing is
        downloaded to fix that.
        """

        name: str = "F2"
        summary: str = (
            "protein-protein interface minimally-frustrated fraction vs the published 14.2%"
        )
        expected: Mapping[str, Any] = _field(
            default_factory=lambda: {"frac_minimally": PUBLISHED_INTERFACE_FRACTION}
        )
        tolerance: Mapping[str, Any] = _field(
            default_factory=lambda: {
                "frac_minimally": PUBLISHED_INTERFACE_TOLERANCE_PP / 100.0
            }
        )

        def measure(self, root: Path, **options: Any) -> Any:
            result = case_interface_fraction(**options)
            if result.status == "SKIP":
                return self.skip(result.detail)
            return self.measured(
                {
                    "frac_minimally": result.measured,
                    "n_cases": len(result.extra.get("per_case", [])),
                    "per_case": result.extra.get("per_case", []),
                },
                result.detail,
            )

    @_dataclass(frozen=True)
    class ReferenceCountsCase(base):  # type: ignore[valid-type,misc]
        """F3 — computed ligand-incident counts against Chen et al.'s per-structure numbers.

        *Proves:* that the new graph counts the **same kind of object** the paper reports.
        Step A4 established the published 4-23 are ligand-residue contacts; the prototype's
        266-407 were protein-protein pairs in a 10 A shell, which is why no selector or
        threshold over them could ever close the gap (step A2, 189 configurations). The
        checked expectation is therefore the *scale* — the computed range must overlap the
        published 4-23 — not agreement structure by structure.

        *Does not prove:* the reproduction. The defaults are 3 of 61 structures at 10 of the
        paper's 1000 decoys, so the counts are smoke-scale and the Pearson r against the
        paper is reported as a diagnostic at an n that supports no inference. The full run's
        cost is recorded in the detail from the per-decoy time measured here.
        """

        name: str = "F3"
        summary: str = (
            "ligand-incident minimally-frustrated counts vs config/pdb_reference_table.csv "
            "(smoke scale)"
        )
        expected: Mapping[str, Any] = _field(
            default_factory=lambda: {"range_overlaps_paper": True}
        )

        def measure(self, root: Path, **options: Any) -> Any:
            processed, params, table = _root_paths(root)
            options.setdefault("processed_dir", processed)
            options.setdefault("params_dir", params)
            options.setdefault("reference_table", table)
            result = case_reference_counts(**options)
            if result.status == "SKIP":
                return self.skip(result.detail)
            low, high = result.measured["range"]
            return self.measured(
                {
                    "range_overlaps_paper": result.status == "PASS",
                    "min_count": low,
                    "max_count": high,
                    "pearson_r": result.measured["pearson_r"],
                    "counts": result.measured["counts"],
                    "paper_counts": result.expected["counts"],
                    "smoke_scale": True,
                    "full_run_core_hours": result.extra["full_run_core_hours"],
                },
                result.detail,
            )

    @_dataclass(frozen=True)
    class PocketRepackEquivalenceCase(base):  # type: ignore[valid-type,misc]
        """F6 — whole-protein repack vs shell-restricted repack, paired decoy by decoy.

        *Proves:* what the shell approximation costs, in rank agreement of the per-contact
        index, and what it buys in wall-clock — both per shell radius. The pairing is exact
        because step C4 gives every (decoy, position) its own RNG substream, so identities
        do not depend on the size of the mutate set.

        *Does not prove:* that either arm is right. Per plan step F6 **a rho below 0.95 is a
        finding about the approximation, not a failure**, so this case pins no expectation on
        rho; it fails only if the two arms cannot be paired at all.
        """

        name: str = "F6"
        summary: str = (
            "shell-restricted vs whole-protein repack: Spearman rho and speed-up per radius"
        )

        def measure(self, root: Path, **options: Any) -> Any:
            processed, params, _ = _root_paths(root)
            options.setdefault("processed_dir", processed)
            options.setdefault("params_dir", params)
            result = case_pocket_repack_equivalence(**options)
            if result.status == "SKIP":
                return self.skip(result.detail)
            measured: dict[str, Any] = {"n_decoys": result.extra["n_decoys"],
                                        "pdb_id": result.extra["pdb_id"]}
            for row in result.measured:
                key = f"{row['shell_A']:.0f}A"
                measured[f"rho_pocket_{key}"] = row["rho_pocket"]
                measured[f"rho_all_{key}"] = row["rho_all"]
                measured[f"speedup_{key}"] = row["speedup"]
            return self.measured(measured, result.detail)

    return [InterfaceFractionCase(), ReferenceCountsCase(), PocketRepackEquivalenceCase()]


def register_all(registry: Any = None) -> list[str]:
    """Publish these cases. Returns the names registered, and is idempotent.

    With no argument the sibling registry ``atomfrust.validation.cases`` is used and the
    cases enter it as ``ValidationCase`` adapters named ``F2`` / ``F3`` / ``F6``, so
    ``atomfrust validate --case F3`` finds them. The import is lazy and every failure is
    swallowed: that module belongs to a different work stream, and this one has to keep
    working — as plain functions in :data:`CASES` — if it is missing or shaped differently.

    A mapping or a two-argument callable passed explicitly is treated as a registry of
    **plain functions** and receives :data:`CASES` under the plan's long names.
    """
    if registry is not None:
        try:
            if callable(registry):
                for name, function in CASES.items():
                    registry(name, function)
            elif hasattr(registry, "__setitem__"):
                for name, function in CASES.items():
                    registry[name] = function
            else:
                return []
        except Exception:
            return []
        return sorted(CASES)

    try:
        from atomfrust.validation import cases as _cases  # type: ignore

        registered: list[str] = []
        for case in _orchestrator_cases(_cases):
            if case.name.upper() in _cases.CASES:
                continue
            _cases.register(case)
            registered.append(case.name)
        return sorted(registered)
    except Exception:
        return []


# Importing this module is enough to publish the cases: `atomfrust/cli/validate.py` imports
# only `atomfrust.validation.cases`, so without this line `--case F3` would report an unknown
# case. Guarded inside `register_all`, which never raises.
register_all()
