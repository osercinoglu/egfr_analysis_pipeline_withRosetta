"""``atomfrust analyze`` — frustration from a finished run directory, with no PyRosetta.

Plan step E3, and the whole of user request 7. Every scientific choice this command makes
is one the run directory deliberately left open: the contact definition and its cutoff, the
sequence-separation rule, the pocket shell, the many-body formula, the index function, the
classification thresholds, ``exclude_fa_rep``, and how many decoys to use. None of them
needs a pose, because §4 of the plan stores **direct** pair energies and never ``E_ij``.

**``E_ij`` is reconstructed, not read.** ``e_direct`` and ``e_fa_rep`` come off disk;
:func:`~atomfrust.energy.effective_energy` applies the ``fa_rep`` switch and
:func:`~atomfrust.energy.many_body_energies` forms the requested formula. The background
sums run over the *selected* contact set, so narrowing the contact definition changes the
value of ``E_ij`` as well as which pairs are reported — that is the point of making the
definition re-specifiable rather than a filter on a stored derived quantity. The prototype
baked all of this in at scoring time (``frustration.py:210-212`` for ``fa_rep``, ``:216``
for Eq. 2), so changing any of it meant regenerating every decoy.

**``--n-decoys N`` is a prefix, and the prefix is exact.** Decoy *i* is seeded
``base_seed + i``, so ``decoy_id < N`` *is* the ensemble an N-decoy run would have produced
— same mutations, same repacks, same energies. It is not a subsample and carries no
sampling error of its own. That is what makes the convergence sweep one run instead of
eight.

**Analysis settings never trigger regeneration.** Requested settings are layered on top of
the run's own resolved settings, then checked with
:meth:`~atomfrust.runstore.RunDir.assert_compatible` at ``pyrosetta_version=None`` — the
version is not read here, because reading it would mean importing PyRosetta to do a job
that requires none. A generation-stage difference exits 3 with a field-by-field diff on
stderr; ``--allow-mismatch`` downgrades that to a report and stamps ``provisional: true``
on every output. A contact cutoff exceeding the stored superset is the same class of
failure — nothing on disk can answer it — so it also exits 3, with the message
:func:`~atomfrust.graph.add_contact_definition` raises.

**Where output goes.** ``systems/<system_id>/analyses/<analysis_id>/`` gets
``manifest.json`` (the analysis settings plus the run's ``regeneration_key`` they were
computed against), ``contacts.parquet`` (the per-pair table of plan §4) and
``summary.json`` (every descriptor in the registry plus every mandatory covariate).
``<analysis_id>`` defaults to a short deterministic digest of the analysis-stage settings,
so two different settings can never overwrite each other and re-running one setting always
lands in the same directory. ``-o`` overrides it with a readable name when that matters.

**Descriptors are not selectable, deliberately.** ``--descriptor`` is validated against the
registry and recorded, but :mod:`atomfrust.analyze.aggregate` evaluates all of them and
emits all seven covariates unconditionally, because a count that travels without its
alternatives and without the pocket size that explains it is the reporting defect the
aggregate module exists to prevent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from atomfrust.analyze.aggregate import (
    COVARIATES,
    DESCRIPTORS,
    pocket_mask,
    shell_nodes,
    summarize,
)
from atomfrust.analyze.classify import class_counts, classify_index, quantile_thresholds
from atomfrust.analyze.zscore import compute_index, decoy_summary, normality_diagnostics
from atomfrust.energy import effective_energy, many_body_energies
from atomfrust.graph import add_contact_definition
from atomfrust.provenance import CONTRACT_VERSION, ManifestVersionError
from atomfrust.runstore import (
    EXIT_REQUIRES_REGENERATION,
    RegenerationRequired,
    RunDir,
    SystemDir,
)
from atomfrust.settings import ClassifySettings, Settings, load_settings, stage_subset

__all__ = [
    "NAME",
    "HELP",
    "register",
    "run",
    "analysis_id",
    "AnalysisError",
    "CONTACTS_COLUMNS",
]

NAME = "analyze"
HELP = "compute frustration from an existing run directory"

#: A run directory that cannot answer the question asked of it, or malformed arguments that
#: argparse cannot express.
EXIT_ANALYSIS_ERROR = 1
EXIT_USAGE = 2

MANIFEST_FILENAME = "manifest.json"
CONTACTS_FILENAME = "contacts.parquet"
SUMMARY_FILENAME = "summary.json"

#: Name of the ``in__<name>`` column this command adds to the stored pairs table. The
#: stored definitions are whatever generation happened to register; membership is always
#: recomputed here so a re-specified cutoff cannot silently fall back to a stored column.
_DEFINITION = "selected"

#: ``contacts.parquet``, in the order plan §4 lists it.
CONTACTS_COLUMNS = (
    "pair_id", "i", "j", "E_native",
    "decoy_mean", "decoy_std", "decoy_median", "decoy_mad", "n_decoys",
    "F", "index_name", "shapiro_W", "shapiro_p", "class",
)

#: ``pocket.reference`` names the atom set; :func:`~atomfrust.analyze.aggregate.pocket_mask`
#: names the stored distance column. ``sidechain_heavy`` has no stored column — the graph
#: writes ``d_ca``, ``d_cb`` and ``d_heavy_min`` and nothing else — so it is rejected rather
#: than silently served by ``d_heavy_min``, which is a different quantity.
_SHELL_REFERENCE = {"ca": "ca", "cb": "cb", "any_heavy": "heavy_min"}


class AnalysisError(RuntimeError):
    """A run directory cannot answer this question. Reported, never raised past :func:`run`."""


# ----------------------------------------------------------------------------- parser


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        NAME,
        help=HELP,
        description=(
            "Recompute the frustration index from a finished run directory. Makes no "
            "PyRosetta call: E_ij is reconstructed from the stored direct pair energies, "
            "so the contact definition, shell, many-body formula, index, thresholds and "
            "decoy count are all chosen here."
        ),
        epilog=(
            "examples:\n"
            "  atomfrust analyze --run-dir runs/gen --shell-A 5.0 --index rank_percentile "
            "-o analyses/s5_rank\n"
            "  atomfrust analyze --run-dir runs/gen --manybody chen_literal "
            "-o analyses/literal\n"
            "  atomfrust analyze --run-dir runs/gen --n-decoys 250 --contact-def heavy_min\n"
            "\n"
            "exit codes:\n"
            "  0  every requested system analysed\n"
            "  1  a system could not be analysed (missing decoys, empty contact set)\n"
            "  2  arguments argparse cannot reject on its own\n"
            "  3  the request needs a new decoy ensemble; the field diff is on stderr\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, metavar="R", help="finished run directory"
    )

    contacts = parser.add_argument_group("contact definition (within the stored superset)")
    contacts.add_argument(
        "--contact-def", choices=("ca_ca", "cb_cb", "heavy_min"),
        help="protein-protein rule; pairs touching a non-protein node always use heavy-min",
    )
    contacts.add_argument(
        "--contact-cutoff-A", type=float, metavar="X",
        help="cutoff for that rule. Exceeding the superset requires regeneration (exit 3).",
    )
    contacts.add_argument(
        "--seq-sep-min", type=int, metavar="K", help="minimum |i-j| within one chain"
    )
    contacts.add_argument(
        "--seq-sep-basis", choices=("resseq", "pose_index"),
        help="what |i-j| counts: PDB numbering, or modelled residues only",
    )

    pocket = parser.add_argument_group("pocket")
    pocket.add_argument(
        "--shell-ref", choices=tuple(_SHELL_REFERENCE), metavar="REF",
        help=f"atom set the shell is measured from: {', '.join(_SHELL_REFERENCE)}",
    )
    pocket.add_argument("--shell-A", type=float, metavar="X", help="shell radius")

    index = parser.add_argument_group("index")
    index.add_argument(
        "--manybody", choices=("pair_retained", "chen_literal", "pair_only"),
        help="many-body formula; background sums run over the selected contact set",
    )
    index.add_argument("--index", choices=("zscore", "rank_percentile", "robust_z"))
    thresholds = index.add_mutually_exclusive_group()
    thresholds.add_argument(
        "--thresholds", metavar="MIN,HIGH",
        help="fixed class cutoffs, e.g. 0.78,-1.0",
    )
    thresholds.add_argument(
        "--thresholds-mode", choices=("fixed", "quantile"),
        help="'quantile' re-derives cutoffs from the pooled F of the analysed systems",
    )

    selection = parser.add_argument_group("ensemble and system selection")
    selection.add_argument(
        "--n-decoys", type=int, metavar="N",
        help="use decoy_id < N. A prefix, and exact: decoy i is seeded base_seed + i.",
    )
    selection.add_argument(
        "--axis", action="append", default=[], metavar="A",
        help="restrict to this decoy axis (repeatable or comma-separated)",
    )
    selection.add_argument(
        "--descriptor", action="append", default=[], metavar="D",
        help="record these descriptors as requested (repeatable). The registry is always "
             "evaluated in full; this cannot narrow the output.",
    )
    selection.add_argument(
        "--systems", metavar="ID[,ID]", help="systems to analyse; default is all of them"
    )

    fa_rep = parser.add_mutually_exclusive_group()
    fa_rep.add_argument(
        "--exclude-fa-rep", dest="exclude_fa_rep", action="store_true", default=None,
        help="subtract the stored weighted fa_rep contribution (the published choice)",
    )
    fa_rep.add_argument(
        "--include-fa-rep", dest="exclude_fa_rep", action="store_false", default=None,
        help="keep fa_rep in the contact energy",
    )

    parser.add_argument(
        "-o", "--out", metavar="analyses/<id>",
        help="analysis id; default is a digest of the analysis settings",
    )
    parser.add_argument(
        "--allow-mismatch", action="store_true",
        help="proceed despite a generation-stage difference, marking outputs provisional",
    )
    parser.add_argument(
        "--config", type=Path, metavar="FILE", help="YAML settings overlay"
    )
    parser.set_defaults(func=run)


# ------------------------------------------------------------------- settings layering


def _put(overlay: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path, skipping ``None`` so an unspecified flag inherits the run's value."""
    if value is None:
        return
    parts = path.split(".")
    node = overlay
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _split_list(values: Iterable[str]) -> list[str]:
    return [v.strip() for value in values for v in str(value).split(",") if v.strip()]


def _parse_thresholds(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--thresholds expects MIN,HIGH, got {text!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"--thresholds expects two numbers, got {text!r}") from None


def _cli_overlay(args: argparse.Namespace) -> dict[str, Any]:
    """Only the flags actually given. Everything else inherits the run's resolved settings."""
    overlay: dict[str, Any] = {}
    _put(overlay, "contacts.definition", args.contact_def)
    _put(overlay, "contacts.cutoff_A", args.contact_cutoff_A)
    _put(overlay, "contacts.seq_sep_min", args.seq_sep_min)
    _put(overlay, "contacts.seq_sep_basis", args.seq_sep_basis)
    _put(overlay, "pocket.reference", args.shell_ref)
    _put(overlay, "pocket.cutoff_A", args.shell_A)
    _put(overlay, "manybody.mode", args.manybody)
    _put(overlay, "analysis.index", args.index)
    _put(overlay, "analysis.n_decoys", args.n_decoys)
    _put(overlay, "energy.exclude_fa_rep", args.exclude_fa_rep)

    axes = _split_list(args.axis)
    if axes:
        _put(overlay, "analysis.axes", axes)

    descriptors = _split_list(args.descriptor)
    if descriptors:
        unknown = sorted(set(descriptors) - set(DESCRIPTORS))
        if unknown:
            raise ValueError(
                f"unknown descriptor(s) {unknown}; available: {sorted(DESCRIPTORS)}"
            )
        _put(overlay, "analysis.descriptors", descriptors)

    if args.thresholds:
        minimal, highly = _parse_thresholds(args.thresholds)
        _put(overlay, "analysis.classify.minimally_frustrated", minimal)
        _put(overlay, "analysis.classify.highly_frustrated", highly)
        _put(overlay, "analysis.classify.mode", "fixed")
    _put(overlay, "analysis.classify.mode", args.thresholds_mode)
    return overlay


def _resolve_settings(run_dir: RunDir, args: argparse.Namespace) -> Settings:
    """Run's resolved settings → ``--config`` → environment → CLI.

    The base is the *run's* settings, not the package defaults: a generation-stage field
    that differs only because this process defaulted it would raise the regeneration error
    the run directory contract reserves for a genuine change.
    """
    base = run_dir.read_settings().model_dump(mode="json")
    if args.config is not None:
        loaded = yaml.safe_load(Path(args.config).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{args.config}: expected a YAML mapping of settings")
        _deep_update(base, loaded)
    settings, _ = load_settings(file_overlay=base, cli_overlay=_cli_overlay(args))
    return settings


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def analysis_id(settings: Settings) -> str:
    """Short digest of the analysis-stage settings.

    Two different settings therefore land in two directories and neither can overwrite the
    other, while re-running one setting always lands in the same place. Generation-stage
    fields are excluded by construction — they are the run's identity, not the analysis's,
    and the run's ``regeneration_key`` is recorded in the analysis manifest instead.
    """
    payload = json.dumps(
        stage_subset(settings, "analysis"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _analysis_dirname(out: str | None, settings: Settings) -> str:
    """``-o`` names one directory under ``analyses/``; a leading ``analyses/`` is redundant."""
    if out is None:
        return analysis_id(settings)
    name = str(out).strip().rstrip("/")
    if name.startswith("analyses/"):
        name = name[len("analyses/"):]
    if not name or "/" in name or "\\" in name:
        raise ValueError(
            f"-o {out!r} must name a single directory under analyses/, e.g. -o s5_rank"
        )
    return name


# ------------------------------------------------------------------------ computation


@dataclass
class _SystemAnalysis:
    """One system's per-pair result, before classification.

    Thresholds are resolved once across every system (``--thresholds-mode quantile``
    calibrates on the pooled distribution), so classification cannot happen until all
    systems have been computed.
    """

    system_id: str
    nodes: pd.DataFrame
    pairs: pd.DataFrame
    frame: pd.DataFrame
    F: np.ndarray
    n_members: int
    n_pairs_superset: int


def _node_codes(pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Dense integer codes for the pair endpoints.

    :func:`~atomfrust.energy.many_body_energies` factorises whatever it is handed, once per
    call; handing it integers instead of node-id strings makes the per-decoy loop below
    cheap without changing a single result.
    """
    codes, _ = pd.factorize(
        pd.concat(
            [pairs["node_i"].astype(str), pairs["node_j"].astype(str)], ignore_index=True
        )
    )
    half = len(pairs)
    return codes[:half], codes[half:]


def _many_body_block(
    node_i: np.ndarray, node_j: np.ndarray, e: np.ndarray, mode: str
) -> np.ndarray:
    """``E_ij`` for a ``(n_members, n_pairs)`` block of decoy energies.

    One row at a time because the background sums are per member: a decoy's ``B_i`` is the
    sum over that decoy's own pair energies, not over the ensemble.
    """
    out = np.empty(e.shape, dtype=np.float64)
    for k in range(e.shape[0]):
        out[k] = many_body_energies(node_i, node_j, e[k], mode)
    return out


def _decoy_matrices(
    system: SystemDir, pair_ids: np.ndarray, settings: Settings
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(e_direct, e_fa_rep)`` as ``(n_members, n_pairs)``, columns ordered like ``pair_ids``."""
    decoys = system.read_decoy_energies(
        n_decoys=settings.analysis.n_decoys, axes=settings.analysis.axes
    )
    if decoys.empty:
        raise AnalysisError(
            "no decoy energies to analyse"
            + (
                f" for axis {list(settings.analysis.axes)}"
                if settings.analysis.axes
                else ""
            )
            + (
                f" within the first {settings.analysis.n_decoys} decoys"
                if settings.analysis.n_decoys is not None
                else ""
            )
        )

    # A member is (axis, decoy_id): decoy ids restart per axis, so pooling on decoy_id
    # alone would collide two different structures' energies into one column.
    decoys = decoys.assign(axis=decoys["axis"].astype(str))
    wide = {
        column: decoys.pivot(index=["axis", "decoy_id"], columns="pair_id", values=column)
        .sort_index()
        .reindex(columns=pair_ids)
        for column in ("e_direct", "e_fa_rep")
    }
    missing = np.flatnonzero(wide["e_direct"].isna().any(axis=0).to_numpy())
    if missing.size:
        raise AnalysisError(
            f"{missing.size} selected pair(s) are absent from the decoy store, e.g. "
            f"pair_id {pair_ids[missing[:5]].tolist()}; the ensemble is incomplete"
        )

    n_members = int(wide["e_direct"].shape[0])
    if n_members < 2:
        raise AnalysisError(
            f"{n_members} decoy(s) available; at least 2 are needed to estimate a spread"
        )
    return (
        wide["e_direct"].to_numpy(dtype=np.float64),
        wide["e_fa_rep"].to_numpy(dtype=np.float64),
        n_members,
    )


def _compute(system: SystemDir, settings: Settings) -> _SystemAnalysis:
    """Everything up to classification, for one system. Reads five files, calls no Rosetta."""
    nodes, superset = system.read_graph()
    n_pairs_superset = len(superset)

    # Raises when the cutoff exceeds the stored superset; `run` turns that into exit 3.
    pairs = add_contact_definition(
        superset, _DEFINITION, settings.contacts, settings.graph.superset
    )
    pairs = pairs[pairs[f"in__{_DEFINITION}"].to_numpy()].reset_index(drop=True)
    if pairs.empty:
        raise AnalysisError(
            f"no pair of the {n_pairs_superset} in the superset satisfies "
            f"{settings.contacts.definition} <= {settings.contacts.cutoff_A} A "
            f"(ligand {settings.contacts.ligand_cutoff_A} A, "
            f"seq_sep_min {settings.contacts.seq_sep_min})"
        )

    native = system.read_native_energies()
    merged = pairs.merge(native[["pair_id", "e_direct", "e_fa_rep"]], on="pair_id", how="left")
    absent = merged["e_direct"].isna().to_numpy()
    if absent.any():
        raise AnalysisError(
            f"{int(absent.sum())} selected pair(s) have no native energy, e.g. pair_id "
            f"{merged.loc[absent, 'pair_id'].head(5).tolist()}"
        )

    pair_ids = pairs["pair_id"].to_numpy(dtype=np.int32)
    decoy_direct, decoy_fa_rep, n_members = _decoy_matrices(system, pair_ids, settings)

    exclude = settings.energy.exclude_fa_rep
    mode = settings.manybody.mode
    code_i, code_j = _node_codes(pairs)

    e_native = effective_energy(
        merged["e_direct"].to_numpy(), merged["e_fa_rep"].to_numpy(), exclude
    )
    E_native = many_body_energies(code_i, code_j, e_native, mode)
    E_decoys = _many_body_block(
        code_i, code_j, effective_energy(decoy_direct, decoy_fa_rep, exclude), mode
    )

    F = compute_index(E_native, E_decoys, settings.analysis.index)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "pair_id": pair_ids,
                    "i": pairs["i"].to_numpy(dtype=np.int32),
                    "j": pairs["j"].to_numpy(dtype=np.int32),
                    "E_native": E_native,
                }
            ),
            decoy_summary(E_decoys),
            pd.DataFrame({"F": F, "index_name": settings.analysis.index}),
            normality_diagnostics(E_decoys)[["shapiro_W", "shapiro_p"]],
        ],
        axis=1,
    )
    return _SystemAnalysis(
        system_id=system.system_id,
        nodes=nodes,
        pairs=pairs,
        frame=frame,
        F=F,
        n_members=n_members,
        n_pairs_superset=n_pairs_superset,
    )


# --------------------------------------------------------------------------- pocket


def _pocket(analysis: _SystemAnalysis, settings: Settings) -> tuple[np.ndarray, list[str]]:
    """The mask descriptors are evaluated over, plus anything the caller should know.

    Selection is column arithmetic over distances the graph already stored, which is why a
    shell can be re-chosen against a finished run at all.
    """
    pocket = settings.pocket
    warnings: list[str] = []

    if pocket.mode == "whole":
        return pocket_mask(analysis.pairs, analysis.nodes, "all"), warnings
    if pocket.mode == "chain_interface":
        return pocket_mask(analysis.pairs, analysis.nodes, "inter_chain"), warnings
    if pocket.mode == "residue_list":
        raise AnalysisError(
            "pocket.mode 'residue_list' needs a residue list, which Settings does not "
            "carry; use 'ligand_shell', 'chain_interface' or 'whole'"
        )
    if pocket.mode != "ligand_shell":
        raise AnalysisError(f"unknown pocket mode {pocket.mode!r}")

    try:
        reference = _SHELL_REFERENCE[pocket.reference]
    except KeyError:
        raise AnalysisError(
            f"shell reference {pocket.reference!r} has no stored distance column; the "
            f"graph writes d_ca, d_cb and d_heavy_min, so only "
            f"{sorted(_SHELL_REFERENCE)} can be re-selected without regeneration"
        ) from None

    lining = shell_nodes(
        analysis.pairs, analysis.nodes, shell_A=pocket.cutoff_A, reference=reference
    )
    if not lining:
        warnings.append(
            f"no node lies within {pocket.cutoff_A} A ({pocket.reference}) of a ligand, "
            "cofactor or metal node; every pocket descriptor is unmeasured"
        )
    return (
        pocket_mask(analysis.pairs, analysis.nodes, "incident_to", node_ids=lining),
        warnings,
    )


def _shell_label(settings: Settings) -> str:
    """Names the pocket in every descriptor column, so two shells cannot be joined as one."""
    if settings.pocket.mode in ("whole", "chain_interface"):
        return settings.pocket.mode
    return f"{settings.pocket.mode}-{settings.pocket.reference}-{settings.pocket.cutoff_A:g}A"


# ---------------------------------------------------------------------------- output


def _jsonable(value: Any) -> Any:
    """NaN and ±inf are not JSON. An unmeasured covariate is written ``null``, not dropped."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_system(
    analysis: _SystemAnalysis,
    settings: Settings,
    thresholds: ClassifySettings,
    directory: Path,
    manifest_common: dict[str, Any],
    requested_descriptors: list[str],
) -> dict[str, Any]:
    labels = classify_index(analysis.F, thresholds)
    frame = analysis.frame.assign(**{"class": labels})[list(CONTACTS_COLUMNS)]

    mask, warnings = _pocket(analysis, settings)
    shell_label = _shell_label(settings)
    row = summarize(
        analysis.pairs.assign(E_native=analysis.frame["E_native"].to_numpy()),
        analysis.nodes,
        analysis.F,
        labels,
        mask,
        index_name=settings.analysis.index,
        shell_label=shell_label,
    )

    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / CONTACTS_FILENAME, index=False)

    summary = {
        "system_id": analysis.system_id,
        "index": settings.analysis.index,
        "manybody": settings.manybody.mode,
        "exclude_fa_rep": settings.energy.exclude_fa_rep,
        "contact_definition": settings.contacts.definition,
        "contact_cutoff_A": settings.contacts.cutoff_A,
        "shell_label": shell_label,
        "thresholds": thresholds.model_dump(mode="json"),
        "n_decoys_used": analysis.n_members,
        "n_pairs_superset": analysis.n_pairs_superset,
        "n_pairs_selected": int(len(analysis.pairs)),
        "class_counts": class_counts(labels),
        "descriptors": {k: v for k, v in row.items() if k.startswith("desc__")},
        "covariates": {k: row[k] for k in COVARIATES},
        "requested_descriptors": requested_descriptors,
        "warnings": warnings,
        "provisional": manifest_common["provisional"],
    }
    _write_json(directory / SUMMARY_FILENAME, summary)
    _write_json(
        directory / MANIFEST_FILENAME,
        {
            **manifest_common,
            "system_id": analysis.system_id,
            "thresholds_resolved": thresholds.model_dump(mode="json"),
            "n_decoys_used": analysis.n_members,
            "n_pairs_selected": int(len(analysis.pairs)),
        },
    )
    return summary


def _print_summary(summary: dict[str, Any], directory: Path, stream: Any) -> None:
    counts = summary["class_counts"]
    print(f"{summary['system_id']}  ->  {directory}", file=stream)
    print(
        f"  contacts  {summary['n_pairs_selected']} selected of "
        f"{summary['n_pairs_superset']} in the superset   "
        f"decoys used: {summary['n_decoys_used']}",
        file=stream,
    )
    print(
        f"  classes   minimal {counts['minimally_frustrated']}   "
        f"neutral {counts['neutral']}   highly {counts['highly_frustrated']}",
        file=stream,
    )
    print(
        f"  pocket    {summary['shell_label']}   "
        f"n_contacts_total={summary['covariates']['n_contacts_total']}   "
        f"n_pocket_residues={summary['covariates']['n_pocket_residues']}",
        file=stream,
    )
    for warning in summary["warnings"]:
        print(f"  warning: {warning}", file=stream)


# ------------------------------------------------------------------------------- run


def run(args: argparse.Namespace) -> int:
    run_dir = RunDir(args.run_dir)
    if not run_dir.manifest_path.exists():
        print(
            f"atomfrust analyze: {args.run_dir} is not a run directory "
            f"({run_dir.manifest_path.name} is missing)",
            file=sys.stderr,
        )
        return EXIT_ANALYSIS_ERROR

    try:
        settings = _resolve_settings(run_dir, args)
        manifest = run_dir.read_manifest()
        directory_name = _analysis_dirname(args.out, settings)
    except (ValueError, OSError, ManifestVersionError, yaml.YAMLError) as exc:
        print(f"atomfrust analyze: {exc}", file=sys.stderr)
        return EXIT_USAGE if isinstance(exc, ValueError) else EXIT_ANALYSIS_ERROR

    # pyrosetta_version=None on purpose: reading it would mean importing PyRosetta to do a
    # job that requires none. The stored version stays in the manifest either way.
    try:
        report = run_dir.assert_compatible(
            settings, pyrosetta_version=None, allow_mismatch=args.allow_mismatch
        )
    except RegenerationRequired as exc:
        print(f"atomfrust analyze: {exc}", file=sys.stderr)
        return exc.exit_code

    provisional = report.verdict == "requires_regeneration"
    if provisional:
        print(
            "atomfrust analyze: --allow-mismatch given; every output is marked provisional",
            file=sys.stderr,
        )

    requested = _split_list([args.systems]) if args.systems else run_dir.systems()
    if not requested:
        print(f"atomfrust analyze: {args.run_dir} contains no systems", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    unknown = sorted(set(requested) - set(run_dir.systems()))
    if unknown:
        print(
            f"atomfrust analyze: no such system(s) {unknown}; present: {run_dir.systems()}",
            file=sys.stderr,
        )
        return EXIT_USAGE

    computed: list[_SystemAnalysis] = []
    failures = 0
    for system_id in requested:
        try:
            computed.append(_compute(run_dir.system(system_id), settings))
        except ValueError as exc:
            # add_contact_definition refuses a cutoff the superset cannot serve. Same class
            # of failure as a generation-stage diff: no stored bytes can answer it.
            print(f"atomfrust analyze: {system_id}: {exc}", file=sys.stderr)
            return EXIT_REQUIRES_REGENERATION
        except (AnalysisError, FileNotFoundError, KeyError) as exc:
            print(f"atomfrust analyze: {system_id}: {exc}", file=sys.stderr)
            failures += 1
    if not computed:
        return EXIT_ANALYSIS_ERROR

    thresholds = settings.analysis.classify
    if thresholds.mode == "quantile":
        # Calibrate on the pooled distribution, never per system: per-system quantiles would
        # force every structure to the same class fractions and destroy the between-system
        # signal (see analyze/classify.py).
        thresholds = quantile_thresholds(
            np.concatenate([a.F for a in computed]) if computed else np.empty(0)
        )

    manifest_common = {
        "analysis_id": directory_name,
        "settings_digest": analysis_id(settings),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contract_version": CONTRACT_VERSION,
        "run_dir": str(Path(args.run_dir).resolve()),
        "run_id": manifest.run_id,
        # What these settings were computed against. A run whose inputs or generation
        # settings later change gets a different key, so a stale analysis is detectable
        # without re-deriving anything.
        "regeneration_key": manifest.regeneration_key,
        "compatibility": {
            "verdict": report.verdict,
            "differences": [
                {"path": d.path, "stored": d.stored, "requested": d.requested, "stage": d.stage}
                for d in report.differences
            ],
        },
        "provisional": provisional,
        "settings": stage_subset(settings, "analysis"),
        "settings_resolved": settings.model_dump(mode="json"),
    }

    requested_descriptors = list(settings.analysis.descriptors)
    for analysis in computed:
        directory = run_dir.system(analysis.system_id).analyses / directory_name
        try:
            summary = _write_system(
                analysis, settings, thresholds, directory, manifest_common,
                requested_descriptors,
            )
        except (AnalysisError, ValueError, OSError) as exc:
            print(f"atomfrust analyze: {analysis.system_id}: {exc}", file=sys.stderr)
            failures += 1
            continue
        _print_summary(summary, directory, sys.stdout)

    return EXIT_ANALYSIS_ERROR if failures else 0
