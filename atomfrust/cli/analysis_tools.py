"""``converge``, ``strata``, ``report`` and ``calibrate`` — plan step E6.

Four commands, one module, because they are the same kind of thing: **queries over runs
that already exist**. None of them generates a decoy, loads a pose or imports PyRosetta;
each reads stored direct pair energies (or stored analysis summaries) and computes. That is
the whole reason plan §4 stores ``e_direct``/``e_fa_rep`` per pair instead of ``E_ij`` — the
many-body formula, the contact definition, the shell, the index and the decoy count are all
chosen here, after the expensive part is over.

``converge`` (R29)
    How many decoys are enough. Subsamples one stored ensemble rather than executing eight
    runs: decoy *i* is seeded ``base_seed + i``, so ``decoy_id < N`` **is** the ensemble an
    N-decoy run would have produced, exactly. Grid points beyond what the run stores are
    skipped and named, because a stored run is routinely shorter than the full sweep and
    that is not an error.

``strata`` (R30, R31)
    Whether the decoy σ — the quantity the index divides by — is the same size in deeply
    buried, polar and large pockets as in shallow, apolar and small ones. If it is not, a
    Z-score looks target-agnostic by construction while still carrying a target-dependent
    scale, and pooling across targets is invalid however well-centred the scores are. The
    axis-redundancy table beside it is R31.

``report`` (D7, plan §2.3)
    Collects per-system summaries into one confound-aware report. Every correlation is a
    triple (raw, partial controlling the pocket-size covariate, OLS coefficient with its
    VIF) and the multiplicity correction is a max-T permutation over the descriptor grid,
    not Benjamini–Hochberg. :func:`~atomfrust.report.collect.render_report` refuses to print
    a headline for a descriptor whose raw CI excludes zero while its partial CI does not,
    and this command cannot switch that off.

``calibrate``
    Definition-specific class thresholds. **Calibration is pooled across every system in the
    cohort and there is no per-system mode**, deliberately — see :func:`run_calibrate`.

Every number here comes from an existing module; nothing statistical is reimplemented.
``E_ij`` reconstruction is *imported* from :mod:`atomfrust.cli.analyze` rather than
re-derived, so a convergence curve and an analysis of the same run cannot drift apart: the
private names below are that module's own reconstruction, used verbatim on purpose.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from atomfrust.analyze.aggregate import LIGAND_KINDS, pocket_mask, shell_nodes
from atomfrust.analyze.classify import quantile_thresholds
from atomfrust.analyze.converge import DEFAULT_GRID, convergence_curve, n_star
from atomfrust.analyze.strata import (
    STRATUM_AXES,
    assign_strata,
    axis_redundancy,
    pocket_descriptors,
    sigma_by_stratum,
)
from atomfrust.analyze.zscore import compute_index, decoy_summary
from atomfrust.energy import effective_energy, many_body_energies
from atomfrust.graph import add_contact_definition

# The E_ij reconstruction of `atomfrust analyze`, imported rather than copied. These are
# private to that module, and that is exactly the point: there must be one implementation of
# "stored direct energies -> E_ij", or `converge` would answer a question about a slightly
# different quantity than the one `analyze` reports.
from atomfrust.cli.analyze import (
    _DEFINITION,
    _SHELL_REFERENCE,
    AnalysisError,
    _decoy_matrices,
    _many_body_block,
    _node_codes,
)
from atomfrust.provenance import CONTRACT_VERSION
from atomfrust.report import collect_analyses, headline_is_permitted, render_report
from atomfrust.runstore import RunDir, SystemDir
from atomfrust.settings import ClassifySettings, Settings, load_settings

__all__ = [
    "NAME",
    "HELP",
    "STRATA_NAME",
    "REPORT_NAME",
    "CALIBRATE_NAME",
    "register",
    "run",
    "run_strata",
    "run_report",
    "run_calibrate",
    "EXIT_ERROR",
    "EXIT_USAGE",
]

NAME = "converge"
HELP = "decoy-count convergence sweep by subsampling one stored run"

STRATA_NAME = "strata"
STRATA_HELP = "decoy sigma across architectural strata, and decoy-axis redundancy"

REPORT_NAME = "report"
REPORT_HELP = "collect analyses into a report with covariate-aware statistics"

CALIBRATE_NAME = "calibrate"
CALIBRATE_HELP = "definition-specific class thresholds, calibrated on the pooled cohort"

#: A request the stored runs cannot answer.
EXIT_ERROR = 1
#: Malformed arguments argparse cannot reject on its own.
EXIT_USAGE = 2

#: R29's acceptance bar: the smallest N whose per-pair ranking agrees with the reference at
#: Spearman rho >= this is N*.
DEFAULT_RHO_THRESHOLD = 0.95


class UsageError(ValueError):
    """Bad arguments. Reported as a message, never as a traceback."""


# --------------------------------------------------------------------------- registration


def _add_index_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index",
        choices=("zscore", "rank_percentile", "robust_z"),
        help="index function; default is the run's own analysis setting",
    )


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register all four E6 commands. One module, four parsers (see :mod:`.verify`)."""
    _register_converge(subparsers)
    _register_strata(subparsers)
    _register_report(subparsers)
    _register_calibrate(subparsers)


def _register_converge(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        NAME,
        help=HELP,
        description=(
            "Decoy-count convergence for a finished run. Subsamples the stored ensemble "
            "instead of re-running: decoy i is seeded base_seed + i, so decoy_id < N is "
            "exactly the ensemble an N-decoy run would have produced. Makes no PyRosetta "
            "call."
        ),
        epilog=(
            "examples:\n"
            "  atomfrust converge --run runs/egfr\n"
            "  atomfrust converge --run runs/egfr --grid 10,25,50,100 --n-boot 200\n"
            "\n"
            "Grid points larger than the stored decoy count are skipped and listed, not\n"
            "treated as errors: a stored run is routinely shorter than the full sweep.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", "--run-dir", dest="run", type=Path, required=True, metavar="R",
                        help="finished run directory")
    parser.add_argument(
        "--grid",
        metavar="N[,N...]",
        default=",".join(str(n) for n in DEFAULT_GRID),
        help=f"decoy counts to evaluate (default {','.join(str(n) for n in DEFAULT_GRID)})",
    )
    parser.add_argument(
        "--n-boot", type=int, default=1000, metavar="N",
        help="bootstrap resamples for the rho confidence interval (default 1000; 0 skips)",
    )
    _add_index_flag(parser)
    parser.add_argument(
        "--axis", metavar="A",
        help="decoy axis to sweep; required only when the run stores more than one, "
             "because a prefix is only a prefix within one axis",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_RHO_THRESHOLD, metavar="RHO",
        help=f"rho at which N* is declared reached (default {DEFAULT_RHO_THRESHOLD})",
    )
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    parser.add_argument(
        "--systems", metavar="ID[,ID]", help="systems to sweep; default is all of them"
    )
    parser.add_argument(
        "-o", "--out", type=Path, metavar="DIR",
        help="output directory (default <run>/converge)",
    )
    parser.set_defaults(func=run)


def _register_strata(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        STRATA_NAME,
        help=STRATA_HELP,
        description=(
            "Decoy sigma across architectural strata (burial, polarity, volume) over a "
            "cohort of finished systems, plus the pairwise redundancy of the decoy axes. "
            "The index divides by sigma, so a sigma that moves with pocket architecture "
            "makes a Z-score target-dependent even when it is perfectly centred."
        ),
        epilog=(
            "examples:\n"
            "  atomfrust strata --runs 'runs/*/systems/*' -o strata/\n"
            "  atomfrust strata --runs 'runs/egfr' --by burial,volume -o strata/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", action="append", required=True, metavar="GLOB",
        help="glob matching run directories or systems/<id> directories (repeatable). "
             "Quote it, or the shell expands it first.",
    )
    parser.add_argument(
        "--by", default=",".join(STRATUM_AXES), metavar="AXIS[,AXIS]",
        help=f"stratification axes to report (default {','.join(STRATUM_AXES)})",
    )
    parser.add_argument(
        "--n-strata", type=int, default=3, metavar="K",
        help="quantile bins per axis (default 3); collapses when there are fewer pockets",
    )
    _add_index_flag(parser)
    parser.add_argument(
        "--n-decoys", type=int, metavar="N", help="use decoy_id < N (default: all stored)"
    )
    parser.add_argument("-o", "--out", type=Path, required=True, metavar="DIR",
                        help="output directory")
    parser.set_defaults(func=run_strata)


def _register_report(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        REPORT_NAME,
        help=REPORT_HELP,
        description=(
            "Collect per-system analysis summaries into one report. Every correlation is "
            "reported as a triple - raw, partial controlling the covariate, and the OLS "
            "coefficient with its VIF - and the multiplicity correction is a max-T "
            "permutation over the descriptor grid, not Benjamini-Hochberg, because a "
            "reported best case is a maximum over a swept grid and BH does not price a "
            "maximum. A descriptor whose raw CI excludes zero while its partial CI does "
            "not gets the covariate warning where its headline would have been; there is "
            "no flag that turns that off."
        ),
        epilog=(
            "examples:\n"
            "  atomfrust report --collect 'runs/*/systems/*/analyses/default' -o report/\n"
            "  atomfrust report --collect results/egfr_frustration_summary.csv -o report/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--collect", action="append", required=True, metavar="GLOB",
        help="glob matching analysis directories, summary.json files, run directories or "
             "a summary CSV (repeatable). Quote it, or the shell expands it first.",
    )
    parser.add_argument("-o", "--out", type=Path, required=True, metavar="DIR",
                        help="output directory")
    parser.add_argument(
        "--permute", type=int, default=10000, metavar="N",
        help="permutations for the max-T adjustment (default 10000)",
    )
    parser.add_argument(
        "--outcome", default="log10_affinity_pM", metavar="COLUMN",
        help="outcome column (default log10_affinity_pM)",
    )
    parser.add_argument(
        "--covariate", default="n_contacts_total", metavar="COLUMN",
        help="covariate every correlation is controlled for (default n_contacts_total, "
             "the pocket contact count a class count is a degree statistic of)",
    )
    parser.add_argument(
        "--descriptor", action="append", default=[], metavar="D",
        help="descriptors to report (repeatable or comma-separated). Default is every "
             "descriptor column present - a subset chosen after seeing the numbers is "
             "precisely the selection the max-T adjustment exists to price.",
    )
    parser.add_argument("--n-boot", type=int, default=10000, metavar="N",
                        help="bootstrap resamples per interval (default 10000)")
    parser.add_argument("--seed", type=int, default=0, help="seed (default 0)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="adjusted-p level a headline must clear (default 0.05)")
    parser.add_argument("--no-plots", dest="plots", action="store_false",
                        help="skip the figures (they need matplotlib)")
    parser.add_argument("--title", default="Frustration-affinity report")
    parser.set_defaults(func=run_report, plots=True)


def _register_calibrate(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        CALIBRATE_NAME,
        help=CALIBRATE_HELP,
        description=(
            "Re-derive the frustration class cutoffs from the observed index distribution "
            "of a cohort. The published 0.78 / -1.0 encode 'top decile / bottom decile' of "
            "one particular null - one contact definition, one decoy construction, one "
            "energy function - so under a swept setting they no longer mark deciles and "
            "the reported class counts move for a purely definitional reason. "
            "CALIBRATION IS POOLED ACROSS EVERY SYSTEM IN THE COHORT, AND THERE IS NO "
            "PER-SYSTEM MODE: per-system quantiles would force every structure to the same "
            "class fractions by construction and destroy the between-structure signal, "
            "which is the entire quantity of interest. --per-system exists only to be "
            "refused, with that explanation."
        ),
        epilog=(
            "examples:\n"
            "  atomfrust calibrate --runs 'runs/*/systems/*' -o calibration/\n"
            "  atomfrust calibrate --runs 'runs/egfr' --definition heavy_min "
            "--cutoff-A 6.0 -o calibration/\n"
            "\n"
            "thresholds.yaml is a settings overlay: pass it to\n"
            "  atomfrust analyze --run-dir R --config calibration/thresholds.yaml\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", action="append", required=True, metavar="GLOB",
        help="glob matching run directories or systems/<id> directories (repeatable). "
             "Every matched system is pooled into one distribution.",
    )
    parser.add_argument(
        "--definition", choices=("ca_ca", "cb_cb", "heavy_min"),
        help="contact definition to calibrate for; default is the run's own",
    )
    parser.add_argument(
        "--cutoff-A", type=float, metavar="X",
        help="cutoff for that definition (must lie within the stored superset)",
    )
    _add_index_flag(parser)
    parser.add_argument(
        "--manybody", choices=("pair_retained", "chen_literal", "pair_only"),
        help="many-body formula; default is the run's own",
    )
    parser.add_argument("--minimal-q", type=float, default=0.9, metavar="Q",
                        help="upper-tail quantile for 'minimally frustrated' (default 0.9)")
    parser.add_argument("--highly-q", type=float, default=0.1, metavar="Q",
                        help="lower-tail quantile for 'highly frustrated' (default 0.1)")
    parser.add_argument(
        "--n-decoys", type=int, metavar="N", help="use decoy_id < N (default: all stored)"
    )
    parser.add_argument(
        "--per-system", action="store_true",
        help="REFUSED. Calibrating each system on its own distribution forces identical "
             "class fractions across structures by construction and destroys the "
             "between-structure signal. Passing this exits 2 with that explanation.",
    )
    parser.add_argument("-o", "--out", type=Path, required=True, metavar="DIR",
                        help="output directory")
    parser.set_defaults(func=run_calibrate)


# ------------------------------------------------------------------------------ helpers


def _split_list(values: Iterable[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [v.strip() for value in values for v in str(value).split(",") if v.strip()]


def _parse_grid(text: str) -> list[int]:
    try:
        grid = sorted({int(v) for v in _split_list(text)})
    except ValueError:
        raise UsageError(f"--grid expects comma-separated integers, got {text!r}") from None
    if not grid:
        raise UsageError("--grid is empty")
    return grid


def _with_overlay(settings: Settings, overlay: dict[str, Any]) -> Settings:
    """Layer a CLI overlay onto a run's own resolved settings.

    The base is the *run's* settings, exactly as ``atomfrust analyze`` does, so a field this
    process merely defaulted differently cannot masquerade as a requested change.
    """
    return load_settings(
        file_overlay=settings.model_dump(mode="json"), cli_overlay=overlay
    )[0]


def _put(overlay: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path, skipping ``None`` so an unspecified flag inherits the run's value."""
    if value is None:
        return
    node = overlay
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _jsonable(value: Any) -> Any:
    """NaN and +-inf are not JSON. An unmeasured quantity is written ``null``, not dropped."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _stamp(command: str, **extra: Any) -> dict[str, Any]:
    return {
        "command": command,
        "contract_version": CONTRACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


# ------------------------------------------------------------------- system resolution


def _system_from_path(path: Path) -> list[SystemDir]:
    """Every system a matched path stands for.

    A run root, a ``systems/`` directory, one ``systems/<id>`` and anything beneath it (an
    analysis directory, for instance) are all things a caller reasonably globs at, so each
    resolves by the layout of plan §4 rather than by the caller having to know which level
    they landed on.
    """
    path = Path(path).resolve()
    if (path / "manifest.json").is_file():
        run = RunDir(path)
        return [run.system(system_id) for system_id in run.systems()]
    if path.name == "systems" and path.is_dir():
        return [SystemDir(path.parent, p.name) for p in sorted(path.iterdir()) if p.is_dir()]

    node = path
    while node != node.parent:
        if node.parent.name == "systems" and (node.parent.parent / "manifest.json").is_file():
            return [SystemDir(node.parent.parent, node.name)]
        node = node.parent
    raise UsageError(
        f"{path} is not a run directory and does not sit under one; expected a directory "
        "holding manifest.json, or a systems/<system_id> directory beneath it"
    )


def _resolve_systems(patterns: Sequence[str]) -> list[SystemDir]:
    """Expand globs into a de-duplicated, ordered list of systems."""
    found: list[SystemDir] = []
    seen: set[tuple[str, str]] = set()
    for pattern in patterns:
        matches = sorted(globlib.glob(str(pattern), recursive=True))
        if not matches:
            raise UsageError(f"no path matches {pattern!r}")
        for match in matches:
            for system in _system_from_path(Path(match)):
                key = (str(Path(system.root).resolve()), system.system_id)
                if key not in seen:
                    seen.add(key)
                    found.append(system)
    if not found:
        raise UsageError(f"no system found under {list(patterns)}")
    return found


def _axes_present(system: SystemDir) -> list[str]:
    """Decoy axes stored for one system, read from the ``axis`` column alone."""
    axes: set[str] = set()
    for path in system.decoy_parts():
        axes.update(pd.read_parquet(path, columns=["axis"])["axis"].astype(str).unique())
    return sorted(axes)


# --------------------------------------------------------------- E_ij reconstruction


@dataclass
class _Reconstruction:
    """One system's ``E_ij``, native and per decoy axis, over the selected contact set."""

    system: SystemDir
    nodes: pd.DataFrame
    pairs: pd.DataFrame
    E_native: np.ndarray
    E_decoys: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def system_id(self) -> str:
        return self.system.system_id

    @property
    def pooled(self) -> np.ndarray:
        """Every axis's decoys stacked — the ensemble ``analyze`` would have used."""
        return np.vstack(list(self.E_decoys.values()))


def _reconstruct(
    system: SystemDir,
    settings: Settings,
    axes: Sequence[str],
    n_decoys: int | None = None,
) -> _Reconstruction:
    """Stored direct energies → ``E_ij``, exactly as :mod:`atomfrust.cli.analyze` does.

    One block per axis rather than one pooled block, because ``converge`` needs a genuine
    ``decoy_id`` prefix (which only exists within an axis) and
    :func:`~atomfrust.analyze.strata.axis_redundancy` needs the axes kept apart. Pooling is
    :attr:`_Reconstruction.pooled`, one ``vstack`` away.
    """
    nodes, superset = system.read_graph()
    pairs = add_contact_definition(
        superset, _DEFINITION, settings.contacts, settings.graph.superset
    )
    pairs = pairs[pairs[f"in__{_DEFINITION}"].to_numpy()].reset_index(drop=True)
    if pairs.empty:
        raise AnalysisError(
            f"no pair of the {len(superset)} in the superset satisfies "
            f"{settings.contacts.definition} <= {settings.contacts.cutoff_A} A"
        )

    native = system.read_native_energies()
    merged = pairs.merge(
        native[["pair_id", "e_direct", "e_fa_rep"]], on="pair_id", how="left"
    )
    absent = merged["e_direct"].isna().to_numpy()
    if absent.any():
        raise AnalysisError(
            f"{int(absent.sum())} selected pair(s) have no native energy, e.g. pair_id "
            f"{merged.loc[absent, 'pair_id'].head(5).tolist()}"
        )

    exclude = settings.energy.exclude_fa_rep
    mode = settings.manybody.mode
    code_i, code_j = _node_codes(pairs)
    E_native = many_body_energies(
        code_i,
        code_j,
        effective_energy(
            merged["e_direct"].to_numpy(), merged["e_fa_rep"].to_numpy(), exclude
        ),
        mode,
    )

    pair_ids = pairs["pair_id"].to_numpy(dtype=np.int32)
    blocks: dict[str, np.ndarray] = {}
    for axis in axes:
        per_axis = _with_overlay(
            settings, {"analysis": {"axes": [axis], "n_decoys": n_decoys}}
        )
        direct, fa_rep, _ = _decoy_matrices(system, pair_ids, per_axis)
        blocks[axis] = _many_body_block(
            code_i, code_j, effective_energy(direct, fa_rep, exclude), mode
        )

    return _Reconstruction(
        system=system, nodes=nodes, pairs=pairs, E_native=E_native, E_decoys=blocks
    )


def _settings_of(system: SystemDir) -> Settings:
    return RunDir(system.root).read_settings()


def _resolve_axes(system: SystemDir, settings: Settings, requested: str | None) -> list[str]:
    """Which axes to read, in a stable order. Empty means the store holds no decoys."""
    present = _axes_present(system)
    wanted = _split_list(requested) or list(settings.analysis.axes or [])
    if not wanted:
        return present
    unknown = sorted(set(wanted) - set(present))
    if unknown:
        raise AnalysisError(
            f"no decoys on axis {unknown}; the store holds {present or 'nothing'}"
        )
    return [axis for axis in present if axis in set(wanted)]


# ------------------------------------------------------------------------- the pocket


def _pocket_selection(
    reconstruction: _Reconstruction, settings: Settings
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(pair_mask, node_mask, warnings)`` for the run's own pocket definition.

    Column arithmetic over distances the graph already stored — the same selection
    ``atomfrust analyze`` makes, which is why a shell can be re-chosen against a finished
    run with no pose in sight.
    """
    pairs, nodes = reconstruction.pairs, reconstruction.nodes
    pocket = settings.pocket
    warnings: list[str] = []
    node_ids = nodes["node_id"].astype(str)

    if pocket.mode == "whole":
        return (
            pocket_mask(pairs, nodes, "all"),
            np.ones(len(nodes), dtype=bool),
            warnings,
        )
    if pocket.mode == "chain_interface":
        mask = pocket_mask(pairs, nodes, "inter_chain")
        touched = set(pairs.loc[mask, "node_i"].astype(str)) | set(
            pairs.loc[mask, "node_j"].astype(str)
        )
        return mask, node_ids.isin(touched).to_numpy(), warnings
    if pocket.mode != "ligand_shell":
        raise AnalysisError(
            f"pocket mode {pocket.mode!r} is not selectable from stored tables; use "
            "'ligand_shell', 'chain_interface' or 'whole'"
        )

    try:
        reference = _SHELL_REFERENCE[pocket.reference]
    except KeyError:
        raise AnalysisError(
            f"shell reference {pocket.reference!r} has no stored distance column; the "
            f"graph writes d_ca, d_cb and d_heavy_min, so only "
            f"{sorted(_SHELL_REFERENCE)} can be re-selected without regeneration"
        ) from None

    lining = shell_nodes(pairs, nodes, shell_A=pocket.cutoff_A, reference=reference)
    if not lining:
        warnings.append(
            f"no node lies within {pocket.cutoff_A} A ({pocket.reference}) of a ligand, "
            "cofactor or metal node; every pocket descriptor is unmeasured"
        )
    pair_mask = pocket_mask(pairs, nodes, "incident_to", node_ids=lining)
    # The binder itself is a pocket *node* — pocket_descriptors counts it in
    # `pocket_heavy_atoms` but excludes it from burial/polarity/volume, which are
    # residue properties.
    binders = nodes["kind"].astype(str).isin(LIGAND_KINDS).to_numpy()
    node_mask = node_ids.isin(set(lining)).to_numpy() | binders
    return pair_mask, node_mask, warnings


# ------------------------------------------------------------------------- converge


def _converge_one(
    system: SystemDir,
    settings: Settings,
    grid: Sequence[int],
    index: str,
    n_boot: int,
    seed: int,
    threshold: float,
    axis: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    axes = _resolve_axes(system, settings, axis)
    if not axes:
        raise AnalysisError("no decoy energies are stored for this system")
    if len(axes) > 1:
        raise AnalysisError(
            f"the store holds {len(axes)} decoy axes {axes}; a decoy_id prefix is only a "
            "prefix within one axis, so name one with --axis"
        )

    reconstruction = _reconstruct(system, settings, axes, n_decoys=max(grid))
    decoys = reconstruction.E_decoys[axes[0]]
    curve = convergence_curve(
        reconstruction.E_native,
        decoys,
        grid=grid,
        index=index,
        n_boot=n_boot,
        seed=seed,
    )
    summary = {
        "system_id": system.system_id,
        "axis": axes[0],
        "n_star": n_star(curve, threshold),
        "threshold": threshold,
        "n_pairs": int(reconstruction.E_native.size),
        **{k: curve.attrs[k] for k in ("reference_n", "n_decoys_available", "skipped_grid_points")},
    }
    return curve.assign(system_id=system.system_id, axis=axes[0]), summary


def run(args: argparse.Namespace) -> int:
    """``atomfrust converge`` — R29, by subsampling rather than by re-running."""
    run_dir = RunDir(args.run)
    if not run_dir.manifest_path.exists():
        print(
            f"atomfrust converge: {args.run} is not a run directory "
            f"({run_dir.manifest_path.name} is missing)",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        grid = _parse_grid(args.grid)
        settings = run_dir.read_settings()
        requested = _split_list(args.systems) or run_dir.systems()
        unknown = sorted(set(requested) - set(run_dir.systems()))
        if unknown:
            raise UsageError(
                f"no such system(s) {unknown}; present: {run_dir.systems()}"
            )
        if not requested:
            raise UsageError(f"{args.run} contains no systems")
    except (UsageError, OSError, ValueError) as exc:
        print(f"atomfrust converge: {exc}", file=sys.stderr)
        return EXIT_USAGE if isinstance(exc, (UsageError, ValueError)) else EXIT_ERROR

    index = args.index or settings.analysis.index
    out_dir = Path(args.out) if args.out else run_dir.root / "converge"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failures = 0
    for system_id in requested:
        try:
            curve, summary = _converge_one(
                run_dir.system(system_id),
                settings,
                grid,
                index,
                int(args.n_boot),
                int(args.seed),
                float(args.threshold),
                args.axis,
            )
        except (AnalysisError, FileNotFoundError, KeyError, ValueError) as exc:
            print(f"atomfrust converge: {system_id}: {exc}", file=sys.stderr)
            failures += 1
            continue
        curves.append(curve)
        summaries.append(summary)

    if not curves:
        return EXIT_ERROR

    table = pd.concat(curves, ignore_index=True)
    table = table[["system_id", "axis"] + [c for c in table.columns if c not in ("system_id", "axis")]]
    table.to_csv(out_dir / "convergence.csv", index=False)
    _write_json(
        out_dir / "convergence.json",
        _stamp(
            NAME,
            run_dir=str(Path(args.run).resolve()),
            index=index,
            grid=grid,
            n_boot=int(args.n_boot),
            seed=int(args.seed),
            threshold=float(args.threshold),
            systems=summaries,
        ),
    )

    print(f"convergence -> {out_dir}")
    for summary in summaries:
        reached = summary["n_star"]
        skipped = summary["skipped_grid_points"]
        print(
            f"  {summary['system_id']}  axis={summary['axis']}  "
            f"pairs={summary['n_pairs']}  stored={summary['n_decoys_available']}  "
            f"reference N={summary['reference_n']}  "
            f"N*={reached if reached is not None else 'not reached'}"
            + (f"  skipped {skipped}" if skipped else "")
        )
    return EXIT_ERROR if failures else 0


# --------------------------------------------------------------------------- strata


def run_strata(args: argparse.Namespace) -> int:
    """``atomfrust strata`` — R30 (sigma by architecture) and R31 (axis redundancy)."""
    try:
        by = _split_list(args.by)
        unknown = sorted(set(by) - set(STRATUM_AXES))
        if unknown:
            raise UsageError(
                f"unknown stratification axis {unknown}; available: {list(STRATUM_AXES)}"
            )
        systems = _resolve_systems(args.runs)
    except UsageError as exc:
        print(f"atomfrust strata: {exc}", file=sys.stderr)
        return EXIT_USAGE

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    redundancy: list[pd.DataFrame] = []
    warnings: list[str] = []
    failures = 0
    index = args.index

    for system in systems:
        try:
            settings = _settings_of(system)
            axes = _resolve_axes(system, settings, None)
            if not axes:
                raise AnalysisError("no decoy energies are stored for this system")
            reconstruction = _reconstruct(system, settings, axes, n_decoys=args.n_decoys)
            pair_mask, node_mask, notes = _pocket_selection(reconstruction, settings)
        except (AnalysisError, FileNotFoundError, KeyError, ValueError) as exc:
            print(f"atomfrust strata: {system.system_id}: {exc}", file=sys.stderr)
            failures += 1
            continue

        warnings.extend(f"{system.system_id}: {note}" for note in notes)
        descriptors = pocket_descriptors(
            reconstruction.nodes, reconstruction.pairs, node_mask
        )
        sigma = decoy_summary(reconstruction.pooled)["decoy_std"].to_numpy()
        pocket_sigma = sigma[pair_mask]
        pocket_sigma = pocket_sigma[np.isfinite(pocket_sigma)]
        rows.append(
            {
                "system_id": system.system_id,
                "run_dir": str(Path(system.root).resolve()),
                "n_pocket_contacts": int(pair_mask.sum()),
                "median_sigma": float(np.median(pocket_sigma)) if pocket_sigma.size else float("nan"),
                "mean_sigma_pocket": float(np.mean(pocket_sigma)) if pocket_sigma.size else float("nan"),
                "n_decoys_used": int(reconstruction.pooled.shape[0]),
                "axes": ",".join(axes),
                **descriptors,
            }
        )

        # Redundancy is per system: the arrays must share one pair ordering, and pair_id is
        # only dense within a system (plan §4).
        per_axis = {
            axis: compute_index(
                reconstruction.E_native, block, index=index or settings.analysis.index
            )
            for axis, block in reconstruction.E_decoys.items()
        }
        redundancy.append(
            axis_redundancy(per_axis).assign(system_id=system.system_id)
        )

    if not rows:
        return EXIT_ERROR

    descriptors_table = assign_strata(pd.DataFrame(rows), n_strata=int(args.n_strata))
    descriptors_table.to_csv(out_dir / "pocket_descriptors.csv", index=False)

    sigma_table = sigma_by_stratum(descriptors_table, sigma_column="median_sigma")
    sigma_table = sigma_table[sigma_table["axis"].isin(by)].reset_index(drop=True)
    sigma_table.to_csv(out_dir / "sigma_by_stratum.csv", index=False)

    redundancy_table = pd.concat(redundancy, ignore_index=True)
    redundancy_table.to_csv(out_dir / "axis_redundancy.csv", index=False)

    cv_across = {
        axis: float(sigma_table.loc[sigma_table["axis"] == axis, "cv_across_strata"].iloc[0])
        for axis in by
        if (sigma_table["axis"] == axis).any()
    }
    _write_json(
        out_dir / "strata.json",
        _stamp(
            STRATA_NAME,
            axes=by,
            n_strata=int(args.n_strata),
            n_systems=len(rows),
            burial_sources=sorted({r["burial_source"] for r in rows}),
            cv_across_strata=cv_across,
            warnings=warnings,
        ),
    )

    print(f"strata -> {out_dir}   {len(rows)} pocket(s)")
    for axis in by:
        value = cv_across.get(axis, float("nan"))
        print(f"  {axis:<9} CV of sigma across strata = {value:.4g}")
    redundant = redundancy_table[redundancy_table["exceeds_threshold"].fillna(False)]
    if len(redundant):
        print(
            f"  {len(redundant)} axis pair(s) exceed the redundancy threshold; see "
            "axis_redundancy.csv"
        )
    for note in warnings:
        print(f"  warning: {note}")
    return EXIT_ERROR if failures else 0


# --------------------------------------------------------------------------- report


def run_report(args: argparse.Namespace) -> int:
    """``atomfrust report`` — D7, and the refusal of an unadjusted headline (plan §2.3)."""
    paths: list[Path] = []
    for pattern in args.collect:
        matches = sorted(globlib.glob(str(pattern), recursive=True))
        if not matches:
            print(f"atomfrust report: no path matches {pattern!r}", file=sys.stderr)
            return EXIT_USAGE
        paths.extend(Path(m) for m in matches)

    descriptors = _split_list(args.descriptor) or None
    out_dir = Path(args.out)
    try:
        table = collect_analyses(paths)
        if table.empty:
            raise ValueError(f"no summary rows under {list(args.collect)}")
        path = render_report(
            table,
            out_dir,
            descriptors=descriptors,
            outcome=args.outcome,
            covariate=args.covariate,
            n_perm=int(args.permute),
            seed=int(args.seed),
            n_boot=int(args.n_boot),
            alpha=float(args.alpha),
            plots=bool(args.plots),
            title=args.title,
        )
    except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as exc:
        print(f"atomfrust report: {exc}", file=sys.stderr)
        return EXIT_ERROR

    results = pd.read_csv(out_dir / "report_table.csv")
    withheld = [
        str(row["descriptor"])
        for _, row in results.iterrows()
        if not headline_is_permitted(row)
    ]
    print(f"report -> {path}")
    print(
        f"  {len(results)} descriptor(s) over n = {int(results['n'].iloc[0])} rows, "
        f"outcome {args.outcome!r}, covariate {args.covariate!r} held fixed"
    )
    print(
        "  headlines withheld under the covariate guard: "
        + (", ".join(withheld) if withheld else "none")
    )
    return 0


# ------------------------------------------------------------------------ calibrate


def run_calibrate(args: argparse.Namespace) -> int:
    """``atomfrust calibrate`` — pooled, definition-specific class thresholds.

    **The pooling is the point, and it is not optional.** 0.78 and −1.0 are quantiles of one
    particular null: they mark roughly the top and bottom decile of the frustration index
    *under the contact definition, decoy construction and energy function they were
    calibrated on*. Change any of those — which is what this whole tool is for — and the F
    distribution shifts and rescales, so the literals select some other quantile and the
    reported "minimally frustrated" count moves for a purely definitional reason.
    Re-deriving the cutoffs from the observed distribution restores their intended meaning.

    Doing that **per system** would destroy the measurement. If each structure's cutoffs are
    its own deciles, then by construction every structure has the same 10% minimally
    frustrated and the same 10% highly frustrated, and the between-structure variation —
    the entire quantity a frustration–affinity correlation is about — is identically zero.
    So there is no per-system mode; ``--per-system`` exists only to be refused with this
    explanation, and the cohort's F values are pooled into one distribution before a single
    quantile is taken (see :func:`~atomfrust.analyze.classify.quantile_thresholds`).
    """
    if args.per_system:
        print(
            "atomfrust calibrate: --per-system is refused. Thresholds must be calibrated "
            "on the pooled distribution across every system in the cohort. Calibrating "
            "each system on its own F distribution makes its class fractions equal to the "
            "chosen quantiles by construction, identically for every structure, so the "
            "between-structure signal - the whole quantity of interest - becomes zero by "
            "definition rather than by measurement.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        systems = _resolve_systems(args.runs)
    except UsageError as exc:
        print(f"atomfrust calibrate: {exc}", file=sys.stderr)
        return EXIT_USAGE

    overlay: dict[str, Any] = {}
    _put(overlay, "contacts.definition", args.definition)
    _put(overlay, "contacts.cutoff_A", args.cutoff_A)
    _put(overlay, "manybody.mode", args.manybody)
    _put(overlay, "analysis.index", args.index)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled: list[np.ndarray] = []
    per_system: list[dict[str, Any]] = []
    failures = 0
    index_name: str | None = None
    definition: str | None = None

    for system in systems:
        try:
            settings = _with_overlay(_settings_of(system), overlay)
            axes = _resolve_axes(system, settings, None)
            if not axes:
                raise AnalysisError("no decoy energies are stored for this system")
            reconstruction = _reconstruct(system, settings, axes, n_decoys=args.n_decoys)
            F = compute_index(
                reconstruction.E_native, reconstruction.pooled, index=settings.analysis.index
            )
        except (AnalysisError, FileNotFoundError, KeyError, ValueError) as exc:
            print(f"atomfrust calibrate: {system.system_id}: {exc}", file=sys.stderr)
            failures += 1
            continue

        index_name = settings.analysis.index
        definition = settings.contacts.definition
        finite = F[np.isfinite(F)]
        pooled.append(finite)
        per_system.append(
            {
                "system_id": system.system_id,
                "run_dir": str(Path(system.root).resolve()),
                "n_contacts": int(F.size),
                "n_finite": int(finite.size),
                "F_median": float(np.median(finite)) if finite.size else float("nan"),
                "F_min": float(finite.min()) if finite.size else float("nan"),
                "F_max": float(finite.max()) if finite.size else float("nan"),
            }
        )

    if not pooled:
        return EXIT_ERROR

    values = np.concatenate(pooled)
    try:
        thresholds = quantile_thresholds(
            values, minimal_q=float(args.minimal_q), highly_q=float(args.highly_q)
        )
    except ValueError as exc:
        print(f"atomfrust calibrate: {exc}", file=sys.stderr)
        return EXIT_ERROR

    overlay_yaml = {
        "analysis": {
            "classify": {
                "minimally_frustrated": thresholds.minimally_frustrated,
                "highly_frustrated": thresholds.highly_frustrated,
                "mode": "quantile",
            }
        }
    }
    (out_dir / "thresholds.yaml").write_text(
        "# Calibrated by `atomfrust calibrate` on the POOLED distribution of "
        f"{len(per_system)} system(s).\n"
        "# Pass to: atomfrust analyze --run-dir R --config thresholds.yaml\n"
        + yaml.safe_dump(overlay_yaml, sort_keys=True)
    )
    _write_json(
        out_dir / "calibration.json",
        _stamp(
            CALIBRATE_NAME,
            calibration="pooled",
            per_system_calibration_refused=(
                "per-system quantiles force identical class fractions across structures by "
                "construction and destroy the between-structure signal"
            ),
            index=index_name,
            contact_definition=definition,
            minimal_q=float(args.minimal_q),
            highly_q=float(args.highly_q),
            n_systems=len(per_system),
            n_values_pooled=int(values.size),
            thresholds=thresholds.model_dump(mode="json"),
            # The published literals, read from the one place they are written down
            # (`ClassifySettings`) rather than restated here — R25.
            published_thresholds=ClassifySettings().model_dump(mode="json"),
            systems=per_system,
        ),
    )

    print(f"calibrate -> {out_dir}")
    print(
        f"  pooled {values.size} F values from {len(per_system)} system(s) "
        f"(index {index_name!r}, definition {definition!r})"
    )
    published = ClassifySettings()
    print(
        f"  minimally_frustrated > {thresholds.minimally_frustrated:.4g}   "
        f"highly_frustrated < {thresholds.highly_frustrated:.4g}   "
        f"(published: {published.minimally_frustrated:g} / "
        f"{published.highly_frustrated:g})"
    )
    if len(per_system) == 1:
        print(
            "  warning: one system in the cohort, so the pooled distribution is that "
            "system's own. The cutoffs are not comparable across structures - add the "
            "rest of the cohort to --runs before using them for a correlation."
        )
    return EXIT_ERROR if failures else 0
