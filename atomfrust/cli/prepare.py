"""``atomfrust prepare`` — turn a spec, or a bare PDB, into a prepared system directory.

Plan step E1, and the whole of user request 8: ``atomfrust prepare --pdb my_complex.pdb
--ligand B:501 -o prepared/`` works with no metadata pipeline, no RCSB lookup and nothing
EGFR-specific. The prototype's entry point was a three-CSV inner join
(``run_pipeline.py:140``) that could only ever see structures the EGFR stages had produced;
here the only input is coordinates plus a selector.

What a prepared directory is::

    prepared/<system_id>/
      system.spec.yaml      the resolved spec, every path absolute
      receptor.pdb          byte-for-byte copy of the coordinates that will be scored
      components.yaml       comp_id <-> rosetta_name <-> params <-> sha256, one row per component
      prepare_report.json   digests, chains, warnings, validation problems

``receptor.pdb`` is a verbatim copy rather than a filtered one because
:func:`atomfrust.pose.load_complex` hands the whole file to Rosetta and applies
``receptor.chains`` when it builds nodes — so the bytes that are scored are all of them, and
a "cleaned" copy here would be a second, unverified preparation step.

**Validation reports everything at once.** :meth:`SystemSpec.check_against_structure` already
collects every problem with the chains and residues that *are* present; this command prints
the whole list and exits non-zero rather than dying on the first one. A prepared directory is
never written for a system that failed.

**The comp_id trap.** ``data/processed/*_clean.pdb`` are legacy Stage-4 outputs whose HETATM
``resName`` has already been rewritten to the Rosetta name: 5HG8 stores ``Z34``, not its true
CCD code ``634``. A spec built by scanning such a file therefore carries the Rosetta name in
``comp_id`` — right for the pose, wrong for reporting (see the warning in
:meth:`SystemSpec.from_pdb`). ``--comp-id A:9001=634`` supplies the true code and moves what
the file actually says into ``rosetta_name``, which is where it belongs.

That split forces one subtlety: the comp_id a *file* carries is the pose-facing name, so this
command validates against :func:`_structure_facing`, a shadow spec whose ligand ``comp_id`` is
``effective_rosetta_name``. Without it any correctly-written spec — ``comp_id: "634"`` with
``rosetta_name: Z34`` — would be reported as inconsistent with its own structure, because
``_check_selector`` compares ``comp_id`` to the file literally. The same literal comparison
lives inside ``load_complex(..., validate=True)``, so a spec carrying the true CCD code is
flagged there too; the report records that as a warning rather than silently writing the
Rosetta name into ``comp_id`` and losing the real one.

No parametrisation happens here — that is :mod:`atomfrust.chem`. A component with no
``params`` is recorded in ``prepare_report.json`` under ``unparametrised`` and named in the
summary, because Rosetta is initialised with ``-in:file:load_PDB_components false``
(``atomfrust.pose.DEFAULT_INIT_FLAGS``): with that flag off, an unparametrised HETATM would be
silently typed from the bundled CCD and scored with atom types nobody chose, so instead it
fails at pose load. A prepared directory with unparametrised components is therefore valid
input for reporting and useless input for ``generate-decoys``, and says so.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from atomfrust.provenance import sha256_file
from atomfrust.spec import (
    LigandSpec,
    ResidueSelector,
    SpecError,
    SystemSet,
    SystemSpec,
)

# The PDB/mmCIF column reader that spec.py itself uses for every spec load. Imported rather
# than re-derived: a second parser is exactly how a comp_id read from column 18-20 in one
# place and column 17-19 in another gets to disagree.
from atomfrust.spec import _read_chains_and_residues as read_chains_and_residues

__all__ = ["NAME", "HELP", "register", "run", "prepare_system"]

NAME = "prepare"
HELP = "build a prepared system from a spec or a bare PDB"

#: A spec that does not match its coordinates, or a system that could not be written.
EXIT_SPEC_ERROR = 1
#: Flag combinations argparse cannot express (``--ligand`` without ``--pdb``).
EXIT_USAGE = 2

SPEC_FILENAME = "system.spec.yaml"
RECEPTOR_FILENAME = "receptor.pdb"
COMPONENTS_FILENAME = "components.yaml"
REPORT_FILENAME = "prepare_report.json"

#: Flags that only make sense while building a spec from bare coordinates.
_PDB_ONLY_FLAGS = ("ligand", "chains", "system_id", "label")


# ----------------------------------------------------------------------- parser


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        NAME,
        help=HELP,
        description=(
            "Build a prepared system directory from a spec file or a bare PDB. "
            "Validates the spec against the actual coordinates and reports every "
            "problem at once."
        ),
        epilog=(
            "examples:\n"
            "  atomfrust prepare --pdb my_complex.pdb --ligand B:501 -o prepared/\n"
            "  atomfrust prepare --pdb 5HG8_clean.pdb --ligand A:9001 "
            "--comp-id A:9001=634 -o prepared/\n"
            "  atomfrust prepare --spec systems.csv -o prepared/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--spec",
        type=Path,
        help="system spec: YAML, JSON or CSV; one system or a whole set",
    )
    source.add_argument(
        "--pdb", type=Path, help="bare coordinates; build the spec from them"
    )

    parser.add_argument(
        "--ligand",
        action="append",
        default=[],
        metavar="CHAIN:RESSEQ",
        help="component to treat as a ligand (repeatable). Omit to autodetect.",
    )
    parser.add_argument(
        "--chains",
        action="append",
        default=[],
        metavar="A,B",
        help="protein chains to keep (repeatable or comma-separated). Omit to keep all.",
    )
    parser.add_argument(
        "--no-autodetect",
        dest="autodetect",
        action="store_false",
        help="do not turn unnamed HETATM residues into components (protein-only)",
    )
    parser.add_argument("--system-id", help="system id; defaults to the file stem + codes")
    parser.add_argument(
        "--covalent-from",
        metavar="CSV",
        help=(
            "attach covalent anchors from a Stage-4 preparation_summary.csv. Without this "
            "nothing populates covalent_anchor, so a covalent complex is prepared as though "
            "it were non-covalent — which is exactly the prototype's failure mode (15 of the "
            "61 EGFR complexes are CYS797-linked and were all scored as non-covalent)."
        ),
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="free-form label, e.g. affinity_pM=23.0 (repeatable)",
    )
    parser.add_argument(
        "--comp-id",
        action="append",
        default=[],
        metavar="CHAIN:RESSEQ=CODE",
        help=(
            "true CCD code for a component whose file says otherwise; the file's name "
            "becomes rosetta_name (634/Z34 trap, repeatable)"
        ),
    )
    parser.add_argument(
        "--params",
        action="append",
        default=[],
        metavar="CHAIN:RESSEQ=PATH",
        help="existing .params for a component (repeatable). No parametrisation is run.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="root directory; one subdirectory per system id is written under it",
    )
    parser.set_defaults(func=run)


# -------------------------------------------------------------------- argument parsing


def _split_chains(values: Iterable[str]) -> tuple[str, ...] | None:
    chains = [c.strip() for value in values for c in value.split(",") if c.strip()]
    return tuple(chains) or None


def _parse_key_value(text: str, what: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key.strip() or not value.strip():
        raise ValueError(f"expected {what}, got {text!r}")
    return key.strip(), value.strip()


def _parse_labels(values: Iterable[str]) -> dict[str, Any]:
    """``KEY=VALUE``, with numbers kept as numbers so YAML round-trips them unchanged."""
    labels: dict[str, Any] = {}
    for item in values:
        key, value = _parse_key_value(item, "KEY=VALUE")
        labels[key] = _coerce_scalar(value)
    return labels


def _coerce_scalar(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_selector_map(
    values: Iterable[str], what: str
) -> dict[tuple[str, int, str], str]:
    out: dict[tuple[str, int, str], str] = {}
    for item in values:
        left, value = _parse_key_value(item, what)
        out[ResidueSelector.parse(left).key()] = value
    return out


# ------------------------------------------------------------------- spec building


def _load_systems(args: argparse.Namespace) -> list[SystemSpec]:
    if args.pdb is not None:
        spec = SystemSpec.from_pdb(
            args.pdb,
            ligand=list(args.ligand) or None,
            system_id=args.system_id,
            chains=_split_chains(args.chains),
            autodetect=args.autodetect,
            **_parse_labels(args.label),
        )
        return [spec]

    path = Path(args.spec)
    if not path.exists():
        raise SpecError(f"spec file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        systems = SystemSet.from_csv_file(path)
    elif suffix == ".json":
        systems = SystemSet.from_json_file(path)
    elif suffix in {".yaml", ".yml"}:
        systems = SystemSet.from_yaml_file(path)
    else:
        raise SpecError(
            f"unrecognised spec extension {suffix!r} in {path.name}; "
            "expected .yaml, .yml, .json or .csv"
        )
    return list(systems)


def _structure_facing(spec: SystemSpec) -> SystemSpec:
    """A shadow spec whose ligand ``comp_id`` is the name the *file* should carry.

    ``_check_selector`` compares ``comp_id`` to the structure literally, but a spec that
    resolves the 634/Z34 collision correctly holds the CCD code in ``comp_id`` and the pose's
    name in ``rosetta_name``. Validating that spec directly would report a mismatch that is
    the whole point of the two fields, so validation runs against ``effective_rosetta_name``.
    """
    ligands = []
    for lig in spec.ligands:
        name = lig.effective_rosetta_name
        if name is not None and name != lig.selector.comp_id:
            lig = lig.model_copy(
                update={"selector": lig.selector.model_copy(update={"comp_id": name})}
            )
        ligands.append(lig)
    return spec.model_copy(update={"ligands": tuple(ligands)})


def _apply_overrides(
    spec: SystemSpec,
    comp_ids: dict[tuple[str, int, str], str],
    params: dict[tuple[str, int, str], str],
    in_file: dict[tuple[str, int, str], str],
) -> tuple[SystemSpec, list[str], set[tuple[str, int, str]]]:
    """Apply ``--comp-id`` / ``--params`` and report any that name nothing.

    A ``--comp-id`` override sets ``rosetta_name`` from what the file actually says, so the
    pose-facing name survives even though ``comp_id`` now reports the true CCD code.
    """
    known = {lig.selector.key() for lig in spec.ligands}
    problems: list[str] = []
    present = ", ".join(sorted(f"{c}:{r}{i}" for c, r, i in known)) or "(none)"
    for flag, keys in (("--comp-id", comp_ids), ("--params", params)):
        for chain, resseq, icode in sorted(set(keys) - known):
            problems.append(
                f"{flag} names {chain}:{resseq}{icode}, which is not a component of "
                f"system {spec.system_id!r}; components: {present}"
            )

    ligands: list[LigandSpec] = []
    for lig in spec.ligands:
        key = lig.selector.key()
        update: dict[str, Any] = {}
        if key in comp_ids:
            update["selector"] = lig.selector.model_copy(
                update={"comp_id": comp_ids[key]}
            )
            actual = in_file.get(key)
            if actual is not None:
                update["rosetta_name"] = actual
        if key in params:
            update["params"] = Path(params[key]).expanduser().resolve()
        ligands.append(lig.model_copy(update=update) if update else lig)
    return spec.model_copy(update={"ligands": tuple(ligands)}), problems, set(comp_ids)


def _absolutised(spec: SystemSpec, receptor_path: Path) -> SystemSpec:
    ligands = tuple(
        lig.model_copy(update={"params": Path(lig.params).expanduser().resolve()})
        if lig.params is not None
        else lig
        for lig in spec.ligands
    )
    return spec.model_copy(
        update={
            "receptor": spec.receptor.model_copy(update={"path": receptor_path}),
            "ligands": ligands,
        }
    )


# ------------------------------------------------------------------------- preparing


def prepare_system(
    spec: SystemSpec,
    out_root: Path,
    comp_ids: dict[tuple[str, int, str], str] | None = None,
    params: dict[tuple[str, int, str], str] | None = None,
) -> dict[str, Any]:
    """Write one prepared directory and return its report. Raises :class:`SpecError`.

    Nothing is written unless the spec validates, so a prepared directory on disk is always a
    directory that passed.
    """
    comp_ids = dict(comp_ids or {})
    params = dict(params or {})

    if spec.receptor.path is None:
        raise SpecError(
            f"system {spec.system_id!r} names pdb_id {spec.receptor.pdb_id!r} but no local "
            "file; fetching is not implemented — pass --pdb or a spec with receptor.path"
        )
    source = Path(spec.receptor.path).expanduser().resolve()
    if not source.exists():
        raise SpecError(f"system {spec.system_id!r}: receptor file not found: {source}")
    if "/" in spec.system_id or "\\" in spec.system_id:
        raise SpecError(
            f"system_id {spec.system_id!r} contains a path separator; it names a directory"
        )

    chains_present, residues = read_chains_and_residues(source)
    spec, problems, overridden = _apply_overrides(spec, comp_ids, params, residues)
    problems += _structure_facing(spec).check_against_structure(source)
    if problems:
        raise SpecError(
            f"system {spec.system_id!r} does not match {source.name}:\n  - "
            + "\n  - ".join(problems)
        )

    prepared = (Path(out_root) / spec.system_id).expanduser().resolve()
    prepared.mkdir(parents=True, exist_ok=True)
    receptor = prepared / RECEPTOR_FILENAME
    source_digest = sha256_file(source)
    shutil.copyfile(source, receptor)
    copy_digest = sha256_file(receptor)
    if copy_digest != source_digest:
        raise SpecError(
            f"system {spec.system_id!r}: {receptor} does not digest to its source "
            f"{source} ({copy_digest} vs {source_digest})"
        )

    resolved = _absolutised(spec, receptor)
    (prepared / SPEC_FILENAME).write_text(resolved.to_yaml())

    components = _component_rows(resolved, overridden)
    (prepared / COMPONENTS_FILENAME).write_text(
        yaml.safe_dump(
            {
                "system_id": resolved.system_id,
                "receptor": str(receptor),
                "components": components,
            },
            sort_keys=False,
            default_flow_style=False,
        )
    )

    unparametrised = [row["selector"] for row in components if row["params"] is None]
    warnings = _warnings(source, resolved, components, overridden, unparametrised)
    report = {
        "system_id": resolved.system_id,
        "prepared_dir": str(prepared),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "receptor": {
            "source_path": str(source),
            "source_sha256": source_digest,
            "sha256": copy_digest,
            "chains_present": sorted(chains_present),
            "chains_kept": (
                list(resolved.receptor.chains)
                if resolved.receptor.chains
                else sorted(chains_present)
            ),
        },
        "components": components,
        "is_protein_only": resolved.is_protein_only,
        "is_covalent": resolved.is_covalent,
        "validation_problems": [],
        "unparametrised": unparametrised,
        "warnings": warnings,
        "files": {
            name: sha256_file(prepared / name)
            for name in (SPEC_FILENAME, RECEPTOR_FILENAME, COMPONENTS_FILENAME)
        },
    }
    (prepared / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _component_rows(
    spec: SystemSpec, overridden: set[tuple[str, int, str]]
) -> list[dict[str, Any]]:
    """One row per component, carrying both names so no consumer has to guess which it has."""
    rows: list[dict[str, Any]] = []
    for lig in spec.ligands:
        sel = lig.selector
        params_path = Path(lig.params) if lig.params else None
        anchor = lig.covalent_anchor
        rows.append(
            {
                "selector": f"{sel.chain}:{sel.resseq}{sel.icode.strip()}",
                "chain": sel.chain,
                "resseq": sel.resseq,
                "icode": sel.icode.strip(),
                "comp_id": sel.comp_id,
                "comp_id_source": "override" if sel.key() in overridden else "file",
                "rosetta_name": lig.effective_rosetta_name,
                "params": str(params_path) if params_path else None,
                "params_sha256": (
                    sha256_file(params_path)
                    if params_path and params_path.exists()
                    else None
                ),
                "covalent_anchor": (
                    f"{anchor.chain}:{anchor.resseq} {anchor.atom}-{anchor.ligand_atom}"
                    if anchor
                    else None
                ),
            }
        )
    return rows


def _warnings(
    source: Path,
    spec: SystemSpec,
    components: list[dict[str, Any]],
    overridden: set[tuple[str, int, str]],
    unparametrised: list[str],
) -> list[str]:
    warnings: list[str] = []
    if unparametrised:
        warnings.append(
            f"no params for {', '.join(unparametrised)}; `atomfrust generate-decoys` will "
            "fail at pose load, because Rosetta runs with -in:file:load_PDB_components "
            "false so an unparametrised component is an error rather than a silent "
            "CCD-typed residue"
        )
    if source.name.endswith("_clean.pdb") and any(
        row["comp_id_source"] == "file" for row in components
    ):
        warnings.append(
            f"{source.name} looks like a legacy Stage-4 output, whose HETATM resName is "
            "already the Rosetta name (5HG8 stores Z34, not 634); comp_id here is therefore "
            "the pose name, not necessarily the CCD code — pass --comp-id CHAIN:RESSEQ=CODE "
            "to record the real one"
        )
    if overridden:
        warnings.append(
            "comp_id was overridden, so system.spec.yaml holds the CCD code while the file "
            "holds the Rosetta name; SystemSpec.check_against_structure compares comp_id to "
            "the file literally, so load_complex(..., validate=True) will flag it until that "
            "check compares against effective_rosetta_name"
        )
    if spec.is_covalent:
        warnings.append(
            "covalent anchor recorded; the linkage itself is not modelled — the anchor "
            "residue is only frozen against mutation"
        )
    return warnings


# --------------------------------------------------------------------------- summary


def _print_summary(report: dict[str, Any], stream: Any) -> None:
    receptor = report["receptor"]
    kept = ", ".join(receptor["chains_kept"])
    all_chains = ", ".join(receptor["chains_present"])
    print(f"{report['system_id']}  ->  {report['prepared_dir']}", file=stream)
    print(
        f"  receptor  {Path(receptor['source_path']).name}  {receptor['sha256'][:19]}...",
        file=stream,
    )
    print(f"  chains    kept: {kept}   (present: {all_chains})", file=stream)
    if report["components"]:
        for row in report["components"]:
            params = Path(row["params"]).name if row["params"] else "(none)"
            print(
                f"  component {row['selector']:<10} comp_id={row['comp_id']} "
                f"rosetta_name={row['rosetta_name']} params={params}"
                + (" [comp_id overridden]" if row["comp_id_source"] == "override" else ""),
                file=stream,
            )
    else:
        print("  component (none) — protein-only system", file=stream)
    print(
        f"  protein-only: {report['is_protein_only']}   "
        f"covalent: {report['is_covalent']}   "
        f"validation problems: {len(report['validation_problems'])}",
        file=stream,
    )
    for warning in report["warnings"]:
        print(f"  warning: {warning}", file=stream)


# ------------------------------------------------------------------------------ run


def _attach_covalent_anchors(systems, csv_path: Path):
    """Attach covalent anchors recovered from Stage 4's summary to each spec.

    The anchor is matched by the spec's `system_id` prefix, which is how the Stage-4 CSV is
    keyed (`pdb_id`). A system with no row is left untouched rather than treated as an
    error — most systems are not covalent.
    """
    from atomfrust.covalent import anchors_from_preparation_summary

    out = []
    for spec in systems:
        pdb_id = spec.system_id.split("_")[0].upper()
        anchors = anchors_from_preparation_summary(csv_path, pdb_id)
        if not anchors or not spec.ligands:
            out.append(spec)
            continue
        # Stage 4 keeps exactly one ligand copy, so a single anchor attaches to it.
        ligands = list(spec.ligands)
        ligands[0] = ligands[0].model_copy(update={"covalent_anchor": anchors[0]})
        out.append(spec.model_copy(update={"ligands": tuple(ligands)}))
    return out


def run(args: argparse.Namespace) -> int:
    if args.spec is not None:
        offenders = [
            f"--{flag.replace('_', '-')}"
            for flag in _PDB_ONLY_FLAGS
            if getattr(args, flag, None)
        ]
        if not args.autodetect:
            offenders.append("--no-autodetect")
        if offenders:
            print(
                f"atomfrust prepare: {', '.join(offenders)} apply to --pdb only; "
                "with --spec the file is the source of truth",
                file=sys.stderr,
            )
            return EXIT_USAGE

    try:
        comp_ids = _parse_selector_map(args.comp_id, "CHAIN:RESSEQ=CODE")
        params = _parse_selector_map(args.params, "CHAIN:RESSEQ=PATH")
        systems = _load_systems(args)
    except (SpecError, ValueError, OSError) as exc:
        print(f"atomfrust prepare: {exc}", file=sys.stderr)
        return EXIT_SPEC_ERROR

    if getattr(args, "covalent_from", None):
        try:
            systems = _attach_covalent_anchors(systems, Path(args.covalent_from))
        except (OSError, ValueError, KeyError) as exc:
            print(f"atomfrust prepare: --covalent-from: {exc}", file=sys.stderr)
            return EXIT_SPEC_ERROR

    failures = 0
    for spec in systems:
        try:
            report = prepare_system(spec, Path(args.out), comp_ids, params)
        except (SpecError, OSError) as exc:
            print(f"atomfrust prepare: {exc}", file=sys.stderr)
            failures += 1
            continue
        _print_summary(report, sys.stdout)
    return EXIT_SPEC_ERROR if failures else 0
