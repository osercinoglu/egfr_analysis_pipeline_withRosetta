"""`atomfrust generate-decoys` — an ensemble, and nothing else (user request 6).

This subcommand produces decoys, their direct pair energies, and the provenance needed to
re-analyse them. It computes **no** frustration index, no classification and no summary, and
it must never import :mod:`atomfrust.analyze` — a test asserts that. The separation is the
whole point of the run-directory contract: generation is the expensive, PyRosetta-bound half,
and every scientific choice that can be deferred (index function, many-body formula, contact
definition, shell, thresholds) is deferred to `atomfrust analyze`, which reads the same bytes
with no Rosetta call.

Three behaviours worth knowing before reading the code:

**Resume is the default.** Decoy ids already on disk are subtracted from the plan
(:func:`~atomfrust.execute.plan_units` takes ``completed``), so re-running with a larger
``--n-decoys`` *extends* the ensemble instead of recomputing it or — as the prototype did at
``run_pipeline.py:241`` — silently doing nothing. ``--restart`` discards the stored decoys of
the requested systems and regenerates them; it does not rewrite the run's settings, because
the stored decoys of *other* systems were produced under those settings.

**A generation-stage settings change is an error, not a merge.** Reopening a run directory
asserts compatibility; a difference in a generation-stage field (score function, decoy
protocol, regions, native repack policy, …) or in an input digest exits **3** with a
field-by-field diff. ``--allow-mismatch`` downgrades that to a warning and stamps
``provisional`` on every system it touches.

**Nothing here is derived from a filename.** The system identity, the ligand position and the
``comp_id`` ↔ ``rosetta_name`` ↔ params triple all come from the spec, which is copied into
``inputs/`` verbatim so a run can be re-derived from its own directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from atomfrust.execute import (
    ShardSpec,
    WorkUnit,
    default_worker_count,
    execute,
    plan_units,
)
from atomfrust.provenance import (
    build_manifest,
    capture_environment,
    sha256_bytes,
    sha256_file,
)
from atomfrust.runstore import RegenerationRequired, RunDir, SystemDir
from atomfrust.settings import Layer, Settings, load_settings
from atomfrust.spec import SystemSet, SystemSpec

__all__ = [
    "NAME",
    "HELP",
    "INDEX_COLUMNS",
    "register",
    "run",
    "overlay_from_args",
    "resolve_settings",
    "load_systems",
    "region_expressions",
    "unsupported_settings",
]

NAME = "generate-decoys"
HELP = "generate a decoy ensemble and stop; computes no index"

#: `decoys/index.parquet`, §4 of the plan. Columns the generator does not fill (a chemotype
#: axis's `source_ref`, a library member's `member_inchikey`) are present and null rather
#: than absent, so the schema does not depend on which axis produced the run.
INDEX_COLUMNS = (
    "decoy_id",
    "axis",
    "generator",
    "scope",
    "identity",
    "placement",
    "seed",
    "source_ref",
    "member_inchikey",
    "n_mutated",
    "backbone_rmsd",
    "wall_s",
    "status",
    "structure_path",
)

# What the engine can actually do today. Every other value the Settings model admits is
# reachable from the plan but not implemented, and is refused by name rather than failing
# somewhere deep in a worker.
_SUPPORTED_AXES = {"identity"}
_SUPPORTED_SCOPES = {"whole_protein", "contact_shell"}
_SUPPORTED_IDENTITIES = {"native", "composition", "uniform20"}

EXIT_OK = 0
EXIT_USAGE = 2


# ------------------------------------------------------------------------ parser


def register(subparsers: Any) -> None:
    """Add the `generate-decoys` parser and set :func:`run` as its ``func`` default."""
    parser = subparsers.add_parser(
        NAME,
        help=HELP,
        description=(
            "Generate a decoy ensemble into a run directory and stop. Computes no "
            "frustration index: run `atomfrust analyze` against the same run directory, "
            "as many times and under as many settings as you like."
        ),
        epilog=(
            "Omitted flags are not defaulted here: they fall through to --config, then the\n"
            "ATOMFRUST__ environment layer, then the built-in settings (n_decoys=1000,\n"
            "relax=mc, identity=native, placement=permute -- the published protocol).\n"
            "Re-running an existing run directory resumes it; a larger --n-decoys extends\n"
            "the ensemble. A generation-stage settings change exits 3 with a diff."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_argument_group("input (exactly one of --spec / --pdb)")
    source.add_argument("--spec", type=Path, help="system spec YAML: one system or a list")
    source.add_argument("--pdb", type=Path, help="a bare PDB, for a one-off system")
    source.add_argument(
        "--ligand",
        action="append",
        metavar="CHAIN:RESSEQ",
        help="ligand position, repeatable. With --pdb only; omit to autodetect components.",
    )
    source.add_argument("--system-id", help="system id for --pdb; defaults to the file stem")
    source.add_argument("--chains", help="comma-separated receptor chains to keep (--pdb)")

    parser.add_argument("--run-dir", type=Path, required=True, help="run directory")
    parser.add_argument("--config", type=Path, help="settings YAML, layered under the flags")

    generation = parser.add_argument_group("generation settings")
    generation.add_argument("--axis", help="comma-separated decoy axes", default=None)
    generation.add_argument("--n-decoys", type=int, help="decoys per axis")
    generation.add_argument(
        "--scope", choices=("whole_protein", "contact_shell", "contact_pair")
    )
    generation.add_argument(
        "--identity", choices=("native", "composition", "uniform20", "layer_stratified")
    )
    generation.add_argument("--placement", choices=("inplace", "permute"))
    generation.add_argument("--mutate-sel", metavar="EXPR", help="residues whose identity changes")
    generation.add_argument("--repack-sel", metavar="EXPR", help="residues that are repacked")
    generation.add_argument("--minimize-sel", metavar="EXPR", help="residues that are minimised")
    generation.add_argument(
        "--repack-shell",
        type=float,
        metavar="R",
        help="intersect all three region selectors with within(R, ligand)",
    )
    generation.add_argument(
        "--relax",
        choices=("min", "mc"),
        help="post-repack relaxation; 'mc' is the published protocol and ~5x the cost",
    )
    generation.add_argument("--mc-cycles", type=int, help="Monte-Carlo cycles when --relax mc")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--workers", type=int, help="worker processes; 0 means every logical CPU"
    )
    runtime.add_argument("--shard", metavar="k/M", help="generate shard k of M; shards never contend")
    runtime.add_argument(
        "--save-structures", action="store_true", help="write decoys/structures/decoy_*.pdb.gz"
    )
    runtime.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing ensemble (the default; the flag is for explicitness)",
    )
    runtime.add_argument(
        "--restart", action="store_true", help="discard stored decoys for these systems"
    )
    runtime.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="proceed despite a generation-stage difference, stamping outputs provisional",
    )

    parser.set_defaults(func=run)


# ------------------------------------------------------------------- settings


def overlay_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """The CLI settings layer: only what was actually passed.

    Absent flags are left out entirely rather than passed as their argparse defaults, so a
    ``--config`` file and the ``ATOMFRUST__`` environment layer keep their say. This is why
    ``--relax`` has no argparse default: the published ``mc`` default lives in
    :mod:`atomfrust.settings` and is not duplicated here.
    """
    decoys: dict[str, Any] = {}
    regions: dict[str, Any] = {}
    runtime: dict[str, Any] = {}

    if args.axis:
        decoys["axes"] = tuple(a.strip() for a in args.axis.split(",") if a.strip())
    if args.n_decoys is not None:
        decoys["n_decoys"] = args.n_decoys
    if args.scope is not None:
        decoys["scope"] = args.scope
    if args.identity is not None:
        decoys["identity"] = args.identity
    if args.placement is not None:
        decoys["placement"] = args.placement
    if args.relax is not None:
        decoys["relax"] = args.relax
    if args.mc_cycles is not None:
        decoys["mc_cycles"] = args.mc_cycles
    if args.repack_shell is not None:
        decoys["repack_shell_A"] = args.repack_shell

    if args.mutate_sel is not None:
        regions["mutate"] = args.mutate_sel
    if args.repack_sel is not None:
        regions["repack"] = args.repack_sel
    if args.minimize_sel is not None:
        regions["minimize"] = args.minimize_sel
    if regions:
        decoys["regions"] = regions

    if args.workers is not None:
        runtime["workers"] = default_worker_count() if args.workers == 0 else args.workers
    if args.shard is not None:
        runtime["shard"] = args.shard
    if args.save_structures:
        runtime["save_structures"] = True

    overlay: dict[str, Any] = {}
    if decoys:
        overlay["decoys"] = decoys
    if runtime:
        overlay["runtime"] = runtime
    return overlay


def resolve_settings(args: argparse.Namespace) -> tuple[Settings, dict[str, Layer]]:
    """Layer defaults → ``--config`` → environment → flags."""
    file_overlay = None
    if args.config is not None:
        file_overlay = yaml.safe_load(Path(args.config).read_text()) or {}
    return load_settings(file_overlay, overlay_from_args(args))


def unsupported_settings(settings: Settings) -> list[str]:
    """Requested generation options the engine cannot honour yet. Empty means go."""
    problems = []
    for axis in settings.decoys.axes:
        if axis not in _SUPPORTED_AXES:
            problems.append(
                f"axis {axis!r} is not implemented (plan stage G supplies the chemotype "
                f"and pose axes); implemented: {', '.join(sorted(_SUPPORTED_AXES))}"
            )
    if settings.decoys.scope not in _SUPPORTED_SCOPES:
        problems.append(
            f"scope {settings.decoys.scope!r} is not implemented; implemented: "
            f"{', '.join(sorted(_SUPPORTED_SCOPES))}"
        )
    if settings.decoys.identity not in _SUPPORTED_IDENTITIES:
        problems.append(
            f"identity {settings.decoys.identity!r} is not implemented; implemented: "
            f"{', '.join(sorted(_SUPPORTED_IDENTITIES))}"
        )
    return problems


def region_expressions(settings: Settings) -> tuple[str, str, str]:
    """The three selector expressions, with ``--repack-shell`` folded in.

    The shell intersects **all three** sets, not just the repacked one.
    :func:`~atomfrust.regions.resolve_regions` refuses ``mutate ⊄ repack`` as a hard error,
    and rightly so: a residue that takes a new identity without being repacked keeps the old
    identity's rotamer. So a shell that narrowed only ``repack`` would either be an error or
    silent corruption; narrowing all three is the only coherent reading.
    """
    regions = settings.decoys.regions
    shell = settings.decoys.repack_shell_A
    if shell is None:
        return regions.mutate, regions.repack, regions.minimize
    within = f"within({float(shell)}, ligand)"
    return (
        f"({regions.mutate}) and {within}",
        f"({regions.repack}) and {within}",
        f"({regions.minimize}) and {within}",
    )


# ---------------------------------------------------------------------- input


def load_systems(args: argparse.Namespace) -> SystemSet:
    """Build the system set from ``--spec`` or ``--pdb``."""
    if (args.spec is None) == (args.pdb is None):
        raise ValueError("pass exactly one of --spec or --pdb")
    if args.spec is not None:
        if args.ligand:
            raise ValueError("--ligand applies to --pdb; put ligands in the spec instead")
        return SystemSet.from_yaml_file(args.spec)

    chains = (
        tuple(c.strip() for c in args.chains.split(",") if c.strip())
        if args.chains
        else None
    )
    spec = SystemSpec.from_pdb(
        args.pdb, ligand=args.ligand, system_id=args.system_id, chains=chains
    )
    return SystemSet(systems=(spec,))


def _absolute(spec: SystemSpec) -> SystemSpec:
    """Resolve receptor and params paths.

    A worker is spawned, not forked, and a shard may run from a different directory; a
    relative path in the spec would then resolve against the wrong root.
    """
    data = spec.model_dump()
    if data["receptor"]["path"] is not None:
        data["receptor"]["path"] = str(Path(data["receptor"]["path"]).resolve())
    for lig in data["ligands"]:
        if lig["params"] is not None:
            lig["params"] = str(Path(lig["params"]).resolve())
    return SystemSpec.model_validate(data)


def _spec_from_yaml(text: str) -> SystemSpec:
    return SystemSpec.model_validate(yaml.safe_load(text) or {})


def _covalent_bonds(spec: SystemSpec, nodes) -> tuple[tuple[str, str], ...]:
    """Node-id pairs forced into the graph by a covalent linkage.

    Delegates to :func:`atomfrust.covalent.bonds_for_graph`, which resolves each anchor
    against the **actual node list** rather than formatting node ids blind. That matters:
    if an anchor names a chain that was dropped at load time, formatting a string produces
    an id no node has, and `build_superset` then dies with a bare
    ``KeyError: bond references an unknown node`` far from the cause.

    Built identically in the parent and in every worker, because the pair table's ``pair_id``
    is the join key between native energies and decoy energies — a bond present on one side
    and not the other would silently shift every id after it. Both sides derive it from the
    same spec and the same nodes, so they agree by construction.
    """
    from atomfrust.covalent import bonds_for_graph

    return bonds_for_graph(spec, nodes)


def _system_digests(spec: SystemSpec) -> dict[str, Any]:
    receptor = spec.receptor.path
    return {
        "receptor_sha256": sha256_file(receptor) if receptor else None,
        "params_sha256": [
            sha256_file(lig.params) for lig in spec.ligands if lig.params is not None
        ],
        "spec_sha256": sha256_bytes(spec.to_yaml().encode()),
    }


# ------------------------------------------------------------- decoy plumbing


def _resolve_regions(nodes: list[Any], geometry: Any, settings: Settings) -> Any:
    from atomfrust.regions import resolve_regions

    mutate, repack, minimize = region_expressions(settings)
    return resolve_regions(nodes, geometry, mutate=mutate, repack=repack, minimize=minimize)


def _generator(context: Any, settings: Settings) -> Any:
    """The axis-A generator, configured from settings.

    ``mutation`` is chosen rather than exposed: ``sequential`` repacks the whole pose and
    :class:`~atomfrust.decoys.identity.IdentityDecoyGenerator` refuses it outright when a
    region is restricted, while ``packer_task`` is the only path that can express a frozen
    region. The two designs the same sequence but land in different rotamer minima, so the
    cheaper-to-reproduce ``sequential`` is kept wherever it is legal.
    """
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    protein = np.array([n.kind == "protein" for n in context.nodes])
    restricted = bool((protein & ~context.regions.repack).any())
    decoys = settings.decoys
    return IdentityDecoyGenerator(
        context,
        scope=decoys.scope,
        identity=decoys.identity,
        placement=decoys.placement,
        base_seed=decoys.base_seed,
        mutation="packer_task" if restricted else "sequential",
        relax=decoys.relax,
        mc_cycles=decoys.mc_cycles,
        native_repack=decoys.native_repack,
    )


def _write_gzipped_pdb(pose: Any, path: Path) -> None:
    """``dump_pdb`` writes to a path, so the gzip has to go through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pose.dump_pdb(str(temporary))
        with temporary.open("rb") as src, gzip.open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    finally:
        temporary.unlink(missing_ok=True)


def _decoy_task(spec_yaml: str, settings_yaml: str, structures_dir: str | None):
    """Build one worker's context once and return the per-unit callable.

    The pose is rebuilt here from ``(receptor.pdb, params)`` rather than pickled: a Rosetta
    pose does not survive a spawn boundary, and PyRosetta is not fork-safe. Everything the
    worker returns is plain arrays and a dict, so nothing Rosetta-owned crosses back.
    """
    from atomfrust.decoys.base import DecoyContext
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex

    spec = _spec_from_yaml(spec_yaml)
    settings = Settings.from_yaml(settings_yaml)
    loaded = load_complex(spec)
    _, pairs = build_graph(
        loaded.nodes, loaded.geometry, settings, bonds=_covalent_bonds(spec, loaded.nodes)
    )
    context = DecoyContext(
        pose=loaded.pose,
        nodes=loaded.nodes,
        pairs=pairs,
        regions=_resolve_regions(loaded.nodes, loaded.geometry, settings),
        settings=settings,
    )
    generator = _generator(context, settings)
    root = Path(structures_dir) if structures_dir else None

    def task(unit: WorkUnit) -> dict[str, Any]:
        result = generator.generate(unit.decoy_id)
        row = dict(result.index_row)
        if root is not None:
            structure = root / f"decoy_{unit.decoy_id:06d}.pdb.gz"
            _write_gzipped_pdb(result.pose, structure)
            row["structure_path"] = str(structure)
        return {
            "pair_id": result.pair_id,
            "e_direct": result.e_direct,
            "e_fa_rep": result.e_fa_rep,
            "index_row": row,
        }

    return task


# ------------------------------------------------------------------- writers


def _energy_writer(system: SystemDir, shard: int, flush_every: int) -> Any:
    """A writer whose part numbering continues an existing shard's.

    :class:`~atomfrust.runstore.DecoyEnergyWriter` numbers from zero on construction, so a
    resumed run would rewrite ``part-<shard>-00000.parquet`` and destroy the decoys already
    stored in it. Resume is the default path here, so the sequence is advanced past whatever
    this shard already wrote.
    """
    writer = system.energy_writer(shard=shard, flush_every=flush_every)
    prefix = f"part-{shard:03d}-"
    sequences = [
        int(path.stem.split("-")[2])
        for path in system.decoy_parts()
        if path.name.startswith(prefix)
    ]
    if sequences:
        writer._sequence = max(sequences) + 1
    return writer


def _write_index(system: SystemDir, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Merge new index rows into ``decoys/index.parquet``, one row per (decoy_id, axis)."""
    path = system.decoys / "index.parquet"
    frames = [pd.read_parquet(path)] if path.exists() else []
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=list(INDEX_COLUMNS))

    frame = pd.concat(frames, ignore_index=True)
    for column in INDEX_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    ordered = list(INDEX_COLUMNS) + [c for c in frame.columns if c not in INDEX_COLUMNS]
    frame = (
        frame[ordered]
        .drop_duplicates(subset=["decoy_id", "axis"], keep="last")
        .sort_values(["axis", "decoy_id"], kind="stable")
        .reset_index(drop=True)
    )
    frame["decoy_id"] = frame["decoy_id"].astype("int32")
    frame.to_parquet(path, index=False)
    return frame


def _log(run: RunDir, **event: Any) -> None:
    run.log_path.parent.mkdir(parents=True, exist_ok=True)
    with run.log_path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


# ---------------------------------------------------------------- per system


def _write_inputs(spec: SystemSpec, system: SystemDir) -> None:
    """``inputs/`` — the exact bytes scored, so the run is re-derivable from itself."""
    (system.inputs / "system.spec.yaml").write_text(spec.to_yaml())
    if spec.receptor.path is not None:
        shutil.copyfile(spec.receptor.path, system.inputs / "receptor.pdb")
    params_dir = system.inputs / "params"
    for lig in spec.ligands:
        if lig.params is None:
            continue
        params_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lig.params, params_dir / Path(lig.params).name)


def _write_components(system: SystemDir, components: Sequence[Any]) -> None:
    """``comp_id`` ↔ ``rosetta_name`` ↔ params ↔ sha256, resolved against the loaded pose.

    Written from the pose rather than from the spec because the pose is where the mapping is
    finally true: it records what Rosetta actually called each component.
    """
    (system.inputs / "components.yaml").write_text(
        yaml.safe_dump(
            {
                "components": [
                    {
                        "node_id": c.node_id,
                        "pose_resnum": c.pose_resnum,
                        "kind": c.kind,
                        "comp_id": c.comp_id,
                        "rosetta_name": c.rosetta_name,
                        "params": str(c.params_path) if c.params_path else None,
                        "params_sha256": c.params_sha256,
                    }
                    for c in components
                ]
            },
            sort_keys=True,
            default_flow_style=False,
        )
    )


def _prepare_native(spec: SystemSpec, settings: Settings, system: SystemDir) -> None:
    """Graph, native pose, native energies and the S5.2 raw-energy control.

    Skipped wholesale when the artifacts are already there: on a resume this is the only
    part of the parent process that needs a pose at all, and rebuilding it would cost a load
    plus a repack for nothing. The stored pair table is authoritative either way — its
    ``pair_id`` is what the decoy energies are keyed by.
    """
    from atomfrust.decoys.base import DecoyContext
    from atomfrust.energy import EnergyEvaluator, raw_interaction_energy
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex

    ready = (
        (system.graph / "pairs.parquet").exists()
        and (system.native / "native_energies.parquet").exists()
        and (system.native / "raw_energy.json").exists()
    )
    if ready:
        return

    loaded = load_complex(spec)
    nodes, pairs = build_graph(
        loaded.nodes, loaded.geometry, settings, bonds=_covalent_bonds(spec, loaded.nodes)
    )
    system.write_graph(nodes, pairs)
    _write_components(system, loaded.components)

    context = DecoyContext(
        pose=loaded.pose,
        nodes=loaded.nodes,
        pairs=pairs,
        regions=_resolve_regions(loaded.nodes, loaded.geometry, settings),
        settings=settings,
    )
    native_pose = _generator(context, settings).prepare_native()
    energies = EnergyEvaluator(native_pose, settings.energy.score_function).pairs(pairs)
    system.write_native_energies(energies)
    system.write_raw_energy(
        raw_interaction_energy(pairs, energies, settings.energy.exclude_fa_rep)
    )
    native_pose.dump_pdb(str(system.native / "native.pdb"))


# ------------------------------------------------------------------ the command


def run(args: argparse.Namespace) -> int:
    """Generate decoys. Returns a process exit code; 3 means "regenerate, do not merge"."""
    created_utc = datetime.now(timezone.utc).isoformat()

    try:
        settings, provenance = resolve_settings(args)
    except Exception as exc:  # pydantic ValidationError, YAML error, missing file
        print(f"atomfrust {NAME}: bad settings: {exc}", file=sys.stderr)
        return EXIT_USAGE

    problems = unsupported_settings(settings)
    if problems:
        for problem in problems:
            print(f"atomfrust {NAME}: {problem}", file=sys.stderr)
        return EXIT_USAGE

    try:
        systems = [_absolute(spec) for spec in load_systems(args)]
    except Exception as exc:
        print(f"atomfrust {NAME}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        shard = ShardSpec.parse(settings.runtime.shard)
    except ValueError as exc:
        print(f"atomfrust {NAME}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    digests = {"systems": {spec.system_id: _system_digests(spec) for spec in systems}}
    environment = capture_environment()
    pyrosetta_version = environment.get("pyrosetta")

    root = Path(args.run_dir)
    provisional = False
    if (root / "manifest.json").exists():
        run_dir = RunDir(root)
        code, provisional = _reopen(
            run_dir, settings, digests, pyrosetta_version, allow_mismatch=args.allow_mismatch
        )
        if code:
            return code
    else:
        manifest = build_manifest(root.name, settings, created_utc, digests)
        run_dir = RunDir.create(root, manifest, settings, environment)

    _log(
        run_dir,
        event="start",
        at=created_utc,
        systems=[spec.system_id for spec in systems],
        n_decoys=settings.decoys.n_decoys,
        axes=list(settings.decoys.axes),
        shard=str(shard),
        workers=settings.runtime.workers,
        cli_layers=sorted(k for k, v in provenance.items() if v is Layer.CLI),
        provisional=provisional,
    )

    axes = list(settings.decoys.axes)
    system_dirs: dict[str, SystemDir] = {}
    for spec in systems:
        system = run_dir.system(spec.system_id)
        if args.restart and system.base.exists():
            # --restart discards stored decoys for these systems. The run's settings are
            # left alone: other systems in this directory were generated under them.
            shutil.rmtree(system.base)
        system_dirs[spec.system_id] = system.ensure()
        _write_inputs(spec, system_dirs[spec.system_id])

    completed = {
        (spec.system_id, axis): system_dirs[spec.system_id].completed_decoy_ids(axis)
        for spec in systems
        for axis in axes
    }
    units = plan_units(
        [spec.system_id for spec in systems],
        settings.decoys.n_decoys,
        axes=axes,
        completed=completed,
        shard=shard,
    )

    for spec in systems:
        _prepare_native(spec, settings, system_dirs[spec.system_id])
        _generate_one_system(
            spec,
            settings,
            run_dir,
            system_dirs[spec.system_id],
            [u for u in units if u.system_id == spec.system_id],
            shard,
            axes,
            provisional,
        )
    return EXIT_OK


def _reopen(
    run_dir: RunDir,
    settings: Settings,
    digests: dict[str, Any],
    pyrosetta_version: str | None,
    allow_mismatch: bool,
) -> tuple[int, bool]:
    """Check an existing run directory and bring its manifest up to date.

    Returns ``(exit_code, provisional)``; a non-zero code means stop.

    Two things happen here that :meth:`RunDir.assert_compatible` cannot do on its own.

    ``decoys.n_decoys`` is normalised to the stored value before comparing. It is a
    generation-stage field — it belongs in the regeneration key, since a run's ensemble size
    is part of what the run *is* — but it is the one generation field whose change does not
    invalidate a single stored decoy: seeds are ``base_seed + decoy_id``, so a 3-decoy
    ensemble contains the 2-decoy one verbatim. Treating a larger ``--n-decoys`` as
    incompatible would make resume impossible, which is the defect this whole contract
    exists to retire (``run_pipeline.py:241``).

    Input digests are compared explicitly, because ``assert_compatible`` recomputes the
    regeneration key *from the stored manifest's own inputs* — a receptor edited in place
    under an unchanged path is invisible to it, and would silently mix two structures'
    decoys in one ensemble.
    """
    manifest = run_dir.read_manifest()
    stored_n = manifest.generation.get("decoys", {}).get("n_decoys", settings.decoys.n_decoys)
    as_stored = settings.model_copy(
        update={"decoys": settings.decoys.model_copy(update={"n_decoys": stored_n})}
    )
    try:
        report = run_dir.assert_compatible(
            as_stored, pyrosetta_version, allow_mismatch=allow_mismatch
        )
    except RegenerationRequired as exc:
        print(
            f"atomfrust {NAME}: {run_dir.root} was generated under different settings.\n"
            f"{exc.report.explain()}\n"
            "Use a new --run-dir, or --allow-mismatch to proceed provisionally.",
            file=sys.stderr,
        )
        return exc.exit_code, False

    provisional = report.verdict == "requires_regeneration"
    if provisional:
        print(
            f"atomfrust {NAME}: proceeding despite a generation-stage difference; "
            "outputs are stamped provisional.",
            file=sys.stderr,
        )

    stored_inputs = dict(manifest.inputs.get("systems", {}))
    changed = [
        f"  systems.{sid}: stored={stored_inputs[sid]!r} requested={new!r}  [generation]"
        for sid, new in digests["systems"].items()
        if sid in stored_inputs and stored_inputs[sid] != new
    ]
    if changed and not allow_mismatch:
        print(
            f"atomfrust {NAME}: input bytes differ from the stored run.\n" + "\n".join(changed),
            file=sys.stderr,
        )
        return RegenerationRequired.exit_code, False

    added = {sid: d for sid, d in digests["systems"].items() if sid not in stored_inputs}
    grew = settings.decoys.n_decoys > stored_n
    if (added or grew) and not provisional:
        stored_inputs.update(added)
        recorded = settings.model_copy(
            update={
                "decoys": settings.decoys.model_copy(
                    update={"n_decoys": max(stored_n, settings.decoys.n_decoys)}
                )
            }
        )
        build_manifest(
            manifest.run_id, recorded, manifest.created_utc, {"systems": stored_inputs}
        ).write(run_dir.manifest_path)
        run_dir.settings_path.write_text(recorded.to_yaml())
    return 0, provisional


def _generate_one_system(
    spec: SystemSpec,
    settings: Settings,
    run_dir: RunDir,
    system: SystemDir,
    units: list[WorkUnit],
    shard: ShardSpec,
    axes: list[str],
    provisional: bool,
) -> None:
    target = settings.decoys.n_decoys * len(axes)
    done_before = sum(len(system.completed_decoy_ids(axis)) for axis in axes)
    system.update_status(
        state="running",
        n_decoys_done=done_before,
        n_decoys_target=target,
        axes=axes,
        shard=str(shard),
        provisional=provisional,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

    structures_dir = None
    if settings.runtime.save_structures:
        structures_dir = str(system.decoys / "structures")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if units:
        writer = _energy_writer(system, shard.index, flush_every=10)
        with writer:

            def on_result(unit: WorkUnit, result: dict[str, Any]) -> None:
                structure = result["index_row"].get("structure_path")
                if structure is not None:
                    # Relative to the run root, so a copied or moved run directory still
                    # resolves. The worker only knows absolute paths.
                    result["index_row"]["structure_path"] = str(
                        Path(structure).relative_to(run_dir.root)
                    )
                writer.add(
                    unit.decoy_id,
                    result["pair_id"],
                    result["e_direct"],
                    result["e_fa_rep"],
                    axis=unit.axis,
                )
                rows.append(result["index_row"])
                _log(
                    run_dir,
                    event="decoy",
                    system_id=unit.system_id,
                    axis=unit.axis,
                    decoy_id=unit.decoy_id,
                    wall_s=result["index_row"].get("wall_s"),
                )
                if settings.runtime.progress:
                    print(
                        f"{unit.system_id} {unit.axis} decoy {unit.decoy_id} "
                        f"({len(rows)}/{len(units)})",
                        file=sys.stderr,
                    )

            execute(
                units,
                _decoy_task,
                task_args=(
                    spec.to_yaml(),
                    settings.to_yaml(),
                    structures_dir,
                ),
                workers=settings.runtime.workers,
                initializer=_worker_init,
                on_result=on_result,
            )

    _write_index(system, rows)
    done = sum(len(system.completed_decoy_ids(axis)) for axis in axes)
    system.update_status(
        state="complete" if done >= target else "partial",
        n_decoys_done=done,
        n_decoys_target=target,
        axes=axes,
        shard=str(shard),
        provisional=provisional,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    _log(
        run_dir,
        event="system_complete",
        system_id=spec.system_id,
        generated=len(rows),
        n_decoys_done=done,
        n_decoys_target=target,
        wall_s=time.perf_counter() - started,
    )


def _worker_init() -> None:
    """Initialise PyRosetta in a spawned worker before any pose is built."""
    from atomfrust.pose import ensure_init

    ensure_init()
