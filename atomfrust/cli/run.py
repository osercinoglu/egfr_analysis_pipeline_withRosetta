"""``atomfrust run`` — ``prepare`` + ``generate-decoys`` + ``analyze``, in one process.

Plan step E4. The accept criterion is *"equals E1+E2+E3 composed"*, so this module composes
and computes nothing: every flag is the stage's own flag, every decision is made by the
stage's own :func:`run`, and the only code here is the plumbing that carries one stage's
output into the next stage's input.

**The parser is built from the stages' parsers, not written out again.** :func:`register`
clones the ``argparse`` actions that :mod:`~atomfrust.cli.prepare`,
:mod:`~atomfrust.cli.generate_decoys` and :mod:`~atomfrust.cli.analyze` register, so a flag
added to any of them appears here with the same spelling, type, default and help text on the
next import — and a flag *removed* there disappears here. A hand-written union would be a
second declaration of thirty-odd flags that drifts the first time one of them changes, which
is exactly the class of bug this command exists to avoid.

Three dests genuinely collide between stages, and the collisions are resolved once, here,
rather than silently by ``argparse``'s conflict handler:

``-o``/``--out``
    ``prepare`` means "root directory to write prepared systems into" (required);
    ``analyze`` means "name of the analysis directory". ``analyze`` keeps the flag, because
    that is the one a ``run`` invocation actually chooses; ``prepare``'s becomes
    ``--prepared-dir``, and defaults to ``<run-dir>/prepared`` so neither of the plan's §5
    worked examples has to pass it.
``--chains``
    ``prepare`` takes a repeatable list, ``generate-decoys`` a single comma-separated
    string. The list form is kept and joined back into a string for ``generate-decoys``.
``--axis``
    ``generate-decoys`` takes a comma-separated string (which axes to *generate*),
    ``analyze`` a repeatable list (which axes to *read*). The string form is kept and handed
    to ``analyze`` as a one-element list, which its own ``_split_list`` re-splits on commas.

``--ligand`` differs only in its default (``[]`` vs ``None``), so the list form is kept and
normalised to ``None`` when empty — ``SystemSpec.from_pdb`` treats an empty list as "this
system has no components", which is not what an omitted flag means.

**Exit codes pass through untouched.** The first stage to fail stops the pipeline and its
code is returned verbatim. In particular **3 survives**: it is the run-directory contract's
"this needs a new decoy ensemble, do not merge" verdict from ``generate-decoys`` and
``analyze``, and flattening it to 1 would turn a precise, recoverable answer into a generic
failure.

**Handover.** ``generate-decoys`` reads the spec ``prepare`` *wrote*
(``<prepared>/<system_id>/system.spec.yaml``), not the one the user passed: that file is the
resolved spec, with absolute paths pointing at the receptor copy whose digest ``prepare``
verified. Several prepared systems are gathered into one ``SystemSet`` file, because
``generate-decoys`` takes a single ``--spec``.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from atomfrust.cli import analyze, generate_decoys, prepare

__all__ = [
    "NAME",
    "HELP",
    "STAGES",
    "register",
    "run",
    "stage_parsers",
    "prepare_namespace",
    "generate_namespace",
    "analyze_namespace",
]

NAME = "run"
HELP = "prepare + generate-decoys + analyze"

#: Malformed flag combinations argparse cannot express, and a handover that failed between
#: two stages that each succeeded. Stage failures return the *stage's* code, never this one.
EXIT_USAGE = 2

#: Stage key -> module. The key is what ``--stop-after`` takes; the module's ``NAME`` is what
#: is printed, so ``generate`` and ``generate-decoys`` never have to be kept in sync by hand.
STAGES: dict[str, Any] = {
    "prepare": prepare,
    "generate": generate_decoys,
    "analyze": analyze,
}

#: Combined ``SystemSet`` written when ``prepare`` produced more than one system, since
#: ``generate-decoys`` accepts a single ``--spec``.
SYSTEMS_FILENAME = "systems.yaml"

#: ``prepare``'s ``-o/--out`` under a non-colliding name. Only the option strings, dest and
#: requiredness change; the type and the action come from ``prepare``'s own declaration.
_PREPARED_DIR_FLAG = "--prepared-dir"


# ----------------------------------------------------------------------------- parser


def stage_parsers() -> dict[str, argparse.ArgumentParser]:
    """The three stage parsers, built by the stages themselves.

    A throwaway ``subparsers`` action is used because ``register`` is the only entry point a
    stage exposes to its own flags; nothing here is dispatched through it.
    """
    holder = argparse.ArgumentParser(add_help=False)
    subparsers = holder.add_subparsers()
    for module in STAGES.values():
        module.register(subparsers)
    return {key: subparsers.choices[module.NAME] for key, module in STAGES.items()}


def _copyable_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [
        action
        for action in parser._actions
        if not isinstance(action, (argparse._HelpAction, argparse._VersionAction))
        and action.option_strings
    ]


def _renamed(action: argparse.Action) -> argparse.Action:
    """``prepare``'s output root, moved off ``-o`` so ``analyze`` can keep it."""
    clone = copy.copy(action)
    clone.option_strings = [_PREPARED_DIR_FLAG]
    clone.dest = "prepared_dir"
    clone.required = False
    clone.default = None
    clone.help = (
        "root directory for the prepared systems (prepare's -o); "
        "default <run-dir>/prepared"
    )
    return clone


def _clone_stage(
    parser: argparse.ArgumentParser,
    source: argparse.ArgumentParser,
    title: str,
    taken_dests: set[str],
    taken_flags: set[str],
) -> None:
    """Add ``source``'s options to ``parser``, skipping anything already claimed.

    Mutually exclusive groups are rebuilt rather than flattened, so ``--thresholds`` and
    ``--thresholds-mode`` stay mutually exclusive here for the same reason they are there.

    A dest is only "claimed" by an *earlier* stage. Within one stage two options routinely
    share a dest — ``analyze``'s ``--exclude-fa-rep`` / ``--include-fa-rep`` are a
    ``store_true``/``store_false`` pair on ``exclude_fa_rep`` — and dropping the second of
    them would delete half a flag pair while leaving its partner working.
    """
    group = parser.add_argument_group(f"{title} stage")
    mutex_index = {
        id(action): number
        for number, mutex in enumerate(source._mutually_exclusive_groups)
        for action in mutex._group_actions
    }
    required_mutex = {
        number: mutex.required
        for number, mutex in enumerate(source._mutually_exclusive_groups)
    }
    mutexes: dict[int, Any] = {}
    claimed_here: set[str] = set()

    for action in _copyable_actions(source):
        if action.dest == "out" and title == prepare.NAME:
            action = _renamed(action)
        if action.dest in taken_dests or taken_flags.intersection(action.option_strings):
            continue
        number = mutex_index.get(id(action))
        container: Any = group
        if number is not None:
            if number not in mutexes:
                # Not `setdefault`: it evaluates its default eagerly, so every member of a
                # group after the first would register an extra empty one on the parser, and
                # argparse refuses to format a usage line containing an empty group.
                mutexes[number] = group.add_mutually_exclusive_group(
                    required=required_mutex[number]
                )
            container = mutexes[number]
        container._add_action(action)
        claimed_here.add(action.dest)
        taken_flags.update(action.option_strings)
    taken_dests.update(claimed_here)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        NAME,
        help=HELP,
        description=(
            "Run the whole pipeline: prepare a system, generate its decoy ensemble, then "
            "analyse it. Every flag is the corresponding stage's own flag, and the first "
            "stage to fail stops the run and returns its exit code unchanged."
        ),
        epilog=(
            "examples:\n"
            "  atomfrust run --spec specs/5GMP.yaml --run-dir runs/egfr "
            "--shell-ref any_heavy --shell-A 6.0 --n-decoys 250\n"
            "  atomfrust run --pdb my_complex.pdb --ligand B:501 --run-dir runs/mine "
            "--n-decoys 250\n"
            "\n"
            "exit codes:\n"
            "  0  every stage that ran succeeded\n"
            "  1  a stage reported a failure\n"
            "  2  arguments argparse cannot reject on its own, or a failed handover\n"
            "  3  a stage needs a new decoy ensemble; the field diff is on stderr\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    control = parser.add_argument_group("pipeline control")
    control.add_argument(
        "--skip-prepare",
        action="store_true",
        help="start at generate-decoys, using --spec/--pdb as given",
    )
    control.add_argument(
        "--stop-after",
        choices=tuple(STAGES),
        default="analyze",
        help="last stage to run (default: analyze, i.e. the whole pipeline)",
    )

    taken_dests = {action.dest for action in parser._actions}
    taken_flags = {flag for action in parser._actions for flag in action.option_strings}
    for key, source in stage_parsers().items():
        _clone_stage(parser, source, STAGES[key].NAME, taken_dests, taken_flags)

    parser.set_defaults(func=run)


# ------------------------------------------------------------------ stage namespaces


def _namespace(
    source: argparse.ArgumentParser, args: argparse.Namespace, **overrides: Any
) -> argparse.Namespace:
    """One stage's namespace, filled from the union namespace.

    The dests come from the stage's parser, so a stage sees exactly the attributes its own
    ``run`` can read and nothing else. ``overrides`` carries the three type adaptations and
    the handover paths; everything else is passed through untouched.
    """
    values: dict[str, Any] = {}
    for action in _copyable_actions(source):
        dest = action.dest
        if dest in overrides:
            values[dest] = overrides[dest]
        else:
            values[dest] = getattr(args, dest, action.default)
    return argparse.Namespace(**values)


def _joined(values: Iterable[str] | None) -> str | None:
    """``--chains A --chains B`` as ``generate-decoys`` wants it: ``"A,B"``, or ``None``."""
    if not values:
        return None
    return ",".join(str(v) for v in values)


def prepare_namespace(
    args: argparse.Namespace, prepared_root: Path
) -> argparse.Namespace:
    return _namespace(stage_parsers()["prepare"], args, out=Path(prepared_root))


def generate_namespace(
    args: argparse.Namespace, spec: Path | None
) -> argparse.Namespace:
    """``spec`` is the file ``prepare`` wrote; ``None`` keeps the user's own input flags.

    With a prepared spec the input flags must be dropped, not merely ignored:
    ``generate-decoys`` refuses ``--ligand`` together with ``--spec`` (the spec is the source
    of truth for components), and it would be refusing a flag the user aimed at ``prepare``.
    """
    source = stage_parsers()["generate"]
    if spec is not None:
        return _namespace(
            source, args, spec=Path(spec), pdb=None, ligand=None, chains=None,
            system_id=None,
        )
    return _namespace(
        source,
        args,
        ligand=list(args.ligand) or None,
        chains=_joined(args.chains),
    )


def analyze_namespace(args: argparse.Namespace) -> argparse.Namespace:
    # `--axis a,b` is one string here and a list there; analyze re-splits on commas itself.
    axis = [args.axis] if getattr(args, "axis", None) else []
    return _namespace(stage_parsers()["analyze"], args, axis=axis)


# --------------------------------------------------------------------------- handover


def _prepared_spec(prepare_args: argparse.Namespace, prepared_root: Path) -> Path:
    """The spec file(s) ``prepare`` just wrote, as one path ``generate-decoys`` can take.

    The system ids come from ``prepare``'s own loader rather than from a directory listing:
    a glob would also pick up systems prepared by an earlier invocation into the same root,
    and silently generate decoys for them.
    """
    systems = prepare._load_systems(prepare_args)
    paths = [
        Path(prepared_root) / spec.system_id / prepare.SPEC_FILENAME for spec in systems
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "prepare reported success but did not write: " + ", ".join(missing)
        )
    if len(paths) == 1:
        return paths[0]

    combined = Path(prepared_root) / SYSTEMS_FILENAME
    combined.write_text(
        yaml.safe_dump(
            {"systems": [yaml.safe_load(path.read_text()) for path in paths]},
            sort_keys=True,
            default_flow_style=False,
        )
    )
    return combined


# ---------------------------------------------------------------------------- summary


def _report(name: str, code: int, seconds: float, stream: Any) -> None:
    status = "ok" if code == 0 else f"FAILED exit {code}"
    print(f"atomfrust run: {name:<16} {status:<14} {seconds:8.1f}s", file=stream)


# -------------------------------------------------------------------------------- run


def run(args: argparse.Namespace) -> int:
    """Compose the three stages. Returns the first non-zero stage exit code, unchanged."""
    if args.skip_prepare and args.stop_after == "prepare":
        print(
            "atomfrust run: --skip-prepare with --stop-after prepare leaves nothing to run",
            file=sys.stderr,
        )
        return EXIT_USAGE
    order = [key for key in STAGES if not (key == "prepare" and args.skip_prepare)]
    order = order[: order.index(args.stop_after) + 1]

    prepared_root = (
        Path(args.prepared_dir)
        if args.prepared_dir is not None
        else Path(args.run_dir) / "prepared"
    )
    spec: Path | None = None

    started = time.perf_counter()
    for key in order:
        module = STAGES[key]
        if key == "prepare":
            stage_args = prepare_namespace(args, prepared_root)
        elif key == "generate":
            stage_args = generate_namespace(args, spec)
        else:
            stage_args = analyze_namespace(args)

        stage_started = time.perf_counter()
        code = int(module.run(stage_args) or 0)
        _report(module.NAME, code, time.perf_counter() - stage_started, sys.stdout)
        if code:
            return code

        if key == "prepare" and "generate" in order:
            try:
                spec = _prepared_spec(stage_args, prepared_root)
            except Exception as exc:  # noqa: BLE001 - reported, never raised past here
                print(f"atomfrust run: {exc}", file=sys.stderr)
                return EXIT_USAGE

    _report("total", 0, time.perf_counter() - started, sys.stdout)
    return 0
