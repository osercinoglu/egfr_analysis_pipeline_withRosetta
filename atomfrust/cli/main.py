"""`atomfrust` console entry point.

Step B1 provides the parser, `--version`, and the dispatch table. Subcommands are
registered as they land (plan steps E1-E7); until then the table is empty and `--help`
reports what is planned rather than pretending the commands exist.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

from atomfrust import __version__

# Modules providing subcommands. Each exposes `register(subparsers)`, which adds its own
# parser and sets `func` as that parser's default — so this table stays a list of names and
# never grows knowledge of any subcommand's flags. A module may register more than one
# command (`verify` also registers `metrics-selftest`).
SUBCOMMAND_MODULES = (
    "atomfrust.cli.prepare",
    "atomfrust.cli.generate_decoys",
    "atomfrust.cli.analyze",
    "atomfrust.cli.run",
    "atomfrust.cli.validate",
    "atomfrust.cli.analysis_tools",
    "atomfrust.cli.verify",
)

# Kept for the tests and for `--help`; populated at parser-build time.
SUBCOMMANDS: dict[str, tuple[str, Callable[[argparse._SubParsersAction], None]]] = {}

# Documented but not yet implemented. Listed in `--help` so the CLI is honest about
# what exists rather than silently offering nothing.
PLANNED: list[tuple[str, str, str]] = []


def _load_subcommands() -> list[Any]:
    """Import each subcommand module.

    Imports are deferred to parser-build time rather than module import time so that
    `atomfrust --version` stays instant and, more importantly, so a broken or
    dependency-missing subcommand cannot take down the whole CLI.
    """
    import importlib

    modules = []
    for path in SUBCOMMAND_MODULES:
        try:
            modules.append(importlib.import_module(path))
        except ModuleNotFoundError:
            # Listed but not yet written. Silent: `--help` already reports what is planned.
            continue
        except Exception as exc:  # noqa: BLE001 - one bad subcommand must not kill the CLI
            print(f"warning: subcommand {path} unavailable: {exc}", file=sys.stderr)
    return modules


def build_parser() -> argparse.ArgumentParser:
    epilog = (
        "not yet implemented:\n"
        + "\n".join(f"  {name:<16} {help_}  [{step}]" for name, help_, step in PLANNED)
        if PLANNED
        else ""
    )
    parser = argparse.ArgumentParser(
        prog="atomfrust",
        description=(
            "Atomistic local frustration for protein-ligand, protein-only and "
            "protein-protein complexes."
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"atomfrust {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for module in _load_subcommands():
        module.register(subparsers)
        SUBCOMMANDS[module.NAME] = (module.HELP, module.register)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
