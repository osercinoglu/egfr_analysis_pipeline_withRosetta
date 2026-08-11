"""``atomfrust validate`` — run the Stage F validation cases against stored expectations.

Plan step E5. A thin adapter: every case lives in :mod:`atomfrust.validation.cases` with its
own expectation, and this module only chooses which ones to run, prints them and decides the
exit code.

```
atomfrust validate --list              # what exists and what each case claims
atomfrust validate --case F1           # one case
atomfrust validate                     # every registered case
atomfrust validate --case F5 --describe    # the full docstring: what it proves
```

Exit codes: ``0`` when nothing FAILed (SKIPs included — a case that could not run is not
evidence), ``1`` when at least one case FAILed, ``2`` when the invocation itself was wrong
(an unknown case name), which is kept distinct so a CI job can tell "the science moved" from
"the command was mistyped".

``--root`` exists because every case reads repository-relative artefacts (``data/``,
``results/``, ``checkpoints/``). It defaults to the current directory, matching every other
command in this package, and pointing it at an empty directory is the supported way to see
each case SKIP cleanly.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from atomfrust.validation import cases as case_module

NAME = "validate"
HELP = "run a named validation case against its stored expectation"

#: Printed under each case name by ``--list``; the full docstring needs ``--describe``.
_LIST_INDENT = " " * 4


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        NAME,
        help=HELP,
        description=(
            "Run the Stage F validation cases. Each case measures one quantity and compares "
            "it against a value stored in the source, so a re-run is a regression test "
            "rather than a print-out."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        metavar="NAME",
        help="case to run (repeatable). Omit to run every registered case.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the registered cases and what each one claims, then exit",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print each selected case's full docstring — what it proves and what it does not",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="repository root the cases read their artefacts from (default: .)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write the results as JSON",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    registered = case_module.all_cases()

    if getattr(args, "list", False):
        if not registered:
            print("no validation cases are registered")
            return 0
        width = max(len(c.name) for c in registered)
        for case in registered:
            print(f"{case.name.ljust(width)}  {case.summary}")
        return 0

    names = getattr(args, "case", None)
    if names:
        selected = []
        for name in names:
            try:
                selected.append(case_module.get_case(name))
            except KeyError as exc:
                print(_one_line(exc))
                return 2
    else:
        selected = registered

    if not selected:
        print("no validation cases are registered")
        return 0

    if getattr(args, "describe", False):
        for case in selected:
            print(f"=== {case.name} — {case.summary}")
            print(textwrap.indent(case.description, _LIST_INDENT))
            print()

    results = [case.run(getattr(args, "root", Path("."))) for case in selected]
    _report(results)

    json_path = getattr(args, "json", None)
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps([r.to_dict() for r in results], indent=2, sort_keys=True) + "\n"
        )
        print(f"\nwrote {json_path}")

    return 1 if any(r.status == "FAIL" for r in results) else 0


def _report(results: list) -> None:
    """One block per case: the verdict line, then measured-vs-expected for checked keys.

    Only keys the case actually asserts are tabulated. Everything else it measured is
    printed as a diagnostics line, because a number nobody compares against anything is
    context, not evidence, and the layout should say which is which.
    """
    width = max([len(r.name) for r in results] + [4])
    print(f"{'case'.ljust(width)}  status  detail")
    print(f"{'-' * width}  ------  ------")
    for result in results:
        print(f"{result.name.ljust(width)}  {result.status:<6}  {result.detail}")

    for result in results:
        measured = result.measured or {}
        expected = result.expected or {}
        tolerance = result.tolerance or {}
        if not measured:
            continue
        print(f"\n{result.name} — measured vs expected")
        for key in expected:
            got = measured.get(key, "<not measured>")
            tol = tolerance.get(key)
            suffix = f" +/- {tol}" if tol is not None else ""
            print(f"  {key:<32} {_format(got):>16}   expected {_format(expected[key])}{suffix}")
        extra = [k for k in measured if k not in expected]
        if extra:
            print(
                "  diagnostics: "
                + ", ".join(f"{k}={_format(measured[k])}" for k in extra)
            )

    counts = {s: sum(1 for r in results if r.status == s) for s in ("PASS", "FAIL", "SKIP")}
    print(
        f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped"
    )


def _format(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split()).strip("'")
