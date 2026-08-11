"""Executable validation cases (plan Stage F).

Every claim this project makes about its own correctness is either a number with a stored
expectation or it is prose. Stage F is the first kind: each case is an object with a name, a
docstring saying what it proves, a ``run()`` that measures something, and an expectation
pinned in code so a re-run is a **regression test** rather than a print-out.

The public surface is deliberately tiny — :class:`~atomfrust.validation.cases.CaseResult`,
:class:`~atomfrust.validation.cases.ValidationCase` and the registry — because
``atomfrust validate`` (``atomfrust/cli/validate.py``) is a thin adapter over it and nothing
else should need to know how a case is built.

Statuses are ``PASS`` / ``FAIL`` / ``SKIP``. Only ``FAIL`` sets a non-zero exit code: a case
that cannot run because its data or PyRosetta is absent is not evidence of anything and must
not be reported as if it were.

Importing this package pulls in numpy, pandas and scipy but **never PyRosetta** — cases that
need it import it inside ``run()`` and ``SKIP`` when it is missing.
"""

from __future__ import annotations

from atomfrust.validation.cases import (
    CASES,
    CaseResult,
    Status,
    ValidationCase,
    all_cases,
    get_case,
    register,
    run_case,
)

__all__ = [
    "CASES",
    "CaseResult",
    "Status",
    "ValidationCase",
    "all_cases",
    "get_case",
    "register",
    "run_case",
]

# Importing the package must publish every case, or `validate --list` silently shows a
# subset and a missing case looks like a case that passed.
from atomfrust.validation import anchors as _anchors  # noqa: E402,F401
