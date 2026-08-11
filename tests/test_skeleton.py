"""B1 smoke tests: the package installs, imports, and exposes a working entry point.

Deliberately marked `unit` — no PyRosetta, no network, no filesystem. This tier must stay
runnable on a machine that has never installed PyRosetta.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_package_imports_and_reports_a_real_version():
    import atomfrust

    assert atomfrust.__version__
    # "0.0.0+unknown" is the fallback when the distribution metadata is missing, which
    # means the editable install did not take. Catching that here is the point.
    assert atomfrust.__version__ != "0.0.0+unknown", (
        "atomfrust is importable but has no distribution metadata — "
        "run `pip install -e . --no-deps`"
    )


def test_cli_version_flag_matches_package_version(capsys):
    import atomfrust
    from atomfrust.cli.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert atomfrust.__version__ in capsys.readouterr().out


def test_bare_invocation_prints_help_and_succeeds(capsys):
    from atomfrust.cli.main import main

    assert main([]) == 0
    assert "atomfrust" in capsys.readouterr().out


def test_console_script_is_on_path():
    """The installed `atomfrust` script runs, not just the importable module."""
    proc = subprocess.run(
        ["atomfrust", "--version"], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "atomfrust" in proc.stdout


def test_importing_atomfrust_does_not_drag_in_pyrosetta():
    """The unit tier must stay usable without PyRosetta installed.

    Importing the package in a subprocess keeps this honest regardless of what other
    tests in the session have already imported.
    """
    code = (
        "import sys; import atomfrust, atomfrust.cli.main; "
        "sys.exit(1 if 'pyrosetta' in sys.modules else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, "importing atomfrust pulled in pyrosetta"
