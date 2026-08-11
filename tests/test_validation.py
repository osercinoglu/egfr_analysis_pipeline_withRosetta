"""E5/F1/F4/F5 tests — the case registry, the `validate` command, and the real cases.

Two tiers, drawn where the cost is:

The **unit** tier never touches PyRosetta, the repository's data or a real case. It builds a
throwaway registry of fake cases — one that PASSes, one that FAILs, one that SKIPs — and
exercises the plumbing that decides what the command does with them: listing, selection,
exit codes, unknown names, and the rule that a SKIP is not a failure. That plumbing is what
E5 delivers; the science lives in the cases themselves.

The **integration** tier runs F1, F4 and F5 for real. Each is written so it either measures
its quantity or SKIPs cleanly, so the tier is green on a machine with no PyRosetta and no
`dvc pull` — and, importantly, the tests assert *that* property rather than assuming it:
every real case is also run against an empty root, where it must SKIP rather than raise.

No test generates a decoy. F1 and F4 read 50-decoy prototype ensembles that are already on
disk and F5 needs none at all, so the whole file is seconds, not minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from atomfrust.cli import validate
from atomfrust.validation import cases as case_module
from atomfrust.validation.cases import CaseResult, ValidationCase

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ fake cases


class _AlwaysPass(ValidationCase):
    """A case that measures 1.0 and expects 1.0."""

    def __init__(self) -> None:
        super().__init__(
            name="FAKEPASS",
            summary="always passes",
            expected={"value": 1.0},
            tolerance={"value": 0.1},
        )

    def measure(self, root: Path, **options) -> CaseResult:
        return self.measured({"value": 1.0, "note": "diagnostic"}, "measured 1.0")


class _AlwaysFail(ValidationCase):
    """A case whose measurement has drifted away from its stored expectation."""

    def __init__(self) -> None:
        super().__init__(
            name="FAKEFAIL",
            summary="always fails",
            expected={"value": 1.0},
            tolerance={"value": 0.1},
        )

    def measure(self, root: Path, **options) -> CaseResult:
        return self.measured({"value": 5.0}, "measured 5.0")


class _AlwaysSkip(ValidationCase):
    """A case whose inputs are absent."""

    def __init__(self) -> None:
        super().__init__(name="FAKESKIP", summary="always skips", expected={"value": 1.0})

    def measure(self, root: Path, **options) -> CaseResult:
        return self.skip("no data")


class _Explodes(ValidationCase):
    """A case with a bug in it."""

    def __init__(self) -> None:
        super().__init__(name="FAKEBOOM", summary="raises", expected={"value": 1.0})

    def measure(self, root: Path, **options) -> CaseResult:
        raise RuntimeError("something went wrong inside the case")


@pytest.fixture
def fake_registry(monkeypatch):
    """Replace the real registry so the CLI tests are independent of the real cases."""
    registry = {c.name: c for c in (_AlwaysPass(), _AlwaysFail(), _AlwaysSkip(), _Explodes())}
    monkeypatch.setattr(case_module, "CASES", registry)
    return registry


def _run(**fields) -> tuple[int, argparse.Namespace]:
    """Invoke `validate` through its own parser, as `main` would."""
    parser = argparse.ArgumentParser()
    validate.register(parser.add_subparsers())
    defaults = {"case": None, "list": False, "describe": False, "root": Path("."), "json": None}
    defaults.update(fields)
    args = argparse.Namespace(**defaults)
    return validate.run(args), args


# ------------------------------------------------------------------------ unit


@pytest.mark.unit
def test_registered_cases_are_the_three_this_step_owns():
    """F1, F4 and F5 exist, are uniquely named and describe themselves."""
    names = {c.name for c in case_module.all_cases()}
    assert {"F1", "F4", "F5"} <= names
    for name in ("F1", "F4", "F5"):
        case = case_module.get_case(name)
        assert case.summary
        assert "Proves" in case.description or "proves" in case.description
        assert case.expected, f"{name} has no stored expectation, so it is not a regression test"


@pytest.mark.unit
def test_lookup_is_case_insensitive_and_registration_refuses_duplicates(fake_registry):
    assert case_module.get_case("fakepass") is fake_registry["FAKEPASS"]
    with pytest.raises(ValueError, match="already registered"):
        case_module.register(_AlwaysPass())


@pytest.mark.unit
def test_unknown_case_names_the_known_ones(fake_registry):
    with pytest.raises(KeyError) as excinfo:
        case_module.get_case("F99")
    message = str(excinfo.value)
    assert "F99" in message
    for name in fake_registry:
        assert name in message


@pytest.mark.unit
def test_pass_and_fail_verdicts_come_from_the_stored_expectation(fake_registry):
    assert case_module.run_case("FAKEPASS").status == "PASS"

    failed = case_module.run_case("FAKEFAIL")
    assert failed.status == "FAIL"
    # The report must carry both numbers, or a failure is not actionable.
    assert "5" in failed.detail and "1" in failed.detail
    assert failed.measured == {"value": 5.0}


@pytest.mark.unit
def test_a_case_that_raises_fails_rather_than_propagating(fake_registry):
    result = case_module.run_case("FAKEBOOM")
    assert result.status == "FAIL"
    assert "RuntimeError" in result.detail


@pytest.mark.unit
def test_list_prints_every_case_and_exits_zero(fake_registry, capsys):
    code, _ = _run(list=True)
    assert code == 0
    out = capsys.readouterr().out
    for name, case in fake_registry.items():
        assert name in out and case.summary in out


@pytest.mark.unit
def test_selecting_one_case_runs_only_that_case(fake_registry, capsys):
    code, _ = _run(case=["FAKEPASS"])
    assert code == 0
    out = capsys.readouterr().out
    assert "FAKEPASS" in out
    assert "FAKEFAIL" not in out
    assert "1 passed, 0 failed, 0 skipped" in out


@pytest.mark.unit
def test_a_failing_case_sets_the_exit_code(fake_registry):
    assert _run(case=["FAKEFAIL"])[0] == 1
    # ... and one failure among several still fails the command.
    assert _run(case=["FAKEPASS", "FAKEFAIL"])[0] == 1


@pytest.mark.unit
def test_a_skipped_case_does_not_set_the_exit_code(fake_registry, capsys):
    code, _ = _run(case=["FAKESKIP"])
    assert code == 0
    assert "SKIP" in capsys.readouterr().out


@pytest.mark.unit
def test_no_selection_runs_every_case(fake_registry, capsys):
    code, _ = _run()
    assert code == 1  # the registry contains failing cases
    out = capsys.readouterr().out
    assert "1 passed, 2 failed, 1 skipped" in out


@pytest.mark.unit
def test_unknown_case_name_exits_two_and_lists_the_known_ones(fake_registry, capsys):
    code, _ = _run(case=["NOSUCHCASE"])
    assert code == 2, "a mistyped case name must not look like a scientific failure"
    out = capsys.readouterr().out
    assert "NOSUCHCASE" in out and "FAKEPASS" in out


@pytest.mark.unit
def test_describe_prints_the_docstring(fake_registry, capsys):
    code, _ = _run(case=["FAKEPASS"], describe=True)
    assert code == 0
    assert "A case that measures 1.0" in capsys.readouterr().out


@pytest.mark.unit
def test_json_output_round_trips(fake_registry, tmp_path):
    target = tmp_path / "nested" / "results.json"
    code, _ = _run(case=["FAKEPASS", "FAKEFAIL"], json=target)
    assert code == 1
    payload = json.loads(target.read_text())
    assert [entry["name"] for entry in payload] == ["FAKEPASS", "FAKEFAIL"]
    assert [entry["status"] for entry in payload] == ["PASS", "FAIL"]
    assert payload[0]["measured"]["value"] == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("name", ["F1", "F4", "F5"])
def test_real_cases_skip_on_an_empty_root(tmp_path, name):
    """No data, no crash: the case reports SKIP and says what it wanted."""
    result = case_module.get_case(name).run(tmp_path)
    assert result.status == "SKIP"
    assert result.detail


@pytest.mark.unit
@pytest.mark.parametrize("name", ["F1", "F5"])
def test_cases_needing_pyrosetta_skip_without_it(monkeypatch, name):
    """On a machine with no PyRosetta these two report SKIP, never FAIL and never raise."""
    monkeypatch.setattr(case_module, "_pyrosetta_available", lambda: False)
    result = case_module.get_case(name).run(REPO_ROOT)
    assert result.status == "SKIP", result.detail


@pytest.mark.unit
def test_f4_needs_no_pyrosetta():
    """F4 reads two stored parquets and a PDB, so it must not consult PyRosetta at all."""
    result = case_module.get_case("F4").run(REPO_ROOT)
    assert result.status in ("PASS", "SKIP"), result.detail
    if result.status == "SKIP":
        assert "PyRosetta" not in result.detail


# ----------------------------------------------------------------- integration


def _require(case_name: str) -> CaseResult:
    """Run a real case against the repository, skipping the test when it SKIPs.

    A SKIP is the case telling us its inputs or PyRosetta are absent, which is a property of
    the machine rather than a defect — so the test skips with the same message instead of
    failing and instead of silently passing.
    """
    result = case_module.get_case(case_name).run(REPO_ROOT)
    if result.status == "SKIP":
        pytest.skip(f"{case_name}: {result.detail}")
    return result


@pytest.mark.integration
def test_f1_lysozyme_core_is_more_minimally_frustrated_than_the_surface():
    result = _require("F1")
    assert result.status == "PASS", result.detail
    measured = result.measured
    assert measured["core_fraction"] > measured["exposed_fraction"]
    assert measured["ci_low"] > 0.0, "the bootstrap CI on the difference must exclude zero"
    assert measured["n_core_contacts"] >= 30 and measured["n_exposed_contacts"] >= 30


@pytest.mark.integration
def test_f4_apo_control_reports_a_ligand_blind_index():
    result = _require("F4")
    assert result.status == "PASS", result.detail
    measured = result.measured
    assert measured["ligand_blind"] is True
    # Structural, not statistical: partner lists are protein-only, so the native reference
    # cannot depend on the ligand at all.
    assert measured["e_native_max_abs_delta"] == 0.0
    assert measured["e_native_n_differing"] == 0
    # The ligand's whole influence arrives through the decoys, so those must differ.
    assert measured["decoy_mean_n_differing"] == measured["n_contacts"]


@pytest.mark.integration
def test_f5_pair_energies_agree_with_both_recomputations():
    """Bounded to three structures; the CLI default covers the plan's twenty."""
    case = case_module.get_case("F5")
    result = case.run(REPO_ROOT, n_structures=3)
    if result.status == "SKIP":
        pytest.skip(f"F5: {result.detail}")
    assert result.status == "PASS", result.detail
    measured = result.measured
    assert measured["max_rel_dev_pair_api"] < 0.01
    assert measured["max_rel_dev_etable_atomic"] < 0.01
    assert measured["max_abs_edgeless_energy"] == 0.0
    assert measured["n_edgeless_pairs_checked"] > 0, "the edgeless assertion must not be vacuous"
    assert measured["n_atom_pair_comparisons"] > 0
    # The genuinely independent route is the atom-level one, and it must be exact to far
    # better than the 1% gate — anything near the gate means the two routes have diverged.
    assert measured["max_rel_dev_etable_atomic"] < 1e-9
