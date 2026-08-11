"""E4 tests — `atomfrust run`.

The unit tier covers the two things a composed command can get wrong that its parts cannot:
the flag union (every stage flag reachable, with the right *type* handed to each stage, and
no stage seeing a flag that is not its own) and the exit-code chain (first failure stops the
pipeline, and its code is returned verbatim — `3` above all, which means "this needs a new
decoy ensemble", not "something went wrong").

The integration tier proves composition actually happened: one `run` on 5GMP at two decoys
and `--relax min` must leave both a decoy store and an analysis directory behind, which no
single stage produces on its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from atomfrust.cli import analyze, prepare
from atomfrust.cli import run as cmd

PROCESSED = Path("data/processed/5GMP_clean.pdb")
PARAMS = Path("data/ligands/params/F62.params")
SYSTEM_ID = "5GMP_F62"


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse through a real subparsers action, the way `main` will."""
    parser = argparse.ArgumentParser(prog="atomfrust")
    subparsers = parser.add_subparsers(dest="command")
    cmd.register(subparsers)
    return parser.parse_args(argv)


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atomfrust")
    subparsers = parser.add_subparsers(dest="command")
    cmd.register(subparsers)
    return subparsers.choices[cmd.NAME]


def _fake_spec(directory: Path, system_id: str = "S1") -> Path:
    """A spec that loads but names no real coordinates. Enough for the unit tier."""
    directory.mkdir(parents=True, exist_ok=True)
    spec = directory / "spec.yaml"
    spec.write_text(
        f"system_id: {system_id}\n"
        f"receptor:\n  path: {directory / 'fake.pdb'}\n"
        "pocket:\n  mode: whole\n"
    )
    return spec


def _fake_prepared(prepared_root: Path, spec: Path, system_id: str = "S1") -> Path:
    """What a successful `prepare` would have written, for a stubbed one."""
    out = prepared_root / system_id / prepare.SPEC_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.read_text())
    return out


class _Recorder:
    """Stand-in stage `run()` that records its namespace and returns a sentinel code."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, argparse.Namespace]] = []
        self.codes: dict[str, int] = {}

    def stub(self, name: str):
        def _stub(args: argparse.Namespace) -> int:
            self.calls.append((name, args))
            return self.codes.get(name, 0)

        return _stub

    def install(self, monkeypatch, **codes: int) -> "_Recorder":
        self.codes.update(codes)
        for key, module in cmd.STAGES.items():
            monkeypatch.setattr(module, "run", self.stub(key))
        return self

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def args(self, name: str) -> argparse.Namespace:
        return next(a for n, a in self.calls if n == name)


# ================================================================== unit tier
# ------------------------------------------------------------ flag composition


@pytest.mark.unit
def test_the_contract_is_the_one_main_dispatches_on():
    assert cmd.NAME == "run"
    assert cmd.HELP
    parser = _run_parser()
    assert parser.get_default("func") is cmd.run


@pytest.mark.unit
def test_every_stage_flag_is_reachable_from_run():
    """The drift guard.

    The union is *built* from the stage parsers, so this cannot fail today; it fails the day
    someone re-declares a flag here by hand and then changes it in only one of the two
    places. `prepare`'s `-o/--out` is the single documented rename and is covered below.
    """
    union = set(_run_parser()._option_string_actions)
    for key, source in cmd.stage_parsers().items():
        for action in source._actions:
            for flag in action.option_strings:
                if key == "prepare" and flag in ("-o", "--out"):
                    continue
                assert flag in union, f"{flag} ({key}) is not reachable from `run`"

    for own in ("--skip-prepare", "--stop-after", "--prepared-dir"):
        assert own in union


@pytest.mark.unit
def test_prepare_output_root_is_the_only_renamed_flag(tmp_path):
    """`-o` means two different things in `prepare` and `analyze`; `analyze` keeps it.

    A run chooses an analysis name far more often than it chooses where the prepared copy
    lands, and the prepared copy has a sane default (`<run-dir>/prepared`) while an analysis
    id does not.
    """
    spec = _fake_spec(tmp_path)
    args = _parse(
        ["run", "--spec", str(spec), "--run-dir", str(tmp_path / "r"), "-o", "s5_rank"]
    )
    assert args.out == "s5_rank"
    assert args.prepared_dir is None
    assert cmd.analyze_namespace(args).out == "s5_rank"
    assert cmd.prepare_namespace(args, tmp_path / "p").out == tmp_path / "p"


@pytest.mark.unit
def test_mutually_exclusive_groups_survive_the_clone(tmp_path):
    """Cloned actions must keep the constraints their own parser gave them."""
    spec = _fake_spec(tmp_path)
    base = ["run", "--run-dir", str(tmp_path / "r")]

    for argv in (
        [*base],  # neither --spec nor --pdb: the required group
        [*base, "--spec", str(spec), "--pdb", "x.pdb"],
        [*base, "--spec", str(spec), "--thresholds", "0.5,-0.5",
         "--thresholds-mode", "quantile"],
        [*base, "--spec", str(spec), "--exclude-fa-rep", "--include-fa-rep"],
    ):
        with pytest.raises(SystemExit):
            _parse(argv)


@pytest.mark.unit
def test_each_stage_sees_its_own_flags_only_with_the_types_it_expects(tmp_path):
    """The three genuine collisions, resolved: `--chains`, `--axis` and `-o/--out`.

    `--chains` is a repeatable list in `prepare` and one comma-separated string in
    `generate-decoys`; `--axis` is a comma-separated string in `generate-decoys` and a
    repeatable list in `analyze`. Handing either the other's form would be a `TypeError` deep
    inside a loader, or worse, a silent single-character chain list.
    """
    spec = _fake_spec(tmp_path)
    args = _parse(
        [
            "run", "--spec", str(spec), "--run-dir", str(tmp_path / "r"),
            "--chains", "A", "--chains", "B",
            "--comp-id", "A:9001=634", "--label", "affinity_pM=23",
            "--n-decoys", "7", "--relax", "min", "--axis", "identity", "--workers", "2",
            "--shell-A", "6.0", "--index", "robust_z", "--include-fa-rep",
            "-o", "myanalysis",
        ]
    )

    prepared = tmp_path / "prep"
    prepare_args = cmd.prepare_namespace(args, prepared)
    assert prepare_args.chains == ["A", "B"]
    assert prepare_args.comp_id == ["A:9001=634"]
    assert prepare_args.label == ["affinity_pM=23"]
    assert prepare_args.out == prepared
    assert not hasattr(prepare_args, "n_decoys")  # a generation flag prepare cannot read

    handover = _fake_prepared(prepared, spec)
    generate_args = cmd.generate_namespace(args, handover)
    assert generate_args.spec == handover
    assert (generate_args.pdb, generate_args.ligand, generate_args.chains) == (None,) * 3
    assert (generate_args.n_decoys, generate_args.relax, generate_args.axis) == (
        7, "min", "identity",
    )
    assert generate_args.workers == 2
    assert not hasattr(generate_args, "shell_A")

    analyze_args = cmd.analyze_namespace(args)
    assert analyze_args.axis == ["identity"]  # analyze re-splits this on commas itself
    assert (analyze_args.n_decoys, analyze_args.index) == (7, "robust_z")
    assert analyze_args.exclude_fa_rep is False
    assert analyze_args.shell_A == 6.0
    assert not hasattr(analyze_args, "relax")


@pytest.mark.unit
def test_without_a_prepare_stage_the_input_flags_pass_through_unchanged(tmp_path):
    args = _parse(
        [
            "run", "--skip-prepare", "--pdb", "my.pdb", "--ligand", "B:501",
            "--chains", "A", "--chains", "B", "--run-dir", str(tmp_path / "r"),
        ]
    )
    generate_args = cmd.generate_namespace(args, None)
    assert generate_args.pdb == Path("my.pdb")
    assert generate_args.ligand == ["B:501"]
    assert generate_args.chains == "A,B"  # the string form generate-decoys splits


@pytest.mark.unit
def test_an_omitted_ligand_flag_is_none_not_an_empty_list(tmp_path):
    """`SystemSpec.from_pdb` reads `[]` as "this system has no components" and `None` as
    "autodetect them", so `prepare`'s `default=[]` cannot be forwarded verbatim."""
    args = _parse(["run", "--skip-prepare", "--pdb", "my.pdb", "--run-dir", str(tmp_path)])
    assert args.ligand == []
    assert cmd.generate_namespace(args, None).ligand is None


# --------------------------------------------------------- exit-code propagation


@pytest.mark.unit
def test_all_three_stages_run_in_order_when_each_succeeds(tmp_path, monkeypatch, capsys):
    spec = _fake_spec(tmp_path)
    prepared = tmp_path / "prepared"
    _fake_prepared(prepared, spec)
    recorder = _Recorder().install(monkeypatch)

    args = _parse(
        ["run", "--spec", str(spec), "--run-dir", str(tmp_path / "r"),
         "--prepared-dir", str(prepared)]
    )
    assert cmd.run(args) == 0
    assert recorder.names == ["prepare", "generate", "analyze"]
    # The handover: generate-decoys reads the spec prepare wrote, not the one passed in.
    assert recorder.args("generate").spec == prepared / "S1" / prepare.SPEC_FILENAME


@pytest.mark.unit
@pytest.mark.parametrize("code", [1, 2, 3])
def test_a_failing_prepare_stops_the_pipeline_with_its_own_code(tmp_path, monkeypatch, code):
    spec = _fake_spec(tmp_path)
    recorder = _Recorder().install(monkeypatch, prepare=code)
    args = _parse(["run", "--spec", str(spec), "--run-dir", str(tmp_path / "r")])

    assert cmd.run(args) == code
    assert recorder.names == ["prepare"]


@pytest.mark.unit
@pytest.mark.parametrize("failing", ["generate", "analyze"])
@pytest.mark.parametrize("code", [1, 3])
def test_a_failing_stage_returns_its_code_unchanged(tmp_path, monkeypatch, failing, code):
    """Exit 3 is the run-directory contract's "requires regeneration" verdict.

    Flattening it to 1 would turn a precise, recoverable answer — regenerate, do not merge —
    into a generic failure, and every caller that branches on it would lose.
    """
    spec = _fake_spec(tmp_path)
    recorder = _Recorder().install(monkeypatch, **{failing: code})
    args = _parse(
        ["run", "--skip-prepare", "--spec", str(spec), "--run-dir", str(tmp_path / "r")]
    )

    assert cmd.run(args) == code
    assert recorder.names == ["generate", "analyze"][: ["generate", "analyze"].index(failing) + 1]


@pytest.mark.unit
def test_the_first_failure_wins_even_when_a_later_stage_would_also_fail(tmp_path, monkeypatch):
    spec = _fake_spec(tmp_path)
    recorder = _Recorder().install(monkeypatch, generate=3, analyze=1)
    args = _parse(
        ["run", "--skip-prepare", "--spec", str(spec), "--run-dir", str(tmp_path / "r")]
    )

    assert cmd.run(args) == 3
    assert recorder.names == ["generate"]


@pytest.mark.unit
def test_a_prepare_that_reports_success_without_writing_its_spec_is_an_error(
    tmp_path, monkeypatch, capsys
):
    """The handover is checked, not assumed: a missing spec must not reach generate-decoys."""
    spec = _fake_spec(tmp_path)
    recorder = _Recorder().install(monkeypatch)
    args = _parse(["run", "--spec", str(spec), "--run-dir", str(tmp_path / "r")])

    assert cmd.run(args) == cmd.EXIT_USAGE
    assert recorder.names == ["prepare"]
    assert "did not write" in capsys.readouterr().err


# ------------------------------------------------------------ partial pipelines


@pytest.mark.unit
def test_stop_after_expresses_a_partial_pipeline(tmp_path, monkeypatch):
    spec = _fake_spec(tmp_path)
    prepared = tmp_path / "prepared"
    _fake_prepared(prepared, spec)

    for stop, expected in (
        ("prepare", ["prepare"]),
        ("generate", ["prepare", "generate"]),
        ("analyze", ["prepare", "generate", "analyze"]),
    ):
        recorder = _Recorder().install(monkeypatch)
        args = _parse(
            ["run", "--spec", str(spec), "--run-dir", str(tmp_path / "r"),
             "--prepared-dir", str(prepared), "--stop-after", stop]
        )
        assert cmd.run(args) == 0
        assert recorder.names == expected, stop


@pytest.mark.unit
def test_skip_prepare_starts_at_generation(tmp_path, monkeypatch):
    spec = _fake_spec(tmp_path)
    recorder = _Recorder().install(monkeypatch)
    args = _parse(
        ["run", "--skip-prepare", "--spec", str(spec), "--run-dir", str(tmp_path / "r")]
    )

    assert cmd.run(args) == 0
    assert recorder.names == ["generate", "analyze"]
    # With no prepare stage there is no handover: the user's own --spec is used as given.
    assert recorder.args("generate").spec == spec


@pytest.mark.unit
def test_skipping_and_stopping_at_prepare_leaves_nothing_to_run(tmp_path, monkeypatch, capsys):
    spec = _fake_spec(tmp_path)
    recorder = _Recorder().install(monkeypatch)
    args = _parse(
        ["run", "--skip-prepare", "--stop-after", "prepare", "--spec", str(spec),
         "--run-dir", str(tmp_path / "r")]
    )

    assert cmd.run(args) == cmd.EXIT_USAGE
    assert recorder.names == []
    assert "nothing to run" in capsys.readouterr().err


# ------------------------------------------------------------------- reporting


@pytest.mark.unit
def test_each_stage_reports_one_line_with_its_wall_clock(tmp_path, monkeypatch, capsys):
    spec = _fake_spec(tmp_path)
    prepared = tmp_path / "prepared"
    _fake_prepared(prepared, spec)
    _Recorder().install(monkeypatch)
    args = _parse(
        ["run", "--spec", str(spec), "--run-dir", str(tmp_path / "r"),
         "--prepared-dir", str(prepared)]
    )

    assert cmd.run(args) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("atomfrust run:")]
    assert len(lines) == 4  # three stages plus the total
    for line, name in zip(lines, ["prepare", "generate-decoys", "analyze", "total"]):
        assert name in line
        assert line.rstrip().endswith("s"), line
        assert "ok" in line


@pytest.mark.unit
def test_a_failed_stage_is_reported_with_its_code(tmp_path, monkeypatch, capsys):
    spec = _fake_spec(tmp_path)
    _Recorder().install(monkeypatch, generate=3)
    args = _parse(
        ["run", "--skip-prepare", "--spec", str(spec), "--run-dir", str(tmp_path / "r")]
    )

    assert cmd.run(args) == 3
    out = capsys.readouterr().out
    assert "generate-decoys" in out and "exit 3" in out
    assert "total" not in out  # a stopped pipeline has no total to report


# =========================================================== integration tier


def _spec_file(directory: Path) -> Path:
    """The same 5GMP/F62 spec the E2 suite uses, so the two tiers exercise one system."""
    path = directory / "5GMP.yaml"
    path.write_text(
        f"system_id: {SYSTEM_ID}\n"
        f"receptor:\n  path: {PROCESSED.resolve()}\n  chains: [A]\n"
        "ligands:\n"
        "  - selector: {chain: A, resseq: 1101, comp_id: F62}\n"
        f"    params: {PARAMS.resolve()}\n"
        "pocket:\n  mode: ligand_shell\n"
    )
    return path


@pytest.mark.integration
def test_one_run_prepares_generates_and_analyses(tmp_path):
    """E4's accept criterion: `run` equals E1 + E2 + E3 composed.

    Two decoys and `--relax min`, because what is under test is that all three stages ran
    against one another's output — not the ensemble's statistics, which the E2 and E3 suites
    already cover at the same cost.
    """
    missing = [str(p) for p in (PROCESSED, PARAMS) if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")

    spec = _spec_file(tmp_path)
    run_dir = tmp_path / "run"
    code = cmd.run(
        _parse(
            ["run", "--spec", str(spec), "--run-dir", str(run_dir),
             "--n-decoys", "2", "--relax", "min", "--workers", "1"]
        )
    )
    assert code == 0

    # E1 ran: a prepared directory, with the receptor copy whose digest prepare verified.
    prepared = run_dir / "prepared" / SYSTEM_ID
    assert (prepared / prepare.SPEC_FILENAME).exists()
    assert (prepared / prepare.RECEPTOR_FILENAME).exists()

    # E2 ran: the decoy store exists and holds both decoys.
    system = run_dir / "systems" / SYSTEM_ID
    parts = sorted((system / "decoys" / "energies").glob("part-*.parquet"))
    assert parts, "no decoy energies were written"
    assert (system / "decoys" / "index.parquet").exists()
    status = json.loads((system / "STATUS.json").read_text())
    assert (status["state"], status["n_decoys_done"]) == ("complete", 2)

    # E3 ran: exactly one analysis directory, with the per-pair table and the summary.
    analyses = sorted(p for p in (system / "analyses").iterdir() if p.is_dir())
    assert len(analyses) == 1, [p.name for p in analyses]
    assert (analyses[0] / analyze.CONTACTS_FILENAME).exists()
    summary = json.loads((analyses[0] / analyze.SUMMARY_FILENAME).read_text())
    assert summary["system_id"] == SYSTEM_ID
    assert summary["n_decoys_used"] == 2
    assert summary["n_pairs_selected"] > 0

    # And the generation half really was driven by the prepared spec, not the input one.
    stored = (system / "inputs" / "system.spec.yaml").read_text()
    assert str(prepared / prepare.RECEPTOR_FILENAME) in stored


@pytest.mark.integration
def test_stop_after_generate_leaves_no_analysis(tmp_path):
    """U6 through `run`: the partial pipeline is expressible and really does stop."""
    missing = [str(p) for p in (PROCESSED, PARAMS) if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")

    spec = _spec_file(tmp_path)
    run_dir = tmp_path / "run"
    assert cmd.run(
        _parse(
            ["run", "--spec", str(spec), "--run-dir", str(run_dir), "--stop-after",
             "generate", "--n-decoys", "2", "--relax", "min", "--workers", "1"]
        )
    ) == 0

    system = run_dir / "systems" / SYSTEM_ID
    assert sorted((system / "decoys" / "energies").glob("part-*.parquet"))
    assert list((system / "analyses").iterdir()) == []
