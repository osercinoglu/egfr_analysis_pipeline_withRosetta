"""E6 tests — ``converge``, ``strata``, ``report`` and ``calibrate``. No PyRosetta.

The run directory is built with the same public API generation would use
(:meth:`RunDir.create`, :meth:`SystemDir.write_graph`, :meth:`SystemDir.energy_writer`), the
way ``tests/test_cli_analyze.py`` does, so these tests exercise the run-directory contract of
plan §4 rather than a fixture invented for them.

Two systems, not one: ``calibrate`` pools across a cohort and ``strata`` bins pockets by
quantile, and neither behaviour is visible with a single pocket.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from atomfrust.analyze.strata import STRATUM_AXES
from atomfrust.cli import analysis_tools as cli
from atomfrust.graph import Geometry, Node, build_graph
from atomfrust.provenance import build_manifest, capture_environment, sha256_file
from atomfrust.runstore import RunDir
from atomfrust.settings import Settings

pytestmark = pytest.mark.unit

CREATED = "2026-08-11T09:00:00Z"
SYSTEM_IDS = ("SYNTH_A", "SYNTH_B")
N_DECOYS = 8

#: The real prototype summary, 61 rows. DVC-tracked, so it may be absent on a fresh clone.
SUMMARY_CSV = Path(__file__).resolve().parents[1] / "results" / "egfr_frustration_summary.csv"


# ------------------------------------------------------------------ synthetic run dir


def _system(spacing: float) -> tuple[list[Node], Geometry]:
    """Six protein residues on a line plus one three-atom ligand beside residues 2-4.

    ``spacing`` is what makes the two systems different pockets: a wider spacing thins the
    contact graph, so burial, degree and the decoy sigma all move.
    """
    nodes: list[Node] = []
    ca, cb, heavy, heavy_node = [], [], [], []
    for k in range(6):
        nodes.append(
            Node(
                node_id=f"A:{101 + k}",
                pose_resnum=k + 1,
                kind="protein",
                chain="A",
                resseq=101 + k,
                icode=" ",
                resname="ALA" if k % 2 else "SER",
                name1="A" if k % 2 else "S",
                n_heavy=2,
                rel_sasa=0.25 + 0.05 * k,
            )
        )
        ca.append([spacing * k, 0.0, 0.0])
        cb.append([spacing * k, 1.5, 0.0])
        heavy.extend([[spacing * k, 0.0, 0.0], [spacing * k, 1.5, 0.0]])
        heavy_node.extend([k, k])

    nodes.append(
        Node(
            node_id="L:LIG:1",
            pose_resnum=7,
            kind="ligand",
            chain="B",
            resseq=901,
            icode=" ",
            resname="LIG",
            name1="X",
            ccd_id="LIG",
            rosetta_name="LIG",
            mutable=False,
            n_heavy=3,
        )
    )
    ca.append([np.nan] * 3)
    cb.append([np.nan] * 3)
    heavy.extend(
        [[2 * spacing, 3.0, 0.0], [2.5 * spacing, 3.0, 0.0], [3 * spacing, 3.0, 0.0]]
    )
    heavy_node.extend([6, 6, 6])

    return nodes, Geometry(
        heavy_xyz=np.array(heavy, dtype=float),
        heavy_node=np.array(heavy_node, dtype=np.int64),
        ca_xyz=np.array(ca, dtype=float),
        cb_xyz=np.array(cb, dtype=float),
    )


def _make_run(root: Path, n_decoys: int = N_DECOYS) -> RunDir:
    settings = Settings()
    receptor = root / "receptor.pdb"
    receptor.parent.mkdir(parents=True, exist_ok=True)
    receptor.write_text("ATOM      1  N   ALA A 101\n")

    manifest = build_manifest(
        "synth", settings, CREATED, {"receptor_sha256": sha256_file(receptor)}
    )
    run = RunDir.create(root / "run", manifest, settings, capture_environment())

    for offset, system_id in enumerate(SYSTEM_IDS):
        system = run.system(system_id).ensure()
        nodes, geometry = _system(spacing=3.0 + 0.5 * offset)
        node_frame, pairs = build_graph(nodes, geometry, settings)
        system.write_graph(node_frame, pairs)

        rng = np.random.default_rng(20260811 + offset)
        pair_ids = pairs["pair_id"].to_numpy()
        native = pd.DataFrame(
            {
                "pair_id": pair_ids.astype(np.int32),
                "e_direct": rng.normal(-1.0, 0.5, len(pair_ids)).astype(np.float32),
                "e_fa_rep": (rng.random(len(pair_ids)) * 0.3).astype(np.float32),
                "has_edge": np.ones(len(pair_ids), dtype=bool),
            }
        )
        system.write_native_energies(native)

        with system.energy_writer(shard=0, flush_every=3) as writer:
            for decoy_id in range(n_decoys):
                writer.add(
                    decoy_id=decoy_id,
                    pair_id=pair_ids,
                    e_direct=native["e_direct"].to_numpy()
                    + rng.normal(0.4, 0.4 + 0.2 * offset, len(pair_ids)),
                    e_fa_rep=native["e_fa_rep"].to_numpy() + rng.random(len(pair_ids)) * 0.1,
                )
        system.update_status(state="complete", n_decoys_done=n_decoys, n_decoys_target=n_decoys)
    return run


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return _make_run(tmp_path).root


def _summary_frame(n: int = 12, seed: int = 3) -> pd.DataFrame:
    """A prototype-shaped summary table. ``report`` needs at least four complete rows."""
    rng = np.random.default_rng(seed)
    contacts = rng.integers(480, 740, n).astype(float)
    frac = rng.normal(0.47, 0.02, n)
    return pd.DataFrame(
        {
            "pdb_id": [f"S{k:03d}" for k in range(n)],
            "n_contacts_total": contacts,
            "n_minimally_frustrated": np.round(contacts * frac),
            "n_neutral": np.round(contacts * (1.0 - frac) * 0.9),
            "n_highly_frustrated": np.round(contacts * (1.0 - frac) * 0.1),
            "frac_minimally": frac,
            "log10_affinity_pM": -0.004 * contacts + rng.normal(0.0, 0.6, n),
        }
    )


# ------------------------------------------------------------------------ invocation


def _parse(command: str, argv: list[str]) -> argparse.Namespace:
    """Parse through a real subparser, so the registration contract is exercised too."""
    parser = argparse.ArgumentParser(prog="atomfrust")
    cli.register(parser.add_subparsers(dest="command"))
    return parser.parse_args([command, *argv])


def _invoke(command: str, *argv: str) -> int:
    args = _parse(command, list(argv))
    return args.func(args)


# --------------------------------------------------------------------- registration


def test_one_module_registers_all_four_commands():
    assert cli.NAME == "converge"
    assert (cli.STRATA_NAME, cli.REPORT_NAME, cli.CALIBRATE_NAME) == (
        "strata", "report", "calibrate"
    )
    assert cli.HELP

    assert _parse("converge", ["--run", "R"]).func is cli.run
    assert _parse("strata", ["--runs", "G", "-o", "O"]).func is cli.run_strata
    assert _parse("report", ["--collect", "G", "-o", "O"]).func is cli.run_report
    assert _parse("calibrate", ["--runs", "G", "-o", "O"]).func is cli.run_calibrate


# ------------------------------------------------------------------------- converge


def test_converge_writes_a_curve_and_an_n_star_per_system(run_dir: Path, capsys):
    out = run_dir / "converge"
    assert _invoke("converge", "--run", str(run_dir), "--grid", "4,8", "--n-boot", "20") == 0

    curve = pd.read_csv(out / "convergence.csv")
    assert set(curve["system_id"]) == set(SYSTEM_IDS)
    assert set(curve["n_decoys"]) == {4, 8}
    assert {"spearman_rho", "rho_ci_low", "rho_ci_high", "sigma_relative_error"} <= set(
        curve.columns
    )
    # The reference row is the self-comparison and must read exactly 1.
    reference = curve[curve["is_reference"]]
    assert len(reference) == len(SYSTEM_IDS)
    assert np.allclose(reference["spearman_rho"].to_numpy(), 1.0)

    payload = json.loads((out / "convergence.json").read_text())
    assert payload["command"] == "converge"
    assert payload["grid"] == [4, 8]
    assert {s["system_id"] for s in payload["systems"]} == set(SYSTEM_IDS)
    for system in payload["systems"]:
        assert system["reference_n"] == 8
        assert system["n_decoys_available"] == N_DECOYS
        assert system["n_star"] in (4, 8, None)
    assert "N*" in capsys.readouterr().out


def test_converge_skips_grid_points_beyond_the_stored_decoy_count(run_dir: Path):
    """A stored run is routinely shorter than the sweep; that is not an error."""
    out = run_dir / "converge"
    assert _invoke(
        "converge", "--run", str(run_dir), "--grid", "4,8,250,2000", "--n-boot", "0"
    ) == 0

    curve = pd.read_csv(out / "convergence.csv")
    assert sorted(set(curve["n_decoys"])) == [4, 8], "no point may exceed the 8 stored decoys"

    payload = json.loads((out / "convergence.json").read_text())
    for system in payload["systems"]:
        assert system["skipped_grid_points"] == [250, 2000]


def test_converge_honours_systems_and_out(run_dir: Path, tmp_path: Path):
    out = tmp_path / "elsewhere"
    assert _invoke(
        "converge", "--run", str(run_dir), "--systems", SYSTEM_IDS[0],
        "--grid", "4,8", "--n-boot", "0", "-o", str(out),
    ) == 0
    curve = pd.read_csv(out / "convergence.csv")
    assert set(curve["system_id"]) == {SYSTEM_IDS[0]}
    assert not (run_dir / "converge").exists()


def test_converge_rejects_an_unknown_system_and_a_non_run_directory(run_dir, tmp_path, capsys):
    assert _invoke("converge", "--run", str(run_dir), "--systems", "NOPE") == cli.EXIT_USAGE
    assert SYSTEM_IDS[0] in capsys.readouterr().err

    assert _invoke("converge", "--run", str(tmp_path / "nowhere")) == cli.EXIT_ERROR
    assert "not a run directory" in capsys.readouterr().err


# --------------------------------------------------------------------------- strata


def test_strata_writes_descriptors_sigma_and_redundancy(run_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "strata"
    assert _invoke("strata", "--runs", f"{run_dir}/systems/*", "-o", str(out)) == 0

    descriptors = pd.read_csv(out / "pocket_descriptors.csv")
    assert set(descriptors["system_id"]) == set(SYSTEM_IDS)
    assert {"mean_burial", "frac_polar", "mean_volume", "median_sigma"} <= set(
        descriptors.columns
    )
    assert descriptors["median_sigma"].notna().all()
    for axis in STRATUM_AXES:
        assert f"stratum_{axis}" in descriptors.columns

    sigma = pd.read_csv(out / "sigma_by_stratum.csv")
    assert set(sigma["axis"]) == set(STRATUM_AXES)
    assert set(sigma.columns) >= {"stratum", "n", "mean_sigma", "cv_sigma", "cv_across_strata"}
    assert sigma["n"].sum() >= len(SYSTEM_IDS)

    redundancy = pd.read_csv(out / "axis_redundancy.csv")
    # One stored axis, so every row is the self-comparison — which must read 1.0, the
    # cheapest available check that the per-system pair ordering is shared.
    assert set(redundancy["system_id"]) == set(SYSTEM_IDS)
    assert (redundancy["is_self"]).all()
    assert np.allclose(redundancy["pearson_r"].to_numpy(), 1.0)

    payload = json.loads((out / "strata.json").read_text())
    assert payload["command"] == "strata"
    assert payload["n_systems"] == len(SYSTEM_IDS)
    assert set(payload["cv_across_strata"]) == set(STRATUM_AXES)
    assert "CV of sigma across strata" in capsys.readouterr().out


def test_strata_by_selects_the_reported_axes(run_dir: Path, tmp_path: Path):
    out = tmp_path / "strata"
    assert _invoke(
        "strata", "--runs", str(run_dir), "--by", "burial,volume", "-o", str(out)
    ) == 0
    sigma = pd.read_csv(out / "sigma_by_stratum.csv")
    assert set(sigma["axis"]) == {"burial", "volume"}


def test_strata_rejects_an_unknown_axis_and_an_empty_glob(run_dir, tmp_path, capsys):
    out = tmp_path / "strata"
    assert _invoke("strata", "--runs", str(run_dir), "--by", "charge", "-o", str(out)) == (
        cli.EXIT_USAGE
    )
    assert "charge" in capsys.readouterr().err

    assert _invoke("strata", "--runs", str(tmp_path / "no-such-*"), "-o", str(out)) == (
        cli.EXIT_USAGE
    )
    assert "no path matches" in capsys.readouterr().err


# --------------------------------------------------------------------------- report


def test_report_writes_the_markdown_the_table_and_the_figure(tmp_path: Path, capsys):
    source = tmp_path / "summary.csv"
    _summary_frame().to_csv(source, index=False)
    out = tmp_path / "report"

    assert _invoke(
        "report", "--collect", str(source), "-o", str(out), "--permute", "50", "--n-boot", "50"
    ) == 0

    assert (out / "report.md").is_file()
    assert (out / "confound.png").is_file()
    table = pd.read_csv(out / "report_table.csv")
    assert "frac_minimally" in set(table["descriptor"])
    # The triple, never a bare correlation.
    assert {"raw_r", "partial_r", "ols_coef", "vif", "p_maxT_adjusted", "headline_permitted"} <= (
        set(table.columns)
    )
    assert "headlines withheld" in capsys.readouterr().out


def test_report_collects_analysis_summaries_from_a_run_layout(tmp_path: Path):
    """``--collect`` globs analysis directories, not only CSVs."""
    frame = _summary_frame(n=8)
    for k, row in frame.iterrows():
        directory = tmp_path / "runs" / "r1" / "systems" / f"S{k}" / "analyses" / "default"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps(row.to_dict()))

    out = tmp_path / "report"
    assert _invoke(
        "report",
        "--collect", str(tmp_path / "runs/*/systems/*/analyses/default"),
        "-o", str(out), "--permute", "50", "--n-boot", "50", "--no-plots",
    ) == 0
    table = pd.read_csv(out / "report_table.csv")
    assert int(table["n"].iloc[0]) == 8
    assert not (out / "confound.png").exists()


@pytest.mark.skipif(not SUMMARY_CSV.is_file(), reason="results/ not pulled (dvc pull)")
def test_report_runs_on_the_real_prototype_summary(tmp_path: Path):
    """61 real rows through the real code path.

    Deliberately asserts **nothing about any correlation**: at n = 61 nothing in this table
    is significant (every raw CI spans zero, max-T adjusted p >= 0.69), so an assertion on a
    particular r would be an assertion about noise. What is checked is that the command
    produces its outputs and that the covariate guard was evaluated on every row.
    """
    out = tmp_path / "report"
    assert _invoke(
        "report", "--collect", str(SUMMARY_CSV), "-o", str(out),
        "--permute", "100", "--n-boot", "100",
    ) == 0

    assert (out / "report.md").is_file()
    table = pd.read_csv(out / "report_table.csv")
    assert len(table) >= 1
    assert int(table["n"].iloc[0]) == 61
    assert table["headline_permitted"].isin([True, False]).all()


def test_report_reports_an_empty_glob_as_usage(tmp_path: Path, capsys):
    assert _invoke("report", "--collect", str(tmp_path / "none-*"), "-o", str(tmp_path)) == (
        cli.EXIT_USAGE
    )
    assert "no path matches" in capsys.readouterr().err


# ------------------------------------------------------------------------ calibrate


def test_calibrate_pools_the_cohort_and_writes_a_settings_overlay(run_dir, tmp_path, capsys):
    out = tmp_path / "calibration"
    assert _invoke("calibrate", "--runs", f"{run_dir}/systems/*", "-o", str(out)) == 0

    overlay = yaml.safe_load((out / "thresholds.yaml").read_text())
    classify = overlay["analysis"]["classify"]
    assert classify["mode"] == "quantile"
    assert classify["minimally_frustrated"] > classify["highly_frustrated"]

    payload = json.loads((out / "calibration.json").read_text())
    assert payload["calibration"] == "pooled"
    assert payload["n_systems"] == len(SYSTEM_IDS)
    # Pooled means exactly that: every finite F of every system is in one distribution.
    assert payload["n_values_pooled"] == sum(s["n_finite"] for s in payload["systems"])
    assert payload["thresholds"]["minimally_frustrated"] == classify["minimally_frustrated"]
    assert "pooled" in capsys.readouterr().out


def test_calibrate_refuses_per_system_calibration(run_dir: Path, tmp_path: Path, capsys):
    """The refusal is the feature: per-system quantiles destroy the between-structure signal."""
    out = tmp_path / "calibration"
    assert _invoke(
        "calibrate", "--runs", str(run_dir), "--per-system", "-o", str(out)
    ) == cli.EXIT_USAGE

    stderr = capsys.readouterr().err
    assert "--per-system is refused" in stderr
    assert "pooled" in stderr
    assert "between-structure signal" in stderr
    assert not out.exists(), "a refused request may not write anything"


def test_calibrate_help_states_the_pooling_requirement(capsys):
    """The constraint has to reach the user who never opens the source."""
    with pytest.raises(SystemExit):
        _parse("calibrate", ["--help"])
    help_text = capsys.readouterr().out

    assert "POOLED" in help_text
    assert "NO PER-SYSTEM MODE" in help_text
    assert "REFUSED" in help_text


def test_calibrate_warns_when_the_cohort_is_a_single_system(run_dir, tmp_path, capsys):
    out = tmp_path / "calibration"
    assert _invoke(
        "calibrate", "--runs", f"{run_dir}/systems/{SYSTEM_IDS[0]}", "-o", str(out)
    ) == 0
    assert "one system in the cohort" in capsys.readouterr().out


# ----------------------------------------------------------------------- no PyRosetta


def test_none_of_the_four_commands_imports_pyrosetta(tmp_path: Path):
    """All four are pure analysis over stored files (plan E6: "against stored runs only").

    Checked in a subprocess because an import that already happened in this interpreter
    cannot be un-happened, and PyRosetta *is* installed on this machine — so an in-process
    assertion would pass for the wrong reason.
    """
    root = _make_run(tmp_path).root
    source = tmp_path / "summary.csv"
    _summary_frame().to_csv(source, index=False)

    code = textwrap.dedent(
        f"""
        import argparse, sys
        from atomfrust.cli import analysis_tools

        parser = argparse.ArgumentParser()
        analysis_tools.register(parser.add_subparsers())

        invocations = [
            ["converge", "--run", {str(root)!r}, "--grid", "4,8", "--n-boot", "0"],
            ["strata", "--runs", {str(root)!r}, "-o", {str(tmp_path / "s")!r}],
            ["report", "--collect", {str(source)!r}, "-o", {str(tmp_path / "r")!r},
             "--permute", "20", "--n-boot", "20"],
            ["calibrate", "--runs", {str(root)!r}, "-o", {str(tmp_path / "c")!r}],
        ]
        for argv in invocations:
            args = parser.parse_args(argv)
            if args.func(args) != 0:
                print("failed:", argv)
                sys.exit(9)

        leaked = sorted(m for m in sys.modules if m.split(".")[0] == "pyrosetta")
        print(leaked)
        sys.exit(1 if leaked else 0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # The check is only meaningful if importing PyRosetta would have been possible.
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib.util as u; "
         "raise SystemExit(0 if u.find_spec('pyrosetta') else 7)"],
        timeout=120,
    )
    assert probe.returncode in (0, 7)
    if probe.returncode == 7:
        pytest.skip("pyrosetta is not installed; the negative result is trivially true")
