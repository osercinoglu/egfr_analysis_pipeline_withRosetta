"""E3 tests — ``atomfrust analyze`` against a synthetic run directory. No PyRosetta.

The run directory is built here with the same public API generation would use
(:meth:`RunDir.create`, :meth:`SystemDir.write_graph`, :meth:`SystemDir.energy_writer`), so
these tests exercise the contract of plan §4 rather than a fixture invented for them.
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

from atomfrust.cli import analyze as cli
from atomfrust.graph import Geometry, Node, build_graph
from atomfrust.provenance import build_manifest, capture_environment, sha256_file
from atomfrust.runstore import EXIT_REQUIRES_REGENERATION, RunDir
from atomfrust.settings import Settings

pytestmark = pytest.mark.unit

CREATED = "2026-08-11T09:00:00Z"
SYSTEM_ID = "SYNTH_LIG"
N_DECOYS = 8


# ------------------------------------------------------------------ synthetic run dir


def _system() -> tuple[list[Node], Geometry]:
    """Six protein residues on a line plus one three-atom ligand beside residues 2-4.

    Spacing is 3 A so the 12 A Ca superset admits pairs up to four residues apart, and the
    ligand sits close enough to be incident to several of them — the analysis needs at
    least one ligand pair or the pocket descriptors would all be unmeasured.
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
                resname="ALA",
                name1="A",
                n_heavy=2,
                rel_sasa=0.25 + 0.05 * k,
            )
        )
        ca.append([3.0 * k, 0.0, 0.0])
        cb.append([3.0 * k, 1.5, 0.0])
        heavy.extend([[3.0 * k, 0.0, 0.0], [3.0 * k, 1.5, 0.0]])
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
    heavy.extend([[6.0, 3.0, 0.0], [7.5, 3.0, 0.0], [9.0, 3.0, 0.0]])
    heavy_node.extend([6, 6, 6])

    geometry = Geometry(
        heavy_xyz=np.array(heavy, dtype=float),
        heavy_node=np.array(heavy_node, dtype=np.int64),
        ca_xyz=np.array(ca, dtype=float),
        cb_xyz=np.array(cb, dtype=float),
    )
    return nodes, geometry


def _make_run(root: Path, settings: Settings | None = None) -> RunDir:
    settings = settings or Settings()
    receptor = root / "receptor.pdb"
    receptor.parent.mkdir(parents=True, exist_ok=True)
    receptor.write_text("ATOM      1  N   ALA A 101\n")

    manifest = build_manifest(
        "synth", settings, CREATED, {"receptor_sha256": sha256_file(receptor)}
    )
    run = RunDir.create(root / "run", manifest, settings, capture_environment())
    system = run.system(SYSTEM_ID).ensure()

    nodes, geometry = _system()
    node_frame, pairs = build_graph(nodes, geometry, settings)
    system.write_graph(node_frame, pairs)

    rng = np.random.default_rng(20260811)
    pair_ids = pairs["pair_id"].to_numpy()
    native = pd.DataFrame(
        {
            "pair_id": pair_ids.astype(np.int32),
            "e_direct": rng.normal(-1.0, 0.5, len(pair_ids)).astype(np.float32),
            "e_fa_rep": rng.random(len(pair_ids)).astype(np.float32) * 0.3,
            "has_edge": np.ones(len(pair_ids), dtype=bool),
        }
    )
    system.write_native_energies(native)

    with system.energy_writer(shard=0, flush_every=3) as writer:
        for decoy_id in range(N_DECOYS):
            writer.add(
                decoy_id=decoy_id,
                pair_id=pair_ids,
                e_direct=native["e_direct"].to_numpy() + rng.normal(0.4, 0.4, len(pair_ids)),
                e_fa_rep=native["e_fa_rep"].to_numpy() + rng.random(len(pair_ids)) * 0.1,
            )
    system.update_status(state="complete", n_decoys_done=N_DECOYS, n_decoys_target=N_DECOYS)
    return run


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return _make_run(tmp_path).root


# ------------------------------------------------------------------------ invocation


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse through a real subparser, so the registration contract is exercised too."""
    parser = argparse.ArgumentParser(prog="atomfrust")
    cli.register(parser.add_subparsers(dest="command"))
    return parser.parse_args([cli.NAME, *argv])


def _analyze(*argv: str) -> int:
    return cli.run(_parse(list(argv)))


def _analyses(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "systems" / SYSTEM_ID / "analyses").iterdir())


def _contacts(directory: Path) -> pd.DataFrame:
    return pd.read_parquet(directory / cli.CONTACTS_FILENAME)


# ------------------------------------------------------------------- registration


def test_registration_contract():
    assert cli.NAME == "analyze"
    assert cli.HELP
    args = _parse(["--run-dir", "R"])
    assert args.func is cli.run
    assert args.run_dir == Path("R")


# ------------------------------------------------------------------- end to end


def test_analyze_writes_the_three_documented_files(run_dir: Path, capsys):
    assert _analyze("--run-dir", str(run_dir)) == 0

    directories = _analyses(run_dir)
    assert len(directories) == 1
    directory = directories[0]
    assert directory.name == cli.analysis_id(Settings())

    for name in (cli.MANIFEST_FILENAME, cli.CONTACTS_FILENAME, cli.SUMMARY_FILENAME):
        assert (directory / name).is_file(), name

    contacts = _contacts(directory)
    assert list(contacts.columns) == list(cli.CONTACTS_COLUMNS)
    assert len(contacts) > 0
    assert (contacts["n_decoys"] == N_DECOYS).all()
    assert set(contacts["class"]) <= {
        "minimally_frustrated", "neutral", "highly_frustrated"
    }
    assert contacts["F"].notna().all()

    summary = json.loads((directory / cli.SUMMARY_FILENAME).read_text())
    from atomfrust.analyze.aggregate import COVARIATES, DESCRIPTORS

    assert set(summary["covariates"]) == set(COVARIATES)
    assert {k.split("__")[1] for k in summary["descriptors"]} == set(DESCRIPTORS)
    assert summary["n_decoys_used"] == N_DECOYS
    assert sum(summary["class_counts"].values()) == len(contacts)

    manifest = json.loads((directory / cli.MANIFEST_FILENAME).read_text())
    stored = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["regeneration_key"] == stored["regeneration_key"]
    assert manifest["settings"]["manybody"]["mode"] == "chen_literal"
    assert manifest["compatibility"]["verdict"] == "ok"
    assert manifest["provisional"] is False

    assert SYSTEM_ID in capsys.readouterr().out


def test_pocket_is_not_empty_so_descriptors_are_measured(run_dir: Path):
    """A ligand-incident pocket is what the covariates describe; an empty one is all NaN."""
    assert _analyze("--run-dir", str(run_dir)) == 0
    summary = json.loads((_analyses(run_dir)[0] / cli.SUMMARY_FILENAME).read_text())
    assert summary["covariates"]["n_contacts_total"] > 0
    assert summary["covariates"]["n_pocket_residues"] > 0
    assert summary["warnings"] == []


# ----------------------------------------------------------------- no PyRosetta


def test_a_full_analyze_never_imports_pyrosetta(tmp_path: Path):
    """User request 7, and the constraint the whole run-directory contract exists to serve.

    Checked in a subprocess because an import that already happened in this interpreter
    cannot be un-happened, and PyRosetta *is* installed on this machine — so an in-process
    assertion would pass for the wrong reason.
    """
    root = _make_run(tmp_path).root
    code = textwrap.dedent(
        f"""
        import argparse, sys
        from atomfrust.cli import analyze

        parser = argparse.ArgumentParser()
        analyze.register(parser.add_subparsers())
        args = parser.parse_args(["analyze", "--run-dir", {str(root)!r}])
        if analyze.run(args) != 0:
            sys.exit(9)
        leaked = sorted(m for m in sys.modules if m.split(".")[0] == "pyrosetta")
        print(leaked)
        sys.exit(1 if leaked else 0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
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


# ------------------------------------------------- two settings, two analyses


def test_two_many_body_modes_coexist_and_disagree(run_dir: Path):
    """Re-specifying the formula is a re-analysis: two directories, both intact, different F."""
    assert _analyze("--run-dir", str(run_dir), "--manybody", "chen_literal") == 0
    assert _analyze("--run-dir", str(run_dir), "--manybody", "pair_retained") == 0

    directories = _analyses(run_dir)
    assert len(directories) == 2, [d.name for d in directories]

    a, b = (_contacts(d) for d in directories)
    pd.testing.assert_series_equal(a["pair_id"], b["pair_id"])
    assert not np.allclose(a["F"].to_numpy(), b["F"].to_numpy())

    modes = {
        json.loads((d / cli.MANIFEST_FILENAME).read_text())["settings"]["manybody"]["mode"]
        for d in directories
    }
    assert modes == {"chen_literal", "pair_retained"}


def test_analysis_id_is_a_digest_of_the_analysis_settings():
    base = Settings()
    other = Settings.model_validate({"analysis": {"index": "robust_z"}})
    generation_only = Settings.model_validate({"decoys": {"n_decoys": 500}})

    assert cli.analysis_id(base) == cli.analysis_id(Settings())
    assert cli.analysis_id(base) != cli.analysis_id(other)
    assert cli.analysis_id(base) == cli.analysis_id(generation_only)


def test_out_names_the_analysis_directory(run_dir: Path):
    assert _analyze("--run-dir", str(run_dir), "-o", "analyses/s5_rank") == 0
    assert [d.name for d in _analyses(run_dir)] == ["s5_rank"]


# ------------------------------------------------------------- decoy prefix


def test_n_decoys_takes_an_exact_prefix_without_touching_the_store(run_dir: Path):
    store = run_dir / "systems" / SYSTEM_ID / "decoys" / "energies"
    before = {p.name: sha256_file(p) for p in sorted(store.glob("part-*.parquet"))}

    assert _analyze("--run-dir", str(run_dir)) == 0
    assert _analyze("--run-dir", str(run_dir), "--n-decoys", "4") == 0

    after = {p.name: sha256_file(p) for p in sorted(store.glob("part-*.parquet"))}
    assert after == before, "analysis must never write to the decoy store"

    by_n = {}
    for directory in _analyses(run_dir):
        frame = _contacts(directory)
        by_n[int(frame["n_decoys"].iloc[0])] = frame
    assert sorted(by_n) == [4, N_DECOYS]
    assert not np.allclose(by_n[4]["F"].to_numpy(), by_n[N_DECOYS]["F"].to_numpy())
    # The native side is a property of the structure, not of how many decoys were read.
    assert np.allclose(by_n[4]["E_native"].to_numpy(), by_n[N_DECOYS]["E_native"].to_numpy())


def test_prefix_matches_a_run_that_only_ever_had_that_many_decoys(tmp_path: Path):
    """"N as a prefix" is exact, not a subsample — the property the sweep depends on."""
    full = _make_run(tmp_path / "full").root
    assert _analyze("--run-dir", str(full), "--n-decoys", "4") == 0
    prefix = _contacts(_analyses(full)[0])

    # A run holding only the first four decoys, written the same way.
    truncated = _make_run(tmp_path / "short").root
    store = truncated / "systems" / SYSTEM_ID / "decoys" / "energies"
    for path in store.glob("part-*.parquet"):
        frame = pd.read_parquet(path)
        kept = frame[frame["decoy_id"] < 4]
        kept.to_parquet(path, index=False) if len(kept) else path.unlink()
    assert cli.run(_parse(["--run-dir", str(truncated)])) == 0
    short = pd.read_parquet(
        sorted((truncated / "systems" / SYSTEM_ID / "analyses").iterdir())[0]
        / cli.CONTACTS_FILENAME
    )
    assert np.allclose(prefix["F"].to_numpy(), short["F"].to_numpy())


# -------------------------------------------------------------- compatibility


def test_analysis_stage_changes_never_require_regeneration(run_dir: Path):
    for argv in (
        ["--index", "rank_percentile"],
        ["--contact-def", "heavy_min", "--contact-cutoff-A", "6.0"],
        ["--shell-A", "5.0"],
        ["--seq-sep-min", "3"],
        ["--include-fa-rep"],
        ["--thresholds", "0.5,-0.5"],
        ["--thresholds-mode", "quantile"],
    ):
        assert _analyze("--run-dir", str(run_dir), *argv) == 0, argv

    verdicts = {
        json.loads((d / cli.MANIFEST_FILENAME).read_text())["compatibility"]["verdict"]
        for d in _analyses(run_dir)
    }
    assert verdicts <= {"ok", "analysis_only"}


def test_a_generation_stage_change_exits_three(run_dir: Path, tmp_path: Path, capsys):
    config = tmp_path / "regen.yaml"
    config.write_text("energy:\n  score_function: ref2015_cart\n")

    assert _analyze("--run-dir", str(run_dir), "--config", str(config)) == (
        EXIT_REQUIRES_REGENERATION
    )
    stderr = capsys.readouterr().err
    assert "energy.score_function" in stderr
    assert "ref2015_cart" in stderr
    assert "Traceback" not in stderr
    assert _analyses(run_dir) == [], "nothing may be written for a refused request"


def test_allow_mismatch_marks_every_output_provisional(run_dir: Path, tmp_path: Path):
    config = tmp_path / "regen.yaml"
    config.write_text("energy:\n  score_function: ref2015_cart\n")

    assert _analyze(
        "--run-dir", str(run_dir), "--config", str(config), "--allow-mismatch"
    ) == 0
    directory = _analyses(run_dir)[0]
    assert json.loads((directory / cli.MANIFEST_FILENAME).read_text())["provisional"] is True
    assert json.loads((directory / cli.SUMMARY_FILENAME).read_text())["provisional"] is True


def test_a_cutoff_beyond_the_superset_names_regeneration(run_dir: Path, capsys):
    """The superset is the ceiling on post-hoc re-selection; nothing stored can answer this."""
    code = _analyze("--run-dir", str(run_dir), "--contact-cutoff-A", "40.0")
    assert code != 0
    stderr = capsys.readouterr().err
    assert "regeneration" in stderr
    assert "superset" in stderr
    assert "Traceback" not in stderr


# --------------------------------------------------------------- argument errors


def test_unknown_descriptor_is_rejected_with_the_registry(run_dir: Path, capsys):
    assert _analyze("--run-dir", str(run_dir), "--descriptor", "nonesuch") == cli.EXIT_USAGE
    assert "frac_minimal" in capsys.readouterr().err


def test_unknown_system_is_rejected(run_dir: Path, capsys):
    assert _analyze("--run-dir", str(run_dir), "--systems", "NOPE") == cli.EXIT_USAGE
    assert SYSTEM_ID in capsys.readouterr().err


def test_a_directory_without_a_manifest_is_not_a_run(tmp_path: Path, capsys):
    assert _analyze("--run-dir", str(tmp_path)) == cli.EXIT_ANALYSIS_ERROR
    assert "not a run directory" in capsys.readouterr().err
