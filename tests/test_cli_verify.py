"""E7 tests — `atomfrust verify` and `atomfrust metrics-selftest`.

The unit tier builds run directories by hand (no PyRosetta, no data): a manifest, a
receptor, a spec and a params file are all `verify` needs to check digests. The integration
tier is the only place a decoy is actually regenerated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from atomfrust.cli import verify
from atomfrust.provenance import build_manifest, capture_environment, sha256_file
from atomfrust.runstore import RunDir
from atomfrust.settings import Settings

CREATED = "2026-08-11T09:00:00Z"

RECEPTOR_TEXT = (
    "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 20.00           N\n"
    "ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00 20.00           C\n"
    "END\n"
)


def _args(**fields) -> argparse.Namespace:
    return argparse.Namespace(**fields)


def _build_run(tmp_path: Path, system_ids=("SYS_LIG",)) -> RunDir:
    """A run directory whose manifest digests match the files under it.

    Files are written before the manifest so the digests are of real bytes; `RunDir.create`
    then lays the run-level files over the top without disturbing them.
    """
    root = tmp_path / "runs" / "r1"
    settings = Settings()
    inputs: dict = {}
    per_system: dict = {}

    for system_id in system_ids:
        directory = root / "systems" / system_id / "inputs"
        (directory / "params").mkdir(parents=True, exist_ok=True)
        (directory / "receptor.pdb").write_text(RECEPTOR_TEXT)
        (directory / "system.spec.yaml").write_text(f"system_id: {system_id}\n")
        (directory / "params" / "LIG.params").write_text("NAME LIG\n")
        per_system[system_id] = {
            "receptor_sha256": sha256_file(directory / "receptor.pdb"),
            "spec_sha256": sha256_file(directory / "system.spec.yaml"),
            "params_sha256": [sha256_file(directory / "params" / "LIG.params")],
        }

    if len(system_ids) == 1:
        inputs = dict(per_system[system_ids[0]])
    else:
        inputs = {"systems": per_system}

    manifest = build_manifest("run-1", settings, CREATED, inputs)
    return RunDir.create(root, manifest, settings, capture_environment())


def _rewrite_manifest(run: RunDir, **fields) -> None:
    data = json.loads(run.manifest_path.read_text())
    data.update(fields)
    run.manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True))


# ============================================================ registration


@pytest.mark.unit
def test_registers_both_subcommands():
    parser = argparse.ArgumentParser()
    verify.register(parser.add_subparsers(dest="command"))

    args = parser.parse_args(["verify", "--run-dir", "runs/r1"])
    assert args.func is verify.run
    assert args.run_dir == Path("runs/r1")
    assert args.replay == 3

    args = parser.parse_args([verify.SELFTEST_NAME, "--emit"])
    assert args.func is verify.run_selftest
    assert args.emit is True
    assert args.golden == verify.GOLDEN_PATH


# ================================================================== verify


@pytest.mark.unit
def test_intact_run_dir_exits_zero(tmp_path: Path, capsys):
    run = _build_run(tmp_path)
    assert verify.run(_args(run_dir=run.root, replay=0)) == 0
    out = capsys.readouterr().out
    assert "0 failed" in out
    assert "receptor.pdb matches" in out


@pytest.mark.unit
def test_a_changed_receptor_fails_and_names_the_file(tmp_path: Path, capsys):
    run = _build_run(tmp_path)
    receptor = run.root / "systems" / "SYS_LIG" / "inputs" / "receptor.pdb"
    receptor.write_text(RECEPTOR_TEXT.replace("ALA", "GLY"))

    assert verify.run(_args(run_dir=run.root, replay=0)) != 0
    out = capsys.readouterr().out
    assert str(receptor) in out
    assert "changed" in out
    assert "FAIL" in out


@pytest.mark.unit
def test_a_missing_input_file_fails(tmp_path: Path, capsys):
    run = _build_run(tmp_path)
    params = run.root / "systems" / "SYS_LIG" / "inputs" / "params" / "LIG.params"
    params.unlink()

    assert verify.run(_args(run_dir=run.root, replay=0)) != 0
    assert "params" in capsys.readouterr().out


@pytest.mark.unit
def test_a_schema_version_mismatch_is_reported_not_raised(tmp_path: Path, capsys):
    run = _build_run(tmp_path)
    _rewrite_manifest(run, schema_version=99)

    assert verify.run(_args(run_dir=run.root, replay=0)) != 0
    out = capsys.readouterr().out
    assert "schema_version" in out
    assert "Traceback" not in out


@pytest.mark.unit
def test_an_unreadable_manifest_is_reported_not_raised(tmp_path: Path, capsys):
    run = _build_run(tmp_path)
    run.manifest_path.write_text("{not json")

    assert verify.run(_args(run_dir=run.root, replay=0)) != 0
    assert "manifest" in capsys.readouterr().out


@pytest.mark.unit
def test_a_missing_manifest_is_reported(tmp_path: Path, capsys):
    run = _build_run(tmp_path)
    run.manifest_path.unlink()

    assert verify.run(_args(run_dir=run.root, replay=0)) != 0
    assert "no manifest" in capsys.readouterr().out


@pytest.mark.unit
def test_an_environment_mismatch_is_information_not_failure(tmp_path: Path, capsys):
    """env_digest excludes cpu_count/platform/machine, so a mismatch means a package moved
    — worth printing, not worth failing."""
    run = _build_run(tmp_path)
    _rewrite_manifest(run, env_digest="sha256:" + "0" * 64)

    assert verify.run(_args(run_dir=run.root, replay=0)) == 0
    out = capsys.readouterr().out
    assert "INFO" in out
    assert "env_digest differs" in out


@pytest.mark.unit
def test_replay_zero_is_skipped_not_failed(tmp_path: Path, capsys):
    run = _build_run(tmp_path)
    assert verify.run(_args(run_dir=run.root, replay=0)) == 0
    assert "--replay 0" in capsys.readouterr().out


@pytest.mark.unit
def test_replay_without_stored_decoys_is_skipped(tmp_path: Path, capsys):
    """Either PyRosetta is absent or there is nothing to replay; both are SKIP, never FAIL."""
    run = _build_run(tmp_path)
    assert verify.run(_args(run_dir=run.root, replay=3)) == 0
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "0 failed" in out


@pytest.mark.unit
def test_flat_digests_on_a_multi_system_run_are_skipped_with_a_reason(tmp_path: Path, capsys):
    run = _build_run(tmp_path, system_ids=("A_LIG", "B_LIG"))
    # The fixture writes per-system digests for a multi-system run; flatten them to the
    # ambiguous shape §4 defines for a single system.
    data = json.loads(run.manifest_path.read_text())
    data["inputs"] = data["inputs"]["systems"]["A_LIG"]
    run.manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    assert verify.run(_args(run_dir=run.root, replay=0)) == 0
    assert "inputs.systems" in capsys.readouterr().out


@pytest.mark.unit
def test_per_system_digests_are_checked_for_every_system(tmp_path: Path, capsys):
    run = _build_run(tmp_path, system_ids=("A_LIG", "B_LIG"))
    assert verify.run(_args(run_dir=run.root, replay=0)) == 0
    out = capsys.readouterr().out
    assert "inputs/A_LIG/receptor_sha256" in out
    assert "inputs/B_LIG/receptor_sha256" in out


# ======================================================== metrics selftest


@pytest.fixture(scope="module")
def emitted_golden(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("golden") / "metrics_golden.json"
    assert verify.run_selftest(_args(emit=True, golden=path)) == 0
    return path


@pytest.mark.unit
def test_selftest_passes_against_the_golden_it_emitted(emitted_golden: Path, capsys):
    assert verify.run_selftest(_args(emit=False, golden=emitted_golden)) == 0
    assert "PASS" in capsys.readouterr().out


@pytest.mark.unit
def test_selftest_passes_against_the_shipped_golden(capsys):
    """S0.3 itself: this environment must reproduce the values the golden file records."""
    assert verify.run_selftest(_args(emit=False, golden=verify.GOLDEN_PATH)) == 0
    assert "PASS" in capsys.readouterr().out


@pytest.mark.unit
def test_selftest_fails_when_a_golden_value_is_perturbed(
    emitted_golden: Path, tmp_path: Path, capsys
):
    golden = json.loads(emitted_golden.read_text())
    golden["cases"]["auroc_point"]["value"] += 1e-9
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(golden))

    assert verify.run_selftest(_args(emit=False, golden=corrupted)) != 0
    out = capsys.readouterr().out
    assert "auroc_point" in out
    assert "FAIL" in out


@pytest.mark.unit
def test_selftest_rejects_a_golden_whose_recorded_digest_does_not_match_its_contents(
    emitted_golden: Path, tmp_path: Path, capsys
):
    """The digest is recomputed from the file's own cases, so the field cannot be trusted
    into agreement."""
    golden = json.loads(emitted_golden.read_text())
    golden["digest"] = "sha256:" + "0" * 64
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(golden))

    assert verify.run_selftest(_args(emit=False, golden=edited)) != 0
    assert "edited since it was written" in capsys.readouterr().out


@pytest.mark.unit
def test_selftest_rejects_a_golden_from_a_different_fixture(
    emitted_golden: Path, tmp_path: Path, capsys
):
    golden = json.loads(emitted_golden.read_text())
    golden["schema_version"] = verify.SELFTEST_SCHEMA + 1
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(golden))

    assert verify.run_selftest(_args(emit=False, golden=stale)) != 0
    assert "schema_version" in capsys.readouterr().out


@pytest.mark.unit
def test_selftest_reports_a_missing_golden_rather_than_raising(tmp_path: Path, capsys):
    assert verify.run_selftest(_args(emit=False, golden=tmp_path / "absent.json")) != 0
    assert "--emit" in capsys.readouterr().out


@pytest.mark.unit
def test_every_public_metric_is_covered():
    """A metric added without a case would leave S0.3 unchecked for that metric."""
    import atomfrust.metrics as metrics

    covered = " ".join(verify.metrics_cases())
    missing = [name for name in metrics.__all__ if name != "Estimate" and name not in covered]
    assert not missing


@pytest.mark.unit
def test_canonicalisation_rounds_to_the_documented_precision():
    """A difference below the 12th significant digit is BLAS noise, not a finding."""
    noisy = 0.7393750000000001234
    assert verify.canonicalise(noisy) == verify.canonicalise(0.73937500000000009)
    assert verify.canonicalise(0.739375) != verify.canonicalise(0.7393751)
    assert verify.canonicalise(np.float32(0.5)) == 0.5
    assert verify.canonicalise(np.array([1, 2])) == [1, 2]


@pytest.mark.unit
def test_selftest_is_deterministic_within_a_process():
    assert verify.selftest_payload()["digest"] == verify.selftest_payload()["digest"]


# ============================================================= integration


PROCESSED = Path("data/processed")
PARAMS = Path("data/ligands/params")


@pytest.mark.integration
def test_replay_reproduces_stored_decoy_energies(tmp_path: Path, capsys):
    """The seeding contract, end to end: decoy 0 regenerated from the recorded settings must
    equal the stored energies bit for bit."""
    from atomfrust.decoys.base import DecoyContext
    from atomfrust.decoys.identity import IdentityDecoyGenerator
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex
    from atomfrust.regions import resolve_regions
    from atomfrust.spec import LigandSpec, Receptor, SystemSpec

    pytest.importorskip("pyrosetta")
    pdb = PROCESSED / "5GMP_clean.pdb"
    params = PARAMS / "F62.params"
    missing = [str(p) for p in (pdb, params) if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")

    system_id = "5GMP_F62"
    root = tmp_path / "runs" / "replay"
    inputs = root / "systems" / system_id / "inputs"
    (inputs / "params").mkdir(parents=True)
    (inputs / "receptor.pdb").write_bytes(pdb.read_bytes())
    (inputs / "params" / "F62.params").write_bytes(params.read_bytes())

    source = SystemSpec.from_pdb(pdb, system_id=system_id)
    spec = source.model_copy(
        update={
            "receptor": Receptor(path=Path("receptor.pdb")),
            "ligands": (
                LigandSpec(
                    selector=source.ligands[0].selector,
                    params=Path("params/F62.params"),
                ),
            ),
        }
    )
    (inputs / "system.spec.yaml").write_text(spec.to_yaml())

    # relax='min' keeps the test to one cheap decoy; n_decoys is the smallest legal value.
    settings = Settings.model_validate(
        {"decoys": {"n_decoys": 2, "base_seed": 42, "relax": "min"}}
    )

    resolved = SystemSpec.from_yaml_file(inputs / "system.spec.yaml")
    loaded = load_complex(resolved)
    nodes_df, pairs = build_graph(loaded.nodes, loaded.geometry, settings)
    regions = resolve_regions(loaded.nodes, loaded.geometry)
    context = DecoyContext(
        pose=loaded.pose,
        nodes=loaded.nodes,
        pairs=pairs,
        regions=regions,
        settings=settings,
    )
    generator = IdentityDecoyGenerator(
        context,
        scope=settings.decoys.scope,
        identity=settings.decoys.identity,
        placement=settings.decoys.placement,
        base_seed=settings.decoys.base_seed,
        relax=settings.decoys.relax,
        mc_cycles=settings.decoys.mc_cycles,
        native_repack=settings.decoys.native_repack,
    )
    result = generator.generate(0)

    manifest = build_manifest(
        "replay-1",
        settings,
        CREATED,
        {
            "receptor_sha256": sha256_file(inputs / "receptor.pdb"),
            "spec_sha256": sha256_file(inputs / "system.spec.yaml"),
            "params_sha256": [sha256_file(inputs / "params" / "F62.params")],
        },
    )
    run = RunDir.create(root, manifest, settings, capture_environment())
    system = run.system(system_id).ensure()
    system.write_graph(nodes_df, pairs)
    with system.energy_writer() as writer:
        writer.add(0, result.pair_id, result.e_direct, result.e_fa_rep)

    assert verify.run(_args(run_dir=run.root, replay=1)) == 0
    out = capsys.readouterr().out
    assert f"replay/{system_id}" in out
    assert "reproduced bit-identically" in out


@pytest.mark.integration
def test_replay_fails_when_a_stored_energy_is_edited(tmp_path: Path, capsys):
    """The replay check must be able to fail; a check that only ever passes proves nothing."""
    import pandas as pd

    pytest.importorskip("pyrosetta")
    pdb = PROCESSED / "5GMP_clean.pdb"
    if not pdb.exists():
        pytest.skip("missing (dvc pull): data/processed/5GMP_clean.pdb")

    test_replay_reproduces_stored_decoy_energies(tmp_path, capsys)
    root = tmp_path / "runs" / "replay"
    system = RunDir(root).system("5GMP_F62")
    part = system.decoy_parts()[0]
    frame = pd.read_parquet(part)
    frame.loc[0, "e_direct"] = np.float32(frame.loc[0, "e_direct"] + 1.0)
    frame.to_parquet(part, index=False)

    assert verify.run(_args(run_dir=root, replay=1)) != 0
    out = capsys.readouterr().out
    assert "e_direct differs" in out
