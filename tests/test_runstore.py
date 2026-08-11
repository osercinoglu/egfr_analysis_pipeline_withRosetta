"""B4/B5 tests — provenance and the run-directory contract. No PyRosetta."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomfrust.provenance import (
    CONTRACT_VERSION,
    SCHEMA_VERSION,
    Manifest,
    ManifestVersionError,
    build_manifest,
    capture_environment,
    env_digest,
    sha256_file,
)
from atomfrust.runstore import (
    EXIT_REQUIRES_REGENERATION,
    RegenerationRequired,
    RunDir,
)
from atomfrust.settings import Settings

pytestmark = pytest.mark.unit

CREATED = "2026-08-11T07:00:00Z"


@pytest.fixture
def run(tmp_path: Path) -> RunDir:
    settings = Settings()
    manifest = build_manifest("run-1", settings, CREATED, {"receptor_sha256": "sha256:aa"})
    return RunDir.create(tmp_path / "runs" / "r1", manifest, settings, capture_environment())


# ============================================================== provenance


def test_digest_changes_when_the_file_changes(tmp_path: Path):
    path = tmp_path / "receptor.pdb"
    path.write_text("ATOM      1  N   ALA A   1\n")
    before = sha256_file(path)
    path.write_text("ATOM      1  N   ALA A   2\n")
    assert sha256_file(path) != before
    assert before.startswith("sha256:")


def test_manifest_round_trips(tmp_path: Path):
    manifest = build_manifest("run-1", Settings(), CREATED, {"receptor_sha256": "sha256:aa"})
    path = tmp_path / "manifest.json"
    manifest.write(path)
    assert Manifest.read(path) == manifest


def test_a_modified_receptor_changes_the_manifest(tmp_path: Path):
    a = build_manifest("r", Settings(), CREATED, {"receptor_sha256": "sha256:aa"})
    b = build_manifest("r", Settings(), CREATED, {"receptor_sha256": "sha256:bb"})
    assert a.inputs != b.inputs
    assert a.regeneration_key != b.regeneration_key


def test_unknown_schema_version_raises_rather_than_guessing(tmp_path: Path):
    manifest = build_manifest("r", Settings(), CREATED)
    path = tmp_path / "manifest.json"
    manifest.write(path)

    data = json.loads(path.read_text())
    data["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(data))
    with pytest.raises(ManifestVersionError, match="schema_version"):
        Manifest.read(path)


def test_unknown_contract_version_raises(tmp_path: Path):
    manifest = build_manifest("r", Settings(), CREATED)
    path = tmp_path / "manifest.json"
    manifest.write(path)
    data = json.loads(path.read_text())
    data["contract_version"] = CONTRACT_VERSION + 99
    path.write_text(json.dumps(data))
    with pytest.raises(ManifestVersionError, match="contract_version"):
        Manifest.read(path)


def test_environment_capture_does_not_import_pyrosetta():
    """The unit tier must stay usable on a machine that has never installed PyRosetta;
    metadata lookup is also seconds faster than an import."""
    import subprocess
    import sys

    code = (
        "import sys; from atomfrust.provenance import capture_environment; "
        "capture_environment(); sys.exit(1 if 'pyrosetta' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], timeout=60).returncode == 0


def test_env_digest_ignores_core_count_and_platform():
    """Results must not depend on how many cores produced them, and requiring an identical
    OS string would make cross-machine comparison impossible."""
    base = capture_environment()
    other = dict(base, cpu_count=999, platform="SomeOtherOS-1.2.3", machine="riscv64")
    assert env_digest(base) == env_digest(other)

    changed = dict(base, packages=dict(base["packages"], numpy="0.0.1"))
    assert env_digest(base) != env_digest(changed)


# ======================================================= run directory shape


def test_create_writes_the_documented_layout(run: RunDir):
    assert run.manifest_path.exists()
    assert run.settings_path.exists()
    assert run.env_path.exists()
    assert (run.root / "systems").is_dir()
    assert run.read_settings() == Settings()
    assert run.read_manifest().run_id == "run-1"


def test_system_directories_are_created_on_demand(run: RunDir):
    system = run.system("5GMP_F62").ensure()
    for path in (system.inputs, system.graph, system.native, system.decoy_energies):
        assert path.is_dir()
    assert run.systems() == ["5GMP_F62"]


def test_status_updates_merge(run: RunDir):
    system = run.system("S").ensure()
    assert system.read_status()["state"] == "absent"
    system.update_status(state="running", n_decoys_target=100)
    system.update_status(n_decoys_done=25)
    status = system.read_status()
    assert (status["state"], status["n_decoys_target"], status["n_decoys_done"]) == (
        "running",
        100,
        25,
    )


# ================================================ acceptance: sharded prefix


def _write_decoys(system, n_decoys: int, n_pairs: int, shards: int) -> None:
    """Write n_decoys across `shards` writers, interleaved as parallel workers would."""
    rng = np.random.default_rng(0)
    writers = [system.energy_writer(shard=s, flush_every=3) for s in range(shards)]
    for decoy_id in range(n_decoys):
        writer = writers[decoy_id % shards]
        writer.add(
            decoy_id=decoy_id,
            pair_id=np.arange(n_pairs),
            e_direct=rng.normal(size=n_pairs),
            e_fa_rep=rng.random(n_pairs),
        )
    for writer in writers:
        writer.flush()


def test_prefix_read_returns_exactly_the_first_n_decoys(run: RunDir):
    """The acceptance criterion, and the property the convergence sweep depends on."""
    system = run.system("S").ensure()
    _write_decoys(system, n_decoys=100, n_pairs=7, shards=4)

    everything = system.read_decoy_energies()
    assert len(everything) == 100 * 7
    assert sorted(everything.decoy_id.unique()) == list(range(100))

    prefix = system.read_decoy_energies(n_decoys=25)
    assert sorted(prefix.decoy_id.unique()) == list(range(25))
    assert len(prefix) == 25 * 7

    # The prefix must be identical to the same rows of the full read — not merely the
    # same size — or "N as a prefix" would be an approximation.
    expected = everything[everything.decoy_id < 25].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, expected)


def test_shards_write_disjoint_files_and_reassemble(run: RunDir):
    system = run.system("S").ensure()
    _write_decoys(system, n_decoys=40, n_pairs=5, shards=4)

    parts = system.decoy_parts()
    assert len(parts) > 4, "flushing should produce several part files per shard"
    shards_seen = {p.name.split("-")[1] for p in parts}
    assert shards_seen == {"000", "001", "002", "003"}

    frame = system.read_decoy_energies()
    counts = frame.groupby("decoy_id").size()
    assert set(counts) == {5}, "every decoy must contribute every pair exactly once"


def test_rows_are_sorted_by_pair_id_within_each_decoy(run: RunDir):
    system = run.system("S").ensure()
    with system.energy_writer(flush_every=1) as writer:
        writer.add(
            decoy_id=0,
            pair_id=np.array([5, 1, 3, 0]),
            e_direct=np.array([0.5, 0.1, 0.3, 0.0]),
            e_fa_rep=np.zeros(4),
        )
    frame = system.read_decoy_energies()
    assert frame.pair_id.tolist() == [0, 1, 3, 5]
    assert frame.e_direct.tolist() == pytest.approx([0.0, 0.1, 0.3, 0.5])


def test_completed_ids_support_resume(run: RunDir):
    system = run.system("S").ensure()
    _write_decoys(system, n_decoys=17, n_pairs=3, shards=2)
    assert system.completed_decoy_ids() == set(range(17))


def test_unflushed_decoys_are_lost_but_flushed_ones_survive(run: RunDir):
    """A killed worker loses at most `flush_every` decoys — not the whole structure, as
    the prototype did when save_every_n_decoys equalled n_decoys."""
    system = run.system("S").ensure()
    writer = system.energy_writer(flush_every=5)
    for decoy_id in range(7):
        writer.add(decoy_id, np.arange(2), np.zeros(2), np.zeros(2))
    # Simulate a kill: no final flush.
    assert system.completed_decoy_ids() == set(range(5))


def test_empty_store_reads_as_an_empty_frame(run: RunDir):
    frame = run.system("S").ensure().read_decoy_energies()
    assert frame.empty
    assert list(frame.columns) == list(
        ("decoy_id", "axis", "pair_id", "e_direct", "e_fa_rep")
    )


def test_axis_filter(run: RunDir):
    system = run.system("S").ensure()
    with system.energy_writer(flush_every=1) as writer:
        writer.add(0, np.arange(2), np.zeros(2), np.zeros(2), axis="identity")
        writer.add(1, np.arange(2), np.zeros(2), np.zeros(2), axis="chemotype")
    assert set(system.read_decoy_energies(axes=["identity"]).decoy_id) == {0}
    assert system.completed_decoy_ids(axis="chemotype") == {1}


def test_no_E_ij_column_is_ever_stored(run: RunDir):
    """The load-bearing schema invariant: E_ij is derived, never persisted. Storing it
    would freeze the many-body formula at generation time."""
    system = run.system("S").ensure()
    _write_decoys(system, n_decoys=3, n_pairs=4, shards=1)
    columns = set(system.read_decoy_energies().columns)
    assert not {"E", "E_ij", "E_native", "F", "F_index"} & columns


# ============================================== acceptance: compatibility


def test_analysis_stage_change_is_compatible(run: RunDir):
    requested = Settings.model_validate(
        {
            "analysis": {"index": "rank_percentile", "n_decoys": 50},
            "manybody": {"mode": "pair_retained"},
            "contacts": {"cutoff_A": 8.0},
        }
    )
    report = run.assert_compatible(requested)
    assert report.verdict == "analysis_only"
    assert report.compatible
    assert any(d.path == "manybody.mode" for d in report.differences)


def test_identical_settings_report_ok(run: RunDir):
    assert run.assert_compatible(Settings()).verdict == "ok"


def test_generation_stage_change_raises_with_exit_code_3(run: RunDir):
    requested = Settings.model_validate({"energy": {"score_function": "ref2015_cart"}})
    with pytest.raises(RegenerationRequired) as exc:
        run.assert_compatible(requested)

    assert exc.value.exit_code == EXIT_REQUIRES_REGENERATION == 3
    message = str(exc.value)
    assert "energy.score_function" in message
    assert "ref2015" in message and "ref2015_cart" in message


@pytest.mark.parametrize(
    "overlay,expected_field",
    [
        ({"decoys": {"n_decoys": 500}}, "decoys.n_decoys"),
        ({"decoys": {"relax": "min"}}, "decoys.relax"),
        ({"decoys": {"native_repack": False}}, "decoys.native_repack"),
        ({"decoys": {"identity": "composition"}}, "decoys.identity"),
        ({"graph": {"superset": {"ca_cutoff_A": 14.0}}}, "graph.superset.ca_cutoff_A"),
    ],
)
def test_generation_fields_each_require_regeneration(run: RunDir, overlay, expected_field):
    with pytest.raises(RegenerationRequired) as exc:
        run.assert_compatible(Settings.model_validate(overlay))
    assert any(d.path == expected_field for d in exc.value.report.differences)


def test_allow_mismatch_downgrades_to_a_report(run: RunDir):
    requested = Settings.model_validate({"energy": {"score_function": "ref2015_cart"}})
    report = run.assert_compatible(requested, allow_mismatch=True)
    assert report.verdict == "requires_regeneration"
    assert not report.compatible


def test_a_different_pyrosetta_version_requires_regeneration(run: RunDir):
    """Same settings, different Rosetta: energies are not comparable."""
    with pytest.raises(RegenerationRequired) as exc:
        run.assert_compatible(Settings(), pyrosetta_version="1999.01")
    assert any(d.path == "regeneration_key" for d in exc.value.report.differences)


def test_report_explains_itself(run: RunDir):
    report = run.check_compatible(
        Settings.model_validate({"decoys": {"n_decoys": 500}})
    )
    text = report.explain()
    assert "generation-stage" in text and "decoys.n_decoys" in text


# ==================================== integration: the contract, end to end


@pytest.mark.integration
def test_a_stored_run_can_be_reanalysed_under_a_different_many_body_formula(tmp_path):
    """The contract's central claim, demonstrated on a real structure.

    Store direct pair energies once, then form BOTH many-body formulas from the same bytes
    with no pose and no PyRosetta. This is what makes a corrected formula a re-analysis
    rather than a regeneration, and it is user request 7.
    """
    processed = Path("data/processed/5GMP_clean.pdb")
    params = Path("data/ligands/params/F62.params")
    if not (processed.exists() and params.exists()):
        pytest.skip("data/ not present (dvc pull)")

    from atomfrust.energy import EnergyEvaluator, effective_energy, many_body_energies
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex
    from atomfrust.provenance import capture_environment, sha256_file
    from atomfrust.spec import LigandSpec, SystemSpec

    settings = Settings()
    spec = SystemSpec.from_pdb(processed, system_id="5GMP_F62")
    spec = spec.model_copy(
        update={
            "ligands": (
                LigandSpec(selector=spec.ligands[0].selector, params=params),
            )
        }
    )

    # --- generate: one pass with PyRosetta -----------------------------------
    manifest = build_manifest(
        "e2e", settings, CREATED, {"receptor_sha256": sha256_file(processed)}
    )
    run = RunDir.create(tmp_path / "run", manifest, settings, capture_environment())
    system = run.system(spec.system_id).ensure()

    loaded = load_complex(spec)
    nodes, pairs = build_graph(loaded.nodes, loaded.geometry, settings)
    system.write_graph(nodes, pairs)

    energies = EnergyEvaluator(loaded.pose).pairs(pairs)
    system.write_native_energies(energies)
    with system.energy_writer(flush_every=2) as writer:
        for decoy_id in range(4):  # stand-ins; Stage C makes them real decoys
            writer.add(
                decoy_id,
                energies.pair_id.to_numpy(),
                energies.e_direct.to_numpy() * (1.0 + 0.01 * decoy_id),
                energies.e_fa_rep.to_numpy(),
            )
    system.update_status(state="complete", n_decoys_done=4, n_decoys_target=4)

    # --- analyse: no pose, no PyRosetta --------------------------------------
    reopened = RunDir(run.root)
    _, stored_pairs = reopened.system(spec.system_id).read_graph()
    stored_native = reopened.system(spec.system_id).read_native_energies()
    merged = stored_pairs.merge(stored_native, on="pair_id")

    e = effective_energy(merged.e_direct, merged.e_fa_rep, exclude_fa_rep=True)
    chen = many_body_energies(merged.node_i.to_numpy(), merged.node_j.to_numpy(), e, "chen_literal")
    retained = many_body_energies(merged.node_i.to_numpy(), merged.node_j.to_numpy(), e, "pair_retained")

    # Both formulas from one stored ensemble, and they genuinely differ.
    assert np.allclose(retained - chen, e, atol=1e-5)
    assert not np.allclose(retained, chen)

    # Switching the formula is an analysis-stage change: no regeneration.
    report = reopened.assert_compatible(
        Settings.model_validate({"manybody": {"mode": "pair_retained"}})
    )
    assert report.verdict == "analysis_only"

    # A decoy prefix is exact.
    assert sorted(reopened.system(spec.system_id).read_decoy_energies(n_decoys=2).decoy_id.unique()) == [0, 1]
    assert reopened.system(spec.system_id).read_status()["state"] == "complete"


# ============================================ D8: the raw-energy control


@pytest.mark.unit
def test_raw_interaction_energy_partitions_the_graph():
    """S5.2's control. Comparing a Rosetta Z-score against a Vina raw score confounds
    normalisation with energy-function quality; only raw-vs-Z within one energy function
    isolates the effect, so the raw number is recorded next to every Z-score."""
    from atomfrust.energy import raw_interaction_energy

    pairs = pd.DataFrame(
        {
            "pair_id": [0, 1, 2, 3],
            "kind_i": ["protein", "protein", "ligand", "protein"],
            "kind_j": ["protein", "ligand", "metal", "ligand"],
        }
    )
    energies = pd.DataFrame(
        {
            "pair_id": [0, 1, 2, 3],
            "e_direct": [-1.0, -2.0, -4.0, -8.0],
            "e_fa_rep": [0.5, 0.5, 1.0, 2.0],
        }
    )
    raw = raw_interaction_energy(pairs, energies, exclude_fa_rep=True)

    # exactly-one-non-protein endpoint -> interaction
    assert raw["n_interaction"] == 2
    assert raw["interaction"] == pytest.approx((-2.0 - 0.5) + (-8.0 - 2.0))
    assert raw["n_intra_protein"] == 1
    assert raw["intra_protein"] == pytest.approx(-1.5)
    assert raw["n_intra_component"] == 1
    assert raw["intra_component"] == pytest.approx(-5.0)
    assert raw["n_total"] == 4

    # A raw score that silently dropped fa_rep would not be raw.
    assert raw["total_with_fa_rep"] == pytest.approx(-15.0)
    assert raw["total"] == pytest.approx(-19.0)


@pytest.mark.unit
def test_raw_energy_round_trips_through_the_run_directory(run: RunDir):
    from atomfrust.energy import raw_interaction_energy

    system = run.system("S").ensure()
    with pytest.raises(FileNotFoundError, match="S5.2"):
        system.read_raw_energy()

    pairs = pd.DataFrame({"pair_id": [0], "kind_i": ["protein"], "kind_j": ["ligand"]})
    energies = pd.DataFrame({"pair_id": [0], "e_direct": [-3.0], "e_fa_rep": [1.0]})
    raw = raw_interaction_energy(pairs, energies)
    system.write_raw_energy(raw)
    assert system.read_raw_energy() == raw


@pytest.mark.unit
def test_a_resumed_shard_does_not_overwrite_its_existing_parts(run: RunDir):
    """Regression: the writer numbered part files from zero on every construction, so a
    second writer on the same shard rewrote `part-000-00000.parquet` and destroyed the
    decoys in it. Silent data loss — nothing errored, the file was simply replaced."""
    system = run.system("S").ensure()

    with system.energy_writer(shard=0, flush_every=1) as writer:
        writer.add(0, np.arange(3), np.zeros(3), np.zeros(3))
    assert system.completed_decoy_ids() == {0}

    with system.energy_writer(shard=0, flush_every=1) as writer:
        writer.add(1, np.arange(3), np.ones(3), np.zeros(3))

    assert system.completed_decoy_ids() == {0, 1}, "resuming destroyed the earlier decoys"
    assert len(system.decoy_parts()) == 2

    # And the recovered ensemble is intact, not merely present.
    frame = system.read_decoy_energies()
    assert sorted(frame.decoy_id.unique()) == [0, 1]
    assert frame[frame.decoy_id == 0].e_direct.tolist() == [0.0, 0.0, 0.0]
    assert frame[frame.decoy_id == 1].e_direct.tolist() == [1.0, 1.0, 1.0]


@pytest.mark.unit
def test_multiple_shards_resume_independently(run: RunDir):
    system = run.system("S").ensure()
    for shard in (0, 1):
        with system.energy_writer(shard=shard, flush_every=1) as writer:
            writer.add(shard, np.arange(2), np.zeros(2), np.zeros(2))
    for shard in (0, 1):
        with system.energy_writer(shard=shard, flush_every=1) as writer:
            writer.add(shard + 10, np.arange(2), np.zeros(2), np.zeros(2))
    assert system.completed_decoy_ids() == {0, 1, 10, 11}
    assert len(system.decoy_parts()) == 4
