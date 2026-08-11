"""E2 tests — `atomfrust generate-decoys`.

The unit tier covers the parser and the flag → settings mapping, plus the structural claim
that gives this subcommand its reason to exist: it generates decoys and computes no index,
so it must not reach the analysis package at all.

The integration tier runs the real thing on 5GMP at two decoys and `--relax min`. Two
decoys is enough to prove every documented path is written, that resume extends rather than
recomputes, and that a generation-stage change is refused; the published `--relax mc` costs
~5x per decoy and proves nothing extra here.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from atomfrust.cli import generate_decoys as cmd

PROCESSED = Path("data/processed/5GMP_clean.pdb")
PARAMS = Path("data/ligands/params/F62.params")


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse through a real subparsers action, the way `main` will."""
    parser = argparse.ArgumentParser(prog="atomfrust")
    subparsers = parser.add_subparsers(dest="command")
    cmd.register(subparsers)
    return parser.parse_args(argv)


# ================================================================== unit tier


@pytest.mark.unit
def test_register_adds_a_parser_that_documents_the_key_flags():
    parser = argparse.ArgumentParser(prog="atomfrust")
    subparsers = parser.add_subparsers(dest="command")
    cmd.register(subparsers)

    assert cmd.NAME in subparsers.choices
    help_text = subparsers.choices[cmd.NAME].format_help()
    for flag in (
        "--spec", "--pdb", "--ligand", "--run-dir", "--axis", "--n-decoys", "--scope",
        "--identity", "--placement", "--mutate-sel", "--repack-sel", "--minimize-sel",
        "--repack-shell", "--relax", "--mc-cycles", "--workers", "--shard",
        "--save-structures", "--resume", "--restart", "--allow-mismatch", "--config",
    ):
        assert flag in help_text, flag
    assert subparsers.choices[cmd.NAME].get_default("func") is cmd.run


@pytest.mark.unit
def test_flags_map_onto_the_settings_overlay():
    args = _parse(
        [
            "generate-decoys", "--pdb", "x.pdb", "--run-dir", "r",
            "--n-decoys", "7", "--scope", "contact_shell", "--identity", "uniform20",
            "--placement", "inplace", "--relax", "min", "--mc-cycles", "3",
            "--mutate-sel", "chain A", "--repack-sel", "protein",
            "--minimize-sel", "protein", "--repack-shell", "8.0",
            "--workers", "2", "--shard", "1/4", "--save-structures",
        ]
    )
    settings, provenance = cmd.resolve_settings(args)

    assert settings.decoys.n_decoys == 7
    assert settings.decoys.scope == "contact_shell"
    assert settings.decoys.identity == "uniform20"
    assert settings.decoys.placement == "inplace"
    assert settings.decoys.relax == "min"
    assert settings.decoys.mc_cycles == 3
    assert settings.decoys.repack_shell_A == 8.0
    assert settings.decoys.regions.mutate == "chain A"
    assert settings.runtime.workers == 2
    assert settings.runtime.shard == "1/4"
    assert settings.runtime.save_structures is True
    # The provenance map is what makes a resolved config explain itself.
    assert provenance["decoys.n_decoys"].value == "cli"


@pytest.mark.unit
def test_omitted_flags_leave_the_published_defaults_alone():
    """`--relax mc` is the settings default because it is the published protocol.

    argparse must not shadow it with a default of its own, or every run would silently drop
    to the cheap `min` relaxation while the manifest claimed otherwise.
    """
    settings, provenance = cmd.resolve_settings(
        _parse(["generate-decoys", "--pdb", "x.pdb", "--run-dir", "r"])
    )
    assert settings.decoys.relax == "mc"
    assert settings.decoys.identity == "native"
    assert settings.decoys.placement == "permute"
    assert settings.decoys.n_decoys == 1000
    assert provenance == {}


@pytest.mark.unit
def test_config_file_is_layered_under_the_flags(tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text("decoys:\n  n_decoys: 20\n  relax: min\n")
    settings, provenance = cmd.resolve_settings(
        _parse(
            ["generate-decoys", "--pdb", "x.pdb", "--run-dir", "r",
             "--config", str(config), "--n-decoys", "5"]
        )
    )
    assert settings.decoys.n_decoys == 5  # flag wins
    assert settings.decoys.relax == "min"  # file wins over the built-in default
    assert provenance["decoys.relax"].value == "file"


@pytest.mark.unit
def test_repack_shell_intersects_all_three_region_selectors():
    """`resolve_regions` refuses mutate not-subset-of repack, so a shell that narrowed only
    the repack set would be an error rather than a restriction."""
    settings, _ = cmd.resolve_settings(
        _parse(
            ["generate-decoys", "--pdb", "x.pdb", "--run-dir", "r", "--repack-shell", "12"]
        )
    )
    mutate, repack, minimize = cmd.region_expressions(settings)
    for expression in (mutate, repack, minimize):
        assert expression.endswith("and within(12.0, ligand)")

    plain, _ = cmd.resolve_settings(_parse(["generate-decoys", "--pdb", "x.pdb", "--run-dir", "r"]))
    assert cmd.region_expressions(plain) == ("protein", "protein", "protein")


@pytest.mark.unit
def test_unimplemented_generation_options_are_refused_by_name(tmp_path, capsys):
    """A chemotype axis is planned (stage G) but absent. It must fail before any pose load,
    naming itself — not somewhere inside a spawned worker."""
    args = _parse(
        ["generate-decoys", "--pdb", "absent.pdb", "--run-dir", str(tmp_path / "r"),
         "--axis", "chemotype"]
    )
    assert cmd.run(args) == 2
    assert "chemotype" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


@pytest.mark.unit
def test_exactly_one_input_source_is_required(tmp_path, capsys):
    args = _parse(["generate-decoys", "--run-dir", str(tmp_path / "r")])
    assert cmd.run(args) == 2
    assert "--spec or --pdb" in capsys.readouterr().err


@pytest.mark.unit
def test_the_module_never_reaches_the_analysis_package():
    """User request 6 is 'decoys, no index'. That is a structural property, not a promise:
    importing this module must not pull in `atomfrust.analyze` by any path."""
    probe = (
        "import sys; import atomfrust.cli.generate_decoys as m; "
        "leaked = sorted(n for n in sys.modules if n.startswith('atomfrust.analyze')); "
        "print(leaked); raise SystemExit(1 if leaked else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"analysis modules imported: {result.stdout}"

    # The subprocess only sees module-level imports; deferred ones hide inside functions.
    tree = ast.parse(Path(cmd.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [name for name in imported if name.startswith("atomfrust.analyze")]


# =========================================================== integration tier


def _require_data() -> None:
    missing = [str(p) for p in (PROCESSED, PARAMS) if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")


def _spec_file(directory: Path) -> Path:
    path = directory / "5GMP.yaml"
    path.write_text(
        "system_id: 5GMP_F62\n"
        f"receptor:\n  path: {PROCESSED.resolve()}\n  chains: [A]\n"
        "ligands:\n"
        "  - selector: {chain: A, resseq: 1101, comp_id: F62}\n"
        f"    params: {PARAMS.resolve()}\n"
        "pocket:\n  mode: ligand_shell\n"
    )
    return path


def _generate(run_dir: Path, spec: Path, n_decoys: int, *extra: str) -> int:
    return cmd.run(
        _parse(
            [
                "generate-decoys", "--spec", str(spec), "--run-dir", str(run_dir),
                "--n-decoys", str(n_decoys), "--relax", "min", "--workers", "1",
                "--save-structures", *extra,
            ]
        )
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """One two-decoy run, shared by the tests below. Ordered: the resume test extends it."""
    _require_data()
    base = tmp_path_factory.mktemp("e2")
    spec = _spec_file(base)
    run_dir = base / "run"
    assert _generate(run_dir, spec, 2) == 0
    return run_dir, spec


@pytest.mark.integration
def test_the_run_directory_contract_is_written_in_full(generated):
    run_dir, _ = generated
    system = run_dir / "systems" / "5GMP_F62"
    for relative in (
        "manifest.json",
        "settings.resolved.yaml",
        "env.json",
        "logs/generate.jsonl",
        "systems/5GMP_F62/STATUS.json",
        "systems/5GMP_F62/inputs/system.spec.yaml",
        "systems/5GMP_F62/inputs/receptor.pdb",
        "systems/5GMP_F62/inputs/components.yaml",
        "systems/5GMP_F62/inputs/params/F62.params",
        "systems/5GMP_F62/graph/nodes.parquet",
        "systems/5GMP_F62/graph/pairs.parquet",
        "systems/5GMP_F62/native/native.pdb",
        "systems/5GMP_F62/native/native_energies.parquet",
        "systems/5GMP_F62/native/raw_energy.json",
        "systems/5GMP_F62/decoys/index.parquet",
    ):
        assert (run_dir / relative).exists(), relative

    parts = sorted((system / "decoys" / "energies").glob("part-*.parquet"))
    assert parts, "no decoy energy parts"

    # No index, no classification, no analysis: that is the whole contract of this command.
    assert list((system / "analyses").iterdir()) == []

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["regeneration_key"].startswith("sha256:")
    assert manifest["generation"]["decoys"]["n_decoys"] == 2

    status = json.loads((system / "STATUS.json").read_text())
    assert status["state"] == "complete"
    assert (status["n_decoys_done"], status["n_decoys_target"]) == (2, 2)

    structures = sorted((system / "decoys" / "structures").glob("decoy_*.pdb.gz"))
    assert [p.name for p in structures] == ["decoy_000000.pdb.gz", "decoy_000001.pdb.gz"]
    with gzip.open(structures[0], "rt") as handle:
        text = handle.read()
    assert text.startswith("HEADER") and "\nATOM  " in text


@pytest.mark.integration
def test_stored_energies_are_direct_pair_energies_keyed_by_pair_id(generated):
    run_dir, _ = generated
    system = run_dir / "systems" / "5GMP_F62"
    pairs = pd.read_parquet(system / "graph" / "pairs.parquet")
    stored = pd.concat(
        [pd.read_parquet(p) for p in sorted((system / "decoys" / "energies").glob("part-*.parquet"))],
        ignore_index=True,
    )
    assert set(stored.columns) == {"decoy_id", "axis", "pair_id", "e_direct", "e_fa_rep"}
    assert sorted(stored.decoy_id.unique()) == [0, 1]
    for decoy_id, group in stored.groupby("decoy_id"):
        assert list(group.pair_id) == sorted(group.pair_id)
        assert set(group.pair_id) == set(pairs.pair_id)

    # The ligand really is in the graph -- an A4 correction that a protein-only pair table
    # would silently undo.
    assert (pairs.kind_i.astype(str).eq("ligand") | pairs.kind_j.astype(str).eq("ligand")).any()


@pytest.mark.integration
def test_index_has_one_row_per_decoy_with_cost_and_the_backbone_invariant(generated):
    run_dir, _ = generated
    index = pd.read_parquet(run_dir / "systems" / "5GMP_F62" / "decoys" / "index.parquet")
    assert len(index) == 2
    assert list(index.decoy_id) == [0, 1]
    for column in cmd.INDEX_COLUMNS:
        assert column in index.columns, column
    assert (index.wall_s > 0).all()
    # C5's post-condition, recorded per decoy: sequence changed, geometry did not.
    assert (index.backbone_rmsd < 1e-6).all()
    assert (index.n_mutated > 0).all()
    assert (index.seed == [42, 43]).all()  # base_seed + decoy_id, verbatim
    # Run-relative, so a copied run directory still resolves its own structures.
    assert list(index.structure_path) == [
        "systems/5GMP_F62/decoys/structures/decoy_000000.pdb.gz",
        "systems/5GMP_F62/decoys/structures/decoy_000001.pdb.gz",
    ]


@pytest.mark.integration
def test_resume_extends_the_ensemble_and_never_rewrites_stored_decoys(generated):
    """The prototype's two defects in one test: `run_pipeline.py:241` short-circuited on a
    finished parquet so a larger n_decoys did nothing, and the shared checkpoint was
    rewritten from zero on every resume."""
    run_dir, spec = generated
    energies = run_dir / "systems" / "5GMP_F62" / "decoys" / "energies"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(energies.glob("part-*"))}

    assert _generate(run_dir, spec, 3) == 0

    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(energies.glob("part-*"))}
    assert before.items() <= after.items(), "an existing part file was rewritten"
    assert len(after) == len(before) + 1

    index = pd.read_parquet(run_dir / "systems" / "5GMP_F62" / "decoys" / "index.parquet")
    assert list(index.decoy_id) == [0, 1, 2]
    stored = pd.concat([pd.read_parquet(p) for p in sorted(energies.glob("part-*"))])
    assert sorted(stored.decoy_id.unique()) == [0, 1, 2]

    status = json.loads((run_dir / "systems" / "5GMP_F62" / "STATUS.json").read_text())
    assert (status["n_decoys_done"], status["n_decoys_target"]) == (3, 3)


@pytest.mark.integration
def test_two_spawned_workers_produce_the_same_ensemble_as_one(generated, tmp_path):
    """Decoy i is seeded base_seed + i, so worker count is a runtime detail and nothing else.

    This is also the only test that exercises the spawn path in this module: a pose cannot
    cross a process boundary, so the worker rebuilds it from (receptor.pdb, params) inside
    the task factory.
    """
    run_dir, spec = generated
    parallel = tmp_path / "parallel"
    assert _generate(parallel, spec, 2, "--workers", "2") == 0

    def _read(root: Path) -> pd.DataFrame:
        parts = sorted((root / "systems" / "5GMP_F62" / "decoys" / "energies").glob("part-*"))
        frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        return (
            frame[frame.decoy_id < 2]
            .sort_values(["decoy_id", "pair_id"], kind="stable")
            .reset_index(drop=True)
        )

    assert _read(parallel).equals(_read(run_dir))


@pytest.mark.integration
def test_a_generation_stage_change_exits_three_with_a_diff(generated, capsys):
    run_dir, spec = generated
    code = _generate(run_dir, spec, 3, "--identity", "uniform20")
    assert code == 3
    message = capsys.readouterr().err
    assert "decoys.identity" in message
    assert "uniform20" in message

    # Refusal is total: nothing was appended under the incompatible settings.
    index = pd.read_parquet(run_dir / "systems" / "5GMP_F62" / "decoys" / "index.parquet")
    assert (index.identity == "native").all()


@pytest.mark.integration
def test_restart_discards_the_stored_ensemble(generated):
    """Runs last, because it destroys the fixture's three-decoy ensemble."""
    run_dir, spec = generated
    system = run_dir / "systems" / "5GMP_F62"
    assert _generate(run_dir, spec, 2, "--restart") == 0

    parts = sorted((system / "decoys" / "energies").glob("part-*"))
    assert len(parts) == 1
    assert sorted(pd.read_parquet(parts[0]).decoy_id.unique()) == [0, 1]
    index = pd.read_parquet(system / "decoys" / "index.parquet")
    assert list(index.decoy_id) == [0, 1]
    # Everything the system needs was rebuilt, not just the decoys.
    assert (system / "native" / "native_energies.parquet").exists()
    assert (system / "graph" / "pairs.parquet").exists()
