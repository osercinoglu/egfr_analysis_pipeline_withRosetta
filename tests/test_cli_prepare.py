"""E1 tests for `atomfrust prepare` — the bare-PDB path (user request 8) and the spec path.

Everything here is `unit`: `atomfrust.spec` parses coordinates by PDB column, so preparing a
system needs no PyRosetta and no network. The one test that touches `data/processed` skips
cleanly when the DVC-tracked tree has not been pulled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from atomfrust.cli import prepare
from atomfrust.spec import LigandSpec, PocketSpec, Receptor, ResidueSelector, SystemSpec

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_5HG8 = REPO_ROOT / "data" / "processed" / "5HG8_clean.pdb"


# ------------------------------------------------------------------ fixtures


def _atom(serial, name, resname, chain, resseq, xyz, het=False, icode=" "):
    """PDB column layout, lifted verbatim from tests/test_spec.py."""
    rec = "HETATM" if het else "ATOM  "
    x, y, z = xyz
    return (
        f"{rec}{serial:5d} {name:^4s} {resname:>3s} {chain:1s}{resseq:4d}{icode:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
    )


@pytest.fixture
def pdb(tmp_path: Path) -> Path:
    """Two protein chains (A, B), a ligand LIG in chain B, a zinc, and a water."""
    lines = []
    n = 1
    for chain in ("A", "B"):
        for resseq in range(1, 4):
            for name in ("N", "CA", "C", "O"):
                lines.append(_atom(n, name, "ALA", chain, resseq, (n * 1.0, 0.0, 0.0)))
                n += 1
    lines.append(_atom(n, "C1", "LIG", "B", 501, (1.0, 1.0, 1.0), het=True)); n += 1
    lines.append(_atom(n, "ZN", " ZN", "A", 601, (2.0, 2.0, 2.0), het=True)); n += 1
    lines.append(_atom(n, "O", "HOH", "A", 701, (9.0, 9.0, 9.0), het=True)); n += 1
    p = tmp_path / "mini.pdb"
    p.write_text("".join(lines) + "END\n")
    return p


def _invoke(*argv: str) -> int:
    """Drive the subcommand exactly as `cli.main` will, through `register` + `func`."""
    parser = argparse.ArgumentParser(prog="atomfrust")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    prepare.register(sub)
    args = parser.parse_args([prepare.NAME, *argv])
    return int(args.func(args) or 0)


def _prepared(out: Path) -> Path:
    """The single system directory under an output root."""
    dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(dirs) == 1, f"expected one prepared system, got {dirs}"
    return dirs[0]


def _report(system_dir: Path) -> dict:
    return json.loads((system_dir / prepare.REPORT_FILENAME).read_text())


# ------------------------------------------------------- registration contract


def test_module_follows_the_registration_contract():
    assert prepare.NAME == "prepare"
    assert prepare.HELP
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    prepare.register(sub)
    assert parser.parse_args(["prepare", "--pdb", "x.pdb", "-o", "out"]).func is prepare.run


# ------------------------------------------------------------- acceptance: U8


def test_bare_pdb_and_ligand_writes_every_documented_file(pdb: Path, tmp_path: Path):
    out = tmp_path / "prepared"
    assert _invoke("--pdb", str(pdb), "--ligand", "B:501", "-o", str(out)) == 0

    system = _prepared(out)
    for name in (
        prepare.SPEC_FILENAME,
        prepare.RECEPTOR_FILENAME,
        prepare.COMPONENTS_FILENAME,
        prepare.REPORT_FILENAME,
    ):
        assert (system / name).is_file(), f"{name} was not written"

    spec = SystemSpec.from_yaml_file(system / prepare.SPEC_FILENAME)
    assert [str(lig.selector.chain) + ":" + str(lig.selector.resseq) for lig in spec.ligands] == [
        "B:501"
    ]
    assert spec.check_against_structure() == []


def test_receptor_copy_is_byte_identical_and_digested(pdb: Path, tmp_path: Path):
    out = tmp_path / "prepared"
    assert _invoke("--pdb", str(pdb), "--ligand", "B:501", "-o", str(out)) == 0

    system = _prepared(out)
    assert (system / prepare.RECEPTOR_FILENAME).read_bytes() == pdb.read_bytes()
    report = _report(system)
    assert report["receptor"]["sha256"] == report["receptor"]["source_sha256"]
    assert report["files"][prepare.RECEPTOR_FILENAME] == report["receptor"]["sha256"]
    # The written spec points at the copy, so the prepared directory is self-contained.
    assert Path(report["prepared_dir"]) == system
    spec = SystemSpec.from_yaml_file(system / prepare.SPEC_FILENAME)
    assert spec.receptor.path == system / prepare.RECEPTOR_FILENAME


def test_written_spec_round_trips(pdb: Path, tmp_path: Path):
    """`system.spec.yaml` reloads to exactly the object that was written."""
    out = tmp_path / "prepared"
    assert (
        _invoke(
            "--pdb", str(pdb),
            "--ligand", "B:501",
            "--chains", "A,B",
            "--label", "affinity_pM=23.0",
            "--label", "set=synthetic",
            "--system-id", "mini_LIG",
            "-o", str(out),
        )
        == 0
    )
    system = out / "mini_LIG"
    reloaded = SystemSpec.from_yaml_file(system / prepare.SPEC_FILENAME)

    expected = SystemSpec(
        system_id="mini_LIG",
        receptor=Receptor(
            path=system / prepare.RECEPTOR_FILENAME, chains=("A", "B")
        ),
        ligands=(
            LigandSpec(
                selector=ResidueSelector(chain="B", resseq=501, comp_id="LIG")
            ),
        ),
        pocket=PocketSpec(mode="ligand_shell"),
        labels={"affinity_pM": 23.0, "set": "synthetic"},
    )
    assert reloaded == expected


def test_autodetect_takes_the_metal_and_leaves_the_water(pdb: Path, tmp_path: Path):
    out = tmp_path / "prepared"
    assert _invoke("--pdb", str(pdb), "-o", str(out)) == 0

    components = yaml.safe_load(
        (_prepared(out) / prepare.COMPONENTS_FILENAME).read_text()
    )["components"]
    assert {row["comp_id"] for row in components} == {"LIG", "ZN"}


def test_no_autodetect_gives_a_protein_only_system(pdb: Path, tmp_path: Path, capsys):
    out = tmp_path / "prepared"
    assert _invoke("--pdb", str(pdb), "--no-autodetect", "-o", str(out)) == 0

    system = _prepared(out)
    report = _report(system)
    assert report["is_protein_only"] is True
    assert report["components"] == []
    spec = SystemSpec.from_yaml_file(system / prepare.SPEC_FILENAME)
    assert spec.ligands == ()
    assert spec.pocket.mode == "whole"
    assert "protein-only" in capsys.readouterr().out


# ------------------------------------------------------------------ components


def test_components_yaml_carries_both_names_and_the_params_digest(
    pdb: Path, tmp_path: Path
):
    params = tmp_path / "LIG.params"
    params.write_text("NAME LIG\n")
    out = tmp_path / "prepared"
    assert (
        _invoke(
            "--pdb", str(pdb),
            "--ligand", "B:501",
            "--comp-id", "B:501=L99",
            "--params", f"B:501={params}",
            "-o", str(out),
        )
        == 0
    )

    (row,) = yaml.safe_load(
        (_prepared(out) / prepare.COMPONENTS_FILENAME).read_text()
    )["components"]
    assert row["selector"] == "B:501"
    assert row["comp_id"] == "L99"            # what reporting must use
    assert row["rosetta_name"] == "LIG"       # what the pose calls it
    assert row["comp_id_source"] == "override"
    assert row["params"] == str(params.resolve())
    assert row["params_sha256"].startswith("sha256:")


def test_comp_id_override_survives_into_the_spec_and_still_validates(
    pdb: Path, tmp_path: Path
):
    out = tmp_path / "prepared"
    assert (
        _invoke(
            "--pdb", str(pdb), "--ligand", "B:501", "--comp-id", "B:501=L99",
            "-o", str(out),
        )
        == 0
    )
    system = _prepared(out)
    (lig,) = SystemSpec.from_yaml_file(system / prepare.SPEC_FILENAME).ligands
    assert lig.selector.comp_id == "L99"
    assert lig.rosetta_name == "LIG"
    assert lig.effective_rosetta_name == "LIG"
    assert _report(system)["validation_problems"] == []


def test_comp_id_override_naming_nothing_is_a_reported_problem(
    pdb: Path, tmp_path: Path, capsys
):
    out = tmp_path / "prepared"
    rc = _invoke(
        "--pdb", str(pdb), "--ligand", "B:501", "--comp-id", "B:999=L99",
        "-o", str(out),
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "--comp-id names B:999" in err
    assert "B:501" in err              # names the components that do exist
    assert not out.exists()            # nothing written for a failed system


def test_missing_params_is_a_warning_not_a_failure(pdb: Path, tmp_path: Path, capsys):
    out = tmp_path / "prepared"
    assert _invoke("--pdb", str(pdb), "--ligand", "B:501", "-o", str(out)) == 0

    report = _report(_prepared(out))
    assert report["unparametrised"] == ["B:501"]
    assert any("generate-decoys" in w for w in report["warnings"])
    assert any("load_PDB_components" in w for w in report["warnings"])
    assert "generate-decoys" in capsys.readouterr().out


# ------------------------------------------------------------------ validation


def test_unknown_chain_exits_non_zero_and_lists_the_chains_present(
    pdb: Path, tmp_path: Path, capsys
):
    out = tmp_path / "prepared"
    rc = _invoke("--pdb", str(pdb), "--ligand", "B:501", "--chains", "Q", "-o", str(out))
    assert rc != 0
    err = capsys.readouterr().err
    assert "chain 'Q' not found" in err
    assert "chains present: A, B" in err


def test_unknown_ligand_exits_non_zero(pdb: Path, tmp_path: Path, capsys):
    out = tmp_path / "prepared"
    assert _invoke("--pdb", str(pdb), "--ligand", "B:999", "-o", str(out)) != 0
    assert "B:999 not found" in capsys.readouterr().err


def test_every_problem_is_reported_in_one_pass(pdb: Path, tmp_path: Path, capsys):
    """Two independent faults; both must appear, not just the first."""
    spec_file = tmp_path / "broken.yaml"
    spec_file.write_text(
        yaml.safe_dump(
            {
                "system_id": "broken",
                "receptor": {"path": str(pdb), "chains": ["Q"]},
                "ligands": [
                    {
                        "selector": {"chain": "B", "resseq": 501, "comp_id": "XXX"},
                        "params": str(tmp_path / "absent.params"),
                    }
                ],
            }
        )
    )
    assert _invoke("--spec", str(spec_file), "-o", str(tmp_path / "prepared")) != 0

    err = capsys.readouterr().err
    assert "chain 'Q' not found" in err
    assert "not 'XXX'" in err                    # comp_id genuinely disagrees with the file
    assert "params file not found" in err


# ---------------------------------------------------------------- spec loading


def test_a_system_set_prepares_every_system(pdb: Path, tmp_path: Path):
    spec_file = tmp_path / "set.yaml"
    spec_file.write_text(
        yaml.safe_dump(
            {
                "systems": [
                    {
                        "system_id": "with_ligand",
                        "receptor": {"path": "mini.pdb"},
                        "ligands": [{"selector": {"chain": "B", "resseq": 501}}],
                        "labels": {"affinity_pM": 23.0},
                    },
                    {
                        "system_id": "apo",
                        "receptor": {"path": "mini.pdb"},
                        "pocket": {"mode": "whole"},
                    },
                ]
            }
        )
    )
    out = tmp_path / "prepared"
    assert _invoke("--spec", str(spec_file), "-o", str(out)) == 0

    assert sorted(p.name for p in out.iterdir()) == ["apo", "with_ligand"]
    assert _report(out / "apo")["is_protein_only"] is True
    assert _report(out / "with_ligand")["components"][0]["selector"] == "B:501"


def test_pdb_only_flags_are_rejected_with_spec(pdb: Path, tmp_path: Path, capsys):
    spec_file = tmp_path / "one.yaml"
    spec_file.write_text(
        yaml.safe_dump(
            {
                "system_id": "one",
                "receptor": {"path": str(pdb)},
                "ligands": [{"selector": {"chain": "B", "resseq": 501}}],
            }
        )
    )
    rc = _invoke("--spec", str(spec_file), "--ligand", "B:501", "-o", str(tmp_path / "p"))
    assert rc == prepare.EXIT_USAGE
    assert "--ligand" in capsys.readouterr().err


def test_unreadable_spec_extension_is_refused(tmp_path: Path, capsys):
    bad = tmp_path / "systems.txt"
    bad.write_text("nope\n")
    assert _invoke("--spec", str(bad), "-o", str(tmp_path / "p")) != 0
    assert "unrecognised spec extension" in capsys.readouterr().err


# --------------------------------------------------- the legacy comp_id trap


@pytest.mark.skipif(
    not LEGACY_5HG8.exists(), reason="data/processed not present (dvc pull)"
)
def test_5hg8_legacy_file_reports_the_rewritten_resname_until_overridden(
    tmp_path: Path, capsys
):
    """5HG8's HETATM resName is already `Z34`; its true CCD code is `634`.

    Without an override the spec can only carry what the file says, and the warning must say
    so. With `--comp-id` the CCD code lands in `comp_id` and the file's name in
    `rosetta_name`.
    """
    naive = tmp_path / "naive"
    assert _invoke("--pdb", str(LEGACY_5HG8), "--ligand", "A:9001", "-o", str(naive)) == 0
    (row,) = yaml.safe_load(
        (_prepared(naive) / prepare.COMPONENTS_FILENAME).read_text()
    )["components"]
    assert row["comp_id"] == "Z34" and row["rosetta_name"] == "Z34"
    assert any("Z34, not 634" in w for w in _report(_prepared(naive))["warnings"])

    fixed = tmp_path / "fixed"
    assert (
        _invoke(
            "--pdb", str(LEGACY_5HG8),
            "--ligand", "A:9001",
            "--comp-id", "A:9001=634",
            "--system-id", "5HG8_634",
            "-o", str(fixed),
        )
        == 0
    )
    (row,) = yaml.safe_load(
        (fixed / "5HG8_634" / prepare.COMPONENTS_FILENAME).read_text()
    )["components"]
    assert row["comp_id"] == "634"
    assert row["rosetta_name"] == "Z34"
    assert _report(fixed / "5HG8_634")["validation_problems"] == []
    assert "comp_id=634 rosetta_name=Z34" in capsys.readouterr().out
