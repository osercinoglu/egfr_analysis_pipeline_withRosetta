"""B3 tests for atomfrust.spec — shape validation, structure validation, loaders, from_pdb."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from atomfrust.spec import (
    LigandSpec,
    PocketSpec,
    Receptor,
    ResidueSelector,
    SpecError,
    SystemSet,
    SystemSpec,
)

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ fixtures


def _atom(serial, name, resname, chain, resseq, xyz, het=False, icode=" "):
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
                lines.append(
                    _atom(n, name, "ALA", chain, resseq, (n * 1.0, 0.0, 0.0))
                )
                n += 1
    lines.append(_atom(n, "C1", "LIG", "B", 501, (1.0, 1.0, 1.0), het=True)); n += 1
    lines.append(_atom(n, "ZN", " ZN", "A", 601, (2.0, 2.0, 2.0), het=True)); n += 1
    lines.append(_atom(n, "O", "HOH", "A", 701, (9.0, 9.0, 9.0), het=True)); n += 1
    p = tmp_path / "mini.pdb"
    p.write_text("".join(lines) + "END\n")
    return p


# ------------------------------------------------------- acceptance: YAML validates


def test_hand_written_yaml_validates(tmp_path: Path, pdb: Path):
    text = f"""
system_id: MINI_LIG
receptor:
  path: {pdb}
  chains: [A, B]
ligands:
  - selector: {{chain: B, resseq: 501, comp_id: LIG}}
    rosetta_name: Z34
pocket:
  mode: ligand_shell
  reference: any_heavy
  cutoff_A: 6.0
labels: {{affinity_pM: 23.0}}
"""
    path = tmp_path / "spec.yaml"
    path.write_text(text)
    spec = SystemSpec.from_yaml_file(path)

    assert spec.system_id == "MINI_LIG"
    assert spec.receptor.chains == ("A", "B")
    assert spec.ligands[0].selector.comp_id == "LIG"
    assert spec.ligands[0].effective_rosetta_name == "Z34"
    assert spec.labels["affinity_pM"] == 23.0
    assert spec.check_against_structure() == []


def test_comp_id_and_rosetta_name_stay_distinct():
    """The 634 -> Z34 trap: reporting uses comp_id, the pose uses rosetta_name."""
    lig = LigandSpec(
        selector=ResidueSelector(chain="A", resseq=1001, comp_id="634"),
        rosetta_name="Z34",
    )
    assert lig.selector.comp_id == "634"
    assert lig.effective_rosetta_name == "Z34"
    # Without an override the pose name is the CCD code.
    plain = LigandSpec(selector=ResidueSelector(chain="A", resseq=1, comp_id="ATP"))
    assert plain.effective_rosetta_name == "ATP"


# ------------------------------------------------- acceptance: protein-only validates


def test_empty_ligands_is_protein_only(pdb: Path):
    spec = SystemSpec(
        system_id="apo",
        receptor=Receptor(path=pdb),
        ligands=(),
        pocket=PocketSpec(mode="whole"),
    )
    assert spec.is_protein_only
    assert not spec.is_covalent
    assert spec.check_against_structure() == []


def test_ligand_shell_without_a_ligand_is_rejected(pdb: Path):
    with pytest.raises(ValidationError, match="needs at least one ligand"):
        SystemSpec(
            system_id="bad",
            receptor=Receptor(path=pdb),
            ligands=(),
            pocket=PocketSpec(mode="ligand_shell"),
        )


def test_chain_interface_needs_two_chains(pdb: Path):
    with pytest.raises(ValidationError, match="at least two chains"):
        SystemSpec(
            system_id="iface",
            receptor=Receptor(path=pdb, chains=("A",)),
            pocket=PocketSpec(mode="chain_interface"),
        )
    ok = SystemSpec(
        system_id="iface",
        receptor=Receptor(path=pdb, chains=("A", "B")),
        pocket=PocketSpec(mode="chain_interface"),
    )
    assert ok.is_protein_only


def test_residue_list_mode_needs_residues(pdb: Path):
    with pytest.raises(ValidationError, match="needs pocket.residues"):
        SystemSpec(
            system_id="rl",
            receptor=Receptor(path=pdb),
            pocket=PocketSpec(mode="residue_list"),
        )


# --------------------------------------- acceptance: missing chain names the chains


def test_absent_chain_error_lists_the_chains_present(pdb: Path):
    spec = SystemSpec(
        system_id="wrong-chain",
        receptor=Receptor(path=pdb, chains=("Q",)),
        pocket=PocketSpec(mode="whole"),
    )
    with pytest.raises(SpecError) as exc:
        spec.validate_against_structure()
    msg = str(exc.value)
    assert "'Q' not found" in msg
    assert "chains present: A, B" in msg


def test_absent_ligand_error_names_the_residue_range(pdb: Path):
    spec = SystemSpec(
        system_id="wrong-resseq",
        receptor=Receptor(path=pdb),
        ligands=(LigandSpec(selector=ResidueSelector(chain="B", resseq=999)),),
    )
    problems = spec.check_against_structure()
    assert len(problems) == 1
    assert "B:999 not found" in problems[0]
    assert "residues" in problems[0]


def test_wrong_comp_id_is_reported(pdb: Path):
    spec = SystemSpec(
        system_id="wrong-comp",
        receptor=Receptor(path=pdb),
        ligands=(
            LigandSpec(selector=ResidueSelector(chain="B", resseq=501, comp_id="XXX")),
        ),
    )
    problems = spec.check_against_structure()
    assert len(problems) == 1 and "is 'LIG'" in problems[0]


def test_all_problems_are_reported_at_once(pdb: Path):
    spec = SystemSpec(
        system_id="many",
        receptor=Receptor(path=pdb, chains=("Q",)),
        ligands=(LigandSpec(selector=ResidueSelector(chain="B", resseq=999)),),
    )
    assert len(spec.check_against_structure()) == 2


def test_missing_file_is_reported_not_raised(tmp_path: Path):
    spec = SystemSpec(
        system_id="nofile",
        receptor=Receptor(path=tmp_path / "nope.pdb"),
        pocket=PocketSpec(mode="whole"),
    )
    assert "not found" in spec.check_against_structure()[0]


# ----------------------------------------------------- acceptance: from_pdb (U8)


def test_from_pdb_with_explicit_ligand_matches_the_yaml_spec(tmp_path: Path, pdb: Path):
    built = SystemSpec.from_pdb(pdb, ligand="B:501", system_id="MINI_LIG")

    text = f"""
system_id: MINI_LIG
receptor: {{path: {pdb}}}
ligands:
  - selector: {{chain: B, resseq: 501, comp_id: LIG}}
pocket: {{mode: ligand_shell}}
"""
    path = tmp_path / "equiv.yaml"
    path.write_text(text)
    written = SystemSpec.from_yaml_file(path)

    assert built == written
    assert built.check_against_structure() == []


def test_from_pdb_fills_in_comp_id_from_the_file(pdb: Path):
    spec = SystemSpec.from_pdb(pdb, ligand="B:501")
    assert spec.ligands[0].selector.comp_id == "LIG"
    assert spec.system_id == "mini_LIG"


def test_from_pdb_autodetects_components_but_not_water_or_protein(pdb: Path):
    spec = SystemSpec.from_pdb(pdb)
    comps = sorted(lig.selector.comp_id for lig in spec.ligands)
    assert comps == ["LIG", "ZN"]  # water excluded, ALA excluded


def test_from_pdb_autodetect_off_gives_protein_only(pdb: Path):
    spec = SystemSpec.from_pdb(pdb, autodetect=False)
    assert spec.is_protein_only and spec.pocket.mode == "whole"


def test_from_pdb_rejects_a_ligand_that_is_not_there(pdb: Path):
    with pytest.raises(SpecError, match="chains present: A, B"):
        SystemSpec.from_pdb(pdb, ligand="B:999")


def test_from_pdb_accepts_multiple_ligands_and_labels(pdb: Path):
    spec = SystemSpec.from_pdb(pdb, ligand=["B:501", "A:601"], affinity_pM=12.0)
    assert len(spec.ligands) == 2
    assert spec.labels == {"affinity_pM": 12.0}


def test_duplicate_ligand_selectors_are_rejected(pdb: Path):
    with pytest.raises(ValidationError, match="duplicate ligand selector"):
        SystemSpec.from_pdb(pdb, ligand=["B:501", "B:501"])


# ------------------------------------------------------------ selector parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("B:501", ("B", 501, "")),
        ("A:52A", ("A", 52, "A")),
        (" B : 501 ", ("B", 501, "")),
        ("A:-3", ("A", -3, "")),
    ],
)
def test_selector_parse(text, expected):
    assert ResidueSelector.parse(text).key() == expected


@pytest.mark.parametrize("bad", ["B501", ":501", "B:", "B:abc", ""])
def test_selector_parse_rejects_junk(bad):
    with pytest.raises(ValueError):
        ResidueSelector.parse(bad)


# -------------------------------------------------------------------- receptor


def test_receptor_requires_exactly_one_source():
    with pytest.raises(ValidationError, match="exactly one"):
        Receptor()
    with pytest.raises(ValidationError, match="exactly one"):
        Receptor(path="a.pdb", pdb_id="5GMP")
    assert Receptor(pdb_id="5GMP").chains is None


def test_empty_chain_tuple_is_rejected():
    with pytest.raises(ValidationError, match="not empty"):
        Receptor(pdb_id="5GMP", chains=())


def test_unknown_key_is_forbidden():
    with pytest.raises(ValidationError):
        SystemSpec.model_validate(
            {"system_id": "x", "receptor": {"pdb_id": "5GMP"}, "typo": 1}
        )


# ------------------------------------------------------------------- covalent


def test_covalent_anchor_round_trips_and_flags_the_system(pdb: Path):
    spec = SystemSpec(
        system_id="cov",
        receptor=Receptor(path=pdb),
        ligands=(
            LigandSpec(
                selector=ResidueSelector(chain="B", resseq=501, comp_id="LIG"),
                covalent_anchor={
                    "chain": "A", "resseq": 1, "atom": "SG", "ligand_atom": "C1"
                },
            ),
        ),
    )
    assert spec.is_covalent
    assert spec.check_against_structure() == []
    assert SystemSpec.model_validate(yaml.safe_load(spec.to_yaml())) == spec


def test_covalent_anchor_pointing_nowhere_is_reported(pdb: Path):
    spec = SystemSpec(
        system_id="cov-bad",
        receptor=Receptor(path=pdb),
        ligands=(
            LigandSpec(
                selector=ResidueSelector(chain="B", resseq=501),
                covalent_anchor={
                    "chain": "A", "resseq": 888, "atom": "SG", "ligand_atom": "C1"
                },
            ),
        ),
    )
    assert any("covalent anchor" in p for p in spec.check_against_structure())


# --------------------------------------------------------------------- loaders


def test_paths_resolve_relative_to_the_spec_file(tmp_path: Path, pdb: Path):
    sub = tmp_path / "specs"
    sub.mkdir()
    rel = Path("..") / pdb.name
    (sub / "s.yaml").write_text(
        yaml.safe_dump(
            {
                "system_id": "rel",
                "receptor": {"path": str(rel)},
                "pocket": {"mode": "whole"},
            }
        )
    )
    spec = SystemSpec.from_yaml_file(sub / "s.yaml")
    assert Path(spec.receptor.path).is_absolute()
    assert spec.check_against_structure() == []


def test_systemset_from_yaml_list_and_dict(tmp_path: Path, pdb: Path):
    one = {"system_id": "a", "receptor": {"path": str(pdb)}, "pocket": {"mode": "whole"}}
    two = {"system_id": "b", "receptor": {"pdb_id": "5GMP"}, "pocket": {"mode": "whole"}}

    as_list = tmp_path / "l.yaml"
    as_list.write_text(yaml.safe_dump([one, two]))
    assert len(SystemSet.from_yaml_file(as_list)) == 2

    as_dict = tmp_path / "d.yaml"
    as_dict.write_text(yaml.safe_dump({"systems": [one, two]}))
    s = SystemSet.from_yaml_file(as_dict)
    assert [x.system_id for x in s] == ["a", "b"]
    assert s.by_id("b").receptor.pdb_id == "5GMP"

    single = tmp_path / "one.yaml"
    single.write_text(yaml.safe_dump(one))
    assert len(SystemSet.from_yaml_file(single)) == 1


def test_systemset_rejects_duplicate_ids(tmp_path: Path):
    one = {"system_id": "a", "receptor": {"pdb_id": "X"}, "pocket": {"mode": "whole"}}
    p = tmp_path / "dup.yaml"
    p.write_text(yaml.safe_dump([one, dict(one)]))
    with pytest.raises(ValidationError, match="duplicate system_id"):
        SystemSet.from_yaml_file(p)


def test_systemset_by_id_error_lists_available(tmp_path: Path):
    p = tmp_path / "s.yaml"
    p.write_text(
        yaml.safe_dump(
            [{"system_id": "a", "receptor": {"pdb_id": "X"}, "pocket": {"mode": "whole"}}]
        )
    )
    with pytest.raises(KeyError, match="have: a"):
        SystemSet.from_yaml_file(p).by_id("zzz")


def test_systemset_from_json(tmp_path: Path, pdb: Path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            [{"system_id": "a", "receptor": {"path": str(pdb)}, "pocket": {"mode": "whole"}}]
        )
    )
    assert SystemSet.from_json_file(p).by_id("a").check_against_structure() == []


def test_systemset_from_csv_with_labels(tmp_path: Path, pdb: Path):
    p = tmp_path / "s.csv"
    p.write_text(
        "system_id,path,chains,ligand,ligand_comp_id,rosetta_name,affinity_pM,set\n"
        f"MINI,{pdb},A|B,B:501,LIG,Z34,23.0,egfr\n"
        f"APO,{pdb},A,,,,,control\n"
    )
    s = SystemSet.from_csv_file(p)
    mini = s.by_id("MINI")
    assert mini.receptor.chains == ("A", "B")
    assert mini.ligands[0].effective_rosetta_name == "Z34"
    assert mini.labels == {"affinity_pM": "23.0", "set": "egfr"}
    assert mini.pocket.mode == "ligand_shell"
    assert mini.check_against_structure() == []

    apo = s.by_id("APO")
    assert apo.is_protein_only and apo.pocket.mode == "whole"


# ------------------------------------------------ real structure, when available


def test_against_a_real_processed_structure():
    """5GMP is a real single-chain kinase + inhibitor prepared by the legacy Stage 4."""
    real = Path("data/processed/5GMP_clean.pdb")
    if not real.exists():
        pytest.skip("data/processed not present (dvc pull)")

    # Autodetect must find exactly the inhibitor, and nothing else: no waters (Stage 4
    # deletes them) and no protein residues misread as components.
    auto = SystemSpec.from_pdb(real, system_id="5GMP_F62")
    assert [lig.selector.comp_id for lig in auto.ligands] == ["F62"]
    assert auto.check_against_structure() == []

    # The explicit selector path must agree with what autodetect found.
    sel = auto.ligands[0].selector
    explicit = SystemSpec.from_pdb(
        real, ligand=f"{sel.chain}:{sel.resseq}", system_id="5GMP_F62"
    )
    assert explicit == auto

    # A real structure is where a wrong residue number is easy to make; the error must
    # say what range is actually there.
    problems = SystemSpec.from_pdb(real, autodetect=False).model_copy(
        update={
            "ligands": (
                LigandSpec(selector=ResidueSelector(chain=sel.chain, resseq=9999)),
            ),
            "pocket": PocketSpec(mode="ligand_shell"),
        }
    ).check_against_structure()
    assert len(problems) == 1 and "residues" in problems[0]


def test_all_processed_structures_parse():
    """Every legacy Stage-4 output must build a valid spec. This is the realistic
    stress set for B6/B7: 61 kinase-inhibitor complexes including the 634 -> Z34 case."""
    pdbs = sorted(Path("data/processed").glob("*_clean.pdb"))
    if not pdbs:
        pytest.skip("data/processed not present (dvc pull)")
    for p in pdbs:
        spec = SystemSpec.from_pdb(p)
        assert len(spec.ligands) == 1, f"{p.name}: {len(spec.ligands)} components"
        assert spec.check_against_structure() == [], p.name


def test_processed_pdbs_carry_the_rosetta_name_not_the_ccd_code():
    """Documents a trap rather than a bug: Stage 4 rewrote the HETATM resName, so 5HG8's
    file says Z34 where its true CCD code is 634. A spec built from such a file is right
    for the pose and wrong for reporting — the real code must be supplied explicitly."""
    p = Path("data/processed/5HG8_clean.pdb")
    if not p.exists():
        pytest.skip("data/processed not present (dvc pull)")

    naive = SystemSpec.from_pdb(p)
    assert naive.ligands[0].selector.comp_id == "Z34"
    assert naive.ligands[0].effective_rosetta_name == "Z34"

    corrected = naive.model_copy(
        update={
            "ligands": (
                LigandSpec(
                    selector=naive.ligands[0].selector.model_copy(
                        update={"comp_id": "634"}
                    ),
                    rosetta_name="Z34",
                ),
            )
        }
    )
    assert corrected.ligands[0].selector.comp_id == "634"
    assert corrected.ligands[0].effective_rosetta_name == "Z34"

    # The corrected spec must VALIDATE, not fail. An earlier version of this test asserted
    # the opposite and so pinned a defect as expected behaviour: `_check_selector` compared
    # `comp_id` to the file literally, which rejected the one spec that gets both names
    # right. The file may legitimately carry either name — a raw PDB has the CCD code, a
    # Stage-4 processed file has the Rosetta name — so both are accepted.
    assert corrected.check_against_structure() == []

    # A genuinely wrong name is still caught, and the message names both acceptable ones.
    wrong = corrected.model_copy(
        update={
            "ligands": (
                LigandSpec(
                    selector=corrected.ligands[0].selector.model_copy(
                        update={"comp_id": "XXX"}
                    ),
                    rosetta_name="YYY",
                ),
            )
        }
    )
    problems = wrong.check_against_structure()
    assert len(problems) == 1
    assert "'XXX' or 'YYY'" in problems[0]
