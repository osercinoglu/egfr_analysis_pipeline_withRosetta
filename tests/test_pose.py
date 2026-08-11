"""B6 integration tests for atomfrust.pose. Requires PyRosetta and data/processed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atomfrust.graph import build_graph
from atomfrust.settings import ContactSettings, Settings
from atomfrust.spec import LigandSpec, SystemSpec

pytestmark = pytest.mark.integration

PROCESSED = Path("data/processed")
PARAMS = Path("data/ligands/params")


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")


def _spec_with_params(pdb: Path, params: Path, system_id: str) -> SystemSpec:
    spec = SystemSpec.from_pdb(pdb, system_id=system_id)
    return spec.model_copy(
        update={
            "ligands": tuple(
                LigandSpec(selector=lig.selector, params=params) for lig in spec.ligands
            )
        }
    )


@pytest.fixture(scope="module")
def loaded_5gmp():
    _require(PROCESSED / "5GMP_clean.pdb", PARAMS / "F62.params")
    from atomfrust.pose import load_complex

    return load_complex(
        _spec_with_params(PROCESSED / "5GMP_clean.pdb", PARAMS / "F62.params", "5GMP")
    )


@pytest.fixture(scope="module")
def protein_lines():
    _require(PROCESSED / "5GMP_clean.pdb")
    src = (PROCESSED / "5GMP_clean.pdb").read_text().splitlines(keepends=True)
    het = [l for l in src if l.startswith("HETATM")]
    rest = [l for l in src if not l.startswith(("HETATM", "END"))]
    return rest, het


# ------------------------------------------------------- the ligand is a node


def test_ligand_loads_as_a_typed_node(loaded_5gmp):
    lc = loaded_5gmp
    assert lc.n_protein == 305
    assert lc.n_components == 1

    component = lc.components[0]
    assert component.node_id == "A:1101"
    assert component.kind == "ligand"
    assert component.comp_id == "F62"
    assert component.params_sha256 is not None

    node = lc.node_by_key("A", 1101)
    assert node.kind == "ligand"
    assert node.n_heavy == 39
    assert not node.mutable, "a ligand must not be identity-randomisable"


def test_geometry_is_consistent_with_the_node_list(loaded_5gmp):
    lc = loaded_5gmp
    geom = lc.geometry
    assert geom.n_nodes == len(lc.nodes)
    assert geom.heavy_node.max() == len(lc.nodes) - 1
    # Every protein residue has a Ca; the ligand does not.
    has_ca = np.isfinite(geom.ca_xyz).all(axis=1)
    assert has_ca.sum() == lc.n_protein
    assert not has_ca[[n.kind != "protein" for n in lc.nodes].index(True)]
    # Heavy-atom counts agree between the node table and the flat atom array.
    counts = np.bincount(geom.heavy_node, minlength=len(lc.nodes))
    assert list(counts) == [n.n_heavy for n in lc.nodes]


# ------------------------------- acceptance: N copies are N distinct nodes


def test_two_ligand_copies_become_two_distinct_nodes(tmp_path, protein_lines):
    """Regression: passing the same .params twice raised
    'residue type F62 already exists in the cache'. N copies share one residue *type*."""
    _require(PARAMS / "F62.params")
    from atomfrust.pose import load_complex

    rest, het = protein_lines
    duplicated = [line[:22] + "1102" + line[26:] for line in het]
    pdb = tmp_path / "two_copies.pdb"
    pdb.write_text("".join(rest + het + duplicated) + "END\n")

    lc = load_complex(_spec_with_params(pdb, PARAMS / "F62.params", "two"))
    assert lc.n_components == 2
    assert sorted(c.node_id for c in lc.components) == ["A:1101", "A:1102"]
    assert {c.comp_id for c in lc.components} == {"F62"}
    assert len({n.node_id for n in lc.nodes}) == len(lc.nodes), "node ids must be unique"


# --------------------------------------- acceptance: metals are typed nodes


def test_metal_loads_as_a_metal_node(tmp_path, protein_lines):
    from atomfrust.pose import load_complex

    rest, _ = protein_lines
    pdb = tmp_path / "with_zn.pdb"
    pdb.write_text(
        "".join(rest)
        + "HETATM 9999 ZN    ZN A9900      10.000  10.000  10.000  1.00  0.00          ZN\n"
        + "END\n"
    )
    lc = load_complex(SystemSpec.from_pdb(pdb, system_id="zn"))
    assert [c.kind for c in lc.components] == ["metal"]
    assert lc.components[0].node_id == "A:9900"
    assert not lc.node_by_key("A", 9900).mutable


# ------------------------------------ acceptance: protein-only has no components


def test_protein_only_has_no_components(tmp_path, protein_lines):
    from atomfrust.pose import load_complex

    rest, _ = protein_lines
    pdb = tmp_path / "apo.pdb"
    pdb.write_text("".join(rest) + "END\n")

    lc = load_complex(SystemSpec.from_pdb(pdb, system_id="apo"))
    assert lc.n_components == 0
    assert len(lc.nodes) == 305
    assert all(n.kind == "protein" for n in lc.nodes)


# ---------------------------- acceptance: 634/Z34 resolves through components


def test_ccd_and_rosetta_names_are_both_preserved():
    """The pose calls it Z34; reporting must still say 634."""
    _require(PROCESSED / "5HG8_clean.pdb", PARAMS / "Z34.params")
    from atomfrust.pose import load_complex

    spec = SystemSpec.from_pdb(PROCESSED / "5HG8_clean.pdb", system_id="5HG8")
    spec = spec.model_copy(
        update={
            "ligands": (
                LigandSpec(
                    selector=spec.ligands[0].selector.model_copy(
                        update={"comp_id": "634"}
                    ),
                    rosetta_name="Z34",
                    params=PARAMS / "Z34.params",
                ),
            )
        }
    )
    component = load_complex(spec, validate=False).components[0]
    assert component.comp_id == "634"
    assert component.rosetta_name == "Z34"


# ------------------------------- cross-validation against the legacy engine


def test_protein_contact_count_reproduces_the_prototype(loaded_5gmp):
    """The strongest available check on the new graph: at the prototype's own definition
    (Ca-Ca 10 A, |i-j| >= 4) it must produce exactly the 1772 pairs stored in
    results/5GMP_F62_frustration.parquet — same number, entirely different code path."""
    parquet = Path("results/5GMP_F62_frustration.parquet")
    if not parquet.exists():
        pytest.skip("results/ not present (dvc pull)")
    import pandas as pd

    lc = loaded_5gmp
    _, pairs = build_graph(
        lc.nodes,
        lc.geometry,
        Settings(),
        definitions={"proto": ContactSettings(cutoff_A=10.0, seq_sep_min=4)},
    )
    protein_only = (pairs.kind_i == "protein") & (pairs.kind_j == "protein")
    got = int((pairs["in__proto"] & protein_only).sum())
    assert got == len(pd.read_parquet(parquet)) == 1772


def test_ligand_contacts_exist_and_scale_with_the_shell(loaded_5gmp):
    """The A4 correction, quantified. The prototype produced zero ligand contacts by
    construction; the published count for 5GMP is 16 minimally frustrated ones, so the
    total must comfortably exceed that at a sane shell radius."""
    lc = loaded_5gmp
    counts = {}
    for cutoff in (4.0, 5.0, 6.0):
        _, pairs = build_graph(
            lc.nodes,
            lc.geometry,
            Settings(),
            definitions={"d": ContactSettings(ligand_cutoff_A=cutoff)},
        )
        touches_ligand = (pairs.kind_i != "protein") | (pairs.kind_j != "protein")
        counts[cutoff] = int((pairs["in__d"] & touches_ligand).sum())

    assert counts[4.0] < counts[5.0] < counts[6.0], counts
    assert counts[6.0] >= 16, "too few ligand contacts to yield the published count"
    assert counts[6.0] < 100, "implausibly many; the shell rule is probably wrong"


# ----------------------------------------------------------------- errors


def test_missing_params_gives_a_diagnosable_error():
    """An unparametrised component must fail loudly, naming itself and the fix.

    Two things are being guarded. Rosetta's default `load_PDB_components` would type F62
    from the bundled CCD instead, so the pose would load and silently bypass the curated
    .params — `DEFAULT_INIT_FLAGS` turns that off. And Rosetta's own failure is a bare
    "Unrecognized residue: F62" with no mention of which system or spec entry is at fault.
    """
    _require(PROCESSED / "5GMP_clean.pdb")
    from atomfrust.pose import load_complex

    spec = SystemSpec.from_pdb(PROCESSED / "5GMP_clean.pdb", system_id="noparams")
    with pytest.raises(ValueError) as exc:
        load_complex(spec)
    message = str(exc.value)
    assert "noparams" in message
    assert "F62" in message
    assert ".params" in message


def test_chain_filter_drops_other_chains(tmp_path, protein_lines):
    from atomfrust.pose import load_complex

    rest, _ = protein_lines
    renamed = [line[:21] + "B" + line[22:] for line in rest[: len(rest) // 2]]
    pdb = tmp_path / "two_chains.pdb"
    pdb.write_text("".join(renamed + rest[len(rest) // 2 :]) + "END\n")

    spec = SystemSpec.from_pdb(pdb, system_id="chains", chains=["A"], autodetect=False)
    lc = load_complex(spec)
    assert {n.chain for n in lc.nodes} == {"A"}
