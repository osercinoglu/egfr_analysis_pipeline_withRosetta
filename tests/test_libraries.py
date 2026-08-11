"""G3 tests for atomfrust.chem.libraries — adapters, roles, has_3d, and the DOE score.

**Fully offline, by construction rather than by skip.** There is no test here that would hit
the network under any flag: the libraries themselves are not redistributable, so the fixtures
under ``tests/data/libraries/`` are a handful of hand-written molecules in each library's real
file layout, and ``scripts/fetch_decoy_libraries.py`` is never invoked. A skipped network test
is a test that runs somewhere; these do not exist.

The ``unit`` tier needs no RDKit either. Roles, provenance, ``has_3d`` and target discovery are
all file parsing; only SMILES round-tripping out of an SDF, the descriptor table and the
InChIKey need RDKit, and those tests carry ``needs_rdkit``. ``doe_score`` is exercised through
its DataFrame path, which is deliberate: the metric is arithmetic over a property table and
should be testable without a chemistry toolkit in the way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomfrust.chem.libraries import (
    ADAPTERS,
    DEKOIS2Adapter,
    DUDEAdapter,
    LocalSDFAdapter,
    MolRecord,
    MUVAdapter,
    ZINCRandomAdapter,
    default_cache_root,
    doe_score,
    get_adapter,
    property_summary,
    rdkit_available,
)
from atomfrust.chem.libraries.base import DecoyLibraryAdapter

FIXTURES = Path(__file__).parent / "data" / "libraries"

needs_rdkit = pytest.mark.skipif(not rdkit_available(), reason="RDKit not installed")


def records_of(adapter, **kwargs) -> list[MolRecord]:
    return list(adapter.records(**kwargs))


def by_role(records) -> dict[str, list[MolRecord]]:
    out: dict[str, list[MolRecord]] = {}
    for record in records:
        out.setdefault(record.role, []).append(record)
    return out


# --------------------------------------------------------------------------------- DUD-E


@pytest.mark.unit
def test_dude_reads_three_roles_from_one_target_directory():
    """The whole point of the role field: one DUD-E target directory holds actives, synthetic
    decoys and measured non-binders, and they are not interchangeable."""
    adapter = get_adapter("dude", FIXTURES)
    assert adapter.available()
    assert adapter.targets() == ["pgh1"]

    groups = by_role(records_of(adapter))
    assert set(groups) == {"active", "property_decoy", "measured_inactive"}
    assert len(groups["active"]) == 4
    assert len(groups["property_decoy"]) == 5
    assert len(groups["measured_inactive"]) == 3
    assert all(r.source == "dude" and r.target == "pgh1" for r in records_of(adapter))


@pytest.mark.unit
def test_dude_source_ids_are_the_librarys_own_identifiers():
    adapter = DUDEAdapter(FIXTURES / "dude")
    groups = by_role(records_of(adapter))
    assert groups["active"][0].source_id == "CHEMBL25"
    assert groups["property_decoy"][0].source_id == "ZINC000012345678"
    assert groups["measured_inactive"][0].source_id == "CHEMBL999801"
    # source_ref is what G6 stores per decoy in decoys/index.parquet.
    assert groups["property_decoy"][0].source_ref == "dude:ZINC000012345678"


@pytest.mark.unit
def test_dude_ism_records_are_not_3d():
    """An .ism file is SMILES. Claiming 3D would make a docking backend skip conformer
    generation and hand Rosetta a molecule with no coordinates."""
    assert all(not r.has_3d for r in records_of(DUDEAdapter(FIXTURES / "dude")))


@pytest.mark.unit
def test_dude_conformer_sdf_is_3d_and_still_only_a_conformer():
    """prefer_sdf reads decoys_final.sdf: coordinates exist, so has_3d is True. It is a free
    molecule's conformer, not a pose — nothing here knows about a receptor."""
    adapter = DUDEAdapter(FIXTURES / "dude", prefer_sdf=True)
    groups = by_role(records_of(adapter))
    assert [r.has_3d for r in groups["property_decoy"]] == [True, True]
    assert groups["property_decoy"][0].source_id == "ZINC000098765432"
    # Measured inactives are SMILES-only and must not vanish in SDF-preferring mode.
    assert len(groups["measured_inactive"]) == 3
    assert all(not r.has_3d for r in groups["measured_inactive"])


@pytest.mark.unit
def test_role_filter_never_mixes_synthetic_with_measured():
    adapter = get_adapter("dude", FIXTURES)
    measured = records_of(adapter, roles=["measured_inactive"])
    synthetic = records_of(adapter, roles=["property_decoy"])
    assert measured and synthetic
    assert {r.role for r in measured} == {"measured_inactive"}
    assert {r.role for r in synthetic} == {"property_decoy"}
    assert not ({r.source_id for r in measured} & {r.source_id for r in synthetic})


@pytest.mark.unit
def test_unknown_role_is_rejected_rather_than_silently_empty():
    with pytest.raises(ValueError, match="unknown role"):
        records_of(get_adapter("dude", FIXTURES), roles=["inactive"])


@pytest.mark.unit
def test_limit_truncates():
    assert len(records_of(get_adapter("dude", FIXTURES), limit=3)) == 3


# ------------------------------------------------------------------------------ DEKOIS 2.0


@pytest.mark.unit
def test_dekois_splits_decoyset_from_actives_by_filename():
    adapter = get_adapter("dekois2", FIXTURES)
    assert adapter.available()
    assert adapter.targets() == ["COX1"]
    groups = by_role(records_of(adapter))
    assert len(groups["property_decoy"]) == 3
    assert len(groups["active"]) == 2
    assert "measured_inactive" not in groups  # DEKOIS has none; MUV and DUD-E do
    assert all(r.source == "dekois2" and r.target == "COX1" for r in records_of(adapter))


@pytest.mark.unit
def test_dekois_has_3d_follows_the_coordinates_not_the_filename():
    """The decoy set was written with a conformer and the actives file with a 2D depiction;
    has_3d must track the z column, which is the only honest source."""
    groups = by_role(records_of(get_adapter("dekois2", FIXTURES)))
    assert all(r.has_3d for r in groups["property_decoy"])
    assert not any(r.has_3d for r in groups["active"])


@pytest.mark.unit
def test_dekois_keeps_numeric_sdf_fields_as_properties():
    groups = by_role(records_of(DEKOIS2Adapter(FIXTURES / "dekois2")))
    assert groups["property_decoy"][0].properties["DEKOIS_MW"] == pytest.approx(193.24)


# ------------------------------------------------------------------------------------ MUV


@pytest.mark.unit
def test_muv_decoys_are_measured_inactives_not_property_decoys():
    """MUV's 'decoys' were assayed. Emitting them as property_decoy would let synthetic and
    measured negatives be pooled, which is exactly what S3.4 must not do."""
    adapter = get_adapter("muv", FIXTURES)
    assert adapter.targets() == ["466"]
    groups = by_role(records_of(adapter))
    assert set(groups) == {"active", "measured_inactive"}
    assert "property_decoy" not in groups
    assert len(groups["active"]) == 3
    assert len(groups["measured_inactive"]) == 4


@pytest.mark.unit
def test_muv_reads_the_third_column_and_keeps_the_pubchem_sid():
    """The .dat header is skipped and SMILES is column 3, not column 1 — reading column 1
    would yield PubChem SIDs as molecules."""
    groups = by_role(records_of(MUVAdapter(FIXTURES / "muv")))
    active = groups["active"][0]
    assert active.smiles == "CC(=O)Nc1ccc(O)cc1"
    assert active.source_id == "14717194"
    assert not active.has_3d
    assert all(r.source == "muv" and r.target == "466" for r in groups["measured_inactive"])


# ----------------------------------------------------------------------------- ZINC control


@pytest.mark.unit
def test_zinc_random_is_a_control_with_no_targets():
    adapter = get_adapter("zinc_random", FIXTURES)
    assert adapter.available()
    assert adapter.targets() == []
    records = records_of(adapter)
    assert len(records) == 5
    assert {r.role for r in records} == {"property_decoy"}
    assert {r.source for r in records} == {"zinc_random"}
    assert records[0].source_id == "ZINC000000123456"
    assert records[0].smiles == "CCN1CCN(CC1)c1ccccn1"  # the header row is not a molecule


@pytest.mark.unit
def test_zinc_random_refuses_a_target():
    """Asking a control for a target means the caller thinks it is a decoy set."""
    with pytest.raises(ValueError, match="no targets"):
        records_of(get_adapter("zinc_random", FIXTURES), target="pgh1")


@pytest.mark.unit
def test_zinc_random_honours_the_role_filter():
    adapter = ZINCRandomAdapter(FIXTURES / "zinc_random")
    assert records_of(adapter, roles=["measured_inactive"]) == []
    assert len(records_of(adapter, roles=["property_decoy"])) == 5


# ------------------------------------------------------------------------------ local files


@pytest.mark.unit
def test_local_infers_roles_from_filenames_and_reads_both_formats():
    adapter = get_adapter("local", FIXTURES)
    assert adapter.available()
    groups = by_role(records_of(adapter))
    assert len(groups["property_decoy"]) == 3   # deepcoy_egfr_decoys.sdf
    assert len(groups["active"]) == 2           # deepcoy_egfr_actives.smi
    assert all(r.has_3d for r in groups["property_decoy"])
    assert not any(r.has_3d for r in groups["active"])
    assert groups["property_decoy"][0].source_id == "DeepCoy_1"
    assert groups["active"][0].source_id == "GEFITINIB"


@pytest.mark.unit
def test_local_role_map_overrides_the_filename_and_source_is_settable():
    """DeepCoy's published sets arrive with the authors' names; provenance should say deepcoy,
    not local, or every hand-supplied set collapses into one bucket in a summary."""
    adapter = LocalSDFAdapter(
        FIXTURES / "local",
        role_map={"deepcoy_egfr_actives.smi": "measured_inactive"},
        source="deepcoy",
    )
    groups = by_role(records_of(adapter))
    assert len(groups["measured_inactive"]) == 2
    assert "active" not in groups
    assert groups["measured_inactive"][0].source_ref.startswith("deepcoy:")


@pytest.mark.unit
def test_local_rejects_an_unknown_role_in_the_role_map():
    with pytest.raises(ValueError, match="not one of"):
        LocalSDFAdapter(FIXTURES / "local", role_map={"*.sdf": "decoy"})


@pytest.mark.unit
def test_local_strict_refuses_to_guess(tmp_path):
    """Named files still parse under strict; an unnameable one raises instead of defaulting,
    because the default (property_decoy) is a claim about the file the caller did not make."""
    assert LocalSDFAdapter(FIXTURES / "local", strict=True).targets() == [""]

    (tmp_path / "batch7.smi").write_text("CCO ETHANOL\n")
    assert list(LocalSDFAdapter(tmp_path).records())[0].role == "property_decoy"
    with pytest.raises(ValueError, match="cannot infer a role"):
        LocalSDFAdapter(tmp_path, strict=True).targets()


@pytest.mark.unit
def test_local_default_role_must_be_a_known_role():
    with pytest.raises(ValueError, match="default_role"):
        LocalSDFAdapter(FIXTURES / "local", default_role="decoy")


# -------------------------------------------------------------------------- missing caches


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_available_is_false_for_a_missing_cache_and_does_not_raise(name, tmp_path):
    """A fresh clone has no cache — licences bar vendoring — so every adapter must degrade to
    'not here' rather than to a traceback in the middle of a sweep."""
    adapter = get_adapter(name, tmp_path / "nothing-here")
    assert adapter.available() is False
    assert adapter.targets() == []
    assert list(adapter.records()) == []


@pytest.mark.unit
def test_adapters_satisfy_the_protocol():
    for name in ADAPTERS:
        assert isinstance(get_adapter(name, FIXTURES), DecoyLibraryAdapter)


@pytest.mark.unit
def test_get_adapter_names_the_known_libraries_on_a_typo():
    with pytest.raises(KeyError) as excinfo:
        get_adapter("dude2", FIXTURES)
    message = str(excinfo.value)
    assert "dude2" in message
    for name in ADAPTERS:
        assert name in message


@pytest.mark.unit
def test_default_cache_root_is_gitignored_and_repo_local():
    root = default_cache_root()
    assert root.name == "libraries" and root.parent.name == ".cache"
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert ".cache/" in gitignore


# ---------------------------------------------------------------------------- descriptors


@pytest.mark.unit
@needs_rdkit
def test_property_summary_has_one_row_per_record_with_the_five_named_descriptors():
    records = records_of(get_adapter("dude", FIXTURES))
    frame = property_summary(records)
    assert len(frame) == len(records)
    for column in ("mw", "hac", "logp", "formal_charge", "rotatable_bonds"):
        assert column in frame.columns
    aspirin = frame.loc[frame["source_id"] == "CHEMBL25"].iloc[0]
    assert aspirin["mw"] == pytest.approx(180.16, abs=0.05)
    assert aspirin["hac"] == 13
    assert aspirin["formal_charge"] == 0
    assert frame["role"].nunique() == 3


@pytest.mark.unit
@needs_rdkit
def test_property_summary_keeps_unparseable_molecules_as_nan_rows():
    bad = MolRecord("not a smiles", None, "local", "X1", "property_decoy")
    frame = property_summary([bad])
    assert len(frame) == 1 and np.isnan(frame.iloc[0]["mw"])


@pytest.mark.unit
@needs_rdkit
def test_inchikey_is_none_unless_requested():
    assert records_of(DUDEAdapter(FIXTURES / "dude"), limit=1)[0].inchikey is None
    with_key = records_of(DUDEAdapter(FIXTURES / "dude", compute_inchikey=True), limit=1)[0]
    assert with_key.inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"  # aspirin


@pytest.mark.unit
@needs_rdkit
def test_sdf_smiles_round_trips():
    record = records_of(DEKOIS2Adapter(FIXTURES / "dekois2"), roles=["active"], limit=1)[0]
    from rdkit import Chem

    assert record.smiles == Chem.CanonSmiles("CC(=O)Oc1ccccc1C(=O)O")


# ----------------------------------------------------------------------------- DOE score


def property_frame(rng: np.random.Generator, n: int, shift: float = 0.0) -> pd.DataFrame:
    """A synthetic property table on the six DOE axes. ``shift`` moves the whole cloud, which
    is what an unmatched decoy set does."""
    return pd.DataFrame(
        {
            "mw": rng.normal(350 + 120 * shift, 40, n),
            "logp": rng.normal(3.0 + 2.5 * shift, 0.8, n),
            "rotatable_bonds": rng.normal(5 + 4 * shift, 1.5, n),
            "hba": rng.normal(5 + 3 * shift, 1.2, n),
            "hbd": rng.normal(2 + 2 * shift, 0.8, n),
            "formal_charge": rng.normal(0 + shift, 0.3, n),
        }
    )


@pytest.mark.unit
def test_doe_is_near_zero_for_decoys_drawn_from_the_actives_own_distribution():
    rng = np.random.default_rng(0)
    actives = property_frame(rng, 60)
    decoys = property_frame(rng, 600)  # same distribution == optimally embedded
    assert doe_score(actives, decoys) < 0.05


@pytest.mark.unit
def test_doe_is_clearly_larger_for_a_mismatched_set():
    rng = np.random.default_rng(0)
    actives = property_frame(rng, 60)
    matched = doe_score(actives, property_frame(rng, 600))
    mismatched = doe_score(actives, property_frame(rng, 600, shift=1.0))
    assert mismatched > 0.25
    assert mismatched > 5 * matched


@pytest.mark.unit
def test_doe_is_bounded_by_one_half():
    rng = np.random.default_rng(1)
    actives = property_frame(rng, 40)
    assert 0.0 <= doe_score(actives, property_frame(rng, 200, shift=20.0)) <= 0.5


@pytest.mark.unit
def test_doe_needs_two_actives_and_a_decoy():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match=">= 2 actives"):
        doe_score(property_frame(rng, 1), property_frame(rng, 10))
    with pytest.raises(ValueError, match=">= 1 decoy"):
        doe_score(property_frame(rng, 10), property_frame(rng, 0))


@pytest.mark.unit
def test_doe_names_a_missing_property_column():
    rng = np.random.default_rng(3)
    frame = property_frame(rng, 10).drop(columns=["hba"])
    with pytest.raises(ValueError, match="hba"):
        doe_score(frame, frame)


@pytest.mark.unit
@needs_rdkit
def test_doe_runs_on_records_end_to_end():
    """The fixture is far too small for the number to mean anything; what is asserted is that
    the records path produces a finite score in range."""
    adapter = get_adapter("dude", FIXTURES)
    score = doe_score(
        records_of(adapter, roles=["active"]), records_of(adapter, roles=["property_decoy"])
    )
    assert 0.0 <= score <= 0.5
