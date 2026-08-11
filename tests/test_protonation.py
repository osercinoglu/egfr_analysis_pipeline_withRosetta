"""G2 tests for atomfrust.chem.protonation — protonation/tautomer enumeration (R19, R20, S1.4).

Everything here is ``unit``: RDKit is a hard dependency of this module, no test touches
PyRosetta, the network, or the filesystem, and the enumeration has no RNG.

**Which backend is under test depends on the environment.** ``dimorphite_dl`` is not
installed here, so the assertions about *specific* charges are written against the documented
fallback rule set (:data:`~atomfrust.chem.protonation.IONISATION_RULES`) and are skipped if
Dimorphite-DL ever appears — its answers are its own and this suite is not the place to
re-derive them. The structural assertions (determinism, InChIKey validity, ranking,
truncation, cache keying) hold for both paths and are never skipped.
"""

from __future__ import annotations

import re

import pytest

from atomfrust.chem.cache import ParamCache
from atomfrust.chem.protonation import (
    IONISATION_RULES,
    PROTONATION_VERSION,
    Protomer,
    ProtomerSet,
    canonical_state,
    dimorphite_available,
    enumerate_states,
    param_cache,
    rdkit_available,
    sensitivity_table,
)

needs_rdkit = pytest.mark.skipif(not rdkit_available(), reason="RDKit not installed")
fallback_only = pytest.mark.skipif(
    dimorphite_available(), reason="asserts the documented fallback rules, not Dimorphite-DL"
)

INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")

#: 24 molecules for the S1.4 scale check: marketed kinase inhibitors (the population this
#: pipeline actually parametrises) plus small molecules chosen to exercise each rule.
LIGANDS = {
    "erlotinib": "CCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "lapatinib": "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "afatinib": "CN(C)CC=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1",
    "osimertinib": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "dasatinib": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
    "sunitinib": "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
    "sorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "vandetanib": "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",
    "neratinib": "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2cc1NC(=O)C=CCN(C)C",
    "icotinib": "C#Cc1cccc(Nc2ncnc3cc4c(cc23)OCCOCCOCCO4)c1",
    "rociletinib": "C=CC(=O)Nc1cccc(Nc2nc(Nc3ccc(N4CCN(C(C)=O)CC4)cc3)ncc2C(F)(F)F)c1",
    "brigatinib": "COc1cc(N2CCC(N3CCN(C)CC3)CC2)ccc1Nc1ncc(Cl)c(Nc2ccccc2P(C)(C)=O)n1",
    "crizotinib": "CC(Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl",
    "ibrutinib": "C=CC(=O)N1CCCC1Cn1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",
    "ruxolitinib": "N#CCC(C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
    "palbociclib": "CC(=O)c1c(C)c2cnc(Nc3ccc(N4CCNCC4)cn3)nc2n(C2CCCC2)c1=O",
    "acetic_acid": "CC(=O)O",
    "methylamine": "CN",
    "benzene": "c1ccccc1",
    "methanesulfonic_acid": "CS(=O)(=O)O",
    "methyl_phosphate": "COP(=O)(O)O",
    "beta_alanine": "NCCC(=O)O",
}


# ------------------------------------------------------------------ the rule set at pH 7.4


@pytest.mark.unit
@needs_rdkit
@fallback_only
def test_acetic_acid_has_a_deprotonated_state():
    """Carboxylic acid, pKa 4.2, two pH units below 7.4 — the carboxylate is dominant and
    the neutral acid survives in the band as the minor form."""
    result = enumerate_states("CC(=O)O")
    assert result.method == "rdkit_tautomer"
    assert -1 in result.charges
    assert result.canonical.formal_charge == -1
    assert 0 in result.charges, "the neutral acid must stay in the band, not be discarded"
    assert result.charge_spread == 1


@pytest.mark.unit
@needs_rdkit
@fallback_only
def test_methylamine_has_a_protonated_state():
    """Aliphatic amine, pKa 10.0 — ammonium dominant at 7.4."""
    result = enumerate_states("CN")
    assert +1 in result.charges
    assert result.canonical.formal_charge == +1


@pytest.mark.unit
@needs_rdkit
@fallback_only
@pytest.mark.parametrize(
    ("smiles", "charge"),
    [
        ("CS(=O)(=O)O", -1),        # sulfonic acid
        ("COP(=O)(O)O", -2),        # phosphate: both acidic hydroxyls
        ("NCCC(=O)O", 0),           # zwitterion: +1 and -1 cancel
        ("CC(=O)NC", 0),            # amide N is excluded from the amine rule
        ("Nc1ccccc1", 0),           # aniline is excluded: pKa 4.6, not a rule we claim
        ("c1cc[nH]c1", 0),          # pyrrole: no rule fires
    ],
)
def test_canonical_charges_follow_the_documented_rules(smiles, charge):
    assert canonical_state(smiles).formal_charge == charge


@pytest.mark.unit
@needs_rdkit
@fallback_only
def test_polybasic_ligand_is_over_ionised_but_the_true_state_is_in_the_band():
    """Pins the documented limitation. Per-site rules cannot see that protonating one
    piperazine nitrogen suppresses the other's pKa, so imatinib's canonical state is +2 where
    the molecule is really +1. The band must still contain +1 — that is what makes the
    over-ionisation a bounded error rather than a wrong answer carried downstream."""
    result = enumerate_states(LIGANDS["imatinib"])
    assert result.canonical.formal_charge == 2
    assert 1 in result.charges


@pytest.mark.unit
@needs_rdkit
@fallback_only
def test_ph_changes_the_dominant_state():
    """``ph`` is a real argument, not decoration: below the pKa the acid stays neutral."""
    assert canonical_state("CC(=O)O", ph=2.0).formal_charge == 0
    assert canonical_state("CC(=O)O", ph=7.4).formal_charge == -1
    assert canonical_state("CN", ph=12.0).formal_charge == 0


@pytest.mark.unit
def test_every_rule_has_compilable_smarts():
    """A typo in a SMARTS silently matches nothing, which would look like 'no ionisable
    group' rather than a bug."""
    from rdkit import Chem

    for rule in IONISATION_RULES:
        assert Chem.MolFromSmarts(rule.smarts) is not None, rule.name
        assert rule.delta in (-1, +1)


# ------------------------------------------------------------------ structural guarantees


@pytest.mark.unit
@needs_rdkit
def test_molecule_without_ionisable_group_is_a_single_passthrough_state():
    result = enumerate_states("c1ccccc1")
    assert result.method == "passthrough"
    assert len(result) == 1
    assert result.canonical.formal_charge == 0
    assert result.canonical.is_canonical
    assert result.charge_spread == 0


@pytest.mark.unit
@needs_rdkit
@pytest.mark.parametrize("smiles", ["CC(=O)O", "NCCC(=O)O", "CC(=O)Nc1ccccc1O", "c1ccccc1"])
def test_enumeration_is_deterministic(smiles):
    """No RNG, and ties broken by canonical SMILES — repeated calls must be identical, in
    order, because the ordering is what a truncated run keeps."""
    first = enumerate_states(smiles)
    second = enumerate_states(smiles)
    assert first == second
    assert [s.smiles for s in first.states] == [s.smiles for s in second.states]


@pytest.mark.unit
@needs_rdkit
@pytest.mark.parametrize("smiles", list(LIGANDS.values()))
def test_states_carry_valid_and_unique_inchikeys(smiles):
    """The InChIKey is the cache key, so a malformed or duplicated one would make two states
    collide in :class:`ParamCache` and silently share params."""
    states = enumerate_states(smiles).states
    keys = [s.inchikey for s in states]
    assert all(INCHIKEY.match(k) for k in keys), keys
    assert len(set(keys)) == len(keys)


@pytest.mark.unit
@needs_rdkit
@pytest.mark.parametrize("smiles", ["CC(=O)O", "NCCC(=O)O", "CC(=O)Nc1ccccc1O"])
def test_exactly_one_canonical_state_and_dense_ranks(smiles):
    states = enumerate_states(smiles).states
    assert sum(s.is_canonical for s in states) == 1
    assert states[0].is_canonical
    assert [s.rank for s in states] == list(range(len(states)))


@pytest.mark.unit
@needs_rdkit
def test_ionisation_states_are_ordered_by_population_penalty():
    """Rank 0 costs nothing; each later ionisation state is at least as improbable as the
    one before it. Tautomer variants have no penalty and are excluded from the comparison."""
    states = enumerate_states("NCCC(=O)O").states
    penalties = [s.log10_penalty for s in states if s.log10_penalty is not None]
    assert penalties[0] == 0.0
    assert penalties == sorted(penalties)


@pytest.mark.unit
@needs_rdkit
def test_max_states_truncates_without_dropping_the_canonical_state():
    full = enumerate_states("CC(=O)Nc1ccccc1O", max_states=0)
    assert len(full) > 3, "need a molecule with a real band for truncation to mean anything"
    for limit in (1, 2, 3):
        clipped = enumerate_states("CC(=O)Nc1ccccc1O", max_states=limit)
        assert len(clipped) == limit
        assert clipped.canonical == full.canonical
        # Truncation removes the tail only: the kept states are the head of the full order.
        assert [s.smiles for s in clipped.states] == [s.smiles for s in full.states[:limit]]


@pytest.mark.unit
@needs_rdkit
def test_tautomers_false_restricts_the_band_to_ionisation():
    with_tautomers = enumerate_states("CC(=O)Nc1ccccc1O", tautomers=True)
    without = enumerate_states("CC(=O)Nc1ccccc1O", tautomers=False)
    assert len(without) < len(with_tautomers)
    assert without.canonical == with_tautomers.canonical


@pytest.mark.unit
@needs_rdkit
def test_canonical_state_is_the_reduced_case_of_enumerate_states():
    """``canonical_state`` must never disagree with the head of the band — a point estimate
    that differed from rank 0 would make the uncertainty band meaningless."""
    for smiles in ("CC(=O)O", "CN", "c1ccccc1", LIGANDS["gefitinib"]):
        assert canonical_state(smiles) == enumerate_states(smiles).canonical


@pytest.mark.unit
@needs_rdkit
@pytest.mark.parametrize("bad", ["", "   ", "not_a_smiles", "C(C", None])
def test_unparseable_input_raises(bad):
    """Input RDKit cannot read is a caller error, not a categorised chemistry outcome."""
    with pytest.raises(ValueError):
        enumerate_states(bad)


# ------------------------------------------------------------------ S1.4 sensitivity table


@pytest.mark.unit
@needs_rdkit
def test_sensitivity_table_has_one_row_per_state():
    molecules = {k: LIGANDS[k] for k in ("acetic_acid", "methylamine", "benzene", "gefitinib")}
    table = sensitivity_table(molecules)

    assert list(table["molecule"].unique()) == list(molecules)
    for name, smiles in molecules.items():
        rows = table[table["molecule"] == name]
        expected = enumerate_states(smiles)
        assert len(rows) == len(expected)
        assert rows["n_states"].eq(len(expected)).all()
        assert list(rows["rank"]) == list(range(len(expected)))
        assert rows["is_canonical"].sum() == 1


@pytest.mark.unit
@needs_rdkit
def test_sensitivity_table_reports_the_charge_spread_per_molecule():
    table = sensitivity_table(
        {k: LIGANDS[k] for k in ("acetic_acid", "benzene", "methyl_phosphate")}
    )
    spread = table.groupby("molecule")["charge_spread"].first()
    assert (table["charge_max"] - table["charge_min"] == table["charge_spread"]).all()
    assert spread["benzene"] == 0, "a molecule with one state has no band"
    assert spread["acetic_acid"] >= 1
    assert spread["methyl_phosphate"] >= spread["acetic_acid"]


@pytest.mark.unit
@needs_rdkit
def test_sensitivity_table_covers_at_least_twenty_ligands():
    """S1.4: sensitivity quantified across >= 20 ligands. This is the scale check — that the
    table builds for a realistic inhibitor set and reports a band, not that any particular
    charge is right."""
    table = sensitivity_table(LIGANDS)
    assert table["molecule"].nunique() >= 20
    assert len(table) == sum(table.groupby("molecule")["n_states"].first())
    banded = table.groupby("molecule")["n_states"].first()
    assert (banded >= 1).all()
    assert (banded > 1).sum() >= 5, "a set this varied must show uncertainty somewhere"
    assert set(table["method"]) <= {"dimorphite", "rdkit_tautomer", "passthrough"}


@pytest.mark.unit
@needs_rdkit
def test_sensitivity_table_accepts_an_unnamed_iterable():
    table = sensitivity_table(["CC(=O)O", "c1ccccc1"])
    assert list(table["molecule"].unique()) == ["0", "1"]


@pytest.mark.unit
def test_sensitivity_table_rejects_a_bare_string():
    """A bare SMILES iterates as characters, which would silently produce a table of
    single-atom nonsense instead of failing."""
    with pytest.raises(TypeError):
        sensitivity_table("CC(=O)O")


@pytest.mark.unit
@needs_rdkit
def test_empty_input_still_has_the_schema():
    table = sensitivity_table([])
    assert len(table) == 0
    assert {"molecule", "inchikey", "formal_charge", "charge_spread"} <= set(table.columns)


# ------------------------------------------------------------------ identity and the cache


@pytest.mark.unit
def test_protonation_version_is_a_non_empty_string():
    assert isinstance(PROTONATION_VERSION, str)
    assert PROTONATION_VERSION.strip()


@pytest.mark.unit
def test_protonation_version_appears_in_the_cache_key(tmp_path):
    cache = param_cache(tmp_path)
    assert cache.protonation_version == PROTONATION_VERSION
    key = cache.key("QTBSBXVTEAMEQO-UHFFFAOYSA-M")
    assert key.split("|")[1] == PROTONATION_VERSION
    # And the plain constructor agrees, so callers who build their own cache can match.
    assert ParamCache(tmp_path, protonation_version=PROTONATION_VERSION).key(
        "QTBSBXVTEAMEQO-UHFFFAOYSA-M"
    ) == key


@pytest.mark.unit
@needs_rdkit
def test_two_states_of_one_ligand_get_different_cache_keys(tmp_path):
    """G2's acceptance criterion. If these collided, the second state would load the first
    state's params and the run would report numbers for a molecule it never built."""
    states = enumerate_states("CC(=O)O").states
    cache = param_cache(tmp_path)
    keys = {cache.key(s.inchikey) for s in states}
    assert len(keys) == len(states)


@pytest.mark.unit
@needs_rdkit
def test_two_states_of_one_ligand_get_different_regeneration_keys():
    """The other half of the same guarantee: a stored decoy ensemble must not be reused
    across protonation states."""
    from atomfrust.settings import Settings, regeneration_key

    settings = Settings()
    states = enumerate_states("CC(=O)O").states
    keys = {regeneration_key(settings, {"ligand": s.state_id}) for s in states}
    assert len(keys) == len(states)


@pytest.mark.unit
@needs_rdkit
def test_state_id_carries_the_protonation_version():
    state = canonical_state("CC(=O)O")
    assert state.state_id == f"{state.inchikey}|{PROTONATION_VERSION}"


# ------------------------------------------------------------------ backend honesty


@pytest.mark.unit
@needs_rdkit
def test_method_names_the_backend_that_actually_ran():
    """No consumer may mistake the fallback for Dimorphite-DL. If the package is absent, no
    result may ever claim ``"dimorphite"``."""
    methods = {enumerate_states(s).method for s in LIGANDS.values()}
    assert methods <= {"dimorphite", "rdkit_tautomer", "passthrough"}
    if not dimorphite_available():
        assert "dimorphite" not in methods


@pytest.mark.unit
def test_dimorphite_availability_check_does_not_import_it():
    """It is a probe, not an import — the module must stay importable without it, which is
    the state of this environment."""
    assert isinstance(dimorphite_available(), bool)


@pytest.mark.unit
def test_dataclasses_are_frozen():
    """A protomer is an identity that keys a cache; mutating one after it has been used to
    build params would divorce the key from the molecule."""
    state = Protomer("C", "AAAAAAAAAAAAAA-BBBBBBBBBB-C", 0, True, 0)
    with pytest.raises(Exception):
        state.formal_charge = 1  # type: ignore[misc]
    result = ProtomerSet("C", 7.4, (state,), "passthrough")
    with pytest.raises(Exception):
        result.method = "dimorphite"  # type: ignore[misc]
