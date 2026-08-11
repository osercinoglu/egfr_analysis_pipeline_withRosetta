"""G8 tests — the binding-site mutation control (methods S4.5).

S4.5 asks for pocket mutations as a *positive* test: a physically grounded measure must
respond when the pocket is mutated, whereas a model that memorises does not. These tests
cover the three ways that claim could be hollow:

* the mutation never reaches the pose (unit: parsing, spec round-trip, validation;
  integration: the node really carries the new residue);
* a stored wild-type ensemble is silently reused for the mutant (unit: the regeneration key
  genuinely changes, checked through the real digest builder, not by assumption);
* the "response" is not localised — i.e. it would look the same for any mutation anywhere
  (integration: a pocket mutation moves pocket energies, a distant one does not).

The integration tier is deliberately decoy-free: one pose load per case, energies straight
off the scored pose. Nothing here needs an ensemble to make its point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomfrust.mutate import (
    AppliedMutation,
    PairedComparison,
    paired_comparison,
    parse_mutation,
)
from atomfrust.spec import LigandSpec, Mutation, Receptor, SpecError, SystemSpec

PROCESSED = Path("data/processed/5GMP_clean.pdb")
PARAMS = Path("data/ligands/params/F62.params")


# =========================================================================== unit


unit = pytest.mark.unit


def _atom(serial, name, resname, chain, resseq, xyz, het=False, icode=" "):
    rec = "HETATM" if het else "ATOM  "
    x, y, z = xyz
    return (
        f"{rec}{serial:5d} {name:^4s} {resname:>3s} {chain:1s}{resseq:4d}{icode:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
    )


@pytest.fixture
def pdb(tmp_path: Path) -> Path:
    """Chains A and B of three ALA each, plus a ligand LIG at B:501."""
    lines = []
    n = 1
    for chain in ("A", "B"):
        for resseq in range(1, 4):
            for name in ("N", "CA", "C", "O"):
                lines.append(_atom(n, name, "ALA", chain, resseq, (n * 1.0, 0.0, 0.0)))
                n += 1
    lines.append(_atom(n, "C1", "LIG", "B", 501, (1.0, 1.0, 1.0), het=True))
    path = tmp_path / "toy.pdb"
    path.write_text("".join(lines) + "END\n")
    return path


def _spec(pdb: Path, mutations=(), **receptor_kw) -> SystemSpec:
    return SystemSpec(
        system_id="toy",
        receptor=Receptor(path=pdb, mutations=mutations, **receptor_kw),
        pocket={"mode": "whole"},
    )


# ------------------------------------------------------------------- parsing


@unit
@pytest.mark.parametrize(
    "text,expected",
    [
        ("A:790:MET", ("A", 790, "", "MET")),
        ("A:790:M", ("A", 790, "", "MET")),
        ("A:790:met", ("A", 790, "", "MET")),
        ("A:52A:TRP", ("A", 52, "A", "TRP")),
        (" A : 790 : W ", ("A", 790, "", "TRP")),
        ("A:-5:GLY", ("A", -5, "", "GLY")),
    ],
)
def test_parse_mutation_accepts_both_code_lengths(text, expected):
    mut = parse_mutation(text)
    assert (mut.chain, mut.resseq, mut.icode, mut.to) == expected


@unit
def test_parse_mutation_passes_a_mutation_through():
    mut = Mutation(chain="A", resseq=790, to="MET")
    assert parse_mutation(mut) is mut


@unit
@pytest.mark.parametrize(
    "text", ["A:790", "A:790:MET:X", ":790:MET", "A:abc:MET", "A:790:XYZ", "A:790:B", "790"]
)
def test_parse_mutation_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        parse_mutation(text)


@unit
def test_target_is_normalised_and_reversible():
    mut = parse_mutation("A:790:m")
    assert mut.to == "MET"
    assert mut.to1 == "M"
    assert str(mut) == "A:790:MET"
    assert str(parse_mutation("A:52A:W")) == "A:52A:TRP"


# --------------------------------------------------------------- spec plumbing


@unit
def test_receptor_is_wild_type_by_default(pdb: Path):
    assert _spec(pdb).receptor.mutations == ()


@unit
def test_spec_round_trips_mutations_through_yaml(pdb: Path, tmp_path: Path):
    spec = _spec(pdb, mutations=(parse_mutation("A:2:TRP"), parse_mutation("B:3:GLY")))
    path = tmp_path / "spec.yaml"
    path.write_text(spec.to_yaml())

    reloaded = SystemSpec.from_yaml_file(path)
    assert reloaded.receptor.mutations == spec.receptor.mutations
    assert [str(m) for m in reloaded.receptor.mutations] == ["A:2:TRP", "B:3:GLY"]


@unit
def test_one_letter_targets_survive_the_round_trip(pdb: Path, tmp_path: Path):
    """The YAML must carry the normalised form, not the shorthand it was written with."""
    spec = _spec(pdb, mutations=(parse_mutation("A:2:W"),))
    assert "to: TRP" in spec.to_yaml()
    path = tmp_path / "spec.yaml"
    path.write_text(spec.to_yaml())
    assert SystemSpec.from_yaml_file(path).receptor.mutations[0].to == "TRP"


@unit
def test_duplicate_mutation_at_one_position_is_rejected(pdb: Path):
    with pytest.raises(ValueError, match="duplicate mutation"):
        _spec(pdb, mutations=(parse_mutation("A:2:TRP"), parse_mutation("A:2:GLY")))


@unit
def test_mutation_at_a_real_position_validates(pdb: Path):
    assert _spec(pdb, mutations=(parse_mutation("A:2:TRP"),)).check_against_structure() == []


@unit
def test_missing_position_is_reported_not_silently_ignored(pdb: Path):
    problems = _spec(pdb, mutations=(parse_mutation("A:99:TRP"),)).check_against_structure()
    assert len(problems) == 1
    assert "mutation target A:99" in problems[0]

    with pytest.raises(SpecError, match="mutation target"):
        _spec(pdb, mutations=(parse_mutation("A:99:TRP"),)).validate_against_structure()


@unit
def test_missing_chain_is_reported(pdb: Path):
    problems = _spec(pdb, mutations=(parse_mutation("Z:2:TRP"),)).check_against_structure()
    assert len(problems) == 1
    assert "chains present: A, B" in problems[0]


@unit
def test_a_non_protein_target_is_rejected(pdb: Path):
    """Mutating the ligand would be nonsense; it must not reach MutateResidue."""
    problems = _spec(pdb, mutations=(parse_mutation("B:501:TRP"),)).check_against_structure()
    assert len(problems) == 1
    assert "not a protein residue" in problems[0]


@unit
def test_a_mutation_on_a_dropped_chain_is_rejected(pdb: Path):
    problems = _spec(
        pdb, mutations=(parse_mutation("B:2:TRP"),), chains=("A",)
    ).check_against_structure()
    assert len(problems) == 1
    assert "would never reach the graph" in problems[0]


# --------------------------------------------------------- the regeneration key


@unit
def test_mutation_changes_the_regeneration_key(pdb: Path):
    """The point of putting the mutation in the spec — checked end to end, not assumed.

    A mutant is a different system: if the key did not move, ``generate-decoys`` would
    accept a stored wild-type ensemble as the mutant's own and S4.5 would be measuring
    nothing.
    """
    from atomfrust.cli.generate_decoys import _system_digests
    from atomfrust.settings import Settings, regeneration_key

    settings = Settings()
    wild = _spec(pdb)
    mutant = _spec(pdb, mutations=(parse_mutation("A:2:TRP"),))
    other = _spec(pdb, mutations=(parse_mutation("A:2:GLY"),))

    def key(spec: SystemSpec) -> str:
        digests = {"systems": {spec.system_id: _system_digests(spec)}}
        return regeneration_key(settings, digests, "2026.30")

    # The receptor file and its params are byte-identical across all three, so only the
    # spec digest can be carrying the difference.
    assert _system_digests(wild)["receptor_sha256"] == _system_digests(mutant)["receptor_sha256"]
    assert _system_digests(wild)["spec_sha256"] != _system_digests(mutant)["spec_sha256"]

    assert key(wild) != key(mutant)
    assert key(mutant) != key(other)
    assert key(wild) == key(_spec(pdb))
    assert key(mutant) == key(_spec(pdb, mutations=(parse_mutation("A:2:W"),)))


@unit
def test_the_mutant_keeps_the_wild_type_system_id(pdb: Path):
    """Pairing is by ``system_id``; distinguishing is by the key. Not the other way round."""
    wild = _spec(pdb)
    mutant = _spec(pdb, mutations=(parse_mutation("A:2:TRP"),))
    assert wild.system_id == mutant.system_id


# -------------------------------------------------------- paired_comparison


def _summary(system_ids, values, metric="desc__frac_minimal__zscore__default"):
    return pd.DataFrame({"system_id": list(system_ids), metric: list(values)})


@unit
def test_paired_comparison_aligns_descriptor_summaries_by_system_id():
    metric = "desc__frac_minimal__zscore__default"
    wild = _summary(["a", "b", "c"], [0.50, 0.40, 0.60])
    mutant = _summary(["c", "a", "b"], [0.55, 0.30, 0.36])  # deliberately out of order

    cmp = paired_comparison(wild, mutant, metric)
    assert isinstance(cmp, PairedComparison)
    assert list(cmp.keys) == ["a", "b", "c"]
    assert list(cmp.groups) == ["a", "b", "c"]
    np.testing.assert_allclose(cmp.delta, [-0.20, -0.04, -0.05], atol=1e-12)
    assert cmp.metric == metric
    assert cmp.unmatched == ()


@unit
def test_paired_comparison_feeds_paired_delta():
    metric = "desc__frac_minimal__zscore__default"
    wild = _summary(["a", "b", "c", "d"], [0.50, 0.40, 0.60, 0.45])
    mutant = _summary(["a", "b", "c", "d"], [0.30, 0.36, 0.55, 0.20])

    estimate = paired_comparison(wild, mutant, metric).estimate(n_boot=2000, seed=1)
    assert estimate.n_groups == 4
    assert estimate.value == pytest.approx(np.mean([-0.20, -0.04, -0.05, -0.25]))
    assert estimate.p_value is not None and estimate.p_value < 0.2
    assert estimate.method.startswith("paired_delta")


@unit
def test_per_contact_comparison_groups_by_the_contact():
    """Within one system every row is a contact, so the contact is the unit."""
    wild = pd.DataFrame({"pair_id": [0, 1, 2], "F": [1.0, 0.5, -0.2]})
    mutant = pd.DataFrame({"pair_id": [0, 1, 2], "F": [0.2, 0.5, -0.9]})

    cmp = paired_comparison(wild, mutant, "F")
    assert list(cmp.keys) == [0, 1, 2]
    assert list(cmp.groups) == [0, 1, 2]
    np.testing.assert_allclose(cmp.delta, [-0.8, 0.0, -0.7])


@unit
def test_per_contact_comparison_groups_by_system_when_several_are_present():
    wild = pd.DataFrame(
        {"pair_id": [0, 1, 2, 3], "system_id": ["a", "a", "b", "b"], "F": [1.0, 0.5, 0.2, 0.1]}
    )
    mutant = pd.DataFrame(
        {"pair_id": [0, 1, 2, 3], "system_id": ["a", "a", "b", "b"], "F": [0.0, 0.5, 0.0, 0.1]}
    )
    cmp = paired_comparison(wild, mutant, "F")
    assert list(cmp.groups) == ["a", "a", "b", "b"]
    assert cmp.estimate(n_boot=0).n_groups == 2


@unit
def test_unmatched_keys_are_dropped_and_reported():
    metric = "desc__frac_minimal__zscore__default"
    wild = _summary(["a", "b", "gone"], [0.5, 0.4, 0.9])
    mutant = _summary(["a", "b", "new"], [0.3, 0.4, 0.1])

    cmp = paired_comparison(wild, mutant, metric)
    assert list(cmp.keys) == ["a", "b"]
    assert set(cmp.unmatched) == {"gone", "new"}


@unit
def test_paired_comparison_accepts_mappings_of_summaries():
    metric = "desc__frac_minimal__zscore__default"
    wild = {"a": {metric: 0.5}, "b": {metric: 0.4}}
    mutant = {"a": {metric: 0.3}, "b": {metric: 0.1}}
    cmp = paired_comparison(wild, mutant, metric)
    np.testing.assert_allclose(cmp.delta, [-0.2, -0.3])


@unit
def test_paired_comparison_refuses_an_unusable_input():
    metric = "desc__frac_minimal__zscore__default"
    with pytest.raises(KeyError, match="no column"):
        paired_comparison(_summary(["a"], [0.5]), _summary(["a"], [0.5]), "desc__missing")
    with pytest.raises(ValueError, match="present in both"):
        paired_comparison(_summary(["a"], [0.5]), _summary(["b"], [0.5]), metric)
    with pytest.raises(ValueError, match="not unique"):
        paired_comparison(_summary(["a", "a"], [0.5, 0.4]), _summary(["a"], [0.5]), metric)


# ==================================================================== integration


integration = pytest.mark.integration


def _require() -> None:
    missing = [str(p) for p in (PROCESSED, PARAMS) if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")


#: 5GMP is the T790M gatekeeper mutant, so A:790 is already MET; TRP at the gatekeeper is
#: the kind of substitution S4.5 has in mind. A:929 is ~31 A from the ligand — the same
#: substitution, so the comparison controls for the amino acid rather than only the site.
POCKET_MUTATION = "A:790:TRP"
DISTANT_MUTATION = "A:929:TRP"
#: Pocket shell the response is measured over: pairs touching any node within 8 A (heavy
#: minimum) of the ligand. Defined once, on the wild type, and reused for both mutants so
#: the two are scored over the same contacts.
SHELL_A = 8.0


def _mutant_spec(*mutations: str) -> SystemSpec:
    spec = SystemSpec.from_pdb(PROCESSED, system_id="5GMP")
    return spec.model_copy(
        update={
            "ligands": tuple(
                LigandSpec(selector=lig.selector, params=PARAMS) for lig in spec.ligands
            ),
            "receptor": spec.receptor.model_copy(
                update={"mutations": tuple(parse_mutation(m) for m in mutations)}
            ),
        }
    )


def _load(*mutations: str):
    """One pose load, its graph, and its direct pair energies. No decoys anywhere."""
    from atomfrust.energy import EnergyEvaluator
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex

    loaded = load_complex(_mutant_spec(*mutations))
    nodes, pairs = build_graph(loaded.nodes, loaded.geometry)
    energies = EnergyEvaluator(loaded.pose).pairs(pairs)
    return loaded, nodes, pairs.merge(energies, on="pair_id")


@pytest.fixture(scope="module")
def wild_type():
    _require()
    return _load()


@pytest.fixture(scope="module")
def pocket_mutant():
    _require()
    return _load(POCKET_MUTATION)


@pytest.fixture(scope="module")
def distant_mutant():
    _require()
    return _load(DISTANT_MUTATION)


def _pocket_energies(pairs: pd.DataFrame, nodes: pd.DataFrame, shell) -> pd.Series:
    """Effective energy of every pair touching the fixed pocket shell, keyed by node pair."""
    from atomfrust.analyze.aggregate import pocket_mask
    from atomfrust.energy import effective_energy

    selected = pairs[pocket_mask(pairs, nodes, "incident_to", node_ids=shell)]
    return pd.Series(
        effective_energy(
            selected["e_direct"].to_numpy(), selected["e_fa_rep"].to_numpy()
        ),
        index=pd.MultiIndex.from_arrays([selected["node_i"], selected["node_j"]]),
    )


@integration
def test_mutated_pose_carries_the_new_residue_at_the_named_position(
    wild_type, pocket_mutant
):
    wild, wt_nodes, _ = wild_type
    mutant, mut_nodes, _ = pocket_mutant

    wt_node = wild.node_by_key("A", 790)
    mut_node = mutant.node_by_key("A", 790)

    assert wt_node.resname == "MET" and wt_node.name1 == "M"
    assert mut_node.resname == "TRP" and mut_node.name1 == "W"
    assert mutant.pose.residue(mut_node.pose_resnum).name3().strip() == "TRP"

    # The node table, not just the pose, reports the mutant — the graph is built from it.
    assert mut_nodes.loc[mut_nodes["node_id"] == "A:790", "resname"].item() == "TRP"
    assert wt_nodes.loc[wt_nodes["node_id"] == "A:790", "resname"].item() == "MET"

    # Nothing else moved identity.
    changed = {
        row.node_id
        for row in mut_nodes.itertuples()
        if row.resname != wt_nodes.set_index("node_id").loc[row.node_id, "resname"]
    }
    assert changed == {"A:790"}


@integration
def test_what_was_applied_is_recorded_on_the_loaded_complex(wild_type, pocket_mutant):
    assert wild_type[0].mutations == ()
    applied = pocket_mutant[0].mutations
    assert applied == (
        AppliedMutation(
            chain="A",
            resseq=790,
            icode="",
            pose_resnum=applied[0].pose_resnum,
            from_resname="MET",
            to_resname="TRP",
        ),
    )
    assert applied[0].changed


@integration
def test_a_mutation_naming_a_missing_position_fails_loudly():
    from atomfrust.pose import load_complex

    _require()
    with pytest.raises(SpecError, match="mutation target A:9999"):
        load_complex(_mutant_spec("A:9999:TRP"))


@integration
def test_wild_type_and_mutant_are_pairable_by_system_id(wild_type, pocket_mutant):
    """Same id on both sides — that is what makes S4.5's paired test possible."""
    wild, wt_nodes, wt_pairs = wild_type
    mutant, mut_nodes, mut_pairs = pocket_mutant

    from atomfrust.analyze.aggregate import shell_nodes

    shell = shell_nodes(wt_pairs, wt_nodes, shell_A=SHELL_A)
    wt_energy = _pocket_energies(wt_pairs, wt_nodes, shell).sum()
    mut_energy = _pocket_energies(mut_pairs, mut_nodes, shell).sum()

    wild_summary = pd.DataFrame({"system_id": ["5GMP"], "pocket_energy": [wt_energy]})
    mutant_summary = pd.DataFrame({"system_id": ["5GMP"], "pocket_energy": [mut_energy]})
    assert wild.spec.system_id == mutant.spec.system_id
    assert wild.mutations == () and mutant.mutations != ()

    cmp = paired_comparison(wild_summary, mutant_summary, "pocket_energy")
    assert list(cmp.keys) == ["5GMP"]
    assert cmp.unmatched == ()
    assert cmp.delta[0] == pytest.approx(mut_energy - wt_energy)
    assert abs(cmp.delta[0]) > 1e-3


@integration
def test_pocket_mutation_moves_pocket_energies_and_a_distant_one_does_not(
    wild_type, pocket_mutant, distant_mutant
):
    """The substantive S4.5 check: the response is real *and* it is localised.

    Both mutations are the same substitution (X→TRP) applied with the same mechanism; the
    only difference is where. A measure that responded equally to both would be responding
    to something other than the pocket.
    """
    from atomfrust.analyze.aggregate import shell_nodes

    _, wt_nodes, wt_pairs = wild_type
    shell = shell_nodes(wt_pairs, wt_nodes, shell_A=SHELL_A)
    assert "A:790" in set(shell), "the pocket mutation must sit in the shell being measured"
    assert "A:929" not in set(shell), "the distant mutation must sit outside it"

    wt = _pocket_energies(wt_pairs, wt_nodes, shell)
    assert len(wt) > 500

    def response(loaded) -> tuple[float, int]:
        _, nodes, pairs = loaded
        joined = pd.concat(
            {"wt": wt, "mut": _pocket_energies(pairs, nodes, shell)}, axis=1
        ).dropna()
        delta = (joined["mut"] - joined["wt"]).abs()
        return float(delta.sum()), int((delta > 1e-6).sum())

    pocket_shift, pocket_moved = response(pocket_mutant)
    distant_shift, distant_moved = response(distant_mutant)

    # Measured on 5GMP, 2026.30: pocket 21.4 REU over 29 contacts, distant exactly 0.0
    # over 0 — a distant substitution cannot touch a pocket pair's energy at all, since
    # every pair energy is local. The thresholds leave room; the gap is categorical.
    assert pocket_shift > 5.0
    assert pocket_moved >= 5
    assert distant_shift < 0.01
    assert distant_moved == 0
    assert pocket_shift > 100 * max(distant_shift, 1e-6)
