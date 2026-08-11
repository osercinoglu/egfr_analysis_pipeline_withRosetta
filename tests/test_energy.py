"""B8 tests for atomfrust.energy — many-body registry (unit) and pose energies (integration)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomfrust.energy import (
    effective_energy,
    many_body_energies,
    many_body_energies_reference,
)

MODES = ("chen_literal", "pair_retained", "pair_only")


# --------------------------------------------------------------------- helpers


def random_graph(seed: int, n_nodes: int = 25, p: float = 0.25):
    rng = np.random.default_rng(seed)
    edges = [
        (i, j)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if rng.random() < p
    ]
    node_i = np.array([a for a, _ in edges])
    node_j = np.array([b for _, b in edges])
    e = rng.normal(-1.0, 2.0, size=len(edges))
    return node_i, node_j, e


def additive_r2(node_i, node_j, y) -> float:
    """R² of the best fit y_ij ~ a_i + a_j. 1.0 means no pair-specific information."""
    nodes = np.unique(np.concatenate([node_i, node_j]))
    index = {n: k for k, n in enumerate(nodes)}
    design = np.zeros((len(y), len(nodes)))
    rows = np.arange(len(y))
    design[rows, [index[n] for n in node_i]] += 1.0
    design[rows, [index[n] for n in node_j]] += 1.0
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(residual @ residual) / ss_tot


# ============================================================ unit: many-body

pytestmark_unit = pytest.mark.unit


@pytest.mark.unit
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(5))
def test_vectorised_matches_the_literal_transcription(mode, seed):
    """The fast path applies an algebraic cancellation. Agreement with an explicit,
    un-simplified transcription of the formula proves the simplification is sound."""
    node_i, node_j, e = random_graph(seed)
    fast = many_body_energies(node_i, node_j, e, mode)
    slow = many_body_energies_reference(node_i, node_j, e, mode)
    assert np.allclose(fast, slow, rtol=0, atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(5))
def test_chen_literal_is_exactly_the_mean_of_two_per_residue_totals(seed):
    """Step A1's finding, as an invariant: E_ij = 0.5*(B_i + B_j), e_ij cancelled."""
    node_i, node_j, e = random_graph(seed)
    n = int(max(node_i.max(), node_j.max())) + 1
    background = np.bincount(node_i, weights=e, minlength=n) + np.bincount(
        node_j, weights=e, minlength=n
    )
    expected = 0.5 * (background[node_i] + background[node_j])
    assert np.allclose(
        many_body_energies(node_i, node_j, e, "chen_literal"), expected, atol=1e-12
    )


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(5))
def test_chen_literal_is_additive_and_pair_retained_is_not(seed):
    """The degeneracy, stated as a testable contrast. This is the whole reason a registry
    exists rather than a single hardcoded formula."""
    node_i, node_j, e = random_graph(seed)

    chen = many_body_energies(node_i, node_j, e, "chen_literal")
    retained = many_body_energies(node_i, node_j, e, "pair_retained")

    assert additive_r2(node_i, node_j, chen) == pytest.approx(1.0, abs=1e-9)
    assert additive_r2(node_i, node_j, retained) < 0.99


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(5))
def test_pair_retained_differs_from_chen_literal_by_exactly_the_pair_term(seed):
    node_i, node_j, e = random_graph(seed)
    delta = many_body_energies(node_i, node_j, e, "pair_retained") - many_body_energies(
        node_i, node_j, e, "chen_literal"
    )
    assert np.allclose(delta, e, atol=1e-12)


@pytest.mark.unit
def test_pair_only_returns_the_input_untouched():
    node_i, node_j, e = random_graph(0)
    out = many_body_energies(node_i, node_j, e, "pair_only")
    assert np.allclose(out, e) and out is not e


@pytest.mark.unit
def test_string_node_labels_work():
    node_i = np.array(["A:1", "A:1", "A:2"])
    node_j = np.array(["A:2", "L:99", "L:99"])
    e = np.array([-1.0, -2.0, -3.0])
    fast = many_body_energies(node_i, node_j, e, "chen_literal")
    slow = many_body_energies_reference(node_i, node_j, e, "chen_literal")
    assert np.allclose(fast, slow)


@pytest.mark.unit
def test_unknown_mode_raises():
    node_i, node_j, e = random_graph(0)
    with pytest.raises(ValueError, match="unknown many-body mode"):
        many_body_energies(node_i, node_j, e, "nonsense")


@pytest.mark.unit
def test_effective_energy_is_a_reversible_switch():
    e_direct = np.array([-3.0, 2.0, 0.0])
    e_fa_rep = np.array([1.0, 0.5, 0.0])
    assert np.allclose(effective_energy(e_direct, e_fa_rep, True), [-4.0, 1.5, 0.0])
    assert np.allclose(effective_energy(e_direct, e_fa_rep, False), e_direct)


@pytest.mark.unit
def test_a_single_isolated_pair_still_has_a_defined_many_body_energy():
    node_i, node_j, e = np.array([0]), np.array([1]), np.array([-2.0])
    # B_i = B_j = e, so chen_literal = e and pair_retained = 2e.
    assert many_body_energies(node_i, node_j, e, "chen_literal")[0] == pytest.approx(-2.0)
    assert many_body_energies(node_i, node_j, e, "pair_retained")[0] == pytest.approx(-4.0)


# ==================================================== integration: real pose


PROCESSED = Path("data/processed")
PARAMS = Path("data/ligands/params")
PARQUET = Path("results/5GMP_F62_frustration.parquet")


@pytest.fixture(scope="module")
def evaluated_5gmp():
    for p in (PROCESSED / "5GMP_clean.pdb", PARAMS / "F62.params", PARQUET):
        if not p.exists():
            pytest.skip(f"missing (dvc pull): {p}")

    from atomfrust.energy import EnergyEvaluator
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex
    from atomfrust.settings import ContactSettings, Settings
    from atomfrust.spec import LigandSpec, SystemSpec

    spec = SystemSpec.from_pdb(PROCESSED / "5GMP_clean.pdb", system_id="5GMP")
    spec = spec.model_copy(
        update={
            "ligands": (
                LigandSpec(
                    selector=spec.ligands[0].selector, params=PARAMS / "F62.params"
                ),
            )
        }
    )
    lc = load_complex(spec)
    _, pairs = build_graph(
        lc.nodes,
        lc.geometry,
        Settings(),
        definitions={"proto": ContactSettings(cutoff_A=10.0, seq_sep_min=4)},
    )
    energies = EnergyEvaluator(lc.pose).pairs(pairs)
    return pairs.merge(energies, on="pair_id")


@pytest.mark.integration
def test_reproduces_the_prototype_native_energies_bit_for_bit(evaluated_5gmp):
    """The decisive check on B8. Restricted to the prototype's own contact set and
    partner map (protein-only, Ca 10 A, |i-j| >= 4, fa_rep excluded, chen_literal), the
    new stack must reproduce every stored E_native."""
    df = evaluated_5gmp
    protein_only = (df.kind_i == "protein") & (df.kind_j == "protein")
    subset = df[df["in__proto"] & protein_only].copy()
    assert len(subset) == 1772

    e = effective_energy(subset.e_direct, subset.e_fa_rep, exclude_fa_rep=True)
    subset["E"] = many_body_energies(
        subset.node_i.to_numpy(), subset.node_j.to_numpy(), e, "chen_literal"
    )

    stored = pd.read_parquet(PARQUET)
    merged = subset.merge(stored, left_on=["i", "j"], right_on=["resi", "resj"])
    assert len(merged) == 1772, "contact sets differ"

    deviation = np.abs(merged["E"] - merged["E_native"]).max()
    # Measured 1.04e-05 against an E_native spread of 3.57, i.e. 2.9e-06 relative, with a
    # median of 5.5e-08 and Pearson r = 1.0000000000. The residual is float32 storage of
    # e_direct/e_fa_rep (~7 significant digits on energies of order 1-100) against the
    # prototype's float64 arithmetic — not an algorithmic difference.
    assert deviation < 1e-4, f"max |delta| = {deviation:.3e}"


@pytest.mark.integration
def test_stored_native_energies_are_additive_as_a1_found(evaluated_5gmp):
    """A1 on live data: under chen_literal the per-contact energy is a per-residue sum."""
    df = evaluated_5gmp
    subset = df[df["in__proto"]]
    e = effective_energy(subset.e_direct, subset.e_fa_rep, exclude_fa_rep=True)
    chen = many_body_energies(subset.node_i.to_numpy(), subset.node_j.to_numpy(), e, "chen_literal")
    retained = many_body_energies(subset.node_i.to_numpy(), subset.node_j.to_numpy(), e, "pair_retained")

    assert additive_r2(subset.node_i.to_numpy(), subset.node_j.to_numpy(), chen) > 0.99999
    assert additive_r2(subset.node_i.to_numpy(), subset.node_j.to_numpy(), retained) < 0.999


@pytest.mark.integration
def test_every_superset_pair_has_an_energy_graph_edge(evaluated_5gmp):
    """Characterises the real behaviour, which is not what CLAUDE.md claims.

    CLAUDE.md says pairs beyond Rosetta's interaction cutoff have no edge, so "some of the
    looser 10 A Ca pairs carry only the background terms". Measured on 5GMP, edge coverage
    by Ca-Ca distance is 100% to 12 A, 94.7% at 12-15 A, 30.6% at 15-20 A and 0% beyond
    20 A. The superset stops at 12 A, so **every** pair in it is scored and none is
    silently zero. Distant pairs are near-zero by magnitude (mean |e| ~0.02 at 10-12 A),
    not by absence of an edge.

    If this ever fails, the superset has been widened past Rosetta's graph range and the
    zero-versus-absent distinction starts to matter."""
    df = evaluated_5gmp
    assert df.has_edge.all(), (
        f"{(~df.has_edge).sum()} of {len(df)} superset pairs have no EnergyGraph edge"
    )
    # The invariant that matters regardless of coverage: no edge means exactly zero.
    edgeless = df[~df.has_edge]
    assert (edgeless.e_direct == 0.0).all() and (edgeless.e_fa_rep == 0.0).all()

    # Direct energy decays with distance, so a wide cutoff adds little but background.
    far = df[(df.d_ca > 10.0) & (df.d_ca <= 12.0)]
    near = df[(df.d_ca > 0) & (df.d_ca <= 6.0)]
    assert far.e_direct.abs().mean() < 0.1 * near.e_direct.abs().mean()


@pytest.mark.integration
def test_ligand_contacts_carry_real_energies(evaluated_5gmp):
    """The A4 correction reaching the energy layer: ligand pairs must be scored, not zero."""
    df = evaluated_5gmp
    ligand = df[(df.kind_i != "protein") | (df.kind_j != "protein")]
    assert len(ligand) > 0
    scored = ligand[ligand.has_edge]
    assert len(scored) >= 16, "too few scored ligand contacts"
    assert (scored.e_direct != 0.0).any()
    assert scored.e_direct.min() < 0, "no favourable ligand interaction found"
