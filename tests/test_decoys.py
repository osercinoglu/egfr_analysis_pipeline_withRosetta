"""C1/C2/C5 tests — decoy protocol, identity generator, backbone invariant."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

PROCESSED = Path("data/processed")
PARAMS = Path("data/ligands/params")


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing (dvc pull): {', '.join(missing)}")


def _context(pdb_stem: str, params_name: str, seq_sep_min: int = 4,
             seq_sep_basis: str = "pose_index"):
    """A DecoyContext over the prototype's own contact definition."""
    from atomfrust.decoys.base import DecoyContext
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex
    from atomfrust.regions import resolve_regions
    from atomfrust.settings import ContactSettings, Settings
    from atomfrust.spec import LigandSpec, SystemSpec

    pdb = PROCESSED / f"{pdb_stem}_clean.pdb"
    params = PARAMS / f"{params_name}.params"
    _require(pdb, params)

    spec = SystemSpec.from_pdb(pdb, system_id=pdb_stem)
    spec = spec.model_copy(
        update={"ligands": (LigandSpec(selector=spec.ligands[0].selector, params=params),)}
    )
    loaded = load_complex(spec)
    settings = Settings()
    _, pairs = build_graph(
        loaded.nodes,
        loaded.geometry,
        settings,
        definitions={
            "proto": ContactSettings(
                cutoff_A=10.0, seq_sep_min=seq_sep_min, seq_sep_basis=seq_sep_basis
            )
        },
    )
    regions = resolve_regions(loaded.nodes, loaded.geometry)
    return (
        DecoyContext(
            pose=loaded.pose,
            nodes=loaded.nodes,
            pairs=pairs,
            regions=regions,
            settings=settings,
        ),
        loaded,
    )


@pytest.fixture(scope="module")
def context_5gmp():
    return _context("5GMP", "F62")


# ================================================================ C1: protocol


def test_null_generator_reproduces_the_native_energies_exactly(context_5gmp):
    """The tightest end-to-end check on extraction: a decoy that changes nothing must
    produce exactly the native energies."""
    from atomfrust.decoys.base import NullGenerator, extract_energies

    context, _ = context_5gmp
    native_ids, native_direct, native_fa = extract_energies(
        context.pose, context.pairs, context.score_function
    )
    result = NullGenerator(context).generate(0)

    assert np.array_equal(result.pair_id, native_ids)
    assert np.array_equal(result.e_direct, native_direct)
    assert np.array_equal(result.e_fa_rep, native_fa)
    assert result.index_row["n_mutated"] == 0


def test_extraction_covers_every_pair_including_ligand_ones(context_5gmp):
    from atomfrust.decoys.base import extract_energies

    context, _ = context_5gmp
    pair_id, e_direct, _ = extract_energies(context.pose, context.pairs, "ref2015")
    assert len(pair_id) == len(context.pairs)
    ligand_mask = (
        (context.pairs.kind_i != "protein") | (context.pairs.kind_j != "protein")
    ).to_numpy()
    assert ligand_mask.sum() > 0
    assert np.abs(e_direct[ligand_mask]).sum() > 0, "ligand pairs must carry energy"


# ============================== C2 acceptance: legacy bit-for-bit reproduction


def test_legacy_protocol_reproduces_a_stored_checkpoint_decoy():
    """The decisive check on C2.

    `checkpoints/1XKK_FMM_ckpt.pkl` holds the prototype's per-decoy many-body energies.
    Regenerating decoy 0 at the same seed under `protocol="legacy"`, with the prototype's
    contact definition and partner map, must reproduce them.

    Every detail of the legacy path matters: the sampling alphabet is
    `list(aa_freq.keys())` in first-appearance order, `np.random.choice` is called once per
    residue rather than once for a vector, and each mutation is applied immediately so
    later draws see earlier ones.
    """
    checkpoint = Path("checkpoints/1XKK_FMM_ckpt.pkl")
    _require(checkpoint)
    from atomfrust.decoys.identity import IdentityDecoyGenerator
    from atomfrust.energy import effective_energy, many_body_energies

    context, _ = _context("1XKK", "FMM")
    stored = pickle.loads(checkpoint.read_bytes())["decoy_energies"]

    generator = IdentityDecoyGenerator(
        context,
        scope="whole_protein",
        identity="composition",
        placement="inplace",
        base_seed=42,
        protocol="legacy",
    )
    result = generator.generate(0)

    pairs = context.pairs
    keep = (
        pairs["in__proto"] & (pairs.kind_i == "protein") & (pairs.kind_j == "protein")
    ).to_numpy()
    assert keep.sum() == 1708, f"contact set is {keep.sum()}, expected the stored 1708"

    e = effective_energy(result.e_direct[keep], result.e_fa_rep[keep], exclude_fa_rep=True)
    E = many_body_energies(
        pairs.node_i.to_numpy()[keep],
        pairs.node_j.to_numpy()[keep],
        e,
        "chen_literal",
    )

    expected = np.array(
        [stored[(int(i), int(j))][0] for i, j in zip(pairs.i[keep], pairs.j[keep])]
    )
    deviation = np.abs(E - expected).max()
    assert deviation < 1e-3, (
        f"max |delta| = {deviation:.3e} against the stored decoy — the legacy protocol "
        "no longer reproduces the prototype"
    )


# ======================================================== C2: the three switches


def test_native_permute_preserves_the_amino_acid_multiset(context_5gmp):
    """The published protocol is a *shuffle*: composition is conserved exactly."""
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    generator = IdentityDecoyGenerator(
        context, identity="native", placement="permute", base_seed=7
    )
    positions = generator.target_positions()
    native = sorted(context.pose.residue(i).name1() for i in positions)

    rng = np.random.default_rng(7)
    drawn = sorted(generator.draw_sequence(positions, rng))
    assert drawn == native, "a permutation must conserve the multiset"


def test_composition_draw_does_not_conserve_the_multiset(context_5gmp):
    """The prototype's i.i.d. draw lets composition fluctuate — the deviation A4 found."""
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    generator = IdentityDecoyGenerator(context, identity="composition", placement="inplace")
    positions = generator.target_positions()
    native = sorted(context.pose.residue(i).name1() for i in positions)

    differs = False
    for seed in range(5):
        drawn = sorted(generator.draw_sequence(positions, np.random.default_rng(seed)))
        differs |= drawn != native
    assert differs, "an i.i.d. draw should not reproduce the native multiset every time"


def test_scope_contact_shell_mutates_only_the_region(context_5gmp):
    from atomfrust.decoys.identity import IdentityDecoyGenerator
    from atomfrust.regions import resolve_regions

    context, loaded = context_5gmp
    shell = resolve_regions(
        loaded.nodes,
        loaded.geometry,
        mutate="protein and within_ca(10.0, ligand)",
        repack="protein",
        minimize="protein",
    )
    scoped = IdentityDecoyGenerator(
        DecoyContextLike(context, shell), scope="contact_shell"
    )
    positions = set(scoped.target_positions())
    expected = set(shell.pose_resnums(loaded.nodes, "mutate"))
    assert positions == expected
    assert 0 < len(positions) < len(
        IdentityDecoyGenerator(context, scope="whole_protein").target_positions()
    )


def DecoyContextLike(context, regions):
    """A shallow copy of a context with different regions."""
    from dataclasses import replace

    return replace(context, regions=regions)


def test_uniform20_can_draw_outside_the_native_alphabet(context_5gmp):
    from atomfrust.decoys.identity import AA_ALPHABET, IdentityDecoyGenerator

    context, _ = context_5gmp
    generator = IdentityDecoyGenerator(context, identity="uniform20")
    drawn = generator.draw_sequence(
        generator.target_positions(), np.random.default_rng(0)
    )
    assert set(drawn) <= set(AA_ALPHABET)
    assert len(set(drawn)) > 15, "a uniform draw should visit most of the alphabet"


def test_seeds_are_base_plus_decoy_id_and_reproducible(context_5gmp):
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    generator = IdentityDecoyGenerator(context, base_seed=100)
    positions = generator.target_positions()

    first = generator.draw_sequence(positions, np.random.default_rng(100 + 3))
    again = generator.draw_sequence(positions, np.random.default_rng(100 + 3))
    other = generator.draw_sequence(positions, np.random.default_rng(100 + 4))
    assert first == again
    assert first != other


# ============================================== C5: the backbone invariant


def test_generated_decoy_leaves_the_backbone_bit_identical(context_5gmp):
    from atomfrust.decoys.identity import IdentityDecoyGenerator, assert_backbone_identical

    context, _ = context_5gmp
    result = IdentityDecoyGenerator(context, base_seed=11).generate(0)
    assert result.index_row["backbone_rmsd"] <= 1e-6
    assert assert_backbone_identical(context.pose, result.pose, tol=1e-6) <= 1e-6
    assert result.index_row["n_mutated"] > 0, "the decoy should differ from the native"


def test_backbone_assertion_actually_fires(context_5gmp):
    """Positive control: without it, the invariant test could pass vacuously."""
    from pyrosetta.rosetta.core.id import AtomID

    from atomfrust.decoys.identity import assert_backbone_identical

    context, _ = context_5gmp
    moved = context.pose.clone()
    residue = moved.residue(1)
    xyz = residue.xyz("CA")
    moved.set_xyz(AtomID(residue.atom_index("CA"), 1), xyz + 1.0)

    with pytest.raises(AssertionError, match="backbone moved"):
        assert_backbone_identical(context.pose, moved, tol=1e-6)


# ==================================== C4: per-position seed substreams


def test_identity_at_a_position_is_independent_of_the_mutate_set(context_5gmp):
    """C4's acceptance. Without substreams the draw at a position depends on how many
    residues precede it in the set, so comparing scopes would differ in *sequence* as
    well as in scope and the comparison would be unpaired."""
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    generator = IdentityDecoyGenerator(
        context, identity="composition", placement="inplace",
        seeding="substream", base_seed=42,
    )
    everything = generator.target_positions()
    assert len(everything) > 250
    small, large = everything[:40], everything

    drawn_small = dict(
        zip(small, generator.draw_sequence(small, np.random.default_rng(0), decoy_id=3))
    )
    drawn_large = dict(
        zip(large, generator.draw_sequence(large, np.random.default_rng(0), decoy_id=3))
    )
    shared = set(small) & set(large)
    assert len(shared) == 40
    assert all(drawn_small[p] == drawn_large[p] for p in shared)


def test_stream_seeding_does_depend_on_the_set(context_5gmp):
    """Positive control: the property is real, not vacuous."""
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    generator = IdentityDecoyGenerator(
        context, identity="composition", placement="inplace", seeding="stream"
    )
    everything = generator.target_positions()
    scattered = everything[::7]

    by_position = dict(zip(everything, generator.draw_sequence(everything, np.random.default_rng(0))))
    drawn_scattered = dict(
        zip(scattered, generator.draw_sequence(scattered, np.random.default_rng(0)))
    )
    # A vectorised draw indexes by *position in the list*, so a scattered subset shifts
    # every draw. (A prefix would coincide, which is why this uses a stride.)
    assert any(by_position[p] != drawn_scattered[p] for p in scattered)


def test_substreams_still_vary_with_decoy_id(context_5gmp):
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    generator = IdentityDecoyGenerator(
        context, identity="composition", placement="inplace", seeding="substream"
    )
    positions = generator.target_positions()[:60]
    a = generator.draw_sequence(positions, np.random.default_rng(0), decoy_id=0)
    b = generator.draw_sequence(positions, np.random.default_rng(0), decoy_id=1)
    again = generator.draw_sequence(positions, np.random.default_rng(0), decoy_id=0)
    assert a == again and a != b


# ============ C3: PackerTask vs sequential — measured, not assumed


def test_packer_task_and_sequential_agree_on_sequence_and_are_compared_on_energy(
    context_5gmp, capsys
):
    """The plan expected 'identical sequences and energies within float tolerance'.

    Sequences must match — both paths draw from the same stream. Energies need not: the two
    consume the packer RNG differently, so rotamer trajectories diverge. Rather than assert
    a tolerance that cannot be justified, this measures the difference and fails only if it
    is large enough to change conclusions.
    """
    import time

    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    common = dict(
        scope="whole_protein", identity="composition", placement="inplace",
        seeding="substream", base_seed=42,
    )
    packer = IdentityDecoyGenerator(context, mutation="packer_task", **common)
    sequential = IdentityDecoyGenerator(context, mutation="sequential", **common)
    assert IdentityDecoyGenerator(context, **common).mutation == "sequential", (
        "sequential is the default: the packer route buys no measurable speed-up"
    )

    positions = packer.target_positions()
    assert packer.draw_sequence(
        positions, np.random.default_rng(42), decoy_id=0, seed=42
    ) == sequential.draw_sequence(
        positions, np.random.default_rng(42), decoy_id=0, seed=42
    ), "the two paths must design the same sequence"

    timings, results = {}, {}
    for name, generator in (("packer_task", packer), ("sequential", sequential)):
        start = time.perf_counter()
        results[name] = generator.generate(0)
        timings[name] = time.perf_counter() - start

    a, b = results["packer_task"], results["sequential"]
    assert np.array_equal(a.pair_id, b.pair_id)
    assert a.index_row["n_mutated"] == b.index_row["n_mutated"]

    delta = np.abs(a.e_direct - b.e_direct)
    spread = float(np.std(b.e_direct))
    relative = float(delta.max() / spread)
    speedup = timings["sequential"] / timings["packer_task"]

    print(
        f"\nC3 packer_task vs sequential on 5GMP decoy 0:"
        f"\n  identical sequence      : yes ({a.index_row['n_mutated']} mutations)"
        f"\n  max |delta e_direct|    : {delta.max():.4f}"
        f"\n  median |delta|          : {np.median(delta):.4f}"
        f"\n  e_direct spread (std)   : {spread:.4f}  -> relative {relative:.3f}"
        f"\n  wall clock              : {timings['packer_task']:.1f}s vs "
        f"{timings['sequential']:.1f}s  ({speedup:.2f}x)"
    )
    # Measured, not assumed. The two design the identical sequence but land in different
    # rotamer minima, so raw e_direct has large outliers (steric clashes, dominated by
    # fa_rep). What matters is the quantity the index is built from.
    from atomfrust.energy import effective_energy, many_body_energies

    node_i = context.pairs.node_i.to_numpy()
    node_j = context.pairs.node_j.to_numpy()
    many = {
        name: many_body_energies(
            node_i, node_j,
            effective_energy(r.e_direct, r.e_fa_rep, exclude_fa_rep=True),
            "chen_literal",
        )
        for name, r in results.items()
    }
    correlation = float(np.corrcoef(many["packer_task"], many["sequential"])[0, 1])
    print(
        f"  E_ij correlation        : {correlation:.6f}"
        f"\n  E_ij max |delta|        : {np.abs(many['packer_task'] - many['sequential']).max():.3f}"
    )
    # Effectively zero (measured ~3e-06): the typical pair is untouched, and the divergence
    # lives in a small tail of clashing rotamers.
    assert np.median(delta) < 1e-3, "the typical pair should be untouched by the packing route"
    assert correlation > 0.99, f"the two routes disagree materially (r = {correlation:.4f})"


# ======================================= C6: symmetric native treatment


def test_native_repack_changes_the_native_pose_but_not_its_backbone(context_5gmp):
    """A4: the paper obtains native energies 'in a similar fashion by omitting the shuffling
    step', so the native is repacked and relaxed too. The prototype scored the crystal pose
    as deposited, making native and decoys asymmetric."""
    from atomfrust.decoys.base import extract_energies
    from atomfrust.decoys.identity import IdentityDecoyGenerator, assert_backbone_identical

    context, _ = context_5gmp
    crystal = IdentityDecoyGenerator(context, native_repack=False).prepare_native()
    assert crystal is context.pose, "native_repack=False must return the crystal pose itself"

    repacked = IdentityDecoyGenerator(context, native_repack=True, relax="min").prepare_native()
    assert repacked is not context.pose
    assert assert_backbone_identical(context.pose, repacked, tol=1e-6) <= 1e-6

    _, crystal_e, crystal_fa = extract_energies(crystal, context.pairs, "ref2015")
    _, repacked_e, repacked_fa = extract_energies(repacked, context.pairs, "ref2015")
    shift = float(
        (repacked_e - repacked_fa).sum() - (crystal_e - crystal_fa).sum()
    )
    print(f"\nC6 native repack shifts total direct energy by {shift:+.1f} REU")
    assert shift < 0.0, "repacking should not raise the native energy"


def test_native_repack_is_deterministic(context_5gmp):
    from atomfrust.decoys.base import extract_energies
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    generator = IdentityDecoyGenerator(context, native_repack=True, base_seed=5)
    a = extract_energies(generator.prepare_native(), context.pairs, "ref2015")[1]
    b = extract_energies(generator.prepare_native(), context.pairs, "ref2015")[1]
    assert np.array_equal(a, b)


# ============================================ C7: region-focused repacking


def test_restricted_repack_touches_fewer_residues_and_is_recorded(context_5gmp):
    from dataclasses import replace

    from atomfrust.decoys.identity import IdentityDecoyGenerator
    from atomfrust.regions import resolve_regions

    context, loaded = context_5gmp
    whole = IdentityDecoyGenerator(
        context, mutation="packer_task", identity="composition", placement="inplace"
    )
    shell_regions = resolve_regions(
        loaded.nodes,
        loaded.geometry,
        mutate="protein and within_ca(8.0, ligand)",
        repack="protein and within_ca(8.0, ligand)",
        minimize="protein and within_ca(8.0, ligand)",
    )
    shell = IdentityDecoyGenerator(
        replace(context, regions=shell_regions),
        scope="contact_shell",
        mutation="packer_task",
        identity="composition",
        placement="inplace",
    )

    assert shell_regions.counts()["repack"] < whole.context.regions.counts()["repack"]

    result = shell.generate(0)
    assert result.index_row["repack_residues"] == shell_regions.counts()["repack"]
    assert result.index_row["wall_s"] > 0
    assert result.index_row["n_mutated"] <= shell_regions.counts()["mutate"]


def test_sequential_refuses_a_restricted_region(context_5gmp):
    """Sequential repacks the whole pose; honouring a region would be a silent lie."""
    from dataclasses import replace

    from atomfrust.decoys.identity import IdentityDecoyGenerator
    from atomfrust.regions import resolve_regions

    context, loaded = context_5gmp
    restricted = resolve_regions(
        loaded.nodes, loaded.geometry,
        mutate="none", repack="protein and within_ca(8.0, ligand)", minimize="none",
    )
    with pytest.raises(ValueError, match="packer_task"):
        IdentityDecoyGenerator(replace(context, regions=restricted), mutation="sequential")


# ================================================ C8: Monte-Carlo relaxation


def test_monte_carlo_relaxation_preserves_the_backbone(context_5gmp):
    from atomfrust.decoys.identity import IdentityDecoyGenerator, assert_backbone_identical

    context, _ = context_5gmp
    result = IdentityDecoyGenerator(
        context, relax="mc", mc_cycles=2, identity="composition", placement="inplace"
    ).generate(0)
    assert result.index_row["relax"] == "mc"
    assert assert_backbone_identical(context.pose, result.pose, tol=1e-6) <= 1e-6


def test_monte_carlo_is_deterministic_and_costs_more_than_one_minimisation(context_5gmp):
    """Characterises `mc` rather than asserting it wins.

    Whether MC reaches a lower energy than a single minimisation is *not* guaranteed and is
    not asserted here. `MonteCarlo.recover_low` selects on the pose **before** the backbone
    restore, and the two relax modes consume the RNG differently, so they are not paired —
    a first attempt at this test asserted `mc <= min` and failed by 10.7 REU on ~2870. That
    comparison belongs in the C8 ablation, where it is measured across structures, not in a
    unit assertion.

    What must hold: the same seed gives the same answer, and more cycles cost more time.
    """
    import pyrosetta

    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    scorefxn = pyrosetta.create_score_function("ref2015")
    common = dict(identity="composition", placement="inplace", base_seed=3)

    generator = IdentityDecoyGenerator(context, relax="mc", mc_cycles=2, **common)
    first, second = generator.generate(0), generator.generate(0)
    assert np.array_equal(first.e_direct, second.e_direct), "mc must be seed-deterministic"

    quick = IdentityDecoyGenerator(context, relax="min", **common).generate(0)
    print(
        f"\nC8 relax on 5GMP decoy 0: min {scorefxn(quick.pose):.1f} REU in "
        f"{quick.index_row['wall_s']:.1f}s | mc(2) {scorefxn(first.pose):.1f} REU in "
        f"{first.index_row['wall_s']:.1f}s"
    )
    assert first.index_row["wall_s"] > quick.index_row["wall_s"], (
        "mc does extra packing cycles and must cost more"
    )


def test_relax_none_is_accepted_and_unknown_modes_raise(context_5gmp):
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context, _ = context_5gmp
    generator = IdentityDecoyGenerator(
        context, relax="none", identity="composition", placement="inplace"
    )
    assert generator.generate(0).index_row["relax"] == "none"

    bad = IdentityDecoyGenerator(context, relax="quantum")
    with pytest.raises(ValueError, match="unknown relax mode"):
        bad.generate(0)
