# C6 + C7 + C8 — Stage C complete, and the native-repack correction is large

**Date:** 2026-08-11 11:30
**Plan steps:** C6, C7, C8 — **Stage C is now complete**
**Artifacts:** `atomfrust/decoys/identity.py`, `tests/test_decoys.py`,
`analysis/ablate_native_and_relax.py`, `analysis/ablation_native_relax.md`

## Current state

**235 tests passing** (200 unit, 35 integration). Legacy pipeline unaffected.
Stages A and B complete; Stage C complete. Next is Stage D (the analysis layer).

## The ablation, 15 decoys × 3 structures

Conditions A and B **share a decoy ensemble** and differ only in the native reference, so
C6's effect is isolated with zero sampling noise between them.

| structure | condition | wall s/decoy | frac_minimal | frac_highly | mean F |
|---|---|---:|---:|---:|---:|
| 5GMP | A prototype | 4.9 | 0.3089 | 0.0677 | 0.331 |
| 5GMP | B +C6 native repack | 4.9 | 0.3792 | **0.0098** | **0.793** |
| 5GMP | C +C6+C8 mc | 23.2 | 0.3822 | 0.0100 | 0.797 |
| 1XKK | A prototype | 5.1 | 0.2886 | 0.0280 | 0.604 |
| 1XKK | B +C6 native repack | 5.1 | 0.3175 | 0.0153 | 0.689 |
| 1XKK | C +C6+C8 mc | 24.8 | 0.3245 | 0.0135 | 0.707 |
| 3POZ | A prototype | 5.9 | 0.3711 | 0.0379 | 0.736 |
| 3POZ | B +C6 native repack | 5.9 | 0.3920 | 0.0116 | 0.848 |
| 3POZ | C +C6+C8 mc | 23.3 | 0.3818 | 0.0107 | 0.856 |

### C6 is not a detail

Repacking the native lowers its energy, which raises `F = (mean_decoy − E_native)/σ` across
the board. The effect is systematic and large:

- **`frac_highly` collapses** — 5GMP 0.068 → 0.010 (−86%), 3POZ 0.038 → 0.012 (−69%),
  1XKK 0.028 → 0.015 (−45%).
- **`mean_F` more than doubles on 5GMP** (0.33 → 0.79) and rises on all three.

Because A and B share decoys exactly, none of this is sampling noise. A4 established that
the paper repacks the native (*"obtained in a similar fashion by omitting the shuffling
step"*) while the prototype scored the crystal pose as deposited. **That asymmetry alone
shifts the whole index**, and it inflated the highly-frustrated count by 2–7×.

### C8 costs about 5× for under one percentage point

MC at 5 cycles: 4.9 → 23.2 s per decoy, and `frac_minimal` moves by 0.003 / 0.007 / −0.010.
`mean_F` moves by <0.02. On this evidence the Monte-Carlo relaxation is faithful to the
paper but close to inert at aggregate level, while multiplying compute by five.

**This is a decision for the PI, not for me.** `relax="mc"` is the settings default because
A4 established it is the published protocol; `relax="min"` is one flag away and is ~5×
cheaper. At the paper's 1000 decoys the difference is roughly 6 core-hours versus 30 per
structure.

### An unplanned consistency check

**`lig_frac_highly` is 0.0000 in every structure and every condition** — not one highly
frustrated ligand contact. `config/pdb_reference_table.csv` records
`paper_highly_frustrated_contacts = 0` for these structures too. That is a qualitative
agreement with the published result that nothing in the pipeline was tuned to produce, and
the first such agreement since the ligand became a node.

Magnitudes are in range but not yet comparable: 5GMP condition B gives ~24 minimally
frustrated ligand-incident pairs against the paper's 16, at 15 decoys with a composition
draw rather than 1000 with a shuffle, and over superset pairs rather than a fixed shell.
Treat it as encouraging, not as reproduction — that is step F3's job.

## What was built

**C6** — `prepare_native()` returns either the crystal pose (`native_repack=False`, the
prototype) or a pose repacked and relaxed under the decoy protocol minus the shuffle
(`native_repack=True`, the paper). Backbone-identical and seed-deterministic, both tested.

**C7** — regions reach the packer task: only `repack` is repacked, `frozen` is prevented
outright. Each decoy records `wall_s` and `repack_residues`, so the cost curve over shell
radius is a groupby rather than a separate experiment. Constructing a `sequential` generator
with a restricted region raises, since sequential repacks the whole pose and would honour
the region silently and wrongly.

**C8** — `relax ∈ {min, mc, none}`. The paper specifies only *"a short Monte-Carlo
relaxation… with the backbone fixed"* — no cycle count, move set or temperature — so the
interpretation (re-anneal rotamers, minimise chi, Metropolis, recover the lowest) is
documented at the implementation rather than left implicit. Backbone stays bit-identical.

## A test I had to correct

My first C8 test asserted MC reaches a lower energy than a single minimisation. It failed by
10.7 REU on ~2870. The premise was wrong: `MonteCarlo.recover_low` selects on the pose
**before** the backbone restore, and the two relax modes consume the RNG differently, so they
are not paired. The test now characterises what must hold — seed-determinism and higher cost
— and the energy question is answered by the ablation across structures instead.

## Next steps

1. **Stage D** — `analyze/zscore.py`, `classify.py`, `aggregate.py`. The ablation above
   computes Z-scores inline because D1/D2 do not exist yet; that duplication should not
   outlive Stage D.
2. The PI decision on `relax`: fidelity (`mc`) versus 5× compute (`min`).
3. Still outstanding: `dvc push`; lower `checkpoint.save_every_n_decoys` to 10 in the legacy
   `config.yaml`.
