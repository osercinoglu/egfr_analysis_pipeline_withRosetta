# A3 complete — the index is ligand-blind, and the native reference provably so

**Date:** 2026-08-11 01:15
**Plan step:** A3 of `plans/frustratometer-ng-plan.md`
**Artifacts:** `analysis/apo_control.py`, `analysis/apo_control_report.md`,
`analysis/apo/5GMP_apo_frustration.parquet`

## Current state

A3 is done. Run on 5GMP (ligand F62, 39 heavy atoms) at 50 decoys, seed 42 — the same seed
the holo run used — on 16 workers alongside the tail of the Stage 6 batch, at the PI's
explicit instruction. No contention was observed (load average ~11 on 64 cores).

Stage A is now complete: A1, A2 and A3 all done. **A4 remains and is the gate.**

## Design property that makes the result strong

The comparison is paired by construction, not by post-hoc matching. `generate_decoy` draws
an identity for each *protein* residue in pose order and skips non-protein residues, and
`native_aa_frequency` counts protein residues only — so holo and apo consume the same RNG
stream and produce **the same decoy sequences**. The only difference between the two runs
is whether the ligand was in the pose while each decoy was repacked, minimised and scored.

The contact set was verified identical (1772 pairs) before any comparison was made.

## What changed

**Verdict: ligand-blind.** Pocket-contact Pearson r = **0.9904**, above the pre-registered
r > 0.95 threshold.

| subset | n | Pearson r | Spearman ρ | mean\|ΔF\| | p95\|ΔF\| | class agreement |
|---|---:|---:|---:|---:|---:|---:|
| all | 1772 | 0.9965 | 0.9974 | 0.0612 | 0.1967 | 98.3% |
| pocket | 235 | 0.9904 | 0.9896 | 0.0933 | 0.2864 | 96.6% |
| non-pocket | 1537 | 0.9971 | 0.9981 | 0.0563 | 0.1876 | 98.5% |

Deleting a 39-heavy-atom inhibitor from the pocket changes the classification of **31 of
1772 contacts (1.7%)**.

**The sharper finding: `E_native` is bit-identical between holo and apo.** Max |Δ| =
**0.000e+00** across all 1772 contacts, 0 differing. This is structural, not numerical
luck:

- `E_ij` sums `e_ik` only over `contacts(i)`, and `contacts` comes from
  `get_protein_contacts` (`src/frustration.py:70-108`), which is protein-only — the ligand
  is never a member of any partner list.
- `pairwise_energy` (`:182-213`) reads the REF2015 `EnergyGraph` edge between two *protein*
  residues, a two-body term the ligand does not participate in.

**The native reference energy provably cannot see the ligand.** The ligand's entire
influence on the index is mediated through the decoys alone — side chains repack
differently when the pocket is occupied, perturbing `decoy_mean` (max |Δ| 0.647) and
`decoy_std` (max |Δ| 0.706). That indirect channel is the whole of the "ligand-aware" claim
in the current implementation.

## Stage A synthesis

The three results form one diagnosis:

- **A1** — `E_ij = 0.5·(B_i + B_j)` exactly; 38/38 structures; only ~4% of the per-contact
  index variance is pair-specific.
- **A2** — no pocket selector or threshold (189 configurations, 50 structures) recovers the
  paper's counts or its affinity relationship; our counts track pocket size to r = 0.791
  while the paper's track it at +0.233; best affinity correlation is +0.280 against the
  published −0.444, i.e. the wrong sign.
- **A3** — the native reference is provably ligand-independent, and the full index is
  ligand-blind at r = 0.9904 on pocket contacts.

**The implemented index is a per-residue, protein-only quantity that happens to be computed
in the presence of a ligand.** That is a complete and coherent diagnosis, and it is not a
dead end: the published quantity carries a real affinity signal in exactly this structure
set (r = −0.444, p = 0.0012, n = 50).

## Blockers

**A4 — check Chen et al. Eq. 2 against the source paper.** Reading task, no compute.
It decides:

1. whether `frustration.py:226-229` is a transcription defect or a faithful rendering;
2. `pair_retained` vs `chen_literal` as the B8 default;
3. whether the 61 completed structures get recomputed.

Note that under `E_ij = e_ij + 0.5·B_i + 0.5·B_j` (sums over *all* partners), the ligand
still would not enter `E_native`, because the partner lists remain protein-only. **Fixing
the many-body formula alone does not make the index ligand-aware** — that requires the
ligand to become a node in the contact graph, which is plan steps B6/B7. A4 and
ligand-as-node are separate fixes and both are needed.

## Next steps

1. **A4** — the paper check. Nothing else should be run for science first.
2. Optional cheap confirmation of A3 on two or three more structures with different ligand
   sizes before this goes in writing. One structure is a screening test, not an estimate.
3. **B1** — package skeleton; independent of A4, can start any time.
4. When the batch finishes: `dvc push`, and lower `checkpoint.save_every_n_decoys` to 10.
