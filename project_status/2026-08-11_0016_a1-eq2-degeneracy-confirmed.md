# A1 complete — the Eq. 2 degeneracy is universal

**Date:** 2026-08-11 00:16
**Plan step:** A1 of `plans/frustratometer-ng-plan.md`
**Artifacts:** `analysis/diagnose_eq2.py`, `analysis/eq2_degeneracy_report.md`

## Current state

Step A1 is done and its acceptance criterion is met. The plan budgeted A1 against the
19 structures finished at the time of writing; the background Stage 6 batch has since
carried that to **38**, so the check ran over all of them.

The Stage 6 batch (the 42 pending PDB IDs, `--mode single --n-jobs 32 --n_decoys 50`) is
still running. Nothing in A1 touched it — the diagnostic is pure numpy over stored
parquets, with BLAS pinned to one thread and `nice -n 19` so it does not contend.

## What changed

**Confirmed: `E_ij` as implemented carries no pair-specific information.**

`contact_energy_eq2` (`src/frustration.py:238-250`) computes
`E_ij = e_ij + 0.5*sum_{k!=j} e_ik + 0.5*sum_{l!=i} e_jl`. Writing `B_i` for the sum over
*all* of i's partners, the excluded sums are `B_i - e_ij` and `B_j - e_ij`, so
`E_ij = 0.5*(B_i + B_j)` and `e_ij` cancels exactly.

Fitting `E_ij ~ a_i + a_j` by least squares on every stored parquet:

| quantity | result |
|---|---|
| `E_native` exactly additive | **38/38 structures** |
| worst max\|residual\| | 8.6e-14 kcal/mol (float64 noise) |
| typical `E_native` spread | 3.59 kcal/mol |
| observations per parameter | ~5.7 (1693 pairs, 295 residues) — not overfitting |
| `R²(decoy_mean)` | 1.0 — the decoy reference is degenerate the same way |
| `R²(decoy_std)` | 0.932–0.951 |
| `R²(F_index)` | 0.952–0.964 |

**Only ~4% of the variance in the per-contact frustration index is pair-specific.** The
numerator `decoy_mean - E_native` is exactly additive, so every departure from additivity
enters through the division by σ — and σ is itself ~94% additive. The index is a
per-residue quantity wearing a per-contact label, and the effect is near-identical across
all 38 structures.

This makes the pocket-size confound in `CLAUDE.md` **derived rather than observed**: if a
contact's value is fixed by its two residues, contact counts are graph-degree statistics,
and `n_contacts_total` out-predicting `n_minimally_frustrated` follows necessarily.

## Blockers

**A4 is the blocker for everything downstream and is a reading task, not a coding task.**
A1 establishes that the *transcription* at `frustration.py:226-229` is degenerate. It does
not establish that Chen et al.'s published Eq. 2 is. If the published sums run over all
partners without excluding `j`, then `E_ij = e_ij + 0.5*B_i + 0.5*B_j`, the pair term
survives with weight 2, and this is a transcription defect — in which case the 38
completed structures were computed on the wrong formula and re-running them is on the
table.

No Stage C or D work should be run for science before A4 is answered.

## Next steps

1. **A4** — check Eq. 2 against the source paper (PI; blocks B8's default).
2. **A2** — sweep thresholds × pocket selectors over the stored parquets to see whether any
   post-hoc setting reproduces the paper's 4–23 counts in
   `config/pdb_reference_table.csv`. Pure post-hoc, no recomputation, runs now.
3. **A3** — apo control on 5GMP: delete the ligand, re-run at 50 decoys, correlate
   per-contact F holo vs apo. Needs one small PyRosetta job; **hold until the batch
   finishes** to avoid core contention.
4. **B1** — package skeleton, can start in parallel with any of the above.
