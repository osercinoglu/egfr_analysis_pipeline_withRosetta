# A2 complete — the count gap is not a selector problem

**Date:** 2026-08-11 00:42
**Plan step:** A2 of `plans/frustratometer-ng-plan.md`
**Artifacts:** `analysis/diagnose_counts.py`, `analysis/count_reproduction_report.md`

## Current state

A2 is done with a **decisive negative result**. 50 structures had both a parquet and a
verified pose-numbering mapping; all 50 appear in `config/pdb_reference_table.csv`.
189 configurations were swept (7 pocket shells × 3 selectors × 9 thresholds).

The Stage 6 batch is still running — 11 structures left of 61 at last check. A2 is pure
post-hoc numpy over stored parquets and does not contend with it.

## Method note worth keeping

The pose→PDB mapping was **verified, not assumed**. For every structure the Cα–Cα contact
set (10 Å, |i−j| ≥ 4) was recomputed from `data/processed/{PDB}_clean.pdb` under the
assumption that pose index *k* is the *k*-th standard residue in file order, then checked
for **exact set equality** against the parquet's pairs. All 50 passed. This makes shell
recomputation from the PDBs trustworthy without loading PyRosetta, and the same trick will
work for any future post-hoc analysis of the legacy parquets.

## What changed

**No combination of pocket selector and threshold reproduces the paper's counts.**

| | |
|---|---|
| configurations landing inside the paper's 4–23 range | **0 of 189** |
| …of those with r > 0.5 vs paper counts | **0** |
| baseline (`ca10`/`either_in_shell`/0.78) | counts 255–407, r = 0.214 (p = 0.135) |
| best r vs paper counts anywhere | 0.308, at a degenerate setting where counts collapse to 0–1 |

Scale *is* reachable — `heavy4`/`both_in_shell`/0.78 gives 3–24 against the paper's 4–23 —
but the per-structure ordering is not. Matching the range is a coincidence of scale.

**Two diagnostics show the problem is upstream of the selector:**

1. **Our counts track pocket size; the paper's do not.** Across the sweep
   `r_vs_pocket_size` reaches **0.791**, while the paper's own counts correlate with `ca10`
   pocket size at only **+0.233**. This is the direct observable consequence of the A1
   degeneracy — an index that reduces to a sum of per-residue terms yields counts that are
   graph-degree statistics.
2. **No setting recovers the affinity relationship.** The paper's counts correlate with
   log₂(affinity_pM) at **r = −0.444, p = 0.0012** (n = 50) — the published signal is real
   and significant in this structure set. The best any swept configuration achieves is
   **r = +0.280, p = 0.049**, and the sign is **opposite**: it would say a more minimally
   frustrated pocket binds *more weakly*.

## Blockers

**A4 is now the gate for everything, and its stakes went up.** A1 showed `E_ij` degenerates
to `0.5(B_i + B_j)`. A2 shows the two knobs that *are* adjustable post-hoc cannot recover
either the paper's counts or its affinity relationship. That leaves the contact definition
and the many-body mode — neither variable from stored data, because `E_ij` is stored and
`e_ij` is not, so `E_corrected = E_chen + e_ij` cannot be formed.

If the many-body transcription at `frustration.py:226-229` is wrong, every stored `E_ij`
is the wrong object and no re-selection repairs it.

## Next steps

1. **A4** — check Chen et al. Eq. 2 against the source paper. Reading task, gates B8's
   default and the re-run decision. Nothing else should be run for science first.
2. **A3** — apo control on 5GMP. Start once the batch finishes; it needs one small
   PyRosetta job. Given A2, its expected result (holo ≈ apo) is now more likely, and
   confirming it would close the argument that the current index is ligand-blind.
3. **B1** — package skeleton. Independent of A4, can start any time.

## Note for the write-up

The negative results from A1 and A2 are worth more than they look. Together they say:
the reimplementation computes a well-defined quantity that is *not* the published one, the
discrepancy is in the energy model rather than in downstream bookkeeping, and the published
quantity does carry a real affinity signal in exactly this structure set (r = −0.444,
p = 0.0012). That is a diagnosis, not a dead end.
