# B6 + B7 complete — the ligand is now a node

**Date:** 2026-08-11 06:20
**Plan steps:** B6 and B7 of `plans/frustratometer-ng-plan.md`
**Artifacts:** `atomfrust/pose.py`, `atomfrust/graph.py`, `tests/test_pose.py`,
`tests/test_graph.py`

## Current state

Both done, all acceptance criteria met. **99 unit + 10 integration = 109 tests passing**
(1.8 s and 7.9 s). Legacy pipeline unaffected — `src/run_pipeline.py --help` runs,
`src/test_frustration.py` still collects 13.

This is the A4 correction, implemented. The Stage 6 batch finished at 61/61.

## The two numbers that matter

**1. The new graph reproduces the prototype exactly.** At the prototype's own definition
(Cα–Cα 10 Å, |i−j| ≥ 4) the new engine produces **1772** protein–protein pairs for 5GMP —
identical to `results/5GMP_F62_frustration.parquet`. Same number, entirely different code
path (typed nodes + KD-tree vs the O(N²) loop). Pinned as a test.

**2. Ligand contacts now exist, at the published scale.** The prototype produced zero by
construction. For 5GMP:

| ligand shell (heavy-atom min) | ligand contacts |
|---:|---:|
| 4.0 Å | 12 |
| 4.5 Å | 15 |
| 5.0 Å | 19 |
| **6.0 Å (default)** | **27** |
| 8.0 Å | 47 |

Chen et al. report **16 minimally frustrated** ligand contacts for 5GMP. Total contacts
must exceed that, so the published number implies a shell of ~5 Å or wider; at the 6 Å
default, 16 of 27 being minimally frustrated is ~59%, close to the ~50% minimal fraction the
prototype saw on protein–protein contacts.

That is the first quantitative evidence that ligand-as-node is the right correction — the
*scale* now matches, where before it was 266–407 against a published 4–23. **It is not yet
evidence the frustration values will match**: no energies have been computed. The
comparison is contacts vs contacts, not counts vs counts.

## Acceptance

| criterion | result |
|---|---|
| B6: two-copy ligand → two distinct nodes | ✓ `A:1101`, `A:1102`, both F62 |
| B6: metal → metal node | ✓ typed `metal` |
| B6: protein-only → zero components | ✓ 305 nodes, 0 components |
| B6: 634/Z34 resolves | ✓ `comp_id=634`, `rosetta_name=Z34` |
| B7: KD-tree == brute force | ✓ exact, 6 seeds × 3 cutoff regimes |
| B7: cross-chain pair not sequence-separated | ✓ `seq_sep = -1`, pair retained |
| B7: ligand outside cutoff still in superset | ✓ 6.1 Å pair present, not a contact at 6.0 |
| B7: narrowing a cutoff is a column filter | ✓ same rows, same distances, monotone subset |

## Three real bugs found while building

1. **Duplicate params crashed multi-copy ligands.** Passing the same `.params` twice raised
   `residue type 'F62' already exists in the cache`. N copies share one residue *type*;
   the list is now deduplicated by resolved path. This is exactly the multi-copy case B6
   was specified to support, so it would have shipped broken.

2. **Rosetta silently typed unparametrised ligands from the bundled CCD.** With
   `load_PDB_components` at its default, loading 5GMP with *no params at all* succeeded and
   produced an F62 ligand node. A run could therefore use CCD-derived atom typing while its
   manifest recorded curated `.params` — undetectably. Now
   `-in:file:load_PDB_components false` is in `DEFAULT_INIT_FLAGS`, so an unparametrised
   component fails at load, and the bare Rosetta exit (`Unrecognized residue: F62`) is
   wrapped to name the system, the component and the fix.

   The prototype was not affected retroactively — Stages 4/5 always supplied params — but
   the hazard was latent and would have bitten the library-scale parametrisation in G1.

3. **Node-id collisions were undetected.** `node_id` is the join key between the node and
   pair tables, so a collision would corrupt the graph rather than fail. A valid PDB cannot
   produce one, but a malformed input can; now checked explicitly.

## Design decisions worth recording

**A named contact definition is per-kind-pair.** A ligand has no Cα, so a `ca_ca` definition
judges protein–protein pairs by Cα–Cα and *any pair touching a non-protein node* by
heavy-atom minimum distance against a separate `ligand_cutoff_A`. Without this split,
selecting `ca_ca` would mark every ligand pair absent and silently restore the protein-only
behaviour A4 identified. Guarded by two tests.

**Sequence separation never filters a ligand pair.** `seq_sep` is −1 for any pair that is
not two protein residues of one chain, and −1 is exempt from the filter. Otherwise a
`seq_sep_min` setting would quietly delete the ligand from the graph.

**Two settings changed.** `graph.superset.heavy_cutoff_A` 6.0 → **8.0**: the superset is the
ceiling on post-hoc re-selection, and at 6.0 it left zero headroom against a 6.0 Å ligand
cutoff, so an 8 Å shell could never be requested without regeneration. Added
`contacts.ligand_cutoff_A` (default 6.0, analysis-stage).

**`graph.py` has no PyRosetta dependency** — geometry in, tables out. All 23 of its tests
run in the unit tier on synthetic coordinates, including the brute-force equivalence check.

## Next steps

1. **B8** (`energy.py`) — `pairwise_energy` lifted, `e_fa_rep` returned unsubtracted, and the
   three-mode many-body registry. With B7 done this is where ligand contacts first acquire
   energies, and where `chen_literal` vs `pair_retained` becomes measurable.
2. B4/B5 (provenance, run store) before anything is persisted.
3. Housekeeping: `dvc push` (the batch finished at 61/61); lower
   `checkpoint.save_every_n_decoys` to 10 in the legacy `config.yaml`.
