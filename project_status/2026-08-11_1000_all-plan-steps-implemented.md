# All 44 plan steps implemented — and what is actually established

**Date:** 2026-08-11 10:00
**Plan:** `plans/frustratometer-ng-plan.md` — Stages A–G complete
**Package:** `atomfrust/` — 11 CLI subcommands, ~825 tests

## Current state

Every step in the implementation plan is done. The distinction that matters for anyone
reading this later is **built** versus **established**, so this note separates them.

## What is established, with evidence

These are measured results that survived independent verification, not claims.

| finding | evidence |
|---|---|
| The published Eq. 2 degenerates | `E_ij = 0.5·(B_i + B_j)` exactly; additive fit R² = 1.000000 on 38/38 structures (A1). A4 confirmed the exclusions are in the paper, so this is a property of the published equation. |
| The prototype counted the wrong object | The paper's counts are ligand–residue contacts (5GMP = 16 ✓, 5EM8 = 4 ✓); the prototype counted protein–protein pairs in a shell (266–407). A2 swept 189 selector/threshold configurations and closed nothing (A4, A2). |
| The prototype's index is ligand-blind | `E_native` bit-identical holo vs apo (max \|Δ\| = 0.000e+00); pocket-contact r = 0.9904 (A3, now regression test F4). |
| The new engine reproduces the old one | Contacts: exactly 1772 for 5GMP (B7). Native energies: max \|Δ\| = 1.04e-05, r = 1.0000000000 (B8). Whole decoys: 1XKK checkpoint reproduced at max \|Δ\| = 1.2e-05 through a stochastic packing step (C2). Class counts: **exact for all 61 structures** (D3). |
| Symmetric native treatment matters | `frac_highly` collapses 0.068 → 0.010 on 5GMP; `mean_F` more than doubles. Conditions share a decoy ensemble, so this is not sampling noise (C6). |
| The PackerTask speed-up does not exist | 4.5 s vs 4.6 s per decoy (C3), independently reconfirmed by F6's 1.08–1.14× shell speed-up. S1.3's assumed order of magnitude is not there. |
| The n = 19 correlations were artifacts | At n = 61 every raw CI spans zero and max-T adjusted p ≥ 0.69. `n_neutral` (ρ = −0.541, p = 0.017) and `n_contacts_total` (ρ = −0.521, p = 0.022) do not survive. Withdrawn in `CLAUDE.md` (D7). |
| The paper's quantity *does* carry signal here | Its own counts vs log₂(affinity) give r = −0.388, p = 0.002 on the same 61 structures. |
| Mutation sensitivity (S4.5) | Pocket mutation at 4.9 Å shifts 21.41 REU over 29 contacts; the same substitution at 31.1 Å shifts **exactly 0.000** (G8). |
| Multiplicity correction is load-bearing | max-T empirical FWER 0.055 at nominal 0.05, where the unadjusted p rejects at 0.42 (D6). |

## What is built but NOT established

Stating these plainly matters more than the count of green tests.

- **The chemotype axis (G6) — the project's novel contribution — is untested.** Its
  positive-control gate **fails** on the only data available: native F62 scores −446 REU
  against fixture members at −722 / −671 / −352, AUROC 0.333 (0.000 residualised). The gate
  correctly refuses to emit any cross-axis redundancy number. Three hand-written molecules
  placed by MCS alignment with no pose search is neither evidence the axis works nor that it
  is broken. It needs a real property-matched library against a real target.
- **S0.2 is unmeasured.** F2 is fully implemented and SKIPs: no multi-chain structure exists
  under `data/`. This is the only ligand-independent numerical gate in the plan, and the
  three test cases (1BRS, 1AY7, 1JTG) are pinned but not downloaded.
- **F3 is smoke-scale only.** Ligand-incident counts are now the right *kind* of object
  (22/27/37 vs 266–407) and the range overlaps the published 4–23, but counts run
  systematically low and move with decoy count. Full reproduction ≈ **82 core-hours**.
- **F6 fails its own bar** (ρ = 0.82/0.86 against ≥ 0.95). Recorded as a finding about the
  shell approximation, not tuned away.
- **Covalent systems fail the pose validity gate by construction.** The CYS797–SG bond at
  1.81 Å reads as a steric clash to a checker with no covalent concept, flooring the pass
  rate at zero for 15 of 61 complexes. The gate records rather than rejects, so nothing is
  silently dropped — but S4.2 cannot be evaluated on covalent systems as it stands.
- **The environment limits three axes.** `smina`, `gnina`, `posebusters` and `dimorphite_dl`
  are all absent. The pose axis is a *perturbation* ensemble, not a docking ensemble; the
  validity gate runs an 8-check `builtin_subset`; protonation runs an RDKit fallback. Every
  one of these reports which path ran, and none can masquerade as the real thing.

## Defects found in the foundation by building on it

Each was found by an agent working under constraints, in code I had written earlier:

1. **Silent data loss on resume** — `DecoyEnergyWriter` numbered part files from zero on
   every construction, so a resumed shard overwrote `part-000-00000.parquet`. Reproduced,
   fixed, pinned.
2. **A correct spec rejected by its own validator** — `_check_selector` compared `comp_id`
   literally, so the one spec that records both `634` (reporting) and `Z34` (pose) failed
   validation. My B3 test had *pinned the defect as expected behaviour*.
3. **`layer(surface)` selected 2 of 129 residues** — Rosetta's cone-weighted LayerSelector
   defaults multiplied by 4 onto a raw neighbour count. Recalibrated.
4. **Rosetta silently typed unparametrised ligands from the bundled CCD**, so a run could
   use CCD atom typing while its manifest claimed curated params. Now
   `-in:file:load_PDB_components false`.
5. **`CLAUDE.md` was wrong about edgeless pairs** — edge coverage is 100 % out to 12 Å, so no
   10 Å Cα pair is edgeless. Corrected, and F5 independently rediscovered it.
6. **The plan's own `pair_retained` formula was degenerate** as written — the same exclusion
   `chen_literal` makes. Corrected in B8.

## Decisions still owed by the PI

1. **`chen_literal` vs `pair_retained` as the scientific object.** The published formula is
   degenerate; the sound one is not what was published. Both are computed. This should be
   stated in any write-up, not inherited.
2. **`relax="mc"` (published, ~5× cost) vs `"min"`.** At 1000 decoys that is ≈ 6 vs 30
   core-hours per structure for a sub-percentage-point change in class fractions.
3. **Whether to re-run the 61 structures** under the corrected protocol. They reproduce the
   prototype exactly, so they are a valid regression baseline — but they are not a
   reproduction of the paper under any treatment.
4. **The pre-registered primary configuration**, before any grid is swept. Without one, max-T
   over the whole grid is the only defensible reporting mode.

## Housekeeping still outstanding

- `dvc push` — the 61 parquets and checkpoints are not pushed.
- `checkpoint.save_every_n_decoys` is still 50 in the legacy `config.yaml`.
- Nothing in this session has been committed.
