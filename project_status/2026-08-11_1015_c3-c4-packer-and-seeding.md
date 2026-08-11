# C3 + C4 — and C3's premise does not survive measurement

**Date:** 2026-08-11 10:15
**Plan steps:** C3 and C4 of `plans/frustratometer-ng-plan.md` (C6–C8 remain)
**Artifacts:** `atomfrust/decoys/identity.py`, `tests/test_decoys.py`

## Current state

**228 tests passing** (200 unit, 28 integration). Legacy pipeline unaffected.
Stage C: C1–C5 done, C6/C7/C8 remain.

## C3: the speed-up is not there, and the default stays sequential

The plan justified a single `PackerTask` as replacing "~300 sequential `MutateResidue` calls
plus a separate repack". Measured on 5GMP:

| | packer_task | sequential |
|---|---:|---:|
| wall clock per decoy | 4.5 s | 4.6 s |

**1.02×. There is no speed-up.** Packing and minimisation dominate; the mutations do not.
The premise was wrong, and it was worth measuring rather than assuming.

The two paths also do not agree. They design the **identical sequence** — 305/305 positions,
verified — but land in different rotamer minima:

| quantity | value |
|---|---|
| median \|Δ e_direct\| | ~3e-06 (the typical pair is untouched) |
| max \|Δ e_direct\| | 586 kcal/mol (a clashing rotamer; `fa_rep`-dominated) |
| `E_ij` correlation (fa_rep excluded) | 0.9982 |
| `E_ij` max \|Δ\| | 9.5, against an `E_native` spread of ~3.6 |

So the divergence is a small tail of clashing rotamers, not a systematic shift — but it is
about 3σ on the pairs it touches, for zero performance gain.

**Decision: `mutation="sequential"` is the default.** `packer_task` is kept for a different
reason than the plan gave — it is the **only** way to express a restricted repack or frozen
region, which is exactly what region-focused decoys (C7, user request 4) require. The
prototype could not express regions at all, having pushed one whole-pose
`RestrictToRepacking()` with no per-residue operation (`frustration.py:347`).

To stop that becoming a silent trap, constructing a generator with `mutation="sequential"`
and a restricted repack region now raises, naming `packer_task` as the fix. Sequential
repacks the whole pose; it would have honoured the region silently and wrongly.

## C4: per-position substreams

Identities are drawn from `SeedSequence(entropy=seed, spawn_key=(decoy_id, position))`, so
the identity at a position no longer depends on how many residues precede it in the mutate
set. Acceptance verified: position-by-position identities are unchanged when the mutate set
grows from 40 residues to 305, at fixed `decoy_id`.

The decoy-level seed stays `base_seed + decoy_id`, so the N-decoy prefix property and
worker-count independence are untouched.

**A limit worth stating:** substreams only apply to the position-independent modes
(`composition`, `uniform20` with `placement="inplace"`). A *permutation* is a property of the
whole set — permuting a 40-element subset cannot agree position-by-position with permuting a
305-element set — so `placement="permute"`, which is the published protocol, keeps a single
stream. Comparisons that vary the mutate set under `permute` are inherently unpaired.

**A test I had to correct.** My first positive control compared a *prefix* of the position
list against the full list and expected them to differ under stream seeding. They do not:
a vectorised `rng.choice(size=N)` is prefix-stable. The set-dependence appears for a
*scattered* subset, where every draw shifts by its index. The control now uses a stride, and
the property it guards is real.

## Next steps

1. **C6** — `native_repack=True` (already the settings default per A4) and the ablation table
   comparing symmetric vs asymmetric native treatment.
2. **C7** — wire the regions through to the packer task, with the cost curve over shell
   radius. `packer_task` is the mechanism; C3's measurement means it should be selected for
   regions, never for speed.
3. **C8** — Monte-Carlo relaxation, which A4 established is the published protocol and which
   the settings default (`relax="mc"`) already names but nothing implements yet.
4. Still outstanding: `dvc push`; lower `checkpoint.save_every_n_decoys` to 10 in the legacy
   `config.yaml`.
