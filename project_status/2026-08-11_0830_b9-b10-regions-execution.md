# B9 + B10 complete — Stage B is done

**Date:** 2026-08-11 08:30
**Plan steps:** B9 and B10 of `plans/frustratometer-ng-plan.md`
**Artifacts:** `atomfrust/regions.py`, `atomfrust/execute.py`,
`tests/test_regions_execute.py`

## Current state

Both done, all acceptance criteria met. **214 tests passing** (200 unit, 16 integration).
Legacy pipeline unaffected.

**Stage B is complete.** The substrate exists end to end: settings → spec → pose → graph →
energies → run store, with region selection and sharded execution. Stage C (the decoy
engine) is unblocked, and it is where the three protocol deviations A4 found actually land.

## B9 — the selector language

A small recursive-descent parser over the grammar in §5, evaluated against the **node table
and geometry**, not a Rosetta pose. That keeps the module PyRosetta-free and unit-testable,
and lets a selection be resolved and inspected before any pose exists; C3 converts the
resulting mask to a `PackerTask` via `pose_resnum`.

**Acceptance: `within_ca(10.0, ligand)` returns exactly the residue set the prototype's
`get_ligand_contacts` (`frustration.py:111`) returns at the same cutoff** — verified against
the legacy function on 5GMP, set equality.

That criterion forced a real distinction. `within(R, expr)` measures heavy-atom to
heavy-atom; the prototype's pocket rule measures **Cα** to ligand heavy atom. They are not
the same set — a long side chain reaching into the site does not qualify its residue under
the Cα rule unless the backbone is close too. Rather than blur them, both exist:
`within_ca` reproduces the prototype exactly, and a test asserts the heavy-atom form is a
strict superset.

Also delivered: `chain`, `resi` (with ranges and negative numbering), `resn`, `layer`
(side-chain neighbour counting, as Rosetta's LayerSelector does by default), `protein`,
`ligand` (every non-protein component — metals and cofactors included, since excluding them
would silently shrink every shell built around them), `water`, `all`, `none`, and
`and`/`or`/`not` with `and` binding tighter.

`ResolvedRegions` enforces the subset invariants as **hard errors**: a mutated residue that
is not repacked would keep a rotamer belonging to its old identity, and a minimised residue
outside the repack set would move without being repacked. Both are silent corruption. An
empty mutate set is legal — a repack-only decoy is a meaningful control.

Non-mutable nodes are intersected out of the mutate set, so a ligand, a metal or a covalent
anchor can never be identity-randomised even when an expression names it (`all` would).

**Deviation:** the `xml:` escape hatch is not implemented. It lexes and raises a message
naming the named selectors instead of a parse error. Deferred rather than half-built.

## B10 — flat work queue and sharding

One flat queue of `(system_id, axis, decoy_id)`. Flat rather than nested because the
prototype's two-level scheme forced `n_jobs_decoys=1` inside every structure worker
(`run_pipeline.py:80`), so neither `--mode all` nor `--mode single` could fill a box unless
the work happened to match its shape.

| property | how it is guaranteed |
|---|---|
| results independent of worker count | tested at 1, 2, 4, 8 workers against a serial reference — exact equality |
| shards disjoint and complete | `decoy_id % n_shards == shard`; 4 shards over 100 decoys reassemble to `range(100)` with no overlap |
| sharded == unsharded | 3 shards merged equal the single-process result |
| resume is subtraction | `plan_units(completed=...)` returns only what is missing |
| a larger `n_decoys` extends | tested: 50 done, 200 requested → plans exactly 50–199 |

That last one retires the prototype's parquet short-circuit (`run_pipeline.py:241`), where
raising `--n_decoys` silently did nothing.

The executor is generic over the task callable, so all of this is tested without PyRosetta;
C2 supplies the real generator. `task_factory` is called once per worker rather than the
task being pickled per unit, so a pose and score function are built once — and a pose is
never pickled, which is why the pool is `spawn`.

## Next steps

Stage C, in order: **C1** (generator protocol) → **C2** (scope × identity × placement, where
the published sequence shuffle lands) → **C3** (single `PackerTask` replacing ~300 sequential
`MutateResidue` calls) → **C4** (per-position seed substreams, without which the C7/F6
pocket-repack comparison is unpaired) → **C5**–**C8**.

Still outstanding on the legacy side: `dvc push` (batch finished 61/61); lower
`checkpoint.save_every_n_decoys` to 10 in `config.yaml`.
