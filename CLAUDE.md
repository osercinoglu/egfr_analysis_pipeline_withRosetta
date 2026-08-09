# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Independent reimplementation of the atomistic frustration method from Chen et al., *Nat. Commun.* 11, 5944 (2020), extended from the paper's 4 EGFR–inhibitor complexes to 61 (51 unique ligands, 15 covalent). Goal: a frustration–affinity correlation, which the paper never computed. The original authors' code was never published, so there is no upstream to diff against — numerical agreement with the paper is not expected, qualitative behaviour (stronger inhibitor → less frustrated pocket) is.

## Environment

Requires a conda env with **PyRosetta** (docs call it `frustrato`). There is no `requirements.txt`; install commands live in `README.md`. Deps: numpy, scipy, pandas, matplotlib, biopython, requests, pyyaml, tqdm, pyarrow, pytest, openbabel-wheel, pyrosetta.

**On this machine no env has PyRosetta installed** (no `frustrato` env exists). Stage 6 and the PyRosetta-marked tests cannot run until it is installed; Stages 1–4 and `scripts/05 --skip-pyrosetta-test` do not need it.

## Commands

Always run from the repo root — `config.yaml` paths are relative to CWD, and `src/run_pipeline.py` imports `frustration` as a top-level module (launch it as a script, **not** `python -m src.run_pipeline`).

```bash
# Stages 1-5 — structure/ligand prep (already run for the current 61-structure set)
python scripts/01_collect_metadata.py
python scripts/02_download_structures.py
python scripts/03_identify_egfr_chain_and_ligand.py
python scripts/04_prepare_complex.py
python scripts/05_prepare_ligand.py

# Every stage has a single-item debug mode:
python scripts/04_prepare_complex.py --pdb-id 5GMP     # scripts 01-04
python scripts/05_prepare_ligand.py --ligand-id 634     # script 05

# Stage 6 — frustration analysis
python src/run_pipeline.py --mode validate --n_decoys 50            # 1LYZ lysozyme sanity check
python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50 --n-jobs 2
python src/run_pipeline.py --mode all --n_decoys 200                # full run
python src/run_pipeline.py --mode all --pdb-ids 1XKK,5GMP --n_decoys 50 --n-jobs 2
python src/run_pipeline.py --mode all --n_decoys 200 --results-dir runs/method-a/results --checkpoints-dir runs/method-a/checkpoints

# Tests (must run from repo root — tests import `src.frustration`)
python -m pytest src/test_frustration.py -v
python -m pytest src/test_frustration.py::test_decoy_backbone_unchanged -v   # single test
```

Cost: roughly 4–5 min per decoy on a ~1700-contact structure. `--mode all --n_decoys 200` over 61 structures is a multi-day job — always prototype at `--n_decoys 50` or on a single structure.
All-mode runs use spawned PyRosetta workers and default to all available logical CPUs;
pass `--n-jobs N` to reduce or explicitly set concurrency.
For a single structure, `--n-jobs N` parallelizes decoys rather than structures.
Pass `--results-dir` and `--checkpoints-dir` together to keep a comparative run's
results and resume state separate from the defaults.

## Architecture

Two halves that communicate only through files on disk.

**Half 1 — `scripts/01`–`05`, data preparation.** Sequential, each stage reads the previous stage's CSV/YAML output. Mostly BioPython + raw PDB/mmCIF text handling; PyRosetta only appears in Stage 5's validation step.

- `01` → `results/metadata/egfr_ligand_inventory.csv` (RCSB Data API metadata, affinity, QC tables). Seeded by `config/pdb_reference_table.csv` (61 PDB IDs + the paper's reference contact counts + affinity in pM).
- `02` → `data/raw_pdb/`, `data/raw_cif/` + `download_manifest.csv` (SHA-256, atom counts, ligand-presence check). Raw files are gitignored.
- `03` → `config/ligand_overrides.yaml` + `chain_ligand_selection.csv`. Picks the EGFR chain by catalytic-motif verification (K745/E762/HRD835-837/DFG855-857/hinge) and the ligand *copy* by distance to the ATP pocket.
- `04` → `data/processed/{PDB}_clean.pdb` + `results/preparation_summary.csv`. Selected chain + one ligand copy only; altloc resolution, missing-residue detection, disulfide scan, covalent bond extracted from mmCIF `_struct_conn`, bridging waters reported before deletion.
- `05` → `data/ligands/params/{LIG}.params` + `ligand_parameterization_status.csv`. Per unique ligand, in priority order: CCD CIF → mol2, else ideal SDF → RDKit → mol2, else extracted ligand PDB → obabel; then `molfile_to_params.py --keep-names`.

**Half 2 — `src/run_pipeline.py` + `src/frustration.py`, Stage 6 analysis.** `frustration.py` is the engine (contacts → Eq. 2 many-body energies → N decoys → Eq. 1 Z-score); `run_pipeline.py` is the driver (pose loading, checkpointing, correlation plot).

**The contract between the halves is `load_candidates()` in `src/run_pipeline.py`.** It inner-joins three CSVs — the Stage 1 inventory, `preparation_summary.csv` (Stage 4), `ligand_parameterization_status.csv` (Stage 5) — and keeps only rows where *both* the complex and its ligand params are `status == "OK"`. Nothing is hardcoded: to add or drop structures, edit `config/pdb_reference_table.csv` and re-run the stages, never a list in the analysis code.

## The method, step by step

### Frustration index

1. **Contact list** (`frustration.py:67`) — every protein residue pair with Cα–Cα ≤ 10 Å and |i−j| ≥ 4. Coarse and Cα-based, even though the energies below are all-atom.
2. **Ligand-pocket residue list** (`frustration.py:108`) — protein residues whose Cα is within 10 Å of any ligand heavy atom. A list of *single residues*, not pairs, and it is **not** fed to the survey; it is only the filter applied in step 9. The ligand never appears in a contact pair — it influences results only by being present in the pose during scoring and repacking.
3. **Direct pairwise energy `e_ij`** (`frustration.py:179`) — REF2015 `EnergyGraph` edge between i and j, `edge.dot(weights)`, then the weighted `fa_rep` contribution subtracted back out when `exclude_fa_rep` is set. Pairs beyond Rosetta's interaction cutoff have no edge and yield `e_ij = 0.0`, so some of the looser 10 Å Cα pairs carry only the background terms from step 4.
4. **Eq. 2 many-body correction** (`frustration.py:213`) — `E_ij = e_ij + 0.5·Σ_{k∈contacts(i),k≠j} e_ik + 0.5·Σ_{l∈contacts(j),l≠i} e_jl`. Partner lists come from `build_contact_partner_map()` over the same step 1 list. This is what makes the index environment-sensitive rather than a bare pair energy.
5. **Native reference pass** (`frustration.py:428`) — score the crystal pose once, compute `E_ij` for every contact.
6. **Decoy ensemble** — see below. Each decoy is rescored and `E_ij` recomputed for the *same residue index pairs* (positions, not identities). One decoy contributes a sample to every contact, which is why a few hundred decoys suffice for ~1700 contacts.
7. **Eq. 1 Z-score** (`frustration.py:489`) — `F_ij = (E_ij_native − mean(E_ij_decoys)) / std(E_ij_decoys, ddof=1)`; σ < 1e-9 → F = 0.
8. **Classify** (`frustration.py:501`) — `F > 0.78` minimally frustrated, `F < −1.0` highly frustrated, else neutral.
9. **Summarize around the ligand** (`frustration.py:525`) — keep contact pairs with at least one partner in the step 2 pocket list, count the three classes. `run_all_egfr` then plots `n_minimally_frustrated` vs `log10(affinity_pM)` and reports Pearson r.

### Decoy generation

Each decoy is a **whole-protein sequence randomization on the native backbone**; one decoy serves every contact at once.

1. **Composition** (`frustration.py:270`) — count the 20 amino acids in *this* protein and normalize. Position-independent: buried and exposed positions draw from the same distribution.
2. **Seeding** (`frustration.py:315`) — seed = base seed + decoy index, applied to Python `random`, NumPy, *and* Rosetta's RNG (`rg().set_seed`) so packing is reproducible too.
3. **Randomize sequence** (`frustration.py:329`) — clone the native pose, then for every protein residue draw a new amino acid from the composition distribution and apply `MutateResidue` immediately. Mutations are applied sequentially to the pose as it is built, so later ones see earlier ones. Non-protein residues (the inhibitor) are skipped and left intact.
4. **Repack** (`frustration.py:342`) — `PackRotamersMover` with `RestrictToRepacking`, backbone untouched.
5. **Chi-only minimization** (`frustration.py:349`) — `MoveMap` with `set_bb(False)`, `set_chi(True)`, `set_jump(False)`; lbfgs MinMover, tol 0.01. Relieves clashes from the repack without moving backbone or jumps.
6. **Hard-restore the backbone** (`frustration.py:366`) — copy native N/CA/C/O coordinates verbatim into the decoy. See the invariant below for why.

Net effect: identical backbone, identical ligand, scrambled sequence, relaxed side chains — so any native-vs-decoy energy difference is attributable to sequence identity at fixed geometry, which is what the Z-score is meant to isolate.

## Invariants and gotchas

- **`ligand_comp_id` vs `rosetta_ligand_comp_id`.** A few real CCD codes collide with Rosetta's internal residue/patch namespace (e.g. `634` → `Z34`). Stage 4 rewrites only the HETATM resName in `_clean.pdb`; the `.params` filename and the pose residue lookup use the *rosetta* name, while checkpoint/result filenames and all reporting use the *real* CCD id. Mixing them up silently produces "ligand not found in pose".
- **`config/ligand_overrides.yaml` is generated, not hand-written** — it's Stage 3's output and Stage 4's only source of chain/ligand-copy truth (61 entries: `egfr_chain`, `ligand_chain`, `ligand_comp_id`, `ligand_resnum`, optional `rosetta_ligand_code`). `chain_selection.default: A` in `config.yaml` is a vestigial fallback and is not what drives selection.
- **Decoy backbones must be bit-identical to native.** `MutateResidue` rebuilds the carbonyl O from idealized geometry and drifts ~0.5–0.8 Å from the crystallographic position, so `generate_decoy()` explicitly restores N/CA/C/O from the native pose after repack+MinMover. This is not a minimizer artifact and cannot be fixed by movemap settings alone; `test_decoy_backbone_unchanged` guards it at 0.05 Å.
- **Ligand atom names must match exactly.** The CCD-CIF-first ordering in Stage 5 exists because params atom names generated any other way (`C1`, `C2`…) don't match the PDB HETATM names (`C16`, `C17`…) and `fill_missing_atoms` then fails. Stage 5 validates the names against *every* processed PDB using that ligand — preserve that check when touching ligand prep.
- **`src/molfile_to_params.py` needs the vendored `src/rosetta_py/`** (not shipped with PyRosetta). `run_molfile_to_params()` sets `PYTHONPATH` to `src/` and runs with `cwd=params_dir`; don't "clean up" either.
- **Z-score sign convention.** Rosetta energies are negative-is-favorable, so `run_frustration_survey()` computes `F = (mean(E_decoy) - E_native) / std(E_decoy)`. More favorable native contacts therefore have positive F and are classified as minimally frustrated. The 1LYZ 50-decoy check produced 43% minimally and 10% highly frustrated contacts; its core minimal fraction (48%) exceeded its surface fraction (38%).
- **Frustration thresholds are hardcoded** at `src/frustration.py:501` (`> 0.78`, `< -1.0`) and duplicated in `run_pipeline.py`'s plot and in the tests. `config.yaml`'s `frustration.thresholds` block is currently **not read by any code** — change both places, or wire the config through.
- **Resume semantics.** `run_frustration_survey` pickles decoy energies to `checkpoints/` every `checkpoint.save_every_n_decoys` decoys. Separately, `run_single_structure` short-circuits entirely if `results/{PDB}_{LIG}_frustration.parquet` already exists. Re-running with a *larger* `--n_decoys` will not extend a completed structure — delete the parquet (and the checkpoint) to force recomputation.
- **Covalent inhibitors are only half-handled.** Stage 4 records the CYS797 linkage from `_struct_conn` into `preparation_summary.csv` (15 structures flagged `is_covalent`), but the CONNECT-record step described in `scripts/05_prepare_ligand.py`'s docstring and in `README.md` **is not implemented** — no code writes it and no `.params` on disk contains one. Stage 6 has no covalent handling at all; those 15 complexes are currently scored as non-covalent.

## State of the repo

61/61 complexes prepared (`preparation_summary.csv` all `OK`), params present for the 51 unique ligands. **Stage 6 has not yet been run on the 61-structure set** — no `results/egfr_frustration_summary.csv`, no `results/*.png`, no `checkpoints/`.

## Stale documents

`README.md` is current. These describe the superseded 25-structure pipeline and will mislead:

- `PROJECT_STATUS.md` — 25 structures, covalent inhibitors *excluded*, old `/home/tugba/...` paths, a "FastRelax → MinMover" next-step that has already been done, and a stage numbering (Stage 0–4) that doesn't match the current Stage 1–6 scheme.
- `notebooks/egfr_frustration_pipeline.ipynb` has been updated for the 61-structure pipeline, but its full Stage 4 survey remains expensive and should be run only after a single-structure prototype.
- `src/search_egfr_structures.py` — legacy RCSB/ChEMBL/BindingDB candidate search, superseded by `scripts/01`. Likewise `results/egfr_*candidates.csv`.
- `src/prepare_structures.py` — still live, but only as a helper library for `scripts/05` (`cif_to_mol2`, `download_ligand_cif`, `run_molfile_to_params`, …). Its own `process_structure()` / `main()` are the old 25-structure entry point.
