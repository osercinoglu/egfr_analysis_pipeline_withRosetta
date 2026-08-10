# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Independent reimplementation of the atomistic frustration method from Chen et al., *Nat. Commun.* 11, 5944 (2020), extended from the paper's 4 EGFR–inhibitor complexes to 61 (51 unique ligands, 15 covalent). Goal: a frustration–affinity correlation, which the paper never computed. The original authors' code was never published, so there is no upstream to diff against — numerical agreement with the paper is not expected, qualitative behaviour (stronger inhibitor → less frustrated pocket) is.

`.github/copilot-instructions.md` covers the same ground for Copilot. When you change a command, invariant, or convention here, update that file too.

## Environment

```bash
conda env create -f environment.yml     # creates env `frustrato` (python 3.10)
conda activate frustrato
# or, if it already exists:
conda env update -n frustrato -f environment.yml --prune
```

`environment.yml` is the only dependency manifest — there is no `requirements.txt`. It installs `pyrosetta-installer`, **not PyRosetta itself**. PyRosetta is a separate step:

```bash
# Resolve the newest release wheel for this Python/platform, then install it
WHEEL=$(python - <<'PY'
import re, urllib.request
base = ("https://west.rosettacommons.org/pyrosetta/release/release/"
        "PyRosetta4.Release.python310.ubuntu.wheel/")
html = urllib.request.urlopen(base, timeout=60).read().decode()
wheels = re.findall(r'href="(pyrosetta-\d+\.\d+[^"]*\.whl)"', html)
key = lambda w: tuple(int(n) for n in re.match(r"pyrosetta-(\d+)\.(\d+)", w).groups())
print(base + max(wheels, key=key))
PY
)
python -m pip install "$WHEEL"      # ~1.7 GB download, several minutes
python -c "import pyrosetta; pyrosetta.init('-mute all'); print('PyRosetta OK')"
```

**Do not use `pyrosetta_installer.install_pyrosetta()`** even though `environment.yml` pulls the package in. It resolves the wheel through `latest.html`, which points at `pyrosetta-0-cp310-cp310-linux_x86_64.whl` — a placeholder that 404s on both mirrors, so the call fails without installing anything. The snippet above reads the same directory listing and picks the newest *versioned* wheel instead. Two related traps: the wheel filename is URL-encoded (`%2B` for `+`) and must stay that way, and on this machine only the **west** mirror is usable — `graylab.jhu.edu` (the installer's `mirror=1`) fails TLS with `CERTIFICATE_VERIFY_FAILED`, so it is not a fallback.

Adjust `python310`/`ubuntu` in the URL if the env's Python version or platform changes; the sibling directories under `.../release/release/` list what exists.

**On this machine `frustrato` is installed and complete, PyRosetta included** — `pyrosetta 2026.30+release.bc091c65b8`, installed 2026-08-10 from the west mirror by the snippet above. The whole pipeline runs. Without PyRosetta only Stages 1–4, `scripts/05 --skip-pyrosetta-test`, and the non-PyRosetta tests work.

## Data sync (DVC)

`data/`, `results/`, and `checkpoints/` are **gitignored and DVC-tracked** — never `git add` them. The remote is Google Cloud Storage:

```bash
git pull && dvc pull     # start of work on any machine
dvc push && git push     # after generating outputs or checkpoints
```

- Remote `storage` → `gs://egfr-analysis-pipeline-withrosetta/` (`.dvc/config`), with `core.autostage = true`, so `dvc add` stages the `.dvc` file for you.
- Tracked as three whole directories: `data.dvc`, `results.dvc`, `checkpoints.dvc`. README's setup section describes a finer-grained layout (`data/processed`, `data/ligands/params` as separate targets) and a different bucket URL — the three whole-dir `.dvc` files at the repo root are what actually exists.
- If `data/processed/` or `results/metadata/` look empty, the fix is `dvc pull`, not re-running the prep stages.
- Auth: `gcloud auth application-default login`, or `GOOGLE_APPLICATION_CREDENTIALS` / `dvc remote modify --local storage credentialpath ...`.

## Commands

Always run from the repo root — `config.yaml` paths are relative to CWD, and `src/run_pipeline.py` imports `frustration` as a top-level module (launch it as a script, **not** `python -m src.run_pipeline`).

```bash
# Stages 1-5 — structure/ligand prep (already run for the current 61-structure set)
python scripts/01_collect_metadata.py
python scripts/02_download_structures.py
python scripts/03_identify_egfr_chain_and_ligand.py
python scripts/04_prepare_complex.py
python scripts/05_prepare_ligand.py            # add --skip-pyrosetta-test without PyRosetta

# Every stage has a single-item debug mode:
python scripts/04_prepare_complex.py --pdb-id 5GMP     # scripts 01-04
python scripts/05_prepare_ligand.py --ligand-id 634    # script 05

# Stage 6 — frustration analysis
python src/run_pipeline.py --mode validate --n_decoys 50            # 1LYZ lysozyme sanity check
python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50 --n-jobs 2
python src/run_pipeline.py --mode all --n_decoys 200                # full run
python src/run_pipeline.py --mode all --pdb-ids 1XKK,5GMP --n_decoys 50 --n-jobs 2
python src/run_pipeline.py --mode all --n_decoys 200 --results-dir runs/method-a/results --checkpoints-dir runs/method-a/checkpoints
python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50 --save-structures-dir runs/method-a/structures

# Tests (must run from repo root — tests import `src.frustration`)
python -m pytest src/test_frustration.py -v
python -m pytest src/test_frustration.py::test_decoy_backbone_unchanged -v   # single test
```

Tests marked `@skip_if_no_pyrosetta` auto-skip when PyRosetta is absent. With PyRosetta installed the baseline is **13 passed** (~33 s, 1LYZ fetched from RCSB); without it, **8 passed, 5 skipped** — so on a machine lacking PyRosetta a green run is not proof the PyRosetta paths work.

Cost: measured at **~6.7 min per decoy per core** on a ~1700-contact structure (12-core box, 2026.30) — the "4–5 min" in older notes is optimistic. At `--n_decoys 50` that is ~32 min per structure with 10 decoy workers. `--mode all --n_decoys 200` over 61 structures is a multi-day job — always prototype at `--n_decoys 50` or on a single structure.

**For a batch of N structures, loop `--mode single` rather than using `--mode all`.** `run_all_egfr` pins `n_jobs_decoys=1` in every worker, so `--mode all --n-jobs 6` uses only 6 cores for 6 structures. Running structures sequentially with `--mode single --n-jobs 10` uses all 10 and finished 6 structures in 3 h 33 m where `--mode all` would have taken ~3.75 h on the same box. This is safe because decoy *i* is seeded `seed + i` in both the sequential and parallel paths (`frustration.py:666` and `:441`), so results are bit-identical regardless of worker count. Follow the batch with one `--mode all --pdb-ids <everything done>` pass to build the summary and plot — completed structures short-circuit on their parquet in seconds.

Do not run other PyRosetta work while a Stage 6 job is live. A concurrent pose-loading script cost ~10 min on one structure by contending for the same cores.

Flags worth knowing:
- `--n-jobs N` — in `--mode all`, spawned workers over *structures*; in `--mode single`, spawned workers over *decoys*. Defaults to all logical CPUs (`default_worker_count()`, `run_pipeline.py:42`).
- `--results-dir` / `--checkpoints-dir` — pass together to keep a comparative run's results and resume state off the defaults.
- `--save-structures-dir ROOT` — writes `ROOT/{PDB}_{LIG}/native.pdb` and `ROOT/{PDB}_{LIG}/decoys/decoy_XXXX.pdb`. Only reachable via this flag (it sets `paths.saved_structures`, which is absent from `config.yaml`). If the structure's parquet already exists the run short-circuits and only `native.pdb` is written — decoys cannot be reconstructed, and the code logs a warning saying so.

## Architecture

Two halves that communicate only through files on disk.

**Half 1 — `scripts/01`–`05`, data preparation.** Sequential, each stage reads the previous stage's CSV/YAML output. Mostly BioPython + raw PDB/mmCIF text handling; PyRosetta only appears in Stage 5's validation step.

- `01` → `results/metadata/egfr_ligand_inventory.csv` (RCSB Data API metadata, affinity, QC tables). Seeded by `config/pdb_reference_table.csv` (61 PDB IDs + the paper's reference contact counts + affinity in pM).
- `02` → `data/raw_pdb/`, `data/raw_cif/` + `download_manifest.csv` (SHA-256, atom counts, ligand-presence check). Raw files are gitignored *and* excluded from DVC — they are re-downloadable.
- `03` → `config/ligand_overrides.yaml` + `chain_ligand_selection.csv`. Picks the EGFR chain by catalytic-motif verification (K745/E762/HRD835-837/DFG855-857/hinge) and the ligand *copy* by distance to the ATP pocket.
- `04` → `data/processed/{PDB}_clean.pdb` + `results/preparation_summary.csv`. Selected chain + one ligand copy only; altloc resolution, missing-residue detection, disulfide scan, covalent bond extracted from mmCIF `_struct_conn`, bridging waters reported before deletion.
- `05` → `data/ligands/params/{LIG}.params` + `results/metadata/ligand_parameterization_status.csv`. Per unique ligand, in priority order: CCD CIF → mol2, else ideal SDF → RDKit → mol2, else extracted ligand PDB → obabel; then `molfile_to_params.py --keep-names`.

**Half 2 — `src/run_pipeline.py` + `src/frustration.py`, Stage 6 analysis.** `frustration.py` is the engine (contacts → Eq. 2 many-body energies → N decoys → Eq. 1 Z-score); `run_pipeline.py` is the driver (pose loading, checkpointing, correlation plot).

**The contract between the halves is `load_candidates()` (`run_pipeline.py:140`).** It inner-joins three CSVs — the Stage 1 inventory, `preparation_summary.csv` (Stage 4), `ligand_parameterization_status.csv` (Stage 5) — and keeps only rows where *both* the complex and its ligand params are `status == "OK"`. Nothing is hardcoded: to add or drop structures, edit `config/pdb_reference_table.csv` and re-run the stages, never a list in the analysis code.

**Parallelism is process-spawn, two levels, never nested.** PyRosetta is not fork-safe, so every worker calls `pyrosetta.init()` itself in an initializer and rebuilds its own pose:
- `--mode all` → `run_all_egfr` (`run_pipeline.py:442`) uses a `spawn` `Pool` with `_worker_init` + `imap_unordered` over structures, and forces `n_jobs_decoys=1` inside each worker.
- `--mode single` → `_generate_parallel_decoys` (`frustration.py:488`) uses a `spawn` pool with `_decoy_worker_init` (`frustration.py:394`), which reloads the pose from `(processed_pdb, params_file)` rather than pickling it, and batches decoys so checkpoints land incrementally.

## The method, step by step

Line numbers below are current but drift with edits — the function names are the stable handle.

### Frustration index

1. **Contact list** (`get_protein_contacts`, `frustration.py:70`) — every protein residue pair with Cα–Cα ≤ 10 Å and |i−j| ≥ 4. Coarse and Cα-based, even though the energies below are all-atom.
2. **Ligand-pocket residue list** (`get_ligand_contacts`, `frustration.py:111`) — protein residues whose Cα is within 10 Å of any ligand heavy atom. A list of *single residues*, not pairs, and it is **not** fed to the survey; it is only the filter applied in step 9. The ligand never appears in a contact pair — it influences results only by being present in the pose during scoring and repacking.
3. **Direct pairwise energy `e_ij`** (`pairwise_energy`, `frustration.py:182`) — REF2015 `EnergyGraph` edge between i and j, `edge.dot(weights)`, then the weighted `fa_rep` contribution subtracted back out when `exclude_fa_rep` is set. Pairs beyond Rosetta's interaction cutoff have no edge and yield `e_ij = 0.0`, so some of the looser 10 Å Cα pairs carry only the background terms from step 4.
4. **Eq. 2 many-body correction** (`contact_energy_eq2`, `frustration.py:216`) — `E_ij = e_ij + 0.5·Σ_{k∈contacts(i),k≠j} e_ik + 0.5·Σ_{l∈contacts(j),l≠i} e_jl`. Partner lists come from `build_contact_partner_map()` (`frustration.py:253`) over the same step 1 list. This is what makes the index environment-sensitive rather than a bare pair energy.
5. **Native reference pass** (`frustration.py:619`) — score the crystal pose once, compute `E_ij` for every contact.
6. **Decoy ensemble** — see below. Each decoy is rescored and `E_ij` recomputed for the *same residue index pairs* (positions, not identities). One decoy contributes a sample to every contact, which is why a few hundred decoys suffice for ~1700 contacts.
7. **Eq. 1 Z-score** (`frustration.py:688`) — `F_ij = (mean(E_ij_decoys) − E_ij_native) / std(E_ij_decoys, ddof=1)`; σ < 1e-9 → F = 0.
8. **Classify** (`frustration.py:703`) — `F > 0.78` minimally frustrated, `F < −1.0` highly frustrated, else neutral.
9. **Summarize around the ligand** (`summarize_ligand_frustration`, `frustration.py:727`) — keep contact pairs with at least one partner in the step 2 pocket list, count the three classes. `run_all_egfr` then plots `n_minimally_frustrated` vs `log10(affinity_pM)` and reports Pearson r.

### Decoy generation

Each decoy is a **whole-protein sequence randomization on the native backbone** (`generate_decoy`, `frustration.py:298`); one decoy serves every contact at once.

1. **Composition** (`native_aa_frequency`, `frustration.py:273`) — count the 20 amino acids in *this* protein and normalize. Position-independent: buried and exposed positions draw from the same distribution.
2. **Seeding** (`frustration.py:320`) — seed = base seed + decoy index, applied to Python `random`, NumPy, *and* Rosetta's RNG (`rg().set_seed`) so packing is reproducible too.
3. **Randomize sequence** (`frustration.py:342`) — clone the native pose, then for every protein residue draw a new amino acid from the composition distribution and apply `MutateResidue` immediately. Mutations are applied sequentially to the pose as it is built, so later ones see earlier ones. Non-protein residues (the inhibitor) are skipped and left intact.
4. **Repack** (`frustration.py:348`) — `PackRotamersMover` with `RestrictToRepacking`, backbone untouched.
5. **Chi-only minimization** (`frustration.py:364`) — `MoveMap` with `set_bb(False)`, `set_chi(True)`, `set_jump(False)`; lbfgs MinMover, tol 0.01. Relieves clashes from the repack without moving backbone or jumps.
6. **Hard-restore the backbone** (`frustration.py:378`) — copy native N/CA/C/O coordinates verbatim into the decoy. See the invariant below for why.

Net effect: identical backbone, identical ligand, scrambled sequence, relaxed side chains — so any native-vs-decoy energy difference is attributable to sequence identity at fixed geometry, which is what the Z-score is meant to isolate.

## Invariants and gotchas

- **`ligand_comp_id` vs `rosetta_ligand_comp_id`.** A few real CCD codes collide with Rosetta's internal residue/patch namespace (e.g. `634` → `Z34`). Stage 4 rewrites only the HETATM resName in `_clean.pdb`; the `.params` filename and the pose residue lookup use the *rosetta* name, while checkpoint/result filenames and all reporting use the *real* CCD id. Mixing them up silently produces "ligand not found in pose".
- **`config/ligand_overrides.yaml` is generated, not hand-written** — it's Stage 3's output and Stage 4's only source of chain/ligand-copy truth (61 entries: `egfr_chain`, `ligand_chain`, `ligand_comp_id`, `ligand_resnum`, optional `rosetta_ligand_code`). `chain_selection.default: A` in `config.yaml` is a vestigial fallback and is not what drives selection.
- **Decoy backbones must be bit-identical to native.** `MutateResidue` rebuilds the carbonyl O from idealized geometry and drifts ~0.5–0.8 Å from the crystallographic position, so `generate_decoy()` explicitly restores N/CA/C/O from the native pose after repack+MinMover. This is not a minimizer artifact and cannot be fixed by movemap settings alone; `test_decoy_backbone_unchanged` guards it at 0.05 Å.
- **Ligand atom names must match exactly.** The CCD-CIF-first ordering in Stage 5 exists because params atom names generated any other way (`C1`, `C2`…) don't match the PDB HETATM names (`C16`, `C17`…) and `fill_missing_atoms` then fails. Stage 5 validates the names against *every* processed PDB using that ligand — preserve that check when touching ligand prep.
- **`data/ligands/params/` holds stale files from the old 25-structure pipeline.** 112 `.params` are on disk; only the 51 in `ligand_parameterization_status.csv` are current. The 25 named `{PDB}_{LIG}.params` are legacy `prepare_structures.py` output — Stage 6 resolves params as `{rosetta_ligand_comp_id}.params`, so the extras are inert, but don't treat a file's presence as evidence a ligand is prepared.
- **`src/molfile_to_params.py` needs the vendored `src/rosetta_py/`** (not shipped with PyRosetta). `run_molfile_to_params()` sets `PYTHONPATH` to `src/` and runs with `cwd=params_dir`; don't "clean up" either.
- **Z-score sign convention.** Rosetta energies are negative-is-favorable, so `run_frustration_survey()` computes `F = (mean(E_decoy) - E_native) / std(E_decoy)`. More favorable native contacts therefore have positive F and are classified as minimally frustrated. The 1LYZ 50-decoy check produced 43% minimally and 10% highly frustrated contacts; its core minimal fraction (48%) exceeded its surface fraction (38%).
- **Frustration thresholds are hardcoded in three places** — `frustration.py:703` (`> 0.78`, `< -1.0`), the plot guides at `run_pipeline.py:415-416`, and `test_frustration.py:140-143`. `config.yaml`'s `frustration.thresholds` block is **not read by any code**; change all three, or wire the config through.
- **Resume semantics.** `run_frustration_survey` pickles decoy energies to `checkpoints/{PDB}_{LIG}_ckpt.pkl` every `checkpoint.save_every_n_decoys` decoys. Separately, `run_single_structure` short-circuits at `run_pipeline.py:241` if `results/{PDB}_{LIG}_frustration.parquet` already exists — it reloads the pose to recompute ligand contacts, then returns the summary without touching decoys. Re-running with a *larger* `--n_decoys` will not extend a completed structure — delete the parquet **and** the checkpoint to force recomputation.
- **`save_every_n_decoys` defaults to 50, which equals the usual `--n_decoys 50`** — so a standard run checkpoints only once, at the very end, and an interruption at decoy 49 restarts from zero. Lower it in `config.yaml` before any long run (`--n_decoys 200` and up), where it actually buys resumability.
- **`egfr_frustration_summary.csv` is overwritten with only the current invocation's structures.** `run_all_egfr` builds the DataFrame from the tasks it ran, so `--mode all --pdb-ids A,B` replaces a 19-row summary with a 2-row one, and `egfr_correlation.png` is redrawn from that subset. The per-structure parquets are the durable record; the summary is not cumulative. Rebuild it by passing every finished PDB ID in one `--mode all` call, or use `--results-dir` for subset runs you want to keep.
- **`n_minimally_frustrated` is confounded by pocket size — do not read the headline plot as a frustration result.** `n_contacts_total` (the ligand-pocket contact count) varies 476–733 across structures and is itself the strongest single predictor of affinity. At n=19 the metrics that correlate significantly with `log10(affinity_pM)` are `n_neutral` (Spearman ρ = −0.541, p = 0.017) and `n_contacts_total` (ρ = −0.521, p = 0.022) — *not* `n_minimally_frustrated` (ρ = −0.347, p = 0.145). The neutral count is by definition the contacts that are not frustrated either way, so its winning is a red flag that the counts are tracking pocket size. `frac_minimally` divides that out: it flips sign (ρ = +0.351 overall; +0.745, p = 0.008 on the 11 WT non-covalent structures), i.e. a *less* frustrated pocket binds *more weakly* — opposite to the paper's hypothesis. Sharpest illustration: 3POZ (23 pM) and 3W2O (8400 pM) share ligand 03P and have `frac_minimally` 0.5007 vs 0.4964 despite 365-fold different affinity. Prefer `frac_minimally`, or regress the count on affinity with `n_contacts_total` as a covariate. `_plot_correlation` and the notebook both still plot the raw count.
- **Covalent inhibitors are only half-handled.** Stage 4 records the CYS797 linkage from `_struct_conn` into `preparation_summary.csv` (15 of 61 flagged `is_covalent`), but the CONNECT-record step claimed by `scripts/05_prepare_ligand.py`'s docstring (lines 19–22: "this script adds the CONNECT record") **is not implemented** — no code writes it and no `.params` on disk contains one. The docstring is the only place that misstates this; `README.md` describes the gap correctly. Stage 6 has no covalent handling at all; those 15 complexes are currently scored as non-covalent.

## State of the repo

- Prep complete: 61/61 complexes `OK` in `preparation_summary.csv`, 51/51 ligands `OK`, 61 `_clean.pdb` files.
- **Stage 6: 19 of 61 structures done, all at `--n_decoys 50`.** Parquets in `results/{PDB}_{LIG}_frustration.parquet` with matching `checkpoints/*_ckpt.pkl` for: 1XKK, 2ITO, 2RGP, 3POZ, 3W2O, 3W2Q, 3W2S, 3W32, 3W33, 4JQ7, 4LI5, 5C8M, 5CAO, 5CAP, 5CAV, 5EM8, 5GMP, 5GTY, 5UGB. The last six (3W2S, 3POZ, 3W32, 5UGB, 5CAV, 4LI5) were added 2026-08-10 and were chosen as a WT non-covalent series to hold the protein sequence constant while affinity varies.
- **Keep new runs at `--n_decoys 50`** unless recomputing all 19. A larger decoy count tightens the decoy-energy σ and shifts the F scale, so mixed-`n_decoys` structures are not comparable in one correlation.
- `results/egfr_frustration_summary.csv` currently holds all 19 rows and `egfr_correlation.png` matches it (Pearson r = −0.297, p = 0.217, n = 19 — not significant; see the pocket-size confound above before interpreting).
- The 1LYZ validation has run (`checkpoints/1LYZ_frustration.parquet`, `results/validation_lysozyme.png`).
- Plots present: `egfr_correlation.png`, `current_scatter.png`, `validation_lysozyme.png`.
- These artifacts live in DVC, not git — `dvc pull` to obtain them, `dvc push` after producing more. **The six new parquets and checkpoints have not been `dvc push`ed yet.**

## Project status tracking

Progress notes live in `project_status/` as an append-only history — add a new file per meaningful update rather than editing older ones. Name them `YYYY-MM-DD_HHMM_short-title.md` so they sort chronologically, and cover current state, what changed, blockers, next steps. Do not recreate a root-level `PROJECT_STATUS.md`.

## Stale documents

`README.md` is current except for the DVC bucket/target details noted above. These describe the superseded 25-structure pipeline and will mislead:

- `project_status/PROJECT_STATUS.md` — the archived baseline snapshot: 25 structures, covalent inhibitors *excluded*, old `/home/tugba/...` paths, a "FastRelax → MinMover" next-step that has already been done, and a Stage 0–4 numbering that doesn't match the current Stage 1–6 scheme. Kept for history; do not treat as current.
- `notebooks/egfr_frustration_pipeline.ipynb` — updated for the 61-structure pipeline, but its full survey remains expensive; run only after a single-structure prototype.
- `src/search_egfr_structures.py` — legacy RCSB/ChEMBL/BindingDB candidate search, superseded by `scripts/01`. Likewise `results/egfr_*candidates.csv`.
- `src/prepare_structures.py` — still live, but only as a helper library for `scripts/05` (`cif_to_mol2`, `download_ligand_cif`, `run_molfile_to_params`, …). Its own `process_structure()` / `main()` are the old 25-structure entry point.
