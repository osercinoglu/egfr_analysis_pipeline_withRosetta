# EGFR Atomistic Frustration Pipeline — Copilot Instructions

## Environment

Bootstrap the workspace with `conda env create -f environment.yml` and activate the
`frustrato` environment before running project commands.

The open-source preparation path works without PyRosetta once that environment exists:

- `scripts/01_collect_metadata.py`
- `scripts/02_download_structures.py`
- `scripts/03_identify_egfr_chain_and_ligand.py`
- `scripts/04_prepare_complex.py`
- `scripts/05_prepare_ligand.py --skip-pyrosetta-test`

Stage 6 (`src/run_pipeline.py`) and full Stage 5 ligand validation still require
PyRosetta after the conda environment is created.

## Commands

Always run from the repo root — `config.yaml` paths are relative to CWD.

```bash
# Data preparation (Stages 1–5, BioPython only, no PyRosetta needed)
python scripts/01_collect_metadata.py
python scripts/02_download_structures.py
python scripts/03_identify_egfr_chain_and_ligand.py
python scripts/04_prepare_complex.py
python scripts/05_prepare_ligand.py

# Single-structure debug mode for scripts 01–04
python scripts/04_prepare_complex.py --pdb-id 5GMP
# Single-ligand debug mode for script 05
python scripts/05_prepare_ligand.py --ligand-id 634 --skip-pyrosetta-test

# Stage 6 — frustration analysis (requires PyRosetta conda env `frustrato`)
python src/run_pipeline.py --mode validate --n_decoys 50      # lysozyme sanity check
python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50
python src/run_pipeline.py --mode all --n_decoys 200          # full run (multi-day)
python src/run_pipeline.py --mode all --pdb-ids 1XKK,5GMP --n_decoys 50 --n-jobs 2
python src/run_pipeline.py --mode all --n_decoys 200 --results-dir runs/method-a/results --checkpoints-dir runs/method-a/checkpoints

# Tests (run from repo root)
python -m pytest src/test_frustration.py -v
python -m pytest src/test_frustration.py::test_decoy_backbone_unchanged -v
```

`run_pipeline.py` must be launched as a script, **not** with `python -m src.run_pipeline`, because `frustration` is imported as a top-level module.

PyRosetta tests are auto-skipped when PyRosetta is not installed. Tests not marked with `@skip_if_no_pyrosetta` can always run.

## Architecture

Two halves that communicate only through files on disk.

**Half 1 — `scripts/01–05`: data preparation**
Each stage reads the previous stage's CSV/YAML output. No PyRosetta except Stage 5's optional validation step.

| Stage | Output |
|-------|--------|
| 01 | `results/metadata/egfr_ligand_inventory.csv` — affinity, QC |
| 02 | `data/raw_pdb/`, `data/raw_cif/`, `download_manifest.csv` |
| 03 | `config/ligand_overrides.yaml`, `chain_ligand_selection.csv` |
| 04 | `data/processed/{PDB}_clean.pdb`, `results/preparation_summary.csv` |
| 05 | `data/ligands/params/{LIG}.params`, `results/metadata/ligand_parameterization_status.csv` |

**Half 2 — `src/frustration.py` + `src/run_pipeline.py`: Stage 6 analysis**
- `frustration.py` — core engine: contacts → Eq. 2 many-body energies → N decoys → Eq. 1 Z-score → classification
- `run_pipeline.py` — driver: pose loading, checkpointing, correlation plot

**The join point** is `load_candidates()` in `run_pipeline.py`. It inner-joins the three Stage 1/4/5 CSVs and keeps only rows where both the complex and its ligand params have `status == "OK"`. To add or drop structures, edit `config/pdb_reference_table.csv` and re-run the stages — never hardcode lists in analysis code.

## Key Conventions

**`ligand_comp_id` vs `rosetta_ligand_comp_id`**  
Some CCD codes collide with Rosetta's internal namespace (e.g. `634` → `Z34`). Stage 4 rewrites only the HETATM `resName` in `_clean.pdb`. The `.params` filename and pose residue lookup use the *rosetta* name; checkpoint/result filenames and all reporting use the *real* CCD id. Mixing them silently produces "ligand not found in pose".

**`config/ligand_overrides.yaml` is generated, not hand-written**  
It is Stage 3's output and Stage 4's sole source of chain/ligand-copy truth (61 entries: `egfr_chain`, `ligand_chain`, `ligand_comp_id`, `ligand_resnum`, optional `rosetta_ligand_code`). The `chain_selection.default: A` in `config.yaml` is a vestigial fallback.

**Frustration thresholds are hardcoded in two places**  
`src/frustration.py:501` (`> 0.78`, `< -1.0`) and `run_pipeline.py`'s plot. The `frustration.thresholds` block in `config.yaml` is not read by any code — change both source files, not just the config.

**Decoy backbone must be bit-identical to native**  
`MutateResidue` drifts the carbonyl O ~0.5–0.8 Å from the crystal position. `generate_decoy()` explicitly restores N/CA/C/O from the native pose after repack+MinMover. `test_decoy_backbone_unchanged` guards this at 0.05 Å tolerance. Backbone restoration is not fixable via movemap settings alone.

**Sign convention**  
Rosetta energies are negative-is-favorable. The implemented index is `(mean(E_decoy) - E_native) / std(E_decoy)`, so a native contact that is more favorable than its decoys has positive F and is classified as minimally frustrated. A 50-decoy 1LYZ validation produced 43% minimally and 10% highly frustrated contacts, with the core fraction (48%) above the surface fraction (38%).

**`src/molfile_to_params.py` requires `src/rosetta_py/`**  
`run_molfile_to_params()` sets `PYTHONPATH` to `src/` and runs with `cwd=params_dir`. Do not relocate either.

**Ligand atom names must match PDB HETATM names exactly**  
Stage 5 uses CCD CIF as the first source so that param atom names match the crystal file names (`C16`, `C17`…) rather than generic sequential names (`C1`, `C2`…). Stage 5 validates names against every processed PDB using that ligand — preserve this check when touching ligand prep.

**Resume semantics**  
`run_frustration_survey` checkpoints decoy energies to `checkpoints/` every N decoys. `run_single_structure` short-circuits if `results/{PDB}_{LIG}_frustration.parquet` already exists. To rerun with more decoys, delete both the parquet and the checkpoint.

**Covalent inhibitors are not fully handled**  
15 structures are flagged `is_covalent` in `preparation_summary.csv`, but no CONNECT records are written to `.params` and Stage 6 has no covalent handling. These are currently scored as non-covalent.

**Performance**  
~4–5 min per decoy on a ~1700-contact structure. Always prototype at `--n_decoys 50` on a single structure before running `--mode all`.
`--mode all` uses spawned workers and defaults to logical CPU count minus two;
override it with `--n-jobs N` when a lower concurrency limit is required.
Use `--results-dir` and `--checkpoints-dir` together to isolate result and resume
files for comparative runs.

## Project Status Tracking

Store project progress under `project_status/`.

- `project_status/PROJECT_STATUS.md` is the historical baseline snapshot moved from the repo root.
- Do not keep a root-level `PROJECT_STATUS.md`.
- For each meaningful progress update, add a new markdown file in `project_status/` instead of overwriting older status files.
- Use filenames like `YYYY-MM-DD_HHMM_short-title.md` so status files remain chronological.
- Each new status entry should summarize the current state, recent changes, blockers, and next steps.
