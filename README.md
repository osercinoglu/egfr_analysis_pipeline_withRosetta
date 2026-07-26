# EGFR Atomistic Frustration Pipeline

Independent reimplementation extending Figure 5 of Chen et al. (2020)
from 4 structures to 61 EGFR–inhibitor complexes (51 unique ligands,
including 15 covalent inhibitors) for a more robust frustration–affinity
correlation.

## Reference
Mingchen Chen et al., "Surveying biomolecular frustration at atomic resolution,"
*Nat. Commun.* 11, 5944 (2020). DOI: [10.1038/s41467-020-19560-9](https://doi.org/10.1038/s41467-020-19560-9)

## Installation

```bash
conda activate frustrato

pip install numpy scipy matplotlib pandas biopython requests pyyaml tqdm pyarrow pytest
pip install openbabel-wheel
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"

# Verify
python -c "import pyrosetta; pyrosetta.init(); print('PyRosetta OK')"
```

## Directory Structure

```
egfr_atomic_resolution/
├── config.yaml                          # All tunable parameters
├── config/
│   ├── pdb_reference_table.csv          # Curated list of 61 PDB IDs + paper reference values
│   └── ligand_overrides.yaml            # Per-structure chain/ligand-copy selection (Stage 3 output)
├── scripts/                             # Structure/ligand preparation pipeline (Stages 1-5)
│   ├── 01_collect_metadata.py           # RCSB Data API metadata + QC tables
│   ├── 02_download_structures.py        # Raw PDB/mmCIF download + integrity checks
│   ├── 03_identify_egfr_chain_and_ligand.py  # Motif-based chain/ligand-copy selection
│   ├── 04_prepare_complex.py            # Cleaned complex PDBs (data/processed/)
│   └── 05_prepare_ligand.py             # Ligand .params generation + validation
├── results/
│   ├── metadata/
│   │   ├── egfr_ligand_inventory.csv    # 61 structures: ligand, affinity, provenance
│   │   ├── chain_ligand_selection.csv   # Stage 3 selection rationale
│   │   ├── ligand_parameterization_status.csv  # Stage 5 per-ligand validation
│   │   └── qc_*.csv                     # Ambiguity/warning subsets
│   ├── preparation_summary.csv          # Stage 4 per-structure summary
│   ├── egfr_frustration_summary.csv     # Stage 6 analysis output
│   ├── egfr_correlation.png             # Fig. 5e-style scatter plot
│   └── validation_lysozyme.png          # Lysozyme validation figure
├── data/
│   ├── raw_pdb/, raw_cif/               # Raw files from RCSB
│   ├── processed/                       # Cleaned complexes (selected chain + target ligand)
│   └── ligands/
│       ├── *.cif / *.mol2               # CCD-derived ligand files
│       └── params/                      # Rosetta .params files (one per unique ligand)
├── checkpoints/       # Decoy intermediate results (for resume)
├── src/
│   ├── frustration.py          # Core engine: Eq. 1, Eq. 2, decoy generation
│   ├── prepare_structures.py   # Shared helpers (CIF/SDF → mol2, molfile_to_params) used by scripts/05
│   ├── run_pipeline.py         # Frustration analysis runner (validate/single/all)
│   ├── search_egfr_structures.py  # Legacy RCSB/ChEMBL candidate search, superseded by scripts/01
│   └── test_frustration.py    # Unit tests
└── notebooks/
    └── egfr_frustration_pipeline.ipynb
```

## Usage

### Stages 1–5 — Structure & Ligand Preparation

Already run for the current 61-structure set; outputs are checked into
`results/`, `data/processed/`, and `data/ligands/params/`. To regenerate or
extend the set:

```bash
python scripts/01_collect_metadata.py
python scripts/02_download_structures.py
python scripts/03_identify_egfr_chain_and_ligand.py
python scripts/04_prepare_complex.py
python scripts/05_prepare_ligand.py

# Single-structure test mode is available on each script, e.g.:
python scripts/01_collect_metadata.py --pdb-id 5GMP
```

### Stage 6 — Validation on Lysozyme

```bash
python src/run_pipeline.py --mode validate --n_decoys 50
# Output: results/validation_lysozyme.png
# Expected: buried core contacts → mostly minimally frustrated
```

### Stage 6 — Full EGFR Analysis

```bash
# Single structure:
python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50

# Quick prototype (all 61 structures):
python src/run_pipeline.py --mode all --n_decoys 50

# Full analysis:
python src/run_pipeline.py --mode all --n_decoys 200
```

The candidate list for `--mode all`/`single` is built from the Stage 1/4/5
outputs (`load_candidates()` in `src/run_pipeline.py`), so it always
reflects the current 61-structure set — nothing is hardcoded.

Interrupted runs resume automatically from checkpoints in `checkpoints/`.

Each decoy costs roughly 4–5 minutes on a mid-size structure (~1700
protein-protein contacts); budget accordingly for `n_decoys` × structure
count.

### Unit Tests

```bash
python -m pytest src/test_frustration.py -v
```

## Key Parameters (config.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contacts.protein_protein_cutoff_A` | 10.0 | Cα–Cα cutoff for protein contacts (Å) |
| `contacts.ligand_protein_cutoff_A` | 10.0 | Ligand heavy-atom–Cα cutoff (Å) |
| `frustration.n_decoys` | 50 | Number of decoys (full run: 200–1000) |
| `frustration.seed` | 42 | Random seed for reproducibility |
| `frustration.exclude_fa_rep` | true | Exclude fa_rep from energy (per paper) |
| `frustration.thresholds.minimally_frustrated` | 0.78 | F_ij > 0.78 |
| `frustration.thresholds.highly_frustrated` | -1.00 | F_ij < -1.0 |
| `chain_selection.default` | A | Fallback chain; actual per-structure chain/ligand selection is driven by `config/ligand_overrides.yaml` |

## Methods Summary

### Frustration Index (Eq. 1)
```
F_ij = (E_ij_native − mean(E_ij_decoy)) / std(E_ij_decoy)
```
Large positive F → minimally frustrated (native contact far more stable than random).

### Many-Body Energy Correction (Eq. 2)
```
E_ij = e_ij + 0.5 * Σ_{k ∈ contacts(i), k≠j} e_ik
             + 0.5 * Σ_{l ∈ contacts(j), l≠i} e_jl
```

### Decoy Generation
1. Compute global amino acid frequency distribution from the native sequence
2. Randomly substitute every position via `MutateResidue`
3. Repack side chains (`PackRotamersMover`, backbone fixed) + 1 pass of
   chi-only `MinMover` to resolve clashes
4. Explicitly restore all backbone atoms (N, CA, C, O) from the native pose —
   `MutateResidue` rebuilds the carbonyl O from idealized geometry, which can
   drift ~0.5–0.8 Å from the (non-ideal) crystallographic position

### Structure Set (61 structures / 51 unique ligands)
- Selected via `scripts/01-03` from RCSB (metadata validation, motif-based
  EGFR chain identification, ATP-pocket-based ligand-copy selection for
  multi-chain/multi-copy structures)
- **Covalent inhibitors are included** (15 structures, e.g. 3IKA, 5HG7-9,
  5J9Y/Z, 5UG8/9/C, 5YU9) — `scripts/04` records the covalent bond via
  mmCIF `_struct_conn`, and `scripts/05` adds a CONNECT record to the
  ligand's `.params`
- Affinity values (`affinity_pM` in `egfr_ligand_inventory.csv`) are a mix
  of Kd/Ki and IC50, log-transformed for correlation; the Kd/Ki vs. IC50
  distinction from the original 25-structure candidate set is not currently
  tracked through the new metadata pipeline

## Disclaimer

This is an independent reimplementation; the original authors' code has not
been published. Numerical results may differ from the paper; the goal is a
qualitatively correct tool (strong inhibitor → more minimal frustration,
weak inhibitor → less).
