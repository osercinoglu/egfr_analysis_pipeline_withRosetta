# EGFR Atomistic Frustration Pipeline

Independent reimplementation extending Figure 5 of Chen et al. (2020)
from 4 structures to 25 EGFR–inhibitor complexes for a more robust
frustration–affinity correlation.

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
├── config.yaml                      # All tunable parameters
├── results/
│   ├── egfr_final_candidates.csv    # 25 EGFR–inhibitor pairs + affinity data
│   ├── egfr_frustration_summary.csv # Analysis output
│   ├── egfr_correlation.png         # Fig. 5e-style scatter plot
│   └── validation_lysozyme.png      # Stage 3 validation figure
├── data/
│   ├── raw_pdb/       # Raw PDB files from RCSB
│   ├── processed/     # Cleaned complexes (chain A + target ligand)
│   └── ligands/
│       ├── *.pdb / *.mol2   # Extracted ligand files
│       └── params/          # Rosetta .params files
├── checkpoints/       # Decoy intermediate results (for resume)
├── src/
│   ├── frustration.py          # Core engine: Eq. 1, Eq. 2, decoy generation
│   ├── prepare_structures.py   # PDB cleaning + params generation
│   ├── run_pipeline.py         # Main runner script
│   ├── search_egfr_structures.py  # RCSB/ChEMBL candidate search
│   └── test_frustration.py    # Unit tests
└── notebooks/
    └── egfr_frustration_pipeline.ipynb
```

## Usage

### Stage 1 — Structure Preparation

```bash
# All 25 structures:
python src/prepare_structures.py --config config.yaml

# Single structure test:
python src/prepare_structures.py --config config.yaml --pdb_id 5GMP
```

### Stage 3 — Validation on Lysozyme

```bash
python src/run_pipeline.py --mode validate --n_decoys 50
# Output: results/validation_lysozyme.png
# Expected: buried core contacts → mostly minimally frustrated
```

### Stage 4 — Full EGFR Analysis

```bash
# Quick prototype:
python src/run_pipeline.py --mode all --n_decoys 50

# Full analysis:
python src/run_pipeline.py --mode all --n_decoys 200
```

Interrupted runs resume automatically from checkpoints in `checkpoints/`.

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
| `chain_selection.default` | A | Chain used for multi-chain structures |

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
1. Compute global amino acid frequency distribution from native sequence
2. Randomly substitute every position (backbone fixed)
3. Repack side-chains + 1-cycle backbone-fixed FastRelax

### Affinity Data (25 structures)
- **Kd (8 structures):** Thermodynamic (ITC/SPR) — preferred
- **IC50 (17 structures):** Enzyme assay — comparable after log-transform
- Covalent inhibitors excluded (4LQM/DJK, 5XDK/8JC, 3IKA/0UN)
- Primary correlation uses all 25; secondary sub-analysis uses Kd-only (8)

## Disclaimer

This is an independent reimplementation; the original authors' code has not
been published. Numerical results may differ from the paper; the goal is a
qualitatively correct tool (strong inhibitor → more minimal frustration,
weak inhibitor → less).
