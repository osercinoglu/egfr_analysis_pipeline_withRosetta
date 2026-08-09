# EGFR Atomistic Frustration Pipeline — Project Status Report

**Date:** June 19, 2026
**Repo:** `/home/tugba/egfr_atomic_resolution`
**Environment:** `conda activate frustrato` (Python 3.10)

---

## 1. What We're Doing: Scientific Goal

### Reference Paper

**Chen et al. (2020), *Nature Communications* 11, 5944**
*"Surveying biomolecular frustration at atomic resolution"*
DOI: [10.1038/s41467-020-19560-9](https://doi.org/10.1038/s41467-020-19560-9)

The paper introduces the concept of **atomistic frustration**, which measures how "optimized" a protein's energy landscape is. How stable an amino acid pair (i, j) is in its native environment is computed by comparison against randomly placed amino acid pairs at the same location.

### The Paper's EGFR Analysis (Figure 5)

The paper makes the following observation for 4 EGFR kinase domain–inhibitor complexes:

> **Stronger-binding inhibitors → contacts around the ligand are less frustrated**
> Weaker-binding inhibitors → more frustrated ligand-protein interface

The paper did this analysis purely for **visualization** purposes (4 structures). They did not perform a correlation analysis, and the original code **was never published**.

---

## 2. Our Contribution: What We Built On Top

### Extension

| Dimension | Paper | This Project |
|---|---|---|
| Number of structures | 4 | **25** |
| Goal | Visualization | **Affinity–frustration correlation** |
| Code | Unpublished | **Independent reimplementation** |
| Affinity | Kd only | Kd + IC50 (mixed, log-transformed) |
| Covalent inhibitors | Included | **Excluded** (3IKA, 4LQM, 5XDK) |

### Core Mathematical Reimplementation

**Equation 1 — Frustration index (Z-score):**
```
F_ij = (E_ij_native − mean(E_ij_decoy)) / std(E_ij_decoy)
```
- `F > 0.78`  → **minimally frustrated** (native is highly stable)
- `F < −1.0`  → **highly frustrated**
- In between  → neutral

**Equation 2 — Many-body pairwise energy correction:**
```
E_ij = e_ij + 0.5 × Σ_{k∈contacts(i), k≠j} e_ik
             + 0.5 × Σ_{l∈contacts(j), l≠i} e_jl
```
- `e_ij`: direct pairwise energy (REF2015, excluding `fa_rep`)
- Background contacts each get half-weighted contributions from the neighbors of both residues

---

## 3. 25 EGFR–Inhibitor Complexes

### Affinity Distribution

| Property | Value |
|---|---|
| Total structures | 25 |
| Kd (thermodynamic) | 8 |
| IC50 (enzymatic) | 17 |
| Affinity range | 0.8 pM – 10 mM |
| log10(pM) range | −0.10 – 7.00 (**~7 orders of magnitude**) |
| Crystal resolution | 1.70 – 2.93 Å |
| Source | 4 papers + 21 BindingDB |

### Special Cases

- **7JXM (EAI045):** Allosteric inhibitor. Protein has chains A/B/C/D; ligand `9LL` is present only in B and D → `chain=B` override.
- **Excluded structures:** 4LQM/DJK, 5XDK/8JC, 3IKA/0UN → covalent inhibitors, not suitable for frustration calculation.

---

## 4. Code Written (~1754 lines)

### `src/frustration.py` (542 lines) — Core Engine

| Function | Description |
|-----------|-----------|
| `get_protein_contacts` | Cα–Cα ≤ 10 Å contact list |
| `get_ligand_contacts` | Ligand heavy-atom – Cα ≤ 10 Å |
| `pairwise_energy` | Direct e_ij (REF2015, excluding fa_rep) |
| `contact_energy_eq2` | Full Equation 2 |
| `native_aa_frequency` | Native AA frequency distribution |
| `generate_decoy` | AA shuffle → side-chain repack → 1 decoy |
| `run_frustration_survey` | N decoys → Z-score DataFrame |
| `summarize_ligand_frustration` | Ligand interface summary |

### `src/prepare_structures.py` (509 lines) — Structure Preparation

**Pipeline (per structure):**

```
Raw PDB  →  clean_pdb()  →  extract_ligand_pdb()
                                    ↓
           RCSB CIF  →  cif_to_mol2() [CCD atom names]
                                    ↓
                     molfile_to_params.py  →  .params
                                    ↓
                           Load pose in PyRosetta
```

**Fallback:** If CIF fails → SDF(RDKit) → last resort PDB(obabel)

### `src/run_pipeline.py` (436 lines) — Main Runner

- `--mode validate`: 1LYZ lysozyme validation
- `--mode single`: single-structure analysis
- `--mode all`: 25 EGFR structures → Pearson r + scatter plot

### `src/test_frustration.py` (267 lines) — 9 Unit Tests

---

## 5. Technical Obstacles Resolved

### 5.1 PyRosetta Installation
**Problem:** Wheel platform incompatibility.
**Solution:** tar.bz2 → manual extraction → `pip install .` from `setup/`.

### 5.2 Missing `rosetta_py`
**Problem:** `molfile_to_params.py` requires the `rosetta_py` module, which isn't included in the PyRosetta package.
**Solution:** Downloaded the pure-Python module from the RosettaCommons/rosetta GitHub repo → `src/rosetta_py/`.

### 5.3 OpenBabel mol2 Valence Error
**Problem:** Aromatic atoms get `ar` bond type → crashes `assign_rosetta_types`.
**Solution:** Used an SDF converted to Kekulé form with RDKit instead.

### 5.4 `fill_missing_atoms` Error (6 structures)
**Problem:** Params atom names (`C1`, `C2`) ≠ PDB HETATM names (`C16`, `C17`).
**Solution:** Generated mol2 from the RCSB CIF's `_chem_comp_atom.atom_id` → **25/25 poses load successfully**.

### 5.5 Multi-Chain Structures
**Problem:** The HETATM filter didn't account for chain, so multiple ligand copies were being included.
**Solution:** Added a `chain == target_chain` condition.

---

## 6. Test Results

```
test_build_contact_partner_map_basic      ✅ PASSED
test_frustration_index_formula            ✅ PASSED
test_frustration_class_thresholds         ✅ PASSED
test_summarize_ligand_frustration_basic   ✅ PASSED
test_get_protein_contacts_count           ✅ PASSED
test_pairwise_energy_symmetric            ✅ PASSED
test_native_aa_frequency_sums_to_one      ✅ PASSED
test_decoy_backbone_unchanged             ❌ FAILED  (0.78 Å > 0.05 Å tolerance)
test_frustration_survey_small             ✅ PASSED

8/9 passing
```

**Remaining failure:** In PyRosetta 2026.25, `FastRelax.set_movemap(mm)` + `mm.set_bb(False)` doesn't fully freeze the backbone. Fix: switch `FastRelax` → `MinMover` (chi-only).

---

## 7. Stage Status

| Stage | Status |
|-------|-------|
| Stage 0: PyRosetta installation | ✅ Done |
| Stage 1: 25 PDB preparation (params + pose) | ✅ Done — 25/25 |
| Stage 2: Frustration engine (Eq.1, Eq.2) | ✅ Done |
| Unit tests | ✅ 8/9 passing |
| Stage 3: Lysozyme validation | 🔲 Pending |
| Stage 4: EGFR analysis + correlation | 🔲 Pending |

---

## 8. Next Steps

```bash
# 1. FastRelax → MinMover fix (frustration.py, ~5 lines)

# 2. Validation
python src/run_pipeline.py --mode validate --n_decoys 50

# 3. Full analysis
python src/run_pipeline.py --mode all --n_decoys 200
```

---

*This project is an independent reimplementation applying Chen et al. (2020)'s atomistic frustration to EGFR–inhibitor complexes. The original authors' code was never published.*
