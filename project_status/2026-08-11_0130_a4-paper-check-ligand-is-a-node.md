# A4 complete — the transcription is faithful, but the paper counts ligand contacts

**Date:** 2026-08-11 01:30
**Plan step:** A4 of `plans/frustratometer-ng-plan.md`
**Source:** Chen et al., *Nat. Commun.* **11**, 5944 (2020), DOI 10.1038/s41467-020-19560-9.
Read from the PMC open-access deposit, [PMC7683549](https://pmc.ncbi.nlm.nih.gov/articles/PMC7683549/).
Equation and prose quoted below are from the raw article HTML, not from a summary.

## Current state

Stage A is complete. A4 answers the blocking question and, unexpectedly, also explains the
count gap that A2 could not close.

## 1. The many-body transcription is FAITHFUL

Eq. 2 exactly as printed in the paper:

```
E_ij = e_ij + 1/2 Σ_{k, k≠j} e_ik + 1/2 Σ_{l, l≠i} e_jl        (2)
```

**The exclusions `k≠j` and `l≠i` are in the paper.** `src/frustration.py:226-250` is a
faithful transcription. `chen_literal` is the published object.

Therefore the degeneracy found in A1 — `E_ij = 0.5·(B_i + B_j)`, with `e_ij` cancelling
exactly — is **a property of the published equation as printed**, not a defect introduced
here. That is a finding about the method, not about this repository.

Worth recording, because it may matter for a write-up: the surrounding prose states the
intent as *"E_ij is defined by considering all the interaction energies that involve
changing any of the two residues that are in contact"*. That intent corresponds to
`B_i + B_j − e_ij` (coefficient 1 on the background sums). The printed formula, with
coefficient 1/2 **and** the exclusions, is not that quantity — it collapses to the mean of
two per-residue totals. We cannot know what their unpublished code computed. What we can
say is that the equation as published does not compute what its own caption describes.

**Plan question Q1 is answered.** `chen_literal` is the published object and is required
for any reproduction claim; `pair_retained` is a deliberate *improvement*, not a bug fix.

## 2. The real gap: the ligand is a node in the paper's contact graph

From the Results:

> *"a strong inhibitor XTF-262 (PDB ID: 5GMP) forms more than ten minimally frustrating
> interactions with its pocket … while a weaker binder 5Q4 forms only three minimally
> frustrated interactions"*

and the Fig. 5 caption: *"frustrations around the ligands only are shown on the right
panel"*.

The published counts are **ligand–residue contacts** — interactions the ligand itself
forms. Cross-checked against `config/pdb_reference_table.csv`:

| PDB | paper count | paper text | affinity |
|---|---:|---|---:|
| 5GMP | 16 | "more than ten" ✓ | 0.8 pM |
| 5EM8 | 4 | "only three" ✓ | 1090 pM |

The full published range 4–23 (mean 12.7) is exactly the number of residues a drug-sized
ligand contacts.

**Our pipeline counts protein–protein contact pairs with at least one partner in a 10 Å
pocket shell: 266–407.** These are different objects, which is why A2's 189-configuration
sweep could not close the gap by re-selecting or re-thresholding — the contacts the paper
counts do not exist in our data at all. `get_protein_contacts` (`frustration.py:70-108`)
excludes non-protein residues, so the ligand is never in a pair.

This also explains A3: the native reference is bit-identical holo vs apo precisely because
the ligand contributes no edges. In the published method it contributes 4–23 of them.

The decoy construction confirms the reading: *"The efficient approximation … mimics the
scenario of ligands searching for their best binding pockets by instead randomly shuffling
the protein sequences to create ensembles of nonspecific binding sites."* The ligand is
held fixed while the pocket around it is randomised — a ligand-anchored calculation.

## 3. Four further implementation deviations, all quoted

| # | Paper | This implementation | Plan step |
|---|---|---|---|
| D1 | *"we randomly shuffle the protein sequence"* — a permutation, preserving the native composition exactly | `np.random.choice(aa_letters, p=aa_probs)` (`frustration.py:337`) — i.i.d. draw with replacement, composition fluctuates | C2 `identity=native, placement=permute` |
| D2 | *"A short Monte-Carlo relaxation is then performed"* | single chi-only MinMover pass (`frustration.py:364`) | C8 — required, not optional |
| D3 | *"The contact energies of native sequence are obtained in a similar fashion by omitting the shuffling step"* → the native **is** repacked and MC-relaxed | native scored as deposited, no repack (`frustration.py:612-621`) | C6 — **`native.repack` default must be `true`** |
| D4 | *"Protein contacts are defined by the CαCα distances … a cutoff of 10 Å"*; the phrase "sequence separation" appears **nowhere** in the paper | `\|i−j\| ≥ 4` applied (`frustration.py:98`) | B7 — make it explicit and default it off for reproduction |

Matching correctly: the 10 Å Cα–Cα cutoff, REF2015, backbone held fixed during repacking,
and removal of the harsh repulsive term (*"simply remove the harsh rapidly varying
repulsive force term"*) — our `exclude_fa_rep` does this.

Decoy count: the paper uses **1000 per contact**; our runs use 50.

**Plan question Q4 is answered:** the native is repacked in the published protocol, so
symmetric native treatment is the default, not an option.

## Stage A synthesis, final

- **A1** — `E_ij = 0.5·(B_i + B_j)` exactly, 38/38 structures; ~4% of the index variance is pair-specific.
- **A2** — no selector or threshold over protein–protein pairs reproduces the published counts.
- **A3** — native reference provably ligand-independent; index ligand-blind at r = 0.9904.
- **A4** — the published counts are ligand–residue contacts; the transcription of Eq. 2 is
  faithful; four further protocol deviations identified.

**These are one coherent story.** The reimplementation computes a protein-only,
per-residue-additive quantity. The paper computes ligand-anchored contact frustration.
A2 and A3 are the observable consequences of the single structural difference that A4
identifies: the ligand is not a node.

## Blockers

None. Stage A is done and the diagnosis is complete.

## Next steps

1. **B1** — package skeleton. The critical path to a meaningful result now runs
   B1 → B2 → B3 → B6 → B7 → B8 → B5 → C1 → C2 → D1 → D2 → D3 → E2 → E3 → F3, and F3 becomes
   a real test only once the ligand is a node.
2. **Re-scope the reproduction target.** F3 must count ligand–residue contacts, not
   protein–protein pairs in a shell. Update its acceptance criterion.
3. The 61 completed structures are not a reproduction of the paper under any post-hoc
   treatment. They remain useful as a bit-for-bit regression target for the new engine.
4. Consider whether `pair_retained` or `chen_literal` is the scientific object going
   forward — now a deliberate choice rather than a bug fix, and one worth stating explicitly
   in any publication.
