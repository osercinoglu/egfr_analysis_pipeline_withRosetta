# atomfrust — implementation plan

A target-agnostic atomistic local-frustration tool for protein–ligand, protein-only and
protein–protein complexes.

**What this document is.** `methods_section_1002_v2.md` is read here as a *requirements source
only*. Every statement in it that implies something the code must do has been extracted into §1
as a technical delta — added, corrected, or modified — and each delta is assigned to a numbered
implementation step in §6. The steps are ordered by dependency.

**What this document deliberately omits.** No months, no work packages, no tiers, no thesis
assignments, no compute budgets, no research schedule. Those belong to the grant, not to the
codebase. Where a methods-doc success criterion (`S1.3`, `D1`, …) states a *testable property of
the code*, it survives here as an acceptance criterion with its ID kept for traceability. Where it
states a research milestone, it was dropped.

**Target.** A new package `atomfrust/` developed in this repository alongside the existing
pipeline. `src/frustration.py`, `src/run_pipeline.py` and `scripts/01`–`05` keep working
untouched until the validation cases in Stage F pass; only then does anything get retired. The
19 completed parquets stay readable throughout — nothing in this plan rewrites them in place.

`frustratometer` is taken on PyPI by the Wolynes-lab AWSEM/DCA tool. Distribution and import name
is **`atomfrust`**; console script `atomfrust`.

---

## 1. Requirements extracted from the methods document

Status against the current code: **OK** present and correct · **PARTIAL** present but incomplete ·
**WRONG** present and contradicts the requirement · **MISSING** absent.

### 1.1 Energy and contact model

| # | Requirement | Doc source | Status | Step |
|---|---|---|---|---|
| R01 | Contacts by Cα–Cα distance at 10 Å | §3.3 | OK (`frustration.py:70-108`) | B7 |
| R02 | Sequence separation excludes local backbone | §3.3, code | WRONG — `\|i−j\|` is applied across chains too (`frustration.py:98`), meaningless for multi-chain | B7 |
| R03 | Separate the harsh repulsive LJ term so clashes do not inflate decoy variance | §3.3 | PARTIAL — subtracted at scoring time (`frustration.py:210-212`), so the choice is baked into stored numbers and cannot be revisited | B8, C1 |
| R04 | Many-body energy: sum all interaction energies involving either partner | §3.4 | WRONG — the transcribed formula degenerates; see §2.1 | B8 |
| R05 | Ligand enters the energy function as a full-atom Rosetta residue type | §3.4 | PARTIAL — ligand is in the pose but is never a node in the contact graph (`frustration.py:111-140` returns single residues used only as a filter) | B6, B7 |
| R06 | Per-contact energies agree with a direct PyRosetta reference to within 1% | S1.2 | MISSING | F5 |
| R07 | Raw REF2015 interaction energy computed alongside every Z-score | §3.7, S5.2 | MISSING | D8 |

### 1.2 Decoy construction

| # | Requirement | Doc source | Status | Step |
|---|---|---|---|---|
| R08 | Randomise residue **identities** of contacting residues | #7, §3.3 | PARTIAL — randomises identities of *all* protein residues, not the contacting set (`frustration.py:333-343`) | C2 |
| R09 | Randomise **locations** of contacting residues | #7, §3.3 | MISSING — no positional axis exists at all | C2 |
| R10 | Side-chain repacking without backbone perturbation | §3.3 | OK (`frustration.py:348-378`, hard restore) | C5 |
| R11 | Short Monte-Carlo relaxation | §3.3 | MISSING — a single chi-only MinMover pass, no MC (`frustration.py:364`) | C8 |
| R12 | Repacking restricted to a shell around the pocket | §3.4, S1.3 | MISSING — the task factory is `RestrictToRepacking()` over the whole pose (`frustration.py:41`, `:347`); no per-residue operation is ever pushed | C7 |
| R13 | One randomisation yields energies for all contacts simultaneously | §3.4, §3.10 | OK — but it contradicts #6's "1000 decoys **per contact**"; treat as a hypothesis, not a settled optimisation | C2 |
| R14 | Decoy count configurable up to 1000 per contact | #6 | PARTIAL — configurable, but re-running with larger N is blocked by the parquet short-circuit (`run_pipeline.py:241`) | B5, D4 |
| R15 | Four decoy axes, each independently invocable | §3.5, S2.1 | MISSING — one hardcoded protocol | C2, G5, G6 |
| R16 | Decoy generators: DUD-E protocol, DeepCoy sets, unmatched random ZINC control | §3.5 | MISSING | G3 |

### 1.3 Ligand handling

| # | Requirement | Doc source | Status | Step |
|---|---|---|---|---|
| R17 | Automated ligand parametrisation at library scale | §3.4, S1.1 | PARTIAL — `scripts/05` handles one crystal ligand at a time; no cache, no failure taxonomy | G1 |
| R18 | OpenFF SMIRNOFF typing, GAFF2 fallback, AM1-BCC charges | §3.4 | MISSING | G1 (deferred, see §5) |
| R19 | Protonation and tautomer enumeration at pH 7.4 | §3.4 | MISSING | G2 |
| R20 | Protonation/tautomer sensitivity carried as an uncertainty band | S1.4 | MISSING | G2 |
| R21 | Cofactors, metals, multiple ligand copies, non-canonical residues | §3.1 (rationale), §3.4 | MISSING — `find_ligand_resnum` (`run_pipeline.py:186`) matches the first residue by `name3()` prefix, so N copies collapse to one | B6 |
| R22 | Covalent ligands bonded to the receptor | implied by the EGFR set | MISSING — `_struct_conn` linkage is recorded (`scripts/04:100-127`) but no CONNECT record is written and Stage 6 ignores it | G7 |
| R23 | Every generated pose passes PoseBusters before analysis | §3.7, S4.2 | MISSING | G4 |

### 1.4 Analysis, statistics, reporting

| # | Requirement | Doc source | Status | Step |
|---|---|---|---|---|
| R24 | Ligand-contacting residues definable in more than one way; adjustable shell | §3.4 (implied), user request 5 | MISSING — one hardcoded 10 Å heavy-atom→Cα rule (`frustration.py:111`) | B7, D3 |
| R25 | Classification thresholds configurable | code | WRONG — hardcoded in three places, and `config.yaml:25-27` is dead | D2 |
| R26 | Non-parametric rank/percentile index where normality fails | D3, S2.4 | MISSING | D1 |
| R27 | Per-pocket normality diagnostics (Shapiro–Wilk, Q–Q) | D3, S2.4 | MISSING | D1 |
| R28 | Five aggregation descriptors, none privileged | §3.5 | PARTIAL — three counts and `frac_minimally` (`frustration.py:727-762`); the plot hardcodes the confounded one (`run_pipeline.py:415-416`) | D3 |
| R29 | Convergence sweep N ∈ {10…2000}, ρ ≥ 0.95 vs N = 2000 | D1, S2.2 | MISSING | D4 |
| R30 | σ across architectural strata (burial, polarity, volume) | D4, S2.5 | MISSING | D5 |
| R31 | Pairwise Z-score correlation across axes | D5, S2.6 | MISSING | D5 |
| R32 | AUROC, BEDROC α = 80.5, adjusted logAUC, EF1%, Pearson, Spearman | §3.9 | MISSING | D6 |
| R33 | Bootstrap 95% CIs, 10 000 resamples, stratified by target | §3.9 | MISSING | D6 |
| R34 | Paired tests across targets, never pooled molecule-level | §3.9 | MISSING | D6 |
| R35 | Multiplicity correction across the descriptor × axis family | §3.9 | MISSING — and BH is the wrong instrument once every parameter is swept; see §2.3 | D6 |
| R36 | One implementation of every metric, shared by all consumers | §3.3, S0.3 | MISSING | D6 |

### 1.5 Infrastructure

| # | Requirement | Doc source | Status | Step |
|---|---|---|---|---|
| R37 | Structured logging enabling regeneration of any reported number from a recorded configuration | §3.3 | MISSING — nothing records what settings produced a parquet | B4, B5 |
| R38 | Identical metric values for a fixed input in two environments | S0.3 | MISSING | E7 |
| R39 | Wall-clock cost reported as a function of pocket size and decoy count | S1.5 | MISSING | C7, E6 |
| R40 | Open-source release with documentation, unit tests, reproducible examples | S3.6 | PARTIAL | E-stage, F-stage |

### 1.6 User-requested features (not from the methods document)

| # | Request | Step |
|---|---|---|
| U1 | Heteroligand as a residue/node in the analysis | B6, B7 |
| U2 | Decoy generation methods for heteroligands | G5, G6 |
| U3 | Decoys drawn from external datasets (DUD-E, DeepCoy, DEKOIS, MUV) | G3 |
| U4 | Decoy generation focused on selected regions | B9, C7 |
| U5 | Ligand-contacting residues defined various ways; adjustable shell/cutoffs | B7, D3 |
| U6 | Generate decoys only, compute no frustration | E2 |
| U7 | Analyse an existing output folder; decoupled generation and analysis | B5, E3 |
| U8 | Accept a custom PDB file | B3, E1 |

---

## 2. Three findings that change the design

These came out of reading the code against the document. The first two are verified numerically in
this repository; both must be settled before any new science is computed, because they determine
what the tool is measuring.

> **STATUS 2026-08-11 — Stage A is complete and has resolved all three.** Read this box before
> the subsections below, which were written before the diagnostics ran.
>
> - **§2.1 is confirmed but re-attributed.** `E_ij = 0.5·(B_i + B_j)` holds exactly in 38/38
>   structures (A1), and A4 established that **the exclusions are in the published Eq. 2** — so
>   the degeneracy belongs to the published equation, not to this transcription. `chen_literal`
>   is the published object. `pair_retained` is a deliberate improvement, not a bug fix.
> - **§2.2's cause is now known, and it is not the selector.** A2 swept 189 configurations over
>   50 structures and closed nothing. A4 found why: **the paper's counts are ligand–residue
>   contacts** ("a strong inhibitor … forms more than ten minimally frustrating interactions
>   with its pocket"; 5GMP = 16 ✓, 5EM8 = 4 ✓). We count protein–protein pairs in a shell. The
>   contacts the paper counts do not exist in our data, because `get_protein_contacts`
>   (`frustration.py:70-108`) excludes non-protein residues.
> - **A3 is the same fact seen from another angle:** `E_native` is bit-identical holo vs apo
>   (max |Δ| = 0), and the index is ligand-blind at pocket-contact r = 0.9904.
>
> **Consequence for this plan: ligand-as-node (B6/B7) is not a feature, it is the correction.**
> Fixing the many-body formula alone would not help — partner lists would remain protein-only.
> A4 also identified four protocol deviations (decoy shuffle vs i.i.d. draw, Monte-Carlo vs
> MinMover, native repacking, sequence separation); see step A4 and its status note.

### 2.1 The many-body formula degenerates — the per-contact index is a per-residue index

Eq. 2 as transcribed at `frustration.py:226-229` and implemented at `:238-250`:

```
E_ij = e_ij + 0.5·Σ_{k∈contacts(i), k≠j} e_ik + 0.5·Σ_{l∈contacts(j), l≠i} e_jl
```

Let `B_i = Σ_{k∈contacts(i)} e_ik` over **all** partners. Then `Σ_{k≠j} e_ik = B_i − e_ij`, and

```
E_ij = e_ij + 0.5·(B_i − e_ij) + 0.5·(B_j − e_ij) = 0.5·(B_i + B_j)
```

`e_ij` cancels exactly. The quantity has no pair-specific content: it is the mean of two
per-residue totals. Verified on `results/1XKK_FMM_frustration.parquet` — fitting `E_native` as
`a_i + a_j` over 1708 contacts gives **R² = 1.000000**, and `decoy_mean` likewise; `F_index` gives
0.961876, the residual arising only from the per-pair σ division.

The caller passes the full partner list including the excluded partner (`frustration.py:610-621`,
`:672-681`) and `contact_energy_eq2` re-excludes it at `:242`/`:247`. That is consistent with its
own docstring — the cancellation is a property of the formula as written, not a coding slip.

Consequences, all of which the plan is built around:

- The pocket-size confound documented in `CLAUDE.md` is **derived, not observed**. If every
  contact's value is fixed by its two residues, contact counts are graph-degree statistics, and
  `n_contacts_total` out-predicting `n_minimally_frustrated` follows necessarily.
- Adding a ligand node under this formula changes little: the ligand would perturb a
  protein–protein key through one term of ~12 in `B_i`.
- Three formulas therefore become selectable (step B8), and the default is not the current one.

### 2.2 The published per-structure counts are on disk and disagree by ~25×

`config/pdb_reference_table.csv` carries Chen et al.'s own per-structure minimally-frustrated
contact counts. Nothing in the pipeline has ever compared against them.

| | paper | this pipeline |
|---|---|---|
| minimally frustrated contacts | 4–23, mean 12.7 | 266–407, mean 316.3 |

Across the 19 completed structures the two are **uncorrelated: Pearson r = 0.163, p = 0.51**.
Meanwhile the paper's own counts against log₂(affinity pM) give **r = −0.388, p = 0.002 (n = 61)**
and **r = −0.646, p = 0.003** on those same 19, against this pipeline's **r = −0.297, p = 0.22**.
(Negative because more minimally frustrated → lower pM; the magnitude is what compares to the
published 0.45.)

So the signal is real in this exact structure set, and we are computing a different object. This
makes reference-count reproduction (step F3) the cheapest and most decisive validation available —
it needs no new structures, no affinity data, and no decoys beyond what is already on disk.

The scale gap of ~25× is itself diagnostic: it is roughly the ratio between "all contacts in a
10 Å-Cα pocket shell" and "contacts a ligand actually touches". That points at the selector and the
contact definition, not at the decoy protocol, and step A2 tests exactly that before any code is
written.

### 2.3 Multiplicity: BH is the wrong correction for a swept-parameter tool

This plan makes contact definition, cutoff, shell, many-body mode, index function, thresholds,
descriptor and axis all selectable. That is the point of the tool, but it means a reported
best-case correlation is a maximum over a grid exceeding 10⁴ configurations. Benjamini–Hochberg
controls FDR over a *fixed* family of pre-specified tests; it does not price a maximum. Step D6
implements a max-T permutation null over the whole grid, and step E6's `report` refuses to print an
unadjusted headline. For scale: at n = 19, r = 0.456 is already the 5% critical value for a
*single* pre-specified test.

---

## 3. Architecture

### 3.1 Package layout

```
atomfrust/          # flat layout, not src/ — see B1
  spec.py          SystemSpec / SystemSet models + loaders (YAML/CSV/JSON)
  settings.py      Settings (pydantic v2, extra="forbid"), layering, regeneration_key
  provenance.py    digests, env capture, manifest read/write
  runstore.py      RunDir: paths, sharded parquet writers/readers, STATUS
  execute.py       flat work queue, sharding, spawn worker init
  pose.py          load_complex(), components manifest, PDBInfo-keyed node lookup
  graph.py         node table, contact definitions, superset graph
  energy.py        pairwise_energy, many-body registry, energy-graph snapshot cache
  regions.py       named residue selectors -> ResolvedRegions
  decoys/          base.py  identity.py  pose.py  chemotype.py  external.py
  chem/            paramize.py  cache.py  codes.py  protonation.py  libraries/
  dock/            base.py  smina.py  gnina.py  mcs_align.py  posebusters.py
  analyze/         zscore.py  classify.py  aggregate.py  strata.py  converge.py
  metrics/         screening.py  inference.py  golden/metrics_golden.json
  report/          plots.py  collect.py
  cli/             main.py + one module per subcommand
  recipes/         egfr.py  cox.py        # emit SystemSpecs; plugins, not dependencies
```

**Lifted verbatim** from the prototype: `pairwise_energy` (`frustration.py:182-213`), the backbone
hard-restore block (`:366-378`) and its 0.05 Å test, `src/molfile_to_params.py` and the vendored
`src/rosetta_py/` with its `PYTHONPATH=src` / `cwd=params_dir` invocation.
**Rewritten:** everything in `run_pipeline.py`, decoy generation, contact construction, checkpointing.
**Dropped:** `load_candidates` (`run_pipeline.py:140`), `scripts/03` motif matching (→
`recipes/egfr.py`), `src/search_egfr_structures.py`, `prepare_structures.main`, the cumulative
summary CSV.

### 3.2 System specification — the target-agnostic input

One hand-writable YAML replaces the three-CSV join at `run_pipeline.py:140-164`. Affinity is a
free-form label, never a required column.

```yaml
system_id: 5GMP_634
receptor:
  path: inputs/5GMP.pdb        # or pdb_id: 5GMP  (fetched + cached)
  chains: [A]                  # omit => all protein chains
ligands:                       # [] => protein-only / protein-protein
  - selector: {chain: A, resseq: 1001, icode: "", comp_id: "634"}
    params: params/Z34.params  # optional; auto-parametrised if absent
    rosetta_name: Z34
    covalent_anchor: {chain: A, resseq: 797, atom: SG, ligand_atom: C24}   # optional
pocket:
  mode: ligand_shell           # ligand_shell | residue_list | chain_interface | whole
  reference: any_heavy         # ca | cb | any_heavy | sidechain_heavy
  cutoff_A: 10.0
labels: {affinity_pM: 23.0, set: egfr, is_covalent: false}
```

`chain_interface` gives protein–protein; `ligands: []` + `pocket.mode: whole` gives protein-only.
The `comp_id` ↔ `rosetta_name` ↔ params triple lives in one place, closing the `634`/`Z34` trap.
Node lookup goes through `PDBInfo (chain, resseq, icode)`, never `name3()` prefix matching
(`run_pipeline.py:186`), so N copies of one ligand are N distinct nodes (R21).

### 3.3 Graph model

Nodes are typed (`protein`, `ligand`, `cofactor`, `metal`, `nucleic`, `noncanonical`). Edge criteria
are per unordered kind-pair:

| Kind pair | Default criterion | Note |
|---|---|---|
| protein–protein | `CaCa(10.0 Å, seq_sep_min=4, scope=same_chain)` | **seq-sep now applies per chain** (fixes R02) |
| protein–ligand / protein–metal / ligand–ligand | `HeavyMin(6.0 Å)` | ligands have no Cα |
| any pair in `bonds` | forced edge, `is_bonded=true` | exempt from distance and seq-sep |

Neighbour search is `scipy.cKDTree` over heavy atoms; the O(N²) loop at `frustration.py:90-108`
does not survive multi-chain systems.

**Superset graph.** Energies are computed once over a permissive union — `(Cα ≤ 12 Å) ∪
(heavy-min ≤ 6 Å)`, `seq_sep ≥ 1` — and every named contact definition becomes a boolean column on
that graph. Narrowing a cutoff is then a filter over stored energies, which is what makes R24/U5/U7
free. Contact **membership is evaluated on the native pose only** and frozen across decoys —
required for the Z-score to be over a fixed index set, and stated as an invariant rather than left
implicit.

### 3.4 Many-body registry

`energy.manybody ∈ {pair_retained, chen_literal, pair_only}`, selectable per kind-pair.

> **Default changed by B2 after A4: `chen_literal`, not `pair_retained`.** A4 confirmed the
> exclusions are in the published Eq. 2, so `chen_literal` is the published object and is
> required for the reproduction target (F3). It is also the degenerate one. Leading with
> the sound formula instead is a one-line flip; this is §10's remaining open question and
> should be decided deliberately.

- `pair_retained`: `E_ij = e_ij + 0.5·B_i + 0.5·B_j`, with `B_i` summed over **all** of
  i's partners. **Corrected in B8:** the formula previously written here excluded `i`
  and `j` from the sums, which is the same exclusion `chen_literal` makes and is
  therefore equally degenerate. Taking the sums over all partners is what retains the
  pair term; note `pair_retained − chen_literal == e_ij` exactly.
  where the sums are taken over the *stored* partner lists with `i` and `j` removed **before**
  summation, so `e_ij` is not double-subtracted. Does not degenerate.
- `chen_literal`: reproduces `frustration.py:238-250` bit-for-bit, i.e. `0.5(B_i + B_j)`. Kept so
  the reproduction run and the degeneracy report are both expressible.
- `pair_only`: `e_ij` alone; the ablation that isolates how much the many-body term contributes.

---

## 4. Run-directory contract and on-disk schemas (normative)

This section is the interface. Steps B5 onward consume it; it is what makes U6, U7, R14 and the
convergence sweep possible.

```
runs/<run_id>/
  manifest.json                     # run-level; see below
  settings.resolved.yaml            # fully resolved Settings after layering
  env.json                          # python, pyrosetta version, OS, CPU, git sha+dirty, pkg versions
  logs/generate.jsonl
  systems/<system_id>/
    STATUS.json                     # {state, n_decoys_done, n_decoys_target, updated_at}
    inputs/
      system.spec.yaml              # exact spec used
      receptor.pdb                  # exact bytes scored
      components.yaml               # comp_id <-> rosetta_name <-> params path <-> sha256
      params/<CODE>.params
    graph/
      nodes.parquet
      pairs.parquet
    native/
      native.pdb
      native_energies.parquet
      raw_energy.json               # R07 control
      pocket.json                   # R30 descriptors
    decoys/
      index.parquet
      energies/part-<shard>-<seq>.parquet
      structures/decoy_%06d.pdb.gz  # only with --save-structures
      members/<member_id>/          # axis D only, only with --save-members
    analyses/<analysis_id>/
      manifest.json
      contacts.parquet
      summary.json
      figures/
```

**`nodes.parquet`**

| column | type | note |
|---|---|---|
| `node_id` | str | `A:745`, `L:634:1`, `M:ZN:1` |
| `pose_resnum` | int32 | |
| `kind` | dict(str) | protein\|ligand\|cofactor\|metal\|nucleic\|noncanonical |
| `chain`,`resseq`,`icode` | str,int32,str | PDBInfo key |
| `resname`,`name1` | str,str | |
| `ccd_id`,`rosetta_name`,`params_sha256` | str? | null for protein |
| `mutable` | bool | eligible for identity randomisation |
| `frozen_reason` | str? | e.g. `covalent_anchor` |
| `n_heavy` | int16 | |
| `rel_sasa` | float32 | native, complex context |

**`pairs.parquet`**

| column | type | note |
|---|---|---|
| `pair_id` | int32 | dense, stable within a system, sorted ascending |
| `node_i`,`node_j` | str | |
| `i`,`j` | int32 | pose resnums |
| `kind_i`,`kind_j` | dict(str) | |
| `same_chain` | bool | |
| `seq_sep` | int32 | −1 when undefined (cross-chain or non-protein) |
| `d_ca`,`d_cb` | float32? | null when either node lacks the atom |
| `d_heavy_min` | float32 | |
| `is_bonded` | bool | |
| `in__<defname>` | bool | one column per registered contact definition |

**`native_energies.parquet`** — `pair_id int32`, `e_direct float32`, `e_fa_rep float32`,
`has_edge bool`, plus optional per-term columns (`fa_atr`, `fa_sol`, `fa_elec`, `hbond_*`,
`lk_ball_wtd`) when `energies.store_terms: true`.

**`decoys/energies/part-<shard>-<seq>.parquet`** — the durable scientific record.

| column | type |
|---|---|
| `decoy_id` | int32 |
| `axis` | dict(str) |
| `pair_id` | int32 |
| `e_direct` | float32 |
| `e_fa_rep` | float32 |

Invariants, each enforced by a test:

- **Direct pair energies only, never `E_ij`.** This single choice is what makes the many-body mode,
  the contact definition, the shell and `exclude_fa_rep` all re-specifiable at analyze time.
- `e_fa_rep` is stored weighted but **not** subtracted, so R03 becomes a post-hoc switch.
- One row group per `decoy_id`, rows sorted by `pair_id`.
- A shard writes only its own files — no shared checkpoint, no lock, no `save_every_n_decoys`
  defect.

**`decoys/index.parquet`** — `decoy_id, axis, generator, scope, identity, placement, seed,
source_ref, member_inchikey, n_mutated, backbone_rmsd, wall_s, status, structure_path`.

**`analyses/<id>/contacts.parquet`** — derived, recomputable with **no PyRosetta call**:
`pair_id, i, j, E_native, decoy_mean, decoy_std, decoy_median, decoy_mad, n_decoys, F, index_name,
shapiro_W, shapiro_p, class`.

**`manifest.json` (schema_version 1)**

```json
{"schema_version":1,"run_id":"...","created_utc":"...",
 "code":{"package_version":"...","git_sha":"...","dirty":false},
 "env_digest":"sha256:...",
 "inputs":{"receptor_sha256":"...","params_sha256":["..."],"spec_sha256":"..."},
 "settings_resolved":{...},
 "generation":{"axes":["identity"],"n_decoys_per_axis":{"identity":1000},
   "base_seed_per_axis":{"identity":42},
   "graph_superset":{"ca_cutoff_A":12.0,"heavy_cutoff_A":6.0,"seq_sep_min":1},
   "score_function":"ref2015","rosetta_version":"2026.30+release.bc091c65b8",
   "manybody":"pair_retained","native_repacked":false,
   "regions":{"mutate":"...","repack":"...","minimize":"..."},
   "store_terms":false,"store_mode":"full"},
 "regeneration_key":"sha256:...","contract_version":1}
```

`regeneration_key = sha256(canonical_json(generation-stage settings ∪ input digests ∪
pyrosetta_version))`. Fields are partitioned by an explicit
`Field(json_schema_extra={"stage": "generation"|"analysis"})` annotation in `settings.py`, so the
split is declared once and cannot drift.

**Re-specifiable at analyze time (no regeneration):** classification thresholds; index function;
many-body mode; contact definition and cutoff *within the superset*; `seq_sep_min ≥ 1`;
pocket/shell definition; `exclude_fa_rep`; descriptor; axis subset; **N as a prefix** —
`--n-decoys 100` on a 1000-decoy run takes `decoy_id < 100`, which is exactly the ensemble a
100-decoy run would produce, because seeds stay `base_seed + decoy_id` verbatim as at
`frustration.py:441`/`:666`. The eight-point convergence sweep (R29) therefore costs one run.

**Requires regeneration (hard error, exit 3, with a field-by-field diff):** score function;
receptor or params bytes; protonation/tautomer state; decoy axis parameters, scope or generator;
repack shell radius; backbone treatment; native repack policy; a cutoff exceeding the superset;
PyRosetta version. `--allow-mismatch` records the diff and stamps `provisional: true` on every
output.

---

## 5. CLI surface

```
atomfrust prepare          --spec S.yaml | --pdb F.pdb [--ligand CHAIN:RESSEQ] -o prepared/
atomfrust generate-decoys  --spec S.yaml --run-dir R [--axis identity,chemotype]
                           [--n-decoys N] [--scope {whole_protein,contact_shell,contact_pair}]
                           [--identity {native,composition,uniform20,layer_stratified}]
                           [--placement {inplace,permute}]
                           [--mutate-sel EXPR] [--repack-sel EXPR] [--minimize-sel EXPR]
                           [--repack-shell R] [--relax {min,mc}] [--mc-cycles N]
                           [--library {dude,dekois,muv,zinc_random,local}] [--library-target KEY]
                           [--dock-backend {smina,gnina,mcs_align,preposed}]
                           [--workers N] [--shard k/M] [--store {full,summary}]
                           [--save-structures] [--save-members] [--resume|--restart]
atomfrust analyze          --run-dir R [--contact-def {ca_ca,cb_cb,heavy_min}]
                           [--contact-cutoff-A X] [--seq-sep-min K] [--shell-ref REF]
                           [--shell-A X] [--manybody {pair_retained,chen_literal,pair_only}]
                           [--index {zscore,rank_percentile,robust_z}]
                           [--thresholds MIN,HIGH | --thresholds-mode quantile]
                           [--n-decoys N] [--axis A] [--descriptor D] [-o analyses/<id>]
atomfrust run              = prepare + generate-decoys + analyze
atomfrust validate         --case {lysozyme-core,interface-fraction,reference-counts,apo-control,
                                   pocket-repack-equivalence,energy-reference}
atomfrust converge         --run R --grid 10,25,50,100,250,500,1000,2000 --n-boot 1000
atomfrust strata           --runs 'runs/*/systems/*' --by burial,polarity,volume
atomfrust report           --collect 'runs/*/systems/*/analyses/default' -o report/ [--permute 10000]
atomfrust calibrate        --runs ... --definition ca_ca -o calibration/
atomfrust verify           --run-dir R          # re-digest inputs, replay 3 decoys (R38)
atomfrust metrics selftest [--emit]             # golden-JSON hash (R36, R38)
```

**Worked examples — one per user request.**

```bash
# (U1) ligand as a node; (U5) shell size; non-degenerate many-body by default
atomfrust run --spec specs/5GMP.yaml --run-dir runs/egfr \
  --shell-ref any_heavy --shell-A 6.0 --n-decoys 250

# (U2) decoys for the heteroligand itself: chemotype axis, MCS-aligned, no docking backend
atomfrust generate-decoys --spec specs/5GMP.yaml --run-dir runs/egfr \
  --axis chemotype --library local --dock-backend mcs_align --n-decoys 250

# (U3) external decoy datasets
atomfrust generate-decoys --spec specs/cox1.yaml --run-dir runs/cox1 \
  --axis chemotype --library dude --library-target pgh1 --dock-backend smina --n-decoys 500

# (U4) region-focused decoys: mutate the pocket only, repack a 12 A shell
atomfrust generate-decoys --spec specs/5GMP.yaml --run-dir runs/pocketonly \
  --axis identity --scope contact_shell \
  --mutate-sel 'within(10.0, ligand)' --repack-sel 'within(12.0, ligand)' --n-decoys 250

# (U6) decoys only, no frustration computed at all
atomfrust generate-decoys --spec specs/5GMP.yaml --run-dir runs/gen --n-decoys 1000 \
  --save-structures            # no analyze step runs; no index is produced

# (U7) re-analyse a finished run under different settings, zero PyRosetta calls
atomfrust analyze --run-dir runs/gen --shell-A 5.0  --index rank_percentile -o analyses/s5_rank
atomfrust analyze --run-dir runs/gen --shell-A 8.0  --manybody chen_literal -o analyses/s8_literal
atomfrust analyze --run-dir runs/gen --n-decoys 250 --contact-def heavy_min -o analyses/n250_heavy

# (U8) a custom hand-written PDB, one shot, no metadata pipeline
atomfrust run --pdb my_complex.pdb --ligand B:501 --run-dir runs/mine --n-decoys 250

# protein-only and protein-protein need no mode flag: the spec decides
atomfrust run --spec specs/1LYZ.yaml --run-dir runs/1lyz --n-decoys 250   # ligands: []
atomfrust run --spec specs/1BRS.yaml --run-dir runs/1brs --n-decoys 250   # pocket.mode: chain_interface
```

Region expressions (step B9) use six named selectors compiled onto PyRosetta `ResidueSelector`s —
`chain X`, `resi RANGES`, `resn NAMES`, `within(R, EXPR)` (heavy-atom neighbourhood),
`layer(core|boundary|surface)`, `ligand`/`protein`/`all`/`none` — combined with `and`/`or`/`not`,
plus an escape hatch `xml:<file>#<name>` via `XmlObjects`. Validated invariants, hard errors not
warnings: `mutate ⊆ repack`, `minimize ⊆ repack`, `frozen = all \ (repack ∪ minimize)`.

---

## 6. Implementation steps

Each step is one reviewable change. **Dep** lists step IDs that must land first. Steps within a
stage are independent unless a dep says otherwise.

### Stage A — Diagnose before building (no new package code)

These run against artifacts already on disk. They decide two design parameters that everything else
inherits, and they are cheap.

---

**A1 — Confirm the Eq. 2 degeneracy across all completed structures** ✅ **DONE 2026-08-11**
*Dep:* none · *Touches:* `analysis/diagnose_eq2.py` (throwaway script, not the package)

> **Result:** degeneracy is **universal — 38/38 structures**, `R²(E_native) = 1.0` with worst
> residual 8.6e-14 kcal/mol against a 3.59 kcal/mol spread. `R²(decoy_mean) = 1.0`;
> `R²(decoy_std)` 0.932–0.951; `R²(F_index)` 0.952–0.964. **Only ~4% of the variance in the
> per-contact index is pair-specific.** Report: `analysis/eq2_degeneracy_report.md`. Note:
> `project_status/2026-08-11_0016_a1-eq2-degeneracy-confirmed.md`.

Fit `E_native ~ a_i + a_j` and `F_index ~ a_i + a_j` by least squares on each of the 19 parquets in
`results/`. Report R² per structure and the residual distribution. §2.1 shows R² = 1.000000 for
1XKK; this generalises it.

*Accept:* a table of 19 R² values plus a one-page note stating whether the degeneracy is universal.
If any structure deviates from R² = 1.0 for `E_native`, stop and find out why — it would mean the
partner map is not what the algebra assumes.

---

**A2 — Find which post-hoc setting reproduces the paper's per-structure counts** ✅ **DONE 2026-08-11**
*Dep:* none · *Touches:* `analysis/diagnose_counts.py`

> **Result: negative, and informative.** 189 configurations swept over 50 structures
> (pose→PDB mapping verified by exact contact-set equality for all 50). **0 configurations
> land inside the paper's 4–23 range**; best r vs paper counts is 0.308 at a degenerate
> setting (counts 0–1). Scale is reachable (`heavy4`/`both_in_shell`/0.78 → 3–24), ordering
> is not. Two diagnostics place the fault upstream of the selector: our counts track pocket
> size up to **r = 0.791** while the paper's track it at only **+0.233**; and the paper's
> counts correlate with log₂(affinity_pM) at **r = −0.444, p = 0.0012** while the best swept
> configuration reaches **+0.280** — the **opposite sign**. Therefore the residual
> explanation is the contact definition or the many-body mode, neither variable post-hoc.
> **This makes A4 the gate.** Report: `analysis/count_reproduction_report.md`. Note:
> `project_status/2026-08-11_0042_a2-count-reproduction-negative.md`.

The stored parquets carry `F_index` per contact pair, so classification thresholds and pocket
selectors can be varied *without recomputing anything*. Sweep: threshold pairs on a grid around
(0.78, −1.0); selector ∈ {all contacts, ≥1 partner in 10 Å Cα shell (current), ≥1 partner in
{4,5,6,8} Å heavy-atom shell, both partners in shell}; and report the resulting count per
structure against `config/pdb_reference_table.csv`.

Contact definition and many-body mode cannot be varied post-hoc from these parquets — note that
limitation in the output rather than faking it.

*Accept:* a table of (selector × threshold) → (count range, Pearson r vs paper counts). Either a
setting lands in the paper's 4–23 range with r > 0.5, in which case the reproduction gap is a
selector problem; or none does, in which case it is an energy/many-body problem and step B8's
default matters more than expected. Both outcomes are actionable.

---

**A3 — Apo control: does the ligand affect the index at all?** ✅ **DONE 2026-08-11**
*Dep:* A1 · *Touches:* `analysis/apo_control.py`, uses existing `src/frustration.py` unchanged

> **Result: LIGAND-BLIND.** 5GMP/F62, 50 decoys, seed 42, contact set verified identical
> (1772 pairs), decoy sequences identical by construction. Pocket-contact Pearson
> **r = 0.9904** (threshold 0.95); 31 of 1772 contacts change class (1.7%) when a
> 39-heavy-atom inhibitor is deleted. **Sharper: `E_native` is bit-identical, max |Δ| =
> 0.000e+00 across all contacts** — structural, because partner lists come from
> `get_protein_contacts` (protein-only) and `pairwise_energy` reads a protein–protein
> `EnergyGraph` edge. The native reference provably cannot see the ligand; the ligand's
> entire influence runs through decoy repacking (`decoy_mean` max |Δ| 0.647, `decoy_std`
> 0.706). **Note for B8:** fixing the many-body formula does *not* make the index
> ligand-aware — partner lists stay protein-only — so A4 and ligand-as-node (B6/B7) are
> separate, both-required fixes. Report: `analysis/apo_control_report.md`. Note:
> `project_status/2026-08-11_0115_a3-apo-control-ligand-blind.md`.

Take one completed structure (5GMP), delete the ligand from `_clean.pdb`, re-run the existing
pipeline at `--n_decoys 50` with identical seed, and correlate per-contact `F_index` holo vs apo
over the shared pair set.

This is the falsification test for the whole approach. Under `chen_literal` the ligand enters only
through `B_i` for pocket residues, so a high holo–apo correlation is expected — the question is how
high.

*Cost:* one structure, 50 decoys. Do not run it while the current batch job is live.

*Accept:* Pearson r and the per-contact distribution of |ΔF|, restricted to pocket contacts and
reported separately for pocket and non-pocket. **If r > 0.95 on pocket contacts, the descriptor is
effectively ligand-blind under the current formula**, and that fact goes at the top of any writeup.

---

**A4 — Settle the many-body formula against the source paper** ✅ **DONE 2026-08-11**
*Dep:* A1 · *Touches:* documentation only

> **Result 1 — the transcription is FAITHFUL.** Eq. 2 as printed in Chen et al. 2020
> ([PMC7683549](https://pmc.ncbi.nlm.nih.gov/articles/PMC7683549/)) is
> `E_ij = e_ij + 1/2 Σ_{k,k≠j} e_ik + 1/2 Σ_{l,l≠i} e_jl`. **The exclusions are in the
> paper.** `chen_literal` is the published object; `frustration.py:226-250` renders it
> correctly. The A1 degeneracy is therefore a property of the *published equation*, not a
> defect here. (Note the tension: the prose says E_ij considers "all the interaction
> energies that involve changing any of the two residues", which would be `B_i + B_j − e_ij`
> at coefficient 1. The printed formula is not that quantity.)
>
> **Result 2 — the real gap: the ligand is a node.** The published counts are
> **ligand–residue contacts**: *"a strong inhibitor XTF-262 (PDB ID: 5GMP) forms more than
> ten minimally frustrating interactions with its pocket … a weaker binder 5Q4 forms only
> three"*, and Fig. 5 shows *"frustrations around the ligands only"*. Cross-checked:
> 5GMP = 16 ✓, 5EM8 = 4 ✓. The 4–23 range is the number of residues a drug contacts. We
> count protein–protein pairs in a shell (266–407) — a different object, which is why A2's
> sweep could not close the gap and why A3 found the native reference ligand-independent.
>
> **Result 3 — four further deviations**, each quoted in
> `project_status/2026-08-11_0130_a4-paper-check-ligand-is-a-node.md`: decoys are a
> *sequence shuffle* (permutation) not an i.i.d. composition draw (→ C2); relaxation is a
> *short Monte-Carlo* not a chi-only MinMover (→ C8); **the native is repacked and relaxed
> too** ("obtained in a similar fashion by omitting the shuffling step") whereas we score
> the crystal pose as-is (→ C6, `native.repack` default becomes `true`); and the paper
> states no sequence-separation criterion at all, while we apply `|i−j| ≥ 4` (→ B7).
> Matching correctly: 10 Å Cα–Cα, REF2015, fixed backbone, `exclude_fa_rep`.
> Decoy count: paper 1000, ours 50.

Read Chen et al. 2020 Eq. 2 and its supplementary definition. Determine whether the exclusion of
`j` from `contacts(i)` is in the paper or was introduced in transcription. If the paper's sums run
over *all* partners, then `E_ij = e_ij + 0.5·B_i + 0.5·B_j`, the pair term survives with weight 2,
and `pair_retained` is simply correct rather than a new variant.

*Accept:* one paragraph in this file naming which of the three registry modes is the published
object, with the equation and page cited. **Nothing in Stage C or D should be run for science
before this is answered.**

---

### Stage B — Foundations

No frustration science here. This stage builds the substrate that makes U6, U7, R14, R24 and R37
possible; getting it wrong is expensive to undo later.

---

**B1 — Package skeleton** ✅ **DONE 2026-08-11**
*Dep:* none · *Touches:* `pyproject.toml`, `atomfrust/__init__.py`, `atomfrust/cli/main.py`,
`tests/test_skeleton.py`

`atomfrust` package, console script entry point, `--version`, three pytest markers
(`unit`/`integration`/`anchor`) registered in `pyproject.toml`. Nothing else.

*Accept:* `pip install -e . --no-deps` succeeds in the `frustrato` env without disturbing
PyRosetta; `atomfrust --version` prints; `pytest -m unit` passes with exit 0.
(Corrected from "collects zero tests and exits 0" — pytest exits 5 when nothing is
collected, so the tier ships with real smoke tests instead.)

> **Result: all criteria met.** `atomfrust 0.1.0.dev0`; `pytest -m unit` → 5 passed, 0.20 s;
> pyrosetta/numpy/pandas/scipy/pydantic all unchanged; legacy `src/run_pipeline.py --help`
> runs and `src/test_frustration.py` still collects its 13 tests.
> **Layout deviation:** the package sits at `atomfrust/` (flat), not `src/atomfrust/`. An
> editable install puts the discovery root on `sys.path`; under a `src/` layout that exports
> `frustration`, `run_pipeline`, `rosetta_py`, `molfile_to_params` and `test_frustration` as
> global top-level names, and `import test_frustration` **initialises PyRosetta**. Strict
> editable mode avoids the leak but does not pick up new modules without a reinstall — a
> silent footgun across the ~15 modules still to come. Flat layout gets both; measured, with
> the leak check reporting NONE. Note:
> `project_status/2026-08-11_0415_b1-package-skeleton.md`.

---

**B2 — `settings.py`: layered, validated, stage-partitioned** ✅ **DONE 2026-08-11**
*Dep:* B1 · *Touches:* `atomfrust/settings.py`, `tests/test_settings.py`

> **Result: all criteria met.** 35 unit tests, 0.42 s. Unknown keys raise at every layer;
> generation changes alter `regeneration_key` while analysis and runtime changes do not;
> YAML round-trips byte-identically.
> **Deviation — three stages, not two:** `runtime` added for `workers`/`shard`/etc. Worker
> count must not enter the key (decoy *i* is seeded `base_seed + i`, so results are
> worker-count independent), and those fields are not re-specifiable analysis semantics
> either. Asserted by test.
> **Defaults now encode the A4 findings**, several differing from the prototype:
> `identity=native`+`placement=permute` (the published shuffle), `relax=mc`,
> `native_repack=True`, `n_decoys=1000`, `contacts.seq_sep_min=1`,
> `graph.include_ligand_nodes=True`, and **`manybody.mode=chen_literal`** — see §3.4 note.
> Note: `project_status/2026-08-11_0450_b2-settings.md`.

Pydantic v2 models with `extra="forbid"`. Layering: package defaults → config file → environment →
CLI override, with the winning layer recorded per field. Every field carries
`json_schema_extra={"stage": "generation"|"analysis"}`. `regeneration_key()` hashes the canonical
JSON of the generation-stage subset.

`extra="forbid"` is what structurally prevents the `config.yaml:25-27` dead-key class of bug (R25):
a key nobody reads is a key that fails validation.

*Accept:* unit tests — an unknown key raises; a generation-stage change alters
`regeneration_key` while an analysis-stage change does not; `settings.resolved.yaml` round-trips
byte-identically.

---

**B3 — `spec.py`: the system specification** ✅ **DONE 2026-08-11**
*Dep:* B2 · *Touches:* `atomfrust/spec.py`, `tests/test_spec.py`

> **Result: all four criteria met.** 76 unit tests, 0.82 s. Validation is two-phase — shape
> (no file I/O, unit-testable) and structure (collects *every* problem, names what is
> actually present). **All 61 `data/processed/*_clean.pdb` build a valid spec**, exactly one
> component each, zero problems.
> **Trap found and pinned:** 5HG8's processed PDB says `Z34`, not its true CCD code `634` —
> Stage 4 rewrote the HETATM `resName`. A spec built from a legacy `_clean.pdb` therefore
> carries the *Rosetta* name in `comp_id`: right for the pose, wrong for reporting. When
> `recipes/egfr.py` generates the 61 specs it must take the true code from
> `config/ligand_overrides.yaml` or the Stage-1 inventory, not from the PDB.
> Note: `project_status/2026-08-11_0520_b3-system-spec.md`.

`SystemSpec` and `SystemSet` per §3.2. Loaders for YAML, CSV and JSON. A `from_pdb()` constructor
that takes a bare PDB path plus an optional `CHAIN:RESSEQ` ligand selector and synthesises a
minimal spec — this is U8.

*Accept:* a hand-written 12-line YAML validates; `from_pdb("x.pdb", ligand="B:501")` produces an
equivalent spec; `ligands: []` validates and sets protein-only mode; a spec naming a chain absent
from the PDB fails with the chain list in the message.

---

**B4 — `provenance.py`: digests, environment, manifest** ✅ **DONE 2026-08-11**
*Dep:* B2 · *Touches:* `atomfrust/provenance.py`, `tests/test_runstore.py`

> **Result: all criteria met.** Manifest round-trips; a modified receptor changes both the
> digest and the `regeneration_key`; an unknown `schema_version` *or* `contract_version`
> raises rather than being interpreted.
> **Two deliberate choices:** `env_digest` excludes `cpu_count`/`platform`/`machine` —
> results must not depend on core count, and demanding an identical OS string would block
> cross-machine comparison (both are still recorded as diagnostics). PyRosetta's version is
> read from distribution metadata, never by importing it, asserted by a subprocess test so
> the unit tier stays usable without it.

SHA-256 of every input file; environment capture (Python, PyRosetta version string, OS, CPU count,
git SHA + dirty flag, installed package versions); manifest read/write against the §4 schema with
`schema_version` enforcement. This is R37.

*Accept:* a manifest written and re-read compares equal; a modified receptor changes
`inputs.receptor_sha256`; reading a manifest with an unknown `schema_version` raises rather than
guessing.

---

**B5 — `runstore.py`: the run directory** ✅ **DONE 2026-08-11**
*Dep:* B4 · *Touches:* `atomfrust/runstore.py`, `tests/test_runstore.py`

> **Result: all criteria met.** 178 tests passing. 100 decoys across 4 shards, prefix of 25
> reads back frame-identical to the same rows of the full read — the prefix is exact, not
> approximate. Analysis-stage changes report `analysis_only`; generation-stage changes raise
> `RegenerationRequired` with `exit_code == 3` and a diff naming the field.
> **End-to-end demonstration (integration):** 5GMP is stored in one PyRosetta pass, then
> reopened and **both** many-body formulas are formed from the same bytes with no pose —
> `pair_retained − chen_literal == e_ij` verified on live data, and the switch reports
> `analysis_only`. This is user request 7, and it means the corrected formula is a
> re-analysis rather than a re-run of 61 structures.
> **Two prototype defects retired:** the parquet short-circuit (`run_pipeline.py:241`) that
> made a larger `--n_decoys` silently do nothing, and the all-or-nothing checkpoint whose
> default interval equalled the usual decoy count. A test kills a writer mid-run and asserts
> flushed decoys survive. A test also asserts **no `E_ij`-like column is ever stored**.
> Note: `project_status/2026-08-11_0745_b4-b5-run-directory.md`.

`RunDir` implementing the §4 layout: path resolution, sharded parquet writers and readers,
`STATUS.json` update, `assert_compatible(settings)` returning a field-by-field diff and the
generation/analysis verdict, `--allow-mismatch` handling.

Two defects of the prototype die here: the parquet short-circuit at `run_pipeline.py:241` that
makes a larger `--n_decoys` silently do nothing (R14), and the single shared checkpoint whose
default interval equals the usual decoy count.

*Accept:* unit tests with no PyRosetta — write 100 synthetic decoys across 4 shards, read back a
prefix of 25 and get exactly `decoy_id < 25`; an analysis-stage settings change passes
`assert_compatible`; a generation-stage change fails with exit code 3 and a diff naming the field.

---

**B6 — `pose.py`: multi-component loading and node identity** ✅ **DONE 2026-08-11**
*Dep:* B3 · *Touches:* `atomfrust/pose.py`, `tests/test_pose.py`

> **Result: all criteria met.** Two-copy ligand → two distinct nodes; metal → `metal` node;
> protein-only → zero components; 634/Z34 keeps both names.
> **Two bugs found:** duplicate `.params` crashed multi-copy ligands (`residue type 'F62'
> already exists`) — now deduplicated by resolved path; and **Rosetta silently typed an
> unparametrised ligand from the bundled CCD**, so a run could use CCD atom typing while its
> manifest claimed curated params. `-in:file:load_PDB_components false` is now in
> `DEFAULT_INIT_FLAGS` and the bare Rosetta exit is wrapped with the system, component and
> fix. Node-id collisions are now detected (it is the join key).
> Note: `project_status/2026-08-11_0620_b6-b7-ligand-is-a-node.md`.

`load_complex(spec) -> (pose, components)`. All params passed through
`pyrosetta.init("-extra_res_fa ... -auto_setup_metals -in:file:load_PDB_components false")` in the
worker initializer for the fixed native; the per-pose dynamic path
(`generate_nonstandard_residue_set`) is retained for the chemotype axis, where a worker scoring
thousands of distinct ligands cannot re-`init()`.

Node lookup is by `PDBInfo (chain, resseq, icode)`. This delivers R21 and replaces
`find_ligand_resnum` (`run_pipeline.py:186`), whose `name3()` prefix match returns the *first*
matching residue and therefore cannot see a second ligand copy, a metal, or a cofactor.

*Accept:* integration tests — a two-copy ligand PDB yields two distinct ligand nodes; a
metal-containing PDB yields a `metal` node; a protein-only PDB yields zero non-protein nodes; the
`634`/`Z34` case resolves through `components.yaml` with no special-casing in calling code.

---

**B7 — `graph.py`: typed nodes, KD-tree neighbours, superset graph** ✅ **DONE 2026-08-11**
*Dep:* B6 · *Touches:* `atomfrust/graph.py`, `tests/test_graph.py`

> **Result: all criteria met**, PyRosetta-free (23 unit tests on synthetic geometry).
> **Cross-validation:** at the prototype's definition (Cα 10 Å, |i−j| ≥ 4) the new graph
> yields **exactly 1772** protein–protein pairs for 5GMP, matching the stored parquet via a
> completely different code path.
> **The A4 correction, quantified:** ligand contacts for 5GMP are 12 / 15 / 19 / **27** / 47
> at 4.0 / 4.5 / 5.0 / **6.0** / 8.0 Å — against zero in the prototype. The published 16
> *minimally frustrated* contacts implies a shell of ~5 Å or wider. Scale now matches; this
> is not yet evidence the frustration values will.
> **Key design point:** a named definition is **per-kind-pair** — a ligand has no Cα, so
> `ca_ca` judges ligand pairs by heavy-atom minimum against `contacts.ligand_cutoff_A`.
> Without that split, selecting `ca_ca` would silently restore protein-only behaviour.
> `seq_sep = -1` on ligand pairs is exempt from sequence-separation filtering for the same
> reason. **Settings changed:** superset `heavy_cutoff_A` 6.0 → 8.0 (headroom for post-hoc
> shells), added `contacts.ligand_cutoff_A`.

Node table and pair table per §4. `scipy.cKDTree` over heavy atoms. Per-kind-pair edge criteria per
§3.3. Named contact definitions (`ca_ca`, `cb_cb`, `heavy_min`) materialised as `in__<defname>`
boolean columns on the superset. Sequence separation applied **per chain** (fixes R02). Contact
membership computed on the native pose only and frozen — asserted, not assumed.

This step delivers R01, R02, R05, R24 and U1/U5 at the graph level.

*Accept:* unit tests on synthetic coordinates — KD-tree pairs match a brute-force O(N²) reference
exactly; two residues in different chains at `|i−j| = 2` are *not* excluded; a ligand heavy atom
6.1 Å from a Cα produces no `heavy_min(6.0)` edge but does appear in the superset; narrowing
`ca_ca` from 10 Å to 8 Å is a pure column filter with no recomputation.

---

**B8 — `energy.py`: pairwise energy and the many-body registry** ✅ **DONE 2026-08-11**
*Dep:* B7, A4 · *Touches:* `atomfrust/energy.py`, `tests/test_energy.py`

> **Result: all criteria met.** 148 tests passing.
> **Reproduction:** all 1772 stored `E_native` values for 5GMP reproduced — max |Δ| =
> **1.04e-05** against a spread of 3.566 (2.9e-06 relative), median 5.45e-08, Pearson
> r = 1.0000000000. The residual is float32 storage vs the prototype's float64, not an
> algorithmic difference.
> `chen_literal` is additive to R² = 1 and `pair_retained` is not, both asserted; the
> vectorised path is checked against a literal un-simplified transcription so the algebraic
> cancellation is proved, not assumed. `EnergyEvaluator` scores the pose **once** — the
> prototype rescored inside every `pairwise_energy` call (`frustration.py:167`).
> **Two doc corrections:** §3.4's `pair_retained` formula was itself degenerate (fixed
> above); and `CLAUDE.md`'s edgeless-pair claim is wrong — measured edge coverage is 100%
> to 12 Å, so no 10 Å Cα pair is edgeless. Note:
> `project_status/2026-08-11_0700_b8-energy.md`.

`pairwise_energy` lifted verbatim from `frustration.py:182-213`, with one change: `e_fa_rep` is
*returned alongside* `e_direct` rather than subtracted, so R03 becomes an analyze-time switch.
Many-body registry per §3.4 with the three modes.

*Accept:* unit tests — `chen_literal` reproduces the prototype bit-for-bit on a stored pose and an
assertion confirms `chen_literal ≡ 0.5(B_i + B_j)` symbolically on synthetic energies;
`pair_retained` breaks additivity (additive-fit R² < 0.99) on the same input; a pair with no
`EnergyGraph` edge yields `e_direct == 0.0` and `has_edge == False`, and that population is counted
and reported rather than silently mixed in.

---

**B9 — `regions.py`: the selector language** ✅ **DONE 2026-08-11**
*Dep:* B7 · *Touches:* `atomfrust/regions.py`, `tests/test_regions_execute.py`

> **Result: all criteria met.** Evaluated against the node table and geometry, not a pose,
> so the module is PyRosetta-free and unit-testable.
> **Acceptance forced a real distinction:** the prototype's `get_ligand_contacts` measures
> **Cα** to ligand heavy atom, while `within(R, expr)` is heavy-to-heavy — different sets. So
> **`within_ca(R, expr)`** was added and verified to return *exactly* the legacy function's
> residue set on 5GMP, with a test asserting `within` is a strict superset.
> `mutate ⊄ repack` and `minimize ⊄ repack` are hard errors (both are silent corruption);
> an empty mutate set is legal; non-mutable nodes are intersected out so a ligand, metal or
> covalent anchor can never be identity-randomised even under `all`.
> **Deviation:** the `xml:` escape hatch is not implemented — it lexes and raises a message
> naming the available selectors rather than a parse error. Deferred, not half-built.

Six named selectors compiled to PyRosetta `ResidueSelector`s per §5, with `and`/`or`/`not` and the
`xml:` escape hatch. `ResolvedRegions(mutate, repack, minimize, frozen)` with the three subset
invariants enforced as hard errors. This is the substrate for U4 and R12.

*Accept:* unit tests — `within(10.0, ligand)` on a known structure returns the residue set the
prototype's `get_ligand_contacts` returns at the same cutoff; `mutate ⊄ repack` raises;
`frozen` is exactly the complement; an empty `mutate` set is legal (it means a repack-only decoy).

---

**B10 — `execute.py`: flat work queue and sharding** ✅ **DONE 2026-08-11**
*Dep:* B5, B6 · *Touches:* `atomfrust/execute.py`, `tests/test_regions_execute.py`

> **Result: all criteria met.** 214 tests passing. Results are identical at 1/2/4/8 workers
> against a serial reference; 4 shards over 100 decoys reassemble to `range(100)` with no
> overlap or gap; 3 merged shards equal the single-process result; resume is subtraction.
> **A larger `--n_decoys` now extends an ensemble** (50 done, 200 requested → plans exactly
> 50–199), retiring the parquet short-circuit at `run_pipeline.py:241`.
> The executor is generic over the task callable, so all of this is tested without
> PyRosetta; C2 supplies the real generator. `task_factory` runs once per worker rather than
> pickling the task per unit, and a pose is never pickled — hence `spawn`.
> Note: `project_status/2026-08-11_0830_b9-b10-regions-execution.md`.

One flat queue of `(system, axis, decoy_id)` units, `spawn` context, worker initializer that calls
`pyrosetta.init()` and rebuilds its own pose from `(receptor.pdb, params)` — never pickling a pose.
`--shard k/M` selects a disjoint subset so independent processes or machines can cooperate without
a lock, because each shard writes only its own parquet files.

This replaces the two-level nesting of `run_all_egfr` (`run_pipeline.py:442`) whose forced
`n_jobs_decoys=1` is the reason `--mode all` underuses the box.

*Accept:* integration test — 8 decoys across 3 shards produce the same 8 `decoy_id`s with no
overlap and no gap; results are bit-identical to a single-shard run because seeding is
`base_seed + decoy_id`; killing one shard mid-run and restarting it with `--resume` produces the
same final set.

---

### Stage C — Decoy engine, protein side

---

**C1 — `decoys/base.py`: the generator protocol** ✅ **DONE 2026-08-11**
*Dep:* B10, B8 · *Touches:* `atomfrust/decoys/base.py`, `tests/test_decoys.py`

> **Result: criterion met.** `NullGenerator` produces energies exactly equal to the stored
> native energies (array equality, not a tolerance). `DecoyContext` holds the per-worker
> state so the executor's task factory builds a pose once.

`DecoyGenerator` protocol: `generate(pose, spec, regions, rng) -> DecoyResult`, where `DecoyResult`
carries the scored pose plus the `index.parquet` row. Energy extraction is shared: one scored pose
in, `(pair_id, e_direct, e_fa_rep)` rows out for the frozen pair set. R13's "one randomisation
serves all contacts" lives here and nowhere else, so the alternative can be tested by swapping the
generator rather than rewriting the engine.

*Accept:* a `NullGenerator` returning the native pose produces decoy energies equal to
`native_energies.parquet` to within float32 rounding — the tightest available end-to-end check on
the extraction path.

---

**C2 — `decoys/identity.py`: scope × identity × placement** ✅ **DONE 2026-08-11**
*Dep:* C1, B9 · *Touches:* `atomfrust/decoys/identity.py`, `tests/test_decoys.py`

> **Result: bit-for-bit reproduction achieved.** Decoys 0 and 1 regenerated under
> `protocol="legacy"` match `checkpoints/1XKK_FMM_ckpt.pkl` at max |Δ| = 1.18e-05 / 1.51e-05
> over 1708 pairs, r = 1.0000000000 (residual is float32 storage). This exercises the whole
> stack and survives a stochastic packing step, so RNG consumption is faithfully reproduced.
> Required matching three details: the alphabet is `list(aa_freq.keys())` in
> **first-appearance order**, `np.random.choice` is called **once per residue**, and each
> mutation is applied immediately so later draws see earlier ones.
> **Silent inconsistency found:** the prototype separates by **pose index**
> (`frustration.py:98`), which across a numbering gap treats sequence-distant residues as
> adjacent. 1XKK (5 gaps) gives 1714 pairs by PDB numbering vs 1708 by pose index; 5GMP
> (2 gaps) gives 1772 either way. Structure-dependent and entirely silent. Now an explicit
> `contacts.seq_sep_basis` setting with both columns stored.
> `identity=native, placement=permute` (the published shuffle) is default and conserves the
> multiset exactly; the prototype's `composition` draw provably does not.
> Note: `project_status/2026-08-11_0930_c1-c2-c5-decoy-engine.md`.

Three orthogonal switches replace the single hardcoded protocol at `frustration.py:298-380`:

| switch | values | meaning |
|---|---|---|
| `scope` | `whole_protein` · `contact_shell` · `contact_pair` | *which* residues are randomised |
| `identity` | `composition` · `native` (native multiset) · `uniform20` · `layer_stratified` | *what* identities exist |
| `placement` | `inplace` · `permute` | *where* they sit |

`scope=whole_protein, identity=composition, placement=inplace` reproduces the prototype exactly.

This is R08 and R09. Note which switch carries the deviation: #7 says "contacting residues" and the
prototype mutates all ~300 (`frustration.py:333-343`), so **`scope` is the gap, not `identity`** —
and under a position-independent composition draw, `permute` is a distributional no-op, so
identity × placement alone cannot explain it. `scope=contact_pair` is the reading consistent with
#6's "1000 decoys **per contact**".

*Accept:* the prototype configuration reproduces stored 1XKK decoy energies bit-for-bit at equal
seed; `scope=contact_shell` mutates exactly the residues `regions.mutate` names; `identity=native`
preserves the native amino-acid multiset exactly; `placement=permute` with `identity=native` is a
pure permutation of the native sequence over the mutate set.

---

**C3 — Single-PackerTask mutation** ✅ **DONE 2026-08-11 — premise not supported**
*Dep:* C2 · *Touches:* `atomfrust/decoys/identity.py`, `tests/test_decoys.py`

> **Result: the speed-up does not exist.** Measured on 5GMP: **4.5 s vs 4.6 s** per decoy
> (1.02×). Packing and minimisation dominate; the ~300 `MutateResidue` calls do not. The
> plan's stated rationale was wrong.
> The two paths design the **identical sequence** (305/305 verified) but land in different
> rotamer minima: median |Δe| ≈ 3e-06, max |Δe| = 586 (a clashing rotamer, `fa_rep`-driven),
> `E_ij` r = 0.9982 with max |Δ| = 9.5 against an `E_native` spread of ~3.6 — roughly 3σ on
> the pairs it touches, for no gain.
> **`mutation="sequential"` is therefore the default.** `packer_task` is kept for a different
> reason than the plan gave: it is the **only** way to express a restricted repack or frozen
> region, which C7 / user request 4 require. Constructing a `sequential` generator with
> restricted regions now raises, naming `packer_task` — sequential repacks the whole pose and
> would otherwise honour the region silently and wrongly.

Replace ~300 sequential `MutateResidue` calls plus a separate repack (`frustration.py:342-348`)
with one `PackerTask` built from per-residue operations: `RestrictAbsentCanonicalAASRLT` pinning
each mutate position to its drawn identity, `RestrictToRepackingRLT` on `repack \ mutate`,
`PreventRepackingRLT` on `frozen`, each wrapped in `OperateOnResidueSubset`. The prototype pushes a
single whole-pose `RestrictToRepacking()` (`frustration.py:347`) and has no per-residue operation at
all, which is why no region restriction is currently expressible.

*Accept:* identical sequences and energies within float tolerance versus C2's sequential path on
20 decoys, plus a wall-clock comparison recorded. Correctness first; the speed-up is the reason but
not the acceptance test.

---

**C4 — Per-position seed substreams** ✅ **DONE 2026-08-11**
*Dep:* C2 · *Touches:* `atomfrust/decoys/identity.py`, `tests/test_decoys.py`

> **Result: criterion met.** Identities come from
> `SeedSequence(entropy=seed, spawn_key=(decoy_id, position))`; position-by-position draws
> are unchanged when the mutate set grows from 40 residues to 305 at fixed `decoy_id`. The
> decoy-level seed stays `base_seed + decoy_id`, so the prefix property and worker-count
> independence are untouched.
> **Limit worth stating:** substreams apply only to the position-independent modes. A
> permutation is a property of the whole set, so `placement="permute"` — the published
> protocol — keeps a single stream, and comparisons that vary the mutate set under `permute`
> are inherently unpaired.
> Note: `project_status/2026-08-11_1015_c3-c4-packer-and-seeding.md`.

Decoy-level seeding stays `base_seed + decoy_id` applied to `random`, NumPy and Rosetta's RNG
exactly as at `frustration.py:319-322` — this is what gives the N-prefix property and worker-count
independence. **Within** a decoy, identities are drawn from
`SeedSequence(entropy=seed, spawn_key=(decoy_id, pose_resnum))`, so the identity at a position does
not depend on mutate-set size or iteration order.

Without this, a run with `--repack-shell 8` and one with `--repack-shell inf` differ in *sequence*
as well as packing, and the pocket-repack equivalence test (F6) is unpaired and therefore
meaningless.

*Accept:* unit test — the identity drawn at position 745 is unchanged when the mutate set grows
from 40 residues to 300, at fixed `decoy_id`.

---

**C5 — Backbone invariant as a runtime post-condition** ✅ **DONE 2026-08-11**
*Dep:* C3 · *Touches:* `atomfrust/decoys/identity.py`, `tests/test_decoys.py`

> **Result: criterion met.** `assert_backbone_identical` runs on **every** decoy at 1e-6 Å,
> rather than once in the suite at 0.05 Å as the prototype did. A positive-control test
> moves an atom and confirms the assertion fires.

Keep the N/CA/C/O hard restore (`frustration.py:366-378`), scoped to `mutate ∪ minimize`, and
promote `test_decoy_backbone_unchanged`'s 0.05 Å check into
`assert_backbone_identical(native, decoy, tol=1e-6)` executed on every decoy under
`--strict` (default on in tests, samplable in production).

*Accept:* the assertion fires if the restore block is deleted; a full run with `--strict` shows
zero violations.

---

**C6 — `native.repack` as a first-class setting** ✅ **DONE 2026-08-11 — large effect**
*Dep:* C3 · *Touches:* `atomfrust/decoys/identity.py`, `analysis/ablate_native_and_relax.py`

> **Result: the asymmetry was material, not cosmetic.** Ablation over 3 structures × 15
> decoys, with conditions A and B **sharing a decoy ensemble** so the effect is isolated with
> zero sampling noise. Repacking the native lowers its energy and raises F throughout:
> `frac_highly` collapses (5GMP 0.068 → 0.010, −86%; 3POZ 0.038 → 0.012; 1XKK 0.028 → 0.015)
> and `mean_F` **more than doubles on 5GMP** (0.33 → 0.79).
> The prototype's crystal-pose native therefore inflated the highly-frustrated count by
> 2–7×. Default is `True`, per A4.

The published protocol is asymmetric: the native is scored as deposited (`frustration.py:612-621`)
while every decoy is repacked and minimised (`:346-364`). Any energy difference from repacking
alone therefore lands entirely in the native–decoy gap. Make it a setting, default `false` to match
the protocol, and produce an ablation table of class fractions with and without symmetric native
treatment.

*Accept:* the ablation table for ≥3 structures. If the offset shifts class fractions by more than a
few percentage points, that is a finding, and the default becomes a decision (§10, Q4).

---

**C7 — Region-focused repacking** ✅ **DONE 2026-08-11** (CLI flags land in E2)
*Dep:* C3, B9 · *Touches:* `atomfrust/decoys/identity.py`, `tests/test_decoys.py`

> **Result: criteria met.** Regions reach the packer task — only `repack` is repacked,
> `frozen` is prevented. Every decoy records `wall_s` and `repack_residues`, so the cost
> curve over shell radius is a groupby rather than a separate experiment. A `sequential`
> generator with a restricted region raises rather than silently repacking everything.

Wire `--mutate-sel`, `--repack-sel`, `--minimize-sel` and `--repack-shell R` through to the
`PackerTask`. This is R12 and U4. Record `wall_s` per decoy in `index.parquet` so cost as a
function of shell radius and pocket size falls out of a groupby (R39).

*Accept:* `--repack-shell 8` touches strictly fewer residues than `inf`, verified from the task's
own `designing_residues`/`repacking_residues`; a cost curve over shell ∈ {6, 8, 10, 12, inf} on one
structure. Equivalence to whole-protein repacking is F6, not here.

---

**C8 — Monte-Carlo relaxation option** ✅ **DONE 2026-08-11 — ~5× cost, <1 pp effect**
*Dep:* C3 · *Touches:* `atomfrust/decoys/identity.py`, `analysis/ablate_native_and_relax.py`

> **Result: implemented and measured.** Backbone stays bit-identical. Cost 4.9 → 23.2 s per
> decoy (~5×); `frac_minimal` moves by 0.003 / 0.007 / −0.010 across the three structures and
> `mean_F` by <0.02. Faithful to the paper but close to inert at aggregate level.
> **PI decision:** `relax="mc"` is the default because A4 established it is the published
> protocol; `relax="min"` is one flag away and ~5× cheaper — roughly 6 vs 30 core-hours per
> structure at the paper's 1000 decoys.
> The paper specifies only "a short Monte-Carlo relaxation… with the backbone fixed", so the
> interpretation (re-anneal rotamers, minimise chi, Metropolis, recover lowest) is documented
> at the implementation rather than left implicit.
> **Unplanned consistency check:** `lig_frac_highly` is **0.0000** in every structure and
> condition, and `config/pdb_reference_table.csv` records 0 published highly-frustrated
> contacts for these structures. First qualitative agreement with the paper since the ligand
> became a node, and nothing was tuned to produce it.
> Note: `project_status/2026-08-11_1130_c6-c7-c8-stage-c-complete.md`.

R11: the document specifies "short Monte-Carlo relaxation"; the prototype does a single chi-only
MinMover pass (`frustration.py:364`). Add `--relax {min,mc}` with `--mc-cycles N`, where `mc` is a
short side-chain-only Monte-Carlo at fixed backbone. Default stays `min` until the ablation says
otherwise.

*Accept:* `--relax mc` leaves the backbone bit-identical (C5's assertion still holds); an ablation
on ≥3 structures reporting class-fraction change and wall-clock cost versus `min`.

---

### Stage D — Analysis (no PyRosetta anywhere in this stage)

---

**D1 — `analyze/zscore.py`: index functions and normality diagnostics** ✅

> ✅ **DONE 2026-08-11.** 26 tests. Reproduces stored `F_index` from stored moments at max deviation **1.78e-15**.
*Dep:* B5 · *Touches:* `atomfrust/analyze/zscore.py`

Three index functions sharing one decoy vector, all computed in one pass:
`zscore` = `(mean − native)/std(ddof=1)` with σ < 1e-9 → 0, matching `frustration.py:688-701`;
`rank_percentile` = `2·((#decoys with E > native) + 0.5·ties)/N − 1`;
`robust_z` = `(median − native)/(1.4826·MAD)`. Per-contact `shapiro_W`, `shapiro_p`, skew and
excess kurtosis stored alongside. This is R26 and R27.

*Accept:* unit tests against closed-form fixtures; `zscore` reproduces stored `F_index` from stored
`E_native`/`decoy_mean`/`decoy_std` on all 19 parquets to float32 tolerance.

---

**D2 — `analyze/classify.py`: one home for thresholds** ✅

> ✅ **DONE 2026-08-11.** 22 tests. Single home for the rule; the anti-triplication guard is an **AST scan**, not grep, because prose in two docstrings spells the numbers out. NaN **raises** rather than falling through to `neutral` — the descriptor CLAUDE.md flags as most suspect.
*Dep:* D1 · *Touches:* `atomfrust/analyze/classify.py`

`classify_index(F, thresholds)` imported by the engine, the plots and the tests — closing the
triplication at `frustration.py:703-708` / `run_pipeline.py:415-416` / `test_frustration.py:139-144`
and the dead config block. This is R25.

Provenance recorded in the docstring and the output manifest: 0.78 / −1.0 are quantiles of the
coarse-grained null inherited from Ferreiro, not universal constants. Under a different contact
definition or decoy axis the same number selects a different quantile. `--thresholds-mode quantile`
plus `atomfrust calibrate` (E6) produce definition-specific cutoffs; the reproduction run is pinned
to `mode: fixed`.

*Accept:* a grep-based test asserting the literals `0.78` and `-1.0` appear in exactly one module;
changing `classify.thresholds` in config changes plot guides and test expectations together.

---

**D3 — `analyze/aggregate.py`: descriptor registry with mandatory covariates** ✅

> ✅ **DONE 2026-08-11.** 23 tests. **Recomputes `n_contacts_total`, `n_minimally_frustrated`, `n_neutral`, `n_highly_frustrated` exactly for all 61 structures** (verified independently). Reports honestly that the pocket residue list is not stored, so it recovers it from the PDBs and verifies pose numbering rather than assuming it.
*Dep:* D2, B7 · *Touches:* `atomfrust/analyze/aggregate.py`

Descriptors as a registry with **no privileged headline** (R28): `count_minimal`, `count_highly`,
`count_neutral`, `frac_minimal`, `frac_highly`, `energy_weighted_sum`, `mean_Z`, `median_Z`,
`mean_Z_top_decile`, `count_minimal_per_pocket_residue`, `net_frustration` — each evaluated against
each index and each shell setting, column-named `desc__{descriptor}__{index}__{shell_hash}`.

Pocket selectors replace `summarize_ligand_frustration` (`frustration.py:727-762`):
`incident_to(node_set)`, `inter_chain`, `within_shell(node, r)`, `all`. Shell radius and reference
atom are analyze-time parameters — this is U5 and R24 at the descriptor level.

Mandatory covariate columns on **every** row: `n_contacts_total`, `n_pocket_residues`,
`mean_residue_degree`, `pocket_sasa_A2`, `ligand_heavy_atoms`, `n_protein_residues`,
`n_resolved_residues`. Given §2.1, these are not optional context — they are the quantities the raw
counts are suspected of tracking.

*Accept:* recomputing `n_minimally_frustrated` at the prototype's settings reproduces
`egfr_frustration_summary.csv` for all 19 structures exactly; changing `--shell-A` changes the
count without touching decoy data; every emitted row carries all seven covariates.

---

**D4 — `analyze/converge.py`: N as a prefix** ✅

> ✅ **DONE 2026-08-11.** 22 tests. σ standard errors reproduce the published 10.10 / 4.48 / 2.24 % at N = 50/250/1000. Prefix exactness asserted frame-by-frame.
*Dep:* D1, B5 · *Touches:* `atomfrust/analyze/converge.py`

The convergence sweep (R29) is subsampling over one stored ensemble, not eight runs, because
`decoy_id < N` is exactly the ensemble an N-decoy run would produce under `base_seed + decoy_id`
seeding.

One caveat must be stated in the output: `CLAUDE.md` warns that mixing `n_decoys` across
*structures* makes them non-comparable, because σ shifts the F scale. That warning concerns
comparing structures at different N; subsampling compares one structure against itself at several
N, which is the intended use and is unaffected. The generated report says so explicitly so the
distinction is not lost.

*Accept:* on a stored 1000-decoy run, the N = 50 prefix reproduces a separately-executed 50-decoy
run bit-for-bit; the ρ-vs-N curve and bootstrap CIs are emitted for ≥1 structure.

---

**D5 — `analyze/strata.py`: σ across strata and axis redundancy** ✅

> ✅ **DONE 2026-08-11.** 32 tests. `cv_across_strata` is the between-stratum number, since small-within/large-between is exactly the failure a Z-score hides.
*Dep:* D1 · *Touches:* `atomfrust/analyze/strata.py`

R30: coefficient of variation of decoy σ across burial / polarity / volume strata, with per-pocket
descriptors written to `native/pocket.json` at generation time so this is a query and not a re-run.
R31: pairwise correlation of per-contact indices across axes, computed from the shared `pair_id`
key.

*Accept:* both run against stored runs with zero PyRosetta calls; stratum assignment is
deterministic and recorded per pocket.

---

**D6 — `metrics/`: one implementation, `Estimate` everywhere** ✅

> ✅ **DONE 2026-08-11.** 34 tests. **max-T null calibration: empirical FWER 0.055 at nominal 0.05, where the unadjusted p rejects at 0.42** — direct evidence §2.3's multiplicity problem is real. BEDROC/EF1% cross-checked against RDKit to 1e-15.
*Dep:* B1 · *Touches:* `atomfrust/metrics/screening.py`, `inference.py`

Pure NumPy/SciPy, no PyRosetta. Every public function returns
`Estimate(value, ci_low, ci_high, n, n_groups, method, n_boot, seed, p_value)` — never a bare
float. AUROC, BEDROC (α = 80.5), adjusted logAUC, EF1%, Pearson, Spearman (R32).
`estimate()` resamples **targets** first, then molecules within target, and raises if `groups=None`
is passed with a screening metric (R33). `paired_delta()` bootstraps per-target differences (R34).
`maxT_permutation(grid, y, n_perm=10000)` permutes the outcome vector and takes the max |r| across
the entire configuration grid — this, not BH, is what a reported maximum requires (R35, §2.3).
All randomness from `np.random.default_rng(seed)`; no global RNG anywhere.

*Accept:* unit tests against closed-form fixtures — AUROC vs Mann–Whitney U; BEDROC at α → 0 equals
AUROC; EF1% on a hand-constructed ranking; BH against a worked example; max-T on a null grid
rejects at the nominal rate over 1000 synthetic replicates.

---

**D7 — `report/`: confound-aware reporting** ✅

> ✅ **DONE 2026-08-11.** 20 tests. Measured on the real 61-structure set: every raw CI spans zero, max-T adjusted p ≥ 0.69 — nothing survives, so the guard never has to fire. Guard verified on synthetic data (raw +0.500, partial +0.066, headline refused).
*Dep:* D6, D3 · *Touches:* `atomfrust/report/collect.py`, `plots.py`

Every correlation emitted as a triple: raw, partial controlling `n_contacts_total`, and OLS
`y ~ descriptor + n_contacts_total` with the descriptor coefficient CI and VIF — plus the
max-T-adjusted p. A test enforces that `report` **refuses to print a headline** for any descriptor
whose partial-correlation CI includes zero while its raw CI does not, and prints the covariate
warning instead.

Default plot is `frac_minimal` with `n_contacts_total` mapped to marker size; the raw count is a
secondary panel labelled "confounded". This is the direct consequence of §2.1 — the confound is
structural, so the reporting layer must be structurally unable to hide it.

*Accept:* fed the current 19-structure summary, `report` prints the covariate warning rather than a
headline r, and the test proving it does so is part of `pytest -m unit`.

---

**D8 — Raw REF2015 interaction energy control** ✅

> ✅ **DONE 2026-08-11.** Raw REF2015 interaction energy stored beside every Z-score. Partitions interaction / intra-protein / intra-component, each with and without `fa_rep`.
*Dep:* B8 · *Touches:* `atomfrust/energy.py`, `runstore.py`

R07: compute and store the raw REF2015 interaction energy of every complex alongside every Z-score,
into `native/raw_energy.json`. Cheap, and it is the only control that separates *normalisation*
from *energy-function quality* — comparing a Rosetta Z-score against a Vina score confounds the two.

*Accept:* present in every run directory; a report column carries raw-vs-Z side by side.

---

### Stage E — CLI

Each subcommand is a thin adapter over the modules above; the logic is already tested by then.

| Step | Command | Dep | Delivers | Accept |
|---|---|---|---|---|
| **E1** | `prepare` | B3, B6, G1 | U8 | a bare PDB + `--ligand B:501` produces a valid prepared system dir |
| **E2** | `generate-decoys` | C2, C7, B10 | **U6** | runs to completion, writes decoys and energies, **produces no index and no analysis** |
| **E3** | `analyze` | D1–D4 | **U7** | runs on a run dir produced by E2 with zero PyRosetta calls; three different `--shell-A` values give three analyses from one generation |
| **E4** | `run` | E1–E3 | — | equals E1+E2+E3 composed; `run` then `analyze` again yields identical numbers |
| **E5** | `validate --case` | F1–F6 | R40 | each case runs standalone and prints pass/fail against a stored expectation |
| **E6** | `converge`,`strata`,`report`,`calibrate` | D4–D7 | R29–R31, R39 | each runs against stored runs only |
| **E7** | `verify`, `metrics selftest` | B4, D6 | R38 | `verify` re-digests inputs and replays 3 decoys, reporting bit-equality; `metrics selftest` hashes a golden JSON |

---

### Stage F — Validation cases

Each is an executable case under `atomfrust validate` with a stored expected value and tolerance,
not a prose claim.

---

**F1 — Lysozyme core/surface signature** ✅

> ✅ **DONE 2026-08-11.** PASS. Core 55.5 % vs exposed 25.3 %, diff +0.302, CI [+0.215, +0.393]. **Exposed a defect in `regions.py`**: `layer(surface)` selected 2 of 129 lysozyme residues because Rosetta's cone-weighted LayerSelector defaults had been multiplied by 4 onto a raw neighbour count. Recalibrated to core ≥ 21 / surface ≤ 12 (~30/40/30); lysozyme now 42/61/26.
*Dep:* E4 · *Source:* S0.1

Protein-only 1LYZ. Core contacts must show a higher minimally-frustrated fraction than surface
contacts. The prototype produced 48% core vs 38% surface at 50 decoys.

*Accept:* core fraction > surface fraction with a bootstrap CI on the difference excluding zero.

---

**F2 — Protein–protein interface fraction** ✅

> ✅ **DONE 2026-08-11.** **SKIP — S0.2 remains unmeasured.** No multi-chain structure exists under `data/` (all 61 processed PDBs are single-chain by Stage-4 construction; `data/raw_pdb/` holds only 1LYZ). Cases **pinned before running** and asserted by a test so they cannot be re-picked: 1BRS A/D, 1AY7 A/B, 1JTG A/B. Fully implemented; runs the moment those files exist.
*Dep:* E4, B7 · *Source:* S0.2

`pocket.mode: chain_interface`, `selector = inter_chain`, no ligand. The published atomistic value
is 14.2% minimally frustrated interface contacts, tolerance ±3 pp.

This is the only ligand-independent numerical gate available, and it exercises the multi-chain
sequence-separation fix (R02) that the single-chain prototype could never test.

*Accept:* a fixed set of ≥3 interface test cases with the computed fraction and its CI recorded.
Pick the cases and pin them before running, not after.

---

**F3 — Per-structure reference-count reproduction** ✅

> ✅ **DONE 2026-08-11.** PASS at smoke scale (3 structures × 10 decoys). **Ligand-incident contacts 22 / 27 / 37 where the prototype had 266–407** — A4's diagnosis reproduces and the object is the right kind. Computed minimal counts 1 / 2 / 11 vs the paper's 4 / 16 / 21: range overlaps 4–23 but runs systematically low. Counts move with decoy count (σ rescales), which is why no full-set number is claimed. Full reproduction ≈ **82 core-hours**.
*Dep:* E4, A2 · *Source:* §2.2

Compare computed minimally-frustrated counts against
`config/pdb_reference_table.csv` across the 19 completed structures. Current state: paper 4–23 vs
ours 266–407, r = 0.163, p = 0.51.

*Accept:* count range overlapping 4–23 and Pearson r versus the paper's counts, both recorded per
configuration. This is the most decisive available check and it needs no new data.

---

**F4 — Apo control** ✅

> ✅ **DONE 2026-08-11.** PASS. Pocket Pearson r = 0.990447; `E_native` max |Δ| = 0.000e+00 over 1772 contacts. The ligand-blindness finding is now a permanent regression test.
*Dep:* E4, A3 · *Source:* §2.1

Promote A3 into a permanent case: holo vs apo per-contact index correlation on a pinned structure.

*Accept:* pocket-contact correlation recorded with a threshold that fails the case if the index is
ligand-blind. The pass threshold is set from A3's measured value, not guessed.

---

**F5 — Per-contact energies vs a direct PyRosetta reference** ✅

> ✅ **DONE 2026-08-11.** PASS. **Two routes, unequally independent, and labelled as such**: route A re-enters the same energy methods (residual 5.9e-08 = float32 rounding, so it tests the lookup layer only); route B rebuilds `fa_atr`/`fa_rep`/`fa_sol` atom-pair by atom-pair and agrees to **1.3e-15**, covering 74 % of |e_direct|. Also found the plan's edgeless-pair check is **vacuous over the superset** (all 87 472 pairs have an edge), so it samples the most distant Cα pairs to exercise that branch.
*Dep:* B8 · *Source:* S1.2

Independent reimplementation of the two-body sum via a second route (residue-pair energies computed
directly rather than through `EnergyGraph`), compared on ≥20 complexes at 1% tolerance.

*Accept:* max relative deviation < 1% across all pairs with an edge; pairs without an edge asserted
to be exactly 0.0 in both.

---

**F6 — Pocket-restricted repacking equivalence** ✅

> ✅ **DONE 2026-08-11.** **Records ρ = 0.823 (8 Å) and 0.857 (12 Å) — both below the 0.95 bar.** Reported as a finding about the shell approximation, not tuned away. Speed-up 1.14× / 1.08×, **independently confirming C3**: pocket-restricted repacking buys essentially nothing, contra S1.3's assumed order of magnitude.
*Dep:* C7, C4 · *Source:* S1.3

Whole-protein repack vs shell-restricted repack on the same structures, same seeds, paired by
`decoy_id` — which is only meaningful because C4 makes identity draws independent of mutate-set
size. Report Spearman ρ on per-contact indices and the wall-clock speed-up.

*Accept:* ρ and speed-up recorded per shell radius. A ρ below 0.95 is a legitimate finding about
the shell approximation, not a test failure to be tuned away.

---

### Stage G — Ligand side

Everything here depends on Stage B's node model and Stage C's generator protocol.

---

**G1 — `chem/paramize.py`: parametrisation at library scale** ✅

> ✅ **DONE 2026-08-11.** 31 tests. Closed failure taxonomy; cache keyed on `inchikey|protonation|conformer`; deterministic letter-first code allocation generalising `634 → Z34`. **Sharp edge pinned**: `allocate_many` is order-independent, repeated `allocate` is not (42/1000 disagree on the collision path) — cross-run stability is the cache's job.
*Dep:* B6 · *Touches:* `atomfrust/chem/paramize.py`, `cache.py`, `codes.py`

`paramize(mol) -> ParamRecord`, cache key `InChIKey|protonation_version|conformer_version`.
Pipeline: RDKit sanitise → conformer (ETKDGv3 + MMFF) → SDF → the existing vendored
`molfile_to_params` invocation, preserving its `PYTHONPATH=src` / `cwd=params_dir` requirements.

Two deltas from `scripts/05_prepare_ligand.py`: the atom-name-match gate (`scripts/05:111-145`) is a
**hard gate for crystal ligands** and is never applied to library molecules, which have no PDB
HETATM names to match; and a **code allocator** assigns 3-character Rosetta codes deterministically
from sorted InChIKeys out of a whitelist probed once against the chemical manager — generalising the
ad-hoc `634 → Z34` fix into a rule.

Failures land in `chem/failures.parquet` under a closed taxonomy: `SANITIZE_FAIL`,
`UNSUPPORTED_ELEMENT`, `CHARGE_UNBALANCED`, `CONFORMER_FAIL`, `ATOMTYPE_UNASSIGNED`, `PARAMS_EMPTY`,
`ROSETTA_LOAD_FAIL`, `ROSETTA_SCORE_NONFINITE`. R17's "all failures categorised" becomes a groupby.

*Accept:* the 51 current EGFR ligands re-parametrise through the new path and load into a pose; a
deliberately broken molecule lands in the taxonomy rather than raising; cache hit on second call.

---

**G2 — `chem/protonation.py`: protonation and tautomers** ✅

> ✅ **DONE 2026-08-11.** 66 tests. `dimorphite_dl` is **absent**, so the RDKit fallback is live and every result reports `method="rdkit_tautomer"` — never `"dimorphite"`. Known limitation pinned by test: sites treated independently, so imatinib over-ionises to +2 where the truth is +1 — the argument for reporting the band, not a point estimate.
*Dep:* G1 · *Touches:* `atomfrust/chem/protonation.py`

R19: enumerate protonation and tautomer states at pH 7.4 (Dimorphite-DL plus RDKit tautomer
canonicalisation), with the chosen state recorded in `components.yaml` and folded into the
`regeneration_key` — a different protonation state is a different molecule and must not silently
reuse a cached analysis.

R20: `--enumerate-states` runs the top-K states as separate systems so downstream results carry an
uncertainty band rather than a point estimate.

*Accept:* ≥20 ligands enumerated; the state count and chosen state recorded per ligand; two states
of the same ligand produce different `regeneration_key`s.

---

**G3 — `chem/libraries/`: external decoy adapters** ✅

> ✅ **DONE 2026-08-11.** 38 tests, fully offline. `role` distinguishes synthetic `property_decoy` from **measured** `measured_inactive`, which is what keeps S3.4's experimental-negative claim from being circular. DOE over DUD-E's own six matching properties. Fetch script in git; caches gitignored.
*Dep:* G1 · *Touches:* `atomfrust/chem/libraries/*.py`

R16 and U3. `DecoyLibraryAdapter` protocol with `DUDEAdapter` (`.ism` SMILES + `decoys_final`
conformers — conformers, not poses), `DEKOIS2Adapter`, `MUVAdapter`, `ZINCRandomAdapter` (the
unmatched negative control), `LocalSDFAdapter`. DeepCoy's **published** DUD-E/DEKOIS decoy sets are
consumed through `LocalSDFAdapter`; the DeepCoy *model* is not a dependency.

Licences bar vendoring: fetch scripts go in git, caches are user-local and gitignored.
`MolRecord` carries `smiles, inchikey, source, source_id, role ∈ {active, property_decoy,
measured_inactive}, has_3d`. The `role` field is what keeps measured non-binders distinguishable
from synthetic decoys downstream.

*Accept:* each adapter yields `MolRecord`s from a small vendored fixture; `has_3d` is honest;
provenance (`source`, `source_id`) survives into `decoys/index.parquet` as `source_ref`.

---

**G4 — `dock/`: pose backends and the PoseBusters gate** ✅

> ✅ **DONE 2026-08-11.** 41 tests. **`smina`, `gnina` and `posebusters` are all absent**; `mcs_align` and `preposed` are live and every row reports `checker="builtin_subset"` — the 8-check subset never masquerades as the real tool, and what it omits is documented.
*Dep:* G1 · *Touches:* `atomfrust/dock/*.py`

Backends are **subprocess-only and probed at runtime** via `available()` — never imported, so a
missing binary degrades one backend rather than breaking the package: `SminaBackend`,
`GninaBackend`, `MCSAlignBackend` (RDKit MCS onto the native ligand — both the zero-dependency
fallback and the docking-free ablation that reveals whether the chemotype axis measures chemotype or
docking), `PrePosedBackend`.

R23: every pose passes PoseBusters before entering analysis; `pose_qc.parquet` records one row per
pose per check.

*Accept:* `MCSAlignBackend` works with only RDKit installed; a deliberately clashing pose is
rejected by the gate and recorded with the failing check named.

---

**G5 — `decoys/pose.py`: the pose axis** ✅

> ✅ **DONE 2026-08-11.** 12 tests. Moves the ligand through the **fold-tree jump**, so internal geometry is rigid by construction, and freezes it during relaxation so the gated pose is the scored pose. Distant protein–protein pairs move by **exactly zero** — the axis is strictly local. **Cross-cutting finding**: the native 5GMP pose *fails* the validity gate because its CYS797–SG covalent bond at 1.81 Å reads as a clash, so every covalent system's pass rate is floored at zero. Neither G4 nor G7 could see this alone.
*Dep:* C1, G4 · *Touches:* `atomfrust/decoys/pose.py`

U2, part of R15. Re-dock the same ligand, keep poses with RMSD-to-native ≥ 2.0 Å, gate on
PoseBusters, score under the identical protocol. Per-contact keys are well defined because the
molecule is unchanged — this is the *easy* ligand axis and should land before the chemotype axis.

*Accept:* native pose ranks above docked decoys, reported as an AUROC with CI.

---

**G6 — `decoys/chemotype.py`: the chemotype axis** ✅

> ✅ **DONE 2026-08-11.** 21 tests. Estimand is the **ligand-node scalar**, not a per-contact Z — a residue-anchored construction over changing molecules yields a mixture whose F degenerates into contact-probability × size. **The positive-control gate FAILS on real data**: native F62 scores −446 REU against members at −722 / −671 / −352, AUROC 0.333 (0.000 residualised). The gate correctly refuses to emit any redundancy number. Three hand-written fixtures placed by MCS with no pose search is not evidence the axis is broken — nor that it works. **The axis is implemented and instrumented; its central claim is untested** and needs a real property-matched library.
*Dep:* C1, G3, G4, G5 · *Touches:* `atomfrust/decoys/chemotype.py`

U2/U3, the remainder of R15, and the conceptually hardest step in the plan.

**The estimand is a ligand-node scalar, not a per-contact Z.** A residue-anchored per-contact
construction over a changing molecule produces a mixture distribution: a point mass at zero (the
decoy molecule never reaches residue *i*, so `pairwise_energy` returns 0.0 with no edge,
`frustration.py:205-207`) plus a continuous part whose scale grows with heavy-atom count. Mean and σ
of that mixture are not interpretable, and F becomes a monotone function of
"contact probability × molecular size". Therefore:

- **Primary quantity:** one scalar per member — the total many-body ligand–site energy over the
  **frozen native pocket shell** — scored with `rank_percentile`, with member MW, heavy-atom count,
  logP, formal charge and rotatable-bond count recorded as covariates and the energy-on-HAC
  regression slope reported and removed.
- **Per-residue decomposition is descriptive only**, restricted to keys with ≥ 80% member occupancy.
- Correspondence rule: anchor on the **protein residue index**, freeze the shell to the native
  pocket, keep zero-energy members (they are data, not missingness), and write
  `n_contacting_members` per key. *Rejected:* recomputing the shell per decoy (the estimand would
  vary per sample); atom-level or pharmacophore correspondence (undefined across topologically
  distinct molecules, which is the axis's entire point).
- **Positive control gate.** Before any cross-axis redundancy result is reportable, the axis must
  show that the native molecule ranks high within its own ensemble (AUROC ≥ 0.75, G5-style).
  Without this gate, a near-degenerate axis-D score would be trivially uncorrelated with axis A and
  the redundancy test could pass by noise.

*Accept:* the positive control passes on ≥1 system before any redundancy number is emitted;
covariate table present for every member; the MCS-align-only ablation reported alongside the docked
result.

---

**G7 — Covalent anchor handling** ✅

> ✅ **DONE 2026-08-11.** 22 tests. Scope held: covalent systems become **identifiable and constrained**, not fully modelled. **15 of 61** flagged, all CYS797–SG. Independently confirmed CLAUDE.md's CONNECT claim: zero of 112 params files contain one while `scripts/05:20-21` says the script adds them. Reported three cross-file changes rather than making them; two were then fixed.
*Dep:* B7, G1 · *Touches:* `atomfrust/spec.py`, `graph.py`, `decoys/identity.py`

R22, partial. Record the `_struct_conn` linkage already extracted by
`scripts/04_prepare_complex.py:100-127` into the spec, set `mutable=false` and
`frozen_reason="covalent_anchor"` on the anchor residue, force the anchor edge with
`is_bonded=true`, exempt it from distance and sequence-separation filters, and report affected
complexes as a separate stratum.

Full covalent chemistry — a real Rosetta bond, a patched residue type, correct valence — is
deferred (§7). This step makes the 15 EGFR covalent complexes *identifiable and excludable* rather
than silently mis-scored, which is the actual current failure mode.

*Accept:* the 15 flagged EGFR complexes carry `is_covalent` through to the analysis output; the
anchor residue is never mutated; a covalent stratum appears in `report`.

---

**G8 — Binding-site mutation control** ✅

> ✅ **DONE 2026-08-11.** 38 tests. **S4.5 demonstrated**: A:790 MET→TRP (4.9 Å from ligand) shifts pocket energies by 21.41 REU over 29 contacts; A:929 LYS→TRP (31.1 Å) by **exactly 0.000** over 0 contacts — same substitution, so the comparison isolates *where*. Regeneration key verified to change via the spec digest while `receptor_sha256` is identical.
*Dep:* E4, B9 · *Touches:* `atomfrust/spec.py`, `cli/prepare.py`

The methods document's S4.5 asks for pocket mutations as a *positive* test — a physically grounded
measure must respond when the pocket is mutated. Mechanically this needs only one thing the code
does not have: the ability to apply a point mutation to the receptor named in the spec and re-run.

Add `receptor.mutations: [{chain: A, resseq: 790, to: MET}]` to the spec, applied at pose-load time
and folded into `regeneration_key`.

*Accept:* a mutated spec produces a different `regeneration_key` and a different pose sequence at
the named position; wild-type and mutant runs are pairable by `system_id` for a paired test.

---

## 7. Deferred and rejected

**Deferred** — buildable later on this architecture, not needed for the corrections above:

| Item | Why deferred |
|---|---|
| Full covalent chemistry (bonded residue types, patches) | G7 makes covalent complexes identifiable, which is what unblocks the science; correct bonding is a large Rosetta chemistry task |
| Site axis (same ligand at other pockets) | needs pocket detection, an independent subsystem; axes A/B/D cover the requirement's core |
| OpenFF/SMIRNOFF as the primary parametriser (R18) | the vendored `molfile_to_params` path is proven on 51 ligands; OpenFF enters as a second backend behind the same `paramize()` interface once G1 is stable |
| Waters as graph nodes | the document names Rosetta's water treatment as a known limitation to be *reported*, not fixed |
| Co-folding backends (Boltz-2, AlphaFold3, Uni-Mol) | heavy dependencies, and the pose axis is served by G4's docking backends |
| Voronoi / contact-area contact definitions | three definitions already exercise the pluggable interface |

**Rejected** outright:

| Item | Why |
|---|---|
| Storing `E_ij` in the decoy record | destroys post-hoc re-specification of the many-body mode, the contact definition, the shell and `exclude_fa_rep` — the single most consequential schema decision in §4 |
| A general-purpose region DSL with a parser generator | six named selectors plus an `xml:` escape hatch cover every stated need |
| Migrating the 19 legacy parquets into the new schema | they lack `pair_id`, per-decoy energies, and provenance; keep them readable, re-run under the new engine when the anchors pass |
| Cluster/SLURM job scaffolding | `--shard k/M` makes any scheduler a one-line wrapper |
| Benjamini–Hochberg as the primary multiplicity control | wrong instrument for a maximum over a swept grid (§2.3) |

---

## 8. Dependency-ordered sequence

```
A1 ─┬─ A3 ── F4
    ├─ A4 ─────────────── B8 ─┬─ D8
    └─ A2 ─────────────── F3  └─ F5
B1 ── B2 ─┬─ B3 ── B6 ─┬─ B7 ─┬─ B8 (above)
          ├─ B4 ── B5   │      ├─ B9 ─┬─ C7
          │             │      └─ G7  └─ G8
          │             └─ B10
          └─ D6 ── D7
B5 ─┬─ D1 ─┬─ D2 ── D3 ── D7
    │      ├─ D4
    │      └─ D5
    └─ C1 ── C2 ─┬─ C3 ─┬─ C4 ── F6
                 │      ├─ C5
                 │      ├─ C6
                 │      ├─ C7 ── F6
                 │      └─ C8
                 └─ (axes)
B6 ── G1 ─┬─ G2
          ├─ G3 ─┐
          └─ G4 ─┴─ G5 ── G6
E1..E7 follow their module deps; F1..F6 follow E4/E5.
```

**Critical path to a scientifically meaningful result:**
`A4 → B1 → B2 → B3 → B6 → B7 → B8 → B5 → C1 → C2 → D1 → D2 → D3 → E2 → E3 → F3`.

That path ends at the reference-count check, which is the cheapest decisive test available. Nothing
on the ligand side (Stage G) is on it.

**Two things worth doing immediately, in parallel with B1:** A2 and A3. Both run against artifacts
already on disk or one small job, both can change what B8's default should be, and neither depends
on a line of new package code.

---

## 9. Coverage matrices

### 9.1 Methods-document requirements → steps

| Step | Requirements delivered |
|---|---|
| A1–A4 | diagnostic; sets R04 |
| B5 | R14, R37 |
| B6 | R05, R21 |
| B7 | R01, R02, R05, R24 |
| B8 | R03, R04 |
| B9 | R12 (substrate) |
| B10 | — (infrastructure) |
| C1 | R03 (stored unsubtracted), R13 (isolated so it is swappable) |
| C2 | R08, R09, R13, R15 |
| C5 | R10 |
| C7 | R12, R39 |
| C8 | R11 |
| D1 | R26, R27 |
| D2 | R25 |
| D3 | R24, R28 |
| D4 | R29 |
| D5 | R30, R31 |
| D6 | R32, R33, R34, R35, R36 |
| D8 | R07 |
| E7 | R38 |
| F5 | R06 |
| F6 | R12 (validated) |
| G1 | R17 |
| G2 | R19, R20 |
| G3 | R16 |
| G4 | R23 |
| G5, G6 | R15 |
| G7 | R22 (partial) |
| E/F stages | R40 |

**Not delivered by any step:** R18 (OpenFF as primary parametriser) — deferred, §7.

### 9.2 User's eight requests → steps

| # | Request | Steps | Where it is visible |
|---|---|---|---|
| U1 | Heteroligand as a residue/node | B6, B7, B8 | `nodes.parquet` `kind`, ligand-incident rows in `pairs.parquet` |
| U2 | Decoy generation for heteroligands | G5, G6 | `--axis pose`, `--axis chemotype` |
| U3 | Decoys from external datasets | G3, G4 | `--library dude --library-target pgh1` |
| U4 | Region-focused decoy generation | B9, C7 | `--mutate-sel`, `--repack-sel`, `--repack-shell` |
| U5 | Flexible contact/shell definitions | B7, D3 | `--contact-def`, `--contact-cutoff-A`, `--shell-ref`, `--shell-A` |
| U6 | Generate decoys only | E2 | `generate-decoys` produces no index by construction |
| U7 | Analyse an existing output folder | B5, E3 | `analyze --run-dir R` with zero PyRosetta calls |
| U8 | Custom PDB input | B3, E1 | `run --pdb my.pdb --ligand B:501` |

---

## 10. Open decisions

Each blocks the step named. These are judgement calls, not missing information.

| # | Decision | Blocks | Default if unanswered |
|---|---|---|---|
| ~~Q1~~ | ~~Which many-body formula is the published object?~~ | — | ✅ **ANSWERED by A4.** Eq. 2 *does* exclude `j`; `chen_literal` is the published object and is required for reproduction claims. `pair_retained` is a deliberate improvement. |
| Q2 | Is reference-count reproduction (F3) a hard gate or a diagnostic? | F3 | **Re-scoped by A4:** F3 must count *ligand–residue* contacts, not protein–protein pairs in a shell. It only becomes testable after B6/B7. Recommend hard gate once meaningful. |
| ~~Q3~~ | ~~Which reading of "locations" is authoritative?~~ | — | ✅ **ANSWERED by A4.** *"we randomly shuffle the protein sequence"* — a permutation over the whole chain: `scope=whole_protein, identity=native, placement=permute`. Our i.i.d. composition draw is the deviation. |
| ~~Q4~~ | ~~Symmetric native repack on or off by default?~~ | — | ✅ **ANSWERED by A4.** *"obtained in a similar fashion by omitting the shuffling step"* — the native **is** repacked and MC-relaxed. Default `true`. |
| Q5 | Which single configuration is pre-registered as primary before any grid is swept? | D6, D7 | none — and without one, §2.3's max-T correction is the only defensible reporting mode |
| Q6 | Do the 19 completed structures get re-run once B8 lands? | — | yes if Q1 changes the formula; they are not comparable across formulas |

**Post-Stage-A: the remaining open questions are Q2, Q5 and Q6.** Q1, Q3 and Q4 are settled
against the source paper.

Q6 ("do the completed structures get re-run?") now has a clear answer: **they are not a
reproduction of the paper under any treatment**, because they count protein–protein pairs where
the paper counts ligand–residue contacts. They stay valuable as a bit-for-bit regression target
for the new engine (steps C2 and D3 depend on that), but no correlation computed from them is a
reproduction claim. Re-running is not a repair of the old numbers; it is a different calculation
that only becomes possible after B6/B7.

The one genuinely new question A4 raises: **which many-body formula is the scientific object going
forward.** `chen_literal` is what the paper printed and is mandatory for any claim of reproducing
r = 0.45. It is also degenerate. `pair_retained` is non-degenerate but is not what was published.
Both should be computed and reported; the choice of which leads is a publication decision, and it
needs stating explicitly rather than silently.
