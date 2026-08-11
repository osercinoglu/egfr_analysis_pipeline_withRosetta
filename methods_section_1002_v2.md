# 3. Materials and Methods

*TÜBİTAK 1002 — Hızlı Destek*
**Project:** Ligand-aware atomistic local frustration as a target-agnostic specificity measure for structure-based virtual screening
**PI:** Onur Serçinoğlu · Computational Structural Biology, Gebze Technical University
**Version:** 2 (reconstructed)

---

## 3.0 Validated parameters

Every quantitative value used in this Methods section, with its source. Values marked *(derived)* follow from standard statistics and are shown with their derivation. Values marked *(target)* are success thresholds set by this project, justified against the literature value in the adjacent row.

| # | Parameter | Value | Source |
|---|---|---|---|
| 1 | EGFR frustration–affinity correlation (published) | Pearson r = 0.45 | Chen et al. 2020, Fig. 5e; count of minimally frustrated interactions vs log₂ affinity (pM) |
| 2 | EGFR reproduction threshold | r ≥ 0.35 *(target)* | Set below #1 to allow implementation variance; materially lower indicates defect |
| 3 | EGFR improvement threshold | r > 0.45 *(target)* | Must exceed #1 to justify a new descriptor |
| 4 | COX inhibitor set | 54 compounds (35 COX-2-selective, 19 non-selective) | Chen et al. 2020 |
| 5 | COX-2 vs COX-1 selectivity effect | +3.5 minimally frustrated interactions (selective class mean) | Chen et al. 2020 |
| 6 | Published decoy count, atomistic frustration | 1000 decoys per contact | Chen et al. 2020 |
| 7 | Atomistic decoy construction | Randomise residue identities **and** locations of contacting residues | Chen et al. 2019/2020 |
| 8 | Atomistic interface minimal frustration | 14.2% (vs 26.0% for AWSEM) | Chen et al. 2020, Suppl. Fig. 8 — quantifies the Rosetta water limitation |
| 9 | Atomistic vs AWSEM agreement | Poor correlation; disjoint non-neutral residue sets | Chen et al. 2019 commentary — coarse-grained cannot validate atomistic |
| 10 | Published atomistic compute cost | Hundreds of core-hours (2019 algorithm) | Chen et al. 2019 supplementary |
| 11 | DUD-E composition | 102 targets; 22,886 clustered actives; 50 decoys per active | Mysinger et al. 2012 |
| 12 | DUD-E target classes | 26 kinases, 15 proteases, 11 nuclear receptors, 5 GPCRs, 2 ion channels, 2 CYP450s, 36 other enzymes, 5 miscellaneous | Mysinger et al. 2012, Fig. 1 |
| 13 | DUD-E clustered actives per target | mean 224 | Mysinger et al. 2012 |
| 14 | **DUD-E experimental decoys** | **9,219 compounds with no measurable affinity to 30 μM; max 1,070 for COX-1 (PGH1)** | Mysinger et al. 2012 — the experimental-negative resource used in WP3 |
| 15 | DUD-E decoy dissimilarity rule | ECFP4; most dissimilar 25% retained | Mysinger et al. 2012 |
| 16 | Bias-reduced DUD-E subset | 47 targets remain after bias removal | Chaput et al. 2016 |
| 17 | Program collapse on bias removal | Glide 30→5, Gold 27→4, FlexX 14→2, Surflex 11→2 targets (BEDROC α = 80.5 > 0.5) | Chaput et al. 2016 |
| 18 | DeepCoy property matching | DOE 0.166 → 0.032 (DUD-E, −81%); 0.109 → 0.038 (DEKOIS 2.0, −66%) | Imrie et al. 2021 |
| 19 | DeepCoy docking hardness | Vina screening AUROC 0.70 → 0.63 | Imrie et al. 2021 |
| 20 | DEKOIS 2.0 composition | 81 targets; 40 actives and 1200 decoys each | Bauer et al. 2013 |
| 21 | MUV composition | 17 targets; ~30 actives and ~15,000 *experimental* inactives each | Rohrer & Baumann 2009 |
| 22 | Uni-Mol Docking V2 pose accuracy | 77%+ < 2.0 Å RMSD; 75%+ pass all checks (up from 62%) | Alcaide et al. 2024, PoseBusters set |
| 23 | Boltz-2 affinity accuracy | Pearson 0.62 on held-out FEP+/OpenFE benchmark; ~1000× faster than FEP | Passaro et al. 2025 |
| 24 | Boltz-2 run-to-run variability | up to ~1.5 kcal/mol | Independent benchmark compilation, 2026 |
| 25 | Co-folding novelty collapse | 75–88% pose accuracy (familiar) → < 25% (unfamiliar targets/chemistry) | Runs N' Poses benchmark |
| 26 | Kinase polypharmacology | 243 clinical inhibitors bind 220 of 518 human kinases | Klaeger et al. 2017 |
| 27 | Z-score σ error at N decoys | σ/√(2(N−1)): 10.1% (N=50), 4.5% (N=250), 2.2% (N=1000) *(derived)* | Standard error of sample SD |
| 28 | Convergence criterion | Spearman ρ ≥ 0.95 vs N = 2000 reference *(target)* | Set so residual ranking error is below the effect sizes in #1 and #5 |

---

## 3.1 Design logic and the question being tested

**The bottleneck this project addresses.** Structure-based virtual screening does not fail primarily from inadequate sampling or inaccurate energy functions. It fails because every method in the stack computes a monotone function of *how favourable* an interaction is, and the field then uses that number to answer *whether a molecule is specifically favoured* at a site. These are different questions. The evidence is consistent across three generations of methods: removing biased targets from DUD-E collapses four commercial docking programs from 30/27/14/11 to 5/4/2/2 successes (#16, #17); machine-learning docking methods produce physically invalid poses and fail on novel sequences; and co-folding models retain enrichment when binding-site residues are mutated, indicating recall rather than recognition, with pose accuracy falling from 75–88% to below 25% on unfamiliar targets (#25). A ligand-property signature, a memorised look-alike complex, and genuine molecular recognition all yield a good score, and nothing in the stack separates them.

**Why local frustration is the right instrument.** A frustration index is not an affinity proxy. It is a Z-score of a native interaction energy against a distribution of decoy energies computed *within the same complex* — structurally, a comparison against alternatives, which is what specificity means. It has three properties no comparator possesses simultaneously: it contains **no fitted parameters**, so it cannot memorise; it is **internally referenced**, so its scale is not target-dependent by construction; and it yields a **per-contact map** rather than a scalar. Atomic resolution is required rather than optional, because coarse-grained AWSEM is parametrised for the twenty canonical amino acids; the original authors state explicitly that the major advantage of atomistic models is that they permit explicit incorporation of ligands and cofactors of large molecular variety (#8).

**What is therefore tested.** The project's claim is *not* that frustration achieves higher absolute accuracy than Boltz-2 or Glide on familiar targets. It almost certainly will not, and does not need to. The claim is that a parameter-free, internally-referenced specificity measure **degrades less along a novelty gradient** than methods that can memorise, and that this property makes it useful precisely where the current stack fails. The primary endpoint is consequently a *slope*, not a level (§3.9).

**Logical chain.** (i) Establish that the calculation is correct, by reproducing a published number (#1). (ii) Characterise the statistical behaviour of the index as a function of its decoy ensemble, since the decoy definition *is* the method. (iii) Test whether the novel ligand-identity decoy axis adds independent information. (iv) Replace experimental structures with predicted ones and test signal retention. (v) Test transfer along the novelty gradient against comparators. Steps (ii) and (iii) contain the experiments capable of falsifying the hypothesis and are scheduled in the first half.

---

## 3.2 Data and target sets

**Tier 1 — Validation anchors.**
- *EGFR:* EGFR kinase–inhibitor complexes from the PDB with measured affinities from PDBbind/BindingDB/ChEMBL (n ≈ 30–40; PDB 5GMP included), reproducing the set type used for #1.
- *COX-1/COX-2:* the 54 inhibitors of #4, each computed against both isoforms.
- *COX-1 experimental negatives:* the 1,070 compounds with no measurable affinity to 30 μM reported for PGH1 in DUD-E (#14). This is the largest experimental-negative set in DUD-E and it belongs to a Tier-1 anchor, allowing the specificity claim to be tested against **measured** non-binders rather than synthetic decoys.

**Tier 2 — Kinase block.** The 26 DUD-E kinases (#12), annotated with conformational state and inhibitor type from KLIFS and KinCoRe, with selectivity labels from Davis 2011, Karaman 2008 and Klaeger 2017. Off-diagonal cross-docking pairs are labelled from measured profiling data and excluded where no evidence exists, because kinase polypharmacology is pervasive (#26).

**Tier 3 — Cross-family screening set.** 30 DUD-E targets spanning ≥5 of the eight classes in #12, maximising overlap with the 47-target bias-reduced subset (#16). Per target: 50 actives and 1,500 decoys (30:1 subsample), giving 46,500 systems. Thirty targets is the floor for leave-family-out to be non-anecdotal; the restriction from 102 is a compute decision, stated as such.

**Tier 4 — Novelty-gradient set.** PLINDER systems stratified by its published split axes — novel ligand, novel pocket, novel protein, and all-novel — providing the ordered difficulty axis on which the primary endpoint is measured.

**Tier 5 — External hold-out.** DEKOIS 2.0 (#20), independent decoy construction, used **once**, at the end of WP5. MUV (#21) supplies a second experimental-negative test.

---

## 3.3 WP0 — Foundation and re-implementation (Months 1–2)

**Rationale.** A survey of available software establishes that **no public atomistic, Rosetta-based frustratometer exists**: the Frustratometer 2 server, frustratometeR, and the available Python implementations are all AWSEM- or DCA-based coarse-grained tools. Re-implementation is therefore the expected path. This raises WP1's cost but is also the reason the resulting tool is a genuine community contribution.

Critically, coarse-grained output **cannot** be used to validate the atomistic implementation, because the two show poor correlation and identify partly disjoint sets of non-neutral residues (#9). Validation must therefore be against the published atomistic numbers (#1, #5) directly.

**Methods.** Confirm PyRosetta licensing and REF2015 availability. Implement the published algorithm: decoy ensembles generated by randomising both residue identities and the locations of contacting residues (#7); side-chain repacking without backbone perturbation; short Monte-Carlo relaxation; contacts by Cα–Cα distance at 10 Å; separation of the harsh repulsive Lennard-Jones term so that steric clashes do not inflate decoy variance. Build the shared evaluation harness: fixed splits, one implementation of every metric, bootstrap intervals, structured logging enabling regeneration of any reported number from a recorded configuration.

**Success criteria.**
- **S0.1** Atomistic frustration runs end-to-end on a reference protein and reproduces the published qualitative signature (highly frustrated contacts at the surface, minimally frustrated in the core).
- **S0.2** On a protein–protein interface test case, the computed fraction of minimally frustrated interface contacts falls within ±3 percentage points of the published 14.2% (#8). *This is a direct numerical reproduction test of the atomistic implementation, independent of any ligand.*
- **S0.3** Harness returns identical metric values for a fixed input in both students' environments.

**Gate.** WP1 does not proceed until S0.1 and S0.2 pass.

---

## 3.4 WP1 — Ligand-aware atomistic energy and parametrisation (Months 1–5; Thesis 1)

**Rationale.** Extending frustration to protein–ligand complexes requires the ligand to enter the energy function, which coarse-grained models cannot do. The bottleneck that has kept such analyses bespoke is ligand parametrisation; automating it is what converts a published demonstration into a general tool.

**Methods.** Automated parametrisation via the Open Force Field toolkit (SMIRNOFF typing) with GAFF2 fallback and AM1-BCC charges; protonation and tautomer enumeration at pH 7.4. Ligands introduced as full-atom Rosetta residue types; contact energies computed under REF2015 with many-body effects included by summing all interaction energies involving either partner, per the reference formulation.

Two efficiency decisions, each validated rather than assumed, address the hundreds-of-core-hours cost of the original approach (#10): repacking restricted to a shell around the pocket, and per-decoy energy caching so that one randomisation yields energies for all contacts simultaneously.

**Success criteria.**
- **S1.1** ≥95% of a 500-ligand stress set (PDBbind + DUD-E actives; spanning charge states, macrocycles, halogens, common metals) parametrises without manual intervention; all failures categorised.
- **S1.2** Per-contact energies agree with a direct PyRosetta reference to within 1% on 20 complexes.
- **S1.3** Pocket-restricted repacking reproduces whole-protein repacking frustration indices at Spearman ρ ≥ 0.95 on those 20 complexes, with the speed-up factor reported.
- **S1.4** Protonation/tautomer sensitivity quantified across ≥20 ligands and carried as an uncertainty band on all downstream results.
- **S1.5** Wall-clock cost per complex reported as a function of pocket size and decoy count, demonstrating ≥10× reduction relative to the naive implementation implied by #10.

---

## 3.5 WP2 — Decoy framework and statistical characterisation (Months 3–9; Thesis 1)

**Rationale.** The index is a Z-score against a decoy ensemble, so the ensemble defines the quantity measured. Published practice is inconsistent and uncharacterised: 50 decoys per active in DUD-E (#11), 1,200 per target in DEKOIS 2.0 (#20), 1,000 per contact in the atomistic frustration method (#6). No convergence analysis exists anywhere in this literature. Supplying it is a self-contained contribution independent of the project's outcome.

**Decoy axes.** Four axes are implemented (reduced from five for feasibility; the fragment axis is deferred):

| Axis | Perturbation | Question | Status |
|---|---|---|---|
| A. Identity/location | Randomise residue identities and positions of contacting residues (#7) | Is this *pocket* unusually favourable? | Published baseline |
| B. Pose | Re-dock the same ligand | Is this *pose* unusually favourable? | Established (docked-ensemble Z-score) |
| C. Site | Same ligand at other pockets | Is this *site* unusually favourable? | Established variant |
| D. **Chemotype** | Replace ligand with property-matched, topologically distinct molecules | Is this *molecule* specifically favoured? | **Novel** |

Axis D is the contribution: existing formulations perturb protein or pose but never ligand identity, which is the variable a screen actually changes. It also carries an efficiency consequence — a screening library *is* a chemotype-decoy ensemble, so on Tier 3 the axis adds negligible marginal docking cost.

**Decoy generators.** Reduced to three from six, protecting the statistical work: the DUD-E protocol (#15) as field standard; DeepCoy as the strongest alternative, having improved DOE by 81% on DUD-E while making decoys genuinely harder (#18, #19); and an unmatched random ZINC sample as negative control establishing the ceiling attributable to trivial property discrimination.

**Optimization experiments.**
- **D1 — Convergence.** Descriptors at N ∈ {10, 25, 50, 100, 250, 500, 1000, 2000} on 20 fixed complexes; N\* is the smallest N reaching ρ ≥ 0.95 against N = 2000 (#28), with bootstrap intervals. *Rationale:* since the index divides by σ, and the standard error of σ is σ/√(2(N−1)), N = 50 propagates ~10% relative error into every value, falling to ~4.5% at N = 250 and ~2.2% at N = 1000 (#27).
- **D2 — Cost-adjusted optimum.** N\* reported against wall-clock cost per axis (identity/location decoys require repacking; chemotype decoys require docking), with a recommended operating point.
- **D3 — Distributional validity.** Normality tested per pocket (Shapiro–Wilk, Q–Q); a rank/percentile non-parametric index implemented and compared wherever violated.
- **D4 — Cross-target invariance (decisive).** σ measured across ≥3 architectural strata (burial, polarity, volume), ≥20 pockets each. A Z-score removes the first-order offset between targets but not systematic differences in σ; this experiment therefore tests the target-agnostic premise directly.
- **D5 — Axis redundancy (decisive).** Pairwise Z-score correlations across axes A–D. If chemotype correlates above 0.8 with identity/location, it adds no information.

**Aggregation.** Five per-complex descriptors compared: the published count of minimally frustrated contacts (#1's descriptor, treated as baseline); energy-weighted sum; mean per-contact Z; pocket-size-normalised variants; fraction of minimally frustrated ligand contacts.

**Success criteria.**
- **S2.1** All four axes implemented and independently invocable.
- **S2.2** N\* determined per axis with bootstrap CIs (D1, D2).
- **S2.3** Three generators benchmarked on a common set reporting DOE, ligand-only classifier AUROC, and docking-hardness; the DeepCoy-over-DUD-E ordering of #18–#19 reproduced within stated tolerance or the discrepancy explained.
- **S2.4** Normality assessed for ≥100 pockets; non-parametric index implemented and compared where violated.
- **S2.5** Coefficient of variation of σ across architectural strata reported as the quantitative statement of target-invariance.
- **S2.6** Chemotype-versus-identity/location Z-score correlation **< 0.8**.

**Contingency.** If S2.5 shows strong architecture dependence or S2.6 fails, the project reports the index as better-calibrated but target-dependent, and WP5 re-scopes from cross-target transfer to within-target rescoring. Both outcomes are publishable; establishing which holds is itself the contribution.

---

## 3.6 WP3 — Validation against published anchors and experimental negatives (Months 6–10; Thesis 1)

**Rationale.** Three separate tests are required and are not interchangeable: reproduce the published result; improve on it; and demonstrate that the measure separates actives from **experimentally measured** non-binders, not merely synthetic decoys. The third addresses the circularity objection that a computational specificity measure validated against computational negatives proves little.

**Methods.** On Tier-1 EGFR, frustration computed at published settings, minimally frustrated interaction count correlated against log₂ affinity (pM), compared directly with #1; alternative aggregations then evaluated on the identical set. On Tier-1 COX, all 54 inhibitors computed against both isoforms and the class difference compared with #5. On the COX-1 experimental-negative set (#14), actives are ranked against the 1,070 measured non-binders. Native-pose discrimination assessed on a redocking set.

**Success criteria.**
- **S3.1** *Reproduction:* Pearson r ≥ 0.35 on EGFR using the published protocol (#2); r ≈ 0.45 confirms faithful implementation. A materially lower value is an implementation defect to debug, **not** a reportable finding.
- **S3.2** *Improvement:* at least one continuous aggregation achieves r > 0.45 (#3) with a bootstrap CI on the improvement excluding zero.
- **S3.3** *Selectivity:* COX-2-selective compounds show a significantly greater gain in minimally frustrated interactions in COX-2 than non-selective compounds, reference effect +3.5 (#5), reported with effect size and 95% CI.
- **S3.4** *Experimental negatives:* AUROC ≥ 0.65 separating COX-1 actives from the 1,070 measured non-binders (#14) — a threshold set above chance and comparable to docking performance on unbiased sets (#19).
- **S3.5** *Pose discrimination:* native poses above docked decoys at AUROC ≥ 0.75.
- **S3.6** Open-source release with documentation, unit tests, and reproducible EGFR/COX examples.

---

## 3.7 WP4 — Structure generation without MD, comparators, and mutation control (Months 4–10; Thesis 2)

**Rationale.** The reference work depended on curated crystal structures, restricting it to systems that already have them. Docking and co-folding now generate complexes directly, which is what makes a frustration-based screen scalable and removes the MD requirement.

**Structure generation.** AutoDock Vina/smina (open baseline and pose generator); GNINA 1.3 (CNN-rescored, on the accuracy–cost Pareto frontier, reports DUD-E screening directly); Uni-Mol Docking V2 (strongest public ML docking, #22); co-folding with Boltz-2 and AlphaFold3, with Protenix-v1 (Apache 2.0) as open cross-check.

**Comparator scores.** Vina/smina score; GNINA CNN score; Boltz-2 affinity (#23); and — **critically — the raw Rosetta REF2015 interaction energy of the same complexes**. This last is the control that makes the central claim interpretable: comparing a Rosetta-derived Z-score against a Vina raw score confounds normalisation with energy-function quality. Only the raw-versus-Z-scored comparison *within the same energy function* isolates the effect of normalisation.

**Quality control.** Every pose passes PoseBusters before entering frustration analysis, since ML docking methods are documented to produce physically invalid poses. Co-folding receives two additional controls: multi-seed replication, because run-to-run variability of up to ~1.5 kcal/mol (#24) can exceed the signal; and a **binding-site mutation control**, applied as a *positive* test for frustration rather than only as a flag for comparators — a physically grounded measure must respond to pocket mutation, whereas memorising models do not.

**Success criteria.**
- **S4.1** All generators and comparators run end-to-end on Tiers 1–2 from identical prepared inputs.
- **S4.2** ≥90% of poses entering analysis pass all PoseBusters checks; generators below this are excluded or reported separately with the failure mode named.
- **S4.3** *Signal retention:* EGFR correlation from docked/co-folded input degrades by < 0.10 in Pearson r relative to crystal input. This licenses the MD-free claim.
- **S4.4** Run-to-run variability quantified over ≥5 seeds per generative method and propagated as error bars downstream.
- **S4.5** *Mutation sensitivity (positive test):* on ≥10 systems with pocket mutations known to abolish binding, the frustration index changes significantly (paired test, p < 0.05 after correction), while the change for at least one co-folding comparator is not significant. This is a direct, cheap demonstration that the measure reflects recognition rather than recall.

---

## 3.8 WP5 — Novelty-gradient transfer and normalisation test (Months 8–12; Thesis 2)

**Rationale.** Absolute accuracy on familiar targets is the wrong endpoint: Boltz-2 correlates strongly with measured potency on such systems (#23) and frustration is unlikely to beat it. The property unique to a parameter-free measure is that it should *degrade less* as target and chemical novelty increase — precisely where co-folding collapses from 75–88% to below 25% (#25). The primary endpoint is therefore the slope of performance across the novelty gradient.

**Methods.** Systems are stratified by PLINDER's novelty axes (novel ligand → novel pocket → novel protein → all novel). For each stratum, screening performance is computed for frustration Z-scores and for every comparator, and the degradation slope across strata is estimated with bootstrap intervals. Secondary evaluation on Tier 3 covers per-target enrichment, pooled cross-target ranking (all molecules ranked against one global threshold — the operation raw scores are known to fail), and leave-family-out partitioning. No parameter is fitted per target; where score fusion requires a weight, it is fitted only on families excluded from evaluation.

Selectivity is assessed on Tier 2 by cross-docking within the 26-kinase block, with off-diagonal labels taken from measured profiling data and unevidenced pairs excluded (#26).

**Success criteria.**
- **S5.1** *Primary endpoint — differential degradation:* across the PLINDER novelty strata, the frustration Z-score's performance slope is significantly shallower than that of at least one co-folding comparator (bootstrap CI on the slope difference excluding zero). Absolute performance may be lower; the slope is the claim.
- **S5.2** *Co-primary — normalisation isolated:* Rosetta Z-scored ranking exceeds **raw Rosetta** ranking in pooled cross-target AUROC by ≥ 0.05, CI excluding zero. Same energy function on both sides; this isolates normalisation from energy-function quality.
- **S5.3** *No regression:* frustration-augmented scoring is not worse than the corresponding raw score on per-target enrichment across Tier 3, within bootstrap error.
- **S5.4** *Bias control:* any Tier-3 improvement is reproduced on the bias-reduced component (#16); improvement present only on the full set is reported as a negative result, given #17.
- **S5.5** *Ligand-only control:* the protein-free equivalent does not achieve comparable performance.
- **S5.6** *Family transfer:* leave-family-out AUROC does not fall below 0.55 for any held-out family; per-family results reported individually.
- **S5.7** *Selectivity:* on the Tier-2 matrix, diagonal systems show significantly lower frustration than evidence-labelled off-diagonal systems, and frustration differences correlate with measured selectivity ratios at Spearman ρ ≥ 0.3.
- **S5.8** *External validation:* primary endpoint re-evaluated once on DEKOIS 2.0 (#20) and MUV (#21); the degradation relative to Tier 3 is reported as the generalisation gap.

---

## 3.9 Evaluation protocol and statistics

All metrics come from the single WP0 implementation. Screening performance is reported as AUROC, BEDROC (α = 80.5, matching #17), adjusted logAUC and EF1%, since early enrichment and global ranking diverge. Correlations are Pearson r for affinity relationships (matching #1) and Spearman ρ for ranking claims. All estimates carry bootstrap 95% CIs (10,000 resamples, stratified by target). Method comparisons use paired tests across targets, never pooled molecule-level tests, since molecules within a target are not independent. Benjamini–Hochberg correction is applied across the descriptor and axis families in WP2–WP3. Degradation slopes (S5.1) are estimated by weighted linear regression across ordered strata with target-level bootstrap. All negative results are reported.

---

## 3.10 Timeline, work distribution and feasibility

```
Month            1   2   3   4   5   6   7   8   9  10  11  12
WP0 Foundation   ████
WP1 Energy       ██████████████████
WP2 Decoys               ██████████████████████████
WP3 Validation                       ██████████████████
WP4 Structures           ██████████████████████████
WP5 Transfer                                 ██████████████████
```

**Distribution.** Thesis 1 (Tuğba Eke): WP1–WP3 — energy function, decoy framework and statistical characterisation, validation. Thesis 2 (scholarship student): WP4–WP5 — structure generation, comparators, mutation control, novelty-gradient transfer. WP0 shared. The single dependency falls at month 9, when WP2's selected axes and counts and WP3's validated descriptors define what Thesis 2 evaluates; Thesis 2 is not blocked before then because WP4 is independent from month 4.

**Feasibility.** The dominant cost is repacking per decoy. Four decisions bring the work within budget, three of them validated rather than assumed: one randomisation yields all contacts simultaneously; pocket-restricted repacking (S1.3) reduces cost by roughly an order of magnitude against the hundreds-of-core-hours baseline (#10); convergence-justified decoy counts (D1) replace the default 1,000 (#6) with an evidence-based N\*; and the screening library doubles as the chemotype-decoy ensemble, so axis D adds negligible marginal docking cost on Tier 3. Compute is staged: Tier 1 (~150 systems) in WP3, Tier 2 (26 targets) from month 8, Tier 3 (46,500 systems) from month 9, with the number of Tier-3 configurations bounded by the D2 operating points.

---

## 3.11 Risk management

| Risk | Detection | Response |
|---|---|---|
| Re-implementation larger than estimated | WP0, months 1–2 | Scope confirmed early; S0.2 gives an objective numerical pass/fail against #8 |
| Reproduction fails (S3.1) | Month 6–7 | Implementation defect; debug against published protocol, do not report as finding |
| Cannot improve on r = 0.45 (S3.2) | Month 8 | Lead with the transfer/normalisation claim; present the tool as means rather than headline |
| σ architecture-dependent (S2.5) | Month 6, by design | Re-scope WP5 to within-target rescoring; report the calibration finding as the contribution |
| Chemotype axis redundant (S2.6) | Month 7, by design | Novelty shifts to the tool, the statistical characterisation and the MD-free pipeline |
| Predicted structures degrade signal (S4.3) | Month 9 | Restrict scalable claims to docked poses of crystal receptors; report the co-folding limitation |
| Rosetta water treatment | Throughout | Quantified limitation (#8: 14.2% vs 26.0% at interfaces); stratify results by pocket polarity and report explicitly |
| Circularity of synthetic negatives | Addressed by design | Experimental negatives in S3.4 (#14) and S5.8 (#21) |
| Compute overrun | Monthly | Reduce Tier-3 configurations to D2 operating points; Tier-3 target count adjustable with 30 as the floor |
