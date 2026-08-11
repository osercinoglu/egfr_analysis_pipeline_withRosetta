#!/usr/bin/env python
"""
Step A3 of plans/frustratometer-ng-plan.md — the apo control.

Does the ligand affect the frustration index at all?

Under the current formulation the ligand never appears in a contact pair
(`get_protein_contacts`, src/frustration.py:70-108, skips non-protein residues). It
influences results only by being present in the pose during scoring, repacking and
minimisation, entering a pocket residue's B_i as one term among ~30. Step A1 established
that E_ij = 0.5*(B_i + B_j) exactly, which dilutes that contribution twice over.

This script deletes the ligand, re-runs the identical protocol at the identical seed, and
correlates per-contact F_index holo vs apo.

The comparison is properly paired by construction. `generate_decoy` draws a new identity
for each *protein* residue in pose order and skips non-protein residues, and
`native_aa_frequency` counts protein residues only. Holo and apo therefore consume the
same RNG stream and produce the *same decoy sequences* at a given seed. The only
difference between the two runs is whether the ligand is present while the decoy is
repacked, minimised and scored -- which is exactly the quantity of interest.

VERDICT THRESHOLD: if Pearson r > 0.95 on pocket contacts, the descriptor is effectively
ligand-blind under the current many-body formula.

Run from the repo root. This one DOES need PyRosetta and real cores:
    OMP_NUM_THREADS=1 nice -n 19 python analysis/apo_control.py --n-jobs 16
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

LIGAND_BLIND_R = 0.95


def build_apo_pdb(holo_pdb: Path, apo_pdb: Path) -> int:
    """Write a copy of holo_pdb with every HETATM line removed. Returns lines dropped."""
    apo_pdb.parent.mkdir(parents=True, exist_ok=True)
    dropped = 0
    with holo_pdb.open() as fin, apo_pdb.open("w") as fout:
        for line in fin:
            if line.startswith(("HETATM", "CONECT")):
                dropped += 1
                continue
            fout.write(line)
    return dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-id", default="5GMP")
    ap.add_argument("--ligand", default="F62", help="CCD id used in the holo parquet name")
    ap.add_argument("--rosetta-ligand", default=None, help="defaults to --ligand")
    ap.add_argument("--n-decoys", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--workdir", default="analysis/apo")
    ap.add_argument("--out", default="analysis/apo_control_report.md")
    ap.add_argument("--shell", default="heavy5", help="pocket shell used to split contacts")
    args = ap.parse_args()

    rosetta_lig = args.rosetta_ligand or args.ligand
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    holo_pdb = Path("data/processed") / f"{args.pdb_id}_clean.pdb"
    holo_parquet = Path("results") / f"{args.pdb_id}_{args.ligand}_frustration.parquet"
    params = Path("data/ligands/params") / f"{rosetta_lig}.params"
    for p in (holo_pdb, holo_parquet, params):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    apo_pdb = work / f"{args.pdb_id}_apo.pdb"
    dropped = build_apo_pdb(holo_pdb, apo_pdb)
    print(f"apo PDB written: {apo_pdb} ({dropped} HETATM/CONECT lines dropped)")

    # ---- pocket sets from the HOLO structure -----------------------------
    from diagnose_counts import SHELLS, load_structure, pocket_sets

    _, ca, heavy_by_res, lig = load_structure(holo_pdb)
    pockets = pocket_sets(ca, heavy_by_res, lig)
    if args.shell not in pockets:
        print(f"unknown shell {args.shell}; have {list(pockets)}", file=sys.stderr)
        return 2
    pocket = pockets[args.shell]
    print(f"ligand heavy atoms: {len(lig)}; pocket ({args.shell}): {len(pocket)} residues; "
          f"pocket (ca10): {len(pockets['ca10'])} residues")

    # ---- load apo pose and verify the contact set is identical -----------
    import pyrosetta
    from pyrosetta import Pose, Vector1

    import frustration as fr

    pyrosetta.init("-mute all")
    apo_pose = Pose()
    res_set = pyrosetta.generate_nonstandard_residue_set(apo_pose, Vector1([str(params)]))
    pyrosetta.pose_from_file(apo_pose, res_set, str(apo_pdb))
    print(f"apo pose loaded: {apo_pose.total_residue()} residues")

    contacts = fr.get_protein_contacts(apo_pose, cutoff=10.0, seq_sep_min=4)
    holo = pd.read_parquet(holo_parquet)
    holo_pairs = set(zip(holo.resi.tolist(), holo.resj.tolist()))
    if set(contacts) != holo_pairs:
        print(
            f"FATAL: apo contact set differs from holo "
            f"({len(contacts)} vs {len(holo_pairs)}, shared {len(set(contacts) & holo_pairs)}). "
            f"The comparison would not be paired.",
            file=sys.stderr,
        )
        return 3
    print(f"contact set verified identical to holo: {len(contacts)} pairs")

    # ---- run the survey on the apo pose ----------------------------------
    apo_parquet = work / f"{args.pdb_id}_apo_frustration.parquet"
    if apo_parquet.exists():
        print(f"reusing existing {apo_parquet}")
        apo = pd.read_parquet(apo_parquet)
    else:
        scorefxn = pyrosetta.create_score_function("ref2015")
        t0 = time.time()
        print(f"running {args.n_decoys} decoys on {args.n_jobs} workers (seed {args.seed})...")
        apo = fr.run_frustration_survey(
            pose=apo_pose,
            scorefxn=scorefxn,
            contacts=contacts,
            n_decoys=args.n_decoys,
            seed=args.seed,
            exclude_fa_rep=True,
            checkpoint_path=str(work / f"{args.pdb_id}_apo_ckpt.pkl"),
            checkpoint_every=10,
            n_jobs=args.n_jobs,
            worker_pose_paths=(str(apo_pdb), str(params)),
            score_function="ref2015",
        )
        apo.to_parquet(apo_parquet, index=False)
        print(f"apo survey done in {(time.time() - t0) / 60:.1f} min -> {apo_parquet}")

    # ---- compare ----------------------------------------------------------
    m = holo.merge(apo, on=["resi", "resj"], suffixes=("_holo", "_apo"))
    if len(m) != len(holo):
        print(f"FATAL: merge lost rows ({len(m)} vs {len(holo)})", file=sys.stderr)
        return 3

    in_pocket = np.array(
        [(i in pocket) or (j in pocket) for i, j in zip(m.resi, m.resj)]
    )
    m["in_pocket"] = in_pocket

    stats = {}
    for label, mask in (("all", np.ones(len(m), bool)),
                        ("pocket", in_pocket),
                        ("non_pocket", ~in_pocket)):
        sub = m[mask]
        if len(sub) < 3:
            continue
        r, p = pearsonr(sub.F_index_holo, sub.F_index_apo)
        rho, _ = spearmanr(sub.F_index_holo, sub.F_index_apo)
        d = (sub.F_index_holo - sub.F_index_apo).abs()
        same_class = (sub.frustration_class_holo == sub.frustration_class_apo).mean()
        stats[label] = {
            "n": len(sub),
            "pearson_r": r,
            "pearson_p": p,
            "spearman_rho": rho,
            "mean_abs_dF": d.mean(),
            "median_abs_dF": d.median(),
            "p95_abs_dF": d.quantile(0.95),
            "max_abs_dF": d.max(),
            "class_agreement": same_class,
        }

    # Where does the difference enter? Report the max absolute delta alongside the
    # correlation: a correlation of 1.000000 and a max delta of exactly 0 are very
    # different claims, and for E_native the second one is the true statement.
    comp = {}
    for col in ("E_native", "decoy_mean", "decoy_std"):
        r, _ = pearsonr(m[f"{col}_holo"], m[f"{col}_apo"])
        d = (m[f"{col}_holo"] - m[f"{col}_apo"]).abs()
        comp[col] = {
            "r": r,
            "max_abs_delta": float(d.max()),
            "n_nonzero": int((d > 1e-12).sum()),
        }

    print("\n" + "=" * 72)
    print(f"A3 apo control — {args.pdb_id} ({args.ligand})")
    print("=" * 72)
    hdr = (f"{'subset':<12} {'n':>6} {'pearson r':>11} {'spearman':>9} "
           f"{'mean|dF|':>9} {'p95|dF|':>8} {'class agree':>12}")
    print(hdr); print("-" * len(hdr))
    for label, s in stats.items():
        print(f"{label:<12} {s['n']:>6} {s['pearson_r']:>11.4f} {s['spearman_rho']:>9.4f} "
              f"{s['mean_abs_dF']:>9.4f} {s['p95_abs_dF']:>8.4f} {s['class_agreement']:>11.1%}")

    print("\nholo-vs-apo agreement of the underlying quantities:")
    for col, c in comp.items():
        print(f"  {col:<12} r = {c['r']:.6f}   max|delta| = {c['max_abs_delta']:.3e}   "
              f"differing = {c['n_nonzero']}/{len(m)}")
    if comp["E_native"]["n_nonzero"] == 0:
        print("  -> E_native is BIT-IDENTICAL: the native reference cannot see the ligand.")

    pr = stats["pocket"]["pearson_r"]
    blind = pr > LIGAND_BLIND_R
    print(f"\nVERDICT: pocket-contact Pearson r = {pr:.4f}")
    print(f"  {'LIGAND-BLIND' if blind else 'ligand has measurable effect'} "
          f"(threshold r > {LIGAND_BLIND_R})")

    write_report(Path(args.out), args, stats, comp, m, pockets, blind, len(lig))
    print(f"\nReport written to {args.out}")
    return 0


def write_report(out: Path, args, stats, comp, m, pockets, blind, n_lig_heavy) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ps = stats["pocket"]
    L = [
        f"# A3 — apo control on {args.pdb_id} ({args.ligand})",
        "",
        "Generated by `analysis/apo_control.py`. Does the ligand affect the frustration",
        "index at all?",
        "",
        "## Design",
        "",
        "The ligand is deleted from `data/processed/{pdb}_clean.pdb` and the identical",
        f"protocol re-run at the identical seed ({args.seed}, {args.n_decoys} decoys).",
        "",
        "The comparison is paired **by construction**: `generate_decoy` draws a new identity",
        "for each protein residue in pose order and skips non-protein residues, and",
        "`native_aa_frequency` counts protein residues only. Holo and apo therefore consume",
        "the same RNG stream and generate the *same decoy sequences*. The sole difference is",
        "whether the ligand is present while each decoy is repacked, minimised and scored.",
        "",
        f"The contact set was verified identical between holo and apo ({len(m)} pairs) before",
        "any comparison was made; a mismatch would abort the run.",
        "",
        f"Ligand heavy atoms: {n_lig_heavy}. Pocket (`{args.shell}`):",
        f"{len(pockets[args.shell])} residues; pocket (`ca10`, the pipeline's own selector):",
        f"{len(pockets['ca10'])} residues.",
        "",
        "## Result",
        "",
        "| subset | n | Pearson r | Spearman rho | mean\\|dF\\| | median\\|dF\\| | p95\\|dF\\| | max\\|dF\\| | class agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in stats.items():
        L.append(
            f"| {label} | {s['n']} | {s['pearson_r']:.4f} | {s['spearman_rho']:.4f} | "
            f"{s['mean_abs_dF']:.4f} | {s['median_abs_dF']:.4f} | {s['p95_abs_dF']:.4f} | "
            f"{s['max_abs_dF']:.4f} | {s['class_agreement']:.1%} |"
        )

    L += [
        "",
        "## Where the ligand's influence enters",
        "",
        "| quantity | Pearson r | max\\|delta\\| | contacts differing |",
        "|---|---:|---:|---:|",
    ]
    for col, c in comp.items():
        L.append(
            f"| `{col}` | {c['r']:.6f} | {c['max_abs_delta']:.3e} | "
            f"{c['n_nonzero']}/{len(m)} |"
        )

    if comp["E_native"]["n_nonzero"] == 0:
        L += [
            "",
            "**`E_native` is bit-identical between holo and apo — max\\|delta\\| is exactly",
            "zero across every contact.** This is structural, not a numerical coincidence:",
            "",
            "- `E_ij` sums `e_ik` only over `contacts(i)`, and `contacts` comes from",
            "  `get_protein_contacts` (`src/frustration.py:70-108`), which is protein-only.",
            "  The ligand is therefore never a member of any partner list.",
            "- `pairwise_energy` (`:182-213`) reads the REF2015 `EnergyGraph` edge between two",
            "  *protein* residues, a two-body term in which the ligand does not participate.",
            "",
            "**The native reference energy provably cannot see the ligand.** The ligand's entire",
            "influence on the index is therefore mediated through the decoys alone: side chains",
            "repack differently when the pocket is occupied, which perturbs `decoy_mean`",
            f"(max\\|delta\\| {comp['decoy_mean']['max_abs_delta']:.3f}) and `decoy_std`",
            f"(max\\|delta\\| {comp['decoy_std']['max_abs_delta']:.3f}). That indirect channel is",
            "the whole of the ligand-awareness in the current implementation.",
        ]

    L += [
        "",
        f"## Verdict: pocket-contact Pearson r = {ps['pearson_r']:.4f}",
        "",
    ]
    if blind:
        L += [
            f"**Above the r > {LIGAND_BLIND_R} threshold: the descriptor is effectively",
            "ligand-blind under the current many-body formula.** Deleting the ligand entirely",
            "leaves the per-contact frustration index on pocket contacts almost unchanged",
            f"(class agreement {ps['class_agreement']:.1%}, median \\|dF\\| "
            f"{ps['median_abs_dF']:.4f}).",
            "",
            "This is the predicted consequence of the A1 degeneracy. `E_ij = 0.5*(B_i + B_j)`",
            "means the ligand's contribution reaches a contact only through one term of ~30 in",
            "a pocket residue's `B_i`, halved again by the 0.5 weighting. A ligand-aware",
            "frustration measure that does not measurably respond to removing the ligand is",
            "not measuring the ligand.",
            "",
            "Taken with A2 -- where no pocket selector or threshold recovered either the",
            "paper's counts or its affinity relationship -- the three Stage-A results form one",
            "diagnosis: **the implemented index is a per-residue, protein-only quantity that",
            "happens to be computed in the presence of a ligand.**",
        ]
    else:
        L += [
            f"**Below the r > {LIGAND_BLIND_R} threshold: the ligand has a measurable effect",
            "on the index.** The descriptor is not ligand-blind, so the A1 degeneracy alone",
            "does not render the calculation vacuous. The reproduction gap identified in A2",
            "must then be explained by the contact definition or the many-body mode rather",
            "than by the ligand being invisible to the energy function.",
        ]

    L += [
        "",
        "## Scope",
        "",
        f"One structure ({args.pdb_id}), {args.n_decoys} decoys. This is a screening test, not",
        "an estimate: it answers whether the effect is near-zero, not how large it is across",
        "the set. If the verdict is ligand-blind, the cheap confirmation is to repeat on two",
        "or three more structures with different ligand sizes before drawing conclusions in",
        "writing.",
        "",
        "The apo run's outputs are kept out of `results/` deliberately -- they live under",
        f"`{args.workdir}/` so they cannot be mistaken for a pipeline structure or picked up",
        "by a summary rebuild.",
    ]
    out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
