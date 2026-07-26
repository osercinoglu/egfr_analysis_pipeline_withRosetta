"""
EGFR Atomistic Frustration Pipeline — Main Runner Script

Used for Stage 3 (validation) and Stage 4 (EGFR analysis).

Usage examples:
  # Validation (lysozyme, quick test):
  python src/run_pipeline.py --mode validate --n_decoys 50

  # Single-structure test:
  python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50

  # All EGFR structures (full analysis):
  python src/run_pipeline.py --mode all --n_decoys 200
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_candidates(cfg: dict) -> pd.DataFrame:
    """
    Builds the list of EGFR-inhibitor structures ready for frustration
    analysis, from the Stage 1/4/5 outputs of scripts/01-05:
      - results/metadata/egfr_ligand_inventory.csv (Stage 1: pdb_id, ligand_comp_id, affinity_pM)
      - results/preparation_summary.csv            (Stage 4: pdb_id -> rosetta_ligand_comp_id, status)
      - results/metadata/ligand_parameterization_status.csv (Stage 5: ligand_comp_id -> status)

    Only rows where both the complex (Stage 4) and its ligand params
    (Stage 5) passed validation ("OK") are included.

    Returns columns: pdb_id, ligand_comp_id, rosetta_ligand_comp_id, affinity_pM
    """
    inventory = pd.read_csv(cfg["paths"]["candidates_csv"])[
        ["pdb_id", "ligand_comp_id", "affinity_pM"]
    ]
    prep = pd.read_csv(cfg["paths"]["prep_summary_csv"])
    prep_ok = prep[prep["status"] == "OK"][["pdb_id", "rosetta_ligand_comp_id"]]

    lig_status = pd.read_csv(cfg["paths"]["ligand_status_csv"])
    validated_ligands = set(lig_status[lig_status["status"] == "OK"]["ligand_comp_id"])

    df = inventory.merge(prep_ok, on="pdb_id", how="inner")
    df = df[df["ligand_comp_id"].isin(validated_ligands)]
    return df.reset_index(drop=True)


def load_pose_with_ligand(processed_pdb: str, params_file: str):
    """Load a pose with the ligand via PyRosetta."""
    import pyrosetta
    from pyrosetta import Pose, Vector1
    from pyrosetta.rosetta.core.chemical import ChemicalManager

    pose = Pose()
    res_set = pyrosetta.generate_nonstandard_residue_set(
        pose, Vector1([params_file])
    )
    pyrosetta.pose_from_file(pose, res_set, processed_pdb)
    return pose


def find_ligand_resnum(pose, rosetta_ligand_comp_id: str) -> int | None:
    """Find the ligand residue number within the pose (matches the residue
    name Rosetta actually uses — see rosetta_ligand_comp_id)."""
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.name3().strip() == rosetta_ligand_comp_id[:3]:
            return i
    return None


def run_single_structure(
    pdb_id: str,
    ligand_comp_id: str,
    cfg: dict,
    n_decoys: int,
    scorefxn,
    seed: int = 0,
    rosetta_ligand_comp_id: str | None = None,
) -> dict | None:
    """
    Runs the frustration analysis for a single EGFR-inhibitor complex.
    Saves results to a checkpoint, or loads directly if already complete.

    ligand_comp_id is the real CCD ligand identity (used for result/checkpoint
    filenames); rosetta_ligand_comp_id is the residue name Rosetta actually
    sees (see scripts/04_prepare_complex.py and config/ligand_overrides.yaml)
    — usually identical to ligand_comp_id, except for the rare cases with an
    internal Rosetta residue/patch name collision (e.g. 5HG8: '634' -> 'Z34').
    """
    import pyrosetta
    from frustration import (
        get_protein_contacts,
        get_ligand_contacts,
        run_frustration_survey,
        summarize_ligand_frustration,
    )

    rosetta_ligand_comp_id = rosetta_ligand_comp_id or ligand_comp_id

    results_dir   = Path(cfg["paths"]["results"])
    ckpt_dir      = Path(cfg["paths"]["checkpoints"])
    proc_dir      = Path(cfg["paths"]["processed"])
    params_dir    = Path(cfg["paths"]["params"])

    processed_pdb = proc_dir / f"{pdb_id}_clean.pdb"
    params_file   = params_dir / f"{rosetta_ligand_comp_id}.params"
    ckpt_file     = ckpt_dir / f"{pdb_id}_{ligand_comp_id}_ckpt.pkl"
    result_file   = results_dir / f"{pdb_id}_{ligand_comp_id}_frustration.parquet"

    # Load the result if it's already complete
    if result_file.exists():
        logger.info(f"{pdb_id}: Result already exists, loading.")
        df = pd.read_parquet(result_file)
        # Recompute ligand contacts (needs the pose)
        pose = load_pose_with_ligand(str(processed_pdb), str(params_file))
        lig_resnum = find_ligand_resnum(pose, rosetta_ligand_comp_id)
        lig_cutoff = cfg["contacts"]["ligand_protein_cutoff_A"]
        if lig_resnum:
            lig_contacts = get_ligand_contacts(pose, lig_resnum, lig_cutoff)
        else:
            lig_contacts = []
        return summarize_ligand_frustration(df, lig_contacts)

    if not processed_pdb.exists() or not params_file.exists():
        logger.warning(f"{pdb_id}: processed file or params not found, skipping.")
        return None

    # Load the pose
    logger.info(f"{pdb_id}: Loading pose...")
    try:
        pose = load_pose_with_ligand(str(processed_pdb), str(params_file))
    except Exception as e:
        logger.error(f"{pdb_id}: Pose loading error: {e}")
        return None

    logger.info(f"  Total residues: {pose.total_residue()}")

    # Protein contacts
    prot_cutoff = cfg["contacts"]["protein_protein_cutoff_A"]
    seq_sep     = cfg["contacts"]["seq_sep_min"]
    contacts = get_protein_contacts(pose, prot_cutoff, seq_sep)
    logger.info(f"  {len(contacts)} protein-protein contacts")

    # Ligand contacts
    lig_resnum = find_ligand_resnum(pose, rosetta_ligand_comp_id)
    if lig_resnum is None:
        logger.warning(f"{pdb_id}: Ligand {rosetta_ligand_comp_id} not found in pose!")
        lig_contacts = []
    else:
        lig_cutoff = cfg["contacts"]["ligand_protein_cutoff_A"]
        lig_contacts = get_ligand_contacts(pose, lig_resnum, lig_cutoff)
        logger.info(f"  {len(lig_contacts)} ligand-protein contact residues")

    # Frustration survey
    t0 = time.time()
    df_frust = run_frustration_survey(
        pose=pose,
        scorefxn=scorefxn,
        contacts=contacts,
        n_decoys=n_decoys,
        seed=seed,
        exclude_fa_rep=cfg["frustration"]["exclude_fa_rep"],
        checkpoint_path=str(ckpt_file),
        checkpoint_every=cfg["checkpoint"]["save_every_n_decoys"],
    )
    elapsed = time.time() - t0
    logger.info(f"  {n_decoys} decoys completed — {elapsed/60:.1f} min")

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    df_frust.to_parquet(result_file, index=False)
    logger.info(f"  Saved: {result_file}")

    summary = summarize_ligand_frustration(df_frust, lig_contacts)
    return summary


def run_validation(cfg: dict, n_decoys: int):
    """
    Stage 3: core/surface frustration validation on lysozyme (1LYZ).
    Expectation: buried core contacts → minimally frustrated,
                 surface contacts → neutral/highly frustrated.
    """
    import pyrosetta
    from frustration import (
        get_protein_contacts,
        run_frustration_survey,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Download 1LYZ
    import requests
    pdb_path = Path(cfg["paths"]["raw_pdb"]) / "1LYZ.pdb"
    if not pdb_path.exists():
        r = requests.get("https://files.rcsb.org/download/1LYZ.pdb", timeout=30)
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.write_bytes(r.content)

    # Load the pose (no ligand)
    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    logger.info(f"Lysozyme: {pose.total_residue()} residues loaded")

    scorefxn = pyrosetta.create_score_function(
        cfg["energy"]["score_function"]
    )

    contacts = get_protein_contacts(
        pose,
        cfg["contacts"]["protein_protein_cutoff_A"],
        cfg["contacts"]["seq_sep_min"],
    )
    logger.info(f"  {len(contacts)} contacts")

    df = run_frustration_survey(
        pose=pose,
        scorefxn=scorefxn,
        contacts=contacts,
        n_decoys=n_decoys,
        seed=cfg["frustration"]["seed"],
        exclude_fa_rep=cfg["frustration"]["exclude_fa_rep"],
        checkpoint_path=str(
            Path(cfg["paths"]["checkpoints"]) / "1LYZ_validation_ckpt.pkl"
        ),
        checkpoint_every=cfg["checkpoint"]["save_every_n_decoys"],
    )

    # Burial estimate via SASA (Biopython)
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.SASA import ShrakeRupley
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("lyz", str(pdb_path))
        sr = ShrakeRupley()
        sr.compute(struct, level="R")
        sasa_map = {}
        for residue in struct.get_residues():
            sasa_map[residue.id[1]] = residue.sasa
    except Exception:
        sasa_map = {}

    # Figure: F_index histogram + scatter against SASA
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: frustration class distribution
    counts = df["frustration_class"].value_counts()
    axes[0].bar(counts.index, counts.values,
                color=["green","gray","red"],
                edgecolor="black")
    axes[0].set_title("Lysozyme Frustration Distribution (1LYZ)")
    axes[0].set_ylabel("Contact count")

    # Right: F_index histogram
    axes[1].hist(df["F_index"], bins=50, color="steelblue", edgecolor="none")
    axes[1].axvline(0.78,  color="green", ls="--", label="min frustrated (>0.78)")
    axes[1].axvline(-1.0,  color="red",   ls="--", label="highly frustrated (<-1)")
    axes[1].set_xlabel("Frustration Index (F_ij)")
    axes[1].set_ylabel("Contact count")
    axes[1].set_title("Frustration Index Distribution")
    axes[1].legend()

    plt.tight_layout()
    fig_path = results_dir / "validation_lysozyme.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Validation figure saved: {fig_path}")

    df.to_parquet(
        Path(cfg["paths"]["checkpoints"]) / "1LYZ_frustration.parquet",
        index=False,
    )

    n_min  = (df["frustration_class"] == "minimally_frustrated").sum()
    n_high = (df["frustration_class"] == "highly_frustrated").sum()
    logger.info(
        f"Lysozyme: %{100*n_min/len(df):.0f} minimally frustrated, "
        f"%{100*n_high/len(df):.0f} highly frustrated"
    )
    return df


def run_all_egfr(cfg: dict, n_decoys: int):
    """
    Frustration analysis + correlation plot for all EGFR structures listed
    by load_candidates() (Stage 1-5 outputs, currently 61 structures / 51
    unique ligands).
    """
    import pyrosetta
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    df_cands = load_candidates(cfg)
    scorefxn = pyrosetta.create_score_function(cfg["energy"]["score_function"])

    results = []
    for _, row in df_cands.iterrows():
        pdb_id         = row["pdb_id"]
        ligand_comp_id = row["ligand_comp_id"]
        rosetta_id     = row["rosetta_ligand_comp_id"]
        aff_pM         = row["affinity_pM"]

        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {pdb_id} (ligand={ligand_comp_id}, "
                    f"affinity={aff_pM:.1f} pM)")

        summary = run_single_structure(
            pdb_id, ligand_comp_id, cfg, n_decoys, scorefxn,
            seed=cfg["frustration"]["seed"],
            rosetta_ligand_comp_id=rosetta_id,
        )
        if summary is None:
            continue

        results.append({
            "pdb_id": pdb_id,
            "ligand_comp_id": ligand_comp_id,
            "affinity_pM": aff_pM,
            "log10_affinity_pM": np.log10(max(aff_pM, 1e-3)),
            **summary,
        })
        logger.info(f"  Ligand pocket: {summary['n_minimally_frustrated']} min-frust / "
                    f"{summary['n_contacts_total']} total contacts")

    df_res = pd.DataFrame(results)

    # Save
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(results_dir / "egfr_frustration_summary.csv", index=False)
    logger.info(f"\nResults saved: {results_dir}/egfr_frustration_summary.csv")

    # Correlation plot
    _plot_correlation(df_res, results_dir)

    return df_res


def _plot_correlation(df: pd.DataFrame, results_dir: Path):
    """Fig. 5e-style scatter plot: min-frustrated contacts vs log(affinity).

    The Stage 1-5 metadata (results/metadata/egfr_ligand_inventory.csv) does
    not distinguish Kd/Ki from IC50, so unlike the original 25-structure
    version this uses a single marker style for all points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    if df.empty:
        return

    x = df["n_minimally_frustrated"].values.astype(float)
    y = df["log10_affinity_pM"].values.astype(float)

    # Pearson correlation
    if len(df) >= 3:
        r, p = stats.pearsonr(x, y)
    else:
        r, p = float("nan"), float("nan")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(x, y,
               c="royalblue", s=80, zorder=5,
               edgecolors="k", linewidths=0.5)

    # Labels
    for _, row in df.iterrows():
        ax.annotate(
            row["pdb_id"],
            (row["n_minimally_frustrated"], row["log10_affinity_pM"]),
            fontsize=7, xytext=(4, 2), textcoords="offset points",
        )

    # Regression line
    if len(df) >= 3 and not np.isnan(r):
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, "k--", lw=1, alpha=0.5)

    ax.set_xlabel("Number of Minimally Frustrated Ligand-Pocket Contacts", fontsize=12)
    ax.set_ylabel("log₁₀(Affinity / pM)", fontsize=12)
    ax.set_title(
        f"EGFR Frustration–Affinity Correlation\n"
        f"Pearson r = {r:.3f}, p = {p:.3f} (n={len(df)})",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = results_dir / "egfr_correlation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Correlation plot: {fig_path}")
    logger.info(f"Pearson r={r:.3f}, p={p:.3f}, n={len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="EGFR Atomistic Frustration Pipeline"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", choices=["validate", "single", "all"],
        default="validate",
        help="validate=lysozyme test, single=single structure, all=all EGFR structures",
    )
    parser.add_argument("--pdb_id", default=None)
    parser.add_argument("--n_decoys", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_decoys = args.n_decoys or cfg["frustration"]["n_decoys"]

    # Initialize PyRosetta
    import pyrosetta
    pyrosetta.init("-mute all")
    logger.info("PyRosetta initialized.")

    if args.mode == "validate":
        run_validation(cfg, n_decoys)

    elif args.mode == "single":
        if not args.pdb_id:
            parser.error("--pdb_id is required for --mode single")
        import pyrosetta
        scorefxn = pyrosetta.create_score_function(cfg["energy"]["score_function"])
        df_cands = load_candidates(cfg)
        row = df_cands[df_cands["pdb_id"] == args.pdb_id]
        if row.empty:
            logger.error(f"{args.pdb_id} not in the candidate list.")
            return
        summary = run_single_structure(
            args.pdb_id, row.iloc[0]["ligand_comp_id"], cfg, n_decoys, scorefxn,
            seed=cfg["frustration"]["seed"],
            rosetta_ligand_comp_id=row.iloc[0]["rosetta_ligand_comp_id"],
        )
        logger.info(f"Result: {summary}")

    elif args.mode == "all":
        run_all_egfr(cfg, n_decoys)


if __name__ == "__main__":
    main()
