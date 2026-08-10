"""
EGFR Atomistic Frustration Pipeline — Main Runner Script

Used for Stage 3 (validation) and Stage 4 (EGFR analysis).

Usage examples:
  # Validation (lysozyme, quick test):
  python src/run_pipeline.py --mode validate --n_decoys 50

  # Single-structure test:
    python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50 --n-jobs 2

  # All EGFR structures (full analysis):
  python src/run_pipeline.py --mode all --n_decoys 200

    # Restrict an all-mode run and use two spawned workers:
    python src/run_pipeline.py --mode all --pdb-ids 1XKK,5GMP --n_decoys 50 --n-jobs 2
"""

import argparse
import logging
import multiprocessing as mp
import os
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

_WORKER_SCOREFXN = None


def default_worker_count() -> int:
    """Use all available logical CPUs by default."""
    return max(1, os.cpu_count() or 1)


def resolve_worker_count(n_jobs: int | None, candidate_count: int) -> int:
    """Resolve a requested worker count without starting idle workers."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    if n_jobs is not None and n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    requested = n_jobs if n_jobs is not None else default_worker_count()
    return min(requested, candidate_count)


def _worker_init(score_function: str) -> None:
    """Initialize an isolated PyRosetta runtime for a spawned worker."""
    import pyrosetta

    global _WORKER_SCOREFXN
    pyrosetta.init("-mute all")
    _WORKER_SCOREFXN = pyrosetta.create_score_function(score_function)


def _worker_run_single(task: dict) -> dict | None:
    """Run one structure in a spawned PyRosetta worker."""
    if _WORKER_SCOREFXN is None:
        raise RuntimeError("PyRosetta worker was not initialized")

    pdb_id = task["pdb_id"]
    summary = run_single_structure(
        pdb_id=pdb_id,
        ligand_comp_id=task["ligand_comp_id"],
        cfg=task["cfg"],
        n_decoys=task["n_decoys"],
        scorefxn=_WORKER_SCOREFXN,
        seed=task["seed"],
        rosetta_ligand_comp_id=task["rosetta_ligand_comp_id"],
        n_jobs_decoys=1,
        structures_dir=task.get("structures_dir"),
    )
    if summary is None:
        return None

    affinity_pM = task["affinity_pM"]
    result = {
        "pdb_id": pdb_id,
        "ligand_comp_id": task["ligand_comp_id"],
        "affinity_pM": affinity_pM,
        "log10_affinity_pM": np.log10(max(affinity_pM, 1e-3)),
        **summary,
    }
    logger.info(
        "%s: Ligand pocket: %s min-frust / %s total contacts",
        pdb_id,
        summary["n_minimally_frustrated"],
        summary["n_contacts_total"],
    )
    return result


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def with_output_directories(
    cfg: dict,
    results_dir: str | None = None,
    checkpoints_dir: str | None = None,
    structures_dir: str | None = None,
) -> dict:
    """Return a config copy with optional Stage 6 output directory overrides."""
    if results_dir is None and checkpoints_dir is None and structures_dir is None:
        return cfg

    updated_cfg = {**cfg, "paths": {**cfg["paths"]}}
    if results_dir is not None:
        updated_cfg["paths"]["results"] = str(Path(results_dir))
    if checkpoints_dir is not None:
        updated_cfg["paths"]["checkpoints"] = str(Path(checkpoints_dir))
    if structures_dir is not None:
        updated_cfg["paths"]["saved_structures"] = str(Path(structures_dir))
    return updated_cfg


def structure_output_directory(
    cfg: dict,
    pdb_id: str,
    ligand_comp_id: str,
) -> Path | None:
    """Return the per-structure directory where native/decoy PDBs should be written."""
    structures_root = cfg["paths"].get("saved_structures")
    if not structures_root:
        return None
    return Path(structures_root) / f"{pdb_id}_{ligand_comp_id}"


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
    n_jobs_decoys: int = 1,
    structures_dir: str | None = None,
) -> dict | None:
    """
    Runs the frustration analysis for a single EGFR-inhibitor complex.
    Saves results to a checkpoint, or loads directly if already complete.

    ligand_comp_id is the real CCD ligand identity (used for result/checkpoint
    filenames); rosetta_ligand_comp_id is the residue name Rosetta actually
    sees (see scripts/04_prepare_complex.py and config/ligand_overrides.yaml)
    — usually identical to ligand_comp_id, except for the rare cases with an
    internal Rosetta residue/patch name collision (e.g. 5HG8: '634' -> 'Z34').
    If structures_dir is set, the native pose and generated decoy poses are
    written there as PDB files.
    """
    import pyrosetta
    from frustration import (
        get_protein_contacts,
        get_ligand_contacts,
        run_frustration_survey,
        save_pose_pdb,
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
    if structures_dir is None:
        structure_dir = structure_output_directory(cfg, pdb_id, ligand_comp_id)
    else:
        structure_dir = Path(structures_dir)
    native_pdb_path = structure_dir / "native.pdb" if structure_dir is not None else None
    decoy_dir = structure_dir / "decoys" if structure_dir is not None else None

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
        if native_pdb_path is not None:
            save_pose_pdb(pose, native_pdb_path)
        if decoy_dir is not None:
            logger.warning(
                "%s: decoy PDB saving was requested, but the existing parquet result "
                "prevents reconstructing decoy structures. Delete the result and "
                "checkpoint to regenerate decoys with PDB output.",
                pdb_id,
            )
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
    if native_pdb_path is not None:
        save_pose_pdb(pose, native_pdb_path)

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
        n_jobs=n_jobs_decoys,
        worker_pose_paths=(str(processed_pdb), str(params_file)),
        score_function=cfg["energy"]["score_function"],
        decoy_structures_dir=str(decoy_dir) if decoy_dir is not None else None,
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
        save_pose_pdb,
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
    structure_dir = None
    if cfg["paths"].get("saved_structures"):
        structure_dir = Path(cfg["paths"]["saved_structures"]) / "1LYZ_validation"
        save_pose_pdb(pose, structure_dir / "native.pdb")

    scorefxn = pyrosetta.create_score_function(
        cfg["energy"]["score_function"]
    )

    contacts = get_protein_contacts(
        pose,
        cfg["contacts"]["protein_protein_cutoff_A"],
        cfg["contacts"]["seq_sep_min"],
    )
    logger.info(f"  {len(contacts)} contacts")
    checkpoints_dir = Path(cfg["paths"]["checkpoints"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    df = run_frustration_survey(
        pose=pose,
        scorefxn=scorefxn,
        contacts=contacts,
        n_decoys=n_decoys,
        seed=cfg["frustration"]["seed"],
        exclude_fa_rep=cfg["frustration"]["exclude_fa_rep"],
        checkpoint_path=str(
            checkpoints_dir / "1LYZ_validation_ckpt.pkl"
        ),
        checkpoint_every=cfg["checkpoint"]["save_every_n_decoys"],
        decoy_structures_dir=(
            str(structure_dir / "decoys") if structure_dir is not None else None
        ),
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
        checkpoints_dir / "1LYZ_frustration.parquet",
        index=False,
    )

    n_min  = (df["frustration_class"] == "minimally_frustrated").sum()
    n_high = (df["frustration_class"] == "highly_frustrated").sum()
    logger.info(
        f"Lysozyme: %{100*n_min/len(df):.0f} minimally frustrated, "
        f"%{100*n_high/len(df):.0f} highly frustrated"
    )
    return df


def run_all_egfr(
    cfg: dict,
    n_decoys: int,
    n_jobs: int | None = None,
    pdb_ids: set[str] | None = None,
):
    """
    Frustration analysis + correlation plot for all EGFR structures listed
    by load_candidates() (Stage 1-5 outputs, currently 61 structures / 51
    unique ligands).

    Uses spawned processes so each worker initializes an independent PyRosetta
    runtime. By default, two logical CPUs remain available for the system.
    """
    import matplotlib
    matplotlib.use("Agg")

    df_cands = load_candidates(cfg)
    if pdb_ids is not None:
        df_cands = df_cands[df_cands["pdb_id"].isin(pdb_ids)].copy()
        missing_ids = pdb_ids - set(df_cands["pdb_id"])
        if missing_ids:
            raise ValueError(
                "Requested PDB IDs are not ready for analysis: "
                + ", ".join(sorted(missing_ids))
            )
    if df_cands.empty:
        raise ValueError("No structures are ready for analysis")

    worker_count = resolve_worker_count(n_jobs, len(df_cands))
    logger.info(
        "Processing %s structures with %s spawned workers (CPU count: %s).",
        len(df_cands),
        worker_count,
        os.cpu_count() or 1,
    )

    tasks = [
        {
            "pdb_id": row.pdb_id,
            "ligand_comp_id": row.ligand_comp_id,
            "rosetta_ligand_comp_id": row.rosetta_ligand_comp_id,
            "affinity_pM": row.affinity_pM,
            "cfg": cfg,
            "n_decoys": n_decoys,
            "seed": cfg["frustration"]["seed"],
            "structures_dir": (
                str(structure_output_directory(cfg, row.pdb_id, row.ligand_comp_id))
                if cfg["paths"].get("saved_structures")
                else None
            ),
        }
        for row in df_cands.itertuples(index=False)
    ]

    results = []
    context = mp.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=_worker_init,
        initargs=(cfg["energy"]["score_function"],),
    ) as pool:
        for result in pool.imap_unordered(_worker_run_single, tasks):
            if result is not None:
                results.append(result)

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values("pdb_id").reset_index(drop=True)

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
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for Stage 6 result files and plots (default: config paths.results).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        help="Directory for Stage 6 checkpoint files (default: config paths.checkpoints).",
    )
    parser.add_argument(
        "--save-structures-dir",
        default=None,
        help=(
            "Directory where native and decoy PDBs should be saved. Files are "
            "written under per-structure subdirectories; omitted means no PDB saving."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        "--n_jobs",
        dest="n_jobs",
        type=int,
        default=None,
        help=(
            "Spawned workers: decoy workers in single mode and structure workers "
            "in all mode (default: all available logical CPUs)."
        ),
    )
    parser.add_argument(
        "--pdb-ids",
        default=None,
        help="Comma-separated PDB IDs to analyze in --mode all.",
    )
    args = parser.parse_args()

    cfg = with_output_directories(
        load_config(args.config),
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        structures_dir=args.save_structures_dir,
    )
    n_decoys = args.n_decoys or cfg["frustration"]["n_decoys"]

    if args.mode == "validate":
        import pyrosetta
        pyrosetta.init("-mute all")
        logger.info("PyRosetta initialized.")
        run_validation(cfg, n_decoys)

    elif args.mode == "single":
        if not args.pdb_id:
            parser.error("--pdb_id is required for --mode single")
        import pyrosetta
        pyrosetta.init("-mute all")
        logger.info("PyRosetta initialized.")
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
            n_jobs_decoys=args.n_jobs or default_worker_count(),
        )
        logger.info(f"Result: {summary}")

    elif args.mode == "all":
        pdb_ids = (
            {pdb_id.strip() for pdb_id in args.pdb_ids.split(",") if pdb_id.strip()}
            if args.pdb_ids
            else None
        )
        run_all_egfr(cfg, n_decoys, n_jobs=args.n_jobs, pdb_ids=pdb_ids)


if __name__ == "__main__":
    main()
