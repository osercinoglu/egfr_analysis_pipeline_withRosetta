"""
EGFR Atomistic Frustration Pipeline — Ana Çalıştırma Scripti

Aşama 3 (validasyon) ve Aşama 4 (EGFR analizi) için kullanılır.

Çalıştırma örnekleri:
  # Validasyon (lysozyme, hızlı test):
  python src/run_pipeline.py --mode validate --n_decoys 50

  # Tek yapı test:
  python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50

  # Tüm EGFR yapıları (tam analiz):
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


def load_pose_with_ligand(processed_pdb: str, params_file: str):
    """PyRosetta ile ligandlı pose yükle."""
    import pyrosetta
    from pyrosetta import Pose, Vector1
    from pyrosetta.rosetta.core.chemical import ChemicalManager

    pose = Pose()
    res_set = pyrosetta.generate_nonstandard_residue_set(
        pose, Vector1([params_file])
    )
    pyrosetta.pose_from_file(pose, res_set, processed_pdb)
    return pose


def find_ligand_resnum(pose, ligand_id: str) -> int | None:
    """Pose içinde ligand residue numarasını bul."""
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.name3().strip() == ligand_id[:3]:
            return i
    return None


def run_single_structure(
    pdb_id: str,
    ligand_id: str,
    cfg: dict,
    n_decoys: int,
    scorefxn,
    seed: int = 0,
) -> dict | None:
    """
    Tek bir EGFR-inhibitör kompleksi için frustrasyon analizi çalıştırır.
    Sonuçları checkpoint'e kaydeder, tamamlanmışsa direkt yükler.
    """
    import pyrosetta
    from src.frustration import (
        get_protein_contacts,
        get_ligand_contacts,
        run_frustration_survey,
        summarize_ligand_frustration,
    )

    results_dir   = Path(cfg["paths"]["results"])
    ckpt_dir      = Path(cfg["paths"]["checkpoints"])
    proc_dir      = Path(cfg["paths"]["processed"])
    params_dir    = Path(cfg["paths"]["params"])

    processed_pdb = proc_dir / f"{pdb_id}_clean.pdb"
    params_file   = params_dir / f"{pdb_id}_{ligand_id}.params"
    ckpt_file     = ckpt_dir / f"{pdb_id}_{ligand_id}_ckpt.pkl"
    result_file   = results_dir / f"{pdb_id}_{ligand_id}_frustration.parquet"

    # Tamamlanmış sonuç varsa yükle
    if result_file.exists():
        logger.info(f"{pdb_id}: Sonuç zaten var, yükleniyor.")
        df = pd.read_parquet(result_file)
        # Ligand kontakları yeniden hesapla (pose gerekiyor)
        pose = load_pose_with_ligand(str(processed_pdb), str(params_file))
        lig_resnum = find_ligand_resnum(pose, ligand_id)
        lig_cutoff = cfg["contacts"]["ligand_protein_cutoff_A"]
        if lig_resnum:
            lig_contacts = get_ligand_contacts(pose, lig_resnum, lig_cutoff)
        else:
            lig_contacts = []
        return summarize_ligand_frustration(df, lig_contacts)

    if not processed_pdb.exists() or not params_file.exists():
        logger.warning(f"{pdb_id}: İşlenmiş dosya veya params bulunamadı, atlanıyor.")
        return None

    # Pose yükle
    logger.info(f"{pdb_id}: Pose yükleniyor...")
    try:
        pose = load_pose_with_ligand(str(processed_pdb), str(params_file))
    except Exception as e:
        logger.error(f"{pdb_id}: Pose yükleme hatası: {e}")
        return None

    logger.info(f"  Toplam residue: {pose.total_residue()}")

    # Protein kontakları
    prot_cutoff = cfg["contacts"]["protein_protein_cutoff_A"]
    seq_sep     = cfg["contacts"]["seq_sep_min"]
    contacts = get_protein_contacts(pose, prot_cutoff, seq_sep)
    logger.info(f"  {len(contacts)} protein-protein kontak")

    # Ligand kontakları
    lig_resnum = find_ligand_resnum(pose, ligand_id)
    if lig_resnum is None:
        logger.warning(f"{pdb_id}: Ligand {ligand_id} pose'da bulunamadı!")
        lig_contacts = []
    else:
        lig_cutoff = cfg["contacts"]["ligand_protein_cutoff_A"]
        lig_contacts = get_ligand_contacts(pose, lig_resnum, lig_cutoff)
        logger.info(f"  {len(lig_contacts)} ligand-protein kontak residue")

    # Frustrasyon survey
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
    logger.info(f"  {n_decoys} decoy tamamlandı — {elapsed/60:.1f} dk")

    # Sonuçları kaydet
    results_dir.mkdir(parents=True, exist_ok=True)
    df_frust.to_parquet(result_file, index=False)
    logger.info(f"  Kaydedildi: {result_file}")

    summary = summarize_ligand_frustration(df_frust, lig_contacts)
    return summary


def run_validation(cfg: dict, n_decoys: int):
    """
    Aşama 3: Lysozyme (1LYZ) üzerinde core/surface frustrasyon validasyonu.
    Beklenti: gömülü core kontaklar → minimally frustrated,
              yüzey kontaklar → neutral/highly frustrated.
    """
    import pyrosetta
    from src.frustration import (
        get_protein_contacts,
        run_frustration_survey,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1LYZ indir
    import requests
    pdb_path = Path(cfg["paths"]["raw_pdb"]) / "1LYZ.pdb"
    if not pdb_path.exists():
        r = requests.get("https://files.rcsb.org/download/1LYZ.pdb", timeout=30)
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.write_bytes(r.content)

    # Pose yükle (ligandsız)
    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    logger.info(f"Lysozyme: {pose.total_residue()} residue yüklendi")

    scorefxn = pyrosetta.create_score_function(
        cfg["energy"]["score_function"]
    )

    contacts = get_protein_contacts(
        pose,
        cfg["contacts"]["protein_protein_cutoff_A"],
        cfg["contacts"]["seq_sep_min"],
    )
    logger.info(f"  {len(contacts)} kontak")

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

    # SASA ile gömülülük tahmini (Biopython)
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

    # Görsel: F_index histogramı + SASA ile scatter
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sol: frustrasyon sınıfı dağılımı
    counts = df["frustration_class"].value_counts()
    axes[0].bar(counts.index, counts.values,
                color=["green","gray","red"],
                edgecolor="black")
    axes[0].set_title("Lysozyme Frustrasyon Dağılımı (1LYZ)")
    axes[0].set_ylabel("Kontak sayısı")

    # Sağ: F_index histogramı
    axes[1].hist(df["F_index"], bins=50, color="steelblue", edgecolor="none")
    axes[1].axvline(0.78,  color="green", ls="--", label="min frustrated (>0.78)")
    axes[1].axvline(-1.0,  color="red",   ls="--", label="highly frustrated (<-1)")
    axes[1].set_xlabel("Frustrasyon İndeksi (F_ij)")
    axes[1].set_ylabel("Kontak sayısı")
    axes[1].set_title("Frustrasyon İndeksi Dağılımı")
    axes[1].legend()

    plt.tight_layout()
    fig_path = results_dir / "validation_lysozyme.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Validasyon grafiği kaydedildi: {fig_path}")

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
    Aşama 4: Tüm 25 EGFR yapısı için frustrasyon analizi + korelasyon grafiği.
    """
    import pyrosetta
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    df_cands = pd.read_csv(cfg["paths"]["candidates_csv"])
    scorefxn = pyrosetta.create_score_function(cfg["energy"]["score_function"])

    results = []
    for _, row in df_cands.iterrows():
        pdb_id    = row["pdb_id"]
        ligand_id = row["ligand_id"]
        aff_pM    = row["affinity_pM"]
        aff_type  = row["affinity_type"]

        logger.info(f"\n{'='*50}")
        logger.info(f"İşleniyor: {pdb_id} (ligand={ligand_id}, "
                    f"{aff_type}={aff_pM:.1f} pM)")

        summary = run_single_structure(
            pdb_id, ligand_id, cfg, n_decoys, scorefxn,
            seed=cfg["frustration"]["seed"]
        )
        if summary is None:
            continue

        results.append({
            "pdb_id": pdb_id,
            "ligand_id": ligand_id,
            "affinity_pM": aff_pM,
            "affinity_type": aff_type,
            "log10_affinity_pM": np.log10(max(aff_pM, 1e-3)),
            **summary,
        })
        logger.info(f"  Ligand pocket: {summary['n_minimally_frustrated']} min-frust / "
                    f"{summary['n_contacts_total']} toplam kontak")

    df_res = pd.DataFrame(results)

    # Kaydet
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(results_dir / "egfr_frustration_summary.csv", index=False)
    logger.info(f"\nSonuçlar kaydedildi: {results_dir}/egfr_frustration_summary.csv")

    # Korelasyon grafiği
    _plot_correlation(df_res, results_dir)

    return df_res


def _plot_correlation(df: pd.DataFrame, results_dir: Path):
    """Fig. 5e benzeri scatter plot: min-frustrated contacts vs log(affinity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    if df.empty:
        return

    x = df["n_minimally_frustrated"].values.astype(float)
    y = df["log10_affinity_pM"].values.astype(float)

    # Pearson korelasyonu
    if len(df) >= 3:
        r, p = stats.pearsonr(x, y)
    else:
        r, p = float("nan"), float("nan")

    fig, ax = plt.subplots(figsize=(8, 6))

    # IC50 ve Kd'yi farklı işaretle
    kd_mask = df["affinity_type"].str.upper().isin(["KD","KI"])
    ic50_mask = ~kd_mask

    ax.scatter(x[kd_mask],   y[kd_mask],
               c="royalblue", s=80, zorder=5,
               edgecolors="k", linewidths=0.5, label="Kd/Ki")
    ax.scatter(x[ic50_mask], y[ic50_mask],
               c="coral", marker="s", s=80, zorder=5,
               edgecolors="k", linewidths=0.5, label="IC50")

    # Etiketler
    for _, row in df.iterrows():
        ax.annotate(
            row["pdb_id"],
            (row["n_minimally_frustrated"], row["log10_affinity_pM"]),
            fontsize=7, xytext=(4, 2), textcoords="offset points",
        )

    # Regresyon çizgisi
    if len(df) >= 3 and not np.isnan(r):
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, "k--", lw=1, alpha=0.5)

    ax.set_xlabel("Minimally Frustrated Ligand-Pocket Kontak Sayısı", fontsize=12)
    ax.set_ylabel("log₁₀(Affinity / pM)", fontsize=12)
    ax.set_title(
        f"EGFR Frustrasyon–Affinite Korelasyonu\n"
        f"Pearson r = {r:.3f}, p = {p:.3f} (n={len(df)})",
        fontsize=12,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = results_dir / "egfr_correlation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Korelasyon grafiği: {fig_path}")
    logger.info(f"Pearson r={r:.3f}, p={p:.3f}, n={len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="EGFR Atomistic Frustration Pipeline"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", choices=["validate", "single", "all"],
        default="validate",
        help="validate=lysozyme test, single=tek yapı, all=tüm EGFR",
    )
    parser.add_argument("--pdb_id", default=None)
    parser.add_argument("--n_decoys", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_decoys = args.n_decoys or cfg["frustration"]["n_decoys"]

    # PyRosetta başlat
    import pyrosetta
    pyrosetta.init("-mute all")
    logger.info("PyRosetta başlatıldı.")

    if args.mode == "validate":
        run_validation(cfg, n_decoys)

    elif args.mode == "single":
        if not args.pdb_id:
            parser.error("--mode single için --pdb_id gerekli")
        import pyrosetta
        scorefxn = pyrosetta.create_score_function(cfg["energy"]["score_function"])
        df_cands = pd.read_csv(cfg["paths"]["candidates_csv"])
        row = df_cands[df_cands["pdb_id"] == args.pdb_id]
        if row.empty:
            logger.error(f"{args.pdb_id} aday listesinde yok.")
            return
        lig_id = row.iloc[0]["ligand_id"]
        summary = run_single_structure(
            args.pdb_id, lig_id, cfg, n_decoys, scorefxn,
            seed=cfg["frustration"]["seed"]
        )
        logger.info(f"Sonuç: {summary}")

    elif args.mode == "all":
        run_all_egfr(cfg, n_decoys)


if __name__ == "__main__":
    main()
