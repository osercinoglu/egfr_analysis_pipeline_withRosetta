"""
Atomistic Frustration Pipeline — Çekirdek Motor

Chen et al., Nature Communications 11, 5944 (2020)
DOI: 10.1038/s41467-020-19560-9

Eq. 1: Frustrasyon indeksi (Z-skoru)
    F_ij = (E_ij_native - mean(E_ij_decoy)) / std(E_ij_decoy)

Eq. 2: Many-body pairwise enerji düzeltmesi
    E_ij = e_ij
         + 0.5 * Σ_{k ∈ contacts(i), k≠j} e_ik
         + 0.5 * Σ_{l ∈ contacts(j), l≠i} e_jl
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PyRosetta — kurulu değilse uyarı ver, import sonrası çalışır
try:
    import pyrosetta
    from pyrosetta import Pose
    from pyrosetta.rosetta.core.scoring import ScoreFunction
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        RestrictToRepacking,
        PreventRepacking,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    from pyrosetta.rosetta.protocols.relax import FastRelax
    _PYROSETTA_AVAILABLE = True
except ImportError:
    _PYROSETTA_AVAILABLE = False
    logger.warning("PyRosetta bulunamadı. Sadece veri hazırlık adımları çalışır.")


# ---------------------------------------------------------------------------
# Yardımcı: PyRosetta başlat
# ---------------------------------------------------------------------------

def init_pyrosetta(silent: bool = True):
    """PyRosetta'yı başlatır (tekrar çağrılsa sorun olmaz)."""
    if not _PYROSETTA_AVAILABLE:
        raise RuntimeError("PyRosetta kurulu değil.")
    flags = "-mute all" if silent else ""
    pyrosetta.init(flags)


# ---------------------------------------------------------------------------
# 1. Kontak listeleri
# ---------------------------------------------------------------------------

def get_protein_contacts(
    pose: "Pose",
    cutoff: float = 10.0,
    seq_sep_min: int = 4,
) -> list[tuple[int, int]]:
    """
    Cα-Cα mesafesine göre protein-protein kontak çiftleri döndürür.

    Yalnızca standart amino asit residue'leri dikkate alır.
    |i - j| < seq_sep_min olan çiftler atlanır (yerel backbone bağları).

    Args:
        pose: PyRosetta Pose nesnesi
        cutoff: Cα-Cα kesme mesafesi (Å)
        seq_sep_min: Minimum sıra ayrımı

    Returns:
        (resi, resj) indeks çiftleri listesi (1-indeksli, i < j)
    """
    n = pose.total_residue()
    contacts = []
    for i in range(1, n + 1):
        ri = pose.residue(i)
        if not ri.is_protein():
            continue
        if not ri.has("CA"):
            continue
        ca_i = ri.xyz("CA")
        for j in range(i + seq_sep_min, n + 1):
            rj = pose.residue(j)
            if not rj.is_protein():
                continue
            if not rj.has("CA"):
                continue
            ca_j = rj.xyz("CA")
            dist = ca_i.distance(ca_j)
            if dist <= cutoff:
                contacts.append((i, j))
    return contacts


def get_ligand_contacts(
    pose: "Pose",
    ligand_resnum: int,
    cutoff: float = 10.0,
) -> list[int]:
    """
    Ligand heavy-atom — protein Cα mesafesine göre kontak yapan protein
    residue'lerinin indekslerini döndürür.

    Herhangi bir ligand heavy-atom'u bir protein Cα'sına cutoff Å'dan
    yakınsa o residue "ligand kontağı" sayılır.

    Args:
        pose: PyRosetta Pose nesnesi
        ligand_resnum: Ligandin pose içindeki residue numarası
        cutoff: heavy-atom – Cα kesme mesafesi (Å)

    Returns:
        Protein residue numaraları listesi (1-indeksli)
    """
    lig_res = pose.residue(ligand_resnum)
    n = pose.total_residue()
    contact_residues = []

    # Ligandın tüm heavy atom koordinatları
    lig_coords = []
    for atom_idx in range(1, lig_res.natoms() + 1):
        atom = lig_res.atom(atom_idx)
        # Hidrojen atla (atom_type_index 18 genelde H, ya da element_type kontrolü)
        if lig_res.atom_is_hydrogen(atom_idx):
            continue
        lig_coords.append(lig_res.xyz(atom_idx))

    for j in range(1, n + 1):
        if j == ligand_resnum:
            continue
        rj = pose.residue(j)
        if not rj.is_protein():
            continue
        if not rj.has("CA"):
            continue
        ca_j = rj.xyz("CA")
        for lc in lig_coords:
            if lc.distance(ca_j) <= cutoff:
                contact_residues.append(j)
                break  # Bu residue için yeterli, diğer atomlara bakma

    return sorted(set(contact_residues))


# ---------------------------------------------------------------------------
# 2. Enerji hesaplama (Eq. 2)
# ---------------------------------------------------------------------------

def _get_energy_graph(pose: "Pose", scorefxn: "ScoreFunction") -> Any:
    """Pose'u skorla ve EnergyGraph nesnesini döndür."""
    scorefxn(pose)
    return pose.energies().energy_graph()


_FA_REP_ST = None  # ScoreType önbelleği

def _fa_rep_score_type():
    """fa_rep ScoreType nesnesini döndürür (lazy init)."""
    global _FA_REP_ST
    if _FA_REP_ST is None:
        from pyrosetta.rosetta.core.scoring import fa_rep
        _FA_REP_ST = fa_rep
    return _FA_REP_ST


def pairwise_energy(
    pose: "Pose",
    scorefxn: "ScoreFunction",
    i: int,
    j: int,
    exclude_fa_rep: bool = True,
) -> float:
    """
    Residue i ve j arasındaki doğrudan pairwise enerjiyi döndürür.

    EnergyGraph üzerinden edge (i,j) toplamı alınır; fa_rep hariç tutulabilir.
    Bu, Eq. 2'deki e_ij terimine karşılık gelir.

    Args:
        pose: Skor edilmiş Pose
        scorefxn: Aktif ScoreFunction
        i, j: Residue indeksleri (1-tabanlı)
        exclude_fa_rep: True → fa_rep terimi toplamdan çıkarılır

    Returns:
        e_ij (Rosetta enerji birimi, kcal/mol~)
    """
    graph = _get_energy_graph(pose, scorefxn)
    edge = graph.find_energy_edge(i, j)
    if edge is None:
        return 0.0

    total = edge.dot(scorefxn.weights())
    if exclude_fa_rep:
        fa_rep_val = edge[_fa_rep_score_type()]
        total -= fa_rep_val * scorefxn.get_weight(_fa_rep_score_type())
    return total


def contact_energy_eq2(
    pose: "Pose",
    scorefxn: "ScoreFunction",
    i: int,
    j: int,
    contact_partners_i: list[int],
    contact_partners_j: list[int],
    exclude_fa_rep: bool = True,
) -> float:
    """
    Eq. 2 many-body enerji düzeltmesi:
        E_ij = e_ij
             + 0.5 * Σ_{k ∈ contacts(i), k≠j} e_ik
             + 0.5 * Σ_{l ∈ contacts(j), l≠i} e_jl

    Args:
        contact_partners_i: i'nin j hariç tüm kontak partnerleri
        contact_partners_j: j'nin i hariç tüm kontak partnerleri

    Returns:
        E_ij (many-body düzeltmeli pairwise enerji)
    """
    e_ij = pairwise_energy(pose, scorefxn, i, j, exclude_fa_rep)

    background_i = 0.0
    for k in contact_partners_i:
        if k != j:
            background_i += pairwise_energy(pose, scorefxn, i, k, exclude_fa_rep)

    background_j = 0.0
    for l_ in contact_partners_j:
        if l_ != i:
            background_j += pairwise_energy(pose, scorefxn, j, l_, exclude_fa_rep)

    return e_ij + 0.5 * background_i + 0.5 * background_j


def build_contact_partner_map(
    contacts: list[tuple[int, int]],
) -> dict[int, list[int]]:
    """
    Kontak listesinden her residue için partner listesi dict'i oluşturur.

    contacts: [(i,j), ...] biçiminde kontak çiftleri
    Returns: {resnum: [partner1, partner2, ...]}
    """
    partners: dict[int, list[int]] = {}
    for i, j in contacts:
        partners.setdefault(i, []).append(j)
        partners.setdefault(j, []).append(i)
    return partners


# ---------------------------------------------------------------------------
# 3. Decoy üretimi
# ---------------------------------------------------------------------------

def native_aa_frequency(pose: "Pose") -> dict[str, float]:
    """
    Protein-spesifik, pozisyon-bağımsız amino asit frekans dağılımı.

    Tüm protein residue'lerinin bir-harfli kodlarını sayarak
    20 amino asit için normalize edilmiş frekans sözlüğü döndürür.

    Returns:
        {'A': 0.08, 'C': 0.02, ...}  (toplam = 1.0)
    """
    from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code
    counts: dict[str, int] = {}
    total = 0
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue
        aa = res.name1()
        counts[aa] = counts.get(aa, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {aa: cnt / total for aa, cnt in counts.items()}


def generate_decoy(
    pose: "Pose",
    scorefxn: "ScoreFunction",
    aa_freq: dict[str, float],
    seed: int | None = None,
) -> "Pose":
    """
    Tek bir decoy pose üretir:
      1. Her protein pozisyonunu aa_freq dağılımından örneklenen amino asitle değiştir
      2. Backbone sabit tutarak side-chain'leri repack et
      3. 1 döngü backbone-sabit FastRelax (yan zincir çarpışmalarını gider)

    Args:
        pose: Orijinal native pose (değiştirilmez — kopya üzerinde çalışılır)
        scorefxn: Skor fonksiyonu
        aa_freq: native_aa_frequency() çıktısı
        seed: Rastgelelik tohumu (None → global state kullanılır)

    Returns:
        Yeni decoy Pose nesnesi (backbone = native, sequence = shuffle)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

    decoy = pose.clone()
    n = decoy.total_residue()

    # Amino asit adları (tek harf → 3 harf + tam Rosetta adı)
    aa_letters = list(aa_freq.keys())
    aa_probs   = list(aa_freq.values())

    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    for i in range(1, n + 1):
        res = decoy.residue(i)
        if not res.is_protein():
            continue
        new_aa_1let = np.random.choice(aa_letters, p=aa_probs)
        # Tek harften Rosetta residue type adına dönüştür
        from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code, name_from_aa
        aa_enum = aa_from_oneletter_code(new_aa_1let)
        res_name = name_from_aa(aa_enum)
        mutator = MutateResidue(i, res_name)
        mutator.apply(decoy)

    # Side-chain repack (backbone sabit)
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    packer = PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(decoy)

    # Kısa backbone-sabit relaxasyon
    relax = FastRelax(scorefxn, 1)
    relax.set_movemap_factory(None)  # varsayılan hareket haritası
    # Backbone dondur: sadece side-chain minimize et
    from pyrosetta.rosetta.core.kinematics import MoveMap
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(True)
    mm.set_jump(False)
    relax.set_movemap(mm)
    relax.apply(decoy)

    return decoy


# ---------------------------------------------------------------------------
# 4. Ana frustrasyon survey
# ---------------------------------------------------------------------------

def run_frustration_survey(
    pose: "Pose",
    scorefxn: "ScoreFunction",
    contacts: list[tuple[int, int]],
    n_decoys: int = 1000,
    seed: int = 0,
    exclude_fa_rep: bool = True,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """
    Frustrasyon indeksini (Eq. 1) tüm kontak çiftleri için hesaplar.

    Algoritma:
      1. Native pose için tüm kontak E_ij değerlerini hesapla (Eq. 2)
      2. n_decoys adet decoy üret, her decoy'dan tüm E_ij'leri topla
      3. Her kontak için Z-skoru hesapla (Eq. 1)

    Checkpoint mekanizması:
      checkpoint_path belirtilirse her checkpoint_every decoy'da ara sonuç
      pickle'a kaydedilir; kaldığı yerden devam edilebilir.

    Args:
        contacts: get_protein_contacts() veya get_ligand_contacts() çıktısı
        n_decoys: Üretilecek decoy sayısı
        seed: Başlangıç rastgelelik tohumu
        checkpoint_path: Checkpoint dosyası yolu (None → kaydetme)
        checkpoint_every: Kaç decoy'da bir checkpoint al

    Returns:
        DataFrame sütunları:
          resi, resj, F_index, E_native, decoy_mean, decoy_std,
          frustration_class
    """
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(iterable, **kwargs):  # type: ignore
            return iterable

    # Kontak partner haritası (Eq. 2 için)
    partner_map = build_contact_partner_map(contacts)

    # --- 1. Native enerjiler ---
    logger.info("Native enerjiler hesaplanıyor...")
    scorefxn(pose)
    native_energies: dict[tuple[int, int], float] = {}
    for i, j in contacts:
        pi = partner_map.get(i, [])
        pj = partner_map.get(j, [])
        native_energies[(i, j)] = contact_energy_eq2(
            pose, scorefxn, i, j, pi, pj, exclude_fa_rep
        )

    # --- 2. Checkpoint yükle (varsa) ---
    decoy_energies: dict[tuple[int, int], list[float]] = {
        c: [] for c in contacts
    }
    start_decoy = 0

    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        decoy_energies = ckpt["decoy_energies"]
        start_decoy = ckpt["completed_decoys"]
        logger.info(f"Checkpoint yüklendi: {start_decoy} decoy tamamlanmış.")

    # --- 3. Decoy döngüsü ---
    aa_freq = native_aa_frequency(pose)

    for d_idx in _tqdm(
        range(start_decoy, n_decoys),
        desc="Decoys",
        initial=start_decoy,
        total=n_decoys,
    ):
        decoy_seed = seed + d_idx
        decoy = generate_decoy(pose, scorefxn, aa_freq, seed=decoy_seed)
        scorefxn(decoy)

        for i, j in contacts:
            pi = partner_map.get(i, [])
            pj = partner_map.get(j, [])
            e = contact_energy_eq2(
                decoy, scorefxn, i, j, pi, pj, exclude_fa_rep
            )
            decoy_energies[(i, j)].append(e)

        # Checkpoint
        if (
            checkpoint_path
            and (d_idx + 1) % checkpoint_every == 0
        ):
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "wb") as f:
                pickle.dump(
                    {"decoy_energies": decoy_energies,
                     "completed_decoys": d_idx + 1},
                    f
                )

    # --- 4. Eq. 1: Z-skoru ---
    rows = []
    for i, j in contacts:
        e_nat  = native_energies[(i, j)]
        vals   = np.array(decoy_energies[(i, j)])
        if len(vals) < 2:
            continue
        mu     = vals.mean()
        sigma  = vals.std(ddof=1)
        if sigma < 1e-9:
            f_idx = 0.0
        else:
            f_idx = (e_nat - mu) / sigma

        # Frustrasyon sınıfı
        if f_idx > 0.78:
            frust_class = "minimally_frustrated"
        elif f_idx < -1.0:
            frust_class = "highly_frustrated"
        else:
            frust_class = "neutral"

        rows.append({
            "resi": i,
            "resj": j,
            "F_index": f_idx,
            "E_native": e_nat,
            "decoy_mean": mu,
            "decoy_std": sigma,
            "frustration_class": frust_class,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. EGFR analizi: ligand-pocket frustrasyon özeti
# ---------------------------------------------------------------------------

def summarize_ligand_frustration(
    df_frustration: pd.DataFrame,
    ligand_contacts: list[int],
) -> dict:
    """
    Ligand-pocket residue'lerini içeren kontak çiftlerini filtreler ve
    frustrasyon dağılımını özetler.

    'Ligand kontağı' = kontak çiftinin en az bir tarafı
    ligand_contacts listesindedir.

    Returns:
        {
          'n_contacts_total': int,
          'n_minimally_frustrated': int,
          'n_neutral': int,
          'n_highly_frustrated': int,
          'frac_minimally': float,
        }
    """
    mask = (
        df_frustration["resi"].isin(ligand_contacts) |
        df_frustration["resj"].isin(ligand_contacts)
    )
    sub = df_frustration[mask]
    n_total = len(sub)
    n_min   = (sub["frustration_class"] == "minimally_frustrated").sum()
    n_neu   = (sub["frustration_class"] == "neutral").sum()
    n_high  = (sub["frustration_class"] == "highly_frustrated").sum()
    return {
        "n_contacts_total": n_total,
        "n_minimally_frustrated": int(n_min),
        "n_neutral": int(n_neu),
        "n_highly_frustrated": int(n_high),
        "frac_minimally": float(n_min / n_total) if n_total > 0 else 0.0,
    }
