"""
Aşama 1: RCSB PDB Data API üzerinden EGFR-inhibitör kompleksleri için
doğrulanmış metadata envanteri oluşturur.

Girdi:  config/pdb_reference_table.csv (Supplementary Fig. 14 listesi:
        pdb_id, paper_minimally_frustrated_contacts,
        paper_highly_frustrated_contacts, affinity_pM)
Çıktı:  results/metadata/egfr_ligand_inventory.csv
        results/metadata/raw_api/{PDB_ID}_*.json  (ham API yanıtları, provenance)
        results/metadata/qc_*.csv  (belirsizlik/uyarı tabloları)

Bu script hiçbir PDB koordinat dosyası indirmez; sadece RCSB Data API
(https://data.rcsb.org/rest/v1/core/...) üzerinden entry/entity/chem_comp
düzeyinde metadata çeker. Yapı dosyalarının indirilmesi Aşama 2'dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("collect_metadata")

RCSB_BASE = "https://data.rcsb.org/rest/v1/core"
REQUEST_DELAY_S = 0.25  # RCSB'ye nazik davranmak için istekler arası bekleme
REQUEST_TIMEOUT_S = 20

EGFR_UNIPROT_ID = "P00533"

# Kristalizasyon ajanları / tampon molekülleri / iyonlar — hiçbiri "hedef inhibitör"
# olarak değerlendirilmeyecek. Bu liste yalnızca ADAY ELEME için kullanılıyor;
# elenen bir bileşen gerçek ligand olamaz demek değil, ama olası değil.
NON_INHIBITOR_BLOCKLIST = {
    "HOH", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4", "1PE", "P6G",
    "DMS", "ACT", "ACY", "MPD", "TRS", "BME", "EPE", "IPA", "IOD", "CL",
    "NA", "K", "MG", "CA", "ZN", "MN", "NI", "CO", "CU", "FE", "BEZ",
    "FMT", "IMD", "NH4", "UNX", "UNK", "SIN", "CIT", "ACE", "MRD",
    "BU3", "BU2", "12P", "15P", "2PE", "MES", "HEPES", "TLA", "OXL",
    "NO3", "BR", "F", "CS", "RB", "LI", "SR", "BA", "CD", "AG", "HG",
    "PB", "PBM", "GD", "EU", "YB", "LU", "LA", "CE", "SM", "PT", "AU",
    "SCN", "AZI", "CO3", "BO4", "PB", "6PE", "PEO", "PEK", "SPD", "SPM",
    "PUT", "DTT", "GSH", "GTT", "TAM", "CXS", "P33", "CAC",
}


@dataclass
class LigandCandidate:
    comp_id: str
    entity_id: str
    name: str | None = None
    formula: str | None = None
    formula_weight: float | None = None
    n_molecules: int | None = None
    auth_asym_ids: list[str] = field(default_factory=list)
    asym_ids: list[str] = field(default_factory=list)
    smiles: str | None = None
    smiles_stereo: str | None = None
    inchikey: str | None = None


class RcsbClient:
    """RCSB Data API için basit, cache'li HTTP istemcisi."""

    def __init__(self, cache_dir: Path, force: bool = False):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.session = requests.Session()

    def _get(self, url: str, cache_key: str) -> dict | None:
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists() and not self.force:
            return json.loads(cache_path.read_text())

        time.sleep(REQUEST_DELAY_S)
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            log.warning("İstek başarısız: %s (%s)", url, exc)
            return None

        if resp.status_code != 200:
            log.warning("HTTP %d: %s", resp.status_code, url)
            return None

        data = resp.json()
        cache_path.write_text(json.dumps(data, indent=2))
        return data

    def entry(self, pdb_id: str) -> dict | None:
        return self._get(f"{RCSB_BASE}/entry/{pdb_id}", f"{pdb_id}_entry")

    def polymer_entity(self, pdb_id: str, entity_id: str) -> dict | None:
        return self._get(
            f"{RCSB_BASE}/polymer_entity/{pdb_id}/{entity_id}",
            f"{pdb_id}_polymer_{entity_id}",
        )

    def nonpolymer_entity(self, pdb_id: str, entity_id: str) -> dict | None:
        return self._get(
            f"{RCSB_BASE}/nonpolymer_entity/{pdb_id}/{entity_id}",
            f"{pdb_id}_nonpolymer_{entity_id}",
        )

    def chemcomp(self, comp_id: str) -> dict | None:
        return self._get(f"{RCSB_BASE}/chemcomp/{comp_id}", f"chemcomp_{comp_id}")

    def nonpolymer_entity_instance(self, pdb_id: str, asym_id: str) -> dict | None:
        return self._get(
            f"{RCSB_BASE}/nonpolymer_entity_instance/{pdb_id}/{asym_id}",
            f"{pdb_id}_nonpolymer_instance_{asym_id}",
        )

    def uniprot_fasta(self, uniprot_id: str) -> str | None:
        cache_path = self.cache_dir / f"uniprot_{uniprot_id}.fasta"
        if cache_path.exists() and not self.force:
            text = cache_path.read_text()
        else:
            time.sleep(REQUEST_DELAY_S)
            try:
                resp = self.session.get(
                    f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta",
                    timeout=REQUEST_TIMEOUT_S,
                )
            except requests.RequestException as exc:
                log.warning("UniProt isteği başarısız: %s (%s)", uniprot_id, exc)
                return None
            if resp.status_code != 200:
                return None
            text = resp.text
            cache_path.write_text(text)
        lines = text.strip().splitlines()
        return "".join(lines[1:]) if len(lines) > 1 else None


def mutation_positions_to_uniprot(
    align_records: list[dict], feature_positions: list[dict]
) -> list[int | None]:
    """Entity seq_id mutasyon konumlarını UniProt numaralandırmasına çevirir."""
    uniprot_positions: list[int | None] = []
    sifts = [a for a in align_records if a.get("reference_database_name") == "UniProt"]
    for pos in feature_positions:
        beg = pos.get("beg_seq_id")
        mapped = None
        for align in sifts:
            for region in align.get("aligned_regions", []):
                e_beg = region["entity_beg_seq_id"]
                e_end = e_beg + region["length"] - 1
                if beg is not None and e_beg <= beg <= e_end:
                    mapped = region["ref_beg_seq_id"] + (beg - e_beg)
                    break
            if mapped is not None:
                break
        uniprot_positions.append(mapped)
    return uniprot_positions


def process_pdb(client: RcsbClient, pdb_id: str) -> dict:
    """Bir PDB kimliği için tüm metadata satırını üretir."""
    pdb_id_u = pdb_id.upper()
    row: dict = {
        "pdb_id": pdb_id_u,
        "egfr_chain": None,
        "uniprot_id": None,
        "mutation_string": None,
        "ligand_comp_id": None,
        "ligand_common_name": None,
        "ligand_systematic_name": None,
        "ligand_chain": None,
        "ligand_resnum": None,
        "ligand_smiles": None,
        "ligand_inchikey": None,
        "covalent_status": "unknown",
        "resolution_A": None,
        "experimental_method": None,
        "deposit_date": None,
        "release_date": None,
        "other_hetero_components": None,
        "n_candidate_ligands": None,
        "n_polymer_entities": None,
        "n_egfr_chains": None,
        "metadata_status": "pending",
        "manual_review_required": False,
        "notes": "",
    }
    notes: list[str] = []

    entry = client.entry(pdb_id_u)
    if entry is None:
        row["metadata_status"] = "ENTRY_FETCH_FAILED"
        row["manual_review_required"] = True
        row["notes"] = "RCSB entry API'sinden veri alınamadı."
        return row

    row["experimental_method"] = entry.get("exptl", [{}])[0].get("method")
    resolution_list = entry.get("rcsb_entry_info", {}).get("resolution_combined")
    row["resolution_A"] = resolution_list[0] if resolution_list else None
    row["deposit_date"] = entry.get("rcsb_accession_info", {}).get("deposit_date")
    row["release_date"] = entry.get("rcsb_accession_info", {}).get(
        "initial_release_date"
    )
    row["title"] = entry.get("struct", {}).get("title")
    inter_mol_covalent = entry.get("rcsb_entry_info", {}).get(
        "inter_mol_covalent_bond_count"
    )
    row["_inter_mol_covalent_bond_count"] = inter_mol_covalent

    ids = entry.get("rcsb_entry_container_identifiers", {})
    polymer_entity_ids = ids.get("polymer_entity_ids") or []
    non_polymer_entity_ids = ids.get("non_polymer_entity_ids") or []
    row["n_polymer_entities"] = len(polymer_entity_ids)

    # --- Polimer entity'ler: EGFR zincirini bul ---
    egfr_chains: list[str] = []
    egfr_uniprot_seen = None
    mutation_strings: list[str] = []
    for pe_id in polymer_entity_ids:
        pe = client.polymer_entity(pdb_id_u, pe_id)
        if pe is None:
            notes.append(f"polymer_entity {pe_id} alınamadı")
            continue
        rp = pe.get("rcsb_polymer_entity", {})
        uniprots = pe.get("rcsb_polymer_entity_container_identifiers", {}).get(
            "uniprot_ids"
        ) or []
        auth_chains = pe.get(
            "rcsb_polymer_entity_container_identifiers", {}
        ).get("auth_asym_ids") or []

        is_egfr = EGFR_UNIPROT_ID in uniprots or (
            rp.get("pdbx_description")
            and "epidermal growth factor receptor" in rp["pdbx_description"].lower()
        )
        if is_egfr:
            egfr_chains.extend(auth_chains)
            egfr_uniprot_seen = EGFR_UNIPROT_ID if EGFR_UNIPROT_ID in uniprots else (
                uniprots[0] if uniprots else None
            )
            align = pe.get("rcsb_polymer_entity_align", [])
            features = pe.get("rcsb_polymer_entity_feature", []) or []
            mut_features = [f for f in features if f.get("type") == "mutation"]
            entity_seq = pe.get("entity_poly", {}).get(
                "pdbx_seq_one_letter_code_can"
            ) or pe.get("entity_poly", {}).get("pdbx_seq_one_letter_code")
            wt_seq = (
                client.uniprot_fasta(egfr_uniprot_seen) if egfr_uniprot_seen else None
            )
            for mf in mut_features:
                fps = mf.get("feature_positions", [])
                mapped = mutation_positions_to_uniprot(align, fps)
                for fp, m in zip(fps, mapped):
                    if m is not None and entity_seq and wt_seq:
                        beg = fp.get("beg_seq_id")
                        mutant_aa = (
                            entity_seq[beg - 1]
                            if beg and beg <= len(entity_seq)
                            else "?"
                        )
                        wt_aa = wt_seq[m - 1] if m <= len(wt_seq) else "?"
                        mutation_strings.append(f"{wt_aa}{m}{mutant_aa}")
                    elif m is not None:
                        mutation_strings.append(f"pos{m}")

    row["egfr_chain"] = ",".join(sorted(set(egfr_chains))) if egfr_chains else None
    row["n_egfr_chains"] = len(set(egfr_chains))
    row["uniprot_id"] = egfr_uniprot_seen
    row["mutation_string"] = (
        ";".join(sorted(set(mutation_strings))) if mutation_strings else None
    )
    if not egfr_chains:
        notes.append("EGFR zinciri UniProt P00533 ile eşleşmedi — MANUEL KONTROL")
        row["manual_review_required"] = True

    # --- Non-polimer entity'ler: ligand adaylarını topla ---
    candidates: list[LigandCandidate] = []
    other_components: list[str] = []
    for npe_id in non_polymer_entity_ids:
        npe = client.nonpolymer_entity(pdb_id_u, npe_id)
        if npe is None:
            notes.append(f"nonpolymer_entity {npe_id} alınamadı")
            continue
        comp_id = npe.get("pdbx_entity_nonpoly", {}).get("comp_id")
        name = npe.get("pdbx_entity_nonpoly", {}).get("name")
        rne = npe.get("rcsb_nonpolymer_entity", {})
        ids_np = npe.get("rcsb_nonpolymer_entity_container_identifiers", {})
        auth_chains_np = ids_np.get("auth_asym_ids") or []

        if comp_id is None:
            continue
        if comp_id in NON_INHIBITOR_BLOCKLIST:
            other_components.append(comp_id)
            continue

        asym_ids_np = ids_np.get("asym_ids") or []
        cand = LigandCandidate(
            comp_id=comp_id,
            entity_id=npe_id,
            name=name,
            formula_weight=rne.get("formula_weight"),
            n_molecules=rne.get("pdbx_number_of_molecules"),
            auth_asym_ids=auth_chains_np,
            asym_ids=asym_ids_np,
        )
        candidates.append(cand)

    row["other_hetero_components"] = ",".join(other_components) if other_components else None
    row["n_candidate_ligands"] = len(candidates)

    if len(candidates) == 0:
        row["metadata_status"] = "NO_LIGAND_FOUND"
        row["manual_review_required"] = True
        notes.append("Blocklist dışında hiçbir non-polimer bileşen bulunamadı.")
    elif len(candidates) > 1:
        row["metadata_status"] = "MULTIPLE_LIGAND_CANDIDATES"
        row["manual_review_required"] = True
        comp_list = ",".join(c.comp_id for c in candidates)
        notes.append(f"Birden fazla aday ligand: {comp_list} — Aşama 3'te geometrik seçim yapılacak.")
        # Şimdilik ilkini birincil aday olarak işaretle; Aşama 3'te doğrulanacak.
        primary = candidates[0]
    else:
        row["metadata_status"] = "OK"
        primary = candidates[0]

    if candidates:
        primary = candidates[0]
        row["ligand_comp_id"] = primary.comp_id
        row["ligand_systematic_name"] = primary.name
        row["ligand_chain"] = ",".join(primary.auth_asym_ids)

        cc = client.chemcomp(primary.comp_id)
        if cc:
            chem_comp = cc.get("chem_comp", {})
            desc = cc.get("rcsb_chem_comp_descriptor", {})
            row["ligand_common_name"] = chem_comp.get("name")
            row["ligand_smiles"] = desc.get("SMILES_stereo") or desc.get("SMILES")
            row["ligand_inchikey"] = desc.get("InChIKey")
        else:
            notes.append(f"chemcomp {primary.comp_id} alınamadı")

        # Instance-düzeyinde otoritatif bilgi: auth_seq_id ve gerçek kovalent bağ anotasyonu.
        # entry-level inter_mol_covalent_bond_count metal koordinasyonu gibi kovalent
        # OLMAYAN bağları da sayabilir; bu yüzden ligand-özel HAS_COVALENT_LINKAGE
        # anotasyonu daha güvenilir birincil kaynaktır.
        resnums = []
        covalent_linkage_found = False
        for asym_id in primary.asym_ids:
            inst = client.nonpolymer_entity_instance(pdb_id_u, asym_id)
            if inst is None:
                continue
            inst_ids = inst.get("rcsb_nonpolymer_entity_instance_container_identifiers", {})
            seq_id = inst_ids.get("auth_seq_id")
            if seq_id is not None:
                resnums.append(str(seq_id))
            annotations = inst.get("rcsb_nonpolymer_instance_annotation", []) or []
            if any(a.get("type") == "HAS_COVALENT_LINKAGE" for a in annotations):
                covalent_linkage_found = True
        row["ligand_resnum"] = ",".join(resnums) if resnums else None

        if covalent_linkage_found:
            row["covalent_status"] = "COVALENT_CONFIRMED"
            row["manual_review_required"] = True
            notes.append(
                "RCSB rcsb_nonpolymer_instance_annotation: HAS_COVALENT_LINKAGE "
                "— ligand EGFR'ye kovalent bağlı (otoritatif kaynak). Aşama 5'te "
                "kovalent params/patch gerekecek."
            )
        elif inter_mol_covalent and inter_mol_covalent > 0:
            # Instance anotasyonu kovalent bağ göstermiyor ama entry-level sayaç > 0:
            # muhtemelen ligand-dışı bir kovalent bağ (örn. disülfid, glikozilasyon).
            row["covalent_status"] = "INTERMOL_COVALENT_BUT_NOT_ON_LIGAND"
            notes.append(
                f"inter_mol_covalent_bond_count={inter_mol_covalent} ama ligand "
                "instance anotasyonunda kovalent bağ yok — bağ ligand dışında olabilir, "
                "Aşama 3'te yapısal kontrol gerekir."
            )
            row["manual_review_required"] = True
        else:
            row["covalent_status"] = "NON_COVALENT"
    else:
        if inter_mol_covalent and inter_mol_covalent > 0:
            row["covalent_status"] = "UNKNOWN_NO_LIGAND"
        else:
            row["covalent_status"] = "NOT_APPLICABLE_NO_LIGAND"

    if row["n_egfr_chains"] and row["n_egfr_chains"] > 1:
        notes.append("Birden fazla EGFR zinciri (asymmetric unit) — Aşama 3'te hangi kopyanın kullanılacağı belirlenecek.")
        row["manual_review_required"] = True

    if row["resolution_A"] is not None and row["resolution_A"] > 3.0:
        notes.append(f"Düşük çözünürlük: {row['resolution_A']} Å")
        row["manual_review_required"] = True

    if row["metadata_status"] == "pending":
        row["metadata_status"] = "OK"

    row["notes"] = " | ".join(notes)
    row["source_url"] = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id_u}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-table",
        type=Path,
        default=Path("config/pdb_reference_table.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metadata"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Önbelleğe alınmış API yanıtlarını yeniden indir.",
    )
    parser.add_argument(
        "--pdb-id",
        type=str,
        default=None,
        help="Tek bir PDB için test amaçlı çalıştır (örn. 5gmp).",
    )
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference_table)
    if args.pdb_id:
        ref_df = ref_df[ref_df["pdb_id"].str.lower() == args.pdb_id.lower()]
        if ref_df.empty:
            raise SystemExit(f"{args.pdb_id} referans tablosunda bulunamadı.")

    cache_dir = args.output_dir / "raw_api"
    client = RcsbClient(cache_dir=cache_dir, force=args.force)

    rows = []
    for i, pdb_id in enumerate(ref_df["pdb_id"], start=1):
        log.info("[%d/%d] %s işleniyor...", i, len(ref_df), pdb_id)
        row = process_pdb(client, pdb_id)
        # Paper referans değerlerini ekle (yalnızca karşılaştırma için — sonuç değil)
        ref_row = ref_df[ref_df["pdb_id"].str.lower() == pdb_id.lower()].iloc[0]
        row["paper_minimally_frustrated_contacts"] = ref_row[
            "paper_minimally_frustrated_contacts"
        ]
        row["paper_highly_frustrated_contacts"] = ref_row[
            "paper_highly_frustrated_contacts"
        ]
        row["affinity_pM"] = ref_row["affinity_pM"]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pdb_id:
        # Tek-yapı test modu: ana envanteri asla üzerine yazma.
        inventory_path = args.output_dir / f"test_single_{args.pdb_id.lower()}.csv"
    else:
        inventory_path = args.output_dir / "egfr_ligand_inventory.csv"
    out_df.to_csv(inventory_path, index=False)
    log.info("Envanter yazıldı: %s (%d satır)", inventory_path, len(out_df))

    # --- QC alt tabloları ---
    qc_specs = {
        "qc_no_ligand_found.csv": out_df["metadata_status"] == "NO_LIGAND_FOUND",
        "qc_multiple_ligand_candidates.csv": out_df["metadata_status"]
        == "MULTIPLE_LIGAND_CANDIDATES",
        "qc_suspected_covalent.csv": out_df["covalent_status"] == "SUSPECTED_COVALENT",
        "qc_multiple_egfr_chains.csv": out_df["n_egfr_chains"].fillna(0) > 1,
        "qc_low_resolution.csv": out_df["resolution_A"].fillna(0) > 3.0,
        "qc_entry_fetch_failed.csv": out_df["metadata_status"] == "ENTRY_FETCH_FAILED",
        "qc_manual_review_required.csv": out_df["manual_review_required"] == True,  # noqa: E712
    }
    for fname, mask in qc_specs.items():
        subset = out_df[mask]
        subset.to_csv(args.output_dir / fname, index=False)
        log.info("  %s: %d satır", fname, len(subset))

    n_total = len(ref_df)
    n_covered = len(out_df)
    log.info("Toplam referans PDB: %d, işlenen: %d", n_total, n_covered)
    if n_total != n_covered:
        log.warning("UYARI: referans tablo ile çıktı satır sayısı uyuşmuyor!")


if __name__ == "__main__":
    main()
