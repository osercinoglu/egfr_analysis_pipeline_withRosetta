"""
EGFR kinase domain - small molecule inhibitor complexes tarama scripti.

RCSB PDB Search API + ChEMBL API + BindingDB kullanarak:
  1. Human EGFR (UniProt P00533) kristal yapılarını listeler
  2. Küçük moleküllü (MW 200-800 Da, ilaç benzeri) ligandları filtreler
  3. ChEMBL üzerinden Kd/Ki binding affinite verisi çeker
  4. Sonuçları CSV'e yazar

Çalıştırma:
  python src/search_egfr_structures.py --out results/egfr_candidates.csv
"""

import requests
import json
import time
import argparse
import pandas as pd
from pathlib import Path

RCSB_SEARCH  = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA    = "https://data.rcsb.org/rest/v1/core"
CHEMBL_BASE  = "https://www.ebi.ac.uk/chembl/api/data"
BDB_BASE     = "https://www.bindingdb.org/axis2/services/BDBService"

# Kristalleştirme katkıları / buffer bileşenleri - dışarıda tutulacak
CRYSTAL_EXCL = {
    "HOH","SO4","GOL","PEG","EDO","MPD","BME","DTT","PO4","MES","TRS",
    "ACE","ACT","FMT","IMD","IPA","DMS","EOH","CIT","TAR","MRD","EPE",
    "HED","1PE","2PE","MOH","P6G","BTB","BU1","BU2","BU3","C8E","NHE",
    "BOG","OGA","XPE","12P","15P","7PE","SIN","SPM","SPD","PUT","TRD",
    "PDO","PGE","PGO","PG4","PCM","CO3","IOD","CA","MG","ZN","CU","FE",
    "MN","CL","K","NA","BR","IOD","F","CD","NI","CO","HG","AU","AG",
    "MPO","ADP","ATP","GTP","GDP","AMP","GMP","APC","ANP","3AN",
    "NO3","ACY","ACO","CMO","NH4","LI","RB","CS","SR","BA","SO3",
    "SUL","CLO","FLO","BRO","PER","TFA","MOO","TNG","EGL","DVT",
    "PGO","TLA","FLC","OXL","FRU","GAL","GLA","XYP","MAN","MLI",
    "CE1","CE","CM","BCT","HIC","MSE","SEP","TPO","PTR","NEP","CSO",
    "OCS","KCX","MLY","M3L","LLP","PYL","SEC","FME","CME","CSD","CSX",
    "XX1", "UNL", "UNX", "DRG",
}

# Makaledeki 4 yapı - bunlar zaten var, listeye eklenecek
KNOWN = [
    {"pdb_id": "5GMP", "ligand_id": "F62",  "kd_pM": 0.8,   "inhibitor": "XTF-262"},
    {"pdb_id": "1XKK", "ligand_id": "FMM",  "kd_pM": 3.0,   "inhibitor": "GW572016"},
    {"pdb_id": "5EM8", "ligand_id": "5Q4",  "kd_pM": 1090.0,"inhibitor": "5Q4"},
    {"pdb_id": "5UGB", "ligand_id": "8BM",  "kd_pM": 161.0, "inhibitor": "8BM"},
]


def search_rcsb_egfr_structures() -> list[str]:
    """RCSB'den human EGFR (P00533) X-ray kristal yapılarını döndürür."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": "P00533",
                        "negation": False
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                        "operator": "exact_match",
                        "value": "UniProt",
                        "negation": False
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 2000},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }
    r = requests.post(RCSB_SEARCH, json=query, timeout=60)
    r.raise_for_status()
    data = r.json()
    ids = [h["identifier"] for h in data.get("result_set", [])]
    print(f"  RCSB: {len(ids)} EGFR X-ray yapısı bulundu")
    return ids


def get_entry_ligands(pdb_id: str) -> list[dict]:
    """Bir PDB entry'sindeki non-polymer (small molecule) ligandları döndürür."""
    url = f"{RCSB_DATA}/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
        # nonpolymer entity listesi
        np_ids = data.get("rcsb_entry_container_identifiers", {}).get(
            "non_polymer_entity_ids", []
        ) or []
        ligands = []
        for np_id in np_ids:
            np_url = f"{RCSB_DATA}/nonpolymer_entity/{pdb_id}/{np_id}"
            rr = requests.get(np_url, timeout=20)
            if rr.status_code != 200:
                continue
            nd = rr.json()
            chem_comp = nd.get("pdbx_entity_nonpoly", {})
            comp_id = chem_comp.get("comp_id", "")
            name    = chem_comp.get("name", "")
            formula_weight = nd.get("rcsb_nonpolymer_entity", {}).get(
                "formula_weight", 0
            ) or 0
            if comp_id and comp_id not in CRYSTAL_EXCL:
                ligands.append({
                    "comp_id": comp_id,
                    "name": name,
                    "mw": float(formula_weight)
                })
        return ligands
    except Exception:
        return []


def get_rcsb_binding_affinity(pdb_id: str, comp_id: str) -> dict | None:
    """
    RCSB'nin binding affinity veritabanından (BindingDB + ChEMBL + MOAD aggregate)
    Kd/Ki/IC50 verisi çeker.
    """
    url = f"{RCSB_DATA}/nonpolymer_entity_instance/{pdb_id}/A/1"
    # nonpolymer instance'larını gezmek yerine doğrudan binding affinity endpoint'i dene
    url2 = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(url2, timeout=20)
        if r.status_code != 200:
            return None
        # rcsb_binding_affinity alanı entry-level'da değil, nonpolymer_entity_instance'da
        # Alternatif: rcsb_entry_info içindeki binding affinity bazen burada
        return None
    except Exception:
        return None


def query_chembl_for_pdb(pdb_id: str, comp_id: str) -> list[dict]:
    """
    ChEMBL API üzerinden bir PDB ligandının EGFR üzerindeki Kd/Ki verilerini çeker.
    PDB comp_id -> ChEMBL molecule cross-reference -> binding assay.
    """
    results = []
    try:
        # 1) PDB ligand kodu -> ChEMBL molecule
        url = f"{CHEMBL_BASE}/molecule.json?xref_src=PDB&xref_id={comp_id}&format=json"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        mols = r.json().get("molecules", [])
        if not mols:
            return []
        chembl_id = mols[0]["molecule_chembl_id"]

        # 2) ChEMBL activity endpoint: EGFR (CHEMBL203) + Kd/Ki
        # EGFR ChEMBL target ID: CHEMBL203
        url2 = (
            f"{CHEMBL_BASE}/activity.json?"
            f"molecule_chembl_id={chembl_id}"
            f"&target_chembl_id=CHEMBL203"
            f"&standard_type__in=Kd,Ki"
            f"&limit=10&format=json"
        )
        r2 = requests.get(url2, timeout=20)
        if r2.status_code != 200:
            return []
        acts = r2.json().get("activities", [])
        for a in acts:
            val  = a.get("standard_value")
            unit = a.get("standard_units", "")
            typ  = a.get("standard_type", "")
            rel  = a.get("standard_relation", "=")
            if val is None:
                continue
            # nM -> pM dönüşümü
            val_f = float(val)
            if unit == "nM":
                val_pM = val_f * 1000
            elif unit == "pM":
                val_pM = val_f
            elif unit == "uM":
                val_pM = val_f * 1e6
            elif unit == "mM":
                val_pM = val_f * 1e9
            else:
                continue
            results.append({
                "chembl_id": chembl_id,
                "affinity_type": typ,
                "affinity_pM": val_pM,
                "relation": rel,
            })
    except Exception:
        pass
    return results


def query_bindingdb_pdb(pdb_id: str) -> list[dict]:
    """BindingDB REST API: PDB ID'ye göre binding data çeker."""
    results = []
    try:
        url = (
            f"{BDB_BASE}/getLigandsByPDB"
            f"?pdb={pdb_id}&response=application/json"
        )
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
        entries = data.get("getLigandsByPDBResponse", {}).get("affinities", [])
        if isinstance(entries, dict):
            entries = [entries]
        for e in entries:
            affinity_str = e.get("affinity", "")
            affinity_type = e.get("affinityType", "")
            # parse value
            try:
                val_nM = float(affinity_str.replace(">","").replace("<","").replace("~","").strip())
            except (ValueError, AttributeError):
                continue
            if affinity_type.upper() in ("KD", "KI"):
                results.append({
                    "affinity_type": affinity_type.upper(),
                    "affinity_pM": val_nM * 1000,
                    "relation": ">=" if ">" in str(affinity_str) else
                                "<=" if "<" in str(affinity_str) else "=",
                })
    except Exception:
        pass
    return results


def query_rcsb_binding_affinity_v2(pdb_id: str) -> list[dict]:
    """
    RCSB'nin /rest/v1/core/entry/{id} endpoint'indeki
    rcsb_binding_affinity aggregate datasını okur.
    """
    url = f"https://data.rcsb.org/graphql"
    query = """
    {
      entry(entry_id: "%s") {
        rcsb_binding_affinity {
          comp_id
          value
          unit
          type
          link
          provenance_code
        }
      }
    }
    """ % pdb_id
    try:
        r = requests.post(url, json={"query": query}, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        affs = (data.get("data", {}).get("entry", {}) or {}).get(
            "rcsb_binding_affinity", []
        ) or []
        results = []
        for a in affs:
            val  = a.get("value")
            unit = a.get("unit", "")
            typ  = a.get("type", "")
            comp = a.get("comp_id", "")
            if val is None:
                continue
            val_f = float(val)
            if unit in ("nM", "nmol/L"):
                val_pM = val_f * 1000
            elif unit in ("pM", "pmol/L"):
                val_pM = val_f
            elif unit in ("uM", "umol/L", "µM"):
                val_pM = val_f * 1e6
            elif unit in ("mM", "mmol/L"):
                val_pM = val_f * 1e9
            else:
                # unit bazen boş ya da farklı
                val_pM = val_f  # bilinmiyor, ham değer
            results.append({
                "comp_id": comp,
                "affinity_type": typ,
                "affinity_pM": val_pM,
                "unit_raw": unit,
            })
        return results
    except Exception:
        return []


def get_entry_resolution(pdb_id: str) -> float:
    """PDB entry'sinin çözünürlüğünü döndürür (Angstrom)."""
    url = f"https://data.rcsb.org/graphql"
    query = '{ entry(entry_id: "%s") { rcsb_entry_info { resolution_combined } } }' % pdb_id
    try:
        r = requests.post(url, json={"query": query}, timeout=15)
        data = r.json()
        res_list = (
            data.get("data", {}).get("entry", {}) or {}
        ).get("rcsb_entry_info", {}).get("resolution_combined", [None])
        if res_list:
            return float(res_list[0])
    except Exception:
        pass
    return 99.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/egfr_candidates.csv")
    parser.add_argument("--res_cutoff", type=float, default=2.8,
                        help="Max resolution (Å)")
    parser.add_argument("--mw_min", type=float, default=200,
                        help="Ligand min MW (Da)")
    parser.add_argument("--mw_max", type=float, default=900,
                        help="Ligand max MW (Da)")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("=== EGFR inhibitör kompleksi tarama ===")
    print(f"Parametreler: res ≤ {args.res_cutoff} Å, MW {args.mw_min}-{args.mw_max} Da")

    # 1. RCSB'den tüm EGFR yapılarını al
    print("\n[1] RCSB PDB taranıyor...")
    all_ids = search_rcsb_egfr_structures()

    # 2. Her yapı için ligand + affinite verisi
    print(f"\n[2] {len(all_ids)} yapı için ligand ve affinite verisi çekiliyor...")
    rows = []
    known_ids = {k["pdb_id"] for k in KNOWN}

    for i, pdb_id in enumerate(all_ids):
        # İlerleme
        if i % 20 == 0:
            print(f"  {i}/{len(all_ids)} işlendi, {len(rows)} aday bulundu...")

        # a) RCSB binding affinity (GraphQL) — en hızlı yol
        rcsb_affs = query_rcsb_binding_affinity_v2(pdb_id)
        if not rcsb_affs:
            continue  # affinite verisi olmayan yapıları atla

        # b) Ligand bilgisi
        ligands = get_entry_ligands(pdb_id)
        drug_ligands = [
            l for l in ligands
            if args.mw_min <= l["mw"] <= args.mw_max
        ]
        if not drug_ligands:
            continue

        # c) Çözünürlük filtresi
        resolution = get_entry_resolution(pdb_id)
        if resolution > args.res_cutoff:
            continue

        # d) Affinite eşleştir
        for lig in drug_ligands:
            # RCSB affiniteleri içinde bu ligandı ara
            matched = [a for a in rcsb_affs if a.get("comp_id", "") == lig["comp_id"]]
            if not matched:
                # comp_id boşsa tüm affiniteleri al
                matched = [a for a in rcsb_affs if not a.get("comp_id")]

            for aff in matched:
                aff_type = aff.get("affinity_type", "?")
                if aff_type.upper() not in ("KD", "KI", "IC50", "EC50"):
                    continue
                rows.append({
                    "pdb_id": pdb_id,
                    "ligand_id": lig["comp_id"],
                    "ligand_name": lig["name"],
                    "ligand_mw": lig["mw"],
                    "resolution_A": resolution,
                    "affinity_type": aff_type,
                    "affinity_pM": aff.get("affinity_pM", None),
                    "unit_raw": aff.get("unit_raw", ""),
                    "source": "RCSB",
                })

        time.sleep(0.05)  # rate limit

    # 3. Bilinen 4 yapıyı ekle (RCSB'de affinite olmasa bile)
    for k in KNOWN:
        if not any(r["pdb_id"] == k["pdb_id"] for r in rows):
            rows.append({
                "pdb_id": k["pdb_id"],
                "ligand_id": k["ligand_id"],
                "ligand_name": k["inhibitor"],
                "ligand_mw": None,
                "resolution_A": None,
                "affinity_type": "Kd",
                "affinity_pM": k["kd_pM"],
                "unit_raw": "pM",
                "source": "paper",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n[!] Hiç sonuç bulunamadı.")
        return

    # 4. Kd > Ki > IC50 öncelik sırası; duplicate pdb_id için en iyi affinity_type seç
    priority = {"KD": 0, "KI": 1, "IC50": 2, "EC50": 3}
    df["priority"] = df["affinity_type"].str.upper().map(priority).fillna(99)
    df = df.sort_values(["pdb_id", "priority", "affinity_pM"])
    df = df.drop_duplicates(subset=["pdb_id", "ligand_id"], keep="first")
    df = df.drop(columns=["priority"])

    # 5. log10(Kd pM) ekle
    import numpy as np
    df["log10_affinity_pM"] = np.log10(df["affinity_pM"].clip(lower=1e-3))

    # 6. Kaydet ve özetle
    df.to_csv(args.out, index=False)
    print(f"\n=== Sonuç: {len(df)} aday yapı ===")
    print(df[["pdb_id","ligand_id","affinity_type","affinity_pM","resolution_A"]].to_string(index=False))
    print(f"\nCSV kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
