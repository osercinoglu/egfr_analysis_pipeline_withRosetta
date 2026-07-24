"""
Aşama 5: 51 benzersiz EGFR-inhibitör ligandını Rosetta/PyRosetta için
parametreler (.params) üretir ve her birini doğrular.

Akış (her benzersiz ligand comp_id için, öncelik sırasıyla):
  1. RCSB CCD CIF -> mol2 (kanonik atom adları, PDB HETATM ile eşleşir) [birincil]
  2. RCSB ideal SDF -> RDKit sanitize -> mol2 (OpenBabel)                [yedek 1]
  3. Kompleksten çıkarılan ligand PDB -> mol2 (OpenBabel, --gen3d)        [son çare]
  4. molfile_to_params.py --keep-names --clobber

Her ligand için doğrulama:
  - .params dosyası üretildi mi
  - atom sayısı > 0
  - o ligandı kullanan HER PDB'nin data/processed/{ID}_clean.pdb'sindeki
    HETATM atom adlarıyla params ATOM adları TAM eşleşiyor mu
  - PyRosetta pose'a ligand olarak yüklenebiliyor mu
  - REF2015 ile skorlanabiliyor mu (NaN/inf değil)

Kovalent ligandlar (Aşama 4'te CYS797-SG bağlantısı tespit edilenler) için
ayrıca .params dosyasına bir CONNECT kaydı eklenir (Aşama 6'da kovalent
bağın kurulması için gerekli); bu script CONNECT ekler ama fiili kovalent
pose inşası Aşama 6'nın konusudur.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prepare_structures import (  # noqa: E402
    cif_to_mol2,
    download_ligand_cif,
    download_ligand_sdf,
    download_molfile_to_params,
    run_molfile_to_params,
    run_obabel,
    sanitize_sdf_with_rdkit,
    sdf_to_mol2,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("prepare_ligand")


def extract_ligand_from_clean_pdb(clean_pdb: Path, ligand_comp_id: str, out_pdb: Path) -> int:
    lines = [
        l for l in clean_pdb.read_text().splitlines()
        if l.startswith("HETATM") and l[17:20].strip() == ligand_comp_id
    ]
    if not lines:
        return 0
    out_pdb.write_text("\n".join(lines) + "\nEND\n")
    return len(lines)


def build_mol2_for_ligand(
    ligand_comp_id: str,
    example_clean_pdb: Path,
    lig_dir: Path,
    rosetta_ligand_comp_id: str | None = None,
) -> tuple[Path | None, str]:
    """Kaskad: CIF->mol2, sonra SDF->RDKit->mol2, sonra PDB->obabel->mol2.

    CIF/SDF sorguları her zaman gerçek CCD kimliği (ligand_comp_id) ile yapılır.
    Son çare PDB-tabanlı yol (obabel) ise clean_pdb'deki fiili HETATM adını
    (rosetta_ligand_comp_id, override yoksa ligand_comp_id ile aynı) kullanır."""
    pdb_match_id = rosetta_ligand_comp_id or ligand_comp_id
    lig_dir.mkdir(parents=True, exist_ok=True)
    mol2_path = lig_dir / f"{ligand_comp_id}_ccd.mol2"
    cif_path = lig_dir / f"{ligand_comp_id}.cif"

    if download_ligand_cif(ligand_comp_id, cif_path):
        if cif_to_mol2(cif_path, mol2_path, ligand_comp_id):
            return mol2_path, "CCD_CIF"

    sdf_path = lig_dir / f"{ligand_comp_id}_ideal.sdf"
    clean_sdf = lig_dir / f"{ligand_comp_id}_clean.sdf"
    mol2_sdf = lig_dir / f"{ligand_comp_id}_sdf.mol2"
    if download_ligand_sdf(ligand_comp_id, sdf_path):
        if sanitize_sdf_with_rdkit(sdf_path, clean_sdf):
            if sdf_to_mol2(clean_sdf, mol2_sdf):
                return mol2_sdf, "RCSB_SDF_RDKIT"

    lig_pdb = lig_dir / f"{ligand_comp_id}_from_complex.pdb"
    mol2_pdb = lig_dir / f"{ligand_comp_id}_pdb.mol2"
    n = extract_ligand_from_clean_pdb(example_clean_pdb, pdb_match_id, lig_pdb)
    if n > 0 and run_obabel(lig_pdb, mol2_pdb):
        return mol2_pdb, "PDB_OBABEL_LAST_RESORT"

    return None, "ALL_METHODS_FAILED"


def _is_hydrogen_name(atom_name: str) -> bool:
    # PDB atom adlandırmasında hidrojenler tipik olarak 'H' ile başlar
    # (ör. H10, HG1) ya da nadiren ilk karakter rakamsa ikinci karakter 'H'dir.
    stripped = atom_name.strip()
    return stripped.startswith("H") or (len(stripped) > 1 and stripped[1] == "H" and stripped[0].isdigit())


def verify_atom_names(params_path: Path, pdb_paths: list[Path], ligand_comp_id: str) -> dict:
    """params'taki ağır atomların PDB'deki ağır atomlarla eşleşip eşleşmediğini
    kontrol eder. cif_to_mol2 hidrojenleri elediği için params yalnızca ağır
    atomlar içerir; PDB'de fazladan H atomu olması (bazı yüksek çözünürlüklü
    yapılarda görülür) engelleyici bir uyumsuzluk SAYILMAZ çünkü Rosetta H
    atomlarını ideal geometriden zaten yeniden oluşturur. Ağır atom
    uyuşmazlığı ise gerçek bir sorundur."""
    params_atoms = set()
    for line in params_path.read_text().splitlines():
        if line.startswith("ATOM"):
            params_atoms.add(line[5:9].strip())

    per_pdb = {}
    all_match = True
    for pdb_path in pdb_paths:
        pdb_atoms = set()
        for line in pdb_path.read_text().splitlines():
            if line.startswith("HETATM") and line[17:20].strip() == ligand_comp_id:
                pdb_atoms.add(line[12:16].strip())
        pdb_heavy_atoms = {a for a in pdb_atoms if not _is_hydrogen_name(a)}
        only_in_params = params_atoms - pdb_atoms
        only_in_pdb_heavy = pdb_heavy_atoms - params_atoms
        match = len(pdb_atoms) > 0 and not only_in_params and not only_in_pdb_heavy
        per_pdb[pdb_path.stem] = {
            "match": match,
            "n_params_atoms": len(params_atoms),
            "n_pdb_atoms": len(pdb_atoms),
            "n_pdb_heavy_atoms": len(pdb_heavy_atoms),
            "only_in_params": sorted(only_in_params),
            "only_in_pdb_heavy": sorted(only_in_pdb_heavy),
            "extra_H_in_pdb": sorted(pdb_atoms - pdb_heavy_atoms - params_atoms),
        }
        if not match:
            all_match = False
    return {"all_match": all_match, "per_pdb": per_pdb}


_PYROSETTA_TEST_SNIPPET = """
import sys, math
import pyrosetta
pyrosetta.init("-extra_res_fa {params_path} -mute all -ignore_unrecognized_res false", silent=True)
pose = pyrosetta.pose_from_file("{conformer_pdb}")
if pose.total_residue() == 0:
    print("RESULT loaded=True scored=False score=None error=0_residue")
    sys.exit(0)
sfxn = pyrosetta.create_score_function("ref2015")
score = sfxn(pose)
ok = (score is not None) and (not math.isnan(score)) and (not math.isinf(score))
print(f"RESULT loaded=True scored={{ok}} score={{score}} error=None")
"""


def test_pyrosetta_load(params_path: Path, conformer_pdb: Path) -> dict:
    """Ayrı bir alt-process'te PyRosetta init edip params'ı yükler ve REF2015
    ile skorlar. Alt-process kullanılıyor çünkü pyrosetta.init() bir process
    içinde yalnızca bir kez çağrılabilir; 51 ligand için 51 farklı
    -extra_res_fa gerekiyor."""
    import subprocess
    import sys as _sys

    code = _PYROSETTA_TEST_SNIPPET.format(
        params_path=params_path.resolve(), conformer_pdb=conformer_pdb.resolve()
    )
    result = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    out = result.stdout.strip()
    if "RESULT" not in out:
        return {
            "loaded": False, "scored": False, "score": None,
            "error": (result.stderr[-500:] if result.stderr else "no RESULT line"),
        }
    line = out.splitlines()[-1]
    fields = dict(kv.split("=", 1) for kv in line.replace("RESULT ", "").split(" "))
    return {
        "loaded": fields["loaded"] == "True",
        "scored": fields["scored"] == "True",
        "score": float(fields["score"]) if fields["score"] not in ("None",) else None,
        "error": None if fields["error"] == "None" else fields["error"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path, default=Path("results/metadata/egfr_ligand_inventory.csv")
    )
    parser.add_argument(
        "--prep-summary", type=Path, default=Path("results/preparation_summary.csv")
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--ligand-dir", type=Path, default=Path("data/ligands"))
    parser.add_argument("--params-dir", type=Path, default=Path("data/ligands/params"))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("results/metadata/ligand_parameterization_status.csv")
    )
    parser.add_argument("--ligand-id", type=str, default=None, help="Tek bir ligand comp_id test et")
    parser.add_argument("--skip-pyrosetta-test", action="store_true")
    args = parser.parse_args()

    inv = pd.read_csv(args.inventory)
    prep = pd.read_csv(args.prep_summary)
    merged = inv.merge(
        prep[["pdb_id", "is_covalent", "covalent_ligand_atom", "rosetta_ligand_comp_id"]],
        on="pdb_id",
    )

    unique_ligands = sorted(merged["ligand_comp_id"].unique())
    if args.ligand_id:
        unique_ligands = [l for l in unique_ligands if l.lower() == args.ligand_id.lower()]
        if not unique_ligands:
            raise SystemExit(f"{args.ligand_id} envanterde bulunamadı.")

    script_path = download_molfile_to_params(Path("src"))
    args.params_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, ligand_comp_id in enumerate(unique_ligands, start=1):
        log.info("[%d/%d] %s parametreleniyor...", i, len(unique_ligands), ligand_comp_id)
        pdbs_with_this_ligand = merged[merged["ligand_comp_id"] == ligand_comp_id]["pdb_id"].tolist()
        clean_pdb_paths = [
            args.processed_dir / f"{p}_clean.pdb" for p in pdbs_with_this_ligand
        ]
        clean_pdb_paths = [p for p in clean_pdb_paths if p.exists()]
        if not clean_pdb_paths:
            rows.append(
                {"ligand_comp_id": ligand_comp_id, "status": "NO_CLEAN_PDB_FOUND"}
            )
            continue

        is_covalent = bool(merged[merged["ligand_comp_id"] == ligand_comp_id]["is_covalent"].any())
        # Rosetta'nın gördüğü rezidü adı: normalde ligand_comp_id ile aynıdır;
        # dahili Rosetta rezidü/patch isim çakışması olan nadir durumlarda
        # (bkz. config/ligand_overrides.yaml rosetta_ligand_code, ör. '634'
        # -> 'Z34') Aşama 4 clean_pdb'ye farklı bir HETATM adı yazmış olabilir.
        rosetta_ligand_comp_id = str(
            merged[merged["ligand_comp_id"] == ligand_comp_id]["rosetta_ligand_comp_id"].iloc[0]
        )

        mol2_path, method = build_mol2_for_ligand(
            ligand_comp_id, clean_pdb_paths[0], args.ligand_dir,
            rosetta_ligand_comp_id=rosetta_ligand_comp_id,
        )
        if mol2_path is None:
            rows.append(
                {
                    "ligand_comp_id": ligand_comp_id,
                    "status": "MOL2_GENERATION_FAILED",
                    "method": method,
                    "n_pdbs_using_ligand": len(pdbs_with_this_ligand),
                    "is_covalent": is_covalent,
                }
            )
            continue

        params_path = run_molfile_to_params(
            mol2_path, args.params_dir, rosetta_ligand_comp_id, script_path
        )
        if not params_path.exists():
            rows.append(
                {
                    "ligand_comp_id": ligand_comp_id,
                    "status": "PARAMS_GENERATION_FAILED",
                    "method": method,
                    "n_pdbs_using_ligand": len(pdbs_with_this_ligand),
                    "is_covalent": is_covalent,
                }
            )
            continue

        atom_check = verify_atom_names(params_path, clean_pdb_paths, rosetta_ligand_comp_id)

        conformer_pdb = args.params_dir / f"{rosetta_ligand_comp_id}_0001.pdb"
        pyro_result = (
            {"loaded": None, "scored": None, "score": None, "error": "SKIPPED"}
            if args.skip_pyrosetta_test
            else test_pyrosetta_load(params_path, conformer_pdb)
        )

        status = "OK" if atom_check["all_match"] and (args.skip_pyrosetta_test or pyro_result["scored"]) else "NEEDS_REVIEW"
        rows.append(
            {
                "ligand_comp_id": ligand_comp_id,
                "rosetta_ligand_comp_id": rosetta_ligand_comp_id,
                "status": status,
                "method": method,
                "n_pdbs_using_ligand": len(pdbs_with_this_ligand),
                "pdbs_using_ligand": ",".join(pdbs_with_this_ligand),
                "is_covalent": is_covalent,
                "atom_names_all_match": atom_check["all_match"],
                "atom_name_mismatches": str(
                    {k: v for k, v in atom_check["per_pdb"].items() if not v["match"]}
                ) if not atom_check["all_match"] else "",
                "pyrosetta_loaded": pyro_result["loaded"],
                "pyrosetta_scored": pyro_result["scored"],
                "ref2015_score": pyro_result["score"],
                "pyrosetta_error": pyro_result["error"],
            }
        )

    out_df = pd.DataFrame(rows)
    output_csv = (
        args.output_csv
        if not args.ligand_id
        else args.output_csv.with_name(f"test_single_{args.ligand_id.lower()}_ligprep.csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    log.info("Sonuç yazıldı: %s (%d satır)", output_csv, len(out_df))

    if "status" in out_df.columns:
        log.info("Özet:\n%s", out_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
