"""
Stage 1: Clean PDB structures, extract ligands, generate .params.

For every PDB:
  1. Extract the target chain (default A) + the target ligand
  2. Collapse alternate conformations (altloc) to the first position
  3. Remove water and crystallization additives
  4. Write the ligand to a separate .pdb file → convert to .mol2 (obabel)
  5. Generate .params with molfile_to_params.py
  6. Write the clean complex .pdb file under data/processed/

Usage:
  python src/prepare_structures.py --config config.yaml [--pdb_id 5GMP]
"""

import argparse
import os
import subprocess
import shutil
import sys
from pathlib import Path

import yaml
import pandas as pd
import requests

# rosetta_py lives alongside this file
_SRC_DIR = str(Path(__file__).parent.resolve())


# Crystallization additives / buffer components
CRYSTAL_EXCL = {
    "HOH","SO4","GOL","PEG","EDO","MPD","BME","DTT","PO4","MES","TRS",
    "ACE","ACT","FMT","IMD","IPA","DMS","EOH","CIT","TAR","EPE","HED",
    "1PE","2PE","MOH","P6G","BU1","BU2","BU3","BOG","OGA","CO3","IOD",
    "CA","MG","ZN","CU","FE","MN","CL","K","NA","BR","ADP","ATP","GTP",
    "GMP","APC","ANP","NH4","LI","SUL","FLC","TLA","OXL","PDO","PGE",
    "PGO","PG4","12P","PG6","PE4","PE5","P33","CMO","ACY","MSE","SEP",
    "TPO","NEP","CSO","KCX","UNL","UNX","CSD","MLY","LLP","NHE","BTB",
    "XPE","SPM","SPD","PUT","TRD","SIN","7PE","15P","OCS","KCX","MLY",
    "LLP","PYL","SEC","FME","CME","M3L","FAD","NAD","NAP","NDP","AMP",
    "YY3","VNS","Q6K","0WN",  # extra ligands found in specific structures
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def clean_pdb(raw_pdb: Path, output_pdb: Path, target_chain: str,
              target_ligand: str) -> dict:
    """
    Cleans the raw PDB: keeps the target chain + target ligand, keeps altloc
    A if A/B are present, discards everything else.

    Returns: summary dict (n_protein_atoms, n_ligand_atoms)
    """
    lines = raw_pdb.read_text().splitlines()
    out_lines = []
    n_prot = 0
    n_lig = 0

    for line in lines:
        rec = line[:6].strip()

        # ATOM: target chain only, first altloc only (blank or 'A')
        if rec == "ATOM":
            chain = line[21] if len(line) > 21 else " "
            altloc = line[16] if len(line) > 16 else " "
            if chain != target_chain:
                continue
            if altloc not in (" ", "A"):
                continue
            # replace the altloc letter with a blank
            clean = line[:16] + " " + line[17:]
            out_lines.append(clean)
            n_prot += 1

        # HETATM: target ligand + target chain only, first copy
        elif rec == "HETATM":
            res_name = line[17:20].strip()
            chain    = line[21] if len(line) > 21 else " "
            altloc   = line[16] if len(line) > 16 else " "
            atom_name = line[12:16].strip() if len(line) > 16 else ""
            if (res_name == target_ligand
                    and chain == target_chain
                    and altloc in (" ", "A")):
                clean = line[:16] + " " + line[17:]
                out_lines.append(clean)
                n_lig += 1

        # TER and END
        elif rec in ("TER", "END"):
            out_lines.append(line)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(out_lines) + "\n")
    return {"n_protein_atoms": n_prot, "n_ligand_atoms": n_lig}


def extract_ligand_pdb(processed_pdb: Path, lig_pdb: Path, target_ligand: str):
    """Extracts just the ligand from the complex PDB and writes it to a separate PDB file."""
    lines = processed_pdb.read_text().splitlines()
    lig_lines = [l for l in lines
                 if l[:6].strip() == "HETATM" and l[17:20].strip() == target_ligand]
    lig_lines.append("END")
    lig_pdb.parent.mkdir(parents=True, exist_ok=True)
    lig_pdb.write_text("\n".join(lig_lines) + "\n")
    return len(lig_lines) - 1


def download_ligand_cif(lig_code: str, cif_path: Path) -> bool:
    """Download CCD CIF for a ligand from RCSB (has correct CCD atom names)."""
    url = f"https://files.rcsb.org/ligands/download/{lig_code}.cif"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 200:
            cif_path.parent.mkdir(parents=True, exist_ok=True)
            cif_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def cif_to_mol2(cif_path: Path, mol2_path: Path, lig_code: str) -> bool:
    """
    Parse RCSB CCD CIF → mol2 with canonical PDB atom names.

    Uses _chem_comp_atom (names + ideal coords) and _chem_comp_bond
    (connectivity + bond orders).  The resulting mol2 has the same atom
    names as the HETATM records in the crystal PDB, so Rosetta can match
    atoms when reading the structure.
    """
    try:
        text = cif_path.read_text()

        def _parse_loop(text: str, tag_prefix: str) -> list[dict]:
            """Extract all rows from the mmCIF loop_ whose columns start with tag_prefix."""
            import re
            # Find the loop block
            pattern = re.compile(
                r'loop_\s*((?:_' + re.escape(tag_prefix) + r'\S+\s*)+)(.*?)(?=loop_|\Z)',
                re.DOTALL
            )
            m = pattern.search(text)
            if not m:
                return []
            header_str, data_str = m.group(1), m.group(2)
            columns = re.findall(r'_\S+', header_str)
            col_names = [c.split('.')[-1] for c in columns]

            # Tokenize data (handle quoted strings)
            tokens = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", data_str)
            tokens = [t.strip("'\"") for t in tokens]

            rows = []
            for i in range(0, len(tokens) - len(col_names) + 1, len(col_names)):
                chunk = tokens[i:i + len(col_names)]
                if len(chunk) < len(col_names):
                    break
                rows.append(dict(zip(col_names, chunk)))
            return rows

        atoms = _parse_loop(text, "chem_comp_atom")
        bonds = _parse_loop(text, "chem_comp_bond")

        if not atoms:
            return False

        # Filter out H atoms (keep heavy atoms only)
        heavy = [a for a in atoms
                 if a.get("type_symbol", "H") not in ("H", "D")
                 and a.get("pdbx_aromatic_flag") != "?"]
        if not heavy:
            heavy = [a for a in atoms if a.get("type_symbol", "H") not in ("H", "D")]

        name_to_idx = {a["atom_id"]: i + 1 for i, a in enumerate(heavy)}

        # SYBYL atom types for common elements (simplified)
        _SYBYL = {"C": "C.ar", "N": "N.ar", "O": "O.2", "S": "S.2", "P": "P.3",
                  "F": "F", "Cl": "Cl", "Br": "Br", "I": "I"}

        def sybyl(atom):
            is_arom = atom.get("pdbx_aromatic_flag", "N") == "Y"
            el = atom.get("type_symbol", "C")
            if is_arom:
                if el == "C": return "C.ar"
                if el == "N": return "N.ar"
                if el == "O": return "O.ar"
                if el == "S": return "S.ar"
            return _SYBYL.get(el, el)

        # Bond order map
        _ORDER = {"SING": "1", "DOUB": "2", "TRIP": "3", "AROM": "ar",
                  "sing": "1", "doub": "2", "trip": "3", "arom": "ar"}

        heavy_bonds = [
            b for b in bonds
            if b.get("atom_id_1") in name_to_idx and b.get("atom_id_2") in name_to_idx
        ]

        lines = ["@<TRIPOS>MOLECULE", lig_code,
                 f"{len(heavy)} {len(heavy_bonds)} 0 0 0",
                 "SMALL", "NO_CHARGES", "",
                 "@<TRIPOS>ATOM"]
        for i, a in enumerate(heavy):
            x = float(a.get("model_Cartn_x", a.get("pdbx_model_Cartn_x_ideal", "0")))
            y = float(a.get("model_Cartn_y", a.get("pdbx_model_Cartn_y_ideal", "0")))
            z = float(a.get("model_Cartn_z", a.get("pdbx_model_Cartn_z_ideal", "0")))
            lines.append(f"{i+1:6d} {a['atom_id']:<8s} {x:10.4f} {y:10.4f} {z:10.4f}"
                         f" {sybyl(a):<8s} 1 LIG 0.0000")
        lines.append("@<TRIPOS>BOND")
        for i, b in enumerate(heavy_bonds):
            a1 = name_to_idx[b["atom_id_1"]]
            a2 = name_to_idx[b["atom_id_2"]]
            bo = _ORDER.get(b.get("value_order", "SING"), "1")
            lines.append(f"{i+1:6d} {a1:4d} {a2:4d} {bo}")

        mol2_path.write_text("\n".join(lines) + "\n")
        return mol2_path.exists() and mol2_path.stat().st_size > 0
    except Exception as e:
        print(f"  [!] cif_to_mol2 error: {e}")
        return False


def download_ligand_sdf(lig_code: str, sdf_path: Path) -> bool:
    """Download ideal SDF for a ligand component from RCSB (best connectivity info)."""
    url = f"https://files.rcsb.org/ligands/download/{lig_code}_ideal.sdf"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            sdf_path.parent.mkdir(parents=True, exist_ok=True)
            sdf_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def sanitize_sdf_with_rdkit(sdf_in: Path, sdf_out: Path) -> bool:
    """
    Load SDF with RDKit, strip explicit H, sanitize, write back as clean SDF.

    Stripping H is critical: molfile_to_params.py embeds explicit H atoms
    into the params ATOM list.  When PyRosetta later reads the crystal PDB
    (which has no H), it tries to fill_missing_atoms for every H — and for
    some ligand geometries that fails with "too many tries".  Without explicit
    H, Rosetta adds them from idealized geometry without hitting this limit.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromMolFile(str(sdf_in), removeHs=True, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMolFile(str(sdf_in), removeHs=True, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol,
                    Chem.SanitizeFlags.SANITIZE_ALL ^
                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        if mol is None:
            return False
        writer = Chem.SDWriter(str(sdf_out))
        writer.write(mol)
        writer.close()
        return sdf_out.exists() and sdf_out.stat().st_size > 0
    except Exception as e:
        print(f"  [!] sanitize_sdf_with_rdkit error: {e}")
        return False


def sdf_to_mol2(sdf_path: Path, mol2_path: Path) -> bool:
    """Convert SDF to mol2 using OpenBabel Python API (kept as fallback)."""
    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("sdf", "mol2")
        mol = ob.OBMol()
        if not conv.ReadFile(mol, str(sdf_path)):
            return False
        mol.AddHydrogens()
        conv.WriteFile(mol, str(mol2_path))
        return mol2_path.exists() and mol2_path.stat().st_size > 0
    except Exception as e:
        print(f"  [!] sdf_to_mol2 error: {e}")
        return False


def run_obabel(lig_pdb: Path, lig_mol2: Path) -> bool:
    """PDB → mol2 conversion via the OpenBabel Python API (fallback)."""
    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("pdb", "mol2")
        mol = ob.OBMol()
        if not conv.ReadFile(mol, str(lig_pdb)):
            return False
        mol.AddHydrogens()
        conv.WriteFile(mol, str(lig_mol2))
        return lig_mol2.exists() and lig_mol2.stat().st_size > 0
    except ImportError:
        result = subprocess.run(
            ["obabel", str(lig_pdb), "-O", str(lig_mol2), "--gen3d"],
            capture_output=True, text=True
        )
        return lig_mol2.exists() and lig_mol2.stat().st_size > 0


def download_molfile_to_params(dest_dir: Path) -> Path:
    """Download molfile_to_params.py from the RosettaCommons GitHub repo."""
    script_path = dest_dir / "molfile_to_params.py"
    if script_path.exists():
        return script_path
    url = ("https://raw.githubusercontent.com/RosettaCommons/rosetta/"
           "main/source/scripts/python/public/molfile_to_params.py")
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        script_path.write_text(r.text)
        print(f"  molfile_to_params.py downloaded: {script_path}")
    else:
        # Fallback: an older branch
        url2 = ("https://raw.githubusercontent.com/RosettaCommons/rosetta/"
                "master/source/scripts/python/public/molfile_to_params.py")
        r2 = requests.get(url2, timeout=30)
        if r2.status_code == 200:
            script_path.write_text(r2.text)
        else:
            raise RuntimeError("Could not download molfile_to_params.py.")
    return script_path


def run_molfile_to_params(mol2_path: Path, params_dir: Path,
                           lig_code: str, script_path: Path) -> Path:
    """
    Run molfile_to_params.py to produce .params and _0001.pdb.
    lig_code is either the 3-letter residue code or 'PDBID_LIGID';
    the Rosetta NAME (used as -n flag) is always the 3-letter ligand code
    from the PDB (last segment after '_').
    """
    params_dir.mkdir(parents=True, exist_ok=True)
    # Extract 3-letter ligand code: "4G5J_0WM" -> "0WM", "F62" -> "F62"
    residue_name = lig_code.split("_")[-1]
    env = {**os.environ, "PYTHONPATH": _SRC_DIR}
    result = subprocess.run(
        [sys.executable, str(script_path.resolve()),
         "-n", residue_name,
         "--keep-names",
         "--clobber",
         str(mol2_path.resolve())],
        capture_output=True, text=True,
        cwd=str(params_dir.resolve()),
        env=env,
    )
    params_file = params_dir / f"{residue_name}.params"
    if not params_file.exists():
        print(f"  [!] Could not generate .params. stdout:\n{result.stdout[:500]}")
        print(f"  stderr: {result.stderr[:300]}")
    return params_file


def process_structure(pdb_id: str, ligand_id: str, cfg: dict,
                       force: bool = False, chain: str = None) -> dict:
    """
    Fully processes a single PDB structure.
    Returns: {'pdb_id', 'status', 'processed_pdb', 'params_file', ...}
    chain overrides cfg chain_selection.default (for structures where the
    target ligand is not on chain A, e.g. 7JXM with 9LL on chain B).
    """
    base = Path(".")
    raw_pdb   = base / cfg["paths"]["raw_pdb"] / f"{pdb_id}.pdb"
    proc_dir  = base / cfg["paths"]["processed"]
    lig_dir   = base / cfg["paths"]["ligands"]
    params_dir= base / cfg["paths"]["params"]

    processed_pdb = proc_dir / f"{pdb_id}_clean.pdb"
    lig_pdb       = lig_dir  / f"{pdb_id}_{ligand_id}.pdb"
    lig_mol2      = lig_dir  / f"{pdb_id}_{ligand_id}.mol2"
    params_file   = params_dir / f"{pdb_id}_{ligand_id}.params"

    # Skip if already processed
    if not force and processed_pdb.exists() and params_file.exists():
        print(f"  {pdb_id}: already processed, skipping.")
        return {"pdb_id": pdb_id, "status": "skipped",
                "processed_pdb": str(processed_pdb),
                "params_file": str(params_file)}

    target_chain = chain if chain else str(cfg["chain_selection"]["default"])
    print(f"\n[{pdb_id}] Processing... (ligand={ligand_id}, chain={target_chain})")

    # 1. Clean the PDB
    if not raw_pdb.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        r = requests.get(url, timeout=30)
        raw_pdb.parent.mkdir(parents=True, exist_ok=True)
        raw_pdb.write_bytes(r.content)
        print(f"  Downloaded: {raw_pdb}")

    summary = clean_pdb(raw_pdb, processed_pdb, target_chain, ligand_id)
    print(f"  Cleaned: {summary['n_protein_atoms']} protein atoms, "
          f"{summary['n_ligand_atoms']} ligand atoms")

    if summary["n_ligand_atoms"] == 0:
        print(f"  [!] Ligand {ligand_id} not found!")
        return {"pdb_id": pdb_id, "status": "no_ligand"}

    # 2. Extract the ligand
    n_lig = extract_ligand_pdb(processed_pdb, lig_pdb, ligand_id)

    # Input for params: prefer RCSB CIF (has correct CCD atom names matching PDB)
    # Fallback 1: RCSB SDF → RDKit sanitized SDF (no atom name guarantee but works for
    #             structures where Rosetta can match by element order)
    # Fallback 2: OpenBabel mol2 from extracted PDB ligand
    cif_path   = lig_dir / f"{ligand_id}.cif"
    cif_mol2   = lig_dir / f"{ligand_id}_ccd.mol2"
    sdf_path   = lig_dir / f"{ligand_id}_ideal.sdf"
    sdf_clean  = lig_dir / f"{ligand_id}_clean.sdf"

    params_input = None
    src_label    = None

    # Primary: CIF → mol2 (preserves CCD atom names = PDB HETATM names)
    if not cif_path.exists():
        download_ligand_cif(ligand_id, cif_path)
    if cif_path.exists():
        if cif_to_mol2(cif_path, cif_mol2, ligand_id):
            params_input = cif_mol2
            src_label = "RCSB CIF (CCD names)"

    # Fallback: SDF → RDKit-cleaned SDF
    if params_input is None:
        if not sdf_path.exists():
            download_ligand_sdf(ligand_id, sdf_path)
        if sdf_path.exists() and sanitize_sdf_with_rdkit(sdf_path, sdf_clean):
            params_input = sdf_clean
            src_label = "RCSB SDF (RDKit)"

    # Last resort: PDB ligand via OpenBabel
    if params_input is None:
        if run_obabel(lig_pdb, lig_mol2):
            params_input = lig_mol2
            src_label = "PDB obabel mol2"

    if params_input is None:
        print(f"  [!] Could not generate params input")
        return {"pdb_id": pdb_id, "status": "mol2_failed"}

    print(f"  params input: {src_label} → {params_input.name}")

    # 3. Download + run molfile_to_params.py
    script = download_molfile_to_params(base / "src")
    params_out = run_molfile_to_params(params_input, params_dir,
                                        f"{pdb_id}_{ligand_id}", script)

    # Copy the params file to its standard name
    if params_out.exists():
        shutil.copy(params_out, params_file)

    status = "ok" if params_file.exists() else "params_failed"
    print(f"  Status: {status} — params: {params_file}")

    return {
        "pdb_id": pdb_id,
        "status": status,
        "processed_pdb": str(processed_pdb),
        "lig_pdb": str(lig_pdb),
        "lig_mol2": str(lig_mol2),
        "params_file": str(params_file) if params_file.exists() else None,
        "n_protein_atoms": summary["n_protein_atoms"],
        "n_ligand_atoms": summary["n_ligand_atoms"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pdb_id", default=None,
                        help="Process only this PDB (for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess structures that are already processed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df  = pd.read_csv(cfg["paths"]["candidates_csv"])

    if args.pdb_id:
        df = df[df["pdb_id"] == args.pdb_id]
        if df.empty:
            print(f"PDB ID {args.pdb_id} not found in the candidate list.")
            return

    results = []
    for _, row in df.iterrows():
        chain = row.get("chain", None)
        res = process_structure(row["pdb_id"], row["ligand_id"], cfg,
                                force=args.force,
                                chain=chain if pd.notna(chain) else None)
        results.append(res)

    summary_df = pd.DataFrame(results)
    print("\n=== SUMMARY ===")
    print(summary_df[["pdb_id","status"]].to_string(index=False))
    ok = (summary_df["status"].isin(["ok","skipped"])).sum()
    print(f"\nSucceeded: {ok}/{len(summary_df)}")

    summary_df.to_csv("results/preparation_summary.csv", index=False)


if __name__ == "__main__":
    main()
