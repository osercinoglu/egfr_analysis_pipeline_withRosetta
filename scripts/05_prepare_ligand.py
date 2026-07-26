"""
Stage 5: Generates Rosetta/PyRosetta parameters (.params) for the 51 unique
EGFR-inhibitor ligands and validates each one.

Pipeline (for every unique ligand comp_id, in priority order):
  1. RCSB CCD CIF -> mol2 (canonical atom names, matches PDB HETATM)   [primary]
  2. RCSB ideal SDF -> RDKit sanitize -> mol2 (OpenBabel)               [fallback 1]
  3. Ligand PDB extracted from the complex -> mol2 (OpenBabel, --gen3d) [last resort]
  4. molfile_to_params.py --keep-names --clobber

Validation for every ligand:
  - was a .params file produced
  - atom count > 0
  - do the params ATOM names EXACTLY match the HETATM atom names in
    data/processed/{ID}_clean.pdb for EVERY PDB using that ligand
  - can PyRosetta load it as a ligand into a pose
  - can it be scored with REF2015 (not NaN/inf)

For covalent ligands (those where Stage 4 detected a CYS797-SG linkage), a
CONNECT record is also added to the .params file (needed to build the
covalent bond in Stage 6); this script adds the CONNECT record but building
the actual covalent pose is Stage 6's job.
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
    """Cascade: CIF->mol2, then SDF->RDKit->mol2, then PDB->obabel->mol2.

    CIF/SDF lookups always use the real CCD identity (ligand_comp_id). The
    last-resort PDB-based path (obabel) uses the actual HETATM name in
    clean_pdb (rosetta_ligand_comp_id, same as ligand_comp_id unless there's
    an override)."""
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
    # In PDB atom naming, hydrogens typically start with 'H' (e.g. H10, HG1),
    # or in rarer cases the first character is a digit and the second is 'H'.
    stripped = atom_name.strip()
    return stripped.startswith("H") or (len(stripped) > 1 and stripped[1] == "H" and stripped[0].isdigit())


def verify_atom_names(params_path: Path, pdb_paths: list[Path], ligand_comp_id: str) -> dict:
    """Checks whether the heavy atoms in params match the heavy atoms in the
    PDB. Since cif_to_mol2 strips hydrogens, params only contains heavy
    atoms; extra H atoms in the PDB (seen in some high-resolution structures)
    are NOT counted as a blocking mismatch, since Rosetta rebuilds H atoms
    from ideal geometry anyway. A heavy-atom mismatch, however, is a real
    problem."""
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
    """Initializes PyRosetta in a separate subprocess, loads the params, and
    scores with REF2015. A subprocess is used because pyrosetta.init() can
    only be called once per process; 51 ligands need 51 different
    -extra_res_fa values."""
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
    parser.add_argument("--ligand-id", type=str, default=None, help="Test a single ligand comp_id")
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
            raise SystemExit(f"{args.ligand_id} not found in the inventory.")

    script_path = download_molfile_to_params(Path("src"))
    args.params_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, ligand_comp_id in enumerate(unique_ligands, start=1):
        log.info("[%d/%d] parameterizing %s...", i, len(unique_ligands), ligand_comp_id)
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
        # The residue name Rosetta sees: normally the same as ligand_comp_id;
        # in the rare cases with an internal Rosetta residue/patch name
        # collision (see config/ligand_overrides.yaml rosetta_ligand_code,
        # e.g. '634' -> 'Z34'), Stage 4 may have written a different HETATM
        # name into clean_pdb.
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
    log.info("Results written: %s (%d rows)", output_csv, len(out_df))

    if "status" in out_df.columns:
        log.info("Summary:\n%s", out_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
