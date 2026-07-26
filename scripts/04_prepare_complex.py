"""
Stage 4: Produces an analysis-specific, cleaned complex structure for
Rosetta/PyRosetta from the EGFR chain + ligand copy selected in Stage 3.

Kept: the selected EGFR chain (standard amino acids), the selected ligand
copy. Removed (default): other protein chains, crystallization agents/
buffer molecules, alternative ligand copies, waters.

Before waters are permanently removed, waters within 4 Å of the ligand are
separately reported (as bridging-water candidates) — see
results/metadata/bridging_waters/.

This script produces:
  - altloc selection (highest-occupancy / 'A' or blank altloc is kept)
  - missing-residue detection via SEQRES vs ATOM comparison
  - Cys-Cys disulfide scan (SG-SG < 2.5 Å)
  - authoritative bond info extracted from mmCIF _struct_conn for covalent
    structures
  - native numbering -> sequential pose numbering map
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import is_aa

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("prepare_complex")

SEQRES_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y",
}


def parse_seqres(pdb_path: Path, chain_id: str) -> list[str]:
    residues = []
    for line in pdb_path.read_text(errors="replace").splitlines():
        if line.startswith("SEQRES") and line[11] == chain_id:
            resnames = line[19:].split()
            residues.extend(resnames)
    return residues


def read_atom_lines(pdb_path: Path) -> list[str]:
    return [
        l for l in pdb_path.read_text(errors="replace").splitlines()
        if l.startswith(("ATOM", "HETATM"))
    ]


def select_altloc(lines: list[str]) -> tuple[list[str], int]:
    """For each (chain, resSeq, atomName), keeps the record with altLoc ' '
    or the highest occupancy; discards the rest. Returns (selected lines,
    number of discarded atoms)."""
    groups: dict[tuple, list[tuple[str, str, float]]] = {}
    for line in lines:
        chain = line[21]
        resseq = line[22:27]
        atomname = line[12:16]
        altloc = line[16]
        occ = float(line[54:60]) if line[54:60].strip() else 1.0
        key = (chain, resseq, atomname)
        groups.setdefault(key, []).append((altloc, line, occ))

    kept_lines = []
    n_removed = 0
    for key, entries in groups.items():
        if len(entries) == 1:
            kept_lines.append(entries[0][1])
            continue
        # Multiple altlocs present: prefer ' ' or 'A', otherwise highest occupancy.
        blank_or_a = [e for e in entries if e[0] in (" ", "A")]
        if blank_or_a:
            best = max(blank_or_a, key=lambda e: e[2])
        else:
            best = max(entries, key=lambda e: e[2])
        kept_lines.append(best[1])
        n_removed += len(entries) - 1
    # Sort by original line order to preserve the source file's order
    order = {l: i for i, l in enumerate(lines)}
    kept_lines.sort(key=lambda l: order[l])
    return kept_lines, n_removed


def get_covalent_bond_info(cif_path: Path, ligand_comp_id: str) -> dict | None:
    """Extracts the ligand's covalent bond info from the mmCIF _struct_conn category."""
    try:
        d = MMCIF2Dict(str(cif_path))
    except Exception:
        return None
    ids = d.get("_struct_conn.conn_type_id", [])
    if not ids:
        return None
    for i, conn_type in enumerate(ids):
        if conn_type != "covale":
            continue
        p1_comp = d["_struct_conn.ptnr1_auth_comp_id"][i]
        p2_comp = d["_struct_conn.ptnr2_auth_comp_id"][i]
        if ligand_comp_id in (p1_comp, p2_comp):
            protein_side = 1 if p2_comp == ligand_comp_id else 2
            ligand_side = 2 if protein_side == 1 else 1
            return {
                "protein_resname": d[f"_struct_conn.ptnr{protein_side}_auth_comp_id"][i],
                "protein_chain": d[f"_struct_conn.ptnr{protein_side}_auth_asym_id"][i],
                "protein_resnum": d[f"_struct_conn.ptnr{protein_side}_auth_seq_id"][i],
                "protein_atom": d[f"_struct_conn.ptnr{protein_side}_label_atom_id"][i],
                "ligand_resname": d[f"_struct_conn.ptnr{ligand_side}_auth_comp_id"][i],
                "ligand_atom": d[f"_struct_conn.ptnr{ligand_side}_label_atom_id"][i],
                "bond_distance_A": d.get("_struct_conn.pdbx_dist_value", ["?"])[i],
            }
    return None


def find_bridging_waters(
    pdb_path: Path, egfr_chain: str, ligand_chain: str, ligand_resnum: str, cutoff: float = 4.0
) -> list[dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = structure[0]
    ligand_coords = []
    for chain in model:
        if chain.id == ligand_chain:
            for res in chain:
                if str(res.id[1]) == str(ligand_resnum):
                    ligand_coords = np.array(
                        [a.coord for a in res if a.element != "H"]
                    )
    if len(ligand_coords) == 0:
        return []

    waters = []
    for chain in model:
        for res in chain:
            if res.resname == "HOH":
                o_atom = res["O"] if "O" in res else next(iter(res), None)
                if o_atom is None:
                    continue
                d = float(np.linalg.norm(ligand_coords - o_atom.coord, axis=1).min())
                if d <= cutoff:
                    waters.append(
                        {
                            "water_chain": chain.id,
                            "water_resnum": res.id[1],
                            "min_dist_to_ligand_A": round(d, 2),
                        }
                    )
    return waters


def detect_missing_residues(
    pdb_path: Path, chain_id: str
) -> tuple[int, list[str]]:
    seqres = parse_seqres(pdb_path, chain_id)
    seqres_len = len(seqres)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return 0, []
    modeled_resnums = sorted(
        r.id[1] for r in chain if is_aa(r, standard=False)
    )
    if not modeled_resnums:
        return seqres_len, ["ALL_MISSING"]
    gaps = []
    for a, b in zip(modeled_resnums, modeled_resnums[1:]):
        if b - a > 1:
            gaps.append(f"{a + 1}-{b - 1}")
    n_missing = seqres_len - len(modeled_resnums)
    return max(n_missing, 0), gaps


def detect_disulfides(pdb_path: Path, chain_id: str) -> list[tuple[int, int, float]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return []
    cys_sg = [
        (res.id[1], res["SG"].coord)
        for res in chain
        if res.resname == "CYS" and "SG" in res
    ]
    bonds = []
    for i in range(len(cys_sg)):
        for j in range(i + 1, len(cys_sg)):
            d = float(np.linalg.norm(cys_sg[i][1] - cys_sg[j][1]))
            if d < 2.5:
                bonds.append((cys_sg[i][0], cys_sg[j][0], round(d, 2)))
    return bonds


def prepare_one(
    pdb_id: str,
    override: dict,
    raw_pdb_dir: Path,
    raw_cif_dir: Path,
    processed_dir: Path,
    mapping_dir: Path,
    bridging_water_dir: Path,
) -> dict:
    pdb_path = raw_pdb_dir / f"{pdb_id}.pdb"
    cif_path = raw_cif_dir / f"{pdb_id}.cif"
    egfr_chain = override["egfr_chain"]
    ligand_chain = override["ligand_chain"]
    ligand_resnum = str(override["ligand_resnum"])
    ligand_comp_id = override["ligand_comp_id"]
    rosetta_ligand_comp_id = override.get("rosetta_ligand_code", ligand_comp_id)

    row: dict = {"pdb_id": pdb_id, "warnings": []}

    all_lines = read_atom_lines(pdb_path)

    protein_lines = [
        l for l in all_lines
        if l.startswith("ATOM")
        and l[21] == egfr_chain
        and l[17:20].strip() in SEQRES_3TO1
    ]
    ligand_lines = [
        l for l in all_lines
        if l.startswith("HETATM")
        and l[17:20].strip() == ligand_comp_id
        and l[21] == ligand_chain
        and l[22:26].strip() == ligand_resnum
    ]

    if not protein_lines:
        row["status"] = "FAILED_NO_PROTEIN_ATOMS"
        return row
    if not ligand_lines:
        row["status"] = "FAILED_NO_LIGAND_ATOMS"
        return row

    protein_lines, n_altloc_removed_p = select_altloc(protein_lines)
    ligand_lines, n_altloc_removed_l = select_altloc(ligand_lines)

    # --- Missing-residue / disulfide / covalent-bond / bridging-water analyses ---
    n_missing, gap_ranges = detect_missing_residues(pdb_path, egfr_chain)
    disulfides = detect_disulfides(pdb_path, egfr_chain)
    bridging_waters = find_bridging_waters(pdb_path, egfr_chain, ligand_chain, ligand_resnum)

    covalent_info = None
    if cif_path.exists():
        covalent_info = get_covalent_bond_info(cif_path, ligand_comp_id)

    if rosetta_ligand_comp_id != ligand_comp_id:
        # Some real CCD codes (e.g. '634') collide with Rosetta's internal
        # residue/patch name space (e.g. the peptoid library), causing a
        # ResidueTypeSetCache error. In that case only the HETATM resName
        # field in clean_pdb is changed, purely for Rosetta's benefit; the
        # real CCD identity is kept as ligand_comp_id in the summary tables.
        ligand_lines = [
            l[:17] + f"{rosetta_ligand_comp_id:>3s}" + l[20:] for l in ligand_lines
        ]

    # --- Write the cleaned PDB ---
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{pdb_id}_clean.pdb"
    with open(out_path, "w") as f:
        f.write(f"REMARK   Generated by Stage 4. Source: {pdb_id}.pdb\n")
        f.write(f"REMARK   EGFR chain: {egfr_chain}, ligand: {ligand_comp_id} "
                f"(chain {ligand_chain}, resnum {ligand_resnum})\n")
        if rosetta_ligand_comp_id != ligand_comp_id:
            f.write(
                f"REMARK   ROSETTA RESIDUE NAME CHANGED: {ligand_comp_id} -> "
                f"{rosetta_ligand_comp_id} (see config/ligand_overrides.yaml, "
                f"rosetta_ligand_code_reason)\n"
            )
        if covalent_info:
            f.write(
                f"REMARK   COVALENT BOND: {covalent_info['protein_resname']}"
                f"{covalent_info['protein_resnum']}.{covalent_info['protein_atom']} - "
                f"{covalent_info['ligand_resname']}.{covalent_info['ligand_atom']} "
                f"({covalent_info['bond_distance_A']} Å)\n"
            )
        for line in protein_lines:
            f.write(line + "\n")
        f.write("TER\n")
        for line in ligand_lines:
            f.write(line + "\n")
        f.write("END\n")

    # --- Residue mapping (native -> sequential pose numbering) ---
    mapping_dir.mkdir(parents=True, exist_ok=True)
    seen_res = []
    for line in protein_lines:
        key = (line[21], line[22:27])
        if key not in [r["_key"] for r in seen_res]:
            seen_res.append(
                {
                    "_key": key,
                    "native_chain": line[21],
                    "native_resnum": line[22:26].strip(),
                    "native_resname": line[17:20].strip(),
                }
            )
    for i, r in enumerate(seen_res, start=1):
        r["pose_resnum"] = i
        del r["_key"]
    ligand_pose_resnum = len(seen_res) + 1
    seen_res.append(
        {
            "native_chain": ligand_chain,
            "native_resnum": ligand_resnum,
            "native_resname": ligand_comp_id,
            "pose_resnum": ligand_pose_resnum,
        }
    )
    mapping_df = pd.DataFrame(seen_res)
    mapping_df.to_csv(mapping_dir / f"{pdb_id}_residue_mapping.csv", index=False)

    if bridging_waters:
        bridging_water_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bridging_waters).to_csv(
            bridging_water_dir / f"{pdb_id}_bridging_waters.csv", index=False
        )

    # --- Summary row ---
    protein_atom_count = len(protein_lines)
    ligand_atom_count = len(ligand_lines)
    n_protein_residues = len(seen_res) - 1

    warnings_list = []
    if n_missing > 0:
        warnings_list.append(f"{n_missing} missing residues (SEQRES vs ATOM): {gap_ranges}")
    if n_altloc_removed_p + n_altloc_removed_l > 0:
        warnings_list.append(
            f"{n_altloc_removed_p + n_altloc_removed_l} altloc atom(s) removed"
        )
    if disulfides:
        warnings_list.append(f"{len(disulfides)} disulfide bond(s): {disulfides}")
    if covalent_info:
        warnings_list.append(
            f"COVALENT: {covalent_info['protein_resname']}{covalent_info['protein_resnum']} - ligand"
        )
    if bridging_waters:
        warnings_list.append(f"{len(bridging_waters)} bridging-water candidate(s) (<4Å ligand)")

    row.update(
        {
            "status": "OK",
            "egfr_chain": egfr_chain,
            "ligand_comp_id": ligand_comp_id,
            "rosetta_ligand_comp_id": rosetta_ligand_comp_id,
            "protein_atom_count": protein_atom_count,
            "ligand_atom_count": ligand_atom_count,
            "n_protein_residues": n_protein_residues,
            "n_missing_residues": n_missing,
            "missing_residue_ranges": ";".join(gap_ranges) if gap_ranges else "",
            "n_altloc_atoms_removed": n_altloc_removed_p + n_altloc_removed_l,
            "n_disulfides": len(disulfides),
            "disulfide_pairs": str(disulfides) if disulfides else "",
            "is_covalent": covalent_info is not None,
            "covalent_protein_residue": (
                f"{covalent_info['protein_resname']}{covalent_info['protein_resnum']}"
                if covalent_info else ""
            ),
            "covalent_protein_atom": covalent_info["protein_atom"] if covalent_info else "",
            "covalent_ligand_atom": covalent_info["ligand_atom"] if covalent_info else "",
            "n_bridging_waters": len(bridging_waters),
            "warnings": " | ".join(warnings_list),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overrides", type=Path, default=Path("config/ligand_overrides.yaml")
    )
    parser.add_argument("--raw-pdb-dir", type=Path, default=Path("data/raw_pdb"))
    parser.add_argument("--raw-cif-dir", type=Path, default=Path("data/raw_cif"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--mapping-dir", type=Path, default=Path("results/metadata/residue_mapping")
    )
    parser.add_argument(
        "--bridging-water-dir", type=Path, default=Path("results/metadata/bridging_waters")
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("results/preparation_summary.csv")
    )
    parser.add_argument("--pdb-id", type=str, default=None)
    args = parser.parse_args()

    with open(args.overrides) as f:
        overrides = yaml.safe_load(f)

    pdb_ids = list(overrides.keys())
    if args.pdb_id:
        pdb_ids = [p for p in pdb_ids if p.lower() == args.pdb_id.lower()]

    rows = []
    for i, pdb_id in enumerate(pdb_ids, start=1):
        log.info("[%d/%d] preparing %s...", i, len(pdb_ids), pdb_id)
        try:
            row = prepare_one(
                pdb_id,
                overrides[pdb_id],
                args.raw_pdb_dir,
                args.raw_cif_dir,
                args.processed_dir,
                args.mapping_dir,
                args.bridging_water_dir,
            )
        except Exception as exc:
            log.error("  error processing %s: %s", pdb_id, exc)
            row = {"pdb_id": pdb_id, "status": "EXCEPTION", "warnings": str(exc)}
        rows.append(row)
        if row.get("status") != "OK":
            log.warning("  -> %s: %s", row.get("status"), row.get("warnings"))

    out_df = pd.DataFrame(rows)
    output_csv = (
        args.output_csv
        if not args.pdb_id
        else args.output_csv.with_name(f"test_single_{args.pdb_id.lower()}_prep.csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    log.info("Summary written: %s (%d rows)", output_csv, len(out_df))

    n_ok = (out_df["status"] == "OK").sum()
    log.info("Summary: OK=%d / %d", n_ok, len(out_df))


if __name__ == "__main__":
    main()
