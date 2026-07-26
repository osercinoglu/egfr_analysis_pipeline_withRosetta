"""
Stage 3: Determines the correct EGFR chain and the correct ligand copy for
each PDB via motif-based structural validation.

Steps:
1. For every protein chain in each PDB, checks whether the standard EGFR
   kinase-domain catalytic motifs (K745 VAIK, E762 aC-helix, HRD 835-837,
   DFG 855-857, hinge ~790-793) are present at the right position with the
   right identity (numbering-consistency verification).
2. Computes how close the ligand (the primary candidate identified in
   Stage 1) is to each chain's ATP binding pocket (minimum heavy-atom
   distances to K745, E762, hinge, HRD-Asp837, DFG-Asp855).
3. For multi-chain structures (2JIU, 3IKA, 5GTY, 5YU9), produces the
   rationale for which chain/ligand copy is selected.
4. Saves the results to config/ligand_overrides.yaml and
   results/metadata/chain_ligand_selection.csv.

This script does NOT modify the structure; it only decides which
chain/ligand copy will be used in Stage 4.
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
from Bio.PDB.Polypeptide import is_aa

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("chain_ligand_selection")

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# EGFR kinase domain catalytic motifs (UniProt P00533 / clinical mutation
# numbering, i.e. the same numbering used for T790M, L858R — verified via
# SIFTS alignment in Stage 1). At mutant positions (790, 858, 719, ...) the
# mutant identity is allowed in place of the expected wild-type identity.
MOTIF_POSITIONS = {
    745: {"expected": "K", "name": "VAIK_Lys"},
    762: {"expected": "E", "name": "alphaC_Glu"},
    835: {"expected": "H", "name": "HRD_His"},
    836: {"expected": "R", "name": "HRD_Arg"},
    837: {"expected": "D", "name": "HRD_Asp"},
    855: {"expected": "D", "name": "DFG_Asp"},
    856: {"expected": "F", "name": "DFG_Phe"},
    857: {"expected": "G", "name": "DFG_Gly"},
}
# Known clinical/engineered mutation positions (if motif validation sees a
# different identity than expected at these positions, it is not counted as
# an error).
KNOWN_MUTATION_POSITIONS = {719, 790, 858, 865, 866, 867, 948}

HINGE_RANGE = range(788, 796)  # hinge region (~788-795), includes gatekeeper T790
POCKET_REFERENCE_ATOMS = {
    "VAIK_Lys745": 745,
    "alphaC_Glu762": 762,
    "HRD_Asp837": 837,
    "DFG_Asp855": 855,
}


def get_chain_residue_map(chain) -> dict[int, str]:
    """Maps every residue number in the chain to its 1-letter code (standard AAs only)."""
    out = {}
    for res in chain:
        if is_aa(res, standard=True):
            out[res.id[1]] = AA3TO1.get(res.resname, "X")
    return out


def calibrate_numbering_offset(res_map: dict[int, str]) -> int | None:
    """Locates the HRD (His-Arg-Asp) and DFG (Asp-Phe-Gly) motifs by sequence
    search and computes the numbering offset relative to the standard
    numbering (HRD=835-837, DFG=855-857). Since different depositors can
    number EGFR relative to different reference points (e.g. including/
    excluding the signal peptide), we search for and confirm the motif for
    every chain instead of assuming fixed positions. Returns None if both
    motifs are found but disagree on the offset (ambiguous -> manual
    review)."""
    positions = sorted(res_map.keys())
    hrd_offsets = [
        835 - p for p in positions
        if res_map.get(p) == "H" and res_map.get(p + 1) == "R" and res_map.get(p + 2) == "D"
        and 700 <= p <= 900  # avoid false positives outside the kinase domain
    ]
    dfg_offsets = [
        855 - p for p in positions
        if res_map.get(p) == "D" and res_map.get(p + 1) == "F" and res_map.get(p + 2) == "G"
        and 700 <= p <= 900
    ]
    if len(hrd_offsets) == 1 and len(dfg_offsets) == 1 and hrd_offsets[0] == dfg_offsets[0]:
        return hrd_offsets[0]
    if len(hrd_offsets) == 1 and not dfg_offsets:
        return hrd_offsets[0]
    if len(dfg_offsets) == 1 and not hrd_offsets:
        return dfg_offsets[0]
    return None  # ambiguous: motif not found, or multiple/conflicting candidates


def check_motifs(res_map: dict[int, str]) -> dict:
    """Checks whether the EGFR catalytic motifs are correct in a chain, first
    calibrating the numbering offset."""
    offset = calibrate_numbering_offset(res_map)
    results: dict = {"_numbering_offset": offset}
    if offset is None:
        results["_n_ok"] = 0
        results["_n_checked"] = 0
        results["_n_missing"] = len(MOTIF_POSITIONS)
        results["_all_ok"] = False
        results["_calibration_status"] = "FAILED_MOTIF_NOT_FOUND"
        return results

    n_ok, n_checked, n_missing = 0, 0, 0
    for pos, info in MOTIF_POSITIONS.items():
        actual = res_map.get(pos - offset)
        if actual is None:
            results[info["name"]] = "MISSING"
            n_missing += 1
            continue
        n_checked += 1
        if actual == info["expected"] or (pos in KNOWN_MUTATION_POSITIONS):
            results[info["name"]] = f"OK({actual})"
            n_ok += 1
        else:
            results[info["name"]] = f"MISMATCH(expected={info['expected']},actual={actual})"
    results["_n_ok"] = n_ok
    results["_n_checked"] = n_checked
    results["_n_missing"] = n_missing
    results["_all_ok"] = (n_ok == n_checked) and n_checked >= 6
    results["_calibration_status"] = "OK" if offset == 0 else f"OFFSET_DETECTED({offset})"
    return results


def get_ligand_heavy_atoms(
    structure, ligand_comp_id: str
) -> list[tuple[str, str, np.ndarray, float, int]]:
    """Returns (chain_id, resseq, coords, mean_occupancy, n_altloc_atoms) for
    every ligand copy in the structure."""
    out = []
    for chain in structure[0]:
        for res in chain:
            if res.resname == ligand_comp_id:
                heavy_atoms = [atom for atom in res if atom.element != "H"]
                coords = np.array([atom.coord for atom in heavy_atoms])
                occupancies = [atom.get_occupancy() or 1.0 for atom in heavy_atoms]
                n_altloc = sum(1 for atom in heavy_atoms if atom.is_disordered())
                mean_occ = float(np.mean(occupancies)) if occupancies else 0.0
                out.append((chain.id, str(res.id[1]), coords, mean_occ, n_altloc))
    return out


def chain_completeness(chain) -> tuple[int, int]:
    """(number of modeled standard-AA residues, number of residues missing a CA)."""
    n_modeled = 0
    n_missing_ca = 0
    for res in chain:
        if is_aa(res, standard=True):
            n_modeled += 1
            if "CA" not in res:
                n_missing_ca += 1
    return n_modeled, n_missing_ca


def min_dist_to_residue(chain, resnum: int, ligand_coords: np.ndarray) -> float | None:
    try:
        res = chain[resnum]
    except KeyError:
        return None
    res_coords = np.array([atom.coord for atom in res if atom.element != "H"])
    if len(res_coords) == 0 or len(ligand_coords) == 0:
        return None
    dists = np.linalg.norm(
        ligand_coords[:, None, :] - res_coords[None, :, :], axis=-1
    )
    return float(dists.min())


def min_dist_to_hinge(chain, ligand_coords: np.ndarray, offset: int = 0) -> float | None:
    best = None
    for pos in HINGE_RANGE:
        d = min_dist_to_residue(chain, pos - offset, ligand_coords)
        if d is not None and (best is None or d < best):
            best = d
    return best


def analyze_pdb(pdb_id: str, ligand_comp_id: str, raw_pdb_dir: Path) -> dict:
    parser = PDBParser(QUIET=True)
    pdb_path = raw_pdb_dir / f"{pdb_id}.pdb"
    structure = parser.get_structure(pdb_id, str(pdb_path))
    model = structure[0]

    protein_chains = [c.id for c in model if any(is_aa(r, standard=True) for r in c)]
    ligand_instances = get_ligand_heavy_atoms(structure, ligand_comp_id)

    chain_reports = {}
    for chain_id in protein_chains:
        res_map = get_chain_residue_map(model[chain_id])
        motif_result = check_motifs(res_map)
        chain_reports[chain_id] = motif_result

    chain_completeness_map = {
        chain_id: chain_completeness(model[chain_id]) for chain_id in protein_chains
    }

    ligand_reports = []
    for lig_chain, lig_resnum, coords, mean_occ, n_altloc in ligand_instances:
        best_chain = None
        best_score = None
        per_chain_dists = {}
        for chain_id in protein_chains:
            chain = model[chain_id]
            offset = chain_reports.get(chain_id, {}).get("_numbering_offset")
            if offset is None:
                # Motif calibration failed for this chain; the offset cannot
                # be assumed, so this chain is excluded from the pocket
                # distance calculation.
                per_chain_dists[chain_id] = {"mean_pocket_dist": None, "calibration": "FAILED"}
                continue
            d_k745 = min_dist_to_residue(chain, 745 - offset, coords)
            d_e762 = min_dist_to_residue(chain, 762 - offset, coords)
            d_hrd = min_dist_to_residue(chain, 837 - offset, coords)
            d_dfg = min_dist_to_residue(chain, 855 - offset, coords)
            d_hinge = min_dist_to_hinge(chain, coords, offset=offset)
            valid = [d for d in [d_k745, d_e762, d_hrd, d_dfg, d_hinge] if d is not None]
            score = float(np.mean(valid)) if valid else None
            per_chain_dists[chain_id] = {
                "d_VAIK_Lys745": d_k745,
                "d_alphaC_Glu762": d_e762,
                "d_HRD_Asp837": d_hrd,
                "d_DFG_Asp855": d_dfg,
                "d_hinge": d_hinge,
                "mean_pocket_dist": score,
            }
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_chain = chain_id

        is_in_pocket = best_score is not None and best_score < 10.0
        n_modeled, n_missing_ca = chain_completeness_map.get(best_chain, (0, 999))
        ligand_reports.append(
            {
                "ligand_chain": lig_chain,
                "ligand_resnum": lig_resnum,
                "closest_protein_chain": best_chain,
                "mean_pocket_distance_A": best_score,
                "is_in_atp_pocket": is_in_pocket,
                "mean_occupancy": mean_occ,
                "n_altloc_atoms": n_altloc,
                "chain_n_modeled_residues": n_modeled,
                "chain_n_missing_ca": n_missing_ca,
                "per_chain_distances": per_chain_dists,
            }
        )

    return {
        "protein_chains": protein_chains,
        "chain_motif_reports": chain_reports,
        "ligand_instances": ligand_reports,
    }


def select_best_copy(analysis: dict, egfr_chains_expected: list[str]) -> dict:
    """Decides which chain/ligand pair to use for multi-copy structures."""
    valid_egfr_chains = [
        c for c in analysis["protein_chains"]
        if analysis["chain_motif_reports"].get(c, {}).get("_all_ok")
    ]
    pocket_ligands = [
        lig for lig in analysis["ligand_instances"] if lig["is_in_atp_pocket"]
    ]

    if not pocket_ligands:
        return {
            "selected_chain": None,
            "selected_ligand_chain": None,
            "selected_ligand_resnum": None,
            "selection_status": "NO_POCKET_LIGAND_FOUND",
            "reason": "No ligand copy is sufficiently close to the ATP pocket (< 10 Å motif average).",
        }

    # Priority order: (1) pocket distance (Stage 3 rule: copies within 1 Å
    # tolerance are considered "equivalent"), and among equivalent copies:
    # (2) the most complete chain (fewest missing CA), (3) highest ligand
    # occupancy, (4) fewest alternate-conformation atoms.
    min_dist = min(lig["mean_pocket_distance_A"] for lig in pocket_ligands)
    near_best = [
        lig for lig in pocket_ligands
        if lig["mean_pocket_distance_A"] <= min_dist + 1.0
    ]
    best = min(
        near_best,
        key=lambda x: (
            x["chain_n_missing_ca"],
            -x["mean_occupancy"],
            x["n_altloc_atoms"],
            x["mean_pocket_distance_A"],
        ),
    )
    n_valid_motif_chains = len(valid_egfr_chains)

    tiebreak_note = (
        f" {len(near_best)} copies were equivalent in distance (± 1 Å); "
        f"disambiguated by completeness (missing CA: {best['chain_n_missing_ca']}), "
        f"occupancy ({best['mean_occupancy']:.2f}) and altloc "
        f"({best['n_altloc_atoms']} atom(s))."
        if len(near_best) > 1
        else ""
    )
    reason = (
        f"Selected ligand copy (chain {best['ligand_chain']}, resnum {best['ligand_resnum']}) "
        f"is closest to the ATP pocket reference points (mean {best['mean_pocket_distance_A']:.2f} Å), "
        f"nearest protein chain {best['closest_protein_chain']}.{tiebreak_note}"
    )
    if len(pocket_ligands) > 1:
        others = [
            f"{lig['ligand_chain']}/{lig['ligand_resnum']} ({lig['mean_pocket_distance_A']:.2f} Å)"
            for lig in pocket_ligands
            if lig is not best
        ]
        reason += f" Other in-pocket copies (not used): {', '.join(others)}."

    return {
        "selected_chain": best["closest_protein_chain"],
        "selected_ligand_chain": best["ligand_chain"],
        "selected_ligand_resnum": best["ligand_resnum"],
        "selection_status": "OK" if n_valid_motif_chains >= 1 else "MOTIF_CHECK_FAILED",
        "reason": reason,
        "n_pocket_ligand_copies": len(pocket_ligands),
        "n_valid_motif_chains": n_valid_motif_chains,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path, default=Path("results/metadata/egfr_ligand_inventory.csv")
    )
    parser.add_argument("--raw-pdb-dir", type=Path, default=Path("data/raw_pdb"))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("results/metadata/chain_ligand_selection.csv")
    )
    parser.add_argument(
        "--overrides-yaml", type=Path, default=Path("config/ligand_overrides.yaml")
    )
    parser.add_argument("--pdb-id", type=str, default=None)
    args = parser.parse_args()

    inv = pd.read_csv(args.inventory)
    if args.pdb_id:
        inv = inv[inv["pdb_id"].str.lower() == args.pdb_id.lower()]

    rows = []
    overrides = {}
    for i, r in enumerate(inv.itertuples(), start=1):
        log.info("[%d/%d] analyzing %s...", i, len(inv), r.pdb_id)
        try:
            analysis = analyze_pdb(r.pdb_id, r.ligand_comp_id, args.raw_pdb_dir)
        except Exception as exc:
            log.error("  error processing %s: %s", r.pdb_id, exc)
            rows.append(
                {
                    "pdb_id": r.pdb_id,
                    "selection_status": "ANALYSIS_ERROR",
                    "reason": str(exc),
                }
            )
            continue

        egfr_chains_expected = str(r.egfr_chain).split(",")
        selection = select_best_copy(analysis, egfr_chains_expected)

        motif_summary = {
            chain_id: f"{rep['_n_ok']}/{rep['_n_checked']} OK ({rep['_n_missing']} missing)"
            for chain_id, rep in analysis["chain_motif_reports"].items()
        }

        row = {
            "pdb_id": r.pdb_id,
            "n_protein_chains": len(analysis["protein_chains"]),
            "n_ligand_copies": len(analysis["ligand_instances"]),
            "motif_summary": str(motif_summary),
            **selection,
        }
        rows.append(row)

        if selection["selected_chain"]:
            overrides[r.pdb_id] = {
                "egfr_chain": selection["selected_chain"],
                "ligand_chain": selection["selected_ligand_chain"],
                "ligand_resnum": selection["selected_ligand_resnum"],
                "ligand_comp_id": r.ligand_comp_id,
                "selection_reason": selection["reason"],
            }

    out_df = pd.DataFrame(rows)
    output_csv = (
        args.output_csv
        if not args.pdb_id
        else args.output_csv.with_name(f"test_single_{args.pdb_id.lower()}_selection.csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    log.info("Selection table written: %s (%d rows)", output_csv, len(out_df))

    if not args.pdb_id:
        args.overrides_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(args.overrides_yaml, "w") as f:
            yaml.dump(overrides, f, sort_keys=True, default_flow_style=False)
        log.info("Ligand override records written: %s", args.overrides_yaml)

    n_ok = (out_df["selection_status"] == "OK").sum()
    n_failed = len(out_df) - n_ok
    log.info("Summary: OK=%d, OTHER=%d (total %d)", n_ok, n_failed, len(out_df))
    if n_failed:
        problem_cols = ["pdb_id", "selection_status", "reason"]
        print(out_df.loc[out_df["selection_status"] != "OK", problem_cols].to_string(index=False))


if __name__ == "__main__":
    main()
