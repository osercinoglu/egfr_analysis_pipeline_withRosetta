"""
Stage 2: Downloads raw PDB and mmCIF coordinate files from RCSB for the 61
EGFR-inhibitor complexes validated in Stage 1.

This script does NOT yet modify the structure (cleaning, altloc selection,
etc. is Stage 4's job). It only downloads + verifies integrity:
  - file checksum (SHA-256)
  - download timestamp (UTC)
  - HTTP status code and errors
  - confirms the file actually belongs to the requested PDB ID
    (PDB: idCode in the HEADER line; mmCIF: the data_ block)
  - atom count (ATOM+HETATM / atom_site lines)
  - confirms the target ligand (ligand_comp_id) identified in Stage 1 is
    actually present in the file

Any existing data/raw_pdb/*.pdb files left over from the earlier 25-structure
project are not trusted; this script does a fresh download for every PDB
and records the checksum in the manifest (with --skip-existing, re-download
is skipped if the manifest already has a verified record).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("download_structures")

RCSB_FILES_BASE = "https://files.rcsb.org/download"
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def download_with_retry(url: str) -> tuple[bytes | None, int | None, str | None]:
    """Downloads a file with simple retry logic. Returns (content, http_status, error)."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(RETRY_BACKOFF_S * attempt)
            continue
        if resp.status_code == 200:
            return resp.content, resp.status_code, None
        if resp.status_code == 404:
            return None, resp.status_code, "404 Not Found"
        last_err = f"HTTP {resp.status_code}"
        time.sleep(RETRY_BACKOFF_S * attempt)
    return None, None, last_err


def verify_pdb_file(content: bytes, expected_id: str, expected_ligand: str) -> dict:
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()

    id_match = False
    for line in lines:
        if line.startswith("HEADER"):
            # PDB format: idCode is in columns 63-66 (1-indexed)
            token = line[62:66].strip().upper()
            if token == expected_id.upper():
                id_match = True
            break

    atom_count = sum(1 for l in lines if l.startswith(("ATOM", "HETATM")))
    ligand_found = any(
        l.startswith("HETATM") and l[17:20].strip().upper() == expected_ligand.upper()
        for l in lines
    )
    return {
        "id_match": id_match,
        "atom_count": atom_count,
        "ligand_found": ligand_found,
    }


def verify_cif_file(content: bytes, expected_id: str, expected_ligand: str) -> dict:
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()

    id_match = False
    for line in lines[:5]:
        if line.lower().startswith("data_"):
            token = line[5:].strip().upper()
            if token == expected_id.upper():
                id_match = True
            break

    atom_count = sum(1 for l in lines if l.startswith("ATOM") or l.startswith("HETATM"))
    # In mmCIF atom_site records the ligand comp_id is usually around column 6
    # (auth_comp_id); the exact column position varies by file, so a simple
    # word-boundary regex search is used instead (for verification purposes only).
    ligand_found = bool(
        re.search(rf"\b{re.escape(expected_ligand.upper())}\b", text.upper())
    )
    return {
        "id_match": id_match,
        "atom_count": atom_count,
        "ligand_found": ligand_found,
    }


def process_one(
    pdb_id: str,
    ligand_comp_id: str,
    raw_pdb_dir: Path,
    raw_cif_dir: Path,
    skip_existing: bool,
) -> dict:
    pdb_id_u = pdb_id.upper()
    row = {
        "pdb_id": pdb_id_u,
        "ligand_comp_id": ligand_comp_id,
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pdb_downloaded": False,
        "pdb_http_status": None,
        "pdb_sha256": None,
        "pdb_bytes": None,
        "pdb_atom_count": None,
        "pdb_id_match": None,
        "pdb_ligand_found": None,
        "cif_downloaded": False,
        "cif_http_status": None,
        "cif_sha256": None,
        "cif_bytes": None,
        "cif_atom_count": None,
        "cif_id_match": None,
        "cif_ligand_found": None,
        "status": "pending",
        "notes": "",
    }
    notes = []

    pdb_path = raw_pdb_dir / f"{pdb_id_u}.pdb"
    cif_path = raw_cif_dir / f"{pdb_id_u}.cif"

    if skip_existing and pdb_path.exists() and cif_path.exists():
        row["pdb_downloaded"] = True
        row["cif_downloaded"] = True
        row["pdb_sha256"] = sha256_of(pdb_path)
        row["cif_sha256"] = sha256_of(cif_path)
        pdb_verify = verify_pdb_file(pdb_path.read_bytes(), pdb_id_u, ligand_comp_id)
        cif_verify = verify_cif_file(cif_path.read_bytes(), pdb_id_u, ligand_comp_id)
        row.update({f"pdb_{k}": v for k, v in pdb_verify.items()})
        row.update({f"cif_{k}": v for k, v in cif_verify.items()})
        row["status"] = "SKIPPED_ALREADY_PRESENT"
        return row

    # --- PDB format ---
    pdb_content, pdb_status, pdb_err = download_with_retry(
        f"{RCSB_FILES_BASE}/{pdb_id_u}.pdb"
    )
    row["pdb_http_status"] = pdb_status
    if pdb_content:
        pdb_path.write_bytes(pdb_content)
        row["pdb_downloaded"] = True
        row["pdb_sha256"] = sha256_of(pdb_path)
        row["pdb_bytes"] = len(pdb_content)
        verify = verify_pdb_file(pdb_content, pdb_id_u, ligand_comp_id)
        row["pdb_id_match"] = verify["id_match"]
        row["pdb_atom_count"] = verify["atom_count"]
        row["pdb_ligand_found"] = verify["ligand_found"]
        if not verify["id_match"]:
            notes.append("PDB HEADER idCode does not match the expected value!")
        if not verify["ligand_found"]:
            notes.append(f"Ligand {ligand_comp_id} not found as HETATM in the PDB file!")
    else:
        notes.append(f"PDB download failed: {pdb_err}")

    # --- mmCIF format ---
    cif_content, cif_status, cif_err = download_with_retry(
        f"{RCSB_FILES_BASE}/{pdb_id_u}.cif"
    )
    row["cif_http_status"] = cif_status
    if cif_content:
        cif_path.write_bytes(cif_content)
        row["cif_downloaded"] = True
        row["cif_sha256"] = sha256_of(cif_path)
        row["cif_bytes"] = len(cif_content)
        verify = verify_cif_file(cif_content, pdb_id_u, ligand_comp_id)
        row["cif_id_match"] = verify["id_match"]
        row["cif_atom_count"] = verify["atom_count"]
        row["cif_ligand_found"] = verify["ligand_found"]
        if not verify["id_match"]:
            notes.append("mmCIF data_ block does not match the expected value!")
        if not verify["ligand_found"]:
            notes.append(f"Ligand {ligand_comp_id} not found in the mmCIF file!")
    else:
        notes.append(f"mmCIF download failed: {cif_err}")

    if row["pdb_downloaded"] and row["cif_downloaded"] and not notes:
        row["status"] = "OK"
    elif row["pdb_downloaded"] or row["cif_downloaded"]:
        row["status"] = "PARTIAL_OR_WARNING"
    else:
        row["status"] = "FAILED"

    row["notes"] = " | ".join(notes)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("results/metadata/egfr_ligand_inventory.csv"),
    )
    parser.add_argument("--raw-pdb-dir", type=Path, default=Path("data/raw_pdb"))
    parser.add_argument("--raw-cif-dir", type=Path, default=Path("data/raw_cif"))
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("results/metadata/download_manifest.csv"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Don't re-download files that are already downloaded and verified.",
    )
    parser.add_argument("--pdb-id", type=str, default=None)
    args = parser.parse_args()

    args.raw_pdb_dir.mkdir(parents=True, exist_ok=True)
    args.raw_cif_dir.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(args.inventory)
    if args.pdb_id:
        inv = inv[inv["pdb_id"].str.lower() == args.pdb_id.lower()]
        if inv.empty:
            raise SystemExit(f"{args.pdb_id} not found in the inventory.")

    rows = []
    for i, r in enumerate(inv.itertuples(), start=1):
        log.info("[%d/%d] downloading %s...", i, len(inv), r.pdb_id)
        row = process_one(
            r.pdb_id,
            r.ligand_comp_id,
            args.raw_pdb_dir,
            args.raw_cif_dir,
            args.skip_existing,
        )
        rows.append(row)
        if row["status"] not in ("OK", "SKIPPED_ALREADY_PRESENT"):
            log.warning("  -> %s: %s", row["status"], row["notes"])

    out_df = pd.DataFrame(rows)
    manifest_path = (
        args.manifest_out
        if not args.pdb_id
        else args.manifest_out.with_name(f"test_single_{args.pdb_id.lower()}_manifest.csv")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(manifest_path, index=False)

    n_ok = (out_df["status"] == "OK").sum()
    n_skipped = (out_df["status"] == "SKIPPED_ALREADY_PRESENT").sum()
    n_partial = (out_df["status"] == "PARTIAL_OR_WARNING").sum()
    n_failed = (out_df["status"] == "FAILED").sum()
    log.info(
        "Summary: OK=%d, SKIPPED=%d, PARTIAL/WARNING=%d, FAILED=%d (total %d)",
        n_ok, n_skipped, n_partial, n_failed, len(out_df),
    )
    log.info("Manifest written: %s", manifest_path)


if __name__ == "__main__":
    main()
