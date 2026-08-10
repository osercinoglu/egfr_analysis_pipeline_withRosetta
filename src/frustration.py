"""
Atomistic Frustration Pipeline — Core Engine

Chen et al., Nature Communications 11, 5944 (2020)
DOI: 10.1038/s41467-020-19560-9

Eq. 1: Frustration index (Z-score)
    F_ij = (mean(E_ij_decoy) - E_ij_native) / std(E_ij_decoy)

Eq. 2: Many-body pairwise energy correction
    E_ij = e_ij
         + 0.5 * Σ_{k ∈ contacts(i), k≠j} e_ik
         + 0.5 * Σ_{l ∈ contacts(j), l≠i} e_jl
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DECOY_WORKER_STATE: dict[str, Any] = {}

# PyRosetta — warn if not installed, works fine after import otherwise
try:
    import pyrosetta
    from pyrosetta import Pose
    from pyrosetta.rosetta.core.scoring import ScoreFunction
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        RestrictToRepacking,
        PreventRepacking,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import (
        PackRotamersMover,
        MinMover,
    )
    _PYROSETTA_AVAILABLE = True
except ImportError:
    _PYROSETTA_AVAILABLE = False
    logger.warning("PyRosetta not found. Only the data preparation steps will work.")


# ---------------------------------------------------------------------------
# Helper: initialize PyRosetta
# ---------------------------------------------------------------------------

def init_pyrosetta(silent: bool = True):
    """Initializes PyRosetta (safe to call more than once)."""
    if not _PYROSETTA_AVAILABLE:
        raise RuntimeError("PyRosetta is not installed.")
    flags = "-mute all" if silent else ""
    pyrosetta.init(flags)


# ---------------------------------------------------------------------------
# 1. Contact lists
# ---------------------------------------------------------------------------

def get_protein_contacts(
    pose: "Pose",
    cutoff: float = 10.0,
    seq_sep_min: int = 4,
) -> list[tuple[int, int]]:
    """
    Returns protein-protein contact pairs based on Cα-Cα distance.

    Only considers standard amino acid residues.
    Pairs with |i - j| < seq_sep_min are skipped (local backbone bonds).

    Args:
        pose: PyRosetta Pose object
        cutoff: Cα-Cα cutoff distance (Å)
        seq_sep_min: Minimum sequence separation

    Returns:
        List of (resi, resj) index pairs (1-indexed, i < j)
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
    Returns the indices of protein residues in contact with the ligand,
    based on ligand heavy-atom — protein Cα distance.

    A residue is counted as a "ligand contact" if any ligand heavy atom is
    within cutoff Å of its Cα.

    Args:
        pose: PyRosetta Pose object
        ligand_resnum: The ligand's residue number within the pose
        cutoff: heavy-atom – Cα cutoff distance (Å)

    Returns:
        List of protein residue numbers (1-indexed)
    """
    lig_res = pose.residue(ligand_resnum)
    n = pose.total_residue()
    contact_residues = []

    # All heavy-atom coordinates of the ligand
    lig_coords = []
    for atom_idx in range(1, lig_res.natoms() + 1):
        atom = lig_res.atom(atom_idx)
        # Skip hydrogens (atom_type_index 18 is usually H, or check element_type)
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
                break  # this residue is enough, no need to check other atoms

    return sorted(set(contact_residues))


# ---------------------------------------------------------------------------
# 2. Energy calculation (Eq. 2)
# ---------------------------------------------------------------------------

def _get_energy_graph(pose: "Pose", scorefxn: "ScoreFunction") -> Any:
    """Scores the pose and returns the EnergyGraph object."""
    scorefxn(pose)
    return pose.energies().energy_graph()


_FA_REP_ST = None  # ScoreType cache

def _fa_rep_score_type():
    """Returns the fa_rep ScoreType object (lazy init)."""
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
    Returns the direct pairwise energy between residues i and j.

    The edge (i,j) total is taken from the EnergyGraph; fa_rep can be
    excluded. This corresponds to the e_ij term in Eq. 2.

    Args:
        pose: A scored Pose
        scorefxn: The active ScoreFunction
        i, j: Residue indices (1-based)
        exclude_fa_rep: True → the fa_rep term is subtracted from the total

    Returns:
        e_ij (Rosetta energy unit, ~kcal/mol)
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
    Eq. 2 many-body energy correction:
        E_ij = e_ij
             + 0.5 * Σ_{k ∈ contacts(i), k≠j} e_ik
             + 0.5 * Σ_{l ∈ contacts(j), l≠i} e_jl

    Args:
        contact_partners_i: all of i's contact partners except j
        contact_partners_j: all of j's contact partners except i

    Returns:
        E_ij (many-body-corrected pairwise energy)
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
    Builds a dict mapping each residue to its list of contact partners.

    contacts: list of contact pairs [(i,j), ...]
    Returns: {resnum: [partner1, partner2, ...]}
    """
    partners: dict[int, list[int]] = {}
    for i, j in contacts:
        partners.setdefault(i, []).append(j)
        partners.setdefault(j, []).append(i)
    return partners


# ---------------------------------------------------------------------------
# 3. Decoy generation
# ---------------------------------------------------------------------------

def native_aa_frequency(pose: "Pose") -> dict[str, float]:
    """
    Protein-specific, position-independent amino acid frequency distribution.

    Counts the one-letter codes of every protein residue and returns a
    normalized frequency dict for the 20 amino acids.

    Returns:
        {'A': 0.08, 'C': 0.02, ...}  (sums to 1.0)
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
    Generates a single decoy pose:
      1. Replace every protein position with an amino acid sampled from aa_freq
      2. Repack side chains with the backbone held fixed
      3. 1 cycle of backbone-fixed FastRelax (resolves side-chain clashes)

    Args:
        pose: The original native pose (not modified — a clone is used)
        scorefxn: The score function
        aa_freq: Output of native_aa_frequency()
        seed: Random seed (None → uses global state)

    Returns:
        A new decoy Pose object (backbone = native, sequence = shuffled)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

    decoy = pose.clone()
    n = decoy.total_residue()

    # Amino acid names (one letter → 3-letter + full Rosetta name)
    aa_letters = list(aa_freq.keys())
    aa_probs   = list(aa_freq.values())

    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    for i in range(1, n + 1):
        res = decoy.residue(i)
        if not res.is_protein():
            continue
        new_aa_1let = np.random.choice(aa_letters, p=aa_probs)
        # Convert the one-letter code to a Rosetta residue type name
        from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code, name_from_aa
        aa_enum = aa_from_oneletter_code(new_aa_1let)
        res_name = name_from_aa(aa_enum)
        mutator = MutateResidue(i, res_name)
        mutator.apply(decoy)

    # Side-chain repack (backbone fixed)
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    packer = PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(decoy)

    # Short backbone-fixed minimization (chi only), to resolve side-chain
    # clashes introduced by the repack.
    from pyrosetta.rosetta.core.kinematics import MoveMap
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(True)
    mm.set_jump(False)
    min_mover = MinMover()
    min_mover.movemap(mm)
    min_mover.score_function(scorefxn)
    min_mover.min_type("lbfgs_armijo_nonmonotone")
    min_mover.tolerance(0.01)
    min_mover.apply(decoy)

    # MutateResidue preserves N/CA/C exactly but rebuilds the carbonyl O from
    # idealized bond geometry, which drifts ~0.5 Å from the (non-ideal)
    # crystallographic position — not a relax/minimizer artifact. Restore all
    # backbone atoms from the native pose explicitly so the decoy backbone is
    # bit-for-bit identical to native, as the frustration algorithm requires.
    from pyrosetta.rosetta.core.id import AtomID
    for i in range(1, n + 1):
        if not pose.residue(i).is_protein():
            continue
        for atom_name in ("N", "CA", "C", "O"):
            if pose.residue(i).has(atom_name) and decoy.residue(i).has(atom_name):
                atom_idx = decoy.residue(i).atom_index(atom_name)
                decoy.set_xyz(AtomID(atom_idx, i), pose.residue(i).xyz(atom_name))

    return decoy


def save_pose_pdb(pose: "Pose", output_path: str | Path) -> None:
    """Write a pose to a PDB file, creating parent directories as needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pose.dump_pdb(str(path))


# ---------------------------------------------------------------------------
# 4. Main frustration survey
# ---------------------------------------------------------------------------

def _decoy_worker_init(
    processed_pdb: str,
    params_file: str,
    score_function: str,
    contacts: list[tuple[int, int]],
    exclude_fa_rep: bool,
    decoy_structures_dir: str | None,
) -> None:
    """Initialize an isolated PyRosetta pose for spawned decoy workers."""
    import pyrosetta
    from pyrosetta import Pose, Vector1

    pyrosetta.init("-mute all")
    pose = Pose()
    residue_set = pyrosetta.generate_nonstandard_residue_set(
        pose, Vector1([params_file])
    )
    pyrosetta.pose_from_file(pose, residue_set, processed_pdb)
    scorefxn = pyrosetta.create_score_function(score_function)
    _DECOY_WORKER_STATE.update(
        pose=pose,
        scorefxn=scorefxn,
        contacts=contacts,
        partner_map=build_contact_partner_map(contacts),
        aa_freq=native_aa_frequency(pose),
        exclude_fa_rep=exclude_fa_rep,
        decoy_structures_dir=decoy_structures_dir,
    )


def _generate_decoy_batch(
    decoy_indices: list[int],
    seed: int,
) -> list[tuple[int, dict[tuple[int, int], float]]]:
    """Generate one deterministic batch of decoys in a spawned worker."""
    if not _DECOY_WORKER_STATE:
        raise RuntimeError("Decoy worker was not initialized")

    pose = _DECOY_WORKER_STATE["pose"]
    scorefxn = _DECOY_WORKER_STATE["scorefxn"]
    contacts = _DECOY_WORKER_STATE["contacts"]
    partner_map = _DECOY_WORKER_STATE["partner_map"]
    aa_freq = _DECOY_WORKER_STATE["aa_freq"]
    exclude_fa_rep = _DECOY_WORKER_STATE["exclude_fa_rep"]
    decoy_structures_dir = _DECOY_WORKER_STATE["decoy_structures_dir"]
    batch = []
    for decoy_index in decoy_indices:
        decoy = generate_decoy(pose, scorefxn, aa_freq, seed=seed + decoy_index)
        if decoy_structures_dir is not None:
            save_pose_pdb(
                decoy,
                Path(decoy_structures_dir) / f"decoy_{decoy_index + 1:04d}.pdb",
            )
        scorefxn(decoy)
        energies = {}
        for i, j in contacts:
            energies[(i, j)] = contact_energy_eq2(
                decoy,
                scorefxn,
                i,
                j,
                partner_map.get(i, []),
                partner_map.get(j, []),
                exclude_fa_rep,
            )
        batch.append((decoy_index, energies))
    return batch


def _generate_decoy_batch_task(
    task: tuple[list[int], int],
) -> list[tuple[int, dict[tuple[int, int], float]]]:
    """Unpack a pool task for incremental parallel result consumption."""
    return _generate_decoy_batch(*task)


def _checkpoint_decoy_energies(
    checkpoint_path: str | None,
    decoy_energies: dict[tuple[int, int], list[float]],
    completed_decoys: int,
) -> None:
    if checkpoint_path is None:
        return
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as checkpoint_file:
        pickle.dump(
            {
                "decoy_energies": decoy_energies,
                "completed_decoys": completed_decoys,
            },
            checkpoint_file,
        )


def _generate_parallel_decoys(
    decoy_energies: dict[tuple[int, int], list[float]],
    start_decoy: int,
    n_decoys: int,
    seed: int,
    n_jobs: int,
    processed_pdb: str,
    params_file: str,
    score_function: str,
    contacts: list[tuple[int, int]],
    exclude_fa_rep: bool,
    checkpoint_path: str | None,
    checkpoint_every: int,
    decoy_structures_dir: str | None,
) -> None:
    """Fill decoy energies with spawned workers and report incremental progress."""
    indices = list(range(start_decoy, n_decoys))
    if not indices:
        return

    worker_count = min(n_jobs, len(indices))
    logger.info(
        "Generating %s decoys with %s spawned workers.", len(indices), worker_count
    )
    tasks = [([decoy_index], seed) for decoy_index in indices]

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    context = mp.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=_decoy_worker_init,
        initargs=(
            processed_pdb,
            params_file,
            score_function,
            contacts,
            exclude_fa_rep,
            decoy_structures_dir,
        ),
    ) as pool:
        completed_decoys = start_decoy
        progress = (
            tqdm(total=len(indices), desc="Parallel decoys", unit="decoy")
            if tqdm is not None
            else None
        )
        try:
            for batch in pool.imap(_generate_decoy_batch_task, tasks):
                for _, energies in batch:
                    for contact, energy in energies.items():
                        decoy_energies[contact].append(energy)
                completed_decoys += len(batch)
                if progress is not None:
                    progress.update(len(batch))
                if (
                    completed_decoys % checkpoint_every == 0
                    or completed_decoys == n_decoys
                ):
                    _checkpoint_decoy_energies(
                        checkpoint_path,
                        decoy_energies,
                        completed_decoys,
                    )
        finally:
            if progress is not None:
                progress.close()


def run_frustration_survey(
    pose: "Pose",
    scorefxn: "ScoreFunction",
    contacts: list[tuple[int, int]],
    n_decoys: int = 1000,
    seed: int = 0,
    exclude_fa_rep: bool = True,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
    n_jobs: int = 1,
    worker_pose_paths: tuple[str, str] | None = None,
    score_function: str | None = None,
    decoy_structures_dir: str | None = None,
) -> pd.DataFrame:
    """
    Computes the frustration index (Eq. 1) for every contact pair.

    Algorithm:
      1. Compute all contact E_ij values for the native pose (Eq. 2)
      2. Generate n_decoys decoys, collecting all E_ij from each one
      3. Compute the Z-score for every contact (Eq. 1)

    Checkpoint mechanism:
      If checkpoint_path is given, an intermediate result is pickled every
      checkpoint_every decoys; the run can be resumed from where it left off.

    Args:
        contacts: output of get_protein_contacts() or get_ligand_contacts()
        n_decoys: number of decoys to generate
        seed: initial random seed
        checkpoint_path: path to the checkpoint file (None → no saving)
        checkpoint_every: how often (in decoys) to checkpoint
        n_jobs: number of spawned workers for decoys (1 → sequential)
        worker_pose_paths: processed PDB and ligand params paths for workers
        score_function: Rosetta score function name used by spawned workers
        decoy_structures_dir: directory where decoy PDBs should be written

    Returns:
        DataFrame with columns:
          resi, resj, F_index, E_native, decoy_mean, decoy_std,
          frustration_class
    """
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(iterable, **kwargs):  # type: ignore
            return iterable

    # Contact partner map (for Eq. 2)
    partner_map = build_contact_partner_map(contacts)

    # --- 1. Native energies ---
    logger.info("Computing native energies...")
    scorefxn(pose)
    native_energies: dict[tuple[int, int], float] = {}
    for i, j in contacts:
        pi = partner_map.get(i, [])
        pj = partner_map.get(j, [])
        native_energies[(i, j)] = contact_energy_eq2(
            pose, scorefxn, i, j, pi, pj, exclude_fa_rep
        )

    # --- 2. Load checkpoint (if present) ---
    decoy_energies: dict[tuple[int, int], list[float]] = {
        c: [] for c in contacts
    }
    start_decoy = 0

    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        decoy_energies = ckpt["decoy_energies"]
        start_decoy = ckpt["completed_decoys"]
        logger.info(f"Checkpoint loaded: {start_decoy} decoys already completed.")

    # --- 3. Decoy loop ---
    aa_freq = native_aa_frequency(pose)
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    if n_jobs > 1:
        if worker_pose_paths is None or score_function is None:
            raise ValueError(
                "Parallel decoys require worker_pose_paths and score_function"
            )
        _generate_parallel_decoys(
            decoy_energies,
            start_decoy,
            n_decoys,
            seed,
            n_jobs,
            *worker_pose_paths,
            score_function,
            contacts,
            exclude_fa_rep,
            checkpoint_path,
            checkpoint_every,
            decoy_structures_dir,
        )
    else:
        for d_idx in _tqdm(
            range(start_decoy, n_decoys),
            desc="Decoys",
            initial=start_decoy,
            total=n_decoys,
        ):
            decoy_seed = seed + d_idx
            decoy = generate_decoy(pose, scorefxn, aa_freq, seed=decoy_seed)
            if decoy_structures_dir is not None:
                save_pose_pdb(
                    decoy,
                    Path(decoy_structures_dir) / f"decoy_{d_idx + 1:04d}.pdb",
                )
            scorefxn(decoy)

            for i, j in contacts:
                pi = partner_map.get(i, [])
                pj = partner_map.get(j, [])
                e = contact_energy_eq2(
                    decoy, scorefxn, i, j, pi, pj, exclude_fa_rep
                )
                decoy_energies[(i, j)].append(e)

            if (d_idx + 1) % checkpoint_every == 0:
                _checkpoint_decoy_energies(
                    checkpoint_path, decoy_energies, d_idx + 1
                )

    # --- 4. Eq. 1: Z-score ---
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
            f_idx = (mu - e_nat) / sigma

        # Frustration class
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
# 5. EGFR analysis: ligand-pocket frustration summary
# ---------------------------------------------------------------------------

def summarize_ligand_frustration(
    df_frustration: pd.DataFrame,
    ligand_contacts: list[int],
) -> dict:
    """
    Filters contact pairs that include a ligand-pocket residue and
    summarizes the frustration distribution.

    A "ligand contact" is a contact pair where at least one side is in
    ligand_contacts.

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
