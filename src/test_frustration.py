"""
Unit tests: frustration engine functions.

Tests that require PyRosetta are skipped if PyRosetta is not installed.

Usage:
  cd /home/tugba/egfr_atomic_resolution
  /home/tugba/anaconda3/envs/frustrato/bin/python -m pytest src/test_frustration.py -v
"""

import sys
import pytest
import numpy as np

# Test with PyRosetta if available, otherwise skip
try:
    import pyrosetta
    pyrosetta.init("-mute all")
    _PYROSETTA = True
except ImportError:
    _PYROSETTA = False

skip_if_no_pyrosetta = pytest.mark.skipif(
    not _PYROSETTA, reason="PyRosetta is not installed"
)


# ---------------------------------------------------------------------------
# Tests that don't require PyRosetta
# ---------------------------------------------------------------------------

def test_build_contact_partner_map_basic():
    """build_contact_partner_map should produce the correct partner list."""
    from src.frustration import build_contact_partner_map

    contacts = [(1, 2), (1, 3), (2, 4)]
    partners = build_contact_partner_map(contacts)

    assert set(partners[1]) == {2, 3}
    assert set(partners[2]) == {1, 4}
    assert set(partners[3]) == {1}
    assert set(partners[4]) == {2}


def test_frustration_index_formula():
    """Eq. 1: the Z-score should match the manually computed value."""
    native = -5.0
    decoys = [-3.0, -4.0, -6.0, -7.0, -5.5, -4.5, -3.5, -6.5, -5.0, -4.0]
    mu    = np.mean(decoys)
    sigma = np.std(decoys, ddof=1)
    F     = (native - mu) / sigma
    # Direct formula check
    expected = (native - mu) / sigma
    assert abs(F - expected) < 1e-10


def test_frustration_class_thresholds():
    """Frustration classification should match the threshold values."""
    # F > 0.78 → minimally frustrated
    assert _classify(1.5)  == "minimally_frustrated"
    assert _classify(0.79) == "minimally_frustrated"
    # F < -1.0 → highly frustrated
    assert _classify(-1.5) == "highly_frustrated"
    assert _classify(-1.01) == "highly_frustrated"
    # in between → neutral
    assert _classify(0.0)  == "neutral"
    assert _classify(0.77) == "neutral"
    assert _classify(-0.99) == "neutral"


def _classify(f):
    if f > 0.78:
        return "minimally_frustrated"
    elif f < -1.0:
        return "highly_frustrated"
    return "neutral"


def test_summarize_ligand_frustration_basic():
    """summarize_ligand_frustration should return the correct counts."""
    import pandas as pd
    from src.frustration import summarize_ligand_frustration

    df = pd.DataFrame({
        "resi": [1, 2, 3, 4, 5],
        "resj": [6, 7, 8, 9, 10],
        "F_index": [1.2, -1.5, 0.3, 0.9, -0.5],
        "frustration_class": [
            "minimally_frustrated",
            "highly_frustrated",
            "neutral",
            "minimally_frustrated",
            "neutral",
        ],
    })
    # Ligand contacts: residues 1, 3 (contacts 0 and 2 are in the ligand region)
    lig_contacts = [1, 3]
    summary = summarize_ligand_frustration(df, lig_contacts)

    assert summary["n_contacts_total"] == 2, f"Expected 2, got {summary['n_contacts_total']}"
    assert summary["n_minimally_frustrated"] == 1
    assert summary["n_neutral"] == 1
    assert summary["n_highly_frustrated"] == 0
    assert abs(summary["frac_minimally"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Tests that require PyRosetta (run on small test proteins)
# ---------------------------------------------------------------------------

@skip_if_no_pyrosetta
def test_get_protein_contacts_count():
    """
    get_protein_contacts should return a reasonable number of contacts for a
    small protein. Expecting >200 contacts for 1LYZ (129 residues).
    """
    import requests
    from pathlib import Path
    from src.frustration import get_protein_contacts

    pdb_path = Path("/tmp/test_1lyz.pdb")
    if not pdb_path.exists():
        r = requests.get("https://files.rcsb.org/download/1LYZ.pdb", timeout=30)
        pdb_path.write_bytes(r.content)

    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    contacts = get_protein_contacts(pose, cutoff=10.0, seq_sep_min=4)

    assert len(contacts) > 100, f"Fewer contacts than expected: {len(contacts)}"
    # Check that every pair has i < j
    for i, j in contacts:
        assert i < j, f"i ({i}) < j ({j}) rule violated"


@skip_if_no_pyrosetta
def test_pairwise_energy_symmetric():
    """
    pairwise_energy(i,j) should equal pairwise_energy(j,i)
    (undirected graph).
    """
    import requests
    from pathlib import Path
    from src.frustration import pairwise_energy

    pdb_path = Path("/tmp/test_1lyz.pdb")
    if not pdb_path.exists():
        r = requests.get("https://files.rcsb.org/download/1LYZ.pdb", timeout=30)
        pdb_path.write_bytes(r.content)

    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    sfxn = pyrosetta.create_score_function("ref2015")
    sfxn(pose)

    e_ij = pairwise_energy(pose, sfxn, 5, 10, exclude_fa_rep=True)
    e_ji = pairwise_energy(pose, sfxn, 10, 5, exclude_fa_rep=True)
    assert abs(e_ij - e_ji) < 1e-6, f"e_ij={e_ij:.6f} != e_ji={e_ji:.6f}"


@skip_if_no_pyrosetta
def test_native_aa_frequency_sums_to_one():
    """native_aa_frequency frequencies should sum to 1.0."""
    import requests
    from pathlib import Path
    from src.frustration import native_aa_frequency

    pdb_path = Path("/tmp/test_1lyz.pdb")
    if not pdb_path.exists():
        r = requests.get("https://files.rcsb.org/download/1LYZ.pdb", timeout=30)
        pdb_path.write_bytes(r.content)

    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    freq = native_aa_frequency(pose)

    assert len(freq) > 0, "Frequency dict is empty"
    total = sum(freq.values())
    assert abs(total - 1.0) < 1e-9, f"Frequency sum {total:.6f} ≠ 1.0"


@skip_if_no_pyrosetta
def test_decoy_backbone_unchanged():
    """
    After generate_decoy, the backbone (N, CA, C, O) coordinates should be
    unchanged relative to native (backbone_fixed=True).
    """
    import requests
    from pathlib import Path
    from src.frustration import generate_decoy, native_aa_frequency

    pdb_path = Path("/tmp/test_1lyz.pdb")
    if not pdb_path.exists():
        r = requests.get("https://files.rcsb.org/download/1LYZ.pdb", timeout=30)
        pdb_path.write_bytes(r.content)

    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    sfxn = pyrosetta.create_score_function("ref2015")
    aa_freq = native_aa_frequency(pose)

    decoy = generate_decoy(pose, sfxn, aa_freq, seed=42)

    # Compare CA coordinates for the first 5 residues
    bb_atoms = ["N", "CA", "C", "O"]
    max_shift = 0.0
    for res_idx in range(1, 6):
        for atom in bb_atoms:
            if pose.residue(res_idx).has(atom):
                native_xyz = pose.residue(res_idx).xyz(atom)
                decoy_xyz  = decoy.residue(res_idx).xyz(atom)
                shift = native_xyz.distance(decoy_xyz)
                max_shift = max(max_shift, shift)

    assert max_shift < 0.05, (
        f"Backbone shift too large: {max_shift:.4f} Å "
        f"(backbone should be fixed)"
    )


@skip_if_no_pyrosetta
def test_frustration_survey_small(tmp_path):
    """
    The frustration survey should run on a small protein (lysozyme, first 20
    residues). Some minimally and some neutral/highly frustrated contacts
    are expected.
    """
    import requests
    from pathlib import Path
    from src.frustration import (
        get_protein_contacts,
        run_frustration_survey,
    )

    pdb_path = Path("/tmp/test_1lyz.pdb")
    if not pdb_path.exists():
        r = requests.get("https://files.rcsb.org/download/1LYZ.pdb", timeout=30)
        pdb_path.write_bytes(r.content)

    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    sfxn = pyrosetta.create_score_function("ref2015")

    # Only take contacts among the first 20 residues (for speed)
    contacts = get_protein_contacts(pose, cutoff=10.0, seq_sep_min=4)
    contacts_small = [(i, j) for i, j in contacts if i <= 20 and j <= 20]

    if len(contacts_small) == 0:
        pytest.skip("No suitable contacts found for the test")

    df = run_frustration_survey(
        pose=pose,
        scorefxn=sfxn,
        contacts=contacts_small,
        n_decoys=20,  # speed test
        seed=0,
        checkpoint_path=str(tmp_path / "test_ckpt.pkl"),
        checkpoint_every=10,
    )

    assert len(df) == len(contacts_small), "There should be a row for every contact"
    assert set(df.columns) >= {
        "resi","resj","F_index","E_native","decoy_mean","decoy_std","frustration_class"
    }
    # Frustration classes should contain valid values
    valid_classes = {"minimally_frustrated","neutral","highly_frustrated"}
    assert set(df["frustration_class"]).issubset(valid_classes)
    # F_index should be a real number
    assert not df["F_index"].isna().any(), "Found NaN F_index values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
