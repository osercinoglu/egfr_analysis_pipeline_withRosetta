"""Typed nodes and the superset contact graph.

This module is **the A4 correction**. Chen et al.'s published counts are ligand–residue
contacts; the prototype's ``get_protein_contacts`` (``frustration.py:70-108``) skips
non-protein residues, so the ligand never appeared in a pair and the native reference was
provably ligand-independent (step A3: ``E_native`` bit-identical holo vs apo). Here every
component — ligand, cofactor, metal, nucleic acid — is a node.

Three design points, each of which fixes a specific prototype defect:

**Per-kind-pair edge criteria.** A ligand has no Cα, so a Cα rule cannot apply to it. A
named definition therefore uses its protein rule for protein–protein pairs and a
heavy-atom-minimum rule for any pair touching a non-protein node. Without this split,
selecting ``ca_ca`` would mark every ligand pair absent and silently restore the
protein-only behaviour — the exact defect being fixed.

**Sequence separation is per chain.** The prototype's ``for j in range(i + seq_sep_min, n)``
(``frustration.py:98``) separates by *pose index*, which equals sequence separation only
within a single chain. Across a chain boundary it excludes residues that are not sequence
neighbours at all. Note the default is now ``seq_sep_min=1`` regardless: step A4 found the
paper states no sequence-separation criterion.

**Superset once, definitions as columns.** Energies are computed over a permissive union of
``(Cα ≤ ca_cutoff) ∪ (heavy-min ≤ heavy_cutoff)``; every named definition is a boolean
column on that graph. Narrowing a cutoff at analysis time is a column filter over stored
energies, never a recomputation — which is what makes user requests 5 and 7 free.

Contact membership is evaluated on the **native pose only** and frozen across decoys. The
Z-score must be taken over a fixed index set; recomputing membership per decoy would change
the estimand from sample to sample.

No PyRosetta here — geometry in, tables out. :mod:`atomfrust.pose` builds the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from atomfrust.settings import ContactSettings, Settings, SupersetSettings

__all__ = [
    "NodeKind",
    "Node",
    "Geometry",
    "PROTEIN_KINDS",
    "nodes_to_frame",
    "build_superset",
    "add_contact_definition",
    "build_graph",
]

NodeKind = Literal["protein", "ligand", "cofactor", "metal", "nucleic", "noncanonical"]

#: Kinds that carry a backbone, and so can be addressed by a Cα/Cβ rule.
PROTEIN_KINDS = frozenset({"protein", "noncanonical"})


@dataclass(frozen=True)
class Node:
    """One residue or component. Identity is its PDB position, never its name."""

    node_id: str
    pose_resnum: int
    kind: NodeKind
    chain: str
    resseq: int
    icode: str
    resname: str
    name1: str = "X"
    ccd_id: str | None = None
    rosetta_name: str | None = None
    params_sha256: str | None = None
    mutable: bool = True
    frozen_reason: str | None = None
    n_heavy: int = 0
    rel_sasa: float = float("nan")

    @property
    def is_protein_like(self) -> bool:
        return self.kind in PROTEIN_KINDS


@dataclass(frozen=True)
class Geometry:
    """Coordinates, node-indexed.

    ``heavy_xyz`` is a flat array of every heavy atom in the system; ``heavy_node`` maps
    each row to its node index. ``ca_xyz`` / ``cb_xyz`` are per node and NaN where the atom
    is absent — glycine has no Cβ, a ligand has neither.
    """

    heavy_xyz: np.ndarray  # (n_atoms, 3) float64
    heavy_node: np.ndarray  # (n_atoms,) int64, index into the node list
    ca_xyz: np.ndarray  # (n_nodes, 3) float64, NaN where absent
    cb_xyz: np.ndarray  # (n_nodes, 3) float64, NaN where absent

    def __post_init__(self) -> None:
        if self.heavy_xyz.shape[0] != self.heavy_node.shape[0]:
            raise ValueError("heavy_xyz and heavy_node disagree on atom count")
        if self.ca_xyz.shape != self.cb_xyz.shape:
            raise ValueError("ca_xyz and cb_xyz disagree on node count")

    @property
    def n_nodes(self) -> int:
        return int(self.ca_xyz.shape[0])


# --------------------------------------------------------------------------- nodes


def nodes_to_frame(nodes: list[Node]) -> pd.DataFrame:
    """The ``nodes.parquet`` table."""
    if not nodes:
        return pd.DataFrame(
            columns=[
                "node_id", "pose_resnum", "kind", "chain", "resseq", "icode", "resname",
                "name1", "ccd_id", "rosetta_name", "params_sha256", "mutable",
                "frozen_reason", "n_heavy", "rel_sasa",
            ]
        )
    df = pd.DataFrame([n.__dict__ for n in nodes])
    df["kind"] = df["kind"].astype("category")
    for col, dtype in (
        ("pose_resnum", "int32"), ("resseq", "int32"),
        ("n_heavy", "int16"), ("rel_sasa", "float32"),
    ):
        df[col] = df[col].astype(dtype)
    return df


# ------------------------------------------------------------------------ superset


def _candidate_pairs_kdtree(
    geom: Geometry, ca_cutoff: float, heavy_cutoff: float
) -> set[tuple[int, int]]:
    """Candidate node pairs from two KD-tree queries: a Cα rule and a heavy-atom rule."""
    pairs: set[tuple[int, int]] = set()

    finite = np.flatnonzero(np.isfinite(geom.ca_xyz).all(axis=1))
    if finite.size > 1:
        tree = cKDTree(geom.ca_xyz[finite])
        for a, b in tree.query_pairs(r=ca_cutoff, output_type="ndarray"):
            i, j = int(finite[a]), int(finite[b])
            pairs.add((i, j) if i < j else (j, i))

    if geom.heavy_xyz.shape[0] > 1:
        tree = cKDTree(geom.heavy_xyz)
        for a, b in tree.query_pairs(r=heavy_cutoff, output_type="ndarray"):
            i, j = int(geom.heavy_node[a]), int(geom.heavy_node[b])
            if i == j:
                continue
            pairs.add((i, j) if i < j else (j, i))

    return pairs


def _candidate_pairs_bruteforce(
    geom: Geometry, ca_cutoff: float, heavy_cutoff: float
) -> set[tuple[int, int]]:
    """O(N^2) reference. Only for tests — the KD-tree path must match it exactly."""
    n = geom.n_nodes
    atoms_by_node = [geom.heavy_xyz[geom.heavy_node == k] for k in range(n)]
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            ca_i, ca_j = geom.ca_xyz[i], geom.ca_xyz[j]
            if np.isfinite(ca_i).all() and np.isfinite(ca_j).all():
                if float(np.linalg.norm(ca_i - ca_j)) <= ca_cutoff:
                    pairs.add((i, j))
                    continue
            if _min_heavy_distance(atoms_by_node[i], atoms_by_node[j]) <= heavy_cutoff:
                pairs.add((i, j))
    return pairs


def _min_heavy_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    diff = a[:, None, :] - b[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=-1)).min())


def _pair_distance(x: np.ndarray, y: np.ndarray) -> float:
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    return float(np.linalg.norm(x - y))


def build_superset(
    nodes: list[Node],
    geom: Geometry,
    superset: SupersetSettings | None = None,
    bonds: tuple[tuple[str, str], ...] = (),
) -> pd.DataFrame:
    """The ``pairs.parquet`` table: every pair energies will be computed over.

    ``bonds`` are node-id pairs forced into the graph regardless of distance or sequence
    separation — a covalent ligand anchor, for instance.
    """
    superset = superset or SupersetSettings()
    if geom.n_nodes != len(nodes):
        raise ValueError(
            f"geometry has {geom.n_nodes} nodes, node list has {len(nodes)}"
        )

    candidates = _candidate_pairs_kdtree(
        geom, superset.ca_cutoff_A, superset.heavy_cutoff_A
    )

    index_of = {n.node_id: k for k, n in enumerate(nodes)}
    forced: set[tuple[int, int]] = set()
    for a, b in bonds:
        if a not in index_of or b not in index_of:
            raise KeyError(f"bond references an unknown node: {a!r}-{b!r}")
        i, j = index_of[a], index_of[b]
        if i == j:
            raise ValueError(f"self-bond on {a!r}")
        forced.add((i, j) if i < j else (j, i))
    candidates |= forced

    atoms_by_node = [geom.heavy_xyz[geom.heavy_node == k] for k in range(len(nodes))]
    rows = []
    for i, j in sorted(candidates):
        ni, nj = nodes[i], nodes[j]
        same_chain = ni.chain == nj.chain
        # Sequence separation is defined only between two protein-like residues of the
        # same chain. -1 everywhere else, and never used as a filter there.
        if same_chain and ni.is_protein_like and nj.is_protein_like:
            seq_sep = abs(ni.resseq - nj.resseq)
            seq_sep_pose = abs(ni.pose_resnum - nj.pose_resnum)
        else:
            seq_sep = seq_sep_pose = -1
        is_bonded = (i, j) in forced
        # The superset filters on the *smaller* of the two bases, so neither can later
        # request a pair the superset threw away.
        if is_bonded is False and superset.seq_sep_min > 1 and seq_sep != -1:
            if min(seq_sep, seq_sep_pose) < superset.seq_sep_min:
                continue
        rows.append(
            {
                "node_i": ni.node_id,
                "node_j": nj.node_id,
                "i": ni.pose_resnum,
                "j": nj.pose_resnum,
                "kind_i": ni.kind,
                "kind_j": nj.kind,
                "same_chain": same_chain,
                "seq_sep": seq_sep,
                "seq_sep_pose": seq_sep_pose,
                "d_ca": _pair_distance(geom.ca_xyz[i], geom.ca_xyz[j]),
                "d_cb": _pair_distance(geom.cb_xyz[i], geom.cb_xyz[j]),
                "d_heavy_min": _min_heavy_distance(atoms_by_node[i], atoms_by_node[j]),
                "is_bonded": is_bonded,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "node_i", "node_j", "i", "j", "kind_i", "kind_j", "same_chain",
                "seq_sep", "seq_sep_pose", "d_ca", "d_cb", "d_heavy_min", "is_bonded",
            ]
        )
    df.insert(0, "pair_id", np.arange(len(df), dtype=np.int32))
    for col, dtype in (
        ("i", "int32"), ("j", "int32"), ("seq_sep", "int32"),
        ("seq_sep_pose", "int32"),
        ("d_ca", "float32"), ("d_cb", "float32"), ("d_heavy_min", "float32"),
    ):
        df[col] = df[col].astype(dtype)
    for col in ("kind_i", "kind_j"):
        df[col] = df[col].astype("category")
    return df


# --------------------------------------------------------------- named definitions


_DISTANCE_COLUMN = {"ca_ca": "d_ca", "cb_cb": "d_cb", "heavy_min": "d_heavy_min"}


def add_contact_definition(
    pairs: pd.DataFrame,
    name: str,
    contacts: ContactSettings | None = None,
    superset: SupersetSettings | None = None,
) -> pd.DataFrame:
    """Add an ``in__<name>`` boolean column. Pure column arithmetic over stored distances.

    The protein rule applies only to protein–protein pairs. **Any pair touching a
    non-protein node is judged by heavy-atom minimum distance against
    ``contacts.ligand_cutoff_A``**, because a ligand has no Cα or Cβ and would otherwise be
    excluded by construction.

    Raises if a requested cutoff exceeds the superset, which cannot be satisfied from stored
    data and requires regeneration.
    """
    contacts = contacts or ContactSettings()
    superset = superset or SupersetSettings()

    protein_ceiling = (
        superset.ca_cutoff_A
        if contacts.definition in ("ca_ca", "cb_cb")
        else superset.heavy_cutoff_A
    )
    if contacts.cutoff_A > protein_ceiling:
        raise ValueError(
            f"contact cutoff {contacts.cutoff_A} A exceeds the superset's "
            f"{protein_ceiling} A for definition {contacts.definition!r}; "
            "this requires regeneration, not re-analysis"
        )
    if contacts.ligand_cutoff_A > superset.heavy_cutoff_A:
        raise ValueError(
            f"ligand cutoff {contacts.ligand_cutoff_A} A exceeds the superset's "
            f"heavy cutoff {superset.heavy_cutoff_A} A; this requires regeneration"
        )

    out = pairs.copy()
    if out.empty:
        out[f"in__{name}"] = pd.Series(dtype=bool)
        return out

    kind_i = out["kind_i"].astype(str)
    kind_j = out["kind_j"].astype(str)
    both_protein = kind_i.isin(PROTEIN_KINDS) & kind_j.isin(PROTEIN_KINDS)

    protein_ok = (
        out[_DISTANCE_COLUMN[contacts.definition]].to_numpy() <= contacts.cutoff_A
    )
    # A missing Cβ (glycine) can never satisfy a cb_cb rule; NaN <= x is already False.
    ligand_ok = out["d_heavy_min"].to_numpy() <= contacts.ligand_cutoff_A

    selected = np.where(both_protein.to_numpy(), protein_ok, ligand_ok)

    if contacts.seq_sep_min > 1:
        column = "seq_sep" if contacts.seq_sep_basis == "resseq" else "seq_sep_pose"
        seq_sep = out[column].to_numpy()
        too_close = (seq_sep != -1) & (seq_sep < contacts.seq_sep_min)
        selected &= ~too_close

    # A forced bond is a contact by definition — distance and separation do not apply.
    selected |= out["is_bonded"].to_numpy()

    out[f"in__{name}"] = selected
    return out


def build_graph(
    nodes: list[Node],
    geom: Geometry,
    settings: Settings | None = None,
    bonds: tuple[tuple[str, str], ...] = (),
    definitions: dict[str, ContactSettings] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``(nodes_df, pairs_df)`` with one ``in__<name>`` column per definition."""
    settings = settings or Settings()
    pairs = build_superset(nodes, geom, settings.graph.superset, bonds)

    if definitions is None:
        definitions = {settings.contacts.definition: settings.contacts}
    for name, contacts in definitions.items():
        pairs = add_contact_definition(pairs, name, contacts, settings.graph.superset)

    return nodes_to_frame(nodes), pairs
