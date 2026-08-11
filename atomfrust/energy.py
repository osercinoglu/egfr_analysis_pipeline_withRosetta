"""Pair energies and the many-body registry.

Two responsibilities, deliberately separated:

**Direct pair energies** come from the REF2015 ``EnergyGraph`` and are the only energetic
quantity ever stored. ``e_fa_rep`` is stored *alongside* ``e_direct`` rather than subtracted
from it, so ``exclude_fa_rep`` becomes an analysis-time switch. The prototype subtracted at
scoring time (``frustration.py:210-212``), baking the choice into every stored number.

**Many-body energies** are derived from stored direct energies by pure array arithmetic —
no pose, no Rosetta. This is what makes the formula re-specifiable against a finished run,
and it is why ``E_ij`` is never written to disk.

The three modes, with ``B_i`` the sum of ``e_ik`` over *all* of i's partners:

===============  ==========================================  ==================
mode             formula                                     degenerate?
===============  ==========================================  ==================
``chen_literal`` ``e_ij + ½Σ_{k≠j} e_ik + ½Σ_{l≠i} e_jl``     **yes**
                 ``= 0.5·(B_i + B_j)``
``pair_retained`` ``e_ij + ½·B_i + ½·B_j``                    no
``pair_only``    ``e_ij``                                     no
===============  ==========================================  ==================

``chen_literal`` is Chen et al. 2020 Eq. 2 exactly as printed — step A4 verified the ``k≠j``
and ``l≠i`` exclusions are in the paper. Substituting ``Σ_{k≠j} e_ik = B_i − e_ij`` shows
``e_ij`` cancels, leaving the mean of two per-residue totals. Step A1 confirmed this
numerically: fitting ``E_native ~ a_i + a_j`` gives R² = 1.000000 on all 38 structures
checked. It carries no pair-specific information, and it is the published object, so it is
required for any reproduction claim.

``pair_retained`` takes the same sums over *all* partners, so the pair term survives with
total weight 2. Note ``pair_retained − chen_literal == e_ij`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from pyrosetta import Pose

__all__ = [
    "ManyBodyMode",
    "raw_interaction_energy",
    "PairEnergy",
    "EnergyEvaluator",
    "effective_energy",
    "many_body_energies",
    "many_body_energies_reference",
]

ManyBodyMode = Literal["chen_literal", "pair_retained", "pair_only"]


@dataclass(frozen=True)
class PairEnergy:
    """One pair's direct energy. ``e_direct`` includes ``fa_rep``; ``e_fa_rep`` is its share."""

    e_direct: float
    e_fa_rep: float
    has_edge: bool


class EnergyEvaluator:
    """Direct pair energies from a scored pose.

    The pose is scored **once** on construction and the ``EnergyGraph`` held. The prototype
    called ``scorefxn(pose)`` inside every ``pairwise_energy`` invocation
    (``frustration.py:167``), i.e. once per pair per background term — tens of thousands of
    rescore calls per decoy where one suffices.

    The evaluator is a read-only view of one pose state. Mutating the pose after
    construction invalidates it; build a new one.
    """

    def __init__(self, pose: Any, score_function: str = "ref2015") -> None:
        import pyrosetta
        from pyrosetta.rosetta.core.scoring import fa_rep

        self.pose = pose
        self.scorefxn = pyrosetta.create_score_function(score_function)
        self.scorefxn(pose)
        self._graph = pose.energies().energy_graph()
        self._weights = self.scorefxn.weights()
        self._fa_rep = fa_rep
        self._fa_rep_weight = self.scorefxn.get_weight(fa_rep)

    def pair(self, i: int, j: int) -> PairEnergy:
        """Direct energy between pose residues ``i`` and ``j``.

        A pair beyond Rosetta's interaction cutoff has no edge and yields exactly 0.0. That
        is a real absence of interaction, not a missing value, but it is worth counting:
        under a permissive Cα cutoff a substantial fraction of pairs can be edgeless, and
        they contribute only through the many-body background terms.
        """
        edge = self._graph.find_energy_edge(i, j)
        if edge is None:
            return PairEnergy(0.0, 0.0, False)
        total = edge.dot(self._weights)
        fa_rep_component = edge[self._fa_rep] * self._fa_rep_weight
        return PairEnergy(float(total), float(fa_rep_component), True)

    def pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Energies for a pair table. Returns ``pair_id, e_direct, e_fa_rep, has_edge``."""
        if pairs.empty:
            return pd.DataFrame(
                {
                    "pair_id": pd.Series(dtype="int32"),
                    "e_direct": pd.Series(dtype="float32"),
                    "e_fa_rep": pd.Series(dtype="float32"),
                    "has_edge": pd.Series(dtype=bool),
                }
            )
        records = [self.pair(int(i), int(j)) for i, j in zip(pairs["i"], pairs["j"])]
        return pd.DataFrame(
            {
                "pair_id": pairs["pair_id"].to_numpy(dtype=np.int32),
                "e_direct": np.fromiter(
                    (r.e_direct for r in records), dtype=np.float32, count=len(records)
                ),
                "e_fa_rep": np.fromiter(
                    (r.e_fa_rep for r in records), dtype=np.float32, count=len(records)
                ),
                "has_edge": np.fromiter(
                    (r.has_edge for r in records), dtype=bool, count=len(records)
                ),
            }
        )


def effective_energy(
    e_direct: np.ndarray, e_fa_rep: np.ndarray, exclude_fa_rep: bool = True
) -> np.ndarray:
    """Apply the ``fa_rep`` switch to stored energies.

    Chen et al. remove the harsh repulsive term because *"the repulsive Lennard-Jones
    interactions … give rise to very strong fluctuating clashes"* that inflate decoy
    variance. Because both components are stored, this is reversible.
    """
    e = np.asarray(e_direct, dtype=np.float64)
    if not exclude_fa_rep:
        return e
    return e - np.asarray(e_fa_rep, dtype=np.float64)


def _node_codes(node_i: np.ndarray, node_j: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Map node labels of any dtype to dense integer codes."""
    codes, uniques = pd.factorize(
        pd.concat([pd.Series(node_i), pd.Series(node_j)], ignore_index=True)
    )
    half = len(node_i)
    return codes[:half], codes[half:], len(uniques)


def many_body_energies(
    node_i: np.ndarray,
    node_j: np.ndarray,
    e: np.ndarray,
    mode: ManyBodyMode = "chen_literal",
) -> np.ndarray:
    """``E_ij`` for every pair, from direct pair energies alone.

    ``B_i`` is accumulated once per node with two ``bincount`` passes, so this is O(P) in the
    number of pairs rather than O(P·degree). The exclusions in the published formula need no
    special handling: they cancel algebraically, which is precisely the finding of step A1.
    """
    e = np.asarray(e, dtype=np.float64)
    if mode == "pair_only":
        return e.copy()

    ci, cj, n_nodes = _node_codes(np.asarray(node_i), np.asarray(node_j))
    background = np.bincount(ci, weights=e, minlength=n_nodes) + np.bincount(
        cj, weights=e, minlength=n_nodes
    )
    half_sum = 0.5 * (background[ci] + background[cj])

    if mode == "chen_literal":
        # e_ij + 0.5*(B_i - e_ij) + 0.5*(B_j - e_ij) == 0.5*(B_i + B_j)
        return half_sum
    if mode == "pair_retained":
        # Sums run over ALL partners, so the pair term is not cancelled.
        return e + half_sum
    raise ValueError(f"unknown many-body mode {mode!r}")


def many_body_energies_reference(
    node_i: np.ndarray,
    node_j: np.ndarray,
    e: np.ndarray,
    mode: ManyBodyMode = "chen_literal",
) -> np.ndarray:
    """Literal, un-simplified transcription of the formulas. O(P·degree); for tests only.

    :func:`many_body_energies` applies the algebraic cancellation. This one performs the
    exclusion explicitly, so agreement between the two proves the simplification is sound
    rather than assumed.
    """
    e = np.asarray(e, dtype=np.float64)
    node_i = np.asarray(node_i)
    node_j = np.asarray(node_j)

    partners: dict[Any, list[tuple[Any, float]]] = {}
    for a, b, value in zip(node_i, node_j, e):
        partners.setdefault(a, []).append((b, value))
        partners.setdefault(b, []).append((a, value))

    out = np.empty(len(e), dtype=np.float64)
    for idx, (a, b, value) in enumerate(zip(node_i, node_j, e)):
        if mode == "pair_only":
            out[idx] = value
            continue
        if mode == "chen_literal":
            bg_a = sum(v for p, v in partners[a] if p != b)
            bg_b = sum(v for p, v in partners[b] if p != a)
        elif mode == "pair_retained":
            bg_a = sum(v for _, v in partners[a])
            bg_b = sum(v for _, v in partners[b])
        else:
            raise ValueError(f"unknown many-body mode {mode!r}")
        out[idx] = value + 0.5 * bg_a + 0.5 * bg_b
    return out


def raw_interaction_energy(
    pairs: pd.DataFrame,
    energies: pd.DataFrame,
    exclude_fa_rep: bool = True,
) -> dict[str, float]:
    """Raw REF2015 energies of one complex, unnormalised.

    Requirement R07 / methods-doc S5.2, and the cheapest control in the whole project.

    The central claim is that an internally-referenced Z-score ranks better than a raw
    score. Comparing a Rosetta-derived Z-score against a Vina raw score cannot test that:
    it confounds *normalisation* with *energy-function quality*. Only raw-versus-Z-scored
    **within the same energy function** isolates the effect, so the raw number must be
    recorded next to every Z-score, at generation time, for free.

    Returned keys:
      ``interaction``     sum over pairs with exactly one non-protein endpoint — the
                          protein-ligand interaction energy.
      ``intra_protein``   sum over protein-protein pairs.
      ``intra_component`` sum over pairs where both endpoints are non-protein.
      ``total``           sum over every pair in the graph.
    Each is also reported with ``_fa_rep`` retained, since ``exclude_fa_rep`` is a post-hoc
    switch and a raw score that silently dropped the repulsive term would not be raw.
    """
    merged = pairs[["pair_id", "kind_i", "kind_j"]].merge(energies, on="pair_id")
    protein_i = merged["kind_i"].astype(str).isin(("protein", "noncanonical")).to_numpy()
    protein_j = merged["kind_j"].astype(str).isin(("protein", "noncanonical")).to_numpy()

    masks = {
        "interaction": protein_i ^ protein_j,
        "intra_protein": protein_i & protein_j,
        "intra_component": ~protein_i & ~protein_j,
        "total": np.ones(len(merged), dtype=bool),
    }
    e_direct = merged["e_direct"].to_numpy(dtype=np.float64)
    e_fa_rep = merged["e_fa_rep"].to_numpy(dtype=np.float64)
    effective = effective_energy(e_direct, e_fa_rep, exclude_fa_rep)

    out: dict[str, float] = {}
    for name, mask in masks.items():
        out[name] = float(effective[mask].sum())
        out[f"{name}_with_fa_rep"] = float(e_direct[mask].sum())
        out[f"n_{name}"] = int(mask.sum())
    out["exclude_fa_rep"] = bool(exclude_fa_rep)
    return out
