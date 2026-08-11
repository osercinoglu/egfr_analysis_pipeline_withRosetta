"""Per-system descriptors, with the size covariates that explain them attached.

**No descriptor is privileged, and no count travels without its size.** That is the whole
design, and it comes from a measurement rather than a preference.

The prototype reported one headline number, ``n_minimally_frustrated``, and plotted it
against affinity (``run_pipeline.py:415-416``). ``CLAUDE.md`` records what that number
actually tracks: ``n_contacts_total`` — the size of the ligand-pocket contact set — ranges
476–733 across structures and is itself the strongest single predictor of affinity. At
n = 19 the quantities that correlate significantly with ``log10(affinity_pM)`` are
``n_neutral`` (Spearman ρ = −0.541, p = 0.017) and ``n_contacts_total`` (ρ = −0.521,
p = 0.022); ``n_minimally_frustrated`` does not (ρ = −0.347, p = 0.145). A *neutral* count
winning is the tell: neutral contacts are by definition the ones that are frustrated in
neither direction, so their count carries no frustration information at all and can only be
counting contacts.

Step A1 supplied the mechanism. Under the published many-body formula (``chen_literal``)
the per-contact index is algebraically a sum of two per-residue terms — R² = 1.000000 for
``E_native ~ a_i + a_j`` on all 38 structures checked — so summing a per-contact class over
a pocket is summing a per-residue quantity weighted by graph degree. A count of contacts is
a degree statistic before it is a frustration statistic.

Two consequences are enforced here rather than left to the caller's discipline:

1. :func:`summarize` emits **every** name in :data:`COVARIATES` on **every** row,
   unconditionally, even when a covariate is unavailable and comes out NaN. There is no
   argument that turns them off, so a caller cannot obtain a count without the size that
   would explain it.
2. :data:`DESCRIPTORS` is a flat registry evaluated in full. The fractions, the means and
   the per-residue-normalised count sit beside the raw counts with equal standing, and the
   column names carry the index and shell they were computed under
   (``desc__{descriptor}__{index}__{shell_label}``) so two settings cannot be silently
   compared as if they were one.

**Pocket definition is a parameter, not a constant.** ``summarize_ligand_frustration``
(``frustration.py:727-762``) hardcoded one rule: protein residues with a Cα within 10 Å of
any ligand heavy atom, then every contact pair with at least one partner in that set.
:func:`pocket_mask` makes the rule selectable and the radius an argument (R24/U5). Every
selector is column arithmetic over ``d_ca`` / ``d_heavy_min`` as *already stored on the
pairs table* — no coordinates are read and no distance is recomputed, which is what lets a
shell be re-chosen against a finished run with no PyRosetta call (U7).

Note what changes when the ligand becomes a graph node: ``incident_to_ligand`` selects
protein–ligand pairs, a handful per system, whereas the prototype's pocket was the
protein–protein pairs *near* the ligand — hundreds. They are different objects and their
counts are not comparable. :func:`shell_nodes` plus ``incident_to`` reproduces the
prototype's object; ``incident_to_ligand`` is the direct one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from atomfrust.analyze.classify import (
    HIGHLY_FRUSTRATED,
    MINIMALLY_FRUSTRATED,
    NEUTRAL,
    class_counts,
    classify_index,
)
from atomfrust.graph import PROTEIN_KINDS
from atomfrust.settings import ClassifySettings

__all__ = [
    "DESCRIPTORS",
    "COVARIATES",
    "LIGAND_KINDS",
    "DescriptorInput",
    "pocket_mask",
    "shell_nodes",
    "covariates",
    "summarize",
    "summarize_many",
]

#: Node kinds treated as "the ligand" by ``incident_to_ligand`` and by the
#: ``ligand_heavy_atoms`` covariate. Cofactors and metals are included because a system
#: whose binder is a metal-bridged fragment is still a binder; nucleic acid is not.
LIGAND_KINDS = frozenset({"ligand", "cofactor", "metal"})

#: Distance columns a shell may be measured on. Keys are the ``reference`` argument of
#: :func:`pocket_mask` and :func:`shell_nodes`; values are columns the graph already carries.
_DISTANCE_COLUMN = {"ca": "d_ca", "cb": "d_cb", "heavy_min": "d_heavy_min"}

#: Columns :func:`summarize` will accept as the native contact energy, in preference order.
#: ``E_native`` is the analysis table's name (plan §4), ``e_direct`` the stored-energy one.
_ENERGY_COLUMNS = ("E_native", "e_direct")

#: Theoretical maximum solvent-accessible area per residue, Å², from Tien et al.,
#: *PLoS ONE* 8(11):e80635 (2013), Table 1 ("theoretical" column). Used only to turn the
#: graph's ``rel_sasa`` fraction back into an area for the ``pocket_sasa_A2`` covariate.
#: A reference table is a choice, so it is named and cited rather than inlined silently;
#: pass an absolute ``sasa_A2`` column on the nodes table to bypass it entirely.
_MAX_ASA_A2 = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


# ------------------------------------------------------------------ pocket selection


def _column(pairs: pd.DataFrame, name: str) -> np.ndarray:
    try:
        return pairs[name].to_numpy()
    except KeyError:
        raise KeyError(
            f"pairs table has no {name!r} column; got {list(pairs.columns)}"
        ) from None


def _reference_distance(pairs: pd.DataFrame, reference: str) -> np.ndarray:
    try:
        column = _DISTANCE_COLUMN[reference]
    except KeyError:
        raise ValueError(
            f"unknown shell reference {reference!r}; available: {sorted(_DISTANCE_COLUMN)}"
        ) from None
    return _column(pairs, column).astype(np.float64)


def pocket_mask(
    pairs: pd.DataFrame,
    nodes: pd.DataFrame,
    selector: str = "incident_to_ligand",
    **kw,
) -> np.ndarray:
    """Which pairs a descriptor is evaluated over. Boolean, ``(len(pairs),)``.

    Selectors:

    ``all``
        Every pair. The unfiltered baseline — the number a pocket count must be read
        against, since a pocket count is a subset of it.
    ``incident_to_ligand``
        Pairs with at least one endpoint of a kind in :data:`LIGAND_KINDS`. Optional
        ``kinds=`` overrides that set. This is the *direct* protein–ligand interface, not
        the prototype's protein–protein shell; see the module docstring.
    ``incident_to``
        ``node_ids=`` an iterable of ``node_id`` values; selects pairs touching one of them.
        ``require_both=True`` narrows to pairs with *both* endpoints in the set —
        ``summarize_ligand_frustration`` used the "either" rule, so ``False`` is the default.
    ``inter_chain``
        Pairs whose endpoints are in different chains (``same_chain == False``).
    ``within_shell``
        ``shell_A=`` radius and ``reference=`` one of ``ca``/``cb``/``heavy_min``; selects
        pairs whose stored distance is within the radius. Non-finite distances (a ligand
        has no Cα) are excluded rather than treated as infinitely far by accident of NaN
        comparison — same outcome here, but stated.

    NaN-safe throughout, and it never touches coordinates: the radius filters a column the
    graph builder already wrote, so re-choosing a shell costs a comparison over a finished
    run (U7).
    """
    n = len(pairs)

    if selector == "all":
        _reject_unused(kw, selector)
        return np.ones(n, dtype=bool)

    if selector == "incident_to_ligand":
        kinds = frozenset(kw.pop("kinds", LIGAND_KINDS))
        _reject_unused(kw, selector)
        kind_i = _column(pairs, "kind_i").astype(str)
        kind_j = _column(pairs, "kind_j").astype(str)
        return np.isin(kind_i, list(kinds)) | np.isin(kind_j, list(kinds))

    if selector == "incident_to":
        if "node_ids" not in kw:
            raise ValueError("selector 'incident_to' requires node_ids=")
        wanted = set(map(str, kw.pop("node_ids")))
        require_both = bool(kw.pop("require_both", False))
        _reject_unused(kw, selector)
        in_i = np.isin(_column(pairs, "node_i").astype(str), list(wanted))
        in_j = np.isin(_column(pairs, "node_j").astype(str), list(wanted))
        return (in_i & in_j) if require_both else (in_i | in_j)

    if selector == "inter_chain":
        _reject_unused(kw, selector)
        return ~_column(pairs, "same_chain").astype(bool)

    if selector == "within_shell":
        if "shell_A" not in kw:
            raise ValueError("selector 'within_shell' requires shell_A=")
        shell_A = float(kw.pop("shell_A"))
        reference = str(kw.pop("reference", "heavy_min"))
        _reject_unused(kw, selector)
        d = _reference_distance(pairs, reference)
        return np.isfinite(d) & (d <= shell_A)

    raise ValueError(
        f"unknown selector {selector!r}; available: 'all', 'incident_to_ligand', "
        "'incident_to', 'inter_chain', 'within_shell'"
    )


def _reject_unused(kw: dict, selector: str) -> None:
    """A misspelled keyword must not silently leave the shell at its default."""
    if kw:
        raise TypeError(f"selector {selector!r} got unexpected keyword(s): {sorted(kw)}")


def shell_nodes(
    pairs: pd.DataFrame,
    nodes: pd.DataFrame,
    shell_A: float = 6.0,
    reference: str = "heavy_min",
    kinds: Iterable[str] = LIGAND_KINDS,
) -> list[str]:
    """Node ids within ``shell_A`` of any node of the given kinds, by a stored distance.

    The two-step pocket the prototype built in one: feeding the result to
    ``pocket_mask(..., 'incident_to', node_ids=...)`` gives the same *shape* of object as
    ``summarize_ligand_frustration`` (``frustration.py:727-762``) — the protein–protein
    contacts lining the site, not the protein–ligand pairs themselves.

    **It is not the prototype's rule, and cannot be.** ``get_ligand_contacts``
    (``frustration.py:111-158``) measures ligand heavy atom to protein Cα; the graph stores
    ``d_ca`` (Cα–Cα, therefore NaN for any pair touching a ligand, which has no Cα) and
    ``d_heavy_min`` (heavy-to-heavy minimum). No stored column is the mixed distance, so
    ``heavy_min`` is the available shell and selects a tighter, differently-shaped set at
    the same radius. Recomputing the mixed distance would need coordinates, which is
    exactly what analysis-time re-specification is meant to avoid.

    Only pairs already in the superset can be seen, so a shell wider than the superset's own
    cutoff silently truncates; the superset is deliberately permissive for that reason.
    """
    kinds = frozenset(map(str, kinds))
    d = _reference_distance(pairs, reference)
    within = np.isfinite(d) & (d <= float(shell_A))
    kind_i = _column(pairs, "kind_i").astype(str)
    kind_j = _column(pairs, "kind_j").astype(str)
    node_i = _column(pairs, "node_i").astype(str)
    node_j = _column(pairs, "node_j").astype(str)

    hit: set[str] = set()
    # A ligand endpoint contributes its *partner*, never itself: the pocket is the set of
    # residues lining the site, and including the ligand would make every ligand pair a
    # pocket pair regardless of the radius.
    hit.update(node_j[within & np.isin(kind_i, list(kinds)) & ~np.isin(kind_j, list(kinds))])
    hit.update(node_i[within & np.isin(kind_j, list(kinds)) & ~np.isin(kind_i, list(kinds))])
    return sorted(hit)


# ----------------------------------------------------------------------- descriptors


@dataclass(frozen=True)
class DescriptorInput:
    """What every descriptor sees: the selected contacts and nothing else.

    ``F`` and ``labels`` are already restricted to the mask, so a descriptor cannot
    accidentally read outside its pocket. ``energy`` is the native contact energy where the
    pairs table carried one and ``None`` otherwise. ``n_pocket_residues`` is passed in
    rather than derived because the normalising descriptor and the covariate must be the
    same number.
    """

    F: np.ndarray
    labels: np.ndarray
    n_pocket_residues: int
    energy: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.F.size)

    @property
    def counts(self) -> dict[str, int]:
        return class_counts(self.labels)


def _fraction(numerator: int, denominator: int) -> float:
    """NaN, not 0.0, on an empty pocket: "no contacts" is not "no frustration"."""
    return float(numerator) / denominator if denominator else math.nan


def count_minimal(d: DescriptorInput) -> float:
    """Contacts classified minimally frustrated. The prototype's headline — and the one
    ``CLAUDE.md`` shows is confounded by pocket size. Kept, not privileged."""
    return float(d.counts[MINIMALLY_FRUSTRATED])


def count_highly(d: DescriptorInput) -> float:
    return float(d.counts[HIGHLY_FRUSTRATED])


def count_neutral(d: DescriptorInput) -> float:
    """The count that "wins" the prototype's affinity correlation, which is the clearest
    evidence the counts track pocket size — neutral contacts carry no frustration signal."""
    return float(d.counts[NEUTRAL])


def frac_minimal(d: DescriptorInput) -> float:
    """``count_minimal / n_contacts_selected`` — the count with pocket size divided out.

    On the 11 WT non-covalent structures this reverses the sign of the published claim
    (ρ = +0.745, p = 0.008: a *less* frustrated pocket binds *more weakly*), which is why
    it is reported next to the count rather than instead of it.
    """
    return _fraction(d.counts[MINIMALLY_FRUSTRATED], d.n)


def frac_highly(d: DescriptorInput) -> float:
    return _fraction(d.counts[HIGHLY_FRUSTRATED], d.n)


def frac_neutral(d: DescriptorInput) -> float:
    return _fraction(d.counts[NEUTRAL], d.n)


def energy_weighted_sum(d: DescriptorInput) -> float:
    """``Σ F·|E_native|`` over selected contacts; NaN when no energy column was supplied.

    Counting classes gives a buried contact and a glancing one equal weight. This weights
    each contact by the magnitude of the energy it actually carries, so it answers "how
    much energy sits in favourable contacts" rather than "how many contacts are favourable".
    The absolute value is deliberate: Rosetta energies are negative-is-favourable, and the
    sign of the frustration statement already lives in ``F``.

    NaN rather than a silent fallback to ``Σ F``: those are different quantities and a
    caller must not get one believing it asked for the other.
    """
    if d.energy is None:
        return math.nan
    if d.n == 0:
        return 0.0
    return float(np.sum(d.F * np.abs(d.energy)))


def mean_Z(d: DescriptorInput) -> float:
    """Mean index over selected contacts — the continuous descriptor, no threshold applied.

    Immune to the classification cutoffs, which ``classify.py`` documents are quantiles of
    one particular null rather than constants, so it is the descriptor to check when a
    count moves after a definition change.
    """
    return float(np.mean(d.F)) if d.n else math.nan


def median_Z(d: DescriptorInput) -> float:
    return float(np.median(d.F)) if d.n else math.nan


def mean_Z_top_decile(d: DescriptorInput) -> float:
    """Mean of the top 10% of ``F``, i.e. of the most minimally frustrated contacts.

    A threshold-free stand-in for ``count_minimal``: it asks how strongly the favourable
    tail is favoured instead of how many contacts clear 0.78. The decile is ``ceil(0.1·n)``
    contacts, at least one, so it is defined for any non-empty pocket.
    """
    if d.n == 0:
        return math.nan
    k = max(1, math.ceil(0.1 * d.n))
    return float(np.mean(np.sort(d.F)[-k:]))


def count_minimal_per_pocket_residue(d: DescriptorInput) -> float:
    """``count_minimal / n_pocket_residues`` — the count normalised by pocket *residues*.

    A second way to divide out size, and not equivalent to :func:`frac_minimal`: this one
    divides by nodes, that one by contacts, and the ratio between them is the mean residue
    degree, which is exactly the quantity A1 identified as what the counts track. Reporting
    both makes the degree dependence visible instead of implicit.
    """
    return _fraction(d.counts[MINIMALLY_FRUSTRATED], d.n_pocket_residues)


def net_frustration(d: DescriptorInput) -> float:
    """``count_minimal − count_highly``. Also a count, and also size-dependent."""
    return float(d.counts[MINIMALLY_FRUSTRATED] - d.counts[HIGHLY_FRUSTRATED])


#: The registry, evaluated in full on every system. Order is stable so column order is.
DESCRIPTORS: dict[str, Callable[[DescriptorInput], float]] = {
    "count_minimal": count_minimal,
    "count_highly": count_highly,
    "count_neutral": count_neutral,
    "frac_minimal": frac_minimal,
    "frac_highly": frac_highly,
    "frac_neutral": frac_neutral,
    "energy_weighted_sum": energy_weighted_sum,
    "mean_Z": mean_Z,
    "median_Z": median_Z,
    "mean_Z_top_decile": mean_Z_top_decile,
    "count_minimal_per_pocket_residue": count_minimal_per_pocket_residue,
    "net_frustration": net_frustration,
}


# ------------------------------------------------------------------------ covariates

#: Emitted on every row, always, by :func:`summarize`. These are not context — they are the
#: quantities the raw counts are suspected of being. See the module docstring.
COVARIATES = (
    "n_contacts_total",
    "n_pocket_residues",
    "mean_residue_degree",
    "pocket_sasa_A2",
    "ligand_heavy_atoms",
    "n_protein_residues",
    "n_resolved_residues",
)


def _selected_nodes(pairs: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Distinct node ids touched by at least one selected pair."""
    if not mask.any():
        return np.empty(0, dtype=object)
    node_i = _column(pairs, "node_i")[mask]
    node_j = _column(pairs, "node_j")[mask]
    return pd.unique(np.concatenate([node_i, node_j]).astype(str))


def _pocket_sasa(nodes: pd.DataFrame, pocket: np.ndarray) -> float:
    """Total accessible area of the pocket nodes, Å².

    Prefers an absolute ``sasa_A2`` column if the nodes table has one. Otherwise converts
    the graph's ``rel_sasa`` fraction with :data:`_MAX_ASA_A2`; residues whose one-letter
    code is not one of the 20 (a ligand, an unresolved node) contribute nothing, and if
    nothing at all is convertible the covariate is NaN rather than 0.0 — an unmeasured
    pocket is not a buried one.
    """
    if len(pocket) == 0:
        return math.nan
    sub = nodes[nodes["node_id"].astype(str).isin(set(pocket.tolist()))]
    if sub.empty:
        return math.nan
    if "sasa_A2" in sub.columns:
        values = pd.to_numeric(sub["sasa_A2"], errors="coerce").to_numpy(dtype=np.float64)
        return float(np.nansum(values)) if np.isfinite(values).any() else math.nan
    if "rel_sasa" not in sub.columns or "name1" not in sub.columns:
        return math.nan
    rel = pd.to_numeric(sub["rel_sasa"], errors="coerce").to_numpy(dtype=np.float64)
    max_asa = sub["name1"].astype(str).map(_MAX_ASA_A2).to_numpy(dtype=np.float64)
    area = rel * max_asa
    return float(np.nansum(area)) if np.isfinite(area).any() else math.nan


def covariates(pairs: pd.DataFrame, nodes: pd.DataFrame, mask: np.ndarray) -> dict:
    """The seven size quantities, all of them, for one system and one mask.

    Four describe the selection (``n_contacts_total``, ``n_pocket_residues``,
    ``mean_residue_degree``, ``pocket_sasa_A2``) and three the system as a whole
    (``ligand_heavy_atoms``, ``n_protein_residues``, ``n_resolved_residues``). A covariate
    that cannot be computed from the tables given comes back NaN and keeps its column;
    dropping it would let a count be reported alone, which is the failure this module
    exists to prevent.

    ``n_pocket_residues`` is the number of distinct nodes incident to a selected pair — a
    property of the mask, so it means the same thing under every selector, and it is the
    denominator ``count_minimal_per_pocket_residue`` uses. ``mean_residue_degree`` is
    ``2·n_contacts/n_nodes`` within the selected subgraph, the degree statistic A1 showed
    the counts are proportional to.
    """
    mask = np.asarray(mask, dtype=bool)
    pocket = _selected_nodes(pairs, mask)
    n_contacts = int(mask.sum())
    n_pocket = int(pocket.size)

    kind = nodes["kind"].astype(str).to_numpy() if "kind" in nodes.columns else np.empty(0, str)
    protein = np.isin(kind, list(PROTEIN_KINDS))
    ligand = np.isin(kind, list(LIGAND_KINDS))
    n_heavy = (
        pd.to_numeric(nodes["n_heavy"], errors="coerce").to_numpy(dtype=np.float64)
        if "n_heavy" in nodes.columns
        else np.full(len(nodes), np.nan)
    )

    return {
        "n_contacts_total": n_contacts,
        "n_pocket_residues": n_pocket,
        "mean_residue_degree": (2.0 * n_contacts / n_pocket) if n_pocket else math.nan,
        "pocket_sasa_A2": _pocket_sasa(nodes, pocket),
        "ligand_heavy_atoms": float(np.nansum(n_heavy[ligand])) if ligand.any() else 0.0,
        "n_protein_residues": int(protein.sum()),
        # A node with no heavy atoms is a position without density. Under the current
        # builder every node has atoms, so this equals n_protein_residues; it is emitted
        # separately so a structure with gaps reports them instead of hiding them.
        "n_resolved_residues": int((protein & (n_heavy > 0)).sum()),
    }


# ------------------------------------------------------------------------- summarise


def summarize(
    pairs: pd.DataFrame,
    nodes: pd.DataFrame,
    F: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    index_name: str = "zscore",
    shell_label: str = "default",
) -> dict:
    """One system, one index, one shell → descriptors plus the mandatory covariates.

    Descriptor keys are ``desc__{descriptor}__{index_name}__{shell_label}``; covariate keys
    are the bare names in :data:`COVARIATES`. The descriptor key carries its settings
    because the same descriptor under two shells is two different numbers, and a bare name
    would let them be averaged or joined as if they were one.

    ``F``, ``labels`` and ``mask`` are all parallel to ``pairs`` and are checked to be.
    ``labels`` is passed in rather than derived here so that thresholds stay a decision made
    once, by :func:`~atomfrust.analyze.classify.classify_index`, at a level where they can
    be recorded.

    An empty mask is legal and returns zeros for the counts and NaN for every fraction and
    mean — a pocket with no contacts is unmeasured, not perfectly neutral.
    """
    n = len(pairs)
    F = np.asarray(F, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    mask = np.asarray(mask, dtype=bool)
    for name, array in (("F", F), ("labels", labels), ("mask", mask)):
        if array.shape != (n,):
            raise ValueError(
                f"{name} has shape {array.shape}, expected ({n},) to match the pairs table"
            )

    energy = None
    for column in _ENERGY_COLUMNS:
        if column in pairs.columns:
            energy = pd.to_numeric(pairs[column], errors="coerce").to_numpy(np.float64)[mask]
            break

    covariate_values = covariates(pairs, nodes, mask)
    data = DescriptorInput(
        F=F[mask],
        labels=labels[mask],
        n_pocket_residues=covariate_values["n_pocket_residues"],
        energy=energy,
    )

    row = {
        f"desc__{name}__{index_name}__{shell_label}": function(data)
        for name, function in DESCRIPTORS.items()
    }
    row.update(covariate_values)
    return row


def summarize_many(
    systems: Mapping[str, Mapping] | Iterable[tuple[str, Mapping]],
    selector: str = "incident_to_ligand",
    index_name: str = "zscore",
    shell_label: str = "default",
    thresholds: ClassifySettings | None = None,
    **selector_kw,
) -> pd.DataFrame:
    """One row per system, same columns for all of them.

    ``systems`` maps a system id to a mapping with ``pairs``, ``nodes`` and ``F``, plus
    optional ``labels`` (classified here from ``F`` if absent) and ``mask`` (built here with
    ``selector`` and ``selector_kw`` if absent). A per-system ``mask`` is how a run with a
    different pocket rule per system is expressed; the shared ``selector`` is the normal case.

    Every row is produced by the same call, so the resulting frame is directly
    correlatable — and because :func:`summarize` cannot omit a covariate, any correlation
    computed from it can be checked against pocket size without going back to the run.
    """
    items = systems.items() if isinstance(systems, Mapping) else systems
    rows = []
    for system_id, payload in items:
        pairs = payload["pairs"]
        nodes = payload["nodes"]
        F = np.asarray(payload["F"], dtype=np.float64)
        labels = payload.get("labels")
        if labels is None:
            labels = classify_index(F, thresholds)
        mask = payload.get("mask")
        if mask is None:
            mask = pocket_mask(pairs, nodes, selector, **selector_kw)
        row = {"system_id": str(system_id)}
        row.update(
            summarize(pairs, nodes, F, labels, mask, index_name, shell_label)
        )
        rows.append(row)

    if not rows:
        columns = ["system_id"] + [
            f"desc__{name}__{index_name}__{shell_label}" for name in DESCRIPTORS
        ] + list(COVARIATES)
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)
