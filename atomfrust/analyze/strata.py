"""σ across architectural strata, and redundancy between decoy axes.

This module answers the two *decisive* experiments of WP2 (methods doc §3.5, D4 and D5).

**Why σ, and not the index.** The frustration index is a Z-score, ``(mean_decoy −
native)/σ``. Centring removes the first-order offset between two targets, so a Z-score
looks target-agnostic by construction — but it does **not** remove a systematic difference
in σ itself. If deeply buried pockets carry a decoy spread twice that of shallow ones, the
same physical stabilisation reports as half the Z-score, and cross-target pooling is
invalid however well-centred the scores are. D4 therefore measures σ directly, stratified
by pocket architecture (burial, polarity, volume), and S2.5 names the *coefficient of
variation* of σ as the quantitative statement of target-invariance: CV ≈ 0 means the scale
is shared, large CV means the index is better-calibrated but target-dependent (and WP5
re-scopes to within-target rescoring — a stated, publishable outcome, not a failure).

**Why axis redundancy.** Four decoy axes are implemented (A identity/location, B pose,
C site, D chemotype). Axis D is the novel contribution, and it only *is* a contribution if
it measures something the published axis does not. S2.6 fixes the bar: chemotype must
correlate **below 0.8** with identity/location on the shared per-contact ordering, or it
adds no information. :data:`REDUNDANCY_THRESHOLD` is that number, in one place.

**Everything here is a query, never a re-run.** Per-pocket descriptors are written at
generation time (``native/pocket.json``) and per-contact indices are stored per axis keyed
on ``pair_id``; this module consumes those tables and computes. No PyRosetta, no I/O, no
plotting, no RNG — so a stratum assignment is reproducible from the stored tables alone,
which is what the D5 acceptance criterion ("deterministic and recorded per pocket")
requires.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy.stats import rankdata

__all__ = [
    "STRATUM_AXES",
    "AXIS_SOURCE_COLUMN",
    "REDUNDANCY_THRESHOLD",
    "POLAR_RESIDUES",
    "RESIDUE_VOLUME_A3",
    "SIDE_CHAIN_VOLUME_A3",
    "pocket_descriptors",
    "assign_strata",
    "sigma_by_stratum",
    "axis_redundancy",
]

#: The three architectural axes of methods-doc D4. Order is fixed: it is the row order of
#: :func:`sigma_by_stratum`, and stable output ordering is part of being deterministic.
STRATUM_AXES = ("burial", "polarity", "volume")

#: Which descriptor column each axis is binned on.
AXIS_SOURCE_COLUMN = {
    "burial": "mean_burial",
    "polarity": "frac_polar",
    "volume": "mean_volume",
}

#: S2.6: above this, the chemotype axis is redundant with identity/location.
REDUNDANCY_THRESHOLD = 0.8

#: Polar and charged side chains; the complement of this set is hydrophobic. CYS is counted
#: polar (its thiol hydrogen-bonds and ionises), consistent with the usual Kyte–Doolittle-
#: adjacent polar/charged partition used for pocket composition.
POLAR_RESIDUES = frozenset(
    {"ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "LYS", "SER", "THR", "TYR", "TRP"}
)

#: Mean residue volumes in Å³, Zamyatnin, A. A. (1972) "Protein volume in solution",
#: Prog. Biophys. Mol. Biol. 24, 107–123, Table 1 — the standard tabulation. These are
#: *whole-residue* volumes (backbone included).
RESIDUE_VOLUME_A3 = {
    "ALA": 88.6, "ARG": 173.4, "ASN": 114.1, "ASP": 111.1, "CYS": 108.5,
    "GLN": 143.8, "GLU": 138.4, "GLY": 60.1, "HIS": 153.2, "ILE": 166.7,
    "LEU": 166.7, "LYS": 168.6, "MET": 162.9, "PHE": 189.9, "PRO": 112.7,
    "SER": 89.0, "THR": 116.1, "TRP": 227.8, "TYR": 193.6, "VAL": 140.0,
}

#: Side-chain volume, taken as the residue volume less glycine's — glycine's side chain is a
#: hydrogen, so its volume is the backbone contribution shared by every residue. Subtracting
#: a constant cannot change any quantile binning; it is done so ``mean_volume`` reads as a
#: side-chain quantity rather than one dominated by a common offset.
SIDE_CHAIN_VOLUME_A3 = {
    name: round(volume - RESIDUE_VOLUME_A3["GLY"], 1)
    for name, volume in RESIDUE_VOLUME_A3.items()
}

_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

#: Node kinds that are amino-acid residues, and so carry burial/polarity/volume. Mirrors
#: :data:`atomfrust.graph.PROTEIN_KINDS`, duplicated rather than imported to keep this
#: module free of any dependency that could drag in a pose.
_RESIDUE_KINDS = frozenset({"protein", "noncanonical"})


# ----------------------------------------------------------------- residue properties


def _canonical_resname(name: object) -> str | None:
    """Three-letter code for a residue name, or ``None`` if it is not a standard one."""
    if not isinstance(name, str):
        return None
    key = name.strip().upper()
    if len(key) == 1:
        key = _ONE_TO_THREE.get(key, key)
    return key if key in RESIDUE_VOLUME_A3 else None


def _residue_degree(pairs: pd.DataFrame) -> pd.Series:
    """Incident-pair count per node id, over the pair table exactly as supplied.

    The caller decides which contact definition applies by filtering ``pairs`` on its
    ``in__<name>`` column first; imposing one here would silently override that choice.
    """
    if pairs.empty:
        return pd.Series(dtype="int64")
    endpoints = pd.concat([pairs["node_i"], pairs["node_j"]], ignore_index=True)
    return endpoints.value_counts()


def _normalise_mask(pocket_mask: np.ndarray, n_nodes: int) -> np.ndarray:
    """Accept a boolean mask over node rows, or positional indices; return a boolean mask."""
    mask = np.asarray(pocket_mask)
    if mask.dtype == bool:
        if mask.shape != (n_nodes,):
            raise ValueError(
                f"pocket_mask has {mask.shape[0]} entries, nodes has {n_nodes} rows"
            )
        return mask
    if mask.size and not np.issubdtype(mask.dtype, np.integer):
        raise TypeError(
            f"pocket_mask must be boolean or integer positional indices, got {mask.dtype}"
        )
    out = np.zeros(n_nodes, dtype=bool)
    if mask.size:
        if mask.min() < 0 or mask.max() >= n_nodes:
            raise IndexError("pocket_mask contains positions outside the node table")
        out[mask.astype(int)] = True
    return out


def pocket_descriptors(
    nodes: pd.DataFrame, pairs: pd.DataFrame, pocket_mask: np.ndarray
) -> dict:
    """Architectural descriptors of one pocket — the D4 stratification variables.

    ``nodes`` is the ``nodes.parquet`` table (``node_id, kind, resname, n_heavy,
    rel_sasa``), ``pairs`` is ``pairs.parquet`` (``node_i, node_j``), and ``pocket_mask``
    selects the pocket, either as a boolean mask over node rows or as positional indices.

    Returned keys:

    ``n_pocket_residues``
        Amino-acid nodes in the pocket. The ligand and any metal are in the mask (they are
        pocket *nodes*) but are not residues and carry no burial/polarity/volume.
    ``mean_burial``
        Mean per-residue burial on [0, 1] when ``rel_sasa`` is usable, else the mean
        residue degree — two different scales, which is exactly why ``burial_source``
        is returned alongside and must be carried into any comparison.
    ``frac_polar``, ``mean_volume``
        Composition over the pocket residues, from :data:`POLAR_RESIDUES` and
        :data:`SIDE_CHAIN_VOLUME_A3`. Non-standard residues are excluded from both rather
        than guessed at; ``n_typed_residues`` reports how many were usable.
    ``pocket_heavy_atoms``
        Heavy atoms over *all* pocket nodes, ligand included — the size covariate.
    ``mean_residue_degree``
        Mean incident-pair count over the pocket nodes.
    ``burial_source``
        ``"rel_sasa"`` or ``"degree"``.
    """
    n_nodes = len(nodes)
    mask = _normalise_mask(pocket_mask, n_nodes)
    pocket = nodes.loc[mask]

    degree = _residue_degree(pairs)
    pocket_degree = (
        degree.reindex(pocket["node_id"]).fillna(0).to_numpy(dtype=float)
        if len(pocket)
        else np.empty(0)
    )

    is_residue = (
        pocket["kind"].astype(str).isin(_RESIDUE_KINDS).to_numpy()
        if len(pocket)
        else np.empty(0, dtype=bool)
    )
    residues = pocket.loc[is_residue]

    # Burial from rel_sasa where the column exists and holds finite values for the pocket
    # residues; otherwise degree, which is monotone in burial but on an unbounded scale.
    rel_sasa = (
        pd.to_numeric(residues.get("rel_sasa"), errors="coerce").to_numpy(dtype=float)
        if "rel_sasa" in residues.columns and len(residues)
        else np.full(len(residues), np.nan)
    )
    usable = np.isfinite(rel_sasa)
    if usable.any():
        burial_source = "rel_sasa"
        mean_burial = float(np.mean(1.0 - np.clip(rel_sasa[usable], 0.0, 1.0)))
    else:
        burial_source = "degree"
        residue_degree = pocket_degree[is_residue] if len(pocket) else np.empty(0)
        mean_burial = float(np.mean(residue_degree)) if residue_degree.size else float("nan")

    canonical = [_canonical_resname(name) for name in residues.get("resname", [])]
    typed = [name for name in canonical if name is not None]
    frac_polar = (
        float(np.mean([name in POLAR_RESIDUES for name in typed])) if typed else float("nan")
    )
    mean_volume = (
        float(np.mean([SIDE_CHAIN_VOLUME_A3[name] for name in typed]))
        if typed
        else float("nan")
    )

    n_heavy = (
        pd.to_numeric(pocket.get("n_heavy"), errors="coerce").fillna(0).sum()
        if "n_heavy" in pocket.columns
        else 0
    )

    return {
        "n_pocket_nodes": int(mask.sum()),
        "n_pocket_residues": int(len(residues)),
        "n_typed_residues": int(len(typed)),
        "mean_burial": mean_burial,
        "burial_source": burial_source,
        "frac_polar": frac_polar,
        "mean_volume": mean_volume,
        "pocket_heavy_atoms": int(n_heavy),
        "mean_residue_degree": (
            float(np.mean(pocket_degree)) if pocket_degree.size else float("nan")
        ),
    }


# ------------------------------------------------------------------------- strata


def _quantile_bin(values: np.ndarray, n_bins: int) -> pd.Categorical:
    """Ordered ``Q1..Qk`` labels from empirical quantiles, collapsing k when it must.

    ``pd.qcut(..., labels=[...])`` raises when tied values force duplicate edges, which is
    the common case with few pockets — so edges are taken explicitly and ``k`` reduced to
    however many distinct ones survive. Fewer pockets than strata therefore yields fewer
    bins, never an exception.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return pd.Categorical([None] * values.size, categories=[], ordered=True)

    k = int(min(n_bins, np.unique(finite).size))
    if k <= 1:
        return pd.Categorical(
            np.where(np.isfinite(values), "Q1", None), categories=["Q1"], ordered=True
        )

    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, k + 1)))
    labels = [f"Q{n + 1}" for n in range(len(edges) - 1)]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True, ordered=True)


def assign_strata(descriptors: pd.DataFrame, n_strata: int = 3) -> pd.DataFrame:
    """Add a ``stratum_<axis>`` column per axis in :data:`STRATUM_AXES`.

    Binning is by quantile **across the supplied pockets**, so a stratum is a statement
    about this cohort and not an absolute class — which is the point: D4 asks whether σ
    moves with architecture *within the set being pooled*. Pockets whose descriptor is
    missing get a missing stratum rather than a default bin, so they cannot inflate a
    stratum's ``n``.

    Deterministic by construction: empirical quantiles of the input, no RNG, row order
    preserved.
    """
    if n_strata < 1:
        raise ValueError(f"n_strata must be >= 1, got {n_strata}")

    out = descriptors.copy()
    for axis in STRATUM_AXES:
        column = AXIS_SOURCE_COLUMN[axis]
        if column not in out.columns:
            raise KeyError(
                f"descriptors is missing {column!r}, the source column for the "
                f"{axis!r} axis; pocket_descriptors() produces it"
            )
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        out[f"stratum_{axis}"] = _quantile_bin(values, n_strata)
    return out


def sigma_by_stratum(
    descriptors: pd.DataFrame, sigma_column: str = "median_sigma"
) -> pd.DataFrame:
    """Per axis and stratum: ``n``, ``mean_sigma``, ``std_sigma``, ``cv_sigma``.

    ``cv_sigma = std_sigma / |mean_sigma|`` is the S2.5 quantity — the spread of the decoy
    scale *within* one architectural stratum, expressed as a scale-free fraction so strata
    with different absolute σ are comparable. ``cv_across_strata`` is carried alongside on
    every row of an axis: the CV of the per-stratum means, i.e. how much the decoy scale
    moves *between* strata. Target-invariance is the claim that both are small; a small
    within and a large between is precisely the failure mode the Z-score hides.

    Requires the ``stratum_<axis>`` columns from :func:`assign_strata`.
    """
    if sigma_column not in descriptors.columns:
        raise KeyError(f"descriptors has no {sigma_column!r} column")

    sigma = pd.to_numeric(descriptors[sigma_column], errors="coerce").to_numpy(dtype=float)

    rows = []
    for axis in STRATUM_AXES:
        column = f"stratum_{axis}"
        if column not in descriptors.columns:
            raise KeyError(
                f"descriptors has no {column!r} column; call assign_strata() first"
            )
        labels = descriptors[column]
        categories = (
            list(labels.cat.categories)
            if isinstance(labels.dtype, pd.CategoricalDtype)
            else sorted({v for v in labels if pd.notna(v)})
        )
        axis_rows = []
        for stratum in categories:
            selected = sigma[(labels == stratum).to_numpy(dtype=bool)]
            selected = selected[np.isfinite(selected)]
            n = int(selected.size)
            mean = float(np.mean(selected)) if n else float("nan")
            # ddof=1: sigma values are a sample of the pockets that could fall in this
            # stratum, not the population. Undefined for a single pocket, and said so.
            std = float(np.std(selected, ddof=1)) if n > 1 else float("nan")
            cv = (
                std / abs(mean)
                if n > 1 and np.isfinite(mean) and mean != 0.0
                else float("nan")
            )
            axis_rows.append(
                {
                    "axis": axis,
                    "stratum": str(stratum),
                    "n": n,
                    "mean_sigma": mean,
                    "std_sigma": std,
                    "cv_sigma": cv,
                }
            )

        means = np.array([r["mean_sigma"] for r in axis_rows], dtype=float)
        means = means[np.isfinite(means)]
        across = (
            float(np.std(means, ddof=1) / abs(np.mean(means)))
            if means.size > 1 and np.mean(means) != 0.0
            else float("nan")
        )
        for row in axis_rows:
            row["cv_across_strata"] = across
        rows.extend(axis_rows)

    return pd.DataFrame(
        rows,
        columns=[
            "axis", "stratum", "n", "mean_sigma", "std_sigma", "cv_sigma",
            "cv_across_strata",
        ],
    )


# --------------------------------------------------------------- axis redundancy


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with no exception on a constant input — that is a real answer of NaN."""
    if x.size < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = float(np.sqrt((xc @ xc) * (yc @ yc)))
    if denominator == 0.0:
        return float("nan")
    return float((xc @ yc) / denominator)


def axis_redundancy(indices: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Pearson and Spearman correlation between decoy axes' per-contact indices.

    Every array is one axis's index over the **same** ``pair_id`` ordering — the arrays are
    positional, so a caller that reindexes one axis and not another silently compares
    different contacts. Equal length is enforced for that reason; alignment to ``pair_id``
    is the caller's job and must happen before this call.

    One row per unordered axis pair, including the diagonal: an axis against itself must
    read 1.0, and its presence is the cheapest available check that the shared ordering
    actually holds. ``exceeds_threshold`` is the S2.6 verdict at
    :data:`REDUNDANCY_THRESHOLD`, left as ``pd.NA`` on the diagonal where it is meaningless.

    Correlations are computed on pairwise-complete rows; a constant or too-short vector
    yields NaN rather than an exception.
    """
    if not indices:
        return pd.DataFrame(
            columns=[
                "axis_a", "axis_b", "n", "pearson_r", "spearman_rho",
                "max_abs_correlation", "is_self", "exceeds_threshold",
            ]
        )

    arrays = {name: np.asarray(values, dtype=float).ravel() for name, values in indices.items()}
    lengths = {name: values.size for name, values in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            "axes must share one pair ordering, so all index arrays must have equal "
            f"length; got {lengths}"
        )

    rows = []
    for name_a, name_b in combinations_with_replacement(arrays, 2):
        a, b = arrays[name_a], arrays[name_b]
        keep = np.isfinite(a) & np.isfinite(b)
        a, b = a[keep], b[keep]
        pearson = _pearson(a, b)
        spearman = (
            _pearson(rankdata(a), rankdata(b)) if a.size >= 2 else float("nan")
        )
        candidates = [abs(v) for v in (pearson, spearman) if np.isfinite(v)]
        max_abs = max(candidates) if candidates else float("nan")
        is_self = name_a == name_b
        rows.append(
            {
                "axis_a": name_a,
                "axis_b": name_b,
                "n": int(a.size),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "max_abs_correlation": max_abs,
                "is_self": is_self,
                "exceeds_threshold": (
                    pd.NA
                    if is_self or not np.isfinite(max_abs)
                    else bool(max_abs >= REDUNDANCY_THRESHOLD)
                ),
            }
        )

    out = pd.DataFrame(rows)
    out["exceeds_threshold"] = out["exceeds_threshold"].astype("boolean")
    return out
