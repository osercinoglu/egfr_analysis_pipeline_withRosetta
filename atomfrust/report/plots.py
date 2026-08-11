"""Figures that put the size covariate in the picture instead of leaving it out.

The prototype plotted ``n_minimally_frustrated`` against ``log10(affinity_pM)`` and called
it the result (``_plot_correlation``, ``run_pipeline.py:524-582``, the raw count read off at
``:539``). ``CLAUDE.md`` records what that plot actually
shows: ``n_contacts_total`` ranges 476-733 across structures and out-predicts the headline
count, and step A1 supplies the reason — under the published many-body formula a per-contact
index is a sum of two per-residue terms, so a class count over a pocket is a graph-degree
statistic. A scatter of the raw count therefore cannot be read without knowing each point's
pocket size.

So the default figure is the size-normalised descriptor with ``n_contacts_total`` mapped to
**marker size**: the confound is visible as a property of every point rather than absent
from the axes. The raw count keeps a panel — hiding it would make the two incomparable — but
that panel is labelled "confounded" in its own title, and it is the secondary one.

Agg backend, selected at import: figures are written to files, never displayed, so a report
renders identically under pytest, over ssh and in CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atomfrust.report.collect import DEFAULT_COVARIATE, resolve_descriptor  # noqa: E402

__all__ = ["marker_sizes", "confound_figure"]

#: Marker area range in points², for the covariate scale. Small enough that a dense pocket
#: does not swallow its neighbours, large enough that the range is legible.
_SIZE_RANGE = (25.0, 320.0)


def marker_sizes(values: np.ndarray, size_range: tuple[float, float] = _SIZE_RANGE) -> np.ndarray:
    """Linear map from covariate value to marker area, min→max over the observed range.

    Linear in *area*, not radius: area is the channel the eye integrates, and mapping to
    radius would make a 1.5x pocket look 2.25x bigger. A constant covariate maps to the
    midpoint rather than dividing by zero.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    low, high = size_range
    if finite.size == 0 or finite.min() == finite.max():
        return np.full(values.shape, 0.5 * (low + high))
    scaled = (values - finite.min()) / (finite.max() - finite.min())
    return low + np.clip(scaled, 0.0, 1.0) * (high - low)


def _scatter(ax, x, y, sizes, xlabel, ylabel, title) -> None:
    ax.scatter(x, y, s=sizes, alpha=0.7, edgecolor="black", linewidth=0.5)
    if x.size >= 3 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 2)
        ax.plot(grid, slope * grid + intercept, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def confound_figure(
    table: pd.DataFrame,
    outcome: str,
    descriptor: str = "frac_minimal",
    count_descriptor: str = "count_minimal",
    covariate: str = DEFAULT_COVARIATE,
    out_path: str | Path = "confound.png",
    dpi: int = 150,
) -> Path:
    """Two panels: the size-normalised descriptor (headline) and the raw count (confounded).

    Marker area encodes ``covariate`` in **both** panels, so the right-hand panel shows the
    reader the same confound the left-hand panel divides out. The right-hand title says
    "confounded" in words — a legend entry would be easy to skip, and this is the one thing
    about that panel a reader must not miss.

    Descriptor names resolve through :func:`~atomfrust.report.collect.resolve_descriptor`,
    so the registry names work against a modern ``desc__...`` table and against the
    prototype's ``results/egfr_frustration_summary.csv`` columns unchanged. A descriptor
    that is genuinely absent is skipped rather than fatal — a run analysed with a descriptor
    subset should still get a figure — but at least one panel must be plottable.

    Returns the path written. The figure is closed before returning: a report renders many
    of these, and an open figure is both a memory leak and a way for one plot's state to
    reach the next.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = []
    for name, label in ((descriptor, "size-normalised"), (count_descriptor, "confounded")):
        try:
            columns.append((name, resolve_descriptor(table, name), label))
        except (KeyError, ValueError):
            continue
    if not columns:
        raise KeyError(
            f"neither {descriptor!r} nor {count_descriptor!r} is a column of the table"
        )
    if covariate not in table.columns:
        raise KeyError(f"covariate {covariate!r} is not a column of the table")
    if outcome not in table.columns:
        raise KeyError(f"outcome {outcome!r} is not a column of the table")

    fig, axes = plt.subplots(1, len(columns), figsize=(5.5 * len(columns), 4.6), squeeze=False)
    try:
        for ax, (name, column, label) in zip(axes[0], columns):
            data = table.loc[:, [column, outcome, covariate]].apply(
                pd.to_numeric, errors="coerce"
            )
            data = data[np.isfinite(data.to_numpy(dtype=np.float64)).all(axis=1)]
            x = data[column].to_numpy(dtype=np.float64)
            y = data[outcome].to_numpy(dtype=np.float64)
            c = data[covariate].to_numpy(dtype=np.float64)
            title = (
                f"{name} — CONFOUNDED by {covariate}"
                if label == "confounded"
                else f"{name} (pocket size divided out)"
            )
            _scatter(ax, x, y, marker_sizes(c), column, outcome, title)
        fig.suptitle(f"marker area ∝ {covariate}  (n = {len(table)})", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    finally:
        plt.close(fig)
    return out_path
