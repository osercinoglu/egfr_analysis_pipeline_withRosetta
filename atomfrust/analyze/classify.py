"""The one home for frustration-class thresholds.

In the prototype the cutoffs 0.78 and −1.0 are written out **three times**, in three
languages, with no link between them:

- ``src/frustration.py:703-708`` — the engine, deciding ``frustration_class`` per contact.
- ``src/run_pipeline.py:415-416`` — the plot guides (``axvline``), which are the *visual*
  claim about where the classes lie.
- ``src/test_frustration.py:139-144`` — a private ``_classify`` helper reimplementing the
  rule, so the test can only ever confirm that the test agrees with itself.

Meanwhile ``config.yaml:25-27`` carries a ``frustration.thresholds`` block that no code
reads. Changing it changes nothing; changing the engine leaves the plot guides and the
test asserting the old rule. That is the defect this module closes: engine, plots and
tests all call :func:`classify_index`, and the numbers themselves live in exactly one
place — :class:`atomfrust.settings.ClassifySettings`, where ``extra="forbid"`` makes a
dead key impossible.

**Provenance of 0.78 and −1.0.** They are not universal constants. They are quantiles of a
*coarse-grained* null distribution, inherited from Ferreiro et al.'s frustratometer and
carried into Chen et al. 2020 unchanged: roughly the top decile and bottom decile of the
frustration index computed under that particular contact definition, that particular decoy
construction and that particular energy function. The number is a proxy for a quantile, and
the mapping between the two holds only for the setup it was calibrated on. Change the
contact definition (Cα–Cα 10 Å → heavy-atom 6 Å), the decoy axis (sequence identity →
pose), or the many-body formula, and the F distribution shifts and rescales — 0.78 then
selects some other quantile, and the reported "minimally frustrated" count changes for a
purely definitional reason. This is exactly the axis this project sweeps, which is why
:func:`quantile_thresholds` exists: it re-derives cutoffs from an observed distribution so
a swept setting is compared against its own null rather than against Ferreiro's.

The reproduction run stays pinned to ``mode="fixed"`` with the published literals; anything
else would not be a reproduction.

**NaN policy.** :func:`classify_index` raises on NaN. Silence would be worse: every
comparison against NaN is false, so a NaN would fall through to ``neutral`` and inflate
precisely the descriptor that is already the most suspect one (``n_neutral`` is the count
that "wins" the affinity correlation in the prototype, which is evidence it tracks pocket
size). A NaN index means the decoy vector was degenerate — zero variance, zero MAD, or
fewer than two decoys — and the caller must decide what that contact means. ``±inf`` is
*not* an error: it is an unambiguous, if extreme, classification.
"""

from __future__ import annotations

import numpy as np

from atomfrust.settings import ClassifySettings

__all__ = [
    "CLASSES",
    "MINIMALLY_FRUSTRATED",
    "NEUTRAL",
    "HIGHLY_FRUSTRATED",
    "classify_index",
    "class_counts",
    "quantile_thresholds",
]

MINIMALLY_FRUSTRATED = "minimally_frustrated"
NEUTRAL = "neutral"
HIGHLY_FRUSTRATED = "highly_frustrated"

#: Canonical order: most favourable first. Callers building count columns or plot
#: categories should iterate this rather than hardcoding the strings, so a renamed class
#: is a one-line change here.
CLASSES = (MINIMALLY_FRUSTRATED, NEUTRAL, HIGHLY_FRUSTRATED)


def _resolve(thresholds: ClassifySettings | None) -> ClassifySettings:
    """Fall back to the published defaults and reject a degenerate ordering.

    ``minimally_frustrated < highly_frustrated`` would make the two regions overlap, and
    the classifier would silently resolve the overlap by evaluation order rather than by
    meaning. ``ClassifySettings`` cannot catch this itself: neither field is invalid alone.
    """
    t = ClassifySettings() if thresholds is None else thresholds
    if t.minimally_frustrated < t.highly_frustrated:
        raise ValueError(
            "thresholds are inverted: minimally_frustrated "
            f"({t.minimally_frustrated}) < highly_frustrated ({t.highly_frustrated})"
        )
    return t


def classify_index(F: np.ndarray, thresholds: ClassifySettings | None = None) -> np.ndarray:
    """Label each frustration index, elementwise.

    ``F > minimally_frustrated`` → minimally frustrated; ``F < highly_frustrated`` →
    highly frustrated; everything else neutral. Both comparisons are **strict**, so a value
    sitting exactly on a threshold is neutral — matching ``frustration.py:703-708``
    verbatim, because stored classifications must not shift under this refactor.

    Recall the sign convention: Rosetta energies are negative-is-favourable and
    ``F = (mean(E_decoy) − E_native)/std``, so a native contact more favourable than its
    decoys has *positive* F. Minimally frustrated is the upper tail.

    ``thresholds.mode`` is deliberately not consulted. This function applies numbers; the
    numbers for ``mode="quantile"`` are produced by :func:`quantile_thresholds` (driven by
    ``atomfrust calibrate``) and passed in already resolved. Self-calibrating per call
    would make each system's classes depend on its own distribution invisibly.

    Returns an object-dtype array shaped like ``F``, so the result drops straight into a
    pandas column without a string-width dtype to widen later.
    """
    t = _resolve(thresholds)
    values = np.asarray(F, dtype=np.float64)

    if np.isnan(values).any():
        n_nan = int(np.isnan(values).sum())
        raise ValueError(
            f"{n_nan} NaN value(s) in F. A NaN index is an undefined classification "
            "(degenerate decoy vector: zero variance, zero MAD, or n < 2), and every "
            "comparison against it is false, so it would be silently counted neutral. "
            "Resolve it upstream — the zscore convention maps sigma < 1e-9 to F = 0 — or "
            "drop those contacts explicitly."
        )

    labels = np.where(
        values > t.minimally_frustrated,
        MINIMALLY_FRUSTRATED,
        np.where(values < t.highly_frustrated, HIGHLY_FRUSTRATED, NEUTRAL),
    )
    return labels.astype(object)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    """Count each class, in :data:`CLASSES` order.

    Every class is present in the result even at zero, so a summary table has stable
    columns across systems. An unrecognised label raises rather than being dropped: a
    dropped label would make the counts silently fail to sum to the number of contacts,
    which is the failure mode this whole module exists to prevent.
    """
    flat = np.asarray(labels, dtype=object).ravel()
    counts = dict.fromkeys(CLASSES, 0)
    if flat.size == 0:
        return counts

    values, freq = np.unique(flat.astype(str), return_counts=True)
    unknown = sorted(set(values) - set(CLASSES))
    if unknown:
        raise ValueError(f"unknown class label(s) {unknown}; expected one of {list(CLASSES)}")
    counts.update({str(v): int(c) for v, c in zip(values, freq)})
    return counts


def quantile_thresholds(
    F: np.ndarray, minimal_q: float = 0.9, highly_q: float = 0.1
) -> ClassifySettings:
    """Definition-specific cutoffs from an observed F distribution.

    The published 0.78 / −1.0 encode "top decile / bottom decile" of one particular null.
    Under a different contact definition, decoy axis or many-body formula the distribution
    moves and those literals no longer mark deciles — so a swept run that keeps them is
    comparing shifted distributions against a fixed ruler. Passing the pooled F values of
    the run being analysed restores the intended meaning of the cutoff.

    Feed this the pooled distribution the classification will be applied to (all contacts
    of all systems in one setting), not a single system: per-system calibration would
    force every structure to the same class fractions by construction and destroy the
    between-structure signal.

    Non-finite values are ignored here — ``±inf`` has no place in a quantile and NaN is
    the degenerate case described in the module docstring. That is a deliberate asymmetry
    with :func:`classify_index`, which refuses NaN: summarising a distribution and
    labelling a specific contact are different obligations.
    """
    for name, q in (("minimal_q", minimal_q), ("highly_q", highly_q)):
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {q}")
    if minimal_q < highly_q:
        raise ValueError(
            f"minimal_q ({minimal_q}) must be >= highly_q ({highly_q}); the upper tail is "
            "the minimally frustrated one"
        )

    values = np.asarray(F, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError(
            f"need at least 2 finite F values to calibrate, got {finite.size} "
            f"of {values.size}"
        )

    return ClassifySettings(
        minimally_frustrated=float(np.quantile(finite, minimal_q)),
        highly_frustrated=float(np.quantile(finite, highly_q)),
        mode="quantile",
    )
