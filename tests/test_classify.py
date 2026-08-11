"""D2 tests for atomfrust.analyze.classify — semantics, boundaries, NaN, calibration, R25."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomfrust.analyze.classify import (
    CLASSES,
    class_counts,
    classify_index,
    quantile_thresholds,
)
from atomfrust.settings import ClassifySettings

pytestmark = pytest.mark.unit

DEFAULTS = ClassifySettings()


# ------------------------------------------------------------------ core semantics


def test_matches_the_prototype_rule():
    """frustration.py:703-708 verbatim: > min -> minimal, < high -> highly, else neutral."""
    F = np.array([5.0, 0.79, 0.78, 0.5, -1.0, -1.01, -3.0])
    assert list(classify_index(F)) == [
        "minimally_frustrated",
        "minimally_frustrated",
        "neutral",
        "neutral",
        "neutral",
        "highly_frustrated",
        "highly_frustrated",
    ]


def test_exactly_on_a_threshold_is_neutral():
    """Both comparisons are strict. Stored classifications must not shift under D2."""
    on_the_line = np.array([DEFAULTS.minimally_frustrated, DEFAULTS.highly_frustrated])
    assert set(classify_index(on_the_line)) == {"neutral"}


def test_labels_are_object_dtype_and_shape_preserving():
    F = np.array([[2.0, 0.0], [-2.0, 0.9]])
    labels = classify_index(F)
    assert labels.dtype == object
    assert labels.shape == F.shape
    assert set(labels.ravel()) <= set(CLASSES)


def test_accepts_a_pandas_series_and_a_python_list():
    expected = ["minimally_frustrated", "neutral", "highly_frustrated"]
    values = [2.0, 0.0, -2.0]
    assert list(classify_index(pd.Series(values))) == expected
    assert list(classify_index(values)) == expected


def test_empty_input_gives_empty_labels_and_zero_counts():
    labels = classify_index(np.array([]))
    assert labels.shape == (0,)
    assert class_counts(labels) == dict.fromkeys(CLASSES, 0)


def test_infinities_classify_rather_than_raise():
    """+-inf is extreme but unambiguous; only NaN is undefined."""
    assert list(classify_index(np.array([np.inf, -np.inf]))) == [
        "minimally_frustrated",
        "highly_frustrated",
    ]


# ------------------------------------------------------------------ custom thresholds


def test_custom_thresholds_move_the_boundaries():
    F = np.array([0.5, 0.0, -0.4])
    tight = ClassifySettings(minimally_frustrated=0.25, highly_frustrated=-0.25)
    assert list(classify_index(F, tight)) == [
        "minimally_frustrated",
        "neutral",
        "highly_frustrated",
    ]
    # The same values under the published defaults are all neutral.
    assert set(classify_index(F)) == {"neutral"}


def test_inverted_thresholds_raise_rather_than_resolving_by_evaluation_order():
    bad = ClassifySettings(minimally_frustrated=-2.0, highly_frustrated=1.0)
    with pytest.raises(ValueError, match="inverted"):
        classify_index(np.array([0.0]), bad)


# -------------------------------------------------------------------------- NaN policy


def test_nan_raises_instead_of_being_counted_neutral():
    """Every comparison against NaN is false, so silence would inflate n_neutral."""
    with pytest.raises(ValueError, match="NaN"):
        classify_index(np.array([1.0, np.nan, -2.0]))


def test_nan_error_reports_how_many():
    with pytest.raises(ValueError, match=r"^2 NaN"):
        classify_index(np.array([np.nan, 0.0, np.nan]))


# ----------------------------------------------------------------------- class_counts


def test_counts_cover_every_class_including_absent_ones():
    counts = class_counts(classify_index(np.array([2.0, 2.0, 0.0])))
    assert counts == {
        "minimally_frustrated": 2,
        "neutral": 1,
        "highly_frustrated": 0,
    }
    assert list(counts) == list(CLASSES)


def test_counts_sum_to_the_number_of_contacts():
    rng = np.random.default_rng(0)
    F = rng.normal(0.0, 1.5, size=500)
    assert sum(class_counts(classify_index(F)).values()) == F.size


def test_counts_accept_a_pandas_column():
    labels = pd.Series(classify_index(np.array([2.0, -2.0, 0.0])))
    assert class_counts(labels) == {
        "minimally_frustrated": 1,
        "neutral": 1,
        "highly_frustrated": 1,
    }


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="unknown class label"):
        class_counts(np.array(["neutral", "frustrated-ish"], dtype=object))


# ------------------------------------------------------------------ quantile_thresholds


def test_quantile_thresholds_recover_the_intended_split():
    """On a synthetic distribution, decile cutoffs must reproduce decile class fractions."""
    F = np.random.default_rng(7).normal(3.0, 2.0, size=20_000)
    calibrated = quantile_thresholds(F, minimal_q=0.9, highly_q=0.1)

    counts = class_counts(classify_index(F, calibrated))
    assert counts["minimally_frustrated"] / F.size == pytest.approx(0.1, abs=0.01)
    assert counts["highly_frustrated"] / F.size == pytest.approx(0.1, abs=0.01)
    assert counts["neutral"] / F.size == pytest.approx(0.8, abs=0.02)


def test_quantile_thresholds_track_a_shifted_distribution():
    """The point of R25: a shifted F scale makes the fixed literals mean something else."""
    rng = np.random.default_rng(11)
    shifted = rng.normal(10.0, 1.0, size=5_000)

    fixed = class_counts(classify_index(shifted))
    assert fixed["minimally_frustrated"] == shifted.size  # 0.78 is now far in the left tail

    calibrated = quantile_thresholds(shifted)
    assert class_counts(classify_index(shifted, calibrated))[
        "minimally_frustrated"
    ] / shifted.size == pytest.approx(0.1, abs=0.02)


def test_quantile_thresholds_are_marked_as_such():
    settings = quantile_thresholds(np.linspace(-3.0, 3.0, 101))
    assert settings.mode == "quantile"
    assert settings.minimally_frustrated > settings.highly_frustrated
    assert isinstance(settings, ClassifySettings)


def test_quantile_thresholds_ignore_non_finite_values():
    clean = np.linspace(-2.0, 2.0, 201)
    dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
    a, b = quantile_thresholds(clean), quantile_thresholds(dirty)
    assert (a.minimally_frustrated, a.highly_frustrated) == pytest.approx(
        (b.minimally_frustrated, b.highly_frustrated)
    )


def test_quantile_thresholds_reject_bad_input():
    F = np.linspace(-1.0, 1.0, 50)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quantile_thresholds(F, minimal_q=1.5)
    with pytest.raises(ValueError, match="must be >="):
        quantile_thresholds(F, minimal_q=0.1, highly_q=0.9)
    with pytest.raises(ValueError, match="at least 2 finite"):
        quantile_thresholds(np.array([np.nan, 1.0]))


# ------------------------------------------------------- R25: the anti-triplication guard


def _float_literals(path: Path) -> set[float]:
    """Numeric literals that reach *code*, negation folded in.

    A plain text grep is the wrong instrument here: the provenance note in
    ``classify.py`` and the field description in ``settings.py`` both spell the numbers
    out in prose, and prose is not a second source of truth. Parsing means only a literal
    a program would evaluate counts.
    """
    found: set[float] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if not isinstance(node.value, bool):
                found.add(float(node.value))
        elif (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            found.add(-float(node.operand.value))
    return found


def test_thresholds_are_written_down_exactly_once_in_the_package():
    """R25. The prototype has these numbers in three modules plus a config block nobody
    reads (frustration.py:703-708, run_pipeline.py:415-416, test_frustration.py:139-144,
    config.yaml:25-27). In atomfrust they are pydantic defaults, in one file, and every
    consumer goes through classify_index."""
    package = Path(inspect.getfile(ClassifySettings)).parent
    assert package.name == "atomfrust"

    holders = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if {DEFAULTS.minimally_frustrated, DEFAULTS.highly_frustrated} <= _float_literals(path)
    }
    assert holders == {"settings.py"}, (
        f"thresholds must live in exactly one module; found in {sorted(holders)}"
    )


def test_classify_does_not_restate_the_numbers():
    """Weaker but sharper than the sweep above: this module must import them, not copy."""
    import atomfrust.analyze.classify as classify

    literals = _float_literals(Path(inspect.getfile(classify)))
    assert DEFAULTS.minimally_frustrated not in literals
    assert DEFAULTS.highly_frustrated not in literals


def test_defaults_are_the_published_values():
    """Pins the numbers themselves, so a change to settings.py is a deliberate act."""
    assert (DEFAULTS.minimally_frustrated, DEFAULTS.highly_frustrated) == (0.78, -1.0)
    assert DEFAULTS.mode == "fixed"
