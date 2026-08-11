"""D7 tests for atomfrust.report — the covariate guard, the triple, and the figures.

The load-bearing test is :func:`test_render_report_withholds_the_headline_of_a_confounded_descriptor`.
Everything else supports it: a confounded descriptor is constructed so that its only route
to the outcome is through ``n_contacts_total``, and the renderer must then print the
covariate warning where its headline would have gone.

The last section runs the report over the real 61-structure summary and prints the measured
triple for each descriptor. It asserts structure only — that the guard's verdict follows
from the two CIs, and that no headline is printed for a descriptor whose partial CI spans
zero — because measuring is the point and n has moved from the 19 of ``CLAUDE.md`` to 61.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from atomfrust.report import (
    COVARIATE_WARNING,
    HEADLINE_MARK,
    collect_analyses,
    correlation_triple,
    default_descriptors,
    headline_is_permitted,
    render_report,
    report_table,
    resolve_descriptor,
)
from atomfrust.report.plots import confound_figure, marker_sizes

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
SUMMARY_CSV = REPO / "results" / "egfr_frustration_summary.csv"

# Small enough to keep the tier under a few seconds; large enough that a percentile
# interval on a strong effect is stable.
N_BOOT = 400
N_PERM = 400


# ------------------------------------------------------------------- synthetic fixtures


def synthetic(n: int = 60, seed: int = 11, genuine: bool = False) -> pd.DataFrame:
    """A table shaped like the real summary, with a known causal structure.

    ``n_minimally_frustrated`` is pocket size plus noise and nothing else, so its
    association with the outcome runs entirely through ``n_contacts_total`` — the situation
    plan §2.1 says the published formula guarantees. ``frac_minimally`` is independent of
    pocket size, and with ``genuine=True`` it is given a real effect on the outcome, which
    is the case the guard must *not* block.
    """
    rng = np.random.default_rng(seed)
    contacts = rng.normal(600.0, 60.0, n)
    frac = rng.normal(0.5, 0.03, n)
    outcome = 0.012 * contacts + rng.normal(0.0, 0.4, n)
    if genuine:
        outcome = outcome + 20.0 * frac
    return pd.DataFrame(
        {
            "pdb_id": [f"S{i:03d}" for i in range(n)],
            "n_contacts_total": contacts,
            "log10_affinity_pM": outcome,
            "n_minimally_frustrated": 0.5 * contacts + rng.normal(0.0, 20.0, n),
            "frac_minimally": frac,
        }
    )


def section(markdown: str, descriptor: str) -> str:
    """The one ``### <descriptor>`` block of a rendered report."""
    for block in markdown.split("### "):
        if block.startswith(descriptor + "\n"):
            return block
    raise AssertionError(f"no section for {descriptor!r} in:\n{markdown}")


def excludes_zero(low, high) -> bool:
    return bool(np.isfinite(low) and np.isfinite(high) and low * high > 0)


# ------------------------------------------------------------------------- collection


def test_collect_analyses_reads_run_layout_and_fills_ids(tmp_path):
    for system, minimal in (("1XKK", 302), ("5GMP", 288)):
        analysis = tmp_path / "systems" / system / "analyses" / "shell6"
        analysis.mkdir(parents=True)
        (analysis / "summary.json").write_text(
            json.dumps(
                {
                    "descriptors": {"desc__count_minimal__zscore__shell6": minimal},
                    "covariates": {"n_contacts_total": 700},
                }
            )
        )

    table = collect_analyses(tmp_path)

    assert sorted(table["system_id"]) == ["1XKK", "5GMP"]
    assert set(table["analysis_id"]) == {"shell6"}
    # Nested sections are flattened onto the descriptor naming convention, not kept as dicts.
    assert "descriptors__desc__count_minimal__zscore__shell6" in table.columns
    assert "covariates__n_contacts_total" in table.columns


def test_collect_analyses_accepts_a_frame_a_csv_and_a_mixture(tmp_path):
    frame = synthetic(n=6)
    csv = tmp_path / "summary.csv"
    frame.to_csv(csv, index=False)

    assert len(collect_analyses(frame)) == 6
    assert len(collect_analyses(csv)) == 6
    assert len(collect_analyses([frame, csv])) == 12
    # A copy, not the caller's frame: the report layer must not mutate its input.
    assert collect_analyses(frame) is not frame


def test_resolve_descriptor_prefers_exact_then_registry_then_legacy():
    registry = pd.DataFrame({"desc__frac_minimal__zscore__ca10": [0.5], "n_contacts_total": [7]})
    legacy = pd.DataFrame({"frac_minimally": [0.5], "n_contacts_total": [7]})

    assert resolve_descriptor(registry, "frac_minimal") == "desc__frac_minimal__zscore__ca10"
    assert resolve_descriptor(registry, "frac_minimally") == "desc__frac_minimal__zscore__ca10"
    assert resolve_descriptor(legacy, "frac_minimal") == "frac_minimally"
    assert resolve_descriptor(legacy, "n_contacts_total") == "n_contacts_total"
    with pytest.raises(KeyError):
        resolve_descriptor(legacy, "mean_Z")


def test_two_shells_of_one_descriptor_must_be_named_explicitly():
    table = pd.DataFrame(
        {"desc__frac_minimal__zscore__ca6": [0.4], "desc__frac_minimal__zscore__ca10": [0.5]}
    )
    with pytest.raises(ValueError, match="matches 2 columns"):
        resolve_descriptor(table, "frac_minimal")


def test_default_descriptors_excludes_the_covariate():
    assert "n_contacts_total" not in default_descriptors(synthetic(n=5))


# ------------------------------------------------------------------------ the guard


def test_a_descriptor_that_only_tracks_the_covariate_loses_its_partial_and_its_headline():
    table = synthetic()
    triple = correlation_triple(
        table, "n_minimally_frustrated", "log10_affinity_pM", n_boot=N_BOOT, seed=0
    )
    raw, partial = triple["raw"], triple["partial"]

    assert abs(raw.value) > 0.4
    assert excludes_zero(raw.ci_low, raw.ci_high)
    assert abs(partial.value) < 0.2
    assert not excludes_zero(partial.ci_low, partial.ci_high)

    row = {
        "raw_ci_low": raw.ci_low,
        "raw_ci_high": raw.ci_high,
        "partial_ci_low": partial.ci_low,
        "partial_ci_high": partial.ci_high,
    }
    assert headline_is_permitted(row) is False


def test_a_genuine_effect_independent_of_the_covariate_survives_the_guard():
    table = synthetic(genuine=True)
    triple = correlation_triple(table, "frac_minimally", "log10_affinity_pM", n_boot=N_BOOT, seed=0)
    raw, partial, ols = triple["raw"], triple["partial"], triple["ols"]

    assert excludes_zero(raw.ci_low, raw.ci_high)
    assert excludes_zero(partial.ci_low, partial.ci_high)
    assert excludes_zero(ols.ci_low, ols.ci_high)
    assert triple["vif"] < 1.5  # independent of pocket size, so nothing is inflated
    assert headline_is_permitted(
        {
            "raw_ci_low": raw.ci_low,
            "raw_ci_high": raw.ci_high,
            "partial_ci_low": partial.ci_low,
            "partial_ci_high": partial.ci_high,
        }
    )


def test_a_missing_partial_ci_fails_the_guard_rather_than_passing_by_default():
    significant_raw = {"raw_ci_low": 0.2, "raw_ci_high": 0.6}
    assert headline_is_permitted({**significant_raw, "partial_ci_low": None, "partial_ci_high": None}) is False
    assert headline_is_permitted({**significant_raw, "partial_ci_low": 0.1, "partial_ci_high": 0.5}) is True
    # An inconclusive raw correlation has no headline to withhold, so the guard is silent.
    assert headline_is_permitted({"raw_ci_low": -0.2, "raw_ci_high": 0.6}) is True


def test_a_descriptor_that_is_the_covariate_reports_no_partial_and_no_headline():
    triple = correlation_triple(
        synthetic(), "n_contacts_total", "log10_affinity_pM", n_boot=N_BOOT, seed=0
    )
    assert np.isnan(triple["partial"].value)
    assert triple["partial"].ci_low is None
    assert np.isinf(triple["vif"])
    assert (
        headline_is_permitted(
            {
                "raw_ci_low": triple["raw"].ci_low,
                "raw_ci_high": triple["raw"].ci_high,
                "partial_ci_low": None,
                "partial_ci_high": None,
            }
        )
        is False
    )


def test_vif_rises_as_the_descriptor_approaches_the_covariate():
    rng = np.random.default_rng(3)
    n = 60
    contacts = rng.normal(600.0, 60.0, n)
    noise = rng.normal(0.0, 1.0, n)
    outcome = rng.normal(0.0, 1.0, n)

    vifs = []
    for spread in (60.0, 20.0, 6.0, 2.0):
        table = pd.DataFrame(
            {
                "n_contacts_total": contacts,
                "log10_affinity_pM": outcome,
                "x": contacts + spread * noise,
            }
        )
        vifs.append(correlation_triple(table, "x", "log10_affinity_pM", n_boot=0)["vif"])

    assert vifs == sorted(vifs)
    assert vifs[0] < 3.0 < vifs[-1]


# ------------------------------------------------------------------------ the table


def test_report_table_carries_the_triple_the_vif_and_the_maxT_adjustment():
    table = synthetic(genuine=True)
    result = report_table(
        table,
        ["n_minimally_frustrated", "frac_minimally"],
        "log10_affinity_pM",
        n_perm=N_PERM,
        seed=0,
        n_boot=N_BOOT,
    )

    expected = {
        "descriptor", "column", "outcome", "covariate", "n",
        "raw_r", "raw_ci_low", "raw_ci_high", "raw_p",
        "partial_r", "partial_ci_low", "partial_ci_high", "partial_p",
        "ols_coef", "ols_ci_low", "ols_ci_high", "ols_p",
        "vif", "p_raw", "p_maxT_adjusted", "headline_permitted",
    }
    assert expected <= set(result.columns)
    assert list(result["descriptor"]) == ["n_minimally_frustrated", "frac_minimally"]
    assert (result["n"] == len(table)).all()
    # The price of the sweep: the adjustment can only move a p-value up.
    assert (result["p_maxT_adjusted"] >= result["p_raw"] - 1e-12).all()
    assert not bool(result.set_index("descriptor").loc["n_minimally_frustrated", "headline_permitted"])
    assert bool(result.set_index("descriptor").loc["frac_minimally", "headline_permitted"])


def test_every_row_shares_one_complete_case_sample():
    table = synthetic(genuine=True)
    table.loc[0, "frac_minimally"] = np.nan
    result = report_table(
        table,
        ["n_minimally_frustrated", "frac_minimally"],
        "log10_affinity_pM",
        n_perm=N_PERM,
        seed=0,
        n_boot=0,
    )
    assert set(result["n"]) == {len(table) - 1}


# ------------------------------------------------------------------------ rendering


def test_render_report_withholds_the_headline_of_a_confounded_descriptor(tmp_path):
    """The behaviour D7 exists for: the covariate warning goes where the headline would."""
    out = render_report(
        synthetic(genuine=True),
        tmp_path,
        outcome="log10_affinity_pM",
        n_perm=N_PERM,
        n_boot=N_BOOT,
        seed=0,
    )
    text = out.read_text()

    confounded = section(text, "n_minimally_frustrated")
    assert COVARIATE_WARNING in confounded
    assert HEADLINE_MARK not in confounded
    # The numbers are still there — the warning exposes the confound, it does not hide it.
    assert "raw r =" in confounded and "partial r" in confounded

    assert "n_minimally_frustrated" in text.split("Headlines withheld under the covariate guard:")[1]
    assert (tmp_path / "report_table.csv").is_file()
    assert (tmp_path / "confound.png").is_file()


def test_render_report_does_print_a_headline_when_the_partial_survives(tmp_path):
    """Without this the guard test above would pass on a renderer that never headlines."""
    out = render_report(
        synthetic(genuine=True),
        tmp_path,
        outcome="log10_affinity_pM",
        n_perm=N_PERM,
        n_boot=N_BOOT,
        seed=0,
    )
    genuine = section(out.read_text(), "frac_minimally")
    assert HEADLINE_MARK in genuine
    assert COVARIATE_WARNING not in genuine


def test_render_report_reports_no_finding_when_nothing_is_significant(tmp_path):
    out = render_report(
        synthetic(genuine=False),
        tmp_path,
        outcome="log10_affinity_pM",
        n_perm=N_PERM,
        n_boot=N_BOOT,
        seed=0,
    )
    noise = section(out.read_text(), "frac_minimally")
    assert HEADLINE_MARK not in noise
    assert "No finding" in noise


# --------------------------------------------------------------------------- plots


def test_marker_sizes_span_the_range_and_survive_a_constant_covariate():
    sizes = marker_sizes(np.array([476.0, 600.0, 733.0]))
    assert sizes[0] < sizes[1] < sizes[2]
    assert np.allclose(marker_sizes(np.array([5.0, 5.0, 5.0])), marker_sizes(np.array([5.0])))


def test_confound_figure_writes_a_file_and_leaves_no_open_figure(tmp_path):
    plt.close("all")
    before = len(plt.get_fignums())

    path = confound_figure(
        synthetic(), "log10_affinity_pM", out_path=tmp_path / "fig.png"
    )

    assert path.is_file() and path.stat().st_size > 0
    assert len(plt.get_fignums()) == before  # no state leaks into the next test


def test_confound_figure_needs_the_outcome_and_the_covariate(tmp_path):
    table = synthetic().drop(columns=["n_contacts_total"])
    with pytest.raises(KeyError):
        confound_figure(table, "log10_affinity_pM", out_path=tmp_path / "fig.png")


# ------------------------------------------------------- the real 61-structure summary


@pytest.mark.skipif(not SUMMARY_CSV.is_file(), reason="results/ not pulled (dvc pull)")
def test_measure_the_real_summary_and_enforce_the_guard_on_what_it_finds(capsys):
    """Run the report over ``results/egfr_frustration_summary.csv`` and print the triple.

    No correlation value is asserted. ``CLAUDE.md``'s numbers are Spearman rhos at n = 19
    and this file now holds 61 rows, so the published values do not apply and pinning any
    replacement would freeze a measurement the pipeline is meant to keep making. What is
    asserted is structural and must hold whatever the numbers are: the guard's verdict
    follows from the two intervals, and the renderer prints no headline for any descriptor
    whose partial CI spans zero.
    """
    table = collect_analyses(SUMMARY_CSV)
    descriptors = [
        "n_minimally_frustrated",
        "n_neutral",
        "n_contacts_total",
        "frac_minimally",
    ]
    result = report_table(
        table, descriptors, "log10_affinity_pM", n_perm=2000, seed=0, n_boot=2000
    )

    with capsys.disabled():
        print(f"\nreal summary: n = {int(result['n'].iloc[0])} systems")
        print(
            result[
                [
                    "descriptor", "raw_r", "raw_ci_low", "raw_ci_high",
                    "partial_r", "partial_ci_low", "partial_ci_high",
                    "ols_coef", "vif", "p_raw", "p_maxT_adjusted", "headline_permitted",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v: .4f}")
        )

    assert int(result["n"].iloc[0]) == len(table)
    for _, row in result.iterrows():
        assert headline_is_permitted(row) == (
            not (
                excludes_zero(row["raw_ci_low"], row["raw_ci_high"])
                and not excludes_zero(row["partial_ci_low"], row["partial_ci_high"])
            )
        ), row["descriptor"]

    # n_contacts_total is the covariate: held against itself it has no partial at all, so
    # there is no interval that could ever clear the renderer's headline condition.
    contacts = result.set_index("descriptor").loc["n_contacts_total"]
    assert np.isnan(contacts["partial_r"])
    assert not excludes_zero(contacts["partial_ci_low"], contacts["partial_ci_high"])


@pytest.mark.skipif(not SUMMARY_CSV.is_file(), reason="results/ not pulled (dvc pull)")
def test_render_report_on_the_real_summary_prints_no_unsupported_headline(tmp_path):
    out = render_report(
        SUMMARY_CSV, tmp_path, outcome="log10_affinity_pM", n_perm=500, n_boot=500, seed=0
    )
    text = out.read_text()
    result = pd.read_csv(tmp_path / "report_table.csv")

    for _, row in result.iterrows():
        block = section(text, row["descriptor"])
        partial_survives = excludes_zero(row["partial_ci_low"], row["partial_ci_high"])
        if not (partial_survives and row["p_maxT_adjusted"] <= 0.05):
            assert HEADLINE_MARK not in block, row["descriptor"]
        if not row["headline_permitted"]:
            assert COVARIATE_WARNING in block, row["descriptor"]
    assert (tmp_path / "confound.png").is_file()
