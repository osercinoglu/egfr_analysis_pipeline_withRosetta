"""Gathering per-system summaries and reporting them so the confound cannot be hidden.

**A correlation is never reported alone here.** Every descriptor gets three numbers, always
together: the raw Pearson r, the partial r controlling ``n_contacts_total``, and the
descriptor coefficient of ``y ~ descriptor + n_contacts_total`` with its VIF. The reason is
step A1, not caution. Under the published many-body formula the per-contact index is
algebraically ``0.5·(B_i + B_j)`` — a sum of two per-residue totals — so summing a
per-contact class over a pocket sums a per-residue quantity weighted by graph degree, and a
"number of minimally frustrated contacts" is a degree statistic before it is a frustration
statistic. ``CLAUDE.md`` records the observation that follows: at n = 19 the quantities that
correlated significantly with ``log10(affinity_pM)`` were ``n_neutral`` (ρ = −0.541,
p = 0.017) and ``n_contacts_total`` (ρ = −0.521, p = 0.022), while the headline
``n_minimally_frustrated`` did not (ρ = −0.347, p = 0.145). A *neutral* count winning is the
tell — neutral contacts carry no frustration information by construction.

**The guard is structural, not advisory.** :func:`headline_is_permitted` is False whenever a
descriptor's raw CI excludes zero and its partial CI does not, and :func:`render_report`
then prints the covariate warning where the headline would have gone. A reporting layer that
merely *offers* the partial correlation would let the confounded number be published anyway;
this one cannot emit it.

**Multiplicity is max-T, not Benjamini-Hochberg.** BH controls FDR over a fixed,
pre-specified family. This tool makes contact definition, cutoff, shell, many-body mode,
index function, thresholds, descriptor and decoy axis all selectable, so a reported best case
is a maximum over >10^4 configurations and BH does not price a maximum (plan §2.3).
:func:`report_table` therefore carries ``p_maxT_adjusted`` from
:func:`~atomfrust.metrics.maxT_permutation`, whose null is the largest |r| over the whole
descriptor grid. For scale: at n = 19, r = 0.456 is already the 5% critical value for a
*single* pre-specified test.

Everything here is pure pandas/NumPy over stored summaries — no run is executed, no
PyRosetta is imported.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from atomfrust.metrics import Estimate, bootstrap_ci, maxT_permutation, pearson
from atomfrust.runstore import RunDir, SystemDir

__all__ = [
    "DEFAULT_COVARIATE",
    "collect_analyses",
    "resolve_descriptor",
    "default_descriptors",
    "correlation_triple",
    "report_table",
    "headline_is_permitted",
    "render_report",
]

#: The covariate every correlation is controlled for unless the caller names another. It is
#: the pocket contact count — the quantity A1 showed the class counts are proportional to.
DEFAULT_COVARIATE = "n_contacts_total"

#: Legacy prototype column names for the D3 descriptor registry, so a report can be run
#: against ``results/egfr_frustration_summary.csv`` and against a modern
#: ``desc__{name}__{index}__{shell}`` table with the same descriptor argument. Resolution is
#: exact-name first, then registry prefix, then these; nothing is guessed by substring.
_LEGACY_NAMES = {
    "frac_minimal": ("frac_minimally",),
    "frac_highly": ("frac_highly_frustrated",),
    "frac_neutral": ("frac_neutral",),
    "count_minimal": ("n_minimally_frustrated",),
    "count_highly": ("n_highly_frustrated",),
    "count_neutral": ("n_neutral",),
}


# --------------------------------------------------------------------------- collection


def _flatten(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested summary sections become ``section__key`` columns.

    ``summary.json`` groups descriptors and covariates into sub-objects; a flat frame is
    what correlates. The separator matches the descriptor naming of
    :mod:`atomfrust.analyze.aggregate` so a flattened key and a registry key look alike.
    """
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{name}__"))
        else:
            flat[name] = value
    return flat


def _rows_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload if isinstance(payload, list) else [payload]
    rows = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(f"{path}: expected an object or list of objects, got {type(record)}")
        row = _flatten(record)
        parent = path.parent
        row.setdefault("analysis_id", parent.name)
        if parent.parent.name == "analyses":
            # systems/<system_id>/analyses/<analysis_id>/summary.json
            row.setdefault("system_id", parent.parent.parent.name)
        rows.append(row)
    return rows


def _summary_paths(directory: Path) -> list[Path]:
    """Which ``summary.json`` files a directory stands for.

    A run root, a system directory, an ``analyses/`` parent and a single analysis directory
    are all things a caller reasonably points at, so each is recognised by the layout of
    plan §4 rather than by requiring the caller to know which level they are on.
    """
    if (directory / "summary.json").is_file():
        return [directory / "summary.json"]
    for pattern in ("*/summary.json", "analyses/*/summary.json", "systems/*/analyses/*/summary.json"):
        found = sorted(directory.glob(pattern))
        if found:
            return found
    found = sorted(directory.rglob("summary.json"))
    if not found:
        raise FileNotFoundError(f"no summary.json under {directory}")
    return found


def _as_items(paths_or_runs: Any) -> list[Any]:
    if isinstance(paths_or_runs, (pd.DataFrame, str, Path, RunDir, SystemDir, Mapping)):
        return [paths_or_runs]
    if isinstance(paths_or_runs, Iterable):
        return list(paths_or_runs)
    raise TypeError(f"cannot collect from {type(paths_or_runs)}")


def collect_analyses(paths_or_runs: Any) -> pd.DataFrame:
    """Gather per-system summary rows into one table, one row per system analysis.

    Accepts, singly or as any iterable of them: a :class:`~pandas.DataFrame` (returned as a
    copy, which is how an in-memory :func:`~atomfrust.analyze.aggregate.summarize_many`
    result enters the report layer), a :class:`~atomfrust.runstore.RunDir` or
    :class:`~atomfrust.runstore.SystemDir`, a ``summary.json``, a directory at any level of
    the run layout, or a CSV of the prototype's
    ``results/egfr_frustration_summary.csv`` shape.

    Rows from different sources keep their own columns; missing ones come back NaN rather
    than being dropped, because a descriptor absent from one analysis is information about
    that analysis and silently intersecting the columns would hide it. ``system_id`` and
    ``analysis_id`` are filled from the path when the payload did not carry them, so rows
    from two analyses of one system stay distinguishable.
    """
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []

    for item in _as_items(paths_or_runs):
        if isinstance(item, pd.DataFrame):
            frames.append(item.copy())
            continue
        if isinstance(item, Mapping):
            rows.append(_flatten(item))
            continue
        if isinstance(item, RunDir):
            paths = [p for sid in item.systems() for p in _summary_paths(item.system(sid).base)]
        elif isinstance(item, SystemDir):
            paths = _summary_paths(item.base)
        else:
            path = Path(item)
            if path.is_dir():
                paths = _summary_paths(path)
            elif not path.exists():
                raise FileNotFoundError(path)
            elif path.suffix.lower() in {".csv", ".tsv"}:
                sep = "\t" if path.suffix.lower() == ".tsv" else ","
                frames.append(pd.read_csv(path, sep=sep))
                continue
            else:
                paths = [path]
        for p in paths:
            rows.extend(_rows_from_json(p))

    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()

    table = pd.concat(frames, ignore_index=True, sort=False)
    leading = [c for c in ("system_id", "pdb_id", "ligand_comp_id", "analysis_id") if c in table]
    return table[leading + [c for c in table.columns if c not in leading]]


# ------------------------------------------------------------------- column resolution


def resolve_descriptor(table: pd.DataFrame, name: str) -> str:
    """Column holding ``name`` — exact, then ``desc__{name}__{index}__{shell}``, then legacy.

    An ambiguous registry prefix raises instead of picking one: two shells are two different
    numbers (see :mod:`atomfrust.analyze.aggregate`), and choosing between them silently is
    exactly the comparison the naming convention exists to prevent.
    """
    if name in table.columns:
        return name
    prefixed = [c for c in table.columns if c.startswith(f"desc__{name}__")]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        raise ValueError(
            f"{name!r} matches {len(prefixed)} columns ({sorted(prefixed)}); name the "
            "index and shell explicitly"
        )
    for legacy in _LEGACY_NAMES.get(name, ()):
        if legacy in table.columns:
            return legacy
    for registry, legacy_names in _LEGACY_NAMES.items():
        if name in legacy_names:
            hit = [c for c in table.columns if c.startswith(f"desc__{registry}__")]
            if len(hit) == 1:
                return hit[0]
    raise KeyError(f"no column for descriptor {name!r}; got {list(table.columns)}")


def default_descriptors(table: pd.DataFrame) -> list[str]:
    """Every descriptor column present, registry-named ones first, legacy ones otherwise.

    The whole registry is the grid the max-T null is built over, so the default is "all of
    them" rather than a chosen few — a subset chosen after seeing the numbers is precisely
    the selection the adjustment is meant to price.
    """
    registry = sorted(c for c in table.columns if c.startswith("desc__"))
    if registry:
        return registry
    legacy = [n for names in _LEGACY_NAMES.values() for n in names]
    return [c for c in table.columns if c in set(legacy)]


# ------------------------------------------------------------------------- the triple


def _complete(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Rows finite in every column named. One row set for every number in a report.

    Per-descriptor complete-case sets would give each row of the report a different n, and
    the max-T null needs one shared observation order across the grid anyway.
    """
    sub = table.loc[:, list(dict.fromkeys(columns))].apply(pd.to_numeric, errors="coerce")
    return sub[np.isfinite(sub.to_numpy(dtype=np.float64)).all(axis=1)]


def _residuals(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _partial_r(rows: np.ndarray) -> float:
    """Partial Pearson r of columns 0 and 1 given the remaining columns.

    Residualising *inside* the bootstrap statistic rather than once outside it is what makes
    the interval a bootstrap of the partial correlation; residualising once and resampling
    the residuals would treat the covariate fit as known and understate the width.
    """
    design = np.column_stack([np.ones(rows.shape[0]), rows[:, 2:]])
    rx = _residuals(rows[:, 0], design)
    ry = _residuals(rows[:, 1], design)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _ols_coefficient(rows: np.ndarray) -> float:
    """Descriptor coefficient of ``y ~ 1 + descriptor + covariate``."""
    design = np.column_stack([np.ones(rows.shape[0]), rows[:, 0], rows[:, 2:]])
    beta, *_ = np.linalg.lstsq(design, rows[:, 1], rcond=None)
    return float(beta[1])


def _ols_p_value(rows: np.ndarray) -> float:
    """Two-sided t-test on the descriptor coefficient, for the observed sample.

    Parametric, like the p on :func:`~atomfrust.metrics.pearson`: it is what the methods
    document reports, and the bootstrap CI beside it is the part that does not assume
    normal errors.
    """
    n, k = rows.shape[0], rows.shape[1]  # predictors = k - 1, plus intercept
    df = n - k
    if df < 1:
        return float("nan")
    design = np.column_stack([np.ones(n), rows[:, 0], rows[:, 2:]])
    beta, *_ = np.linalg.lstsq(design, rows[:, 1], rcond=None)
    residual = rows[:, 1] - design @ beta
    sigma2 = float(residual @ residual) / df
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    if not np.isfinite(se) or se == 0:
        return float("nan")
    return float(2 * stats.t.sf(abs(beta[1] / se), df))


def _vif(x: np.ndarray, covariates: np.ndarray) -> float:
    """Variance inflation factor of the descriptor against the covariate block.

    ``1/(1 - R²)`` from regressing the descriptor on the covariates. Reported because the
    OLS coefficient is only interpretable while it is finite: as descriptor and covariate
    approach collinearity the coefficient's standard error diverges, and a wide CI then
    means "cannot be separated", not "no effect".
    """
    design = np.column_stack([np.ones(x.size), covariates])
    residual = _residuals(x, design)
    ss_total = float(((x - x.mean()) ** 2).sum())
    if ss_total == 0:
        return float("inf")
    r2 = 1.0 - float(residual @ residual) / ss_total
    r2 = min(max(r2, 0.0), 1.0)
    return float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)


def _maybe_bootstrap(
    rows: np.ndarray, statistic, n_boot: int, seed: int, method: str
) -> Estimate:
    """Bootstrap CI, or a point estimate when the statistic is degenerate on this sample.

    :func:`~atomfrust.metrics.bootstrap_ci` raises when every replicate is non-finite, which
    happens exactly when descriptor and covariate cannot be separated. Falling back to a
    CI-less estimate is the conservative direction: :func:`headline_is_permitted` treats an
    absent partial CI as a failed guard, so an unseparable descriptor loses its headline
    rather than acquiring one.
    """
    point = float(statistic(rows))
    if n_boot <= 0:
        return Estimate(value=point, n=rows.shape[0], method="point", seed=seed)
    try:
        est = bootstrap_ci(rows, statistic, n_boot=n_boot, seed=seed)
    except ValueError:
        return Estimate(value=point, n=rows.shape[0], method=f"{method}+degenerate", seed=seed)
    return est


def correlation_triple(
    table: pd.DataFrame,
    descriptor: str,
    outcome: str,
    covariate: str = DEFAULT_COVARIATE,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """The three numbers that must always be reported together, for one descriptor.

    ``raw``
        Pearson r of descriptor vs outcome, with a percentile bootstrap CI.
    ``partial``
        Partial Pearson r controlling ``covariate``, CI by bootstrapping the whole
        residualisation. Its p is the standard partial-correlation t on ``n − 2 − k`` df.
    ``ols``
        Descriptor coefficient of ``outcome ~ descriptor + covariate``, bootstrap CI, and a
        parametric t-test p. On the same scale as the outcome, so it answers "how much
        outcome per unit descriptor, at fixed pocket size".

    Also returns ``vif`` (descriptor against the covariate block) and ``n``. The raw
    estimate on its own is the number ``CLAUDE.md`` warns about; this function exists so
    that obtaining it costs the same call as obtaining its refutation.

    A descriptor that *is* the covariate is legal and reports partial and OLS as NaN with
    no CI. It is not an error to ask how ``n_contacts_total`` correlates with affinity —
    that question is the whole reason for the covariate — but nothing is left of it once it
    is held fixed, and the missing CI makes the guard refuse it a headline.
    """
    x_col = resolve_descriptor(table, descriptor)
    data = _complete(table, [x_col, outcome, covariate])
    if len(data) < 4:
        raise ValueError(
            f"{descriptor!r}: {len(data)} complete rows over "
            f"({x_col!r}, {outcome!r}, {covariate!r}); need at least 4"
        )

    x = data[x_col].to_numpy(dtype=np.float64)
    y = data[outcome].to_numpy(dtype=np.float64)
    c = data[covariate].to_numpy(dtype=np.float64)
    rows = np.column_stack([x, y, c])
    n = rows.shape[0]

    raw = pearson(x, y, n_boot=n_boot, seed=seed)

    if x_col == covariate:
        nothing = Estimate(value=float("nan"), n=n, method="undefined_self_covariate", seed=seed)
        return {
            "descriptor": descriptor,
            "column": x_col,
            "outcome": outcome,
            "covariate": covariate,
            "raw": raw,
            "partial": nothing,
            "ols": nothing,
            "vif": float("inf"),
            "n": n,
        }

    partial_ci = _maybe_bootstrap(rows, _partial_r, n_boot, seed, "partial_pearson")
    partial_value = partial_ci.value
    df = n - 2 - (rows.shape[1] - 2)
    if np.isfinite(partial_value) and df > 0 and abs(partial_value) < 1.0:
        t = partial_value * np.sqrt(df / (1.0 - partial_value**2))
        partial_p = float(2 * stats.t.sf(abs(t), df))
    else:
        partial_p = float("nan")
    partial = Estimate(
        value=partial_value,
        ci_low=partial_ci.ci_low,
        ci_high=partial_ci.ci_high,
        n=n,
        method=f"partial_pearson|{covariate}" + (f"+{partial_ci.method}" if n_boot > 0 else ""),
        n_boot=partial_ci.n_boot,
        seed=seed,
        p_value=partial_p,
    )

    ols_ci = _maybe_bootstrap(rows, _ols_coefficient, n_boot, seed, "ols_coef")
    ols = Estimate(
        value=ols_ci.value,
        ci_low=ols_ci.ci_low,
        ci_high=ols_ci.ci_high,
        n=n,
        method=f"ols_coef|{outcome}~{descriptor}+{covariate}"
        + (f"+{ols_ci.method}" if n_boot > 0 else ""),
        n_boot=ols_ci.n_boot,
        seed=seed,
        p_value=_ols_p_value(rows),
    )

    return {
        "descriptor": descriptor,
        "column": x_col,
        "outcome": outcome,
        "covariate": covariate,
        "raw": raw,
        "partial": partial,
        "ols": ols,
        "vif": _vif(x, c[:, None]),
        "n": n,
    }


# ------------------------------------------------------------------------- the table


def report_table(
    table: pd.DataFrame,
    descriptors: Sequence[str],
    outcome: str,
    covariate: str = DEFAULT_COVARIATE,
    n_perm: int = 10000,
    seed: int = 0,
    n_boot: int = 10000,
) -> pd.DataFrame:
    """One row per descriptor: raw, partial, OLS, VIF, and the max-T-adjusted p.

    ``p_maxT_adjusted`` comes from :func:`~atomfrust.metrics.maxT_permutation` over the
    whole descriptor grid, **not** from Benjamini-Hochberg. BH controls FDR over a fixed,
    pre-specified family; this tool sweeps >10^4 configurations (contact definition, cutoff,
    shell, many-body mode, index, thresholds, descriptor, axis) and a reported best case is
    a *maximum* over that grid, which BH does not price (plan §2.3). The max-T null is the
    distribution of the largest |r| over the grid under a permuted outcome, so ``p_raw``
    beside it shows the price of the sweep as the gap between the two. At n = 19, r = 0.456
    is already the 5% critical value for a single pre-specified test — the grid makes that
    threshold optimistic, not conservative.

    All rows share one complete-case observation set, over the outcome, the covariate and
    every descriptor: the permutation null needs one observation order across the grid, and
    a per-descriptor n would make the rows incomparable.

    ``headline_permitted`` is :func:`headline_is_permitted` evaluated on the row, carried in
    the table so that a caller who never reads the docstring still cannot format a headline
    from a confounded row without stepping over a False.
    """
    if len(descriptors) == 0:
        raise ValueError("descriptors is empty")

    columns = {name: resolve_descriptor(table, name) for name in descriptors}
    data = _complete(table, [*columns.values(), outcome, covariate])
    if len(data) < 4:
        raise ValueError(
            f"{len(data)} rows complete over the outcome, the covariate and all "
            f"{len(columns)} descriptors; need at least 4"
        )

    y = data[outcome].to_numpy(dtype=np.float64)
    grid = {name: data[column].to_numpy(dtype=np.float64) for name, column in columns.items()}
    permutation = maxT_permutation(grid, y, n_perm=n_perm, seed=seed).set_index("config")

    rows = []
    for name in descriptors:
        triple = correlation_triple(
            data, columns[name], outcome, covariate, n_boot=n_boot, seed=seed
        )
        raw, partial, ols = triple["raw"], triple["partial"], triple["ols"]
        row = {
            "descriptor": name,
            "column": columns[name],
            "outcome": outcome,
            "covariate": covariate,
            "n": triple["n"],
            "raw_r": raw.value,
            "raw_ci_low": raw.ci_low,
            "raw_ci_high": raw.ci_high,
            "raw_p": raw.p_value,
            "partial_r": partial.value,
            "partial_ci_low": partial.ci_low,
            "partial_ci_high": partial.ci_high,
            "partial_p": partial.p_value,
            "ols_coef": ols.value,
            "ols_ci_low": ols.ci_low,
            "ols_ci_high": ols.ci_high,
            "ols_p": ols.p_value,
            "vif": triple["vif"],
            "p_raw": float(permutation.loc[name, "p_raw"]),
            "p_maxT_adjusted": float(permutation.loc[name, "p_adjusted"]),
            "n_perm": int(n_perm),
            "n_boot": int(n_boot),
            "seed": int(seed),
        }
        row["headline_permitted"] = headline_is_permitted(row)
        rows.append(row)

    return pd.DataFrame(rows)


def _excludes_zero(low: Any, high: Any) -> bool:
    """True only when both bounds exist, are finite, and lie on the same side of zero.

    A missing bound is not evidence of an interval that excludes zero, so it reads False —
    which for the partial correlation means the guard fires.
    """
    try:
        low, high = float(low), float(high)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(low) and np.isfinite(high) and low * high > 0)


def headline_is_permitted(row: Mapping[str, Any]) -> bool:
    """False when the raw correlation excludes zero but the partial does not.

    That is the whole condition, and it is deliberately the *only* one: a descriptor whose
    apparent association with the outcome disappears once ``n_contacts_total`` is held fixed
    has not measured frustration, it has measured pocket size (plan §2.1). The prototype's
    ``n_minimally_frustrated`` headline is exactly this shape, which is why the check is a
    function the renderer must pass rather than a paragraph in a docstring.

    A row whose partial CI is missing also fails, because the guard cannot be shown to have
    been cleared and "not checked" must not read as "checked and fine".
    """
    return not (
        _excludes_zero(row.get("raw_ci_low"), row.get("raw_ci_high"))
        and not _excludes_zero(row.get("partial_ci_low"), row.get("partial_ci_high"))
    )


# --------------------------------------------------------------------------- rendering

COVARIATE_WARNING = "COVARIATE WARNING"
HEADLINE_MARK = "**Headline**"


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "inf" if np.isinf(value) else "n/a"
    return f"{value:.{digits}f}"


def _interval(low: Any, high: Any) -> str:
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _section(row: Mapping[str, Any], alpha: float) -> str:
    """One descriptor's verdict. Exactly one of three states, and the guard outranks both.

    ``headline`` requires all three of: the guard passed, the partial CI excludes zero, and
    the max-T-adjusted p clears ``alpha``. The adjusted p is in the condition because an
    unadjusted headline over a swept grid is the failure mode of §2.3, not a stylistic one.
    """
    permitted = headline_is_permitted(row)
    partial_survives = _excludes_zero(row.get("partial_ci_low"), row.get("partial_ci_high"))
    try:
        adjusted = float(row.get("p_maxT_adjusted"))
    except (TypeError, ValueError):
        adjusted = float("nan")
    adjusted_ok = bool(np.isfinite(adjusted)) and adjusted <= alpha

    self_covariate = str(row.get("column", row["descriptor"])) == str(row["covariate"])
    partial_line = (
        f"- partial r: undefined — this descriptor *is* `{row['covariate']}`\n"
        if self_covariate
        else f"- partial r (controlling `{row['covariate']}`) = {_fmt(row['partial_r'])} "
        f"{_interval(row['partial_ci_low'], row['partial_ci_high'])}\n"
    )
    ols_line = (
        "- OLS: not identified, and VIF = inf\n"
        if self_covariate
        else f"- OLS coefficient in `{row['outcome']} ~ {row['descriptor']} + {row['covariate']}` "
        f"= {_fmt(row['ols_coef'])} {_interval(row['ols_ci_low'], row['ols_ci_high'])}, "
        f"VIF = {_fmt(row['vif'], 2)}\n"
    )
    common = (
        f"- raw r = {_fmt(row['raw_r'])} {_interval(row['raw_ci_low'], row['raw_ci_high'])}\n"
        + partial_line
        + ols_line
        + f"- p_raw = {_fmt(row['p_raw'], 4)}, p_maxT_adjusted = "
        f"{_fmt(row['p_maxT_adjusted'], 4)}, n = {int(row['n'])}\n"
    )

    header = f"### {row['descriptor']}\n\n"
    if self_covariate and permitted:
        return (
            header
            + "No finding: the descriptor is the covariate, so nothing is left of it once "
            "the covariate is held fixed. Its raw correlation is reported because it is the "
            "quantity every other row is controlled for, not as a frustration result.\n\n"
            + common
        )
    if not permitted:
        return (
            header
            + f"{COVARIATE_WARNING}: the raw correlation excludes zero, the partial "
            f"correlation controlling `{row['covariate']}` does not. No headline is "
            "reported for this descriptor. The association is accounted for by pocket "
            "size; under the many-body formula of plan §2.1 a class count over a pocket "
            "is a graph-degree statistic before it is a frustration statistic.\n\n"
            + common
        )
    if partial_survives and adjusted_ok:
        return (
            header
            + f"{HEADLINE_MARK} — partial r = {_fmt(row['partial_r'])} "
            f"{_interval(row['partial_ci_low'], row['partial_ci_high'])} at fixed "
            f"`{row['covariate']}`, max-T-adjusted p = {_fmt(row['p_maxT_adjusted'], 4)}.\n\n"
            + common
        )
    reason = (
        "the partial CI includes zero"
        if not partial_survives
        else f"the max-T-adjusted p exceeds {alpha}"
    )
    return header + f"No finding: {reason}.\n\n" + common


def render_report(
    table: pd.DataFrame,
    out_dir: str | Path,
    descriptors: Sequence[str] | None = None,
    outcome: str = "log10_affinity_pM",
    covariate: str = DEFAULT_COVARIATE,
    n_perm: int = 10000,
    seed: int = 0,
    n_boot: int = 10000,
    alpha: float = 0.05,
    plots: bool = True,
    title: str = "Frustration-affinity report",
) -> Path:
    """Write ``report.md``, ``report_table.csv`` and the figures; return the markdown path.

    The rendering rule is the point of the module: a descriptor for which
    :func:`headline_is_permitted` is False gets the covariate warning where its headline
    would have been, and the string :data:`HEADLINE_MARK` never appears in its section. The
    numbers are still printed — suppressing them would hide the confound rather than expose
    it — but they are printed under the warning, which is what a reader has to read past.

    ``table`` may be anything :func:`collect_analyses` accepts; it is passed through it, so
    a run directory and an in-memory frame are the same call here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = collect_analyses(table)
    names = list(descriptors) if descriptors is not None else default_descriptors(frame)
    if not names:
        raise ValueError("no descriptor columns found; pass descriptors= explicitly")

    results = report_table(
        frame, names, outcome, covariate, n_perm=n_perm, seed=seed, n_boot=n_boot
    )
    results.to_csv(out_dir / "report_table.csv", index=False)

    figures: list[Path] = []
    if plots:
        from atomfrust.report.plots import confound_figure

        figures.append(
            confound_figure(frame, outcome, covariate=covariate, out_path=out_dir / "confound.png")
        )

    withheld = [r["descriptor"] for _, r in results.iterrows() if not r["headline_permitted"]]
    lines = [
        f"# {title}",
        "",
        f"Outcome `{outcome}` · covariate held `{covariate}` · n = {int(results['n'].iloc[0])} "
        f"· {n_boot} bootstrap resamples · {n_perm} permutations · seed {seed}",
        "",
        "Every correlation is reported as a triple — raw, partial controlling the covariate, "
        "and the OLS coefficient with its VIF — and the multiplicity correction is a max-T "
        "permutation over the descriptor grid, not Benjamini-Hochberg: a reported best case "
        "is a maximum over the configuration grid, which BH does not price (plan §2.3).",
        "",
        f"Headlines withheld under the covariate guard: "
        f"{', '.join(withheld) if withheld else 'none'}.",
        "",
        "## Findings",
        "",
    ]
    lines.extend(_section(row, alpha) for _, row in results.iterrows())
    if figures:
        lines.append("## Figures\n")
        lines.extend(f"- `{p.name}`" for p in figures)
        lines.append("")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines))
    return path
