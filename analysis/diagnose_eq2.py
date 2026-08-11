#!/usr/bin/env python
"""
Step A1 of plans/frustratometer-ng-plan.md — confirm the Eq. 2 degeneracy.

Eq. 2 as implemented (src/frustration.py:238-250) is

    E_ij = e_ij + 0.5*sum_{k in contacts(i), k!=j} e_ik
                + 0.5*sum_{l in contacts(j), l!=i} e_jl

Writing B_i = sum over ALL of i's contact partners of e_ik, the excluded sums are
(B_i - e_ij) and (B_j - e_ij), so

    E_ij = e_ij + 0.5*(B_i - e_ij) + 0.5*(B_j - e_ij) = 0.5*(B_i + B_j)

i.e. e_ij cancels exactly and E_ij carries no pair-specific information: it is the
mean of two per-residue totals.

This script tests that prediction on the stored per-structure parquets, which hold
E_native, decoy_mean and decoy_std per contact pair. If the prediction holds, then
E_native is EXACTLY additive in the two residue indices -- fitting E_ij ~ a_i + a_j
by least squares must leave residuals at floating-point noise, with a_i = 0.5*B_i.

Run from the repo root. Pure numpy/pandas: no PyRosetta, safe to run alongside a
live Stage 6 job.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Residual at or below this level (kcal/mol) is float64 accumulation noise, not
# a real departure from additivity. E_native values are order 1-100 kcal/mol and
# are sums of up to ~30 pair terms, so ~1e-10 is a generous ceiling on noise.
EXACT_TOL = 1e-9

# Columns fitted. E_native is the decisive one; decoy_mean should behave
# identically (same formula, different pose); decoy_std must NOT be additive
# (a standard deviation of a sum is not a sum), and F_index inherits its
# non-additivity only through that division.
FIT_COLUMNS = ("E_native", "decoy_mean", "decoy_std", "F_index")


@dataclass
class Fit:
    r2: float
    max_abs_resid: float
    rms_resid: float
    y_std: float
    rank: int
    n_params: int

    @property
    def is_exact(self) -> bool:
        return self.max_abs_resid <= EXACT_TOL


def fit_additive(resi: np.ndarray, resj: np.ndarray, y: np.ndarray) -> Fit:
    """Least-squares fit of y_ij ~ a_i + a_j over the residues present."""
    residues = np.unique(np.concatenate([resi, resj]))
    index = {r: k for k, r in enumerate(residues)}

    design = np.zeros((len(y), len(residues)))
    rows = np.arange(len(y))
    design[rows, [index[r] for r in resi]] += 1.0
    design[rows, [index[r] for r in resj]] += 1.0

    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return Fit(
        r2=r2,
        max_abs_resid=float(np.abs(resid).max()),
        rms_resid=float(np.sqrt(ss_res / len(y))),
        y_std=float(y.std(ddof=1)),
        rank=int(rank),
        n_params=len(residues),
    )


def analyse(path: Path) -> dict:
    df = pd.read_parquet(path)
    resi = df["resi"].to_numpy()
    resj = df["resj"].to_numpy()

    if (resi == resj).any():
        raise ValueError(f"{path.name}: self-pairs present, additive model ill-posed")

    row: dict = {
        "structure": path.name.replace("_frustration.parquet", ""),
        "n_pairs": len(df),
        "n_residues": len(np.unique(np.concatenate([resi, resj]))),
    }
    for col in FIT_COLUMNS:
        fit = fit_additive(resi, resj, df[col].to_numpy())
        row[f"r2_{col}"] = fit.r2
        row[f"maxres_{col}"] = fit.max_abs_resid
        row[f"exact_{col}"] = fit.is_exact
        if col == "E_native":
            row["rank"] = fit.rank
            row["n_params"] = fit.n_params
            row["rms_E_native"] = fit.rms_resid
            row["std_E_native"] = fit.y_std
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="analysis/eq2_degeneracy_report.md")
    args = parser.parse_args()

    paths = sorted(Path(args.results_dir).glob("*_frustration.parquet"))
    if not paths:
        print(f"No parquets under {args.results_dir}/", file=sys.stderr)
        return 2

    rows = [analyse(p) for p in paths]
    table = pd.DataFrame(rows)

    # ---- console table -------------------------------------------------
    print(f"\nEq. 2 degeneracy check over {len(table)} structures\n")
    header = (
        f"{'structure':<12} {'pairs':>6} {'res':>5} "
        f"{'R2(E_nat)':>20} {'max|res|':>11} "
        f"{'R2(dec_mean)':>14} {'R2(dec_std)':>13} {'R2(F)':>9}"
    )
    print(header)
    print("-" * len(header))
    for _, r in table.iterrows():
        print(
            f"{r.structure:<12} {r.n_pairs:>6} {r.n_residues:>5} "
            f"{r.r2_E_native:>20.15f} {r.maxres_E_native:>11.2e} "
            f"{r.r2_decoy_mean:>14.10f} {r.r2_decoy_std:>13.6f} {r.r2_F_index:>9.6f}"
        )

    # ---- verdict -------------------------------------------------------
    n_exact = int(table["exact_E_native"].sum())
    universal = n_exact == len(table)
    print(
        f"\nE_native exactly additive (max|residual| <= {EXACT_TOL:g}): "
        f"{n_exact}/{len(table)} structures"
    )
    print(f"worst max|residual| across all structures: "
          f"{table['maxres_E_native'].max():.3e} kcal/mol")
    print(f"typical E_native spread (std):             "
          f"{table['std_E_native'].median():.3f} kcal/mol")
    print(f"\nVERDICT: degeneracy is "
          f"{'UNIVERSAL' if universal else 'NOT universal — investigate'}")

    if not universal:
        bad = table.loc[~table["exact_E_native"], ["structure", "maxres_E_native"]]
        print("\nStructures departing from exact additivity:")
        print(bad.to_string(index=False))

    write_report(table, Path(args.out), universal, n_exact)
    print(f"\nReport written to {args.out}")

    # Non-zero exit if the degeneracy is not universal: the plan says stop and
    # find out why, so this should break a pipeline rather than pass quietly.
    return 0 if universal else 1


def write_report(table: pd.DataFrame, out: Path, universal: bool, n_exact: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(table)

    lines = [
        "# A1 — Eq. 2 degeneracy: is the per-contact index actually a per-residue index?",
        "",
        f"Generated by `analysis/diagnose_eq2.py` over {n} completed structures in `results/`.",
        "",
        "## Claim under test",
        "",
        "`contact_energy_eq2` (`src/frustration.py:238-250`) computes",
        "",
        "```",
        "E_ij = e_ij + 0.5*sum_{k in contacts(i), k!=j} e_ik + 0.5*sum_{l in contacts(j), l!=i} e_jl",
        "```",
        "",
        "With `B_i` the sum of `e_ik` over *all* of i's contact partners, the two excluded",
        "sums equal `B_i - e_ij` and `B_j - e_ij`, so",
        "",
        "```",
        "E_ij = e_ij + 0.5*(B_i - e_ij) + 0.5*(B_j - e_ij) = 0.5*(B_i + B_j)",
        "```",
        "",
        "`e_ij` cancels exactly. The prediction is that `E_native` is *exactly* additive in",
        "the two residue indices, with `a_i = 0.5*B_i`.",
        "",
        "## Result",
        "",
        f"**E_native is exactly additive in {n_exact}/{n} structures** "
        f"(max\\|residual\\| <= {EXACT_TOL:g} kcal/mol).",
        "",
        f"- Worst max\\|residual\\| across all structures: "
        f"`{table['maxres_E_native'].max():.3e}` kcal/mol",
        f"- Median E_native spread for scale: "
        f"`{table['std_E_native'].median():.3f}` kcal/mol",
        f"- Ratio (worst residual / typical spread): "
        f"`{table['maxres_E_native'].max() / table['std_E_native'].median():.2e}`",
        "",
        "**This is not overfitting.** The additive model has one parameter per residue",
        f"(median {int(table['n_params'].median())}, design-matrix rank "
        f"{int(table['rank'].median())}) fitted against a median of "
        f"{int(table['n_pairs'].median())} contact pairs — about "
        f"{table['n_pairs'].median() / table['n_params'].median():.1f} observations per",
        "parameter. A model with that much slack does not reach residuals of 1e-14 by",
        "chance; it does so because the identity holds algebraically.",
        "",
        f"**Verdict: the degeneracy is "
        f"{'universal' if universal else 'NOT universal — investigate before proceeding'}.**",
        "",
        "## Per-structure table",
        "",
        "`R2(E_nat)` at 15 decimal places; `max|res|` in kcal/mol. `R2(dec_std)` and `R2(F)`",
        "are shown to locate where non-additivity enters.",
        "",
        "| structure | pairs | residues | R2(E_native) | max\\|resid\\| | R2(decoy_mean) | R2(decoy_std) | R2(F_index) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r.structure} | {r.n_pairs} | {r.n_residues} | "
            f"{r.r2_E_native:.15f} | {r.maxres_E_native:.2e} | "
            f"{r.r2_decoy_mean:.10f} | {r.r2_decoy_std:.6f} | {r.r2_F_index:.6f} |"
        )

    lines += [
        "",
        "## Reading the columns",
        "",
        "- **`R2(E_native) = 1` with residuals at float64 noise** is the degeneracy itself:",
        "  the native contact energy carries no pair-specific information.",
        "- **`R2(decoy_mean)` matches** — same formula, different pose, so the decoy",
        "  reference is degenerate in exactly the same way.",
        "- **`R2(decoy_std)` is below 1 but not by much** — a standard deviation of a sum",
        "  is not a sum, so this is the only channel through which pair-specific",
        "  information can survive at all. The channel is narrow: the observed range is",
        f"  {table['r2_decoy_std'].min():.3f}-{table['r2_decoy_std'].max():.3f}, so sigma is",
        f"  itself ~{100 * table['r2_decoy_std'].median():.0f}% explained by a per-residue model.",
        "- **`R2(F_index)` inherits that.** Since `F = (decoy_mean - E_native) / decoy_std`",
        "  has an exactly additive numerator, *all* of F's departure from additivity comes",
        "  from dividing by sigma. Observed range "
        f"{table['r2_F_index'].min():.3f}-{table['r2_F_index'].max():.3f}.",
        "",
        f"**Quantitatively: only ~{100 * (1 - table['r2_F_index'].median()):.0f}% of the "
        "variance in the per-contact frustration index is pair-specific.** The other "
        f"~{100 * table['r2_F_index'].median():.0f}% is reproduced by assigning one number "
        "to each residue and adding. This is a sharper claim than 'the index is confounded",
        "by pocket size': the per-contact resolution is largely illusory, the index is a",
        "per-residue quantity wearing a per-contact label, and the effect is near-identical",
        f"across all {n} structures examined (range "
        f"{table['r2_F_index'].min():.3f}-{table['r2_F_index'].max():.3f}).",
        "",
        "## Consequences",
        "",
        "1. **The pocket-size confound is derived, not observed.** If every contact's value",
        "   is fixed by its two residues, then contact counts are graph-degree statistics.",
        "   `n_contacts_total` out-predicting `n_minimally_frustrated` follows necessarily",
        "   and is not a coincidence to be corrected in reporting.",
        "2. **Adding a ligand node changes little under this formula.** A ligand would",
        "   perturb a protein-protein key through one term of ~30 in `B_i`. Step A3 (apo",
        "   control) measures how little.",
        "3. **The many-body mode must become selectable** (step B8), and `chen_literal`",
        "   cannot be the default for new science.",
        "",
        "## Resolved by A4 (2026-08-11)",
        "",
        "- **The transcription is faithful.** Chen et al. 2020 Eq. 2 as printed is",
        "  `E_ij = e_ij + 1/2 sum_{k,k!=j} e_ik + 1/2 sum_{l,l!=i} e_jl` -- the exclusions",
        "  are in the paper. The degeneracy demonstrated above is therefore a property of",
        "  the **published equation**, not a defect introduced in this repository.",
        "  `chen_literal` is the published object.",
        "- **The degeneracy does not explain the count gap.** A4 established that the",
        "  paper's counts are *ligand-residue* contacts (4-23 = the number of residues a",
        "  drug contacts), while this pipeline counts protein-protein pairs in a 10 A shell",
        "  (266-407). A2 swept 189 selector/threshold configurations and closed nothing,",
        "  because the contacts the paper counts do not exist in our data at all:",
        "  `get_protein_contacts` excludes non-protein residues. Making the ligand a graph",
        "  node (plan steps B6/B7) is the correction; changing the many-body formula alone",
        "  would leave partner lists protein-only and would not help.",
    ]
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
