#!/usr/bin/env python
"""C6 / C8 ablation — symmetric native treatment and Monte-Carlo relaxation.

Three conditions, chosen so two of them share a decoy ensemble:

  A  native_repack=False, relax=min   the prototype: crystal native, one chi-only MinMover
  B  native_repack=True,  relax=min   C6 only — the paper repacks the native too (step A4)
  C  native_repack=True,  relax=mc    C6 + C8 — the paper's "short Monte-Carlo relaxation"

A and B differ **only** in the native reference, so they reuse the same decoys. That is not
a shortcut: it isolates C6's effect exactly, with zero sampling noise between the two.

Reports the frustration class fractions each condition produces. The Z-score is computed
inline here because the analysis modules (plan steps D1/D2) do not exist yet; the arithmetic
is the same and this file is a diagnostic, not a pipeline stage.

Run from the repo root:
    OMP_NUM_THREADS=1 nice -n 19 python analysis/ablate_native_and_relax.py --decoys 15
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

MINIMAL, HIGHLY = 0.78, -1.0
_STATE: dict = {}


def build_context(pdb_stem: str, params_name: str):
    from atomfrust.decoys.base import DecoyContext
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex
    from atomfrust.regions import resolve_regions
    from atomfrust.settings import Settings
    from atomfrust.spec import LigandSpec, SystemSpec

    spec = SystemSpec.from_pdb(
        Path("data/processed") / f"{pdb_stem}_clean.pdb", system_id=pdb_stem
    )
    spec = spec.model_copy(
        update={
            "ligands": (
                LigandSpec(
                    selector=spec.ligands[0].selector,
                    params=Path("data/ligands/params") / f"{params_name}.params",
                ),
            )
        }
    )
    loaded = load_complex(spec)
    settings = Settings()
    _, pairs = build_graph(loaded.nodes, loaded.geometry, settings)
    regions = resolve_regions(loaded.nodes, loaded.geometry)
    return DecoyContext(
        pose=loaded.pose,
        nodes=loaded.nodes,
        pairs=pairs,
        regions=regions,
        settings=settings,
    )


def _init(pdb_stem: str, params_name: str, relax: str) -> None:
    from atomfrust.decoys.identity import IdentityDecoyGenerator

    context = build_context(pdb_stem, params_name)
    _STATE["context"] = context
    _STATE["generator"] = IdentityDecoyGenerator(
        context,
        identity="composition",
        placement="inplace",
        mutation="sequential",
        relax=relax,
        mc_cycles=5,
        base_seed=42,
    )


def _one_decoy(decoy_id: int):
    from atomfrust.energy import effective_energy, many_body_energies

    context, generator = _STATE["context"], _STATE["generator"]
    result = generator.generate(decoy_id)
    e = effective_energy(result.e_direct, result.e_fa_rep, exclude_fa_rep=True)
    E = many_body_energies(
        context.pairs.node_i.to_numpy(), context.pairs.node_j.to_numpy(), e, "chen_literal"
    )
    return E, result.index_row["wall_s"]


def _native_energy(pdb_stem: str, params_name: str, relax: str, native_repack: bool):
    from atomfrust.decoys.base import extract_energies
    from atomfrust.decoys.identity import IdentityDecoyGenerator
    from atomfrust.energy import effective_energy, many_body_energies

    context = build_context(pdb_stem, params_name)
    generator = IdentityDecoyGenerator(
        context,
        identity="composition",
        placement="inplace",
        mutation="sequential",
        relax=relax,
        mc_cycles=5,
        base_seed=42,
        native_repack=native_repack,
    )
    _, e_direct, e_fa_rep = extract_energies(
        generator.prepare_native(), context.pairs, context.score_function
    )
    e = effective_energy(e_direct, e_fa_rep, exclude_fa_rep=True)
    return many_body_energies(
        context.pairs.node_i.to_numpy(), context.pairs.node_j.to_numpy(), e, "chen_literal"
    ), context


def classify(native: np.ndarray, decoys: np.ndarray) -> dict:
    """Eq. 1 with the published thresholds. Inlined until D1/D2 exist."""
    mean = decoys.mean(axis=0)
    sigma = decoys.std(axis=0, ddof=1)
    F = np.where(sigma < 1e-9, 0.0, (mean - native) / np.where(sigma < 1e-9, 1.0, sigma))
    total = len(F)
    return {
        "frac_minimal": float((F > MINIMAL).sum() / total),
        "frac_highly": float((F < HIGHLY).sum() / total),
        "frac_neutral": float(((F <= MINIMAL) & (F >= HIGHLY)).sum() / total),
        "mean_F": float(F.mean()),
        "median_sigma": float(np.median(sigma)),
    }


def run_structure(pdb_stem: str, params_name: str, n_decoys: int, workers: int) -> list[dict]:
    rows = []
    context_pairs = None
    ensembles = {}
    for relax in ("min", "mc"):
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init, initargs=(pdb_stem, params_name, relax)) as pool:
            produced = pool.map(_one_decoy, range(n_decoys))
        ensembles[relax] = (
            np.stack([E for E, _ in produced]),
            float(np.mean([w for _, w in produced])),
        )

    for label, relax, native_repack in (
        ("A prototype", "min", False),
        ("B +C6 native repack", "min", True),
        ("C +C6+C8 mc", "mc", True),
    ):
        native, context = _native_energy(pdb_stem, params_name, relax, native_repack)
        context_pairs = context.pairs
        decoys, wall = ensembles[relax]
        ligand = (
            (context_pairs.kind_i != "protein") | (context_pairs.kind_j != "protein")
        ).to_numpy()
        row = {"structure": pdb_stem, "condition": label, "wall_s_per_decoy": wall}
        row.update(classify(native, decoys))
        row.update(
            {f"lig_{k}": v for k, v in classify(native[ligand], decoys[:, ligand]).items()}
        )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoys", type=int, default=15)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default="analysis/ablation_native_relax.md")
    args = parser.parse_args()

    targets = [("5GMP", "F62"), ("1XKK", "FMM"), ("3POZ", "03P")]
    rows: list[dict] = []
    for pdb_stem, params_name in targets:
        if not (Path("data/processed") / f"{pdb_stem}_clean.pdb").exists():
            print(f"skip {pdb_stem}: no processed structure")
            continue
        print(f"running {pdb_stem} ({args.decoys} decoys x 2 relax modes)...", flush=True)
        rows.extend(run_structure(pdb_stem, params_name, args.decoys, args.workers))

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# C6 / C8 ablation — native treatment and relaxation\n\n"
        f"{args.decoys} decoys per condition, `identity=composition, placement=inplace`, "
        "`chen_literal`, `exclude_fa_rep`, thresholds 0.78 / -1.0.\n\n"
        "Conditions A and B share a decoy ensemble and differ only in the native reference, "
        "so C6's effect is isolated with zero sampling noise between them.\n\n"
        "`frac_*` are over all superset pairs; `lig_frac_*` over ligand-incident pairs only.\n\n"
        + table.to_markdown(index=False, floatfmt=".4f")
        + "\n"
    )
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
