"""The decoy generator protocol and shared energy extraction.

One randomisation yields energies for **every** contact at once: a decoy is scored once and
read off for the whole frozen pair set. Chen et al. describe this as the efficiency that
makes the atomistic method tractable, but note the tension recorded in the plan — the paper
also says "1000 decoys **per contact**", so treating one ensemble as serving all contacts is
a departure worth testing rather than a settled optimisation. Isolating extraction here means
the alternative can be tried by swapping the generator, not by rewriting the engine.

Contact membership is fixed on the native pose and never recomputed per decoy: the Z-score
must be taken over a fixed index set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from atomfrust.energy import EnergyEvaluator
from atomfrust.graph import Node
from atomfrust.regions import ResolvedRegions
from atomfrust.settings import Settings

__all__ = [
    "DecoyContext",
    "DecoyResult",
    "DecoyGenerator",
    "NullGenerator",
    "extract_energies",
]


@dataclass
class DecoyContext:
    """Per-worker state, built once and reused for every decoy that worker handles.

    Holds the native pose, the frozen pair table, and the resolved regions. Building this is
    the expensive part (pose load, graph construction), which is why the executor calls a
    task *factory* per worker rather than pickling a task per unit.
    """

    pose: Any
    nodes: list[Node]
    pairs: pd.DataFrame
    regions: ResolvedRegions
    settings: Settings = field(default_factory=Settings)

    @property
    def pair_ids(self) -> np.ndarray:
        return self.pairs["pair_id"].to_numpy(dtype=np.int32)

    @property
    def score_function(self) -> str:
        return self.settings.energy.score_function


@dataclass
class DecoyResult:
    """One decoy: its energies and the row describing how it was made."""

    decoy_id: int
    pair_id: np.ndarray
    e_direct: np.ndarray
    e_fa_rep: np.ndarray
    index_row: dict[str, Any]
    pose: Any = None


@runtime_checkable
class DecoyGenerator(Protocol):
    """Anything that can produce a decoy from a context."""

    axis: str

    def generate(self, decoy_id: int) -> DecoyResult:  # pragma: no cover - protocol
        ...


def extract_energies(
    pose: Any, pairs: pd.DataFrame, score_function: str = "ref2015"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a pose once and read every pair off the energy graph.

    Returns ``(pair_id, e_direct, e_fa_rep)``. ``e_fa_rep`` is returned unsubtracted so the
    ``exclude_fa_rep`` choice stays an analysis-time switch.
    """
    evaluator = EnergyEvaluator(pose, score_function)
    frame = evaluator.pairs(pairs)
    return (
        frame["pair_id"].to_numpy(dtype=np.int32),
        frame["e_direct"].to_numpy(dtype=np.float32),
        frame["e_fa_rep"].to_numpy(dtype=np.float32),
    )


class NullGenerator:
    """Returns the native pose untouched.

    Not a toy: it is the tightest available end-to-end check on the extraction path, since
    its "decoy" energies must equal the stored native energies exactly. Also the natural
    baseline for asking what a decoy protocol actually changes.
    """

    axis = "null"

    def __init__(self, context: DecoyContext) -> None:
        self.context = context

    def generate(self, decoy_id: int) -> DecoyResult:
        pair_id, e_direct, e_fa_rep = extract_energies(
            self.context.pose, self.context.pairs, self.context.score_function
        )
        return DecoyResult(
            decoy_id=decoy_id,
            pair_id=pair_id,
            e_direct=e_direct,
            e_fa_rep=e_fa_rep,
            index_row={
                "decoy_id": decoy_id,
                "axis": self.axis,
                "generator": "null",
                "n_mutated": 0,
                "backbone_rmsd": 0.0,
                "status": "ok",
            },
        )
