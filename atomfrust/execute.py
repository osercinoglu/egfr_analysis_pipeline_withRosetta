"""Work planning, sharding, and spawned execution.

One flat queue of ``(system_id, axis, decoy_id)`` units. Flat rather than nested because the
prototype's two-level scheme forced ``n_jobs_decoys=1`` inside every structure worker
(``run_pipeline.py:80``), so ``--mode all --n-jobs 32`` gave 32 structures at one core each
and ``--mode single`` gave one structure at 32 cores — neither able to fill a box when the
work did not happen to match its shape. A flat queue has no such coupling.

Three properties:

**Results do not depend on worker count.** Decoy *i* is seeded ``base_seed + i`` and units
are independent, so one worker, thirty-two workers and four shards on four machines produce
identical output. Asserted by test rather than assumed.

**Shards are disjoint by construction.** ``decoy_id % n_shards == shard`` needs no
coordination, and each shard writes only its own part files (see
:class:`~atomfrust.runstore.DecoyEnergyWriter`), so there is no lock anywhere.

**Resume is subtraction.** :func:`plan_units` takes the set already on disk and returns what
is left. Nothing is recomputed, and a larger ``n_decoys`` extends an ensemble rather than
being silently ignored as it was at ``run_pipeline.py:241``.

The executor is generic over the task callable so it can be exercised without PyRosetta;
step C2 supplies the real decoy generator.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

__all__ = [
    "WorkUnit",
    "ShardSpec",
    "plan_units",
    "default_worker_count",
    "run_serial",
    "run_parallel",
    "execute",
]


@dataclass(frozen=True, order=True)
class WorkUnit:
    """One decoy of one axis of one system — the atom of scheduling."""

    system_id: str
    axis: str
    decoy_id: int

    def seed(self, base_seed: int) -> int:
        """Decoy *i* is seeded ``base_seed + i``. This is the whole basis of reproducibility."""
        return base_seed + self.decoy_id


@dataclass(frozen=True)
class ShardSpec:
    """``k/M`` — shard ``k`` of ``M``, zero-based."""

    index: int = 0
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"shard count must be >= 1, got {self.count}")
        if not 0 <= self.index < self.count:
            raise ValueError(f"shard index {self.index} out of range for {self.count} shards")

    @classmethod
    def parse(cls, text: str | None) -> ShardSpec:
        if text is None or not text.strip():
            return cls()
        if "/" not in text:
            raise ValueError(f"expected 'k/M', got {text!r}")
        left, _, right = text.partition("/")
        try:
            return cls(int(left), int(right))
        except ValueError as exc:
            raise ValueError(f"expected 'k/M' with integers, got {text!r}") from exc

    def owns(self, unit: WorkUnit) -> bool:
        return unit.decoy_id % self.count == self.index

    def filter(self, units: Iterable[WorkUnit]) -> list[WorkUnit]:
        return [u for u in units if self.owns(u)]

    def __str__(self) -> str:
        return f"{self.index}/{self.count}"


def plan_units(
    system_ids: Sequence[str],
    n_decoys: int,
    axes: Sequence[str] = ("identity",),
    completed: dict[tuple[str, str], set[int]] | None = None,
    shard: ShardSpec | None = None,
) -> list[WorkUnit]:
    """Everything still to do, in a deterministic order.

    ``completed`` maps ``(system_id, axis)`` to decoy ids already stored. Passing it turns a
    re-run into a resume; omitting it plans the full ensemble.
    """
    completed = completed or {}
    shard = shard or ShardSpec()
    units: list[WorkUnit] = []
    for system_id in system_ids:
        for axis in axes:
            done = completed.get((system_id, axis), set())
            for decoy_id in range(n_decoys):
                if decoy_id in done:
                    continue
                unit = WorkUnit(system_id, axis, decoy_id)
                if shard.owns(unit):
                    units.append(unit)
    return units


def default_worker_count() -> int:
    return max(1, os.cpu_count() or 1)


# --------------------------------------------------------------------- runners

_TASK: Callable[[WorkUnit], Any] | None = None


def _pool_initializer(
    initializer: Callable[..., Any] | None,
    initargs: tuple,
    task_factory: Callable[..., Callable[[WorkUnit], Any]],
    task_args: tuple,
) -> None:
    """Build per-worker state once.

    A Rosetta pose is never pickled: the worker rebuilds it from
    ``(receptor.pdb, params)``. PyRosetta is not fork-safe, so the pool always uses
    ``spawn`` and every worker calls ``pyrosetta.init`` itself.
    """
    global _TASK
    if initializer is not None:
        initializer(*initargs)
    _TASK = task_factory(*task_args)


def _pool_call(unit: WorkUnit) -> Any:
    if _TASK is None:  # pragma: no cover - defensive
        raise RuntimeError("worker was not initialised")
    return _TASK(unit)


def run_serial(
    units: Sequence[WorkUnit], task: Callable[[WorkUnit], Any]
) -> Iterator[tuple[WorkUnit, Any]]:
    """Run in this process. The reference path that parallel results must match."""
    for unit in units:
        yield unit, task(unit)


def run_parallel(
    units: Sequence[WorkUnit],
    task_factory: Callable[..., Callable[[WorkUnit], Any]],
    task_args: tuple = (),
    workers: int = 1,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple = (),
    chunksize: int = 1,
) -> Iterator[tuple[WorkUnit, Any]]:
    """Run on a ``spawn`` pool, yielding results as they complete.

    ``task_factory`` is called once per worker rather than the task being pickled per unit,
    so expensive per-worker state (a pose, a score function) is built once.
    """
    if not units:
        return
    worker_count = max(1, min(workers, len(units)))
    if worker_count == 1:
        yield from run_serial(units, task_factory(*task_args))
        return

    context = mp.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=_pool_initializer,
        initargs=(initializer, initargs, task_factory, task_args),
    ) as pool:
        for unit, result in zip(
            units, pool.imap(_pool_call, units, chunksize=chunksize)
        ):
            yield unit, result


def execute(
    units: Sequence[WorkUnit],
    task_factory: Callable[..., Callable[[WorkUnit], Any]],
    task_args: tuple = (),
    workers: int = 1,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple = (),
    on_result: Callable[[WorkUnit, Any], None] | None = None,
) -> int:
    """Run every unit, streaming results to ``on_result``. Returns the number completed.

    Results are streamed rather than collected so a writer can flush incrementally: an
    interrupted run keeps everything already flushed.
    """
    completed = 0
    for unit, result in run_parallel(
        units,
        task_factory,
        task_args=task_args,
        workers=workers,
        initializer=initializer,
        initargs=initargs,
    ):
        if on_result is not None:
            on_result(unit, result)
        completed += 1
    return completed
