"""Binding-site mutation control — plan step G8, methods success criterion S4.5.

**Why this exists.** S4.5 asks for pocket mutations as a *positive* test: a physically
grounded measure must respond when the pocket is mutated, whereas a model that has
memorised its training complexes does not. Stated as a criterion — *"on ≥10 systems with
pocket mutations known to abolish binding, the frustration index changes significantly
(paired test, p < 0.05 after correction), while the change for at least one co-folding
comparator is not significant"* — it is the cheapest available demonstration that the
measure reflects recognition rather than recall.

Mechanically that needs exactly one capability the pipeline lacked: **apply a point
mutation to the receptor named in a spec and re-run.** Everything else is already in place —
the graph, the energies, the decoy ensemble and the paired statistics
(:func:`atomfrust.metrics.paired_delta`) do not care whether a residue is the crystal one.

Three decisions worth stating, because each is a place this could go quietly wrong:

* **The mutation lives in the spec, not in a runtime flag.** ``regeneration_key`` digests
  the spec YAML (``cli/generate_decoys.py:369-377`` builds ``spec_sha256``, which
  ``settings.regeneration_key`` folds into ``inputs``), so a mutant cannot silently reuse a
  stored wild-type ensemble. A mutation expressed as a runtime flag would have been
  invisible to that key, which is the failure this arrangement forecloses.
* **The mutant keeps its wild-type ``system_id``.** That is what makes the two runs
  *pairable*: S4.5 is a paired test, and the pair is (wild type, mutant) of the same system.
  The two are told apart by ``regeneration_key`` and by ``LoadedComplex.mutations``, never by
  the id.
* **The substitution is applied at pose load, before nodes are built.** So ``Node.resname``
  and ``Node.name1`` report the mutant, the pair table is built from mutant geometry, and no
  downstream stage needs to know a mutation happened.

The new side chain is placed at its ideal rotamer by ``MutateResidue`` and is *not* relaxed
here. Relaxation is the generation stage's business, under whatever protocol the run
already specifies (``decoys.native_repack`` repacks the native pose, and every decoy is
repacked by construction). Doing it here would apply a second, unrecorded protocol to one
residue.

No PyRosetta at import time — the ``unit`` tier must stay runnable without it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from atomfrust.spec import Mutation

__all__ = [
    "AppliedMutation",
    "PairedComparison",
    "parse_mutation",
    "apply_mutations",
    "paired_comparison",
]


@dataclass(frozen=True)
class AppliedMutation:
    """What a mutation actually did to a pose — recorded, never inferred.

    ``from_resname`` is read off the pose *before* the substitution, so a spec that asks for
    a residue the structure already carries is visible as ``changed == False`` rather than
    passing for a real perturbation.
    """

    chain: str
    resseq: int
    icode: str
    pose_resnum: int
    from_resname: str
    to_resname: str

    @property
    def changed(self) -> bool:
        return self.from_resname != self.to_resname

    def __str__(self) -> str:
        return (
            f"{self.from_resname}{self.chain}:{self.resseq}{self.icode}"
            f"{self.to_resname} (pose {self.pose_resnum})"
        )


def parse_mutation(text: str | Mutation) -> Mutation:
    """``"A:790:MET"`` → :class:`~atomfrust.spec.Mutation`. One- or three-letter targets.

    A :class:`~atomfrust.spec.Mutation` passes through unchanged, so call sites can accept
    either form without branching.
    """
    if isinstance(text, Mutation):
        return text
    return Mutation.parse(text)


def _pdb_index(pose: Any, info: Any) -> dict[tuple[str, int, str], int]:
    """``(chain, resseq, icode) -> pose resnum``. The same key the node list uses."""
    index: dict[tuple[str, int, str], int] = {}
    for resnum in range(1, pose.total_residue() + 1):
        key = (
            info.chain(resnum).strip(),
            int(info.number(resnum)),
            info.icode(resnum).strip(),
        )
        index.setdefault(key, resnum)
    return index


def apply_mutations(
    pose: Any,
    mutations: Iterable[Mutation | str],
    pdb_info: Any = None,
) -> tuple[AppliedMutation, ...]:
    """Apply point substitutions to ``pose`` in place; return what was applied.

    Positions are addressed by ``(chain, resseq, icode)`` from ``PDBInfo`` — never by pose
    index, which depends on what else loaded — and every position is resolved against the
    *pre-mutation* pose, so the substitutions are independent of one another and of their
    order.

    A position that does not exist, or that is not a protein residue, raises. Silence there
    would produce the worst possible outcome for S4.5: a "mutant" run that is the wild type
    with a different regeneration key, indistinguishable in the results from a measure that
    fails to respond.
    """
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    wanted = tuple(parse_mutation(m) for m in mutations)
    if not wanted:
        return ()

    info = pdb_info if pdb_info is not None else pose.pdb_info()
    if info is None:
        raise ValueError(
            "pose has no PDBInfo; a mutation is addressed by PDB position, so there is "
            "nothing to resolve it against"
        )

    index = _pdb_index(pose, info)
    applied: list[AppliedMutation] = []
    for mut in wanted:
        resnum = index.get(mut.key())
        if resnum is None:
            chain_positions = sorted(r for (c, r, _) in index if c == mut.chain)
            hint = (
                f"; chain {mut.chain} spans {chain_positions[0]}-{chain_positions[-1]}"
                if chain_positions
                else f"; chains present: {', '.join(sorted({c for c, _, _ in index}))}"
            )
            raise KeyError(f"mutation target {mut} is not in the pose{hint}")
        residue = pose.residue(resnum)
        if not residue.is_protein():
            raise ValueError(
                f"mutation target {mut} is {residue.name3().strip()!r}, not a protein "
                "residue; only amino acids can be substituted"
            )
        before = residue.name3().strip()
        if before != mut.to:
            MutateResidue(resnum, mut.to).apply(pose)
        applied.append(
            AppliedMutation(
                chain=mut.chain,
                resseq=mut.resseq,
                icode=mut.icode.strip(),
                pose_resnum=resnum,
                from_resname=before,
                to_resname=mut.to,
            )
        )
    return tuple(applied)


# --------------------------------------------------------------- paired comparison


#: Key columns tried, in order, when the caller does not name one. Finest grain first: a
#: per-contact table carries both ``pair_id`` and (often) ``system_id``, and there the pair
#: is the row and the system is the *group*. A per-descriptor summary has only ``system_id``,
#: so it falls through to it.
_KEY_CANDIDATES = ("pair_id", "node_id", "system_id")


@dataclass(frozen=True)
class PairedComparison:
    """Wild-type and mutant values aligned on a shared key, plus their difference.

    ``frame`` has one row per matched key with columns ``key``, ``group``, ``wild_type``,
    ``mutant`` and ``delta`` (mutant − wild type: the *response* to the mutation, signed so
    that "the measure moved" is a non-zero mean). Feed it to
    :func:`atomfrust.metrics.paired_delta` directly, or call :meth:`estimate`.
    """

    frame: pd.DataFrame
    metric: str
    unmatched: tuple[Any, ...] = ()

    @property
    def keys(self) -> np.ndarray:
        return self.frame["key"].to_numpy()

    @property
    def groups(self) -> np.ndarray:
        return self.frame["group"].to_numpy()

    @property
    def wild_type(self) -> np.ndarray:
        return self.frame["wild_type"].to_numpy(dtype=float)

    @property
    def mutant(self) -> np.ndarray:
        return self.frame["mutant"].to_numpy(dtype=float)

    @property
    def delta(self) -> np.ndarray:
        return self.frame["delta"].to_numpy(dtype=float)

    def __len__(self) -> int:
        return len(self.frame)

    def estimate(self, n_boot: int = 10000, seed: int = 0):
        """The S4.5 paired test: mean per-group ``mutant − wild_type`` with a sign-flip p.

        Thin by design — the statistic belongs to :mod:`atomfrust.metrics`, and this is only
        the call that puts the right three arrays into it.
        """
        from atomfrust.metrics import paired_delta

        return paired_delta(
            self.mutant, self.wild_type, groups=self.groups, n_boot=n_boot, seed=seed
        )


def _as_frame(summary: Any, metric: str, key: str, side: str) -> pd.DataFrame:
    """Coerce a summary to a frame carrying ``key`` and ``metric``.

    Accepts what the analysis layer actually produces: a
    :func:`~atomfrust.analyze.aggregate.summarize_many` frame, a mapping of
    ``system_id -> summarize(...) dict``, or a single flat :func:`summarize` dict.
    """
    if isinstance(summary, pd.DataFrame):
        frame = summary
    elif isinstance(summary, Mapping):
        values = list(summary.values())
        if values and all(isinstance(v, Mapping) for v in values):
            frame = pd.DataFrame(
                [{key: k, **dict(v)} for k, v in summary.items()]
            )
        else:
            row = dict(summary)
            row.setdefault(key, row.get("system_id", "system"))
            frame = pd.DataFrame([row])
    elif isinstance(summary, Sequence):
        frame = pd.DataFrame(list(summary))
    else:
        raise TypeError(
            f"{side} summary is {type(summary).__name__}; expected a DataFrame, a mapping "
            "of id -> summary, or a summary dict"
        )

    for column in (key, metric):
        if column not in frame.columns:
            raise KeyError(
                f"{side} summary has no column {column!r}; columns: "
                f"{', '.join(map(str, frame.columns))}"
            )
    return frame


def _resolve_key(wild: Any, mutant: Any, key: str | None) -> str:
    if key is not None:
        return key
    for candidate in _KEY_CANDIDATES:
        if _has_column(wild, candidate) and _has_column(mutant, candidate):
            return candidate
    return "system_id"


def _has_column(summary: Any, column: str) -> bool:
    if isinstance(summary, pd.DataFrame):
        return column in summary.columns
    if isinstance(summary, Mapping):
        values = list(summary.values())
        if values and all(isinstance(v, Mapping) for v in values):
            return all(column in v or column == "system_id" for v in values)
        return column in summary
    return False


def paired_comparison(
    wild_type_summary: Any,
    mutant_summary: Any,
    metric: str,
    *,
    key: str | None = None,
    group: str | None = None,
) -> PairedComparison:
    """Align a wild-type and a mutant summary and return their paired difference.

    ``metric`` is a column name — a descriptor key such as
    ``desc__frac_minimal__zscore__default`` for the per-descriptor form, or an index column
    such as ``F`` for the per-contact form. The two summaries are inner-joined on ``key``,
    which defaults to the finest-grained of ``pair_id``, ``node_id``, ``system_id`` present
    in both. For the per-descriptor form that is ``system_id`` — and the mutant keeps its
    wild-type ``system_id``, which is exactly what makes the pair a pair.

    ``group`` is the unit :func:`~atomfrust.metrics.paired_delta` collapses to before
    testing, and it matters. Across systems it is ``system_id`` (auto-detected), so each
    system weighs the same however many contacts it contributed — the rule of R34. Within
    one system, where every row is a contact of the same system, it defaults to the key
    itself, which makes the test a plain sign-flip over contacts; that is a within-system
    statement and must not be reported as if it were evidence across systems.

    Keys present on one side only are dropped and listed in ``unmatched`` rather than
    filled: a contact that exists in the mutant and not the wild type has no partner, and
    imputing one would invent the very signal being measured.
    """
    key = _resolve_key(wild_type_summary, mutant_summary, key)
    wild = _as_frame(wild_type_summary, metric, key, "wild-type")
    mut = _as_frame(mutant_summary, metric, key, "mutant")

    group_column = group
    if group_column is None:
        group_column = "system_id" if ("system_id" in wild.columns and key != "system_id") else key
    if group_column not in wild.columns:
        raise KeyError(
            f"wild-type summary has no group column {group_column!r}; columns: "
            f"{', '.join(map(str, wild.columns))}"
        )

    # Built column by column rather than by ``rename``: when the key *is* the group column
    # a rename mapping collapses to one entry and silently loses the key.
    left = pd.DataFrame(
        {
            "key": wild[key].to_numpy(),
            "group": wild[group_column].to_numpy(),
            "wild_type": wild[metric].to_numpy(),
        }
    )
    right = pd.DataFrame({"key": mut[key].to_numpy(), "mutant": mut[metric].to_numpy()})
    if left["key"].duplicated().any() or right["key"].duplicated().any():
        raise ValueError(
            f"key {key!r} is not unique within a summary; a paired comparison needs one "
            "row per key on each side"
        )

    merged = left.merge(right, on="key", how="inner")
    if merged.empty:
        raise ValueError(
            f"no {key} value is present in both summaries; the mutant must keep the "
            "wild-type id for the two to pair"
        )
    merged["wild_type"] = pd.to_numeric(merged["wild_type"], errors="coerce")
    merged["mutant"] = pd.to_numeric(merged["mutant"], errors="coerce")
    merged["delta"] = merged["mutant"] - merged["wild_type"]

    matched = set(merged["key"])
    unmatched = tuple(
        k for k in list(left["key"]) + list(right["key"]) if k not in matched
    )
    return PairedComparison(
        frame=merged.reset_index(drop=True),
        metric=metric,
        unmatched=tuple(dict.fromkeys(unmatched)),
    )
