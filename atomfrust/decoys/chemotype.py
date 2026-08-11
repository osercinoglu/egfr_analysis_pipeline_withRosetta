"""Axis D — the chemotype decoy: the site is fixed and the *molecule* changes.

Axis A randomises the protein sequence, axis B randomises where the native molecule sits.
Axis D asks the question the other two cannot: **is this site favourable to *this molecule*
specifically, or to any molecule of roughly this size?** It is the project's novel
contribution and, as plan step G6 says, its hardest conceptual problem. The plan resolved
that problem; this module implements the resolution rather than re-deriving one.

The estimand: a ligand-node scalar, not a per-contact Z
------------------------------------------------------

Axes A and B can be scored per contact because their pair keys are well defined: the same
residue *i* and the same ligand exist in every decoy, so ``(i, j)`` names the same physical
object in the native and in every sample. Axis D has no such guarantee. A decoy molecule is
a *different molecule* — different heavy-atom count, different shape, different reach — so a
residue-anchored per-contact construction samples a **mixture distribution**:

* a **point mass at exactly zero**, contributed by every member whose atoms never come close
  enough to residue *i* for Rosetta to build an ``EnergyGraph`` edge. That zero is not a
  missing value; :meth:`atomfrust.energy.EnergyEvaluator.pair` returns ``PairEnergy(0.0,
  0.0, False)`` for an edgeless pair, which is a true statement about the absence of an
  interaction;
* plus a **continuous part whose scale grows with heavy-atom count**, because a bigger
  molecule makes more and larger contributions to every contact it does reach.

Mean and σ of that mixture are not interpretable, and ``F = (mean − native)/σ`` degenerates
into a monotone function of *contact probability × molecular size*. The unit test
``test_a_naive_per_contact_z_measures_occupancy_and_size`` pins this: with an interaction
strength that is *identical* at every residue by construction, the naive per-contact Z still
varies by a factor of several across residues purely because their occupancies differ.

So the primary quantity is **one scalar per library member**: the total many-body
ligand–site energy over the **frozen native pocket shell**
(:func:`ligand_site_energies`), scored with ``rank_percentile``
(:func:`atomfrust.analyze.zscore.rank_percentile`, chosen because it is distribution-free
and this ensemble is emphatically not Gaussian), with member MW, heavy-atom count, cLogP,
formal charge and rotatable-bond count recorded as covariates and the **energy-on-heavy-atom-
count regression slope reported and removed** (:func:`size_regression`). Size is the
confound that would otherwise reproduce, on the ligand axis, exactly the pocket-size
confound CLAUDE.md records on the protein axis.

Per-residue numbers are still produced — :meth:`ChemotypeDecoyGenerator.per_residue_decomposition`
— but they are **descriptive only**, restricted to keys with ≥ 80 % member occupancy, and
every row says so in an ``interpretation`` column. They are for looking at, not for testing.

The correspondence rule, and what was rejected
----------------------------------------------

Anchor on the **protein residue index**; freeze the shell to the native pocket; keep
zero-energy members (they are data, not missingness); write ``n_contacting_members`` per
key. Two alternatives were considered and rejected in the plan, and are deliberately *not*
implemented here:

* **Recomputing the shell per decoy.** The estimand would then vary from sample to sample —
  a bigger molecule would be scored over a bigger site — which is the same error as
  recomputing contact membership per decoy (:mod:`atomfrust.graph` refuses that too).
* **Atom-level or pharmacophore correspondence.** Undefined across topologically distinct
  molecules, which is the entire point of this axis. :func:`atomfrust.dock.base.symmetry_rmsd`
  already returns ``None`` for such a pair for exactly this reason.

The positive-control gate
-------------------------

Before any cross-axis redundancy number may be reported, the axis must show that **the
native molecule ranks high within its own ensemble** — AUROC ≥ 0.75
(:data:`NATIVE_RANK_AUROC_MIN`), the G5-style control. Without it, a near-degenerate axis-D
score would be trivially uncorrelated with axis A and the S2.6 redundancy test would pass by
noise rather than by evidence. :meth:`ChemotypeDecoyGenerator.cross_axis_redundancy` raises
:class:`PositiveControlFailed` until the gate passes; there is no flag to turn that off.

Placement: MCS alignment is the experiment, not the fallback
------------------------------------------------------------

``smina``, ``gnina`` and ``posebusters`` are absent on this machine
(:func:`atomfrust.dock.available_backends` reports only ``mcs_align`` and ``preposed``), so
the docking-free route is the live one. Per the plan that is **not merely a fallback**: it is
the ablation that distinguishes "frustration discriminates chemotypes" from "the docking
program refuses to pose non-binders". A docking backend given a non-binder can decline to
find a good pose, and a frustration index computed on the resulting bad geometry would then
be reporting the docking program's opinion, not the energy function's.
:class:`~atomfrust.dock.mcs_align.MCSAlignBackend` places every molecule by the same
deterministic geometric rule, with no scoring function and no search, so a signal that
survives it cannot be the search's doing. Pass ``backend=SminaBackend()`` once a binary
exists and run both; the two numbers are the ablation.

Protocol symmetry
-----------------

Every member is placed, then its **pocket shell is repacked and chi-minimised with the
ligand frozen**, using axis B's own :class:`~atomfrust.decoys.pose.PoseDecoyGenerator`
relaxation rather than a private copy of it — the two ligand axes must not drift apart on
what "relaxed" means. The **native is relaxed by the identical protocol** before it is
scored (:meth:`ChemotypeDecoyGenerator.prepare_native`): relaxing the decoys but not the
native would hand the decoys a free energy decrease and quietly rig the positive-control
gate against the native.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atomfrust.analyze.zscore import compute_index
from atomfrust.decoys.base import DecoyContext, DecoyResult, extract_energies
from atomfrust.decoys.pose import (
    PoseDecoyGenerator,
    ligand_heavy_coords,
    ligand_mol_from_pose,
)
from atomfrust.energy import effective_energy, many_body_energies

__all__ = [
    "ChemotypeDecoyGenerator",
    "PositiveControlFailed",
    "NATIVE_RANK_AUROC_MIN",
    "MIN_KEY_OCCUPANCY",
    "MEMBER_COVARIATES",
    "SiteEnergies",
    "SizeRegression",
    "ligand_site_energies",
    "native_pocket_shell",
    "size_regression",
    "rank_percentile_scores",
    "rank_within",
    "auroc_against",
    "per_residue_frame",
]

#: The positive control the plan requires before any redundancy number is reportable: the
#: native molecule must rank in the top quarter of its own ensemble. Not tunable per call by
#: design — a gate whose threshold moves with the result it is gating is not a gate.
NATIVE_RANK_AUROC_MIN = 0.75

#: Member occupancy below which a per-residue key is dropped from the descriptive
#: decomposition. A key most members never reach has a per-member distribution that is
#: mostly the point mass at zero, so its mean and σ describe reach, not chemistry.
MIN_KEY_OCCUPANCY = 0.8

#: The covariates the plan names. :func:`atomfrust.chem.libraries.property_summary` also
#: returns ``hba``/``hbd``; they are carried through because they are free, but ``hac`` is
#: the one the regression removes.
MEMBER_COVARIATES = ("mw", "hac", "logp", "formal_charge", "rotatable_bonds")


class PositiveControlFailed(RuntimeError):
    """Raised when a redundancy result is requested before the axis has earned it.

    Distinct from an error: the computation would succeed and return a number. The number
    would simply not mean what a reader would take it to mean, because an axis that cannot
    rank the native molecule above its own decoys is uncorrelated with *everything*.
    """


# ====================================================================== the estimand
#
# Everything in this section is pure array/table arithmetic — no pose, no Rosetta, no RDKit —
# so the estimand is testable without any of them, and re-specifiable against a finished run
# for the same reason `atomfrust.energy.many_body_energies` is.


@dataclass(frozen=True)
class SiteEnergies:
    """One member's ligand–site energy, decomposed over the frozen native pocket shell.

    ``total`` is the primary quantity: the sum of the many-body contact energy over every
    frozen pair joining the ligand node to a shell residue. ``per_key`` is the descriptive
    decomposition, in frozen-pair-table order so that the vectors of two different members
    are aligned element for element — which is the whole reason the shell is frozen.

    ``n_contacting`` counts keys where the member actually reached the residue (Rosetta
    built an energy-graph edge). The complementary count, ``len(per_key) - n_contacting``, is
    the size of the point mass at zero for this member: data, not missingness.
    """

    total: float
    total_direct: float
    per_key: pd.DataFrame
    n_contacting: int

    @property
    def n_keys(self) -> int:
        return int(len(self.per_key))


def ligand_site_energies(
    pairs: pd.DataFrame,
    e_direct: np.ndarray,
    e_fa_rep: np.ndarray,
    *,
    ligand_resnum: int,
    shell_resnums: Sequence[int],
    mode: str = "chen_literal",
    exclude_fa_rep: bool = True,
) -> SiteEnergies:
    """Total many-body ligand–site energy over a **fixed** shell, plus its per-key parts.

    ``pairs`` is the frozen native pair table and ``e_direct``/``e_fa_rep`` are one member's
    energies over it, in the same order — i.e. exactly what
    :func:`atomfrust.decoys.base.extract_energies` returns. The many-body sum is formed over
    the *whole* pair table before restriction, because the background terms ``B_i`` in Eq. 2
    run over all of a node's partners; restricting first would silently redefine the formula.

    A pair with no energy-graph edge contributes exactly ``0.0`` and is counted in
    ``n_contacting`` as absent. That zero is kept, never dropped: dropping it would make the
    estimand depend on which members happened to reach which residues.
    """
    e = effective_energy(e_direct, e_fa_rep, exclude_fa_rep)
    many_body = many_body_energies(
        pairs["node_i"].to_numpy(), pairs["node_j"].to_numpy(), e, mode  # type: ignore[arg-type]
    )

    i = pairs["i"].to_numpy(dtype=np.int64)
    j = pairs["j"].to_numpy(dtype=np.int64)
    incident = (i == int(ligand_resnum)) | (j == int(ligand_resnum))
    partner = np.where(i == int(ligand_resnum), j, i)
    shell = np.asarray(sorted({int(r) for r in shell_resnums}), dtype=np.int64)
    mask = incident & np.isin(partner, shell)

    raw_direct = np.asarray(e_direct, dtype=np.float64)[mask]
    raw_fa_rep = np.asarray(e_fa_rep, dtype=np.float64)[mask]
    # `has_edge` is not stored on the pair table, so it is recovered the only honest way:
    # an edgeless pair is exactly (0.0, 0.0), and a real edge is measured to float precision.
    contacts = (raw_direct != 0.0) | (raw_fa_rep != 0.0)

    node_i = pairs["node_i"].to_numpy()[mask]
    node_j = pairs["node_j"].to_numpy()[mask]

    per_key = pd.DataFrame(
        {
            "pair_id": pairs["pair_id"].to_numpy()[mask],
            "pose_resnum": partner[mask].astype(np.int32),
            # The partner's node id, picked by which endpoint the ligand sits on — never by
            # assuming the ligand is always `j`, which is true of this pair table only
            # because the ligand happens to be the last residue in the pose.
            "node_id": np.where(i[mask] == int(ligand_resnum), node_j, node_i),
            "e_manybody": many_body[mask],
            "e_effective": e[mask],
            "e_direct": raw_direct,
            "contacts": contacts,
        }
    )
    return SiteEnergies(
        total=float(many_body[mask].sum()),
        total_direct=float(e[mask].sum()),
        per_key=per_key,
        n_contacting=int(contacts.sum()),
    )


@dataclass(frozen=True)
class SizeRegression:
    """OLS of member energy on heavy-atom count — the confound, measured then removed.

    ``slope`` is in energy units per heavy atom. Reporting it is not decoration: if it is
    large and the raw scores rank the way size ranks, the axis has measured molecular weight
    with extra steps, which is the ligand-side analogue of the ``n_contacts_total`` confound
    CLAUDE.md records for the protein side.
    """

    slope: float
    intercept: float
    r: float
    n: int

    def residuals(self, energies: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """``energy − (intercept + slope·size)``. NaN in, NaN out."""
        energies = np.asarray(energies, dtype=np.float64)
        sizes = np.asarray(sizes, dtype=np.float64)
        return energies - (self.intercept + self.slope * sizes)

    def to_dict(self) -> dict[str, float]:
        return {
            "size_slope": self.slope,
            "size_intercept": self.intercept,
            "size_r": self.r,
            "size_n": float(self.n),
        }


def size_regression(energies: np.ndarray, sizes: np.ndarray) -> SizeRegression:
    """Fit ``energy ~ a + b·size`` over the finite pairs only.

    Fewer than three usable members, or a constant size, gives ``slope = 0`` and
    ``intercept = mean`` — a refusal to fit rather than a fit to nothing, so that
    residualising a degenerate ensemble leaves it centred instead of destroying it.
    """
    energies = np.asarray(energies, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    ok = np.isfinite(energies) & np.isfinite(sizes)
    n = int(ok.sum())
    if n == 0:
        return SizeRegression(0.0, 0.0, float("nan"), 0)
    y, x = energies[ok], sizes[ok]
    if n < 3 or float(np.ptp(x)) == 0.0:
        return SizeRegression(0.0, float(y.mean()), float("nan"), n)
    slope, intercept = np.polyfit(x, y, 1)
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(x, y)[0, 1])
    return SizeRegression(float(slope), float(intercept), r, n)


def rank_percentile_scores(values: np.ndarray, *, block: int = 512) -> np.ndarray:
    """Each member's ``rank_percentile`` against **the other** members.

    Delegates to :func:`atomfrust.analyze.zscore.compute_index` rather than reimplementing
    the tail fraction, so the sign convention (+1 = more favourable than every other member,
    Rosetta energies being negative-is-favourable) is the package's single definition.

    Leave-one-out, not self-inclusive, for one reason: the native is ranked against the
    members (it is not one of them), and a member ranked *including itself* would be on a
    scale shorter by a factor ``(n−1)/n``. The gate and the member column have to be
    comparable, so both exclude the thing being ranked.

    NaN members — those that failed to place, parametrise or load — take no part and come
    back NaN. Fewer than three scored members returns all-NaN: leaving one out of two leaves
    an ensemble of one, and a rank within an ensemble of one is not a rank.

    ``block`` bounds the working set: the comparison matrix is ``(n−1, block)`` rather than
    ``(n−1, n)``, which matters at library scale (MUV ships ~15,000 molecules per target).
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    ok = np.isfinite(values)
    ensemble = values[ok]
    n = ensemble.size
    if n < 3:
        return out
    scores = np.empty(n, dtype=np.float64)
    for start in range(0, n, max(1, block)):
        columns = range(start, min(start + max(1, block), n))
        others = np.empty((n - 1, len(columns)), dtype=np.float64)
        for position, k in enumerate(columns):
            others[:, position] = np.delete(ensemble, k)
        scores[start : start + others.shape[1]] = compute_index(
            ensemble[list(columns)], others, index="rank_percentile"
        )
    out[ok] = scores
    return out


def rank_within(value: float, ensemble: np.ndarray) -> float:
    """``rank_percentile`` of one value against an ensemble it is not a member of."""
    ensemble = np.asarray(ensemble, dtype=np.float64)
    ensemble = ensemble[np.isfinite(ensemble)]
    if not np.isfinite(value) or ensemble.size < 2:
        return float("nan")
    return float(
        compute_index(np.array([value]), ensemble.reshape(-1, 1), index="rank_percentile")[0]
    )


def auroc_against(value: float, ensemble: np.ndarray) -> float:
    """Probability that ``value`` is more favourable than a random ensemble member.

    Ties count a half, so a value equal to every member scores 0.5 rather than 0 or 1. With
    a single positive this is the Mann–Whitney AUROC exactly, and it is the affine image
    ``(rank_percentile + 1)/2`` of :func:`rank_within` — reported as an AUROC because that is
    the form G5 states its own positive control in.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    ensemble = ensemble[np.isfinite(ensemble)]
    if not np.isfinite(value) or ensemble.size == 0:
        return float("nan")
    wins = float((ensemble > value).sum())
    ties = float((ensemble == value).sum())
    return (wins + 0.5 * ties) / ensemble.size


def per_residue_frame(
    keys: pd.DataFrame,
    member_manybody: np.ndarray,
    member_contacts: np.ndarray,
    native_manybody: np.ndarray,
    *,
    min_occupancy: float = MIN_KEY_OCCUPANCY,
) -> pd.DataFrame:
    """The descriptive per-residue table. **Not an index; not to be thresholded.**

    ``member_manybody`` is ``(n_members, n_keys)`` and ``member_contacts`` the matching
    boolean "did this member reach this residue" matrix; ``keys`` carries one row per key in
    the same column order. Keys below ``min_occupancy`` are dropped, because their per-member
    distribution is mostly the point mass at zero and its moments describe reach rather than
    chemistry. Every surviving row carries ``n_contacting_members`` and an ``interpretation``
    column saying ``descriptive_only``, so a table that escapes into a report cannot be read
    as a Z-score by someone who did not read this docstring.
    """
    member_manybody = np.atleast_2d(np.asarray(member_manybody, dtype=np.float64))
    member_contacts = np.atleast_2d(np.asarray(member_contacts, dtype=bool))
    native_manybody = np.asarray(native_manybody, dtype=np.float64)
    n_members = member_manybody.shape[0]

    if member_manybody.shape[1] != len(keys) or member_contacts.shape != member_manybody.shape:
        raise ValueError(
            f"key/member shape mismatch: keys={len(keys)}, energies={member_manybody.shape}, "
            f"contacts={member_contacts.shape}"
        )

    usable = np.isfinite(member_manybody)
    n_scored = usable.all(axis=1).sum() if n_members else 0
    n_contacting = (member_contacts & usable).sum(axis=0)
    occupancy = n_contacting / n_scored if n_scored else np.zeros(len(keys))

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(member_manybody, axis=0) if n_members else np.full(len(keys), np.nan)
        sd = (
            np.nanstd(member_manybody, axis=0, ddof=1)
            if n_members > 1
            else np.full(len(keys), np.nan)
        )

    frame = keys.copy().reset_index(drop=True)
    frame["n_members_scored"] = int(n_scored)
    frame["n_contacting_members"] = n_contacting.astype(np.int32)
    frame["occupancy"] = occupancy
    frame["native_e_manybody"] = native_manybody
    frame["mean_member_e_manybody"] = mean
    frame["sd_member_e_manybody"] = sd
    frame["native_rank_percentile"] = [
        rank_within(native_manybody[k], member_manybody[:, k]) for k in range(len(frame))
    ]
    frame["min_occupancy"] = float(min_occupancy)
    frame["interpretation"] = "descriptive_only"
    kept = frame.loc[frame["occupancy"] >= float(min_occupancy)].reset_index(drop=True)
    kept.attrs["descriptive_only"] = True
    kept.attrs["note"] = (
        "Per-residue values are descriptive. The axis-D estimand is the ligand-node scalar "
        "(member_scores); a per-contact Z over a changing molecule is a mixture of a point "
        "mass at zero and a size-scaled continuous part."
    )
    kept.attrs["n_keys_before_occupancy_filter"] = int(len(frame))
    return kept


def native_pocket_shell(
    pose: Any, nodes: Sequence[Any], ligand_resnum: int, shell_A: float = 6.0
) -> list[int]:
    """Protein residues within ``shell_A`` of the **native** ligand's heavy atoms.

    Computed once, on the native pose, and never recomputed for a member — that is the
    correspondence rule. Recomputing it per member would let a larger molecule be scored over
    a larger site, so the estimand would vary from sample to sample and the ensemble would no
    longer be an ensemble of anything.
    """
    ligand_xyz = ligand_heavy_coords(pose, int(ligand_resnum))
    shell: list[int] = []
    for node in nodes:
        if node.kind not in ("protein", "noncanonical"):
            continue
        residue = pose.residue(node.pose_resnum)
        points = np.asarray(
            [
                (residue.xyz(k).x, residue.xyz(k).y, residue.xyz(k).z)
                for k in range(1, residue.nheavyatoms() + 1)
            ],
            dtype=float,
        ).reshape(-1, 3)
        if points.size == 0:
            continue
        deltas = points[:, None, :] - ligand_xyz[None, :, :]
        if np.sqrt((deltas**2).sum(axis=2)).min() <= shell_A:
            shell.append(int(node.pose_resnum))
    return sorted(shell)


# ==================================================================== the generator


@dataclass
class _ScoredMember:
    """One member's outcome — success and failure in the same shape, as in ``ParamRecord``."""

    index: int
    status: str
    site: SiteEnergies | None = None
    row: dict[str, Any] = field(default_factory=dict)
    pose: Any = None
    e_direct: np.ndarray | None = None
    e_fa_rep: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.site is not None


@dataclass
class ChemotypeDecoyGenerator:
    """Axis D generator: swap the molecule, hold the site.

    ``members`` is the decoy library — :class:`~atomfrust.chem.libraries.MolRecord`\\ s from
    any adapter, with their ``role`` intact so a synthetic ``property_decoy`` can never be
    silently pooled with a ``measured_inactive``.

    ``shell_A`` freezes the pocket. It defaults to 6.0 Å, matching
    ``ContactSettings.ligand_cutoff_A``, so the shell is the same set of residues the contact
    definition would call ligand contacts.

    The interesting outputs are :meth:`member_scores` (the primary table),
    :meth:`native_rank_auroc` (the gate) and :meth:`per_residue_decomposition` (descriptive).
    :meth:`generate` exists so the axis satisfies
    :class:`~atomfrust.decoys.base.DecoyGenerator` like every other axis, but note what its
    ``DecoyResult`` means: the per-pair energies are the raw material for the scalar and for
    the descriptive decomposition, **not** a vector to Z-score. ``index_row`` says so in
    ``per_contact_index_valid``.
    """

    context: DecoyContext
    members: Sequence[Any]
    base_seed: int = 42
    axis: str = "chemotype"
    shell_A: float = 6.0

    #: Placement backend. ``None`` builds :class:`~atomfrust.dock.mcs_align.MCSAlignBackend`,
    #: which is the docking-free ablation described in the module docstring — and the only
    #: route available on this machine.
    backend: Any | None = None

    #: Which component is replaced. ``None`` when the system has exactly one poseable one.
    ligand_resnum: int | None = None

    #: Defaults come from ``context.settings`` so axis D cannot silently disagree with the
    #: rest of the run about what an energy is.
    many_body_mode: str | None = None
    exclude_fa_rep: bool | None = None

    #: Repack + chi-minimise the pocket around the placed molecule (axis B's protocol), and
    #: do the same to the native before scoring it. Turning this off scores raw MCS geometry,
    #: where ``fa_rep`` dominates everything.
    relax: bool = True

    #: Validity gate, applied to the pose that is actually scored.
    checker: str = "auto"

    #: Where ``.params`` and the per-member complex PDBs are written. ``None`` uses a
    #: temporary directory owned by this generator.
    work_dir: Path | None = None

    #: Optional :class:`~atomfrust.chem.cache.ParamCache`; a library re-run should not
    #: re-parametrise every molecule.
    param_cache: Any | None = None

    #: The positive control the redundancy output is gated on.
    gate_auroc: float = NATIVE_RANK_AUROC_MIN

    def __post_init__(self) -> None:
        from atomfrust.chem import CodeAllocator

        # Axis B owns the ligand-finding, receptor-writing and pocket-relaxation helpers.
        # Reusing the object rather than copying its methods is what keeps the two ligand
        # axes from drifting apart on what "the pocket" and "relaxed" mean.
        self._poser = PoseDecoyGenerator(
            self.context,
            base_seed=self.base_seed,
            ligand_resnum=self.ligand_resnum,
            checker=self.checker,
            minimise=self.relax,
        )
        self._resnum = self._poser.pose_resnum
        if self.backend is None:
            from atomfrust.dock import MCSAlignBackend

            self.backend = MCSAlignBackend()
        self._allocator = CodeAllocator()
        self._tempdir: Any | None = None
        self._shell: list[int] | None = None
        self._native_mol: Any | None = None
        self._receptor: Path | None = None
        self._scored: dict[int, _ScoredMember] = {}
        self._native: _ScoredMember | None = None
        self._native_pose: Any | None = None

    # ------------------------------------------------------------------ identity

    @property
    def pose_resnum(self) -> int:
        """Pose residue number of the component this axis replaces."""
        return self._resnum

    @property
    def generator_name(self) -> str:
        """``chemotype/mcs_align`` — the placement route, readable before any decoy exists."""
        return f"{self.axis}/{getattr(self.backend, 'name', 'backend')}"

    @property
    def mode(self) -> str:
        return self.many_body_mode or self.context.settings.manybody.mode

    @property
    def fa_rep_excluded(self) -> bool:
        if self.exclude_fa_rep is None:
            return bool(self.context.settings.energy.exclude_fa_rep)
        return bool(self.exclude_fa_rep)

    @property
    def shell_resnums(self) -> list[int]:
        """The frozen native pocket shell, computed once and reused for every member."""
        if self._shell is None:
            self._shell = native_pocket_shell(
                self.context.pose, self.context.nodes, self._resnum, self.shell_A
            )
        return self._shell

    # ------------------------------------------------------------------ the axis

    def generate(self, decoy_id: int) -> DecoyResult:
        """Score member ``decoy_id`` of the library.

        ``decoy_id`` indexes ``members`` — unlike axes A and B, where it is a seed index and
        the ensemble is unbounded, this axis's ensemble *is* the library and asking for
        member 500 of a 50-molecule library is a bug rather than a 500th draw.
        """
        if not 0 <= int(decoy_id) < len(self.members):
            raise IndexError(
                f"decoy_id {decoy_id} is out of range for a {len(self.members)}-member "
                "library; on this axis a decoy is a library member, not a draw"
            )
        scored = self._score_member(int(decoy_id))
        n_pairs = len(self.context.pairs)
        empty = np.full(n_pairs, np.nan, dtype=np.float32)
        return DecoyResult(
            decoy_id=int(decoy_id),
            pair_id=self.context.pair_ids,
            e_direct=scored.e_direct if scored.e_direct is not None else empty,
            e_fa_rep=scored.e_fa_rep if scored.e_fa_rep is not None else empty.copy(),
            pose=scored.pose,
            index_row=dict(scored.row),
        )

    def prepare_native(self) -> Any:
        """The native reference: the crystal pose, relaxed under the members' protocol.

        Axis B can return the crystal pose untouched because its decoys differ from it only
        by a displacement. Here they differ by a repack, so an unrelaxed native would be
        compared against relaxed decoys and would lose the gate for a protocol asymmetry
        rather than for a chemical reason.
        """
        if self._native_pose is None:
            pose = self.context.pose.clone()
            if self.relax:
                self._poser._relax_pocket(pose)
            self._native_pose = pose
        return self._native_pose

    # ------------------------------------------------------------------ outputs

    def member_scores(self) -> pd.DataFrame:
        """**The primary output.** One row per library member, plus one for the native.

        Columns: identity (``source``, ``source_id``, ``source_ref``, ``role``, ``smiles``,
        ``inchikey``, ``rosetta_code``), the covariates
        (:data:`MEMBER_COVARIATES` plus ``hba``/``hbd``), the estimand (``e_site``, its
        size-residualised twin ``e_site_resid``, and both scored with ``rank_percentile``),
        the reach diagnostics (``n_contacting_keys`` / ``n_keys``), the validity verdict and
        ``status``.

        The native row carries ``is_native=True`` and ``member_index=-1``. It is *not* part
        of the ensemble the ranks are taken over — it is ranked *against* it, which is the
        positive control — and it is excluded from the size regression for the same reason.

        ``frame.attrs`` carries the run-level facts a caller should not have to recompute:
        the size-regression coefficients, the native's AUROC, whether the gate passed, the
        placement route, the shell size and the many-body mode.
        """
        rows: list[dict[str, Any]] = []
        for index in range(len(self.members)):
            rows.append(dict(self._score_member(index).row))
        native = self._score_native()
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=list(native.row))

        member_energy = frame["e_site"].to_numpy(dtype=np.float64) if len(frame) else np.array([])
        member_size = frame["hac"].to_numpy(dtype=np.float64) if len(frame) else np.array([])
        fit = size_regression(member_energy, member_size)

        native_row = dict(native.row)
        frame = pd.concat([frame, pd.DataFrame([native_row])], ignore_index=True)

        is_native = frame["is_native"].to_numpy(dtype=bool)
        energy = frame["e_site"].to_numpy(dtype=np.float64)
        size = frame["hac"].to_numpy(dtype=np.float64)
        residual = fit.residuals(energy, size)

        ensemble = energy[~is_native]
        ensemble_resid = residual[~is_native]

        frame["e_site_resid"] = residual
        frame["rank_percentile"] = np.where(
            is_native,
            [rank_within(v, ensemble) for v in energy],
            rank_percentile_scores(np.where(is_native, np.nan, energy)),
        )
        frame["rank_percentile_resid"] = np.where(
            is_native,
            [rank_within(v, ensemble_resid) for v in residual],
            rank_percentile_scores(np.where(is_native, np.nan, residual)),
        )

        auroc = auroc_against(float(native.row["e_site"]), ensemble)
        auroc_resid = auroc_against(float(residual[is_native][0]), ensemble_resid)
        frame.attrs.update(
            {
                "axis": self.axis,
                "generator": self.generator_name,
                "placement_backend": getattr(self.backend, "name", "backend"),
                "docking_free_ablation": getattr(self.backend, "name", "") == "mcs_align",
                "many_body_mode": self.mode,
                "exclude_fa_rep": self.fa_rep_excluded,
                "shell_A": self.shell_A,
                "n_shell_residues": len(self.shell_resnums),
                "n_members": len(self.members),
                "n_scored": int(np.isfinite(ensemble).sum()),
                "native_e_site": float(native.row["e_site"]),
                "native_rank_auroc": auroc,
                "native_rank_auroc_resid": auroc_resid,
                "gate_auroc": float(self.gate_auroc),
                "gate_passed": bool(np.isfinite(auroc) and auroc >= self.gate_auroc),
                "estimand": "ligand_node_scalar",
                "per_contact_index_valid": False,
                **fit.to_dict(),
            }
        )
        return frame

    def native_rank_auroc(self, *, residualised: bool = False) -> float:
        """**The positive-control gate.** P(native beats a random member), ties a half.

        ``residualised=True`` asks the same question of the size-corrected score, which is
        the harder version: a native that only wins because it is large has not shown that
        the site prefers *it*.
        """
        scores = self.member_scores()
        key = "native_rank_auroc_resid" if residualised else "native_rank_auroc"
        return float(scores.attrs[key])

    def gate_passed(self) -> bool:
        """Whether the axis has earned the right to report a redundancy number."""
        auroc = self.native_rank_auroc()
        return bool(np.isfinite(auroc) and auroc >= self.gate_auroc)

    def per_residue_decomposition(self, min_occupancy: float = MIN_KEY_OCCUPANCY) -> pd.DataFrame:
        """Descriptive per-residue table over keys ≥ ``min_occupancy`` member occupancy.

        See :func:`per_residue_frame`. This is not an index and must not be thresholded like
        one; the ``interpretation`` column repeats that on every row.
        """
        native = self._score_native()
        if native.site is None:
            raise RuntimeError("the native could not be scored, so there is nothing to decompose")
        keys = native.site.per_key[["pair_id", "pose_resnum", "node_id"]].copy()

        scored = [self._score_member(k) for k in range(len(self.members))]
        usable = [s for s in scored if s.ok]
        if not usable:
            return per_residue_frame(
                keys,
                np.empty((0, len(keys))),
                np.empty((0, len(keys)), dtype=bool),
                native.site.per_key["e_manybody"].to_numpy(),
                min_occupancy=min_occupancy,
            )
        energies = np.vstack([s.site.per_key["e_manybody"].to_numpy() for s in usable])  # type: ignore[union-attr]
        contacts = np.vstack([s.site.per_key["contacts"].to_numpy() for s in usable])  # type: ignore[union-attr]
        frame = per_residue_frame(
            keys,
            energies,
            contacts,
            native.site.per_key["e_manybody"].to_numpy(),
            min_occupancy=min_occupancy,
        )
        frame.attrs["axis"] = self.axis
        frame.attrs["estimand"] = "ligand_node_scalar (this table is not it)"
        return frame

    def cross_axis_redundancy(
        self,
        other: Mapping[str, float] | pd.Series | Sequence[float],
        *,
        other_axis: str = "identity",
        residualised: bool = True,
    ) -> pd.DataFrame:
        """Correlate this axis's per-member score against another axis's — **if gated**.

        ``other`` is either keyed by ``source_ref`` (a mapping or a Series) or positional
        (one value per member, in ``members`` order). The result is a one-row frame with
        Spearman and Pearson coefficients, ``n``, and the gate evidence that licenses it.

        Raises :class:`PositiveControlFailed` when the native does not rank at
        ``gate_auroc`` within its own ensemble. This is S2.6's precondition, not a
        convenience check: a near-degenerate axis-D score is uncorrelated with *every* other
        axis, so an ungated redundancy test passes by noise and would be read as evidence
        that the axes measure different things.
        """
        scores = self.member_scores()
        auroc = float(scores.attrs["native_rank_auroc"])
        if not (np.isfinite(auroc) and auroc >= self.gate_auroc):
            raise PositiveControlFailed(
                f"axis {self.axis!r} has not passed its positive control: the native molecule "
                f"ranks at AUROC {auroc:.3f} within its own ensemble, below the required "
                f"{self.gate_auroc:.2f}. A redundancy number computed now would measure noise "
                "— an axis that cannot rank the native above its decoys is uncorrelated with "
                "everything. Fix the axis (placement, relaxation, library) and re-run; there "
                "is no override."
            )

        members = scores.loc[~scores["is_native"].astype(bool)].reset_index(drop=True)
        if isinstance(other, Mapping) or isinstance(other, pd.Series):
            mapping = dict(other)
            values = np.array(
                [float(mapping.get(ref, np.nan)) for ref in members["source_ref"]], dtype=float
            )
        else:
            values = np.asarray(list(other), dtype=float)
            if values.shape[0] != len(members):
                raise ValueError(
                    f"positional `other` has {values.shape[0]} values but there are "
                    f"{len(members)} members; pass a mapping keyed by source_ref instead"
                )

        column = "rank_percentile_resid" if residualised else "rank_percentile"
        mine = members[column].to_numpy(dtype=float)
        ok = np.isfinite(mine) & np.isfinite(values)
        n = int(ok.sum())
        if n >= 3:
            from scipy import stats

            spearman = stats.spearmanr(mine[ok], values[ok])
            pearson = stats.pearsonr(mine[ok], values[ok])
            rho, rho_p = float(spearman.statistic), float(spearman.pvalue)
            r, r_p = float(pearson.statistic), float(pearson.pvalue)
        else:
            rho = rho_p = r = r_p = float("nan")

        return pd.DataFrame(
            [
                {
                    "axis": self.axis,
                    "other_axis": other_axis,
                    "score_column": column,
                    "n": n,
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                    "pearson_r": r,
                    "pearson_p": r_p,
                    "native_rank_auroc": auroc,
                    "gate_auroc": float(self.gate_auroc),
                    "gate_passed": True,
                }
            ]
        )

    # ------------------------------------------------------------------ scoring one member

    def _score_member(self, index: int) -> _ScoredMember:
        if index in self._scored:
            return self._scored[index]
        started = time.perf_counter()
        record = self.members[index]
        seed = self.base_seed + index
        row: dict[str, Any] = {
            "decoy_id": index,
            "member_index": index,
            "axis": self.axis,
            "generator": self.generator_name,
            "seed": seed,
            "is_native": False,
            "source": getattr(record, "source", ""),
            "source_id": getattr(record, "source_id", ""),
            "source_ref": getattr(record, "source_ref", ""),
            "role": getattr(record, "role", ""),
            "smiles": getattr(record, "smiles", ""),
            "inchikey": getattr(record, "inchikey", None),
            "estimand": "ligand_node_scalar",
            "per_contact_index_valid": False,
        }
        row.update(self._covariates(getattr(record, "smiles", "")))

        scored = self._score_smiles(
            row, getattr(record, "smiles", ""), seed=seed, tag=f"member{index:04d}"
        )
        scored.index = index
        scored.row["wall_s"] = time.perf_counter() - started
        self._scored[index] = scored
        return scored

    def _score_native(self) -> _ScoredMember:
        """Score the crystal ligand under the members' protocol, minus the placement step.

        The native is *not* re-placed by the backend: MCS-aligning a molecule onto itself is
        the identity map plus embedding noise, and the crystal pose is the reference the
        whole project is about. It *is* relaxed identically — see :meth:`prepare_native`.
        """
        if self._native is not None:
            return self._native
        from rdkit import Chem

        pose = self.prepare_native()
        smiles = ""
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(self._native_ligand_mol())))
        except Exception:  # pragma: no cover - a residue type RDKit cannot express
            smiles = ""
        node = next(n for n in self.context.nodes if n.pose_resnum == self._resnum)
        row: dict[str, Any] = {
            "decoy_id": -1,
            "member_index": -1,
            "axis": self.axis,
            "generator": f"{self.axis}/native",
            "seed": self.base_seed,
            "is_native": True,
            "source": "native",
            "source_id": str(node.ccd_id or node.resname),
            "source_ref": f"native:{node.ccd_id or node.resname}",
            "role": "active",
            "smiles": smiles,
            "inchikey": None,
            "rosetta_code": node.rosetta_name or node.resname,
            "estimand": "ligand_node_scalar",
            "per_contact_index_valid": False,
        }
        row.update(self._covariates(smiles))
        pair_id, e_direct, e_fa_rep = extract_energies(
            pose, self.context.pairs, self.context.score_function
        )
        site = self._site(e_direct, e_fa_rep)
        row.update(self._site_columns(site))
        row.update(self._verdict(pose))
        row["status"] = "ok"
        self._native = _ScoredMember(
            index=-1, status="ok", site=site, row=row, pose=pose,
            e_direct=e_direct, e_fa_rep=e_fa_rep,
        )
        return self._native

    def _score_smiles(
        self, row: dict[str, Any], smiles: str, *, seed: int, tag: str
    ) -> _ScoredMember:
        """Place → parametrise → build a complex → relax → score. Failures are rows.

        Every stage that can fail on a *molecule* (as opposed to on the code) is caught and
        turned into a ``status``, because a screening library's whole point is that some of
        its members are unusable and the count of those is a result.
        """
        blank = self._site_columns(None)
        row.update(blank)
        row.setdefault("rosetta_code", "")
        row.update(
            {"pose_valid": None, "pose_checker": "not_run", "pose_checks_run": 0,
             "pose_failed_checks": ""}
        )

        if not smiles:
            row["status"] = "no_smiles"
            return _ScoredMember(index=-1, status=row["status"], row=row)

        # --- placement -------------------------------------------------------------
        try:
            poses = self.backend.pose(
                smiles,
                self._receptor_pdb(),
                reference_ligand=self._native_ligand_mol(),
                n_poses=1,
                seed=seed,
            )
        except Exception as exc:
            row["status"] = f"placement_failed: {type(exc).__name__}: {exc}"
            return _ScoredMember(index=-1, status="placement_failed", row=row)
        if not poses:
            row["status"] = "placement_failed: backend returned no poses"
            return _ScoredMember(index=-1, status="placement_failed", row=row)

        placed = poses[0]
        row["placement"] = str(placed.metadata.get("placement", ""))
        row["n_mcs_atoms"] = placed.metadata.get("n_mcs_atoms")
        row["mcs_rmsd"] = placed.metadata.get("mcs_rmsd")

        # --- parametrisation --------------------------------------------------------
        from atomfrust.chem import paramize
        from rdkit import Chem

        mol = Chem.MolFromMolBlock(placed.molblock, removeHs=False)
        if mol is None:
            row["status"] = "placement_failed: unparseable molblock"
            return _ScoredMember(index=-1, status="placement_failed", row=row)

        record = paramize(
            mol,
            out_dir=self._workdir(),
            cache=self.param_cache,
            allocator=self._allocator,
        )
        row["rosetta_code"] = record.rosetta_code
        row["inchikey"] = row.get("inchikey") or record.inchikey or None
        if not record.ok:
            row["status"] = f"param_{record.failure.value}"  # type: ignore[union-attr]
            row["message"] = record.message
            return _ScoredMember(index=-1, status=row["status"], row=row)

        # --- pose -------------------------------------------------------------------
        try:
            pose = self._load_member_pose(record, tag)
        except Exception as exc:
            row["status"] = f"pose_load_failed: {type(exc).__name__}: {exc}"
            return _ScoredMember(index=-1, status="pose_load_failed", row=row)

        if self.relax:
            self._poser._relax_pocket(pose)

        pair_id, e_direct, e_fa_rep = extract_energies(
            pose, self.context.pairs, self.context.score_function
        )
        site = self._site(e_direct, e_fa_rep)
        row.update(self._site_columns(site))
        row.update(self._verdict(pose))
        row["status"] = "ok"
        return _ScoredMember(
            index=-1, status="ok", site=site, row=row, pose=pose,
            e_direct=e_direct, e_fa_rep=e_fa_rep,
        )

    # ------------------------------------------------------------------ plumbing

    def _site(self, e_direct: np.ndarray, e_fa_rep: np.ndarray) -> SiteEnergies:
        return ligand_site_energies(
            self.context.pairs,
            e_direct,
            e_fa_rep,
            ligand_resnum=self._resnum,
            shell_resnums=self.shell_resnums,
            mode=self.mode,
            exclude_fa_rep=self.fa_rep_excluded,
        )

    @staticmethod
    def _site_columns(site: SiteEnergies | None) -> dict[str, Any]:
        if site is None:
            return {
                "e_site": float("nan"),
                "e_site_direct": float("nan"),
                "n_keys": 0,
                "n_contacting_keys": 0,
                "frac_keys_contacted": float("nan"),
            }
        return {
            "e_site": site.total,
            "e_site_direct": site.total_direct,
            "n_keys": site.n_keys,
            "n_contacting_keys": site.n_contacting,
            "frac_keys_contacted": (site.n_contacting / site.n_keys) if site.n_keys else float("nan"),
        }

    def _covariates(self, smiles: str) -> dict[str, float]:
        """MW, HAC, cLogP, formal charge, rotatable bonds (+ HBA/HBD) for one SMILES.

        Delegates to :func:`atomfrust.chem.libraries.property_summary` so the descriptor
        definitions are the library layer's, not a second set that could drift from the ones
        :func:`~atomfrust.chem.libraries.doe_score` matches on.
        """
        from atomfrust.chem.libraries import MolRecord, PROPERTY_COLUMNS, property_summary

        if not smiles:
            return {name: float("nan") for name in PROPERTY_COLUMNS}
        try:
            frame = property_summary(
                [MolRecord(smiles=smiles, inchikey=None, source="", source_id="", role="active")]
            )
        except Exception:  # pragma: no cover - RDKit absent
            return {name: float("nan") for name in PROPERTY_COLUMNS}
        return {name: float(frame[name].iloc[0]) for name in PROPERTY_COLUMNS}

    def _workdir(self) -> Path:
        if self.work_dir is not None:
            path = Path(self.work_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path
        if self._tempdir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="atomfrust_chemotype_")
        return Path(self._tempdir.name)

    def _native_ligand_mol(self) -> Any:
        """The native ligand as RDKit sees Rosetta's ``ResidueType`` — bond orders included.

        :func:`~atomfrust.decoys.pose.ligand_mol_from_pose` rather than a PDB round trip: the
        MCS that places every member is computed against this molecule, and a perceived-bond
        reference would make every aromatic ring a set of single bonds and change which atoms
        the maximum common substructure matches.
        """
        if self._native_mol is None:
            self._native_mol = ligand_mol_from_pose(self.context.pose, self._resnum)
        return self._native_mol

    def _receptor_pdb(self) -> Path:
        """The native receptor minus the ligand being replaced, written once."""
        if self._receptor is None:
            source = self._poser.receptor_pdb(self.context.pose)
            target = self._workdir() / "receptor.pdb"
            target.write_text(source.read_text())
            self._receptor = target
        return self._receptor

    def _load_member_pose(self, record: Any, tag: str) -> Any:
        """Build ``receptor + this molecule`` and load it with the member's ``.params``.

        The coordinates come from ``{code}_0001.pdb``, the conformer PDB
        ``molfile_to_params`` writes beside the params file. That file is used rather than
        the molblock for one reason: its **atom names are the params atom names**, in params
        order, so ``fill_missing_atoms`` cannot fail on a name mismatch — the failure mode
        CLAUDE.md records as the reason Stage 5's CCD-CIF-first cascade exists. Its
        coordinates are the placed ones, because :func:`~atomfrust.chem.paramize.paramize`
        keeps an existing conformer rather than re-embedding (verified: centroid agrees with
        the placed molecule to 1e-4 Å).

        The molecule is written at the **native ligand's chain and residue number**, so the
        member pose has the same residue numbering as the native and the frozen pair table
        applies to it unchanged. That is what makes the pair index a valid anchor.

        Known limit: a library that lists the *same* molecule twice allocates it the same
        Rosetta code twice, and Rosetta refuses to register a residue type it already holds
        ("residue type 'XYZ' already exists in the cache"). The second copy is recorded as
        ``pose_load_failed`` rather than crashing the run — deduplicate on ``inchikey``
        upstream if a library has duplicates.
        """
        import pyrosetta
        from pyrosetta import Pose, Vector1

        conformer = Path(record.params_path).with_name(f"{record.rosetta_code}_0001.pdb")
        if not conformer.exists():
            raise FileNotFoundError(f"molfile_to_params wrote no conformer PDB at {conformer}")

        info = self.context.pose.pdb_info()
        chain = info.chain(self._resnum)
        resseq = int(info.number(self._resnum))

        het: list[str] = []
        for line in conformer.read_text().splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            het.append("HETATM" + line[6:21] + chain + f"{resseq:>4d}" + line[26:])

        receptor = self._receptor_pdb().read_text().splitlines()
        keep = [line for line in receptor if line.startswith(("ATOM", "HETATM"))]
        complex_pdb = self._workdir() / f"{tag}_{record.rosetta_code}_complex.pdb"
        complex_pdb.write_text("\n".join(keep + het) + "\nTER\nEND\n")

        pose = Pose()
        residue_set = pyrosetta.generate_nonstandard_residue_set(
            pose, Vector1([str(record.params_path)])
        )
        pyrosetta.pose_from_file(pose, residue_set, str(complex_pdb))
        if pose.total_residue() != self.context.pose.total_residue():
            raise ValueError(
                f"member pose has {pose.total_residue()} residues, native has "
                f"{self.context.pose.total_residue()}; the frozen pair table would not apply"
            )
        return pose

    def _verdict(self, pose: Any) -> dict[str, Any]:
        """Run the S4.2 gate on the pose that is about to be scored.

        ``rmsd_to_reference`` is ``None`` and stays ``None``: a different molecule has no
        atom correspondence with the native and therefore no RMSD to it —
        :func:`atomfrust.dock.base.symmetry_rmsd` returns ``None`` for the same reason. That
        absence is precisely what distinguishes this axis from G5, where the RMSD is the
        acceptance criterion.
        """
        from rdkit import Chem

        from atomfrust.dock import check_poses
        from atomfrust.dock.base import Pose3D

        try:
            mol = ligand_mol_from_pose(pose, self._resnum)
            pose3d = Pose3D(
                molblock=Chem.MolToMolBlock(mol),
                source=self.generator_name,
                rmsd_to_reference=None,
                metadata={"smiles": Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))},
            )
            qc = check_poses(
                [pose3d], self._poser.receptor_pdb(pose), checker=self.checker
            )
        except Exception as exc:
            return {
                "pose_valid": None,
                "pose_checker": "unavailable",
                "pose_checks_run": 0,
                "pose_failed_checks": f"{type(exc).__name__}: {exc}",
            }

        ran = qc[~qc["skipped"].astype(bool)]
        failed = sorted(ran.loc[~ran["passed"].astype(bool), "check"].tolist())
        return {
            "pose_valid": bool(len(ran) > 0 and not failed),
            "pose_checker": str(qc["checker"].iloc[0]) if len(qc) else "unavailable",
            "pose_checks_run": int(len(ran)),
            "pose_failed_checks": ",".join(failed),
        }
