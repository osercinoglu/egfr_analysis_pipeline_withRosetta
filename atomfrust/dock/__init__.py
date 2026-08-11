"""Pose generation and the validity gate (plan step G4).

Where a ligand's coordinates come from, and whether they are allowed to be used.

**Backends are subprocess-only and probed at runtime.** Nothing here imports a docking
program. :meth:`~atomfrust.dock.base.PoseBackend.available` is a ``shutil.which`` lookup that
returns False rather than raising, so a machine without GNINA loses the GNINA backend and
keeps the package: ``import atomfrust.dock`` works, MCS alignment works, the gate works. That
is what makes a survey over "which generators can this machine run?" a question with an
answer instead of a sequence of import errors.

Four backends:

* :class:`~atomfrust.dock.smina.SminaBackend` — the open docking baseline (methods §3.7).
* :class:`~atomfrust.dock.gnina.GninaBackend` — CNN-rescored docking; smina's command line
  with a network on top.
* :class:`~atomfrust.dock.mcs_align.MCSAlignBackend` — RDKit-only placement onto the native
  ligand by maximum common substructure. It is the **docking-free ablation**: it is what
  tells us whether the chemotype axis (G6) measures chemotype or measures the docking
  program, since a signal that survives placement-without-search cannot be the search's
  doing. It is not a fallback of convenience.
* :class:`~atomfrust.dock.preposed.PrePosedBackend` — coordinates that already exist, passed
  through byte-for-byte, and gated like everything else.

**The gate is mandatory.** :func:`~atomfrust.dock.posebusters.check_poses` runs before
analysis, not after (R23), because ML docking methods are documented to produce physically
invalid poses and S4.2 requires >=90% of the poses that enter analysis to pass. It uses the
real ``posebusters`` package when importable and a clearly-labelled built-in subset when not;
the ``checker`` column names which one ran, so a subset result can never be presented as the
criterion itself.
"""

from __future__ import annotations

from atomfrust.dock.base import (
    BACKEND_NAMES,
    BackendUnavailable,
    Pose3D,
    PoseBackend,
    SubprocessBackend,
    available_backends,
    get_backend,
    list_backends,
)
from atomfrust.dock.gnina import GninaBackend
from atomfrust.dock.mcs_align import MCSAlignBackend
from atomfrust.dock.posebusters import (
    POSEBUSTERS_CHECKS,
    QC_COLUMNS,
    S4_2_MIN_PASS_RATE,
    CheckTolerances,
    check_poses,
    filter_valid,
    meets_s4_2,
    passing_pose_indices,
    pose_pass_rate,
    posebusters_available,
)
from atomfrust.dock.preposed import PrePosedBackend
from atomfrust.dock.smina import SminaBackend

__all__ = [
    "Pose3D",
    "PoseBackend",
    "SubprocessBackend",
    "BackendUnavailable",
    "BACKEND_NAMES",
    "get_backend",
    "list_backends",
    "available_backends",
    "MCSAlignBackend",
    "SminaBackend",
    "GninaBackend",
    "PrePosedBackend",
    "check_poses",
    "POSEBUSTERS_CHECKS",
    "QC_COLUMNS",
    "S4_2_MIN_PASS_RATE",
    "CheckTolerances",
    "posebusters_available",
    "pose_pass_rate",
    "meets_s4_2",
    "passing_pose_indices",
    "filter_valid",
]
