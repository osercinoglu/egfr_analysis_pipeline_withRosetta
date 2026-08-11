"""
atomfrust — atomistic local frustration for protein–ligand, protein-only and
protein–protein complexes.

A target-agnostic reimplementation of the atomistic frustration method of
Chen et al., *Nat. Commun.* **11**, 5944 (2020), built around two corrections
established by the Stage A diagnostics (see `plans/frustratometer-ng-plan.md`):

1. The ligand is a **node in the contact graph**. The published counts are
   ligand–residue contacts; the prototype counted protein–protein pairs near the
   ligand, which is a different quantity (step A4).
2. The many-body contact energy is **selectable**. The published Eq. 2 collapses to
   ``E_ij = 0.5 * (B_i + B_j)``, carrying no pair-specific information (step A1);
   reproducing it is required for a reproduction claim, but it is not the only
   defensible formula.

Nothing here is implemented yet beyond the package skeleton — this is step B1.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("atomfrust")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
