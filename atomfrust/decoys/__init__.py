"""Decoy generators.

Each axis is a separate generator behind one protocol (:mod:`atomfrust.decoys.base`), so
axes are independently invocable — methods-doc S2.1.
"""

from atomfrust.decoys.base import DecoyContext, DecoyResult, NullGenerator, extract_energies

__all__ = ["DecoyContext", "DecoyResult", "NullGenerator", "extract_energies"]
