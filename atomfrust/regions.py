"""Region selectors — which residues get mutated, repacked, minimised, or frozen.

This is the substrate for user request 4 (focus decoy generation on selected regions) and
for requirement R12 (repacking restricted to a shell around the pocket, methods-doc S1.3).
The prototype had none of it: one whole-pose ``RestrictToRepacking()``
(``frustration.py:347``) with no per-residue operation anywhere, so no region could be
expressed.

Grammar::

    expr    := term ('or' term)*
    term    := factor ('and' factor)*
    factor  := 'not' factor | '(' expr ')' | primary
    primary := all | none | protein | ligand | water
             | chain A[,B]        | resi 10-20,45   | resn ALA,GLY
             | within(R, expr)    | within_ca(R, expr)
             | layer(core|boundary|surface)

Selectors are evaluated against the node table and geometry from :mod:`atomfrust.graph`,
**not** against a Rosetta pose. That keeps the whole module PyRosetta-free and unit-testable,
and it means a selection can be resolved and inspected before any pose exists. Step C3
converts the resulting node set to a ``PackerTask`` via ``pose_resnum``.

:class:`ResolvedRegions` enforces the three invariants as hard errors, not warnings: a
mutated residue that is not repacked would keep a rotamer belonging to its old identity, and
a minimised residue outside the repack set would move without being re-packed. Both are
silent corruption, so they are refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from atomfrust.graph import Geometry, Node

__all__ = [
    "SelectorError",
    "select",
    "ResolvedRegions",
    "resolve_regions",
    "LAYER_NEIGHBOUR_CUTOFFS",
]

#: Neighbour counts separating layers, on the metric actually computed below: the number of
#: other residue centres (Cbeta, or Calpha for glycine) within :data:`LAYER_RADIUS_A`.
#:
#: **These are not Rosetta's LayerSelector numbers and must not be confused with them.** An
#: earlier version of this module took Rosetta's `use_sidechain_neighbors` defaults (core
#: 5.2, surface 2.0) and multiplied them by 4 to fit this metric. That was an unjustified
#: fudge: Rosetta's figure is a *cone-weighted* count that discounts neighbours the side
#: chain does not point at, so it lives on a different scale entirely. The consequence was
#: measurable — `layer(surface)` selected 2 of 129 lysozyme residues, leaving the strict
#: core-versus-surface comparison in validation case F1 with no denominator.
#:
#: Calibrated instead against the real distribution: over 5GMP and 1XKK the count has
#: median 17, 10th percentile 9 and 90th percentile 24, and these cutoffs give roughly a
#: 30/40/30 core/boundary/surface split on both. Anything depending on the exact split
#: should say so, because this is a threshold on a proxy, not a physical boundary.
LAYER_NEIGHBOUR_CUTOFFS = {"core": 21, "surface": 12}
LAYER_RADIUS_A = 10.0


class SelectorError(ValueError):
    """A selector expression is malformed or names something that does not exist."""


# ------------------------------------------------------------------ tokenizer

# `range` must precede `number`, or "10-20" would lex as 10 followed by an unlexable
# "-20". Negative residue numbering is supported on both ends.
_TOKEN = re.compile(
    r"""\s*(?:
        (?P<lparen>\()
      | (?P<rparen>\))
      | (?P<comma>,)
      | (?P<xmlref>xml:[^\s()]+)
      | (?P<range>-?\d+\s*-\s*-?\d+)
      | (?P<number>-?\d+(?:\.\d+)?)
      | (?P<word>[A-Za-z_][A-Za-z0-9_:]*)
    )""",
    re.VERBOSE,
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        match = _TOKEN.match(text, position)
        if not match or match.end() == position:
            raise SelectorError(f"cannot parse {text!r} at offset {position}")
        kind = match.lastgroup or ""
        tokens.append((kind, match.group(kind)))
        position = match.end()
    return tokens


# --------------------------------------------------------------------- parser

Predicate = Callable[[Sequence["Node"], "Geometry"], np.ndarray]


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        if self.pos >= len(self.tokens):
            raise SelectorError("unexpected end of expression")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, kind: str, value: str | None = None) -> str:
        got_kind, got_value = self.take()
        if got_kind != kind or (value is not None and got_value.lower() != value):
            raise SelectorError(f"expected {value or kind}, got {got_value!r}")
        return got_value

    def parse(self) -> Predicate:
        predicate = self.parse_or()
        if self.peek() is not None:
            raise SelectorError(f"trailing input at token {self.peek()[1]!r}")
        return predicate

    def parse_or(self) -> Predicate:
        left = self.parse_and()
        while (token := self.peek()) and token[0] == "word" and token[1].lower() == "or":
            self.take()
            right = self.parse_and()
            left = _binary(left, right, np.logical_or)
        return left

    def parse_and(self) -> Predicate:
        left = self.parse_factor()
        while (token := self.peek()) and token[0] == "word" and token[1].lower() == "and":
            self.take()
            right = self.parse_factor()
            left = _binary(left, right, np.logical_and)
        return left

    def parse_factor(self) -> Predicate:
        token = self.peek()
        if token is None:
            raise SelectorError("unexpected end of expression")
        if token[0] == "word" and token[1].lower() == "not":
            self.take()
            inner = self.parse_factor()
            return lambda nodes, geom: ~inner(nodes, geom)
        if token[0] == "lparen":
            self.take()
            inner = self.parse_or()
            self.expect("rparen")
            return inner
        return self.parse_primary()

    def parse_primary(self) -> Predicate:
        kind, value = self.take()
        if kind == "xmlref":
            raise SelectorError(
                f"the xml: escape hatch is not implemented ({value!r}). Use the named "
                "selectors — chain / resi / resn / within / layer / protein / ligand / "
                "water — combined with and/or/not."
            )
        if kind != "word":
            raise SelectorError(f"expected a selector name, got {value!r}")
        name = value.lower()

        if name == "all":
            return lambda nodes, geom: np.ones(len(nodes), bool)
        if name == "none":
            return lambda nodes, geom: np.zeros(len(nodes), bool)
        if name == "protein":
            return lambda nodes, geom: np.array([n.kind == "protein" for n in nodes])
        if name == "noncanonical":
            return lambda nodes, geom: np.array([n.kind == "noncanonical" for n in nodes])
        if name == "ligand":
            # "ligand" means every non-protein component: a metal or cofactor is as much a
            # part of the site as an inhibitor, and excluding them here would silently
            # shrink every shell built around them.
            return lambda nodes, geom: np.array(
                [n.kind not in ("protein", "noncanonical") for n in nodes]
            )
        if name == "water":
            return lambda nodes, geom: np.array([n.resname in ("HOH", "WAT", "TP3") for n in nodes])
        if name == "chain":
            chains = {c.upper() for c in self._name_list()}
            return lambda nodes, geom: np.array([n.chain.upper() in chains for n in nodes])
        if name == "resn":
            names = {c.upper() for c in self._name_list()}
            return lambda nodes, geom: np.array([n.resname.upper() in names for n in nodes])
        if name == "resi":
            ranges = self._ranges()
            return lambda nodes, geom: np.array(
                [any(lo <= n.resseq <= hi for lo, hi in ranges) for n in nodes]
            )
        if name in ("within", "within_ca"):
            self.expect("lparen")
            radius = float(self.expect("number"))
            self.expect("comma")
            inner = self.parse_or()
            self.expect("rparen")
            if name == "within":
                return lambda nodes, geom: _within(radius, inner(nodes, geom), geom)
            return lambda nodes, geom: _within_ca(radius, inner(nodes, geom), geom)
        if name == "layer":
            self.expect("lparen")
            layer = self.expect("word").lower()
            self.expect("rparen")
            if layer not in ("core", "boundary", "surface"):
                raise SelectorError(f"unknown layer {layer!r}")
            return lambda nodes, geom: _layer(layer, nodes, geom)
        raise SelectorError(f"unknown selector {value!r}")

    def _name_list(self) -> list[str]:
        names = [self.expect("word")]
        while (token := self.peek()) and token[0] == "comma":
            self.take()
            names.append(self.expect("word"))
        return names

    def _ranges(self) -> list[tuple[int, int]]:
        """``10-20,45,-3`` -> [(10, 20), (45, 45), (-3, -3)]."""
        ranges: list[tuple[int, int]] = []
        while True:
            kind, value = self.take()
            if kind == "range":
                lo_text, hi_text = re.match(r"(-?\d+)\s*-\s*(-?\d+)", value).groups()
                lo, hi = int(lo_text), int(hi_text)
                if lo > hi:
                    raise SelectorError(f"inverted residue range {value!r}")
                ranges.append((lo, hi))
            elif kind == "number":
                if "." in value:
                    raise SelectorError(f"residue numbers must be integers, got {value!r}")
                ranges.append((int(value), int(value)))
            else:
                raise SelectorError(f"expected a residue number or range, got {value!r}")
            token = self.peek()
            if token and token[0] == "comma":
                self.take()
                continue
            return ranges


def _binary(left: Predicate, right: Predicate, op) -> Predicate:
    return lambda nodes, geom: op(left(nodes, geom), right(nodes, geom))


def _within(radius: float, mask: np.ndarray, geom: "Geometry") -> np.ndarray:
    """Residues with a heavy atom within ``radius`` of any heavy atom of the mask set."""
    from scipy.spatial import cKDTree

    out = np.zeros(len(mask), bool)
    if not mask.any():
        return out
    reference = geom.heavy_xyz[np.isin(geom.heavy_node, np.flatnonzero(mask))]
    if reference.size == 0:
        return out
    tree = cKDTree(reference)
    hit_atoms = tree.query_ball_point(geom.heavy_xyz, r=radius, return_length=True) > 0
    out[np.unique(geom.heavy_node[hit_atoms])] = True
    return out


def _within_ca(radius: float, mask: np.ndarray, geom: "Geometry") -> np.ndarray:
    """Residues whose **Ca** lies within ``radius`` of any heavy atom of the mask set.

    This is the prototype's pocket rule (``get_ligand_contacts``, ``frustration.py:111``)
    and exists so that definition can be reproduced exactly. It is a strictly narrower set
    than ``within`` at the same radius, since Ca is itself one of the heavy atoms measured
    there — a long side chain reaching into the site does not qualify its residue unless the
    backbone is close too.
    """
    from scipy.spatial import cKDTree

    out = np.zeros(len(mask), bool)
    if not mask.any():
        return out
    reference = geom.heavy_xyz[np.isin(geom.heavy_node, np.flatnonzero(mask))]
    if reference.size == 0:
        return out
    finite = np.flatnonzero(np.isfinite(geom.ca_xyz).all(axis=1))
    if finite.size == 0:
        return out
    tree = cKDTree(reference)
    hit = tree.query_ball_point(geom.ca_xyz[finite], r=radius, return_length=True) > 0
    out[finite[hit]] = True
    return out


def _layer(layer: str, nodes: Sequence["Node"], geom: "Geometry") -> np.ndarray:
    """Burial by neighbour count within :data:`LAYER_RADIUS_A`.

    Similar in spirit to Rosetta's LayerSelector but **not the same metric**: Rosetta cones
    the count toward the side-chain direction, this counts plainly. See
    :data:`LAYER_NEIGHBOUR_CUTOFFS` for why that distinction cost a validation case.
    """
    from scipy.spatial import cKDTree

    centres = np.where(
        np.isfinite(geom.cb_xyz).all(axis=1, keepdims=True), geom.cb_xyz, geom.ca_xyz
    )
    valid = np.isfinite(centres).all(axis=1)
    counts = np.zeros(len(nodes), dtype=float)
    if valid.sum() > 1:
        tree = cKDTree(centres[valid])
        neighbours = tree.query_ball_point(centres[valid], r=LAYER_RADIUS_A, return_length=True)
        counts[valid] = neighbours - 1  # exclude self

    core = counts >= LAYER_NEIGHBOUR_CUTOFFS["core"]
    surface = counts <= LAYER_NEIGHBOUR_CUTOFFS["surface"]
    if layer == "core":
        return core & valid
    if layer == "surface":
        return surface & valid
    return ~core & ~surface & valid


def select(expression: str, nodes: Sequence["Node"], geom: "Geometry") -> np.ndarray:
    """Evaluate a selector expression to a boolean mask over ``nodes``."""
    if not expression or not expression.strip():
        raise SelectorError("empty selector expression")
    predicate = _Parser(_tokenize(expression)).parse()
    mask = np.asarray(predicate(nodes, geom), dtype=bool)
    if mask.shape != (len(nodes),):
        raise SelectorError("selector produced a mask of the wrong shape")
    return mask


# ------------------------------------------------------------------- regions


@dataclass(frozen=True)
class ResolvedRegions:
    """Concrete residue sets for one decoy protocol, as boolean masks over the node list."""

    mutate: np.ndarray
    repack: np.ndarray
    minimize: np.ndarray

    @property
    def frozen(self) -> np.ndarray:
        """Everything touched by neither repacking nor minimisation."""
        return ~(self.repack | self.minimize)

    def counts(self) -> dict[str, int]:
        return {
            "mutate": int(self.mutate.sum()),
            "repack": int(self.repack.sum()),
            "minimize": int(self.minimize.sum()),
            "frozen": int(self.frozen.sum()),
        }

    def pose_resnums(self, nodes: Sequence["Node"], which: str) -> list[int]:
        mask = getattr(self, which) if which != "frozen" else self.frozen
        return [n.pose_resnum for n, keep in zip(nodes, mask) if keep]


def resolve_regions(
    nodes: Sequence["Node"],
    geom: "Geometry",
    mutate: str = "protein",
    repack: str = "protein",
    minimize: str = "protein",
    mutable_only: bool = True,
) -> ResolvedRegions:
    """Resolve three selector expressions, enforcing the subset invariants.

    ``mutable_only`` intersects the mutate set with nodes marked mutable, so a ligand, a
    metal or a covalent anchor can never be identity-randomised even if an expression
    names it. That is a safety net, not a substitute for writing the expression correctly.
    """
    mutate_mask = select(mutate, nodes, geom)
    repack_mask = select(repack, nodes, geom)
    minimize_mask = select(minimize, nodes, geom)

    if mutable_only:
        allowed = np.array([n.mutable for n in nodes], dtype=bool)
        blocked = mutate_mask & ~allowed
        mutate_mask = mutate_mask & allowed
    else:
        blocked = np.zeros(len(nodes), bool)

    problems = []
    if (extra := mutate_mask & ~repack_mask).any():
        problems.append(
            f"mutate is not a subset of repack: {_examples(nodes, extra)} would take a new "
            "identity while keeping the old identity's rotamer"
        )
    if (extra := minimize_mask & ~repack_mask).any():
        problems.append(
            f"minimize is not a subset of repack: {_examples(nodes, extra)} would move "
            "without being re-packed"
        )
    if problems:
        raise SelectorError("; ".join(problems))

    regions = ResolvedRegions(mutate_mask, repack_mask, minimize_mask)
    if blocked.any():
        # Not an error: `protein` is the natural expression and a covalent anchor is
        # legitimately excluded from it. Worth knowing about, so it is recorded.
        object.__setattr__(regions, "_blocked", _examples(nodes, blocked))
    return regions


def _examples(nodes: Sequence["Node"], mask: np.ndarray, limit: int = 3) -> str:
    ids = [n.node_id for n, keep in zip(nodes, mask) if keep]
    shown = ", ".join(ids[:limit])
    return f"{shown}{'' if len(ids) <= limit else f' (+{len(ids) - limit} more)'}"
