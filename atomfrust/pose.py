"""Pose loading and node identity.

Turns a :class:`~atomfrust.spec.SystemSpec` into a Rosetta pose plus the typed node list and
geometry that :mod:`atomfrust.graph` consumes.

Two prototype defects are fixed here:

**Identity is a PDB position.** ``find_ligand_resnum`` (``run_pipeline.py:186``) scanned for
the first residue whose ``name3()`` matched a code prefix and returned it. Two copies of a
ligand collapsed to one, a cofactor was invisible, and a code that happened to prefix-match
something else silently picked the wrong residue. Here every node is keyed by
``(chain, resseq, icode)`` from ``PDBInfo``, which is unique.

**Components are typed, not assumed.** ``is_protein()`` was the only distinction the
prototype drew. A pose can also hold metals, cofactors, nucleic acids and non-canonical
residues, all of which are nodes in the contact graph and some of which must never be
mutated.

This is the only module in the package that imports PyRosetta, and it does so lazily so the
``unit`` test tier stays importable on a machine that has never installed it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from atomfrust.graph import Geometry, Node, NodeKind
from atomfrust.mutate import AppliedMutation, apply_mutations
from atomfrust.spec import SystemSpec

if TYPE_CHECKING:  # pragma: no cover
    from pyrosetta import Pose

__all__ = ["Component", "LoadedComplex", "load_complex", "classify_residue", "ensure_init"]

_INITIALISED = False


#: ``-in:file:load_PDB_components false`` is not cosmetic. With it on (Rosetta's default) a
#: HETATM whose ``.params`` was never supplied is silently typed from the bundled PDB
#: Chemical Component Dictionary: the pose loads, the ligand appears as a node, and nothing
#: reports that the curated parametrisation was bypassed. A run would then use CCD-derived
#: atom typing while its manifest claimed otherwise. Off, an unparametrised component fails
#: loudly at load time, which is the only moment it is cheap to diagnose.
DEFAULT_INIT_FLAGS = "-mute all -in:file:load_PDB_components false"


def ensure_init(extra_flags: str = "") -> None:
    """Initialise PyRosetta once per process. Safe to call repeatedly.

    PyRosetta can only be initialised once, so the first caller's flags win. Workers call
    this in their initializer; nothing else should call ``pyrosetta.init`` directly.
    """
    global _INITIALISED
    import pyrosetta

    if not _INITIALISED:
        pyrosetta.init(f"{DEFAULT_INIT_FLAGS} {extra_flags}".strip())
        _INITIALISED = True


@dataclass(frozen=True)
class Component:
    """A non-protein component, resolved across the spec, the params and the pose.

    Holds the ``comp_id`` / ``rosetta_name`` / params triple in one place so no call site has
    to guess which name it needs. ``comp_id`` is for reporting; ``rosetta_name`` is what the
    pose calls the residue.
    """

    node_id: str
    pose_resnum: int
    kind: NodeKind
    comp_id: str | None
    rosetta_name: str
    params_path: Path | None
    params_sha256: str | None


@dataclass
class LoadedComplex:
    pose: Any
    nodes: list[Node]
    geometry: Geometry
    components: list[Component]
    spec: SystemSpec
    #: What ``receptor.mutations`` actually did, read off the pose rather than restated from
    #: the spec — empty for a wild-type system. See :mod:`atomfrust.mutate`.
    mutations: tuple[AppliedMutation, ...] = ()

    @property
    def n_protein(self) -> int:
        return sum(1 for n in self.nodes if n.kind == "protein")

    @property
    def n_components(self) -> int:
        return len(self.components)

    def node_by_key(self, chain: str, resseq: int, icode: str = "") -> Node:
        for n in self.nodes:
            if (n.chain, n.resseq, n.icode) == (chain, resseq, icode.strip()):
                return n
        raise KeyError(f"no node at {chain}:{resseq}{icode}")


def classify_residue(residue: Any) -> NodeKind:
    """Type a Rosetta residue. Order matters: metals and nucleics before the ligand catch-all."""
    if residue.is_protein():
        # Rosetta counts non-canonical backbones as protein; separate them so they can be
        # excluded from identity randomisation without excluding them from the graph.
        return "protein" if residue.type().is_canonical_aa() else "noncanonical"
    if residue.is_metal():
        return "metal"
    if residue.is_DNA() or residue.is_RNA() or residue.is_NA():
        return "nucleic"
    return "ligand"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _node_id(chain: str, resseq: int, icode: str) -> str:
    """``A:745``, or ``A:52A`` with an insertion code. Unique because a PDB position is."""
    return f"{chain}:{resseq}{icode.strip()}"


def load_complex(
    spec: SystemSpec,
    *,
    include_water: bool = False,
    validate: bool = True,
) -> LoadedComplex:
    """Load a pose from a spec and build its node list and geometry.

    Every ``.params`` named by the spec is registered, so a system with several distinct
    components loads in one pass. Waters are dropped by default: treating them as nodes is
    deferred, and the published method does not model them.

    ``spec.receptor.mutations`` are applied here, after the pose is built and before the
    node list, and recorded on :attr:`LoadedComplex.mutations`.
    """
    import pyrosetta
    from pyrosetta import Pose, Vector1

    if spec.receptor.path is None:
        raise ValueError(
            f"system {spec.system_id!r} has no local receptor path; "
            "fetching by pdb_id is not implemented yet"
        )
    pdb_path = Path(spec.receptor.path)
    if validate:
        spec.validate_against_structure(pdb_path)

    ensure_init()

    # Deduplicate by resolved path. N copies of one ligand share a single residue *type*:
    # handing Rosetta the same .params twice raises
    #   "residue type 'F62' already exists in the cache".
    # The type is registered once; each copy instantiates from it.
    params: list[str] = []
    seen_params: set[Path] = set()
    for lig in spec.ligands:
        if lig.params is None:
            continue
        resolved = Path(lig.params).resolve()
        if resolved in seen_params:
            continue
        seen_params.add(resolved)
        params.append(str(lig.params))

    pose = Pose()
    try:
        if params:
            residue_set = pyrosetta.generate_nonstandard_residue_set(pose, Vector1(params))
            pyrosetta.pose_from_file(pose, residue_set, str(pdb_path))
        else:
            pyrosetta.pose_from_file(pose, str(pdb_path))
    except RuntimeError as exc:
        # Rosetta exits with a bare "Unrecognized residue: XXX" and no indication of which
        # system or which spec entry is at fault. With load_PDB_components off (see
        # DEFAULT_INIT_FLAGS) this is the expected failure for an unparametrised component,
        # so it is worth naming the cause and the fix.
        named = {
            lig.selector.comp_id
            for lig in spec.ligands
            if lig.params is None and lig.selector.comp_id
        }
        hint = (
            f" Component(s) {', '.join(sorted(named))} have no `params` in the spec; "
            "supply a .params file for each, or run the parametrisation step first."
            if named
            else ""
        )
        raise ValueError(
            f"system {spec.system_id!r}: Rosetta could not build a pose from "
            f"{pdb_path.name}.{hint} Original error: {exc}"
        ) from exc

    params_by_key = {
        lig.selector.key(): lig for lig in spec.ligands
    }

    info = pose.pdb_info()
    if info is None:
        raise ValueError(f"{pdb_path} loaded without PDBInfo; cannot key nodes by position")

    # Before the node list, so `resname`/`name1` and every distance downstream describe the
    # mutant. A mutation applied after this point would leave the graph describing a
    # structure the pose no longer is. See atomfrust.mutate for why this is a spec field.
    applied = apply_mutations(pose, spec.receptor.mutations, info)

    keep_chains = set(spec.receptor.chains) if spec.receptor.chains else None

    nodes: list[Node] = []
    components: list[Component] = []
    heavy_xyz: list[np.ndarray] = []
    heavy_node: list[int] = []
    ca_xyz: list[np.ndarray] = []
    cb_xyz: list[np.ndarray] = []

    for resnum in range(1, pose.total_residue() + 1):
        residue = pose.residue(resnum)
        if residue.is_virtual_residue():
            continue
        if residue.name3().strip() in {"HOH", "TP3", "WAT"} and not include_water:
            continue

        chain = info.chain(resnum).strip()
        resseq = int(info.number(resnum))
        icode = info.icode(resnum).strip()
        if keep_chains is not None and chain not in keep_chains:
            continue

        kind = classify_residue(residue)
        key = (chain, resseq, icode)
        node_id = _node_id(chain, resseq, icode)
        lig = params_by_key.get(key)

        params_path = Path(lig.params) if (lig and lig.params) else None
        component: Component | None = None
        if kind != "protein":
            component = Component(
                node_id=node_id,
                pose_resnum=resnum,
                kind=kind,
                comp_id=(lig.selector.comp_id if lig else residue.name3().strip()),
                rosetta_name=residue.name3().strip(),
                params_path=params_path,
                params_sha256=_sha256(params_path) if params_path else None,
            )
            components.append(component)

        # A covalent anchor must never be mutated: its identity is load-bearing chemistry.
        frozen_reason = None
        for other in spec.ligands:
            anchor = other.covalent_anchor
            if anchor is not None and anchor.key() == key:
                frozen_reason = "covalent_anchor"
                break

        n_heavy = 0
        node_index = len(nodes)
        for atom_index in range(1, residue.natoms() + 1):
            if residue.is_virtual(atom_index):
                continue
            if residue.atom_type(atom_index).element().strip() == "H":
                continue
            xyz = residue.xyz(atom_index)
            heavy_xyz.append(np.array([xyz.x, xyz.y, xyz.z]))
            heavy_node.append(node_index)
            n_heavy += 1

        ca_xyz.append(_atom_or_nan(residue, "CA"))
        cb_xyz.append(_atom_or_nan(residue, "CB"))

        nodes.append(
            Node(
                node_id=node_id,
                pose_resnum=resnum,
                kind=kind,
                chain=chain,
                resseq=resseq,
                icode=icode,
                resname=residue.name3().strip(),
                name1=residue.name1() if kind == "protein" else "X",
                ccd_id=component.comp_id if component else None,
                rosetta_name=component.rosetta_name if component else None,
                params_sha256=component.params_sha256 if component else None,
                mutable=(kind == "protein" and frozen_reason is None),
                frozen_reason=frozen_reason,
                n_heavy=n_heavy,
            )
        )

    geometry = Geometry(
        heavy_xyz=np.asarray(heavy_xyz, dtype=float).reshape(-1, 3),
        heavy_node=np.asarray(heavy_node, dtype=np.int64),
        ca_xyz=np.asarray(ca_xyz, dtype=float).reshape(-1, 3),
        cb_xyz=np.asarray(cb_xyz, dtype=float).reshape(-1, 3),
    )

    _check_node_ids_unique(nodes, pdb_path)
    _check_components_resolved(spec, components, pdb_path)
    return LoadedComplex(
        pose=pose,
        nodes=nodes,
        geometry=geometry,
        components=components,
        spec=spec,
        mutations=applied,
    )


def _atom_or_nan(residue: Any, name: str) -> np.ndarray:
    if not residue.has(name):
        return np.full(3, np.nan)
    xyz = residue.xyz(name)
    return np.array([xyz.x, xyz.y, xyz.z])


def _check_components_resolved(
    spec: SystemSpec, components: list[Component], pdb_path: Path
) -> None:
    """Every ligand the spec names must have become a node.

    Silence here is how the prototype produced "ligand not found in pose" much later and far
    from the cause.
    """
    found = {c.node_id for c in components}
    missing = [
        str(lig.selector)
        for lig in spec.ligands
        if _node_id(lig.selector.chain, lig.selector.resseq, lig.selector.icode)
        not in found
    ]
    if missing:
        raise ValueError(
            f"system {spec.system_id!r}: ligand(s) {', '.join(missing)} named by the spec "
            f"did not load as components from {pdb_path.name}. "
            f"Components found: {', '.join(sorted(found)) or '(none)'}. "
            "A missing .params entry is the usual cause."
        )


def _check_node_ids_unique(nodes: list[Node], pdb_path: Path) -> None:
    """``node_id`` is the join key between the node and pair tables, so a collision would
    silently corrupt the graph rather than fail. A well-formed PDB cannot produce one —
    ``(chain, resseq, icode)`` is unique per model — so this catches malformed input."""
    from collections import Counter

    duplicates = [k for k, n in Counter(n.node_id for n in nodes).items() if n > 1]
    if duplicates:
        raise ValueError(
            f"{pdb_path.name}: duplicate node ids {', '.join(sorted(duplicates))}. "
            "Two residues share a (chain, resseq, icode) position, which a valid PDB "
            "cannot do; renumber the offending residues before loading."
        )
