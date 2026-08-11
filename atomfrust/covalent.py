"""Covalent ligands: identifiable and constrained, not silently mis-scored.

Fifteen of the 61 EGFR complexes carry an acrylamide warhead bonded to CYS797-SG. The
prototype scores all 61 identically: ``scripts/04_prepare_complex.py`` extracts the linkage
from the mmCIF ``_struct_conn`` category (``get_covalent_bond_info``, lines 100-126) and
writes it to ``results/preparation_summary.csv``, and nothing downstream reads it. Stage 6
therefore mutates the anchor cysteine like any other residue and treats the ligand as a
free binder, so those fifteen numbers are pooled with the other 46 without a marker.

**Full covalent chemistry is deferred** (plan §7): a real Rosetta bond, a patched residue
type and correct valence are a large chemistry task. This module does the part that
unblocks the science, which is three narrow things:

* the anchor residue is **frozen** — ``mutable=False``, ``frozen_reason="covalent_anchor"``
  — so no decoy can mutate away the chemistry that holds the ligand in place;
* the anchor–ligand pair is **forced into the contact graph** via ``build_superset``'s
  ``bonds=`` argument, which exempts it from distance and sequence-separation filters. A
  bonded pair at 1.8 Å would survive any cutoff anyway; forcing it makes the pair's presence
  a property of the chemistry rather than of the currently selected shell;
* covalent complexes get a **reporting stratum**, so they are never pooled silently.

Nothing here writes a CONNECT record. ``scripts/05_prepare_ligand.py``'s docstring (lines
19-22) claims it adds one to the ``.params`` file; that is false — no code writes it, no
``.params`` on disk contains one, and the ``molfile_to_params`` invocation
(``src/prepare_structures.py:332-358``) passes only ``-n``, ``--keep-names`` and
``--clobber``. The claim is a documentation defect, recorded here because a reader looking
for the covalent bond will otherwise go hunting for it in the params.

Two parsers, because the linkage is available from two places and the mmCIF is not always
on disk (``data/raw_cif/`` is re-downloadable and excluded from DVC):
:func:`anchors_from_struct_conn` reads the authoritative source, and
:func:`anchors_from_preparation_summary` recovers the same anchor from the Stage-4 output
the prototype already produces.

No PyRosetta here — text and tables in, :class:`~atomfrust.spec.CovalentAnchor` objects out.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from atomfrust.spec import CovalentAnchor, SystemSpec, WATER_NAMES, _is_standard_residue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from atomfrust.graph import Node

__all__ = [
    "COVALENT_FROZEN_REASON",
    "anchors_from_struct_conn",
    "anchors_from_preparation_summary",
    "bonds_for_graph",
    "apply_covalent_constraints",
    "covalent_stratum",
]

#: The single spelling of the freeze marker. ``atomfrust.pose.load_complex`` writes this
#: string when it sees a spec anchor; anything comparing against it must use this name.
COVALENT_FROZEN_REASON = "covalent_anchor"

#: ``_struct_conn.conn_type_id`` values that are a covalent link to a ligand. ``covale_base``
#: and ``covale_sugar`` are nucleic-acid and glycan variants of the same bond; ``disulf``,
#: ``metalc`` and ``hydrog`` are deliberately absent — a disulfide is not a ligand anchor.
_COVALENT_CONN_TYPES = frozenset({"covale", "covale_base", "covale_phosphate", "covale_sugar"})

#: mmCIF's two spellings of "no value".
_MISSING = frozenset({"?", ".", ""})

#: ``covalent_protein_residue`` in the Stage-4 summary is a concatenation, e.g. ``CYS797``.
_RESIDUE_LABEL = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]{0,2}?)\s*(-?\d+)\s*([A-Za-z]?)\s*$")


# --------------------------------------------------------------------------- mmCIF


def _cif_value(text: object) -> str:
    """One ``_struct_conn`` field, with mmCIF's null spellings collapsed to ``""``."""
    value = str(text).strip().strip("'\"")
    return "" if value in _MISSING else value


def anchors_from_struct_conn(
    cif_path: Path | str, ligand_comp_ids: Iterable[str] | None = None
) -> list[CovalentAnchor]:
    """Parse mmCIF ``_struct_conn`` for covalent linkages between a ligand and the receptor.

    The authoritative source: the depositor states the bond, so it needs no distance
    heuristic. Returns one anchor per ``covale`` row that joins a receptor residue to a
    component, in file order, deduplicated. A file with no such row yields ``[]`` — which is
    the answer for 46 of the 61 EGFR structures and must not be an error.

    ``ligand_comp_ids`` names which side is the ligand. When omitted, the ligand side is
    whichever partner is not a standard residue: that resolves the acrylamide case, and it
    deliberately drops rows where *both* partners are standard (a disulfide or a peptide
    link deposited as ``covale`` is not a ligand anchor) and rows where *neither* is (a
    glycan-to-glycan bond has no receptor side).

    Chain, residue number and insertion code are taken from the ``auth`` fields, because
    those are what a PDB file — and therefore ``PDBInfo`` and every node id — carries. Atom
    names come from ``label_atom_id``, the only atom-name field ``_struct_conn`` has.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    cif_path = Path(cif_path)
    data = MMCIF2Dict(str(cif_path))

    conn_types = data.get("_struct_conn.conn_type_id")
    if not conn_types:
        return []

    wanted = {c.strip().upper() for c in ligand_comp_ids} if ligand_comp_ids else None

    anchors: list[CovalentAnchor] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for row in range(len(conn_types)):
        if _cif_value(conn_types[row]).lower() not in _COVALENT_CONN_TYPES:
            continue

        partners = [_partner(data, side, row) for side in (1, 2)]
        if any(p is None for p in partners):
            continue

        ligand_side = _ligand_side(partners, wanted)
        if ligand_side is None:
            continue
        ligand = partners[ligand_side]
        receptor = partners[1 - ligand_side]

        anchor = CovalentAnchor(
            chain=receptor["chain"],
            resseq=receptor["resseq"],
            icode=receptor["icode"],
            atom=receptor["atom"],
            ligand_atom=ligand["atom"],
        )
        key = (*anchor.key(), anchor.atom, anchor.ligand_atom)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(anchor)

    return anchors


def _partner(data: Mapping[str, Any], side: int, row: int) -> dict[str, Any] | None:
    """One side of a ``_struct_conn`` row, or ``None`` if it is not addressable."""
    def field(name: str, default: str = "") -> str:
        values = data.get(name)
        if not values or row >= len(values):
            return default
        return _cif_value(values[row])

    chain = field(f"_struct_conn.ptnr{side}_auth_asym_id")
    resseq = field(f"_struct_conn.ptnr{side}_auth_seq_id")
    comp = field(f"_struct_conn.ptnr{side}_auth_comp_id")
    atom = field(f"_struct_conn.ptnr{side}_label_atom_id")
    icode = field(f"_struct_conn.pdbx_ptnr{side}_PDB_ins_code")
    if not chain or not atom or not comp:
        return None
    try:
        number = int(resseq)
    except ValueError:
        return None
    return {
        "chain": chain,
        "resseq": number,
        "icode": icode[:1],
        "comp_id": comp.upper(),
        "atom": atom,
    }


def _ligand_side(partners: Sequence[Mapping[str, Any]], wanted: set[str] | None) -> int | None:
    """Index of the ligand partner, or ``None`` when the row is not a ligand anchor."""
    if wanted is not None:
        matched = [k for k, p in enumerate(partners) if p["comp_id"] in wanted]
        # Exactly one side must be the ligand: two matches is a ligand-ligand bond with no
        # receptor side, zero is a link between two residues the caller did not ask about.
        return matched[0] if len(matched) == 1 else None
    component = [
        k
        for k, p in enumerate(partners)
        if not _is_standard_residue(p["comp_id"]) and p["comp_id"] not in WATER_NAMES
    ]
    return component[0] if len(component) == 1 else None


# ------------------------------------------------------------------- Stage-4 CSV


def anchors_from_preparation_summary(
    csv_path: Path | str, pdb_id: str
) -> list[CovalentAnchor]:
    """Recover anchors from the Stage-4 output the prototype already produces.

    ``results/preparation_summary.csv`` carries the linkage in four columns:
    ``is_covalent`` (bool), ``covalent_protein_residue`` (``CYS797`` — name and number
    concatenated, no chain), ``covalent_protein_atom`` (``SG``) and ``covalent_ligand_atom``
    (``CAR``). One row per PDB id, so this returns zero or one anchor.

    **The chain comes from ``egfr_chain``, not from the linkage.** Stage 4 reads a chain from
    ``_struct_conn`` but does not write it to the summary; ``egfr_chain`` is the chain it
    kept in ``data/processed/{PDB}_clean.pdb``, which is the file a spec points at and the
    only chain a pose built from it has. Recovering the anchor's chain any other way would
    name a chain that is not in the structure. Use :func:`anchors_from_struct_conn` on the
    mmCIF when the deposited chain matters.
    """
    csv_path = Path(csv_path)
    table = pd.read_csv(csv_path, dtype={"pdb_id": str})
    rows = table.loc[table["pdb_id"].astype(str).str.upper() == str(pdb_id).upper()]
    if rows.empty:
        raise KeyError(
            f"{pdb_id!r} is not in {csv_path.name}; it has {len(table)} rows "
            f"({', '.join(map(str, table['pdb_id'].head(5)))}, ...)"
        )

    row = rows.iloc[0]
    if not _is_true(row.get("is_covalent")):
        return []

    label = str(row.get("covalent_protein_residue", "") or "")
    match = _RESIDUE_LABEL.match(label)
    if match is None:
        raise ValueError(
            f"{pdb_id}: cannot read a residue number out of covalent_protein_residue "
            f"{label!r}; expected a name and number, e.g. 'CYS797'"
        )
    _resname, resseq, icode = match.groups()

    chain = str(row.get("egfr_chain", "") or "").strip()
    if not chain:
        raise ValueError(f"{pdb_id}: is_covalent is set but egfr_chain is empty")

    return [
        CovalentAnchor(
            chain=chain,
            resseq=int(resseq),
            icode=icode,
            atom=str(row["covalent_protein_atom"]).strip(),
            ligand_atom=str(row["covalent_ligand_atom"]).strip(),
        )
    ]


def _is_true(value: object) -> bool:
    """Truthiness for a CSV cell, where ``"False"`` is a non-empty string."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "t", "y"}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(value)


# ----------------------------------------------------------------------- graph


def bonds_for_graph(spec: SystemSpec, nodes: Sequence["Node"]) -> tuple[tuple[str, str], ...]:
    """Node-id pairs to force into the contact graph, from the spec's covalent anchors.

    Feeds ``build_superset(..., bonds=...)``, which adds each pair to the candidate set and
    exempts it from the sequence-separation filter, and ``add_contact_definition``, which
    marks any ``is_bonded`` pair as a contact regardless of distance. So the anchor edge is
    present under every contact definition and cutoff a later analysis selects.

    Both endpoints are resolved by PDB position against the node list rather than trusted
    from the spec: ``build_superset`` raises a bare ``KeyError`` on an unknown node id, and
    an anchor naming a chain that was dropped at load time is a spec error worth naming.
    """
    by_key = {(n.chain, n.resseq, n.icode.strip()): n.node_id for n in nodes}

    bonds: list[tuple[str, str]] = []
    for ligand in spec.ligands:
        anchor = ligand.covalent_anchor
        if anchor is None:
            continue
        anchor_id = by_key.get(anchor.key())
        ligand_id = by_key.get(ligand.selector.key())
        if anchor_id is None:
            raise ValueError(
                f"system {spec.system_id!r}: covalent anchor "
                f"{anchor.chain}:{anchor.resseq}{anchor.icode} is not a node — the residue "
                "is absent from the structure or its chain was dropped at load time"
            )
        if ligand_id is None:
            raise ValueError(
                f"system {spec.system_id!r}: ligand {ligand.selector} is not a node, so its "
                "covalent anchor cannot be bonded to anything"
            )
        if anchor_id == ligand_id:
            raise ValueError(
                f"system {spec.system_id!r}: covalent anchor and ligand are the same "
                f"residue ({anchor_id}); a residue cannot be bonded to itself"
            )
        pair = (anchor_id, ligand_id)
        if pair not in bonds:
            bonds.append(pair)

    return tuple(bonds)


def apply_covalent_constraints(
    nodes: Sequence["Node"], anchors: Iterable[CovalentAnchor] = ()
) -> list["Node"]:
    """Return nodes with the anchor residue marked ``mutable=False``,
    ``frozen_reason='covalent_anchor'``.

    An anchor cysteine that mutates to alanine in a decoy leaves the ligand bonded to
    nothing, so the decoy no longer differs from the native by sequence alone — which is the
    single assumption the Z-score rests on. ``resolve_regions(mutable_only=True)`` intersects
    the mutate set with ``mutable``, so freezing the node here is enough to keep it out of
    every generator's target set.

    ``anchors`` is optional because ``atomfrust.pose.load_complex`` already applies the
    spec's anchors while building nodes. Called with no anchors this re-enforces whatever
    markers the nodes carry — a node tagged ``covalent_anchor`` but still ``mutable`` is the
    one inconsistency that would defeat the freeze silently. Idempotent either way.

    :class:`~atomfrust.graph.Node` is frozen, so this returns new nodes; the input list is
    untouched.
    """
    keys = {a.key() for a in anchors}
    out: list["Node"] = []
    for node in nodes:
        is_anchor = (node.chain, node.resseq, node.icode.strip()) in keys
        if not is_anchor and node.frozen_reason != COVALENT_FROZEN_REASON:
            out.append(node)
            continue
        out.append(
            dataclasses.replace(
                node, mutable=False, frozen_reason=COVALENT_FROZEN_REASON
            )
        )
    return out


# ------------------------------------------------------------------- reporting


def covalent_stratum(
    table: pd.DataFrame, spec_by_id: Mapping[str, SystemSpec]
) -> pd.Series:
    """Boolean per system — the reporting stratum, so covalent complexes are never pooled
    silently.

    A covalent ligand's affinity is not the same measurement as a reversible one: the
    residence time is chemistry, not equilibrium binding, and the frustration index computed
    without the bond is a different estimand again while full covalent chemistry is deferred.
    Pooling the 15 with the 46 is therefore a stratification error, and the point of this
    function is that the split is available to every report as one column.

    Resolution order per row, by ``system_id``: the spec's anchors
    (:attr:`~atomfrust.spec.SystemSpec.is_covalent`), then its ``labels["is_covalent"]``,
    then an ``is_covalent`` column on the table itself. A system with no spec and no column
    is ``False`` — "not known to be covalent", which is what an unlabelled system is; the
    label is what makes the stratum, and inventing ``NA`` here would only push the same
    decision downstream.

    Returns a boolean Series named ``is_covalent``, aligned to ``table``'s index.
    """
    if "system_id" not in table.columns:
        raise KeyError(
            "covalent_stratum joins on 'system_id'; the table has "
            f"[{', '.join(map(str, table.columns[:8]))}]"
        )

    fallback = table["is_covalent"] if "is_covalent" in table.columns else None

    values = []
    for position, system_id in enumerate(table["system_id"]):
        spec = spec_by_id.get(system_id)
        if spec is not None:
            values.append(bool(spec.is_covalent) or _is_true(spec.labels.get("is_covalent")))
        elif fallback is not None:
            values.append(_is_true(fallback.iloc[position]))
        else:
            values.append(False)

    return pd.Series(values, index=table.index, dtype=bool, name="is_covalent")
