"""System specification — the target-agnostic input.

One hand-writable YAML replaces the three-CSV EGFR join at ``run_pipeline.py:140-164``.
Nothing here knows what EGFR is: a system is a receptor, zero or more non-protein
components, and a pocket definition.

```yaml
system_id: 5GMP_634
receptor:
  path: inputs/5GMP.pdb        # or pdb_id: 5GMP
  chains: [A]                  # omit => all protein chains
  mutations:                   # omit => wild type; a mutant is a different system
    - {chain: A, resseq: 790, to: MET}
ligands:                       # [] => protein-only / protein-protein
  - selector: {chain: A, resseq: 1001, comp_id: "634"}
    params: params/Z34.params  # optional; auto-parametrised if absent
    rosetta_name: Z34
pocket:
  mode: ligand_shell           # ligand_shell | residue_list | chain_interface | whole
  reference: any_heavy
  cutoff_A: 10.0
labels: {affinity_pM: 23.0}
```

Two invariants this file exists to hold:

* **The ``comp_id`` ↔ ``rosetta_name`` ↔ params triple lives in exactly one place.** A few
  real CCD codes collide with Rosetta's internal namespace (``634`` → ``Z34``); in the
  prototype the params filename and pose lookup used one name while reporting used the
  other, and mixing them produced "ligand not found in pose".
* **Ligand identity is a PDB position, not a residue-name prefix.** The prototype's
  ``find_ligand_resnum`` (``run_pipeline.py:186``) matched ``name3()`` and returned the
  *first* hit, so N copies of a ligand collapsed to one and cofactors were invisible. A
  selector here is ``(chain, resseq, icode)``, which is unique.

Validation is two-phase on purpose. ``SystemSpec.model_validate`` checks shape only and
touches no files, so it stays unit-testable. :meth:`SystemSpec.validate_against_structure`
checks the spec against the actual coordinates and reports *every* problem at once with the
available chains and components listed — a spec naming a chain that is not there should say
which chains are.

No PyRosetta anywhere in this module.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "SpecError",
    "ResidueSelector",
    "CovalentAnchor",
    "Mutation",
    "Receptor",
    "LigandSpec",
    "PocketSpec",
    "SystemSpec",
    "SystemSet",
    "WATER_NAMES",
    "normalise_residue_code",
]

# Excluded from ligand autodetection. Everything else that is a HETATM becomes a node —
# metals and cofactors included, deliberately (they are part of the energy landscape).
WATER_NAMES = frozenset({"HOH", "DOD", "WAT", "H2O", "TIP", "TIP3", "SOL"})

_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
_THREE_TO_ONE = {three: one for one, three in _ONE_TO_THREE.items()}


def normalise_residue_code(code: str) -> str:
    """``'m'``, ``'M'``, ``'met'`` → ``'MET'``. Raises ``ValueError`` on anything else.

    Only the 20 canonical amino acids are accepted. A mutation target outside them is either
    a typo or a non-canonical chemistry that needs its own residue type, and both are worth
    a loud failure at spec-validation time rather than a Rosetta error at pose load.
    """
    text = str(code).strip().upper()
    if len(text) == 1:
        if text in _ONE_TO_THREE:
            return _ONE_TO_THREE[text]
        raise ValueError(
            f"unknown one-letter amino-acid code {code!r}; expected one of "
            f"{''.join(sorted(_ONE_TO_THREE))}"
        )
    if text in _THREE_TO_ONE:
        return text
    raise ValueError(
        f"unknown amino-acid code {code!r}; expected a one- or three-letter canonical code"
    )


class SpecError(ValueError):
    """A spec is structurally valid but inconsistent with the coordinates it names."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResidueSelector(_Base):
    """A residue's PDB position. ``(chain, resseq, icode)`` is unique within a model."""

    chain: str = Field(min_length=1, max_length=2)
    resseq: int
    icode: str = Field(default="", max_length=1)
    comp_id: str | None = Field(
        default=None,
        description="Expected CCD code. Checked against the file when given.",
    )

    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode.strip())

    def __str__(self) -> str:
        icode = self.icode.strip()
        base = f"{self.chain}:{self.resseq}{icode}"
        return f"{base} ({self.comp_id})" if self.comp_id else base

    @classmethod
    def parse(cls, text: str) -> ResidueSelector:
        """Parse ``CHAIN:RESSEQ`` or ``CHAIN:RESSEQICODE``, e.g. ``B:501`` or ``A:52A``."""
        if ":" not in text:
            raise ValueError(f"expected CHAIN:RESSEQ, got {text!r}")
        chain, _, rest = text.partition(":")
        rest = rest.strip()
        if not chain.strip():
            raise ValueError(f"empty chain in {text!r}")
        icode = ""
        if rest and rest[-1].isalpha():
            rest, icode = rest[:-1], rest[-1]
        if not rest.lstrip("-").isdigit():
            raise ValueError(f"expected an integer residue number in {text!r}")
        return cls(chain=chain.strip(), resseq=int(rest), icode=icode)


class CovalentAnchor(_Base):
    """A covalent linkage between a ligand atom and a receptor atom.

    Recorded so the anchor residue can be frozen and the edge forced. Full covalent
    chemistry is deferred; this makes covalent systems identifiable rather than silently
    mis-scored, which is the prototype's actual failure mode.
    """

    chain: str
    resseq: int
    icode: str = Field(default="", max_length=1)
    atom: str = Field(description="Receptor atom name, e.g. SG")
    ligand_atom: str

    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode.strip())


class Mutation(_Base):
    """A point substitution applied to the receptor at pose-load time.

    Plan step G8 / methods S4.5: pocket mutation is the *positive* control for a physically
    grounded measure — it must respond when the pocket is changed. The substitution lives in
    the spec, not in a runtime flag, because a mutant is a different system: the spec is
    digested into ``regeneration_key`` (``cli/generate_decoys.py:369-377``), so a stored
    wild-type ensemble can never be silently reused for a mutant.

    ``to`` accepts a one- or three-letter canonical code and is stored three-letter upper
    case, which is what ``MutateResidue`` wants and what ``Node.resname`` will read back.
    """

    chain: str = Field(min_length=1, max_length=2)
    resseq: int
    icode: str = Field(default="", max_length=1)
    to: str = Field(
        description="Target residue: 'M', 'met' and 'MET' all normalise to 'MET'."
    )

    @field_validator("to", mode="before")
    @classmethod
    def _normalise_target(cls, value: Any) -> Any:
        return normalise_residue_code(value) if isinstance(value, str) else value

    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode.strip())

    @property
    def to1(self) -> str:
        """The one-letter form of the target, for comparing against ``Node.name1``."""
        return _THREE_TO_ONE[self.to]

    def __str__(self) -> str:
        return f"{self.chain}:{self.resseq}{self.icode.strip()}:{self.to}"

    @classmethod
    def parse(cls, text: str) -> Mutation:
        """Parse ``CHAIN:RESSEQ:TO``, e.g. ``A:790:MET``, ``A:790:M`` or ``A:52A:TRP``."""
        parts = str(text).split(":")
        if len(parts) != 3:
            raise ValueError(f"expected CHAIN:RESSEQ:TO, got {text!r}")
        chain, position, target = (p.strip() for p in parts)
        if not chain:
            raise ValueError(f"empty chain in {text!r}")
        icode = ""
        if position and position[-1].isalpha():
            position, icode = position[:-1], position[-1]
        if not position.lstrip("-").isdigit():
            raise ValueError(f"expected an integer residue number in {text!r}")
        return cls(chain=chain, resseq=int(position), icode=icode, to=target)


class Receptor(_Base):
    path: Path | None = None
    pdb_id: str | None = None
    chains: tuple[str, ...] | None = Field(
        default=None, description="Protein chains to keep. None keeps all."
    )
    mutations: tuple[Mutation, ...] = Field(
        default=(),
        description=(
            "Point substitutions applied at pose load, before nodes are built. Empty for a "
            "wild-type system; a mutant keeps the same system_id so the two are pairable."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Receptor:
        if (self.path is None) == (self.pdb_id is None):
            raise ValueError("receptor requires exactly one of 'path' or 'pdb_id'")
        if self.chains is not None and len(self.chains) == 0:
            raise ValueError("receptor.chains may be omitted, but not empty")
        seen: set[tuple[str, int, str]] = set()
        for mut in self.mutations:
            if mut.key() in seen:
                raise ValueError(
                    f"duplicate mutation at {mut.chain}:{mut.resseq}{mut.icode.strip()}; "
                    "a position can be mutated once, and the second would silently win"
                )
            seen.add(mut.key())
        return self


class LigandSpec(_Base):
    selector: ResidueSelector
    params: Path | None = Field(
        default=None, description="Rosetta .params. Auto-parametrised when absent."
    )
    rosetta_name: str | None = Field(
        default=None,
        max_length=3,
        description="Residue name Rosetta uses, when it differs from comp_id (634 -> Z34).",
    )
    covalent_anchor: CovalentAnchor | None = None

    @property
    def effective_rosetta_name(self) -> str | None:
        """What the pose will call this residue. Never guess this at a call site."""
        if self.rosetta_name:
            return self.rosetta_name
        return self.selector.comp_id


class PocketSpec(_Base):
    mode: Literal["ligand_shell", "residue_list", "chain_interface", "whole"] = "ligand_shell"
    reference: Literal["ca", "cb", "any_heavy", "sidechain_heavy"] = "any_heavy"
    cutoff_A: float = Field(default=10.0, gt=0)
    residues: tuple[ResidueSelector, ...] = ()
    chains: tuple[str, ...] = Field(
        default=(),
        description="For chain_interface: the two sides. Empty uses all receptor chains.",
    )


class SystemSpec(_Base):
    system_id: str = Field(min_length=1)
    receptor: Receptor
    ligands: tuple[LigandSpec, ...] = ()
    pocket: PocketSpec = PocketSpec()
    labels: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form. Affinity lives here; it is never a required column.",
    )

    @model_validator(mode="after")
    def _pocket_mode_is_satisfiable(self) -> SystemSpec:
        mode = self.pocket.mode
        if mode == "ligand_shell" and not self.ligands:
            raise ValueError(
                "pocket.mode='ligand_shell' needs at least one ligand; "
                "use mode='whole' for protein-only or 'chain_interface' for an interface"
            )
        if mode == "residue_list" and not self.pocket.residues:
            raise ValueError("pocket.mode='residue_list' needs pocket.residues")
        if mode == "chain_interface":
            sides = self.pocket.chains or self.receptor.chains or ()
            if len(sides) < 2:
                raise ValueError(
                    "pocket.mode='chain_interface' needs at least two chains, via "
                    "pocket.chains or receptor.chains"
                )
        seen: set[tuple[str, int, str]] = set()
        for lig in self.ligands:
            if lig.selector.key() in seen:
                raise ValueError(f"duplicate ligand selector {lig.selector}")
            seen.add(lig.selector.key())
        return self

    @property
    def is_protein_only(self) -> bool:
        return not self.ligands

    @property
    def is_covalent(self) -> bool:
        return any(lig.covalent_anchor is not None for lig in self.ligands)

    # ------------------------------------------------------------------ paths

    def with_paths_relative_to(self, base: Path) -> SystemSpec:
        """Resolve relative ``path``/``params`` against ``base`` (the spec's directory)."""
        base = Path(base)
        data = self.model_dump()
        rp = data["receptor"]["path"]
        if rp is not None and not Path(rp).is_absolute():
            data["receptor"]["path"] = str((base / rp).resolve())
        for lig in data["ligands"]:
            if lig["params"] is not None and not Path(lig["params"]).is_absolute():
                lig["params"] = str((base / lig["params"]).resolve())
        return SystemSpec.model_validate(data)

    # -------------------------------------------------------------- structure

    def validate_against_structure(self, pdb_path: Path | None = None) -> None:
        """Check the spec against real coordinates. Raises :class:`SpecError` on any problem.

        Every problem is collected before raising, and the message lists what *is* present,
        so a wrong chain or a mistyped residue number is diagnosable in one pass.
        """
        problems = self.check_against_structure(pdb_path)
        if problems:
            raise SpecError(
                f"spec {self.system_id!r} does not match its structure:\n  - "
                + "\n  - ".join(problems)
            )

    def check_against_structure(self, pdb_path: Path | None = None) -> list[str]:
        """Return a list of human-readable problems. Empty means consistent."""
        path = Path(pdb_path) if pdb_path is not None else self.receptor.path
        if path is None:
            return ["no local path to validate against (receptor uses pdb_id)"]
        path = Path(path)
        if not path.exists():
            return [f"receptor file not found: {path}"]

        chains, residues = _read_chains_and_residues(path)
        problems: list[str] = []
        chain_list = ", ".join(sorted(chains)) or "(none)"

        for chain in self.receptor.chains or ():
            if chain not in chains:
                problems.append(
                    f"receptor chain {chain!r} not found in {path.name}; "
                    f"chains present: {chain_list}"
                )
        for chain in self.pocket.chains:
            if chain not in chains:
                problems.append(
                    f"pocket chain {chain!r} not found in {path.name}; "
                    f"chains present: {chain_list}"
                )

        for lig in self.ligands:
            # The file may legitimately carry EITHER name. A raw PDB has the CCD code; a
            # Stage-4 processed file has the Rosetta name, because Stage 4 rewrites the
            # HETATM resName (5HG8 stores Z34, not 634). Insisting on comp_id alone would
            # reject a *correctly written* spec — the one that records the true CCD code for
            # reporting and the Rosetta name for the pose.
            accept = {
                name
                for name in (lig.selector.comp_id, lig.effective_rosetta_name)
                if name
            }
            problems.extend(
                _check_selector(lig.selector, residues, chains, path, "ligand", accept)
            )
            anchor = lig.covalent_anchor
            if anchor is not None and anchor.key() not in residues:
                problems.append(
                    f"covalent anchor {anchor.chain}:{anchor.resseq} not found in {path.name}"
                )
            if lig.params is not None and not Path(lig.params).exists():
                problems.append(f"params file not found: {lig.params}")

        for sel in self.pocket.residues:
            problems.extend(_check_selector(sel, residues, chains, path, "pocket residue"))

        # A mutation names a position that must exist and must be a protein residue. Left
        # unchecked, a mistyped number becomes a silent no-op: the pose loads, the run
        # completes, and the "mutant" is the wild type under a different regeneration key.
        for mut in self.receptor.mutations:
            sel = ResidueSelector(chain=mut.chain, resseq=mut.resseq, icode=mut.icode)
            found_problems = _check_selector(
                sel, residues, chains, path, "mutation target"
            )
            if found_problems:
                problems.extend(found_problems)
                continue
            found = residues[sel.key()]
            if found.upper() not in _MUTATABLE:
                problems.append(
                    f"mutation target {mut} is {found!r} in {path.name}, which is not a "
                    "protein residue; only amino acids can be substituted"
                )
            if self.receptor.chains and mut.chain not in self.receptor.chains:
                problems.append(
                    f"mutation target {mut} names chain {mut.chain!r}, which "
                    f"receptor.chains ({', '.join(self.receptor.chains)}) drops, so the "
                    "substitution would never reach the graph"
                )

        return problems

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str | Path,
        ligand: str | Sequence[str] | None = None,
        *,
        system_id: str | None = None,
        chains: Sequence[str] | None = None,
        autodetect: bool = True,
        **labels: Any,
    ) -> SystemSpec:
        """Build a spec from a bare PDB — the custom-structure path (user request U8).

        ``ligand`` is one or more ``CHAIN:RESSEQ`` selectors. When omitted and ``autodetect``
        is on, every non-water HETATM residue becomes a component: metals and cofactors
        included, deliberately, since they are nodes in the contact graph. Pass
        ``autodetect=False`` for a protein-only system in a file that contains heteroatoms.

        .. warning::
           ``comp_id`` is read from the file, so it is only the real CCD code if the file
           says so. The legacy Stage-4 outputs in ``data/processed/*_clean.pdb`` have the
           HETATM ``resName`` **already rewritten to the Rosetta name** — 5HG8 stores
           ``Z34``, not its true CCD code ``634``. A spec built from such a file therefore
           carries the Rosetta name in ``comp_id``, which is right for the pose and wrong
           for reporting. When the true CCD code matters, supply it explicitly (from
           ``config/ligand_overrides.yaml`` or the Stage-1 inventory) and put the Rosetta
           name in ``rosetta_name`` where it belongs.
        """
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            raise SpecError(f"PDB not found: {pdb_path}")

        present_chains, residues = _read_chains_and_residues(pdb_path)

        if ligand is not None:
            wanted = [ligand] if isinstance(ligand, str) else list(ligand)
            selectors = []
            for text in wanted:
                sel = ResidueSelector.parse(text)
                comp = residues.get(sel.key())
                if comp is None:
                    raise SpecError(
                        f"ligand {sel} not found in {pdb_path.name}; "
                        f"chains present: {', '.join(sorted(present_chains)) or '(none)'}"
                    )
                selectors.append(sel.model_copy(update={"comp_id": comp}))
        elif autodetect:
            selectors = [
                ResidueSelector(chain=c, resseq=r, icode=i, comp_id=comp)
                for (c, r, i), comp in sorted(residues.items())
                if comp not in WATER_NAMES and not _is_standard_residue(comp)
            ]
        else:
            selectors = []

        ligands = tuple(LigandSpec(selector=s) for s in selectors)
        if system_id is None:
            stem = pdb_path.stem
            codes = "_".join(dict.fromkeys(s.comp_id or "LIG" for s in selectors))
            system_id = f"{stem}_{codes}" if codes else stem

        return cls(
            system_id=system_id,
            receptor=Receptor(
                path=pdb_path, chains=tuple(chains) if chains is not None else None
            ),
            ligands=ligands,
            pocket=PocketSpec(mode="ligand_shell" if ligands else "whole"),
            labels=dict(labels),
        )

    # ---------------------------------------------------------------- loaders

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> SystemSpec:
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        return cls.model_validate(data).with_paths_relative_to(path.parent)

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_defaults=False),
            sort_keys=True,
            default_flow_style=False,
        )


class SystemSet(_Base):
    """An ordered collection of systems with unique ids."""

    systems: tuple[SystemSpec, ...] = ()

    @model_validator(mode="after")
    def _ids_are_unique(self) -> SystemSet:
        seen: set[str] = set()
        for s in self.systems:
            if s.system_id in seen:
                raise ValueError(f"duplicate system_id {s.system_id!r}")
            seen.add(s.system_id)
        return self

    def __len__(self) -> int:
        return len(self.systems)

    def __iter__(self) -> Iterator[SystemSpec]:  # type: ignore[override]
        return iter(self.systems)

    def by_id(self, system_id: str) -> SystemSpec:
        for s in self.systems:
            if s.system_id == system_id:
                return s
        raise KeyError(
            f"no system {system_id!r}; have: {', '.join(s.system_id for s in self.systems)}"
        )

    # ---------------------------------------------------------------- loaders

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> SystemSet:
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        return cls._from_obj(data).\
            _resolved(path.parent)

    @classmethod
    def from_json_file(cls, path: str | Path) -> SystemSet:
        path = Path(path)
        return cls._from_obj(json.loads(path.read_text()))._resolved(path.parent)

    @classmethod
    def _from_obj(cls, data: Any) -> SystemSet:
        if isinstance(data, list):
            return cls(systems=tuple(SystemSpec.model_validate(d) for d in data))
        if isinstance(data, dict) and "systems" in data:
            return cls.model_validate(data)
        return cls(systems=(SystemSpec.model_validate(data),))

    def _resolved(self, base: Path) -> SystemSet:
        return SystemSet(systems=tuple(s.with_paths_relative_to(base) for s in self.systems))

    @classmethod
    def from_csv_file(cls, path: str | Path) -> SystemSet:
        """Flat one-row-per-system CSV.

        Recognised columns: ``system_id``, ``path`` or ``pdb_id``, ``chains`` (``|``- or
        ``;``-separated), ``ligand`` (``CHAIN:RESSEQ``), ``ligand_comp_id``, ``params``,
        ``rosetta_name``, ``pocket_mode``, ``pocket_reference``, ``pocket_cutoff_A``.
        Every other column becomes a label, so affinity and set membership ride along
        without a schema change.
        """
        import csv

        path = Path(path)
        known = {
            "system_id", "path", "pdb_id", "chains", "ligand", "ligand_comp_id",
            "params", "rosetta_name", "pocket_mode", "pocket_reference", "pocket_cutoff_A",
        }
        systems: list[SystemSpec] = []
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                ligands: tuple[LigandSpec, ...] = ()
                if row.get("ligand"):
                    sel = ResidueSelector.parse(row["ligand"])
                    if row.get("ligand_comp_id"):
                        sel = sel.model_copy(update={"comp_id": row["ligand_comp_id"]})
                    ligands = (
                        LigandSpec(
                            selector=sel,
                            params=Path(row["params"]) if row.get("params") else None,
                            rosetta_name=row.get("rosetta_name") or None,
                        ),
                    )
                pocket = PocketSpec(
                    mode=row.get("pocket_mode") or ("ligand_shell" if ligands else "whole"),
                    reference=row.get("pocket_reference") or "any_heavy",
                    cutoff_A=float(row["pocket_cutoff_A"]) if row.get("pocket_cutoff_A") else 10.0,
                )
                chains = _split_list(row.get("chains"))
                systems.append(
                    SystemSpec(
                        system_id=row["system_id"],
                        receptor=Receptor(
                            path=Path(row["path"]) if row.get("path") else None,
                            pdb_id=row.get("pdb_id") or None,
                            chains=chains,
                        ),
                        ligands=ligands,
                        pocket=pocket,
                        labels={
                            k: v for k, v in row.items() if k not in known and v not in (None, "")
                        },
                    )
                )
        return SystemSet(systems=tuple(systems))._resolved(path.parent)


# ------------------------------------------------------------------- helpers

# Standard residues are excluded from ligand autodetection: MSE and friends appear as
# HETATM but are backbone, not components.
_STANDARD = frozenset(
    """ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
       MSE SEC PYL ASX GLX UNK
       A C G U T DA DC DG DT DU""".split()
)


# What may sit at a mutated position in the input file. MSE/SEC/PYL are protein residues
# written as HETATM; the nucleotides in _STANDARD are not, which is why this is its own set.
_MUTATABLE = frozenset(_THREE_TO_ONE) | {"MSE", "SEC", "PYL"}


def _is_standard_residue(comp_id: str) -> bool:
    return comp_id.strip().upper() in _STANDARD


def _split_list(text: str | None) -> tuple[str, ...] | None:
    if not text:
        return None
    for sep in ("|", ";", ","):
        if sep in text:
            return tuple(p.strip() for p in text.split(sep) if p.strip())
    return (text.strip(),)


def _read_chains_and_residues(
    path: Path,
) -> tuple[set[str], dict[tuple[str, int, str], str]]:
    """Chain ids and ``(chain, resseq, icode) -> comp_id`` from a PDB or mmCIF.

    Parsed from ATOM/HETATM columns directly rather than through a structure library: this
    runs on every spec load, needs no connectivity, and must not depend on PyRosetta.
    """
    chains: set[str] = set()
    residues: dict[tuple[str, int, str], str] = {}

    if path.suffix.lower() in {".cif", ".mmcif"}:
        from Bio.PDB.MMCIFParser import MMCIFParser

        model = MMCIFParser(QUIET=True).get_structure("s", str(path))[0]
        for chain in model:
            chains.add(chain.id)
            for res in chain:
                het, resseq, icode = res.id
                residues[(chain.id, resseq, icode.strip())] = res.get_resname().strip()
        return chains, residues

    with path.open(errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21].strip() or " "
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip()
            comp = line[17:20].strip()
            chains.add(chain)
            residues.setdefault((chain, resseq, icode), comp)
    return chains, residues


def _check_selector(
    sel: ResidueSelector,
    residues: dict[tuple[str, int, str], str],
    chains: set[str],
    path: Path,
    what: str,
    accept: set[str] | None = None,
) -> list[str]:
    """``accept`` is the set of residue names the file may carry for this selector. It is
    wider than ``comp_id`` for a ligand — see the caller."""
    if sel.chain not in chains:
        return [
            f"{what} {sel} names chain {sel.chain!r}, not in {path.name}; "
            f"chains present: {', '.join(sorted(chains)) or '(none)'}"
        ]
    found = residues.get(sel.key())
    if found is None:
        in_chain = sorted(r for (c, r, _) in residues if c == sel.chain)
        hint = (
            f" residues {in_chain[0]}-{in_chain[-1]} in chain {sel.chain}"
            if in_chain
            else ""
        )
        return [f"{what} {sel} not found in {path.name};{hint}"]
    expected = accept if accept else ({sel.comp_id} if sel.comp_id is not None else set())
    if expected and found not in expected:
        wanted = " or ".join(repr(n) for n in sorted(expected))
        return [f"{what} {sel} is {found!r} in {path.name}, not {wanted}"]
    return []
