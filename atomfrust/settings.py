"""Layered, validated, stage-partitioned settings.

Three properties this module exists to guarantee:

1. **No dead keys.** Every model sets ``extra="forbid"``. A key nobody reads is a key that
   fails validation, which structurally prevents the defect at ``config.yaml:25-27`` in the
   prototype, where a ``frustration.thresholds`` block sat in the config and was read by no
   code while the real thresholds were hardcoded in three places.

2. **Stage partitioning.** Every leaf field declares
   ``json_schema_extra={"stage": ...}`` with one of:

   - ``"generation"`` — changing it invalidates a stored decoy ensemble. Enters
     :func:`regeneration_key`.
   - ``"analysis"`` — re-specifiable against an existing run directory with no PyRosetta
     call. Does *not* enter the key.
   - ``"runtime"`` — affects neither the scientific object nor the analysis (worker counts,
     sharding). Must never enter the key: results are required to be identical regardless
     of how many workers produced them, because decoy *i* is seeded ``base_seed + i``.

   :func:`check_all_fields_declare_a_stage` enforces that no field escapes this, so adding a
   field without deciding its stage is an error rather than a silent default.

3. **Provenance.** :func:`load_settings` layers defaults → file → environment → CLI and
   records which layer won each field, so a resolved config can explain itself.

Defaults follow the *published* protocol wherever step A4 established it from Chen et al.
2020 — see ``project_status/2026-08-11_0130_a4-paper-check-ligand-is-a-node.md``. Where a
default here differs from the prototype's behaviour, that is deliberate and flagged inline.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo

__all__ = [
    "Settings",
    "Layer",
    "load_settings",
    "stage_subset",
    "regeneration_key",
    "check_all_fields_declare_a_stage",
    "ENV_PREFIX",
]

ENV_PREFIX = "ATOMFRUST__"
STAGES = ("generation", "analysis", "runtime")


class Layer(str, Enum):
    """Where a resolved field's value came from. Later layers win."""

    DEFAULT = "default"
    FILE = "file"
    ENV = "env"
    CLI = "cli"


def _gen(desc: str, **kw: Any) -> Any:
    return Field(description=desc, json_schema_extra={"stage": "generation"}, **kw)


def _ana(desc: str, **kw: Any) -> Any:
    return Field(description=desc, json_schema_extra={"stage": "analysis"}, **kw)


def _run(desc: str, **kw: Any) -> Any:
    return Field(description=desc, json_schema_extra={"stage": "runtime"}, **kw)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- energy


class EnergySettings(_Base):
    score_function: str = _gen(
        "Rosetta score function. Chen et al. use REF2015.", default="ref2015"
    )
    store_terms: bool = _gen(
        "Store per-term energy columns alongside the total. Costs disk, buys per-term "
        "post-hoc analysis.",
        default=False,
    )
    exclude_fa_rep: bool = _ana(
        "Subtract the weighted fa_rep contribution. Analysis-stage because e_fa_rep is "
        "stored unsubtracted, so this is a post-hoc switch — unlike the prototype, which "
        "baked the choice in at scoring time (frustration.py:210-212).",
        default=True,
    )


# --------------------------------------------------------------------------- graph


class SupersetSettings(_Base):
    """The permissive pair set energies are computed over.

    Named contact definitions are boolean columns on this superset, so narrowing a cutoff
    at analysis time is a filter rather than a recomputation. A cutoff exceeding the
    superset requires regeneration.
    """

    ca_cutoff_A: float = _gen("Ca-Ca cutoff for the superset.", default=12.0, gt=0)
    heavy_cutoff_A: float = _gen(
        "Heavy-atom minimum-distance cutoff for the superset. Applies to pairs involving "
        "a ligand, metal or cofactor, which have no Ca. Must exceed any ligand contact "
        "cutoff an analysis may ask for: the superset is the ceiling on post-hoc "
        "re-selection, so 8 A here buys shells up to 8 A without regeneration.",
        default=8.0,
        gt=0,
    )
    seq_sep_min: int = _gen(
        "Minimum |i-j| within one chain for the superset. 1 admits everything.",
        default=1,
        ge=1,
    )


class GraphSettings(_Base):
    superset: SupersetSettings = SupersetSettings()
    include_ligand_nodes: bool = _gen(
        "Treat ligands, cofactors and metals as nodes in the contact graph. THE correction "
        "identified by step A4: the published counts are ligand-residue contacts, while the "
        "prototype counted only protein-protein pairs (get_protein_contacts, "
        "frustration.py:70-108, skips non-protein residues).",
        default=True,
    )


# ------------------------------------------------------------------------- contacts


class ContactSettings(_Base):
    definition: Literal["ca_ca", "cb_cb", "heavy_min"] = _ana(
        "Which named contact definition to select from the superset.", default="ca_ca"
    )
    cutoff_A: float = _ana(
        "Cutoff for the chosen definition. Must not exceed the superset's.",
        default=10.0,
        gt=0,
    )
    ligand_cutoff_A: float = _ana(
        "Heavy-atom minimum-distance cutoff for pairs involving a non-protein node. A Ca "
        "rule cannot apply to a ligand, so without this a 'ca_ca' definition would mark "
        "every ligand pair as absent and silently restore the protein-only behaviour that "
        "step A4 identified as the reproduction gap.",
        default=6.0,
        gt=0,
    )
    seq_sep_basis: Literal["resseq", "pose_index"] = _ana(
        "What |i-j| counts. 'resseq' uses PDB numbering, which is the biological sequence "
        "separation and is correct across a disordered loop. 'pose_index' counts only "
        "*modelled* residues and is what the prototype used (`for j in range(i + "
        "seq_sep_min, n)`, frustration.py:98) — across a numbering gap it treats residues "
        "that are far apart in sequence as adjacent. The two disagree on 6 of 1714 pairs "
        "in 1XKK (5 numbering gaps) and on none in 5GMP (2 gaps), so the difference is "
        "structure-dependent and silent. Kept selectable so stored results stay checkable.",
        default="resseq",
    )
    seq_sep_min: int = _ana(
        "Minimum |i-j| within one chain. Default 1 (no filter): Chen et al. define contacts "
        "purely by Ca-Ca distance and state no sequence-separation criterion (step A4). The "
        "prototype applied |i-j| >= 4 (frustration.py:98).",
        default=1,
        ge=1,
    )


class PocketSettings(_Base):
    mode: Literal["ligand_shell", "residue_list", "chain_interface", "whole"] = _ana(
        "How the pocket is defined.", default="ligand_shell"
    )
    reference: Literal["ca", "cb", "any_heavy", "sidechain_heavy"] = _ana(
        "Atom set the shell distance is measured from.", default="any_heavy"
    )
    cutoff_A: float = _ana("Shell radius.", default=10.0, gt=0)


# ------------------------------------------------------------------------- many-body


class ManyBodySettings(_Base):
    mode: Literal["chen_literal", "pair_retained", "pair_only"] = _ana(
        "Many-body contact energy formula. Analysis-stage because direct e_ij is stored and "
        "E_ij is never stored, so any formula can be re-formed post-hoc.\n"
        "  chen_literal   E_ij = e_ij + 1/2*sum_{k!=j} e_ik + 1/2*sum_{l!=i} e_jl.\n"
        "                 Exactly as printed in Chen et al. 2020 Eq. 2 (step A4 verified the "
        "                 exclusions are in the paper). Collapses to 0.5*(B_i + B_j): e_ij "
        "                 cancels, so it carries no pair-specific information (step A1, "
        "                 R^2 = 1.000000 over 38 structures). Required for any reproduction "
        "                 claim, and the default for that reason.\n"
        "  pair_retained  Background sums run over ALL partners, so the pair term survives. "
        "                 Non-degenerate, but not what was published.\n"
        "  pair_only      e_ij alone; the ablation isolating the many-body contribution.",
        default="chen_literal",
    )


# --------------------------------------------------------------------------- decoys


class RegionSettings(_Base):
    mutate: str = _gen("Selector for residues whose identity is randomised.", default="protein")
    repack: str = _gen("Selector for residues that are repacked.", default="protein")
    minimize: str = _gen("Selector for residues that are minimised.", default="protein")


class DecoySettings(_Base):
    axes: tuple[Literal["identity", "pose", "chemotype"], ...] = _gen(
        "Decoy axes to generate. Each is independently invocable.", default=("identity",)
    )
    n_decoys: int = _gen(
        "Decoys per axis. Chen et al. use 1000 per contact; the prototype used 50, which "
        "propagates ~10% relative error into every sigma.",
        default=1000,
        ge=2,
    )
    base_seed: int = _gen(
        "Decoy i is seeded base_seed + i. This is what makes results independent of worker "
        "count and makes an N-decoy prefix of a larger run exact.",
        default=42,
    )
    scope: Literal["whole_protein", "contact_shell", "contact_pair"] = _gen(
        "Which residues are randomised.", default="whole_protein"
    )
    identity: Literal["native", "composition", "uniform20", "layer_stratified"] = _gen(
        "Where new residue identities come from. Default 'native' (the native multiset, "
        "preserved exactly) because Chen et al. say 'we randomly shuffle the protein "
        "sequence' — a permutation (step A4). The prototype drew i.i.d. from the "
        "composition (frustration.py:337), which lets composition fluctuate.",
        default="native",
    )
    placement: Literal["inplace", "permute"] = _gen(
        "Whether identities are redrawn in place or permuted across positions. 'permute' "
        "with identity='native' is the published sequence shuffle.",
        default="permute",
    )
    relax: Literal["min", "mc"] = _gen(
        "Post-repack relaxation. Default 'mc': Chen et al. specify 'A short Monte-Carlo "
        "relaxation' (step A4). The prototype ran a single chi-only MinMover pass "
        "(frustration.py:364). NOTE: 'mc' is implemented by plan step C8; until it lands, "
        "pass relax='min' explicitly.",
        default="mc",
    )
    mc_cycles: int = _gen(
        "Monte-Carlo cycles when relax='mc'. The paper says 'short' without a number.",
        default=10,
        ge=1,
    )
    repack_shell_A: float | None = _gen(
        "Restrict repacking to this shell around the pocket. None repacks everything "
        "selected by regions.repack.",
        default=None,
    )
    native_repack: bool = _gen(
        "Repack and relax the native pose under the same protocol as decoys. Default True: "
        "Chen et al. state the native contact energies are 'obtained in a similar fashion by "
        "omitting the shuffling step' (step A4). The prototype scored the crystal pose as "
        "deposited (frustration.py:612-621), making native and decoys asymmetric.",
        default=True,
    )
    regions: RegionSettings = RegionSettings()


# ------------------------------------------------------------------------- analysis


class ClassifySettings(_Base):
    minimally_frustrated: float = _ana(
        "F above this is minimally frustrated.", default=0.78
    )
    highly_frustrated: float = _ana("F below this is highly frustrated.", default=-1.0)
    mode: Literal["fixed", "quantile"] = _ana(
        "'fixed' uses the literals above; 'quantile' uses definition-specific cutoffs from "
        "`atomfrust calibrate`. The 0.78 / -1.0 pair are quantiles of a coarse-grained null "
        "inherited from Ferreiro, not universal constants.",
        default="fixed",
    )


class AnalysisSettings(_Base):
    index: Literal["zscore", "rank_percentile", "robust_z"] = _ana(
        "Index function applied to the decoy vector.", default="zscore"
    )
    n_decoys: int | None = _ana(
        "Use only the first N decoys (decoy_id < N). Exact, because seeds are "
        "base_seed + decoy_id. None uses all.",
        default=None,
    )
    axes: tuple[str, ...] | None = _ana(
        "Restrict analysis to these axes. None uses all present.", default=None
    )
    descriptors: tuple[str, ...] = _ana(
        "Per-system descriptors to compute. No descriptor is privileged; pocket-size "
        "covariates are always emitted alongside.",
        default=("count_minimal", "count_highly", "count_neutral", "frac_minimal", "mean_Z"),
    )
    classify: ClassifySettings = ClassifySettings()


# -------------------------------------------------------------------------- runtime


class RuntimeSettings(_Base):
    workers: int = _run("Spawned worker processes.", default=1, ge=1)
    shard: str | None = _run(
        "Shard selector 'k/M'. Shards write disjoint files, so no lock is needed.",
        default=None,
    )
    save_structures: bool = _run("Write decoy PDBs.", default=False)
    progress: bool = _run("Show a progress bar.", default=True)


# ------------------------------------------------------------------------- top level


class Settings(_Base):
    energy: EnergySettings = EnergySettings()
    graph: GraphSettings = GraphSettings()
    contacts: ContactSettings = ContactSettings()
    pocket: PocketSettings = PocketSettings()
    manybody: ManyBodySettings = ManyBodySettings()
    decoys: DecoySettings = DecoySettings()
    analysis: AnalysisSettings = AnalysisSettings()
    runtime: RuntimeSettings = RuntimeSettings()

    def to_yaml(self) -> str:
        """Canonical YAML. Round-trips byte-identically through :meth:`from_yaml`."""
        return yaml.safe_dump(
            self.model_dump(mode="json"), sort_keys=True, default_flow_style=False
        )

    @classmethod
    def from_yaml(cls, text: str) -> Settings:
        return cls.model_validate(yaml.safe_load(text) or {})


# ----------------------------------------------------------------------- stage logic


def _stage_of(info: FieldInfo) -> str | None:
    extra = info.json_schema_extra
    return extra.get("stage") if isinstance(extra, dict) else None


def _iter_leaves(
    model: type[BaseModel], prefix: str = ""
) -> Iterator[tuple[str, FieldInfo]]:
    """Yield (dotted_path, FieldInfo) for every leaf field, recursing into submodels."""
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield from _iter_leaves(annotation, prefix=f"{path}.")
        else:
            yield path, info


def check_all_fields_declare_a_stage(model: type[BaseModel] = Settings) -> list[str]:
    """Return dotted paths of leaf fields missing a valid ``stage``. Empty means clean."""
    bad = []
    for path, info in _iter_leaves(model):
        if _stage_of(info) not in STAGES:
            bad.append(path)
    return bad


def stage_subset(settings: Settings, stage: str) -> dict[str, Any]:
    """Nested dict of only the leaves declared as ``stage``."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

    dumped = settings.model_dump(mode="json")
    out: dict[str, Any] = {}
    for path, info in _iter_leaves(type(settings)):
        if _stage_of(info) != stage:
            continue
        parts = path.split(".")
        src: Any = dumped
        for p in parts:
            src = src[p]
        dst = out
        for p in parts[:-1]:
            dst = dst.setdefault(p, {})
        dst[parts[-1]] = src
    return out


def regeneration_key(
    settings: Settings,
    input_digests: Mapping[str, Any] | None = None,
    pyrosetta_version: str | None = None,
) -> str:
    """``sha256`` over generation-stage settings, input digests and the PyRosetta version.

    Two runs sharing this key produce the same decoy ensemble. Analysis-stage and
    runtime-stage fields are excluded by construction: re-specifying an index function or
    doubling the worker count must not invalidate a stored ensemble.
    """
    payload = {
        "generation": stage_subset(settings, "generation"),
        "inputs": dict(input_digests or {}),
        "pyrosetta_version": pyrosetta_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# --------------------------------------------------------------------------- layering


def _deep_merge(
    base: dict[str, Any],
    overlay: Mapping[str, Any],
    layer: Layer,
    provenance: dict[str, Layer],
    prefix: str = "",
) -> dict[str, Any]:
    for key, value in overlay.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value, layer, provenance, prefix=f"{path}.")
        else:
            base[key] = value
            provenance[path] = layer
    return base


def _canonical_path(parts: list[str], model: type[BaseModel] = Settings) -> list[str]:
    """Resolve case-insensitive path segments against real field names.

    Environment variables are conventionally upper-case, but field names are not uniformly
    lower-case: ``repack_shell_A`` and ``cutoff_A`` carry a capital Angstrom. Naively
    lower-casing produces ``repack_shell_a``, which ``extra="forbid"`` then rejects — so the
    override would fail on a correctly-spelled variable.

    A segment that matches nothing is passed through unchanged, so a genuine typo still
    surfaces as a validation error rather than being silently dropped.
    """
    resolved: list[str] = []
    current: type[BaseModel] | None = model
    for part in parts:
        match = None
        if current is not None:
            for field_name in current.model_fields:
                if field_name.lower() == part:
                    match = field_name
                    break
        if match is None:
            resolved.append(part)
            current = None
            continue
        resolved.append(match)
        annotation = current.model_fields[match].annotation  # type: ignore[union-attr]
        current = (
            annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel)
            else None
        )
    return resolved


def _env_overlay(
    environ: Mapping[str, str], model: type[BaseModel] = Settings
) -> dict[str, Any]:
    """``ATOMFRUST__DECOYS__N_DECOYS=200`` -> ``{"decoys": {"n_decoys": 200}}``."""
    out: dict[str, Any] = {}
    for raw_key, raw_value in environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in raw_key[len(ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue
        parts = _canonical_path(parts, model)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def load_settings(
    file_overlay: Mapping[str, Any] | None = None,
    cli_overlay: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Settings, dict[str, Layer]]:
    """Layer defaults → file → env → CLI.

    Returns the validated settings and a dotted-path → :class:`Layer` map recording which
    layer won each field that was explicitly set. Fields absent from the map came from
    defaults.

    An unknown key at any layer raises ``pydantic.ValidationError`` — that is the whole
    point of ``extra="forbid"``, and it is what makes a dead config key impossible.
    """
    merged = Settings().model_dump(mode="json")
    provenance: dict[str, Layer] = {}

    for overlay, layer in (
        (file_overlay, Layer.FILE),
        (_env_overlay(os.environ if environ is None else environ), Layer.ENV),
        (cli_overlay, Layer.CLI),
    ):
        if overlay:
            _deep_merge(merged, overlay, layer, provenance)

    return Settings.model_validate(merged), provenance
