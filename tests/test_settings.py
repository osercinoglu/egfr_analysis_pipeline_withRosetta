"""B2 tests for atomfrust.settings — validation, stage partitioning, layering, round-trip."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atomfrust.settings import (
    ENV_PREFIX,
    Layer,
    Settings,
    check_all_fields_declare_a_stage,
    load_settings,
    regeneration_key,
    stage_subset,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------- no dead keys (R25)


def test_unknown_key_raises_rather_than_being_ignored():
    """The config.yaml:25-27 defect class: a key nobody reads must not validate."""
    with pytest.raises(ValidationError):
        Settings.model_validate({"frustration": {"thresholds": {"minimally": 0.78}}})


def test_unknown_nested_key_raises():
    with pytest.raises(ValidationError):
        Settings.model_validate({"decoys": {"not_a_real_field": 1}})


def test_unknown_key_raises_through_the_layering_path():
    with pytest.raises(ValidationError):
        load_settings(file_overlay={"decoys": {"typo_here": 3}}, environ={})


def test_constraints_are_enforced():
    with pytest.raises(ValidationError):
        Settings.model_validate({"decoys": {"n_decoys": 1}})  # ge=2
    with pytest.raises(ValidationError):
        Settings.model_validate({"contacts": {"cutoff_A": -1.0}})  # gt=0
    with pytest.raises(ValidationError):
        Settings.model_validate({"manybody": {"mode": "not_a_mode"}})


# ------------------------------------------------------------------ stage partition


def test_every_leaf_field_declares_a_stage():
    """Adding a field without deciding its stage is an error, not a silent default."""
    assert check_all_fields_declare_a_stage() == []


def test_stage_subsets_are_disjoint_and_total():
    s = Settings()
    leaves = {
        stage: _flatten(stage_subset(s, stage))
        for stage in ("generation", "analysis", "runtime")
    }
    gen, ana, run = leaves["generation"], leaves["analysis"], leaves["runtime"]
    assert not (gen & ana) and not (gen & run) and not (ana & run)
    assert gen and ana and run


def _flatten(d, prefix=""):
    out = set()
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _flatten(v, f"{prefix}{k}.")
        else:
            out.add(f"{prefix}{k}")
    return out


# --------------------------------------------------------------- regeneration key


def test_generation_change_alters_the_key():
    base = Settings()
    changed = Settings.model_validate({"energy": {"score_function": "ref2015_cart"}})
    assert regeneration_key(base) != regeneration_key(changed)


@pytest.mark.parametrize(
    "overlay",
    [
        {"analysis": {"index": "rank_percentile"}},
        {"analysis": {"n_decoys": 100}},
        {"analysis": {"classify": {"minimally_frustrated": 1.5}}},
        {"manybody": {"mode": "pair_retained"}},
        {"contacts": {"cutoff_A": 8.0}},
        {"pocket": {"cutoff_A": 5.0}},
        {"energy": {"exclude_fa_rep": False}},
    ],
)
def test_analysis_changes_do_not_alter_the_key(overlay):
    """These are exactly the knobs that must be re-specifiable against a stored run."""
    assert regeneration_key(Settings()) == regeneration_key(
        Settings.model_validate(overlay)
    )


@pytest.mark.parametrize(
    "overlay",
    [{"runtime": {"workers": 32}}, {"runtime": {"shard": "1/4"}}],
)
def test_runtime_changes_do_not_alter_the_key(overlay):
    """Worker count must not affect results: decoy i is seeded base_seed + i."""
    assert regeneration_key(Settings()) == regeneration_key(
        Settings.model_validate(overlay)
    )


def test_key_covers_input_digests_and_pyrosetta_version():
    s = Settings()
    a = regeneration_key(s, {"receptor_sha256": "aa"}, "2026.30")
    assert a != regeneration_key(s, {"receptor_sha256": "bb"}, "2026.30")
    assert a != regeneration_key(s, {"receptor_sha256": "aa"}, "2026.31")
    assert a == regeneration_key(s, {"receptor_sha256": "aa"}, "2026.30")


def test_key_is_order_independent():
    s = Settings()
    assert regeneration_key(s, {"a": "1", "b": "2"}) == regeneration_key(
        s, {"b": "2", "a": "1"}
    )


# ----------------------------------------------------------------------- round-trip


def test_yaml_round_trips_byte_identically():
    original = Settings()
    once = original.to_yaml()
    twice = Settings.from_yaml(once).to_yaml()
    assert once == twice
    assert Settings.from_yaml(once) == original


def test_yaml_round_trip_survives_non_default_values():
    s = Settings.model_validate(
        {
            "decoys": {"n_decoys": 250, "relax": "min", "repack_shell_A": 8.0},
            "analysis": {"index": "robust_z", "axes": ["identity", "chemotype"]},
        }
    )
    assert Settings.from_yaml(s.to_yaml()) == s


# ------------------------------------------------------------------------- layering


def test_layer_precedence_cli_beats_env_beats_file():
    settings, prov = load_settings(
        file_overlay={"decoys": {"n_decoys": 100, "base_seed": 7}},
        cli_overlay={"decoys": {"n_decoys": 300}},
        environ={f"{ENV_PREFIX}DECOYS__N_DECOYS": "200"},
    )
    assert settings.decoys.n_decoys == 300
    assert prov["decoys.n_decoys"] is Layer.CLI
    assert settings.decoys.base_seed == 7
    assert prov["decoys.base_seed"] is Layer.FILE


def test_env_beats_file_when_no_cli():
    settings, prov = load_settings(
        file_overlay={"decoys": {"n_decoys": 100}},
        environ={f"{ENV_PREFIX}DECOYS__N_DECOYS": "200"},
    )
    assert settings.decoys.n_decoys == 200
    assert prov["decoys.n_decoys"] is Layer.ENV


def test_unset_fields_keep_defaults_and_have_no_provenance():
    settings, prov = load_settings(environ={})
    assert settings == Settings()
    assert prov == {}


def test_env_values_are_parsed_not_left_as_strings():
    settings, _ = load_settings(
        environ={
            f"{ENV_PREFIX}DECOYS__N_DECOYS": "200",
            f"{ENV_PREFIX}ENERGY__EXCLUDE_FA_REP": "false",
            f"{ENV_PREFIX}DECOYS__REPACK_SHELL_A": "8.5",
        }
    )
    assert settings.decoys.n_decoys == 200
    assert settings.energy.exclude_fa_rep is False
    assert settings.decoys.repack_shell_A == 8.5


def test_deep_merge_does_not_clobber_sibling_fields():
    settings, _ = load_settings(
        file_overlay={"decoys": {"n_decoys": 100}},
        cli_overlay={"decoys": {"base_seed": 9}},
        environ={},
    )
    assert (settings.decoys.n_decoys, settings.decoys.base_seed) == (100, 9)
    assert settings.decoys.identity == "native"  # untouched default survives


# ------------------------------------------------- defaults encode the A4 findings


def test_defaults_follow_the_published_protocol():
    """Step A4 read these off Chen et al. 2020; they are not arbitrary choices."""
    s = Settings()
    assert s.decoys.identity == "native" and s.decoys.placement == "permute"  # shuffle
    assert s.decoys.relax == "mc"  # "A short Monte-Carlo relaxation"
    assert s.decoys.native_repack is True  # "in a similar fashion by omitting shuffling"
    assert s.decoys.n_decoys == 1000  # "1000 appropriately distributed decoys"
    assert s.contacts.seq_sep_min == 1  # no sequence-separation criterion in the paper
    assert s.manybody.mode == "chen_literal"  # the published Eq. 2
    assert s.graph.include_ligand_nodes is True  # the A4 correction


def test_env_resolves_field_names_with_capital_letters():
    """Regression: naive lower-casing turned REPACK_SHELL_A into an unknown key."""
    settings, prov = load_settings(
        environ={
            f"{ENV_PREFIX}DECOYS__REPACK_SHELL_A": "8.5",
            f"{ENV_PREFIX}CONTACTS__CUTOFF_A": "7.0",
            f"{ENV_PREFIX}GRAPH__SUPERSET__CA_CUTOFF_A": "14.0",
        }
    )
    assert settings.decoys.repack_shell_A == 8.5
    assert settings.contacts.cutoff_A == 7.0
    assert settings.graph.superset.ca_cutoff_A == 14.0
    assert prov["decoys.repack_shell_A"] is Layer.ENV


def test_env_typo_still_raises():
    """Case-insensitive resolution must not swallow genuinely wrong names."""
    with pytest.raises(ValidationError):
        load_settings(environ={f"{ENV_PREFIX}DECOYS__NDECOYS": "200"})


def test_stage_checker_actually_detects_a_missing_stage():
    """Positive control: without this, test_every_leaf_field_declares_a_stage could pass
    vacuously if the checker were broken."""
    from pydantic import BaseModel, Field

    class Bad(BaseModel):
        declared: int = Field(default=1, json_schema_extra={"stage": "generation"})
        undeclared: int = 2
        wrong_stage: int = Field(default=3, json_schema_extra={"stage": "nonsense"})

    assert sorted(check_all_fields_declare_a_stage(Bad)) == ["undeclared", "wrong_stage"]


def test_stage_checker_recurses_into_submodels():
    from pydantic import BaseModel, Field

    class Inner(BaseModel):
        ok: int = Field(default=1, json_schema_extra={"stage": "analysis"})
        missing: int = 2

    class Outer(BaseModel):
        inner: Inner = Inner()

    assert check_all_fields_declare_a_stage(Outer) == ["inner.missing"]
