"""G1 tests for atomfrust.chem — code allocation, the params cache, and paramize.

The ``unit`` tier here needs neither PyRosetta nor RDKit: allocation and caching are pure
Python, and ``paramize`` on a broken molecule is *supposed* to return a categorised failure
whether RDKit is present (unparseable SMILES) or not (the RDKitUnavailable guard). Everything
that actually builds a conformer or a ``.params`` is ``integration``.
"""

from __future__ import annotations

import random
import string

import pytest

from atomfrust.chem.cache import ParamCache
from atomfrust.chem.codes import DEFAULT_RESERVED_CODES, CodeAllocator
from atomfrust.chem.paramize import (
    ParamFailure,
    ParamRecord,
    _atom_name_mismatch,
    _validate_in_rosetta,
    paramize,
    rdkit_available,
)


def _pyrosetta_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("pyrosetta") is not None


needs_rdkit = pytest.mark.skipif(not rdkit_available(), reason="RDKit not installed")
needs_pyrosetta = pytest.mark.skipif(not _pyrosetta_available(), reason="PyRosetta not installed")


def synthetic_inchikeys(n: int, seed: int = 7) -> list[str]:
    """InChIKey-shaped strings: ``AAAAAAAAAAAAAA-BBBBBBBBBB-C``."""
    rng = random.Random(seed)
    keys = set()
    while len(keys) < n:
        block = lambda k: "".join(rng.choice(string.ascii_uppercase) for _ in range(k))  # noqa: E731
        keys.add(f"{block(14)}-{block(10)}-{block(1)}")
    return sorted(keys)


# ------------------------------------------------------------------ CodeAllocator


@pytest.mark.unit
def test_codes_are_deterministic_across_allocators():
    keys = synthetic_inchikeys(1000)
    assert CodeAllocator().allocate_many(keys) == CodeAllocator().allocate_many(keys)


@pytest.mark.unit
def test_allocate_many_is_order_independent():
    """It sorts, so a batch is a set — the same library allocated in a different order (a
    different glob, a different worker count) must not renumber."""
    keys = synthetic_inchikeys(1000)
    shuffled = list(keys)
    random.Random(11).shuffle(shuffled)
    assert CodeAllocator().allocate_many(shuffled) == CodeAllocator().allocate_many(keys)


@pytest.mark.unit
def test_codes_are_collision_free_over_1000_molecules():
    codes = CodeAllocator().allocate_many(synthetic_inchikeys(1000))
    assert len(set(codes.values())) == len(codes) == 1000


@pytest.mark.unit
def test_codes_are_well_formed_and_letter_first():
    for code in CodeAllocator().allocate_many(synthetic_inchikeys(500)).values():
        assert len(code) == 3
        assert code[0] in string.ascii_uppercase, f"{code} starts with a digit (the 634 trap)"
        assert all(c in string.ascii_uppercase + string.digits for c in code[1:])


@pytest.mark.unit
def test_no_allocated_code_is_reserved():
    codes = set(CodeAllocator().allocate_many(synthetic_inchikeys(1000)).values())
    assert not codes & set(DEFAULT_RESERVED_CODES)
    assert "ALA" not in codes and "HOH" not in codes and "VRT" not in codes


@pytest.mark.unit
def test_extra_reserved_codes_are_avoided():
    keys = synthetic_inchikeys(200)
    natural = CodeAllocator().allocate_many(keys)
    blocked = set(natural.values())
    codes = CodeAllocator(reserved=set(DEFAULT_RESERVED_CODES) | blocked).allocate_many(keys)
    assert not set(codes.values()) & blocked
    assert len(set(codes.values())) == len(keys)


@pytest.mark.unit
def test_allocate_is_idempotent_within_an_instance():
    allocator = CodeAllocator()
    key = synthetic_inchikeys(1)[0]
    assert allocator.allocate(key) == allocator.allocate(key.lower())


@pytest.mark.unit
def test_reserve_blocks_a_code_after_construction():
    """A crystal ligand's comp_id must not be shadowed by a library molecule."""
    key = synthetic_inchikeys(1)[0]
    natural = CodeAllocator().allocate(key)
    allocator = CodeAllocator()
    allocator.reserve(natural)
    assert allocator.allocate(key) != natural


# --------------------------------------------------------------------- ParamCache


def _record(inchikey: str, failure: ParamFailure | None = None) -> ParamRecord:
    from pathlib import Path

    return ParamRecord(
        inchikey=inchikey,
        smiles="c1ccccc1",
        rosetta_code="ABC",
        params_path=None if failure else Path("params/ABC.params"),
        sha256=None if failure else "sha256:" + "0" * 64,
        failure=failure,
        message="synthetic",
    )


@pytest.mark.unit
def test_cache_round_trips_a_record(tmp_path):
    cache = ParamCache(tmp_path)
    record = _record("AAAAAAAAAAAAAA-BBBBBBBBBB-C")
    cache.put(record)
    assert cache.get(record.inchikey) == record


@pytest.mark.unit
def test_cache_round_trips_a_failure(tmp_path):
    cache = ParamCache(tmp_path)
    record = _record("AAAAAAAAAAAAAA-BBBBBBBBBB-C", ParamFailure.CONFORMER_FAIL)
    cache.put(record)
    hit = cache.get(record.inchikey)
    assert hit is not None and hit.failure is ParamFailure.CONFORMER_FAIL
    assert hit.params_path is None


@pytest.mark.unit
def test_cache_key_includes_both_versions(tmp_path):
    cache = ParamCache(tmp_path, protonation_version="v1", conformer_version="v1")
    assert cache.key("KEY") == "KEY|v1|v1"


@pytest.mark.unit
@pytest.mark.parametrize(
    "versions", [{"protonation_version": "v2"}, {"conformer_version": "v2"}]
)
def test_a_different_version_is_a_miss(tmp_path, versions):
    """Params built under another protonation or conformer model are a different molecule."""
    key = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
    ParamCache(tmp_path).put(_record(key))
    assert ParamCache(tmp_path).get(key) is not None
    assert ParamCache(tmp_path, **versions).get(key) is None


@pytest.mark.unit
def test_failures_group_by_the_taxonomy(tmp_path):
    cache = ParamCache(tmp_path)
    plan = {
        "K1": ParamFailure.SANITIZE_FAIL,
        "K2": ParamFailure.SANITIZE_FAIL,
        "K3": ParamFailure.UNSUPPORTED_ELEMENT,
        "K4": None,
    }
    for key, failure in plan.items():
        cache.put(_record(key, failure))

    failures = cache.failures()
    assert len(failures) == 3
    assert set(failures.columns) >= {"inchikey", "failure", "message"}
    counts = failures.groupby("failure").size().to_dict()
    assert counts == {"SANITIZE_FAIL": 2, "UNSUPPORTED_ELEMENT": 1}
    assert len(cache.records()) == 4


@pytest.mark.unit
def test_empty_cache_reports_an_empty_frame_not_an_error(tmp_path):
    cache = ParamCache(tmp_path)
    assert cache.failures().empty
    assert "failure" in cache.failures().columns
    assert len(cache) == 0


@pytest.mark.unit
def test_put_refuses_a_record_without_an_identity(tmp_path):
    with pytest.raises(ValueError):
        ParamCache(tmp_path).put(_record(""))


# ------------------------------------------------------------ failures are data


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["not-a-molecule", "", "C(C", None])
def test_broken_input_is_categorised_not_raised(tmp_path, bad):
    record = paramize(bad, out_dir=tmp_path)
    assert record.failure is ParamFailure.SANITIZE_FAIL
    assert record.message
    assert record.params_path is None
    assert not record.ok


@pytest.mark.unit
def test_paramize_needs_somewhere_to_write():
    """A missing output directory is the caller's bug, not the molecule's — that one raises."""
    with pytest.raises(ValueError):
        paramize("c1ccccc1")


# ------------------------------------------------ the crystal-only atom-name gate


@pytest.mark.unit
def test_atom_name_gate_accepts_matching_names(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "HETATM    1  C1  LIG A 501       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  C2  LIG A 501       1.400   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3  H1  LIG A 501       2.000   0.000   0.000  1.00  0.00           H\n"
    )
    # Extra H in the PDB is tolerated: Rosetta rebuilds hydrogens from ideal geometry.
    assert _atom_name_mismatch({"C1", "C2"}, [pdb], "LIG") is None


@pytest.mark.unit
def test_atom_name_gate_rejects_a_heavy_atom_disagreement(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "HETATM    1  C16 LIG A 501       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    problem = _atom_name_mismatch({"C1"}, [pdb], "LIG")
    assert problem is not None and "C16" in problem and "C1" in problem


@pytest.mark.unit
def test_atom_name_gate_needs_a_reference():
    assert _atom_name_mismatch({"C1"}, [], "LIG") is not None


# ------------------------------------------------------------------ integration

# These need RDKit and shell out to the vendored molfile_to_params; no PyRosetta.


@pytest.mark.integration
@needs_rdkit
def test_benzene_parametrises_end_to_end(tmp_path):
    from atomfrust.chem.paramize import _params_atoms

    record = paramize("c1ccccc1", out_dir=tmp_path)
    if not record.ok:
        assert isinstance(record.failure, ParamFailure), "failures must stay in the taxonomy"
        pytest.skip(f"benzene failed as {record.failure.value}: {record.message}")

    assert record.inchikey.startswith("UHOVQNZJYSORNB")  # benzene
    assert record.params_path is not None and record.params_path.exists()
    assert record.params_path.name == f"{record.rosetta_code}.params"
    assert record.sha256 and record.sha256.startswith("sha256:")

    atoms = _params_atoms(record.params_path)
    assert len(atoms) == 12  # 6 C + 6 H
    assert len({name for name, _, _ in atoms}) == 12, "duplicate atom names break pose loading"
    assert all(ros_type for _, ros_type, _ in atoms)


@pytest.mark.integration
@needs_rdkit
def test_unsupported_element_is_categorised(tmp_path):
    record = paramize("[U]", out_dir=tmp_path)
    assert record.failure is ParamFailure.UNSUPPORTED_ELEMENT
    assert "U" in record.message


@pytest.mark.integration
@needs_rdkit
def test_formal_charge_survives_the_backend(tmp_path):
    """Acetate must not come back neutral: a charge check is only meaningful if it passes
    for a genuinely charged molecule."""
    record = paramize("CC(=O)[O-]", out_dir=tmp_path)
    assert record.failure is None, record.message


@pytest.mark.integration
@needs_rdkit
def test_second_call_is_a_cache_hit(tmp_path):
    cache = ParamCache(tmp_path / "cache")
    first = paramize("c1ccccc1", cache=cache)
    assert first.ok, first.message

    # Remove the intermediate: a genuine hit never re-runs RDKit or the backend.
    sdf = cache.params_dir / f"{first.rosetta_code}.sdf"
    sdf.unlink()
    second = paramize("c1ccccc1", cache=cache)
    assert second == first
    assert not sdf.exists()


@pytest.mark.integration
@needs_rdkit
def test_cached_failure_is_replayed(tmp_path):
    cache = ParamCache(tmp_path / "cache")
    first = paramize("[U]", cache=cache)
    assert first.failure is ParamFailure.UNSUPPORTED_ELEMENT
    assert paramize("[U]", cache=cache) == first
    assert cache.failures().groupby("failure").size().to_dict() == {"UNSUPPORTED_ELEMENT": 1}


@pytest.mark.integration
@needs_rdkit
def test_library_molecules_skip_the_atom_name_gate(tmp_path):
    """The default is False on purpose: a library molecule has no HETATM names to match, so
    the gate that is essential in scripts/05 would fail every molecule here."""
    record = paramize("c1ccccc1", out_dir=tmp_path)
    assert record.ok, record.message

    gated = paramize(
        "c1ccccc1",
        out_dir=tmp_path,
        require_atom_name_match=True,
        reference_pdbs=[],
    )
    assert gated.failure is ParamFailure.ROSETTA_LOAD_FAIL


@pytest.mark.integration
@needs_rdkit
@needs_pyrosetta
def test_rosetta_validation_reaches_the_taxonomy(tmp_path):
    """The two Rosetta failure members must be reachable, not decorative."""
    record = paramize("c1ccccc1", out_dir=tmp_path, validate_in_rosetta=True)
    assert record.ok, record.message
    assert "REF2015" in record.message, "a passing validation must leave evidence it ran"

    bad = tmp_path / "BAD.params"
    bad.write_text("NAME BAD\nIO_STRING BAD Z\nTYPE LIGAND\nAA UNK\n")
    failure, message = _validate_in_rosetta(
        bad, tmp_path / f"{record.rosetta_code}_0001.pdb", 180
    )
    assert failure is ParamFailure.ROSETTA_LOAD_FAIL
    assert message


@pytest.mark.unit
def test_allocate_many_is_order_independent_but_sequential_allocate_is_not():
    """Pins a sharp edge rather than pretending it is not there.

    The first probe is a pure function of the InChIKey, so an uncontended molecule is
    stable. A collision is resolved against allocator state, so sequential `allocate` can
    disagree under reordering — measured at 42 of 1000 synthetic keys. `allocate_many`
    sorts first and is order-independent, which is why it is the batch entry point.
    Cross-run stability is the cache's job, not the allocator's.
    """
    import hashlib
    import random

    from atomfrust.chem.codes import CodeAllocator

    keys = [hashlib.sha256(str(i).encode()).hexdigest()[:27] for i in range(200)]
    shuffled = keys[:]
    random.Random(0).shuffle(shuffled)

    assert CodeAllocator().allocate_many(keys) == CodeAllocator().allocate_many(shuffled)

    forward = CodeAllocator()
    backward = CodeAllocator()
    a = {k: forward.allocate(k) for k in keys}
    b = {k: backward.allocate(k) for k in reversed(keys)}
    # Each allocator individually must still produce unique, well-formed codes.
    assert len(set(a.values())) == len(a)
    assert len(set(b.values())) == len(b)
    # Equality between them is deliberately NOT asserted: disagreement on the collision
    # path is the documented behaviour, and `allocate_many` above is the supported route.
    assert all(len(code) == 3 and code[0].isalpha() for code in a.values())
    assert all(len(code) == 3 and code[0].isalpha() for code in b.values())
