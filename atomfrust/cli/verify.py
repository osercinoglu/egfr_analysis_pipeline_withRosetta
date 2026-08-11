"""``atomfrust verify`` and ``atomfrust metrics-selftest`` — the two reproducibility checks.

Success criterion S0.3 of the methods document is *"harness returns identical metric values
for a fixed input in both students' environments"*. Two independent things stand between a
stored number and that guarantee, and one command addresses each.

``verify`` asks whether a run directory still describes the machine it is sitting on: the
bytes it was computed from, the environment it was computed in, and — with PyRosetta
present — whether the recorded protocol still reproduces the stored decoy energies exactly.
A run whose receptor has been edited since it was written is not a reproducible artefact
even if every number inside it is self-consistent, and nothing else in the package notices.

``metrics-selftest`` asks whether *this* machine computes the same metric values as the
machine that wrote the golden file. It needs no run directory, no PyRosetta and no data, so
it is the check the two students can run on day one, which is exactly what S0.3 asks for.

Statuses are ``PASS``/``FAIL``/``SKIP`` plus ``INFO``. Only ``FAIL`` affects the exit code:
``SKIP`` means a check could not run (no PyRosetta, ``--replay 0``, nothing stored) and
``INFO`` means a difference was found that is not by itself a defect — see
:func:`_check_environment`.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from atomfrust.provenance import (
    Manifest,
    ManifestVersionError,
    capture_environment,
    env_digest,
    sha256_bytes,
    sha256_file,
)
from atomfrust.runstore import RunDir, SystemDir

NAME = "verify"
HELP = "re-digest a run's inputs and replay decoys for reproducibility"

SELFTEST_NAME = "metrics-selftest"
SELFTEST_HELP = "recompute every metric on fixed inputs and hash against a golden file"

#: Shipped golden file. ``--golden`` points the command at a different one, which is how a
#: test can corrupt a copy without touching the repository's.
GOLDEN_PATH = Path(__file__).resolve().parents[1] / "metrics" / "golden" / "metrics_golden.json"

#: Bumped when the case set or the canonicalisation changes, so an old golden file is
#: rejected as stale instead of being compared field-by-field against a different fixture.
SELFTEST_SCHEMA = 1

#: Seed for every synthetic input below. Fixed input + fixed seed is the whole premise.
SELFTEST_SEED = 20260811

#: Significant digits every float is rounded to before hashing.
#:
#: An unrounded float64 hash would be a bad check: a bootstrap percentile, a Pearson r and a
#: permutation count are all sums and dot products whose last bits depend on summation order,
#: and summation order depends on the BLAS build, its threading and the numpy version. Two
#: correct environments would therefore hash differently for no scientific reason, and the
#: check would be abandoned as noisy — the usual fate of a too-strict golden test.
#:
#: 12 is chosen from both ends. Above: accumulated rounding over the ~10^2-10^4 element
#: reductions here is ~1e-15 relative (unit roundoff 1.1e-16 times sqrt(n)), so 12 digits is
#: three orders of magnitude clear of it. Below: nothing in this package reports more than 4
#: significant digits, so a difference too small for 12 digits to see is too small to change
#: any reported number. A genuine divergence — a different tie rule, a changed RNG stream, a
#: different BEDROC formula — moves far more than the 12th digit.
SIGNIFICANT_DIGITS = 12

Status = Literal["PASS", "FAIL", "SKIP", "INFO"]

#: Filenames the run-directory contract (§4) fixes for each manifest input key. The manifest
#: stores digests, not paths, so the path has to come from the contract.
CONVENTIONAL_INPUT_FILES = {
    "receptor_sha256": "receptor.pdb",
    "spec_sha256": "system.spec.yaml",
    "components_sha256": "components.yaml",
}

#: List-valued: one digest per ``.params`` file, order-insensitive.
PARAMS_DIGEST_KEY = "params_sha256"


@dataclass(frozen=True)
class Check:
    """One row of the report. ``detail`` is a full sentence, not a code."""

    name: str
    status: Status
    detail: str = ""


# --------------------------------------------------------------------- registration


def register(subparsers: argparse._SubParsersAction) -> None:
    verify = subparsers.add_parser(
        NAME,
        help=HELP,
        description=(
            "Check that a run directory still describes the files and environment it was "
            "computed from, and that its stored decoy energies can be reproduced exactly."
        ),
    )
    verify.add_argument(
        "--run-dir", required=True, type=Path, help="run directory to verify"
    )
    verify.add_argument(
        "--replay",
        type=int,
        default=3,
        metavar="N",
        help="replay the first N stored decoys per system and require bit-identical "
        "energies (default 3; 0 skips, and so does a machine without PyRosetta)",
    )
    verify.set_defaults(func=run)

    selftest = subparsers.add_parser(SELFTEST_NAME, help=SELFTEST_HELP)
    selftest.add_argument(
        "--emit",
        action="store_true",
        help="write the golden file from this environment instead of comparing against it",
    )
    selftest.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_PATH,
        help=f"golden file to compare against or write (default {GOLDEN_PATH})",
    )
    selftest.set_defaults(func=run_selftest)


# --------------------------------------------------------------------------- verify


def run(args: argparse.Namespace) -> int:
    run_dir = RunDir(args.run_dir)
    checks: list[Check] = []

    manifest = None
    try:
        manifest = run_dir.read_manifest()
        checks.append(
            Check(
                "manifest",
                "PASS",
                f"schema {manifest.schema_version}, contract {manifest.contract_version}, "
                f"run_id={manifest.run_id}, created {manifest.created_utc}",
            )
        )
    except FileNotFoundError:
        checks.append(Check("manifest", "FAIL", f"no manifest at {run_dir.manifest_path}"))
    except (ManifestVersionError, ValueError) as exc:
        # ValueError covers both json.JSONDecodeError and pydantic's ValidationError. The
        # message is reported as a message: a traceback out of `verify` would say the tool
        # broke, when what actually happened is that the run directory is unreadable.
        checks.append(Check("manifest", "FAIL", _one_line(exc)))

    if manifest is None:
        # Every remaining check reads the manifest. Continuing would produce a cascade of
        # failures with one cause, which buries the cause.
        return _report(checks)

    checks.extend(_check_inputs(run_dir, manifest))
    checks.append(_check_environment(run_dir, manifest))
    checks.extend(_check_replay(run_dir, int(args.replay)))
    return _report(checks)


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())


def _report(checks: list[Check]) -> int:
    width = max((len(c.name) for c in checks), default=5)
    print(f"{'check'.ljust(width)}  status  detail")
    print(f"{'-' * width}  ------  ------")
    for check in checks:
        print(f"{check.name.ljust(width)}  {check.status:<6}  {check.detail}")

    failed = [c for c in checks if c.status == "FAIL"]
    counts = {s: sum(1 for c in checks if c.status == s) for s in ("PASS", "FAIL", "SKIP", "INFO")}
    print(
        f"\n{counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['SKIP']} skipped, {counts['INFO']} informational"
    )
    return 1 if failed else 0


# ------------------------------------------------------------------- input digests


def _check_inputs(run_dir: RunDir, manifest: Manifest) -> list[Check]:
    """Re-digest every file the manifest names and report anything that moved.

    The manifest records digests keyed by role (``receptor_sha256``), not paths, so the path
    comes from the run-directory contract. Two shapes are accepted: the flat form of §4,
    which describes a single-system run, and ``inputs.systems.<system_id>.<key>`` for a run
    holding several systems, where a single flat ``receptor_sha256`` would be ambiguous. An
    entry may also be an explicit ``{"path": ..., "sha256": ...}`` mapping, which is checked
    verbatim.
    """
    entries = dict(manifest.inputs)
    per_system = entries.pop("systems", None)
    if not entries and not per_system:
        return [Check("inputs", "SKIP", "the manifest records no input digests")]

    checks: list[Check] = []
    if isinstance(per_system, Mapping):
        for system_id, sub in sorted(per_system.items()):
            if isinstance(sub, Mapping):
                checks.extend(_check_system_inputs(run_dir.system(system_id), dict(sub)))
            else:
                checks.append(
                    Check(f"inputs/{system_id}", "SKIP", f"expected a mapping, got {type(sub).__name__}")
                )

    if entries:
        systems = run_dir.systems()
        if len(systems) == 1:
            checks.extend(_check_system_inputs(run_dir.system(systems[0]), entries))
        elif not systems:
            checks.append(
                Check(
                    "inputs",
                    "FAIL",
                    f"the manifest records {sorted(entries)} but the run has no systems/ "
                    "directory, so there are no files to compare them against",
                )
            )
        else:
            checks.append(
                Check(
                    "inputs",
                    "SKIP",
                    f"the run holds {len(systems)} systems but the manifest carries one flat "
                    "digest set; record inputs.systems.<system_id>.<key> so each digest "
                    "names the file it belongs to",
                )
            )
    return checks


def _check_system_inputs(system: SystemDir, entries: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    for key, expected in sorted(entries.items()):
        label = f"inputs/{system.system_id}/{key}"
        if isinstance(expected, Mapping) and "path" in expected:
            path = Path(expected["path"])
            checks.append(
                _check_one_file(
                    label, path if path.is_absolute() else system.root / path, expected.get("sha256")
                )
            )
        elif key == PARAMS_DIGEST_KEY:
            checks.append(_check_params(label, system, expected))
        elif key in CONVENTIONAL_INPUT_FILES:
            checks.append(
                _check_one_file(label, system.inputs / CONVENTIONAL_INPUT_FILES[key], expected)
            )
        else:
            checks.append(
                Check(label, "SKIP", "not a known file reference; no path to re-digest it from")
            )
    return checks


def _check_one_file(label: str, path: Path, expected: Any) -> Check:
    if not isinstance(expected, str):
        return Check(label, "SKIP", f"expected a digest string, got {type(expected).__name__}")
    if not path.exists():
        return Check(label, "FAIL", f"{path} is missing; the manifest records {expected}")
    actual = sha256_file(path)
    if actual != expected:
        return Check(
            label,
            "FAIL",
            f"{path} changed: manifest {expected}, on disk {actual}. The run no longer "
            "describes this file, so its stored numbers are not reproducible from it.",
        )
    return Check(label, "PASS", f"{path} matches {actual}")


def _check_params(label: str, system: SystemDir, expected: Any) -> Check:
    """Params digests are a set, not a sequence: the order they were listed in is not data."""
    if not isinstance(expected, (list, tuple)):
        return Check(label, "SKIP", f"expected a list of digests, got {type(expected).__name__}")
    directory = system.inputs / "params"
    paths = sorted(directory.glob("*.params"))
    if not paths:
        return Check(label, "FAIL", f"{directory} holds no .params but the manifest records {len(expected)}")

    actual = {sha256_file(p): p for p in paths}
    missing = sorted(set(expected) - set(actual))
    added = sorted(f"{actual[d].name} ({d})" for d in set(actual) - set(expected))
    if missing or added:
        return Check(
            label,
            "FAIL",
            f"params changed: {len(missing)} recorded digest(s) absent {missing}, "
            f"{len(added)} unrecorded file(s) present {added}",
        )
    return Check(label, "PASS", f"{len(paths)} params file(s) in {directory} match")


# --------------------------------------------------------------------- environment


def _check_environment(run_dir: RunDir, manifest: Manifest) -> Check:
    """Compare the live environment digest against the stored one. Information, not failure.

    ``env_digest`` (``atomfrust/provenance.py``) hashes only ``python``, ``pyrosetta`` and
    the tracked package versions. ``cpu_count``, ``platform`` and ``machine`` are excluded
    on purpose: a result that depended on the core count would be a defect in this package,
    and demanding an identical OS string would make the cross-machine comparison S0.3 asks
    for impossible from the start. Both are still recorded in ``env.json`` as diagnostics.

    So a mismatch here means exactly one thing — a package version or the Python version
    moved. That is worth knowing and worth printing, but it is not evidence that the run is
    wrong; the digest checks above and the replay below are what decide that.
    """
    environment = capture_environment()
    digest = env_digest(environment)
    if digest == manifest.env_digest:
        return Check("environment", "PASS", f"env_digest matches ({digest})")

    moved = _environment_differences(run_dir, environment)
    detail = f"env_digest differs: manifest {manifest.env_digest}, here {digest}"
    if moved:
        detail += " — " + "; ".join(moved)
    else:
        detail += " — env.json is absent, so the moving package cannot be named"
    detail += ". Informational: a package or Python version moved, which does not by itself invalidate the run."
    return Check("environment", "INFO", detail)


def _environment_differences(run_dir: RunDir, environment: dict[str, Any]) -> list[str]:
    """Name what moved, by diffing ``env.json`` against the live capture."""
    if not run_dir.env_path.exists():
        return []
    try:
        stored = json.loads(run_dir.env_path.read_text())
    except (OSError, ValueError):
        return []

    moved = []
    for key in ("python", "pyrosetta"):
        if stored.get(key) != environment.get(key):
            moved.append(f"{key} {stored.get(key)} -> {environment.get(key)}")
    stored_packages = stored.get("packages") or {}
    live_packages = environment.get("packages") or {}
    for name in sorted(set(stored_packages) | set(live_packages)):
        if stored_packages.get(name) != live_packages.get(name):
            moved.append(f"{name} {stored_packages.get(name)} -> {live_packages.get(name)}")
    return moved


# -------------------------------------------------------------------------- replay


def _check_replay(run_dir: RunDir, n_replay: int) -> list[Check]:
    """Regenerate the first ``n_replay`` stored decoys and require bit-identical energies.

    This is the check that actually tests the seeding contract end to end: decoy *i* is
    seeded ``base_seed + i``, so regenerating it from the recorded settings must reproduce
    the stored ``e_direct``/``e_fa_rep`` exactly — not to a tolerance.
    """
    if n_replay <= 0:
        return [Check("replay", "SKIP", "--replay 0")]
    if importlib.util.find_spec("pyrosetta") is None:
        return [
            Check(
                "replay",
                "SKIP",
                "PyRosetta is not installed, so no decoy can be regenerated; the digest and "
                "environment checks above still ran",
            )
        ]

    systems = run_dir.systems()
    if not systems:
        return [Check("replay", "SKIP", "the run holds no systems")]
    try:
        settings = run_dir.read_settings()
    except (OSError, ValueError) as exc:
        return [Check("replay", "FAIL", f"settings.resolved.yaml is unreadable: {_one_line(exc)}")]

    return [_replay_system(run_dir.system(system_id), settings, n_replay) for system_id in systems]


def _replay_system(system: SystemDir, settings: Any, n_replay: int) -> Check:
    label = f"replay/{system.system_id}"
    stored = system.read_decoy_energies()
    if stored.empty:
        return Check(label, "SKIP", "no decoy energies are stored for this system")

    stored = stored[stored["axis"].astype(str) == "identity"]
    if stored.empty:
        return Check(label, "SKIP", "no decoys on the identity axis; replay covers that axis only")

    spec_path = system.inputs / "system.spec.yaml"
    if not spec_path.exists():
        return Check(
            label,
            "FAIL",
            f"{spec_path} is missing; a run whose spec is gone cannot be regenerated at all",
        )

    try:
        generator = _build_generator(system, settings, spec_path)
    except Exception as exc:  # noqa: BLE001 - any failure here is a reportable finding
        return Check(label, "FAIL", f"could not rebuild the decoy protocol: {_one_line(exc)}")

    decoy_ids = sorted({int(v) for v in stored["decoy_id"].unique()})[:n_replay]
    problems = []
    for decoy_id in decoy_ids:
        try:
            result = generator.generate(decoy_id)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"decoy {decoy_id}: generation failed: {_one_line(exc)}")
            continue
        difference = _compare_energies(stored[stored["decoy_id"] == decoy_id], result)
        if difference is not None:
            problems.append(f"decoy {decoy_id}: {difference}")

    if problems:
        return Check(label, "FAIL", "; ".join(problems))
    return Check(label, "PASS", f"decoys {decoy_ids} reproduced bit-identically")


def _build_generator(system: SystemDir, settings: Any, spec_path: Path) -> Any:
    from atomfrust.decoys.base import DecoyContext
    from atomfrust.decoys.identity import IdentityDecoyGenerator
    from atomfrust.graph import build_graph
    from atomfrust.pose import load_complex
    from atomfrust.regions import resolve_regions
    from atomfrust.spec import SystemSpec

    spec = SystemSpec.from_yaml_file(spec_path)
    loaded = load_complex(spec)

    # The stored graph wins over a rebuilt one. Replay must score the pairs the run scored;
    # rebuilding would silently substitute a different pair set if a graph setting moved,
    # and the resulting mismatch would be blamed on the decoy protocol.
    try:
        _, pairs = system.read_graph()
    except (FileNotFoundError, OSError):
        _, pairs = build_graph(loaded.nodes, loaded.geometry, settings)

    regions = resolve_regions(
        loaded.nodes,
        loaded.geometry,
        mutate=settings.decoys.regions.mutate,
        repack=settings.decoys.regions.repack,
        minimize=settings.decoys.regions.minimize,
    )
    context = DecoyContext(
        pose=loaded.pose,
        nodes=loaded.nodes,
        pairs=pairs,
        regions=regions,
        settings=settings,
    )
    # `mutation` has no settings field, so replay uses the generator's default. A run made
    # with `mutation="packer_task"` therefore reports a mismatch rather than passing
    # quietly — the two paths land in different rotamer minima (see decoys/identity.py).
    return IdentityDecoyGenerator(
        context,
        scope=settings.decoys.scope,
        identity=settings.decoys.identity,
        placement=settings.decoys.placement,
        base_seed=settings.decoys.base_seed,
        relax=settings.decoys.relax,
        mc_cycles=settings.decoys.mc_cycles,
        native_repack=settings.decoys.native_repack,
    )


def _compare_energies(stored: pd.DataFrame, result: Any) -> str | None:
    """``None`` when the replay is bit-identical, else a one-line description of the gap."""
    stored = stored.sort_values("pair_id", kind="stable")
    order = np.argsort(result.pair_id, kind="stable")

    stored_ids = stored["pair_id"].to_numpy(dtype=np.int32)
    if not np.array_equal(stored_ids, np.asarray(result.pair_id)[order]):
        return (
            f"pair set differs: {stored_ids.size} pairs stored, {order.size} regenerated"
            if stored_ids.size != order.size
            else "pair ids differ from the stored graph"
        )

    for column, regenerated in (
        ("e_direct", np.asarray(result.e_direct)[order]),
        ("e_fa_rep", np.asarray(result.e_fa_rep)[order]),
    ):
        expected = stored[column].to_numpy(dtype=np.float32)
        regenerated = regenerated.astype(np.float32)
        if not np.array_equal(expected, regenerated):
            differing = int(np.count_nonzero(expected != regenerated))
            worst = float(np.max(np.abs(expected - regenerated)))
            return (
                f"{column} differs on {differing} of {expected.size} pairs, "
                f"max |delta| = {worst:.4g}"
            )
    return None


# ----------------------------------------------------------------- metrics selftest


def metrics_cases() -> dict[str, Any]:
    """Every public function of :mod:`atomfrust.metrics` on one fixed synthetic fixture.

    The fixture is deliberately small and fully determined by :data:`SELFTEST_SEED`: 5
    targets of 40 molecules, 8 actives each, scores separated by 0.8 sigma. Bootstrap and
    permutation counts are part of the fixture, not tuning knobs — changing one changes
    every dependent value and so requires a new golden file.

    ``groups`` is passed wherever a resampled estimate is asked for, because the screening
    metrics raise on ``groups=None`` with ``n_boot > 0`` (methods §3.9: molecules within a
    target are not independent). Every target carries both classes, so no replicate is
    degenerate and no replicate is silently dropped.
    """
    from atomfrust.metrics import (
        adjusted_logauc,
        auroc,
        bedroc,
        benjamini_hochberg,
        bootstrap_ci,
        enrichment_factor,
        maxT_permutation,
        paired_delta,
        pearson,
        spearman,
    )

    rng = np.random.default_rng(SELFTEST_SEED)
    n_targets, per_target, actives_per_target = 5, 40, 8

    blocks = []
    for _ in range(n_targets):
        block = np.zeros(per_target)
        block[rng.permutation(per_target)[:actives_per_target]] = 1.0
        blocks.append(block)
    labels = np.concatenate(blocks)
    groups = np.repeat([f"T{k}" for k in range(n_targets)], per_target)
    scores = rng.normal(loc=0.8 * labels, scale=1.0)

    x = rng.normal(size=labels.size)
    y = 0.6 * x + rng.normal(size=labels.size)
    method_a = rng.normal(size=labels.size)
    method_b = method_a - 0.25 + rng.normal(scale=0.5, size=labels.size)

    n_grid = 25
    outcome = rng.normal(size=n_grid)
    grid = {f"cfg{k}": 0.2 * k * outcome + rng.normal(size=n_grid) for k in range(6)}

    # The worked example of Benjamini & Hochberg (1995), so a wrong step-up is visible as a
    # wrong number rather than only as a hash mismatch.
    bh_p = [
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
        0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
    ]

    n_boot, n_perm = 200, 500
    return {
        "auroc_point": auroc(scores, labels),
        "auroc_cluster_bootstrap": auroc(scores, labels, groups=groups, n_boot=n_boot, seed=7),
        "bedroc_alpha_80.5": bedroc(scores, labels),
        "bedroc_alpha_20": bedroc(scores, labels, alpha=20.0),
        "adjusted_logauc": adjusted_logauc(scores, labels),
        "enrichment_factor_1pct": enrichment_factor(scores, labels),
        "enrichment_factor_5pct": enrichment_factor(scores, labels, fraction=0.05),
        "pearson_point": pearson(x, y),
        "pearson_cluster_bootstrap": pearson(x, y, groups=groups, n_boot=n_boot, seed=7),
        "spearman_point": spearman(x, y),
        "spearman_cluster_bootstrap": spearman(x, y, groups=groups, n_boot=n_boot, seed=7),
        "paired_delta": paired_delta(method_a, method_b, groups=groups, n_boot=n_boot, seed=7),
        "maxT_permutation": maxT_permutation(grid, outcome, n_perm=n_perm, seed=3),
        "benjamini_hochberg": benjamini_hochberg(bh_p),
        "bootstrap_ci_mean_score": bootstrap_ci(
            np.column_stack([scores, labels]),
            lambda rows: float(rows[:, 0].mean()),
            groups=groups,
            n_boot=n_boot,
            seed=7,
        ),
    }


def canonicalise(obj: Any) -> Any:
    """JSON-able form with every float rounded to :data:`SIGNIFICANT_DIGITS`.

    Rounding happens here and nowhere else, so the golden file, the digest and the printed
    diff all agree on what "the same value" means.
    """
    if isinstance(obj, float) or isinstance(obj, np.floating):
        value = float(obj)
        if not np.isfinite(value):
            return str(value)  # "nan"/"inf" survive JSON; the bare float does not
        return float(f"{value:.{SIGNIFICANT_DIGITS}g}")
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, pd.DataFrame):
        return {name: canonicalise(obj[name].tolist()) for name in obj.columns}
    if isinstance(obj, pd.Series):
        return canonicalise(obj.tolist())
    if isinstance(obj, np.ndarray):
        return [canonicalise(v) for v in obj.tolist()]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return canonicalise(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): canonicalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonicalise(v) for v in obj]
    raise TypeError(f"cannot canonicalise {type(obj).__name__}")


def selftest_payload() -> dict[str, Any]:
    """The full golden-file body: schema, rounding, cases and their digest."""
    cases = {name: canonicalise(value) for name, value in metrics_cases().items()}
    return {
        "schema_version": SELFTEST_SCHEMA,
        "significant_digits": SIGNIFICANT_DIGITS,
        "seed": SELFTEST_SEED,
        "digest": _digest_cases(cases),
        "cases": cases,
    }


def _digest_cases(cases: Mapping[str, Any]) -> str:
    return sha256_bytes(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode())


def run_selftest(args: argparse.Namespace) -> int:
    golden_path = Path(args.golden)
    payload = selftest_payload()
    print(f"{len(payload['cases'])} cases, {SIGNIFICANT_DIGITS} significant digits")
    print(f"computed digest: {payload['digest']}")

    if args.emit:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {golden_path}")
        return 0

    if not golden_path.exists():
        print(f"FAIL: no golden file at {golden_path}; write one with --emit")
        return 1
    try:
        golden = json.loads(golden_path.read_text())
    except ValueError as exc:
        print(f"FAIL: {golden_path} is not readable JSON: {_one_line(exc)}")
        return 1

    if golden.get("schema_version") != SELFTEST_SCHEMA:
        print(
            f"FAIL: {golden_path} has schema_version {golden.get('schema_version')!r}, this "
            f"build writes {SELFTEST_SCHEMA}. The fixture changed, so the stored values "
            "describe different inputs; refresh it with --emit."
        )
        return 1
    if golden.get("significant_digits") != SIGNIFICANT_DIGITS:
        print(
            f"FAIL: {golden_path} was rounded to {golden.get('significant_digits')!r} "
            f"significant digits, this build uses {SIGNIFICANT_DIGITS}"
        )
        return 1

    # The digest is recomputed from the golden file's own cases rather than read from its
    # `digest` field. Trusting the stored field would let a hand-edited value pass: the
    # field would still hold the digest of the values as originally written.
    golden_cases = golden.get("cases", {})
    stored_digest = _digest_cases(golden_cases)
    print(f"golden digest:   {stored_digest}  ({golden_path})")

    differing = _differing_cases(golden_cases, payload["cases"])
    if differing:
        print(f"FAIL: {len(differing)} case(s) differ from the golden file")
        for name, expected, actual in differing:
            print(f"  {name}\n    golden: {expected}\n    here:   {actual}")
        return 1

    if golden.get("digest") != stored_digest:
        print(
            f"FAIL: the values match, but {golden_path} records digest "
            f"{golden.get('digest')} for contents that hash to {stored_digest}; the file "
            "has been edited since it was written. Rewrite it with --emit."
        )
        return 1

    print("PASS: every metric value matches the golden file")
    return 0


def _differing_cases(
    golden: Mapping[str, Any], computed: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Per-case diff, so a mismatch points at a function instead of at a hash."""
    return [
        (name, golden.get(name, "<absent>"), computed.get(name, "<absent>"))
        for name in sorted(set(golden) | set(computed))
        if golden.get(name, "<absent>") != computed.get(name, "<absent>")
    ]
