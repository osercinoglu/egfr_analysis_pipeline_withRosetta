# B1 complete — `atomfrust` package skeleton, installed at flat layout

**Date:** 2026-08-11 04:15
**Plan step:** B1 of `plans/frustratometer-ng-plan.md`
**Artifacts:** `pyproject.toml`, `atomfrust/__init__.py`, `atomfrust/cli/__init__.py`,
`atomfrust/cli/main.py`, `tests/test_skeleton.py`

## Current state

B1 is done, all acceptance criteria met. Stage B has begun; B2 (`settings.py`) is unblocked
and every dependency it needs is already in the env (pydantic 2.13.4 confirmed).

The Stage 6 batch is at 51/61 with two workers left.

## What was built

- `atomfrust` package with `__version__` resolved from distribution metadata.
- `atomfrust` console script, `--version`, and an empty `SUBCOMMANDS` dispatch table. The
  parser's epilog lists the ten planned subcommands with the plan step that delivers each,
  so `--help` is honest about what does not exist yet rather than silently offering nothing.
- Three pytest markers registered (`unit` / `integration` / `anchor`).
- Five `unit` smoke tests: package imports, version is real metadata (not the
  `0.0.0+unknown` fallback, which is how a failed editable install would present), CLI
  `--version` matches the package, bare invocation prints help and exits 0, the console
  script is on PATH, and **importing `atomfrust` does not drag in PyRosetta** — checked in a
  subprocess so it stays true regardless of what else the session imported.

## Deviation from the plan: flat layout, not `src/atomfrust/`

The plan specified `src/atomfrust/`. That was changed after measuring what an editable
install actually does. Both setuptools editable modes were tested:

| mode | new modules picked up | namespace leak |
|---|---|---|
| `editable_mode=strict` | **No** — needs `pip install -e .` per new file; symlink tree under `build/` | none |
| lenient (default), `src/` layout | yes | **`frustration`, `run_pipeline`, `rosetta_py`, `molfile_to_params`, `test_frustration`, `prepare_structures` all importable as top-level names, machine-wide** |
| lenient, **flat layout** (chosen) | yes | **none** |

An editable install puts the discovery root on `sys.path`. Under a `src/` layout that root
is `src/`, which exports the legacy modules globally — and `import test_frustration`
**initialises PyRosetta as a side effect**, which is a genuine hazard, not just untidiness.
Strict mode avoids the leak but silently fails to see new modules, which over the ~15
modules Stage B–G adds would surface as `ImportError`s that look like typos.

Flat layout gets both: the discovery root is the repo root, which is already on `sys.path`
for every documented command since they all run from there. Verified: `atomfrust` resolves
to the source tree, a newly created module is importable with no reinstall, and the leak
check reports `NONE`.

`plans/frustratometer-ng-plan.md` §3.1 has been updated to match.

## Verification

| check | result |
|---|---|
| `pyrosetta` version and path unchanged | 2026.30+release.bc091c65b8 ✓ |
| numpy / pandas / scipy / pydantic unchanged | 2.2.6 / 2.3.3 / 1.15.2 / 2.13.4 ✓ |
| `atomfrust --version` | `atomfrust 0.1.0.dev0` ✓ |
| `pytest -m unit` | 5 passed in 0.20 s, exit 0 ✓ |
| `python src/run_pipeline.py --help` | runs ✓ |
| `python -m pytest src/test_frustration.py --collect-only` | 13 tests collected ✓ (matches the CLAUDE.md baseline) |
| `analysis/*.py` still import | ✓ |
| namespace leak | NONE ✓ |

Installed with `pip install -e . --no-deps`. The `--no-deps` is deliberate and documented in
`pyproject.toml`: every runtime dependency is already satisfied by `environment.yml`, and
letting pip resolve them risks it touching the conda env's PyRosetta — a 1.7 GB
reinstall from a mirror that only partly works on this machine. PyRosetta is deliberately
absent from `dependencies` for the same reason: naming it would make pip try PyPI, where
the placeholder wheel 404s.

## Note on the acceptance criterion

The plan's wording was "`pytest -m unit` collects zero tests and exits 0". That is not
achievable as written — pytest returns exit code 5 when nothing is collected. Delivered
instead: five real `unit` tests that pass with exit 0, which tests the markers *and* the
install. The plan text has been corrected.

## Next steps

1. **B2** — `settings.py`: pydantic v2, `extra="forbid"`, stage-partitioned fields,
   `regeneration_key()`. Nothing blocks it.
2. B3 (`spec.py`) then B6/B7 (pose + graph) are the critical path — B6/B7 are where the
   ligand becomes a node, which A4 identified as *the* correction.
3. Housekeeping when the batch finishes: `dvc push`, and lower
   `checkpoint.save_every_n_decoys` to 10.
