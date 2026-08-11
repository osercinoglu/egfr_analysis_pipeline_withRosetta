"""The run directory — the interface everything else consumes.

Layout is §4 of the plan. Three properties this module exists to guarantee:

**Direct energies only.** ``E_ij`` is never written. The decoy store holds ``e_direct`` and
``e_fa_rep`` per pair, so the many-body formula, the contact definition, the shell, the
index function and ``exclude_fa_rep`` are all re-specifiable against a finished run. That
single choice is what makes user requests 5, 6 and 7 possible, and it is why a corrected
many-body formula does not require regenerating decoys.

**N is a prefix.** Decoy *i* is seeded ``base_seed + i``, so the first N decoys of a
1000-decoy run *are* the ensemble a N-decoy run would produce. Reading a prefix is exact,
not an approximation, which turns the convergence sweep from eight runs into one.

**Shards never contend.** Each shard writes only its own part files, so there is no shared
checkpoint and no lock. This retires two prototype defects at once: the single
``{PDB}_{LIG}_ckpt.pkl`` whose default save interval equalled the usual decoy count (so an
interruption at decoy 49 of 50 lost everything), and the parquet short-circuit at
``run_pipeline.py:241`` which made re-running with a larger ``--n_decoys`` silently do
nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from atomfrust.provenance import Manifest
from atomfrust.settings import Settings, regeneration_key, stage_subset

__all__ = [
    "EXIT_REQUIRES_REGENERATION",
    "RegenerationRequired",
    "FieldDiff",
    "CompatibilityReport",
    "DecoyEnergyWriter",
    "SystemDir",
    "RunDir",
    "DECOY_ENERGY_COLUMNS",
]

#: Exit code for "this run directory cannot answer that question without regenerating".
EXIT_REQUIRES_REGENERATION = 3

DECOY_ENERGY_COLUMNS = ("decoy_id", "axis", "pair_id", "e_direct", "e_fa_rep")

Verdict = Literal["ok", "analysis_only", "requires_regeneration"]


class RegenerationRequired(RuntimeError):
    """Raised when requested settings need a new decoy ensemble."""

    exit_code = EXIT_REQUIRES_REGENERATION

    def __init__(self, report: CompatibilityReport) -> None:
        super().__init__(report.explain())
        self.report = report


@dataclass(frozen=True)
class FieldDiff:
    path: str
    stored: Any
    requested: Any
    stage: str


@dataclass
class CompatibilityReport:
    verdict: Verdict
    differences: list[FieldDiff] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        return self.verdict != "requires_regeneration"

    def explain(self) -> str:
        if self.verdict == "ok":
            return "settings match the stored run exactly"
        header = {
            "analysis_only": "settings differ only in analysis-stage fields; "
            "the stored decoys remain valid",
            "requires_regeneration": "settings differ in generation-stage fields; "
            "the stored decoys cannot answer this",
        }[self.verdict]
        lines = [
            f"  {d.path}: stored={d.stored!r} requested={d.requested!r}  [{d.stage}]"
            for d in self.differences
        ]
        return header + "\n" + "\n".join(lines)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


class DecoyEnergyWriter:
    """Append-only writer for one shard.

    Buffers whole decoys and flushes to a new part file every ``flush_every`` of them.
    Partial work therefore survives an interruption at flush granularity, and because a
    shard owns its files exclusively there is nothing to lock.
    """

    def __init__(
        self,
        directory: Path,
        shard: int = 0,
        flush_every: int = 10,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard = int(shard)
        self.flush_every = max(1, int(flush_every))
        self._buffer: list[pd.DataFrame] = []
        self._pending = 0
        # Resume past this shard's existing part files. Numbering from zero on every
        # construction would make a resumed shard overwrite `part-<shard>-00000.parquet`
        # and destroy the decoys already in it — silent data loss, since the writer has no
        # reason to think the file it is creating already existed.
        self._sequence = self._next_sequence()

    def _next_sequence(self) -> int:
        existing = self.directory.glob(f"part-{self.shard:03d}-*.parquet")
        used = []
        for path in existing:
            try:
                used.append(int(path.stem.rsplit("-", 1)[1]))
            except (IndexError, ValueError):  # a file we did not write; leave it alone
                continue
        return max(used) + 1 if used else 0

    def add(
        self,
        decoy_id: int,
        pair_id: np.ndarray,
        e_direct: np.ndarray,
        e_fa_rep: np.ndarray,
        axis: str = "identity",
    ) -> None:
        order = np.argsort(pair_id, kind="stable")
        frame = pd.DataFrame(
            {
                "decoy_id": np.full(len(pair_id), int(decoy_id), dtype=np.int32),
                "axis": pd.Categorical([axis] * len(pair_id)),
                "pair_id": np.asarray(pair_id, dtype=np.int32)[order],
                "e_direct": np.asarray(e_direct, dtype=np.float32)[order],
                "e_fa_rep": np.asarray(e_fa_rep, dtype=np.float32)[order],
            }
        )
        self._buffer.append(frame)
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        frame = pd.concat(self._buffer, ignore_index=True)
        path = self.directory / f"part-{self.shard:03d}-{self._sequence:05d}.parquet"
        frame.to_parquet(path, index=False)
        self._buffer.clear()
        self._pending = 0
        self._sequence += 1
        return path

    def __enter__(self) -> DecoyEnergyWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()


@dataclass
class SystemDir:
    """Paths and IO for one system inside a run."""

    root: Path
    system_id: str

    @property
    def base(self) -> Path:
        return self.root / "systems" / self.system_id

    @property
    def inputs(self) -> Path:
        return self.base / "inputs"

    @property
    def graph(self) -> Path:
        return self.base / "graph"

    @property
    def native(self) -> Path:
        return self.base / "native"

    @property
    def decoys(self) -> Path:
        return self.base / "decoys"

    @property
    def decoy_energies(self) -> Path:
        return self.decoys / "energies"

    @property
    def analyses(self) -> Path:
        return self.base / "analyses"

    @property
    def status_path(self) -> Path:
        return self.base / "STATUS.json"

    def ensure(self) -> SystemDir:
        for path in (self.inputs, self.graph, self.native, self.decoy_energies, self.analyses):
            path.mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------------- status

    def read_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {"state": "absent", "n_decoys_done": 0, "n_decoys_target": 0}
        return json.loads(self.status_path.read_text())

    def update_status(self, **fields: Any) -> dict[str, Any]:
        status = self.read_status()
        status.update(fields)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return status

    # -------------------------------------------------------------- graph

    def write_graph(self, nodes: pd.DataFrame, pairs: pd.DataFrame) -> None:
        self.graph.mkdir(parents=True, exist_ok=True)
        nodes.to_parquet(self.graph / "nodes.parquet", index=False)
        pairs.to_parquet(self.graph / "pairs.parquet", index=False)

    def read_graph(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            pd.read_parquet(self.graph / "nodes.parquet"),
            pd.read_parquet(self.graph / "pairs.parquet"),
        )

    # ------------------------------------------------------------- native

    def write_native_energies(self, energies: pd.DataFrame) -> None:
        self.native.mkdir(parents=True, exist_ok=True)
        energies.to_parquet(self.native / "native_energies.parquet", index=False)

    def read_native_energies(self) -> pd.DataFrame:
        return pd.read_parquet(self.native / "native_energies.parquet")

    def write_raw_energy(self, raw: dict[str, Any]) -> None:
        """The S5.2 control (see `atomfrust.energy.raw_interaction_energy`).

        Written at generation time because it is free there and impossible to reconstruct
        later without the pose.
        """
        self.native.mkdir(parents=True, exist_ok=True)
        (self.native / "raw_energy.json").write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n"
        )

    def read_raw_energy(self) -> dict[str, Any]:
        path = self.native / "raw_energy.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is absent; this run predates the S5.2 raw-energy control or was "
                "written by a different tool"
            )
        return json.loads(path.read_text())

    # ------------------------------------------------------------- decoys

    def energy_writer(self, shard: int = 0, flush_every: int = 10) -> DecoyEnergyWriter:
        return DecoyEnergyWriter(self.decoy_energies, shard=shard, flush_every=flush_every)

    def decoy_parts(self) -> list[Path]:
        return sorted(self.decoy_energies.glob("part-*.parquet"))

    def read_decoy_energies(
        self,
        n_decoys: int | None = None,
        axes: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Read the decoy store, optionally truncated to the first ``n_decoys``.

        The prefix is **exact**, not a sample: decoy *i* is seeded ``base_seed + i``, so
        ``decoy_id < N`` is precisely the ensemble an N-decoy run would have produced.
        """
        parts = self.decoy_parts()
        if not parts:
            return pd.DataFrame(
                {
                    "decoy_id": pd.Series(dtype="int32"),
                    "axis": pd.Series(dtype="object"),
                    "pair_id": pd.Series(dtype="int32"),
                    "e_direct": pd.Series(dtype="float32"),
                    "e_fa_rep": pd.Series(dtype="float32"),
                }
            )
        frames = []
        for path in parts:
            frame = pd.read_parquet(path)
            if n_decoys is not None:
                frame = frame[frame["decoy_id"] < int(n_decoys)]
            if axes is not None:
                frame = frame[frame["axis"].astype(str).isin(set(axes))]
            if len(frame):
                frames.append(frame)
        if not frames:
            return pd.read_parquet(parts[0]).iloc[:0]
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["decoy_id", "pair_id"], kind="stable").reset_index(drop=True)

    def completed_decoy_ids(self, axis: str | None = None) -> set[int]:
        """Decoy ids already on disk — the basis for resuming without recomputation."""
        ids: set[int] = set()
        for path in self.decoy_parts():
            frame = pd.read_parquet(path, columns=["decoy_id", "axis"])
            if axis is not None:
                frame = frame[frame["axis"].astype(str) == axis]
            ids.update(int(v) for v in frame["decoy_id"].unique())
        return ids


@dataclass
class RunDir:
    """A run directory. Create with :meth:`create`, reopen with the constructor."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def settings_path(self) -> Path:
        return self.root / "settings.resolved.yaml"

    @property
    def env_path(self) -> Path:
        return self.root / "env.json"

    @property
    def log_path(self) -> Path:
        return self.root / "logs" / "generate.jsonl"

    def system(self, system_id: str) -> SystemDir:
        return SystemDir(self.root, system_id)

    def systems(self) -> list[str]:
        base = self.root / "systems"
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []

    # ------------------------------------------------------------- create

    @classmethod
    def create(
        cls,
        root: str | Path,
        manifest: Manifest,
        settings: Settings,
        environment: dict[str, Any] | None = None,
    ) -> RunDir:
        run = cls(Path(root))
        run.root.mkdir(parents=True, exist_ok=True)
        (run.root / "systems").mkdir(exist_ok=True)
        run.log_path.parent.mkdir(exist_ok=True)
        manifest.write(run.manifest_path)
        run.settings_path.write_text(settings.to_yaml())
        if environment is not None:
            run.env_path.write_text(
                json.dumps(environment, indent=2, sort_keys=True) + "\n"
            )
        return run

    def read_manifest(self) -> Manifest:
        return Manifest.read(self.manifest_path)

    def read_settings(self) -> Settings:
        return Settings.from_yaml(self.settings_path.read_text())

    # ------------------------------------------------------ compatibility

    def check_compatible(
        self, settings: Settings, pyrosetta_version: str | None = None
    ) -> CompatibilityReport:
        """Compare requested settings against the stored run.

        Analysis-stage differences are fine — that is the whole point of storing direct
        energies. Generation-stage differences are not, and the report names every field so
        the cause is obvious without diffing YAML by eye.
        """
        manifest = self.read_manifest()
        stored_settings = self.read_settings()

        stored_gen = _flatten(manifest.generation)
        requested_gen = _flatten(stage_subset(settings, "generation"))
        differences = [
            FieldDiff(path, stored_gen.get(path), requested_gen.get(path), "generation")
            for path in sorted(set(stored_gen) | set(requested_gen))
            if stored_gen.get(path) != requested_gen.get(path)
        ]
        if differences:
            return CompatibilityReport("requires_regeneration", differences)

        # Generation settings match. A key mismatch then means the *inputs* or the
        # PyRosetta version moved, which is equally disqualifying and easy to miss.
        requested_key = regeneration_key(
            settings, manifest.inputs, pyrosetta_version
        )
        if pyrosetta_version is not None and requested_key != manifest.regeneration_key:
            return CompatibilityReport(
                "requires_regeneration",
                [
                    FieldDiff(
                        "regeneration_key",
                        manifest.regeneration_key,
                        requested_key,
                        "generation",
                    )
                ],
            )

        stored_ana = _flatten(stage_subset(stored_settings, "analysis"))
        requested_ana = _flatten(stage_subset(settings, "analysis"))
        analysis_differences = [
            FieldDiff(path, stored_ana.get(path), requested_ana.get(path), "analysis")
            for path in sorted(set(stored_ana) | set(requested_ana))
            if stored_ana.get(path) != requested_ana.get(path)
        ]
        if analysis_differences:
            return CompatibilityReport("analysis_only", analysis_differences)
        return CompatibilityReport("ok", [])

    def assert_compatible(
        self,
        settings: Settings,
        pyrosetta_version: str | None = None,
        allow_mismatch: bool = False,
    ) -> CompatibilityReport:
        """Raise :class:`RegenerationRequired` (exit code 3) on a generation-stage change.

        ``allow_mismatch`` downgrades the error to a report, which callers must then stamp
        as provisional on every output they write.
        """
        report = self.check_compatible(settings, pyrosetta_version)
        if report.verdict == "requires_regeneration" and not allow_mismatch:
            raise RegenerationRequired(report)
        return report
