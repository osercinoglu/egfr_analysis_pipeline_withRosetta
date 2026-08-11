"""Content-addressed store for parametrisation results.

A library is re-parametrised constantly — a new decoy set, a rerun, a second target that
shares half its molecules — and every parametrisation costs an RDKit embedding plus a
subprocess. The cache makes the second pass free.

**The key is not the molecule alone.** It is
``"{inchikey}|{protonation_version}|{conformer_version}"``. An InChIKey does not distinguish
protonation states or tautomers (G2 enumerates them at pH 7.4), and it says nothing about how
the conformer was generated. Bumping either version is therefore a cache miss for every
molecule, which is the point: silently reusing params built under a different protonation
model would make a run's numbers unreproducible from its manifest. This mirrors
:func:`atomfrust.settings.regeneration_key` — identity includes the recipe, not just the input.

**Failures are cached too.** A molecule that cannot be parametrised must not be retried on
every pass, and its category is the deliverable: :meth:`failures` is the ``groupby`` that
answers "all failures categorised" (methods document S1.1).

**One file per record.** No shared index to read-modify-write, so parallel workers writing
different molecules never contend and never corrupt each other — the same reasoning that
made :mod:`atomfrust.runstore` shard its parquet output.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from atomfrust.chem.paramize import ParamRecord

__all__ = ["ParamCache", "RECORD_COLUMNS"]

RECORD_COLUMNS = (
    "inchikey", "smiles", "rosetta_code", "params_path", "sha256", "failure", "message",
)


class ParamCache:
    """Records under ``root/records/``, params files under ``root/params/``."""

    def __init__(
        self,
        root: Path | str,
        protonation_version: str = "v1",
        conformer_version: str = "v1",
    ) -> None:
        self.root = Path(root)
        self.protonation_version = protonation_version
        self.conformer_version = conformer_version
        self.records_dir = self.root / "records"
        self.params_dir = self.root / "params"

    def key(self, inchikey: str) -> str:
        """The cache key. Read it in a failure report and you know why something missed."""
        return f"{inchikey}|{self.protonation_version}|{self.conformer_version}"

    def _path(self, inchikey: str) -> Path:
        """Digest the key rather than using it as a filename: an InChIKey is filesystem-safe
        but the version fields are caller-supplied strings and need not be."""
        digest = hashlib.sha256(self.key(inchikey).encode()).hexdigest()[:32]
        return self.records_dir / f"{digest}.json"

    def get(self, inchikey: str) -> ParamRecord | None:
        path = self._path(inchikey)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            # A truncated record is a miss, not a crash: a half-written file from a killed
            # worker must never take down the next run.
            return None
        return ParamRecord.from_dict(data["record"])

    def put(self, record: ParamRecord) -> None:
        """Write atomically, so a concurrent :meth:`get` sees the old record or the new one."""
        if not record.inchikey:
            raise ValueError("cannot cache a record without an inchikey")
        self.records_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(record.inchikey)
        payload = {
            "key": self.key(record.inchikey),
            "protonation_version": self.protonation_version,
            "conformer_version": self.conformer_version,
            "record": record.to_dict(),
        }
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)

    # ------------------------------------------------------------------ reporting

    def records(self) -> pd.DataFrame:
        """Every record under this cache root, whatever its version key.

        Deliberately not filtered to the current versions: a report about what a library did
        should show the molecules that were parametrised under an older protonation model,
        not hide them.
        """
        rows = []
        for path in sorted(self.records_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            row = dict(data["record"])
            row["protonation_version"] = data.get("protonation_version", "")
            row["conformer_version"] = data.get("conformer_version", "")
            rows.append(row)
        columns = [*RECORD_COLUMNS, "protonation_version", "conformer_version"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows)[columns]

    def failures(self) -> pd.DataFrame:
        """One row per failed molecule. ``df.groupby("failure").size()`` is the S1.1 table."""
        df = self.records()
        return df[df["failure"].notna()].reset_index(drop=True)

    def write_failures(self, path: Path | str | None = None) -> Path:
        """Materialise ``failures.parquet`` — the plan's ``chem/failures.parquet``."""
        target = Path(path) if path is not None else self.root / "failures.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.failures().to_parquet(target, index=False)
        return target

    def __len__(self) -> int:
        return sum(1 for _ in self.records_dir.glob("*.json")) if self.records_dir.exists() else 0
