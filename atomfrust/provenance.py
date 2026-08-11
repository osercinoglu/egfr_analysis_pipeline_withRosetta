"""Digests, environment capture, and the run manifest.

Requirement R37 from the methods document: *"structured logging enabling regeneration of any
reported number from a recorded configuration"*. Nothing in the prototype recorded what
settings produced a parquet, so a stored result could not be traced to the run that made it.

Three pieces:

* :func:`sha256_file` — content digests for every input that can change an answer.
* :func:`capture_environment` — Python, platform, CPU, git SHA + dirty flag, package
  versions, PyRosetta version. Read from distribution metadata, **never** by importing
  PyRosetta, so the ``unit`` tier stays usable without it.
* :class:`Manifest` — the ``manifest.json`` of §4, with ``schema_version`` enforced on read.
  An unknown version raises instead of being interpreted optimistically.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atomfrust.settings import Settings, regeneration_key, stage_subset

__all__ = [
    "SCHEMA_VERSION",
    "CONTRACT_VERSION",
    "sha256_file",
    "sha256_bytes",
    "capture_environment",
    "env_digest",
    "CodeInfo",
    "Manifest",
    "ManifestVersionError",
    "build_manifest",
]

#: Bumped when the manifest layout changes incompatibly.
SCHEMA_VERSION = 1
#: Bumped when the run-directory layout changes incompatibly.
CONTRACT_VERSION = 1

#: Packages whose version can change a number. Deliberately short — capturing the whole
#: environment makes every manifest differ and the digest useless as a comparison.
TRACKED_PACKAGES = ("atomfrust", "numpy", "pandas", "pyarrow", "scipy", "pydantic")


class ManifestVersionError(ValueError):
    """A manifest was written by an incompatible version of the contract."""


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Streaming digest, so a large structure file does not have to be held in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class CodeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_version: str
    git_sha: str | None = None
    dirty: bool = False


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def capture_code_info(repo: Path | None = None) -> CodeInfo:
    """Package version plus git state. A dirty tree is recorded, never hidden."""
    from atomfrust import __version__

    sha = _git("rev-parse", "HEAD", cwd=repo)
    status = _git("status", "--porcelain", cwd=repo)
    return CodeInfo(
        package_version=__version__,
        git_sha=sha,
        dirty=bool(status) if status is not None else False,
    )


def _package_version(name: str) -> str | None:
    """Version from distribution metadata. Does not import the package.

    This matters for PyRosetta specifically: importing it costs seconds and pulls in a
    database, and the unit tier must run on a machine that has never installed it.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def capture_environment(repo: Path | None = None) -> dict[str, Any]:
    """Everything about the machine that could change a number."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "pyrosetta": _package_version("pyrosetta"),
        "packages": {n: _package_version(n) for n in TRACKED_PACKAGES},
        "code": capture_code_info(repo).model_dump(),
    }


def env_digest(environment: dict[str, Any]) -> str:
    """Digest of the reproducibility-relevant subset.

    ``cpu_count`` and ``platform`` are excluded: results must not depend on how many cores
    ran them, and requiring an identical OS string would make cross-machine comparison
    impossible. Both are still *recorded* in ``env.json`` — they are diagnostics, not part
    of the identity.
    """
    relevant = {
        "python": environment.get("python"),
        "pyrosetta": environment.get("pyrosetta"),
        "packages": environment.get("packages"),
    }
    return sha256_bytes(
        json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
    )


class Manifest(BaseModel):
    """``manifest.json``. See §4 of the plan; this model is the normative form."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    contract_version: int = CONTRACT_VERSION
    run_id: str
    created_utc: str
    code: CodeInfo
    env_digest: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    settings_resolved: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    regeneration_key: str

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2, sort_keys=True) + "\n"

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        data = json.loads(Path(path).read_text())
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ManifestVersionError(
                f"{path}: manifest schema_version {version!r}, this build writes "
                f"{SCHEMA_VERSION}. Refusing to interpret it — a silently mis-read manifest "
                "is worse than no manifest."
            )
        contract = data.get("contract_version")
        if contract != CONTRACT_VERSION:
            raise ManifestVersionError(
                f"{path}: run-directory contract_version {contract!r}, this build "
                f"implements {CONTRACT_VERSION}."
            )
        return cls.model_validate(data)


def build_manifest(
    run_id: str,
    settings: Settings,
    created_utc: str,
    input_digests: dict[str, Any] | None = None,
    repo: Path | None = None,
) -> Manifest:
    """Assemble a manifest from live settings and environment.

    ``created_utc`` is passed in rather than read from the clock so callers stay
    deterministic and testable.
    """
    environment = capture_environment(repo)
    digests = dict(input_digests or {})
    return Manifest(
        run_id=run_id,
        created_utc=created_utc,
        code=CodeInfo.model_validate(environment["code"]),
        env_digest=env_digest(environment),
        inputs=digests,
        settings_resolved=settings.model_dump(mode="json"),
        generation=stage_subset(settings, "generation"),
        regeneration_key=regeneration_key(
            settings, digests, environment.get("pyrosetta")
        ),
    )
