"""smina — the open docking baseline, driven as a subprocess.

smina is an AutoDock Vina fork with a scriptable scoring function and an ``--autobox_ligand``
flag that derives the search box from the native ligand. It is the open baseline named in
methods §3.7 and the pose generator for the pose axis (G5).

**Subprocess, never an import.** smina is a C++ binary with no Python bindings worth the
dependency, so :meth:`available` is ``shutil.which("smina")`` and nothing here fails to
import when it is absent (see :mod:`atomfrust.dock.base`).

**The box comes from the reference ligand when there is one.** ``--autobox_ligand <native>``
plus a small ``--autobox_add`` margin reproduces the standard re-docking setup; without a
reference the whole receptor becomes the box, which is blind docking — a different, much
harder experiment that should be labelled as such rather than entered by accident. The
metadata records which of the two ran (``blind``).

**The score is a comparator, not the measurement.** ``minimizedAffinity`` (kcal/mol, negative
is better) goes into :attr:`~atomfrust.dock.base.Pose3D.score` so it can be compared against
the Rosetta-derived Z-score under methods §3.7's control — but only against the *raw* Rosetta
interaction energy, since comparing a Z-score to a raw Vina score confounds normalisation
with energy-function quality.

**A pose from here is a candidate, not an input.** Every pose goes through
:func:`atomfrust.dock.posebusters.check_poses` before analysis (R23, S4.2).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from atomfrust.dock.base import (
    Chem,
    Pose3D,
    SubprocessBackend,
    as_mol_with_conformer,
    embedded_mol,
    molblock_from_mol,
    require_rdkit,
    symmetry_rmsd,
)

__all__ = ["SminaBackend"]


class SminaBackend(SubprocessBackend):
    """Dock one molecule into one receptor with smina.

    ``exhaustiveness`` is Vina's search effort. It is a *reproducibility* parameter as much
    as an accuracy one: at low exhaustiveness the same seed still lands in different minima
    across builds, so a comparison of pass rates (S4.2) between generators has to hold it
    fixed and record it, which is why it appears in every pose's metadata.

    ``cpu`` defaults to 1. Docking is called from inside the package's spawn pools, and a
    binary that helpfully grabs every core turns an N-way parallel run into an N-way
    oversubscribed one.
    """

    name = "smina"
    binary = "smina"
    version_args = ("--version",)

    #: SDF property holding the backend's own score, best-first in the output file.
    score_property = "minimizedAffinity"
    #: Appended to every command line. Split out so :class:`~atomfrust.dock.gnina.GninaBackend`
    #: can add its CNN flags without restating the shared invocation.
    extra_args: tuple[str, ...] = ()

    def __init__(
        self,
        binary: str | None = None,
        *,
        exhaustiveness: int = 8,
        autobox_add: float = 4.0,
        cpu: int = 1,
        timeout_s: float = 900.0,
    ) -> None:
        super().__init__(binary=binary, timeout_s=timeout_s)
        self.exhaustiveness = exhaustiveness
        self.autobox_add = autobox_add
        self.cpu = cpu

    def pose(
        self,
        smiles_or_mol: str | Any,
        receptor_pdb: Path | str,
        reference_ligand: str | Path | Any | None = None,
        n_poses: int = 1,
        seed: int = 0,
    ) -> list[Pose3D]:
        """Dock and return up to ``n_poses`` poses, best-scoring first.

        Raises :class:`~atomfrust.dock.base.BackendUnavailable` when the binary is missing —
        callers that want a question rather than an exception ask :meth:`available` first.
        """
        require_rdkit()
        binary = self._require_binary()
        receptor = Path(receptor_pdb)
        if not receptor.exists():
            raise ValueError(f"receptor not found: {receptor}")

        reference = (
            as_mol_with_conformer(reference_ligand) if reference_ligand is not None else None
        )
        ligand = embedded_mol(smiles_or_mol, seed=seed)

        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as tmp:
            tmpdir = Path(tmp)
            ligand_sdf = tmpdir / "ligand.sdf"
            _write_sdf(ligand, ligand_sdf)

            if reference is not None:
                box_source = tmpdir / "reference.sdf"
                _write_sdf(reference, box_source)
                blind = False
            else:
                # smina accepts any structure file here; the receptor itself boxes everything.
                box_source = receptor
                blind = True

            out_sdf = tmpdir / "docked.sdf"
            command = [
                binary,
                "--receptor", str(receptor),
                "--ligand", str(ligand_sdf),
                "--autobox_ligand", str(box_source),
                "--autobox_add", f"{self.autobox_add:g}",
                "--num_modes", str(max(1, n_poses)),
                "--exhaustiveness", str(self.exhaustiveness),
                "--seed", str(seed),
                "--cpu", str(self.cpu),
                "--out", str(out_sdf),
                *self.extra_args,
            ]
            result = self._run(command)
            if not out_sdf.exists() or out_sdf.stat().st_size == 0:
                raise RuntimeError(
                    f"{self.name} produced no output (exit {result.returncode}). "
                    f"stderr: {(result.stderr or '')[-400:]}"
                )
            molblocks = self._read_output(out_sdf)

        version = self.version()
        poses: list[Pose3D] = []
        for mode, (molblock, score) in enumerate(molblocks[: max(1, n_poses)]):
            rmsd = None
            if reference is not None:
                try:
                    rmsd = symmetry_rmsd(Chem.MolFromMolBlock(molblock, removeHs=False), reference)
                except Exception:
                    rmsd = None
            poses.append(
                Pose3D(
                    molblock=molblock,
                    source=self.name,
                    score=score,
                    rmsd_to_reference=rmsd,
                    metadata={
                        "binary": binary,
                        "version": version,
                        "command": " ".join(command),
                        "mode": mode,
                        "seed": seed,
                        "exhaustiveness": self.exhaustiveness,
                        "autobox_add": self.autobox_add,
                        "blind": blind,
                        "score_property": self.score_property,
                        "smiles": _input_smiles(smiles_or_mol, ligand),
                    },
                )
            )
        if not poses:
            raise RuntimeError(f"{self.name} wrote an output file containing no poses")
        return poses

    def _read_output(self, out_sdf: Path) -> list[tuple[str, float | None]]:
        """Molblock and score per docked mode, in the order smina wrote them (best first).

        The molblock is taken from RDKit's re-serialisation rather than the raw text because
        smina writes its scores as SDF *properties*, and the property block has to be dropped
        for the molblock to parse standalone.
        """
        supplier = Chem.SDMolSupplier(str(out_sdf), removeHs=False, sanitize=True)
        records: list[tuple[str, float | None]] = []
        for mol in supplier:
            if mol is None:
                continue
            records.append((molblock_from_mol(mol), _score_of(mol, self.score_property)))
        return records


def _score_of(mol: Any, prop: str) -> float | None:
    for name in (prop, "minimizedAffinity", "CNNaffinity", "affinity"):
        if mol.HasProp(name):
            try:
                return float(mol.GetProp(name))
            except ValueError:
                continue
    return None


def _write_sdf(mol: Any, path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()


def _input_smiles(smiles_or_mol: str | Any, prepared: Any) -> str:
    """The identity the pose is supposed to have — the reference for the gate's stereo and
    formal-charge checks, which otherwise have nothing to compare a docked pose against."""
    try:
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
            return Chem.MolToSmiles(mol) if mol is not None else ""
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(prepared)))
    except Exception:
        return ""
