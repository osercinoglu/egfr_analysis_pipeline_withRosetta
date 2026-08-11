#!/usr/bin/env python
"""Populate the local decoy-library cache. Prints a plan by default; downloads only with --download.

The DUD-E, DEKOIS 2.0 and MUV licences permit use but not redistribution, so this script is in
git and its output is not: everything lands under
:func:`atomfrust.chem.libraries.default_cache_root` (``<repo>/.cache/libraries``, gitignored,
overridable with ``$ATOMFRUST_LIBRARY_CACHE``). Nothing here runs during tests — the test suite
reads tiny hand-written fixtures under ``tests/data/libraries/`` instead, so a green suite is
never evidence that a URL below still resolves.

**Two kinds of source, and the difference is deliberate.**

*Direct* sources have stable file URLs and are downloaded. *Manual* sources are printed as a
landing page with instructions, because their distribution goes through a form or a
registration step that a script cannot honestly automate. Guessing a plausible-looking URL for
those would produce a 404 at best and a silently wrong file at worst, so this script does not
guess: pass ``--url`` when you know the real one.

Direct (DUD-E, per target ``<t>``)::

    https://dude.docking.org/targets/<t>/actives_final.ism      SMILES, actives
    https://dude.docking.org/targets/<t>/decoys_final.ism       SMILES, property-matched decoys
    https://dude.docking.org/targets/<t>/actives_final.sdf.gz   conformers   (--with-conformers)
    https://dude.docking.org/targets/<t>/decoys_final.sdf.gz    conformers   (--with-conformers)
    https://dude.docking.org/db/subsets/all/all.tar.gz          all 102 targets (--all-targets)

Direct (ZINC control, per ``--tranche AA/AAAA``)::

    https://files.docking.org/2D/<AA>/<AAAA>.smi

Manual::

    https://www.dekois.com/                       DEKOIS 2.0, per-target SDF sets
    https://dude.docking.org/targets/<t>          DUD-E experimental (measured-inactive) sets
    https://www.tu-braunschweig.de/pharmchem/forschung/baumann/muv    MUV .dat tables

**The measured-inactive files are the ones worth the manual effort.** DUD-E's 9,219 compounds
with no measurable affinity to 30 µM — 1,070 of them for COX-1/PGH1 — are the experimental
negatives behind success criterion S3.4, and MUV's ~15,000 per target are the second such
resource (methods document #14, #21). Save them into the target directory as
``inactives_final.ism`` (``SMILES <id>`` per line); that is the filename
:class:`~atomfrust.chem.libraries.dude.DUDEAdapter` maps to ``role="measured_inactive"``. A
name it does not recognise is skipped rather than imported under the wrong role, which is the
right failure: a synthetic decoy counted as a measured non-binder would turn S3.4 back into
the circular experiment it exists to replace.

**The ZINC sample is an artefact, not a query.** ``--library zinc_random`` reservoir-samples
the downloaded tranches with an explicit ``--seed`` and writes the sample plus its seed, so the
negative control is a fixed file that can be cited, not something re-randomised per run.

Usage::

    python scripts/fetch_decoy_libraries.py --library dude --target pgh1            # plan only
    python scripts/fetch_decoy_libraries.py --library dude --target pgh1 --download
    python scripts/fetch_decoy_libraries.py --library zinc_random --tranche BA/BAAA \\
        --sample-size 5000 --seed 1234 --download
    python scripts/fetch_decoy_libraries.py --library dekois2                       # instructions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atomfrust.chem.libraries import ADAPTERS, default_cache_root, get_adapter  # noqa: E402

DUDE_TARGET_BASE = "https://dude.docking.org/targets"
DUDE_ALL_TARBALL = "https://dude.docking.org/db/subsets/all/all.tar.gz"
ZINC_2D_BASE = "https://files.docking.org/2D"
DEKOIS_LANDING = "https://www.dekois.com/"
MUV_LANDING = "https://www.tu-braunschweig.de/pharmchem/forschung/baumann/muv"

USER_AGENT = "atomfrust-fetch/0.1 (+https://github.com/; academic use)"


@dataclass(frozen=True)
class Download:
    url: str
    dest: Path
    note: str = ""


# ------------------------------------------------------------------------------ planning


def plan_dude(root: Path, targets: list[str], with_conformers: bool, all_targets: bool) -> list[Download]:
    library = root / "dude"
    if all_targets:
        return [
            Download(
                DUDE_ALL_TARBALL,
                library / "all.tar.gz",
                "unpack in place: it expands to all/<target>/, which the adapter reads",
            )
        ]
    names = ["actives_final.ism", "decoys_final.ism"]
    if with_conformers:
        # Conformers, not poses: free-molecule 3D used as docking input, never as a binding mode.
        names += ["actives_final.sdf.gz", "decoys_final.sdf.gz"]
    return [
        Download(f"{DUDE_TARGET_BASE}/{t}/{name}", library / "all" / t / name)
        for t in targets
        for name in names
    ]


def plan_zinc(root: Path, tranches: list[str]) -> list[Download]:
    library = root / "zinc_random"
    downloads = []
    for tranche in tranches:
        code = tranche.strip("/")
        prefix, _, leaf = code.partition("/")
        if not leaf:  # "BAAA" -> "BA/BAAA"
            prefix, leaf = code[:2], code
        downloads.append(
            Download(f"{ZINC_2D_BASE}/{prefix}/{leaf}.smi", library / "tranches" / f"{leaf}.smi")
        )
    return downloads


MANUAL = {
    "dekois2": (
        DEKOIS_LANDING,
        "Download the per-target sets (<TARGET>_*decoyset.sdf plus the actives SDF) into "
        "{dest}. Tier-5 hold-out: use once, at the end of WP5 (S5.8).",
    ),
    "muv": (
        MUV_LANDING,
        "Download cmp_list_MUV_<aid>_actives.dat and cmp_list_MUV_<aid>_decoys.dat into {dest}. "
        "The 'decoys' are assayed non-binders and the adapter labels them measured_inactive.",
    ),
    "local": (
        "https://github.com/oxpig/DeepCoy",
        "DeepCoy's published decoy sets go in {dest}; name files *_actives / *_decoys / "
        "*_inactives so LocalSDFAdapter can infer the role, or pass role_map.",
    ),
}


# ------------------------------------------------------------------------------ fetching


def fetch(download: Download, timeout: float) -> tuple[bool, str]:
    """Stream one URL to disk atomically. Returns ``(ok, message)``; never raises for HTTP."""
    download.dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = download.dest.with_suffix(download.dest.suffix + ".part")
    request = urllib.request.Request(download.url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"
    tmp.replace(download.dest)
    return True, digest.hexdigest()


def record_provenance(library_dir: Path, entries: list[dict]) -> Path:
    """``FETCH.json`` beside the data: URL, sha256 and time per file.

    A cache is only usable as evidence if a run can say where its molecules came from, and the
    files themselves carry no provenance."""
    path = library_dir / "FETCH.json"
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text()).get("files", [])
        except (OSError, json.JSONDecodeError):
            existing = []
    by_dest = {entry["dest"]: entry for entry in existing}
    for entry in entries:
        by_dest[entry["dest"]] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"files": sorted(by_dest.values(), key=lambda e: e["dest"])}, indent=2)
    )
    return path


def build_zinc_sample(library_dir: Path, size: int, seed: int) -> Path | None:
    """Reservoir-sample the downloaded tranches into one seeded file.

    Reservoir rather than "read all, then sample": a ZINC tranche is millions of lines and the
    sample has to be drawable on a laptop. One pass, O(size) memory, and the seed is in the
    filename so two controls cannot be confused.
    """
    tranche_dir = library_dir / "tranches"
    files = sorted(tranche_dir.glob("*.smi"))
    if not files:
        return None
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for path in files:
        with path.open(errors="replace") as handle:
            for lineno, raw in enumerate(handle):
                line = raw.strip()
                if not line or (lineno == 0 and line.lower().startswith("smiles")):
                    continue
                seen += 1
                if len(reservoir) < size:
                    reservoir.append(line)
                else:
                    j = rng.randrange(seen)
                    if j < size:
                        reservoir[j] = line
    out = library_dir / f"zinc_random_seed{seed}.smi"
    out.write_text("smiles zinc_id\n" + "\n".join(reservoir) + "\n")
    print(f"  sampled {len(reservoir)} of {seen} molecules -> {out}")
    return out


# ------------------------------------------------------------------------------ reporting


def summarise(name: str, root: Path) -> None:
    """Read the cache back through the adapter — the only check that the layout is right."""
    adapter = get_adapter(name, root)
    if not adapter.available():
        print(f"  {name}: adapter reports not available at {root / name}")
        return
    targets = adapter.targets()
    print(f"  {name}: {len(targets)} target(s): {', '.join(targets[:8])}"
          f"{' ...' if len(targets) > 8 else ''}")
    counts: dict[str, int] = {}
    for record in adapter.records(limit=200_000):
        counts[record.role] = counts.get(record.role, 0) + 1
    for role in sorted(counts):
        print(f"    {role}: {counts[role]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--library", action="append", choices=[*sorted(ADAPTERS), "all"],
        help="repeatable; default: dude",
    )
    parser.add_argument("--target", action="append", help="DUD-E target, repeatable (e.g. pgh1)")
    parser.add_argument("--all-targets", action="store_true", help="DUD-E: the 102-target tarball")
    parser.add_argument("--with-conformers", action="store_true", help="DUD-E: also the SDF conformers")
    parser.add_argument("--tranche", action="append", help="ZINC 2D tranche, e.g. BA/BAAA")
    parser.add_argument("--sample-size", type=int, default=5000, help="ZINC control size")
    parser.add_argument("--seed", type=int, default=1234, help="ZINC sampling seed (recorded)")
    parser.add_argument("--root", type=Path, default=None, help="cache root [default: repo .cache/libraries]")
    parser.add_argument("--url", action="append", help="extra URL to fetch into the library dir")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--download", action="store_true", help="actually fetch; otherwise print the plan")
    args = parser.parse_args(argv)

    root = args.root or default_cache_root()
    libraries = args.library or ["dude"]
    if "all" in libraries:
        libraries = sorted(ADAPTERS)
    targets = args.target or []

    downloads: list[Download] = []
    for name in libraries:
        if name == "dude":
            if not targets and not args.all_targets:
                print("dude: no --target given; pass --target pgh1 (or --all-targets)")
            else:
                downloads += plan_dude(root, targets, args.with_conformers, args.all_targets)
        elif name == "zinc_random":
            if not args.tranche:
                print(f"zinc_random: pass --tranche AA/AAAA (browse {ZINC_2D_BASE}/)")
            else:
                downloads += plan_zinc(root, args.tranche)
        if name in MANUAL:
            url, instructions = MANUAL[name]
            print(f"{name}: manual download from {url}")
            print(f"  {instructions.format(dest=root / name)}")
    for url in args.url or []:
        downloads.append(Download(url, root / libraries[0] / Path(url).name, "from --url"))

    if not downloads:
        return 0
    print(f"\n{len(downloads)} file(s) planned under {root}:")
    for download in downloads:
        print(f"  {download.url}\n    -> {download.dest}"
              + (f"  ({download.note})" if download.note else ""))
    if not args.download:
        print("\nplan only; re-run with --download to fetch")
        return 0

    failures = 0
    provenance: dict[Path, list[dict]] = {}
    for download in downloads:
        ok, message = fetch(download, args.timeout)
        print(f"  {'OK  ' if ok else 'FAIL'} {download.url} {message[:16] if ok else message}")
        if not ok:
            failures += 1
            continue
        library_dir = download.dest.parent
        while library_dir.parent != root and library_dir != root:
            library_dir = library_dir.parent
        provenance.setdefault(library_dir, []).append(
            {
                "url": download.url,
                "dest": str(download.dest),
                "sha256": message,
                "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    for library_dir, entries in provenance.items():
        record_provenance(library_dir, entries)

    if "zinc_random" in libraries and args.tranche:
        build_zinc_sample(root / "zinc_random", args.sample_size, args.seed)

    print("\ncache contents:")
    for name in libraries:
        summarise(name, root)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
