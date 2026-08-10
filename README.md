# EGFR Atomistic Frustration Pipeline

Independent reimplementation of the atomistic frustration method from Chen et al.
(2020), extended from the paper's 4 EGFR-inhibitor complexes to 61 structures
(51 unique ligands, including 15 covalent inhibitors) to test a
frustration-affinity correlation that the paper never computed.

## Reference

Mingchen Chen et al., "Surveying biomolecular frustration at atomic resolution,"
*Nat. Commun.* 11, 5944 (2020).
DOI: [10.1038/s41467-020-19560-9](https://doi.org/10.1038/s41467-020-19560-9)

## Current Repository State

- Stages 1-5 have already been run for the current 61-structure set.
- `data/processed/` contains 61 cleaned complexes.
- `data/ligands/params/` contains params and conformer files for the 51 unique ligands.
  Note that the directory also holds stale `{PDB}_{LIG}.params` files left over from the
  superseded 25-structure pipeline; only the 51 listed in
  `results/metadata/ligand_parameterization_status.csv` are current.
- Stage 6 has been run for **19 of the 61 structures**, all at `--n_decoys 50`:
  1XKK, 2ITO, 2RGP, 3POZ, 3W2O, 3W2Q, 3W2S, 3W32, 3W33, 4JQ7, 4LI5, 5C8M, 5CAO,
  5CAP, 5CAV, 5EM8, 5GMP, 5GTY, 5UGB. Each has a
  `results/{PDB}_{LIG}_frustration.parquet` and a matching `checkpoints/*_ckpt.pkl`.
  `results/egfr_frustration_summary.csv` and `results/egfr_correlation.png` cover all 19.
  The 1LYZ validation has also been run (`results/validation_lysozyme.png`).
  These outputs are DVC-tracked, not committed — run `dvc pull` to obtain them.
- Keep new runs at `--n_decoys 50` unless recomputing all 19. A different decoy count
  changes the decoy-energy standard deviation and therefore the Z-score scale, so
  structures run at different `--n_decoys` are not comparable in one correlation.
- 15 structures are flagged as covalent in `results/preparation_summary.csv`, but Stage 6
  does not currently model covalent bonding and existing ligand params do not contain
  ligand-specific `CONNECT` records.

### Interpreting the correlation

The current 19-structure result is **Pearson r = -0.297, p = 0.217** — not significant.
More importantly, the plotted metric `n_minimally_frustrated` is confounded by pocket
size and should not be read as a frustration result on its own. The ligand-pocket
contact count varies from 476 to 733 across these structures, and every count metric
inherits that variation. Ranked by Spearman correlation against `log10(affinity_pM)`:

| metric | Spearman rho | p |
|---|---|---|
| `n_neutral` | -0.541 | 0.017 |
| `n_contacts_total` | -0.521 | 0.022 |
| `n_highly_frustrated` | -0.455 | 0.050 |
| `frac_minimally` | +0.351 | 0.141 |
| `n_minimally_frustrated` | -0.347 | 0.145 |
| `frac_highly` | -0.172 | 0.482 |

The two metrics reaching p < 0.05 are pocket size and the count of *neutral* contacts —
the ones that are not frustrated in either direction. A frustration-driven effect would
not rank that way. Dividing pocket size out reverses the sign: `frac_minimally` is
+0.351 overall and +0.745 (p = 0.008) on the 11 WT non-covalent structures, meaning a
*less* frustrated pocket binds *more weakly*, opposite to the paper's hypothesis.
The clearest single case is ligand 03P, which appears in both 3POZ (23 pM) and 3W2O
(8400 pM): despite a 365-fold affinity difference their `frac_minimally` values are
0.5007 and 0.4964.

Treat this as a hypothesis to test rather than a conclusion — n is small, the set mixes
WT with T790M/L858R and covalent complexes, and several metrics were tested without
correction. Prefer `frac_minimally`, or regress the raw count on affinity with
`n_contacts_total` as a covariate. Both `_plot_correlation()` and the notebook still
plot the raw count.

## Environment Setup

Run everything from the repository root. `config.yaml` uses paths relative to the
current working directory, and `src/run_pipeline.py` imports `frustration` as a
top-level module.

This repository now includes a reproducible conda spec:

```bash
conda env create -f environment.yml
conda activate frustrato
```

If the environment already exists:

```bash
conda env update -n frustrato -f environment.yml --prune
conda activate frustrato
```

The environment file includes the notebook and preparation dependencies used in
this repository, including `biopython`, `rdkit`, and the Python `openbabel`
bindings via `openbabel-wheel`. PyRosetta still needs a separate install.

Resolve the newest release wheel from the RosettaCommons west mirror and install
it directly:

```bash
WHEEL=$(python - <<'PY'
import re, urllib.request
base = ("https://west.rosettacommons.org/pyrosetta/release/release/"
        "PyRosetta4.Release.python310.ubuntu.wheel/")
html = urllib.request.urlopen(base, timeout=60).read().decode()
wheels = re.findall(r'href="(pyrosetta-\d+\.\d+[^"]*\.whl)"', html)
key = lambda w: tuple(int(n) for n in re.match(r"pyrosetta-(\d+)\.(\d+)", w).groups())
print(base + max(wheels, key=key))
PY
)
python -m pip install "$WHEEL"      # ~1.7 GB download, several minutes
python -c "import pyrosetta; pyrosetta.init('-mute all'); print('PyRosetta OK')"
```

PyRosetta is free for academic use but requires a licence — see
https://els2.comotion.uw.edu/product/pyrosetta for terms.

### Why not `pyrosetta_installer`

`environment.yml` installs the `pyrosetta-installer` helper, but
`pyrosetta_installer.install_pyrosetta()` does not currently work. It resolves
the download through `latest.html`, which points at
`pyrosetta-0-cp310-cp310-linux_x86_64.whl`; that placeholder returns HTTP 404 on
both mirrors, so the call aborts without installing anything. The snippet above
reads the same directory listing and selects the newest versioned wheel instead.

Two further caveats:

- The wheel filename is URL-encoded (`%2B` for `+`). Keep it encoded when passing
  the URL to `pip`.
- Only the **west** mirror
  (`west.rosettacommons.org`) is reliably reachable. The east mirror
  (`graylab.jhu.edu`, the installer's `mirror=1`) can fail TLS verification with
  `CERTIFICATE_VERIFY_FAILED` behind some networks, so it is not a dependable
  fallback.

Adjust `python310` and `ubuntu` in the URL if the environment's Python version or
platform differs; the sibling directories under `.../release/release/` show what
is available.

## Data Sync Across Codespaces and a Workstation

For cross-machine use, the most practical approach is to keep code in Git and
sync large derived data with DVC backed by Google Cloud Storage.

### What to sync with DVC

Three whole directories are tracked, one `.dvc` file each at the repo root:

| directory | `.dvc` file | contents |
|---|---|---|
| `data/` | `data.dvc` | `data/processed/` and `data/ligands/` |
| `results/` | `results.dvc` | parquets, summary CSV, plots, `results/metadata/` |
| `checkpoints/` | `checkpoints.dvc` | `*_ckpt.pkl` decoy-energy checkpoints |

`data/raw_pdb/` and `data/raw_cif/` are excluded via `.gitignore`/`.dvcignore` — they
are re-downloadable from RCSB by rerunning Stage 2.

Because these are directory-level outputs, changing any file inside one means the whole
directory has to be re-added before it can be pushed. See **Daily workflow** below —
this is the single most common way to lose work here.

### Install DVC with Google Cloud Storage support

On both your workstation and any Codespace where you plan to work:

```bash
pip install "dvc[gs]"
```

### Install the Google Cloud CLI

You need the `gcloud` CLI if you want to authenticate with user credentials via
Application Default Credentials.

For Linux (including GitHub Codespaces), a practical user-local install is:

```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
```

Then open a new shell or source the generated shell config, and verify:

```bash
gcloud version
```

If you prefer, you can instead install `gcloud` using your system package
manager or another official method documented by Google Cloud.

### One-time repository setup

**This has already been done in this repository** — DVC is initialized and the remote
`storage` points at `gs://egfr-analysis-pipeline-withrosetta/` (see `.dvc/config`,
which also sets `core.autostage = true` so `dvc add` stages the `.dvc` file for you).
The steps below are recorded for reference, or for setting up a fresh clone against a
different bucket.

Create a Google Cloud Storage bucket, then initialize DVC and add the remote:

```bash
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

dvc remote add -d storage gs://<your-bucket>/
```

### Configure Google Cloud authentication

Four things have to line up for `dvc push`/`dvc pull` to work: the Python GCS
libraries, the gcloud CLI, credentials, and a **quota project**. The last one is the
step that is easy to miss.

```bash
# 1. Python side (already satisfied in the `frustrato` env)
pip install "dvc[gs]"          # provides dvc-gs + gcsfs + google-cloud-storage

# 2. Credentials — opens a browser, writes
#    ~/.config/gcloud/application_default_credentials.json
gcloud init
gcloud auth application-default login

# 3. Quota project — REQUIRED, and separate from `gcloud config set project`
gcloud auth application-default set-quota-project <your-gcp-project-id>

# 4. Verify end to end
dvc status --cloud             # should print "Cache and remote 'storage' are in sync"
```

Step 3 is the one people skip. `gcloud config set project` (or `gcloud init`) sets the
project for the **CLI**; it does not set the project that **client libraries** bill API
calls to. Without step 3, `google.auth.default()` emits:

```
UserWarning: Your application has authenticated using end user credentials from
Google Cloud SDK without a quota project. You might receive a "quota exceeded" or
"API not enabled" error.
```

Google Cloud Storage happens to tolerate this and DVC keeps working, so the warning is
easy to dismiss — but the state is fragile, and other Google APIs fail outright. Step 3
writes `quota_project_id` into the ADC file so the setting persists across shells.

`GOOGLE_CLOUD_QUOTA_PROJECT` is an alternative way to supply the same value, but it only
lives as long as the shell that exports it. If you use it, put it in your shell profile —
otherwise the setup will work in one terminal and warn in the next.

#### Current state of this machine

Verified 2026-08-10:

| item | value |
|---|---|
| gcloud SDK | 579.0.0 (`/snap/bin/gcloud`) |
| active account | `onursercin@gmail.com` |
| `gcloud config` project | `githubrepodvcs` |
| ADC file | present, type `authorized_user` |
| **ADC `quota_project_id`** | **not set** — step 3 has not been run |
| `GOOGLE_CLOUD_QUOTA_PROJECT` | set to `personal-dashboard-497418` in the current shell only, not in any profile |
| bucket | `gs://egfr-analysis-pipeline-withrosetta/`, US, uniform bucket-level access on |
| bucket access | read and write confirmed (legacy owner IAM roles) |
| Python deps | dvc 3.67.1, dvc-gs 3.1.0, gcsfs 2026.7.0, google-cloud-storage 3.13.1 |

Push and pull work today, but only because an unpersisted environment variable is
supplying a quota project belonging to an unrelated GCP project. In a fresh terminal
that variable is absent and the ADC file has no `quota_project_id`, so operations fall
back to the warned state above. To fix it permanently:

```bash
gcloud auth application-default set-quota-project githubrepodvcs
```

Optionally, pin the project for DVC itself so it does not depend on discovery:

```bash
dvc remote modify storage projectname githubrepodvcs
```

#### Non-interactive setups (Codespaces, CI)

Use a service account key instead of user credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-service-account.json
```

or, scoped to this repository only:

```bash
dvc remote modify --local storage credentialpath /path/to/gcs-service-account.json
```

`--local` writes to `.dvc/config.local`, which is gitignored. Service accounts carry
their own project, so no quota-project step is needed. Grant the account
`roles/storage.objectAdmin` on the bucket. Do not commit secrets or service account
keys.

### Start tracking this repository's derived data

Also already done. What actually exists is three whole-directory `.dvc` files at the
repo root — `data.dvc`, `results.dvc`, `checkpoints.dvc` — not the finer-grained
`data/processed` and `data/ligands/params` targets an earlier draft of this section
described:

```bash
dvc add data
dvc add results
dvc add checkpoints

git add data.dvc results.dvc checkpoints.dvc .gitignore
git commit -m "Track analysis data with DVC"
```

`data/raw_pdb/` and `data/raw_cif/` are excluded by `.dvcignore`/`.gitignore` — they are
re-downloadable from RCSB by re-running Stage 2.

### Push the initial dataset

```bash
dvc push
git push
```

### Daily workflow

Before starting work on either machine:

```bash
git pull
dvc pull
```

After generating new outputs or checkpoints:

```bash
dvc add results          # re-hash the directory; do NOT skip this
dvc add checkpoints      # only the directories that actually changed
dvc push
git commit -m "..."      # results.dvc / checkpoints.dvc are auto-staged
git push
```

**`dvc push` alone is not enough, and it fails silently.** `results/` and
`checkpoints/` are tracked as whole directories, so `results.dvc` records a single hash
for the entire directory. Writing new parquet files does not update that hash and does
not put the new files into the DVC cache — `dvc add` is the step that does both. Until
you run it, `dvc push` sees nothing new to upload and exits 0, which looks exactly like
a successful backup.

The symptom is a `dvc status` that reports the directory as modified while
`dvc status --cloud` claims everything is in sync:

```console
$ dvc status
results.dvc:
        changed outs:
                modified:           results

$ dvc status --cloud
Cache and remote 'storage' are in sync.     # <- misleading: nothing was ever added
```

If you see that pair, run `dvc add` on the modified directories and push again. Check
`dvc status` before every push; it names exactly which directories need re-adding.
`core.autostage = true` is set in `.dvc/config`, so `dvc add` stages the rewritten
`.dvc` files in git for you — you still have to commit them.

That explicit push/pull workflow is more reliable than trying to run a
background two-way sync daemon inside Codespaces, which may stop after idle
timeout or Codespace shutdown.

### What works without PyRosetta

- `scripts/01_collect_metadata.py`
- `scripts/02_download_structures.py`
- `scripts/03_identify_egfr_chain_and_ligand.py`
- `scripts/04_prepare_complex.py`
- `scripts/05_prepare_ligand.py --skip-pyrosetta-test`
- The non-PyRosetta tests in `src/test_frustration.py`

### What still requires PyRosetta

- `scripts/05_prepare_ligand.py` without `--skip-pyrosetta-test`
- `python src/run_pipeline.py --mode validate ...`
- `python src/run_pipeline.py --mode single ...`
- `python src/run_pipeline.py --mode all ...`
- The PyRosetta-marked tests in `src/test_frustration.py`

## Commands

There is no dedicated build step and no lint target in this repository. The main
entry points are the stage scripts and the pytest suite.

### Stages 1-5 - structure and ligand preparation

```bash
python scripts/01_collect_metadata.py
python scripts/02_download_structures.py
python scripts/03_identify_egfr_chain_and_ligand.py
python scripts/04_prepare_complex.py
python scripts/05_prepare_ligand.py
```

Single-item debug modes:

```bash
python scripts/01_collect_metadata.py --pdb-id 5GMP
python scripts/02_download_structures.py --pdb-id 5GMP
python scripts/03_identify_egfr_chain_and_ligand.py --pdb-id 5GMP
python scripts/04_prepare_complex.py --pdb-id 5GMP
python scripts/05_prepare_ligand.py --ligand-id 634 --skip-pyrosetta-test
```

### Stage 6 - frustration analysis

Always launch the driver as a script, **not** with `python -m src.run_pipeline`.

```bash
python src/run_pipeline.py --mode validate --n_decoys 50
python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50 --n-jobs 2
python src/run_pipeline.py --mode all --n_decoys 50
python src/run_pipeline.py --mode all --n_decoys 200
# Limit an all-mode run to selected structures and explicitly use two workers
python src/run_pipeline.py --mode all --pdb-ids 1XKK,5GMP --n_decoys 50 --n-jobs 2
# Save the native pose plus every generated decoy PDB for later inspection
python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50 \
  --save-structures-dir runs/method-a/structures
# Keep a comparative run and its resume checkpoints outside the default folders
python src/run_pipeline.py --mode all --n_decoys 200 \
  --results-dir runs/method-a/results \
  --checkpoints-dir runs/method-a/checkpoints
```

Runtime is substantial: one decoy on a ~1700-contact structure costs about
**6.7 minutes of single-core time** (measured on a 12-core machine with PyRosetta
2026.30; an earlier estimate of 4-5 minutes proved optimistic). At `--n_decoys 50`
that is roughly 32 minutes per structure using 10 decoy workers, and a full
`--mode all --n_decoys 200` run is multi-day. Prototype with `--mode validate`,
`--mode single`, or `--mode all --n_decoys 50` first.

`--mode all` uses spawned worker processes, initializing PyRosetta separately in
each process. It defaults to all available logical CPUs. Use `--n-jobs N` (or
`--n_jobs N`) to override that count.
Use `--pdb-ids ID1,ID2` to restrict an all-mode run without changing the candidate
tables; this is useful for a parallel smoke test.

**To run a batch of structures, loop `--mode single` rather than using `--mode all`.**
`run_all_egfr()` forces `n_jobs_decoys=1` inside each worker, so `--mode all --n-jobs 6`
occupies only 6 cores for 6 structures. Running them one at a time with
`--mode single --n-jobs 10` uses all 10 and is meaningfully faster — six structures took
3 h 33 m that way, against an estimated 3 h 45 m for `--mode all` on the same machine.
Results are unaffected: decoy *i* is seeded `seed + i` in both the sequential and the
parallel code paths, so output is identical regardless of worker count. Finish the batch
with a single `--mode all --pdb-ids <every completed ID>` pass to rebuild
`egfr_frustration_summary.csv` and the correlation plot; already-completed structures
short-circuit on their existing parquet in seconds.

Note that `--mode all` writes the summary from only the structures in that invocation,
so a `--pdb-ids` subset run replaces the full summary with a smaller one. The
per-structure parquets are the durable record.

Avoid running other PyRosetta work while a Stage 6 job is live — a concurrent
pose-loading script added roughly 10 minutes to one structure through CPU contention.

For `--mode single`, `--n-jobs N` parallelizes decoy generation for that one
structure. For `--mode all`, it parallelizes structures instead; each structure
then generates decoys sequentially to avoid nested worker pools.

Stage 6 writes per-structure parquet files, the all-mode summary, and plots under
`--results-dir` (default: `paths.results` in `config.yaml`). Use
`--checkpoints-dir` (default: `paths.checkpoints`) with the same custom run root
to keep decoy-resume state separate. The preparation scripts already expose their
own output arguments, such as `--output-dir` or `--output-csv`.
Use `--save-structures-dir` to write `native.pdb` plus `decoys/decoy_XXXX.pdb`
under one subdirectory per analyzed structure. If a parquet result already
exists, the native pose can still be dumped, but decoy PDBs require deleting the
existing parquet and checkpoint so the decoys are regenerated.

### Tests

```bash
python -m pytest src/test_frustration.py -v
python -m pytest src/test_frustration.py::test_decoy_backbone_unchanged -v
```

If PyRosetta is not installed, the PyRosetta-marked tests are skipped automatically.

## High-Level Architecture

The codebase has two halves that communicate only through files on disk.

### Half 1 - `scripts/01` through `scripts/05`

This is the preparation pipeline. Each stage reads the previous stage's CSV/YAML
output and writes the next one.

| Stage | Main output |
|-------|-------------|
| 01 | `results/metadata/egfr_ligand_inventory.csv` |
| 02 | `data/raw_pdb/`, `data/raw_cif/`, `results/metadata/download_manifest.csv` |
| 03 | `config/ligand_overrides.yaml`, `results/metadata/chain_ligand_selection.csv` |
| 04 | `data/processed/{PDB}_clean.pdb`, `results/preparation_summary.csv` |
| 05 | `data/ligands/params/{LIG}.params`, `results/metadata/ligand_parameterization_status.csv` |

### Half 2 - `src/frustration.py` and `src/run_pipeline.py`

- `src/frustration.py` is the engine: contact generation, Eq. 2 many-body energies,
  decoy generation, Eq. 1 Z-scores, and class assignment.
- `src/run_pipeline.py` is the driver: pose loading, candidate selection, checkpointing,
  validation, spawned parallel per-structure runs, and the final affinity correlation plot.

The contract between the two halves is `load_candidates()` in `src/run_pipeline.py`.
It inner-joins:

- `results/metadata/egfr_ligand_inventory.csv`
- `results/preparation_summary.csv`
- `results/metadata/ligand_parameterization_status.csv`

Only rows where both the prepared complex and the ligand params have `status == "OK"`
are kept. To add or remove structures, edit `config/pdb_reference_table.csv` and rerun
the preparation stages; there is no hardcoded structure list in the analysis code.

## Key Conventions and Gotchas

### `ligand_comp_id` versus `rosetta_ligand_comp_id`

Some CCD codes collide with Rosetta residue names (for example `634` becomes `Z34`).
Stage 4 may rewrite the ligand `resName` in the cleaned PDB, but reporting and
metadata still use the real CCD code. `.params` lookup and pose loading use the
Rosetta-safe identifier.

### `config/ligand_overrides.yaml` is generated output

This file is produced by Stage 3 and is the source of truth for EGFR chain choice
and ligand-copy selection. `config.yaml`'s `chain_selection.default` is only a fallback.

### Frustration thresholds are hardcoded

`config.yaml` contains a `frustration.thresholds` block, but the live thresholds are
currently hardcoded in `src/frustration.py` and duplicated in the plotting logic in
`src/run_pipeline.py`.

### Decoy backbones must be restored explicitly

`MutateResidue` perturbs the carbonyl oxygen even when the backbone is notionally
fixed. `generate_decoy()` therefore hard-restores `N`, `CA`, `C`, and `O` coordinates
from the native pose after repack/minimization, and `test_decoy_backbone_unchanged`
exists specifically to guard that invariant.

### Stage 5 can be run without PyRosetta validation

`scripts/05_prepare_ligand.py` supports `--skip-pyrosetta-test`. That is the intended
path in environments where the ligand params should be regenerated but PyRosetta is
not installed yet.

### Covalent handling is incomplete

Stage 4 records covalent link information from mmCIF `_struct_conn`, but Stage 6 does
not currently apply covalent chemistry and the checked-in params files do not include
ligand-specific `CONNECT` records. The 15 covalent complexes are presently analyzed
as if they were non-covalent.

### Resume semantics

`run_frustration_survey()` checkpoints decoy energies under `checkpoints/`, but
`run_single_structure()` also short-circuits if
`results/{PDB}_{LIG}_frustration.parquet` already exists. To rerun a structure with
more decoys, delete both the result parquet and the matching checkpoint.

Checkpointing happens every `checkpoint.save_every_n_decoys` decoys, which defaults to
50 — the same as the usual `--n_decoys 50`. A standard run therefore checkpoints only
once, at the very end, and an interruption at decoy 49 restarts from zero. Lower that
value in `config.yaml` before any long run, where resumability actually matters.

## Methods Summary

### Frustration index (Eq. 1)

```text
F_ij = (mean(E_ij_decoy) - E_ij_native) / std(E_ij_decoy)
```

### Many-body correction (Eq. 2)

```text
E_ij = e_ij
     + 0.5 * Σ_{k ∈ contacts(i), k≠j} e_ik
     + 0.5 * Σ_{l ∈ contacts(j), l≠i} e_jl
```

### Decoy generation

1. Estimate the amino-acid composition from the native structure.
2. Randomize the full protein sequence on the native backbone.
3. Repack side chains.
4. Run chi-only minimization.
5. Restore backbone atom coordinates from the native pose.

## Disclaimer

This is an independent reimplementation. The original paper's code was not
published, so exact numerical agreement is not expected; the target is qualitatively
correct behavior.
