# EGFR Atomistic Frustration Pipeline

Independent reimplementation of the atomistic frustration method from Chen et al.
(2020), extended from the paper's 4 EGFR-inhibitor complexes to 61 structures
(51 unique ligands, including 15 covalent inhibitors) to test a
frustration-affinity correlation that the paper never computed.

**The repository holds two implementations.**

| | |
|---|---|
| `src/` + `scripts/` | The original EGFR-specific pipeline. All 61 structures have been run. Still works, still the source of every stored result. |
| `atomfrust/` | A target-agnostic rewrite — any protein-ligand, protein-only or protein-protein complex. Built to `plans/frustratometer-ng-plan.md`. |

The rewrite exists because a set of diagnostics run against the original pipeline
found that it computes a well-defined quantity which is **not** the published one.
That finding, and the evidence for it, is summarised under
[What the diagnostics found](#what-the-diagnostics-found) — read it before
interpreting any number in `results/`.

## Quick Start

On a machine with nothing installed:

```bash
git clone <repo-url>
cd egfr_analysis_pipeline_withRosetta
./setup.sh
```

`setup.sh` bootstraps everything in seven idempotent stages: Miniforge, the
`frustrato` conda environment, PyRosetta, the Google Cloud CLI, credentials, the
DVC-tracked data, and a verification pass. Re-run it at any time — each stage detects
what is already done, so an interrupted setup resumes rather than restarting.

There is **one** step it cannot automate: `gcloud auth application-default login` opens
a browser and waits for you to sign in. Everything else is unattended. For genuinely
unattended runs (CI, or a collaborator with a shared key) use a service-account key:

```bash
./setup.sh --credentials /path/to/key.json --yes
```

Useful variants:

```bash
./setup.sh --skip-auth --no-data          # code only, no cloud access needed
./setup.sh --quota-project my-gcp-project # avoids the ADC quota-project prompt
./setup.sh --skip-pyrosetta               # skip the 1.7 GB download for now
./setup.sh --help                         # all flags
```

Expect 25-40 minutes on a fresh machine, dominated by the PyRosetta download.
The sections below document what each stage does, and how to perform the steps by
hand if you would rather not use the script.

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
- Stage 6 has been run for **all 61 structures**, at `--n_decoys 50` (completed
  2026-08-11). Each has a `results/{PDB}_{LIG}_frustration.parquet` and a matching
  `checkpoints/*_ckpt.pkl`; `results/egfr_frustration_summary.csv` holds all 61 rows.
  The 1LYZ validation has also been run (`results/validation_lysozyme.png`).
  These outputs are DVC-tracked, not committed — run `dvc pull` to obtain them.
- Keep new runs at `--n_decoys 50` unless recomputing all 19. A different decoy count
  changes the decoy-energy standard deviation and therefore the Z-score scale, so
  structures run at different `--n_decoys` are not comparable in one correlation.
- 15 structures are flagged as covalent in `results/preparation_summary.csv`, but Stage 6
  does not currently model covalent bonding and existing ligand params do not contain
  ligand-specific `CONNECT` records.

### Interpreting the correlation

**At the full 61 structures, nothing is significant.** Against `log10(affinity_pM)`:

| metric | Pearson r | p | Spearman rho | p |
|---|---:|---:|---:|---:|
| `n_minimally_frustrated` | +0.038 | 0.77 | +0.023 | 0.86 |
| `n_neutral` | -0.121 | 0.35 | -0.126 | 0.33 |
| `n_contacts_total` | -0.068 | 0.60 | -0.086 | 0.51 |
| `frac_minimally` | +0.131 | 0.31 | +0.171 | 0.19 |

An earlier version of this file reported a 19-structure result in which `n_neutral`
(rho = -0.541, p = 0.017) and `n_contacts_total` (rho = -0.521, p = 0.022) reached
p < 0.05. **Those are withdrawn.** They were found while examining four descriptors at
an n where the 5% critical value for a *single* pre-specified test is already r = 0.456,
and they do not survive the full set. The n = 19 subset was also a deliberately chosen
WT non-covalent series, so part of the shift is composition rather than sample size.

The pocket-size confound is still structurally present — `n_contacts_total` ranges 476
to 733, and every count metric inherits that variation — it simply no longer produces a
significant correlation to be confounded by.

**For contrast, the paper's own counts do carry signal on these same 61 structures.**
`config/pdb_reference_table.csv` records Chen et al.'s per-structure minimally-frustrated
counts; against log2(affinity_pM) they give **r = -0.388, p = 0.002**. The published
quantity has a real relationship here while this pipeline's has none — which is what the
diagnostics below explain.

## What the diagnostics found

Four checks were run against the stored results before any new code was written. Each is
reproducible from `analysis/`, and each is recorded in `project_status/`.

**1. The many-body equation collapses.** With `B_i` the sum of `e_ik` over *all* of i's
contact partners, the published Eq. 2 reduces algebraically:

```text
E_ij = e_ij + 0.5*(B_i - e_ij) + 0.5*(B_j - e_ij) = 0.5*(B_i + B_j)
```

`e_ij` cancels exactly, so the "per-contact" energy carries no pair-specific information —
it is the mean of two per-residue totals. Fitting `E_native` as `a_i + a_j` gives
**R^2 = 1.000000 on 38/38 structures** (max residual 8.6e-14 against a 3.59 kcal/mol
spread). Only ~4% of the per-contact index variance is pair-specific.

The exclusions are in the published paper, verified against the source — so this is a
property of the published equation, not a transcription error. It also explains the
pocket-size confound: if a contact's value is fixed by its two residues, contact counts
are graph-degree statistics.

**2. The counts are of the wrong object.** The paper's counts are **ligand-residue**
contacts — "a strong inhibitor XTF-262 (PDB 5GMP) forms more than ten minimally
frustrating interactions with its pocket", and the reference table gives 5GMP = 16,
5EM8 = 4, range 4-23. This pipeline counts protein-protein pairs in a 10 A pocket shell:
266-407, uncorrelated with the paper at r = 0.163, p = 0.51. A sweep of 189
selector/threshold configurations closed none of the gap.

**3. The index is ligand-blind.** Delete the ligand from 5GMP and re-run at the same
seed: pocket-contact Pearson r = **0.9904**, and `E_native` is **bit-identical**
(max |delta| = 0.000e+00). This is structural — `get_protein_contacts` excludes
non-protein residues, so the ligand is in no partner list and cannot reach `E_native`.
Its entire influence runs through side-chain repacking in the decoys.

**4. Four protocol deviations from the paper**, each quoted in
`project_status/2026-08-11_0130_a4-paper-check-ligand-is-a-node.md`:

| | paper | this pipeline |
|---|---|---|
| decoy sequence | *"randomly shuffle the protein sequence"* (a permutation) | i.i.d. draw from the composition |
| relaxation | *"a short Monte-Carlo relaxation"* | one chi-only MinMover pass |
| native reference | *"obtained in a similar fashion by omitting the shuffling step"* — the native is repacked too | crystal pose scored as deposited |
| sequence separation | no criterion stated | `\|i-j\| >= 4` applied |

The native-reference asymmetry alone is large: repacking the native collapses
`frac_highly` from 0.068 to 0.010 on 5GMP and more than doubles `mean_F`.

**None of this makes the stored results wrong** — they are a faithful, reproducible
computation. They are a computation of something other than what the paper reports.

## The `atomfrust` package

The target-agnostic rewrite. Any protein-ligand, protein-only or protein-protein complex;
no EGFR-specific anything.

```bash
pip install -e . --no-deps     # --no-deps is deliberate; see below
atomfrust --help
```

Eleven subcommands:

```bash
atomfrust prepare --pdb my_complex.pdb --ligand B:501 -o prepared/
atomfrust generate-decoys --spec S.yaml --run-dir R --n-decoys 250   # no index computed
atomfrust analyze --run-dir R --shell-A 5.0 --index rank_percentile  # no PyRosetta at all
atomfrust run | validate | converge | strata | report | calibrate | verify
atomfrust metrics-selftest
```

**Two design choices carry most of the value.**

*Direct pair energies are stored; `E_ij` never is.* The many-body formula, contact
definition, shell, index function and `exclude_fa_rep` are therefore all chosen at
analysis time, against a finished run, with no Rosetta call. Re-analysing under different
settings is free; only a change to how decoys were *generated* requires regeneration, and
the run directory says which is which and exits 3 rather than guessing.

*Decoy i is seeded `base_seed + i`.* So results do not depend on worker count or sharding,
and the first N decoys of a 1000-decoy run **are** the ensemble an N-decoy run would have
produced — making the convergence sweep one run rather than eight.

`--no-deps` is deliberate: every runtime dependency is already satisfied by
`environment.yml`, and letting pip resolve them risks it touching the conda environment's
PyRosetta — a 1.7 GB reinstall. PyRosetta is absent from `dependencies` for the same
reason; naming it sends pip to PyPI, where the placeholder wheel 404s.

### What is established, and what is not

The package has ~844 tests (`pytest -m unit` is PyRosetta-free and runs in ~45 s;
`-m integration` needs PyRosetta and `dvc pull`). Test count is not evidence of
scientific validity, so the two are separated here.

**Established** — the new engine reproduces the old one at four independent levels:
contacts (exactly 1772 for 5GMP), native energies (r = 1.0000000000), whole decoys
regenerated against a stored checkpoint through a stochastic packing step
(max |delta| 1.2e-05), and class counts (**exact for all 61 structures**). The ligand is
now a node, and 5GMP has 27 ligand-incident contacts at a 6 A shell where the prototype
had none. Mutating a pocket residue shifts pocket energies by 21.41 REU while the same
substitution 31 A away shifts exactly 0.000.

**Not established:**

- **The chemotype axis — the intended novel contribution — is untested.** Its
  positive-control gate currently *fails*: the native ligand does not rank high within its
  own decoy ensemble (AUROC 0.333). The gate correctly refuses to emit a cross-axis
  redundancy number. Three hand-written fixture molecules is not evidence either way; it
  needs a real property-matched library.
- **The protein-protein interface gate (S0.2) is unmeasured** — no multi-chain structure
  exists under `data/`. The case is implemented and skips; 1BRS, 1AY7 and 1JTG are pinned.
- **Reference-count reproduction is smoke-scale only.** Ligand-incident counts are now the
  right kind of object and the range overlaps the published 4-23, but they run
  systematically low and move with decoy count. A full run is ~82 core-hours.
- **Pocket-restricted repacking misses its own bar** (rho = 0.82/0.86 against >= 0.95), and
  buys no measurable speed-up — 1.08-1.14x, consistent with a direct measurement showing
  packing is not the bottleneck.
- **Covalent complexes fail the pose validity gate by construction**: a 1.81 A S-C bond
  reads as a steric clash to a checker with no covalent concept.

`project_status/2026-08-11_1000_all-plan-steps-implemented.md` is the full version of this
list, with evidence for each line.

### Environment-dependent behaviour

`smina`, `gnina`, `posebusters` and `dimorphite_dl` are **not installed** here. The
affected components degrade explicitly rather than silently: the pose axis is a
*perturbation* ensemble rather than a docking ensemble, the validity gate runs an 8-check
built-in subset that reports `checker="builtin_subset"`, and protonation runs an RDKit
fallback reporting `method="rdkit_tautomer"`. None can be mistaken for the real thing, and
installing any of them requires no code change.

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

### Giving a collaborator access to the data bucket

Two ways. Prefer the first.

#### Option A — grant their Google account (recommended)

No shared secret, revocable per person, and their actions are attributable in the audit
log. Run these once, as the bucket owner:

```bash
# Read + write objects: exactly what `dvc pull` and `dvc push` need.
gcloud storage buckets add-iam-policy-binding gs://egfr-analysis-pipeline-withrosetta \
    --member="user:collaborator@example.com" \
    --role="roles/storage.objectUser"

# Lets them use your project as their ADC quota project, so they do not need
# a GCP project of their own. Skip if they already have one.
gcloud projects add-iam-policy-binding githubrepodvcs \
    --member="user:collaborator@example.com" \
    --role="roles/serviceusage.serviceUsageConsumer"
```

`roles/storage.objectUser` covers read, write, and delete of objects plus listing.
Use `roles/storage.objectViewer` instead if you want them read-only (`dvc pull` but not
`dvc push`). This bucket has uniform bucket-level access enabled, so IAM is the only
mechanism in play — object ACLs are disabled and cannot be used.

The collaborator then just runs:

```bash
./setup.sh --quota-project githubrepodvcs
```

To revoke, swap `add-iam-policy-binding` for `remove-iam-policy-binding`.

#### Option B — share a service-account key

Use this when the collaborator cannot be added to the GCP project, or for CI. The key
is a **long-lived credential in a file**: anyone holding it has the granted access,
there is no per-person attribution, and revoking it cuts off everyone at once.

```bash
# Create a service account and give it object read/write on the bucket only
gcloud iam service-accounts create egfr-dvc \
    --project githubrepodvcs \
    --display-name "EGFR pipeline DVC access"

gcloud storage buckets add-iam-policy-binding gs://egfr-analysis-pipeline-withrosetta \
    --member="serviceAccount:egfr-dvc@githubrepodvcs.iam.gserviceaccount.com" \
    --role="roles/storage.objectUser"

# Generate the key file to hand over
gcloud iam service-accounts keys create ~/egfr-dvc-key.json \
    --iam-account="egfr-dvc@githubrepodvcs.iam.gserviceaccount.com"
```

Send `egfr-dvc-key.json` over a private channel — never email, chat, or a git repo.
The collaborator runs:

```bash
./setup.sh --credentials /path/to/egfr-dvc-key.json
```

which writes the path into `.dvc/config.local` (gitignored, never shared) and requires
no gcloud install, no browser login, and no quota project — a service account carries
its own project.

Rotate or delete the key when the collaboration ends:

```bash
gcloud iam service-accounts keys list --iam-account="egfr-dvc@githubrepodvcs.iam.gserviceaccount.com"
gcloud iam service-accounts keys delete KEY_ID --iam-account="egfr-dvc@githubrepodvcs.iam.gserviceaccount.com"
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

There is no dedicated build step and no lint target in this repository. The main entry
points are the stage scripts, the `atomfrust` console script, and the pytest suites.

### `atomfrust` — the target-agnostic package

```bash
pip install -e . --no-deps

# a custom PDB, end to end, no metadata pipeline
atomfrust run --pdb my_complex.pdb --ligand B:501 --run-dir runs/mine --n-decoys 250

# generate decoys only — computes no frustration index at all
atomfrust generate-decoys --spec S.yaml --run-dir R --n-decoys 1000 --save-structures

# re-analyse a finished run under different settings; makes zero PyRosetta calls
atomfrust analyze --run-dir R --shell-A 5.0  --index rank_percentile -o analyses/s5
atomfrust analyze --run-dir R --manybody pair_retained                -o analyses/retained
atomfrust analyze --run-dir R --n-decoys 250                          -o analyses/n250

# region-focused decoys: mutate the pocket, repack a wider shell
atomfrust generate-decoys --spec S.yaml --run-dir R --scope contact_shell \
  --mutate-sel 'protein and within_ca(10.0, ligand)' \
  --repack-sel 'protein and within(12.0, ligand)'

# protein-only and protein-protein need no mode flag — the spec decides
atomfrust run --spec specs/1LYZ.yaml --run-dir runs/1lyz    # ligands: []
atomfrust run --spec specs/1BRS.yaml --run-dir runs/1brs    # chain_interface

atomfrust validate --list                 # the F1-F6 cases and what each proves
atomfrust validate --case F4              # holo vs apo: is the index ligand-blind?
atomfrust metrics-selftest                # the S0.3 cross-environment hash
atomfrust converge | strata | report | calibrate | verify
```

Tests: `pytest -m unit` is PyRosetta-free and runs in ~45 s; `pytest -m integration`
needs PyRosetta and `dvc pull`.

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

Two suites, run separately.

```bash
# atomfrust — three tiers, selected by marker
python -m pytest -m unit          # ~760 tests, no PyRosetta, no network, ~45 s
python -m pytest -m integration   # ~85 tests, needs PyRosetta and `dvc pull`
python -m pytest                  # everything, ~7 min

# the legacy pipeline
python -m pytest src/test_frustration.py -v
python -m pytest src/test_frustration.py::test_decoy_backbone_unchanged -v
```

Both must be run from the repository root. If PyRosetta is not installed, the
PyRosetta-marked tests skip automatically — so a green `-m unit` run on a machine without
it is not evidence the PyRosetta paths work.

## High-Level Architecture

Two implementations, described in turn.

### `atomfrust/` — the target-agnostic package

```text
spec.py        one hand-writable YAML replaces the three-CSV EGFR join
pose.py        the only module importing PyRosetta; nodes keyed by PDB position
graph.py       typed nodes, KD-tree neighbours, superset graph  (PyRosetta-free)
energy.py      REF2015 pair energies + the selectable many-body registry
regions.py     selector language: chain / resi / resn / within / layer / ...
execute.py     one flat work queue, spawn workers, disjoint shards
runstore.py    the run-directory contract — see below
decoys/        identity (axis A), pose (axis B), chemotype (axis D)
analyze/       z-scores, classification, descriptors, convergence, strata
metrics/       screening + inference, everything returns an Estimate
report/        confound-aware reporting that can refuse to print a headline
chem/ dock/    parametrisation, decoy libraries, pose backends, validity gate
validation/    the F1-F6 cases with stored expectations
cli/           one module per subcommand; main.py stays a table of names
```

The run directory is the interface between generation and analysis:

```text
runs/<run_id>/manifest.json, settings.resolved.yaml, env.json
  systems/<system_id>/
    inputs/    the exact bytes scored, plus params and digests
    graph/     nodes.parquet, pairs.parquet
    native/    native_energies.parquet, raw_energy.json
    decoys/    energies/part-<shard>-<seq>.parquet, index.parquet
    analyses/<analysis_id>/  contacts.parquet, summary.json
```

`decoys/energies/` stores **direct** pair energies — never `E_ij`. That single choice is
what lets the many-body formula, contact definition, shell, index function and
`exclude_fa_rep` all be re-chosen afterwards without touching Rosetta.

### The legacy pipeline — two halves communicating through files on disk

#### Half 1 - `scripts/01` through `scripts/05`

This is the preparation pipeline. Each stage reads the previous stage's CSV/YAML
output and writes the next one.

| Stage | Main output |
|-------|-------------|
| 01 | `results/metadata/egfr_ligand_inventory.csv` |
| 02 | `data/raw_pdb/`, `data/raw_cif/`, `results/metadata/download_manifest.csv` |
| 03 | `config/ligand_overrides.yaml`, `results/metadata/chain_ligand_selection.csv` |
| 04 | `data/processed/{PDB}_clean.pdb`, `results/preparation_summary.csv` |
| 05 | `data/ligands/params/{LIG}.params`, `results/metadata/ligand_parameterization_status.csv` |

#### Half 2 - `src/frustration.py` and `src/run_pipeline.py`

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

### Frustration thresholds are hardcoded (legacy pipeline only)

`config.yaml` contains a `frustration.thresholds` block that **no code reads**. The live
thresholds are hardcoded in three places: `src/frustration.py`, the plotting logic in
`src/run_pipeline.py`, and again in `src/test_frustration.py` — so the test confirms only
that it agrees with itself.

Fixed in `atomfrust`: the numbers live once, as pydantic defaults, and the classification
rule lives once in `atomfrust/analyze/classify.py`. A test AST-scans the package and fails
if both literals appear anywhere else. It has already caught one regression.

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

Stage 4 records covalent link information from mmCIF `_struct_conn`, but Stage 6 does not
apply covalent chemistry. Verified: **zero of the 112 checked-in `.params` files contain a
`CONNECT` record**, despite `scripts/05_prepare_ligand.py`'s docstring claiming the script
adds them. The 15 covalent complexes are analysed as if they were non-covalent.

`atomfrust` narrows the gap without closing it. Anchors are recovered from
`_struct_conn` or the Stage-4 summary (`atomfrust prepare --covalent-from`), the anchor
residue is frozen against mutation, the bond is forced into the contact graph regardless
of distance, and covalent systems are reportable as their own stratum — so they are
**identifiable and constrained** rather than silently mis-scored. Full covalent chemistry
(a real Rosetta bond, a patched residue type, correct valence) is deferred.

One consequence worth knowing: a 1.81 A S-C covalent bond reads as a steric clash to a
pose-validity checker that has no covalent concept, so every covalent system's pose pass
rate is floored at zero. The gate therefore records rather than rejects.

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

Rosetta energies are negative-is-favourable, so a more favourable native contact gives a
*positive* index and is classified as minimally frustrated.

### Many-body correction (Eq. 2)

```text
E_ij = e_ij
     + 0.5 * Σ_{k ∈ contacts(i), k≠j} e_ik
     + 0.5 * Σ_{l ∈ contacts(j), l≠i} e_jl
```

**This collapses.** Writing `B_i` for the sum over *all* of i's partners, the excluded
sums are `B_i - e_ij` and `B_j - e_ij`, so `E_ij = 0.5*(B_i + B_j)` and the pair term
cancels exactly — see [What the diagnostics found](#what-the-diagnostics-found). The
exclusions are in the published paper, so this is a property of the equation as printed.

`atomfrust` makes the formula selectable rather than assumed:

| mode | formula | degenerate? |
|---|---|---|
| `chen_literal` | as printed above | **yes** — required for any reproduction claim |
| `pair_retained` | sums over **all** partners, so the pair term survives | no |
| `pair_only` | `e_ij` alone | no |

Note `pair_retained - chen_literal == e_ij` exactly. Which of these is the scientific
object going forward is an open decision, not something to inherit silently.

### Decoy generation

What this pipeline does:

1. Estimate the amino-acid composition from the native structure.
2. Randomize the full protein sequence on the native backbone (i.i.d. draw).
3. Repack side chains.
4. Run chi-only minimization.
5. Restore backbone atom coordinates from the native pose.

What the paper describes: *"we randomly shuffle the protein sequence and then repack the
resulting sequence onto the backbone"* — a **permutation**, conserving the native
amino-acid multiset exactly — followed by *"a short Monte-Carlo relaxation"*, with the
native contact energies *"obtained in a similar fashion by omitting the shuffling step"*,
i.e. the native is repacked and relaxed too. 1000 decoys per contact.

Step 5 is not optional and is not a minimiser artifact: `MutateResidue` rebuilds the
carbonyl O from idealised geometry and drifts ~0.5-0.8 A from the crystallographic
position. `atomfrust` enforces backbone identity as a runtime post-condition on every
decoy at 1e-6 A rather than as a single test at 0.05 A.

## Disclaimer

This is an independent reimplementation. The original paper's code was not published, so
exact numerical agreement was never the target; qualitatively correct behaviour was.

That framing turned out to matter less than expected. The diagnostics above show the
original pipeline computes a *different quantity* from the one the paper reports — not a
noisier estimate of the same one — so "no exact agreement expected" was not the reason the
numbers disagreed. `atomfrust` reproduces this pipeline exactly where the two overlap,
which is what makes the remaining differences attributable to the method rather than to
the implementation.
