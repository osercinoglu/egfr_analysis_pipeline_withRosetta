#!/usr/bin/env bash
#
# One-command setup for the EGFR Atomistic Frustration Pipeline.
#
#   git clone <repo> && cd egfr_analysis_pipeline_withRosetta && ./setup.sh
#
# Assumes nothing is installed. Bootstraps, in order: Miniforge (conda), the
# `frustrato` environment, PyRosetta, the Google Cloud CLI, credentials, the
# DVC-tracked data, and a verification pass.
#
# Every stage is idempotent — re-running after a failure resumes rather than
# redoing. Run ./setup.sh --help for flags.
#
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults (override via flags)
# ---------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-frustrato}"
QUOTA_PROJECT="${QUOTA_PROJECT:-}"
CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
GCLOUD_ROOT="${GCLOUD_ROOT:-$HOME/google-cloud-sdk}"
BUCKET="gs://egfr-analysis-pipeline-withrosetta"

SKIP_CONDA=0 SKIP_PYROSETTA=0 SKIP_AUTH=0 NO_DATA=0 SKIP_TESTS=0 ASSUME_YES=0

# Stage outcomes, filled in as we go, printed by the summary at the end.
declare -A STATUS
for s in conda env pyrosetta gcloud auth data tests; do STATUS[$s]="skipped"; done

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'
    BLU=$'\033[34m'; DIM=$'\033[2m'; RST=$'\033[0m'
else
    BOLD='' RED='' GRN='' YLW='' BLU='' DIM='' RST=''
fi
step()  { printf '\n%s==> %s%s\n' "$BOLD$BLU" "$*" "$RST"; }
info()  { printf '    %s\n' "$*"; }
ok()    { printf '    %s✓%s %s\n' "$GRN" "$RST" "$*"; }
warn()  { printf '    %s!%s %s\n' "$YLW" "$RST" "$*" >&2; }
die()   { printf '\n%sERROR:%s %s\n' "$RED$BOLD" "$RST" "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: ./setup.sh [options]

Bootstraps everything needed to run the EGFR frustration pipeline.

Options:
  --env-name NAME        Conda environment name (default: frustrato)
  --quota-project ID     GCP project to bill API calls to. Required for
                         user credentials; without it Google emits a
                         "no quota project" warning on every call.
  --credentials PATH     Service-account key JSON. Enables fully unattended
                         setup and is how a collaborator with a shared key
                         authenticates. Also read from
                         GOOGLE_APPLICATION_CREDENTIALS.
  --conda-root PATH      Where to install Miniforge (default: ~/miniforge3)
  --skip-conda           Use whatever conda is already on PATH
  --skip-pyrosetta       Do not install PyRosetta (~1.7 GB download)
  --skip-auth            Do not install gcloud or configure credentials
  --no-data              Do not run `dvc pull`
  --skip-tests           Do not run the test suite at the end
  --yes, -y              Non-interactive. Never prompts. Requires
                         --credentials (or an already-working ADC) for the
                         data stage; without credentials it warns, skips the
                         data, and still completes the code setup.
  --help, -h             Show this message

Examples:
  ./setup.sh                                   # full interactive setup
  ./setup.sh --quota-project my-gcp-project    # skip the ADC quota prompt
  ./setup.sh --credentials ~/key.json --yes    # unattended / CI / collaborator
  ./setup.sh --skip-auth --no-data             # code only, no cloud access
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)       ENV_NAME="${2:?--env-name needs a value}"; shift 2 ;;
        --quota-project)  QUOTA_PROJECT="${2:?--quota-project needs a value}"; shift 2 ;;
        --credentials)    CREDENTIALS="${2:?--credentials needs a value}"; shift 2 ;;
        --conda-root)     CONDA_ROOT="${2:?--conda-root needs a value}"; shift 2 ;;
        --skip-conda)     SKIP_CONDA=1; shift ;;
        --skip-pyrosetta) SKIP_PYROSETTA=1; shift ;;
        --skip-auth)      SKIP_AUTH=1; shift ;;
        --no-data)        NO_DATA=1; shift ;;
        --skip-tests)     SKIP_TESTS=1; shift ;;
        -y|--yes)         ASSUME_YES=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                die "Unknown option: $1  (try --help)" ;;
    esac
done

confirm() {  # confirm "prompt" -> 0 yes / 1 no ; auto-yes under --yes
    [[ $ASSUME_YES -eq 1 ]] && return 0
    local reply
    read -r -p "    $1 [Y/n] " reply </dev/tty || return 1
    [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------
step "1/7  Preflight"

OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
    Linux|Darwin) ;;
    *) die "Unsupported OS: $OS. This script supports Linux and macOS." ;;
esac
for tool in git curl; do
    command -v "$tool" >/dev/null 2>&1 || die "'$tool' is required but not installed."
done
[[ -f environment.yml ]] || die "environment.yml not found — run this from the repo root."
ok "$OS/$ARCH, git and curl present, repo root is $REPO_ROOT"

# ---------------------------------------------------------------------------
# 2. Conda (Miniforge)
# ---------------------------------------------------------------------------
step "2/7  Conda"

find_conda() {
    local c
    for c in "$CONDA_ROOT/bin/conda" "$HOME/miniforge3/bin/conda" \
             "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
        [[ -x "$c" ]] && { echo "$c"; return 0; }
    done
    command -v conda 2>/dev/null && return 0
    return 1
}

CONDA_BIN="$(find_conda || true)"

if [[ -z "$CONDA_BIN" && $SKIP_CONDA -eq 0 ]]; then
    info "No conda found. Miniforge will be installed to $CONDA_ROOT (~400 MB, no sudo)."
    info "Miniforge is used rather than Miniconda because environment.yml declares"
    info "conda-forge only, avoiding the Anaconda default-channel terms entirely."
    confirm "Install Miniforge now?" || die "Cannot continue without conda. Install it, then re-run with --skip-conda."
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${OS}-${ARCH}.sh"
    tmp="$(mktemp -t miniforge.XXXXXX.sh)"
    info "Downloading $url"
    curl -fsSL "$url" -o "$tmp" || die "Miniforge download failed. Check the URL for $OS/$ARCH."
    bash "$tmp" -b -p "$CONDA_ROOT" >/dev/null
    rm -f "$tmp"
    CONDA_BIN="$CONDA_ROOT/bin/conda"
    STATUS[conda]="installed"
    ok "Miniforge installed to $CONDA_ROOT"
elif [[ -z "$CONDA_BIN" ]]; then
    die "--skip-conda given but no conda found on PATH."
else
    STATUS[conda]="present"
    ok "Using conda at $CONDA_BIN"
fi

# Make `conda activate` usable inside this non-interactive shell.
CONDA_BASE="$("$CONDA_BIN" info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 3. Environment
# ---------------------------------------------------------------------------
step "3/7  Conda environment '$ENV_NAME'"

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    info "Environment exists — updating from environment.yml"
    # Deliberately NOT --prune. PyRosetta is installed by stage 4 via pip and is
    # absent from environment.yml, so pruning can tear it out and force a 1.7 GB
    # re-download on every re-run. The README's --prune form is for a manual
    # cleanup, not for this idempotent path.
    "$CONDA_BIN" env update -n "$ENV_NAME" -f environment.yml
    STATUS[env]="updated"
else
    info "Creating environment (this takes a few minutes)"
    "$CONDA_BIN" env create -f environment.yml -n "$ENV_NAME"
    STATUS[env]="created"
fi

conda activate "$ENV_NAME"
ENV_PY="$(command -v python)"
PY_TAG="$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
ok "Environment ready — python $($ENV_PY -V 2>&1 | cut -d' ' -f2) at $ENV_PY"

# ---------------------------------------------------------------------------
# 4. PyRosetta
# ---------------------------------------------------------------------------
step "4/7  PyRosetta"

if [[ $SKIP_PYROSETTA -eq 1 ]]; then
    warn "Skipped (--skip-pyrosetta). Stage 6 and 5 of 13 tests will not run."
elif python -c 'import pyrosetta' >/dev/null 2>&1; then
    STATUS[pyrosetta]="present"
    ok "Already installed: $(python -c 'import pyrosetta,sys; sys.stdout.write(pyrosetta.__version__ if hasattr(pyrosetta,"__version__") else "unknown")' 2>/dev/null || echo present)"
else
    # Platform tag, mirroring pyrosetta_installer.get_pyrosetta_os()
    if [[ "$OS" == "Darwin" ]]; then
        [[ "$ARCH" == "arm64" ]] && PR_OS="m1" || PR_OS="mac"
    elif [[ "$ARCH" == "aarch64" ]]; then PR_OS="aarch64"
    else PR_OS="ubuntu"
    fi

    info "Resolving newest release wheel (python${PY_TAG}, ${PR_OS})"
    # NOTE: the packaged pyrosetta_installer.install_pyrosetta() is NOT used —
    # it resolves via latest.html, which points at a placeholder wheel that
    # 404s on both mirrors. We read the same listing and take the newest
    # *versioned* wheel instead.
    WHEEL="$(python - "$PY_TAG" "$PR_OS" <<'PY'
import re, sys, urllib.request
py_tag, pr_os = sys.argv[1], sys.argv[2]
base = ("https://west.rosettacommons.org/pyrosetta/release/release/"
        f"PyRosetta4.Release.python{py_tag}.{pr_os}.wheel/")
try:
    html = urllib.request.urlopen(base, timeout=60).read().decode()
except Exception as exc:
    sys.exit(f"could not reach {base}: {exc}")
wheels = re.findall(r'href="(pyrosetta-\d+\.\d+[^"]*\.whl)"', html)
if not wheels:
    sys.exit(f"no versioned wheels listed at {base}")
key = lambda w: tuple(int(n) for n in re.match(r"pyrosetta-(\d+)\.(\d+)", w).groups())
print(base + max(wheels, key=key))
PY
)" || die "Could not resolve a PyRosetta wheel. The east mirror (graylab.jhu.edu) often fails TLS; only west is used here."

    info "Installing ${WHEEL##*/}"
    info "~1.7 GB download — expect 15-25 minutes on a typical connection."
    python -m pip install --progress-bar off "$WHEEL"
    STATUS[pyrosetta]="installed"
    ok "PyRosetta installed"
fi

# ---------------------------------------------------------------------------
# 5. Google Cloud CLI
# ---------------------------------------------------------------------------
step "5/7  Google Cloud CLI"

GCLOUD_BIN=""
if [[ $SKIP_AUTH -eq 1 ]]; then
    warn "Skipped (--skip-auth)."
elif [[ -n "$CREDENTIALS" ]]; then
    # A service-account key needs no gcloud at all.
    [[ -f "$CREDENTIALS" ]] || die "--credentials file not found: $CREDENTIALS"
    STATUS[gcloud]="not needed (service-account key)"
    ok "Using service-account key — gcloud CLI not required"
else
    for g in "$(command -v gcloud 2>/dev/null || true)" "$GCLOUD_ROOT/bin/gcloud"; do
        [[ -n "$g" && -x "$g" ]] && { GCLOUD_BIN="$g"; break; }
    done
    if [[ -z "$GCLOUD_BIN" ]]; then
        info "gcloud not found. It will be installed to $GCLOUD_ROOT (~150 MB, no sudo)."
        if confirm "Install the Google Cloud CLI now?"; then
            case "$OS" in
                Linux)  gurl="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-${ARCH}.tar.gz" ;;
                Darwin) [[ "$ARCH" == "arm64" ]] \
                          && gurl="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-arm.tar.gz" \
                          || gurl="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-x86_64.tar.gz" ;;
            esac
            tmpd="$(mktemp -d)"
            curl -fsSL "$gurl" -o "$tmpd/gcloud.tar.gz" || die "gcloud download failed ($gurl)"
            tar -xzf "$tmpd/gcloud.tar.gz" -C "$(dirname "$GCLOUD_ROOT")"
            rm -rf "$tmpd"
            "$GCLOUD_ROOT/install.sh" --quiet --path-update true >/dev/null
            GCLOUD_BIN="$GCLOUD_ROOT/bin/gcloud"
            STATUS[gcloud]="installed"
            ok "gcloud installed to $GCLOUD_ROOT (restart your shell to get it on PATH)"
        else
            warn "Continuing without gcloud — the data stage will be skipped."
            NO_DATA=1
        fi
    else
        STATUS[gcloud]="present"
        ok "Using gcloud at $GCLOUD_BIN"
    fi
fi

# ---------------------------------------------------------------------------
# 6. Credentials + data
# ---------------------------------------------------------------------------
step "6/7  Credentials and data"

adc_works() { python - <<'PY' >/dev/null 2>&1
import google.auth
from google.cloud import storage
c, p = google.auth.default()
storage.Client(credentials=c, project=p).bucket(
    "egfr-analysis-pipeline-withrosetta").exists()
PY
}

if [[ $SKIP_AUTH -eq 1 ]]; then
    warn "Credentials skipped (--skip-auth)."
    NO_DATA=1
elif [[ -n "$CREDENTIALS" ]]; then
    # Repo-local so it never lands in the shared .dvc/config. config.local is gitignored.
    dvc remote modify --local storage credentialpath "$CREDENTIALS"
    export GOOGLE_APPLICATION_CREDENTIALS="$CREDENTIALS"
    STATUS[auth]="service-account key"
    ok "Service-account key wired into .dvc/config.local (gitignored)"
elif adc_works; then
    STATUS[auth]="existing ADC"
    ok "Existing credentials work against $BUCKET"
elif [[ $ASSUME_YES -eq 1 ]]; then
    warn "--yes with no working credentials and no --credentials key."
    warn "Skipping the data stage; code setup will still complete."
    NO_DATA=1
    STATUS[auth]="none (unattended)"
elif [[ -z "$GCLOUD_BIN" ]]; then
    warn "No gcloud available to sign in with — skipping credentials and data."
    warn "Either install gcloud and re-run, or pass --credentials with a key file."
    NO_DATA=1
    STATUS[auth]="no gcloud"
else
    info "A browser window will open for Google sign-in. This is the one step"
    info "that cannot be automated."
    if confirm "Sign in to Google Cloud now?"; then
        "$GCLOUD_BIN" auth application-default login
        # The quota project is separate from `gcloud config set project`: it
        # decides which project API calls are billed to. Without it every call
        # warns and some APIs refuse outright.
        if [[ -z "$QUOTA_PROJECT" ]]; then
            QUOTA_PROJECT="$("$GCLOUD_BIN" config get-value project 2>/dev/null || true)"
            [[ "$QUOTA_PROJECT" == "(unset)" ]] && QUOTA_PROJECT=""
        fi
        if [[ -n "$QUOTA_PROJECT" ]]; then
            "$GCLOUD_BIN" auth application-default set-quota-project "$QUOTA_PROJECT" || \
                warn "Could not set quota project '$QUOTA_PROJECT' — you may need roles/serviceusage.serviceUsageConsumer on it."
            ok "Quota project set to $QUOTA_PROJECT"
        else
            warn "No quota project set. Re-run with --quota-project ID to silence"
            warn "the 'authenticated without a quota project' warning."
        fi
        STATUS[auth]="interactive ADC"
    else
        warn "Skipping credentials — data will not be pulled."
        NO_DATA=1
        STATUS[auth]="declined"
    fi
fi

if [[ $NO_DATA -eq 1 ]]; then
    warn "Data stage skipped. Run 'dvc pull' later once credentials are in place."
else
    # Guard against clobbering local work. `dvc pull` checks out the version
    # recorded in the .dvc files; if a directory has been modified since the
    # last `dvc add`, pulling can overwrite results that were never uploaded.
    # This is a real hazard on a machine that has run Stage 6 — the fix is
    # `dvc add <dir> && dvc push`, not a pull.
    # POSIX character class, not \s — BSD grep on macOS does not support \s.
    if [[ -n "$(dvc status 2>/dev/null | grep -E '^[[:space:]]+(modified|deleted):' || true)" ]]; then
        warn "Local DVC-tracked directories have uncommitted changes:"
        dvc status 2>/dev/null | sed 's/^/      /'
        warn "Skipping 'dvc pull' — it would overwrite these with the last-added"
        warn "version and lose work. Run 'dvc add <dir> && dvc push' first."
        STATUS[data]="skipped (local changes)"
    else
        info "Pulling DVC-tracked data (~45 MB: processed structures, params, results)"
        if dvc pull; then
            STATUS[data]="pulled"
            ok "Data in place"
        else
            STATUS[data]="FAILED"
            warn "dvc pull failed. Check bucket access — see README, 'Giving a collaborator access'."
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------
step "7/7  Verification"

if [[ $SKIP_TESTS -eq 1 ]]; then
    warn "Tests skipped (--skip-tests)."
else
    if python -m pytest src/test_frustration.py -q 2>&1 | tail -3; then
        STATUS[tests]="passed"
    else
        STATUS[tests]="FAILED"
        warn "Test suite reported failures — see output above."
    fi
fi

# `set -o pipefail` is active: a failing `ls` in these pipelines would abort the
# script, so count with find, which succeeds on an empty/missing directory.
count_files() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l | tr -d ' '; }
n_processed=$(count_files data/processed '*_clean.pdb')
n_params=$(count_files data/ligands/params '*.params')
n_results=$(count_files results '*_frustration.parquet')
has_pyrosetta=$(python -c 'import pyrosetta' >/dev/null 2>&1 && echo yes || echo no)

printf '\n%s%s Setup summary %s\n' "$BOLD" "════════════════" "$RST"
for s in conda env pyrosetta gcloud auth data tests; do
    printf '  %-12s %s\n' "$s" "${STATUS[$s]}"
done
printf '\n  %-28s %s\n' "PyRosetta importable" "$has_pyrosetta"
printf '  %-28s %s / 61\n' "processed structures"  "$n_processed"
printf '  %-28s %s\n'      "ligand params files"   "$n_params"
printf '  %-28s %s / 61\n' "Stage 6 results"       "$n_results"

cat <<EOF

${BOLD}Next steps${RST}
  conda activate $ENV_NAME
  python src/run_pipeline.py --mode validate --n_decoys 50     # ~10 min sanity check
  python src/run_pipeline.py --mode single --pdb_id 5GMP --n_decoys 50 --n-jobs 10

Re-run ./setup.sh at any time — every stage detects what is already done.
EOF

if [[ "${STATUS[tests]}" == "FAILED" || "${STATUS[data]}" == "FAILED" ]]; then
    exit 1
fi
