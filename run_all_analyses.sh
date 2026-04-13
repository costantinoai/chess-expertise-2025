#!/usr/bin/env bash
################################################################################
# run_all_analyses.sh -- pipeline runner for chess-expertise-2025
#
# Usage:
#   ./run_all_analyses.sh [LEVEL]
#
# LEVEL (default: group)
#   all           - run every stage in order: fmriprep -> spm -> subject-level -> group
#   fmriprep      - invoke the external fMRIPrep pipeline (not implemented here; stub)
#   spm           - invoke chess-glm MATLAB first-/second-level GLM scripts (stub;
#                   prints the MATLAB commands to run)
#   subject-level - run every subject-level script that writes into BIDS/derivatives/
#                   (MATLAB stubs + the two Python subject scripts: behavioral and
#                   manifold)
#   group         - run every Python 1x/8x/9x script that writes into
#                   results/<analysis>/{data,tables,figures}/ (the default and the
#                   only path that does NOT re-run MATLAB or external pipelines)
#
# Script numbering convention:
#   0x = subject-level  (writes per-subject data to BIDS/derivatives/)
#   1x = group-level    (reads derivatives → writes group aggregates to results/)
#   8x = tables         (reads results/ → writes formatted tables)
#   9x = figures        (reads results/ → writes rendered figures)
#   x is the same digit for related scripts within a pipeline.
#
# The `group` level is what you normally want after the BIDS derivatives
# (bundles C + D + E) are in place. It regenerates the entire `results/`
# tree locally from the per-subject derivatives.
#
# Options:
#   -p, --python PATH  Explicit Python interpreter (overrides conda env discovery)
#   -d, --debug        Enable CHESS_LOG_LEVEL=DEBUG for verbose logs
#   -h, --help         Show this help and exit
################################################################################

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================
CONDA_ENV="chess-expertise"
PYTHON_BIN=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEVEL="group"
DEBUG_MODE=false

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
print_ok()      { echo -e "${GREEN}[OK]${NC} $*"; }
print_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
print_err()     { echo -e "${RED}[ERROR]${NC} $*"; }

# ==============================================================================
# Argument parsing
# ==============================================================================
show_usage() {
  sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's|^# \?||'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -d|--debug)
      DEBUG_MODE=true
      shift
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    all|fmriprep|spm|subject-level|group)
      LEVEL="$1"
      shift
      ;;
    *)
      print_err "Unknown argument: $1"
      show_usage
      exit 1
      ;;
  esac
done

if [[ "$DEBUG_MODE" == "true" ]]; then
  export CHESS_LOG_LEVEL=DEBUG
  print_info "Debug logging enabled"
fi

# ==============================================================================
# Python interpreter discovery
# ==============================================================================
resolve_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    echo "$PYTHON_BIN"
    return
  fi
  # Prefer mamba/conda env by name
  local candidate=""
  if command -v mamba >/dev/null 2>&1; then
    candidate="$(mamba env list 2>/dev/null | awk -v env="$CONDA_ENV" '$1==env {print $NF}' || true)"
  fi
  if [[ -z "$candidate" ]] && command -v conda >/dev/null 2>&1; then
    candidate="$(conda env list 2>/dev/null | awk -v env="$CONDA_ENV" '$1==env {print $NF}' || true)"
  fi
  if [[ -n "$candidate" && -x "$candidate/bin/python" ]]; then
    echo "$candidate/bin/python"
    return
  fi
  # Fall back to system python
  command -v python || command -v python3
}

PY="$(resolve_python)"
if [[ -z "$PY" ]]; then
  print_err "No Python interpreter found. Pass --python or activate the conda env."
  exit 1
fi
print_info "Using Python: $PY"

# Ensure the repo is importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ==============================================================================
# Stage definitions
# ==============================================================================
# Python group-level scripts: everything that reads from BIDS derivatives
# (or the unified results/ tree) and writes into
# results/<analysis>/{data,tables,figures}/.
#
# Order within each analysis is deterministic so plotting scripts always see
# the latest table outputs.
declare -a GROUP_SCRIPTS=(
  # ── Numbering convention ────────────────────────────────────────────────
  #   0x = subject-level  (writes per-subject data to BIDS/derivatives/)
  #   1x = group-level    (reads from derivatives, writes group stats to results/)
  #   8x = tables         (reads from results/, writes formatted tables)
  #   9x = figures        (reads from results/, writes rendered figures)
  #   x is the same digit for related subject/group/table/figure scripts.
  #
  #   Subject-level scripts (0x) are in SUBJECT_LEVEL_SCRIPTS_PY below.
  #   This array lists everything from group-level onward.
  # ────────────────────────────────────────────────────────────────────────

  # chess-behavioral (group stats -> tables -> figures)
  "chess-behavioral/11_behavioral_rsa_group.py"
  "chess-behavioral/81_table_behavioral_correlations.py"
  "chess-behavioral/91_plot_behavioral_panels.py"

  # chess-manifold (group stats -> tables -> figures)
  "chess-manifold/11_manifold_group.py"
  "chess-manifold/81_table_manifold_pr.py"
  "chess-manifold/91_plot_manifold_panels.py"

  # chess-mvpa (group over MATLAB-produced per-subject TSVs)
  "chess-mvpa/11_mvpa_group_rsa.py"
  "chess-mvpa/12_mvpa_group_decoding.py"
  "chess-mvpa/81_table_mvpa_rsa.py"
  "chess-mvpa/82_table_mvpa_decoding.py"
  "chess-mvpa/91_plot_mvpa_rsa.py"
  "chess-mvpa/92_plot_mvpa_decoding.py"

  # chess-neurosynth (group-only reads of searchlight + SPM smoothed group maps)
  "chess-neurosynth/11_univariate_neurosynth.py"
  "chess-neurosynth/12_rsa_neurosynth.py"
  "chess-neurosynth/81_table_neurosynth_univariate.py"
  "chess-neurosynth/82_table_neurosynth_rsa.py"
  "chess-neurosynth/91_plot_neurosynth_univariate.py"
  "chess-neurosynth/92_plot_neurosynth_rsa.py"

  # chess-supplementary/behavioral-reliability
  "chess-supplementary/behavioral-reliability/11_behavioral_split_half_reliability.py"
  "chess-supplementary/behavioral-reliability/12_marginal_split_half.py"
  "chess-supplementary/behavioral-reliability/81_table_split_half_reliability.py"
  "chess-supplementary/behavioral-reliability/91_plot_reliability_panels.py"

  # chess-supplementary/eyetracking
  "chess-supplementary/eyetracking/11_eye_decoding_group.py"
  "chess-supplementary/eyetracking/81_table_eyetracking_decoding.py"
  "chess-supplementary/eyetracking/91_plot_eyetracking_decoding.py"

  # chess-supplementary/mvpa-finer (Python group over MATLAB-appended TSVs)
  "chess-supplementary/mvpa-finer/11_mvpa_finer_group_rsa.py"
  "chess-supplementary/mvpa-finer/12_mvpa_finer_group_decoding.py"
  "chess-supplementary/mvpa-finer/81_table_mvpa_finer_rsa.py"
  "chess-supplementary/mvpa-finer/82_table_mvpa_finer_decoding.py"
  "chess-supplementary/mvpa-finer/82_table_mvpa_extended_dimensions.py"
  "chess-supplementary/mvpa-finer/91_plot_mvpa_finer_panel.py"

  # chess-supplementary/neurosynth-terms (single figure over atlas terms)
  "chess-supplementary/neurosynth-terms/91_plot_neurosynth_terms.py"

  # chess-supplementary/rdm-intercorrelation
  "chess-supplementary/rdm-intercorrelation/11_rdm_intercorrelation.py"
  "chess-supplementary/rdm-intercorrelation/81_table_rdm_intercorr.py"
  "chess-supplementary/rdm-intercorrelation/91_plot_rdm_intercorr.py"

  # chess-supplementary/rsa-rois
  "chess-supplementary/rsa-rois/11_rsa_roi_group.py"
  "chess-supplementary/rsa-rois/81_table_rsa_rois.py"
  "chess-supplementary/rsa-rois/82_table_roi_maps_rsa.py"
  "chess-supplementary/rsa-rois/91_plot_rsa_rois.py"

  # chess-supplementary/run-matching (Python group over MATLAB-produced TSVs)
  "chess-supplementary/run-matching/11_pr_run_matched.py"
  "chess-supplementary/run-matching/12_group_rsa_run_matched.py"
  "chess-supplementary/run-matching/13_compare_run_matched.py"
  "chess-supplementary/run-matching/81_table_rsa_run_matched.py"
  "chess-supplementary/run-matching/82_table_pr_run_matched.py"

  # chess-supplementary/skill-gradient
  "chess-supplementary/skill-gradient/11_skill_gradient_group.py"
  "chess-supplementary/skill-gradient/91_plot_skill_gradient.py"

  # chess-supplementary/subcortical-rois (Python group over MATLAB TSVs)
  "chess-supplementary/subcortical-rois/11_subcortical_group_rsa.py"
  "chess-supplementary/subcortical-rois/12_subcortical_group_decoding.py"
  "chess-supplementary/subcortical-rois/91_plot_subcortical_rsa.py"
  "chess-supplementary/subcortical-rois/92_plot_atlas_on_mni.py"
  "chess-supplementary/subcortical-rois/93_plot_subcortical_decoding.py"

  # chess-supplementary/task-engagement
  "chess-supplementary/task-engagement/11_task_engagement_group.py"
  "chess-supplementary/task-engagement/12_familiarisation_group.py"
  "chess-supplementary/task-engagement/13_quantify_preference_drivers.py"
  "chess-supplementary/task-engagement/91_plot_novice_diagnostics.py"
  "chess-supplementary/task-engagement/92_plot_preference_features.py"
  "chess-supplementary/task-engagement/93_plot_gradient_panel.py"

  # chess-supplementary/univariate-rois
  "chess-supplementary/univariate-rois/11_univariate_roi_group.py"
  "chess-supplementary/univariate-rois/81_table_univariate_rois.py"
  "chess-supplementary/univariate-rois/82_table_roi_maps_univ.py"
  "chess-supplementary/univariate-rois/91_plot_univariate_rois.py"
)

# Subject-level scripts that WRITE into BIDS/derivatives/. Running these will
# re-compute derivative pipelines. MATLAB entries are intentionally left as
# comments: the task-12 refactor freezes MATLAB subject pipelines, and the
# group scripts above already consume the existing derivatives on disk.
declare -a SUBJECT_LEVEL_SCRIPTS_PY=(
  # 0x scripts: write per-subject data to BIDS/derivatives/
  "chess-behavioral/01_behavioral_rsa_subject.py"
  "chess-manifold/01_manifold_subject.py"
  "chess-supplementary/subcortical-rois/01_prepare_atlas.py"
  "chess-supplementary/rsa-rois/01_rsa_roi_subject.py"
  "chess-supplementary/univariate-rois/01_univariate_roi_subject.py"
  "chess-supplementary/eyetracking/01_eye_decoding_subject.py"
  "chess-supplementary/task-engagement/01_task_engagement_subject.py"
  "chess-supplementary/task-engagement/02_familiarisation_subject.py"
  "chess-supplementary/skill-gradient/01_skill_gradient_subject.py"
)

declare -a SUBJECT_LEVEL_SCRIPTS_MATLAB=(
  "chess-mvpa/01_roi_mvpa_subject.m"
  "chess-mvpa/04_searchlight_rsa.m"
  "chess-supplementary/mvpa-finer/01_roi_decoding_fine.m"
  "chess-supplementary/run-matching/01_roi_rsa_run_matched.m"
  "chess-supplementary/subcortical-rois/subcortical_rsa.m"
)

declare -a SPM_SCRIPTS_MATLAB=(
  "chess-glm/01_spm_glm_firstlevel.m"
  "chess-glm/02_spm_second_level_within.m"
  "chess-glm/03_spm_second_level_two_sample.m"
)

# ==============================================================================
# Runners
# ==============================================================================
run_python_scripts() {
  local label="$1"
  shift
  local -a scripts=("$@")
  local n=${#scripts[@]}
  local i=0
  local failed=0
  print_info "[$label] running $n Python scripts sequentially"
  for script in "${scripts[@]}"; do
    i=$((i+1))
    local path="${REPO_ROOT}/${script}"
    if [[ ! -f "$path" ]]; then
      print_warn "[$label] ($i/$n) missing: $script"
      continue
    fi
    print_info "[$label] ($i/$n) $script"
    if "$PY" "$path"; then
      print_ok "[$label] ($i/$n) $script"
    else
      print_err "[$label] ($i/$n) FAILED: $script"
      failed=$((failed+1))
    fi
  done
  if (( failed > 0 )); then
    print_err "[$label] $failed / $n scripts failed"
    return 1
  fi
  print_ok "[$label] all $n Python scripts completed"
}

run_fmriprep_stub() {
  print_warn "[fmriprep] fMRIPrep is an external pipeline and must be invoked directly;"
  print_warn "           there is no in-repo driver. Expected command (adjust to your setup):"
  echo "  fmriprep <BIDS_ROOT> <BIDS_DERIVATIVES>/fmriprep participant \\"
  echo "           --participant-label ... --output-spaces MNI152NLin2009cAsym:res-2 T1w \\"
  echo "           --nprocs ... --omp-nthreads ..."
  print_warn "[fmriprep] skipping in this run."
}

run_spm_stub() {
  print_warn "[spm] SPM first-/second-level GLMs run in MATLAB. The chess-glm scripts"
  print_warn "      live at these paths and must be invoked from MATLAB:"
  for s in "${SPM_SCRIPTS_MATLAB[@]}"; do
    echo "  matlab -batch \"run('${REPO_ROOT}/${s}'); exit\""
  done
  print_warn "[spm] skipping in this run (refactor policy: no MATLAB recomputation)."
}

run_subject_level_stub() {
  print_warn "[subject-level] MATLAB subject-level scripts are FROZEN in the task-12"
  print_warn "                refactor. Python subject-level scripts are safe to run."
  print_info "[subject-level] Python subject scripts:"
  run_python_scripts "subject-level/python" "${SUBJECT_LEVEL_SCRIPTS_PY[@]}"
  print_info "[subject-level] MATLAB subject scripts (invoke manually if needed):"
  for s in "${SUBJECT_LEVEL_SCRIPTS_MATLAB[@]}"; do
    echo "  matlab -batch \"run('${REPO_ROOT}/${s}'); exit\""
  done
}

# ==============================================================================
# Main
# ==============================================================================
print_info "Level: $LEVEL"
print_info "Repo:  $REPO_ROOT"
print_info "Python path set so 'common' and 'analyses' packages are importable."

case "$LEVEL" in
  all)
    run_fmriprep_stub
    run_spm_stub
    run_subject_level_stub
    run_python_scripts "group" "${GROUP_SCRIPTS[@]}"
    ;;
  fmriprep)
    run_fmriprep_stub
    ;;
  spm)
    run_spm_stub
    ;;
  subject-level)
    run_subject_level_stub
    ;;
  group)
    run_python_scripts "group" "${GROUP_SCRIPTS[@]}"
    ;;
  *)
    print_err "Unknown LEVEL: $LEVEL"
    show_usage
    exit 1
    ;;
esac

print_ok "run_all_analyses.sh ($LEVEL) finished."
