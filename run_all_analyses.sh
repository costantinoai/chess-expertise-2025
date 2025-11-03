#!/bin/bash
################################################################################
# run_all_analyses.sh - Automated Analysis Pipeline Runner
################################################################################
#
# Purpose:
#   1. Backup all existing results folders to ./backup/<timestamp>_full-results.zip
#   2. Delete backed-up results folders
#   3. Disable pylustrator in CONFIG (always off during runs)
#   4. Run all analysis scripts (01_*, 81_*, 91_*, etc.) in parallel by folder
#   5. Re-enable pylustrator in CONFIG (always on after run, even on failure)
#   6. Generate summary report
#
# Usage:
#   ./run_all_analyses.sh [--levels LEVELS]
#
# Options:
#   -l, --levels LEVELS   Comma-separated list selecting which stages to run.
#                         Valid values: analysis, tables, figures
#                         Examples:
#                           --levels analysis
#                           --levels tables,figures
#                           --levels analysis,tables,figures
#
# Environment & Requirements:
#   - Set the conda environment name at the top (CONDA_ENV). Ensure all required
#     packages for this repo are installed in that environment (see environment.yml
#     or requirements.txt).
#   - zip utility
#
# Quick start (examples):
#   cd /home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2025
#   # Recommended (conda-forge):
#   mamba env create -f environment.yml   # or: conda env create -f environment.yml
#   conda activate chess-expertise
#   # Alternative (requirements.txt):
#   conda create -n chess-expertise python=3.11
#   conda activate chess-expertise && pip install -r requirements.txt
#   # Run everything
#   ./run_all_analyses.sh --levels analysis,tables,figures
#
################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ==============================================================================
# Configuration
# ==============================================================================

CONDA_ENV="ml"  # Choose your conda environment here
PYTHON_BIN=""
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="./backup"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${REPO_ROOT}/common/constants.py"
LOG_DIR="${REPO_ROOT}/results-bundle/logs/${TIMESTAMP}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default stage selection (run everything unless --levels is provided)
RUN_ANALYSIS=true
RUN_TABLES=true
RUN_FIGURES=true
# Sequential mode: when true, do NOT launch background jobs and tee logs to console
SEQUENTIAL=false
# Debug mode: when true, set CHESS_LOG_LEVEL=DEBUG for verbose logging
DEBUG_MODE=false

# Note: No thread env tuning; keep runtime simple and predictable.

# ==============================================================================
# Argument Parsing
# ==============================================================================

show_usage() {
  cat <<EOF
Usage: $0 [--levels LEVELS] [--sequential] [--debug] [--python PATH]

Options:
  -l, --levels LEVELS   Comma-separated list selecting which stages to run.
                        Valid values: analysis, tables, figures
                        Examples:
                          --levels analysis
                          --levels tables,figures
                          --levels analysis,tables,figures
  -s, --sequential      Run jobs sequentially and stream logs to console
  -d, --debug           Enable verbose logging (CHESS_LOG_LEVEL=DEBUG)
  -p, --python PATH     Use explicit Python interpreter (overrides conda env)
  -h, --help            Show this help and exit
EOF
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -l|--levels)
        if [[ $# -lt 2 ]]; then
          echo "Error: --levels requires an argument" >&2
          show_usage
          exit 2
        fi
        # Reset defaults when explicitly selecting levels
        RUN_ANALYSIS=false
        RUN_TABLES=false
        RUN_FIGURES=false
        IFS=',' read -r -a _levels <<< "$2"
        for _lv in "${_levels[@]}"; do
          # Normalize token: lowercase and strip whitespace
          _tok=${_lv,,}
          _tok=${_tok//[[:space:]]/}
          case "$_tok" in
            analysis)
              RUN_ANALYSIS=true
              ;;
            tables)
              RUN_TABLES=true
              ;;
            figures)
              RUN_FIGURES=true
              ;;
            "")
              ;;
            *)
              echo "Error: invalid level '$_lv'. Valid: analysis, tables, figures" >&2
              exit 2
              ;;
          esac
        done
        shift 2
        ;;
      -p|--python)
        if [[ $# -lt 2 ]]; then
          echo "Error: --python requires a path argument" >&2
          show_usage
          exit 2
        fi
        PYTHON_BIN="$2"
        shift 2
        ;;
      -s|--sequential)
        SEQUENTIAL=true
        shift 1
        ;;
      -d|--debug)
        DEBUG_MODE=true
        shift 1
        ;;
      -h|--help)
        show_usage
        exit 0
        ;;
      *)
        echo "Error: unknown argument '$1'" >&2
        show_usage
        exit 2
        ;;
    esac
  done
fi

# Set logging level based on debug mode
if $DEBUG_MODE; then
  export CHESS_LOG_LEVEL="DEBUG"
else
  export CHESS_LOG_LEVEL="INFO"
fi

# Analysis folders (in logical order)
ANALYSIS_FOLDERS=(
  "chess-behavioral"
  "chess-manifold"
  "chess-mvpa"
  "chess-neurosynth"
  "chess-supplementary/behavioral-reliability"
  "chess-supplementary/dataset-viz"
  "chess-supplementary/eyetracking"
  "chess-supplementary/mvpa-finer"
  "chess-supplementary/neurosynth-terms"
  "chess-supplementary/rdm-intercorrelation"
  "chess-supplementary/rsa-rois"
  "chess-supplementary/univariate-rois"
)

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
  echo -e "\n${BLUE}========================================${NC}"
  echo -e "${BLUE}$1${NC}"
  echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
  echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
  echo -e "${RED}✗${NC} $1"
}

print_info() {
  echo -e "${BLUE}→${NC} $1"
}

# Function to selectively clean results based on selected levels
clean_results_selective() {
  local results_dir=$1

  # If running everything (default without --levels), delete entire results directory
  if $RUN_ANALYSIS && $RUN_TABLES && $RUN_FIGURES; then
    rm -rf "$results_dir"
    print_info "Deleted (all levels): $results_dir"
    return 0
  fi

  # Otherwise, selectively clean based on levels
  if [ ! -d "$results_dir" ]; then
    return 0
  fi

  # Clean tables subfolder
  if $RUN_TABLES && [ -d "$results_dir/tables" ]; then
    rm -rf "$results_dir/tables"
    print_info "Deleted tables: $results_dir/tables"
  fi

  # Clean figures subfolder
  if $RUN_FIGURES && [ -d "$results_dir/figures" ]; then
    rm -rf "$results_dir/figures"
    print_info "Deleted figures: $results_dir/figures"
  fi

  # Clean analysis artifacts (files in results root, excluding subfolders and .log/.py files)
  if $RUN_ANALYSIS; then
    find "$results_dir" -maxdepth 1 -type f ! -name "*.log" ! -name "*.py" -delete 2>/dev/null || true
    print_info "Deleted analysis artifacts: $results_dir/* (except tables/, figures/, *.log, *.py)"
  fi
}

# ==============================================================================
# Phase 1: Setup & Validation
# ==============================================================================

print_header "PHASE 1: SETUP & VALIDATION"

# Change to repo root
cd "$REPO_ROOT"
print_info "Working directory: $REPO_ROOT"

# Check conda/miniforge
if ! command -v conda &> /dev/null; then
  print_error "conda not found. Please install miniforge/conda."
  exit 1
fi
print_success "conda found"

# Resolve Python interpreter
if [[ -n "$PYTHON_BIN" ]]; then
  print_info "Using explicit Python interpreter: $PYTHON_BIN"
else
  # Derive conda root from conda executable without invoking conda subcommands
  CONDA_BIN=$(command -v conda)
  CONDA_ROOT=$(dirname "$(dirname "$CONDA_BIN")")
  PYTHON_BIN="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
  print_info "Resolved conda root: $CONDA_ROOT"
  print_info "Using env '${CONDA_ENV}' Python: $PYTHON_BIN"
fi

# Validate Python interpreter
if [[ ! -x "$PYTHON_BIN" ]]; then
  print_error "Python interpreter not found or not executable: $PYTHON_BIN"
  echo "Hint: Ensure conda env '${CONDA_ENV}' exists under miniforge, or pass --python /path/to/python"
  exit 1
fi
print_success "Python interpreter is available"

# Check CONFIG file
if [ ! -f "$CONFIG_FILE" ]; then
  print_error "CONFIG file not found: $CONFIG_FILE"
  exit 1
fi
print_success "CONFIG file found"

# Check zip utility
if ! command -v zip &> /dev/null; then
  print_error "zip utility not found. Please install it (apt install zip)."
  exit 1
fi
print_success "zip utility found"

print_header "STAGE SELECTION"
print_info "Run analysis scripts:  ${RUN_ANALYSIS}"
print_info "Run table scripts:     ${RUN_TABLES}"
print_info "Run figure scripts:    ${RUN_FIGURES}"
print_info "Sequential mode:       ${SEQUENTIAL}"
print_info "Log level:             ${CHESS_LOG_LEVEL}"
print_info "Python:                ${PYTHON_BIN}"
if [[ "$RUN_ANALYSIS$RUN_TABLES$RUN_FIGURES" == "falsefalsefalse" ]]; then
  print_error "No stages selected. Use --levels with one or more of: analysis,tables,figures"
  exit 2
fi

START_TIME=$(date +%s)

# ==============================================================================
# Phase 2: Backup Results
# ==============================================================================

print_header "PHASE 2: BACKUP RESULTS"

# Clean results-bundle directory from previous runs
RESULTS_BUNDLE_DIR="${REPO_ROOT}/results-bundle"
if [ -d "$RESULTS_BUNDLE_DIR" ]; then
  print_info "Removing previous results-bundle directory..."
  rm -rf "$RESULTS_BUNDLE_DIR"
  print_success "Previous results-bundle directory removed"
fi

# Create log directory (saved under results-bundle)
mkdir -p "$LOG_DIR"
print_success "Log directory created: $LOG_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"
print_success "Backup directory ready: $BACKUP_DIR"

# Find all results directories (exclude backup folder itself)
print_info "Searching for results directories..."
RESULTS_DIRS=$(find . -type d -name "results" -not -path "*/backup/*" -not -path "*/.git/*" | sort)

if [ -z "$RESULTS_DIRS" ]; then
  print_warning "No results directories found. Nothing to backup."
  SKIP_BACKUP=true
else
  RESULTS_COUNT=$(echo "$RESULTS_DIRS" | wc -l)
  print_info "Found $RESULTS_COUNT results directories:"
  echo "$RESULTS_DIRS" | sed 's/^/  - /'

  # Create archive
  BACKUP_FILE="${BACKUP_DIR}/${TIMESTAMP}_full-results.zip"
  print_info "Creating backup archive: $BACKUP_FILE"

  # Use zip with relative paths (suppress verbose output)
  echo "$RESULTS_DIRS" | xargs zip -r -q "$BACKUP_FILE"

  # Verify archive was created and has content
  if [ ! -f "$BACKUP_FILE" ]; then
    print_error "Backup archive was not created!"
    exit 1
  fi

  BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
  if [ ! -s "$BACKUP_FILE" ]; then
    print_error "Backup archive is empty!"
    exit 1
  fi

  print_success "Backup created successfully (size: $BACKUP_SIZE)"

  # Selectively clean results directories based on selected levels
  print_info "Cleaning results directories based on selected levels..."
  echo "$RESULTS_DIRS" | while read -r dir; do
    if [ -d "$dir" ]; then
      clean_results_selective "$dir"
    fi
  done
  print_success "Results directories cleaned based on selected levels"

  SKIP_BACKUP=false
fi

# ==============================================================================
# Phase 3: Update .gitignore
# ==============================================================================

print_header "PHASE 3: UPDATE .gitignore"

GITIGNORE_FILE="${REPO_ROOT}/.gitignore"

if grep -q "^backup/" "$GITIGNORE_FILE" 2>/dev/null; then
  print_success "backup/ already in .gitignore"
else
  echo "backup/" >> "$GITIGNORE_FILE"
  print_success "Added backup/ to .gitignore"
fi

# ==============================================================================
# Phase 3.5: Prepare Manuscript Output Dirs
# ==============================================================================

# (Removed) Manuscript dir preparation is handled by saving utilities

# ==============================================================================
# Phase 4: Disable Pylustrator
# ==============================================================================

print_header "PHASE 4: DISABLE PYLUSTRATOR"

# Backup original CONFIG and force-disable for this run
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"
if grep -q "'ENABLE_PYLUSTRATOR'" "$CONFIG_FILE"; then
  sed -i "s/'ENABLE_PYLUSTRATOR': *True/'ENABLE_PYLUSTRATOR': False/" "$CONFIG_FILE"
  sed -i "s/'ENABLE_PYLUSTRATOR': *False/'ENABLE_PYLUSTRATOR': False/" "$CONFIG_FILE"
  print_success "Pylustrator disabled in CONFIG for this run"
else
  print_error "Could not find ENABLE_PYLUSTRATOR setting in CONFIG"
  exit 1
fi

# Always re-enable at the end, even on failure/interrupt
PYLU_RESTORED=false
restore_pylustrator() {
  if [ "$PYLU_RESTORED" = true ]; then return 0; fi
  if grep -q "'ENABLE_PYLUSTRATOR'" "$CONFIG_FILE"; then
    sed -i "s/'ENABLE_PYLUSTRATOR': *False/'ENABLE_PYLUSTRATOR': True/" "$CONFIG_FILE"
    sed -i "s/'ENABLE_PYLUSTRATOR': *True/'ENABLE_PYLUSTRATOR': True/" "$CONFIG_FILE"
    print_info "Pylustrator re-enabled in CONFIG"
  else
    print_warning "ENABLE_PYLUSTRATOR setting not found during restore"
  fi
  rm -f "${CONFIG_FILE}.bak" || true
  PYLU_RESTORED=true
}
trap restore_pylustrator EXIT INT TERM

# ==============================================================================
# Phase 5: Run Analyses (Parallel)
# ==============================================================================

print_header "PHASE 5: RUN ANALYSES (PARALLEL)"

# Function to run all scripts in a single analysis folder
run_analysis_folder() {
  local folder=$1
  local folder_name=$(echo "$folder" | tr '/' '_')
  local logfile="${LOG_DIR}/${folder_name}.log"

  # Helper local logger: tee to console in sequential mode, else only to logfile
  local _tee_cmd="cat >> \"$logfile\""
  if $SEQUENTIAL; then
    # shellcheck disable=SC2016
    _tee_cmd='tee -a "$logfile"'
  fi
  eval "echo \"========================================\" | $_tee_cmd"
  eval "echo \"Analysis: $folder\" | $_tee_cmd"
  eval "echo \"Started: $(date)\" | $_tee_cmd"
  eval "echo \"========================================\" | $_tee_cmd"
  eval "echo \"\" | $_tee_cmd"

  # Find all Python scripts matching pattern [0-9][0-9]_*.py (only .py files!)
  mapfile -t scripts_array < <(find "$folder" -maxdepth 1 -type f -name "[0-9][0-9]_*.py" 2>/dev/null | sort)

  if [ ${#scripts_array[@]} -eq 0 ]; then
    eval "echo \"No analysis scripts found in $folder\" | $_tee_cmd"
    return 0
  fi

  # Filter by selected stages
  local selected_scripts=()
  for script in "${scripts_array[@]}"; do
    local base
    base=$(basename "$script")
    local nn
    nn=${base:0:2}
    # Interpret as decimal ignoring leading zeros
    local n
    n=$((10#$nn))
    local category="other"
    if (( n >= 90 && n <= 99 )); then
      category="figures"
    elif (( n >= 80 && n <= 89 )); then
      category="tables"
    elif (( n >= 0 && n <= 79 )); then
      category="analysis"
    fi

    case "$category" in
      analysis)
        $RUN_ANALYSIS && selected_scripts+=("$script")
        ;;
      tables)
        $RUN_TABLES && selected_scripts+=("$script")
        ;;
      figures)
        $RUN_FIGURES && selected_scripts+=("$script")
        ;;
    esac
  done

  if [ ${#selected_scripts[@]} -eq 0 ]; then
    eval "echo \"No scripts match selected levels in $folder\" | $_tee_cmd"
    return 0
  fi

  local script_count=${#selected_scripts[@]}
  eval "echo \"Found $script_count Python script(s) to run after level filtering\" | $_tee_cmd"
  eval "echo \"\" | $_tee_cmd"

  # Track failures for this folder
  local failed_count=0
  local failed_scripts=()

  # Run each script in order
  local script_num=0
  for script in "${selected_scripts[@]}"; do
    script_num=$((script_num + 1))
    local script_name=$(basename "$script")

    eval "echo \"----------------------------------------\" | $_tee_cmd"
    eval "echo \"[$script_num/$script_count] Running: $script_name\" | $_tee_cmd"
    eval "echo \"Started: $(date)\" | $_tee_cmd"
    eval "echo \"----------------------------------------\" | $_tee_cmd"

    # Run with conda, stream to console in sequential mode while logging
    local exit_code=0
    if $SEQUENTIAL; then
      if ! "$PYTHON_BIN" "$script" 2>&1 | tee -a "$logfile"; then
        exit_code=$?
      fi
    else
      if ! "$PYTHON_BIN" "$script" >> "$logfile" 2>&1; then
        exit_code=$?
      fi
    fi

    # Log result with prominent error markers
    if [ $exit_code -eq 0 ]; then
      eval "echo \"✓ SUCCESS: $script_name\" | $_tee_cmd"
    else
      eval "echo \"\" | $_tee_cmd"
      eval "echo \"╔═══════════════════════════════════════════════════════════════╗\" | $_tee_cmd"
      eval "echo \"║ ERROR: SCRIPT FAILED                                          ║\" | $_tee_cmd"
      eval "echo \"╚═══════════════════════════════════════════════════════════════╝\" | $_tee_cmd"
      eval "echo \"✗ FAILED: $script_name (exit code: $exit_code)\" | $_tee_cmd"
      eval "echo \"  Folder: $folder\" | $_tee_cmd"
      eval "echo \"  Script: $script\" | $_tee_cmd"
      eval "echo \"  Logfile: $logfile\" | $_tee_cmd"
      eval "echo \"╚═══════════════════════════════════════════════════════════════╝\" | $_tee_cmd"
      eval "echo \"\" | $_tee_cmd"

      # Track failure
      failed_count=$((failed_count + 1))
      failed_scripts+=("$script")

      # Write to global error log for summary
      echo "ERROR|$folder|$script_name|$exit_code|$logfile" >> "${LOG_DIR}/ERRORS.log"
    fi

    eval "echo \"Finished: $(date)\" | $_tee_cmd"
    eval "echo \"\" | $_tee_cmd"
  done

  eval "echo \"========================================\" | $_tee_cmd"
  eval "echo \"Analysis complete: $folder\" | $_tee_cmd"
  if [ $failed_count -gt 0 ]; then
    eval "echo \"FAILED: $failed_count of $script_count script(s) failed\" | $_tee_cmd"
  else
    eval "echo \"SUCCESS: All $script_count script(s) completed successfully\" | $_tee_cmd"
  fi
  eval "echo \"Finished: $(date)\" | $_tee_cmd"
  eval "echo \"========================================\" | $_tee_cmd"

  # Return non-zero exit code if any script failed
  return $failed_count
}

# Export function and variables for subshells
export -f run_analysis_folder
export PYTHON_BIN
export CONDA_ENV
export LOG_DIR
export RUN_ANALYSIS
export RUN_TABLES
export RUN_FIGURES

# Launch all analyses in parallel
if $SEQUENTIAL; then
  print_info "Launching ${#ANALYSIS_FOLDERS[@]} analysis jobs sequentially..."
  echo ""
  FAILED_JOBS=0
  JOB_NUM=0
  for folder in "${ANALYSIS_FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
      folder_name=$(echo "$folder" | tr '/' '_')
      print_info "Starting: $folder → ${LOG_DIR}/${folder_name}.log"
      if run_analysis_folder "$folder"; then
        JOB_NUM=$((JOB_NUM + 1))
        print_success "Job $JOB_NUM completed"
      else
        JOB_NUM=$((JOB_NUM + 1))
        print_error "Job $JOB_NUM failed"
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi
    else
      print_warning "Folder not found, skipping: $folder"
    fi
  done
  echo ""
  if [ $FAILED_JOBS -eq 0 ]; then
    print_success "All analysis jobs completed successfully"
  else
    print_warning "$FAILED_JOBS job(s) failed. Check logs in $LOG_DIR"
  fi
  PARALLEL_JOBS_LAUNCHED=0
else
  print_info "Launching ${#ANALYSIS_FOLDERS[@]} parallel analysis jobs..."
  echo ""

  PIDS=()
  for folder in "${ANALYSIS_FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
      folder_name=$(echo "$folder" | tr '/' '_')
      print_info "Starting: $folder → ${LOG_DIR}/${folder_name}.log"
      run_analysis_folder "$folder" &
      PIDS+=($!)
    else
      print_warning "Folder not found, skipping: $folder"
    fi
  done

  echo ""
  print_info "Waiting for ${#PIDS[@]} analysis jobs to complete..."
  print_info "Monitor progress: tail -f ${LOG_DIR}/*.log"
  echo ""

  # Wait for all background jobs and track failures
  FAILED_JOBS=0
  JOB_NUM=0
  for pid in "${PIDS[@]}"; do
    JOB_NUM=$((JOB_NUM + 1))
    if wait $pid; then
      print_success "Job $JOB_NUM (PID $pid) completed successfully"
    else
      print_error "Job $JOB_NUM (PID $pid) failed"
      FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
  done

  echo ""
  if [ $FAILED_JOBS -eq 0 ]; then
    print_success "All analysis jobs completed successfully"
  else
    print_warning "$FAILED_JOBS job(s) failed. Check logs in $LOG_DIR"
  fi
  PARALLEL_JOBS_LAUNCHED=${#PIDS[@]}
fi

# ==============================================================================
# Phase 6: Restore Pylustrator
# ==============================================================================

print_header "PHASE 6: RESTORE PYLUSTRATOR"
restore_pylustrator

# ==============================================================================
# Phase 7: Summary Report
# ==============================================================================

print_header "PHASE 7: SUMMARY REPORT"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    PIPELINE EXECUTION SUMMARY                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ "$SKIP_BACKUP" = false ]; then
  echo "  Backup Archive:    $BACKUP_FILE"
  echo "  Backup Size:       $BACKUP_SIZE"
  echo "  Folders Backed Up: $RESULTS_COUNT"
else
  echo "  Backup:            Skipped (no results found)"
fi

echo ""
echo "  Analysis Jobs:     ${#ANALYSIS_FOLDERS[@]} folders processed"
echo "  Parallel Jobs:     ${PARALLEL_JOBS_LAUNCHED} launched"
echo "  Failed Jobs:       $FAILED_JOBS"
echo ""
echo "  Log Directory:     $LOG_DIR"
echo "  Total Runtime:     ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""
PYLU_FLAG=$(grep -n "ENABLE_PYLUSTRATOR" "$CONFIG_FILE" | sed -n '1p' | sed -E "s/.*ENABLE_PYLUSTRATOR': *([A-Za-z]+).*/\1/")
echo "  Pylustrator:       $PYLU_FLAG"
echo "  Levels Run:        analysis=${RUN_ANALYSIS}, tables=${RUN_TABLES}, figures=${RUN_FIGURES}"
echo ""

# Display detailed error summary if any scripts failed
if [ -f "${LOG_DIR}/ERRORS.log" ]; then
  TOTAL_FAILED_SCRIPTS=$(wc -l < "${LOG_DIR}/ERRORS.log")
  echo "╔════════════════════════════════════════════════════════════════╗"
  echo "║                        ERROR SUMMARY                           ║"
  echo "╚════════════════════════════════════════════════════════════════╝"
  echo ""
  echo -e "${RED}  Total Failed Scripts: $TOTAL_FAILED_SCRIPTS${NC}"
  echo ""
  echo "  Failed Scripts Details:"
  echo "  ─────────────────────────────────────────────────────────────"

  # Parse and display each error in a readable format
  while IFS='|' read -r folder script_name exit_code logfile; do
    echo ""
    echo -e "  ${RED}✗${NC} $script_name (exit code: $exit_code)"
    echo "    Folder:  $folder"
    echo "    Logfile: $logfile"
  done < "${LOG_DIR}/ERRORS.log"

  echo ""
  echo "  ─────────────────────────────────────────────────────────────"
  echo ""
  echo "  To investigate errors:"
  echo "    • View specific log:  cat $LOG_DIR/<folder_name>.log"
  echo "    • Search for errors:  grep -n 'ERROR:' $LOG_DIR/*.log"
  echo "    • View all errors:    cat ${LOG_DIR}/ERRORS.log"
  echo ""
fi

if [ $FAILED_JOBS -eq 0 ]; then
  print_success "Pipeline completed successfully!"
else
  print_warning "Pipeline completed with errors. Review logs for details."
fi

echo ""
echo "Next steps:"
echo "  1. Review logs:     ls -lh ${LOG_DIR}/"
echo "  2. Check results:   ls -lh chess-*/results/"
echo "  3. View manuscript: ls -lh /home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-manuscript/write/figures/"
echo ""

# Exit with failure code if any jobs failed
# ==============================================================================
# Phase 8: Export Manuscript Bundle (timestamped)
# ==============================================================================

# Exit code based on job results
if [ $FAILED_JOBS -gt 0 ]; then
  exit 1
else
  exit 0
fi
