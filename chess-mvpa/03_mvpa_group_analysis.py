#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVPA Group Analysis (ROI-level)

Loads subject-level MVPA artifacts produced by 01_roi_decoding_main.m from
`BIDS/derivatives/mvpa/` and computes group statistics per ROI:

- Experts vs chance (one-sample, one-tailed > chance)
- Novices vs chance (one-sample, one-tailed > chance)
- Experts vs novices (Welch t-test, two-tailed) with FDR correction

Analysis-only: saves structured artifacts for downstream plotting/reporting.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Import paths for common and local modules
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(script_dir))

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_participants_with_expertise, load_roi_metadata
from common.neuro_utils import get_roi_names_and_colors
from common.report_utils import write_group_stats_outputs
from common.io_utils import resolve_latest_dir

from modules.mvpa_io import (
    find_subject_tsvs,
    load_subject_tsv,
    build_group_dataframe,
)
from modules.mvpa_group import compute_group_stats_for_method


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Select which MVPA directory under BIDS/derivatives/mvpa to read from.
# If None, auto-pick latest Glasser-22 MVPA directory.
MVPA_DIR_NAME = None

# Methods to analyze (subfolders within the MVPA results folder)
METHODS = ["svm", "rsa_corr"]

SVM_CHANCE = CONFIG['MVPA_SVM_CHANCE_DEFAULTS']

RSA_CHANCE = 0.0  # correlation at chance


# -----------------------------------------------------------------------------
# Helper: resolve MVPA base directory under BIDS derivatives
# -----------------------------------------------------------------------------
def resolve_mvpa_dir() -> Path:
    base = CONFIG["BIDS_MVPA"]
    if MVPA_DIR_NAME is not None:
        return Path(base) / MVPA_DIR_NAME
    # Select the most recent Glasser-22 cortices run (no fallbacks)
    return resolve_latest_dir(base, pattern="*_glasser_cortices_bilateral")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

# Set up analysis outputs (timestamped under chess-mvpa/results)
config, out_dir, logger = setup_analysis(
    analysis_name="mvpa_group",
    results_base=script_dir / "results",
    script_file=__file__,
)

mvpa_dir = resolve_mvpa_dir()
logger.info(f"Using MVPA source: {mvpa_dir}")

# Load participants and ROI metadata
participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG["BIDS_PARTICIPANTS"], bids_root=CONFIG["BIDS_ROOT"]
)
logger.info(f"Participants: {n_exp} experts, {n_nov} novices")

roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])  # standardized: roi_id, roi_name, family, color
default_roi_names, _ = get_roi_names_and_colors(CONFIG["ROI_GLASSER_22"])  # names/colors helper

artifact_index = {}

for method in METHODS:
    method_dir = mvpa_dir / method
    if not method_dir.exists():
        logger.warning(f"Missing method directory: {method_dir}; skipping")
        continue

    # Discover and load subject TSVs
    files = find_subject_tsvs(method_dir)
    logger.info(f"[{method}] Found {len(files)} subject TSVs")

    df = build_group_dataframe(files, participants, default_roi_names)
    # df columns: subject, expert(bool), target, <ROI 1> ... <ROI N>
    # Determine ROI columns from DataFrame to handle passthrough cases
    roi_names = [c for c in df.columns if c not in ["subject", "expert", "target"]]

    # Build chance mapping per method
    if method == 'svm':
        chance_map = {t: float(SVM_CHANCE.get(t, np.nan)) for t in sorted(df['target'].unique())}
    else:
        chance_map = {t: RSA_CHANCE for t in sorted(df['target'].unique())}

    method_results = compute_group_stats_for_method(
        df_method=df,
        roi_names=roi_names,
        method=method,
        chance_map=chance_map,
        alpha=CONFIG['ALPHA_FDR'],
    )

    # Save CSVs per target
    for tgt, blocks in method_results.items():
        write_group_stats_outputs(out_dir, method, tgt, blocks)

    artifact_index[method] = method_results

# Save combined artifacts
with open(out_dir / "mvpa_group_stats.pkl", "wb") as f:
    pickle.dump(artifact_index, f)

logger.info("Saved group statistics artifacts")
log_script_end(logger)
logger.info(f"All outputs saved to: {out_dir}")
