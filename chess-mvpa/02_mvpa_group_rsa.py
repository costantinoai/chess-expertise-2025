#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVPA Group Analysis — RSA Correlations (ROI-level)

Loads subject-level RSA correlation artifacts from `BIDS/derivatives/mvpa/rsa_corr/`
and computes group statistics per ROI:

- Experts vs chance (one-sample vs 0)
- Novices vs chance (one-sample vs 0)
- Experts vs novices (Welch t-test, two-tailed) with FDR correction

Analysis-only: saves structured artifacts for downstream plotting.
"""

import os
import sys
from pathlib import Path
import pickle

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = Path(__file__).parent

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import (
    get_participants_with_expertise,
    load_roi_metadata,
)
from common.neuro_utils import get_roi_names_and_colors
from common.report_utils import write_group_stats_outputs
from common.io_utils import resolve_latest_dir

from modules.mvpa_io import (
    find_subject_tsvs,
    build_group_dataframe,
)
from modules.mvpa_group import compute_group_stats_for_method


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MVPA_DIR_NAME = None  # If None, auto-pick latest Glasser-22 run


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

config, out_dir, logger = setup_analysis(
    analysis_name="mvpa_group_rsa",
    results_base=script_dir / "results",
    script_file=__file__,
)

mvpa_dir = resolve_latest_dir(CONFIG["BIDS_MVPA"], pattern="*_glasser_cortices_bilateral", specific_name=MVPA_DIR_NAME)
logger.info(f"Using MVPA source: {mvpa_dir}")

participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG["BIDS_PARTICIPANTS"], bids_root=CONFIG["BIDS_ROOT"]
)
logger.info(f"Participants: {n_exp} experts, {n_nov} novices")

roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])  # standardized
default_roi_names, _ = get_roi_names_and_colors(CONFIG["ROI_GLASSER_22"])  # helper

artifact_index = {}

method = "rsa_corr"
method_dir = mvpa_dir / method
if not method_dir.exists():
    raise FileNotFoundError(f"Missing method directory: {method_dir}")

files = find_subject_tsvs(method_dir)
logger.info(f"[{method}] Found {len(files)} subject TSVs")

df = build_group_dataframe(files, participants, default_roi_names)
roi_names = [c for c in df.columns if c not in ["subject", "expert", "target"]]

chance_map = {t: float(CONFIG.get('CHANCE_LEVEL_RSA', 0.0)) for t in sorted(df['target'].unique())}

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

with open(out_dir / "mvpa_group_stats.pkl", "wb") as f:
    pickle.dump(artifact_index, f)

logger.info("Saved group statistics artifacts (RSA)")
log_script_end(logger)
logger.info(f"All outputs saved to: {out_dir}")
