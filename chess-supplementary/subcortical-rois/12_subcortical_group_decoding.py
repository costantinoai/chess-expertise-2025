"""
Subcortical Group Analysis - Decoding (ROI-level, CAB-NP Atlas)
===============================================================

Mirrors chess-mvpa/03_mvpa_group_decoding.py for subcortical ROIs.
Pickle is used for compatibility with the existing cortical pipeline
plotting infrastructure.

METHODS
=======

Data
----
Subject-level SVM decoding accuracies from
BIDS/derivatives/fmriprep_spm-unsmoothed_decoding-subcortical/.
Participants: N=40 (20 experts, 20 novices). ROIs: 9 bilateral subcortical.

Statistical Testing
-------------------
1. Experts vs Chance: one-sample one-tailed t-test (greater).
2. Novices vs Chance: one-sample one-tailed t-test (greater).
3. Experts vs Novices: Welch two-sample two-tailed t-test.
FDR correction (BH, alpha=0.05) across 9 ROIs.

Outputs
-------
- ttest_svm_{target}_{comparison}.csv
- subcortical_group_stats.pkl (svm block)
"""

from pathlib import Path
import pickle
import numpy as np


from common import CONFIG, setup_or_reuse_analysis_dir, log_script_end
from common.bids_utils import (
    get_participants_with_expertise,
    load_roi_metadata,
    load_stimulus_metadata,
    derive_target_chance_from_stimuli,
)
from common.neuro_utils import get_roi_names_and_colors
from common.report_utils import write_group_stats_outputs

from analyses.mvpa.io import find_subject_tsvs, build_group_dataframe
from analyses.mvpa.group import (
    compute_per_roi_group_comparison,
    compute_per_roi_vs_chance_tests,
    split_data_by_target_and_group,
)

# =============================================================================
# 1. SETUP
# =============================================================================

results_dir, logger, _ = setup_or_reuse_analysis_dir(
    __file__, analysis_name="subcortical_rois"
)

# =============================================================================
# 2. LOAD DATA
# =============================================================================

decoding_dir = CONFIG["BIDS_MVPA_DECODING_SUBCORTICAL"]
if not decoding_dir.exists():
    raise FileNotFoundError(f"Missing subcortical decoding directory: {decoding_dir}")
logger.info(f"Using subcortical decoding source: {decoding_dir}")

participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG["BIDS_PARTICIPANTS"], bids_root=CONFIG["BIDS_ROOT"]
)
logger.info(f"Participants: {n_exp} experts, {n_nov} novices")

roi_info = load_roi_metadata(CONFIG["ROI_CABNP"])
default_roi_names, _ = get_roi_names_and_colors(CONFIG["ROI_CABNP"])

# =============================================================================
# 3. BUILD GROUP DATAFRAME
# =============================================================================

files = find_subject_tsvs(decoding_dir)
logger.info(f"Found {len(files)} subject TSVs in {decoding_dir.name}")

df = build_group_dataframe(files, participants, default_roi_names)
roi_names = [c for c in df.columns if c not in ["participant_id", "expert", "target"]]

stim = load_stimulus_metadata(return_all=True)
targets = sorted(df['target'].dropna().unique())
chance_map = derive_target_chance_from_stimuli(targets, stimuli_df=stim)
logger.info(f"Targets and chance levels: {chance_map}")

# =============================================================================
# 4. STATISTICAL TESTS PER TARGET
# =============================================================================

method_results = {}

for tgt in targets:
    logger.info(f"  Processing target: {tgt}")
    expert_data, novice_data = split_data_by_target_and_group(df, tgt, roi_names)
    chance_level = float(chance_map.get(tgt, np.nan))
    logger.info(f"    Experts: {expert_data.shape}, Novices: {novice_data.shape}, chance: {chance_level}")

    group_comparison = compute_per_roi_group_comparison(
        expert_data=expert_data, novice_data=novice_data,
        roi_names=roi_names, alpha=CONFIG['ALPHA_FDR'], confidence_level=0.95,
    )

    expert_vs_chance, novice_vs_chance = compute_per_roi_vs_chance_tests(
        expert_data=expert_data, novice_data=novice_data,
        roi_names=roi_names, chance_level=chance_level,
        alpha=CONFIG['ALPHA_FDR'], alternative='greater', confidence_level=0.95,
    )

    method_results[tgt] = {
        'welch_expert_vs_novice': group_comparison['test_results'],
        'experts_vs_chance': expert_vs_chance['test_results'],
        'novices_vs_chance': novice_vs_chance['test_results'],
        'chance': chance_level,
        'experts_desc': group_comparison['expert_desc'],
        'novices_desc': group_comparison['novice_desc'],
    }

    welch_df = group_comparison['test_results']
    sig = welch_df[welch_df['p_val_fdr'] < CONFIG['ALPHA_FDR']]['ROI_Name'].tolist()
    logger.info(f"    FDR-significant (exp vs nov): {sig if sig else 'none'}")

# =============================================================================
# 5. SAVE RESULTS
# =============================================================================

for tgt, blocks in method_results.items():
    write_group_stats_outputs(results_dir, "svm", tgt, blocks)

# Save the decoding half into its own file so the RSA and decoding
# group stages do not collide under the unified results/ tree. The
# matching RSA-stage file is produced by 02_subcortical_group_rsa.py and
# both halves are merged by analyses.mvpa.io.load_mvpa_group_stats.
artifact_index_path = results_dir / "subcortical_group_stats_svm.pkl"
with open(artifact_index_path, "wb") as f:
    pickle.dump({"svm": method_results}, f)

logger.info("Saved group statistics artifacts (subcortical decoding)")
log_script_end(logger)
