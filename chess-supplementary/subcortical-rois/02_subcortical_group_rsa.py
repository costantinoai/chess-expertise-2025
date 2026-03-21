"""
Subcortical Group Analysis - RSA Correlations (ROI-level, CAB-NP Atlas)
=======================================================================

METHODS
=======

Rationale
---------
This exploratory analysis extends the cortical RSA analysis (Glasser 22-region
parcellation) to subcortical structures using the Cole-Anticevic Brain-wide
Network Partition (CAB-NP; Ji et al., 2019, NeuroImage). Subject-level RSA
correlation coefficients were computed for 9 bilateral subcortical ROIs
(hippocampus, amygdala, caudate, putamen, pallidum, thalamus, nucleus
accumbens, cerebellum, brainstem) using the same procedure as the cortical
analysis.

Data
----
Subject-level RSA correlation coefficients were computed by correlating neural
RDMs with theoretical model RDMs within each subcortical ROI. Neural RDMs were
constructed from trial-wise beta estimates within each ROI, then correlated
(Spearman) with model RDMs. Subject-level correlations are stored as TSV files
in BIDS/derivatives/mvpa-rsa-subcortical/.

Participants: N=40 (20 experts, 20 novices).
Stimuli: 40 chess board positions (20 check, 20 non-check).
ROIs: 9 bilateral subcortical regions (CAB-NP atlas).

For RSA correlations, chance level = 0 (null hypothesis: no correlation).

Group-Level Statistical Testing
--------------------------------
Identical to the cortical analysis:

1. **Experts vs Chance (zero)**: One-sample one-tailed t-test (greater).
   Null: mu_expert <= 0.

2. **Novices vs Chance (zero)**: One-sample one-tailed t-test (greater).
   Null: mu_novice <= 0.

3. **Experts vs Novices**: Welch two-sample two-tailed t-test.
   Null: mu_expert = mu_novice.

False Discovery Rate (FDR) Correction
--------------------------------------
Benjamini-Hochberg FDR correction across the 9 subcortical ROIs (not 22),
applied separately within each model target and test type. Alpha = 0.05.

Outputs
-------
- ttest_rsa_corr_{target}_experts_vs_novices.csv
- ttest_rsa_corr_{target}_experts_vs_chance.csv
- ttest_rsa_corr_{target}_novices_vs_chance.csv
- subcortical_group_stats.pkl
"""

import os
import sys
from pathlib import Path
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common import CONFIG, setup_or_reuse_analysis_dir, log_script_end
from common.bids_utils import (
    get_participants_with_expertise,
    load_roi_metadata,
)
from common.neuro_utils import get_roi_names_and_colors
from common.report_utils import write_group_stats_outputs

# Reuse the MVPA I/O and group modules (identical logic, different atlas)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'chess-mvpa')))
from modules.mvpa_io import (
    find_subject_tsvs,
    build_group_dataframe,
)
from modules.mvpa_group import (
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

# Locate subject-level subcortical RSA results
rsa_dir = CONFIG["BIDS_MVPA_RSA_SUBCORTICAL"]
if not rsa_dir.exists():
    raise FileNotFoundError(f"Missing subcortical RSA directory: {rsa_dir}")
logger.info(f"Using subcortical RSA source: {rsa_dir}")

# Load participant list with expertise labels
participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG["BIDS_PARTICIPANTS"], bids_root=CONFIG["BIDS_ROOT"]
)
logger.info(f"Participants: {n_exp} experts, {n_nov} novices")

# Load ROI metadata for CAB-NP subcortical atlas
roi_info = load_roi_metadata(CONFIG["ROI_CABNP"])
default_roi_names, _ = get_roi_names_and_colors(CONFIG["ROI_CABNP"])
logger.info(f"Subcortical ROIs ({len(default_roi_names)}): {default_roi_names}")

# =============================================================================
# 3. BUILD GROUP DATAFRAME
# =============================================================================

# Find all subject-level TSV files
files = find_subject_tsvs(rsa_dir)
logger.info(f"Found {len(files)} subject TSVs in {rsa_dir.name}")

# Load and consolidate into a single DataFrame
df = build_group_dataframe(files, participants, default_roi_names)
roi_names = [c for c in df.columns if c not in ["participant_id", "expert", "target"]]
logger.info(f"ROI columns in dataframe ({len(roi_names)}): {roi_names}")

# RSA chance level = 0 (null: no correlation)
chance_level = float(CONFIG.get('CHANCE_LEVEL_RSA', 0.0))

# =============================================================================
# 4. STATISTICAL TESTS PER TARGET
# =============================================================================

targets = sorted(df['target'].dropna().unique())
logger.info(f"Targets: {targets}")
method_results = {}

for tgt in targets:
    logger.info(f"  Processing target: {tgt}")

    # Extract expert and novice data arrays
    expert_data, novice_data = split_data_by_target_and_group(df, tgt, roi_names)
    logger.info(f"    Experts: {expert_data.shape}, Novices: {novice_data.shape}")

    # Between-group comparison: Welch's t-test with FDR
    group_comparison = compute_per_roi_group_comparison(
        expert_data=expert_data,
        novice_data=novice_data,
        roi_names=roi_names,
        alpha=CONFIG['ALPHA_FDR'],
        confidence_level=0.95,
    )

    # Within-group vs-chance tests: one-sample t-tests with FDR
    expert_vs_chance, novice_vs_chance = compute_per_roi_vs_chance_tests(
        expert_data=expert_data,
        novice_data=novice_data,
        roi_names=roi_names,
        chance_level=chance_level,
        alpha=CONFIG['ALPHA_FDR'],
        alternative='greater',
        confidence_level=0.95,
    )

    # Package results (same structure as cortical pipeline)
    method_results[tgt] = {
        'welch_expert_vs_novice': group_comparison['test_results'],
        'experts_vs_chance': expert_vs_chance['test_results'],
        'novices_vs_chance': novice_vs_chance['test_results'],
        'chance': chance_level,
        'experts_desc': group_comparison['expert_desc'],
        'novices_desc': group_comparison['novice_desc'],
    }

    # Log key results
    welch_df = group_comparison['test_results']
    sig_rois = welch_df[welch_df['p_val_fdr'] < CONFIG['ALPHA_FDR']]['ROI_Name'].tolist()
    logger.info(f"    FDR-significant ROIs (expert vs novice): {sig_rois if sig_rois else 'none'}")

# =============================================================================
# 5. SAVE RESULTS
# =============================================================================

# Save human-readable CSVs
for tgt, blocks in method_results.items():
    write_group_stats_outputs(results_dir, "rsa_corr", tgt, blocks)

# Save pickle for plotting (same format as cortical mvpa_group_stats.pkl)
# NOTE: pickle used here for compatibility with the existing cortical pipeline
# plotting infrastructure which expects this format
artifact_index_path = results_dir / "subcortical_group_stats.pkl"
if artifact_index_path.exists():
    try:
        with open(artifact_index_path, "rb") as f:
            prev = pickle.load(f)
    except Exception:
        prev = {}
else:
    prev = {}
prev["rsa_corr"] = method_results
with open(artifact_index_path, "wb") as f:
    pickle.dump(prev, f)

logger.info("Saved group statistics artifacts (subcortical RSA)")
logger.info(f"All outputs saved to: {results_dir}")
log_script_end(logger)
