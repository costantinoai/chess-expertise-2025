"""
Participation Ratio (PR) Manifold Analysis
===========================================

Analyzes the effective dimensionality of neural representations in chess
experts vs novices using the participation ratio metric.

**What is Participation Ratio?**
Quantifies how many dimensions are actively used for representation.
Higher PR = more distributed, lower PR = more specialized.

**Analysis Steps:**
1. Load atlas, ROI metadata, participants → data.load_atlas_and_metadata()
2. Compute PR for all subjects (loop)
3. Compute summary stats by group → analysis.summarize_pr_by_group()
4. Statistical tests (Welch + FDR) → analysis.compare_groups_welch_fdr()
5. Train classifier for feature importance → models.train_logreg_on_pr()
6. Compute PCA embedding (2D) → models.compute_pca_2d()
7. Compute decision boundary → models.compute_2d_decision_boundary()
8. Prepare heatmap data → data.pivot_pr_long_to_subject_roi()
9. PR vs voxel correlations → data.correlate_pr_with_roi_size()
10. Save all results

**Outputs:**
- pr_results.pkl: Complete results for plotting script
- pr_long_format.csv: Subject-level PR values
- pr_summary_stats.csv: Group means and CIs
- pr_statistical_tests.csv: Test results with FDR
- analysis.log: Execution log
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(script_dir))

import pickle
import pandas as pd

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_subject_list
from common.neuro_utils import load_atlas

from modules.data import (
    load_atlas_and_metadata,
    pivot_pr_long_to_subject_roi,
    correlate_pr_with_roi_size,
)
from modules.models import (
    train_logreg_on_pr,
    compute_pca_2d,
    compute_2d_decision_boundary,
)
from modules.analysis import (
    summarize_pr_by_group,
    compare_groups_welch_fdr,
)
from modules.pr_computation import compute_subject_roi_prs

# =============================================================================
# Configuration
# =============================================================================

ATLAS_PATH = CONFIG['ROI_GLASSER_22_ATLAS']
ROI_INFO_PATH = CONFIG['ROI_GLASSER_22'] / 'region_info.tsv'
GLM_BASE_PATH = CONFIG['BIDS_GLM_UNSMOOTHED']
PARTICIPANTS_PATH = CONFIG['BIDS_PARTICIPANTS']
ALPHA = CONFIG['ALPHA']

# =============================================================================
# Setup
# =============================================================================

config, output_dir, logger = setup_analysis(
    analysis_name="manifold",
    results_base=script_dir / "results",
    script_file=__file__,
)

# =============================================================================
# Load Data
# =============================================================================

# Load atlas and metadata
atlas_data, roi_labels, roi_info, participants = load_atlas_and_metadata(
    atlas_path=ATLAS_PATH,
    roi_info_path=ROI_INFO_PATH,
    participants_path=PARTICIPANTS_PATH,
    load_atlas_func=load_atlas
)

# Get subject lists
all_subjects = get_subject_list()
expert_subjects = get_subject_list(group='expert')
novice_subjects = get_subject_list(group='novice')

# =============================================================================
# Compute Participation Ratios
# =============================================================================

# Compute PR per subject (simple loop; clearer execution order)
logger.info(f"Starting PR computation for {len(all_subjects)} subjects, {len(roi_labels)} ROIs")
records = []
for subject_id in all_subjects:
    pr_values, voxel_counts = compute_subject_roi_prs(
        subject_id=subject_id,
        atlas_data=atlas_data,
        roi_labels=roi_labels,
        base_path=GLM_BASE_PATH,
    )
    for roi_idx, roi_label in enumerate(roi_labels):
        records.append({
            'subject_id': subject_id,
            'ROI_Label': int(roi_label),
            'PR': pr_values[roi_idx],
            'n_voxels': voxel_counts[roi_idx],
        })

pr_df = pd.DataFrame(records)

# Log summary statistics
n_valid = pr_df['PR'].notna().sum()
n_total = len(pr_df)

# =============================================================================
# Summary Statistics at the group level
# =============================================================================

# Computes mean, CI, SEM per group and ROI
summary_stats = summarize_pr_by_group(
    pr_df=pr_df,
    participants_df=participants,
    roi_labels=roi_labels,
    confidence_level=0.95
)

# Save summary stats
summary_stats.to_csv(output_dir / "pr_summary_stats.csv", index=False)

# =============================================================================
# Statistical Tests Experts vs Novices
# =============================================================================

# Runs Welch t-tests (by ROI) with FDR correction (across ROIs)
stats_results = compare_groups_welch_fdr(
    pr_df=pr_df,
    participants_df=participants,
    roi_labels=roi_labels,
    alpha=ALPHA
)

# Log significant results
sig_fdr = stats_results['significant_fdr'].sum()
if sig_fdr > 0:
    sig_rois = stats_results[stats_results['significant_fdr']].merge(
        roi_info[['ROI_idx', 'roi_name']],
        left_on='ROI_Label',
        right_on='ROI_idx',
        how='left'
    )

stats_results.to_csv(output_dir / "pr_statistical_tests.csv", index=False)

# =============================================================================
# Train Classifier in PR Space (Expert vs Novice)
# =============================================================================

# Train logistic regression
clf, scaler, all_pr_scaled, labels = train_logreg_on_pr(
    pr_df=pr_df,
    participants=participants,
    roi_labels=roi_labels,
    random_seed=CONFIG['RANDOM_SEED']
)

# =============================================================================
# PCA Embedding (2D)
# =============================================================================

# Compute PCA
pca2d, coords2d, explained2d = compute_pca_2d(
    data_scaled=all_pr_scaled,
    n_components=2,
    random_seed=CONFIG['RANDOM_SEED']
)

# =============================================================================
# Decision Boundary of Classification (2D)
# =============================================================================

# Compute boundary grid
xx, yy, Z = compute_2d_decision_boundary(
    coords_2d=coords2d,
    labels=labels,
    random_seed=CONFIG['RANDOM_SEED']
)

# =============================================================================
# Save data for visualizations
# =============================================================================

# Reshape PR data for heatmap
pr_matrix, n_experts = pivot_pr_long_to_subject_roi(
    pr_df=pr_df,
    participants=participants,
    roi_labels=roi_labels
)

# Compute PR vs voxel correlations
group_avg, diff_data, stats_vox = correlate_pr_with_roi_size(
    pr_df=pr_df,
    participants=participants,
    roi_info=roi_info
)

# =============================================================================
# Save Results
# =============================================================================

results = {
    'pr_long_format': pr_df,
    'roi_info': roi_info,
    'participants': participants,
    'roi_labels': roi_labels,
    'summary_stats': summary_stats,
    'stats_results': stats_results,
    'classifier': clf,
    'scaler': scaler,
    'pca2d': {
        'coords': coords2d,
        'explained': explained2d,
        'labels': labels,
        'boundary': {'xx': xx, 'yy': yy, 'Z': Z},
        'components': pca2d.components_,
    },
    'pr_matrix': {
        'matrix': pr_matrix,
        'n_experts': int(n_experts),
    },
    'voxel_corr': {
        'group_avg': group_avg,
        'diff_data': diff_data,
        'stats': stats_vox,
    },
    'config': {
        'atlas_path': str(ATLAS_PATH),
        'glm_path': str(GLM_BASE_PATH),
        'alpha': ALPHA,
        'n_experts': len(expert_subjects),
        'n_novices': len(novice_subjects),
        'n_rois': len(roi_labels),
    }
}

with open(output_dir / "pr_results.pkl", 'wb') as f:
    pickle.dump(results, f)

log_script_end(logger)
