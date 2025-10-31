"""
Participation Ratio (PR) Manifold Analysis

METHODS
=======

Rationale
---------
Neural population activity can be conceptualized as trajectories in a high-
dimensional state space. The participation ratio (PR) quantifies the effective
dimensionality of these representations—how many dimensions are actively used
versus how concentrated activity is along a few dominant axes. We hypothesized
that chess expertise alters the dimensionality of neural representations in
task-relevant brain regions.

Data
----
Trial-wise beta estimates were extracted from unsmoothed first-level GLMs for
each of 40 participants (20 experts, 20 novices) across 40 chess stimuli (20
check, 20 non-check). Beta values were extracted from 22 bilateral cortical
regions (Glasser multimodal parcellation). Each ROI's beta matrix has shape
(n_stimuli, n_voxels).

Participation Ratio Computation
--------------------------------
For each participant and each ROI, we computed the participation ratio (PR)
from the beta matrix B (40 stimuli × n_voxels):

1. Center B by subtracting the mean across stimuli for each voxel.
2. Compute the covariance matrix C = B^T B / n_stimuli (voxels × voxels).
3. Perform eigenvalue decomposition on C to obtain eigenvalues λ_i.
4. Compute PR using the formula:

   PR = (Σ λ_i)^2 / Σ (λ_i^2)

PR ranges from 1 (activity concentrated along one dimension) to n_voxels
(activity uniformly distributed across all dimensions). Higher PR indicates
more distributed, higher-dimensional representations.

Group-Level Statistical Testing
--------------------------------
PR values were grouped by expertise (experts vs novices) for each ROI. Three
statistical tests were conducted per ROI:

1. **Welch two-sample t-test**: Comparing expert and novice mean PR values.
   Null hypothesis: μ_expert = μ_novice.
   Implementation: scipy.stats.ttest_ind with equal_var=False.

2. **False Discovery Rate (FDR) correction**: Applied across 22 ROIs using the
   Benjamini-Hochberg procedure (alpha=0.05).
   Implementation: statsmodels.stats.multitest.multipletests with
   method='fdr_bh'.

3. **Effect size**: Cohen's d was computed as (mean_expert − mean_novice) /
   pooled_std.

Classification Analysis
-----------------------
To assess whether PR profiles distinguish experts from novices, we trained a
logistic regression classifier on the 22-dimensional PR feature space (one
feature per ROI). Classification was performed using leave-one-out cross-
validation (LOOCV) to maximize training data while providing unbiased accuracy
estimates.

**Permutation test for significance**: To test whether classification accuracy
exceeded chance, we performed a permutation test with 10,000 iterations. In
each iteration, group labels were randomly shuffled, and LOOCV accuracy was
recomputed. The p-value was computed as the proportion of permuted accuracies
exceeding the observed accuracy.

Dimensionality Reduction and Visualization
-------------------------------------------
To visualize expertise differences in PR space, we performed principal
component analysis (PCA) on the standardized 22-dimensional PR features. The
first two principal components (PC1, PC2) captured the largest sources of
variance in PR profiles. PCA allows 2D visualization while preserving as much
variance as possible.

A logistic regression decision boundary was fitted in the 2D PCA space to
visualize the linear separability of expert and novice PR profiles.

Statistical Assumptions and Limitations
----------------------------------------
- **Normality**: t-tests assume normally distributed PR values within each
  group and ROI. With n=20 per group, the central limit theorem provides
  robustness.
- **Independence**: PR values are assumed independent across participants but
  may share common noise sources (scanner drift, task strategies).
- **ROI size**: PR is sensitive to the number of voxels in each ROI. We
  computed correlations between PR and ROI size to assess this confound.
- **Dimensionality interpretation**: PR quantifies spread across dimensions
  but does not identify which dimensions are functionally meaningful.

Outputs
-------
All results are saved to results/<timestamp>_manifold/:
- pr_results.pkl: Complete results dictionary (for plotting scripts)
- pr_long_format.csv: Subject-level PR values (long format)
- pr_summary_stats.csv: Group means, CIs, SEMs per ROI
- pr_statistical_tests.csv: Welch t-tests, FDR-corrected q-values, Cohen's d
- pr_classification_tests.csv: Classification accuracy, permutation p-values
- 01_manifold_analysis.py: Copy of this script
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
script_dir = Path(__file__).parent

import pickle
import pandas as pd

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_subject_list, get_group_summary
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
    evaluate_classification_significance,
)
from modules.analysis import (
    summarize_pr_by_group,
    compare_groups_welch_fdr,
)
from modules.pr_computation import compute_subject_roi_prs
from modules.models import build_feature_matrix
from common.group_stats import get_descriptives_per_roi

# =============================================================================
# Configuration
# =============================================================================

ATLAS_PATH = CONFIG["ROI_GLASSER_22_ATLAS"]
ROI_INFO_PATH = CONFIG["ROI_GLASSER_22"] / "region_info.tsv"
GLM_BASE_PATH = CONFIG["BIDS_GLM_UNSMOOTHED"]
PARTICIPANTS_PATH = CONFIG["BIDS_PARTICIPANTS"]
ALPHA = CONFIG["ALPHA"]

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

# Load the Glasser atlas (3D volume with integer labels per voxel), ROI metadata
# (names, hemisphere assignments), and participant information (IDs, expertise labels).
# The atlas defines 22 bilateral cortical regions selected for chess-related processing.
atlas_data, roi_labels, roi_info, participants = load_atlas_and_metadata(
    atlas_path=ATLAS_PATH,
    roi_info_path=ROI_INFO_PATH,
    participants_path=PARTICIPANTS_PATH,
    load_atlas_func=load_atlas,
)

all_subjects = get_subject_list()
group_summary = get_group_summary()

# =============================================================================
# Compute Participation Ratios
# =============================================================================

# For each subject and each ROI, compute the participation ratio (PR) from trial-wise
# beta estimates. PR quantifies effective dimensionality: how many dimensions of the
# neural state space are actively used to represent the 40 stimuli. High PR = distributed
# representation across many dimensions; low PR = concentrated along few dimensions.
#
# Algorithm:
# 1. Load beta matrix (40 stimuli × n_voxels) for each ROI from unsmoothed GLM
# 2. Center by subtracting mean across stimuli
# 3. Compute covariance matrix C = B^T B / n_stimuli
# 4. Eigenvalue decomposition of C
# 5. PR = (Σλ)² / Σ(λ²), where λ are eigenvalues
#
# PR ranges from 1 (one dominant dimension) to n_voxels (uniform distribution).
logger.info(
    f"Starting PR computation for {len(all_subjects)} subjects, {len(roi_labels)} ROIs"
)
records = []
for subject_id in all_subjects:
    pr_values, voxel_counts = compute_subject_roi_prs(
        subject_id=subject_id,
        atlas_data=atlas_data,
        roi_labels=roi_labels,
        base_path=GLM_BASE_PATH,
    )
    # Store as long-format table: one row per subject-ROI combination
    for roi_idx, roi_label in enumerate(roi_labels):
        records.append(
            {
                "subject_id": subject_id,
                "ROI_Label": int(roi_label),
                "PR": pr_values[roi_idx],
                "n_voxels": voxel_counts[roi_idx],
            }
        )

pr_df = pd.DataFrame(records)

n_valid = pr_df["PR"].notna().sum()
n_total = len(pr_df)

# =============================================================================
# Summary Statistics at the group level
# =============================================================================

# Compute descriptive statistics per ROI per group: mean PR, standard error,
# and 95% confidence intervals. These summary statistics characterize the
# typical dimensionality within each expertise group.
summary_stats = summarize_pr_by_group(
    pr_df=pr_df,
    participants_df=participants,
    roi_labels=roi_labels,
    confidence_level=0.95,
)

summary_stats.to_csv(output_dir / "pr_summary_stats.csv", index=False)

# =============================================================================
# Statistical Tests Experts vs Novices
# =============================================================================

# Test whether PR differs between experts and novices in each ROI using Welch's
# two-sample t-test (allows unequal variances). Apply Benjamini-Hochberg FDR
# correction across the 22 ROIs to control for multiple comparisons.
# Hypothesis: experts and novices differ in representational dimensionality.
stats_results = compare_groups_welch_fdr(
    pr_df=pr_df, participants_df=participants, roi_labels=roi_labels, alpha=ALPHA
)

# Identify and log which ROIs show significant expertise differences after FDR correction
sig_fdr = stats_results["significant_fdr"].sum()
if sig_fdr > 0:
    sig_rois = stats_results[stats_results["significant_fdr"]].merge(
        roi_info[["roi_id", "roi_name"]],
        left_on="ROI_Label",
        right_on="roi_id",
        how="left",
    )

stats_results.to_csv(output_dir / "pr_statistical_tests.csv", index=False)

# =============================================================================
# Train Classifier in PR Space (Expert vs Novice)
# =============================================================================

# Train a logistic regression classifier on the 22-dimensional PR feature space
# (one PR value per ROI per participant). This tests whether PR profiles can
# distinguish experts from novices. Features are standardized (z-scored) before
# training. This provides feature importance (classifier coefficients) indicating
# which ROIs contribute most to expertise classification.
clf, scaler, all_pr_scaled, labels = train_logreg_on_pr(
    pr_df=pr_df,
    participants=participants,
    roi_labels=roi_labels,
    random_seed=CONFIG["RANDOM_SEED"],
)

# =============================================================================
# PCA Embedding (2D)
# =============================================================================

# Reduce the 22-dimensional PR space to 2D using principal component analysis (PCA)
# for visualization. PC1 and PC2 capture the largest sources of variance in PR profiles.
# This allows visual inspection of expert/novice clustering and identification of
# ROIs contributing most to each PC (via component loadings).
pca2d, coords2d, explained2d = compute_pca_2d(
    data_scaled=all_pr_scaled, n_components=2, random_seed=CONFIG["RANDOM_SEED"]
)

# =============================================================================
# Decision Boundary of Classification (2D)
# =============================================================================

# Fit a logistic regression decision boundary in 2D PCA space to visualize
# linear separability. The boundary is computed on a dense grid spanning the
# 2D space, providing probability values at each point. This allows visualization
# of classification confidence regions.
xx, yy, Z = compute_2d_decision_boundary(
    coords_2d=coords2d, labels=labels, random_seed=CONFIG["RANDOM_SEED"]
)

# =============================================================================
# Classification Significance Tests (ROI space and PCA-2D)
# =============================================================================

# Test whether classification accuracy exceeds chance using permutation tests.
# We perform leave-one-out cross-validation (LOOCV) to estimate true accuracy,
# then shuffle group labels 10,000 times to build a null distribution. P-value =
# proportion of permuted accuracies ≥ observed accuracy. This tests whether
# PR profiles genuinely distinguish expertise or if accuracy is spurious.
#
# Test in two spaces:
# 1. Full 22-dimensional ROI space (all PR features)
# 2. 2D PCA space (testing if even low-dimensional projection is informative)
logger.info("Running significance tests for classification accuracy...")

cls_test_roi = evaluate_classification_significance(
    pr_df=pr_df,
    participants=participants,
    roi_labels=roi_labels,
    space="roi",
    random_seed=CONFIG["RANDOM_SEED"],
    n_splits=None,
    n_permutations=10000,
)

cls_test_pca2d = evaluate_classification_significance(
    pr_df=pr_df,
    participants=participants,
    roi_labels=roi_labels,
    space="pca2d",
    random_seed=CONFIG["RANDOM_SEED"],
    n_splits=None,
    n_permutations=10000,
)

# Save a compact CSV summary
cls_summary_df = pd.DataFrame(
    [
        {
            "space": "roi",
            "cv_accuracy_mean": cls_test_roi["cv_accuracy_mean"],
            "cv_accuracy_std": cls_test_roi["cv_accuracy_std"],
            "n_splits": cls_test_roi["n_splits"],
            "n_subjects": cls_test_roi["n_subjects"],
            "n_experts": cls_test_roi["n_experts"],
            "n_novices": cls_test_roi["n_novices"],
            "perm_pvalue": cls_test_roi["perm_pvalue"],
            "perm_null_mean": cls_test_roi["perm_null_mean"],
            "perm_null_std": cls_test_roi["perm_null_std"],
            "n_permutations": cls_test_roi["n_permutations"],
        },
        {
            "space": "pca2d",
            "cv_accuracy_mean": cls_test_pca2d["cv_accuracy_mean"],
            "cv_accuracy_std": cls_test_pca2d["cv_accuracy_std"],
            "n_splits": cls_test_pca2d["n_splits"],
            "n_subjects": cls_test_pca2d["n_subjects"],
            "n_experts": cls_test_pca2d["n_experts"],
            "n_novices": cls_test_pca2d["n_novices"],
            "perm_pvalue": cls_test_pca2d["perm_pvalue"],
            "perm_null_mean": cls_test_pca2d["perm_null_mean"],
            "perm_null_std": cls_test_pca2d["perm_null_std"],
            "n_permutations": cls_test_pca2d["n_permutations"],
        },
    ]
)
cls_summary_df.to_csv(output_dir / "pr_classification_tests.csv", index=False)

# =============================================================================
# Save data for visualizations
# =============================================================================

# Reshape PR data for heatmap
pr_matrix, n_experts = pivot_pr_long_to_subject_roi(
    pr_df=pr_df, participants=participants, roi_labels=roi_labels
)

# Compute PR vs voxel correlations
group_avg, diff_data, stats_vox = correlate_pr_with_roi_size(
    pr_df=pr_df, participants=participants, roi_info=roi_info
)

# =============================================================================
# Save Results
# =============================================================================

# Compute per-ROI descriptive tuples (mean, CI) for experts/novices using shared helper
X_all, y_labels, n_exp_ct, n_nov_ct = build_feature_matrix(
    pr_df=pr_df, participants=participants, roi_labels=roi_labels
)
expert_vals = X_all[:n_exp_ct, :]
novice_vals = X_all[n_exp_ct:, :]
experts_desc = get_descriptives_per_roi(expert_vals, confidence_level=0.95)
novices_desc = get_descriptives_per_roi(novice_vals, confidence_level=0.95)

results = {
    "pr_long_format": pr_df,
    "roi_info": roi_info,
    "participants": participants,
    "roi_labels": roi_labels,
    "summary_stats": summary_stats,
    "stats_results": stats_results,
    "experts_desc": experts_desc,
    "novices_desc": novices_desc,
    "classifier": clf,
    "scaler": scaler,
    "pca2d": {
        "coords": coords2d,
        "explained": explained2d,
        "labels": labels,
        "boundary": {"xx": xx, "yy": yy, "Z": Z},
        "components": pca2d.components_,
    },
    "pr_matrix": {
        "matrix": pr_matrix,
        "n_experts": int(n_experts),
    },
    "voxel_corr": {
        "group_avg": group_avg,
        "diff_data": diff_data,
        "stats": stats_vox,
    },
    "classification_tests": {
        "roi": cls_test_roi,
        "pca2d": cls_test_pca2d,
    },
    "config": {
        "atlas_path": str(ATLAS_PATH),
        "glm_path": str(GLM_BASE_PATH),
        "alpha": ALPHA,
        "n_experts": int(group_summary["n_expert"]),
        "n_novices": int(group_summary["n_novice"]),
        "n_rois": len(roi_labels),
    },
}

with open(output_dir / "pr_results.pkl", "wb") as f:
    pickle.dump(results, f)

log_script_end(logger)
