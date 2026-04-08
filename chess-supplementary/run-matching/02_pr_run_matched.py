"""
Participation Ratio (PR) Manifold Analysis — RUN-MATCHED CONTROL

This is a verbatim copy of chess-manifold/01_manifold_subject.py +
chess-manifold/02_manifold_group.py (fused into a single script) with
ONE change: both experts and novices are capped at exactly 8 runs
(expert group mean) to verify that unequal run counts do not drive the
reported effects. See reviewer R2 Major Comment 3.

The only modification is a local version of load_spm_beta_images() that
filters out runs > MAX_RUNS before averaging betas. Everything else
(atlas, PR computation, group stats, classification) is identical.

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
feature per ROI). Classification was performed using stratified K-fold cross-
validation with up to 5 folds, limited by the smallest class size, to estimate
out-of-sample accuracy.

**Permutation test for significance**: To test whether classification accuracy
exceeded chance, we performed a permutation test with 10,000 iterations. In
each iteration, group labels were randomly shuffled, the same cross-validation
procedure was recomputed, and the p-value was computed as the proportion of
permuted accuracies exceeding the observed accuracy.

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
- **Independence**: PR values are assumed independent across participants but
  may share common noise sources (scanner drift, task strategies).
- **ROI size**: PR is sensitive to the number of voxels in each ROI. We
  computed correlations between PR and ROI size to assess this confound.
- **Dimensionality interpretation**: PR quantifies spread across dimensions
  but does not identify which dimensions are functionally meaningful.

Outputs
-------
All results are saved to results/supplementary/run-matching/data/:
- pr_results.pkl: Complete results dictionary (for plotting scripts)
- pr_long_format.csv: Subject-level PR values (long format)
- pr_summary_stats.csv: Group means, CIs, SEMs per ROI
- pr_statistical_tests.csv: Welch t-tests, FDR-corrected q-values, Cohen's d
- pr_classification_tests.csv: Classification accuracy, permutation p-values
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

script_dir = Path(__file__).parent

import pickle  # noqa: S301 — used for internal analysis artifacts only
import numpy as np
import nibabel as nib
import scipy.io as sio
import pandas as pd

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_subject_list, get_group_summary
from common.neuro_utils import load_atlas
from common.spm_utils import _normalize_subject_id, _get_spm_dir, _get_beta_filename

from analyses.manifold.data import (
    load_atlas_and_metadata,
    pivot_pr_long_to_subject_roi,
    correlate_pr_with_roi_size,
)
from analyses.manifold.models import (
    train_logreg_on_pr,
    compute_pca_2d,
    compute_2d_decision_boundary,
    evaluate_classification_significance,
)
from analyses.manifold.analysis import (
    summarize_pr_by_group,
    compare_groups_welch_fdr,
)
from analyses.manifold.pr_computation import participation_ratio
from analyses.manifold.models import build_feature_matrix
from common.group_stats import get_descriptives_per_roi

import logging
_logger = logging.getLogger(__name__)

# =============================================================================
# Run-matching configuration
# =============================================================================
#
# Expert run distribution: 8 subjects have 6 runs, 12 have 10 runs.
# Novice run distribution: 2 subjects have 6 runs, 1 has 7, 17 have 10.
#
# To equate the two groups, we cap 8 novices at 6 runs so that both groups
# have identical distributions: 8 subjects x 6 runs + 12 subjects x 10 runs
# = 168 total runs per group. Expert data is completely untouched.
#
# Which novices are capped:
#   - sub-01, sub-02: already have 6 runs (no data lost)
#   - sub-39: has 7 runs, capped to 6 (loses 1 run)
#   - sub-15, sub-17, sub-18, sub-19, sub-21: have 10 runs, capped to 6
#     (these are the 5 lowest-ID ten-run novices, selected deterministically)
#
# The remaining 12 novices (sub-25 through sub-44) keep all 10 runs.
NOVICES_CAP6 = {'sub-01', 'sub-02', 'sub-15', 'sub-17', 'sub-18', 'sub-19', 'sub-21', 'sub-39'}


# =============================================================================
# Inlined run-capped beta-loading functions
# =============================================================================
#
# These are verbatim copies of two functions from common/spm_utils.py and
# chess-manifold/modules/pr_computation.py, with ONE modification:
# load_spm_beta_images_capped() accepts a max_runs parameter and skips any
# beta image belonging to a run number > max_runs before averaging.
#
# The original functions always average ALL available runs. By filtering here,
# we reduce the number of runs contributing to the averaged beta map for
# capped subjects, exactly as if they had been scanned for fewer runs.
#
# SPM encodes run number in regressor names: "Sn(3) C12*bf(1)" means
# condition C12 from run 3. The original regex ignores the run number;
# our modified version captures it and uses it for filtering.

def load_spm_beta_images_capped(
    subject_id: str,
    glm_dir: Path,
    max_runs: Optional[int] = None,
    spm_filename: str = "SPM.mat",
) -> Dict[str, nib.Nifti1Image]:
    """Load SPM betas averaged across runs, optionally capping at max_runs."""
    glm_dir = Path(glm_dir)
    subject_id_norm = _normalize_subject_id(subject_id)
    spm_mat_path = glm_dir / subject_id_norm / "exp" / spm_filename

    if not spm_mat_path.is_file():
        raise FileNotFoundError(f"SPM.mat not found: {spm_mat_path}")

    spm_dict = sio.loadmat(spm_mat_path.as_posix(), struct_as_record=False, squeeze_me=True)
    SPM = spm_dict["SPM"]
    beta_info = SPM.Vbeta
    regressor_names = SPM.xX.name

    # Modified regex: capture run number in group 1
    pattern = re.compile(r"Sn\((\d+)\)\s+(.*?)\*bf\(1\)")

    condition_to_indices: Dict[str, list] = {}
    for i, reg_name in enumerate(regressor_names):
        m = pattern.match(reg_name)
        if m:
            run_num = int(m.group(1))
            cond = m.group(2)
            # --- Only change vs original: skip runs above cap ---
            if max_runs is not None and run_num > max_runs:
                continue
            condition_to_indices.setdefault(cond, []).append(i)

    spm_dir = _get_spm_dir(SPM, spm_mat_path.parent)
    averaged: Dict[str, nib.Nifti1Image] = {}

    for cond, idxs in condition_to_indices.items():
        if not idxs:
            continue
        sum_data: Optional[np.ndarray] = None
        affine = header = None
        for idx in idxs:
            beta_fname = _get_beta_filename(beta_info[idx])
            beta_path = spm_dir / beta_fname
            img = nib.load(beta_path.as_posix())
            data = img.get_fdata(dtype=np.float32)
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.float32)
                affine, header = img.affine, img.header
            sum_data += data
        assert sum_data is not None
        avg = sum_data / float(len(idxs))
        averaged[cond] = nib.Nifti1Image(avg, affine=affine, header=header)

    n_runs_used = max_runs if max_runs is not None else "all"
    _logger.info(f"[Subject {subject_id}] Loaded {len(averaged)} conditions (runs: {n_runs_used})")
    return averaged


def compute_subject_roi_prs_capped(
    subject_id: str,
    atlas_data: np.ndarray,
    roi_labels: np.ndarray,
    base_path: Path,
    max_runs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PR per ROI using run-capped beta loading."""
    _logger.info(f"[Subject {subject_id}] Computing PR for {len(roi_labels)} ROIs (max_runs={max_runs})")

    try:
        averaged_betas = load_spm_beta_images_capped(subject_id, Path(base_path), max_runs=max_runs)
        conditions = sorted(averaged_betas.keys())

        roi_data: Dict[int, np.ndarray] = {}
        for roi_label in roi_labels:
            mask = atlas_data == roi_label
            n_vox = int(mask.sum())
            if n_vox == 0:
                raise ValueError(f"ROI {roi_label} has 0 voxels in atlas.")
            mat = np.zeros((len(conditions), n_vox), dtype=np.float32)
            for ci, cname in enumerate(conditions):
                beta_vals = averaged_betas[cname].get_fdata()
                mat[ci, :] = beta_vals[mask]
            roi_data[int(roi_label)] = mat
    except Exception as e:
        _logger.error(f"[Subject {subject_id}] Failed to extract ROI matrices: {e}")
        return (np.full(len(roi_labels), np.nan, dtype=np.float32),
                np.zeros(len(roi_labels), dtype=int))

    pr_values = np.full(len(roi_labels), np.nan, dtype=np.float32)
    voxel_counts = np.zeros(len(roi_labels), dtype=int)

    for idx, roi_label in enumerate(roi_labels):
        roi_matrix = roi_data[int(roi_label)]
        pr_values[idx] = participation_ratio(roi_matrix)
        voxel_counts[idx] = roi_matrix.shape[1]

    _logger.info(f"[Subject {subject_id}] PR computation completed")
    return pr_values, voxel_counts

# =============================================================================
# Setup
# =============================================================================

config, output_dir, logger = setup_analysis(
    analysis_name="manifold_run_matched",
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
    atlas_path=CONFIG["ROI_GLASSER_22_ATLAS"],
    roi_info_path=CONFIG["ROI_GLASSER_22"] / "region_info.tsv",
    participants_path=CONFIG["BIDS_PARTICIPANTS"],
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
    # --- Run-matching (ONLY CHANGE vs original script) ---
    # For novices in NOVICES_CAP6, average only the first 6 runs of betas.
    # For all other subjects (all experts + 12 remaining novices), use all
    # available runs (max_runs=None passes through to the original logic).
    # This ensures both groups have matched run distributions (8x6 + 12x10).
    max_runs = 6 if subject_id in NOVICES_CAP6 else None
    pr_values, voxel_counts = compute_subject_roi_prs_capped(
        subject_id=subject_id,
        atlas_data=atlas_data,
        roi_labels=roi_labels,
        base_path=CONFIG["SPM_GLM_UNSMOOTHED"],
        max_runs=max_runs,
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
    pr_df=pr_df, participants_df=participants, roi_labels=roi_labels, alpha=CONFIG["ALPHA"]
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
# We estimate accuracy with the same stratified K-fold procedure used for the
# main classifier, then shuffle group labels 10,000 times to build a null
# distribution. P-value = proportion of permuted accuracies ≥ observed
# accuracy. This tests whether PR profiles genuinely distinguish expertise or
# if accuracy is spurious.
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
        "atlas_path": str(CONFIG["ROI_GLASSER_22_ATLAS"]),
        "glm_path": str(CONFIG["SPM_GLM_UNSMOOTHED"]),
        "alpha": CONFIG["ALPHA"],
        "n_experts": int(group_summary["n_expert"]),
        "n_novices": int(group_summary["n_novice"]),
        "n_rois": len(roi_labels),
    },
}

with open(output_dir / "pr_results.pkl", "wb") as f:
    pickle.dump(results, f)

log_script_end(logger)
