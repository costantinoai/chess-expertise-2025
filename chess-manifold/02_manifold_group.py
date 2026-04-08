#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manifold -- group stage
=======================

Reads per-subject participation-ratio (PR) values from the
``fmriprep_spm-unsmoothed_manifold`` BIDS derivative (produced by
``01_manifold_subject.py``), runs all group-level statistics and
classification analyses, and writes outputs to

    results/manifold/data/

Outputs (unchanged from the pre-refactor monolith for regression diff):

    pr_summary_stats.csv           # group means, CIs, SEMs per ROI
    pr_statistical_tests.csv       # Welch t-tests + FDR + Cohen's d
    pr_classification_tests.csv    # ROI + PCA 2D permutation tests
    pr_results.pkl                 # complete results dictionary
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_subject_list, get_group_summary
from common.neuro_utils import load_atlas
from common.group_stats import get_descriptives_per_roi

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
    build_feature_matrix,
)
from analyses.manifold.analysis import (
    summarize_pr_by_group,
    compare_groups_welch_fdr,
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
config, _, logger = setup_analysis(
    analysis_name="02_manifold_group",
    results_base=CONFIG["RESULTS_ROOT"] / "manifold" / "logs",
    script_file=__file__,
)

MANIFOLD_ROOT: Path = CONFIG["BIDS_MANIFOLD"]
OUTPUT_DIR: Path = CONFIG["RESULTS_ROOT"] / "manifold" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_FILE_SUFFIX = (
    "_space-MNI152NLin2009cAsym_roi-glasser_desc-pr_values.tsv"
)


# ---------------------------------------------------------------------------
# Load per-subject PR values
# ---------------------------------------------------------------------------
logger.info("Loading atlas and participant metadata...")
_, roi_labels, roi_info, participants = load_atlas_and_metadata(
    atlas_path=CONFIG["ROI_GLASSER_22_ATLAS"],
    roi_info_path=CONFIG["ROI_GLASSER_22"] / "region_info.tsv",
    participants_path=CONFIG["BIDS_PARTICIPANTS"],
    load_atlas_func=load_atlas,
)
all_subjects = get_subject_list()
group_summary = get_group_summary()

logger.info(f"Reading per-subject PR values from {MANIFOLD_ROOT}...")
records: list[dict] = []
missing: list[str] = []
for subject_id in all_subjects:
    path = MANIFOLD_ROOT / subject_id / f"{subject_id}{SUBJECT_FILE_SUFFIX}"
    if not path.is_file():
        logger.warning(f"  {subject_id}: no TSV at {path} -- skipping")
        missing.append(subject_id)
        continue
    df = pd.read_csv(path, sep="\t")
    for _, row in df.iterrows():
        records.append(
            {
                "subject_id": subject_id,
                "ROI_Label": int(row["ROI_Label"]),
                "PR": float(row["PR"]),
                "n_voxels": int(row["n_voxels"]),
            }
        )

pr_df = pd.DataFrame(records)
logger.info(
    f"Loaded PR values for {pr_df['subject_id'].nunique()} subjects x "
    f"{pr_df['ROI_Label'].nunique()} ROIs ({len(missing)} subjects missing)."
)


# ---------------------------------------------------------------------------
# Group summary statistics
# ---------------------------------------------------------------------------
summary_stats = summarize_pr_by_group(
    pr_df=pr_df,
    participants_df=participants,
    roi_labels=roi_labels,
    confidence_level=0.95,
)
summary_stats.to_csv(OUTPUT_DIR / "pr_summary_stats.csv", index=False)
logger.info("  Saved pr_summary_stats.csv")


# ---------------------------------------------------------------------------
# Welch t-tests + FDR
# ---------------------------------------------------------------------------
stats_results = compare_groups_welch_fdr(
    pr_df=pr_df,
    participants_df=participants,
    roi_labels=roi_labels,
    alpha=CONFIG["ALPHA"],
)
stats_results.to_csv(OUTPUT_DIR / "pr_statistical_tests.csv", index=False)
logger.info("  Saved pr_statistical_tests.csv")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
clf, scaler, all_pr_scaled, labels = train_logreg_on_pr(
    pr_df=pr_df,
    participants=participants,
    roi_labels=roi_labels,
    random_seed=CONFIG["RANDOM_SEED"],
)

pca2d, coords2d, explained2d = compute_pca_2d(
    data_scaled=all_pr_scaled, n_components=2, random_seed=CONFIG["RANDOM_SEED"]
)
xx, yy, Z = compute_2d_decision_boundary(
    coords_2d=coords2d, labels=labels, random_seed=CONFIG["RANDOM_SEED"]
)


# ---------------------------------------------------------------------------
# Classification significance (permutation)
# ---------------------------------------------------------------------------
logger.info("Running permutation tests for classification accuracy...")
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
cls_summary_df.to_csv(OUTPUT_DIR / "pr_classification_tests.csv", index=False)
logger.info("  Saved pr_classification_tests.csv")


# ---------------------------------------------------------------------------
# PR heatmap and voxel-correlation
# ---------------------------------------------------------------------------
pr_matrix, n_experts_mat = pivot_pr_long_to_subject_roi(
    pr_df=pr_df, participants=participants, roi_labels=roi_labels
)
group_avg, diff_data, stats_vox = correlate_pr_with_roi_size(
    pr_df=pr_df, participants=participants, roi_info=roi_info
)


# ---------------------------------------------------------------------------
# Descriptive per-ROI tuples
# ---------------------------------------------------------------------------
X_all, y_labels, n_exp_ct, n_nov_ct = build_feature_matrix(
    pr_df=pr_df, participants=participants, roi_labels=roi_labels
)
expert_vals = X_all[:n_exp_ct, :]
novice_vals = X_all[n_exp_ct:, :]
experts_desc = get_descriptives_per_roi(expert_vals, confidence_level=0.95)
novices_desc = get_descriptives_per_roi(novice_vals, confidence_level=0.95)


# ---------------------------------------------------------------------------
# Save complete results dict
# ---------------------------------------------------------------------------
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
        "n_experts": int(n_experts_mat),
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
        "manifold_path": str(CONFIG["BIDS_MANIFOLD"]),
        "alpha": CONFIG["ALPHA"],
        "n_experts": int(group_summary["n_expert"]),
        "n_novices": int(group_summary["n_novice"]),
        "n_rois": len(roi_labels),
    },
}

with open(OUTPUT_DIR / "pr_results.pkl", "wb") as f:
    pickle.dump(results, f)
logger.info(f"  Saved pr_results.pkl under {OUTPUT_DIR}")

log_script_end(logger)
