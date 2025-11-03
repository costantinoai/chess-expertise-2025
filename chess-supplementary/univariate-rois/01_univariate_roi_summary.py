#!/usr/bin/env python3
"""
Univariate ROI Summary (Glasser-180 Bilateral) — Supplementary Analysis

METHODS
=======

Overview
--------
This analysis summarizes subject-level univariate contrast maps within the 180
bilateral Glasser cortical ROIs and performs expert vs. novice group comparisons
per ROI. Two first-level contrasts are considered:
1) Checkmate > Non-checkmate (con_0001)
2) All > Rest (con_0002)

Each Glasser-180 ROI is bilateral, averaging left and right hemisphere voxels.

Data
----
- Participants: All subjects in participants.tsv (n_experts, n_novices derived
  at runtime via BIDS participants table).
- Imaging: First-level SPM contrasts from CONFIG['SPM_GLM_DIR']/smooth4.
- Volumetric atlas: Glasser-180 bilateral atlas in MNI152NLin2009cAsym space (180 ROIs total).
- ROI metadata: region_info.tsv from CONFIG['ROI_GLASSER_180'].

Procedure
---------
1. Set up a timestamped results directory and log configuration (seed, paths).
2. Load participants and derive expert vs novice groups.
3. Load the Glasser-180 bilateral volumetric atlas and initialize NiftiLabelsMasker.
4. Load ROI metadata (180 bilateral ROIs).
5. For each subject and contrast:
   - Load the volumetric contrast image
   - Extract 180 bilateral ROI means using NiftiLabelsMasker
6. For each contrast:
   - Form expert and novice matrices (subjects × 180 ROIs)
   - Run Welch's t-tests per ROI with Benjamini–Hochberg FDR correction (180 tests)
   - Compute per-group descriptive means and 95% CIs per ROI
7. Save artifacts: per-contrast subject×ROI TSVs and a consolidated
   univ_group_stats.pkl containing statistics.

Statistical Tests
-----------------
- Welch two-sample t-test (unequal variances) per ROI comparing experts vs
  novices. 95% confidence intervals are reported for the group difference.
- Multiple comparisons: FDR correction (Benjamini–Hochberg) at alpha=0.05,
  applied across all 180 bilateral ROIs.

Outputs
-------
All outputs are saved to results/<timestamp>_univariate_rois/:
- univ_subject_roi_means_{contrast}.tsv (subject × ROI tables per contrast)
- univ_group_stats.pkl (dict with per-contrast Welch table and descriptives)
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Enable repo root imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_participants_with_expertise, load_roi_metadata
from common.neuro_utils import load_nifti, map_glasser_roi_to_harvard_oxford
from common.group_stats import get_descriptives_per_roi
from common.stats_utils import per_roi_welch_and_fdr
from nilearn.maskers import NiftiLabelsMasker

from modules.io import UNIV_CONTRASTS, find_subject_contrast_path


# ============================================================================
# Setup
# ============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="univariate_rois",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)

logger.info("=" * 80)
logger.info("UNIVARIATE ROI SUMMARY (GLASSER-180 BILATERAL, VOLUME-BASED)")
logger.info("=" * 80)

# Participants
participants, (n_exp, n_nov) = get_participants_with_expertise()
logger.info(f"Participants loaded: {len(participants)} (experts={n_exp}, novices={n_nov})")

# Load bilateral volumetric atlas and ROI metadata
atlas_path = CONFIG['ROI_GLASSER_180_ATLAS']
atlas_img = load_nifti(atlas_path)
masker = NiftiLabelsMasker(labels_img=str(atlas_path), standardize=False, strategy='mean')
logger.info(f"Loaded Glasser-180 bilateral atlas from: {atlas_path}")

roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_180'])
# Metadata has 360 rows (180 left + 180 right), but bilateral atlas has only 180 labels
# Use only left hemisphere rows (1-180) which correspond to bilateral atlas labels
roi_info = roi_info[roi_info['roi_id'] <= 180].copy()
roi_ids = roi_info['roi_id'].to_numpy()
n_rois = len(roi_ids)
logger.info(f"Loaded {n_rois} bilateral ROIs")

# Map each Glasser ROI to Harvard-Oxford anatomical labels via center of mass
logger.info("Mapping Glasser ROIs to Harvard-Oxford anatomical labels...")
roi_info['harvard_oxford_label'] = [
    map_glasser_roi_to_harvard_oxford(roi_id, atlas_img, threshold=0.25)
    for roi_id in roi_ids
]
logger.info("Harvard-Oxford mapping complete")


# ============================================================================
# Per-subject ROI means (volume-based extraction, bilateral)
# ============================================================================

glm_base = Path(CONFIG['SPM_GLM_SMOOTH4'])

# Storage for all contrasts
subject_rows = {k: [] for k in UNIV_CONTRASTS.keys()}
expert_vals = {k: [] for k in UNIV_CONTRASTS.keys()}
novice_vals = {k: [] for k in UNIV_CONTRASTS.keys()}

for sub_id, is_expert in participants:
    for con_code, con_label in UNIV_CONTRASTS.items():
        path = find_subject_contrast_path(sub_id, con_code, glm_base)
        img = load_nifti(path)

        # Extract all 180 bilateral ROI means at once
        roi_means = masker.fit_transform(img).flatten()  # shape: (180,)

        subject_rows[con_code].append({
            'subject': sub_id,
            **{f"roi_{rid}": v for rid, v in zip(roi_ids, roi_means)}
        })

        if is_expert:
            expert_vals[con_code].append(roi_means)
        else:
            novice_vals[con_code].append(roi_means)

# Write per-contrast subject×ROI TSVs
for con_code, rows in subject_rows.items():
    df = pd.DataFrame(rows)
    fname = f"univ_subject_roi_means_{con_code}.tsv"
    df.to_csv(out_dir / fname, sep="\t", index=False)
    logger.info(f"Saved subject ROI means for {con_code} → {out_dir / fname}")


# ============================================================================
# Group statistics per contrast
# ============================================================================

group_index = {"contrasts": {}}

for con_code, con_label in UNIV_CONTRASTS.items():
    X_exp = np.vstack(expert_vals[con_code]) if expert_vals[con_code] else np.empty((0, n_rois))
    X_nov = np.vstack(novice_vals[con_code]) if novice_vals[con_code] else np.empty((0, n_rois))

    # Standard t-test with FDR (all 180 bilateral ROIs) - equal variances assumption
    welch_df = per_roi_welch_and_fdr(X_exp, X_nov, roi_ids, alpha=CONFIG['ALPHA_FDR'], equal_var=True)

    # Descriptives
    exp_desc = get_descriptives_per_roi(X_exp) if X_exp.size else []
    nov_desc = get_descriptives_per_roi(X_nov) if X_nov.size else []

    group_index["contrasts"][con_code] = {
        "label": con_label,
        "welch_expert_vs_novice": welch_df,
        "experts_desc": exp_desc,
        "novices_desc": nov_desc,
    }

    logger.info(f"Statistics computed for {con_code}")

with open(out_dir / "univ_group_stats.pkl", "wb") as f:
    pickle.dump(group_index, f)
logger.info(f"Saved group statistics → {out_dir / 'univ_group_stats.pkl'}")

# Save ROI metadata with Harvard-Oxford labels for table generation
roi_info.to_csv(out_dir / "roi_info_with_ho_labels.tsv", sep="\t", index=False)
logger.info(f"Saved ROI metadata with Harvard-Oxford labels → {out_dir / 'roi_info_with_ho_labels.tsv'}")

logger.info("Analysis complete. Results in: %s", out_dir)
logger.info("=" * 80)
log_script_end(logger)
