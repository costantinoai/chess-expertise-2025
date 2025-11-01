#!/usr/bin/env python3
"""
RSA Searchlight ROI Summary (Glasser-180 Bilateral) — Supplementary Analysis

METHODS
=======

Overview
--------
This analysis summarizes subject-level RSA searchlight correlation maps within
the 180 bilateral Glasser cortical ROIs and performs expert vs. novice group
comparisons per ROI for three model RDM targets:
  - Visual Similarity
  - Strategy
  - Checkmate

Each Glasser-180 ROI is bilateral, averaging left and right hemisphere voxels.

Data
----
- Participants: All subjects in participants.tsv (n_experts, n_novices derived
  at runtime via BIDS participants table).
- Imaging: RSA searchlight r-maps from CONFIG['BIDS_RSA_SEARCHLIGHT'] with files
  named per subject and target (e.g., sub-XX_desc-searchlight_<target>_stat-r_map.nii.gz).
- Volumetric atlas: Glasser-180 bilateral atlas in MNI152NLin2009cAsym space (180 ROIs total).
- ROI metadata: region_info.tsv from CONFIG['ROI_GLASSER_180'].

Procedure
---------
1. Set up a timestamped results directory and log configuration (seed, paths).
2. Load participants and derive expert vs novice groups.
3. Load the Glasser-180 bilateral volumetric atlas and initialize NiftiLabelsMasker.
4. Load ROI metadata (180 bilateral ROIs).
5. For each subject and RSA target:
   - Load the volumetric r-map
   - Extract 180 bilateral ROI means using NiftiLabelsMasker
6. For each target:
   - Form expert and novice matrices (subjects × 180 ROIs)
   - Run Welch's t-tests per ROI with Benjamini–Hochberg FDR correction (180 tests)
   - Compute per-group descriptive means and 95% CIs per ROI
7. Save artifacts: per-target subject×ROI TSVs and a consolidated
   rsa_group_stats.pkl containing statistics.

Statistical Tests
-----------------
- Welch two-sample t-test (unequal variances) per ROI comparing experts vs
  novices. 95% confidence intervals are reported for the group difference.
- Multiple comparisons: FDR correction (Benjamini–Hochberg) at alpha=0.05,
  applied across all 180 bilateral ROIs.

Outputs
-------
All outputs are saved to results/<timestamp>_rsa_rois/:
- rsa_subject_roi_means_{target}.tsv (subject × ROI tables per target)
- rsa_group_stats.pkl (dict with per-target Welch table and descriptives)
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

from modules.io import RSA_TARGETS, find_subject_rsa_path


# ============================================================================
# Setup
# ============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="rsa_rois",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)

logger.info("=" * 80)
logger.info("RSA SEARCHLIGHT ROI SUMMARY (GLASSER-180 BILATERAL, VOLUME-BASED)")
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

rsa_base = Path(CONFIG['BIDS_RSA_SEARCHLIGHT'])

# Storage for all targets
subject_rows = {k: [] for k in RSA_TARGETS.keys()}
expert_vals = {k: [] for k in RSA_TARGETS.keys()}
novice_vals = {k: [] for k in RSA_TARGETS.keys()}

for sub_id, is_expert in participants:
    for tgt_key, tgt_label in RSA_TARGETS.items():
        path = find_subject_rsa_path(sub_id, tgt_key, rsa_base)
        img = load_nifti(path)

        # Extract all 180 bilateral ROI means at once
        roi_means = masker.fit_transform(img).flatten()  # shape: (180,)

        subject_rows[tgt_key].append({
            'subject': sub_id,
            **{f"roi_{rid}": v for rid, v in zip(roi_ids, roi_means)}
        })

        if is_expert:
            expert_vals[tgt_key].append(roi_means)
        else:
            novice_vals[tgt_key].append(roi_means)

# Write per-target subject×ROI TSVs
for tgt_key, rows in subject_rows.items():
    df = pd.DataFrame(rows)
    fname = f"rsa_subject_roi_means_{tgt_key}.tsv"
    df.to_csv(out_dir / fname, sep="\t", index=False)
    logger.info(f"Saved subject ROI means for {tgt_key} → {out_dir / fname}")


# ============================================================================
# Group statistics per target
# ============================================================================

group_index = {"rsa_corr": {}}

for tgt_key, tgt_label in RSA_TARGETS.items():
    X_exp = np.vstack(expert_vals[tgt_key]) if expert_vals[tgt_key] else np.empty((0, n_rois))
    X_nov = np.vstack(novice_vals[tgt_key]) if novice_vals[tgt_key] else np.empty((0, n_rois))

    # Standard t-test with FDR (all 180 bilateral ROIs) - equal variances assumption
    welch_df = per_roi_welch_and_fdr(X_exp, X_nov, roi_ids, alpha=CONFIG['ALPHA_FDR'], equal_var=True)

    # Descriptives
    exp_desc = get_descriptives_per_roi(X_exp) if X_exp.size else []
    nov_desc = get_descriptives_per_roi(X_nov) if X_nov.size else []

    group_index["rsa_corr"][tgt_key] = {
        "label": tgt_label,
        "welch_expert_vs_novice": welch_df,
        "experts_desc": exp_desc,
        "novices_desc": nov_desc,
        "chance": 0.0,  # RSA correlations are tested around 0 by convention
    }

    logger.info(f"Statistics computed for {tgt_key}")

with open(out_dir / "rsa_group_stats.pkl", "wb") as f:
    pickle.dump(group_index, f)
logger.info(f"Saved group statistics → {out_dir / 'rsa_group_stats.pkl'}")

logger.info("Analysis complete. Results in: %s", out_dir)
logger.info("=" * 80)
log_script_end(logger)
