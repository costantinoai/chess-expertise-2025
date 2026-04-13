#!/usr/bin/env python3
"""
RSA Searchlight ROI Group Statistics (Glasser-180 Bilateral) — Supplementary Analysis

METHODS
=======

Overview
--------
This script reads per-subject ROI mean TSVs produced by
``01_rsa_roi_subject.py`` (stored in BIDS derivatives) and performs
expert vs. novice group comparisons per ROI for three model RDM targets:
  - Visual Similarity
  - Strategy
  - Checkmate

Each Glasser-180 ROI is bilateral, averaging left and right hemisphere voxels.

Data
----
- Participants: All subjects in participants.tsv (n_experts, n_novices derived
  at runtime via BIDS participants table).
- Per-subject ROI means: TSVs from
  ``BIDS/derivatives/fmriprep_spm-unsmoothed_rsa-rois/sub-XX/`` written by
  ``01_rsa_roi_subject.py``.
- Volumetric atlas: Glasser-180 bilateral atlas in MNI152NLin2009cAsym space
  (used only for Harvard-Oxford mapping, not NIfTI extraction).
- ROI metadata: region_info.tsv from CONFIG['ROI_GLASSER_180'].

Procedure
---------
1. Set up the results directory and log configuration.
2. Load participants and derive expert vs novice groups.
3. Load the Glasser-180 bilateral volumetric atlas (for H-O mapping).
4. Load ROI metadata (180 bilateral ROIs).
5. Read per-subject ROI mean TSVs from BIDS derivatives (written by 01).
6. For each target:
   - Form expert and novice matrices (subjects x 180 ROIs)
   - Run Welch's t-tests per ROI with Benjamini-Hochberg FDR correction (180 tests)
   - Compute per-group descriptive means and 95% CIs per ROI
7. Save group-level artifacts: rsa_group_stats.pkl and roi_info TSV.

Statistical Tests
-----------------
- Welch two-sample t-test (unequal variances) per ROI comparing experts vs
  novices. 95% confidence intervals are reported for the group difference.
- Multiple comparisons: FDR correction (Benjamini-Hochberg) at alpha=0.05,
  applied across all 180 bilateral ROIs.

Outputs
-------
All outputs are saved to results/supplementary/rsa-rois/data/:
- rsa_group_stats.pkl (dict with per-target Welch table and descriptives)
- roi_info_with_ho_labels.tsv (ROI metadata with Harvard-Oxford labels)

Per-subject data lives in BIDS/derivatives/ (GDPR compliance).
"""

from pathlib import Path
import pickle  # noqa: S403 — trusted internal data only
import numpy as np
import pandas as pd

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_participants_with_expertise, get_subject_list, load_roi_metadata
from common.neuro_utils import load_nifti, map_glasser_roi_to_harvard_oxford
from common.group_stats import get_descriptives_per_roi
from common.stats_utils import per_roi_welch_and_fdr

from analyses.rsa_rois.io import RSA_TARGETS, _BIDS_DESC_FOR_TARGET


# ============================================================================
# Setup
# ============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="11_rsa_rois_group",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)

logger.info("=" * 80)
logger.info("RSA SEARCHLIGHT ROI GROUP STATISTICS (GLASSER-180 BILATERAL)")
logger.info("=" * 80)

# Participants
participants, (n_exp, n_nov) = get_participants_with_expertise()
logger.info(f"Participants loaded: {len(participants)} (experts={n_exp}, novices={n_nov})")

# Load bilateral volumetric atlas (needed for Harvard-Oxford mapping) and ROI metadata
atlas_path = CONFIG['ROI_GLASSER_180_ATLAS']
atlas_img = load_nifti(atlas_path)
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
# Read per-subject ROI means from BIDS derivatives (written by 01)
# ============================================================================

RSA_ROIS_ROOT: Path = (
    Path(CONFIG['BIDS_RSA_SEARCHLIGHT']).parent / 'fmriprep_spm-unsmoothed_rsa-rois'
)
logger.info(f"Reading per-subject ROI means from: {RSA_ROIS_ROOT}")

all_subjects = get_subject_list()
expert_set = {sub_id for sub_id, is_expert in participants if is_expert}

expert_vals = {k: [] for k in RSA_TARGETS.keys()}
novice_vals = {k: [] for k in RSA_TARGETS.keys()}
missing = []

for sub_id in all_subjects:
    is_expert = sub_id in expert_set
    for tgt_key in RSA_TARGETS.keys():
        bids_desc = _BIDS_DESC_FOR_TARGET[tgt_key]
        tsv_name = (
            f"{sub_id}_space-MNI152NLin2009cAsym"
            f"_roi-glasser180_desc-rsamean"
            f"_target-{bids_desc}_rois.tsv"
        )
        tsv_path = RSA_ROIS_ROOT / sub_id / tsv_name
        if not tsv_path.is_file():
            logger.warning(f"  {sub_id}/{tgt_key}: no TSV at {tsv_path} -- skipping")
            missing.append((sub_id, tgt_key))
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        roi_cols = [c for c in df.columns if c.startswith("roi_")]
        roi_means = df[roi_cols].values.flatten()  # shape: (180,)

        if is_expert:
            expert_vals[tgt_key].append(roi_means)
        else:
            novice_vals[tgt_key].append(roi_means)

logger.info(
    f"Loaded per-subject ROI means for {len(all_subjects)} subjects "
    f"({len(missing)} subject/target pairs missing)"
)


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

# Save ROI metadata with Harvard-Oxford labels for table generation
roi_info.to_csv(out_dir / "roi_info_with_ho_labels.tsv", sep="\t", index=False)
logger.info(f"Saved ROI metadata with Harvard-Oxford labels → {out_dir / 'roi_info_with_ho_labels.tsv'}")

logger.info("Analysis complete. Results in: %s", out_dir)
logger.info("=" * 80)
log_script_end(logger)
