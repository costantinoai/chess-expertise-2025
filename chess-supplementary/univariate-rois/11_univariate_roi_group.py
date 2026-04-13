#!/usr/bin/env python3
"""
Univariate ROI Summary -- group stage (Glasser-180 Bilateral)
=============================================================

Reads per-subject univariate ROI means from the
``fmriprep_spm-smoothed_univariate-rois`` BIDS derivative (produced by
``01_univariate_roi_subject.py``), runs expert-vs-novice group comparisons
per ROI, and writes outputs to

    results/<timestamp>_univariate_rois/

This script writes only GROUP-LEVEL aggregates to the results/ tree.
Per-subject ROI means remain in BIDS/derivatives/ (written by the
subject-level script 01_univariate_roi_subject.py).

Contrasts
---------
1) Checkmate > Non-checkmate (con_0001)
2) All > Rest (con_0002)

Each Glasser-180 ROI is bilateral, averaging left and right hemisphere voxels.

Procedure
---------
1. Set up a timestamped results directory and log configuration (seed, paths).
2. Load participants and derive expert vs novice groups.
3. Load the Glasser-180 bilateral volumetric atlas for ROI metadata.
4. Load ROI metadata (180 bilateral ROIs) and map to Harvard-Oxford labels.
5. For each subject and contrast, read per-subject ROI means from derivatives.
6. For each contrast:
   - Form expert and novice matrices (subjects x 180 ROIs)
   - Run Welch's t-tests per ROI with Benjamini-Hochberg FDR correction
   - Compute per-group descriptive means and 95% CIs per ROI
7. Save artifacts: univ_group_stats.pkl containing statistics.

Statistical Tests
-----------------
- Welch two-sample t-test (unequal variances) per ROI comparing experts vs
  novices. 95% confidence intervals are reported for the group difference.
- Multiple comparisons: FDR correction (Benjamini-Hochberg) at alpha=0.05,
  applied across all 180 bilateral ROIs.

Outputs
-------
All outputs are saved to results/<timestamp>_univariate_rois/:
- univ_group_stats.pkl (dict with per-contrast Welch table and descriptives)
- roi_info_with_ho_labels.tsv (ROI metadata with Harvard-Oxford mapping)
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Enable repo root imports
from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_participants_with_expertise, load_roi_metadata
from common.neuro_utils import load_nifti, map_glasser_roi_to_harvard_oxford
from common.group_stats import get_descriptives_per_roi
from common.stats_utils import per_roi_welch_and_fdr

from analyses.univariate_rois.io import UNIV_CONTRASTS


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
# Load per-subject ROI means from derivatives
# ============================================================================

DERIV_ROOT: Path = CONFIG["BIDS_UNIVARIATE_ROIS"]
SUBJECT_FILE_TEMPLATE = (
    "{sub}_space-MNI152NLin2009cAsym_roi-glasser180"
    "_desc-univmean_contrast-{con}_rois.tsv"
)

logger.info(f"Reading per-subject ROI means from {DERIV_ROOT}...")

expert_vals = {k: [] for k in UNIV_CONTRASTS.keys()}
novice_vals = {k: [] for k in UNIV_CONTRASTS.keys()}
missing: list[str] = []

for sub_id, is_expert in participants:
    for con_code, con_label in UNIV_CONTRASTS.items():
        tsv_name = SUBJECT_FILE_TEMPLATE.format(sub=sub_id, con=con_code)
        path = DERIV_ROOT / sub_id / tsv_name
        if not path.is_file():
            logger.warning(f"  {sub_id} {con_code}: no TSV at {path} -- skipping")
            missing.append(f"{sub_id}_{con_code}")
            continue

        df = pd.read_csv(path, sep="\t")
        roi_means = df["roi_mean"].to_numpy()  # shape: (180,)

        if is_expert:
            expert_vals[con_code].append(roi_means)
        else:
            novice_vals[con_code].append(roi_means)

n_loaded = sum(len(v) for v in expert_vals.values()) + sum(len(v) for v in novice_vals.values())
logger.info(
    f"Loaded {n_loaded} subject-contrast ROI vectors ({len(missing)} missing)."
)


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
logger.info(f"Saved group statistics -> {out_dir / 'univ_group_stats.pkl'}")

# Save ROI metadata with Harvard-Oxford labels for table generation
roi_info.to_csv(out_dir / "roi_info_with_ho_labels.tsv", sep="\t", index=False)
logger.info(f"Saved ROI metadata with Harvard-Oxford labels -> {out_dir / 'roi_info_with_ho_labels.tsv'}")

logger.info("Analysis complete. Results in: %s", out_dir)
logger.info("=" * 80)
log_script_end(logger)
