#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Univariate ROI Extraction -- per-subject stage
===============================================

For each participant and first-level contrast, extracts the mean activation
within each of the 180 bilateral Glasser cortical ROIs from the smoothed
SPM contrast image and writes a BIDS-named TSV under

    BIDS/derivatives/fmriprep_spm-smoothed_univariate-rois/sub-XX/
        sub-XX_space-MNI152NLin2009cAsym_roi-glasser180_desc-univmean_contrast-{con_code}_rois.tsv

with a root-level sidecar JSON documenting the columns.  The pipeline's
``dataset_description.json`` is created on first run.

The subject-level outputs feed into ``11_univariate_roi_group.py`` which runs
all group-level statistics (Welch t-tests, FDR correction, descriptives).

Contrasts
---------
- con_0001 : Checkmate > Non-checkmate
- con_0002 : All > Rest

Columns in the per-subject TSV
-------------------------------
- roi_id   : integer atlas label (1..180)
- roi_mean : mean contrast value across voxels in that bilateral ROI
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_subject_list
from common.neuro_utils import load_nifti
from nilearn.maskers import NiftiLabelsMasker

from analyses.univariate_rois.io import UNIV_CONTRASTS, find_subject_contrast_path


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
config, _, logger = setup_analysis(
    analysis_name="01_univariate_roi_subject",
    results_base=CONFIG["RESULTS_ROOT"] / "univariate_rois" / "logs",
    script_file=__file__,
)

DERIV_ROOT: Path = CONFIG["BIDS_UNIVARIATE_ROIS"]
DERIV_ROOT.mkdir(parents=True, exist_ok=True)

SUBJECT_FILE_TEMPLATE = (
    "{sub}_space-MNI152NLin2009cAsym_roi-glasser180"
    "_desc-univmean_contrast-{con}_rois.tsv"
)
SIDECAR_STEM = (
    "space-MNI152NLin2009cAsym_roi-glasser180_desc-univmean_rois"
)


# ---------------------------------------------------------------------------
# Pipeline descriptor and root-level sidecar
# ---------------------------------------------------------------------------
def write_pipeline_descriptor(root: Path) -> None:
    descriptor = {
        "Name": "fmriprep_spm-smoothed_univariate-rois",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "fmriprep_spm-smoothed_univariate-rois",
                "Description": (
                    "Per-subject mean univariate contrast values within "
                    "each of the 180 bilateral cortical regions of the "
                    "Glasser atlas, extracted from smoothed SPM first-level "
                    "contrast images (con_0001: Checkmate > Non-checkmate; "
                    "con_0002: All > Rest)."
                ),
                "CodeURL": "https://github.com/costantinoai/chess-expertise-2025",
            }
        ],
        "SourceDatasets": [{"URL": "../fmriprep_spm-smoothed"}],
    }
    (root / "dataset_description.json").write_text(
        json.dumps(descriptor, indent=2) + "\n"
    )


def write_root_sidecar(root: Path) -> None:
    sidecar = {
        "Description": (
            "Per-subject mean univariate contrast values for each ROI in the "
            "Glasser-180 bilateral cortical atlas, extracted from smoothed "
            "SPM first-level contrast images. Each row is one ROI for one "
            "subject; values inherit this sidecar via the BIDS inheritance "
            "principle."
        ),
        "RoiAtlas": "glasser180-bilateral",
        "Columns": {
            "roi_id": {
                "Description": "Integer atlas label for the bilateral ROI (1..180)."
            },
            "roi_mean": {
                "Description": (
                    "Mean contrast value (parameter estimate) across all "
                    "voxels in the bilateral ROI."
                )
            },
        },
    }
    (root / f"{SIDECAR_STEM}.json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
logger.info("=" * 80)
logger.info("UNIVARIATE ROI EXTRACTION -- PER-SUBJECT (GLASSER-180, VOLUME-BASED)")
logger.info("=" * 80)

logger.info("Writing pipeline descriptor and sidecar...")
write_pipeline_descriptor(DERIV_ROOT)
write_root_sidecar(DERIV_ROOT)

# Load bilateral volumetric atlas and build masker
atlas_path = CONFIG["ROI_GLASSER_180_ATLAS"]
masker = NiftiLabelsMasker(labels_img=str(atlas_path), standardize=False, strategy="mean")
logger.info(f"Loaded Glasser-180 bilateral atlas from: {atlas_path}")

# ROI IDs are 1..180 (bilateral labels in the atlas)
roi_ids = list(range(1, 181))

glm_base = Path(CONFIG["SPM_GLM_SMOOTH4"])
all_subjects = get_subject_list()
logger.info(
    f"Extracting ROI means for {len(all_subjects)} subjects "
    f"across {len(UNIV_CONTRASTS)} contrasts x {len(roi_ids)} ROIs"
)

written = 0
skipped = 0
for subject_id in all_subjects:
    sub_dir = DERIV_ROOT / subject_id
    sub_dir.mkdir(parents=True, exist_ok=True)

    for con_code, con_label in UNIV_CONTRASTS.items():
        try:
            path = find_subject_contrast_path(subject_id, con_code, glm_base)
        except FileNotFoundError as err:
            logger.warning(f"  {subject_id} {con_code}: missing -- skipping ({err})")
            skipped += 1
            continue

        img = load_nifti(path)

        # Extract all 180 bilateral ROI means at once
        roi_means = masker.fit_transform(img).flatten()  # shape: (180,)

        rows = []
        for roi_id, roi_mean in zip(roi_ids, roi_means):
            rows.append({"roi_id": int(roi_id), "roi_mean": float(roi_mean)})

        subject_df = pd.DataFrame(rows)
        tsv_name = SUBJECT_FILE_TEMPLATE.format(sub=subject_id, con=con_code)
        tsv_path = sub_dir / tsv_name
        subject_df.to_csv(tsv_path, sep="\t", index=False)
        logger.info(f"  {subject_id} {con_code}: {len(subject_df)} ROIs -> {tsv_path.name}")
        written += 1

logger.info(
    f"Wrote {written} subject-level TSVs under {DERIV_ROOT} "
    f"({skipped} subject-contrast pairs skipped)."
)
logger.info("=" * 80)
log_script_end(logger)
