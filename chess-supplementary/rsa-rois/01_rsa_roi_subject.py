#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA Searchlight ROI -- per-subject stage
=========================================

For each participant and RSA target (visual_similarity, strategy, checkmate),
loads the volumetric RSA searchlight r-map from
``BIDS/derivatives/fmriprep_spm-unsmoothed_searchlight-rsa/``,
extracts 180 bilateral Glasser ROI means via NiftiLabelsMasker,
and writes a BIDS-named TSV under

    BIDS/derivatives/fmriprep_spm-unsmoothed_rsa-rois/sub-XX/
        sub-XX_space-MNI152NLin2009cAsym_roi-glasser180_desc-rsamean_target-{target}_rois.tsv

with a root-level ``dataset_description.json`` and sidecar JSON
documenting the columns.

The per-subject outputs feed into ``11_rsa_roi_group.py`` which reads
these TSVs and runs all group-level statistics.

Columns in each per-subject TSV:
- subject_id : BIDS subject identifier (e.g. sub-01)
- roi_1 ... roi_180 : mean RSA searchlight r-value within each bilateral
  Glasser ROI
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_subject_list
from common.neuro_utils import load_nifti

from analyses.rsa_rois.io import RSA_TARGETS, find_subject_rsa_path, _BIDS_DESC_FOR_TARGET


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
config, _, logger = setup_analysis(
    analysis_name="01_rsa_rois_subject",
    results_base=CONFIG["RESULTS_ROOT"] / "supplementary" / "rsa-rois" / "logs",
    script_file=__file__,
)

# Derivatives output root
RSA_ROIS_ROOT: Path = (
    Path(CONFIG["BIDS_RSA_SEARCHLIGHT"]).parent / "fmriprep_spm-unsmoothed_rsa-rois"
)
RSA_ROIS_ROOT.mkdir(parents=True, exist_ok=True)

logger.info("=" * 80)
logger.info("RSA SEARCHLIGHT ROI -- PER-SUBJECT EXTRACTION (GLASSER-180 BILATERAL)")
logger.info("=" * 80)
logger.info(f"Output derivatives root: {RSA_ROIS_ROOT}")


# ---------------------------------------------------------------------------
# Pipeline descriptor and root-level sidecar
# ---------------------------------------------------------------------------
def write_pipeline_descriptor(root: Path) -> None:
    descriptor = {
        "Name": "fmriprep_spm-unsmoothed_rsa-rois",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "fmriprep_spm-unsmoothed_rsa-rois",
                "Description": (
                    "Per-subject mean RSA searchlight correlation (r) values "
                    "within each of the 180 bilateral cortical regions of the "
                    "Glasser atlas, extracted from volumetric searchlight r-maps."
                ),
                "CodeURL": "https://github.com/costantinoai/chess-expertise-2025",
            }
        ],
        "SourceDatasets": [{"URL": "../fmriprep_spm-unsmoothed_searchlight-rsa"}],
    }
    (root / "dataset_description.json").write_text(
        json.dumps(descriptor, indent=2) + "\n"
    )


def write_root_sidecar(root: Path) -> None:
    sidecar = {
        "Description": (
            "Per-subject mean RSA searchlight correlation (r) values for "
            "each of the 180 bilateral Glasser cortical ROIs, extracted from "
            "volumetric searchlight r-maps. Each row is one subject; columns "
            "roi_1..roi_180 give the mean r-value within that ROI. Values "
            "inherit this sidecar via the BIDS inheritance principle."
        ),
        "RoiAtlas": "glasser180-bilateral",
        "Columns": {
            "subject_id": {
                "Description": "BIDS subject identifier (e.g. sub-01)."
            },
            "roi_N": {
                "Description": (
                    "Mean RSA searchlight r-value within bilateral Glasser "
                    "ROI N (N = 1..180). Computed as the arithmetic mean of "
                    "all voxel-wise r-values falling within the ROI mask."
                )
            },
        },
    }
    sidecar_stem = (
        "space-MNI152NLin2009cAsym_roi-glasser180_desc-rsamean_rois"
    )
    (root / f"{sidecar_stem}.json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# Atlas setup
# ---------------------------------------------------------------------------
logger.info("Writing pipeline descriptor and sidecar...")
write_pipeline_descriptor(RSA_ROIS_ROOT)
write_root_sidecar(RSA_ROIS_ROOT)

atlas_path = CONFIG["ROI_GLASSER_180_ATLAS"]
masker = NiftiLabelsMasker(labels_img=str(atlas_path), standardize=False, strategy="mean")
logger.info(f"Loaded Glasser-180 bilateral atlas from: {atlas_path}")

roi_ids = list(range(1, 181))  # 180 bilateral ROIs labelled 1..180
n_rois = len(roi_ids)

rsa_base = Path(CONFIG["BIDS_RSA_SEARCHLIGHT"])
all_subjects = get_subject_list()

logger.info(
    f"Extracting ROI means for {len(all_subjects)} subjects x "
    f"{len(RSA_TARGETS)} targets x {n_rois} ROIs"
)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
written = 0
skipped = 0

for subject_id in all_subjects:
    for tgt_key, tgt_label in RSA_TARGETS.items():
        # Load searchlight r-map
        try:
            nifti_path = find_subject_rsa_path(subject_id, tgt_key, rsa_base)
        except FileNotFoundError as err:
            logger.warning(f"  {subject_id}/{tgt_key}: missing r-map -- skipping ({err})")
            skipped += 1
            continue

        img = load_nifti(nifti_path)

        # Extract 180 bilateral ROI means
        roi_means = masker.fit_transform(img).flatten()  # shape: (180,)

        # Build single-row DataFrame
        row = {"subject_id": subject_id}
        row.update({f"roi_{rid}": float(v) for rid, v in zip(roi_ids, roi_means)})
        subject_df = pd.DataFrame([row])

        # Write to derivatives
        bids_desc = _BIDS_DESC_FOR_TARGET[tgt_key]
        sub_dir = RSA_ROIS_ROOT / subject_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        tsv_name = (
            f"{subject_id}_space-MNI152NLin2009cAsym"
            f"_roi-glasser180_desc-rsamean"
            f"_target-{bids_desc}_rois.tsv"
        )
        tsv_path = sub_dir / tsv_name
        subject_df.to_csv(tsv_path, sep="\t", index=False)
        logger.info(f"  {subject_id}/{tgt_key}: {n_rois} ROIs -> {tsv_path.name}")
        written += 1

logger.info(
    f"Wrote {written} subject-level TSVs under {RSA_ROIS_ROOT} "
    f"({skipped} subject/target combinations skipped)."
)
log_script_end(logger)
