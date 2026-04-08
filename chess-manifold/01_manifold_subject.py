#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manifold -- per-subject stage
=============================

For each participant, reads trial-wise beta estimates from the unsmoothed
SPM first-level GLM derivative, computes the participation ratio (PR) in
every Glasser-22 bilateral cortical ROI, and writes a BIDS-named TSV
under

    BIDS/derivatives/fmriprep_spm-unsmoothed_manifold/sub-XX/
        sub-XX_space-MNI152NLin2009cAsym_roi-glasser_desc-pr_values.tsv

with a root-level sidecar JSON documenting the columns. The pipeline's
``dataset_description.json`` is created on first run.

The subject-level outputs feed into ``02_manifold_group.py`` which runs
all group statistics and classifier training.

Columns in the per-subject TSV:
- ROI_Label : integer atlas label (1..22)
- roi_name  : human-readable ROI name from region_info.tsv
- PR        : participation ratio (float)
- n_voxels  : number of voxels contributing to that ROI for this subject
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_subject_list
from common.neuro_utils import load_atlas

from analyses.manifold.data import load_atlas_and_metadata
from analyses.manifold.pr_computation import compute_subject_roi_prs


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
script_dir = Path(__file__).parent
config, _, logger = setup_analysis(
    analysis_name="manifold_subject",
    results_base=script_dir / "results",
    script_file=__file__,
)

MANIFOLD_ROOT: Path = CONFIG["BIDS_MANIFOLD"]
MANIFOLD_ROOT.mkdir(parents=True, exist_ok=True)

SUBJECT_FILE_SUFFIX = (
    "_space-MNI152NLin2009cAsym_roi-glasser_desc-pr_values.tsv"
)
SIDECAR_STEM = "space-MNI152NLin2009cAsym_roi-glasser_desc-pr_values"


# ---------------------------------------------------------------------------
# Pipeline descriptor and root-level sidecar
# ---------------------------------------------------------------------------
def write_pipeline_descriptor(root: Path) -> None:
    descriptor = {
        "Name": "fmriprep_spm-unsmoothed_manifold",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "fmriprep_spm-unsmoothed_manifold",
                "Description": (
                    "Per-subject neural participation ratio (PR) values in "
                    "the 22 bilateral cortical regions of the Glasser atlas, "
                    "computed from unsmoothed SPM first-level beta estimates "
                    "(40 stimuli x n_voxels per ROI). PR is the effective "
                    "dimensionality of the neural state space, defined as "
                    "(sum of eigenvalues)^2 / sum of eigenvalues^2."
                ),
                "CodeURL": "https://github.com/costantinoai/chess-expertise-2025",
            }
        ],
        "SourceDatasets": [{"URL": "../fmriprep_spm-unsmoothed"}],
    }
    (root / "dataset_description.json").write_text(
        json.dumps(descriptor, indent=2) + "\n"
    )


def write_root_sidecar(root: Path) -> None:
    sidecar = {
        "Description": (
            "Per-subject participation ratio (PR) values for each ROI in the "
            "Glasser-22 bilateral cortical atlas, computed from unsmoothed "
            "SPM first-level beta estimates. Each row is one ROI for one "
            "subject; values inherit this sidecar via the BIDS inheritance "
            "principle."
        ),
        "RoiAtlas": "glasser22-bilateral",
        "Columns": {
            "ROI_Label": {
                "Description": "Integer atlas label for the ROI (1..22)."
            },
            "roi_name": {
                "Description": "Human-readable ROI name from region_info.tsv."
            },
            "PR": {
                "Description": (
                    "Participation ratio = (sum(lambda_i))^2 / sum(lambda_i^2), "
                    "where lambda_i are the eigenvalues of the ROI's "
                    "(n_stimuli x n_voxels) beta covariance matrix. "
                    "Ranges from 1 (one dominant dimension) to n_voxels "
                    "(uniform distribution)."
                )
            },
            "n_voxels": {
                "Description": "Number of voxels contributing to the ROI for this subject."
            },
        },
    }
    (root / f"{SIDECAR_STEM}.json").write_text(json.dumps(sidecar, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
logger.info("Writing pipeline descriptor and sidecar...")
write_pipeline_descriptor(MANIFOLD_ROOT)
write_root_sidecar(MANIFOLD_ROOT)

logger.info("Loading atlas and ROI metadata...")
atlas_data, roi_labels, roi_info, _ = load_atlas_and_metadata(
    atlas_path=CONFIG["ROI_GLASSER_22_ATLAS"],
    roi_info_path=CONFIG["ROI_GLASSER_22"] / "region_info.tsv",
    participants_path=CONFIG["BIDS_PARTICIPANTS"],
    load_atlas_func=load_atlas,
)
roi_name_by_label = {
    int(row["roi_id"]): row["roi_name"]
    for _, row in roi_info[["roi_id", "roi_name"]].iterrows()
}

all_subjects = get_subject_list()
logger.info(
    f"Computing PR for {len(all_subjects)} subjects across {len(roi_labels)} ROIs"
)

written = 0
skipped = 0
for subject_id in all_subjects:
    try:
        pr_values, voxel_counts = compute_subject_roi_prs(
            subject_id=subject_id,
            atlas_data=atlas_data,
            roi_labels=roi_labels,
            base_path=CONFIG["SPM_GLM_UNSMOOTHED"],
        )
    except FileNotFoundError as err:
        logger.warning(f"  {subject_id}: missing SPM.mat -- skipping ({err})")
        skipped += 1
        continue

    rows = []
    for roi_idx, roi_label in enumerate(roi_labels):
        rows.append(
            {
                "ROI_Label": int(roi_label),
                "roi_name": roi_name_by_label.get(int(roi_label), ""),
                "PR": float(pr_values[roi_idx]),
                "n_voxels": int(voxel_counts[roi_idx]),
            }
        )
    subject_df = pd.DataFrame(rows)
    sub_dir = MANIFOLD_ROOT / subject_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = sub_dir / f"{subject_id}{SUBJECT_FILE_SUFFIX}"
    subject_df.to_csv(tsv_path, sep="\t", index=False)
    logger.info(f"  {subject_id}: {len(subject_df)} ROIs -> {tsv_path.name}")
    written += 1

logger.info(
    f"Wrote {written} subject-level TSVs under {MANIFOLD_ROOT} ({skipped} subjects skipped)."
)
log_script_end(logger)
