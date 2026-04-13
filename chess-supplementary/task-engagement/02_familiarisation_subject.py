#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Familiarisation Task Accuracy -- per-subject stage
===================================================

For each participant, reads the BIDS familiarisation behavioural TSV,
computes detection accuracy and move accuracy, and writes three files
under

    BIDS/derivatives/task-engagement/sub-XX/
        sub-XX_desc-familiarisation_accuracy.tsv
        sub-XX_desc-familiarisation_responsematrix.tsv
        sub-XX_desc-familiarisation_detectionmatrix.tsv

The accuracy TSV (one row per subject) and the two matrix TSVs feed into
``12_familiarisation_group.py`` which runs all group comparisons.

Columns in the per-subject accuracy TSV:
- participant_id         : participant identifier (sub-XX)
- group                  : expert or novice
- n_checkmate            : number of checkmate boards shown
- n_noncheckmate         : number of non-checkmate boards shown
- cm_hits                : checkmate boards correctly identified
- cm_misses              : checkmate boards missed
- nc_correct_rejections  : non-checkmate boards correctly left empty
- nc_false_alarms        : non-checkmate boards incorrectly responded to
- detection_acc          : overall detection accuracy (hits + correct rejections) / total
- n_cm_responded         : number of checkmate boards where participant typed a move
- move_correct_count     : number of correct first moves
- move_acc_all_cm        : move accuracy over all checkmate boards
- move_acc_responded     : move accuracy over checkmate boards with a response

Note: Two participants (sub-07, sub-39) have no familiarisation data
(no Pavlovia record found) and are excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end

from analyses.task_engagement.io import load_familiarisation_data, MISSING_SUBJECTS


# ============================================================================
# Setup
# ============================================================================

config, _, logger = setup_analysis(
    analysis_name="02_familiarisation_subject",
    results_base=CONFIG["RESULTS_ROOT"] / "supplementary" / "task-engagement" / "logs",
    script_file=__file__,
)

TASK_ENGAGEMENT_ROOT: Path = CONFIG["BIDS_TASK_ENGAGEMENT"]
TASK_ENGAGEMENT_ROOT.mkdir(parents=True, exist_ok=True)

ACCURACY_SUFFIX = "_desc-familiarisation_accuracy.tsv"
RESPONSE_MATRIX_SUFFIX = "_desc-familiarisation_responsematrix.tsv"
DETECTION_MATRIX_SUFFIX = "_desc-familiarisation_detectionmatrix.tsv"

ACCURACY_SIDECAR_STEM = "desc-familiarisation_accuracy"
RESPONSE_MATRIX_SIDECAR_STEM = "desc-familiarisation_responsematrix"
DETECTION_MATRIX_SIDECAR_STEM = "desc-familiarisation_detectionmatrix"


# ============================================================================
# Root-level sidecars
# ============================================================================

def write_accuracy_sidecar(root: Path) -> None:
    """Write the root-level sidecar for accuracy TSVs."""
    sidecar = {
        "Description": (
            "Per-subject familiarisation task accuracy. Each row is one "
            "participant. Detection accuracy measures whether the participant "
            "correctly distinguished checkmate from non-checkmate boards. "
            "Move accuracy measures whether the correct first move was "
            "identified on checkmate boards."
        ),
        "Columns": {
            "participant_id": {"Description": "Participant identifier (sub-XX)."},
            "group": {"Description": "Expertise group: 'expert' or 'novice'."},
            "n_checkmate": {"Description": "Number of checkmate boards shown."},
            "n_noncheckmate": {"Description": "Number of non-checkmate boards shown."},
            "cm_hits": {"Description": "Checkmate boards correctly identified."},
            "cm_misses": {"Description": "Checkmate boards missed."},
            "nc_correct_rejections": {"Description": "Non-checkmate boards correctly left empty."},
            "nc_false_alarms": {"Description": "Non-checkmate boards incorrectly responded to."},
            "detection_acc": {"Description": "Overall detection accuracy."},
            "n_cm_responded": {"Description": "Number of checkmate boards where participant typed a move."},
            "move_correct_count": {"Description": "Number of correct first moves."},
            "move_acc_all_cm": {"Description": "Move accuracy over all checkmate boards."},
            "move_acc_responded": {"Description": "Move accuracy over checkmate boards with a response."},
        },
    }
    (root / f"{ACCURACY_SIDECAR_STEM}.json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )


def write_response_matrix_sidecar(root: Path) -> None:
    """Write the root-level sidecar for response matrix TSVs."""
    sidecar = {
        "Description": (
            "Per-subject move correctness matrix for checkmate boards. "
            "Each row is the participant; each column is a stimulus ID. "
            "Values indicate whether the participant's first-move response "
            "was correct (1) or not (0)."
        ),
    }
    (root / f"{RESPONSE_MATRIX_SIDECAR_STEM}.json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )


def write_detection_matrix_sidecar(root: Path) -> None:
    """Write the root-level sidecar for detection matrix TSVs."""
    sidecar = {
        "Description": (
            "Per-subject detection correctness matrix for all boards. "
            "Each row is the participant; each column is a stimulus ID. "
            "Values indicate whether the participant correctly detected "
            "the board type (1 = correct, 0 = incorrect)."
        ),
    }
    (root / f"{DETECTION_MATRIX_SIDECAR_STEM}.json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )


# ============================================================================
# Load data
# ============================================================================

logger.info("Loading familiarisation data from BIDS TSVs...")
data = load_familiarisation_data()

n_subjects = data['participant_id'].nunique()
n_experts = data[data['group'] == 'expert']['participant_id'].nunique()
n_novices = data[data['group'] == 'novice']['participant_id'].nunique()
logger.info(f"Loaded {len(data)} trials from {n_subjects} participants "
            f"({n_experts} experts, {n_novices} novices)")
if MISSING_SUBJECTS:
    logger.info(f"Missing participants (no Pavlovia data): {sorted(MISSING_SUBJECTS)}")


# ============================================================================
# Write sidecars
# ============================================================================

logger.info("Writing root-level sidecars...")
write_accuracy_sidecar(TASK_ENGAGEMENT_ROOT)
write_response_matrix_sidecar(TASK_ENGAGEMENT_ROOT)
write_detection_matrix_sidecar(TASK_ENGAGEMENT_ROOT)


# ============================================================================
# Per-subject computation
# ============================================================================

logger.info("Computing per-subject familiarisation accuracy...")

acc_rows = []
resp_rows = []
det_rows = []
for sub_id in sorted(data['participant_id'].unique()):
    sdf = data[data['participant_id'] == sub_id]
    group = sdf['group'].iloc[0]

    cm = sdf[sdf['is_checkmate'] == 1]
    nc = sdf[sdf['is_checkmate'] == 0]

    # Detection accuracy (coarse)
    n_cm = len(cm)
    n_nc = len(nc)
    cm_hits = int(cm['detection_correct'].sum())
    nc_correct_rej = int(nc['detection_correct'].sum())
    detection_acc = (cm_hits + nc_correct_rej) / (n_cm + n_nc) if (n_cm + n_nc) > 0 else np.nan

    # Move accuracy (fine) — among checkmate boards only
    cm_responded = cm[cm['responded'] == 1]
    n_cm_responded = len(cm_responded)
    if n_cm_responded > 0:
        move_correct_count = int(cm_responded['move_correct'].sum())
        move_acc = move_correct_count / n_cm if n_cm > 0 else np.nan
        move_acc_responded = move_correct_count / n_cm_responded
    else:
        move_correct_count = 0
        move_acc = 0.0
        move_acc_responded = np.nan

    acc_rows.append({
        'participant_id': sub_id,
        'group': group,
        'n_checkmate': n_cm,
        'n_noncheckmate': n_nc,
        'cm_hits': cm_hits,
        'cm_misses': n_cm - cm_hits,
        'nc_correct_rejections': nc_correct_rej,
        'nc_false_alarms': n_nc - nc_correct_rej,
        'detection_acc': detection_acc,
        'n_cm_responded': n_cm_responded,
        'move_correct_count': move_correct_count,
        'move_acc_all_cm': move_acc,
        'move_acc_responded': move_acc_responded,
    })

    # Response matrix row (checkmate boards: move correctness)
    cm_sub = sdf[sdf['is_checkmate'] == 1].copy()
    resp_row = cm_sub.set_index('stim_id')['move_correct'].fillna(0)
    resp_row.name = sub_id
    resp_rows.append(resp_row)

    # Detection matrix row (all boards: detection correctness)
    det_row = sdf.set_index('stim_id')['detection_correct'].fillna(0)
    det_row.name = sub_id
    det_rows.append(det_row)

    logger.info(
        f"  {sub_id}: detection_acc={detection_acc:.3f}, "
        f"move_acc={move_acc:.3f} ({n_cm_responded}/{n_cm} responded)"
    )

# Write single stacked files (all subjects in one file each)
acc_df = pd.DataFrame(acc_rows)
acc_df.to_csv(TASK_ENGAGEMENT_ROOT / "familiarisation_accuracy.tsv", sep="\t", index=False)

resp_df = pd.DataFrame(resp_rows)
resp_df.index.name = 'participant_id'
resp_df.to_csv(TASK_ENGAGEMENT_ROOT / "familiarisation_response_matrix.tsv", sep="\t")

det_df = pd.DataFrame(det_rows)
det_df.index.name = 'participant_id'
det_df.to_csv(TASK_ENGAGEMENT_ROOT / "familiarisation_detection_matrix.tsv", sep="\t")

logger.info(
    f"Wrote {len(acc_df)} subjects to familiarisation_{{accuracy,response_matrix,detection_matrix}}.tsv"
)
log_script_end(logger)
