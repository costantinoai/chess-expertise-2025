#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioral RSA -- per-subject stage
===================================

For each participant, reads BIDS events, converts 1-back preference responses
into pairwise (better, worse) comparison rows, aggregates those rows into a
per-subject count table, and writes the table as a BIDS-named TSV under

    BIDS/derivatives/behavioral-rsa/sub-XX/sub-XX_desc-preference_pairs.tsv

together with a root-level sidecar JSON documenting the columns. The
pipeline's ``dataset_description.json`` is created on first run.

The aggregation is mathematically identical to counting the concatenated
raw-trial rows (addition is associative), so the downstream group stage
``02_behavioral_rsa_group.py`` produces byte-identical numerical results
to the pre-refactor monolithic ``01_behavioral_rsa.py``.

Outputs (under BIDS derivatives, NOT the repo):
    BIDS/derivatives/behavioral-rsa/
    ├── dataset_description.json
    ├── desc-preference_pairs.json          (root-level sidecar, inherited)
    └── sub-XX/
        └── sub-XX_desc-preference_pairs.tsv
            columns: better, worse, count
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import (
    CONFIG,
    setup_analysis,
    log_script_end,
    get_participants_with_expertise,
)
from analyses.behavioral.data_loading import load_participant_trial_data
from analyses.behavioral.rdm_utils import create_pairwise_df, aggregate_pairwise_counts


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
config, _, logger = setup_analysis(
    analysis_name="01_behavioral_rsa_subject",
    results_base=CONFIG["RESULTS_ROOT"] / "behavioral" / "logs",
    script_file=__file__,
)

BIDS_ROOT = CONFIG["BIDS_ROOT"]
BEHAVIORAL_RSA_ROOT: Path = CONFIG["BIDS_BEHAVIORAL_RSA"]
BEHAVIORAL_RSA_ROOT.mkdir(parents=True, exist_ok=True)

SUBJECT_FILE_SUFFIX = "_desc-preference_pairs.tsv"
SIDECAR_STEM = "desc-preference_pairs"


# ---------------------------------------------------------------------------
# Pipeline descriptor and root-level sidecar
# ---------------------------------------------------------------------------
def write_pipeline_descriptor(root: Path) -> None:
    """Write dataset_description.json for the behavioral-rsa pipeline."""
    descriptor = {
        "Name": "behavioral-rsa",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "behavioral-rsa",
                "Description": (
                    "Per-subject pairwise preference counts derived from the "
                    "1-back behavioral task events. For each (better, worse) "
                    "stimulus pair, stores the number of times the subject "
                    "chose the first over the second."
                ),
                "CodeURL": "https://github.com/costantinoai/chess-expertise-2025",
            }
        ],
        "SourceDatasets": [{"URL": "../../"}],
    }
    (root / "dataset_description.json").write_text(
        json.dumps(descriptor, indent=2) + "\n"
    )


def write_root_sidecar(root: Path) -> None:
    """Write the root-level sidecar describing the per-subject TSV columns.

    Per the BIDS inheritance principle, a single sidecar at the pipeline
    root is inherited by all matching files below (sub-XX/*_desc-preference_pairs.tsv).
    """
    sidecar = {
        "Description": (
            "Per-subject pairwise preference counts derived from the 1-back "
            "behavioral task. Each row is one (better, worse) stimulus pair; "
            "the count column is the number of times the subject chose "
            "'better' over 'worse' across all valid trials. Trials with "
            "preference = 'n/a' are skipped; cross-run transitions are "
            "excluded."
        ),
        "Columns": {
            "better": {
                "Description": "Stimulus ID (1-40) of the preferred board in the pair."
            },
            "worse": {
                "Description": "Stimulus ID (1-40) of the non-preferred board in the pair."
            },
            "count": {
                "Description": "Number of trials on which this subject chose 'better' over 'worse'.",
                "Units": "count",
            },
        },
    }
    (root / f"{SIDECAR_STEM}.json").write_text(json.dumps(sidecar, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------
def write_subject(subject_id: str, pairs: pd.DataFrame) -> Path:
    """Write a per-subject aggregated pair-count TSV and return its path."""
    sub_dir = BEHAVIORAL_RSA_ROOT / subject_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = sub_dir / f"{subject_id}{SUBJECT_FILE_SUFFIX}"
    # Stable, deterministic column order
    pairs = pairs[["better", "worse", "count"]]
    pairs.to_csv(tsv_path, sep="\t", index=False)
    return tsv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
logger.info("Writing pipeline descriptor and sidecar...")
write_pipeline_descriptor(BEHAVIORAL_RSA_ROOT)
write_root_sidecar(BEHAVIORAL_RSA_ROOT)

logger.info("Loading participant information...")
participants_list, (n_experts, n_novices) = get_participants_with_expertise()
logger.info(f"Loaded {n_experts} experts and {n_novices} novices")

logger.info("Extracting pairwise preferences per subject...")
written = 0
skipped = 0
for subject_id, is_expert in participants_list:
    trial_df = load_participant_trial_data(subject_id, is_expert, BIDS_ROOT)
    if trial_df is None:
        logger.warning(f"  {subject_id}: no valid trial data -- skipping")
        skipped += 1
        continue

    raw_pairs = create_pairwise_df(trial_df)
    if len(raw_pairs) == 0:
        logger.warning(f"  {subject_id}: no valid pairwise trials -- skipping")
        skipped += 1
        continue

    # aggregate_pairwise_counts accepts a list of pair DataFrames (one per
    # participant). For the per-subject derivative we wrap the single
    # subject's raw pairs in a list so the same helper is reused.
    subject_counts = aggregate_pairwise_counts([raw_pairs])
    tsv_path = write_subject(subject_id, subject_counts)
    logger.info(
        f"  {subject_id}: {len(subject_counts)} unique pairs -> {tsv_path.name}"
    )
    written += 1

logger.info(
    f"Wrote {written} subject-level TSVs under {BEHAVIORAL_RSA_ROOT} ({skipped} subjects skipped)."
)
log_script_end(logger)
