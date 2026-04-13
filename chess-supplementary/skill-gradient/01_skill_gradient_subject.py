#!/usr/bin/env python3
"""
Skill-Gradient Subject-Level — Enriched Per-Subject Table

METHODS
=======

Overview
--------
This script creates an enriched per-subject table that merges behavioural
and neural metrics for each participant. The table is used by downstream
group-level correlation scripts (11_skill_gradient_group.py) and plotting
scripts (91_plot_skill_gradient.py).

Data Sources
------------
- Participants: BIDS participants.tsv (participant_id, group, Elo rating)
- PR: BIDS/derivatives/fmriprep_spm-unsmoothed_manifold per-subject TSVs
- RSA: BIDS/derivatives/fmriprep_spm-unsmoothed_rsa per-subject TSVs
- Decoding: BIDS/derivatives/fmriprep_spm-unsmoothed_decoding per-subject TSVs
- Familiarisation: results/supplementary/task-engagement/data/
  familiarisation_subject_accuracy.csv (behavioural accuracy from
  12_familiarisation_group.py)

Outputs
-------
- familiarisation_subject_enriched.csv: Per-subject table with columns:
    participant_id, group, rating, move_acc_all_cm, mean_pr,
    rsa_checkmate, rsa_strategy, rsa_visual_similarity,
    dec_checkmate, dec_strategy
"""

from pathlib import Path

import pandas as pd

# Enable repo root imports
from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from utils import compute_subject_mean_pr, load_bids_tsvs


# ============================================================================
# Setup
# ============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="skill_gradient",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)


# ============================================================================
# Load participant metadata
# ============================================================================

participants = pd.read_csv(config['BIDS_PARTICIPANTS'], sep='\t')
logger.info(f"Participants: {len(participants)} total")

# Start with participant_id, group, rating
fam_file = (Path(config['RESULTS_ROOT']) / 'supplementary' / 'task-engagement' /
            'data' / 'familiarisation_subject_accuracy.csv')

if not fam_file.exists():
    raise FileNotFoundError(
        f"Familiarisation accuracy file not found: {fam_file}\n"
        "Run chess-supplementary/task-engagement/12_familiarisation_group.py first."
    )

fam_df = pd.read_csv(fam_file)
logger.info(f"Loaded familiarisation data: {len(fam_df)} subjects")

# Merge rating from participants.tsv
fam_df = fam_df.merge(
    participants[['participant_id', 'rating']],
    on='participant_id', how='left',
)


# ============================================================================
# Load PR from BIDS derivatives (per-subject TSVs)
# ============================================================================

logger.info("Loading per-subject PR from BIDS derivatives...")

MANIFOLD_ROOT = Path(config['BIDS_MANIFOLD'])
SUBJECT_FILE_SUFFIX = "_space-MNI152NLin2009cAsym_roi-glasser_desc-pr_values.tsv"

pr_records = []
for sub_dir in sorted(MANIFOLD_ROOT.iterdir()):
    if not sub_dir.is_dir() or not sub_dir.name.startswith('sub-'):
        continue
    subject_id = sub_dir.name
    tsv_path = sub_dir / f"{subject_id}{SUBJECT_FILE_SUFFIX}"
    if not tsv_path.is_file():
        logger.warning(f"  {subject_id}: no PR TSV at {tsv_path} -- skipping")
        continue
    df = pd.read_csv(tsv_path, sep='\t')
    for _, row in df.iterrows():
        pr_records.append({
            'subject_id': subject_id,
            'ROI_Label': int(row['ROI_Label']),
            'PR': float(row['PR']),
        })

pr_long = pd.DataFrame(pr_records)
logger.info(f"PR data: {pr_long['subject_id'].nunique()} subjects, "
            f"{pr_long['ROI_Label'].nunique()} ROIs")

subject_pr = compute_subject_mean_pr(pr_long)
fam_df = fam_df.merge(subject_pr, on='participant_id', how='left')


# ============================================================================
# Load RSA from BIDS derivatives (per-subject TSVs)
# ============================================================================

logger.info("Loading RSA data...")
rsa_df = load_bids_tsvs(config['BIDS_MVPA_RSA'])
roi_cols = [c for c in rsa_df.columns if c not in ['subject', 'target']]
logger.info(f"RSA data: {rsa_df['subject'].nunique()} subjects, "
            f"{len(roi_cols)} ROIs, targets: {sorted(rsa_df['target'].unique())}")

for target in ['checkmate', 'strategy', 'visual_similarity']:
    rsa_target = rsa_df[rsa_df['target'] == target].copy()
    if len(rsa_target) > 0:
        rsa_target[f'rsa_{target}'] = rsa_target[roi_cols].astype(float).mean(axis=1)
        fam_df = fam_df.merge(
            rsa_target[['subject', f'rsa_{target}']].rename(
                columns={'subject': 'participant_id'}),
            on='participant_id', how='left',
        )


# ============================================================================
# Load decoding from BIDS derivatives (per-subject TSVs)
# ============================================================================

logger.info("Loading decoding data...")
dec_df = load_bids_tsvs(config['BIDS_MVPA_DECODING'])
dec_roi_cols = [c for c in dec_df.columns if c not in ['subject', 'target']]
logger.info(f"Decoding data: {dec_df['subject'].nunique()} subjects, "
            f"{len(dec_roi_cols)} ROIs, targets: {sorted(dec_df['target'].unique())}")

for target in ['checkmate', 'strategy']:
    dec_target = dec_df[dec_df['target'] == target].copy()
    if len(dec_target) > 0:
        dec_target[f'dec_{target}'] = dec_target[dec_roi_cols].astype(float).mean(axis=1)
        fam_df = fam_df.merge(
            dec_target[['subject', f'dec_{target}']].rename(
                columns={'subject': 'participant_id'}),
            on='participant_id', how='left',
        )


# ============================================================================
# Save enriched per-subject table
# ============================================================================

DERIV_DIR = Path(config['BIDS_SKILL_GRADIENT']) if 'BIDS_SKILL_GRADIENT' in config else Path(config['BIDS_ROOT']).parent / 'derivatives' / 'skill-gradient'
DERIV_DIR.mkdir(parents=True, exist_ok=True)
fam_df.to_csv(DERIV_DIR / 'familiarisation_subject_enriched.csv', index=False)
logger.info(f"Saved familiarisation_subject_enriched.csv → {DERIV_DIR} ({len(fam_df)} subjects)")

neural_cols = ['mean_pr', 'rsa_checkmate', 'rsa_strategy',
               'rsa_visual_similarity', 'dec_checkmate', 'dec_strategy']
for nc in neural_cols:
    if nc in fam_df.columns:
        n_valid = fam_df[nc].notna().sum()
        logger.info(f"  {nc}: {n_valid} valid values")

log_script_end(logger)
