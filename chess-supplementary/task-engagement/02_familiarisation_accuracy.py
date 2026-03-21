#!/usr/bin/env python3
"""
Familiarisation Task Accuracy — Supplementary Analysis

METHODS
=======

Overview
--------
This analysis quantifies pre-scan familiarisation task performance for both
expert and novice groups. Participants viewed 40 chess positions (20 checkmate,
20 non-checkmate) and were instructed to type the first move leading to
checkmate if one exists, or leave the field empty and click continue otherwise.

Two accuracy levels are computed:
1. **Detection accuracy** (coarse): Did the participant correctly distinguish
   checkmate from non-checkmate boards? A response on a checkmate board counts
   as a hit; leaving a non-checkmate board empty counts as a correct rejection.
2. **Move accuracy** (fine): Among checkmate boards where the participant
   responded, did they identify a correct first move? Correct moves were
   determined via Stockfish analysis (stored in stimuli.tsv:correct_moves).
   Move matching handles missing disambiguation, check/capture symbols, and
   Dutch notation (D=Q, T=R, L=B, H=N).

Note: Two participants (sub-07, sub-39) have no familiarisation data
(no Pavlovia record found) and are excluded.

Data
----
- Participants: All subjects in participants.tsv with familiarisation data
  (n=39 of 41).
- Behavioural: BIDS familiarisation TSVs at
  BIDS/sub-XX/beh/sub-XX_task-familiarisation_beh.tsv
- Stimulus metadata: stimuli.tsv (correct_moves column from Stockfish)

Statistical Tests
-----------------
- One-sample t-test vs 50% chance (detection) per group
- Welch two-sample t-test comparing expert vs novice accuracy
- Cohen's d effect size (pooled SD)

Outputs
-------
- familiarisation_subject_accuracy.csv: Per-subject accuracy table
- familiarisation_group_stats.csv: Group-level summary statistics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Enable repo root imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end

from modules.io import load_familiarisation_data, MISSING_SUBJECTS


# ============================================================================
# Setup
# ============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="familiarisation_accuracy",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
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
# Per-subject accuracy
# ============================================================================

logger.info("Computing per-subject accuracy...")

subject_rows = []
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

    subject_rows.append({
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

subj_df = pd.DataFrame(subject_rows)
subj_df.to_csv(out_dir / 'familiarisation_subject_accuracy.csv', index=False)
logger.info(f"Saved per-subject accuracy ({len(subj_df)} subjects)")

# ============================================================================
# Group-level statistics
# ============================================================================

logger.info("Computing group-level statistics...")

experts = subj_df[subj_df['group'] == 'expert']
novices = subj_df[subj_df['group'] == 'novice']


def group_stats(df, metric, label):
    """Compute descriptive and inferential stats for one metric."""
    vals = df[metric].dropna()
    n = len(vals)
    if n == 0:
        return None

    result = {
        'metric': label,
        'group': df['group'].iloc[0] if len(df) > 0 else 'unknown',
        'n': n,
        'mean': vals.mean(),
        'sd': vals.std(),
        'median': vals.median(),
        'min': vals.min(),
        'max': vals.max(),
    }

    # One-sample t-test vs 0.5 (chance)
    if n > 1:
        t, p = stats.ttest_1samp(vals, 0.5)
        result['t_vs_chance'] = t
        result['p_vs_chance'] = p

    return result


def comparison_stats(e_vals, n_vals, label):
    """Welch t-test and Cohen's d comparing two groups."""
    e = e_vals.dropna()
    n = n_vals.dropna()
    if len(e) < 2 or len(n) < 2:
        return None

    t, p = stats.ttest_ind(e, n, equal_var=False)
    pooled_sd = np.sqrt((e.std()**2 + n.std()**2) / 2)
    d = (e.mean() - n.mean()) / pooled_sd if pooled_sd > 0 else np.nan

    return {
        'metric': label,
        'group': 'expert_vs_novice',
        'n': f"{len(e)}/{len(n)}",
        'mean': f"{e.mean():.3f}/{n.mean():.3f}",
        'sd': f"{e.std():.3f}/{n.std():.3f}",
        't_welch': t,
        'p_welch': p,
        'cohens_d': d,
    }


stats_rows = []

for metric, label in [
    ('detection_acc', 'Detection accuracy (all boards)'),
    ('move_acc_all_cm', 'Move accuracy (all checkmate boards)'),
    ('move_acc_responded', 'Move accuracy (responded checkmate boards)'),
]:
    for grp_df in [experts, novices]:
        s = group_stats(grp_df, metric, label)
        if s:
            stats_rows.append(s)

    comp = comparison_stats(experts[metric], novices[metric], label)
    if comp:
        stats_rows.append(comp)

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(out_dir / 'familiarisation_group_stats.csv', index=False)

# ============================================================================
# Print summary
# ============================================================================

logger.info("")
logger.info("=" * 72)
logger.info("FAMILIARISATION TASK RESULTS")
logger.info("=" * 72)

for metric, label in [
    ('detection_acc', 'Detection accuracy (all boards)'),
    ('move_acc_all_cm', 'Move accuracy (all checkmate boards)'),
    ('move_acc_responded', 'Move accuracy (responded CM boards)'),
]:
    logger.info("")
    logger.info(f"--- {label} ---")
    for grp_name, grp_df in [('Expert', experts), ('Novice', novices)]:
        vals = grp_df[metric].dropna()
        if len(vals) > 0:
            msg = f"  {grp_name}: M={vals.mean():.3f}, SD={vals.std():.3f}, n={len(vals)}"
            if len(vals) > 1:
                t, p = stats.ttest_1samp(vals, 0.5)
                msg += f" (vs 50%: t={t:.2f}, p={p:.4f})"
            logger.info(msg)

    e = experts[metric].dropna()
    n = novices[metric].dropna()
    if len(e) > 1 and len(n) > 1:
        t, p = stats.ttest_ind(e, n, equal_var=False)
        pooled_sd = np.sqrt((e.std()**2 + n.std()**2) / 2)
        d = (e.mean() - n.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        logger.info(f"  Group: t={t:.2f}, p={p:.4f}, d={d:.2f}")

logger.info("")
logger.info("Per-subject summary:")
logger.info(subj_df[['participant_id', 'group', 'detection_acc',
                      'move_acc_all_cm', 'move_acc_responded']].to_string(index=False))

# ============================================================================
# Per-board accuracy
# ============================================================================

logger.info("")
logger.info("Computing per-board accuracy matrix...")

# Build subject x board response matrix
cm_data = data[data['is_checkmate'] == 1].copy()
response_matrix = cm_data.pivot_table(
    index='participant_id', columns='stim_id', values='move_correct',
    aggfunc='first'
).fillna(0)
response_matrix.to_csv(out_dir / 'familiarisation_response_matrix.csv')
logger.info(f"Response matrix: {response_matrix.shape[0]} subjects x {response_matrix.shape[1]} boards")

# Full response matrix including NC boards (responded vs not)
full_matrix = data.pivot_table(
    index='participant_id', columns='stim_id', values='detection_correct',
    aggfunc='first'
).fillna(0)
full_matrix.to_csv(out_dir / 'familiarisation_detection_matrix.csv')

log_script_end(logger)
