#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Engagement Diagnostics -- per-subject stage
=================================================

For each participant, reads BIDS fMRI events, computes four behavioural
diagnostics of task engagement, and writes a single per-subject TSV under

    BIDS/derivatives/task-engagement/sub-XX/
        sub-XX_desc-taskengagement_metrics.tsv

with a root-level sidecar JSON documenting the columns. The pipeline's
``dataset_description.json`` is created on first run.

The subject-level outputs feed into ``11_task_engagement_group.py`` which
runs all group comparisons (Welch t-tests, Fisher z-test, Cohen's d).

Columns in the per-subject TSV:
- subject        : participant identifier (sub-XX)
- group          : expert or novice
- response_rate  : proportion of non-first trials with a valid response
- prop_check_preferred : proportion of visual-pair comparisons where
                         the checkmate board was preferred
- n_pairs        : number of valid visual-pair 1-back comparisons
- prop_transitive : proportion of transitive triples among all testable triples
- n_triples      : number of testable transitive triples
- cnc_r          : per-subject Pearson correlation between checkmate and
                   non-checkmate board selection frequencies within visual pairs
- n_valid_pairs  : number of visual pairs with data for both C and NC boards

Note: sub-04 is excluded (button box malfunction during scanning).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import load_stimulus_metadata


# ============================================================================
# Setup
# ============================================================================

config, _, logger = setup_analysis(
    analysis_name="01_task_engagement_subject",
    results_base=CONFIG["RESULTS_ROOT"] / "supplementary" / "task-engagement" / "logs",
    script_file=__file__,
)

BIDS_ROOT = Path(config['BIDS_ROOT'])
TASK_ENGAGEMENT_ROOT: Path = CONFIG["BIDS_TASK_ENGAGEMENT"]
TASK_ENGAGEMENT_ROOT.mkdir(parents=True, exist_ok=True)

SUBJECT_FILE_SUFFIX = "_desc-taskengagement_metrics.tsv"
SIDECAR_STEM = "desc-taskengagement_metrics"


# ============================================================================
# Pipeline descriptor and root-level sidecar
# ============================================================================

def write_pipeline_descriptor(root: Path) -> None:
    """Write dataset_description.json for the task-engagement pipeline."""
    descriptor = {
        "Name": "task-engagement",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "task-engagement",
                "Description": (
                    "Per-subject behavioural diagnostics from the fMRI "
                    "preference task: response rate, checkmate preference, "
                    "within-subject transitivity, and board preference "
                    "profile (C-NC within-pair correlation)."
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
    """Write the root-level sidecar describing the per-subject TSV columns."""
    sidecar = {
        "Description": (
            "Per-subject task-engagement diagnostics derived from the fMRI "
            "1-back preference task. Each row is one participant; columns "
            "capture response rate, checkmate preference, transitivity, and "
            "board preference profile."
        ),
        "Columns": {
            "subject": {
                "Description": "Participant identifier (sub-XX)."
            },
            "group": {
                "Description": "Expertise group: 'expert' or 'novice'."
            },
            "response_rate": {
                "Description": (
                    "Proportion of non-first trials with a valid preference "
                    "response (i.e., not n/a). First trials in each run are "
                    "excluded because no comparison board has been shown."
                )
            },
            "prop_check_preferred": {
                "Description": (
                    "Among consecutive 1-back visual pairs (checkmate board "
                    "adjacent to its matched non-checkmate partner), how often "
                    "the checkmate board was preferred."
                )
            },
            "n_pairs": {
                "Description": "Number of valid visual-pair 1-back comparisons."
            },
            "prop_transitive": {
                "Description": (
                    "Proportion of transitive triples (A > B > C => A > C?) "
                    "among all testable triples from pairwise preferences."
                )
            },
            "n_triples": {
                "Description": "Number of testable transitive triples."
            },
            "cnc_r": {
                "Description": (
                    "Per-subject Pearson r between checkmate and non-checkmate "
                    "board selection frequencies within visual pairs (20 pairs)."
                )
            },
            "n_valid_pairs": {
                "Description": (
                    "Number of visual pairs with valid frequency data for both "
                    "checkmate and non-checkmate boards."
                )
            },
        },
    }
    (root / f"{SIDECAR_STEM}.json").write_text(json.dumps(sidecar, indent=2) + "\n")


# ============================================================================
# Load participants and stimuli
# ============================================================================

ppt = pd.read_csv(config['BIDS_PARTICIPANTS'], sep="\t")
n_experts = (ppt['group'] == 'expert').sum()
n_novices = (ppt['group'] == 'novice').sum()
logger.info(f"Participants: {len(ppt)} ({n_experts} experts, {n_novices} novices)")

stim = load_stimulus_metadata(return_all=True)
# Column is 'check' after load_stimulus_metadata standardisation
check_col = 'check' if 'check' in stim.columns else 'check_status'

# Build visual pair mapping: each checkmate stim_id -> its NC counterpart
check_to_nc = {}
for v in stim['visual'].unique():
    pair = stim[stim['visual'] == v]
    c_row = pair[pair[check_col] == 'checkmate']
    nc_row = pair[pair[check_col] == 'non_checkmate']
    if len(c_row) == 1 and len(nc_row) == 1:
        check_to_nc[int(c_row['stim_id'].values[0])] = int(nc_row['stim_id'].values[0])

logger.info(f"Visual pairs found: {len(check_to_nc)}")

# ============================================================================
# Load all behavioural events from fMRI task
# ============================================================================

logger.info("Loading fMRI behavioural events from BIDS...")

all_events = []
for sub_dir in sorted(BIDS_ROOT.glob("sub-*")):
    if not sub_dir.is_dir():
        continue
    func_dir = sub_dir / "func"
    if not func_dir.exists():
        continue
    sub_id = sub_dir.name
    for ev_file in sorted(func_dir.glob("*_events.tsv")):
        df = pd.read_csv(ev_file, sep="\t")
        df['subject'] = sub_id
        run_str = ev_file.stem.split("run-")[1].split("_")[0]
        df['run'] = int(run_str)
        all_events.append(df)

events = pd.concat(all_events, ignore_index=True)
events = events.merge(
    ppt[['participant_id', 'group']],
    left_on='subject', right_on='participant_id', how='left',
)
logger.info(f"Total trials: {len(events)}")

# Exclude sub-04: button box malfunction during scanning (no valid responses)
n_before = len(events)
events = events[events['subject'] != 'sub-04'].reset_index(drop=True)
logger.info(f"Excluded sub-04 (button box issue): {n_before - len(events)} trials removed, "
            f"{len(events)} remaining")


# ============================================================================
# Pre-compute shared columns
# ============================================================================

events['has_response'] = events['preference'].notna() & (events['preference'] != 'n/a')

# Mark first trial of each run (no comparison possible)
events['is_first_trial'] = False
for (sub, run), grp in events.groupby(['subject', 'run']):
    events.loc[grp.index[0], 'is_first_trial'] = True

non_first = events[~events['is_first_trial']]
valid_trials = non_first[non_first['has_response']].copy()
valid_trials['current_chosen'] = (valid_trials['preference'] == 'current_preferred').astype(int)

# Per-subject, per-board mean selection frequency (used for board preference)
board_freq = (
    valid_trials
    .groupby(['subject', 'group', 'stim_id'])['current_chosen']
    .mean()
    .reset_index()
    .rename(columns={'current_chosen': 'freq'})
)

# Visual pair lists
c_ids = sorted(check_to_nc.keys())
nc_ids = [check_to_nc[c] for c in c_ids]


# ============================================================================
# Per-subject computation
# ============================================================================

logger.info("Writing pipeline descriptor and sidecar...")
write_pipeline_descriptor(TASK_ENGAGEMENT_ROOT)
write_root_sidecar(TASK_ENGAGEMENT_ROOT)

logger.info("Computing per-subject task engagement metrics...")

all_rows = []
skipped = 0

for sub in sorted(events['subject'].unique()):
    sub_events = events[events['subject'] == sub]
    group = sub_events['group'].iloc[0]

    # ------------------------------------------------------------------
    # 1. Response rate
    # ------------------------------------------------------------------
    sub_non_first = non_first[non_first['subject'] == sub]
    response_rate = sub_non_first['has_response'].mean() if len(sub_non_first) > 0 else np.nan

    # ------------------------------------------------------------------
    # 2. Checkmate preference
    # ------------------------------------------------------------------
    n_pairs = 0
    n_check_preferred = 0

    for run, run_events in sub_events.groupby('run'):
        run_events = run_events.sort_values('onset').reset_index(drop=True)
        for i in range(1, len(run_events)):
            prev_stim = int(run_events.loc[i - 1, 'stim_id'])
            curr_stim = int(run_events.loc[i, 'stim_id'])
            pref = run_events.loc[i, 'preference']

            if pref == 'n/a' or pd.isna(pref):
                continue

            # Check if this is a visual pair (C-NC or NC-C)
            is_pair = False
            curr_is_check = False

            if prev_stim in check_to_nc and check_to_nc[prev_stim] == curr_stim:
                is_pair = True
                curr_is_check = False
            elif curr_stim in check_to_nc and check_to_nc[curr_stim] == prev_stim:
                is_pair = True
                curr_is_check = True

            if is_pair:
                n_pairs += 1
                if pref == 'current_preferred' and curr_is_check:
                    n_check_preferred += 1
                elif pref != 'current_preferred' and not curr_is_check:
                    n_check_preferred += 1

    prop_check_preferred = n_check_preferred / n_pairs if n_pairs > 0 else np.nan

    # ------------------------------------------------------------------
    # 3. Within-subject transitivity
    # ------------------------------------------------------------------
    pref_counts = {}

    for run, run_events in sub_events.groupby('run'):
        run_events = run_events.sort_values('onset').reset_index(drop=True)
        for i in range(1, len(run_events)):
            prev_stim = int(run_events.loc[i - 1, 'stim_id'])
            curr_stim = int(run_events.loc[i, 'stim_id'])
            pref = run_events.loc[i, 'preference']

            if pref == 'n/a' or pd.isna(pref):
                continue

            if pref == 'current_preferred':
                winner, loser = curr_stim, prev_stim
            else:
                winner, loser = prev_stim, curr_stim

            pref_counts[(winner, loser)] = pref_counts.get((winner, loser), 0) + 1

    # Build preference matrix: A > B if A preferred over B more often
    stim_ids = sorted({s for pair in pref_counts for s in pair})
    preferred = {}
    for a in stim_ids:
        for b in stim_ids:
            if a == b:
                continue
            ab = pref_counts.get((a, b), 0)
            ba = pref_counts.get((b, a), 0)
            if ab > ba:
                preferred[(a, b)] = True
            elif ba > ab:
                preferred[(a, b)] = False

    # Count transitive triples
    n_triples = 0
    n_transitive = 0
    for a, b, c in combinations(stim_ids, 3):
        for x, y, z in [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]:
            if preferred.get((x, y)) is True and preferred.get((y, z)) is True:
                n_triples += 1
                if preferred.get((x, z)) is True:
                    n_transitive += 1
                break

    prop_transitive = n_transitive / n_triples if n_triples > 0 else np.nan

    # ------------------------------------------------------------------
    # 4. Board preference profile (C-NC within-pair correlation)
    # ------------------------------------------------------------------
    sub_freq = board_freq[board_freq['subject'] == sub]
    sub_board_map = sub_freq.set_index('stim_id')['freq']

    c_freqs_sub = np.array([sub_board_map.get(c, np.nan) for c in c_ids])
    nc_freqs_sub = np.array([sub_board_map.get(nc, np.nan) for nc in nc_ids])
    valid_mask = ~np.isnan(c_freqs_sub) & ~np.isnan(nc_freqs_sub)

    if valid_mask.sum() >= 3:
        cnc_r, _ = stats.pearsonr(c_freqs_sub[valid_mask], nc_freqs_sub[valid_mask])
    else:
        cnc_r = np.nan

    n_valid_pairs = int(valid_mask.sum())

    all_rows.append({
        'subject': sub,
        'group': group,
        'response_rate': response_rate,
        'n_pairs': n_pairs,
        'n_check_preferred': n_check_preferred,
        'prop_check_preferred': prop_check_preferred,
        'n_triples': n_triples,
        'n_transitive': n_transitive,
        'prop_transitive': prop_transitive,
        'cnc_r': cnc_r,
        'n_valid_pairs': n_valid_pairs,
    })
    logger.info(
        f"  {sub}: response_rate={response_rate:.3f}, "
        f"check_pref={prop_check_preferred:.3f} ({n_pairs} pairs), "
        f"transitivity={prop_transitive:.3f} ({n_triples} triples), "
        f"cnc_r={cnc_r:.3f}"
    )

# Write single stacked TSV (all subjects in one file)
metrics_df = pd.DataFrame(all_rows)
tsv_path = TASK_ENGAGEMENT_ROOT / "task_engagement_metrics.tsv"
metrics_df.to_csv(tsv_path, sep="\t", index=False)
logger.info(
    f"Wrote {len(metrics_df)} subjects to {tsv_path.name} "
    f"({skipped} subjects skipped)."
)
log_script_end(logger)
