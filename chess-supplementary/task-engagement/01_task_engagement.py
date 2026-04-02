#!/usr/bin/env python3
"""
Novice Diagnostics — Supplementary Analysis

METHODS
=======

Overview
--------
This analysis runs four behavioural diagnostics on the fMRI preference task
to characterise how novices engage with the stimuli and whether their
preferences reflect perceptual (visual) or relational (checkmate/strategy)
features. All analyses compare expert and novice groups.

Analyses
--------
1. **Response rate**: Proportion of non-first trials with a valid preference
   response (i.e., not n/a). First trials in each run are excluded because
   no comparison board has yet been shown.

2. **Checkmate preference**: Among consecutive 1-back visual pairs (a
   checkmate board adjacent to its visually matched non-checkmate partner),
   how often is the checkmate board preferred? Tested against 50% chance.

3. **Within-subject transitivity**: For each subject, pairwise preference
   relations are extracted from 1-back comparisons, and transitive triples
   (A > B > C => A > C?) are evaluated. Because the 1-back task yields
   incomplete pairwise data, absolute transitivity values are reported
   descriptively and the inferential focus is the expert-vs-novice group
   comparison.

4. **Board preference profile**: For each group, compute per-board mean
   selection frequency and correlate checkmate vs non-checkmate frequencies
   within visual pairs (20 pairs). A high C-NC correlation indicates that
   perceptual similarity drives preferences (both members of a visual pair
   are liked/disliked together). A low or negative correlation indicates
   relational structure drives preferences.

Data
----
- Participants: All subjects in participants.tsv
- Behavioural: BIDS fMRI events at BIDS/sub-XX/func/*_events.tsv
- Stimulus metadata: stimuli.tsv (visual column for pair mapping)

Statistical Tests
-----------------
- Welch two-sample t-tests comparing expert vs novice
- One-sample t-tests vs interpretable chance baselines (50% for checkmate
  preference)
- Cohen's d effect size (pooled SD)
- Fisher z-test for group difference in C-NC correlations

Outputs
-------
- response_rate.csv: Per-subject response rates
- checkmate_preference.csv: Per-subject checkmate preference proportions
- transitivity.csv: Per-subject transitivity proportions
- board_preference_profile.csv: Per-subject C-NC within-pair correlations
- board_preference_group.csv: Group-level C and NC board frequencies
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Enable repo root imports
from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.stats_utils import (
    compare_independent_groups,
    compute_mean_ci_and_ttest_vs_value,
    fisher_z_test_independent_correlations,
)


# ============================================================================
# Setup
# ============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="novice_diagnostics",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)

BIDS_ROOT = Path(config['BIDS_ROOT'])


# ============================================================================
# Load participants and stimuli
# ============================================================================

ppt = pd.read_csv(config['BIDS_PARTICIPANTS'], sep="\t")
n_experts = (ppt['group'] == 'expert').sum()
n_novices = (ppt['group'] == 'novice').sum()
logger.info(f"Participants: {len(ppt)} ({n_experts} experts, {n_novices} novices)")

from common.bids_utils import load_stimulus_metadata
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
# 1. Response Rate
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("1. RESPONSE RATE")
logger.info("=" * 60)

events['has_response'] = events['preference'].notna() & (events['preference'] != 'n/a')

# Mark first trial of each run (no comparison possible)
events['is_first_trial'] = False
for (sub, run), grp in events.groupby(['subject', 'run']):
    events.loc[grp.index[0], 'is_first_trial'] = True

non_first = events[~events['is_first_trial']]
resp_rate = (
    non_first.groupby(['subject', 'group'])['has_response']
    .mean()
    .reset_index()
    .rename(columns={'has_response': 'response_rate'})
)

for g in ['expert', 'novice']:
    vals = resp_rate.loc[resp_rate['group'] == g, 'response_rate']
    logger.info(
        f"  {g.capitalize()}: M={vals.mean():.3f}, SD={vals.std():.3f}, "
        f"range=[{vals.min():.3f}, {vals.max():.3f}]"
    )

exp_rr = resp_rate.loc[resp_rate['group'] == 'expert', 'response_rate']
nov_rr = resp_rate.loc[resp_rate['group'] == 'novice', 'response_rate']
resp_rate_cmp = compare_independent_groups(exp_rr, nov_rr)
logger.info(
    f"  t({resp_rate_cmp['df']:.1f})={resp_rate_cmp['t_stat']:.3f}, "
    f"p={resp_rate_cmp['p_value']:.4f}, Cohen's d={resp_rate_cmp['cohens_d']:.3f}"
)

resp_rate.to_csv(out_dir / "response_rate.csv", index=False)

# ============================================================================
# 2. Checkmate Preference (visual-pair 1-back comparisons)
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("2. CHECKMATE PREFERENCE (visual-pair 1-back comparisons)")
logger.info("=" * 60)

check_pref_results = []

for sub, sub_events in events.groupby('subject'):
    group = sub_events['group'].iloc[0]
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

    prop = n_check_preferred / n_pairs if n_pairs > 0 else np.nan
    check_pref_results.append({
        'subject': sub, 'group': group,
        'n_pairs': n_pairs, 'n_check_preferred': n_check_preferred,
        'prop_check_preferred': prop,
    })

cp_df = pd.DataFrame(check_pref_results)

for g in ['expert', 'novice']:
    vals = cp_df.loc[cp_df['group'] == g, 'prop_check_preferred'].dropna()
    _, _, _, t_1s, p_1s = compute_mean_ci_and_ttest_vs_value(vals, popmean=0.5)
    mean_pairs = cp_df.loc[cp_df['group'] == g, 'n_pairs'].mean()
    logger.info(
        f"  {g.capitalize()}: M={vals.mean():.3f}, SD={vals.std():.3f}, "
        f"n_pairs_mean={mean_pairs:.1f}"
    )
    logger.info(f"    vs 50%: t={t_1s:.3f}, p={p_1s:.4f}")

exp_cp = cp_df.loc[cp_df['group'] == 'expert', 'prop_check_preferred'].dropna()
nov_cp = cp_df.loc[cp_df['group'] == 'novice', 'prop_check_preferred'].dropna()
checkmate_pref_cmp = compare_independent_groups(exp_cp, nov_cp)
logger.info(
    f"  Group diff: t({checkmate_pref_cmp['df']:.1f})={checkmate_pref_cmp['t_stat']:.3f}, "
    f"p={checkmate_pref_cmp['p_value']:.4f}, Cohen's d={checkmate_pref_cmp['cohens_d']:.3f}"
)

cp_df.to_csv(out_dir / "checkmate_preference.csv", index=False)

# ============================================================================
# 3. Within-Subject Transitivity
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("3. WITHIN-SUBJECT TRANSITIVITY")
logger.info("=" * 60)

trans_results = []

for sub, sub_events in events.groupby('subject'):
    group = sub_events['group'].iloc[0]

    # Build pairwise preference counts
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

    prop_trans = n_transitive / n_triples if n_triples > 0 else np.nan
    trans_results.append({
        'subject': sub, 'group': group,
        'n_triples': n_triples, 'n_transitive': n_transitive,
        'prop_transitive': prop_trans,
    })

tr_df = pd.DataFrame(trans_results)

for g in ['expert', 'novice']:
    vals = tr_df.loc[tr_df['group'] == g, 'prop_transitive'].dropna()
    logger.info(f"  {g.capitalize()}: M={vals.mean():.3f}, SD={vals.std():.3f}")

exp_tr = tr_df.loc[tr_df['group'] == 'expert', 'prop_transitive'].dropna()
nov_tr = tr_df.loc[tr_df['group'] == 'novice', 'prop_transitive'].dropna()
transitivity_cmp = compare_independent_groups(exp_tr, nov_tr)
logger.info(
    f"  Group diff: t({transitivity_cmp['df']:.1f})={transitivity_cmp['t_stat']:.3f}, "
    f"p={transitivity_cmp['p_value']:.4f}, Cohen's d={transitivity_cmp['cohens_d']:.3f}"
)
logger.info(
    "  Absolute transitivity values are descriptive only for the sparse 1-back design; "
    "no one-sample baseline test is reported."
)

tr_df.to_csv(out_dir / "transitivity.csv", index=False)

# ============================================================================
# 4. Board Preference Profile (C-NC within-pair correlation)
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("4. BOARD PREFERENCE PROFILE")
logger.info("=" * 60)

# For each subject, compute per-board selection frequency:
# how often each board is judged "current_preferred" when it appears as the
# current stimulus (excluding first trials and non-responses).
valid_trials = non_first[non_first['has_response']].copy()

# Binary: was the current board preferred?
valid_trials['current_chosen'] = (valid_trials['preference'] == 'current_preferred').astype(int)

# Per-subject, per-board mean selection frequency
board_freq = (
    valid_trials
    .groupby(['subject', 'group', 'stim_id'])['current_chosen']
    .mean()
    .reset_index()
    .rename(columns={'current_chosen': 'freq'})
)

# Build visual pair list: 20 pairs of (C_stim_id, NC_stim_id)
c_ids = sorted(check_to_nc.keys())
nc_ids = [check_to_nc[c] for c in c_ids]

# --- Group-level C-NC correlation ---
group_profile_rows = []

for g in ['expert', 'novice']:
    g_freq = board_freq[board_freq['group'] == g]
    # Mean frequency per board across subjects in this group
    group_board_mean = g_freq.groupby('stim_id')['freq'].mean()

    c_freqs = [group_board_mean.get(c, np.nan) for c in c_ids]
    nc_freqs = [group_board_mean.get(nc, np.nan) for nc in nc_ids]

    c_arr = np.array(c_freqs)
    nc_arr = np.array(nc_freqs)
    valid_mask = ~np.isnan(c_arr) & ~np.isnan(nc_arr)

    if valid_mask.sum() >= 3:
        r, p_val = stats.pearsonr(c_arr[valid_mask], nc_arr[valid_mask])
        rho, p_rho = stats.spearmanr(c_arr[valid_mask], nc_arr[valid_mask])
        logger.info(
            f"  {g.capitalize()} group C-NC: r = {r:.2f}, p = {p_val:.4f}; "
            f"rho = {rho:.2f}, p = {p_rho:.4f} (n_pairs = {valid_mask.sum()})"
        )
        logger.info(
            f"  {g.capitalize()} mean C freq = {np.nanmean(c_arr):.3f}, "
            f"mean NC freq = {np.nanmean(nc_arr):.3f}"
        )

    for i, (c, nc) in enumerate(zip(c_ids, nc_ids)):
        group_profile_rows.append({
            'group': g, 'pair_idx': i,
            'c_stim_id': c, 'nc_stim_id': nc,
            'c_freq': c_freqs[i], 'nc_freq': nc_freqs[i],
        })

group_profile_df = pd.DataFrame(group_profile_rows)
group_profile_df.to_csv(out_dir / "board_preference_group.csv", index=False)

# --- Per-subject C-NC correlation ---
subject_corr_rows = []

for sub in sorted(board_freq['subject'].unique()):
    sub_freq = board_freq[board_freq['subject'] == sub]
    group = sub_freq['group'].iloc[0]
    sub_board_map = sub_freq.set_index('stim_id')['freq']

    c_freqs_sub = np.array([sub_board_map.get(c, np.nan) for c in c_ids])
    nc_freqs_sub = np.array([sub_board_map.get(nc, np.nan) for nc in nc_ids])
    valid_mask = ~np.isnan(c_freqs_sub) & ~np.isnan(nc_freqs_sub)

    if valid_mask.sum() >= 3:
        r_sub, _ = stats.pearsonr(c_freqs_sub[valid_mask], nc_freqs_sub[valid_mask])
    else:
        r_sub = np.nan

    subject_corr_rows.append({
        'subject': sub, 'group': group,
        'cnc_r': r_sub, 'n_valid_pairs': int(valid_mask.sum()),
    })

subject_corr_df = pd.DataFrame(subject_corr_rows)
subject_corr_df.to_csv(out_dir / "board_preference_profile.csv", index=False)

# Report per-subject correlations by group
for g in ['expert', 'novice']:
    vals = subject_corr_df.loc[subject_corr_df['group'] == g, 'cnc_r'].dropna()
    _, _, _, t_1s, p_1s = compute_mean_ci_and_ttest_vs_value(vals, popmean=0.0)
    logger.info(
        f"  {g.capitalize()} per-subject C-NC r: M={vals.mean():.2f}, "
        f"SD={vals.std():.2f}, n={len(vals)}, vs 0: t={t_1s:.2f}, p={p_1s:.4f}"
    )

# Fisher z-test for group difference in group-level correlations
exp_group_freq = board_freq[board_freq['group'] == 'expert'].groupby('stim_id')['freq'].mean()
nov_group_freq = board_freq[board_freq['group'] == 'novice'].groupby('stim_id')['freq'].mean()

exp_c = np.array([exp_group_freq.get(c, np.nan) for c in c_ids])
exp_nc = np.array([exp_group_freq.get(nc, np.nan) for nc in nc_ids])
nov_c = np.array([nov_group_freq.get(c, np.nan) for c in c_ids])
nov_nc = np.array([nov_group_freq.get(nc, np.nan) for nc in nc_ids])

exp_valid = ~np.isnan(exp_c) & ~np.isnan(exp_nc)
nov_valid = ~np.isnan(nov_c) & ~np.isnan(nov_nc)

r_exp, _ = stats.pearsonr(exp_c[exp_valid], exp_nc[exp_valid])
r_nov, _ = stats.pearsonr(nov_c[nov_valid], nov_nc[nov_valid])

z_diff, p_diff = fisher_z_test_independent_correlations(
    r_exp,
    int(exp_valid.sum()),
    r_nov,
    int(nov_valid.sum()),
)
logger.info(f"  Fisher z-test (group C-NC): z={z_diff:.2f}, p={p_diff:.4f}")

# Per-subject group difference (t-test on per-subject correlations)
exp_subj_r = subject_corr_df.loc[subject_corr_df['group'] == 'expert', 'cnc_r'].dropna()
nov_subj_r = subject_corr_df.loc[subject_corr_df['group'] == 'novice', 'cnc_r'].dropna()
subj_corr_cmp = compare_independent_groups(exp_subj_r, nov_subj_r)
logger.info(
    f"  Per-subject C-NC r group diff: t({subj_corr_cmp['df']:.1f})={subj_corr_cmp['t_stat']:.2f}, "
    f"p={subj_corr_cmp['p_value']:.4f}, Cohen's d={subj_corr_cmp['cohens_d']:.2f}"
)

# ============================================================================

log_script_end(logger)
