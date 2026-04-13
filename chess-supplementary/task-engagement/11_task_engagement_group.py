#!/usr/bin/env python3
"""
Novice Diagnostics -- group stage
==================================

Reads per-subject task engagement metrics from the ``task-engagement``
BIDS derivative (produced by ``01_task_engagement_subject.py``), and
runs all group comparisons:

- Welch two-sample t-tests comparing expert vs novice
- One-sample t-tests vs interpretable chance baselines (50% for checkmate
  preference)
- Cohen's d effect size (pooled SD)
- Fisher z-test for group difference in C-NC correlations
- Group-level board preference profile (C-NC correlation from averaged
  board frequencies)

Outputs (under the unified results tree, NOT derivatives):
- response_rate.csv: Anonymous per-subject response rates (group + metric)
- checkmate_preference.csv: Anonymous per-subject checkmate preference
- transitivity.csv: Anonymous per-subject transitivity metrics
- board_preference_profile.csv: Anonymous per-subject C-NC correlations
- board_preference_group.csv: Group-level C and NC board frequencies
- condition_stats_anonymous.csv: Per-subject per-condition response and
  choice rates (no subject IDs)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Enable repo root imports
from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import load_stimulus_metadata
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
TASK_ENGAGEMENT_ROOT: Path = CONFIG["BIDS_TASK_ENGAGEMENT"]


# ============================================================================
# Load per-subject metrics from derivatives
# ============================================================================

logger.info("Loading per-subject task engagement metrics from derivatives...")

metrics_path = TASK_ENGAGEMENT_ROOT / "task_engagement_metrics.tsv"
if not metrics_path.exists():
    raise FileNotFoundError(
        f"Metrics file not found: {metrics_path}. "
        "Run 01_task_engagement_subject.py first."
    )

metrics = pd.read_csv(metrics_path, sep="\t")
n_experts = (metrics['group'] == 'expert').sum()
n_novices = (metrics['group'] == 'novice').sum()
logger.info(f"Loaded metrics for {len(metrics)} subjects "
            f"({n_experts} experts, {n_novices} novices)")


# ============================================================================
# 1. Response Rate — group comparison
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("1. RESPONSE RATE")
logger.info("=" * 60)

for g in ['expert', 'novice']:
    vals = metrics.loc[metrics['group'] == g, 'response_rate']
    logger.info(
        f"  {g.capitalize()}: M={vals.mean():.3f}, SD={vals.std():.3f}, "
        f"range=[{vals.min():.3f}, {vals.max():.3f}]"
    )

exp_rr = metrics.loc[metrics['group'] == 'expert', 'response_rate']
nov_rr = metrics.loc[metrics['group'] == 'novice', 'response_rate']
resp_rate_cmp = compare_independent_groups(exp_rr, nov_rr)
logger.info(
    f"  t({resp_rate_cmp['df']:.1f})={resp_rate_cmp['t_stat']:.3f}, "
    f"p={resp_rate_cmp['p_value']:.4f}, Cohen's d={resp_rate_cmp['cohens_d']:.3f}"
)


# ============================================================================
# 2. Checkmate Preference — group comparison
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("2. CHECKMATE PREFERENCE (visual-pair 1-back comparisons)")
logger.info("=" * 60)

for g in ['expert', 'novice']:
    vals = metrics.loc[metrics['group'] == g, 'prop_check_preferred'].dropna()
    _, _, _, t_1s, p_1s = compute_mean_ci_and_ttest_vs_value(vals, popmean=0.5)
    mean_pairs = metrics.loc[metrics['group'] == g, 'n_pairs'].mean()
    logger.info(
        f"  {g.capitalize()}: M={vals.mean():.3f}, SD={vals.std():.3f}, "
        f"n_pairs_mean={mean_pairs:.1f}"
    )
    logger.info(f"    vs 50%: t={t_1s:.3f}, p={p_1s:.4f}")

exp_cp = metrics.loc[metrics['group'] == 'expert', 'prop_check_preferred'].dropna()
nov_cp = metrics.loc[metrics['group'] == 'novice', 'prop_check_preferred'].dropna()
checkmate_pref_cmp = compare_independent_groups(exp_cp, nov_cp)
logger.info(
    f"  Group diff: t({checkmate_pref_cmp['df']:.1f})={checkmate_pref_cmp['t_stat']:.3f}, "
    f"p={checkmate_pref_cmp['p_value']:.4f}, Cohen's d={checkmate_pref_cmp['cohens_d']:.3f}"
)


# ============================================================================
# 3. Within-Subject Transitivity — group comparison
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("3. WITHIN-SUBJECT TRANSITIVITY")
logger.info("=" * 60)

for g in ['expert', 'novice']:
    vals = metrics.loc[metrics['group'] == g, 'prop_transitive'].dropna()
    logger.info(f"  {g.capitalize()}: M={vals.mean():.3f}, SD={vals.std():.3f}")

exp_tr = metrics.loc[metrics['group'] == 'expert', 'prop_transitive'].dropna()
nov_tr = metrics.loc[metrics['group'] == 'novice', 'prop_transitive'].dropna()
transitivity_cmp = compare_independent_groups(exp_tr, nov_tr)
logger.info(
    f"  Group diff: t({transitivity_cmp['df']:.1f})={transitivity_cmp['t_stat']:.3f}, "
    f"p={transitivity_cmp['p_value']:.4f}, Cohen's d={transitivity_cmp['cohens_d']:.3f}"
)
logger.info(
    "  Absolute transitivity values are descriptive only for the sparse 1-back design; "
    "no one-sample baseline test is reported."
)


# ============================================================================
# 4. Board Preference Profile (C-NC within-pair correlation) — group stats
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("4. BOARD PREFERENCE PROFILE")
logger.info("=" * 60)

# ---- Load stimuli to build visual pair mapping (needed for group-level
#      board frequency correlation) ----
stim = load_stimulus_metadata(return_all=True)
check_col = 'check' if 'check' in stim.columns else 'check_status'

check_to_nc = {}
for v in stim['visual'].unique():
    pair = stim[stim['visual'] == v]
    c_row = pair[pair[check_col] == 'checkmate']
    nc_row = pair[pair[check_col] == 'non_checkmate']
    if len(c_row) == 1 and len(nc_row) == 1:
        check_to_nc[int(c_row['stim_id'].values[0])] = int(nc_row['stim_id'].values[0])

c_ids = sorted(check_to_nc.keys())
nc_ids = [check_to_nc[c] for c in c_ids]

# ---- Re-compute board selection frequencies from BIDS events for the
#      group-level C-NC correlation (needed for Fisher z-test on group
#      means and the group profile CSV). ----
ppt = pd.read_csv(config['BIDS_PARTICIPANTS'], sep="\t")

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
events = events[events['subject'] != 'sub-04'].reset_index(drop=True)

events['has_response'] = events['preference'].notna() & (events['preference'] != 'n/a')
events['is_first_trial'] = False
for (sub, run), grp in events.groupby(['subject', 'run']):
    events.loc[grp.index[0], 'is_first_trial'] = True

non_first = events[~events['is_first_trial']]
valid_trials = non_first[non_first['has_response']].copy()
valid_trials['current_chosen'] = (valid_trials['preference'] == 'current_preferred').astype(int)

board_freq = (
    valid_trials
    .groupby(['subject', 'group', 'stim_id'])['current_chosen']
    .mean()
    .reset_index()
    .rename(columns={'current_chosen': 'freq'})
)

# --- Group-level C-NC correlation ---
group_profile_rows = []

for g in ['expert', 'novice']:
    g_freq = board_freq[board_freq['group'] == g]
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

# --- Per-subject C-NC correlation: group comparison ---
# Report per-subject correlations from derivatives
for g in ['expert', 'novice']:
    vals = metrics.loc[metrics['group'] == g, 'cnc_r'].dropna()
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
exp_subj_r = metrics.loc[metrics['group'] == 'expert', 'cnc_r'].dropna()
nov_subj_r = metrics.loc[metrics['group'] == 'novice', 'cnc_r'].dropna()
subj_corr_cmp = compare_independent_groups(exp_subj_r, nov_subj_r)
logger.info(
    f"  Per-subject C-NC r group diff: t({subj_corr_cmp['df']:.1f})={subj_corr_cmp['t_stat']:.2f}, "
    f"p={subj_corr_cmp['p_value']:.4f}, Cohen's d={subj_corr_cmp['cohens_d']:.2f}"
)

# ============================================================================
# 5. Per-condition stats (anonymous, for plotting)
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("5. PER-CONDITION STATS (anonymous CSV for plotting)")
logger.info("=" * 60)

check_ids_set = set(check_to_nc.keys())

cond_rows = []
for sub_id in events['subject'].unique():
    sub_ev = events[events['subject'] == sub_id]
    group = sub_ev['group'].iloc[0]
    for run, grp in sub_ev.groupby('run'):
        grp = grp.sort_values('onset').reset_index(drop=True)
        for i in range(1, len(grp)):
            curr = int(grp.iloc[i]['stim_id'])
            prev = int(grp.iloc[i - 1]['stim_id'])
            pref = str(grp.iloc[i]['preference'])
            c_c = curr in check_ids_set
            c_p = prev in check_ids_set
            has = pref not in ('n/a', 'nan') and pd.notna(grp.iloc[i]['preference'])

            if c_c and not c_p:
                cond = 'Advantageous'
            elif not c_c and c_p:
                cond = 'Non-advantageous'
            else:
                cond = 'Same status'

            cond_rows.append({
                'subject': sub_id, 'group': group, 'condition': cond,
                'responded': int(has),
                'chose_current': int(has and pref == 'current_preferred'),
            })

cond_df = pd.DataFrame(cond_rows)

# Per-subject per-condition aggregation
sub_cond = (cond_df.groupby(['subject', 'group', 'condition'])
            .agg(n_trials=('responded', 'count'),
                 n_responded=('responded', 'sum'),
                 n_current=('chose_current', 'sum'))
            .reset_index())
sub_cond['resp_rate'] = sub_cond['n_responded'] / sub_cond['n_trials']
sub_cond['p_current'] = sub_cond['n_current'] / sub_cond['n_responded'].clip(lower=1)

# Save anonymous version (drop subject column)
sub_cond[['group', 'condition', 'resp_rate', 'p_current']].to_csv(
    out_dir / "condition_stats_anonymous.csv", index=False)
logger.info("  Wrote condition_stats_anonymous.csv")

# ============================================================================

log_script_end(logger)
