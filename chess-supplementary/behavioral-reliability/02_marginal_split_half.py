#!/usr/bin/env python3
"""
Marginal Split-Half Reliability of Board Selection Frequencies

METHODS
=======

Overview
--------
This analysis complements the pairwise RDM split-half (script 01) by
computing the split-half reliability of **marginal board selection
frequencies** — the per-board rate at which each stimulus is chosen as
"current preferred" when it appears in the 1-back task, averaged across
all opponents.

Rationale
---------
The 1-back design produces two levels of behavioral data:

1. **Pairwise**: each specific pair (i, j) is compared ~7 times per
   half-sample. The RDM split-half tests this sparse pairwise structure.
2. **Marginal**: each board appears ~160 times per half-sample (across all
   opponents). The marginal split-half tests this dense per-board structure.

The supplementary analyses (checkmate preference, C-NC within-pair
correlation) are built on marginal frequencies, so the marginal split-half
is the appropriate reliability measure for those analyses.

Procedure
---------
For each group (experts, novices) and for each of 1000 random participant
splits (same bootstrap approach as script 01):

1. Split participants into two halves.
2. For each half, compute per-board selection frequency: how often each of
   the 40 boards is chosen as "current preferred" when it appears.
3. Correlate the 40-element frequency vectors between halves (Spearman).
4. Apply Spearman-Brown correction.

Additionally, for each split:
- Compute the number of pairwise comparisons per pair (for the frequency
  distribution panel).
- Compute the number of board presentations per board.

Data
----
- BIDS events at CONFIG['BIDS_ROOT']/sub-XX/func/*_events.tsv

Outputs
-------
- marginal_split_half_summary.csv: Bootstrap CIs and p-values
- marginal_bootstrap_experts.csv, _novices.csv: Per-iteration r values
- pair_frequency_experts.csv, _novices.csv: Per-pair comparison counts
- board_presentations_experts.csv, _novices.csv: Per-board presentation counts
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'chess-behavioral'))

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_participants_with_expertise
from modules.data_loading import load_participant_trial_data
from modules.rdm_utils import create_pairwise_df, aggregate_pairwise_counts

config, out_dir, logger = setup_analysis(
    analysis_name="marginal_split_half",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)

BIDS_ROOT = Path(CONFIG['BIDS_ROOT'])
stim_df = pd.read_csv(CONFIG['STIMULI_FILE'], sep='\t')
check_ids = set(stim_df.loc[stim_df['check_status'] == 'checkmate', 'stim_id'].astype(int))
N_STIM = 40
N_BOOT = 1000


def spearman_brown(r):
    return 2 * r / (1 + r) if r != -1 else np.nan


def bootstrap_p(samples, null=0.0):
    n = len(samples)
    le = (np.sum(samples <= null) + 1) / (n + 1)
    ge = (np.sum(samples >= null) + 1) / (n + 1)
    return float(2.0 * min(le, ge))


# ============================================================================
# Load all data
# ============================================================================

logger.info("=" * 60)
logger.info("LOADING DATA")
logger.info("=" * 60)

participants_list, _ = get_participants_with_expertise()

# Per-subject: pairwise data (for pair frequency) and trial events (for marginal)
data_by_group = {'expert': {'pw': [], 'ev': [], 'ids': []},
                 'novice': {'pw': [], 'ev': [], 'ids': []}}

for sub_id, is_expert in participants_list:
    group = 'expert' if is_expert else 'novice'
    trials = load_participant_trial_data(sub_id, is_expert, bids_root=BIDS_ROOT)
    if trials is None:
        continue
    pw = create_pairwise_df(trials)
    if len(pw) == 0:
        continue
    pw['sub_id'] = sub_id
    data_by_group[group]['pw'].append(pw)
    data_by_group[group]['ids'].append(sub_id)

    ev_rows = []
    for _, row in trials.iterrows():
        pref = str(row.get('preference', 'n/a'))
        if pref in ('n/a', 'nan'):
            continue
        ev_rows.append({
            'sub_id': sub_id,
            'stim_id': int(row['stim_id']),
            'chosen': 1 if pref == 'current_preferred' else 0,
        })
    data_by_group[group]['ev'].extend(ev_rows)
    logger.info(f"  {sub_id} ({group}): {len(pw)} pairwise, {len(ev_rows)} events")


# ============================================================================
# Bootstrap marginal split-half
# ============================================================================

summary_rows = []

for group_name in ['expert', 'novice']:
    gd = data_by_group[group_name]
    ids = gd['ids']
    all_pw = pd.concat(gd['pw'], ignore_index=True)
    ev_df = pd.DataFrame(gd['ev'])

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{group_name.upper()} (n={len(ids)})")
    logger.info(f"{'=' * 60}")

    rng = np.random.RandomState(CONFIG['RANDOM_SEED'])
    r_marginal = []

    for it in range(N_BOOT):
        sh = rng.permutation(len(ids))
        h1_ids = [ids[i] for i in sh[:len(ids) // 2]]
        h2_ids = [ids[i] for i in sh[len(ids) // 2:]]

        h1_ev = ev_df[ev_df['sub_id'].isin(h1_ids)]
        h2_ev = ev_df[ev_df['sub_id'].isin(h2_ids)]

        h1_freq = h1_ev.groupby('stim_id')['chosen'].mean()
        h2_freq = h2_ev.groupby('stim_id')['chosen'].mean()
        common = sorted(set(h1_freq.index) & set(h2_freq.index))

        if len(common) >= 10:
            r, _ = stats.spearmanr(h1_freq[common], h2_freq[common])
            r_marginal.append(r)

        if (it + 1) % 250 == 0:
            logger.info(f"  Bootstrap {it + 1}/{N_BOOT}...")

    r_arr = np.array(r_marginal)
    r_full = np.array([spearman_brown(r) for r in r_arr])

    p_boot = bootstrap_p(r_full, null=0.0)

    logger.info(f"\n  Marginal split-half:")
    logger.info(f"    r_half: M={r_arr.mean():.4f} "
                f"[{np.percentile(r_arr, 2.5):.4f}, {np.percentile(r_arr, 97.5):.4f}]")
    logger.info(f"    r_full (SB): M={r_full.mean():.4f} "
                f"[{np.percentile(r_full, 2.5):.4f}, {np.percentile(r_full, 97.5):.4f}]")
    logger.info(f"    p_boot = {p_boot:.4e}")

    pd.DataFrame({'r_half': r_arr, 'r_full': r_full}).to_csv(
        out_dir / f"marginal_bootstrap_{group_name}s.csv", index=False)

    summary_rows.append({
        'group': group_name,
        'n_iterations': len(r_arr),
        'r_half_mean': r_arr.mean(),
        'r_half_ci': f"[{np.percentile(r_arr, 2.5):.4f}, {np.percentile(r_arr, 97.5):.4f}]",
        'r_full_mean': r_full.mean(),
        'r_full_ci': f"[{np.percentile(r_full, 2.5):.4f}, {np.percentile(r_full, 97.5):.4f}]",
        'r_full_p_boot': p_boot,
    })

    # ── One canonical split (seed=42) for per-half distributions ─────
    # All per-half statistics (pair frequency, board presentations, marginal
    # scatter) are computed from the SAME split for internal consistency.
    TRI = np.tril_indices(N_STIM, k=-1)
    pair_is_cross = np.array([
        (TRI[0][k] + 1 in check_ids) != (TRI[1][k] + 1 in check_ids)
        for k in range(len(TRI[0]))
    ])

    sh0 = np.random.RandomState(CONFIG['RANDOM_SEED']).permutation(len(ids))
    h1_ids = [ids[i] for i in sh0[:len(ids) // 2]]
    h2_ids = [ids[i] for i in sh0[len(ids) // 2:]]

    # ── Pair frequency per half ──────────────────────────────────────
    h1_pw = all_pw[all_pw['sub_id'].isin(h1_ids)]
    h2_pw = all_pw[all_pw['sub_id'].isin(h2_ids)]

    def _pair_freq(pw_subset):
        agg = aggregate_pairwise_counts([pw_subset])
        cmat = np.zeros((N_STIM, N_STIM), dtype=int)
        if 'count' in agg.columns:
            for _, r in agg.iterrows():
                cmat[int(r['better']) - 1, int(r['worse']) - 1] = int(r['count'])
        else:
            for _, r in agg.iterrows():
                cmat[int(r['better']) - 1, int(r['worse']) - 1] += 1
        return (cmat + cmat.T)[TRI]

    h1_pair_freq = _pair_freq(h1_pw)
    h2_pair_freq = _pair_freq(h2_pw)

    # Save average of both halves (representative per-half distribution)
    avg_pair_freq = (h1_pair_freq + h2_pair_freq) / 2.0

    pd.DataFrame({
        'stim_i': TRI[0] + 1, 'stim_j': TRI[1] + 1,
        'h1_frequency': h1_pair_freq,
        'h2_frequency': h2_pair_freq,
        'avg_frequency': avg_pair_freq,
        'is_cross_category': pair_is_cross,
    }).to_csv(out_dir / f"pair_frequency_{group_name}s.csv", index=False)

    logger.info(f"\n  Pair frequency per half: "
                f"H1 M={h1_pair_freq.mean():.1f} [{h1_pair_freq.min()}-{h1_pair_freq.max()}], "
                f"H2 M={h2_pair_freq.mean():.1f} [{h2_pair_freq.min()}-{h2_pair_freq.max()}]")

    # ── Board presentations per half ─────────────────────────────────
    h1_ev = ev_df[ev_df['sub_id'].isin(h1_ids)]
    h2_ev = ev_df[ev_df['sub_id'].isin(h2_ids)]

    h1_board = h1_ev.groupby('stim_id').size().reindex(range(1, N_STIM + 1), fill_value=0)
    h2_board = h2_ev.groupby('stim_id').size().reindex(range(1, N_STIM + 1), fill_value=0)
    board_check = ['checkmate' if s in check_ids else 'non_checkmate'
                    for s in range(1, N_STIM + 1)]

    pd.DataFrame({
        'stim_id': range(1, N_STIM + 1),
        'h1_presentations': h1_board.values,
        'h2_presentations': h2_board.values,
        'check_status': board_check,
    }).to_csv(out_dir / f"board_presentations_{group_name}s.csv", index=False)

    logger.info(f"  Board presentations per half: "
                f"H1 M={h1_board.mean():.0f} [{h1_board.min()}-{h1_board.max()}], "
                f"H2 M={h2_board.mean():.0f} [{h2_board.min()}-{h2_board.max()}]")

    # ── Marginal frequency for scatter (same split) ──────────────────
    h1_freq = h1_ev.groupby('stim_id')['chosen'].mean()
    h2_freq = h2_ev.groupby('stim_id')['chosen'].mean()
    common = sorted(set(h1_freq.index) & set(h2_freq.index))

    scatter_df = pd.DataFrame({
        'stim_id': common,
        'half1_freq': [h1_freq[s] for s in common],
        'half2_freq': [h2_freq[s] for s in common],
        'check_status': ['checkmate' if s in check_ids else 'non_checkmate' for s in common],
    })
    scatter_df.to_csv(out_dir / f"marginal_scatter_{group_name}s.csv", index=False)

    r_scatter, _ = stats.spearmanr(scatter_df['half1_freq'], scatter_df['half2_freq'])
    logger.info(f"  Marginal scatter (one split): r={r_scatter:.3f}")

# Save summary
pd.DataFrame(summary_rows).to_csv(out_dir / "marginal_split_half_summary.csv", index=False)

log_script_end(logger)
