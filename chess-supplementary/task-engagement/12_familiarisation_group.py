#!/usr/bin/env python3
"""
Familiarisation Task Accuracy -- group stage
=============================================

Reads per-subject familiarisation accuracy from the ``task-engagement``
BIDS derivative (produced by ``02_familiarisation_subject.py``), and
runs all group comparisons:

- One-sample t-test vs 50% chance (detection) per group
- Welch two-sample t-test comparing expert vs novice accuracy
- Cohen's d effect size (pooled SD)

Outputs (under the unified results tree, NOT derivatives):
- familiarisation_group_stats.csv: Group-level summary statistics
"""

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
)


# ============================================================================
# Setup
# ============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="familiarisation_accuracy",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)

TASK_ENGAGEMENT_ROOT: Path = CONFIG["BIDS_TASK_ENGAGEMENT"]


# ============================================================================
# Load per-subject accuracy from derivatives
# ============================================================================

logger.info("Loading per-subject familiarisation accuracy from derivatives...")

acc_path = TASK_ENGAGEMENT_ROOT / "familiarisation_accuracy.tsv"
if not acc_path.exists():
    raise FileNotFoundError(
        f"Accuracy file not found: {acc_path}. "
        "Run 02_familiarisation_subject.py first."
    )

subj_df = pd.read_csv(acc_path, sep="\t")
n_experts = (subj_df['group'] == 'expert').sum()
n_novices = (subj_df['group'] == 'novice').sum()
logger.info(f"Loaded accuracy for {len(subj_df)} subjects "
            f"({n_experts} experts, {n_novices} novices)")


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
        _, _, _, t, p = compute_mean_ci_and_ttest_vs_value(vals, popmean=0.5)
        result['t_vs_chance'] = t
        result['p_vs_chance'] = p

    return result


def comparison_stats(e_vals, n_vals, label):
    """Welch t-test and Cohen's d comparing two groups."""
    e = np.asarray(e_vals.dropna(), dtype=float)
    n = np.asarray(n_vals.dropna(), dtype=float)
    if len(e) < 2 or len(n) < 2:
        return None

    comparison = compare_independent_groups(e, n)

    return {
        'metric': label,
        'group': 'expert_vs_novice',
        'n': f"{comparison['n_group1']}/{comparison['n_group2']}",
        'mean': f"{comparison['mean1']:.3f}/{comparison['mean2']:.3f}",
        'sd': f"{e.std(ddof=1):.3f}/{n.std(ddof=1):.3f}",
        't_welch': comparison['t_stat'],
        'p_welch': comparison['p_value'],
        'cohens_d': comparison['cohens_d'],
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
                _, _, _, t, p = compute_mean_ci_and_ttest_vs_value(vals, popmean=0.5)
                msg += f" (vs 50%: t={t:.2f}, p={p:.4f})"
            logger.info(msg)

    e = experts[metric].dropna()
    n = novices[metric].dropna()
    if len(e) > 1 and len(n) > 1:
        comparison = compare_independent_groups(e, n)
        logger.info(
            f"  Group: t={comparison['t_stat']:.2f}, "
            f"p={comparison['p_value']:.4f}, d={comparison['cohens_d']:.2f}"
        )

logger.info("")
logger.info("Per-subject summary:")
logger.info(subj_df[['participant_id', 'group', 'detection_acc',
                      'move_acc_all_cm', 'move_acc_responded']].to_string(index=False))

log_script_end(logger)
