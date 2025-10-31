#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split-Half Reliability Analysis of Behavioral RDMs

METHODS
=======

Rationale
---------
Representational similarity analysis requires reliable measurements of
similarity structure. To assess the internal consistency of our behavioral
RDMs, we perform a split-half reliability analysis with Spearman–Brown
correction. We quantify uncertainty using non-parametric bootstrap across
random participant splits and keep inference internally consistent: bootstrap
confidence intervals and bootstrap p-values (no parametric t-tests on
bootstrap replicates).

Data
----
We analyze pairwise preference judgments from N=40 participants (20 experts,
20 novices). On each trial, participants indicate which of two consecutively
presented chess positions they prefer, yielding pairwise comparisons between
all 40 stimulus boards for each participant.

Split-Half Procedure
--------------------
For each participant group (experts, novices) separately:

1. Participant Splitting: Randomly split participants within each group into
   two halves (n=10 per half). Repeat for 1,000 random splits to build a
   sampling distribution of the reliability estimator.

2. Half-Sample RDMs: For each half, aggregate pairwise preference counts and
   compute a 40×40 behavioral RDM.

3. Split-Half Correlation: Compute Spearman correlation between the two half
   RDMs; this is the half-sample reliability r_half.

4. Spearman–Brown Correction: Estimate full-sample reliability using the
   prophecy formula r_full = (2 × r_half) / (1 + r_half).

5. Bootstrap Intervals and p-values: Using the distribution of r_full across
   the 1,000 random splits, compute 95% CIs via the percentile method (2.5th
   and 97.5th percentiles). For significance vs. zero, compute a two-sided
   bootstrap p-value using the sign-proportion approach with a +1 correction:

       p_boot = 2 × min{ (#{r_full ≤ 0} + 1) / (n + 1),
                         (#{r_full ≥ 0} + 1) / (n + 1) }.

Between-Group Similarity and Group Difference
--------------------------------------------
- Between-group similarity (Experts vs Novices) is computed by correlating an
  expert half-RDM with a novice half-RDM on each iteration, then applying the
  Spearman–Brown correction; report bootstrap CI and p_boot as above.
- Group difference in reliability is assessed by the bootstrap distribution of
  Δ = r_full(experts) − r_full(novices) computed across iterations; report the
  percentile 95% CI and a two-sided bootstrap p-value for Δ vs 0 using the same
  sign-proportion rule.

Interpretation
--------------
Following conventional psychometric guidelines:
- r_full ≥ 0.90: Excellent reliability
- r_full ≥ 0.80: Good reliability
- r_full ≥ 0.70: Acceptable reliability
- r_full < 0.70: Questionable reliability

Outputs
-------
All results are saved to results/<timestamp>_behavioral_split_half/:
- reliability_metrics.pkl: Full reliability statistics for table generation
- reliability_summary.csv: Human-readable summary (bootstrap CIs and p_boot)
- split_rdm_distributions.npz: Bootstrap distributions
- 01_behavioral_split_half_reliability.py: Copy of this script
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple
from scipy import stats

# Get absolute paths
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
repo_root = os.path.abspath(os.path.join(script_dir, '../..'))

# Import LOCAL modules FIRST (before adding chess-behavioral to path)
local_modules_path = os.path.join(script_dir, 'modules')
sys.path.insert(0, local_modules_path)
from split_half_utils import (
    spearman_brown_correction,
    bootstrap_split_half_reliability,
)
sys.path.pop(0)  # Remove local modules path

# Add repo root to sys.path
sys.path.insert(0, repo_root)

# Add chess-behavioral to path (for modules.data_loading, modules.rdm_utils)
chess_behavioral_dir = os.path.join(repo_root, 'chess-behavioral')
sys.path.insert(0, chess_behavioral_dir)

from common import (
    CONFIG,
    setup_analysis,
    log_script_end,
    get_participants_with_expertise,
)

# Reuse behavioral data loading from chess-behavioral
from modules.data_loading import load_participant_trial_data
from modules.rdm_utils import create_pairwise_df, aggregate_pairwise_counts, compute_symmetric_rdm

# Reuse RSA correlation from common
from common.rsa_utils import correlate_rdms


# ============================================================================
# Helper Functions
# ============================================================================
# Deduplicated: reliability utilities are imported from modules/split_half_utils


def bootstrap_between_group_reliability(
    pairwise_data_group1: pd.DataFrame,
    participant_ids_group1: List[str],
    pairwise_data_group2: pd.DataFrame,
    participant_ids_group2: List[str],
    n_iterations: int = 10000,
    method: str = 'spearman',
    random_state: int = 42,
    participant_col: str = 'sub_id'
) -> Dict:
    """
    Bootstrap between-group reliability by comparing random halves from two groups.

    For each iteration, randomly sample half of participants from each group,
    compute RDMs from pooled data, and correlate them.

    Parameters
    ----------
    pairwise_data_group1, pairwise_data_group2 : pd.DataFrame
        Pairwise comparison data for each group
    participant_ids_group1, participant_ids_group2 : List[str]
        Participant IDs for each group
    n_iterations : int
        Number of bootstrap iterations
    method : str
        Correlation method ('spearman' or 'pearson')
    random_state : int
        Random seed for reproducibility
    participant_col : str
        Column name for participant IDs

    Returns
    -------
    Dict with r_half, r_full arrays and summary statistics
    """
    rng = np.random.RandomState(random_state)

    half_size_g1 = len(participant_ids_group1) // 2
    half_size_g2 = len(participant_ids_group2) // 2

    r_half_values = []
    r_full_values = []

    for i in range(n_iterations):
        # Random sample from each group
        g1_indices = rng.choice(len(participant_ids_group1), size=half_size_g1, replace=False)
        g2_indices = rng.choice(len(participant_ids_group2), size=half_size_g2, replace=False)

        g1_ids = [participant_ids_group1[idx] for idx in g1_indices]
        g2_ids = [participant_ids_group2[idx] for idx in g2_indices]

        # Filter pairwise data for selected participants
        g1_data = pairwise_data_group1[pairwise_data_group1[participant_col].isin(g1_ids)]
        g2_data = pairwise_data_group2[pairwise_data_group2[participant_col].isin(g2_ids)]

        # Aggregate and compute RDMs
        g1_agg = aggregate_pairwise_counts([g1_data])
        g2_agg = aggregate_pairwise_counts([g2_data])

        rdm_g1 = compute_symmetric_rdm(g1_agg)
        rdm_g2 = compute_symmetric_rdm(g2_agg)

        # Correlate between groups
        r, p, ci_l, ci_u = correlate_rdms(rdm_g1, rdm_g2, method=method)

        # Apply Spearman-Brown correction
        r_corrected = spearman_brown_correction(r)

        r_half_values.append(r)
        r_full_values.append(r_corrected)

    # Compute summary statistics
    r_half_arr = np.array(r_half_values)
    r_full_arr = np.array(r_full_values)

    return {
        'r_half': r_half_arr,
        'r_full': r_full_arr,
        'mean_r_half': float(np.mean(r_half_arr)),
        'mean_r_full': float(np.mean(r_full_arr)),
        'ci_r_half': (
            float(np.percentile(r_half_arr, 2.5)),
            float(np.percentile(r_half_arr, 97.5))
        ),
        'ci_r_full': (
            float(np.percentile(r_full_arr, 2.5)),
            float(np.percentile(r_full_arr, 97.5))
        ),
    }


def _bootstrap_p_two_sided(samples: np.ndarray, null: float = 0.0) -> float:
    """Two-sided bootstrap p-value by sign-proportion with +1 correction.

    p = 2 * min( Pr(samples <= null), Pr(samples >= null) ),
    where probabilities use (count + 1)/(n + 1) to avoid zero p.
    """
    n = len(samples)
    le = (np.sum(samples <= null) + 1) / (n + 1)
    ge = (np.sum(samples >= null) + 1) / (n + 1)
    return float(2.0 * min(le, ge))


# ============================================================================
# Setup
# ============================================================================

config, output_dir, logger = setup_analysis(
    analysis_name="behavioral_split_half",
    results_base=Path("results"),
    script_file=__file__,
)

# ============================================================================
# Load Data
# ============================================================================

logger.info("=" * 80)
logger.info("LOADING BEHAVIORAL DATA")
logger.info("=" * 80)

# Get participants
participants_list, (n_experts, n_novices) = get_participants_with_expertise()
logger.info(f"Total participants: {len(participants_list)} ({n_experts} experts, {n_novices} novices)")

# Separate expert and novice IDs
expert_ids = [sub_id for sub_id, is_expert in participants_list if is_expert]
novice_ids = [sub_id for sub_id, is_expert in participants_list if not is_expert]

logger.info("\nLoading trial data and converting to pairwise comparisons...")

expert_pairwise_list = []
for sub_id in expert_ids:
    trials = load_participant_trial_data(sub_id, is_expert=True, bids_root=Path(CONFIG['BIDS_ROOT']))
    if trials is not None:
        pairwise = create_pairwise_df(trials)
        pairwise['sub_id'] = sub_id
        expert_pairwise_list.append(pairwise)
        logger.info(f"  {sub_id}: {len(pairwise)} pairwise comparisons")

expert_pairwise_all = pd.concat(expert_pairwise_list, ignore_index=True)
logger.info(f"\nExpert pairwise data: {len(expert_pairwise_all)} total comparisons")

novice_pairwise_list = []
for sub_id in novice_ids:
    trials = load_participant_trial_data(sub_id, is_expert=False, bids_root=Path(CONFIG['BIDS_ROOT']))
    if trials is not None:
        pairwise = create_pairwise_df(trials)
        pairwise['sub_id'] = sub_id
        novice_pairwise_list.append(pairwise)
        logger.info(f"  {sub_id}: {len(pairwise)} pairwise comparisons")

novice_pairwise_all = pd.concat(novice_pairwise_list, ignore_index=True)
logger.info(f"\nNovice pairwise data: {len(novice_pairwise_all)} total comparisons")

# ============================================================================
# Compute Split-Half Reliability
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("COMPUTING SPLIT-HALF RELIABILITY")
logger.info("=" * 80)

results = {}
distributions = {}

for group_name, pairwise_data, participant_ids in [
    ('experts', expert_pairwise_all, expert_ids),
    ('novices', novice_pairwise_all, novice_ids)
]:
    logger.info(f"\n{group_name.upper()}")
    logger.info("-" * 80)
    logger.info(f"n={len(participant_ids)} participants")

    logger.info("\nBootstrapping random splits (n=1,000)...")
    logger.info("This may take a few minutes...")

    bootstrap_results = bootstrap_split_half_reliability(
        pairwise_data,
        participant_ids,
        n_iterations=1000,
        method='spearman',
        random_state=CONFIG['RANDOM_SEED']
    )

    logger.info(f"\nResults:")
    logger.info(f"  Half-sample correlation (r_half):")
    logger.info(f"    Mean = {bootstrap_results['mean_r_half']:.4f}")
    logger.info(f"    95% CI = [{bootstrap_results['ci_r_half'][0]:.4f}, "
                f"{bootstrap_results['ci_r_half'][1]:.4f}]")

    logger.info(f"\n  Full-sample reliability (r_full, Spearman-Brown corrected):")
    logger.info(f"    Mean = {bootstrap_results['mean_r_full']:.4f}")
    logger.info(f"    95% CI = [{bootstrap_results['ci_r_full'][0]:.4f}, "
                f"{bootstrap_results['ci_r_full'][1]:.4f}]")

    # Interpret reliability
    r_full = bootstrap_results['mean_r_full']
    if r_full >= 0.90:
        interpretation = "Excellent reliability"
    elif r_full >= 0.80:
        interpretation = "Good reliability"
    elif r_full >= 0.70:
        interpretation = "Acceptable reliability"
    else:
        interpretation = "Questionable reliability"

    logger.info(f"\n  Interpretation: {interpretation}")

    results[f'{group_name}_within'] = bootstrap_results
    distributions[f'{group_name}_within_r_half'] = bootstrap_results['r_half']
    distributions[f'{group_name}_within_r_full'] = bootstrap_results['r_full']

# ============================================================================
# Between-Group Analysis
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("BETWEEN-GROUP RELIABILITY (EXPERTS vs NOVICES)")
logger.info("=" * 80)

logger.info("\nBootstrapping random splits (n=1,000)...")
logger.info("This may take a few minutes...")

between_results = bootstrap_between_group_reliability(
    expert_pairwise_all,
    expert_ids,
    novice_pairwise_all,
    novice_ids,
    n_iterations=1000,
    method='spearman',
    random_state=CONFIG['RANDOM_SEED']
)

logger.info(f"\nResults:")
logger.info(f"  Half-sample correlation (r_half):")
logger.info(f"    Mean = {between_results['mean_r_half']:.4f}")
logger.info(f"    95% CI = [{between_results['ci_r_half'][0]:.4f}, "
            f"{between_results['ci_r_half'][1]:.4f}]")

logger.info(f"\n  Full-sample reliability (r_full, Spearman-Brown corrected):")
logger.info(f"    Mean = {between_results['mean_r_full']:.4f}")
logger.info(f"    95% CI = [{between_results['ci_r_full'][0]:.4f}, "
            f"{between_results['ci_r_full'][1]:.4f}]")

results['between_groups'] = between_results
distributions['between_groups_r_half'] = between_results['r_half']
distributions['between_groups_r_full'] = between_results['r_full']

# ============================================================================
# Statistical Tests
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("BOOTSTRAP INFERENCE")
logger.info("=" * 80)

# One-sample bootstrap p-values (H0: r_full = 0)
for key in ['experts_within', 'novices_within', 'between_groups']:
    r_full_samples = results[key]['r_full']
    p_boot = _bootstrap_p_two_sided(r_full_samples, null=0.0)
    results[key]['p_boot_full'] = p_boot
    logger.info(f"{key}: mean r_full = {results[key]['mean_r_full']:.4f}, "
                f"95% CI = [{results[key]['ci_r_full'][0]:.4f}, {results[key]['ci_r_full'][1]:.4f}], "
                f"p_boot = {p_boot:.4f}")

# Group difference in reliability (experts - novices), bootstrap Δ
exp_full = results['experts_within']['r_full']
nov_full = results['novices_within']['r_full']
n_delta = min(len(exp_full), len(nov_full))
delta_full = exp_full[:n_delta] - nov_full[:n_delta]
delta_mean = float(np.mean(delta_full))
delta_ci = (
    float(np.percentile(delta_full, 2.5)),
    float(np.percentile(delta_full, 97.5))
)
delta_p = _bootstrap_p_two_sided(delta_full, null=0.0)

results['experts_vs_novices_diff'] = {
    'delta_full': delta_full,
    'mean_delta_full': delta_mean,
    'ci_delta_full': delta_ci,
    'p_boot_delta_full': float(delta_p),
}

logger.info("\nGroup difference (experts − novices):")
logger.info(f"  Δ r_full = {delta_mean:.4f}, 95% CI = [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}], p_boot = {delta_p:.4f}")

# ============================================================================
# Save Artifacts
# ============================================================================

logger.info("\n" + "=" * 80)
logger.info("SAVING RESULTS")
logger.info("=" * 80)

# Save full results dictionary
pkl_path = output_dir / 'reliability_metrics.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
logger.info(f"✓ Saved reliability metrics: {pkl_path.name}")

# Save bootstrap distributions
npz_path = output_dir / 'split_rdm_distributions.npz'
np.savez_compressed(npz_path, **distributions)
logger.info(f"✓ Saved bootstrap distributions: {npz_path.name}")

# Create human-readable summary table (bootstrap CIs and p_boot)
summary_rows = []

for group_label, dict_key in [("Experts (within)", 'experts_within'), ("Novices (within)", 'novices_within')]:
    val = results[dict_key]
    summary_rows.append({
        'comparison': group_label,
        'n_iterations': len(val['r_full']),
        'r_half_mean': f"{val['mean_r_half']:.4f}",
        'r_half_ci': f"[{val['ci_r_half'][0]:.4f}, {val['ci_r_half'][1]:.4f}]",
        'r_full_mean': f"{val['mean_r_full']:.4f}",
        'r_full_ci': f"[{val['ci_r_full'][0]:.4f}, {val['ci_r_full'][1]:.4f}]",
        'r_full_p_boot': f"{val['p_boot_full']:.4e}",
    })

val = results['between_groups']
summary_rows.append({
    'comparison': 'Between-groups (E vs N)',
    'n_iterations': len(val['r_full']),
    'r_half_mean': f"{val['mean_r_half']:.4f}",
    'r_half_ci': f"[{val['ci_r_half'][0]:.4f}, {val['ci_r_half'][1]:.4f}]",
    'r_full_mean': f"{val['mean_r_full']:.4f}",
    'r_full_ci': f"[{val['ci_r_full'][0]:.4f}, {val['ci_r_full'][1]:.4f}]",
    'r_full_p_boot': f"{val['p_boot_full']:.4e}",
})

# Group difference row
diff = results['experts_vs_novices_diff']
summary_rows.append({
    'comparison': 'Δ reliability (E − N)',
    'n_iterations': len(diff['delta_full']),
    'r_half_mean': '',
    'r_half_ci': '',
    'r_full_mean': f"{diff['mean_delta_full']:.4f}",
    'r_full_ci': f"[{diff['ci_delta_full'][0]:.4f}, {diff['ci_delta_full'][1]:.4f}]",
    'r_full_p_boot': f"{diff['p_boot_delta_full']:.4e}",
})

summary_df = pd.DataFrame(summary_rows)
csv_path = output_dir / 'reliability_summary.csv'
summary_df.to_csv(csv_path, index=False)
logger.info(f"✓ Saved summary table: {csv_path.name}")

logger.info("\n" + "=" * 80)
logger.info("SUMMARY TABLE")
logger.info("=" * 80)
logger.info(f"\n{summary_df.to_string(index=False)}")

# ============================================================================
# End
# ============================================================================

log_script_end(logger)
