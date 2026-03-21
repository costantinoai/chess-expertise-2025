#!/usr/bin/env python3
"""
Skill-Gradient Analysis — Supplementary Analysis

METHODS
=======

Overview
--------
This analysis tests whether neural measures scale continuously with chess
skill using two complementary skill proxies:

1. **Elo rating** (general chess strength): Correlated with PR, RSA model
   fit, and decoding accuracy within the expert group (n=20, Elo 1751-2269).
2. **Familiarisation accuracy** (stimulus-specific competence): Move accuracy
   on the 20 checkmate boards used in fMRI, correlated with all neural metrics
   across all participants (n=38) and within experts only.

All correlations are computed across the 22 coarsened bilateral Glasser ROI
groups. FDR correction (Benjamini-Hochberg) is applied across ROIs within
each measure.

Data
----
- Participants: BIDS participants.tsv (Elo in 'rating' column)
- PR: chess-manifold/results/manifold/pr_results.pkl
- RSA: BIDS/derivatives/mvpa-rsa per-subject TSVs
- Decoding: BIDS/derivatives/mvpa-decoding per-subject TSVs
- Familiarisation: task-engagement/results/familiarisation_accuracy/
  familiarisation_subject_accuracy.csv (behavioural only; neural enrichment
  is computed within this script)

Outputs
-------
- elo_pr_correlations.csv: Elo x PR per ROI + mean
- elo_rsa_correlations.csv: Elo x RSA (checkmate, strategy) per ROI + mean
- elo_decoding_correlations.csv: Elo x decoding per ROI + mean
- elo_correlations_all.csv: Combined Elo results
- familiarisation_neural_correlations.csv: Familiarisation accuracy x neural metrics
"""

import pickle  # nosec B403 — trusted internal analysis outputs
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
from utils import compute_subject_mean_pr, load_bids_tsvs
from common.stats_utils import apply_fdr_correction


# ============================================================================
# Setup
# ============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="skill_gradient",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)


# ============================================================================
# Helper functions
# ============================================================================

def compute_correlations(x, y):
    """Compute Pearson and Spearman correlations."""
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    return r_p, p_p, r_s, p_s


# ============================================================================
# Load data
# ============================================================================

# Participants
participants = pd.read_csv(config['BIDS_PARTICIPANTS'], sep='\t')
experts = participants[
    (participants['group'] == 'expert') & participants['rating'].notna()
].copy()
elo_map = dict(zip(experts['participant_id'], experts['rating']))
expert_ids = set(elo_map.keys())
logger.info(
    f"Experts with Elo: {len(expert_ids)}, "
    f"range: {min(elo_map.values()):.0f}-{max(elo_map.values()):.0f}"
)

# PR data (trusted internal analysis output)
pr_pkl = Path(config['REPO_ROOT']) / 'chess-manifold/results/manifold/pr_results.pkl'
with open(pr_pkl, 'rb') as f:  # nosec B301 — trusted internal pkl
    pr_data = pickle.load(f)
pr_long = pr_data['pr_long_format']
roi_info = pr_data['roi_info']
roi_id_to_name = dict(zip(roi_info['roi_id'], roi_info['pretty_name']))
logger.info(f"PR data: {pr_long['subject_id'].nunique()} subjects, "
            f"{pr_long['ROI_Label'].nunique()} ROIs")

# RSA data
logger.info("Loading RSA data...")
rsa_df = load_bids_tsvs(config['BIDS_MVPA_RSA'])
roi_cols = [c for c in rsa_df.columns if c not in ['subject', 'target']]
logger.info(f"RSA data: {rsa_df['subject'].nunique()} subjects, "
            f"{len(roi_cols)} ROIs, targets: {sorted(rsa_df['target'].unique())}")

# Decoding data
logger.info("Loading decoding data...")
dec_df = load_bids_tsvs(config['BIDS_MVPA_DECODING'])
dec_roi_cols = [c for c in dec_df.columns if c not in ['subject', 'target']]
logger.info(f"Decoding data: {dec_df['subject'].nunique()} subjects, "
            f"{len(dec_roi_cols)} ROIs, targets: {sorted(dec_df['target'].unique())}")


# ============================================================================
# Elo x PR correlations (22 ROIs)
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("ELO x PR CORRELATIONS")
logger.info("=" * 60)

pr_expert = pr_long[pr_long['subject_id'].isin(expert_ids)].copy()
pr_expert['elo'] = pr_expert['subject_id'].map(elo_map)

pr_results = []
for roi_label in sorted(pr_expert['ROI_Label'].unique()):
    sub = pr_expert[pr_expert['ROI_Label'] == roi_label]
    r_p, p_p, r_s, p_s = compute_correlations(sub['elo'], sub['PR'])
    pr_results.append({
        'measure': 'PR', 'roi_id': roi_label,
        'roi_name': roi_id_to_name.get(roi_label, f'ROI_{roi_label}'),
        'n': len(sub), 'pearson_r': r_p, 'pearson_p': p_p,
        'spearman_rho': r_s, 'spearman_p': p_s,
    })

# PR mean across ROIs
expert_mean_pr = compute_subject_mean_pr(pr_long, subject_ids=expert_ids)
elo_vals = expert_mean_pr['participant_id'].map(elo_map)
r_p, p_p, r_s, p_s = compute_correlations(elo_vals, expert_mean_pr['mean_pr'])
pr_results.append({
    'measure': 'PR', 'roi_id': 'mean', 'roi_name': 'Mean (22 ROIs)',
    'n': len(expert_mean_pr), 'pearson_r': r_p, 'pearson_p': p_p,
    'spearman_rho': r_s, 'spearman_p': p_s,
})

pr_res_df = pd.DataFrame(pr_results)
roi_mask = pr_res_df['roi_id'] != 'mean'
_, pr_res_df.loc[roi_mask, 'pearson_p_fdr'] = apply_fdr_correction(
    pr_res_df.loc[roi_mask, 'pearson_p'].values)
_, pr_res_df.loc[roi_mask, 'spearman_p_fdr'] = apply_fdr_correction(
    pr_res_df.loc[roi_mask, 'spearman_p'].values)

row = pr_res_df[pr_res_df['roi_id'] == 'mean'].iloc[0]
logger.info(f"  Mean PR x Elo: r={row['pearson_r']:.3f}, p={row['pearson_p']:.3f}; "
            f"rho={row['spearman_rho']:.3f}, p={row['spearman_p']:.3f}")

for _, row in pr_res_df[(pr_res_df['roi_id'] != 'mean') & (pr_res_df['pearson_p'] < 0.10)].iterrows():
    logger.info(f"  {row['roi_name']}: r={row['pearson_r']:.3f}, p={row['pearson_p']:.3f}, "
                f"p_fdr={row['pearson_p_fdr']:.3f}")

pr_res_df.to_csv(out_dir / 'elo_pr_correlations.csv', index=False)


# ============================================================================
# Elo x RSA correlations (22 ROIs)
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("ELO x RSA CORRELATIONS")
logger.info("=" * 60)

rsa_expert = rsa_df[rsa_df['subject'].isin(expert_ids)].copy()
rsa_expert['elo'] = rsa_expert['subject'].map(elo_map)

rsa_results = []
for target in ['checkmate', 'strategy']:
    sub = rsa_expert[rsa_expert['target'] == target]
    for roi_col in roi_cols:
        vals = sub[roi_col].astype(float)
        r_p, p_p, r_s, p_s = compute_correlations(sub['elo'], vals)
        rsa_results.append({
            'measure': f'RSA_{target}', 'roi_id': roi_col,
            'roi_name': roi_col.replace('_', ' '),
            'n': len(sub), 'pearson_r': r_p, 'pearson_p': p_p,
            'spearman_rho': r_s, 'spearman_p': p_s,
        })
    # Mean across 22 ROIs
    mean_rsa = sub[roi_cols].astype(float).mean(axis=1)
    r_p, p_p, r_s, p_s = compute_correlations(sub['elo'], mean_rsa)
    rsa_results.append({
        'measure': f'RSA_{target}', 'roi_id': 'mean',
        'roi_name': 'Mean (22 ROIs)',
        'n': len(sub), 'pearson_r': r_p, 'pearson_p': p_p,
        'spearman_rho': r_s, 'spearman_p': p_s,
    })

rsa_res_df = pd.DataFrame(rsa_results)
for target in ['checkmate', 'strategy']:
    mask = (rsa_res_df['measure'] == f'RSA_{target}') & (rsa_res_df['roi_id'] != 'mean')
    _, rsa_res_df.loc[mask, 'pearson_p_fdr'] = apply_fdr_correction(
        rsa_res_df.loc[mask, 'pearson_p'].values)
    _, rsa_res_df.loc[mask, 'spearman_p_fdr'] = apply_fdr_correction(
        rsa_res_df.loc[mask, 'spearman_p'].values)

for target in ['checkmate', 'strategy']:
    row = rsa_res_df[(rsa_res_df['measure'] == f'RSA_{target}') &
                     (rsa_res_df['roi_id'] == 'mean')].iloc[0]
    logger.info(f"  RSA {target} x Elo (mean): r={row['pearson_r']:.3f}, p={row['pearson_p']:.3f}; "
                f"rho={row['spearman_rho']:.3f}, p={row['spearman_p']:.3f}")

rsa_res_df.to_csv(out_dir / 'elo_rsa_correlations.csv', index=False)


# ============================================================================
# Elo x Decoding correlations (22 ROIs)
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("ELO x DECODING CORRELATIONS")
logger.info("=" * 60)

dec_expert = dec_df[dec_df['subject'].isin(expert_ids)].copy()
dec_expert['elo'] = dec_expert['subject'].map(elo_map)

dec_results = []
for target in ['checkmate', 'strategy']:
    sub = dec_expert[dec_expert['target'] == target]
    if len(sub) == 0:
        logger.warning(f"  No decoding data for target '{target}'")
        continue
    for roi_col in dec_roi_cols:
        vals = sub[roi_col].astype(float)
        r_p, p_p, r_s, p_s = compute_correlations(sub['elo'], vals)
        dec_results.append({
            'measure': f'Dec_{target}', 'roi_id': roi_col,
            'roi_name': roi_col.replace('_', ' '),
            'n': len(sub), 'pearson_r': r_p, 'pearson_p': p_p,
            'spearman_rho': r_s, 'spearman_p': p_s,
        })
    # Mean across ROIs
    mean_dec = sub[dec_roi_cols].astype(float).mean(axis=1)
    r_p, p_p, r_s, p_s = compute_correlations(sub['elo'], mean_dec)
    dec_results.append({
        'measure': f'Dec_{target}', 'roi_id': 'mean',
        'roi_name': 'Mean (22 ROIs)',
        'n': len(sub), 'pearson_r': r_p, 'pearson_p': p_p,
        'spearman_rho': r_s, 'spearman_p': p_s,
    })

dec_res_df = pd.DataFrame(dec_results)
for target in ['checkmate', 'strategy']:
    mask = (dec_res_df['measure'] == f'Dec_{target}') & (dec_res_df['roi_id'] != 'mean')
    if mask.sum() > 0:
        _, dec_res_df.loc[mask, 'pearson_p_fdr'] = apply_fdr_correction(
            dec_res_df.loc[mask, 'pearson_p'].values)
        _, dec_res_df.loc[mask, 'spearman_p_fdr'] = apply_fdr_correction(
            dec_res_df.loc[mask, 'spearman_p'].values)

for target in ['checkmate', 'strategy']:
    rows = dec_res_df[(dec_res_df['measure'] == f'Dec_{target}') &
                      (dec_res_df['roi_id'] == 'mean')]
    if len(rows) > 0:
        row = rows.iloc[0]
        logger.info(f"  Dec {target} x Elo (mean): r={row['pearson_r']:.3f}, p={row['pearson_p']:.3f}; "
                    f"rho={row['spearman_rho']:.3f}, p={row['spearman_p']:.3f}")

dec_res_df.to_csv(out_dir / 'elo_decoding_correlations.csv', index=False)


# ============================================================================
# Combine all Elo results
# ============================================================================

all_elo = pd.concat([pr_res_df, rsa_res_df, dec_res_df], ignore_index=True)
all_elo.to_csv(out_dir / 'elo_correlations_all.csv', index=False)

logger.info("")
logger.info("=== FDR-significant results (any measure) ===")
any_sig = False
for _, row in all_elo.iterrows():
    if pd.notna(row.get('pearson_p_fdr')) and row['pearson_p_fdr'] < 0.05:
        logger.info(f"  {row['measure']} - {row['roi_name']}: r={row['pearson_r']:.3f}, "
                    f"p_fdr={row['pearson_p_fdr']:.3f}")
        any_sig = True
if not any_sig:
    logger.info("  None")


# ============================================================================
# Familiarisation accuracy as stimulus-specific skill proxy
# ============================================================================

logger.info("")
logger.info("=" * 60)
logger.info("FAMILIARISATION ACCURACY x NEURAL METRICS")
logger.info("=" * 60)

fam_file = (Path(config['REPO_ROOT']) / 'chess-supplementary' / 'task-engagement' /
            'results' / 'familiarisation_accuracy' / 'familiarisation_subject_accuracy.csv')

if fam_file.exists():
    fam_df = pd.read_csv(fam_file)
    logger.info(f"Loaded familiarisation data: {len(fam_df)} subjects")

    # Enrich with neural metrics (mean across 22 ROIs per subject)
    # PR
    subject_pr = compute_subject_mean_pr(pr_long)
    fam_df = fam_df.merge(subject_pr, on='participant_id', how='left')

    # RSA (mean across ROIs per subject per target model)
    for target in ['checkmate', 'strategy', 'visual_similarity']:
        rsa_target = rsa_df[rsa_df['target'] == target].copy()
        if len(rsa_target) > 0:
            rsa_target[f'rsa_{target}'] = rsa_target[roi_cols].astype(float).mean(axis=1)
            fam_df = fam_df.merge(
                rsa_target[['subject', f'rsa_{target}']].rename(columns={'subject': 'participant_id'}),
                on='participant_id', how='left',
            )

    # Decoding (mean across ROIs per subject per target)
    for target in ['checkmate', 'strategy']:
        dec_target = dec_df[dec_df['target'] == target].copy()
        if len(dec_target) > 0:
            dec_target[f'dec_{target}'] = dec_target[dec_roi_cols].astype(float).mean(axis=1)
            fam_df = fam_df.merge(
                dec_target[['subject', f'dec_{target}']].rename(columns={'subject': 'participant_id'}),
                on='participant_id', how='left',
            )

    # Save enriched CSV for plotting script
    fam_df.to_csv(out_dir / 'familiarisation_subject_enriched.csv', index=False)

    neural_cols = ['mean_pr', 'rsa_checkmate', 'rsa_strategy',
                   'rsa_visual_similarity', 'dec_checkmate', 'dec_strategy']
    skill_col = 'move_acc_all_cm'

    # Note: pooled (all participants) correlations include between-group variance.
    # Expert-only correlations isolate the within-group skill gradient.
    fam_valid = fam_df[fam_df[skill_col].notna()].copy()
    logger.info(f"  Participants with familiarisation + neural data: {len(fam_valid)}")

    fam_corr_rows = []
    for nc in neural_cols:
        if nc not in fam_valid.columns:
            continue
        valid = fam_valid[[skill_col, nc, 'group']].dropna()

        # All participants
        if len(valid) >= 5:
            r, p = stats.pearsonr(valid[skill_col], valid[nc])
            logger.info(f"  All (n={len(valid)}): {skill_col} x {nc}: r={r:.3f}, p={p:.4f}")
            fam_corr_rows.append({
                'skill_proxy': skill_col, 'neural_metric': nc,
                'sample': 'all', 'n': len(valid),
                'pearson_r': r, 'pearson_p': p,
            })

        # Experts only
        exp_valid = valid[valid['group'] == 'expert']
        if len(exp_valid) >= 5:
            r, p = stats.pearsonr(exp_valid[skill_col], exp_valid[nc])
            logger.info(f"  Experts (n={len(exp_valid)}): {skill_col} x {nc}: r={r:.3f}, p={p:.4f}")
            fam_corr_rows.append({
                'skill_proxy': skill_col, 'neural_metric': nc,
                'sample': 'expert', 'n': len(exp_valid),
                'pearson_r': r, 'pearson_p': p,
            })

    fam_corr_df = pd.DataFrame(fam_corr_rows)

    # Apply FDR correction across all familiarisation correlations per sample
    for sample in fam_corr_df['sample'].unique():
        mask = fam_corr_df['sample'] == sample
        raw_p = fam_corr_df.loc[mask, 'pearson_p'].values
        _, fdr_p = apply_fdr_correction(raw_p)
        fam_corr_df.loc[mask, 'pearson_p_fdr'] = fdr_p

    fam_corr_df.to_csv(out_dir / 'familiarisation_neural_correlations.csv', index=False)
else:
    logger.warning(f"Familiarisation file not found: {fam_file}")


# ============================================================================

log_script_end(logger)
