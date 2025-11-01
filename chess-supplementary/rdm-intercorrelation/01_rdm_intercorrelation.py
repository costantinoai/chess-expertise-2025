#!/usr/bin/env python3
"""
RDM Intercorrelation Analysis — Supplementary Material

METHODS
=======

Rationale
---------
This analysis quantifies the relationships among three theoretical model
representational dissimilarity matrices (RDMs) derived from chess stimulus
features. The goal is to assess shared and unique variance explained by each
model RDM through pairwise correlations, partial correlations controlling for
other models, and hierarchical variance partitioning. Understanding the
intercorrelations among model RDMs is critical for interpreting which aspects
of stimulus structure drive behavioral and neural similarity patterns.

Data
----
Stimulus metadata was loaded from the repository-wide stimulus TSV. All 40
chess stimuli (20 check, 20 non-check) were included, yielding 780 unique
pairwise dissimilarities per RDM (upper triangle of 40×40 matrix, excluding
diagonal).

Model RDMs
----------
Three theoretical RDMs were derived from stimulus feature annotations:

1. **Check Status** (`check`): Binary categorical RDM encoding whether both
   stimuli share the same check status (dissimilarity = 0 if both check or
   both non-check; = 1 otherwise). Balanced design: 20 check, 20 non-check.

2. **Strategy** (`strategy`): Binary categorical RDM encoding whether both
   stimuli share the same tactical theme (e.g., fork, pin, skewer, back-rank
   mate). Dissimilarity = 0 if same strategy; = 1 otherwise. Five strategy
   categories.

3. **Visual Similarity** (`visual`): Binary categorical RDM encoding whether
   both stimuli belong to the same visual cluster (based on piece configuration
   patterns). Dissimilarity = 0 if same cluster; = 1 otherwise. Four visual
   clusters.

Each RDM is a 40×40 symmetric matrix where entry (i,j) = 0 if stimuli i and j
share the feature, and 1 otherwise. Diagonal entries are zero.

Statistical Analyses
--------------------

**Pairwise RDM Correlations**

Spearman rank correlations were computed between all pairs of model RDMs using
vectorized upper-triangle entries (780 pairwise dissimilarities per RDM,
excluding diagonal). Correlation magnitude quantifies the overlap in
representational structure between models. High correlations indicate that two
models capture similar patterns of stimulus dissimilarity.

**Partial Correlations**

For each target RDM and each predictor RDM, Spearman partial correlations were
computed while controlling for all remaining model RDMs as covariates. This
isolates the unique relationship between target and predictor after accounting
for shared variance with other models.

The partial correlation was computed by:
1. Regressing the target RDM on covariate RDMs (ordinary least squares),
   extracting residuals
2. Regressing the predictor RDM on the same covariate RDMs, extracting
   residuals
3. Computing Spearman correlation between the two residual vectors
4. Computing p-values from the t-distribution with df = n_observations −
   n_covariates − 2

Example: The partial correlation between Visual and Strategy RDMs (controlling
for Check RDM) reveals whether Visual and Strategy share representational
structure beyond any common relationship with check status.

**Variance Partitioning**

For each target RDM, hierarchical variance partitioning quantified the
proportion of variance (R²) explained uniquely by each predictor RDM, shared
across predictors, and unexplained. This analysis uses commonality analysis
principles to decompose total explained variance into unique and shared
components.

Method:
1. Fit full model: target ~ all predictors (ordinary least squares regression)
2. Compute R²_full (total variance explained by all predictors)
3. For each predictor k:
   - Fit reduced model: target ~ all predictors except k
   - Compute R²_reduced(k)
   - Unique variance for k: R²_unique(k) = R²_full − R²_reduced(k)
4. Shared variance: R²_shared = R²_full − Σ R²_unique
5. Unexplained variance: R²_unexplained = 1 − R²_full

All regressions operated on vectorized upper-triangle entries (780
observations per RDM). Predictors and target were z-scored before regression
to ensure comparable coefficients.

Statistical Assumptions and Limitations
----------------------------------------
- **Independence assumption**: RDM entries are treated as independent
  observations for correlation and regression analyses. In reality, RDM entries
  sharing stimuli exhibit dependence. This is standard practice in RSA but may
  inflate significance; results should be interpreted cautiously.

- **Sample size**: 780 pairwise dissimilarities provide reasonable power for
  correlation and regression, but the effective degrees of freedom are lower
  due to entry dependencies.

- **Binary features**: All model RDMs encode categorical features as binary
  dissimilarity (0=same category, 1=different category), which may not capture
  graded relationships within categories.

Outputs
-------
All results are saved to results/<timestamp>_rdm_intercorrelation/:
- pairwise_correlations.tsv: 3×3 Spearman correlation matrix
- partial_correlations.tsv: Target, predictor, covariates, r_partial, p, dof
- variance_partitioning_<target>.tsv: Unique, shared, unexplained R² per target
- variance_partitioning_all.tsv: Combined variance partitioning for all targets
- 01_rdm_intercorrelation.py: Copy of this script
"""

import sys
from pathlib import Path

# Enable imports from repo root
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import load_stimulus_metadata
from common.rsa_utils import create_model_rdm, compute_pairwise_rdm_correlations
from common.stats_utils import partial_correlation_rdms, variance_partitioning_rdms

# === Setup ===
config, out_dir, logger = setup_analysis(
    analysis_name="rdm_intercorrelation",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)

logger.info("=" * 80)
logger.info("RDM INTERCORRELATION ANALYSIS")
logger.info("=" * 80)

# Load stimulus feature table containing categorical annotations (check status,
# strategy type, visual cluster) for all 40 chess positions. These features will
# be used to construct theoretical model RDMs encoding different aspects of
# stimulus similarity.
logger.info("Loading stimulus metadata...")
stim_df = load_stimulus_metadata(return_all=True)
logger.info(f"Loaded {len(stim_df)} stimuli")
logger.info(f"Available features: {list(stim_df.columns)}")

# Define three model RDMs to analyze: check status, strategy, and visual similarity.
# Each is a binary categorical RDM where dissimilarity=0 if stimuli share the same
# category and dissimilarity=1 otherwise. We use only these three to avoid collinearity
# (e.g., checkmate is a subset of check, so they would be highly correlated).
model_specs = {
    'check': ('check', True),
    'strategy': ('strategy', True),
    'visual': ('visual', True),
}

logger.info(f"Computing {len(model_specs)} model RDMs: {list(model_specs.keys())}")

# Construct model RDMs from stimulus feature vectors. For categorical features,
# create_model_rdm generates binary dissimilarity matrices (0=same category, 1=different).
model_rdms = {}
for model_name, (col_name, is_categorical) in model_specs.items():
    if col_name not in stim_df.columns:
        logger.warning(f"Column '{col_name}' not found. Skipping {model_name}.")
        continue

    values = stim_df[col_name].values
    rdm = create_model_rdm(values, is_categorical=is_categorical)
    model_rdms[model_name] = rdm
    logger.info(f"  Created {model_name} RDM ({rdm.shape[0]}×{rdm.shape[0]})")

model_names = list(model_rdms.keys())

# Sanity check: need at least two RDMs for correlation/partial correlation
if len(model_rdms) < 2:
    raise RuntimeError(
        "At least two model RDMs are required. "
        "Check stimulus metadata columns or update model_specs."
    )

# Order models consistently (using CONFIG if available, otherwise alphabetical)
model_names = [
    name for name in CONFIG.get('MODEL_ORDER', []) if name in model_rdms
]
model_names += [name for name in model_rdms if name not in model_names]
logger.info(f"Model processing order: {model_names}")

# === Analysis 1: Pairwise Correlations ===
# Compute Spearman rank correlations between all pairs of model RDMs to quantify
# their overlap. High correlations indicate that two models capture similar stimulus
# relationships (e.g., if check and strategy are correlated, stimuli in check might
# tend to share strategic themes). Correlations are computed on vectorized upper
# triangles (780 unique pairwise dissimilarities per 40×40 RDM).
logger.info("")
logger.info("Computing pairwise RDM correlations...")

pairwise_corr = compute_pairwise_rdm_correlations(model_rdms, method='spearman')
pairwise_corr = pairwise_corr.loc[model_names, model_names]

# Save symmetric matrix (r only) for quick reference
pairwise_corr.to_csv(out_dir / "pairwise_correlations.tsv", sep='\t', float_format='%.4f')
logger.info(f"Saved pairwise correlations to {out_dir / 'pairwise_correlations.tsv'}")

# Also compute raw + FDR p-values for unique pairs (upper triangle)
from scipy.stats import spearmanr
from common.stats_utils import apply_fdr_correction

pair_rows = []
tri_indices = []
for i, name_i in enumerate(model_names):
    for j, name_j in enumerate(model_names):
        if i < j:
            # Vectorize upper triangle of both RDMs
            n_stim = model_rdms[name_i].shape[0]
            tri = np.triu_indices(n_stim, k=1)
            x = model_rdms[name_i][tri]
            y = model_rdms[name_j][tri]
            r, p = spearmanr(x, y)
            pair_rows.append({'rdm1': name_i, 'rdm2': name_j, 'r': float(r), 'p_raw': float(p)})
            tri_indices.append((i, j))

pair_df = pd.DataFrame(pair_rows)
if not pair_df.empty:
    _, p_fdr = apply_fdr_correction(pair_df['p_raw'].values, alpha=0.05, method='fdr_bh')
    pair_df['p_fdr'] = p_fdr
    pair_df.to_csv(out_dir / "pairwise_correlations_long.tsv", sep='\t', index=False, float_format='%.4f')
    logger.info(f"Saved pairwise correlation p-values to {out_dir / 'pairwise_correlations_long.tsv'}")

# Log pairwise correlation magnitudes
logger.info("")
logger.info("Pairwise correlation matrix:")
for i, name_i in enumerate(model_names):
    for j, name_j in enumerate(model_names):
        if i < j:
            r = pairwise_corr.loc[name_i, name_j]
            logger.info(f"  {name_i} <-> {name_j}: r={r:.3f}")

# === Analysis 2: Partial Correlations ===
# Compute partial correlations to isolate unique relationships between pairs of RDMs
# after controlling for all other RDMs. For example, the partial correlation between
# check and strategy (controlling for visual) reveals whether check and strategy
# share variance beyond their common association with visual similarity.
#
# Method: Regress target on covariates (OLS), regress predictor on covariates (OLS),
# then compute Spearman correlation between residuals. P-values from t-distribution
# with dof = n_observations - n_covariates - 2.
logger.info("")
logger.info("Computing partial correlations (controlling for other RDMs)...")

partial_results = []

for target_name in model_names:
    for predictor_name in model_names:
        if target_name == predictor_name:
            continue

        # Covariates: all RDMs except target and predictor
        covariate_names = [name for name in model_names if name not in [target_name, predictor_name]]

        if not covariate_names:
            continue

        covariate_rdms = [model_rdms[name] for name in covariate_names]

        result = partial_correlation_rdms(
            model_rdms[target_name],
            model_rdms[predictor_name],
            covariate_rdms,
            method='spearman'
        )

        partial_results.append({
            'target': target_name,
            'predictor': predictor_name,
            'covariates': ','.join(covariate_names),
            'r_partial': result['r'],
            'p_partial': result['p'],
            'dof': result['dof']
        })

        logger.info(
            f"  {target_name} ~ {predictor_name} | {','.join(covariate_names)}: "
            f"r={result['r']:.3f}, p={result['p']:.4f}"
        )

partial_df = pd.DataFrame(partial_results)
if not partial_df.empty:
    # Apply FDR across all partial tests
    _, p_fdr = apply_fdr_correction(partial_df['p_partial'].values, alpha=0.05, method='fdr_bh')
    partial_df['p_fdr'] = p_fdr
partial_df.to_csv(out_dir / "partial_correlations.tsv", sep='\t', index=False, float_format='%.4f')
logger.info(f"Saved partial correlations to {out_dir / 'partial_correlations.tsv'}")

# === Analysis 3: Variance Partitioning ===
# Decompose each model RDM's variance into unique and shared components using
# hierarchical variance partitioning (commonality analysis). For each target RDM,
# fit a full regression model with all other RDMs as predictors, then compute:
# - R²_full: total variance explained by all predictors
# - R²_unique(k): unique variance explained by predictor k (difference between
#   full model and reduced model excluding k)
# - R²_shared: variance shared across multiple predictors (R²_full - sum of uniques)
# - R²_residual: unexplained variance (1 - R²_full)
#
# This reveals which aspects of each model RDM are independent vs overlapping with
# other models, helping interpret what each model uniquely captures.
logger.info("")
logger.info("Performing variance partitioning for each target RDM...")

var_part_frames = []

for target_name in model_names:
    # All other RDMs serve as predictors
    predictor_names = [name for name in model_names if name != target_name]
    predictor_rdms_dict = {name: model_rdms[name] for name in predictor_names}

    # Fit OLS regressions: target ~ all predictors, and reduced models (target ~ all except one)
    var_part_df = variance_partitioning_rdms(
        model_rdms[target_name],
        predictor_rdms_dict
    )

    var_part_df.insert(0, 'target', target_name)
    var_part_frames.append(var_part_df)

    out_file = out_dir / f"variance_partitioning_{target_name}.tsv"
    var_part_df.to_csv(out_file, sep='\t', index=False, float_format='%.4f')

    # Log variance components
    logger.info(f"  Target: {target_name}")
    logger.info(f"    Full R² = {var_part_df['r2_full'].iloc[0]:.4f}")
    for pred_name in predictor_names:
        unique_col = f"unique_{pred_name}"
        if unique_col in var_part_df.columns:
            logger.info(f"    Unique ({pred_name}) = {var_part_df[unique_col].iloc[0]:.4f}")
    logger.info(f"    Shared = {var_part_df['shared'].iloc[0]:.4f}")
    logger.info(f"    Residual = {var_part_df['residual'].iloc[0]:.4f}")
    logger.info(f"    Saved to {out_file}")

# Combine variance partitioning results for all targets into one table
combined_var_part = pd.concat(var_part_frames, ignore_index=True)
combined_var_part.to_csv(out_dir / "variance_partitioning_all.tsv", sep='\t', index=False, float_format='%.4f')
logger.info("")
logger.info(f"Combined variance partitioning saved to {out_dir / 'variance_partitioning_all.tsv'}")

# === End ===
logger.info("")
logger.info("=" * 80)
logger.info("RDM intercorrelation analysis complete!")
logger.info(f"Results saved to: {out_dir}")
logger.info("=" * 80)

log_script_end(logger)
