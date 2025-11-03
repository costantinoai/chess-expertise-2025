#!/usr/bin/env python3
"""
RDM Intercorrelation — LaTeX Table Generation
==============================================

This script generates publication-ready LaTeX tables summarizing the relationships
between different representational dissimilarity matrices (RDMs) using pairwise
correlations, partial correlations, and variance partitioning.

METHODS (Academic Manuscript Section)
--------------------------------------
Summary tables were generated to characterize the relationships between
theoretical model RDMs (e.g., checkmate status, strategy type, visual similarity).
Understanding how these RDMs relate to each other is critical for interpreting
their unique and shared contributions to neural and behavioral representations.

Three complementary analyses were performed:

1. **Pairwise Correlations**: For each pair of model RDMs, we computed Spearman
   rank correlations to quantify their similarity. Statistical significance was
   assessed using permutation tests (10,000 iterations), and p-values were
   corrected for multiple comparisons using the Benjamini-Hochberg false
   discovery rate (FDR) procedure (α=0.05). The pairwise correlation matrix
   provides an overview of which model RDMs share variance and which are
   relatively independent.

2. **Partial Correlations**: To isolate the unique relationship between pairs
   of RDMs while controlling for confounding influences from other RDMs, we
   computed partial Spearman correlations. For each target-predictor pair, we
   controlled for all other model RDMs as covariates. This analysis reveals
   whether pairwise correlations are driven by direct relationships or indirect
   associations through third variables.

3. **Variance Partitioning**: For each target RDM, we performed hierarchical
   variance partitioning to decompose its total variance into components
   attributable to each predictor RDM. We report:
   - Full R²: Total variance explained by all predictors combined
   - Unique variance: Variance explained exclusively by each predictor
   - Shared variance: Variance explained jointly by multiple predictors
   - Residual variance: Unexplained variance (1 - R²)

   This analysis quantifies how much each model RDM uniquely contributes to
   explaining others, versus how much is redundantly explained by multiple
   models.

Inputs
------
- pairwise_correlations.tsv: Symmetric correlation matrix (N×N model RDMs)
- pairwise_correlations_long.tsv: Long-format pairwise correlations with p-values and FDR q-values
- partial_correlations.tsv: Partial correlations controlling for other RDMs
  - Columns: target, predictor, covariates, r_partial, p_partial, p_fdr
- variance_partitioning_all.tsv: Variance components for each target RDM
  - Columns: target, r2_full, unique_{model}, shared, residual

Outputs
-------
- tables/table_pairwise_correlations.tex: Upper-triangle correlation matrix
- tables/table_pairwise_correlations_p.tex: Long-format table with p-values
- tables/table_partial_correlations.tex: Partial correlation table
- tables/table_variance_partitioning.tex: Variance partitioning results

Dependencies
------------
- common: CONFIG, logging_utils, io_utils
- modules: pretty_model_label for display names

Usage
-----
python chess-supplementary/rdm-intercorrelation/81_table_rdm_intercorr.py

Supplementary Analysis: RDM intercorrelation and variance partitioning
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Ensure repo root is on sys.path for 'common' imports
_cur = os.path.dirname(__file__)
for _up in (os.path.join(_cur, '..'), os.path.join(_cur, '..', '..')):
    _cand = os.path.abspath(_up)
    if os.path.isdir(os.path.join(_cand, 'common')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from common import CONFIG, setup_script, log_script_end
from modules import pretty_model_label


# ============================================================================
# Configuration & Setup
# ============================================================================

# Locate the latest RDM intercorrelation results directory
results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='rdm_intercorrelation',
    output_subdirs=['tables'],
    log_name='table_rdm_intercorr.log',
)

logger.info("=" * 80)
logger.info("GENERATING LATEX TABLES FOR RDM INTERCORRELATION")
logger.info("=" * 80)

# Output directory for tables (created by find_latest_results_directory)
tables_dir = dirs['tables']
logger.info(f"Tables will be saved to: {tables_dir}")

# ============================================================================
# Load Analysis Results
# ============================================================================

logger.info("")
logger.info("Loading analysis results...")

# Define paths to all input TSV files
pairwise_file = results_dir / "pairwise_correlations.tsv"  # Symmetric correlation matrix
pairwise_long_file = results_dir / "pairwise_correlations_long.tsv"  # Long format with p-values
partial_file = results_dir / "partial_correlations.tsv"  # Partial correlations
var_part_file = results_dir / "variance_partitioning_all.tsv"  # Variance partitioning

# Ensure required pairwise correlations file exists
if not pairwise_file.exists():
    raise FileNotFoundError(f"Pairwise correlations file not found: {pairwise_file}")

# Load pairwise correlation matrix (required)
# This is a symmetric N×N matrix where N = number of model RDMs
pairwise_df = pd.read_csv(pairwise_file, sep='\t', index_col=0)
logger.info(f"Loaded pairwise correlations: {pairwise_df.shape}")

# Load long-format pairwise correlations with p-values (optional)
# Contains columns: rdm1, rdm2, r, p_raw, p_fdr
if pairwise_long_file.exists():
    pairwise_long_df = pd.read_csv(pairwise_long_file, sep='\t')
    logger.info(f"Loaded pairwise p-values: {pairwise_long_df.shape}")
else:
    pairwise_long_df = None

# Load partial correlations (optional)
# Contains: target, predictor, covariates, r_partial, p_partial, p_fdr
if partial_file.exists():
    partial_df = pd.read_csv(partial_file, sep='\t')
    logger.info(f"Loaded partial correlations: {partial_df.shape}")
else:
    partial_df = None
    logger.warning(f"Partial correlations file not found: {partial_file}")

# Load variance partitioning results (optional)
# Contains: target, r2_full, unique_{model}, shared, residual
if var_part_file.exists():
    var_part_df = pd.read_csv(var_part_file, sep='\t')
    logger.info(f"Loaded variance partitioning: {var_part_df.shape}")
else:
    var_part_df = None
    logger.warning(f"Variance partitioning file not found: {var_part_file}")

# ============================================================================
# Table 1: Pairwise Correlations (Matrix Format) - CSV only
# ============================================================================

logger.info("")
logger.info("Generating pairwise RDM correlations CSV...")

# Reorder models according to CONFIG['MODEL_ORDER'] if available
# This ensures consistent ordering across analyses
model_order = [m for m in CONFIG.get('MODEL_ORDER', []) if m in pairwise_df.index]
model_order += [m for m in pairwise_df.index if m not in model_order]  # Add any remaining models
pairwise_df = pairwise_df.loc[model_order, model_order]  # Reindex both rows and columns

# Create pretty display labels for model names (e.g., "check" -> "Checkmate Status")
pretty_labels = {key: pretty_model_label(key) for key in model_order}
model_names = model_order

# Build upper-triangle table (since correlation matrix is symmetric)
# Only show correlations where i < j to avoid redundancy
table1_data = []
for i, model_i in enumerate(model_names):
    row_data = {'RDM': pretty_labels.get(model_i, model_i.capitalize())}  # Row label
    for j, model_j in enumerate(model_names):
        if i < j:  # Upper triangle: show correlation value
            r = pairwise_df.iloc[i, j]
            row_data[pretty_labels.get(model_j, model_j.capitalize())] = f"{r:.3f}"
        else:  # Lower triangle and diagonal: show dash
            row_data[pretty_labels.get(model_j, pretty_model_label(model_j))] = '-'
    table1_data.append(row_data)

# Convert to DataFrame and save CSV
table1_df = pd.DataFrame(table1_data)
table1_file = tables_dir / "table_pairwise_correlations.csv"
table1_df.to_csv(table1_file, index=False)
logger.info(f"Saved pairwise correlations CSV to {table1_file}")

# ============================================================================
# Table 1b: Pairwise Correlations with P-Values (Long Format) - CSV only
# ============================================================================

# If long-format p-values are available, create a second table with statistical info
if pairwise_long_df is not None:
    logger.info("")
    logger.info("Generating pairwise correlations with p-values CSV...")

    # Format long-format table with pretty model names and p-values
    pairwise_long_df = pairwise_long_df.copy()
    pairwise_long_df['RDM 1'] = pairwise_long_df['rdm1'].map(pretty_model_label)  # First RDM pretty name
    pairwise_long_df['RDM 2'] = pairwise_long_df['rdm2'].map(pretty_model_label)  # Second RDM pretty name
    pairwise_long_df['r'] = pairwise_long_df['r'].round(3)  # Round correlation to 3 decimals
    pairwise_long_df['p'] = pairwise_long_df['p_raw'].apply(lambda x: f"{x:.3g}")  # Format raw p-value
    if 'p_fdr' in pairwise_long_df.columns:  # Add FDR-corrected p-values if available
        pairwise_long_df['pFDR'] = pairwise_long_df['p_fdr'].apply(lambda x: f"{x:.3g}")

    # Select columns to display in table
    display_cols = ['RDM 1','RDM 2','r','p'] + (['pFDR'] if 'pFDR' in pairwise_long_df.columns else [])

    # Save CSV
    table1b_file = tables_dir / "table_pairwise_correlations_p.csv"
    pairwise_long_df[display_cols].to_csv(table1b_file, index=False)
    logger.info(f"Saved pairwise correlations with p-values CSV to {table1b_file}")

# ============================================================================
# Table 2: Variance Partitioning - CSV only
# ============================================================================

# If variance partitioning results are available, generate table
if var_part_df is not None:
    logger.info("")
    logger.info("Generating variance partitioning CSV...")

    # Identify columns containing unique variance for each predictor
    # These columns are named 'unique_{model_name}'
    predictor_cols = [col for col in var_part_df.columns if col.startswith('unique_')]
    predictor_names = [col.replace('unique_', '') for col in predictor_cols]  # Extract model names

    # Select columns to include in table
    # Order: target, full R², unique variance columns, shared variance, residual
    table2_cols = ['target', 'r2_full'] + predictor_cols + ['shared', 'residual']
    table2_df = var_part_df[table2_cols].copy()

    # Create pretty column names for display
    rename_map = {
        'target': 'Target RDM',  # RDM being explained
        'r2_full': 'Full R^2',  # Total variance explained
        'shared': 'Shared',  # Variance explained by multiple predictors
        'residual': 'Residual'  # Unexplained variance (1 - R²)
    }
    # Add unique variance columns with pretty predictor names
    for pred in predictor_names:
        label = pretty_model_label(pred)  # Get pretty name (e.g., "check" -> "Checkmate Status")
        rename_map[f'unique_{pred}'] = f'Unique ({label})'  # Format as "Unique (Pretty Name)"

    # Apply column renaming
    table2_df = table2_df.rename(columns=rename_map)

    # Convert target RDM names to pretty format (e.g., "check" -> "Checkmate Status")
    table2_df['Target RDM'] = table2_df['Target RDM'].map(pretty_model_label)

    # Save CSV
    table2_file = tables_dir / "table_variance_partitioning.csv"
    table2_df.to_csv(table2_file, index=False, float_format='%.3f')
    logger.info(f"Saved variance partitioning CSV to {table2_file}")

# ============================================================================
# Table 3: Partial Correlations - CSV only
# ============================================================================

# If partial correlation results are available, generate table
if partial_df is not None:
    logger.info("")
    logger.info("Generating partial correlations CSV...")

    # Copy dataframe for formatting
    # Partial correlations show the unique relationship between target and predictor
    # after removing shared variance with covariates
    table3_df = partial_df.copy()

    # Format columns with pretty model names
    table3_df['Target RDM'] = table3_df['target'].map(pretty_model_label)  # Target being explained
    table3_df['Predictor'] = table3_df['predictor'].map(pretty_model_label)  # Predictor of interest
    table3_df['Controlled For'] = table3_df['covariates'].str.replace(',', ', ').str.title()  # Covariates
    table3_df['Partial r'] = table3_df['r_partial'].apply(lambda x: f"{x:.3f}")  # Partial correlation
    table3_df['p-value'] = table3_df['p_partial'].apply(lambda x: f"{x:.3g}")  # Raw p-value
    if 'p_fdr' in table3_df.columns:  # Add FDR-corrected p-value if available
        table3_df['pFDR'] = table3_df['p_fdr'].apply(lambda x: f"{x:.3g}")

    # Select columns to display in table
    cols = ['Target RDM', 'Predictor', 'Controlled For', 'Partial r', 'p-value']
    if 'pFDR' in table3_df.columns:
        cols.append('pFDR')  # Add FDR column if present
    table3_display = table3_df[cols]

    # Save CSV
    table3_file = tables_dir / "table_partial_correlations.csv"
    table3_display.to_csv(table3_file, index=False)
    logger.info(f"Saved partial correlations CSV to {table3_file}")

# ============================================================================
# Finish
# ============================================================================

logger.info("")
logger.info("=" * 80)
logger.info("CSV table generation complete!")
logger.info(f"CSV files saved to: {tables_dir}")
logger.info("=" * 80)

log_script_end(logger)
