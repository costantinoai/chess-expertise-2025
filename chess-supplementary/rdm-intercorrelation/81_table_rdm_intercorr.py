#!/usr/bin/env python3
"""
Generate LaTeX tables for RDM intercorrelation analysis.

This script loads results from 01_rdm_intercorrelation.py and generates
publication-ready LaTeX tables for:
1. Pairwise correlations between RDMs
2. Partial correlations controlling for confounds
3. Variance partitioning for each target RDM

Tables are saved to tables/ subdirectory within the latest results folder.
"""

import sys
from pathlib import Path

# Enable imports from repo root
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
from common import CONFIG
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from modules import pretty_model_label


# === Find latest results directory ===
results_base = Path(__file__).parent / "results"
results_dir = find_latest_results_directory(
    results_base,
    pattern="*_rdm_intercorrelation",
    create_subdirs=["tables"],
    require_exists=True,
    verbose=True,
)

extra = {"RESULTS_DIR": str(results_dir)}
config, _, logger = setup_analysis_in_dir(
    results_dir,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name="table_rdm_intercorr.log",
)

logger.info("=" * 80)
logger.info("GENERATING LATEX TABLES FOR RDM INTERCORRELATION")
logger.info("=" * 80)

# Create tables subdirectory
tables_dir = results_dir / "tables"
logger.info(f"Tables will be saved to: {tables_dir}")

# === Load data ===
logger.info("")
logger.info("Loading analysis results...")

pairwise_file = results_dir / "pairwise_correlations.tsv"
pairwise_long_file = results_dir / "pairwise_correlations_long.tsv"
partial_file = results_dir / "partial_correlations.tsv"
var_part_file = results_dir / "variance_partitioning_all.tsv"

if not pairwise_file.exists():
    raise FileNotFoundError(f"Pairwise correlations file not found: {pairwise_file}")

pairwise_df = pd.read_csv(pairwise_file, sep='\t', index_col=0)
logger.info(f"Loaded pairwise correlations: {pairwise_df.shape}")

if pairwise_long_file.exists():
    pairwise_long_df = pd.read_csv(pairwise_long_file, sep='\t')
    logger.info(f"Loaded pairwise p-values: {pairwise_long_df.shape}")
else:
    pairwise_long_df = None

if partial_file.exists():
    partial_df = pd.read_csv(partial_file, sep='\t')
    logger.info(f"Loaded partial correlations: {partial_df.shape}")
else:
    partial_df = None
    logger.warning(f"Partial correlations file not found: {partial_file}")

if var_part_file.exists():
    var_part_df = pd.read_csv(var_part_file, sep='\t')
    logger.info(f"Loaded variance partitioning: {var_part_df.shape}")
else:
    var_part_df = None
    logger.warning(f"Variance partitioning file not found: {var_part_file}")

# === Table 1: Pairwise Correlations (matrix) ===
logger.info("")
logger.info("Generating Table 1: Pairwise RDM correlations...")

model_order = [m for m in CONFIG.get('MODEL_ORDER', []) if m in pairwise_df.index]
model_order += [m for m in pairwise_df.index if m not in model_order]
pairwise_df = pairwise_df.loc[model_order, model_order]

pretty_labels = {key: pretty_model_label(key) for key in model_order}
model_names = model_order

# Only show upper triangle (since matrix is symmetric)
table1_data = []
for i, model_i in enumerate(model_names):
    row_data = {'RDM': pretty_labels.get(model_i, model_i.capitalize())}
    for j, model_j in enumerate(model_names):
        if i < j:
            r = pairwise_df.iloc[i, j]
            row_data[pretty_labels.get(model_j, model_j.capitalize())] = f"{r:.3f}"
        else:
            row_data[pretty_labels.get(model_j, pretty_model_label(model_j))] = '-'
    table1_data.append(row_data)

table1_df = pd.DataFrame(table1_data)

# Generate LaTeX
latex1 = table1_df.to_latex(
    index=False,
    escape=False,
    caption='Pairwise Spearman correlations between model RDMs. Only upper triangle shown (matrix is symmetric).',
    label='tab:rdm_pairwise_corr'
)

table1_file = tables_dir / "table_pairwise_correlations.tex"
with open(table1_file, 'w') as f:
    f.write(latex1)
logger.info(f"Saved Table 1 to {table1_file}")

# === Table 1b: Pairwise Correlations with p-values (long format) ===
if pairwise_long_df is not None:
    logger.info("")
    logger.info("Generating Table 1b: Pairwise correlations with p-values...")

    # Pretty names
    pairwise_long_df = pairwise_long_df.copy()
    pairwise_long_df['RDM 1'] = pairwise_long_df['rdm1'].map(pretty_model_label)
    pairwise_long_df['RDM 2'] = pairwise_long_df['rdm2'].map(pretty_model_label)
    pairwise_long_df['r'] = pairwise_long_df['r'].round(3)
    pairwise_long_df['p'] = pairwise_long_df['p_raw'].apply(lambda x: f"{x:.3g}")
    if 'p_fdr' in pairwise_long_df.columns:
        pairwise_long_df['pFDR'] = pairwise_long_df['p_fdr'].apply(lambda x: f"{x:.3g}")

    display_cols = ['RDM 1','RDM 2','r','p'] + (['pFDR'] if 'pFDR' in pairwise_long_df.columns else [])
    latex1b = pairwise_long_df[display_cols].to_latex(
        index=False,
        escape=False,
        caption='Pairwise Spearman correlations between model RDMs with raw and FDR-corrected p-values.',
        label='tab:rdm_pairwise_corr_p'
    )

    table1b_file = tables_dir / "table_pairwise_correlations_p.tex"
    with open(table1b_file, 'w') as f:
        f.write(latex1b)
    logger.info(f"Saved Table 1b to {table1b_file}")

# === Table 2: Variance Partitioning ===
if var_part_df is not None:
    logger.info("")
    logger.info("Generating Table 2: Variance partitioning...")

    # Identify unique and shared columns
    predictor_cols = [col for col in var_part_df.columns if col.startswith('unique_')]
    predictor_names = [col.replace('unique_', '') for col in predictor_cols]

    # Select relevant columns
    table2_cols = ['target', 'r2_full'] + predictor_cols + ['shared', 'residual']
    table2_df = var_part_df[table2_cols].copy()

    # Rename columns for display
    rename_map = {
        'target': 'Target RDM',
        'r2_full': 'Full $R^2$',
        'shared': 'Shared',
        'residual': 'Residual'
    }
    for pred in predictor_names:
        label = pretty_model_label(pred)
        rename_map[f'unique_{pred}'] = f'Unique ({label})'

    table2_df = table2_df.rename(columns=rename_map)

    # Capitalize target names
    table2_df['Target RDM'] = table2_df['Target RDM'].map(pretty_model_label)

    # Generate LaTeX
    latex2 = table2_df.to_latex(
        index=False,
        escape=False,
        float_format='%.3f',
        caption='Variance partitioning for each target RDM. Full $R^2$ shows total variance explained by all predictors. Unique values show variance explained exclusively by each predictor. Shared shows variance explained jointly by multiple predictors. Residual shows unexplained variance. All components sum to 1.0.',
        label='tab:rdm_variance_partitioning'
    )

    table2_file = tables_dir / "table_variance_partitioning.tex"
    with open(table2_file, 'w') as f:
        f.write(latex2)
    logger.info(f"Saved Table 2 to {table2_file}")

# === Table 3: Partial Correlations (selected comparisons) ===
if partial_df is not None:
    logger.info("")
    logger.info("Generating Table 3: Partial correlations (selected)...")

    # For readability, only show partial correlations where target != predictor
    # and organize by target
    table3_df = partial_df.copy()

    # Format for display
    table3_df['Target RDM'] = table3_df['target'].map(pretty_model_label)
    table3_df['Predictor'] = table3_df['predictor'].map(pretty_model_label)
    table3_df['Controlled For'] = table3_df['covariates'].str.replace(',', ', ').str.title()
    table3_df['Partial $r$'] = table3_df['r_partial'].apply(lambda x: f"{x:.3f}")
    table3_df['$p$-value'] = table3_df['p_partial'].apply(lambda x: f"{x:.3g}")
    if 'p_fdr' in table3_df.columns:
        table3_df['$p_{FDR}$'] = table3_df['p_fdr'].apply(lambda x: f"{x:.3g}")

    # Select columns for table
    cols = ['Target RDM', 'Predictor', 'Controlled For', 'Partial $r$', '$p$-value']
    if '$p_{FDR}$' in table3_df.columns:
        cols.append('$p_{FDR}$')
    table3_display = table3_df[cols]

    # Generate LaTeX
    latex3 = table3_display.to_latex(
        index=False,
        escape=False,
        caption='Partial Spearman correlations between RDMs controlling for confounding RDMs. Each row shows the correlation between target and predictor RDMs after partialing out the influence of covariates.',
        label='tab:rdm_partial_corr'
    )

    table3_file = tables_dir / "table_partial_correlations.tex"
    with open(table3_file, 'w') as f:
        f.write(latex3)
    logger.info(f"Saved Table 3 to {table3_file}")

# === Summary ===
logger.info("")
logger.info("=" * 80)
logger.info("LaTeX table generation complete!")
logger.info(f"Tables saved to: {tables_dir}")
logger.info("=" * 80)

log_script_end(logger)
