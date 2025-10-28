#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Behavioral RSA Results - Visualization Only

This script loads precomputed behavioral RDM results from 01_behavioral_rsa.py
and generates all figures and LaTeX tables for the manuscript.

NO ANALYSIS: This script only creates visualizations from saved results.

Inputs (from results directory):
    - expert_behavioral_rdm.npy: Expert group RDM
    - novice_behavioral_rdm.npy: Novice group RDM
    - expert_directional_dsm.npy: Expert directional preference matrix
    - novice_directional_dsm.npy: Novice directional preference matrix
    - pairwise_data.pkl: Pairwise comparison DataFrames
    - model_rdms.pkl: Model RDM matrices (dict)
    - correlation_results.pkl: Correlation statistics

Outputs (saved to same results directory):
    - All figures as PDF files (publication-ready, 300 dpi)
    - LaTeX tables for manuscript (.tex files)
    - Figure summary log
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add paths for imports
script_dir = Path(__file__).parent
repo_root = script_dir.parent

sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(script_dir))

from common import CONFIG, load_stimulus_metadata
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory, copy_script_to_results
from common.report_utils import create_figure_summary, generate_latex_table
from common.plotting_utils import plot_rdm, compute_stimulus_palette, plot_2d_embedding
from modules.plotting import (
    plot_choice_frequency,
    plot_model_correlations,
)

# ============================================================================
# Configuration
# ============================================================================

# IMPORTANT: Set this to the results directory you want to visualize
# Use the full timestamped directory name, e.g., "20251017-174511_behavioral_rsa"
# Or use None to automatically find the most recent results directory
RESULTS_DIR_NAME = None  # Set to None for latest, or specify directory name

# Base results directory
RESULTS_BASE = script_dir / "results"

# ============================================================================
# Find Results Directory and Create Subdirectories
# ============================================================================

# Use centralized function - handles finding directory AND creating subdirs
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern="*_behavioral_rsa",
    specific_name=RESULTS_DIR_NAME,
    create_subdirs=["figures", "tables"],  # Automatically creates these
    require_exists=True,
    verbose=True,  # Prints which directory is being used
)

# Get subdirectory paths (already created by find_latest_results_directory)
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ============================================================================
# Setup (DRY via common.setup_analysis_in_dir)
# ============================================================================

extra = {
    "RESULTS_DIR": str(RESULTS_DIR),
    "FIGURES_DIR": str(FIGURES_DIR),
    "TABLES_DIR": str(TABLES_DIR),
}
config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name="plotting.log",
)

# ============================================================================
# Load All Results
# ============================================================================

logger.info("Loading precomputed results...")

# Load RDMs
expert_rdm = np.load(RESULTS_DIR / "expert_behavioral_rdm.npy")
novice_rdm = np.load(RESULTS_DIR / "novice_behavioral_rdm.npy")
logger.info(
    f"  Loaded behavioral RDMs: expert={expert_rdm.shape}, novice={novice_rdm.shape}"
)

# Load DSMs
expert_dsm = np.load(RESULTS_DIR / "expert_directional_dsm.npy")
novice_dsm = np.load(RESULTS_DIR / "novice_directional_dsm.npy")
logger.info(
    f"  Loaded directional DSMs: expert={expert_dsm.shape}, novice={novice_dsm.shape}"
)

# Load model RDMs
with open(RESULTS_DIR / "model_rdms.pkl", "rb") as f:
    model_rdms_dict = pickle.load(f)
logger.info(f"  Loaded {len(model_rdms_dict)} model RDMs: {list(model_rdms_dict.keys())}")

# Load correlation results
with open(RESULTS_DIR / "correlation_results.pkl", "rb") as f:
    correlation_results = pickle.load(f)
expert_corrs = correlation_results["expert"]
novice_corrs = correlation_results["novice"]
model_columns = correlation_results["model_columns"]
logger.info(f"  Loaded correlation results for {len(model_columns)} models")

# Load pairwise data
with open(RESULTS_DIR / "pairwise_data.pkl", "rb") as f:
    pairwise_data = pickle.load(f)
expert_pairwise = pairwise_data["expert_pairwise"]
novice_pairwise = pairwise_data["novice_pairwise"]
logger.info(
    f"  Loaded pairwise data: {len(expert_pairwise)} expert, {len(novice_pairwise)} novice comparisons"
)


# Load stimulus metadata (for colors/alphas in plots)
stimuli_df = load_stimulus_metadata()
logger.info(f"  Loaded stimulus metadata for {len(stimuli_df)} stimuli")

logger.info("All data loaded successfully\n")

# ============================================================================
# Generate Figures
# ============================================================================

logger.info("Generating figures...")

# Precompute stimulus colors/alphas once
strat_colors, strat_alphas = compute_stimulus_palette(stimuli_df)

# ============================================================================
# CRITICAL: Compute global colorscale for ALL behavioral RDMs/DSMs
# ============================================================================
# To enable visual comparison, all behavioral matrices (symmetric RDMs and
# directional DSMs, for both experts and novices) must use the same colorscale.
# We compute the maximum absolute value across all 4 matrices and use it as
# vmin=-global_max and vmax=+global_max for all plots.

global_max_abs = max(
    np.abs(expert_rdm).max(),
    np.abs(novice_rdm).max(),
    np.abs(expert_dsm).max(),
    np.abs(novice_dsm).max()
)
behavioral_vmin = -global_max_abs
behavioral_vmax = global_max_abs

logger.info(f"  Global colorscale for behavioral RDMs: [{behavioral_vmin:.2f}, {behavioral_vmax:.2f}]")

# 1. Expert Behavioral RDM
plot_rdm(
    expert_rdm,
    title="Behavioral RDM",
    subtitle="Experts",
    output_path=FIGURES_DIR / "behavioral_rdm_Experts.pdf",
    colors=strat_colors,
    alphas=strat_alphas,
    colorbar_label="Preference Dissimilarity",
    vmin=behavioral_vmin,
    vmax=behavioral_vmax,
)
logger.info("  ✓ Expert behavioral RDM")

# 2. Novice Behavioral RDM
plot_rdm(
    novice_rdm,
    title="Behavioral RDM",
    subtitle="Novices",
    output_path=FIGURES_DIR / "behavioral_rdm_Novices.pdf",
    colors=strat_colors,
    alphas=strat_alphas,
    colorbar_label="Preference Dissimilarity",
    vmin=behavioral_vmin,
    vmax=behavioral_vmax,
)
logger.info("  ✓ Novice behavioral RDM")

# 3. Expert Directional DSM (uses CMAP_BRAIN with center=0)
plot_rdm(
    expert_dsm,
    title="Directional Preference",
    subtitle="Experts",
    output_path=FIGURES_DIR / "directional_dsm_Experts.pdf",
    colors=strat_colors,
    alphas=strat_alphas,
    colorbar_label="Preference",
    vmin=behavioral_vmin,
    vmax=behavioral_vmax,
)
logger.info("  ✓ Expert directional DSM")

# 4. Novice Directional DSM (uses CMAP_BRAIN with center=0)
plot_rdm(
    novice_dsm,
    title="Directional Preference",
    subtitle="Novices",
    output_path=FIGURES_DIR / "directional_dsm_Novices.pdf",
    colors=strat_colors,
    alphas=strat_alphas,
    colorbar_label="Preference",
    vmin=behavioral_vmin,
    vmax=behavioral_vmax,
)
logger.info("  ✓ Novice directional DSM")

# 5. MDS Embeddings (use precomputed coordinates)
expert_mds_coords = np.load(RESULTS_DIR / "expert_mds_coords.npy")
novice_mds_coords = np.load(RESULTS_DIR / "novice_mds_coords.npy")

plot_2d_embedding(
    coords=expert_mds_coords,
    title="MDS Embedding of RDM",
    subtitle="Experts",
    output_path=FIGURES_DIR / "mds_embedding_Experts.pdf",
    point_colors=strat_colors,
    point_alphas=strat_alphas,
)
logger.info("  ✓ Expert MDS embedding")

plot_2d_embedding(
    coords=novice_mds_coords,
    title="MDS Embedding of RDM",
    subtitle="Novices",
    output_path=FIGURES_DIR / "mds_embedding_Novices.pdf",
    point_colors=strat_colors,
    point_alphas=strat_alphas,
)
logger.info("  ✓ Novice MDS embedding")

# 6. Choice Frequency
plot_choice_frequency(
    expert_pairwise, "Experts", FIGURES_DIR / "choice_frequency_Experts.pdf", stimuli_df
)
logger.info("  ✓ Expert choice frequency")

plot_choice_frequency(
    novice_pairwise, "Novices", FIGURES_DIR / "choice_frequency_Novices.pdf", stimuli_df
)
logger.info("  ✓ Novice choice frequency")

# 7. Model RDMs (all use CMAP_BRAIN with center=0)
for model_name, model_rdm in model_rdms_dict.items():
    plot_rdm(
        model_rdm,
        title=f"{model_name.capitalize()} Model",
        output_path=FIGURES_DIR / f"model_rdm_{model_name}_Experts.pdf",
        colors=strat_colors,
        alphas=strat_alphas,
        show_colorbar=False
    )

logger.info(f"  ✓ {len(model_rdms_dict)} model RDM figures")

# 8. Correlation Comparison (Expert vs Novice)
plot_model_correlations(
    expert_corrs,
    novice_corrs,
    FIGURES_DIR / "behavioral_model_correlations_comparison.pdf",
)
logger.info("  ✓ Correlation comparison figure")

# ============================================================================
# Generate LaTeX Tables
# ============================================================================

logger.info("\nGenerating LaTeX tables...")

# Load summary DataFrame (keep p-values as strings to preserve scientific notation)
summary_df = pd.read_csv(
    RESULTS_DIR / "correlation_summary.csv", dtype={"p_Experts": str, "p_Novices": str}
)

# Define multicolumn headers for Expert vs Novice comparison
multicolumn_headers = {
    "Experts": ["r_Experts", "95%_CI_Experts", "p_Experts"],
    "Novices": ["r_Novices", "95%_CI_Novices", "p_Novices"],
}

# Use centralized function to generate LaTeX table with multicolumn headers
latex_path = generate_latex_table(
    df=summary_df,
    output_path=TABLES_DIR / "correlation_table.tex",
    caption="Behavioral RDM correlations with model RDMs (Experts vs. Novices). "
    "Pearson correlation coefficients with 95\\% confidence intervals computed via bootstrapping.",
    label="tab:behavioral_correlations",
    multicolumn_headers=multicolumn_headers,
    column_format="lcccccc",
    escape=False,
    logger=logger,
)

logger.info(f"  ✓ LaTeX table: {latex_path.name}")

# ============================================================================
# Create Figure Summary
# ============================================================================

logger.info("\nCreating figure summary...")

# Use centralized function to create figure summary
summary_path = create_figure_summary(
    results_dir=RESULTS_DIR,
    figures_dir=FIGURES_DIR,
    tables_dir=TABLES_DIR,
    analysis_name="Behavioral RSA",
    logger=logger,
)

# ============================================================================
# Finish
# ============================================================================

log_script_end(logger)
logger.info(f"\n{'='*80}")
logger.info(f"All outputs saved to: {RESULTS_DIR}")
logger.info(f"{'='*80}")
