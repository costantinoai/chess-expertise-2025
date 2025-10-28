"""
Participation Ratio Results Visualization
==========================================

This script loads results from 01_manifold_analysis.py and generates all
publication-ready figures and tables.

Outputs:
- pr_roi_group_means.pdf: Group means (experts vs novices) per ROI
- pr_roi_differences.pdf: Expert − Novice differences per ROI
- pr_voxels_*.pdf: Scatter plots of PR vs voxel count (separate files)
- pr_mds_projection.pdf: PCA projection (2D)
- pr_matrix_and_loadings.pdf: PR matrix heatmap + PCA loadings
- pr_feature_importance.pdf: ROI contributions to classification
- pr_results_table.tex: LaTeX table for publication
- pr_results_table.csv: CSV version of results table

Usage:
    python 02_plot_manifold_results.py

Note: Requires results from 01_manifold_analysis.py to exist.
"""

import sys
from pathlib import Path

# Setup import paths
script_dir = Path(__file__).parent
repo_root = script_dir.parent

sys.path.insert(0, str(repo_root))    # Enables: from common import ...
sys.path.insert(0, str(script_dir))   # Enables: from modules import ...

import pickle
import pandas as pd
from common import CONFIG
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from modules.plotting import (
    plot_pr_roi_bars,
    plot_pr_voxel_correlations,
    plot_pr_matrix_and_loadings,
    plot_pr_feature_importance,
    PLOT_PARAMS
)
from common.plotting import plot_2d_embedding, COLORS_EXPERT_NOVICE
from modules.tables import generate_pr_results_table

# ============================================================================
# Configuration
# ============================================================================

# IMPORTANT: Set this to the results directory you want to visualize
# Use the full timestamped directory name, e.g., "20251018-233210_manifold"
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
    pattern="*_manifold",
    specific_name=RESULTS_DIR_NAME,
    create_subdirs=["figures", "tables"],  # Automatically creates these
    require_exists=True,
    verbose=True,  # Prints which directory is being used
)

# Get subdirectory paths (already created by find_latest_results_directory)
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ============================================================================
# Setup (via common.setup_analysis_in_dir)
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
# Load Results
# ============================================================================

results_path = RESULTS_DIR / "pr_results.pkl"

if not results_path.exists():
    raise FileNotFoundError(
        f"Results file not found: {results_path}\n"
        "Run 01_manifold_analysis.py first."
    )

with open(results_path, 'rb') as f:
    results = pickle.load(f)

# Unpack results
pr_df = results['pr_long_format']
summary_stats = results['summary_stats']
stats_results = results['stats_results']
roi_info = results['roi_info']
participants = results['participants']
roi_labels = results['roi_labels']
classifier = results['classifier']
scaler = results['scaler']
pca2d = results.get('pca2d', None)
pr_matrix_pack = results.get('pr_matrix', None)
voxel_corr = results.get('voxel_corr', None)

# ============================================================================
# Plot 1: ROI Comparison (Combined Two-Panel Figure)
# ============================================================================

combined_path = plot_pr_roi_bars(
    summary_stats=summary_stats,
    stats_results=stats_results,
    roi_info=roi_info,
    output_dir=FIGURES_DIR,
    alpha=config['ALPHA'],
    use_fdr=True,
    figsize=(19, 17),
)

# ============================================================================
# Plot 2: PR vs Voxel Count Correlations
# ============================================================================

if voxel_corr is None:
    raise RuntimeError("voxel_corr not found in results. Re-run 01_manifold_analysis.py.")

plot_pr_voxel_correlations(
    group_avg=voxel_corr['group_avg'],
    diff_data=voxel_corr['diff_data'],
    stats=voxel_corr['stats'],
    output_dir=FIGURES_DIR,
    figsize = (10,10)
)

# ============================================================================
# Plot 3: MDS of PR Profiles
# ============================================================================

coords = pca2d['coords']
expl = pca2d['explained']
labels = pca2d['labels']
bnd = pca2d['boundary']

# Map labels to colors
point_colors = [COLORS_EXPERT_NOVICE['expert'] if lbl == 1 else COLORS_EXPERT_NOVICE['novice'] for lbl in labels]
point_alphas = [0.7] * len(labels)

# Create custom params with doubled text sizes for MDS projection
mds_params = PLOT_PARAMS.copy()

plot_2d_embedding(
    coords=coords,
    title='PCA Projection of PR Profiles',
    subtitle='',
    output_path=FIGURES_DIR / 'pr_mds_projection.pdf',
    point_colors=point_colors,
    point_alphas=point_alphas,
    x_label=f'PC1 ({expl[0]:.1f}% var)',
    y_label=f'PC2 ({expl[1]:.1f}% var)',
    fill={
        'xx': bnd['xx'],
        'yy': bnd['yy'],
        'Z': bnd['Z'],
        'colors': [COLORS_EXPERT_NOVICE['novice'], COLORS_EXPERT_NOVICE['expert']],
        'alpha': 0.15,
        'levels': [0, 0.5, 1],
    },
    params=mds_params,  # Use doubled text sizes
)

# ============================================================================
# Plot 4: PR Matrix + PCA Loadings (stacked)
# ============================================================================

if pr_matrix_pack is None or pca2d is None:
    raise RuntimeError("pr_matrix or pca2d missing in results. Re-run 01_manifold_analysis.py.")

plot_pr_matrix_and_loadings(
    pr_matrix=pr_matrix_pack['matrix'],
    n_experts=pr_matrix_pack['n_experts'],
    roi_labels=roi_labels,
    output_dir=FIGURES_DIR,
    pca_components=pca2d['components'],
    roi_info=roi_info,
    figsize=(10, 14),  # Increased height by 4x total for better readability
)

# ============================================================================
# Plot 5: Feature Importance
# ============================================================================

plot_pr_feature_importance(
    clf=classifier,
    roi_info=roi_info,
    output_dir=FIGURES_DIR,
    top_n=10,
    figsize = (14,8)
)


# ============================================================================
# Generate LaTeX Table
# ============================================================================

tex_path, csv_path = generate_pr_results_table(
    summary_stats=summary_stats,
    stats_results=stats_results,
    roi_info=roi_info,
    output_dir=TABLES_DIR,
    use_fdr=True,
)


log_script_end(logger)
