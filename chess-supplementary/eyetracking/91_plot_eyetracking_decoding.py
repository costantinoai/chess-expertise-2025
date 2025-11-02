"""
Generate Eyetracking Decoding Supplementary Figure
==================================================

Creates a publication-ready two-panel figure showing expert vs novice decoding
performance from eyetracking data. Compares two feature representations:
2D gaze coordinates (x, y) and displacement from center. Uses stratified
cross-validation fold accuracies to visualize decoding reliability.

Figure Produced
---------------

Eyetracking Decoding Panel (2 panels side-by-side)
- File: figures/panels/eyetracking_decoding_panel.svg (and .pdf)
- Individual axes saved to figures/: eyetracking_decoding_*.svg/pdf

Panel A: 2D Gaze (x, y) Features
- Shows cross-validation fold accuracies as jittered dots
- Mean accuracy displayed as horizontal orange line
- 95% CI shown as shaded band
- Chance level (0.5) shown as gray dashed line

Panel B: Displacement Features
- Same visualization as Panel A but for displacement from center
- Displacement = Euclidean distance from board center

Visual Elements (both panels):
- Blue dots: Individual fold accuracies (with horizontal jitter for visibility)
- Orange line: Mean accuracy across folds
- Orange band: 95% confidence interval
- Gray dashed line: Chance level (0.5 for binary classification)
- Y-axis: Accuracy (0 to 1.0)
- X-axis: Hidden (strip plot, position has no meaning)

Inputs
------
- results_xy.json (from eyetracking decoding analysis)
  Contains: fold_accuracies, mean_accuracy, ci_low, ci_high, n_folds
- results_displacement.json (from eyetracking decoding analysis)
  Contains: fold_accuracies, mean_accuracy, ci_low, ci_high, n_folds

Dependencies
------------
- common.plotting primitives (apply_nature_rc, save_axes_svgs)
- Strict I/O: fails if expected results are missing; no silent fallbacks

Usage
-----
python chess-supplementary/eyetracking/91_plot_eyetracking_decoding.py
"""

import os
import sys
from pathlib import Path
import json

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
script_dir = Path(__file__).parent

import numpy as np
import matplotlib.pyplot as plt

from common.plotting import (
    apply_nature_rc,
    save_axes_svgs,
    save_panel_pdf,
    figure_size,
    PLOT_PARAMS,
    COLORS_WONG,
)
from common.io_utils import find_latest_results_directory
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

# =============================================================================
# Configuration and Results Loading
# =============================================================================
# Find latest eyetracking decoding results directory and load JSON outputs
# containing cross-validation fold accuracies for both feature sets

apply_nature_rc()  # Apply Nature journal style to all figures

# Find latest results directory (fails if not found)
RESULTS_DIR = find_latest_results_directory(
    script_dir / 'results',
    pattern='*_eyetracking_decoding',
    create_subdirs=['figures'],
    require_exists=True,
    verbose=True,
)
FIGURES_DIR = RESULTS_DIR / 'figures'

# Initialize logging in existing results directory
config, out_dir, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
)

# Load decoding results for both feature representations
# Each JSON contains: fold_accuracies (list), mean_accuracy, ci_low, ci_high, n_folds
with open(RESULTS_DIR / 'results_xy.json', 'r') as f:
    results_xy = json.load(f)       # 2D gaze coordinates (x, y)
with open(RESULTS_DIR / 'results_displacement.json', 'r') as f:
    results_disp = json.load(f)     # Displacement from center

logger.info(f"Loaded results: xy (n_folds={results_xy['n_folds']}), displacement (n_folds={results_disp['n_folds']})")

# Extract xy features data
# - fold_accuracies: accuracy for each cross-validation fold
# - mean_accuracy: average accuracy across folds
# - ci_low, ci_high: 95% confidence interval bounds
xy_acc = np.array(results_xy['fold_accuracies'])
xy_mean = results_xy['mean_accuracy']
xy_ci_low = results_xy['ci_low']
xy_ci_high = results_xy['ci_high']

# Extract displacement features data (same structure as xy)
disp_acc = np.array(results_disp['fold_accuracies'])
disp_mean = results_disp['mean_accuracy']
disp_ci_low = results_disp['ci_low']
disp_ci_high = results_disp['ci_high']

# =============================================================================
# Create Two-Panel Figure
# =============================================================================
# Panel A: 2D gaze (x, y) features
# Panel B: Displacement from center features
# Both panels show fold accuracies, mean, 95% CI, and chance level

fig, axes = plt.subplots(1, 2, figsize=figure_size(columns=2, height_mm=100), sharey=True)

# -----------------------------------------------------------------------------
# Panel A: 2D Gaze (x, y) Features
# -----------------------------------------------------------------------------
# Shows decoding performance using raw 2D gaze coordinates
# Jittered strip plot allows visualization of all fold accuracies
ax = axes[0]

# Plot individual fold accuracies as jittered dots
# Jitter is horizontal only (vertical position = actual accuracy)
x_jitter = np.random.default_rng(seed=42).normal(0, 0.02, size=len(xy_acc))
ax.scatter(
    x_jitter, xy_acc,
    alpha=0.6,                     # Semi-transparent for overlapping points
    s=PLOT_PARAMS['marker_size'],  # Point size from central settings
    color=COLORS_WONG['blue'],     # Blue color from central palette
    zorder=3,                      # Draw on top of lines/bands
    label='Fold accuracies'
)

# Plot mean accuracy as horizontal line
ax.axhline(
    xy_mean,
    color=COLORS_WONG['orange'],   # Orange color from central palette
    linestyle='-',                 # Solid line
    linewidth=PLOT_PARAMS['plot_linewidth'],
    label='Mean accuracy',
    zorder=2                       # Behind dots, in front of CI band
)

# Plot 95% confidence interval as shaded band
ax.axhspan(
    xy_ci_low, xy_ci_high,
    alpha=0.2,                     # Transparent band
    color=COLORS_WONG['orange'],   # Orange (matches mean line)
    zorder=1                       # Behind everything else
)

# Plot chance level (0.5 for binary classification)
ax.axhline(
    0.5,
    color='gray',                  # Gray color
    linestyle='--',                # Dashed line
    linewidth=PLOT_PARAMS['plot_linewidth'],
    label='Chance',
    zorder=2
)

# Configure panel A axes and labels
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.0)                # Full accuracy range
ax.set_xlim(-0.15, 0.15)           # Jitter range
ax.set_xticks([])                  # No x-ticks (strip plot)
ax.set_title('2D gaze (x, y)', fontsize=PLOT_PARAMS['font_size_title'])
ax.legend(loc='lower right', frameon=False, fontsize=PLOT_PARAMS['font_size_legend'])

# -----------------------------------------------------------------------------
# Panel B: Displacement Features
# -----------------------------------------------------------------------------
# Shows decoding performance using displacement from board center
# Same visualization format as Panel A for easy comparison
ax = axes[1]

# Plot individual fold accuracies as jittered dots
x_jitter = np.random.default_rng(seed=42).normal(0, 0.02, size=len(disp_acc))
ax.scatter(
    x_jitter, disp_acc,
    alpha=0.6,
    s=PLOT_PARAMS['marker_size'],  # Point size from central settings
    color=COLORS_WONG['blue'],     # Blue (same as Panel A)
    zorder=3,
    label='Fold accuracies'
)

# Plot mean accuracy as horizontal line
ax.axhline(
    disp_mean,
    color=COLORS_WONG['orange'],   # Orange (same as Panel A)
    linestyle='-',
    linewidth=PLOT_PARAMS['plot_linewidth'],
    label='Mean accuracy',
    zorder=2
)

# Plot 95% confidence interval as shaded band
ax.axhspan(
    disp_ci_low, disp_ci_high,
    alpha=0.2,
    color=COLORS_WONG['orange'],
    zorder=1
)

# Plot chance level
ax.axhline(
    0.5,
    color='gray',
    linestyle='--',
    linewidth=PLOT_PARAMS['plot_linewidth'],
    label='Chance',
    zorder=2
)

# Configure panel B axes and labels
ax.set_ylim(0, 1.0)                # Full accuracy range
ax.set_xlim(-0.15, 0.15)           # Jitter range
ax.set_xticks([])                  # No x-ticks (strip plot)
ax.set_title('Displacement', fontsize=PLOT_PARAMS['font_size_title'])
ax.legend(loc='lower right', frameon=False, fontsize=PLOT_PARAMS['font_size_legend'])


# =============================================================================
# Save Figures (Individual Axes and Full Panel)
# =============================================================================
# Save both individual axes (for modular figure assembly) and complete panel

# Save individual axes as separate SVG/PDF files
save_axes_svgs(fig, FIGURES_DIR, prefix='eyetracking_decoding')

# Save complete panel as single SVG/PDF file
(FIGURES_DIR / 'panels').mkdir(exist_ok=True)
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'eyetracking_decoding_panel.pdf')

logger.info(f"Saved figures to {FIGURES_DIR}")
logger.info("✓ Panel: eyetracking decoding complete")

log_script_end(logger)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(8.900000/2.54, 5.000000/2.54, forward=True)
plt.figure(1).axes[0].legend(loc=(0.4821, 0.7996), frameon=False)
plt.figure(1).axes[1].legend(loc=(0.4821, 0.7996), frameon=False)
#% end: automatic generated code from pylustrator
plt.show()
