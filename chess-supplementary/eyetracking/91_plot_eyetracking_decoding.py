"""
Eyetracking decoding plot: two-panel figure showing fold accuracies for xy and displacement features.

Creates a 2-panel figure:
- Left: xy features (2D gaze coordinates)
- Right: displacement features (distance from center)

Each panel shows:
- Individual fold accuracies as dots (stripplot with jitter)
- Mean accuracy as solid horizontal line
- Chance level (0.5) as dashed horizontal line
- 95% CI as shaded band

Run from IDE or CLI (no arguments required).
"""

import os
import sys
from pathlib import Path
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
script_dir = Path(__file__).parent

import numpy as np
import matplotlib.pyplot as plt

from common.plotting import apply_nature_rc, save_axes_svgs, save_panel_svg, save_axes_pdfs, save_panel_pdf
from common.io_utils import find_latest_results_directory
from common.logging_utils import setup_analysis_in_dir, log_script_end


apply_nature_rc()

RESULTS_DIR = find_latest_results_directory(
    script_dir / 'results',
    pattern='*_eyetracking_decoding',
    create_subdirs=['figures'],
    require_exists=True,
    verbose=True,
)
FIGURES_DIR = RESULTS_DIR / 'figures'

config, out_dir, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
)

# Load results for both feature sets
with open(RESULTS_DIR / 'results_xy.json', 'r') as f:
    results_xy = json.load(f)
with open(RESULTS_DIR / 'results_displacement.json', 'r') as f:
    results_disp = json.load(f)

logger.info(f"Loaded results: xy (n_folds={results_xy['n_folds']}), displacement (n_folds={results_disp['n_folds']})")

# Extract data
xy_acc = np.array(results_xy['fold_accuracies'])
xy_mean = results_xy['mean_accuracy']
xy_ci_low = results_xy['ci_low']
xy_ci_high = results_xy['ci_high']

disp_acc = np.array(results_disp['fold_accuracies'])
disp_mean = results_disp['mean_accuracy']
disp_ci_low = results_disp['ci_low']
disp_ci_high = results_disp['ci_high']

# Create figure with 2 panels
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# Panel A: xy features
ax = axes[0]
# Plot fold accuracies as strip plot (with jitter)
x_jitter = np.random.default_rng(seed=42).normal(0, 0.02, size=len(xy_acc))
ax.scatter(x_jitter, xy_acc, alpha=0.6, s=40, color='#1f77b4', zorder=3, label='Fold accuracies')
# Mean accuracy
ax.axhline(xy_mean, color='#ff7f0e', linestyle='-', linewidth=2, label='Mean accuracy', zorder=2)
# Confidence interval
ax.axhspan(xy_ci_low, xy_ci_high, alpha=0.2, color='#ff7f0e', zorder=1)
# Chance level
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Chance', zorder=2)

ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.0)
ax.set_xlim(-0.15, 0.15)
ax.set_xticks([])
ax.set_title('2D gaze (x, y)', fontsize=10)
ax.spines['bottom'].set_visible(False)
ax.legend(loc='lower right', frameon=False, fontsize=8)

# Panel B: displacement features
ax = axes[1]
# Plot fold accuracies as strip plot (with jitter)
x_jitter = np.random.default_rng(seed=42).normal(0, 0.02, size=len(disp_acc))
ax.scatter(x_jitter, disp_acc, alpha=0.6, s=40, color='#1f77b4', zorder=3, label='Fold accuracies')
# Mean accuracy
ax.axhline(disp_mean, color='#ff7f0e', linestyle='-', linewidth=2, label='Mean accuracy', zorder=2)
# Confidence interval
ax.axhspan(disp_ci_low, disp_ci_high, alpha=0.2, color='#ff7f0e', zorder=1)
# Chance level
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Chance', zorder=2)

ax.set_ylim(0, 1.0)
ax.set_xlim(-0.15, 0.15)
ax.set_xticks([])
ax.set_title('Displacement', fontsize=10)
ax.spines['bottom'].set_visible(False)
ax.legend(loc='lower right', frameon=False, fontsize=8)

plt.tight_layout()

# Save individual axes and assembled panel
save_axes_svgs(fig, FIGURES_DIR, prefix='eyetracking_decoding')
save_axes_pdfs(fig, FIGURES_DIR, prefix='eyetracking_decoding')

# Save assembled panel
(FIGURES_DIR / 'panels').mkdir(exist_ok=True)
save_panel_svg(fig, FIGURES_DIR / 'panels' / 'eyetracking_decoding_panel.svg')
save_panel_pdf(fig, FIGURES_DIR / 'panels' / 'eyetracking_decoding_panel.pdf')

logger.info(f"Saved figures to {FIGURES_DIR}")
log_script_end(logger)

plt.show()
