#!/usr/bin/env python3
"""
Generate RDM Intercorrelation Supplementary Figures (Pylustrator)
=================================================================
[docstring unchanged]
"""
import sys
from pathlib import Path

# Add parent (repo root) to sys.path for 'common' and 'modules'
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import CONFIG first to check pylustrator flag
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.plotting import (
    apply_nature_rc,
    save_axes_svgs,
    figure_size,
    save_panel_pdf,
)
from modules.plotting import (
    plot_correlation_bars,
    plot_variance_partition_bars,
)

# =============================================================================
# Configuration and Results Loading
# =============================================================================
script_dir = Path(__file__).parent
results_base = script_dir / "results"

results_dir = find_latest_results_directory(
    results_base,
    pattern="*_rdm_intercorrelation",
    create_subdirs=["figures"],
    require_exists=True,
    verbose=True,
)

figures_dir = results_dir / "figures"
panels_dir = figures_dir / "panels"
panels_dir.mkdir(parents=True, exist_ok=True)

extra = {"RESULTS_DIR": str(results_dir), "FIGURES_DIR": str(figures_dir)}
config, _, logger = setup_analysis_in_dir(
    results_dir=results_dir,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name="pylustrator_rdm_intercorr.log",
)

logger.info("=" * 80)
logger.info("PLOTTING RDM INTERCORRELATION ANALYSIS")
logger.info("=" * 80)

# =============================================================================
# Common Data Loading
# =============================================================================
pairwise_file = results_dir / "pairwise_correlations.tsv"
partial_file = results_dir / "partial_correlations.tsv"
var_part_file = results_dir / "variance_partitioning_all.tsv"

if not pairwise_file.exists():
    raise FileNotFoundError(f"Missing pairwise correlations: {pairwise_file}")
if not partial_file.exists():
    raise FileNotFoundError(f"Missing partial correlations: {partial_file}")
if not var_part_file.exists():
    raise FileNotFoundError(f"Missing variance partitioning: {var_part_file}")

pairwise_df = pd.read_csv(pairwise_file, sep="\t", index_col=0)
partial_df = pd.read_csv(partial_file, sep="\t")
var_part_df = pd.read_csv(var_part_file, sep="\t")

model_order = [m for m in CONFIG['MODEL_ORDER'] if m in pairwise_df.index]
model_order += [m for m in pairwise_df.index if m not in model_order]
pairwise_df = pairwise_df.loc[model_order, model_order]

partial_lookup = {
    (row.target, row.predictor): row.r_partial
    for row in partial_df.itertuples()
}

logger.info("Loaded analysis outputs successfully.")

apply_nature_rc()
colorblind_palette = sns.color_palette("colorblind")

model_colors = {
    'visual': colorblind_palette[0],     # Blue
    'strategy': colorblind_palette[1],   # Orange
    'check': colorblind_palette[2],      # Green
}
color_shared = colorblind_palette[3]         # Red
color_unexplained = colorblind_palette[4]    # Purple

logger.info("Colors assigned from seaborn colorblind palette:")
logger.info(f"  Visual: {model_colors['visual']}")
logger.info(f"  Strategy: {model_colors['strategy']}")
logger.info(f"  Checkmate: {model_colors['check']}")
logger.info(f"  Shared: {color_shared}")
logger.info(f"  Unexplained: {color_unexplained}")

# =============================================================================
# Single Figure (contains both panels)
# =============================================================================
# Combine sizes from the two original figures into one slightly taller canvas.
fig = plt.figure(1, figsize=figure_size(columns=2, height_mm=220))

# -----------------------------------------------------------------------------
# Panel 1: Pairwise vs Partial Correlations (axes labeled "Corr_*")
# -----------------------------------------------------------------------------
logger.info("Building Panel 1 axes (pairwise vs partial correlations)...")

for idx, target in enumerate(model_order, start=1):
    predictors = [m for m in model_order if m != target]
    if not predictors:
        continue
    ax = plt.axes()
    ax.set_label(f"Corr_{target}")
    plot_correlation_bars(
        ax=ax,
        target=target,
        predictors=predictors,
        pairwise_df=pairwise_df,
        partial_lookup=partial_lookup,
        model_colors=model_colors,
        ylabel=(idx == 1),
    )

# -----------------------------------------------------------------------------
# Panel 2: Variance Partitioning (axes labeled "VarPart_*")
# -----------------------------------------------------------------------------
logger.info("Building Panel 2 axes (variance partitioning)...")

for idx, target in enumerate(model_order, start=1):
    ax = plt.axes()
    ax.set_label(f"VarPart_{target}")
    plot_variance_partition_bars(
        ax=ax,
        target=target,
        var_part_df=var_part_df,
        model_colors=model_colors,
        color_shared=color_shared,
        color_unexplained=color_unexplained,
        ylabel=(idx == 1),
    )

# Setup ax_dict for pylustrator (one figure containing both sets of axes)
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}

# =============================================================================
# Show for interactive editing
# =============================================================================
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(8.880000/2.54, 9.290000/2.54, forward=True)
plt.figure(1).ax_dict["Corr_check"].set(position=[0.7127, 0.596, 0.2641, 0.2949], yticks=[], yticklabels=[])
plt.figure(1).ax_dict["Corr_check"].texts[0].set(position=(-0.1689, -0.1981))
plt.figure(1).ax_dict["Corr_check"].texts[1].set(position=(0.8311, 0.3672))
plt.figure(1).ax_dict["Corr_check"].texts[2].set(position=(0.1689, -0.1896))
plt.figure(1).ax_dict["Corr_check"].texts[3].set(position=(1.169, 0.3635))
plt.figure(1).ax_dict["Corr_strategy"].set(position=[0.427, 0.596, 0.2641, 0.2949], yticks=[], yticklabels=[])
plt.figure(1).ax_dict["Corr_strategy"].texts[1].set(position=(0.865, 0.3672))
plt.figure(1).ax_dict["Corr_strategy"].texts[3].set(position=(1.135, 0.3635))
plt.figure(1).ax_dict["Corr_strategy"].text(-0.1447, 1.2331, 'RDMs Inter-correlation', transform=plt.figure(1).ax_dict["Corr_strategy"].transAxes, fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["Corr_strategy"].texts[4].new
plt.figure(1).ax_dict["Corr_visual"].set(position=[0.1345, 0.596, 0.2641, 0.2949], yticks=[-0.8, -0.4, 0., 0.4, 0.8], yticklabels=['-0.80', '-0.40', '0.00', '0.40', '0.80'])
plt.figure(1).ax_dict["Corr_visual"].texts[1].set(position=(0.8446, -0.1896))
plt.figure(1).ax_dict["Corr_visual"].texts[3].set(position=(1.149, -0.1896))
plt.figure(1).ax_dict["Corr_visual"].text(-0.1044, 1.2331, 'a', transform=plt.figure(1).ax_dict["Corr_visual"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["Corr_visual"].texts[4].new
plt.figure(1).ax_dict["VarPart_check"].set(position=[0.7127, 0.1209, 0.2641, 0.2949], yticks=[], yticklabels=[])
plt.figure(1).ax_dict["VarPart_strategy"].set(position=[0.427, 0.1209, 0.2641, 0.2949], yticks=[], yticklabels=[])
plt.figure(1).ax_dict["VarPart_strategy"].text(-0.1820, 1.2331, 'RDM Variance Explained', transform=plt.figure(1).ax_dict["VarPart_strategy"].transAxes, fontsize=7., weight='bold')  # id=plt.figure(1).ax_dict["VarPart_strategy"].texts[4].new
plt.figure(1).ax_dict["VarPart_visual"].set(position=[0.1345, 0.1209, 0.2641, 0.2949], yticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], yticklabels=['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'])
plt.figure(1).ax_dict["VarPart_visual"].text(-0.1044, 1.2331, 'b', transform=plt.figure(1).ax_dict["VarPart_visual"].transAxes, fontsize=8., weight='bold')  # id=plt.figure(1).ax_dict["VarPart_visual"].texts[4].new
#% end: automatic generated code from pylustrator
plt.show()

# =============================================================================
# Save (combined figure)
# =============================================================================
# Save all axes SVGs and a single combined PDF (filenames updated accordingly)
save_axes_svgs(fig, figures_dir, "rdm_intercorr_combined")
save_panel_pdf(fig, panels_dir / "rdm_intercorr_combined.pdf")

logger.info("✓ Combined figure (pairwise/partial + variance partitioning) complete")

log_script_end(logger)
