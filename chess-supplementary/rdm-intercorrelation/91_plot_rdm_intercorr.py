#!/usr/bin/env python3
"""
Pylustrator-ready figures for the RDM intercorrelation supplementary analysis.

Methods
-------
Colors are assigned from the seaborn colorblind palette to ensure consistent
visual coding across all panels:
- Visual Similarity: colorblind[0]
- Strategy: colorblind[1]
- Checkmate: colorblind[2]
- Shared variance: colorblind[3]
- Unexplained variance: colorblind[4]

Partial correlations are displayed using lightened versions of the base model
colors (via `common.plotting.lighten_color` with amount=0.45).

Correlation panels show pairwise vs partial correlations as grouped bars with
x-axis labels identifying predictors. Variance partitioning panels show stacked
bars ordered by MODEL_ORDER (Visual, Strategy, Check, Shared, Unexplained).
All bars display numeric values on top.

After arranging axes in pylustrator and saving, rerun to inject layout code and
export individual axes as SVGs in `figures/` and assembled panels in `figures/panels/`.
"""

import sys
from pathlib import Path

# Enable imports from repo root
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
    save_panel_svg,
    figure_size,
    save_axes_pdfs,
    save_panel_pdf,
)
from modules.plotting import (
    plot_correlation_bars,
    plot_variance_partition_bars,
)


# =============================================================================
# Configuration and results directory
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
# Load data
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

# Determine model order from CONFIG
model_order = [m for m in CONFIG['MODEL_ORDER'] if m in pairwise_df.index]
model_order += [m for m in pairwise_df.index if m not in model_order]
pairwise_df = pairwise_df.loc[model_order, model_order]

partial_lookup = {
    (row.target, row.predictor): row.r_partial
    for row in partial_df.itertuples()
}

logger.info("Loaded analysis outputs successfully.")

# =============================================================================
# Colors: Seaborn colorblind palette for consistent model coding
# =============================================================================

apply_nature_rc()

# Seaborn colorblind palette
colorblind_palette = sns.color_palette("colorblind")

# Assign colors based on MODEL_ORDER for consistency across all figures
model_colors = {
    'visual': colorblind_palette[0],
    'strategy': colorblind_palette[1],
    'check': colorblind_palette[2],
}

# Special colors for variance partitioning components
color_shared = colorblind_palette[3]
color_unexplained = colorblind_palette[4]

logger.info("Colors assigned from seaborn colorblind palette:")
logger.info(f"  Visual: {model_colors['visual']}")
logger.info(f"  Strategy: {model_colors['strategy']}")
logger.info(f"  Checkmate: {model_colors['check']}")
logger.info(f"  Shared: {color_shared}")
logger.info(f"  Unexplained: {color_unexplained}")

# =============================================================================
# Figure 1: Pairwise vs Partial correlations
# =============================================================================

logger.info("Building Figure 1 axes (pairwise vs partial correlations)...")
fig1 = plt.figure(1, figsize=figure_size(columns=2, height_mm=120))

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

# =============================================================================
# Figure 2: Variance partitioning stacked bars
# =============================================================================

logger.info("Building Figure 2 axes (variance partitioning)...")
fig2 = plt.figure(2, figsize=figure_size(columns=2, height_mm=110))

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

# =============================================================================
# Pylustrator layout placeholder (to be populated after interactive arrangement)
# =============================================================================

#% start: automatic generated code from pylustrator
# (Arrange axes in pylustrator, then save to inject layout code here.)
#% end: automatic generated code from pylustrator

#% start: automatic generated code from pylustrator
plt.figure(2).ax_dict = {ax.get_label(): ax for ax in plt.figure(2).axes}
import matplotlib as mpl
getattr(plt.figure(2), '_pylustrator_init', lambda: ...)()
plt.figure(2).ax_dict["VarPart_check"].set(position=[0.7161, 0.678, 0.2644, 0.2617])
plt.figure(2).ax_dict["VarPart_strategy"].set(position=[0.3878, 0.678, 0.2644, 0.2617])
plt.figure(2).ax_dict["VarPart_strategy"].spines[['right', 'top']].set_visible(False)
plt.figure(2).ax_dict["VarPart_visual"].set(position=[0.05954, 0.678, 0.2644, 0.2617])
plt.figure(2).ax_dict["VarPart_visual"].spines[['right', 'top']].set_visible(False)
#% end: automatic generated code from pylustrator
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).ax_dict["Corr_check"].set(position=[0.5432, 0.6896, 0.1483, 0.2678])
plt.figure(1).ax_dict["Corr_strategy"].set(position=[0.3144, 0.6896, 0.1483, 0.2678])
plt.figure(1).ax_dict["Corr_visual"].set(position=[0.08563, 0.6896, 0.1483, 0.2678])
#% end: automatic generated code from pylustrator
plt.show()

# =============================================================================
# Export axes and panels
# =============================================================================

logger.info("Saving individual axes and assembled panels...")

fig1 = plt.figure(1)
save_axes_svgs(fig1, figures_dir, "rdm_intercorr_fig1")
save_panel_svg(fig1, panels_dir / "rdm_intercorr_fig1.svg")
save_axes_pdfs(fig1, figures_dir, "rdm_intercorr_fig1")
save_panel_pdf(fig1, panels_dir / "rdm_intercorr_fig1.pdf")

fig2 = plt.figure(2)
save_axes_svgs(fig2, figures_dir, "rdm_intercorr_fig2")
save_panel_svg(fig2, panels_dir / "rdm_intercorr_fig2.svg")
save_axes_pdfs(fig2, figures_dir, "rdm_intercorr_fig2")
save_panel_pdf(fig2, panels_dir / "rdm_intercorr_fig2.pdf")

# =============================================================================
# Done
# =============================================================================

logger.info("=" * 80)
logger.info("RDM intercorrelation plotting complete.")
logger.info(f"Figures saved to {figures_dir}")
logger.info(f"Panels saved to {panels_dir}")
logger.info("=" * 80)

log_script_end(logger)
