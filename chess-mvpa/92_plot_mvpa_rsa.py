"""
Pylustrator-driven MVPA panels — RSA only.

Figure: ROI RSA correlations (Expert vs Novice) for three targets
  - Visual Similarity, Strategy, Checkmate

Arrange interactively in pylustrator and save to inject layout code.

Usage:
    python 92_plot_mvpa_rsa.py
"""

import sys
import os
import pickle
from pathlib import Path

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = Path(__file__).parent

# Import CONFIG first to check pylustrator flag
from common import CONFIG

# Conditionally start pylustrator BEFORE creating any figures
if CONFIG['ENABLE_PYLUSTRATOR']:
    import pylustrator
    pylustrator.start()

import numpy as np
import matplotlib.pyplot as plt
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata
from common.plotting import (
    apply_nature_rc,
    plot_grouped_bars_on_ax,
    set_axis_title,
    compute_ylim_range,
    format_roi_labels_and_colors,
    style_spines,
    PLOT_PARAMS,
    save_axes_svgs,
    save_panel_svg,
    save_axes_pdfs,
    save_panel_pdf,
    embed_figure_on_ax,
    plot_pial_views_triplet,
    create_standalone_colorbar,
    CMAP_BRAIN,
    create_roi_group_legend,
)
from nibabel.freesurfer import io as fsio
import pandas as pd
from modules.mvpa_plot_utils import extract_mvpa_bar_data


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR_NAME = None
RESULTS_BASE = script_dir / "results"

MAIN_TARGETS = ['visual_similarity', 'strategy', 'checkmate']
RSA_TITLES = {
    'visual_similarity': 'Visual Similarity RSA',
    'strategy': 'Strategy RSA',
    'checkmate': 'Checkmate RSA',
}


# =============================================================================
# Load results
# =============================================================================

RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern="*_mvpa_group_rsa",
    specific_name=RESULTS_DIR_NAME,
    create_subdirs=["figures"],
    require_exists=True,
    verbose=True,
)

FIGURES_DIR = RESULTS_DIR / "figures"

extra = {"RESULTS_DIR": str(RESULTS_DIR), "FIGURES_DIR": str(FIGURES_DIR)}
config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name="pylustrator_mvpa_rsa.log",
)

logger.info("Loading MVPA RSA group statistics...")
with open(RESULTS_DIR / "mvpa_group_stats.pkl", "rb") as f:
    group_stats = pickle.load(f)

roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])
apply_nature_rc()

# Figure: RSA correlations (3 axes, no layout)
rsa_data = extract_mvpa_bar_data(group_stats, roi_info, MAIN_TARGETS, method='rsa_corr', subtract_chance=False)

all_vals = []
for d in rsa_data.values():
    all_vals.extend(d['exp_means'])
    all_vals.extend(d['nov_means'])
ylim_rsa = compute_ylim_range(all_vals, padding_pct=0.15)

fig1 = plt.figure(1)

for idx, tgt in enumerate(MAIN_TARGETS):
    if tgt not in rsa_data:
        continue
    data = rsa_data[tgt]
    roi_names = data['roi_names']
    roi_colors = data['roi_colors']
    label_colors = data['label_colors']
    x = np.arange(len(roi_names))

    ax = plt.axes()
    ax.set_label(f'RSA_{idx+1}_{tgt}')

    plot_grouped_bars_on_ax(
        ax=ax,
        x_positions=x,
        group1_values=data['exp_means'],
        group1_cis=data['exp_cis'],
        group1_color=roi_colors,
        group2_values=data['nov_means'],
        group2_cis=data['nov_cis'],
        group2_color=roi_colors,
        group1_label='Experts',
        group2_label='Novices',
        comparison_pvals=data['pvals'],
        ylim=ylim_rsa,
        params=PLOT_PARAMS
    )

    ax.set_ylabel(PLOT_PARAMS['ylabel_correlation_r'], fontsize=PLOT_PARAMS['font_size_label'])
    ax.tick_params(axis='y', labelsize=PLOT_PARAMS['font_size_tick'])
    set_axis_title(ax, title=RSA_TITLES[tgt])

    if idx == 0:
        ax.legend(frameon=False, loc='upper right', ncol=1, fontsize=PLOT_PARAMS['font_size_legend'])

    style_spines(ax, visible_spines=['left', 'bottom'], params=PLOT_PARAMS)
    ax.set_xlim(-0.5, len(roi_names) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, rotation=30, ha='right', fontsize=PLOT_PARAMS['font_size_tick'])
    for ticklabel, color in zip(ax.get_xticklabels(), label_colors):
        ticklabel.set_color(color)

fig1.ax_dict = {ax.get_label(): ax for ax in fig1.axes}

# =============================================================================
# Add 3×1 left pial surface axes (FDR-significant ROIs only)
# =============================================================================

# Build per-vertex left-hemisphere textures: Δr (Experts−Novices) where pFDR<alpha; others NaN
alpha_fdr = float(CONFIG.get('ALPHA_FDR', 0.05))

# Load surface annotations and Glasser-180 metadata (must contain a 'region22_id' column for mapping)
labels_l, _, _ = fsio.read_annot(CONFIG['ROI_GLASSER_180_ANNOT_L'])

roi180_info_path = CONFIG['ROI_GLASSER_180'] / 'region_info.tsv'
roi180_df = pd.read_csv(roi180_info_path, sep='\t')
# Determine id column name (roi_id preferred, accept 'index' or 'ROI_idx')
if 'roi_id' in roi180_df.columns:
    roi_id_col = 'roi_id'
elif 'index' in roi180_df.columns:
    roi_id_col = 'index'
elif 'ROI_idx' in roi180_df.columns:
    roi_id_col = 'ROI_idx'
else:
    raise RuntimeError(f"Expected an ROI id column ('roi_id'|'index'|'ROI_idx') in {roi180_info_path}")
if 'region22_id' not in roi180_df.columns:
    raise RuntimeError(
        "Missing 'region22_id' column in Glasser-180 region_info.tsv.\n"
        "Please add an integer column 'region22_id' mapping each left-hemisphere MMP parcel (roi_id 1–180)\n"
        "to one of the 22 bilateral regions, per the Glasser paper grouping."
    )

# Use only left hemisphere rows (1..180) to match fsaverage left annot
roi180_left = roi180_df[roi180_df[roi_id_col] <= 180].copy()

# Mapping from 180 parcel id -> 22-region id
map_180_to_22 = dict(zip(roi180_left[roi_id_col].astype(int), roi180_left['region22_id'].astype(int)))

left_textures_by_target = {}
delta_values_by_target = {}
for tgt in MAIN_TARGETS:
    if 'rsa_corr' not in group_stats or tgt not in group_stats['rsa_corr']:
        raise RuntimeError(f"Missing RSA group stats for target '{tgt}'")
    welch = group_stats['rsa_corr'][tgt]['welch_expert_vs_novice']
    if not {'ROI_Label', 'mean_diff', 'p_val_fdr'}.issubset(set(welch.columns)):
        raise RuntimeError(f"Welch dataframe missing required columns for target '{tgt}'")

    roi22_ids = welch['ROI_Label'].to_numpy()
    delta_r = welch['mean_diff'].to_numpy()
    p_fdr = welch['p_val_fdr'].to_numpy()
    include_mask = (p_fdr < alpha_fdr) & np.isfinite(delta_r)

    # Map 22-region deltas (sig-only) to 180 parcels (left hemi)
    delta_map = {int(rid): (float(val) if keep else np.nan) for rid, val, keep in zip(roi22_ids.astype(int), delta_r, include_mask)}

    # Expand to per-180 values
    roi_ids_180 = []
    roi_vals_180 = []
    for rid180, rid22 in map_180_to_22.items():
        roi_ids_180.append(int(rid180))
        roi_vals_180.append(delta_map.get(int(rid22), np.nan))
    roi_ids_180 = np.asarray(roi_ids_180, dtype=int)
    roi_vals_180 = np.asarray(roi_vals_180, dtype=float)

    # Build per-vertex texture for left hemisphere
    from common.neuro_utils import roi_values_to_surface_texture
    tex_left = roi_values_to_surface_texture(
        labels=labels_l,
        roi_labels=roi_ids_180,
        roi_values=roi_vals_180,
        include_mask=None,
        default_value=np.nan,
    )
    left_textures_by_target[tgt] = tex_left
    delta_values_by_target[tgt] = roi_vals_180

# Shared color scale across the three targets: vmin=0, vmax=max positive Δr (rounded to 1 decimal)
all_positive = []
for tgt, vals in delta_values_by_target.items():
    if vals is None:
        continue
    vals = np.asarray(vals)
    if vals.size:
        pos = vals[np.isfinite(vals) & (vals > 0)]
        if pos.size:
            all_positive.append(np.nanmax(pos))

if not all_positive:
    raise RuntimeError("No positive significant Δr values found across targets; cannot set color scale with vmin=0.")

vmax_raw = float(np.max(all_positive))
vmax_surf = round(vmax_raw, 1)
if vmax_surf <= 0:
    raise RuntimeError(f"Computed vmax <= 0 after rounding: raw={vmax_raw:.4f}, rounded={vmax_surf:.1f}")
vmin_surf = 0.0

# Create and embed one left-lateral pial surface per target
for idx, tgt in enumerate(MAIN_TARGETS):
    ax = plt.axes()
    ax.set_label(f'RSA_Pial_{idx+1}_{tgt}')
    fig_pial = plot_pial_views_triplet(
        data={'left': left_textures_by_target[tgt]},
        hemi='left',
        views=('lateral', 'medial'),
        title='',
        threshold=None,
        vmin=vmin_surf,
        vmax=vmax_surf,
    )
    embed_figure_on_ax(ax, fig_pial, title=f"{RSA_TITLES[tgt]} — Left Pial (FDR sig)")

# Add a standalone horizontal colorbar for Δr across all pial panels
ax_cbar = plt.axes()
ax_cbar.set_label('RSA_Pial_Colorbar')
cbar_fig = create_standalone_colorbar(
    cmap=CMAP_BRAIN,
    vmin=vmin_surf,
    vmax=vmax_surf,
    orientation='horizontal',
    label='Δr (Experts − Novices)'
)
embed_figure_on_ax(ax_cbar, cbar_fig, title='')

# Add a 1-row legend (ROI groups) for pylustrator layout
ax_legend = plt.axes()
ax_legend.set_label('RSA_Pial_Legend')
legend_fig = create_roi_group_legend(
    roi_metadata_path=CONFIG['ROI_GLASSER_22'] / 'region_info.tsv',
    output_path=None,
    single_row=True,
    colorblind=False,
)
embed_figure_on_ax(ax_legend, legend_fig, title='')

# Update axis dictionary (for pylustrator)
fig1.ax_dict = {ax.get_label(): ax for ax in fig1.axes}

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).ax_dict["RSA_1_visual_similarity"].set(position=[0.05595, 0.7798, 0.4479, 0.1829])
plt.figure(1).ax_dict["RSA_2_strategy"].set(position=[0.05595, 0.4436, 0.4479, 0.1829])
plt.figure(1).ax_dict["RSA_3_checkmate"].set(position=[0.05595, 0.1074, 0.4479, 0.1829])
#% end: automatic generated code from pylustrator
plt.show()

save_axes_svgs(fig1, FIGURES_DIR, 'mvpa_rsa')
save_axes_pdfs(fig1, FIGURES_DIR, 'mvpa_rsa')
save_panel_svg(fig1, FIGURES_DIR / 'panels' / 'mvpa_rsa_panel.svg')
save_panel_pdf(fig1, FIGURES_DIR / 'panels' / 'mvpa_rsa_panel.pdf')

log_script_end(logger)
