"""
Plotting helpers for neurosynth analyses (local, style from common.plotting_utils).
"""

from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting, surface, datasets
from nilearn import image
from plotly.subplots import make_subplots

from common.plotting import (
    apply_nature_rc,
    CMAP_BRAIN,
    PLOT_PARAMS,
    plot_grouped_bars_with_ci,
    plot_grouped_bars_on_ax,
    COLORS_EXPERT_NOVICE,
    plot_flat_pair,
    plot_flat_hemisphere,
)


def plot_map(arr, ref_img, title: str, outpath: Path | str, thresh: float = 1e-5):
    """
    Glass brain plot for a 3D array using CMAP_BRAIN and consistent style.
    """
    apply_nature_rc()
    img = image.new_img_like(ref_img, arr)
    disp = plotting.plot_glass_brain(
        img,
        display_mode='lyrz',
        colorbar=True,
        cmap=CMAP_BRAIN,
        symmetric_cbar=True,
        plot_abs=False,
        threshold=thresh,
    )
    disp.title(title, size=14, color='black', bgcolor='white', weight='bold')
    disp.savefig(str(outpath))
    plt.close('all')


from common.plotting.surfaces import _save_plotly  # centralized


def plot_surface_map(img, title: str, threshold: float | None, output_file: Path | str):
    """
    Plot a 2×2 grid (left/right × medial/lateral) using Plotly engine.
    """
    fsavg = datasets.fetch_surf_fsaverage()
    # Use pial (non-inflated) surfaces to emulate flat view
    # views = [('medial', 'left'), ('lateral', 'left'), ('medial', 'right'), ('lateral', 'right')]
    views = [('lateral', 'left'), ('lateral', 'right')]
    view_angles = {
        'lateral': dict(x=1.0, y=1.0, z=50.0),
        'medial': dict(x=-1.0, y=0.0, z=0.0),
        'dorsal': dict(x=0.0, y=0.0, z=1.0),
        'ventral': dict(x=0.0, y=0.0, z=-1.0),
        'posterior': dict(x=0.0, y=-1.0, z=0.0),
    }

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.02, vertical_spacing=0.02,
        subplot_titles=["", "", "", ""],
    )

    # Symmetric color range across views
    # Compute a representative texture once per hemi to define vmin/vmax
    tex_l = surface.vol_to_surf(img, fsavg.pial_left)
    tex_r = surface.vol_to_surf(img, fsavg.pial_right)
    vmax = float(np.nanmax(np.abs(np.concatenate([tex_l, tex_r]))))
    vmin = -vmax

    for i, (view, hemi) in enumerate(views, start=1):
        row = 1 if i <= 2 else 2
        col = i if i <= 2 else i - 2
        texture = tex_l if hemi == 'left' else tex_r
        sub = plotting.plot_surf_stat_map(
            fsavg.pial_left if hemi == 'left' else fsavg.pial_right,
            texture,
            hemi=hemi,
            view=view,
            bg_map=fsavg.sulc_left if hemi == 'left' else fsavg.sulc_right,
            colorbar=(i == 4),
            threshold=threshold,
            cmap=CMAP_BRAIN,
            engine="plotly",
            vmin=vmin,
            vmax=vmax,
            title=None,
        )
        sub_fig = sub.figure
        for tr in sub_fig.data:
            # Adjust colorbar on last subplot only
            if i == 4 and hasattr(tr, 'colorbar'):
                tr.colorbar.len = 0.8
                tr.colorbar.thickness = 16
                # Set ticks at [min, 0, max]
                tmin = float(np.nanmin(texture))
                tmax = float(np.nanmax(texture))
                tr.colorbar.tickvals = [tmin, 0.0, tmax]
                tr.colorbar.ticktext = [f"{tmin:.2f}", "0", f"{tmax:.2f}"]
                tr.colorbar.tickfont = dict(size=14, family=PLOT_PARAMS['font_family'])
            fig.add_trace(tr, row=row, col=col)
        fig.update_scenes(
            dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(eye=view_angles[view]),
                aspectmode='data',
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=int(PLOT_PARAMS['font_size_title']*1.1), family=PLOT_PARAMS['font_family'])),
        font=dict(size=int(PLOT_PARAMS['font_size_tick']*1.1), family=PLOT_PARAMS['font_family']),
        showlegend=False,
        margin=dict(t=40, l=0, r=0, b=0),
        width=1200,
        height=800,
    )
    _save_plotly(fig, title, output_file)


# =============================================================================
# On-axis helpers used by pylustrator scripts
# =============================================================================

def zero_cis(values):
    """Return zero-width CIs for a list of values, as (v, v) pairs."""
    return [(v, v) for v in values]


def plot_correlations_on_ax(ax, df_pos: 'pd.DataFrame', df_neg: 'pd.DataFrame', title: str):
    """
    Plot paired bars for POS vs NEG correlations onto an existing axis.

    Uses centralized grouped bar plotting and PLOT_PARAMS styling.
    """
    import numpy as np
    terms = [t.title() for t in df_pos['term']]
    r_pos = df_pos['r'].to_numpy().tolist()
    r_neg = df_neg['r'].to_numpy().tolist()
    cis_pos = zero_cis(r_pos)
    cis_neg = zero_cis(r_neg)

    green = COLORS_EXPERT_NOVICE.get('expert', '#198019')
    red = COLORS_EXPERT_NOVICE.get('novice', '#a90f0f')

    x = np.arange(len(terms))
    plot_grouped_bars_on_ax(
        ax=ax,
        x_positions=x,
        group1_values=r_pos,
        group1_cis=cis_pos,
        group1_color=green,
        group2_values=r_neg,
        group2_cis=cis_neg,
        group2_color=red,
        group1_label='POS',
        group2_label='NEG',
        ylim=None,
        params=PLOT_PARAMS,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=30, ha='right', fontsize=PLOT_PARAMS['font_size_tick'])
    ax.set_ylabel('Correlation (z)', fontsize=PLOT_PARAMS['font_size_label'])
    from common.plotting import set_axis_title
    set_axis_title(ax, title=title)


def plot_differences_on_ax(ax, df_diff: 'pd.DataFrame', title: str):
    """
    Plot Δr bars with sign-colored bars onto an existing axis.
    """
    import numpy as np
    terms = [t.title() for t in df_diff['term']]
    diffs = df_diff['r_diff'].to_numpy().tolist()
    cis = zero_cis(diffs)
    green = COLORS_EXPERT_NOVICE.get('expert', '#198019')
    red = COLORS_EXPERT_NOVICE.get('novice', '#a90f0f')
    colors = [green if (isinstance(v, (int, float)) and np.isfinite(v) and v >= 0) else red for v in diffs]

    x = np.arange(len(terms))
    plot_grouped_bars_on_ax(
        ax=ax,
        x_positions=x,
        group1_values=diffs,
        group1_cis=cis,
        group1_color=colors,
        params=PLOT_PARAMS,
        bar_width_multiplier=2.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=30, ha='right', fontsize=PLOT_PARAMS['font_size_tick'])
    ax.set_ylabel('ΔCorrelation (z)', fontsize=PLOT_PARAMS['font_size_label'])
    from common.plotting import set_axis_title
    set_axis_title(ax, title=title)


# Surface embedding helper (DRY) - scripts should call plot_flat_pair then embed_figure_on_ax
from common.plotting import embed_figure_on_ax


def plot_correlations(df_pos, df_neg, df_diff, run_id: str, out_fig: Path | str):
    """
    Paired bars for POS vs NEG correlations using common.plotting_utils.

    Reuses plot_grouped_bars_with_ci for consistent styling and DRY.
    """
    terms = [t.title() for t in df_pos['term']]
    r_pos = df_pos['r'].to_numpy().tolist()
    r_neg = df_neg['r'].to_numpy().tolist()

    cis_pos = list(zip(df_pos.get('CI_low', np.nan).to_numpy(), df_pos.get('CI_high', np.nan).to_numpy()))
    cis_neg = list(zip(df_neg.get('CI_low', np.nan).to_numpy(), df_neg.get('CI_high', np.nan).to_numpy()))

    # Colors and y-limits (legacy style: green=positive, red=negative; symmetric ylim)
    green = COLORS_EXPERT_NOVICE.get('expert', '#198019')
    red = COLORS_EXPERT_NOVICE.get('novice', '#a90f0f')
    # Fixed publication y-limits for correlations
    ylim = (-0.15, 0.30)

    # Plot with centralized parameters (no manual overrides)
    plot_grouped_bars_with_ci(
        group1_values=r_pos,
        group2_values=r_neg,
        group1_cis=cis_pos,
        group2_cis=cis_neg,
        x_labels=terms,
        # No significance stars for neurosynth plots
        group1_pvals=None,
        group2_pvals=None,
        group1_label=None,
        group2_label=None,
        group1_color=green,
        group2_color=red,
        ylabel='Correlation (z)',
        title=None,
        subtitle=str(run_id),
        ylim=ylim,
        add_zero_line=True,
        output_path=Path(out_fig),
    )


def plot_difference(df_diff, run_id: str, out_fig: Path | str):
    """
    Plot Δr = r_pos − r_neg with 95% CI and FDR significance stars using common plotting.
    """
    terms = [t.title() for t in df_diff['term']]
    diffs = df_diff['r_diff'].to_numpy().tolist()
    cis = list(zip(df_diff.get('CI_low', np.nan).to_numpy(), df_diff.get('CI_high', np.nan).to_numpy()))
    # No significance stars for neurosynth plots
    pvals = None

    # Per-bar colors by sign
    green = COLORS_EXPERT_NOVICE.get('expert', '#198019')
    red = COLORS_EXPERT_NOVICE.get('novice', '#a90f0f')
    bar_colors = [green if (isinstance(v, (int, float)) and np.isfinite(v) and v >= 0) else red for v in diffs]
    # Fixed publication y-limits for correlation differences
    ylim = (-0.20, 0.35)

    # Hide error bars in single-bar plot as well
    # Plot with centralized parameters (no manual overrides)
    # NOTE: figsize will be computed automatically by auto_bar_figure_size()
    plot_grouped_bars_with_ci(
        group1_values=diffs,
        group1_cis=cis,
        x_labels=terms,
        group2_values=None,
        group1_pvals=None,
        group1_label='',
        group2_label='',
        group1_color=bar_colors,
        ylabel='ΔCorrelation (z)',
        title=None,
        subtitle=None,
        ylim=ylim,
        add_zero_line=True,
        output_path=Path(out_fig),
        show_legend=True,
    )


def _plot_hemisphere_flat(
    fig, flat_mesh, pial_mesh, texture, hemi: str,
    vmin: float, vmax: float, threshold: float | None,
    row: int, col: int, show_colorbar: bool = False
) -> tuple:
    """
    Helper function to plot a single hemisphere on a flat surface.

    Parameters
    ----------
    fig : plotly figure
        The figure to add traces to
    flat_mesh : surface mesh
        The flat surface mesh (preferred)
    pial_mesh : surface mesh
        The pial surface mesh (fallback)
    texture : array
        The texture data to plot
    hemi : str
        Hemisphere ('left' or 'right')
    vmin, vmax : float
        Color scale limits
    threshold : float or None
        Threshold for displaying values
    row, col : int
        Subplot position
    show_colorbar : bool
        Whether to show colorbar

    Returns
    -------
    tmin, tmax : float
        Min and max texture values for colorbar
    """
    # Use flat mesh if available, otherwise fallback to pial
    mesh = flat_mesh if flat_mesh is not None else pial_mesh

    sub = plotting.plot_surf_stat_map(
        mesh,
        texture,
        hemi=hemi,
        view='dorsal',
        colorbar=show_colorbar,
        threshold=threshold,
        cmap=CMAP_BRAIN,
        engine='plotly',
        vmin=vmin if vmax > 0 else None,
        vmax=vmax if vmax > 0 else None,
        title=None,
    )

    # Extract min/max for colorbar
    tmin = float(np.nanmin(texture))
    tmax = float(np.nanmax(texture))

    # Add traces and style colorbar if present
    for tr in sub.figure.data:
        if show_colorbar and hasattr(tr, 'colorbar') and tr.colorbar is not None:
            tr.colorbar.len = 0.8
            tr.colorbar.thickness = 16
            tr.colorbar.tickvals = [tmin, 0.0, tmax]
            tr.colorbar.ticktext = [f"{tmin:.2f}", "0", f"{tmax:.2f}"]
            tr.colorbar.tickfont = dict(size=14, family=PLOT_PARAMS['font_family'])
        fig.add_trace(tr, row=row, col=col)

    return tmin, tmax


# Removed thin wrapper plot_surface_map_flat; use common.plotting.plot_flat_pair directly
