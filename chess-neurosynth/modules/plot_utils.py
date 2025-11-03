"""
Plotting helpers for neurosynth analyses (local, style from common.plotting_utils).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    from common.plotting import compute_ylim_range
    vmin, vmax = compute_ylim_range(tex_l, tex_r, symmetric=True, padding_pct=0.0)

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

def load_term_corr_triple(results_dir: Path, stem: str):
    """
    Load the trio of term-correlation CSV files (positive, negative, difference).

    Parameters
    ----------
    results_dir : Path
        Directory containing exported correlation CSVs
    stem : str
        Filename stem, e.g., 'spmT_exp-gt-nonexp_all-gt-rest'

    Returns
    -------
    (df_pos, df_neg, df_diff) : tuple of pandas.DataFrame
        DataFrames for positive, negative, and difference correlations
    """
    pos = pd.read_csv(results_dir / f"{stem}_term_corr_positive.csv")
    neg = pd.read_csv(results_dir / f"{stem}_term_corr_negative.csv")
    diff = pd.read_csv(results_dir / f"{stem}_term_corr_difference.csv")
    return pos, neg, diff

def zero_cis(values):
    """Return zero-width CIs for a list of values, as (v, v) pairs."""
    return [(v, v) for v in values]


def plot_correlations_on_ax(
    ax,
    df_pos: 'pd.DataFrame',
    df_neg: 'pd.DataFrame',
    title: str,
    subtitle: str = None,
    ylim=None
):
    """
    Plot paired bars for POS vs NEG correlations onto an existing axis.

    Uses centralized grouped bar plotting and PLOT_PARAMS styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    df_pos : pd.DataFrame
        DataFrame with positive correlations (columns: 'term', 'r')
    df_neg : pd.DataFrame
        DataFrame with negative correlations (columns: 'term', 'r')
    title : str
        Main plot title (bold)
    subtitle : str, optional
        Plot subtitle (normal weight), e.g., contrast name or context
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). If None, uses automatic scaling.
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
        group1_label='Positive z-map',
        group2_label='Negative z-map',
        ylim=ylim,
        # DRY formatting in helper
        y_label='Correlation (z)',
        title=title,
        subtitle=subtitle,
        xtick_labels=terms,
        x_tick_rotation=30,
        x_tick_align='right',
        visible_spines=['left','bottom'],
        show_legend=True,
        legend_loc='upper right',
        params=PLOT_PARAMS,
    )


def plot_differences_on_ax(
    ax,
    df_diff: 'pd.DataFrame',
    title: str,
    subtitle: str = None,
    ylim=None
):
    """
    Plot Δr bars with sign-colored bars onto an existing axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    df_diff : pd.DataFrame
        DataFrame with correlation differences (columns: 'term', 'r_diff')
    title : str
        Main plot title (bold)
    subtitle : str, optional
        Plot subtitle (normal weight), e.g., contrast description
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). If None, uses automatic scaling.
    """
    import numpy as np
    from matplotlib.patches import Patch

    terms = [t.title() for t in df_diff['term']]
    diffs = df_diff['r_diff'].to_numpy().tolist()
    # cis = zero_cis(diffs)
    green = COLORS_EXPERT_NOVICE.get('expert', '#198019')
    red = COLORS_EXPERT_NOVICE.get('novice', '#a90f0f')
    colors = [green if (isinstance(v, (int, float)) and np.isfinite(v) and v >= 0) else red for v in diffs]

    # legend wants exactly one entry per unique color:
    labels_for_colors = ['Exp > Nov', 'Nov > Exp']  # must match unique colors order (first appearance)


    x = np.arange(len(terms))
    plot_grouped_bars_on_ax(
        ax=ax,
        x_positions=x,
        group1_values=diffs,
        # group1_cis=cis,
        group1_color=colors,
        group1_label=labels_for_colors,
        params=PLOT_PARAMS,
        bar_width_multiplier=2.0,
        ylim=ylim,
        # DRY formatting in helper
        y_label='ΔCorrelation (z)',
        title=title,
        subtitle=subtitle,
        xtick_labels=terms,
        x_tick_rotation=30,
        x_tick_align='right',
        visible_spines=['left','bottom'],
    )
