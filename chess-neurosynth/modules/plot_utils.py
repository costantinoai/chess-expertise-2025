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


def _save_plotly(fig, title: str, output_file: Path | str):
    """
    Save Plotly figure to HTML and PDF (requires kaleido); silently skip on error.

    Notes
    -----
    - Always writes a sidecar HTML for interactive inspection
    - Primary static export is PDF to match publication requirements
    - If PDF export fails (e.g., kaleido missing), we do not raise
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # HTML
    html_path = output_file.with_suffix('.html')
    try:
        fig.write_html(str(html_path))
    except Exception:
        pass
    # Optional static export via kaleido (PDF) — can be disabled via env
    skip_static = os.environ.get('SKIP_PLOTLY_STATIC_EXPORT', '').strip().lower() in {'1', 'true', 'yes'}
    if not skip_static:
        try:
            pdf_path = output_file.with_suffix('.pdf')
            fig.write_image(str(pdf_path))
        except Exception:
            # Fallback: ignore if kaleido not installed or fails
            pass


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


def plot_surface_map_flat(img, title: str, threshold: float | None, output_file: Path | str):
    """
    Publication-style flat surface plot (left and right hemispheres).

    - Samples volume→surface on pial meshes (fsaverage, high-res)
    - Renders on fsaverage flat meshes for left and right hemispheres
    - Central title + per-hemisphere subtitles
    - Left hemisphere: rotated 90° clockwise around y-axis (lateral view)
    - Right hemisphere: flat surface, rotated 90° counterclockwise around y-axis
    """
    # Fetch matching-resolution meshes for sampling and flat rendering
    fsavg_fetch = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    # Use flat meshes from fetch_surf_fsaverage to ensure both hemispheres are flat
    flat_left = getattr(fsavg_fetch, 'flat_left', None)
    flat_right = getattr(fsavg_fetch, 'flat_right', None)

    # Sample volume to pial
    tex_l = surface.vol_to_surf(img, fsavg_fetch.pial_left)
    tex_r = surface.vol_to_surf(img, fsavg_fetch.pial_right)
    vmax = float(np.nanmax(np.abs(np.concatenate([tex_l, tex_r])))) if tex_l.size and tex_r.size else float(np.nanmax(np.abs(tex_l)))
    vmin = -vmax

    # Two-panel plot (Left, Right)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.05,
    )

    # Plot left hemisphere using helper
    _plot_hemisphere_flat(
        fig, flat_left, fsavg_fetch.pial_left, tex_l,
        hemi='left', vmin=vmin, vmax=vmax, threshold=threshold,
        row=1, col=1, show_colorbar=False
    )

    # Plot right hemisphere using helper (with colorbar)
    tmin, tmax = _plot_hemisphere_flat(
        fig, flat_right, fsavg_fetch.pial_right, tex_r,
        hemi='right', vmin=vmin, vmax=vmax, threshold=threshold,
        row=1, col=2, show_colorbar=False
    )

    # Camera angles - ORIGINAL:
    # Both hemispheres: dorsal view from above with 90° CW in-plane rotation
    cam_left_cw = dict(eye=dict(x=0.0, y=0.0, z=2.1), up=dict(x=0.0, y=1.0, z=.0))
    cam_right_cw = dict(eye=dict(x=-0.0, y=0.0, z=2.1), up=dict(x=0.0, y=1.0, z=0.0))

    fig.update_scenes(
        dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', camera=cam_left_cw),
        row=1, col=1
    )
    fig.update_scenes(
        dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', camera=cam_right_cw),
        row=1, col=2
    )

    # Update lighting on all mesh traces to eliminate harsh shadows
    # High ambient light with low specular creates a flat, evenly-lit appearance
    for trace in fig.data:
        if hasattr(trace, 'lighting'):
            trace.lighting = dict(
                ambient=0.2,      # High ambient light (reduces shadows)
                diffuse=.7,      # Moderate diffuse reflection
                specular=0.1,     # Low specular highlights
                roughness=1,    # High roughness (matte finish)
                fresnel=0.0       # No fresnel effect
            )
        if hasattr(trace, 'lightposition'):
            # Position light at camera location for consistent illumination
            trace.lightposition = dict(x=0, y=0, z=1000)

    # Layout with central title and hemisphere subtitles
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=int(PLOT_PARAMS['font_size_title']*1.3), family=PLOT_PARAMS['font_family'])),
        font=dict(size=int(PLOT_PARAMS['font_size_tick']*1.1), family=PLOT_PARAMS['font_family']),
        showlegend=False,
        margin=dict(t=60, l=0, r=0, b=0),
        width=1400,
        height=850,
        annotations=[
            dict(text='Left Hemisphere', x=0.25, y=0.98, xref='paper', yref='paper', showarrow=False,
                 font=dict(size=int(PLOT_PARAMS['font_size_title']), family=PLOT_PARAMS['font_family'])),
            dict(text='Right Hemisphere', x=0.75, y=0.98, xref='paper', yref='paper', showarrow=False,
                 font=dict(size=int(PLOT_PARAMS['font_size_title']), family=PLOT_PARAMS['font_family'])),
        ]
    )
    _save_plotly(fig, title, output_file)
