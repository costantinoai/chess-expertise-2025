#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surface plotting primitives (flat hemispheres) for Nature-compliant figures.

Provides:
- plot_flat_pair(): Left/Right flat surfaces using Plotly engine
- plot_flat_hemisphere(): Single flat hemisphere using Plotly engine

Notes
-----
- Uses nilearn's fsaverage meshes and vol_to_surf sampling
- Saves interactive HTML and optional static PDF (via kaleido) when output_file is provided
- Styling (fonts) follows PLOT_PARAMS
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable, Tuple
import logging

import numpy as np
from nilearn import plotting, surface, datasets

from plotly.subplots import make_subplots

from .colors import CMAP_BRAIN
from .style import PLOT_PARAMS
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _mpl_to_plotly_scale(cmap, n: int = 256) -> List[Tuple[float, str]]:
    """
    Convert a Matplotlib colormap to a Plotly colorscale definition.

    Returns list of (position, 'rgb(r,g,b)') with position in [0,1].
    """
    import matplotlib as mpl
    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)
    scale = []
    for i in range(n):
        x = i / (n - 1)
        r, g, b, _ = cmap(x)
        scale.append((x, f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"))
    return scale


def _save_plotly(fig, title: str, output_file: Path | str | None):
    """
    Save Plotly figure to HTML and optionally PDF (requires kaleido).

    - Writes an interactive HTML when `output_file` is provided (raises on write errors)
    - Attempts a PDF export using kaleido unless disabled via env var
      `SKIP_PLOTLY_STATIC_EXPORT`.
    """
    if output_file is None:
        return
    import os
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # HTML (fail fast on error)
    html_path = output_file.with_suffix('.html')
    fig.write_html(str(html_path))
    # Optional static export via kaleido (PDF-only; no PNG)
    skip_static = os.environ.get('SKIP_PLOTLY_STATIC_EXPORT', '').strip().lower() in {'1', 'true', 'yes'}
    if not skip_static:
        pdf_path = output_file.with_suffix('.pdf')
        fig.write_image(str(pdf_path))


def _flat_meshes():
    """Fetch fsaverage meshes; return (flat_left, flat_right, pial_left, pial_right).

    Notes
    -----
    If flat meshes are unavailable in the local nilearn dataset, falls back to
    pial meshes and logs this event (no silent fallback).
    """
    fsavg = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    flat_left = getattr(fsavg, 'flat_left', None)
    flat_right = getattr(fsavg, 'flat_right', None)
    if flat_left is None or flat_right is None:
        logger.debug("Flat meshes not found in fsaverage; falling back to pial meshes")
    return flat_left, flat_right, fsavg.pial_left, fsavg.pial_right


def compute_surface_symmetric_range(imgs: Iterable) -> Tuple[float, float]:
    """
    Compute symmetric vmin/vmax for a collection of volume images using
    surface projections onto fsaverage pial meshes.

    Falls back to volume absolute max if surface projection fails.

    Parameters
    ----------
    imgs : iterable of NIfTI-like images

    Returns
    -------
    (vmin, vmax) : tuple of floats
        Symmetric range (-absmax, absmax)
    """
    try:
        fsavg = datasets.fetch_surf_fsaverage()
        pial_left, pial_right = fsavg.pial_left, fsavg.pial_right
        vmax = 0.0
        for zimg in imgs:
            try:
                texl = surface.vol_to_surf(zimg, pial_left)
                texr = surface.vol_to_surf(zimg, pial_right)
                vmax = max(vmax, float(np.nanmax(np.abs(np.concatenate([texl, texr])))))
            except Exception:
                arr = zimg.get_fdata()
                vmax = max(vmax, float(np.nanmax(np.abs(arr))))
        return (-vmax, vmax)
    except Exception:
        vmax = 0.0
        for zimg in imgs:
            arr = zimg.get_fdata()
            vmax = max(vmax, float(np.nanmax(np.abs(arr))))
        return (-vmax, vmax)


def _plot_hemisphere_flat(fig, mesh_flat, mesh_pial, texture, hemi: str,
                          vmin: float, vmax: float,
                          threshold: Optional[float], row: int, col: int,
                          show_colorbar: bool = False,
                          colorbar_horizontal: bool = False):
    """Internal helper: add a single flat hemisphere to a plotly subplot."""
    mesh = mesh_flat if mesh_flat is not None else mesh_pial
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

    tmin = float(np.nanmin(texture))
    tmax = float(np.nanmax(texture))

    for tr in sub.figure.data:
        if show_colorbar and hasattr(tr, 'colorbar') and tr.colorbar is not None:
            tr.colorbar.thickness = 16
            tr.colorbar.tickvals = [tmin, 0.0, tmax]
            tr.colorbar.ticktext = [f"{tmin:.2f}", "0", f"{tmax:.2f}"]
            tr.colorbar.tickfont = dict(size=14, family=PLOT_PARAMS['font_family'])
            if colorbar_horizontal:
                tr.colorbar.orientation = 'h'
                tr.colorbar.x = 0.5
                tr.colorbar.y = -0.08
                tr.colorbar.xanchor = 'center'
                tr.colorbar.len = 0.6
        fig.add_trace(tr, row=row, col=col)

    return tmin, tmax


def plot_flat_pair(
    data,
    title: str,
    threshold: float | None = None,
    output_file: Path | str | None = None,
    show_hemi_labels: bool = False,
    show_colorbar: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    show_directions: bool = True,
):
    """
    Publication-style flat surface plot (left and right hemispheres), Plotly engine.

    Creates a flatmap visualization with minimal white space and optional anatomical
    direction labels (A/P/D/V).

    Parameters
    ----------
    img : 3D Nifti-like image or array-like
        Statistical map in volume space
    title : str
        Figure title
    threshold : float or None
        Value threshold; values below are masked
    output_file : Path or str, optional
        If provided, saves HTML and attempts a PDF via kaleido
    show_hemi_labels : bool
        If True, show "Left hemi" / "Right hemi" labels
    show_colorbar : bool
        If True, show colorbar
    vmin, vmax : float or None
        Color scale limits
    show_directions : bool
        If True, add anatomical direction labels (A/P/D/V) and hemisphere labels
    """
    flat_left, flat_right, pial_left, pial_right = _flat_meshes()

    # Determine textures from input (NIfTI volume, dict, or tuple/list)
    tex_l: np.ndarray
    tex_r: np.ndarray
    if isinstance(data, dict) and 'left' in data and 'right' in data:
        tex_l = np.asarray(data['left'])
        tex_r = np.asarray(data['right'])
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        tex_l = np.asarray(data[0])
        tex_r = np.asarray(data[1])
    else:
        # Assume NIfTI-like image
        img = data
        tex_l = surface.vol_to_surf(img, pial_left)
        tex_r = surface.vol_to_surf(img, pial_right)

    # Symmetric color scale if not provided
    if vmin is None or vmax is None:
        vmax_local = float(np.nanmax(np.abs(np.concatenate([tex_l, tex_r]))))
        vmin_local = -vmax_local
    else:
        vmax_local = float(vmax)
        vmin_local = float(vmin)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.01,  # Reduced spacing between hemispheres
    )

    _plot_hemisphere_flat(fig, flat_left, pial_left, tex_l,
                          hemi='left', vmin=vmin_local, vmax=vmax_local, threshold=threshold,
                          row=1, col=1, show_colorbar=False)

    # Show a single horizontal colorbar on the right hemisphere traces (shared scale)
    _plot_hemisphere_flat(fig, flat_right, pial_right, tex_r,
                          hemi='right', vmin=vmin_local, vmax=vmax_local, threshold=threshold,
                          row=1, col=2, show_colorbar=show_colorbar, colorbar_horizontal=True)

    cam_left = dict(eye=dict(x=0.0, y=0.0, z=2.1), up=dict(x=0.0, y=1.0, z=0.0))
    cam_right = dict(eye=dict(x=-0.0, y=0.0, z=2.1), up=dict(x=0.0, y=1.0, z=0.0))

    fig.update_scenes(
        dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', camera=cam_left),
        row=1, col=1
    )
    fig.update_scenes(
        dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', camera=cam_right),
        row=1, col=2
    )

    for trace in fig.data:
        if hasattr(trace, 'lighting'):
            trace.lighting = dict(ambient=0.2, diffuse=.7, specular=0.1, roughness=1, fresnel=0.0)
        if hasattr(trace, 'lightposition'):
            trace.lightposition = dict(x=0, y=0, z=1000)

    # Build annotations list
    annotations = []

    # Add hemisphere labels if requested
    if show_hemi_labels:
        annotations.extend([
            dict(text='Left hemi', x=0.25, y=0.98, xref='paper', yref='paper', showarrow=False,
                 font=dict(size=int(PLOT_PARAMS['font_size_title']), family=PLOT_PARAMS['font_family'])),
            dict(text='Right hemi', x=0.75, y=0.98, xref='paper', yref='paper', showarrow=False,
                 font=dict(size=int(PLOT_PARAMS['font_size_title']), family=PLOT_PARAMS['font_family'])),
        ])

    # Add anatomical direction labels if requested
    if show_directions:
        # Font size for direction labels (large for visibility)
        dir_font_size = 40
        dir_font = dict(size=dir_font_size, family=PLOT_PARAMS['font_family'],
                       color='lightgray')

        # Vertical center of the plot area (where P should go)
        vcenter = 0.5

        # Horizontal positions
        # Left A: at left edge
        left_a_x = 0.01
        # Right A: at right edge
        right_a_x = 0.99
        # P: centered between the two hemispheres
        p_x = 0.5

        # Add the labels
        annotations.extend([
            # P - center (Posterior)
            dict(text='P', x=p_x, y=vcenter, xref='paper', yref='paper',
                 showarrow=False, font=dir_font),

            # A - left (Anterior)
            dict(text='A', x=left_a_x, y=vcenter, xref='paper', yref='paper',
                 showarrow=False, font=dir_font),

            # A - right (Anterior)
            dict(text='A', x=right_a_x, y=vcenter, xref='paper', yref='paper',
                 showarrow=False, font=dir_font),

            # D - dorsal (above P)
            dict(text='D', x=p_x, y=vcenter + 0.35, xref='paper', yref='paper',
                 showarrow=False, font=dir_font),

            # V - ventral (below P)
            dict(text='V', x=p_x, y=vcenter - 0.35, xref='paper', yref='paper',
                 showarrow=False, font=dir_font),

            # Left hemi
            dict(text='Left Hemisphere', x=p_x-0.25, y=vcenter + 0.45, xref='paper', yref='paper',
                 showarrow=False, font=dir_font),

            # Right hemi
            dict(text='Right Hemisphere', x=p_x+0.25, y=vcenter + 0.45, xref='paper', yref='paper',
                 showarrow=False, font=dir_font),
        ])

    layout_kwargs = dict(
        showlegend=False,
        # Minimize margins to reduce white space
        margin=dict(t=0, l=10, r=10, b=10),
        width=1600,
        height=850,
    )

    if annotations:
        layout_kwargs["annotations"] = annotations

    fig.update_layout(**layout_kwargs)

    _save_plotly(fig, title, output_file)
    return fig


def plot_flat_hemisphere(
    img,
    hemi: str,
    title: Optional[str] = None,
    threshold: float | None = None,
    output_file: Path | str | None = None,
    show_hemi_label: bool = False,
    show_colorbar: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Single-hemisphere flat surface plot using Plotly engine.
    """
    flat_left, flat_right, pial_left, pial_right = _flat_meshes()
    if hemi not in {'left', 'right'}:
        raise ValueError("hemi must be 'left' or 'right'")

    pial = pial_left if hemi == 'left' else pial_right
    flat = flat_left if hemi == 'left' else flat_right

    tex = surface.vol_to_surf(img, pial)
    if vmin is None or vmax is None:
        vmax_local = float(np.nanmax(np.abs(tex)))
        vmin_local = -vmax_local
    else:
        vmax_local = float(vmax)
        vmin_local = float(vmin)

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    _plot_hemisphere_flat(fig, flat, pial, tex, hemi=hemi, vmin=vmin_local, vmax=vmax_local,
                          threshold=threshold, row=1, col=1, show_colorbar=show_colorbar, colorbar_horizontal=True)

    cam = dict(eye=dict(x=0.0, y=0.0, z=2.1), up=dict(x=0.0, y=1.0, z=0.0))
    fig.update_scenes(
        dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', camera=cam),
        row=1, col=1
    )
    for trace in fig.data:
        if hasattr(trace, 'lighting'):
            trace.lighting = dict(ambient=0.2, diffuse=.7, specular=0.1, roughness=1, fresnel=0.0)
        if hasattr(trace, 'lightposition'):
            trace.lightposition = dict(x=0, y=0, z=1000)

    layout_kwargs = {}
    if title:
        layout_kwargs["title"] = dict(text=title, x=0.5, font=dict(size=int(PLOT_PARAMS['font_size_title']*1.3), family=PLOT_PARAMS['font_family']))
    if show_hemi_label:
        layout_kwargs["annotations"] = [
            dict(text=f"{hemi.title()} hemi", x=0.5, y=0.98, xref='paper', yref='paper', showarrow=False,
                 font=dict(size=int(PLOT_PARAMS['font_size_title']), family=PLOT_PARAMS['font_family']))
        ]
    # No global coloraxis; colorbar handled by hemisphere trace
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    _save_plotly(fig, title or 'flat_surface', output_file)
    return fig


def embed_figure_on_ax(ax, fig, title: str = ''):
    """
    Embed any figure (matplotlib or Plotly) into a matplotlib axis for pylustrator layouts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to embed the figure into
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure to embed
    title : str, optional
        Title to add above the embedded figure

    Notes
    -----
    - Renders figure to in-memory PNG buffer
    - Embeds the rasterized image in the provided axis
    - Closes matplotlib figures after rendering
    - No disk writes
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO
    from .helpers import set_axis_title

    buf = BytesIO()

    # Detect figure type and render appropriately
    if hasattr(fig, 'savefig'):
        # Matplotlib figure
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif hasattr(fig, 'write_image'):
        # Plotly figure
        from plotly.io import to_image
        img_bytes = to_image(fig, format='png', scale=2)
        buf.write(img_bytes)
    else:
        raise TypeError(f"Unsupported figure type: {type(fig)}")

    buf.seek(0)

    # Embed in provided axis
    ax.set_axis_off()
    ax.imshow(mpimg.imread(buf))
    if title:
        set_axis_title(ax, title=title)
