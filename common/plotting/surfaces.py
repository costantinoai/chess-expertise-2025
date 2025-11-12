#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surface plotting primitives (flat hemispheres) for Nature-compliant figures.

Provides:
- plot_flat_pair(): Left/Right flat surfaces using Plotly engine
- plot_flat_hemisphere(): Single flat hemisphere using Plotly engine
- plot_pial_hemisphere(): Single pial hemisphere (e.g., lateral view) using Plotly engine

Notes
-----
- Uses nilearn's fsaverage meshes and vol_to_surf sampling
- Saves interactive HTML and optional static PDF (via kaleido) when output_file is provided
- Styling (fonts) follows PLOT_PARAMS
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import logging

import numpy as np
from nilearn import plotting, surface, datasets

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .colors import CMAP_BRAIN
from .style import PLOT_PARAMS
import matplotlib.pyplot as plt
from nilearn.datasets import load_fsaverage_data

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
    html_path = output_file.with_suffix(".html")
    fig.write_html(str(html_path))
    # Optional static export via kaleido (PDF-only; no PNG)
    skip_static = os.environ.get("SKIP_PLOTLY_STATIC_EXPORT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not skip_static:
        pdf_path = output_file.with_suffix(".pdf")
        fig.write_image(str(pdf_path))
    fig.show()

def _flat_meshes():
    """Fetch fsaverage meshes; return (flat_left, flat_right, pial_left, pial_right).

    Notes
    -----
    If flat meshes are unavailable in the local nilearn dataset, falls back to
    pial meshes and logs this event (no silent fallback).
    """
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage7")
    flat_left = getattr(fsavg, "flat_left", None)
    flat_right = getattr(fsavg, "flat_right", None)
    if flat_left is None or flat_right is None:
        logger.debug("Flat meshes not found in fsaverage; falling back to pial meshes")
    return flat_left, flat_right, fsavg.pial_left, fsavg.pial_right


def _pial_meshes():
    """Fetch fsaverage pial meshes; return (pial_left, pial_right).

    Raises on failure — no silent fallbacks.
    """
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    return fsavg.pial_left, fsavg.pial_right


def _extract_roi_boundary_edges(
    mesh,
    roi_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D edge coordinates where ROI boundaries occur on a surface mesh.

    Parameters
    ----------
    mesh : tuple, object, or str/Path
        Surface mesh from nilearn (either tuple of (coords, faces), object
        with .coordinates and .faces attributes, or path to mesh file).
    roi_labels : np.ndarray
        Per-vertex integer labels (same length as vertices). Vertices with
        value > 0 are considered part of an ROI. Boundaries are drawn where
        adjacent vertices have different non-zero labels.

    Returns
    -------
    x, y, z : np.ndarray
        1D arrays of edge coordinates with None separators for discontinuous
        line segments (Plotly convention). Format: [x1, x2, None, x3, x4, None, ...]
    """
    # Extract coordinates and faces from mesh
    if isinstance(mesh, tuple):
        coords, faces = mesh
    elif isinstance(mesh, (str, Path)):
        # Mesh is a file path - load it using nilearn
        coords, faces = surface.load_surf_mesh(str(mesh))
    else:
        coords = mesh.coordinates if hasattr(mesh, 'coordinates') else mesh.coords
        faces = mesh.faces

    coords = np.asarray(coords)
    faces = np.asarray(faces)
    roi_labels = np.asarray(roi_labels).astype(int)

    # Collect boundary edges (avoid duplicates using set)
    edge_coords = []
    seen_edges = set()

    for face in faces:
        v1, v2, v3 = face
        # Check all three edges of the triangle
        edges = [(v1, v2), (v2, v3), (v3, v1)]

        for va, vb in edges:
            # Create canonical edge key (sorted to handle bidirectional)
            edge_key = tuple(sorted([va, vb]))
            if edge_key in seen_edges:
                continue

            label_a = roi_labels[va]
            label_b = roi_labels[vb]

            # Draw boundary if labels are different AND at least one is an ROI (>0)
            # This captures:
            # 1. ROI outline vs background (one is >0, other is 0)
            # 2. Boundary between different ROIs (both >0 but different)
            if label_a != label_b and (label_a > 0 or label_b > 0):
                edge_coords.append((coords[va], coords[vb]))
                seen_edges.add(edge_key)

    # Convert to Plotly format: separate x, y, z arrays with None separators
    if not edge_coords:
        # No boundaries found, return empty arrays
        return np.array([]), np.array([]), np.array([])

    x_coords = []
    y_coords = []
    z_coords = []

    for pt1, pt2 in edge_coords:
        x_coords.extend([pt1[0], pt2[0], None])
        y_coords.extend([pt1[1], pt2[1], None])
        z_coords.extend([pt1[2], pt2[2], None])

    return np.array(x_coords), np.array(y_coords), np.array(z_coords)


def _compute_roi_centroids(
    mesh,
    roi_labels: np.ndarray,
) -> dict:
    """
    Compute centroid coordinates for each ROI on a surface mesh.

    Uses a robust approach that ensures the label point is actually inside the ROI
    by finding the vertex closest to the geometric centroid that has the correct label.

    Parameters
    ----------
    mesh : tuple, object, or str/Path
        Surface mesh from nilearn.
    roi_labels : np.ndarray
        Per-vertex integer labels. Centroids computed for each unique value > 0.

    Returns
    -------
    dict
        Mapping of roi_id -> (x, y, z) centroid coordinates guaranteed to be inside ROI.
    """
    # Extract coordinates from mesh
    if isinstance(mesh, tuple):
        coords, _ = mesh
    elif isinstance(mesh, (str, Path)):
        coords, _ = surface.load_surf_mesh(str(mesh))
    else:
        coords = mesh.coordinates if hasattr(mesh, 'coordinates') else mesh.coords

    coords = np.asarray(coords)
    roi_labels = np.asarray(roi_labels).astype(int)

    centroids = {}
    unique_rois = np.unique(roi_labels)

    for roi_id in unique_rois:
        if roi_id == 0:  # Skip background
            continue
        # Get all vertices belonging to this ROI
        mask = roi_labels == roi_id
        roi_coords = coords[mask]

        # Compute geometric centroid
        geometric_centroid = roi_coords.mean(axis=0)

        # Find the vertex in this ROI that is closest to the geometric centroid
        # This guarantees the label point is actually inside the ROI
        distances = np.linalg.norm(roi_coords - geometric_centroid, axis=1)
        closest_idx = np.argmin(distances)
        best_centroid = roi_coords[closest_idx]

        centroids[roi_id] = tuple(best_centroid)

    return centroids


def _plot_hemisphere_flat(
    fig,
    mesh_flat,
    mesh_pial,
    texture,
    hemi: str,
    vmin: float,
    vmax: float,
    threshold: Optional[float],
    row: int,
    col: int,
    show_colorbar: bool = False,
    colorbar_horizontal: bool = False,
    contour_labels: Optional[np.ndarray] = None,
    contour_color: str = "black",
    contour_width: float = 2.0,
    roi_text_labels: Optional[dict] = None,
):
    """Internal helper: add a single flat hemisphere to a plotly subplot.

    Parameters
    ----------
    roi_text_labels : dict, optional
        Mapping of roi_id -> text label to display at ROI centroid.
        Only used if contour_labels is also provided.
    """
    mesh = mesh_flat if mesh_flat is not None else mesh_pial
    sub = plotting.plot_surf_stat_map(
        mesh,
        texture,
        hemi=hemi,
        view="dorsal",
        colorbar=show_colorbar,
        threshold=threshold,
        cmap=CMAP_BRAIN,
        engine="plotly",
        vmin=vmin if vmax > 0 else None,
        vmax=vmax if vmax > 0 else None,
        title=None,
    )

    tmin = float(np.nanmin(texture))
    tmax = float(np.nanmax(texture))

    for tr in sub.figure.data:
        if show_colorbar and hasattr(tr, "colorbar") and tr.colorbar is not None:
            tr.colorbar.thickness = 16
            tr.colorbar.tickvals = [tmin, 0.0, tmax]
            tr.colorbar.ticktext = [f"{tmin:.2f}", "0", f"{tmax:.2f}"]
            tr.colorbar.tickfont = dict(size=14, family=PLOT_PARAMS["font_family"])
            if colorbar_horizontal:
                tr.colorbar.orientation = "h"
                tr.colorbar.x = 0.5
                tr.colorbar.y = -0.08
                tr.colorbar.xanchor = "center"
                tr.colorbar.len = 0.6
        fig.add_trace(tr, row=row, col=col)

    # Add contour lines if requested
    if contour_labels is not None:
        x, y, z = _extract_roi_boundary_edges(mesh, contour_labels)
        if len(x) > 0:  # Only add trace if boundaries were found
            # Offset z slightly to ensure contours render above surface (skip None values)
            z_offset = []
            for val in z:
                if val is None:
                    z_offset.append(None)
                else:
                    z_offset.append(val + 5.0)
            contour_trace = go.Scatter3d(
                x=x,
                y=y,
                z=z_offset,
                mode='lines',
                line=dict(color=contour_color, width=contour_width),
                showlegend=False,
                hoverinfo='skip',
            )
            fig.add_trace(contour_trace, row=row, col=col)

        # Add text labels at ROI centroids if provided
        if roi_text_labels is not None and len(roi_text_labels) > 0:
            centroids = _compute_roi_centroids(mesh, contour_labels)
            for roi_id, label_text in roi_text_labels.items():
                if roi_id in centroids:
                    cx, cy, cz = centroids[roi_id]
                    # Offset z-coordinate to ensure text is above surface (prevents occlusion)
                    cz_offset = cz + 10.0
                    text_trace = go.Scatter3d(
                        x=[cx],
                        y=[cy],
                        z=[cz_offset],
                        mode='text',
                        text=[label_text],
                        textfont=dict(
                            size=32,
                            color='black',
                            family=PLOT_PARAMS["font_family"],
                        ),
                        textposition='middle center',
                        showlegend=False,
                        hoverinfo='skip',
                    )
                    fig.add_trace(text_trace, row=row, col=col)

    return tmin, tmax


def plot_flat_pair(
    textures: Tuple[np.ndarray, np.ndarray],
    title: str = "",
    threshold: float | None = None,
    output_file: Path | str | None = None,
    show_hemi_labels: bool = False,
    show_colorbar: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    show_directions: bool = True,
    roi_contours: dict | None = None,
):
    """
    Publication-style flat surface plot (left and right hemispheres), Plotly engine.

    Plots pre-computed surface textures. Callers must project volumes to surfaces
    before calling this function (see common.neuro_utils.project_volume_to_surfaces).

    Parameters
    ----------
    textures : tuple of (tex_left, tex_right)
        Pre-computed surface textures for left and right hemispheres.
        Each should be a 1D numpy array of per-vertex values.
    title : str, default=''
        Figure title (used for file metadata and when embedding via embed_figure_on_ax)
    subtitle : str, optional
        Figure subtitle (used when embedding via embed_figure_on_ax)
    threshold : float or None
        Value threshold; values below are masked
    output_file : Path or str, optional
        If provided, saves HTML and attempts a PDF via kaleido
    show_hemi_labels : bool, default=False
        If True, show "Left hemi" / "Right hemi" labels
    show_colorbar : bool, default=False
        If True, show colorbar
    vmin, vmax : float, required
        Color scale limits. Must be provided by caller.
        Use common.plotting.compute_ylim_range to compute from textures.
    show_directions : bool, default=True
        If True, add anatomical direction labels (A/P/D/V) and hemisphere labels
    roi_contours : dict, optional
        Dictionary specifying pre-computed ROI contours to draw and label. Set to None
        (default) to disable contours. Required keys:
        - 'contours_left': np.ndarray - Per-vertex integer labels for left hemisphere
        - 'contours_right': np.ndarray - Per-vertex integer labels for right hemisphere
        Optional keys:
        - 'labels': dict[int, str] - Mapping of roi_id -> text label to display at centroids
        - 'color': str, default='black' - Contour line color
        - 'width': float, default=2.0 - Contour line width in pixels

    Raises
    ------
    ValueError
        If vmin or vmax not provided, or if textures is invalid.

    Examples
    --------
    >>> # Basic usage: Project volume to surfaces and plot
    >>> from common.neuro_utils import project_volume_to_surfaces
    >>> from common.plotting import compute_ylim_range, plot_flat_pair
    >>>
    >>> tex_l, tex_r = project_volume_to_surfaces(nifti_volume)
    >>> vmin, vmax = compute_ylim_range(tex_l, tex_r, symmetric=True, padding_pct=0.0)
    >>> fig = plot_flat_pair(
    ...     textures=(tex_l, tex_r),
    ...     title='Brain Activation',
    ...     vmin=vmin,
    ...     vmax=vmax,
    ...     show_colorbar=True
    ... )

    >>> # With ROI contours: Generate contours and labels separately
    >>> from common.neuro_utils import create_glasser22_contours
    >>>
    >>> # Generate contours for specific Glasser-22 regions
    >>> contours_l, contours_r = create_glasser22_contours(['dLPFC', 'PCC', 'V1', 'TPOJ'])
    >>>
    >>> # Plot with contours and labels
    >>> fig = plot_flat_pair(
    ...     textures=(tex_l, tex_r),
    ...     roi_contours={
    ...         'contours_left': contours_l,
    ...         'contours_right': contours_r,
    ...         'labels': {1: 'V1', 15: 'TPOJ', 18: 'PCC', 22: 'dLPFC'},
    ...         'color': 'black',
    ...         'width': 2.0
    ...     },
    ...     vmin=vmin,
    ...     vmax=vmax,
    ...     show_colorbar=True
    ... )

    See Also
    --------
    common.neuro_utils.project_volume_to_surfaces : Project NIfTI volumes to surfaces
    common.neuro_utils.create_glasser22_contours : Generate contours for Glasser-22 regions
    common.plotting.compute_ylim_range : Compute symmetric or anchored ranges
    """
    # Validate inputs
    if not isinstance(textures, (tuple, list)) or len(textures) != 2:
        raise ValueError("textures must be a tuple of (tex_left, tex_right)")
    if vmin is None or vmax is None:
        raise ValueError(
            "vmin and vmax are required. Use compute_ylim_range to compute from textures."
        )

    flat_left, flat_right, pial_left, pial_right = _flat_meshes()

    # Extract textures
    tex_l = np.asarray(textures[0])
    tex_r = np.asarray(textures[1])

    # Parse roi_contours dictionary if provided
    contours_l = None
    contours_r = None
    roi_text_labels = None
    contour_color = 'black'
    contour_width = 2.0

    if roi_contours is not None:
        if not isinstance(roi_contours, dict):
            raise ValueError("roi_contours must be a dictionary")

        # Required keys
        if 'contours_left' not in roi_contours or 'contours_right' not in roi_contours:
            raise ValueError("roi_contours must contain 'contours_left' and 'contours_right' keys")

        contours_l = np.asarray(roi_contours['contours_left'])
        contours_r = np.asarray(roi_contours['contours_right'])

        # Validate shapes
        if contours_l.shape[0] != tex_l.shape[0]:
            raise ValueError(
                f"contours_left length ({contours_l.shape[0]}) must match tex_left length ({tex_l.shape[0]})"
            )
        if contours_r.shape[0] != tex_r.shape[0]:
            raise ValueError(
                f"contours_right length ({contours_r.shape[0]}) must match tex_right length ({tex_r.shape[0]})"
            )

        # Optional keys
        roi_text_labels = roi_contours.get('labels', None)
        contour_color = roi_contours.get('color', 'black')
        contour_width = roi_contours.get('width', 2.0)

    vmin_local = float(vmin)
    vmax_local = float(vmax)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.01,  # Reduced spacing between hemispheres
    )

    _plot_hemisphere_flat(
        fig,
        flat_left,
        pial_left,
        tex_l,
        hemi="left",
        vmin=vmin_local,
        vmax=vmax_local,
        threshold=threshold,
        row=1,
        col=1,
        show_colorbar=False,
        contour_labels=contours_l,
        contour_color=contour_color,
        contour_width=contour_width,
        roi_text_labels=roi_text_labels,
    )

    # Show a single horizontal colorbar on the right hemisphere traces (shared scale)
    _plot_hemisphere_flat(
        fig,
        flat_right,
        pial_right,
        tex_r,
        hemi="right",
        vmin=vmin_local,
        vmax=vmax_local,
        threshold=threshold,
        row=1,
        col=2,
        show_colorbar=show_colorbar,
        colorbar_horizontal=True,
        contour_labels=contours_r,
        contour_color=contour_color,
        contour_width=contour_width,
        roi_text_labels=roi_text_labels,
    )

    cam_left = dict(eye=dict(x=0.0, y=0.0, z=2.1), up=dict(x=0.0, y=1.0, z=0.0))
    cam_right = dict(eye=dict(x=-0.0, y=0.0, z=2.1), up=dict(x=0.0, y=1.0, z=0.0))

    fig.update_scenes(
        dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=cam_left,
        ),
        row=1,
        col=1,
    )
    fig.update_scenes(
        dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=cam_right,
        ),
        row=1,
        col=2,
    )

    for trace in fig.data:
        if hasattr(trace, "lighting"):
            trace.lighting = dict(
                ambient=0.2, diffuse=0.7, specular=0.1, roughness=1, fresnel=0.0
            )
        if hasattr(trace, "lightposition"):
            trace.lightposition = dict(x=0, y=0, z=1000)

    # Build annotations list
    annotations = []

    # Add hemisphere labels if requested
    if show_hemi_labels:
        annotations.extend(
            [
                dict(
                    text="Left hemi",
                    x=0.25,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(
                        size=int(PLOT_PARAMS["font_size_title"]),
                        family=PLOT_PARAMS["font_family"],
                    ),
                ),
                dict(
                    text="Right hemi",
                    x=0.75,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(
                        size=int(PLOT_PARAMS["font_size_title"]),
                        family=PLOT_PARAMS["font_family"],
                    ),
                ),
            ]
        )

    # Add anatomical direction labels if requested
    if show_directions:
        # Font size for direction labels (increased for better visibility on flat projections)
        dir_font_size = 50  # Was 40, increased for better readability
        dir_font = dict(
            size=dir_font_size,
            family=PLOT_PARAMS["font_family"],
            color="darkgray"  # Was "lightgray", darker for better contrast
        )

        # Vertical center of the plot area (where P should go)
        vcenter = 0.5

        # Horizontal positions
        # Left A: at left edge (moved further left for better positioning)
        left_a_x = 0.005  # Was 0.01
        # Right A: at right edge (moved further right for better positioning)
        right_a_x = 0.995  # Was 0.99
        # P: centered between the two hemispheres
        p_x = 0.5

        # Add the labels
        annotations.extend(
            [
                # P - center (Posterior)
                dict(
                    text="P",
                    x=p_x,
                    y=vcenter,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dir_font,
                ),
                # A - left (Anterior)
                dict(
                    text="A",
                    x=left_a_x,
                    y=vcenter,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dir_font,
                ),
                # A - right (Anterior)
                dict(
                    text="A",
                    x=right_a_x,
                    y=vcenter,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dir_font,
                ),
                # D - dorsal (above P)
                dict(
                    text="D",
                    x=p_x,
                    y=vcenter + 0.35,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dir_font,
                ),
                # V - ventral (below P)
                dict(
                    text="V",
                    x=p_x,
                    y=vcenter - 0.35,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dir_font,
                ),
                # Left hemi
                dict(
                    text="Left Hemisphere",
                    x=p_x - 0.25,
                    y=vcenter + 0.45,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dir_font,
                ),
                # Right hemi
                dict(
                    text="Right Hemisphere",
                    x=p_x + 0.25,
                    y=vcenter + 0.45,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dir_font,
                ),
            ]
        )

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
    if hemi not in {"left", "right"}:
        raise ValueError("hemi must be 'left' or 'right'")

    pial = pial_left if hemi == "left" else pial_right
    flat = flat_left if hemi == "left" else flat_right

    tex = surface.vol_to_surf(img, pial)
    if vmin is None or vmax is None:
        vmax_local = float(np.nanmax(np.abs(tex)))
        vmin_local = -vmax_local
    else:
        vmax_local = float(vmax)
        vmin_local = float(vmin)

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    _plot_hemisphere_flat(
        fig,
        flat,
        pial,
        tex,
        hemi=hemi,
        vmin=vmin_local,
        vmax=vmax_local,
        threshold=threshold,
        row=1,
        col=1,
        show_colorbar=show_colorbar,
        colorbar_horizontal=True,
    )

    cam = dict(eye=dict(x=0.0, y=0.0, z=2.1), up=dict(x=0.0, y=1.0, z=0.0))
    fig.update_scenes(
        dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=cam,
        ),
        row=1,
        col=1,
    )
    for trace in fig.data:
        if hasattr(trace, "lighting"):
            trace.lighting = dict(
                ambient=0.2, diffuse=0.7, specular=0.1, roughness=1, fresnel=0.0
            )
        if hasattr(trace, "lightposition"):
            trace.lightposition = dict(x=0, y=0, z=1000)

    layout_kwargs = {}
    if title:
        layout_kwargs["title"] = dict(
            text=title,
            x=0.5,
            font=dict(
                size=int(PLOT_PARAMS["font_size_title"] * 1.3),
                family=PLOT_PARAMS["font_family"],
            ),
        )
    if show_hemi_label:
        layout_kwargs["annotations"] = [
            dict(
                text=f"{hemi.title()} hemi",
                x=0.5,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(
                    size=int(PLOT_PARAMS["font_size_title"]),
                    family=PLOT_PARAMS["font_family"],
                ),
            )
        ]
    # No global coloraxis; colorbar handled by hemisphere trace
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    _save_plotly(fig, title or "flat_surface", output_file)
    return fig


def embed_figure_on_ax(ax, fig, title: str = "", subtitle: str = None):
    """
    Embed any figure (matplotlib or Plotly) into a matplotlib axis for pylustrator layouts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to embed the figure into
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure to embed
    title : str, optional
        Main title to add above the embedded figure (bold)
    subtitle : str, optional
        Subtitle to add below the title (normal weight)

    Notes
    -----
    - Renders figure to in-memory PNG buffer
    - Embeds the rasterized image in the provided axis
    - Closes matplotlib figures after rendering
    - No disk writes
    - Uses centralized set_axis_title for consistent styling
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO
    from .helpers import set_axis_title

    buf = BytesIO()

    # Detect figure type and render appropriately
    if hasattr(fig, "savefig"):
        # Matplotlib figure - DRY: use centralized DPI
        fig.savefig(buf, format="png", dpi=PLOT_PARAMS['dpi'], bbox_inches="tight")
        plt.close(fig)
    elif hasattr(fig, "write_image"):
        # Plotly figure
        from plotly.io import to_image

        img_bytes = to_image(fig, format="png", scale=2)
        buf.write(img_bytes)
    else:
        raise TypeError(f"Unsupported figure type: {type(fig)}")

    buf.seek(0)

    # Embed in provided axis
    ax.set_axis_off()

    # Check for zorder metadata from source figure (e.g., ROI legends)
    if hasattr(fig, "_roi_legend_zorder"):
        ax.set_zorder(fig._roi_legend_zorder)  # type: ignore[attr-defined]

    ax.imshow(mpimg.imread(buf))
    if title or subtitle:
        set_axis_title(ax, title=title, subtitle=subtitle)


def plot_pial_hemisphere(
    data,
    hemi: str = "left",
    view: str = "lateral",
    *,
    title: str | None = None,
    threshold: float | None = None,
    show_colorbar: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Single-hemisphere pial surface plot using Plotly engine.

    Parameters
    ----------
    data : NIfTI-like image or array-like
        Either a 3D NIfTI image (volume) to sample onto the pial surface or a
        1D per-vertex texture for the requested hemisphere. If an array is
        provided, its length must match the fsaverage pial vertex count.
        Alternatively, accepts a dict with key 'left' or 'right'.
    hemi : {'left','right'}
        Hemisphere to display (default: 'left').
    view : str
        Surface view (e.g., 'lateral', 'medial', 'ventral', 'dorsal').
    title : str, optional
        Title for the figure (unused when embedding; kept for completeness).
    threshold : float or None
        Values below threshold are masked.
    show_colorbar : bool
        Whether to display a colorbar for this hemisphere.
    vmin, vmax : float or None
        Color scale limits. If None, computed from the texture range (symmetric).

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with a single pial hemisphere scene.
    """
    if hemi not in {"left", "right"}:
        raise ValueError("hemi must be 'left' or 'right'")

    pial_left, pial_right = _pial_meshes()
    mesh = pial_left if hemi == "left" else pial_right

    # Determine texture based on input type
    tex = None
    if isinstance(data, dict) and hemi in data:
        tex = np.asarray(data[hemi])
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        tex = np.asarray(data[0] if hemi == "left" else data[1])
    else:
        # Assume NIfTI-like image; sample to surface
        tex = surface.vol_to_surf(data, mesh)

    # Symmetric color scale if not provided
    if vmin is None or vmax is None:
        vmax_local = float(np.nanmax(np.abs(tex))) if np.any(np.isfinite(tex)) else 1.0
        vmin_local = -vmax_local
    else:
        vmax_local = float(vmax)
        vmin_local = float(vmin)

    # Create single scene with surf stat map
    sub = plotting.plot_surf_stat_map(
        mesh,
        tex,
        hemi=hemi,
        view=view,
        colorbar=show_colorbar,
        threshold=threshold,
        cmap=CMAP_BRAIN,
        engine="plotly",
        vmin=vmin_local if vmax_local > 0 else None,
        vmax=vmax_local if vmax_local > 0 else None,
        title=None,
    )

    fig = sub.figure

    # Style colorbar if present
    for tr in fig.data:
        if show_colorbar and hasattr(tr, "colorbar") and tr.colorbar is not None:
            tr.colorbar.thickness = 16
            tr.colorbar.tickfont = dict(size=14, family=PLOT_PARAMS["font_family"])

    # Hide axes; set a reasonable camera
    cam = dict(eye=dict(x=0.0, y=0.0, z=2.1), up=dict(x=0.0, y=1.0, z=0.0))
    fig.update_scenes(
        dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=cam,
        )
    )

    for trace in fig.data:
        if hasattr(trace, "lighting"):
            trace.lighting = dict(
                ambient=0.2, diffuse=0.7, specular=0.1, roughness=1, fresnel=0.0
            )
        if hasattr(trace, "lightposition"):
            trace.lightposition = dict(x=0, y=0, z=1000)

    # Compact margins; embedding will handle final size
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(
                    size=int(PLOT_PARAMS["font_size_title"]),
                    family=PLOT_PARAMS["font_family"],
                ),
            )
        )

    return fig


def plot_pial_views_triplet(
    data,
    *,
    hemi: str = "left",
    views: Tuple[str, ...] = ("lateral", "medial", "ventral"),
    title: str | None = None,
    threshold: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Create a multi-row (len(views) × 1) pial surface figure (single hemisphere)
    for the given views.

    Parameters
    ----------
    data : NIfTI-like image, 1D texture array, or dict {'left': array}
        Data to render. If a 1D array is provided, it is interpreted as the
        per-vertex texture for the requested hemisphere. If a volume is given,
        it is sampled to the pial surface. Dict allows explicit hemisphere.
    hemi : {'left','right'}
        Hemisphere to display.
    views : tuple of str
        View names, typically 2 or 3 (e.g., ('lateral','medial') or
        ('lateral','medial','ventral')).
    title : str, optional
        Optional figure title (unused when embedding; kept for completeness).
    threshold : float, optional
        Threshold below which values are masked.
    vmin, vmax : float, optional
        Color scale limits. If None, computed from data (symmetric around 0).

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly Figure with three scenes stacked vertically.
    """

    if hemi not in {"left", "right"}:
        raise ValueError("hemi must be 'left' or 'right'")
    n_rows = len(views)
    if n_rows not in (2, 3):
        raise ValueError("views must contain 2 or 3 view names")

    pial_left, pial_right = _pial_meshes()
    mesh = pial_left if hemi == "left" else pial_right

    # Determine texture from input
    if isinstance(data, dict) and hemi in data:
        tex = np.asarray(data[hemi])
    elif hasattr(data, "shape") and data.ndim == 1:
        tex = np.asarray(data)
    else:
        tex = surface.vol_to_surf(data, mesh)

    # Determine color range
    if vmin is None or vmax is None:
        vmax_local = float(np.nanmax(np.abs(tex))) if np.any(np.isfinite(tex)) else 1.0
        vmin_local = -vmax_local
    else:
        vmax_local = float(vmax)
        vmin_local = float(vmin)

    fsaverage_data = load_fsaverage_data(mesh='fsaverage7', data_type="sulcal")

    fig, axes = plt.subplots(
        nrows=len(views),
        ncols=1,
        figsize=(3, 3 * len(views)),  # narrower width; 3×6 for 2 rows per guidance
        subplot_kw={'projection': '3d'}
    )

    # Handle the case len(views) == 1 (axes would be a single Axes, not iterable)
    if len(views) == 1:
        axes = [axes]

    for ax, view in zip(axes, views):
        plotting.plot_surf_stat_map(
            mesh, tex,
            hemi=hemi, view=view,
            colorbar=False, threshold=threshold,
            cmap=CMAP_BRAIN,
            vmin=vmin_local,
            vmax=vmax_local,
            axes=ax,                 # draw on this row's axis
            engine="matplotlib",
            title=None,
            bg_map=fsaverage_data,
            darkness=0.8
        )

    # Optional overall title
    if title:
        fig.suptitle(
            title,
            x=0.5,
            fontsize=int(PLOT_PARAMS["font_size_title"]),
            fontfamily=PLOT_PARAMS["font_family"],
        )
        # Make room for the suptitle
        plt.subplots_adjust(top=0.9)

    # Use negative vertical spacing to bring rows closer together
    # Tight layout is avoided here because it can override custom hspace
    plt.subplots_adjust(hspace=-0.6)

    return fig
