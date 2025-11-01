"""
Utilities for creating chess dataset grid visualizations.

This module provides functions to generate publication-ready grids of chess board images
from stimulus metadata, with optional colored borders encoding stimulus properties.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common import (
    PLOT_PARAMS,
    apply_nature_rc,
    figure_size,
    save_figure,
    compute_stimulus_palette,
)
from common.bids_utils import load_stimulus_metadata


def _collect_image_paths(
    images_dir: Optional[Path],
    stimuli_df=None,
    image_col: Optional[str] = None,
    allowed_ext: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
) -> List[Path]:
    """
    Resolve image paths for the full chess stimulus set.

    Strategy:
    1. If stimuli_df has an image path column, use it
    2. Else, if images_dir is provided, try common filename patterns based on stim_id
    3. Sort results by stim_id when available

    Parameters
    ----------
    images_dir : Path, optional
        Base directory containing stimulus images.
    stimuli_df : DataFrame, optional
        Stimulus metadata. If None, loads from BIDS.
    image_col : str, optional
        Column name containing image paths. Auto-detected if None.
    allowed_ext : Sequence[str]
        Valid image file extensions.

    Returns
    -------
    List[Path]
        Sorted list of image file paths.
    """
    # Load stimulus metadata if not provided
    if stimuli_df is None:
        stimuli_df = load_stimulus_metadata(return_all=True)

    # Auto-detect image column if not specified
    if image_col is None:
        candidate_cols = [
            c for c in stimuli_df.columns
            if any(k in c.lower() for k in ["file", "path", "image"])
        ]
        image_col = candidate_cols[0] if candidate_cols else None

    paths: List[Path] = []

    # Strategy 1: Use image column from metadata
    if image_col is not None and image_col in stimuli_df.columns:
        for _, row in stimuli_df.iterrows():
            p = Path(str(row[image_col])).expanduser()
            # Make relative paths absolute using images_dir
            if not p.is_absolute() and images_dir is not None:
                p = Path(images_dir) / p
            paths.append(p)

    # Strategy 2: Pattern-based lookup in images_dir
    elif images_dir is not None:
        images_dir = Path(images_dir)
        stim_ids = (
            list(stimuli_df["stim_id"].astype(int).tolist())
            if "stim_id" in stimuli_df.columns
            else []
        )

        if stim_ids:
            # Try common patterns: stim_{id}.ext or {id}.ext
            for sid in stim_ids:
                found = None
                for pat in [f"stim_{sid}", f"{sid}"]:
                    for ext in allowed_ext:
                        cand = images_dir / f"{pat}{ext}"
                        if cand.exists():
                            found = cand
                            break
                    if found:
                        break

                # Fallback: glob search for sid in filename
                if not found:
                    matches = list(images_dir.glob(f"*{sid}*"))
                    found = matches[0] if matches else None

                if found:
                    paths.append(found)
        else:
            # Last resort: take all images in directory
            for ext in allowed_ext:
                paths.extend(sorted(images_dir.glob(f"*{ext}")))
    else:
        raise ValueError(
            "Cannot resolve image paths: provide stimuli metadata with an image column or specify images_dir"
        )

    # Remove duplicates and non-existing files
    uniq: List[Path] = []
    seen = set()
    for p in paths:
        if p is None:
            continue
        if p.exists() and p not in seen:
            uniq.append(p)
            seen.add(p)

    # Sort by stim_id order if available
    if stimuli_df is not None and "stim_id" in stimuli_df.columns:
        id_to_idx = {
            int(r["stim_id"]): i
            for i, r in stimuli_df.reset_index().iterrows()
            if "stim_id" in r
        }

        def _key(p: Path):
            """Extract stim_id from filename for sorting."""
            name = p.stem
            m = re.search(r"(\d+)$", name)
            return id_to_idx.get(int(m.group(1))) if m else 1e9

        uniq.sort(key=_key)
    else:
        uniq.sort()

    return uniq


def create_chess_dataset_grid(
    images_dir: Optional[Path] = None,
    image_col: Optional[str] = None,
    n_cols: int = 10,
    max_images: Optional[int] = None,
    title: Optional[str] = "Chess Board Stimuli",
    output_path: Optional[Path] = None,
    params: dict | None = None,
) -> plt.Figure:
    """
    Create a plain grid figure showing the full chess board stimulus set.

    Parameters
    ----------
    images_dir : Path, optional
        Base directory containing board images.
    image_col : str, optional
        Column in stimuli.tsv containing image paths.
    n_cols : int, default=10
        Number of images per row.
    max_images : int, optional
        Maximum number of images to display.
    title : str, optional
        Figure title.
    output_path : Path, optional
        If provided, save figure to this path.
    params : dict, optional
        PLOT_PARAMS override.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    if params is None:
        params = PLOT_PARAMS

    apply_nature_rc(params)

    # Load stimulus metadata and resolve image paths
    stimuli_df = load_stimulus_metadata(return_all=True)
    paths = _collect_image_paths(images_dir, stimuli_df, image_col)

    if max_images is not None:
        paths = paths[:max_images]

    if len(paths) == 0:
        raise ValueError("No images found to plot")

    n = len(paths)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n / n_cols))

    # Calculate figure dimensions (double column width, proportional height)
    fig_w, _ = figure_size(columns=2)
    row_height_mm = 18.0
    fig_h = min(n_rows * row_height_mm / 25.4, 10.0)

    # Create subplot grid
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(n_rows, n_cols)

    # Plot each image
    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        if idx < n:
            img = plt.imread(str(paths[idx]))
            ax.imshow(img)
        else:
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=params["font_size_title"], fontweight="bold")

    if output_path is not None:
        save_figure(fig, Path(output_path))

    return fig


def create_chess_dataset_grid_bordered(
    images_dir: Optional[Path] = None,
    image_col: Optional[str] = None,
    n_cols: int = 5,
    title: str = "Full Dataset",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a chess dataset grid with colored borders encoding stimulus properties.

    Borders use colors from the stimulus palette where:
    - Color (green/red) indicates checkmate status
    - Alpha intensity encodes strategy rank within each group

    Each image is labeled with: S{stim_id} • C/NC • SY{strategy+1} • P{visual+1}

    Parameters
    ----------
    images_dir : Path, optional
        Base directory for images.
    image_col : str, optional
        Column with image filenames/paths (auto-detected if None).
    n_cols : int, default=5
        Columns per row (rows inferred from image count).
    title : str, default="Full Dataset"
        Figure suptitle.
    output_path : Path, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        The created figure with bordered grid.

    Notes
    -----
    Figure width is fixed to 186 mm (Nature double column); height scales
    proportionally based on the number of rows.
    """
    # Apply Nature RC style and load metadata
    apply_nature_rc(PLOT_PARAMS)
    stimuli_df = load_stimulus_metadata(return_all=True)
    paths = _collect_image_paths(images_dir, stimuli_df, image_col)

    if len(paths) == 0:
        raise ValueError("No images found to plot")

    # Compute stimulus palette (colors and alphas based on check/strategy)
    palette_colors, palette_alphas = compute_stimulus_palette(stimuli_df)

    # Build lookup tables: stim_id -> index and filename -> index
    id_to_idx = {}
    if "stim_id" in stimuli_df.columns:
        id_to_idx = {
            int(r["stim_id"]): i
            for i, r in stimuli_df.reset_index().iterrows()
        }

    fname_to_idx = {}
    if "filename" in stimuli_df.columns:
        fname_to_idx = {
            str(r["filename"]).rsplit(".", 1)[0]: i
            for i, r in stimuli_df.iterrows()
        }

    def color_alpha_for(p: Path):
        """Look up border color and alpha for a given image path."""
        stem = p.stem
        # Try to extract stim_id from filename
        m = re.search(r"(\d+)$", stem)
        if m and id_to_idx:
            sid = int(m.group(1))
            if sid in id_to_idx:
                idx = id_to_idx[sid]
                return palette_colors[idx], float(palette_alphas[idx])
        # Fallback to filename lookup
        if stem in fname_to_idx:
            idx = fname_to_idx[stem]
            return palette_colors[idx], float(palette_alphas[idx])
        # Default color if no match
        return "#4C78A8", 1.0

    # Calculate grid dimensions
    n = len(paths)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n / n_cols))

    # Fixed width (186mm), height scaled from 5×8 base layout
    MM = 1.0 / 25.4
    width_in = 186.0 * MM
    base_aspect = 24.0 / 15.0  # Original 5×8 grid aspect
    row_scale = (n_rows / 8.0) if n_rows > 0 else 1.0
    col_scale = (5.0 / n_cols) if n_cols > 0 else 1.0
    height_in = width_in * base_aspect * row_scale * col_scale

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_in, height_in))
    axes = np.array(axes).reshape(n_rows, n_cols)

    # Suptitle (50% larger than standard)
    fig.suptitle(
        title,
        fontsize=PLOT_PARAMS["font_size_title"] * 1.5,
        fontweight="bold"
    )

    # Plot each image with colored border and label
    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.set_axis_off()

        if idx >= n:
            continue

        # Load and display image
        img = plt.imread(str(paths[idx]))
        ax.imshow(img)

        # Build stimulus label: S{stim} • C/NC • SY{strategy+1} • P{visual+1}
        try:
            # Resolve stimulus row from metadata
            stem = paths[idx].stem
            row = None
            m = re.search(r"(\d+)$", stem)

            # Try stim_id lookup
            if m and 'stim_id' in stimuli_df.columns:
                sid = int(m.group(1))
                hit = stimuli_df[stimuli_df['stim_id'] == sid]
                if not hit.empty:
                    row = hit.iloc[0]

            # Fallback to filename lookup
            if row is None and 'filename' in stimuli_df.columns:
                hit = stimuli_df[
                    stimuli_df['filename'].astype(str).str.contains(stem, regex=False)
                ]
                if not hit.empty:
                    row = hit.iloc[0]

            if row is not None:
                stim_id = int(row.get('stim_id', idx + 1))

                # Check status (C = checkmate, NC = non-checkmate)
                check_raw = str(row.get('check', '')).lower()
                check_tag = 'C' if 'check' in check_raw or check_raw == '1' else 'NC'

                # Strategy and visual ranks (1-indexed for display)
                strat_series = pd.to_numeric(
                    stimuli_df.get('strategy'), errors='coerce'
                ) if 'strategy' in stimuli_df.columns else None
                vis_series = pd.to_numeric(
                    stimuli_df.get('visual'), errors='coerce'
                ) if 'visual' in stimuli_df.columns else None

                strat_val = int(row.get('strategy', 0))
                vis_val = int(row.get('visual', 0))

                # Adjust to 1-indexed if metadata is 0-indexed
                strat_disp = (
                    strat_val + 1
                    if (strat_series is not None and
                        pd.notna(strat_series.min()) and
                        int(strat_series.min()) == 0)
                    else strat_val
                )
                vis_disp = (
                    vis_val + 1
                    if (vis_series is not None and
                        pd.notna(vis_series.min()) and
                        int(vis_series.min()) == 0)
                    else vis_val
                )
            else:
                # Fallback if metadata lookup fails
                stim_id = idx + 1
                check_tag = 'C'
                strat_disp = 1
                vis_disp = 1

            # Format label with bold stim_id
            tag = fr"$\bf{{S{stim_id}}}$•{check_tag}•SY{strat_disp}•P{vis_disp}"
            ax.set_title(
                tag,
                fontsize=PLOT_PARAMS['font_size_label'] * 1.5,
                pad=2,
                loc='center'
            )
        except Exception:
            # Skip labeling if any error occurs
            pass

        # Draw colored border using stimulus palette
        color, _alpha = color_alpha_for(paths[idx])
        rect = Rectangle(
            (0, 0), 1, 1,
            fill=False,
            transform=ax.transAxes,
            edgecolor=color,
            linewidth=6.0,
            alpha=1.0,
            zorder=10
        )
        ax.add_patch(rect)

    # Adjust layout
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    except Exception:
        plt.tight_layout()

    if output_path is not None:
        save_figure(fig, Path(output_path))

    return fig
