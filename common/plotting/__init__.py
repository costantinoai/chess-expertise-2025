"""
Centralized plotting utilities for Nature-compliant figures.

Organization:
- style.py: PLOT_PARAMS, rcParams, figure sizing
- colors.py: Color palettes, CMAP_BRAIN
- helpers.py: Range computation, axis formatting
- panels.py: Multi-panel composition
- bars.py: Bar plots with automatic sizing
- heatmaps.py: RDM/heatmap plots
- scatter.py: 2D embeddings

All files complete and ready to use!
"""

# Style and configuration
from .style import (
    PLOT_PARAMS,
    apply_nature_rc,
    figure_size,
    auto_bar_figure_size,
)

# Colors
from .colors import (
    CMAP_BRAIN,
    COLORS_EXPERT_NOVICE,
    COLORS_CHECKMATE_NONCHECKMATE,
    COLORS_WONG,
    compute_stimulus_palette,
)

# Helpers
from .helpers import (
    compute_symmetric_range,
    compute_ylim_range,
    format_axis_commas,
    label_axes,
    style_spines,
    hide_ticks,
    save_figure,
    sanitize_label_to_filename,
    save_axes_svgs,
    save_panel_svg,
    format_roi_labels_and_colors,
)

# Titles (pylustrator handles layout; only titles retained)
from .helpers import (
    set_axis_title,
)

# Bar plots
from .bars import (
    plot_grouped_bars_with_ci,
    plot_grouped_bars_on_ax,
    plot_counts_on_ax,
)

# Heatmaps (RDMs)
from .heatmaps import (
    plot_rdm,
    plot_rdm_on_ax,
    add_rdm_category_bars,
    add_roi_color_legend,
    plot_matrix_on_ax,
)

# Scatter plots (2D embeddings)
from .scatter import (
    plot_2d_embedding,
    plot_2d_embedding_on_ax,
)

# Surface plots (flat hemispheres via Plotly)
from .surfaces import (
    plot_flat_pair,
    plot_flat_hemisphere,
)

__all__ = [
    # Style
    'PLOT_PARAMS',
    'apply_nature_rc',
    'figure_size',
    'auto_bar_figure_size',

    # Colors
    'CMAP_BRAIN',
    'COLORS_EXPERT_NOVICE',
    'COLORS_CHECKMATE_NONCHECKMATE',
    'COLORS_WONG',
    'compute_stimulus_palette',

    # Helpers
    'compute_symmetric_range',
    'compute_ylim_range',
    'format_axis_commas',
    'label_axes',
    'style_spines',
    'hide_ticks',
    'save_figure',
    'sanitize_label_to_filename',
    'save_axes_svgs',
    'save_panel_svg',
    'format_roi_labels_and_colors',

    # Titles
    'set_axis_title',

    # Bars
    'plot_grouped_bars_with_ci',
    'plot_grouped_bars_on_ax',
    'plot_counts_on_ax',

    # Heatmaps
    'plot_rdm',
    'plot_rdm_on_ax',
    'add_rdm_category_bars',
    'add_roi_color_legend',
    'plot_matrix_on_ax',

    # Scatter
    'plot_2d_embedding',
    'plot_2d_embedding_on_ax',

    # Surfaces
    'plot_flat_pair',
    'plot_flat_hemisphere',
]
