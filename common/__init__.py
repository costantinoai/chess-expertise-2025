"""
Common utilities for chess expertise analyses.

This package provides shared functionality across all analyses:
- constants: CONFIG dictionary with all paths and parameters
- plotting: Nature-compliant plotting utilities (modular structure)
- logging_utils: Logging and analysis setup functions
- neuro_utils: Brain data loading and manipulation
- bids_utils: BIDS path helpers and participant information
- rsa_utils: RSA correlation and model RDM functions

Example Usage
-------------
>>> from common import CONFIG, setup_analysis
>>> from common import COLORS_EXPERT_NOVICE, apply_nature_rc
>>> from common import get_subject_list, load_nifti
"""

__version__ = "0.1.0"

# Configuration
from .constants import CONFIG
# Provide module-level proxies for commonly used labels/order
MODEL_ORDER = CONFIG.get('MODEL_ORDER')
MODEL_LABELS = CONFIG.get('MODEL_LABELS')
MODEL_LABELS_PRETTY = CONFIG.get('MODEL_LABELS_PRETTY')

# Plotting (Nature-compliant, modular structure)
from .plotting import (
    # Style
    PLOT_PARAMS,
    apply_nature_rc,
    figure_size,
    auto_bar_figure_size,
    # Colors
    CMAP_BRAIN,
    COLORS_EXPERT_NOVICE,
    COLORS_CHECKMATE_NONCHECKMATE,
    COLORS_WONG,
    compute_stimulus_palette,
    # Helpers
    compute_ylim_range,
    format_axis_commas,
    label_axes,
    style_spines,
    hide_ticks,
    save_figure,
    sanitize_label_to_filename,
    save_axes_svgs,
    save_panel_pdf,
    format_roi_labels_and_colors,
    # Titles
    set_axis_title,
    create_standalone_colorbar,
    # Legends
    create_roi_group_legend,
    # Bars
    plot_grouped_bars_with_ci,
    plot_grouped_bars_on_ax,
    plot_counts_on_ax,
    # Heatmaps
    plot_rdm,
    plot_rdm_on_ax,
    add_rdm_category_bars,
    add_roi_color_legend,
    plot_matrix_on_ax,
    # Scatter
    plot_2d_embedding,
    plot_2d_embedding_on_ax,
    # Surfaces
    plot_flat_pair,
    plot_flat_hemisphere,
    plot_pial_hemisphere,
)

# Logging and setup
from .logging_utils import setup_analysis, setup_analysis_in_dir, log_script_end
from .script_utils import setup_script, setup_or_reuse_analysis_dir

# IO utilities
from .io_utils import find_latest_results_directory

# Reporting
from .report_utils import (
    create_figure_summary,
    format_correlation_summary,
    generate_latex_table,
    create_correlation_table,
    save_results_metadata,
)
from .table_utils import (
    load_results_pickle,
    generate_expert_novice_table,
    generate_roi_table_from_blocks,
)

# BIDS utilities
from .bids_utils import (
    get_subject_list,
    load_participants_tsv,
    get_participants_with_expertise,
    load_stimulus_metadata,
    load_roi_metadata,
)

# RSA utilities
from .rsa_utils import create_model_rdm, correlate_rdms, correlate_rdm_with_models, compute_pairwise_rdm_reliability

# Neuroimaging IO (required dependency for neuro utilities)
from .neuro_utils import load_nifti, create_glasser22_contours

__all__ = [
    # Configuration
    'CONFIG',
    'MODEL_ORDER',
    'MODEL_LABELS',
    'MODEL_LABELS_PRETTY',
    # Plotting - Style
    'PLOT_PARAMS',
    'apply_nature_rc',
    'figure_size',
    'auto_bar_figure_size',
    # Plotting - Colors
    'CMAP_BRAIN',
    'COLORS_EXPERT_NOVICE',
    'COLORS_CHECKMATE_NONCHECKMATE',
    'COLORS_WONG',
    'compute_stimulus_palette',
    # Plotting - Helpers
    'compute_ylim_range',
    'format_axis_commas',
    'label_axes',
    'style_spines',
    'hide_ticks',
    'save_figure',
    'sanitize_label_to_filename',
    'save_axes_svgs',
    'save_panel_pdf',
    'format_roi_labels_and_colors',
    # Plotting - Titles
    'set_axis_title',
    'create_standalone_colorbar',
    'create_roi_group_legend',
    # Plotting - Bars
    'plot_grouped_bars_with_ci',
    'plot_grouped_bars_on_ax',
    'plot_counts_on_ax',
    # Plotting - Heatmaps
    'plot_rdm',
    'plot_rdm_on_ax',
    'add_rdm_category_bars',
    'add_roi_color_legend',
    'plot_matrix_on_ax',
    # Plotting - Scatter
    'plot_2d_embedding',
    'plot_2d_embedding_on_ax',
    # Plotting - Surfaces
    'plot_flat_pair',
    'plot_flat_hemisphere',
    'plot_pial_hemisphere',
    # Logging and setup
    'setup_analysis',
    'setup_analysis_in_dir',
    'log_script_end',
    'setup_script',
    'setup_or_reuse_analysis_dir',
    # IO utilities
    'find_latest_results_directory',
    # Reporting
    'create_figure_summary',
    'format_correlation_summary',
    'generate_latex_table',
    'create_correlation_table',
    'save_results_metadata',
    'load_results_pickle',
    'generate_expert_novice_table',
    'generate_roi_table_from_blocks',
    # BIDS utilities
    'get_subject_list',
    'load_participants_tsv',
    'get_participants_with_expertise',
    'load_stimulus_metadata',
    'load_roi_metadata',
    # RSA utilities
    'create_model_rdm',
    'correlate_rdms',
    'correlate_rdm_with_models',
    'compute_pairwise_rdm_reliability',
    # Neuro IO
    'load_nifti',
    'create_glasser22_contours',
]
