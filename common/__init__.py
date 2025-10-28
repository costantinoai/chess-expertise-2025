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
    compute_symmetric_range,
    compute_ylim_range,
    format_axis_commas,
    label_axes,
    style_spines,
    hide_ticks,
    save_figure,
    # Panels
    make_panel_grid,
    set_axis_title,
    add_panel_label,
    # Bars
    plot_grouped_bars_with_ci,
    plot_grouped_bars_on_ax,
    # Heatmaps
    plot_rdm,
    plot_rdm_on_ax,
    add_rdm_category_bars,
    add_roi_color_legend,
    # Scatter
    plot_2d_embedding,
    plot_2d_embedding_on_ax,
)

# Logging and setup
from .logging_utils import setup_analysis, setup_analysis_in_dir, log_script_end

# Reporting
from .report_utils import (
    create_figure_summary,
    format_correlation_summary,
    generate_latex_table,
    create_correlation_table,
    save_results_metadata,
)

# BIDS utilities
from .bids_utils import (
    get_subject_list, 
    load_participants_tsv,
    get_participants_with_expertise,
    load_stimulus_metadata,
)

# RSA utilities
from .rsa_utils import create_model_rdm, correlate_rdms, correlate_rdm_with_models, compute_pairwise_rdm_reliability

# Neuroimaging IO (optional dependency)
try:
    from .neuro_utils import load_nifti
except Exception:  # Defer hard failure until actually used
    def load_nifti(*args, **kwargs):  # type: ignore
        raise ImportError("load_nifti requires nibabel. Please install nibabel to use neuro_utils.")

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
    'compute_symmetric_range',
    'compute_ylim_range',
    'format_axis_commas',
    'label_axes',
    'style_spines',
    'hide_ticks',
    'save_figure',
    # Plotting - Panels
    'make_panel_grid',
    'set_axis_title',
    'add_panel_label',
    # Plotting - Bars
    'plot_grouped_bars_with_ci',
    'plot_grouped_bars_on_ax',
    # Plotting - Heatmaps
    'plot_rdm',
    'plot_rdm_on_ax',
    'add_rdm_category_bars',
    'add_roi_color_legend',
    # Plotting - Scatter
    'plot_2d_embedding',
    'plot_2d_embedding_on_ax',
    # Logging and setup
    'setup_analysis',
    'setup_analysis_in_dir',
    'log_script_end',
    # Reporting
    'create_figure_summary',
    'format_correlation_summary',
    'generate_latex_table',
    'create_correlation_table',
    'save_results_metadata',
    # BIDS utilities
    'get_subject_list',
    'load_participants_tsv',
    'get_participants_with_expertise',
    'load_stimulus_metadata',
    # RSA utilities
    'create_model_rdm',
    'correlate_rdms',
    'correlate_rdm_with_models',
    'compute_pairwise_rdm_reliability',
    # Neuro IO
    'load_nifti',
]
