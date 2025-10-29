"""
Behavioral analysis modules for chess expertise study.

This package provides utilities for behavioral RSA analysis:
- data_loading: Subject and stimulus data loading
- rdm_utils: RDM/DSM computation and correlation functions
- plotting: Behavioral-specific plotting functions
"""

__version__ = "0.1.0"

from .data_loading import (
    load_trial_data_from_events_tsv,
    load_participant_trial_data,
)

from .rdm_utils import (
    create_pairwise_df,
    compute_symmetric_rdm,
    compute_directional_dsm,
    correlate_with_all_models
)

# Plotting helpers are provided centrally via common.plotting.

__all__ = [
    # Data loading
    'load_trial_data_from_events_tsv',
    'load_participant_trial_data',
    # RDM utilities
    'create_pairwise_df',
    'compute_symmetric_rdm',
    'compute_directional_dsm',
    'correlate_with_all_models',
    # Plotting helpers live in common.plotting
]
