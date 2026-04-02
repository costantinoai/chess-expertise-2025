"""
Eyetracking supplementary analysis modules.

Contains IO and feature-preparation utilities for decoding analyses.
"""

from .io import (
    find_eyetracking_tsvs,
    load_eyetracking_dataframe,
)
from .features import (
    prepare_run_level_features,
)

__all__ = [
    'find_eyetracking_tsvs',
    'load_eyetracking_dataframe',
    'prepare_run_level_features',
]

