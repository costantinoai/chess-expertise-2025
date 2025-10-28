"""
GLM utilities for neurosynth RSA group analysis (local module).

Provides small helpers for building design matrices consistent with
`nilearn.glm.second_level.SecondLevelModel`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_design_matrix(n_experts: int, n_novices: int) -> pd.DataFrame:
    """
    Create a second-level design matrix with intercept and group regressor.

    Parameters
    ----------
    n_experts : int
        Number of expert subjects
    n_novices : int
        Number of novice subjects

    Returns
    -------
    pandas.DataFrame
        Columns: 'intercept', 'group' (+1 for experts, -1 for novices)
    """
    intercept = np.ones(n_experts + n_novices)
    group = np.concatenate([np.ones(n_experts), -np.ones(n_novices)])
    return pd.DataFrame({'intercept': intercept, 'group': group})

