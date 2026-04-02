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

    This function builds a design matrix suitable for nilearn's SecondLevelModel
    to test for group differences between experts and novices. The design uses
    effects coding where experts are coded as +1 and novices as -1, making the
    intercept represent the grand mean and the group coefficient represent half
    the difference between groups.

    Parameters
    ----------
    n_experts : int
        Number of expert subjects
    n_novices : int
        Number of novice subjects

    Returns
    -------
    pandas.DataFrame
        Design matrix with shape (n_experts + n_novices, 2) containing:
        - 'intercept': All ones (grand mean across both groups)
        - 'group': +1 for experts, -1 for novices (tests group difference)

    Notes
    -----
    - Effects coding (±1) differs from dummy coding (0/1)
    - The intercept beta represents the average activation across all subjects
    - The group beta represents (expert_mean - novice_mean) / 2
    - To get the full difference: multiply group beta by 2

    Examples
    --------
    >>> # Create design matrix for 20 experts and 24 novices
    >>> dm = build_design_matrix(n_experts=20, n_novices=24)
    >>> print(dm.shape)
    (44, 2)
    >>> print(dm.head())
       intercept  group
    0        1.0    1.0
    1        1.0    1.0
    ...
    >>> print(dm.tail())
       intercept  group
    42       1.0   -1.0
    43       1.0   -1.0
    """
    intercept = np.ones(n_experts + n_novices)
    group = np.concatenate([np.ones(n_experts), -np.ones(n_novices)])
    return pd.DataFrame({'intercept': intercept, 'group': group})

