"""
Feature preparation for eyetracking decoding analyses.

Converts long-format eyetracking time series into fixed-length per-run
feature vectors suitable for scikit-learn classifiers.
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
import pandas as pd


def _choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Pick a consistent set of columns to represent the time series.

    Preference order:
      1) 'displacement'
      2) ['x_coordinate','y_coordinate']
    Raises if neither is present.
    """
    cols = df.columns
    if 'displacement' in cols:
        return ['displacement']
    xy = [c for c in ['x_coordinate', 'y_coordinate'] if c in cols]
    if len(xy) == 2:
        return xy
    raise ValueError("Eyetracking TSV must contain 'displacement' or both 'x_coordinate' and 'y_coordinate'.")


def prepare_run_level_features(
    df: pd.DataFrame,
    feature_type: str = 'auto',
    min_length_threshold: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build fixed-length per-run feature vectors and labels.

    - Groups by subject and run
    - Selects feature columns (displacement or x/y)
    - Truncates all runs to the minimum length across runs for consistency
    - Flattens each run's time series into a 1D vector

    Parameters
    ----------
    df : DataFrame
        Must contain 'subject', 'run', 'expert' plus time series columns
    feature_type : {'auto', 'xy', 'displacement'}, default='auto'
        Which features to extract:
        - 'auto': auto-select (displacement if available, else x/y)
        - 'xy': use x_coordinate and y_coordinate
        - 'displacement': use displacement only
    min_length_threshold : int, default=50
        Minimum acceptable timepoints after truncation; below raises ValueError

    Returns
    -------
    X : ndarray, shape (n_runs, n_features)
    y : ndarray, shape (n_runs,)
        Boolean labels (True=expert)
    groups : ndarray, shape (n_runs,)
        Group labels (subject IDs) for group-aware CV
    """
    # Select feature columns based on feature_type
    if feature_type == 'auto':
        feat_cols = _choose_feature_columns(df)
    elif feature_type == 'xy':
        if not all(c in df.columns for c in ['x_coordinate', 'y_coordinate']):
            raise ValueError("feature_type='xy' requires 'x_coordinate' and 'y_coordinate' columns")
        feat_cols = ['x_coordinate', 'y_coordinate']
    elif feature_type == 'displacement':
        if 'displacement' not in df.columns:
            raise ValueError("feature_type='displacement' requires 'displacement' column")
        feat_cols = ['displacement']
    else:
        raise ValueError(f"Invalid feature_type='{feature_type}'. Must be 'auto', 'xy', or 'displacement'.")
    if 'subject' not in df.columns or 'run' not in df.columns or 'expert' not in df.columns:
        raise ValueError("DataFrame must include 'subject', 'run', 'expert' columns")

    # Compute min length across runs for chosen feature set
    lengths = (
        df.groupby(['subject', 'run'])[feat_cols]
        .apply(lambda g: len(g))
        .values
        .tolist()
    )
    if not lengths:
        return np.empty((0, 0)), np.array([]), np.array([])
    min_len = int(np.min(lengths))
    if min_len < min_length_threshold:
        raise ValueError(f"Minimum run length {min_len} < threshold {min_length_threshold}.")

    rows = []
    labels = []
    groups = []

    # Sort for deterministic ordering
    grouped = df.groupby(['subject', 'run'], sort=True)
    for (sub, run), g in grouped:
        g_sel = g[feat_cols].iloc[:min_len].to_numpy()
        vec = np.nan_to_num(g_sel, nan=0.0).astype(float).ravel()
        rows.append(vec)
        labels.append(bool(g['expert'].iloc[0]))
        groups.append(sub)

    X = np.vstack(rows)
    y = np.array(labels, dtype=bool)
    groups = np.array(groups)
    return X, y, groups

