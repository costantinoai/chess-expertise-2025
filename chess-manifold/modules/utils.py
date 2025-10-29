"""
Utility helpers for manifold module.

Currently includes:
- ensure_roi_order: reindex DataFrame columns to a specific ROI order
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


def ensure_roi_order(df: pd.DataFrame, roi_labels: Iterable[int]) -> pd.DataFrame:
    """
    Reindex DataFrame columns to match roi_labels order, preserving other columns.

    If a ROI is missing, inserts a column filled with NaN to preserve shape.
    """
    roi_labels = list(roi_labels)
    # Non-ROI columns remain untouched; apply ordering to ROI columns only
    non_roi = [c for c in df.columns if c not in roi_labels]
    # Create any missing ROI columns to keep shape consistent
    out = df.copy()
    for lbl in roi_labels:
        if lbl not in out.columns:
            out[lbl] = np.nan
    return out[non_roi + roi_labels]


__all__ = [
    'ensure_roi_order',
]

