"""
MVPA I/O helpers (analysis-specific)

Utilities to discover and load subject-level TSV artifacts produced by
CoSMoMVPA MATLAB scripts for decoding (svm) and RSA correlations (rsa_corr).

These functions are intentionally minimal and reuse common utilities upstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import pandas as pd
import re
import numpy as np

from common.io_utils import sanitize_matlab_varnames, find_subject_tsvs
from common.bids_utils import to_sub_id

## find_subject_tsvs now provided by common.io_utils


def parse_subject_id_from_path(path: Path) -> str:
    """Extract subject ID (XX) from a path like .../sub-XX/filename.tsv."""
    m = re.search(r"sub-([0-9]+)", str(path))
    return m.group(1) if m else ""


def load_subject_tsv(path: Path) -> pd.DataFrame:
    """
    Load a subject-level TSV into a DataFrame with subject metadata.

    Adds columns:
    - participant_id: 'XX' string
    - target: preserved from file

    Returns
    -------
    pd.DataFrame
        Columns: participant_id, target, ROI columns...
    """
    df = pd.read_csv(path, sep="\t")
    if "target" not in df.columns:
        # Some legacy outputs used 'regressor'
        if "regressor" in df.columns:
            df = df.rename(columns={"regressor": "target"})
        else:
            raise ValueError(f"TSV missing 'target' column: {path}")

    participant_id = parse_subject_id_from_path(path)
    df.insert(0, "participant_id", participant_id)
    return df


def build_group_dataframe(
    files: Iterable[Path],
    participants_list: List[Tuple[str, bool]],
    roi_names: List[str],
) -> pd.DataFrame:
    """
    Concatenate subject TSVs and add 'expert' boolean from participants.

    Parameters
    ----------
    files : iterable of Path
        Subject-level TSV paths
    participants_list : list of (sub_id, is_expert)
        From common.bids_utils.get_participants_with_expertise
    roi_names : list of str
        ROI names to keep/align as columns

    Returns
    -------
    pd.DataFrame
        Columns: participant_id, expert, target, ROI columns (roi_names order)
    """
    rows: List[pd.DataFrame] = []
    expert_lookup: Dict[str, bool] = {sid: is_exp for sid, is_exp in participants_list}

    # MATLAB TSVs may use different ROI naming; prepare sanitized target names
    roi_names_sanitized = sanitize_matlab_varnames(roi_names)
    rename_map = {san: orig for san, orig in zip(roi_names_sanitized, roi_names)}

    for f in files:
        sub_df = load_subject_tsv(f)
        sid = sub_df["participant_id"].iloc[0]
        # participants.tsv uses 'sub-XX' IDs; TSVs store 'XX'. Normalize.
        key = to_sub_id(sid)
        sub_df.insert(1, "expert", bool(expert_lookup.get(key, False)))

        # Identify ROI columns present in file
        present_roi_cols = [c for c in sub_df.columns if c not in ("participant_id", "expert", "target")]

        # CRITICAL FIX: Sanitize the MATLAB column names to match expected names
        # MATLAB outputs have characters like + and - that need to be removed
        present_roi_cols_sanitized = sanitize_matlab_varnames(present_roi_cols)
        col_rename_from_matlab = {orig: san for orig, san in zip(present_roi_cols, present_roi_cols_sanitized)}
        sub_df = sub_df.rename(columns=col_rename_from_matlab)

        # Now check overlap with sanitized expected names
        overlap = set(present_roi_cols_sanitized).intersection(roi_names_sanitized)
        if len(overlap) >= max(1, int(0.5 * len(roi_names))):
            # Create missing sanitized columns to preserve shape
            for col in roi_names_sanitized:
                if col not in sub_df.columns:
                    sub_df[col] = np.nan
            keep_cols = ["participant_id", "expert", "target"] + roi_names_sanitized
            sub_df = sub_df[keep_cols].rename(columns=rename_map)
            sub_df = sub_df[["participant_id", "expert", "target"] + roi_names]
        else:
            # Passthrough: keep whatever ROI columns exist; do not rename
            raise ValueError(
                f"Low ROI name overlap for {f.name} (overlap={len(overlap)}/{len(roi_names)})."
            )
            sub_df = sub_df[["participant_id", "expert", "target"] + present_roi_cols_sanitized]
        rows.append(sub_df)

    if not rows:
        return pd.DataFrame(columns=["participant_id", "expert", "target"] + roi_names)
    return pd.concat(rows, axis=0, ignore_index=True)


__all__ = [
    "find_subject_tsvs",
    "load_subject_tsv",
    "build_group_dataframe",
]
