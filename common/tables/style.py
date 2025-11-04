"""
LaTeX table styling primitives (siunitx + booktabs) used across analyses.

Provides helpers to infer column formats (S for numeric) and a high-level
generator that applies repository-wide defaults (booktabs, no resizebox).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from ..report_utils import generate_latex_table

# Repository-wide defaults
DEFAULT_DECIMALS = 3


def infer_column_format(
    df: pd.DataFrame,
    multicolumn_headers: Optional[Dict[str, List[str]]] = None,
    *,
    first_col_align: str = 'l',
    insert_bar_between_groups: bool = True,
) -> str:
    """
    Infer a LaTeX column format string using siunitx S for numeric columns.

    - First column uses `first_col_align` (default 'l')
    - Other columns: 'S' if numeric dtype else 'c'
    - If multicolumn headers are provided and insert_bar_between_groups=True,
      a vertical bar '|' is inserted before the start of the second group.
    """
    if df.shape[1] == 0:
        return first_col_align

    fmts = [first_col_align]
    for col in df.columns[1:]:
        fmts.append('S' if pd.api.types.is_numeric_dtype(df[col]) else 'c')

    # Optional visual separation between first and second group
    if multicolumn_headers and insert_bar_between_groups:
        # Determine total columns in the first group after the first label col
        group_names = list(multicolumn_headers.keys())
        if len(group_names) >= 2:
            first_group = multicolumn_headers[group_names[0]]
            split_idx = 1 + len(first_group)  # after first label col
            fmts = fmts[:split_idx] + ['|'] + fmts[split_idx:]

    return ''.join(fmts)


def build_c_only_colspec(
    df: pd.DataFrame,
    multicolumn_headers: Optional[Dict[str, List[str]]] = None,
    *,
    first_col_align: str = 'l',
) -> str:
    """
    Build a LaTeX colspec using only 'c' after the first label column,
    inserting vertical bars '|' between multicolumn groups and before any
    trailing ungrouped columns.

    Examples
    --------
    - No groups, 6 columns => 'lccccc'
    - Three groups (3,3,3) + 1 trailing col => 'lccc|ccc|ccc|c'
    """
    ncols = df.shape[1]
    if ncols <= 0:
        return first_col_align

    if not multicolumn_headers:
        return first_col_align + 'c' * (ncols - 1)

    cols = list(df.columns)
    data_cols = cols[1:]

    colspec_parts: List[str] = []
    grouped_flat: List[str] = []
    for group in multicolumn_headers.keys():
        group_cols = [c for c in multicolumn_headers[group] if c in data_cols]
        grouped_flat.extend(group_cols)
        if group_cols:
            colspec_parts.append('c' * len(group_cols))

    remaining = [c for c in data_cols if c not in grouped_flat]

    colspec = first_col_align
    if colspec_parts:
        colspec += '|'.join(colspec_parts)
        if remaining:
            colspec += '|' + 'c' * len(remaining)
    else:
        colspec += 'c' * len(remaining)

    return colspec


def generate_styled_table(
    *,
    df: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
    multicolumn_headers: Optional[Dict[str, List[str]]] = None,
    column_format: Optional[str] = None,
    logger=None,
    manuscript_name: Optional[str] = None,
    force_c_alignment: bool = False,
) -> Path:
    """
    Generate a LaTeX table with standard repository style (booktabs + siunitx).

    - Infers a column format if not provided
    - Delegates to common.report_utils.generate_latex_table
    """
    # Auto-infer multicolumn headers if not provided and column names follow
    # the pattern '<base>_<Group>' (e.g., 'Δr_Visual Similarity').
    if multicolumn_headers is None:
        inferred = _infer_multicolumn_headers(df)
        if inferred:
            multicolumn_headers = inferred

    # Sanitize leftmost label column: replace underscores with spaces
    try:
        if df.shape[1] > 0:
            first_col = df.columns[0]
            if pd.api.types.is_object_dtype(df[first_col]) or pd.api.types.is_string_dtype(df[first_col]):
                df = df.copy()
                df[first_col] = df[first_col].map(lambda v: v.replace('_', ' ') if isinstance(v, str) else v)
    except Exception:
        # Strict policy: do not fail table generation because of label sanitation
        pass

    if column_format is None:
        column_format = (
            build_c_only_colspec(df, multicolumn_headers)
            if force_c_alignment or multicolumn_headers is not None
            else infer_column_format(df, multicolumn_headers)
        )

    return generate_latex_table(
        df=df,
        output_path=output_path,
        caption=caption,
        label=label,
        column_format=column_format,
        multicolumn_headers=multicolumn_headers,
        escape=False,
        logger=logger,
        manuscript_name=manuscript_name,
        # Fit tables to linewidth to match manuscript examples
        wrap_with_resizebox=True,
        use_booktabs=True,
    )


def _infer_multicolumn_headers(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Infer multicolumn header groups from DataFrame columns using '<base>_<Group>'.

    Returns a dict mapping group name -> ordered list of original columns.
    The first column is treated as a label and never grouped.
    """
    if df.shape[1] <= 2:
        return {}

    cols = list(df.columns)
    data_cols = cols[1:]
    groups: Dict[str, List[str]] = {}

    # Preferred order of sub-columns inside a group, when present
    preferred = [
        'Δr', 'Δacc', 'Δ', 'Mean', 'M_diff', 'r', '95% CI', 't', 'p', 'pFDR', 'p_raw'
    ]

    # Helper: rank base token by preference
    def base_rank(col: str) -> int:
        base = col.split('_', 1)[0].strip()
        try:
            return preferred.index(base)
        except ValueError:
            return len(preferred) + 1

    for col in data_cols:
        if '_' not in col:
            # Ungrouped column — leave ungrouped
            continue
        base, group = col.split('_', 1)
        group = group.strip()
        groups.setdefault(group, []).append(col)

    # Sort each group's columns by preferred order where possible, otherwise stable
    for g, clist in list(groups.items()):
        clist_sorted = sorted(clist, key=base_rank)
        groups[g] = clist_sorted

    # Drop spurious groups with only a single column (avoids accidental grouping
    # on tokens like 'M_diff' in simple two-line tables)
    groups = {g: cols for g, cols in groups.items() if len(cols) >= 2}

    # If nothing could be grouped, return empty dict
    if not groups:
        return {}
    return groups
