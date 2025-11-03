"""
LaTeX table styling primitives (siunitx + booktabs) used across analyses.

Provides helpers to infer column formats (S for numeric) and a high-level
generator that applies repository-wide defaults (booktabs + resizebox).
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
) -> Path:
    """
    Generate a LaTeX table with standard repository style (booktabs + resizebox + siunitx).

    - Infers a column format if not provided
    - Delegates to common.report_utils.generate_latex_table
    """
    if column_format is None:
        column_format = infer_column_format(df, multicolumn_headers)

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
        wrap_with_resizebox=True,
        use_booktabs=True,
    )

