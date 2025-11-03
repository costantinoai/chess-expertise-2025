#!/usr/bin/env python3
"""
Table generation utilities to eliminate boilerplate in 81_/82_ scripts.

Patterns consolidated:
 - Load results from a pickle with logging
 - Format into a DataFrame via a supplied formatter function
 - Generate LaTeX (with optional multicolumn headers) and CSV side-by-side
 - Optional manuscript copy via common.report_utils

Strict behavior: explicit errors on missing keys and invalid inputs. No silent
fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable
import logging
import pickle
import pandas as pd

from .report_utils import save_table_with_manuscript_copy
from .tables.style import infer_column_format, generate_styled_table


def load_results_pickle(
    results_dir: Path,
    pickle_name: str,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Load a results pickle from a results directory, with logging.
    """
    p = Path(results_dir) / pickle_name
    if logger:
        logger.info(f"Loading results: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Missing results pickle: {p}")
    with open(p, 'rb') as f:
        data = pickle.load(f)
    if logger:
        logger.info("Loaded results successfully")
    return data


def generate_expert_novice_table(
    *,
    results_dir: Path,
    output_dir: Path,
    table_name: str,
    caption: str,
    label: str,
    formatter_func: Callable[[Dict], pd.DataFrame],
    pickle_name: Optional[str] = None,
    data: Optional[Dict] = None,
    column_format: Optional[str] = None,
    manuscript_name: Optional[str] = None,
    save_csv: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Standard workflow for expert-vs-novice tables built from a results dict.

    Either 'pickle_name' or preloaded 'data' must be provided.
    The 'formatter_func' converts the data dict to a display DataFrame.
    """
    if (pickle_name is None) == (data is None):
        raise ValueError("Provide exactly one of pickle_name or data")

    if data is None:
        data = load_results_pickle(results_dir, pickle_name, logger)

    df = formatter_func(data)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("formatter_func must return a pandas.DataFrame")

    # Infer multicolumn headers from suffixes if possible
    expert_cols = [c for c in df.columns if 'Expert' in c or 'Experts' in c]
    novice_cols = [c for c in df.columns if 'Novice' in c or 'Novices' in c]
    multicolumn = {}
    if expert_cols:
        multicolumn['Experts'] = expert_cols
    if novice_cols:
        multicolumn['Novices'] = novice_cols

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{table_name}.tex"

    if column_format is None:
        column_format = infer_column_format(df, multicolumn)

    generate_styled_table(
        df=df,
        output_path=tex_path,
        caption=caption,
        label=label,
        column_format=column_format,
        multicolumn_headers=multicolumn if multicolumn else None,
        logger=logger,
    )

    csv_path: Optional[Path] = None
    if save_csv:
        csv_path = out_dir / f"{table_name}.csv"
        df.to_csv(csv_path, index=False)
        if logger:
            logger.info(f"CSV saved: {csv_path}")

    if manuscript_name:
        # Save a manuscript copy next to the LaTeX output
        content = tex_path.read_text(encoding='utf-8')
        save_table_with_manuscript_copy(content, tex_path, manuscript_name=manuscript_name, logger=logger)

    if logger:
        logger.info(f"LaTeX saved: {tex_path}")

    return tex_path, csv_path


def generate_roi_table_from_blocks(
    *,
    blocks: Dict,
    roi_info: pd.DataFrame,
    output_dir: Path,
    table_name: str,
    caption: str,
    label: str,
    subtract_chance: float = 0.0,
    manuscript_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    column_format: str = 'lSc|Sc|Sc|cc',
    csv_only: bool = False,
) -> Tuple[Optional[Path], Path]:
    """
    Generate a standard expert-vs-novice ROI table given pre-extracted blocks.

    Expects keys in 'blocks':
      - 'welch_expert_vs_novice' : DataFrame
      - 'experts_desc'           : list[tuple]
      - 'novices_desc'           : list[tuple]

    Parameters
    ----------
    csv_only : bool, default=False
        If True, only generate CSV file (skip LaTeX generation).

    Returns
    -------
    tex_path : Optional[Path]
        Path to LaTeX file, or None if csv_only=True
    csv_path : Path
        Path to CSV file
    """
    from .report_utils import format_roi_stats_table

    required = ['welch_expert_vs_novice', 'experts_desc', 'novices_desc']
    missing = [k for k in required if k not in blocks]
    if missing:
        raise KeyError(f"Missing keys in blocks: {missing}")

    df = format_roi_stats_table(
        blocks['welch_expert_vs_novice'],
        blocks['experts_desc'],
        blocks['novices_desc'],
        roi_info,
        subtract_chance=subtract_chance,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{table_name}.csv"

    # Save CSV
    df.to_csv(csv_path, index=False)
    if logger:
        logger.info(f"CSV saved: {csv_path}")

    # Skip LaTeX generation if csv_only=True
    if csv_only:
        if logger:
            logger.info(f"Skipping LaTeX generation (csv_only=True)")
        return None, csv_path

    # Generate LaTeX table
    tex_path = out_dir / f"{table_name}.tex"

    # Multicolumn grouping for standard ROI tables
    multicolumn = {
        'Experts': ['Expert_mean', 'Expert_CI'],
        'Novices': ['Novice_mean', 'Novice_CI'],
        'Experts−Novices': ['Delta_mean', 'Delta_CI'],
    }

    generate_styled_table(
        df=df,
        output_path=tex_path,
        caption=caption,
        label=label,
        column_format=column_format,
        multicolumn_headers=multicolumn,
        logger=logger,
    )

    if manuscript_name:
        content = tex_path.read_text(encoding='utf-8')
        save_table_with_manuscript_copy(content, tex_path, manuscript_name=manuscript_name, logger=logger)

    if logger:
        logger.info(f"LaTeX saved: {tex_path}")

    return tex_path, csv_path


__all__ = [
    'load_results_pickle',
    'generate_expert_novice_table',
    'generate_roi_table_from_blocks',
]
