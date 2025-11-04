"""
Table helpers for neurosynth analyses (thin wrappers over common.report_utils).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Dict
from common.tables import generate_styled_table, build_c_only_colspec


def save_latex_correlation_tables(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    df_diff: pd.DataFrame,
    run_id: str,
    out_dir: Path | str,
    ) -> None:
    """
    Save three LaTeX tables for positive, negative, and difference correlations.
    Uses generate_latex_table under the hood.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(df: pd.DataFrame, base_name: str, caption: str, label: str):
        path = out_dir / f"{base_name}.tex"
        # Use centralized styled generator for consistency and validation
        generate_styled_table(
            df=df,
            output_path=path,
            caption=caption,
            label=label,
            column_format=build_c_only_colspec(df),
        )

    _save(
        df_pos,
        base_name=f"{run_id}_positive_zmap",
        caption=f"{run_id} — Positive z-map: correlations with each term map.",
        label=f"tab:{run_id}_pos",
    )

    _save(
        df_neg,
        base_name=f"{run_id}_negative_zmap",
        caption=f"{run_id} — Negative z-map: correlations with each term map.",
        label=f"tab:{run_id}_neg",
    )

    _save(
        df_diff,
        base_name=f"{run_id}_difference_zmap",
        caption=f"{run_id} — Difference in correlations (positive $-$ negative).",
        label=f"tab:{run_id}_diff",
    )


def save_latex_combined_pos_neg_diff(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    df_diff: pd.DataFrame,
    run_id: str,
    out_path: Path | str,
) -> Path:
    """
    Create a single LaTeX table per run with multicolumn headers for
    Positive, Negative, and Difference results.

    Columns under each group: r, CI_low, CI_high, p_fdr.
    """
    out_path = Path(out_path)

    # Normalize keys
    def _sel(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        keep = ['term'] + [c for c in cols if c in df.columns]
        return df[keep].copy()

    pos = _sel(df_pos, ['r', 'CI_low', 'CI_high', 'p_fdr'])
    neg = _sel(df_neg, ['r', 'CI_low', 'CI_high', 'p_fdr'])
    dif = _sel(df_diff, ['r_diff', 'CI_low', 'CI_high', 'p_fdr'])

    # Merge on term
    merged = pos.merge(neg, on='term', suffixes=('_pos', '_neg'))
    merged = merged.merge(dif, on='term')

    # Rename for grouping clarity
    rename_map = {
        'r_pos': 'r', 'CI_low_pos': 'CI_low', 'CI_high_pos': 'CI_high', 'p_fdr_pos': 'p_fdr',
        'r_neg': 'r', 'CI_low_neg': 'CI_low', 'CI_high_neg': 'CI_high', 'p_fdr_neg': 'p_fdr',
        'r_diff': 'r', 'CI_low': 'CI_low', 'CI_high': 'CI_high', 'p_fdr': 'p_fdr',
    }
    # Build final columns
    final_cols = ['term',
                  'r_pos', 'CI_low_pos', 'CI_high_pos', 'p_fdr_pos',
                  'r_neg', 'CI_low_neg', 'CI_high_neg', 'p_fdr_neg',
                  'r_diff', 'CI_low', 'CI_high', 'p_fdr']
    merged = merged[final_cols].rename(columns=rename_map)

    multicolumn = {
        'Positive z-map': ['r', 'CI_low', 'CI_high', 'p_fdr'],
        'Negative z-map': ['r', 'CI_low', 'CI_high', 'p_fdr'],
        'Difference (pos $-$ neg)': ['r', 'CI_low', 'CI_high', 'p_fdr'],
    }

    caption = f"{run_id} — Correlations with Neurosynth term maps (positive/negative/difference)."
    label = f"tab:{run_id}_combined"

    return generate_styled_table(
        df=merged,
        output_path=out_path,
        caption=caption,
        label=label,
        multicolumn_headers=multicolumn,
        column_format=build_c_only_colspec(merged, multicolumn_headers=multicolumn),
    )


def generate_latex_multicolumn_table(
    data_dict: Dict[str, pd.DataFrame],
    output_path: Path | str,
    table_type: str = "diff",
    caption: str = "",
    label: str = "",
    pretty_names: Dict[str, str] | None = None,
) -> Path:
    """
    Build a multicolumn LaTeX table across multiple patterns (e.g., RSA models).

    Parameters
    ----------
    data_dict : dict
        Mapping from key (pattern) -> DataFrame with columns dependent on table_type.
    table_type : {'pos','neg','diff'}
        Selects which columns to extract from each DataFrame:
        - 'pos'/'neg': ['r','CI_low','CI_high','p_fdr']
        - 'diff':      ['r_diff','CI_low','CI_high','p_fdr']
    pretty_names : dict, optional
        Mapping from key to pretty group name for the multicolumn header.
    """
    output_path = Path(output_path)

    if table_type == 'diff':
        cols = ['r_diff', 'CI_low', 'CI_high', 'p_fdr']
        rename_base = {'r_diff': 'r', 'CI_low': 'CI_low', 'CI_high': 'CI_high', 'p_fdr': 'p_fdr'}
    else:
        cols = ['r', 'CI_low', 'CI_high', 'p_fdr']
        rename_base = {'r': 'r', 'CI_low': 'CI_low', 'CI_high': 'CI_high', 'p_fdr': 'p_fdr'}

    # Start with term column
    merged = None
    multicolumn = {}

    for key, df in data_dict.items():
        group = pretty_names.get(key, key) if pretty_names else key
        dfk = df[['term'] + [c for c in cols if c in df.columns]].copy()
        # Rename columns with suffix per group for uniqueness
        ren = {c: f"{rename_base[c]}_{group}" for c in cols if c in dfk.columns}
        dfk = dfk.rename(columns=ren)
        if merged is None:
            merged = dfk
        else:
            merged = merged.merge(dfk, on='term', how='outer')
        # Track group columns for multicolumn header
        multicolumn[group] = [v for k, v in ren.items()]

    # Order columns: term first, then groups in insertion order
    ordered_cols = ['term']
    for group, cols_ in multicolumn.items():
        ordered_cols.extend(cols_)
    merged = merged[ordered_cols]

    return generate_styled_table(
        df=merged,
        output_path=output_path,
        caption=caption,
        label=label,
        multicolumn_headers=multicolumn,
        column_format=build_c_only_colspec(merged, multicolumn_headers=multicolumn),
    )
