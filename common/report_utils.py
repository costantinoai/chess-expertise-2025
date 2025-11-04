"""
Reporting utilities for generating figure summaries and LaTeX tables.

Primary API for LaTeX/reporting. Prefer using functions in this module for:
- DataFrame → LaTeX table generation (with optional multicolumn headers)
- Prebuilt correlation result tables (experts vs novices)
- Output summaries and metadata saving

This module is the single source of truth for LaTeX/reporting helpers.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .formatters import format_ci, format_p_cell, shorten_roi_name
import logging


def create_figure_summary(
    results_dir: Path,
    figures_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    analysis_name: str = "Analysis",
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Create a text summary of all figures and tables in a results directory.

    This function scans a results directory for PDF figures and LaTeX tables,
    then generates a human-readable summary file listing all outputs.

    Parameters
    ----------
    results_dir : Path
        Results directory containing figures/ and tables/ subdirectories
    figures_dir : Path, optional
        Custom figures directory. If None, uses results_dir/figures
    tables_dir : Path, optional
        Custom tables directory. If None, uses results_dir/tables
    analysis_name : str, default="Analysis"
        Name of the analysis for the summary header
    logger : logging.Logger, optional
        Logger instance for logging progress

    Returns
    -------
    Path
        Path to the created summary file (results_dir/output_summary.txt)

    Notes
    -----
    - Lists all .pdf files in figures_dir
    - Lists all .tex files in tables_dir
    - Includes timestamp and total counts
    - Summary is saved as plain text for easy reading

    Example
    -------
    >>> # Create summary after generating all outputs
    >>> summary_path = create_figure_summary(
    ...     results_dir=Path("results/20251018-120000_behavioral_rsa"),
    ...     analysis_name="Behavioral RSA",
    ...     logger=logger
    ... )
    """
    results_dir = Path(results_dir)

    # Determine subdirectories
    if figures_dir is None:
        figures_dir = results_dir / "figures"
    if tables_dir is None:
        tables_dir = results_dir / "tables"

    # Get lists of files
    figure_list = sorted(figures_dir.glob("*.pdf")) if figures_dir.exists() else []
    table_list = sorted(tables_dir.glob("*.tex")) if tables_dir.exists() else []

    # Create summary text
    summary_text = f"""
{analysis_name.upper()} - FIGURES AND TABLES
{'=' * 80}

Results from: {results_dir.name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total figures: {len(figure_list)}
Total tables: {len(table_list)}

"""

    # Add figures section
    if figure_list:
        summary_text += "Figures (in figures/):\n"
        for fig_path in figure_list:
            summary_text += f"  - {fig_path.name}\n"
    else:
        summary_text += "No figures found.\n"

    summary_text += "\n"

    # Add tables section
    if table_list:
        summary_text += "LaTeX tables (in tables/):\n"
        for table_path in table_list:
            summary_text += f"  - {table_path.name}\n"
    else:
        summary_text += "No LaTeX tables found.\n"

    # Save summary
    summary_path = results_dir / "output_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    if logger:
        logger.info(f"\n{summary_text}")
        logger.info(f"Summary saved to: {summary_path}")

    return summary_path


def write_group_stats_outputs(
    out_dir: Path,
    method: str,
    target: str,
    blocks: Dict[str, pd.DataFrame],
) -> None:
    """
    Write standard group statistics outputs to CSV files for a method/target.

    Parameters
    ----------
    out_dir : Path
        Base output directory
    method : str
        Analysis method (e.g., 'svm', 'rsa_corr')
    target : str
        Target name (e.g., 'checkmate')
    blocks : dict
        Dictionary containing keys:
          - 'welch_expert_vs_novice' : DataFrame
          - 'experts_vs_chance' : DataFrame
          - 'novices_vs_chance' : DataFrame
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks['welch_expert_vs_novice'].to_csv(out_dir / f"ttest_{method}_{target}_experts_vs_novices.csv", index=False)
    blocks['experts_vs_chance'].to_csv(out_dir / f"ttest_{method}_{target}_experts_vs_chance.csv", index=False)
    blocks['novices_vs_chance'].to_csv(out_dir / f"ttest_{method}_{target}_novices_vs_chance.csv", index=False)


def format_correlation_summary(
    expert_results: List[Tuple[str, float, float, float, float]],
    novice_results: List[Tuple[str, float, float, float, float]],
    model_columns: List[str],
    model_labels_map: Optional[Dict[str, str]] = None,
    ci_brackets: bool = True,
    p_sci: bool = False,
    exp_p_fdr: Optional[Dict[str, float]] = None,
    nov_p_fdr: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Create a standardized correlation summary DataFrame for export.

    Parameters
    ----------
    expert_results : list of tuple
        Each tuple: (model_name, r, p, ci_lower, ci_upper)
    novice_results : list of tuple
        Same structure as expert_results
    model_columns : list of str
        Model column keys to display, in desired order
    model_labels_map : dict, optional
        Mapping from model key to pretty label for the 'Model' column
    ci_brackets : bool, default=True
        If True, formats CI as "[low, high]" strings; else separate cols
    p_sci : bool, default=True
        If True, p-values are formatted in scientific notation strings

    Returns
    -------
    pd.DataFrame
        Columns: Model, r_Experts, 95%_CI_Experts, p_Experts, pFDR_Experts,
                 r_Novices, 95%_CI_Novices, p_Novices, pFDR_Novices
        Ordered by model_columns
    """
    # Convert results to dict by model name for quick lookup
    exp_by_name = {name: (r, p, ci_l, ci_u) for name, r, p, ci_l, ci_u in expert_results}
    nov_by_name = {name: (r, p, ci_l, ci_u) for name, r, p, ci_l, ci_u in novice_results}

    rows = []
    for key in model_columns:
        er, ep, eci_l, eci_u = exp_by_name.get(key, (float('nan'), float('nan'), float('nan'), float('nan')))
        nr, npv, nci_l, nci_u = nov_by_name.get(key, (float('nan'), float('nan'), float('nan'), float('nan')))

        label = key
        if model_labels_map and key in model_labels_map:
            label = model_labels_map[key].replace('\n', ' ')

        # Build row with proper column ordering (group expert cols together, then novice cols)
        row = {
            'Model': label,
            'r_Experts': f"{er:.3f}",
            '95%_CI_Experts': f"\\numrange{{{eci_l:.3f}}}{{{eci_u:.3f}}}" if ci_brackets else None,
            'p_Experts': format_p_cell(ep),
        }
        # Add pFDR_Experts immediately after other expert columns
        if isinstance(exp_p_fdr, dict) and key in exp_p_fdr:
            pexp_fdr = exp_p_fdr[key]
            row['pFDR_Experts'] = format_p_cell(pexp_fdr)
        # Now add all novice columns
        row.update({
            'r_Novices': f"{nr:.3f}",
            '95%_CI_Novices': f"\\numrange{{{nci_l:.3f}}}{{{nci_u:.3f}}}" if ci_brackets else None,
            'p_Novices': format_p_cell(npv),
        })
        # Add pFDR_Novices immediately after other novice columns
        if isinstance(nov_p_fdr, dict) and key in nov_p_fdr:
            pnov_fdr = nov_p_fdr[key]
            row['pFDR_Novices'] = format_p_cell(pnov_fdr)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_latex_table(
    df: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
    column_format: Optional[str] = None,
    multicolumn_headers: Optional[Dict[str, List[str]]] = None,
    escape: bool = False,
    logger: Optional[logging.Logger] = None,
    manuscript_name: Optional[str] = None,
    wrap_with_resizebox: bool = False,
    use_booktabs: bool = True,
    na_rep: str = "--",
) -> Path:
    """
    Generate a publication-ready LaTeX table from a DataFrame.

    This function creates a LaTeX table with support for multi-level column
    headers (e.g., "Experts" and "Novices" as top-level groups with sub-columns
    for r, p, CI, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Data to convert to LaTeX table
    output_path : Path
        Path to save the .tex file
    caption : str
        LaTeX table caption
    label : str
        LaTeX label for referencing (e.g., "tab:behavioral_correlations")
    column_format : str, optional
        LaTeX column format string (e.g., "lcccccc")
        If None, auto-generated based on DataFrame columns
    multicolumn_headers : dict, optional
        Dictionary mapping group names to lists of column names
        Example: {"Experts": ["r_Experts", "p_Experts"], "Novices": ["r_Novices", "p_Novices"]}
        If provided, generates \\multicolumn headers in LaTeX
    escape : bool, default=False
        Whether to escape special LaTeX characters in cell values
        Set to False if using LaTeX math mode (e.g., $\\it{r}$)
    logger : logging.Logger, optional
        Logger instance for logging progress
    manuscript_name : str, optional
        If provided, also copy the table to CONFIG['MANUSCRIPT_TABLES_DIR']
        using this filename (e.g., 'roi_maps_rsa.tex')

    Returns
    -------
    Path
        Path to the created .tex file

    Notes
    -----
    - Uses pandas.DataFrame.to_latex() for basic table generation
    - Supports multi-level column headers via \\multicolumn{n}{c}{Header}
    - Caption and label are automatically added
    - Table is wrapped in \\begin{table}...\\end{table} environment

    Example
    -------
    >>> # Simple table without multicolumn headers
    >>> df = pd.DataFrame({
    ...     "Model": ["Checkmate", "Strategy"],
    ...     "r": [0.49, 0.20],
    ...     "p": [0.001, 0.05]
    ... })
    >>> generate_latex_table(
    ...     df, Path("tables/results.tex"),
    ...     caption="Correlation results",
    ...     label="tab:correlations"
    ... )

    >>> # Table with multicolumn headers for Experts vs Novices
    >>> df = pd.DataFrame({
    ...     "Model": ["Checkmate", "Strategy"],
    ...     "r_Experts": [0.49, 0.20],
    ...     "p_Experts": [0.001, 0.05],
    ...     "r_Novices": [0.05, 0.03],
    ...     "p_Novices": [0.80, 0.85]
    ... })
    >>> multicolumn = {
    ...     "Experts": ["r_Experts", "p_Experts"],
    ...     "Novices": ["r_Novices", "p_Novices"]
    ... }
    >>> generate_latex_table(
    ...     df, Path("tables/group_comparison.tex"),
    ...     caption="Expert vs Novice correlations",
    ...     label="tab:group_comparison",
    ...     multicolumn_headers=multicolumn
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-generate column format if not provided
    if column_format is None:
        # First column left-aligned, rest centered
        column_format = "l" + "c" * (len(df.columns) - 1)

    # Preflight: ensure columns designated as 'S' are numeric
    def _parse_colspec(fmt: str) -> List[str]:
        return [ch for ch in fmt if ch.strip() and ch != '|']

    colspec_tokens = _parse_colspec(column_format)
    if len(colspec_tokens) != len(df.columns):
        raise ValueError(
            f"column_format token count ({len(colspec_tokens)}) does not match number of DataFrame columns ({len(df.columns)})."
        )
    for idx, tok in enumerate(colspec_tokens):
        if tok == 'S' and not pd.api.types.is_numeric_dtype(df.iloc[:, idx]):
            raise TypeError(
                f"Column '{df.columns[idx]}' mapped to 'S' (siunitx numeric) but dtype is not numeric. "
                "Convert the column to float or change column_format."
            )

    # Sanitize caption: escape bare % outside math mode
    def _escape_caption_text(text: str) -> str:
        if text is None:
            return text
        out: List[str] = []
        in_math = False
        prev_char: Optional[str] = None
        for ch in text:
            if ch == '$':
                in_math = not in_math
                out.append(ch)
            elif ch == '%' and not in_math:
                # Avoid double-escaping already-escaped sequences like "\%"
                if prev_char != '\\':
                    out.append(r"\%")
                else:
                    # Already escaped: keep as-is
                    out.append(ch)
            else:
                out.append(ch)
            prev_char = ch
        return ''.join(out)

    caption_safe = _escape_caption_text(caption)

    # Generate basic LaTeX table
    try:
        latex_table = df.to_latex(
            index=False,
            caption=caption_safe,
            label=label,
            escape=escape,
            column_format=column_format,
            na_rep=na_rep,
            multicolumn=True,
            multicolumn_format='c',
            bold_rows=False,
            longtable=False,
            float_format="%.3f",
            # booktabs is accepted in recent pandas; guard with try
            booktabs=use_booktabs,
        )
    except TypeError:
        # Fallback if pandas version lacks some args
        latex_table = df.to_latex(
            index=False,
            caption=caption_safe,
            label=label,
            escape=escape,
            column_format=column_format,
            na_rep=na_rep,
            multicolumn=True,
            multicolumn_format='c',
            bold_rows=False,
            longtable=False,
            float_format="%.3f",
        )

    # Add multicolumn headers if provided
    if multicolumn_headers is not None:
        latex_table = _add_multicolumn_headers(
            latex_table, df.columns.tolist(), multicolumn_headers
        )
    else:
        # No multicolumns: stylize math tokens and escape % in header row
        latex_table = _stylize_and_escape_simple_header(latex_table)

    # Sanitize known unicode/control sequences that break LaTeX compilers
    # - Replace Unicode minus and en-dash with math minus
    # - Replace stray BEL-alpha artifacts (\x07lpha) with \alpha
    latex_table = (
        latex_table
        .replace('\u2212', '$-$')
        .replace('−', '$-$')
        .replace('–', '--')
        .replace('\x07lpha', '\\alpha')
    )

    # Enforce capital Delta in any delta math tokens
    latex_table = latex_table.replace('$\\delta', '$\\Delta')

    # Normalize math tokens in headers to requested forms
    latex_table = latex_table.replace('$p_\\mathrm{FDR}$', '$p_{FDR}$')
    latex_table = latex_table.replace('$M_{\\text{diff}}$', '$M_{diff}$')

    # Optionally wrap the tabular with a resizebox to fit linewidth
    if wrap_with_resizebox:
        latex_table = _wrap_tabular_with_resizebox(latex_table)

    # Validate LaTeX tabular before saving
    errors = validate_latex_table(latex_table)
    if errors:
        msg = "LaTeX table validation failed:\n- " + "\n- ".join(errors)
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    # Save to file (and optionally to manuscript folder)
    save_table_with_manuscript_copy(
        latex_table,
        output_path,
        manuscript_name=manuscript_name,
        logger=logger
    )

    return output_path


def create_correlation_table(
    expert_results: List[Tuple[str, float, float, float, float]],
    novice_results: List[Tuple[str, float, float, float, float]],
    model_labels: Optional[Dict[str, str]] = None,
    caption: str = "Behavioral RDM correlations with model RDMs",
    label: str = "tab:behavioral_correlations",
    exp_p_fdr: Optional[Dict[str, float]] = None,
    nov_p_fdr: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Create a correlation summary DataFrame for expert vs novice results.

    Parameters
    ----------
    expert_results : list of tuple
        Each: (column, r, p, ci_lower, ci_upper)
    novice_results : list of tuple
        Each: (column, r, p, ci_lower, ci_upper)
    model_labels : dict, optional
        Mapping from column names to pretty labels
    caption : str
        Table caption
    label : str
        LaTeX label for cross-referencing

    Returns
    -------
    pd.DataFrame
        Table with columns grouped for Experts and Novices
    """
    if model_labels is None:
        model_labels = {
            'check': 'Checkmate',
            'visual': 'Visual Similarity',
            'strategy': 'Strategy'
        }

    # Build rows using formatters (numeric columns remain numeric for siunitx)
    df_rows = []
    for exp_res, nov_res in zip(expert_results, novice_results):
        key = exp_res[0]
        label_pretty = model_labels.get(key, key.capitalize()) if model_labels else key
        er, ep, ecl, ech = exp_res[1:]
        nr, npv, ncl, nch = nov_res[1:]
        # Build row with proper column ordering (group expert cols together, then novice cols)
        row = {
            'Model': label_pretty,
            'r_Experts': float(er),
            '95%_CI_Experts': format_ci(ecl, ech, precision=3, latex=False, use_numrange=True),
            'p_Experts': format_p_cell(ep),
        }
        # Add pFDR_Experts immediately after other expert columns
        if isinstance(exp_p_fdr, dict) and key in exp_p_fdr:
            row['pFDR_Experts'] = format_p_cell(exp_p_fdr[key])
        # Now add all novice columns
        row.update({
            'r_Novices': float(nr),
            '95%_CI_Novices': format_ci(ncl, nch, precision=3, latex=False, use_numrange=True),
            'p_Novices': format_p_cell(npv),
        })
        # Add pFDR_Novices immediately after other novice columns
        if isinstance(nov_p_fdr, dict) and key in nov_p_fdr:
            row['pFDR_Novices'] = format_p_cell(nov_p_fdr[key])
        df_rows.append(row)

    return pd.DataFrame(df_rows)


def save_latex_table(
    latex_code: str,
    output_path: Path
) -> Path:
    """Save LaTeX table code to file and return the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_code)
    return output_path


def format_roi_stats_table(
    welch_df: pd.DataFrame,
    exp_desc: List[Tuple[float, float, float]],
    nov_desc: List[Tuple[float, float, float]],
    roi_info: pd.DataFrame,
    subtract_chance: float = 0.0,
) -> pd.DataFrame:
    """
    Format ROI-level group statistics into a standardized table.

    Combines Welch t-test results with per-group descriptives (mean, CI) and
    ROI metadata (pretty names) into a publication-ready DataFrame with
    columns: ROI, Expert_mean, Expert_CI, Novice_mean, Novice_CI, Delta_mean,
    Delta_CI, p_raw, pFDR. Both raw and FDR-corrected p-values are included
    to satisfy the correlation reporting policy.

    Parameters
    ----------
    welch_df : pd.DataFrame
        Welch t-test results with columns: ROI_Label, mean_diff, ci95_low,
        ci95_high, p_val_fdr
    exp_desc : list of tuple
        Expert group descriptives: [(mean, ci_low, ci_high), ...] per ROI
    nov_desc : list of tuple
        Novice group descriptives: [(mean, ci_low, ci_high), ...] per ROI
    roi_info : pd.DataFrame
        ROI metadata with columns: roi_id, pretty_name (from load_roi_metadata)
    subtract_chance : float, default=0.0
        Value to subtract from means and CIs (e.g., for SVM decoding accuracy)

    Returns
    -------
    pd.DataFrame
        Formatted table with columns: ROI, Expert_mean, Expert_CI,
        Novice_mean, Novice_CI, Delta_mean, Delta_CI, pFDR

    Notes
    -----
    - Used by table generation scripts (81_*) across MVPA and supplementary analyses
    - Handles NaN values gracefully (displays as '--')
    - CI formatting: [low, high] with 3 decimal places
    - p-values in scientific notation (3 sig figs)

    Example
    -------
    >>> welch = per_roi_welch_and_fdr(expert_data, novice_data, roi_labels)
    >>> exp_desc = get_descriptives_per_roi(expert_data)
    >>> nov_desc = get_descriptives_per_roi(novice_data)
    >>> roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
    >>> df = format_roi_stats_table(welch, exp_desc, nov_desc, roi_info)
    """
    # Merge pretty names from ROI metadata
    df = welch_df.merge(
        roi_info[['roi_id', 'pretty_name']],
        left_on='ROI_Label',
        right_on='roi_id',
        how='left'
    )
    df['ROI'] = df['pretty_name'].map(shorten_roi_name)

    # Helper to format (mean, ci_low, ci_high) triplets
    def _fmt_triplet(t):
        m, lo, hi = t
        if pd.isna(m) or pd.isna(lo) or pd.isna(hi):
            return float('nan'), '{--}{--}'
        m_adj = float(m - subtract_chance)
        lo_adj = float(lo - subtract_chance)
        hi_adj = float(hi - subtract_chance)
        return m_adj, f"\\numrange{{{lo_adj:.3f}}}{{{hi_adj:.3f}}}"

    # Format expert and novice descriptives
    exp_vals, exp_cis = zip(*(_fmt_triplet(t) for t in exp_desc)) if exp_desc else ([], [])
    nov_vals, nov_cis = zip(*(_fmt_triplet(t) for t in nov_desc)) if nov_desc else ([], [])

    # Build output DataFrame (include both raw and FDR-corrected p-values)
    df_out = pd.DataFrame({
        'ROI': df['ROI'],
        'Expert_mean': list(exp_vals),
        'Expert_CI': list(exp_cis),
        'Novice_mean': list(nov_vals),
        'Novice_CI': list(nov_cis),
        'Delta_mean': df['mean_diff'].astype(float),
        'Delta_CI': pd.Series(zip(df['ci95_low'], df['ci95_high'])).map(
            lambda x: '{--}{--}' if any(pd.isna(list(x))) else f"\\numrange{{{float(x[0]):.3f}}}{{{float(x[1]):.3f}}}"
        ),
        'p_raw': df['p_val'].map(lambda p: format_p_cell(p) if pd.notna(p) else '--'),
        'pFDR': df['p_val_fdr'].map(lambda p: format_p_cell(p) if pd.notna(p) else '--'),
    })
    return df_out


def save_table_with_manuscript_copy(
    latex_table: str,
    output_path: Path,
    manuscript_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Save a LaTeX table to results directory and optionally copy to manuscript folder.

    This function saves the LaTeX table to the specified output path and, if
    MANUSCRIPT_TABLES_DIR is configured and exists, also copies it to the
    manuscript tables folder for automatic inclusion in the LaTeX manuscript.

    Parameters
    ----------
    latex_table : str
        The complete LaTeX table code to save
    output_path : Path
        Primary save location (in results/tables/)
    manuscript_name : str, optional
        Name for the manuscript copy (e.g., 'rsa_main_dims.tex').
        If None, no manuscript copy is created.
    logger : logging.Logger, optional
        Logger instance for logging progress

    Returns
    -------
    Path
        Path to the primary saved table file

    Notes
    -----
    - Always saves to output_path (results directory)
    - If manuscript_name is provided and MANUSCRIPT_TABLES_DIR exists, also
      copies to manuscript folder
    - If MANUSCRIPT_TABLES_DIR is configured but doesn't exist, logs a warning
      and skips manuscript copy

    Example
    -------
    >>> from common import CONFIG
    >>> latex_code = r"\\begin{table}...\\end{table}"
    >>> save_table_with_manuscript_copy(
    ...     latex_code,
    ...     Path("results/tables/mvpa_rsa_summary.tex"),
    ...     manuscript_name="rsa_main_dims.tex",
    ...     logger=logger
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate before saving (strict: raise on errors)
    errors = validate_latex_table(latex_table)
    if errors:
        msg = "LaTeX table validation failed:\n- " + "\n- ".join(errors)
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    # Save to primary location (results directory)
    with open(output_path, 'w') as f:
        f.write(latex_table)

    if logger:
        logger.info(f"LaTeX table saved to: {output_path}")

    # Copy to manuscript tables directory (configured consolidated location)
    from . import CONFIG
    manuscript_dir = CONFIG.get('MANUSCRIPT_TABLES_DIR')
    if (manuscript_dir is not None) and (manuscript_name is not None):
        manuscript_dir = Path(manuscript_dir)
        manuscript_dir.mkdir(parents=True, exist_ok=True)
        manuscript_path = manuscript_dir / manuscript_name
        with open(manuscript_path, 'w') as f:
            f.write(latex_table)
        if logger:
            logger.info(f"Table copied to manuscript: {manuscript_path}")

    return output_path


__all__ = [
    'create_figure_summary',
    'format_correlation_summary',
    'generate_latex_table',
    'create_correlation_table',
    'save_latex_table',
    'save_results_metadata',
    'format_roi_stats_table',
    'save_table_with_manuscript_copy',
    'validate_latex_table',
    'compile_latex_table',
]


def _add_multicolumn_headers(
    latex_table: str,
    column_names: List[str],
    multicolumn_groups: Dict[str, List[str]]
) -> str:
    """
    Add multi-level column headers to a LaTeX table.

    This internal helper function modifies a LaTeX table string to include
    \\multicolumn{n}{c}{Header} commands for grouped columns.

    Parameters
    ----------
    latex_table : str
        Original LaTeX table string (output from df.to_latex())
    column_names : list of str
        List of DataFrame column names
    multicolumn_groups : dict
        Dictionary mapping group names to lists of column names in that group

    Returns
    -------
    str
        Modified LaTeX table with multicolumn headers

    Notes
    -----
    - Inserts a new header row after \\toprule
    - Assumes first column (e.g., "Model") is not part of any multicolumn group
    - Uses \\cmidrule to separate groups visually

    Example
    -------
    Given columns: ["Model", "r_Experts", "p_Experts", "r_Novices", "p_Novices"]
    And multicolumn_groups: {"Experts": ["r_Experts", "p_Experts"], "Novices": ["r_Novices", "p_Novices"]}

    Generates header row:
    & \\multicolumn{2}{c}{Experts} & \\multicolumn{2}{c}{Novices} \\\\
    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}
    Model & r & p & r & p \\\\
    """
    # Find the position of \\toprule to insert the multicolumn row
    lines = latex_table.split('\n')
    toprule_idx = None
    header_idx = None

    for i, line in enumerate(lines):
        if '\\toprule' in line:
            toprule_idx = i
        if toprule_idx is not None and i > toprule_idx and '&' in line and header_idx is None:
            header_idx = i
            break

    if toprule_idx is None or header_idx is None:
        # Could not find the right position, return original table
        return latex_table

    # Build multicolumn header row
    multicolumn_row_parts = []
    cmidrule_parts = []

    # Determine which columns are in which group
    col_to_group = {}
    for group_name, cols in multicolumn_groups.items():
        for col in cols:
            col_to_group[col] = group_name

    # Track current position in column list
    current_col_idx = 0
    processed_groups = set()

    # First column (e.g., "ROI") is usually standalone; optionally multirow
    first_col_name = column_names[0]
    do_multirow = first_col_name.strip() == 'ROI'
    if first_col_name not in col_to_group:
        if do_multirow:
            multicolumn_row_parts.append(rf"\\multirow{{2}}{{*}}{{{first_col_name}}}")
        else:
            multicolumn_row_parts.append("")  # Empty for first column
        current_col_idx = 1
    else:
        current_col_idx = 0

    # Process remaining columns
    while current_col_idx < len(column_names):
        col = column_names[current_col_idx]

        # Check if this column is part of a group
        if col in col_to_group:
            group_name = col_to_group[col]
            group_cols = multicolumn_groups[group_name]

            # Only add multicolumn header once per group
            if group_name not in processed_groups:
                n_cols = len(group_cols)
                multicolumn_row_parts.append(f"\\multicolumn{{{n_cols}}}{{c}}{{{group_name}}}")

                # Add cmidrule for this group
                start_idx = current_col_idx + 1  # +1 because LaTeX column indices start at 1
                end_idx = start_idx + n_cols - 1
                cmidrule_parts.append(f"\\cmidrule(lr){{{start_idx}-{end_idx}}}")

                processed_groups.add(group_name)
                current_col_idx += n_cols
            else:
                current_col_idx += 1
        else:
            # Standalone column (not in any group)
            multicolumn_row_parts.append("")
            current_col_idx += 1

    # Construct the multicolumn header row
    multicolumn_row = " & ".join(multicolumn_row_parts) + " \\\\"
    cmidrule_row = " ".join(cmidrule_parts)

    # Insert the multicolumn row after \\toprule
    lines.insert(toprule_idx + 1, multicolumn_row)
    lines.insert(toprule_idx + 2, cmidrule_row)

    # Modify the original header row to use short, styled column names
    # (remove group suffix and apply math/escaping conventions)
    original_header = lines[header_idx + 2]  # +2 because we inserted 2 lines
    header_parts = original_header.split('&')

    new_header_parts = []
    for i, part in enumerate(header_parts):
        col_name = column_names[i].strip() if i < len(column_names) else part.strip()

        # Extract short base name (remove group suffix after the last underscore)
        base = col_name.split('_')[0] if '_' in col_name else col_name

        # Map to standardized LaTeX-friendly labels
        def stylize(token: str) -> str:
            t = token.strip()
            # Escape percent
            t = t.replace('%', r'\%')
            # Specific header tokens → LaTeX math/text
            mapping = {
                'pFDR': r'$p_\mathrm{FDR}$',
                'p': r'$p$',
                't': r'$t$',
                'M_diff': r'$M_{\text{diff}}$',
                'r': r'$r$',
            }
            if t in mapping:
                return mapping[t]
            # Delta-prefixed tokens (e.g., 'Δr', 'Δacc', 'Δ')
            if t.startswith('Δ'):
                rest = t[1:]
                # For 'Δr' prefer lower-case delta and math r
                if rest == 'r':
                    return r'$\delta r$'
                elif rest:
                    # Lower-case delta followed by base token (non-math)
                    return rf'$\delta\,{rest}$'
                else:
                    return r'$\delta$'
            return t

        # For ROI multirow, leave the first cell of the second header row empty
        if i == 0 and first_col_name.strip() == 'ROI':
            new_header_parts.append("")
            continue
        short_name = stylize(base)
        new_header_parts.append(short_name)

    new_header_row = " & ".join(new_header_parts) + " \\\\"
    lines[header_idx + 2] = new_header_row

    return '\n'.join(lines)


def _wrap_tabular_with_resizebox(latex_table: str) -> str:
    """Wrap the tabular environment with a resizebox to match page width."""
    lines = latex_table.split('\n')
    # Find begin/end tabular
    begin_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith('\\begin{tabular}')), None)
    end_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith('\\end{tabular}')), None)
    if begin_idx is None or end_idx is None:
        return latex_table
    lines.insert(begin_idx, r"\resizebox{\linewidth}{!}{%")
    lines.insert(end_idx + 2, r"}")
    return '\n'.join(lines)


def _escape_header_percents(latex_table: str) -> str:
    """Escape % signs in the header row when not using multicolumn headers."""
    lines = latex_table.split('\n')
    toprule_idx = None
    header_idx = None
    for i, line in enumerate(lines):
        if '\\toprule' in line:
            toprule_idx = i
        if toprule_idx is not None and i > toprule_idx and '&' in line and header_idx is None:
            header_idx = i
            break
    if header_idx is None:
        return latex_table
    # Escape % outside math in the header row
    row = lines[header_idx]
    out = []
    in_math = False
    for ch in row:
        if ch == '$':
            in_math = not in_math
            out.append(ch)
        elif ch == '%' and not in_math:
            out.append(r'\%')
        else:
            out.append(ch)
    lines[header_idx] = ''.join(out)
    return '\n'.join(lines)

def validate_latex_table(latex_table: str) -> List[str]:
    """
    Validate a LaTeX table string for common issues that break compilation.

    Checks:
    - Presence of a single tabular environment with column spec
    - Column count consistency (rows have expected number of & separators)
    - No unescaped % characters inside tabular content

    Returns a list of error strings (empty if valid).
    """
    errors: List[str] = []
    lines = latex_table.split('\n')
    # Find begin/end tabular and extract colspec
    begin_idx = next((i for i, ln in enumerate(lines) if '\\begin{tabular}' in ln), None)
    end_idx = next((i for i, ln in enumerate(lines) if '\\end{tabular}' in ln), None)
    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        errors.append("Missing or malformed tabular environment")
        return errors
    import re
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}", lines[begin_idx])
    if not m:
        errors.append("Could not parse tabular column specification")
        return errors
    colspec = m.group(1)
    tokens = [ch for ch in colspec if ch.strip() and ch != '|']
    ncols = len(tokens)
    # Determine start of body (after last \midrule if present)
    midrules = [i for i, ln in enumerate(lines) if '\\midrule' in ln]
    body_start = (midrules[-1] + 1) if midrules else (begin_idx + 1)
    content_lines = lines[body_start:end_idx]

    # Also check header region for unescaped % characters
    header_lines = lines[begin_idx:end_idx]
    for ln in header_lines:
        # Skip empty and structural lines
        s = ln.strip()
        if not s or any(tag in s for tag in ('\\begin{tabular}', '\\end{tabular}', '\\toprule', '\\midrule', '\\bottomrule', '\\cmidrule')):
            continue
        # Detect unescaped % in header region
        for j, ch in enumerate(ln):
            if ch == '%':
                if j == 0 or ln[j-1] != '\\':
                    errors.append("Unescaped % in header: '" + ln.strip() + "'")
                    break

    def _is_data_row(ln: str) -> bool:
        ln_stripped = ln.strip()
        if not ln_stripped:
            return False
        if any(tag in ln_stripped for tag in ('\\toprule', '\\midrule', '\\bottomrule', '\\cmidrule')):
            return False
        return ln_stripped.endswith('\\')

    for ln in content_lines:
        # Error on unescaped % inside tabular content
        for j, ch in enumerate(ln):
            if ch == '%':
                if j == 0 or ln[j-1] != '\\':
                    errors.append("Unescaped % in tabular content: '" + ln.strip() + "'")
                    break
        if not _is_data_row(ln):
            continue
        # Skip column count check for multicolumn rows
        if '\\multicolumn' in ln:
            continue
        ampersands = ln.count('&')
        if ampersands != ncols - 1:
            errors.append(
                f"Row has {ampersands+1} columns but tabular spec defines {ncols}: '{ln.strip()}'"
            )
    return errors


def compile_latex_table(
    latex_table: str,
    *,
    engine: str = 'pdflatex',
    work_dir: Optional[Path] = None,
    timeout_s: int = 20,
) -> Tuple[bool, str]:
    """
    Attempt to compile a LaTeX table into a PDF using a minimal document.

    Parameters
    ----------
    latex_table : str
        The LaTeX table code (including the table environment) to compile.
    engine : str, default 'pdflatex'
        LaTeX engine to use ('pdflatex' recommended for compatibility).
    work_dir : Path, optional
        Directory to write temporary files; if None, uses a temp directory.
    timeout_s : int, default 20
        Maximum seconds to allow the compilation process to run.

    Returns
    -------
    (ok, log) : Tuple[bool, str]
        ok is True when compilation succeeds (exit code 0), False otherwise.
        log contains stdout/stderr or diagnostic message if engine not found.

    Notes
    -----
    - This is optional; call when a LaTeX engine is available in the environment.
    - No network access is required; it shells out to the local LaTeX engine.
    - Strict: no silent fallbacks — failure returns False with full log.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which(engine) is None:
        return False, f"LaTeX engine '{engine}' not found in PATH"

    # Minimal wrapper document
    doc = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{multirow}\n"
        "\\usepackage{siunitx}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{graphicx}\n"
        "\\begin{document}\n"
        + latex_table +
        "\n\\end{document}\n"
    )

    # Prepare working directory
    cleanup = False
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory()
        work_dir = Path(tmp.name)
        cleanup = True
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    tex_path = work_dir / 'table_test.tex'
    tex_path.write_text(doc, encoding='utf-8')

    cmd = [engine, '-interaction=nonstopmode', '-halt-on-error', 'table_test.tex']
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
            text=True,
        )
        ok = (proc.returncode == 0)
        log = proc.stdout
    except subprocess.TimeoutExpired as e:
        ok = False
        log = f"LaTeX compilation timed out after {timeout_s}s\n{e}"

    # Cleanup temp directory if we created it
    if cleanup:
        try:
            tmp.cleanup()
        except Exception:
            pass

    return ok, log


def save_results_metadata(
    results_dir: Path,
    analysis_name: str,
    parameters: Dict,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Save analysis metadata (parameters, timestamp, etc.) to results directory.

    Parameters
    ----------
    results_dir : Path
        Results directory
    analysis_name : str
        Name of the analysis
    parameters : dict
        Dictionary of analysis parameters (e.g., random seed, paths, etc.)
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    Path
        Path to the saved metadata file

    Example
    -------
    >>> params = {
    ...     "RANDOM_SEED": 42,
    ...     "BIDS_ROOT": "/data/BIDS",
    ...     "MODEL_COLUMNS": ["check", "strategy", "visual"]
    ... }
    >>> save_results_metadata(results_dir, "Behavioral RSA", params, logger)
    """
    results_dir = Path(results_dir)
    metadata_path = results_dir / "analysis_metadata.txt"

    metadata_text = f"""
{analysis_name.upper()} - ANALYSIS METADATA
{'=' * 80}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results directory: {results_dir.name}

Parameters:
"""

    for key, value in parameters.items():
        metadata_text += f"  {key}: {value}\n"

    metadata_text += "\n" + "=" * 80 + "\n"

    with open(metadata_path, 'w') as f:
        f.write(metadata_text)

    if logger:
        logger.info(f"Metadata saved to: {metadata_path}")

    return metadata_path
def _stylize_and_escape_simple_header(latex_table: str) -> str:
    """Stylize math tokens (p, pFDR, t, M_diff, r, Δ*) and escape % in a simple header row.

    Applies when no multicolumn headers are used. Rewrites the header line following
    \toprule, converting known tokens to LaTeX math and escaping percent signs.
    """
    lines = latex_table.split('\n')
    toprule_idx = None
    header_idx = None
    for i, line in enumerate(lines):
        if '\\toprule' in line:
            toprule_idx = i
        if toprule_idx is not None and i > toprule_idx and '&' in line and header_idx is None:
            header_idx = i
            break
    if header_idx is None:
        return latex_table

    row = lines[header_idx]
    # Remove trailing \\ for processing, will add back later
    has_trail = row.rstrip().endswith('\\')
    row_clean = row.rstrip()
    if has_trail:
        row_clean = row_clean[:-2].rstrip()

    cells = [c.strip() for c in row_clean.split('&')]

    def style_cell(cell: str) -> str:
        # Escape % outside of math (these headers are text by default)
        cell_esc = cell.replace('%', r'\%')
        t = cell_esc
        # Exact token mappings
        if t == 'pFDR':
            return '$p_{FDR}$'
        if t == 'p':
            return '$p$'
        if t == 't':
            return '$t$'
        if t == 'M_diff':
            return '$M_{diff}$'
        if t == 'r':
            return '$r$'
        # Delta-prefixed
        if t.startswith('Δ'):
            rest = t[1:]
            if rest == 'r':
                return '$\\Delta r$'
            return f'$\\Delta\\,{rest}$' if rest else '$\\Delta$'
        return t

    new_cells = [style_cell(c) for c in cells]
    new_row = ' & '.join(new_cells)
    if has_trail:
        new_row += ' \\\\'
    lines[header_idx] = new_row
    return '\n'.join(lines)
