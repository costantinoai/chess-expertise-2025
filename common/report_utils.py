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
from .formatters import format_pvalue_latex, format_ci as fmt_ci
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
    p_sci: bool = True,
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
        Columns: Model, r_Experts, 95%_CI_Experts, p_Experts,
                 r_Novices, 95%_CI_Novices, p_Novices
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

        row = {
            'Model': label,
            'r_Experts': f"{er:.3f}",
            '95%_CI_Experts': f"[{eci_l:.3f}, {eci_u:.3f}]" if ci_brackets else None,
            'p_Experts': f"{ep:.3e}" if p_sci else f"{ep:.3f}",
            'r_Novices': f"{nr:.3f}",
            '95%_CI_Novices': f"[{nci_l:.3f}, {nci_u:.3f}]" if ci_brackets else None,
            'p_Novices': f"{npv:.3e}" if p_sci else f"{npv:.3f}",
        }
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
    logger: Optional[logging.Logger] = None
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

    # Generate basic LaTeX table
    latex_table = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=escape,
        column_format=column_format
    )

    # Add multicolumn headers if provided
    if multicolumn_headers is not None:
        latex_table = _add_multicolumn_headers(
            latex_table, df.columns.tolist(), multicolumn_headers
        )

    # Save to file
    with open(output_path, 'w') as f:
        f.write(latex_table)

    if logger:
        logger.info(f"LaTeX table saved to: {output_path}")

    return output_path


def create_correlation_table(
    expert_results: List[Tuple[str, float, float, float, float]],
    novice_results: List[Tuple[str, float, float, float, float]],
    model_labels: Optional[Dict[str, str]] = None,
    caption: str = "Behavioral RDM correlations with model RDMs",
    label: str = "tab:behavioral_correlations"
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

    # Build rows using formatters
    df_rows = []
    for exp_res, nov_res in zip(expert_results, novice_results):
        key = exp_res[0]
        label_pretty = model_labels.get(key, key.capitalize()) if model_labels else key
        er, ep, ecl, ech = exp_res[1:]
        nr, npv, ncl, nch = nov_res[1:]
        df_rows.append({
            'Model': label_pretty,
            'r_Experts': f"{er:.3f}",
            '95%_CI_Experts': fmt_ci(ecl, ech, precision=3, latex=False),
            'p_Experts': f"{ep:.3e}",
            'r_Novices': f"{nr:.3f}",
            '95%_CI_Novices': fmt_ci(ncl, nch, precision=3, latex=False),
            'p_Novices': f"{npv:.3e}",
        })

    return pd.DataFrame(df_rows)


__all__ = [
    'create_figure_summary',
    'format_correlation_summary',
    'generate_latex_table',
    'create_correlation_table',
    'save_results_metadata',
]


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


__all__ = [
    'create_figure_summary',
    'format_correlation_summary',
    'generate_latex_table',
    'create_correlation_table',
    'save_latex_table',
    'save_results_metadata',
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

    # First column (e.g., "Model") is usually standalone
    first_col_name = column_names[0]
    if first_col_name not in col_to_group:
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

    # Modify the original header row to use short column names
    # (remove group prefix, e.g., "r_Experts" -> "r")
    original_header = lines[header_idx + 2]  # +2 because we inserted 2 lines
    header_parts = original_header.split('&')

    new_header_parts = []
    for i, part in enumerate(header_parts):
        col_name = column_names[i].strip() if i < len(column_names) else part.strip()

        # Shorten column name if it's part of a group
        if col_name in col_to_group:
            # Extract short name (remove group suffix)
            # E.g., "r_Experts" -> "r", "95%_CI_Experts" -> "95% CI"
            short_name = col_name.split('_')[0]  # Take first part before underscore
            new_header_parts.append(short_name)
        else:
            new_header_parts.append(part.strip())

    new_header_row = " & ".join(new_header_parts) + " \\\\"
    lines[header_idx + 2] = new_header_row

    return '\n'.join(lines)


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
