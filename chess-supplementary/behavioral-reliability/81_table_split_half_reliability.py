#!/usr/bin/env python3
"""
Split-Half Reliability — LaTeX table generation

Loads reliability_metrics.pkl from the latest behavioral split-half analysis
and builds a LaTeX table showing within-group and between-group reliability
with corrected and uncorrected correlations, 95% CIs, and bootstrap p-values.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.formatters import format_ci, format_pvalue_latex


def generate_split_half_table(results, output_dir, filename_tex='split_half_reliability.tex'):
    """
    Generate LaTeX table for split-half reliability results.

    Parameters
    ----------
    results : dict
        Dictionary with 'experts_within', 'novices_within', 'between_groups',
        and 'experts_vs_novices_diff' keys from 01_behavioral_split_half_reliability.py
    output_dir : Path
        Directory to save table
    filename_tex : str
        Output filename for LaTeX table

    Returns
    -------
    tex_path : Path
        Path to saved LaTeX file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    exp_within = results['experts_within']
    nov_within = results['novices_within']
    between = results['between_groups']
    diff = results['experts_vs_novices_diff']

    # Build table rows
    lines = []
    lines.append(r"\begin{table}[p]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Group} & \textbf{Condition} & \textbf{$r_{\text{half}}$} & \textbf{$r_{\text{full}}$} & \textbf{95\% CI} & \textbf{$p$} \\")
    lines.append(r"\midrule")

    # Experts - Within
    ci_low, ci_high = exp_within['ci_r_full']
    lines.append(
        f"Experts & Within  & {exp_within['mean_r_half']:.3f}  & "
        f"{exp_within['mean_r_full']:.3f}  & "
        f"{format_ci(ci_low, ci_high, precision=3, latex=False)} & "
        f"{format_pvalue_latex(exp_within['p_boot_full'], threshold=0.001)} \\\\"
    )

    # Novices - Within
    ci_low, ci_high = nov_within['ci_r_full']
    lines.append(
        f"Novices & Within  & {nov_within['mean_r_half']:.3f}  & "
        f"{nov_within['mean_r_full']:.3f}  & "
        f"{format_ci(ci_low, ci_high, precision=3, latex=False)} & "
        f"{format_pvalue_latex(nov_within['p_boot_full'], threshold=0.001)} \\\\"
    )

    # Between-groups
    ci_low, ci_high = between['ci_r_full']
    lines.append(
        f"Between & Experts vs Novices  & {between['mean_r_half']:.3f}  & "
        f"{between['mean_r_full']:.3f}  & "
        f"{format_ci(ci_low, ci_high, precision=3, latex=False)} & "
        f"{format_pvalue_latex(between['p_boot_full'], threshold=0.001)} \\\\"
    )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textbf{Group Comparison (Bootstrap difference)}} \\")
    lines.append(r"\midrule")

    # Group comparison: Experts vs Novices (difference in r_full)
    ci_low, ci_high = diff['ci_delta_full']
    lines.append(
        r"\multicolumn{2}{l}{Within: Experts vs. Novices}  & "
        r"\textemdash & "
        f"{diff['mean_delta_full']:.3f} & "
        f"{format_ci(ci_low, ci_high, precision=3, latex=False)} & "
        f"{format_pvalue_latex(diff['p_boot_delta_full'], threshold=0.001)} \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\caption{Split-half reliability of behavioral RDMs for Experts and Novices. "
                 r"$r_{\text{half}}$ = uncorrected split-half correlation; "
                 r"$r_{\text{full}}$ = Spearman-Brown corrected reliability. "
                 r"Within-group reliability: correlation between two random halves of participants within each group. "
                 r"Between-group: correlation between random half of Experts and random half of Novices. "
                 r"All correlations computed across 1,000 random splits. "
                 r"95\% CI and $p$-values reported for $r_{\text{full}}$ only. "
                 r"$p$-values are two-sided bootstrap sign-proportion tests (H$_0$: 0). "
                 r"Group comparison uses the bootstrap distribution of $\Delta = r_{\text{full}}^{E} - r_{\text{full}}^{N}$.}")
    lines.append(r"\label{supptab:bh_splithalf}")
    lines.append(r"\end{table}")

    # Write to file
    tex_path = output_dir / filename_tex
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))

    return tex_path


# ============================================================================
# Main
# ============================================================================

RESULTS_BASE = Path(__file__).parent / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_behavioral_split_half',
    create_subdirs=['tables'],
    require_exists=True,
    verbose=True,
)

extra = {"RESULTS_DIR": str(RESULTS_DIR)}
config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name='tables_split_half.log',
)

tables_dir = RESULTS_DIR / 'tables'

# Load results
with open(RESULTS_DIR / 'reliability_metrics.pkl', 'rb') as f:
    results = pickle.load(f)

# Generate table
tex_path = generate_split_half_table(
    results=results,
    output_dir=tables_dir,
    filename_tex='split_half_reliability.tex',
)

logger.info(f"Saved split-half reliability LaTeX table: {tex_path}")
log_script_end(logger)
