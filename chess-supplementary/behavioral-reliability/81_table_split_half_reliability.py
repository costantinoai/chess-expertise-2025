#!/usr/bin/env python3
"""
Split-Half Reliability — LaTeX Table Generation
================================================

This script generates a publication-ready LaTeX table summarizing the split-half
reliability of behavioral representational dissimilarity matrices (RDMs) for
expert and novice chess players.

METHODS (Academic Manuscript Section)
--------------------------------------
Summary tables were generated to assess the reliability of behavioral RDMs using
split-half correlation analysis. To evaluate the stability of representational
structure within and between expertise groups, we repeatedly split participants
into two random halves (1,000 iterations) and computed correlations between the
RDMs from each half.

For each expertise group and comparison type, we report:

1. **Uncorrected split-half correlation (r_half)**: Mean Spearman correlation
   between RDMs computed from two random halves of participants. This represents
   the reliability of the RDM based on half the sample size.

2. **Spearman-Brown corrected reliability (r_full)**: Reliability estimate
   corrected for the full sample size using the Spearman-Brown prophecy formula:
   r_full = (2 × r_half) / (1 + r_half). This provides an estimate of the
   reliability one would obtain with the complete participant sample.

3. **95% confidence intervals**: Computed from the bootstrap distribution of
   1,000 random splits, providing uncertainty estimates for the corrected
   reliability.

4. **Statistical significance**: Two-sided bootstrap sign-proportion tests
   (H₀: r = 0) testing whether the reliability significantly differs from zero.

Three types of comparisons were performed:

- **Within-group (Experts)**: Correlation between random halves of expert
  participants, assessing the stability of expert representational structure.

- **Within-group (Novices)**: Correlation between random halves of novice
  participants, assessing the stability of novice representational structure.

- **Between-group**: Correlation between a random half of experts and a random
  half of novices, quantifying the similarity of representational structure
  across expertise levels.

- **Group comparison**: Bootstrap test of the difference between expert and
  novice within-group reliability (Δ = r_full^Experts - r_full^Novices),
  testing whether one group shows more stable representations than the other.

Inputs
------
- reliability_metrics.pkl: Dictionary containing (from 01_behavioral_split_half_reliability.py):
  - 'experts_within': Within-group reliability for experts (mean_r_half, mean_r_full, ci_r_full, p_boot_full)
  - 'novices_within': Within-group reliability for novices
  - 'between_groups': Between-group reliability (experts vs novices)
  - 'experts_vs_novices_diff': Bootstrap test of group difference (mean_delta_full, ci_delta_full, p_boot_delta_full)

Outputs
-------
- tables/split_half_reliability.tex: LaTeX table with all reliability metrics

Dependencies
------------
- common.formatters: format_ci, format_pvalue_latex for value formatting
- common.logging_utils: Logging setup
- common.io_utils: Results directory finder

Usage
-----
python chess-supplementary/behavioral-reliability/81_table_split_half_reliability.py

Supplementary Analysis: Behavioral RDM reliability assessment
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
from common import setup_script, log_script_end
from common.report_utils import save_table_with_manuscript_copy
from common.formatters import format_ci, format_p_cell


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
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Extract Reliability Data
    # ========================================================================

    # Extract within-group and between-group reliability metrics
    exp_within = results['experts_within']  # Expert within-group reliability
    nov_within = results['novices_within']  # Novice within-group reliability
    between = results['between_groups']  # Between-group reliability
    diff = results['experts_vs_novices_diff']  # Group difference statistics

    # ========================================================================
    # Build LaTeX Table Structure
    # ========================================================================

    # Initialize table lines with LaTeX formatting
    lines = []
    # Begin table environment with full-page placement
    lines.append(r"\begin{table}[p]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{%")  # Resize to fit page width
    lines.append(r"\begin{tabular}{llSScc}")
    lines.append(r"\toprule")

    # Table header with column labels
    lines.append(r"\textbf{Group} & \textbf{Condition} & \textbf{$r_{\text{half}}$} & \textbf{$r_{\text{full}}$} & \textbf{95\% CI} & \textbf{$p$} \\")
    lines.append(r"\midrule")

    # ========================================================================
    # Within-Group Reliability: Experts
    # ========================================================================

    # Format expert within-group reliability metrics
    # r_half: uncorrected split-half correlation
    # r_full: Spearman-Brown corrected reliability
    # CI: 95% confidence interval for r_full
    # p: bootstrap p-value testing H0: r_full = 0
    ci_low, ci_high = exp_within['ci_r_full']
    lines.append(
        f"Experts & Within  & {exp_within['mean_r_half']:.3f}  & "
        f"{exp_within['mean_r_full']:.3f}  & "
        f"{format_ci(ci_low, ci_high, precision=3, latex=False)} & "
        f"{format_p_cell(exp_within['p_boot_full'])} \\\\"
    )

    # ========================================================================
    # Within-Group Reliability: Novices
    # ========================================================================

    # Format novice within-group reliability metrics
    ci_low, ci_high = nov_within['ci_r_full']
    lines.append(
        f"Novices & Within  & {nov_within['mean_r_half']:.3f}  & "
        f"{nov_within['mean_r_full']:.3f}  & "
        f"{format_ci(ci_low, ci_high, precision=3, latex=False)} & "
        f"{format_p_cell(nov_within['p_boot_full'])} \\\\"
    )

    # ========================================================================
    # Between-Group Reliability
    # ========================================================================

    # Format between-group reliability (experts vs novices)
    # This measures similarity of RDM structure across expertise levels
    ci_low, ci_high = between['ci_r_full']
    lines.append(
        f"Between & Experts vs Novices  & {between['mean_r_half']:.3f}  & "
        f"{between['mean_r_full']:.3f}  & "
        f"{format_ci(ci_low, ci_high, precision=3, latex=False)} & "
        f"{format_p_cell(between['p_boot_full'])} \\\\"
    )

    # ========================================================================
    # Group Comparison Section
    # ========================================================================

    # Add section header for group comparison
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textbf{Group Comparison (Bootstrap difference)}} \\")
    lines.append(r"\midrule")

    # Format group comparison: difference between expert and novice within-group reliability
    # Tests whether one group has more stable/reliable representations than the other
    # Δ = r_full(Experts) - r_full(Novices)
    ci_low, ci_high = diff['ci_delta_full']
    lines.append(
        r"\multicolumn{2}{l}{Within: Experts vs. Novices}  & "
        r"\textemdash & "  # No r_half for difference
        f"{diff['mean_delta_full']:.3f} & "  # Mean difference in r_full
        f"{format_ci(ci_low, ci_high, precision=3, latex=False)} & "  # 95% CI for difference
        f"{format_p_cell(diff['p_boot_delta_full'])} \\\\"  # Bootstrap p-value
    )

    # Close table structure
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")

    # ========================================================================
    # Table Caption
    # ========================================================================

    # Add comprehensive caption explaining all table elements
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

    # ========================================================================
    # Write LaTeX File
    # ========================================================================

    # Write all table lines to LaTeX file and copy to manuscript folder
    tex_path = output_dir / filename_tex
    latex_table = '\n'.join(lines)
    save_table_with_manuscript_copy(
        latex_table,
        tex_path,
        manuscript_name='bh_rdm_splithalf.tex',  # Copy to final_results/tables/
        logger=None  # No logger in this function context
    )

    return tex_path


# ============================================================================
# Configuration & Setup
# ============================================================================

# Locate the latest behavioral split-half reliability results directory
results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='behavioral_split_half',
    output_subdirs=['tables'],
    log_name='tables_split_half.log',
)
RESULTS_DIR = results_dir
tables_dir = dirs['tables']

# ============================================================================
# Load Reliability Results
# ============================================================================

logger.info("Loading split-half reliability metrics from pickle file...")

# Load the reliability_metrics.pkl file generated by 01_behavioral_split_half_reliability.py
# This file contains all reliability statistics from bootstrap analysis
with open(RESULTS_DIR / 'reliability_metrics.pkl', 'rb') as f:
    results = pickle.load(f)

# ============================================================================
# Generate LaTeX Table
# ============================================================================

logger.info("Generating split-half reliability LaTeX table...")

# Generate comprehensive reliability table with within-group, between-group,
# and group comparison statistics
tex_path = generate_split_half_table(
    results=results,  # Reliability metrics from bootstrap analysis
    output_dir=tables_dir,  # Output directory for table
    filename_tex='split_half_reliability.tex',  # LaTeX output filename
)

# ============================================================================
# Finish
# ============================================================================

logger.info(f"Saved split-half reliability LaTeX table: {tex_path}")
log_script_end(logger)
