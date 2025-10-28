#!/usr/bin/env python3
"""
Neurosynth Plotting Script (from saved CSVs)

This script reloads correlation CSVs (positive/negative/difference) from the
most recent neurosynth results directories and regenerates the correlation
bar plots and difference plots with a canonical term order.

It does not recompute correlations—only plotting—and keeps analysis vs plotting
separation per CLAUDE.md.
"""

import sys
from pathlib import Path
import pandas as pd

# Import path management (per CLAUDE.md)
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))   # common/
sys.path.insert(0, str(script_dir))  # chess-neurosynth/modules

from common import CONFIG
from common.logging_utils import setup_analysis_in_dir
from common.io_utils import find_latest_results_directory

from modules.plot_utils import plot_correlations, plot_difference, plot_map, plot_surface_map_flat
from modules.tables import save_latex_correlation_tables, save_latex_combined_pos_neg_diff, generate_latex_multicolumn_table
from modules.io_utils import reorder_by_term
from nilearn import image


## Uses modules.io_utils.reorder_by_term (DRY)


def _find_csv_triples(results_dir: Path) -> list[tuple[str, Path, Path, Path]]:
    """Return list of (run_id, pos_csv, neg_csv, diff_csv) in results_dir."""
    triples = []
    # Identify files ending with patterns
    pos_files = sorted(results_dir.glob("*_term_corr_positive.csv"))
    for pos in pos_files:
        base = pos.name.replace("_term_corr_positive.csv", "")
        neg = results_dir / f"{base}_term_corr_negative.csv"
        diff = results_dir / f"{base}_term_corr_difference.csv"
        if neg.exists() and diff.exists():
            triples.append((base, pos, neg, diff))
    return triples


def _pretty_title(run_id: str, title_prefix: str) -> str:
    """Return a publication title for surface/brain maps.

    - For RSA (title_prefix starts with 'RSA'), converts searchlight_* to pretty labels.
    - Otherwise returns f"{title_prefix}: {run_id}".
    """
    if title_prefix.startswith('RSA') and run_id.startswith('searchlight_'):
        key = run_id.split('searchlight_')[-1].lower()
        mapping = {
            'checkmate': 'Checkmate',
            'strategy': 'Strategy',
            'visualsimilarity': 'Visual Similarity',
        }
        return mapping.get(key, run_id)
    return f"{title_prefix}: {run_id}"


def _process_analysis_dir(results_dir: Path, title_prefix: str, term_order: list[str]) -> tuple[dict, dict, dict]:
    """DRY helper: plot glass/flat maps, correlation bars, and write LaTeX tables."""
    extra = {"RESULTS_DIR": str(results_dir)}
    _, _, logger = setup_analysis_in_dir(results_dir, __file__, extra_config=extra, log_name='plotting.log')
    logger.info(f"Using results: {results_dir}")

    figures_dir = results_dir / 'figures'
    tables_dir = results_dir / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Maps
    for zmap in sorted(results_dir.glob('zmap_*.nii*')):
        run_id = zmap.stem.replace('zmap_', '')
        z_img = image.load_img(str(zmap))
        title_str = _pretty_title(run_id, title_prefix)
        plot_map(z_img.get_fdata(), z_img, title=title_str, outpath=figures_dir / f"{run_id}_glass.pdf", thresh=1e-5)
        plot_surface_map_flat(z_img, title=title_str, threshold=0, output_file=figures_dir / f"{run_id}_surface_flat.pdf")

    # Correlations and tables
    all_pos, all_neg, all_diff = {}, {}, {}
    for run_id, pos_csv, neg_csv, diff_csv in _find_csv_triples(results_dir):
        df_pos = pd.read_csv(pos_csv)
        df_neg = pd.read_csv(neg_csv)
        df_diff = pd.read_csv(diff_csv)

        df_pos = reorder_by_term(df_pos)
        df_neg = reorder_by_term(df_neg)
        df_diff = reorder_by_term(df_diff)

        plot_correlations(df_pos, df_neg, df_diff, run_id=run_id, out_fig=figures_dir / f"{run_id}_term_correlations.pdf")
        plot_difference(df_diff, run_id=run_id, out_fig=figures_dir / f"{run_id}_term_correlation_differences.pdf")
        save_latex_correlation_tables(df_pos, df_neg, df_diff, run_id=run_id, out_dir=tables_dir)
        save_latex_combined_pos_neg_diff(df_pos, df_neg, df_diff, run_id=run_id, out_path=tables_dir / f"{run_id}_combined.tex")

        all_pos[run_id] = df_pos
        all_neg[run_id] = df_neg
        all_diff[run_id] = df_diff

    return all_pos, all_neg, all_diff


# Run plotting (no CLI/if-main per CLAUDE.md)
term_order = CONFIG.get('NEUROSYNTH_TERM_ORDER', [])

# Univariate
try:
    uni_dir = find_latest_results_directory(script_dir / 'results', pattern='*_neurosynth_univariate')
    _process_analysis_dir(uni_dir, "Univariate", term_order)
except Exception as e:
    print(f"Univariate plotting skipped: {e}")

# RSA (and combined tables across patterns)
try:
    rsa_dir = find_latest_results_directory(script_dir / 'results', pattern='*_neurosynth_rsa')
    all_pos, all_neg, all_diff = _process_analysis_dir(rsa_dir, "RSA Searchlight", term_order)
    tables_dir = rsa_dir / 'tables'
    if all_diff:
        generate_latex_multicolumn_table(all_diff, tables_dir / 'rsa_searchlight_diff.tex', 'diff', 'RSA searchlight: Δr (pos − neg) vs term maps.', 'tab:rsa_searchlight_diff')
    if all_pos:
        generate_latex_multicolumn_table(all_pos, tables_dir / 'rsa_searchlight_pos.tex', 'pos', 'RSA searchlight: positive z-maps vs term maps.', 'tab:rsa_searchlight_pos')
    if all_neg:
        generate_latex_multicolumn_table(all_neg, tables_dir / 'rsa_searchlight_neg.tex', 'neg', 'RSA searchlight: negative z-maps vs term maps.', 'tab:rsa_searchlight_neg')
except Exception as e:
    print(f"RSA plotting skipped: {e}")
