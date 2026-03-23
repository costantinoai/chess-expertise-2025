#!/usr/bin/env python3
"""
Compare run-matched control results with original (full-data) results.

Loads pre-computed group-level statistics from the original and run-matched
analyses and checks whether the pattern of significant results is preserved.
No statistics are recomputed here -- all tests use the exact same pipeline
as the original analyses (Welch t-tests, FDR correction).

Compares:
  1. PR: pr_statistical_tests.csv (Welch t-tests, FDR, Cohen's d per ROI)
  2. PR classification: pr_classification_tests.csv (permutation accuracy)
  3. RSA: experts_vs_novices CSV per target (Welch t-tests, FDR per ROI)

Usage:
    python compare_run_matched_vs_original.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from common import CONFIG

script_dir = Path(__file__).parent
repo_root = script_dir.parents[1]

# =============================================================================
# 1. PR comparison
# =============================================================================

print("=" * 70)
print("PARTICIPATION RATIO: Original vs Run-Matched")
print("=" * 70)

orig_pr_path = repo_root / "chess-manifold" / "results" / "manifold" / "pr_statistical_tests.csv"
matched_pr_path = script_dir / "results" / "manifold_run_matched" / "pr_statistical_tests.csv"

if orig_pr_path.exists() and matched_pr_path.exists():
    orig = pd.read_csv(orig_pr_path)
    matched = pd.read_csv(matched_pr_path)

    comp = orig[['ROI_Label', 't_stat', 'p_val_fdr', 'significant_fdr', 'cohen_d']].merge(
        matched[['ROI_Label', 't_stat', 'p_val_fdr', 'significant_fdr', 'cohen_d']],
        on='ROI_Label', suffixes=('_orig', '_matched'))

    print(f"\n{'ROI':>3} | {'t_orig':>7} | {'t_match':>7} | {'sig_o':>5} | {'sig_m':>5} | {'d_orig':>6} | {'d_match':>7}")
    print("-" * 70)
    for _, r in comp.iterrows():
        flag = " <<<" if r.significant_fdr_orig != r.significant_fdr_matched else ""
        print(f"{int(r.ROI_Label):>3} | {r.t_stat_orig:>7.2f} | {r.t_stat_matched:>7.2f} | "
              f"{str(r.significant_fdr_orig):>5} | {str(r.significant_fdr_matched):>5} | "
              f"{r.cohen_d_orig:>6.2f} | {r.cohen_d_matched:>7.2f}{flag}")

    n_sig_orig = comp['significant_fdr_orig'].sum()
    n_sig_matched = comp['significant_fdr_matched'].sum()
    n_changed = (comp['significant_fdr_orig'] != comp['significant_fdr_matched']).sum()
    print(f"\nFDR-significant ROIs: original={n_sig_orig}, run-matched={n_sig_matched}, changed={n_changed}")

    # Classification accuracy
    orig_cls_path = orig_pr_path.parent / "pr_classification_tests.csv"
    matched_cls_path = matched_pr_path.parent / "pr_classification_tests.csv"

    if orig_cls_path.exists() and matched_cls_path.exists():
        orig_cls = pd.read_csv(orig_cls_path)
        matched_cls = pd.read_csv(matched_cls_path)
        print("\nClassification accuracy:")
        for _, r in orig_cls.iterrows():
            mr = matched_cls[matched_cls['space'] == r['space']].iloc[0]
            print(f"  {r['space']}: orig={r['cv_accuracy_mean']:.3f} (p={r['perm_pvalue']:.4f}), "
                  f"matched={mr['cv_accuracy_mean']:.3f} (p={mr['perm_pvalue']:.4f})")
else:
    missing = []
    if not orig_pr_path.exists():
        missing.append(f"original: {orig_pr_path}")
    if not matched_pr_path.exists():
        missing.append(f"matched: {matched_pr_path}")
    print(f"PR results not found: {', '.join(missing)}")

# =============================================================================
# 2. RSA comparison
# =============================================================================

print("\n" + "=" * 70)
print("RSA: Original vs Run-Matched")
print("=" * 70)

# Pre-computed group stats CSVs from the original and run-matched pipelines.
# Both are produced by the same script (02_mvpa_group_rsa.py / group_rsa_run_matched.py)
# using identical statistical functions (Welch t-test, FDR correction).
orig_rsa_stats = repo_root / "chess-mvpa" / "results" / "mvpa_group"
matched_rsa_stats = script_dir / "results" / "mvpa_group_run_matched"

# The three main RSA dimensions reported in Fig 4
MAIN_TARGETS = ['checkmate', 'strategy', 'visual_similarity']

if orig_rsa_stats.exists() and matched_rsa_stats.exists():
    for target in MAIN_TARGETS:
        # Group comparison CSV: experts vs novices per ROI
        orig_csv = orig_rsa_stats / f"ttest_rsa_corr_{target}_experts_vs_novices.csv"
        matched_csv = matched_rsa_stats / f"ttest_rsa_corr_{target}_experts_vs_novices.csv"

        if not orig_csv.exists() or not matched_csv.exists():
            print(f"\n--- {target} --- SKIPPED (CSV not found)")
            continue

        orig_df = pd.read_csv(orig_csv)
        matched_df = pd.read_csv(matched_csv)

        # Merge on ROI name
        roi_col = orig_df.columns[0]  # First column is the ROI identifier
        comp = orig_df[[roi_col, 't_stat', 'p_val_fdr', 'significant_fdr']].merge(
            matched_df[[roi_col, 't_stat', 'p_val_fdr', 'significant_fdr']],
            on=roi_col, suffixes=('_orig', '_matched'))

        print(f"\n--- {target} ---")
        print(f"{'ROI':>25} | {'t_orig':>7} | {'t_match':>7} | {'sig_o':>5} | {'sig_m':>5}")
        print("-" * 60)
        for _, r in comp.iterrows():
            flag = " <<<" if r.significant_fdr_orig != r.significant_fdr_matched else ""
            print(f"{r[roi_col]:>25} | {r.t_stat_orig:>7.2f} | {r.t_stat_matched:>7.2f} | "
                  f"{str(r.significant_fdr_orig):>5} | {str(r.significant_fdr_matched):>5}{flag}")

        n_sig_o = comp['significant_fdr_orig'].sum()
        n_sig_m = comp['significant_fdr_matched'].sum()
        n_changed = (comp['significant_fdr_orig'] != comp['significant_fdr_matched']).sum()
        print(f"  FDR-significant ROIs: orig={n_sig_o}, matched={n_sig_m}, changed={n_changed}")
else:
    missing = []
    if not orig_rsa_stats.exists():
        missing.append(f"original: {orig_rsa_stats}")
    if not matched_rsa_stats.exists():
        missing.append(f"matched: {matched_rsa_stats}")
    print(f"RSA group stats not found: {', '.join(missing)}")
    print("Run group_rsa_run_matched.py after roi_mvpa_run_matched.m completes.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
