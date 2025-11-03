"""
Eyetracking Decoding — LaTeX Table Generation (Summary Metrics)
================================================================

This script generates a publication-ready LaTeX table summarizing machine
learning classification performance for decoding chess expertise from
eye-tracking features.

METHODS (Academic Manuscript Section)
--------------------------------------
Summary tables were generated from eyetracking-based expertise classification
results. Using a support vector machine (SVM) classifier with linear kernel and
L2 regularization, we trained binary classifiers to discriminate expert from
novice chess players based on eye-tracking features extracted during the chess
evaluation task. Classification was performed using leave-one-subject-out
cross-validation (LOSO-CV) to ensure independence between training and test sets.

Two feature sets were evaluated:
1. **Displacement features**: Temporal dynamics of eye movements (velocity,
   acceleration, displacement magnitude)
2. **XY position features**: Spatial patterns of gaze locations on the board

For each feature set, we report:

1. **Accuracy (CV mean, t-test)**: Mean classification accuracy across LOSO-CV
   folds with 95% CI (t distribution). Statistical significance assessed via a
   one-sample t-test vs chance level (0.5).

2. **Accuracy (pooled, binomial)**: Accuracy computed by pooling all out-of-fold
   predictions; statistical significance assessed via exact binomial test vs p0=0.5.
   95% CI reported using Wilson method.

2. **Balanced accuracy**: Arithmetic mean of sensitivity (true positive rate)
   and specificity (true negative rate), providing a metric robust to class
   imbalance.

3. **F1 score**: Harmonic mean of precision and recall, quantifying the balance
   between false positives and false negatives.

4. **ROC AUC**: Area under the receiver operating characteristic curve,
   measuring the classifier's ability to discriminate between classes across
   all decision thresholds (0.5 = chance, 1.0 = perfect discrimination).

These metrics collectively characterize the classifier's performance and
generalization ability, demonstrating whether eye-tracking patterns reliably
differentiate expertise levels.

Inputs
------
- results_displacement.json: JSON file with displacement feature results
- results_xy.json: JSON file with XY position feature results
  Each containing at minimum:
  - 'mean_accuracy': Mean accuracy across CV folds
  - 'ci_low', 'ci_high': 95% CI bounds for CV-mean accuracy (t CI)
  - 'p_value' or 'p_value_ttest': One-sample t-test p-value (H₀: accuracy = 0.5)
  - 'balanced_accuracy': Mean balanced accuracy
  - 'f1_score': Mean F1 score
  - 'roc_auc': Mean ROC AUC
  Optionally (recommended; computed by 01_eye_decoding.py):
  - 'pooled_accuracy': Accuracy pooled across all out-of-fold predictions
  - 'pooled_ci_low', 'pooled_ci_high': 95% CI (Wilson) for pooled accuracy
  - 'p_value_binomial': Exact binomial p-value vs p0=0.5

Outputs
-------
- tables/eyetracking_decoding_summary.tex: LaTeX table with all metrics
- tables/eyetracking_decoding_summary.csv: CSV version for reference

Dependencies
------------
- common.report_utils: generate_latex_table for LaTeX formatting
- common.logging_utils: Logging setup
- common.io_utils: Results directory finder

Usage
-----
python chess-supplementary/eyetracking/81_table_eyetracking_decoding.py

Supplementary Analysis: Eyetracking-based expertise classification
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd

# Ensure repo root is on sys.path for 'common' imports
_cur = os.path.dirname(__file__)
for _up in (os.path.join(_cur, '..'), os.path.join(_cur, '..', '..')):
    _cand = os.path.abspath(_up)
    if os.path.isdir(os.path.join(_cand, 'common')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break
from common import setup_script, log_script_end
from common.tables import generate_styled_table
from common.formatters import format_p_cell, format_ci
from common.stats_utils import binomial_test_from_predictions

# ============================================================================
# Configuration & Setup
# ============================================================================

# Locate the latest eyetracking decoding results directory
# This script reads from the most recent timestamped results folder
results_dir, logger, dirs = setup_script(
    __file__,
    results_pattern='eyetracking_decoding',
    output_subdirs=['tables'],
    log_name='tables_eyetracking.log',
)
RESULTS_DIR = results_dir

# Output directory for tables (created by find_latest_results_directory)
tables_dir = RESULTS_DIR / 'tables'

# ============================================================================
# Helper Function: Build Metric Rows
# ============================================================================

def build_metric_rows(res: dict, feature_label: str) -> list:
    """
    Build table rows for a single feature set.

    Parameters
    ----------
    res : dict
        Results dictionary from JSON file
    feature_label : str
        Label for the feature set (e.g., 'Displacement', 'XY Position')

    Returns
    -------
    list
        List of row dictionaries for DataFrame
    """
    rows = []
    # Accuracy (CV mean, t-test)
    p_t = res.get('p_value_ttest', res.get('p_value', float('nan')))
    rows.append(dict(
        Feature=feature_label,
        Metric='Accuracy (CV mean, t-test)',
        Estimate=float(res['mean_accuracy']),
        CI=format_ci(float(res['ci_low']), float(res['ci_high']), precision=3, latex=False),
        p_value=format_p_cell(p_t),
    ))

    # Accuracy (pooled, binomial)
    if all(k in res for k in ('pooled_accuracy', 'pooled_ci_low', 'pooled_ci_high', 'p_value_binomial')):
        pooled_acc = res['pooled_accuracy']
        pooled_lo = res['pooled_ci_low']
        pooled_hi = res['pooled_ci_high']
        p_bin = res['p_value_binomial']
    else:
        # Fallback: compute from y_true/y_pred if present
        if 'y_true' in res and 'y_pred' in res:
            # Not silent: log that we are computing binomial test from raw predictions
            # (requires external logger in calling scope, else no-op)
            try:
                import logging
                logging.getLogger(__name__).info(
                    "Binomial fields missing; computing from y_true/y_pred (Wilson CI)."
                )
            except Exception:
                pass
            pooled_acc, pooled_lo, pooled_hi, p_bin, _, _ = binomial_test_from_predictions(
                res['y_true'], res['y_pred'], p_null=0.5, alternative='two-sided', confidence_level=0.95, ci_method='wilson'
            )
        else:
            pooled_acc = float('nan'); pooled_lo = float('nan'); pooled_hi = float('nan'); p_bin = float('nan')
    rows.append(dict(
        Feature=feature_label,
        Metric='Accuracy (pooled, binomial)',
        Estimate=float(pooled_acc) if pooled_acc == pooled_acc else float('nan'),
        CI=(format_ci(float(pooled_lo), float(pooled_hi), precision=3, latex=False)
            if pooled_lo == pooled_lo and pooled_hi == pooled_hi else ''),
        p_value=(format_p_cell(p_bin) if p_bin == p_bin else ''),
    ))

    return [
        *rows,
        # Balanced accuracy (no CI or p-value)
        dict(
            Feature=feature_label,
            Metric='Balanced Accuracy',
            Estimate=float(res['balanced_accuracy']),
            CI='',
            p_value='',
        ),
        # F1 score (no CI or p-value)
        dict(
            Feature=feature_label,
            Metric='F1',
            Estimate=float(res['f1_score']),
            CI='',
            p_value='',
        ),
        # ROC AUC (no CI or p-value)
        dict(
            Feature=feature_label,
            Metric='ROC AUC',
            Estimate=float(res['roc_auc']),
            CI='',
            p_value='',
        ),
    ]

# ============================================================================
# Load Decoding Results
# ============================================================================

logger.info("Loading eyetracking decoding results from JSON files...")

# Load displacement feature results
with open(RESULTS_DIR / 'results_displacement.json', 'r') as fh:
    res_displacement = json.load(fh)
logger.info("Loaded displacement feature results")

# Load XY position feature results
with open(RESULTS_DIR / 'results_xy.json', 'r') as fh:
    res_xy = json.load(fh)
logger.info("Loaded XY position feature results")

# ============================================================================
# Build Summary Table DataFrame
# ============================================================================

logger.info("Building eyetracking decoding summary table...")

# Combine results from both feature sets
all_rows = []
all_rows.extend(build_metric_rows(res_displacement, 'Displacement'))
all_rows.extend(build_metric_rows(res_xy, 'XY Position'))

df = pd.DataFrame(all_rows)

logger.info(f"Created summary table with {len(df)} rows ({len(df)//2} feature sets)")

# ============================================================================
# Generate LaTeX Table
# ============================================================================

logger.info("Generating LaTeX table...")

# Save CSV version for reference
csv_path = tables_dir / 'eyetracking_decoding_summary.csv'
df.to_csv(csv_path, index=False)
logger.info(f"Saved CSV table: {csv_path}")

# Generate LaTeX table with all classification metrics
# column_format: left-aligned feature type, left-aligned metric names, three centered data columns
tex_path = generate_styled_table(
    df=df,  # Summary metrics DataFrame
    output_path=tables_dir / 'eyetracking_decoding_summary.tex',  # Output LaTeX file
    caption='Eyetracking decoding summary metrics for displacement and XY position features.',  # Table caption
    label='tab:eyetracking_decoding_summary',  # LaTeX label for references
    column_format='llScc',  # Two text cols, S for Estimate, then CI and p as text
    logger=logger,
    manuscript_name='et_decoding.tex',  # Copy to final_results/tables/
)

# ============================================================================
# Finish
# ============================================================================

logger.info(f"Saved eyetracking decoding LaTeX table: {tex_path}")
log_script_end(logger)
