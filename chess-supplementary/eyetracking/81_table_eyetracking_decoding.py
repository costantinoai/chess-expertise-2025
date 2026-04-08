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
evaluation task. Classification was performed using stratified group k-fold
cross-validation (k=20, grouped by subject) so that all runs from a held-out
participant remained out of sample.

Two feature sets were evaluated:
1. **Displacement features**: Distance from screen center at each timepoint
2. **XY position features**: Spatial patterns of gaze locations on the board

For each feature set, we report:

1. **Accuracy (subject mean, t-test)**: Mean subject-level out-of-fold accuracy
   with 95% CI (t distribution). Statistical significance assessed via a
   one-sample t-test vs chance level (0.5) across subjects.

2. **Accuracy (pooled runs, descriptive)**: Accuracy computed by pooling all
   out-of-fold run predictions. Because runs are nested within subjects, this
   quantity is reported descriptively only. A Wilson 95% CI is provided.

3. **Balanced accuracy**: Arithmetic mean of sensitivity (true positive rate)
   and specificity (true negative rate), providing a metric robust to class
   imbalance.

4. **F1 score**: Harmonic mean of precision and recall, quantifying the balance
   between false positives and false negatives.

5. **ROC AUC**: Area under the receiver operating characteristic curve,
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
  - 'mean_accuracy': Mean subject-level out-of-fold accuracy
  - 'ci_low', 'ci_high': 95% CI bounds for subject-mean accuracy (t CI)
  - 'p_value' or 'p_value_ttest': One-sample t-test p-value (H₀: accuracy = 0.5)
  - 'balanced_accuracy': Mean balanced accuracy
  - 'f1_score': Mean F1 score
  - 'roc_auc': Mean ROC AUC
  - 'pooled_accuracy': Accuracy pooled across all out-of-fold run predictions
  - 'pooled_ci_low', 'pooled_ci_high': 95% CI (Wilson) for pooled run accuracy

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

from pathlib import Path
import json
import pandas as pd

from common import setup_script, log_script_end
from common.tables import generate_styled_table
from common.formatters import format_p_cell, format_ci

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

# Output directory for tables (lives at results/<analysis>/tables/, not
# results/<analysis>/data/tables/ — setup_script already resolved it).
tables_dir = dirs['tables']

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
    # Accuracy (subject mean, t-test)
    p_t = res.get('p_value_ttest', res.get('p_value', float('nan')))
    rows.append(dict(
        Feature=feature_label,
        Metric='Accuracy (subject mean, t-test)',
        Estimate=float(res['mean_accuracy']),
        CI=format_ci(float(res['ci_low']), float(res['ci_high']), precision=3, latex=False, use_numrange=True),
        p_value=format_p_cell(p_t),
    ))

    # Accuracy (pooled runs, descriptive only)
    pooled_acc = res.get('pooled_accuracy', float('nan'))
    pooled_lo = res.get('pooled_ci_low', float('nan'))
    pooled_hi = res.get('pooled_ci_high', float('nan'))
    rows.append(dict(
        Feature=feature_label,
        Metric='Accuracy (pooled runs, descriptive)',
        Estimate=float(pooled_acc) if pooled_acc == pooled_acc else float('nan'),
        CI=(format_ci(float(pooled_lo), float(pooled_hi), precision=3, latex=False, use_numrange=True)
            if pooled_lo == pooled_lo and pooled_hi == pooled_hi else ''),
        p_value='',
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
    column_format='llccc',  # Two text cols, then three centered data columns
    logger=logger,
    manuscript_name='et_decoding.tex',  # Copy to final_results/tables/
)

# ============================================================================
# Finish
# ============================================================================

logger.info(f"Saved eyetracking decoding LaTeX table: {tex_path}")
log_script_end(logger)
