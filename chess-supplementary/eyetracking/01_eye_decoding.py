"""
Eyetracking decoding (experts vs novices) — separate analyses for xy and displacement features.

METHODS
=======

Overview
--------
This analysis tests whether eyetracking time-series features discriminate chess
experts from novices. Two feature sets are evaluated independently: (1) two-
dimensional gaze coordinates (x, y), and (2) displacement from screen center.
Features are constructed per run as fixed-length vectors by truncating to the
minimum number of timepoints across runs.

Data
----
- Eyetracking TSVs under BIDS derivatives:
  CONFIG['BIDS_EYETRACK']/sub-XX/func/*eyetrack.tsv
  with corresponding .json metadata files.
- Participants file: CONFIG['BIDS_PARTICIPANTS'] (for expert labels).
- N=40 participants (20 experts, 20 novices), 357 total runs.

Procedure
---------
1. Load all TSVs and metadata; add subject, run, and expert columns.
2. Run two independent SVM decoding analyses:
   a) Features = x_coordinate, y_coordinate (2D gaze position)
   b) Features = displacement (distance from screen center)
3. For each feature set:
   - Truncate runs to common length (minimum across runs) and flatten per-run vectors.
   - Train linear SVM in StratifiedGroupKFold CV (k=20 folds, group=subject).
   - Compute fold accuracies, mean accuracy, 95% CI via Student's t, and one-sample
     t-test vs chance=0.5.

Statistical Tests
-----------------
- One-sample t-test: Tests whether mean fold accuracy differs from chance (0.5).
- 95% Confidence Interval: Computed using Student's t-distribution on fold accuracies.

Outputs
-------
Saved under results/<ts>_eyetracking_decoding/:
- results_xy.json: metrics and predictions for xy features
- results_displacement.json: metrics and predictions for displacement features
- fold_accuracies_xy.csv: per-fold accuracies for xy
- fold_accuracies_displacement.csv: per-fold accuracies for displacement
- copies of this script and logs
"""

import os
import sys
from pathlib import Path
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
script_dir = Path(__file__).parent

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_curve, auc
from common.stats_utils import compute_mean_ci_and_ttest_vs_value

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end

from modules import load_eyetracking_dataframe, prepare_run_level_features


def run_svm_cv(X, y, groups, feature_label, logger, n_splits=20):
    """
    Run stratified group k-fold CV with linear SVM.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Labels (bool)
    groups : ndarray
        Group identifiers (subject IDs)
    feature_label : str
        Label for logging (e.g., 'xy', 'displacement')
    logger : Logger
        Logger instance
    n_splits : int, default=20
        Number of CV folds

    Returns
    -------
    dict
        Results dictionary with metrics and fold accuracies
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=CONFIG.get('RANDOM_SEED', 42))

    y_true, y_pred, y_prob, fold_acc = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups[test_idx]

        logger.info(f"  [{feature_label}] Fold {fold_idx:02d}: held-out subjects = {np.unique(groups_test)}")

        # Train SVM
        svm = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=CONFIG.get('RANDOM_SEED', 42)))
        svm.fit(X_train, y_train)

        # Predict
        fold_pred = svm.predict(X_test)
        fold_pred_proba = svm.predict_proba(X_test)[:, 1]
        fold_accuracy = accuracy_score(y_test, fold_pred)
        fold_acc.append(fold_accuracy)

        logger.info(f"  [{feature_label}] Fold {fold_idx:02d}: accuracy = {fold_accuracy:.3f}")

        y_true.extend(y_test)
        y_pred.extend(fold_pred)
        y_prob.extend(fold_pred_proba)

    # Convert to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    fold_acc = np.array(fold_acc)

    # Compute overall metrics
    mean_acc, ci_low, ci_high, t_stat, p_val = compute_mean_ci_and_ttest_vs_value(
        fold_acc, popmean=0.5, alternative='two-sided', confidence_level=0.95
    )
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    logger.info(f"[{feature_label}] Mean accuracy: {mean_acc:.3f}, 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    logger.info(f"[{feature_label}] t-test vs chance=0.5: t={t_stat:.3f}, p={p_val:.4f}")
    logger.info(f"[{feature_label}] Balanced accuracy: {balanced_acc:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}")

    return {
        'feature_type': feature_label,
        'n_folds': n_splits,
        'mean_accuracy': float(mean_acc),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'balanced_accuracy': float(balanced_acc),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'fold_accuracies': fold_acc.tolist(),
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob.tolist(),
    }


# Set up analysis
config, out_dir, logger = setup_analysis(
    analysis_name="eyetracking_decoding",
    results_base=script_dir / 'results',
    script_file=__file__,
)

# Load eyetracking long-format data from canonical derivatives folder
et_root = CONFIG['BIDS_EYETRACK']
df = load_eyetracking_dataframe(et_root)
if df.empty:
    raise FileNotFoundError(f"No eyetracking TSVs found under: {et_root}")

logger.info(f"Loaded eyetracking data: {len(df)} rows, {df['subject'].nunique()} subjects, {len(df.groupby(['subject','run']))} runs")

# ============================================================================
# Analysis 1: xy features (2D gaze coordinates)
# ============================================================================
logger.info("=" * 80)
logger.info("Analysis 1: 2D gaze coordinates (x, y)")
logger.info("=" * 80)

X_xy, y_xy, groups_xy = prepare_run_level_features(df, feature_type='xy')
logger.info(f"xy features: X={X_xy.shape}, n_runs={len(y_xy)}, n_subjects={len(np.unique(groups_xy))}")

results_xy = run_svm_cv(X_xy, y_xy, groups_xy, feature_label='xy', logger=logger, n_splits=20)

# Save xy results
pd.DataFrame({'fold_accuracy': results_xy['fold_accuracies']}).to_csv(out_dir / 'fold_accuracies_xy.csv', index=False)
with open(out_dir / 'results_xy.json', 'w') as f:
    json.dump(results_xy, f, indent=2)

# ============================================================================
# Analysis 2: displacement features (distance from center)
# ============================================================================
logger.info("=" * 80)
logger.info("Analysis 2: Displacement from screen center")
logger.info("=" * 80)

X_disp, y_disp, groups_disp = prepare_run_level_features(df, feature_type='displacement')
logger.info(f"displacement features: X={X_disp.shape}, n_runs={len(y_disp)}, n_subjects={len(np.unique(groups_disp))}")

results_disp = run_svm_cv(X_disp, y_disp, groups_disp, feature_label='displacement', logger=logger, n_splits=20)

# Save displacement results
pd.DataFrame({'fold_accuracy': results_disp['fold_accuracies']}).to_csv(out_dir / 'fold_accuracies_displacement.csv', index=False)
with open(out_dir / 'results_displacement.json', 'w') as f:
    json.dump(results_disp, f, indent=2)

logger.info("=" * 80)
logger.info("Summary:")
logger.info(f"  xy: accuracy={results_xy['mean_accuracy']:.3f}, p={results_xy['p_value']:.4f}")
logger.info(f"  displacement: accuracy={results_disp['mean_accuracy']:.3f}, p={results_disp['p_value']:.4f}")
logger.info("=" * 80)

log_script_end(logger)
