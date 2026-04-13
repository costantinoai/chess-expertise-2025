"""
Eyetracking decoding (experts vs novices) -- group-level stage.

Reads per-subject accuracy CSVs and full results JSONs from the BIDS
derivatives tree (produced by ``01_eye_decoding_subject.py``), computes
group-level statistics, and writes GROUP-LEVEL ONLY outputs to

    results/eyetracking/data/

This script writes only group-level aggregates to the results/ tree.
Per-subject data remains in BIDS/derivatives/ (written by the
subject-level script 01_eye_decoding_subject.py).

Outputs
-------
Saved under results/eyetracking/data/:
- results_xy.json: group-level metrics for XY features (no subject data)
- results_displacement.json: group-level metrics for displacement features (no subject data)
- fold_accuracies_xy.csv: per-fold accuracies for XY
- fold_accuracies_displacement.csv: per-fold accuracies for displacement

Group-level JSON keys (per feature set):
- feature_type, n_folds, n_subjects, inference_unit
- mean_accuracy, ci_low, ci_high, t_statistic, p_value, p_value_ttest
- fold_mean_accuracy, fold_ci_low, fold_ci_high, fold_accuracies
- pooled_accuracy, pooled_ci_low, pooled_ci_high, n_correct, n_total
- balanced_accuracy, f1_score, roc_auc
- anonymous_subject_accuracies: list of {group, accuracy} dicts (no subject IDs)

Stripped from group JSON (remain in derivatives only):
- subject_accuracy_records, subject_accuracies, y_true, y_pred, y_prob, subject_ids
"""

from pathlib import Path
import json

import pandas as pd

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end


# Keys that contain per-subject or per-run data and must not appear in
# the group-level results JSON (GDPR: no subject data in results/).
_SUBJECT_LEVEL_KEYS = {
    'subject_accuracy_records',
    'subject_accuracies',
    'y_true',
    'y_pred',
    'y_prob',
    'subject_ids',
}


def load_and_strip(deriv_root: Path, feature_label: str, logger):
    """
    Load full results JSON from derivatives, validate against per-subject
    accuracy CSV, and return a group-level-only dict.

    Parameters
    ----------
    deriv_root : Path
        BIDS derivatives directory for eyetracking decoding.
    feature_label : str
        Feature set identifier ('xy' or 'displacement').
    logger : Logger
        Logger instance.

    Returns
    -------
    dict
        Group-level results dictionary (subject-level keys removed).
    """
    json_path = deriv_root / f'results_{feature_label}.json'
    csv_path = deriv_root / f'subject_accuracies_{feature_label}.csv'

    if not json_path.is_file():
        raise FileNotFoundError(
            f"Missing derivatives JSON: {json_path}. "
            "Run 01_eye_decoding_subject.py first."
        )
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing derivatives CSV: {csv_path}. "
            "Run 01_eye_decoding_subject.py first."
        )

    with open(json_path, 'r') as f:
        full_results = json.load(f)

    subject_df = pd.read_csv(csv_path)
    logger.info(
        f"[{feature_label}] Loaded {len(subject_df)} subject rows from "
        f"{csv_path.name}, full JSON from {json_path.name}"
    )

    # Sanity check: CSV subject count matches JSON
    n_csv = len(subject_df)
    n_json = full_results.get('n_subjects', -1)
    if n_csv != n_json:
        logger.warning(
            f"[{feature_label}] Subject count mismatch: CSV has {n_csv}, "
            f"JSON has {n_json}"
        )

    # Strip subject-level keys
    group_results = {
        k: v for k, v in full_results.items()
        if k not in _SUBJECT_LEVEL_KEYS
    }

    logger.info(
        f"[{feature_label}] mean_accuracy={group_results['mean_accuracy']:.3f}, "
        f"95% CI=[{group_results['ci_low']:.3f}, {group_results['ci_high']:.3f}], "
        f"p={group_results['p_value']:.4f}"
    )
    return group_results


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
config, out_dir, logger = setup_analysis(
    analysis_name="11_eye_decoding_group",
    results_base=CONFIG["RESULTS_ROOT"] / "eyetracking" / "logs",
    script_file=__file__,
)

DERIV_ROOT: Path = CONFIG['BIDS_EYETRACK_DECODING']

# ============================================================================
# Load and process XY results
# ============================================================================
logger.info("=" * 80)
logger.info("Loading group-level results: 2D gaze coordinates (x, y)")
logger.info("=" * 80)

results_xy = load_and_strip(DERIV_ROOT, 'xy', logger)

# Save group-level JSON (no subject data)
with open(out_dir / 'results_xy.json', 'w') as f:
    json.dump(results_xy, f, indent=2)

# Save fold accuracies CSV (not per-subject)
pd.DataFrame({'fold_accuracy': results_xy['fold_accuracies']}).to_csv(
    out_dir / 'fold_accuracies_xy.csv', index=False
)

logger.info(f"Saved results_xy.json and fold_accuracies_xy.csv to {out_dir}")

# ============================================================================
# Load and process displacement results
# ============================================================================
logger.info("=" * 80)
logger.info("Loading group-level results: Displacement from screen center")
logger.info("=" * 80)

results_disp = load_and_strip(DERIV_ROOT, 'displacement', logger)

# Save group-level JSON (no subject data)
with open(out_dir / 'results_displacement.json', 'w') as f:
    json.dump(results_disp, f, indent=2)

# Save fold accuracies CSV (not per-subject)
pd.DataFrame({'fold_accuracy': results_disp['fold_accuracies']}).to_csv(
    out_dir / 'fold_accuracies_displacement.csv', index=False
)

logger.info(f"Saved results_displacement.json and fold_accuracies_displacement.csv to {out_dir}")

# ============================================================================
# Summary
# ============================================================================
logger.info("=" * 80)
logger.info("Summary:")
logger.info(f"  xy: accuracy={results_xy['mean_accuracy']:.3f}, p={results_xy['p_value']:.4f}")
logger.info(f"  displacement: accuracy={results_disp['mean_accuracy']:.3f}, p={results_disp['p_value']:.4f}")
logger.info("=" * 80)

log_script_end(logger)
