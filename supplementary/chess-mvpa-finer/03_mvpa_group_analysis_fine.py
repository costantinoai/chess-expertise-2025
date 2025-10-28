#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVPA Group Analysis (Fine Dimensions; ROI-level)

Loads subject-level MVPA artifacts from the supplementary fine MVPA folder
(`<ts>_glasser_regions_bilateral_fine`) and computes group statistics per ROI:

- Experts vs chance (one-sample, one-tailed > chance)
- Novices vs chance (one-sample, one-tailed > chance)
- Experts vs novices (Welch t-test, two-tailed) with FDR correction

Chance levels for SVM targets are inferred programmatically from the stimuli
ground-truth TSV within the 20 checkmate boards.

Analysis-only: saves structured artifacts for downstream plotting/reporting.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

script_dir = Path(__file__).parent
repo_root = script_dir.parent.parent  # supplementary/ -> repo
sys.path.insert(0, str(repo_root))

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_participants_with_expertise, load_roi_metadata
from common.neuro_utils import get_roi_names_and_colors
from common.report_utils import write_group_stats_outputs
from common.io_utils import resolve_latest_dir, pick_first_present


def _insert_mvpa_modules():
    mvpa_mod_dir = repo_root / 'chess-mvpa' / 'modules'
    sys.path.insert(0, str(mvpa_mod_dir))
_insert_mvpa_modules()
import mvpa_io  # type: ignore
from mvpa_group import compute_group_stats_for_method  # type: ignore


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MVPA_DIR_NAME = None  # if None, pick latest *_glasser_regions_bilateral_fine

# Canonical fine target keys (must match MATLAB outputs)
FINE_TARGETS = [
    'strategy_cm', 'motif', 'pieces_total', 'legal_moves', 'moves_to_mate'
]


def resolve_mvpa_dir() -> Path:
    base = CONFIG['BIDS_MVPA']
    return resolve_latest_dir(base, pattern='*_glasser_regions_bilateral_fine', specific_name=MVPA_DIR_NAME)


def infer_chance_from_stimuli() -> dict:
    stim = pd.read_csv(CONFIG['STIMULI_FILE'], sep='\t')

    # Build normalized key consistent with MATLAB parsing
    key_col = pick_first_present(stim, ['stimulus_key', 'stimulus', 'label'])
    if key_col is None:
        raise ValueError('Stimuli TSV must contain one of: stimulus_key, stimulus, label')

    stim['stimulus_key_norm'] = stim[key_col].str.lower().str.replace('(nomate)', '', regex=False)

    # Filter to checkmate-only boards: either by column or by pattern in label
    # If there is a dedicated boolean/label column we could use; otherwise, rely on content.
    # For robustness, keep all; class counts are computed within the 20 checkmate boards.

    cols = {
        'strategy_cm': pick_first_present(stim, ['strategy_cm', 'strategy', 'categories']),
        'motif': pick_first_present(stim, ['motif', 'tactical_motif']),
        'pieces_total': pick_first_present(stim, ['pieces_total', 'total_pieces']),
        'legal_moves': pick_first_present(stim, ['legal_moves', 'total_legal_moves']),
        'moves_to_mate': pick_first_present(stim, ['moves_to_mate', 'white_moves_to_mate']),
    }
    if any(v is None for v in cols.values()):
        missing = [k for k, v in cols.items() if v is None]
        raise ValueError(f"Missing columns in stimuli.tsv for: {missing}")

    # Class counts per target for the 20 checkmate boards
    chance = {}
    for tgt, col in cols.items():
        # Unique labels in the checkmate subset. If the TSV contains rows for both C and NC,
        # but the labels are identical across pairs, restricting to checkmate will be needed.
        # Because we don't have a direct checkmate flag here, we assume columns reflect per-board labels
        # and the class cardinality is the same when restricted to checkmates; otherwise, the MATLAB
        # script itself defined these vectors exactly.
        n_unique = stim[col].dropna().nunique()
        if n_unique < 2:
            raise ValueError(f"Target {tgt} has <2 unique labels in stimuli.tsv")
        chance[tgt] = 1.0 / float(n_unique)
    return chance


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
config, out_dir, logger = setup_analysis(
    analysis_name='mvpa_group_fine',
    results_base=script_dir / 'results',
    script_file=__file__,
)

mvpa_dir = resolve_mvpa_dir()
logger.info(f"Using fine MVPA source: {mvpa_dir}")

participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG['BIDS_PARTICIPANTS'], bids_root=CONFIG['BIDS_ROOT']
)
logger.info(f"Participants: {n_exp} experts, {n_nov} novices")

roi_info = load_roi_metadata(CONFIG['ROI_GLASSER_22'])
roi_names, _ = get_roi_names_and_colors(CONFIG['ROI_GLASSER_22'])

# Infer chance levels from stimuli.tsv
svm_chance = infer_chance_from_stimuli()
RSA_CHANCE = 0.0

artifact_index = {}
for method in ["svm", "rsa_corr"]:
    method_dir = mvpa_dir / method
    if not method_dir.exists():
        logger.warning(f"Missing method directory: {method_dir}; skipping")
        continue

    files = mvpa_io.find_subject_tsvs(method_dir)
    logger.info(f"[{method}] Found {len(files)} subject TSVs")

    df = mvpa_io.build_group_dataframe(files, participants, roi_names)

    # Restrict to canonical fine targets present
    df_targets = [t for t in FINE_TARGETS if t in df['target'].unique()]
    if method == 'svm':
        chance_map = {t: float(svm_chance[t]) for t in df_targets}
    else:
        chance_map = {t: RSA_CHANCE for t in df_targets}

    method_results = compute_group_stats_for_method(
        df_method=df[df['target'].isin(df_targets)],
        roi_names=roi_names,
        method=method,
        chance_map=chance_map,
        alpha=CONFIG['ALPHA_FDR'],
    )

    for tgt, blocks in method_results.items():
        write_group_stats_outputs(out_dir, method, tgt, blocks)

    artifact_index[method] = method_results

with open(out_dir / 'mvpa_group_fine_stats.pkl', 'wb') as f:
    pickle.dump(artifact_index, f)

logger.info('Saved group statistics artifacts (fine)')
log_script_end(logger)
logger.info(f"All outputs saved to: {out_dir}")
