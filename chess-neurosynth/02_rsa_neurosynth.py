#!/usr/bin/env python3
"""
Neurosynth RSA Correlation (Searchlight Experts>Novices z-map vs term maps)

Pipeline (per model RDM pattern)
--------------------------------
1) Find subject-level searchlight maps matching pattern
2) Split files by group using participants.tsv (experts vs novices)
3) Fisher z-transform subject maps
4) Second-level GLM (nilearn) with intercept and group (+1/-1)
5) Compute contrast 'group' as z-score map (Experts > Novices)
6) Split z into Z+ and Z−
7) Correlate with Neurosynth term maps (bootstrap CI, FDR)
8) Save CSV, LaTeX; render glass brain and surface; bar/diff plots

Notes
-----
- Inputs are local copies only; do not read /data/projects/...
- This script follows CLAUDE.md conventions (no CLI args; analysis vs plotting separation)
"""

import os
import sys
from pathlib import Path

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = Path(__file__).parent

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.glm.second_level import SecondLevelModel

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import get_subject_list
from common.neuro_utils import load_nifti, fisher_z_transform

from common.io_utils import find_nifti_files, split_by_group
from modules.io_utils import load_term_maps, reorder_by_term
from modules.maps_utils import (
    split_zmap_by_sign,
    compute_all_zmap_correlations,
)
from modules.glm_utils import build_design_matrix


# =====================
# Configuration (local)
# =====================
N_BOOTSTRAPS = 10000
PLOT_THRESH = 1e-5
SMOOTHING_FWHM_MM = None

# Patterns and pretty labels
PATTERNS = {
    'searchlight_checkmate': "Checkmate | RSA searchlight",
    'searchlight_strategy': "Strategy | RSA searchlight",
    'searchlight_visualSimilarity': "Visual Similarity | RSA searchlight",
}


# 1) Setup
extra = {
    'N_BOOTSTRAPS': N_BOOTSTRAPS,
    'ALPHA_FDR': CONFIG['ALPHA_FDR'],
    'PLOT_THRESH': PLOT_THRESH,
    'SMOOTHING_FWHM_MM': SMOOTHING_FWHM_MM,
}
config, output_dir, logger = setup_analysis(
    analysis_name="neurosynth_rsa",
    results_base=script_dir / 'results',
    script_file=__file__,
    extra_config=extra,
)

# 2) Groups from participants.tsv
experts = get_subject_list('expert')
novices = get_subject_list('novice')
logger.info(f"Experts: {len(experts)}; Novices: {len(novices)}")

# 3) Term maps
term_dir = CONFIG['NEUROSYNTH_TERMS_DIR']
term_maps = load_term_maps(term_dir)

# 4) RSA searchlight directory
rsa_root = CONFIG['BIDS_RSA_SEARCHLIGHT']

all_pos = {}
all_neg = {}
all_diff = {}

# 5) Process each pattern
for pattern, pretty in PATTERNS.items():

    logger.info(f"Processing pattern: {pattern} → {pretty}")
    files = find_nifti_files(rsa_root, pattern=pattern)
    logger.info(f"Found {len(files)} matching subject maps for pattern '{pattern}'")

    exp_files, nov_files = split_by_group(files, experts, novices)
    logger.info(f"  Experts: {len(exp_files)} maps; Novices: {len(nov_files)} maps")

    # Load and Fisher z-transform subject maps
    z_exp = [fisher_z_transform(load_nifti(f)) for f in exp_files]
    z_nov = [fisher_z_transform(load_nifti(f)) for f in nov_files]
    z_all = z_exp + z_nov

    # 5a) Second-level GLM
    design = build_design_matrix(len(z_exp), len(z_nov))
    slm = SecondLevelModel(smoothing_fwhm=SMOOTHING_FWHM_MM, n_jobs=-1)
    slm = slm.fit(z_all, design_matrix=design)
    con_img = slm.compute_contrast('group', output_type='z_score')
    z_map = con_img.get_fdata()

    # Save group z-score map for plotting script
    safe_base = pattern.replace(' ', '_')
    con_img.to_filename(str(output_dir / f"zmap_{safe_base}.nii.gz"))

    # 5b) Correlations with term maps
    z_pos, z_neg = split_zmap_by_sign(z_map)
    df_pos, df_neg, df_diff = compute_all_zmap_correlations(
        z_pos, z_neg, term_maps, ref_img=con_img,
        n_boot=N_BOOTSTRAPS, fdr_alpha=CONFIG['ALPHA_FDR'], ci_alpha=0.05, random_state=CONFIG['RANDOM_SEED']
    )

    # Reorder by canonical term order
    df_pos = reorder_by_term(df_pos)
    df_neg = reorder_by_term(df_neg)
    df_diff = reorder_by_term(df_diff)

    # Save outputs per pattern (plots/LaTeX handled in 02_plot)
    df_pos.to_csv(output_dir / f"{safe_base}_term_corr_positive.csv", index=False)
    df_neg.to_csv(output_dir / f"{safe_base}_term_corr_negative.csv", index=False)
    df_diff.to_csv(output_dir / f"{safe_base}_term_corr_difference.csv", index=False)
    # Keep for combined tables — handled in plotting script
    key = pattern.split('_', 1)[1] if '_' in pattern else pattern
    all_pos[key] = df_pos
    all_neg[key] = df_neg
    all_diff[key] = df_diff

# Done
log_script_end(logger)
