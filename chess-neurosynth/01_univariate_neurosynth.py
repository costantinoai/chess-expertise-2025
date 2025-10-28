#!/usr/bin/env python3
"""
Neurosynth Univariate Correlation (Group GLM T-maps vs term maps)

Pipeline
--------
1) Load group-level GLM T-maps from CONFIG['BIDS_SPM_GROUP']
2) Convert T -> signed two-tailed Z (retain sign)
3) Split into Z+ and Z−
4) Correlate with Neurosynth term maps (bootstrap CI, FDR)
5) Save CSV, LaTeX; render glass brain and surface; bar/diff plots

Notes
-----
- Inputs are local copies only; do not read /data/projects/...
- This script follows CLAUDE.md conventions (no CLI args; analysis vs plotting separation)
"""

import sys
from pathlib import Path

# Import path management (per CLAUDE.md)
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))   # common/
sys.path.insert(0, str(script_dir))  # chess-neurosynth/modules

import numpy as np
import pandas as pd
from nilearn import image

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
 

from modules.io_utils import (
    find_nifti_files,
    load_term_maps,
    extract_run_label,
    reorder_by_term,
    find_group_tmaps,
)
from modules.maps_utils import (
    t_to_two_tailed_z,
    split_zmap_by_sign,
    compute_all_zmap_correlations,
)
# Plotting and LaTeX tables are handled in 02_plot_neurosynth.py


# =====================
# Configuration (local)
# =====================
DOF = 38                 # Degrees of freedom for group T-maps (set to manuscript value)
N_BOOTSTRAPS = 10000
ALPHA_FDR = 0.05
PLOT_THRESH = 1e-5


"""Analysis-only script: univariate neurosynth correlations.

Follows CLAUDE.md: no CLI/if-main. Configure and run sequentially.
"""

# 1) Setup logging and timestamped results directory
extra = {
    'DOF': DOF,
    'N_BOOTSTRAPS': N_BOOTSTRAPS,
    'ALPHA_FDR': ALPHA_FDR,
    'PLOT_THRESH': PLOT_THRESH,
}
config, output_dir, logger = setup_analysis(
    analysis_name="neurosynth_univariate",
    results_base=script_dir / 'results',
    script_file=__file__,
    extra_config=extra,
)

    # Analysis-only: do not create figures/tables here (plotting script handles it)

# 2) Load term maps
term_dir = CONFIG['NEUROSYNTH_TERMS_DIR']
term_maps = load_term_maps(term_dir)
logger.info(f"Loaded {len(term_maps)} term maps from {term_dir}")

# 3) Discover group-level T maps (only GLM-smooth4/group)
group_dir = CONFIG['BIDS_SPM_GROUP'] / 'GLM-smooth4' / 'group'
t_files = find_group_tmaps(group_dir)
logger.info(f"Found {len(t_files)} group T-map(s) for analysis in {group_dir}")

# 4) Process each T-map: convert to Z, split by sign, save zmap, compute correlations
for t_path in t_files:
    if not t_path.name.startswith('spmT_'):
        continue
    
    run_label = extract_run_label(t_path)
    safe_base = run_label.replace(' > ', '-gt-').replace(' ', '_')
    logger.info(f"Processing: {run_label}")

    # 4a) Load and convert to signed two-tailed z
    t_img = image.load_img(str(t_path))
    t_data = t_img.get_fdata()
    z_map = t_to_two_tailed_z(t_data, dof=DOF)
    z_pos, z_neg = split_zmap_by_sign(z_map)

    # 4b) Save z-map (NIfTI) for plotting script
    z_img = image.new_img_like(t_img, z_map)
    z_img.to_filename(str(output_dir / f"zmap_{safe_base}.nii.gz"))

    # 4c) Correlate z+ and z− vs each term map, with bootstrap/FDR
    df_pos, df_neg, df_diff = compute_all_zmap_correlations(
        z_pos, z_neg, term_maps, ref_img=z_img,
        n_boot=N_BOOTSTRAPS, fdr_alpha=ALPHA_FDR, ci_alpha=0.05, random_state=CONFIG['RANDOM_SEED']
    )

    # 4d) Reorder rows to canonical term order for consistent outputs
    df_pos = reorder_by_term(df_pos)
    df_neg = reorder_by_term(df_neg)
    df_diff = reorder_by_term(df_diff)

    # 4e) Save CSV results (plotting/LaTeX handled in 03_plot_neurosynth.py)
    df_pos.to_csv(output_dir / f"{safe_base}_term_corr_positive.csv", index=False)
    df_neg.to_csv(output_dir / f"{safe_base}_term_corr_negative.csv", index=False)
    df_diff.to_csv(output_dir / f"{safe_base}_term_corr_difference.csv", index=False)

# Done
log_script_end(logger)
