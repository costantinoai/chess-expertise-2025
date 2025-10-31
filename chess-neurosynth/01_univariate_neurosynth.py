#!/usr/bin/env python3
"""
Neurosynth Univariate Correlation (Group GLM T-maps vs term maps)

METHODS
=======

Rationale
---------
To interpret the functional significance of brain regions showing expertise-
related differences in univariate GLM contrasts, we correlated group-level
t-statistic maps with Neurosynth meta-analytic term maps. This data-driven
approach identifies cognitive functions preferentially associated with regions
where experts show stronger (or weaker) task-related activations compared to
novices.

Data
----
Group-level statistical parametric maps (SPMs) were computed using second-
level GLMs in SPM12. Each T-map represents a group contrast (e.g., Experts >
Novices) with t-statistics at each voxel. T-maps are stored as NIfTI files in
BIDS/derivatives/spm/GLM-smooth4/group/.

Participants: N=40 (20 experts, 20 novices).
Degrees of freedom: 38 (n_subjects − 2 for two-sample t-test).

Neurosynth term maps: Meta-analytic association z-score maps for cognitive
terms (e.g., "working memory", "visual attention", "semantic processing").
Term maps were downloaded from the Neurosynth database and resampled to match
the GLM map resolution.

T-to-Z Conversion
-----------------
Group-level T-maps were converted to signed two-tailed z-scores while
preserving the sign (direction) of the effect. For each voxel:

1. Compute the two-tailed p-value from the t-statistic using the t-
   distribution with 38 degrees of freedom.
2. Convert the p-value to a two-tailed z-score using the inverse cumulative
   standard normal distribution.
3. Assign the sign of the original t-statistic to the z-score.

This transformation standardizes effect magnitudes across contrasts and
allows comparison with Neurosynth z-score maps, which are also two-tailed.

Sign-Specific Map Splitting
----------------------------
The group z-map was split into two separate maps:
- Z+ (positive z-values, zeros elsewhere): regions with stronger expert
  activations
- Z− (absolute negative z-values, zeros elsewhere): regions with stronger
  novice activations

This allows separate functional interpretation of regions with expertise-
enhanced vs expertise-reduced activations.

Neurosynth Term Correlation
----------------------------
For each sign-specific map (Z+, Z−), we computed spatial correlations with
each Neurosynth term map. Only non-zero voxels in the z-map were included in
the correlation (to focus on regions with group differences). Correlations
were computed as Pearson r across voxel values.

For each term, we also computed the difference in correlations (r_pos − r_neg)
to identify terms that are differentially associated with expert-enhanced
versus expert-reduced regions. Positive differences indicate terms more
strongly associated with regions where experts show higher activations.

Statistical Assumptions and Limitations
----------------------------------------
- **Spatial independence**: Voxels are treated as independent observations for
  correlation analysis. In reality, fMRI voxels exhibit spatial autocorrelation
  due to smoothing (FWHM=4mm) and hemodynamic spread. Correlations are
  descriptive and should be interpreted as effect size measures rather than
  statistical tests.
- **Neurosynth circularity**: If the current study overlaps with Neurosynth
  database studies, term maps may reflect the current dataset, inflating
  correlations. The Neurosynth database is large (>10,000 studies), mitigating
  this concern.

Outputs
-------
All results are saved to results/<timestamp>_neurosynth_univariate/:
- zmap_<contrast>.nii.gz: Signed z-score map (converted from T-map)
- <contrast>_term_corr_positive.csv: Z+ term correlations (term, r)
- <contrast>_term_corr_negative.csv: Z− term correlations (term, r)
- <contrast>_term_corr_difference.csv: Correlation differences (term, r_pos, r_neg, r_diff)
- 01_univariate_neurosynth.py: Copy of this script
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

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
 

from common.io_utils import find_nifti_files
from modules.io_utils import (
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
DOF = 38  # Degrees of freedom for group T-maps (n=40 subjects, 2 groups → df=38)


"""Analysis-only script: univariate neurosynth correlations.

Follows CLAUDE.md: no CLI/if-main. Configure and run sequentially.
"""

# 1) Setup logging and timestamped results directory
extra = {
    'DOF': DOF,
}
config, output_dir, logger = setup_analysis(
    analysis_name="neurosynth_univariate",
    results_base=script_dir / 'results',
    script_file=__file__,
    extra_config=extra,
)

    # Analysis-only: do not create figures/tables here (plotting script handles it)

# Load Neurosynth meta-analytic term maps for functional interpretation.
# Same as RSA script: these are z-score maps representing cognitive term associations.
term_dir = CONFIG['NEUROSYNTH_TERMS_DIR']
term_maps = load_term_maps(term_dir)
logger.info(f"Loaded {len(term_maps)} term maps from {term_dir}")

# Find group-level statistical parametric maps (SPMs) from SPM12 second-level GLMs.
# These are T-maps representing group contrasts (e.g., Experts > Novices) computed
# from first-level beta images. We analyze only the smoothed (4mm FWHM) GLM results.
group_dir = CONFIG['BIDS_SPM_GROUP'] / 'GLM-smooth4' / 'group'
t_files = find_group_tmaps(group_dir)
logger.info(f"Found {len(t_files)} group T-map(s) for analysis in {group_dir}")

# Process each T-map: convert T-statistics to z-scores, split by sign, and correlate
# with Neurosynth term maps to identify cognitive functions associated with group differences.
for t_path in t_files:
    if not t_path.name.startswith('spmT_'):
        continue

    # Extract a human-readable label from the filename (e.g., "Experts > Novices")
    run_label = extract_run_label(t_path)
    safe_base = run_label.replace(' > ', '-gt-').replace(' ', '_')
    logger.info(f"Processing: {run_label}")

    # Load T-map and convert to signed two-tailed z-scores.
    # T-to-z conversion: for each voxel, compute two-tailed p-value from t-statistic
    # (df=38 for 40 subjects − 2 groups), convert p to z-score using inverse normal CDF,
    # and preserve the sign of the original t-statistic. This standardizes effect sizes
    # and makes them comparable to Neurosynth z-maps.
    t_img = image.load_img(str(t_path))
    t_data = t_img.get_fdata()
    z_map = t_to_two_tailed_z(t_data, dof=DOF)

    # Split z-map by sign to separately characterize regions with expert-enhanced
    # vs expert-reduced activations. Z+ = positive z-values; Z− = absolute negative z-values.
    z_pos, z_neg = split_zmap_by_sign(z_map)

    # Save z-map for visualization in plotting scripts
    z_img = image.new_img_like(t_img, z_map)
    z_img.to_filename(str(output_dir / f"zmap_{safe_base}.nii.gz"))

    # Correlate Z+ and Z− with each Neurosynth term map across voxels.
    # This identifies which cognitive functions are most strongly associated
    # with regions showing expertise-related activation differences.
    df_pos, df_neg, df_diff = compute_all_zmap_correlations(
        z_pos, z_neg, term_maps, ref_img=z_img
    )

    # Reorder rows by canonical term order for consistency
    df_pos = reorder_by_term(df_pos)
    df_neg = reorder_by_term(df_neg)
    df_diff = reorder_by_term(df_diff)

    # Save per-contrast results as CSVs
    df_pos.to_csv(output_dir / f"{safe_base}_term_corr_positive.csv", index=False)
    df_neg.to_csv(output_dir / f"{safe_base}_term_corr_negative.csv", index=False)
    df_diff.to_csv(output_dir / f"{safe_base}_term_corr_difference.csv", index=False)

# Done
log_script_end(logger)
