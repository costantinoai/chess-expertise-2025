#!/usr/bin/env python3
"""
Neurosynth RSA Correlation (Searchlight Experts>Novices z-map vs term maps)

METHODS
=======

Rationale
---------
To interpret the functional significance of brain regions showing expertise-
related differences in RSA searchlight analyses, we correlated group-contrast
z-score maps with Neurosynth meta-analytic term maps. This data-driven
approach identifies cognitive functions preferentially associated with regions
where experts show stronger (or weaker) neural-model correlations compared to
novices.

Data
----
Subject-level RSA searchlight maps were previously computed by correlating
whole-brain searchlight RDMs with theoretical model RDMs (check status,
strategy, visual similarity). Each map is a 3D NIfTI volume with correlation
coefficients at each voxel.

Participants: N=40 (20 experts, 20 novices).
Model RDMs: Check status, strategy, visual similarity.

Neurosynth term maps: Meta-analytic association z-score maps for cognitive
terms (e.g., "working memory", "visual attention", "semantic processing").
Term maps were downloaded from the Neurosynth database and resampled to match
the searchlight map resolution.

Second-Level Group Analysis
----------------------------
For each model RDM pattern, we performed the following:

1. **Fisher z-transformation**: Subject-level correlation maps were Fisher
   z-transformed to stabilize variance for group-level inference.

2. **Group GLM**: A second-level general linear model (GLM) was fitted with
   two regressors: an intercept (group mean) and a group contrast (experts =
   +1, novices = −1). Implementation: nilearn.glm.second_level.SecondLevelModel.

3. **Group contrast map**: The group contrast was computed as a z-score map
   representing Experts > Novices. Positive z-values indicate voxels where
   experts show stronger RSA correlations; negative z-values indicate voxels
   where novices show stronger correlations.

4. **Sign-specific maps**: The group z-map was split into two separate maps:
   Z+ (positive z-values, zeros elsewhere) and Z− (absolute negative z-values,
   zeros elsewhere). This allows separate functional interpretation of regions
   with expertise-enhanced vs expertise-reduced correlations.

Neurosynth Term Correlation
----------------------------
For each sign-specific map (Z+, Z−), we computed spatial correlations with
each Neurosynth term map. Only non-zero voxels in the z-map were included in
the correlation (to focus on regions with group differences). Correlations
were computed as Pearson r across voxel values.

For each term, we also computed the difference in correlations (r_pos − r_neg)
to identify terms that are differentially associated with expert-enhanced
versus expert-reduced regions. Positive differences indicate terms more
strongly associated with regions where experts show higher correlations.

Bootstrap Statistical Inference
--------------------------------
To provide robust confidence intervals and p-values accounting for spatial
dependencies in brain imaging data, we employed bootstrap resampling:

1. **Individual correlations** (r_pos, r_neg):
   - Method: Percentile bootstrap using Pingouin implementation
   - Resamples: 10,000 bootstrap samples
   - Procedure: For each bootstrap iteration, voxels are resampled with
     replacement, and Pearson correlation is recomputed
   - CI estimation: 95% confidence intervals computed using the percentile
     method (2.5th and 97.5th percentiles of bootstrap distribution)
   - P-value: Two-sided test based on the bootstrap distribution
   - Implementation: common.stats_utils.correlate_vectors_bootstrap()

2. **Correlation differences** (Δr = r_pos − r_neg):
   - Method: Custom percentile bootstrap for paired differences
   - Resamples: 10,000 bootstrap samples
   - Procedure: For each bootstrap iteration:
     a. Resample voxel indices with replacement
     b. Compute r_x = corr(term, Z+) on resampled data
     c. Compute r_y = corr(term, Z−) on resampled data
     d. Store difference: Δr_i = r_x − r_y
   - CI estimation: 95% confidence intervals from percentile method
     (2.5th and 97.5th percentiles of Δr bootstrap distribution)
   - P-value: Two-sided test computed as 2 × min(P(Δr ≤ 0), P(Δr ≥ 0))
   - Implementation: modules.maps_utils.bootstrap_corr_diff()

3. **Multiple testing correction**:
   - Method: False Discovery Rate (FDR) correction using Benjamini-Hochberg
     procedure
   - Applied separately to positive, negative, and difference correlations
   - Significance level: α = 0.05 (5% FDR)
   - Implementation: common.stats_utils.apply_fdr_correction()

The bootstrap approach addresses spatial autocorrelation by resampling voxels
as units, providing empirical confidence intervals that reflect sampling
variability without assuming independence. The percentile method is
distribution-free and robust to non-normality.

Statistical Assumptions and Limitations
----------------------------------------
- **Spatial independence**: Voxels are treated as independent observations for
  correlation analysis. In reality, fMRI voxels exhibit spatial autocorrelation
  due to smoothing and hemodynamic spread. Correlations are descriptive and
  should be interpreted as effect size measures rather than statistical tests.
- **Neurosynth circularity**: If the current study overlaps with Neurosynth
  database studies, term maps may reflect the current dataset, inflating
  correlations. The Neurosynth database is large (>10,000 studies), mitigating
  this concern.

Outputs
-------
All results are saved to results/<timestamp>_neurosynth_rsa/:
- zmap_<pattern>.nii.gz: Group z-score map (Experts > Novices)
- <pattern>_term_corr_positive.csv: Z+ term correlations (term, r)
- <pattern>_term_corr_negative.csv: Z− term correlations (term, r)
- <pattern>_term_corr_difference.csv: Correlation differences (term, r_pos, r_neg, r_diff)
- 02_rsa_neurosynth.py: Copy of this script
"""

import os
import sys
from pathlib import Path

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = Path(__file__).parent

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
SMOOTHING_FWHM_MM = None

# Patterns and pretty labels
PATTERNS = {
    'searchlight_checkmate': "Checkmate | RSA searchlight",
    'searchlight_strategy': "Strategy | RSA searchlight",
    'searchlight_visual_similarity': "Visual Similarity | RSA searchlight",
}


# 1) Setup
extra = {
    'SMOOTHING_FWHM_MM': SMOOTHING_FWHM_MM,
}
config, output_dir, logger = setup_analysis(
    analysis_name="neurosynth_rsa",
    results_base=script_dir / 'results',
    script_file=__file__,
    extra_config=extra,
)

# Load participant IDs grouped by expertise for splitting subject-level maps
experts = get_subject_list('expert')
novices = get_subject_list('novice')
logger.info(f"Experts: {len(experts)}; Novices: {len(novices)}")

# Load Neurosynth meta-analytic term maps (z-score maps for cognitive terms like
# "working memory", "visual attention", etc.). These are whole-brain maps derived
# from >10,000 studies, representing the probability of a term being mentioned
# when a voxel is activated. We'll correlate our group contrast maps with these
# to identify which cognitive functions are associated with expertise differences.
term_dir = CONFIG['NEUROSYNTH_TERMS_DIR']
term_maps = load_term_maps(term_dir)

# Subject-level RSA searchlight maps are stored in BIDS derivatives
rsa_root = CONFIG['BIDS_RSA_SEARCHLIGHT']

# Storage for combined results across all patterns (for multi-panel tables)
all_pos = {}
all_neg = {}
all_diff = {}

# Process each RSA model pattern (checkmate, strategy, visual similarity).
# For each, we'll compute a group contrast z-map (Experts > Novices) and correlate
# it with Neurosynth term maps to identify functional associations.
for pattern, pretty in PATTERNS.items():

    logger.info(f"Processing pattern: {pattern} → {pretty}")

    # Find subject-level searchlight correlation maps matching this pattern.
    # Each file is a 3D NIfTI volume with correlation coefficients at each voxel.
    files = find_nifti_files(rsa_root, pattern=pattern)
    logger.info(f"Found {len(files)} matching subject maps for pattern '{pattern}'")

    # Split files by expertise group based on subject IDs
    exp_files, nov_files = split_by_group(files, experts, novices)
    logger.info(f"  Experts: {len(exp_files)} maps; Novices: {len(nov_files)} maps")

    # Load subject maps and apply Fisher z-transformation to stabilize variance.
    # Fisher z-transform: z = arctanh(r), which maps correlations [-1,1] to [-inf,+inf]
    # and makes their sampling distribution approximately normal for group inference.
    z_exp = [fisher_z_transform(load_nifti(f)) for f in exp_files]
    z_nov = [fisher_z_transform(load_nifti(f)) for f in nov_files]
    z_all = z_exp + z_nov

    # Fit a second-level general linear model (GLM) to test for group differences.
    # Design matrix has two columns: intercept (overall mean) and group contrast
    # (experts=+1, novices=-1). This allows us to test: are experts' correlations
    # higher than novices' at each voxel?
    design = build_design_matrix(len(z_exp), len(z_nov))
    slm = SecondLevelModel(smoothing_fwhm=SMOOTHING_FWHM_MM, n_jobs=-1)
    slm = slm.fit(z_all, design_matrix=design)

    # Compute the group contrast as a z-score map. Positive z-values = voxels where
    # experts show stronger RSA correlations; negative z-values = voxels where novices
    # show stronger correlations. Z-scores quantify effect size in standard deviations.
    con_img = slm.compute_contrast('group', output_type='z_score')
    z_map = con_img.get_fdata()

    # Save the group z-map for visualization in plotting scripts
    safe_base = pattern.replace(' ', '_')
    con_img.to_filename(str(output_dir / f"zmap_{safe_base}.nii.gz"))

    # Split the z-map by sign to separately analyze regions with expert-enhanced
    # vs expert-reduced correlations. Z+ contains positive z-values (zeros elsewhere);
    # Z− contains absolute negative z-values (zeros elsewhere). This allows separate
    # functional characterization of regions showing opposite expertise effects.
    z_pos, z_neg = split_zmap_by_sign(z_map)

    # Correlate Z+ and Z− with each Neurosynth term map. For each term, compute
    # Pearson correlation across voxels (excluding zeros). This identifies which
    # cognitive functions are most strongly associated with regions showing
    # expertise-related differences in RSA correlations.
    df_pos, df_neg, df_diff = compute_all_zmap_correlations(
        z_pos, z_neg, term_maps, ref_img=con_img
    )

    # Reorder rows by canonical term order for consistency across analyses
    df_pos = reorder_by_term(df_pos)
    df_neg = reorder_by_term(df_neg)
    df_diff = reorder_by_term(df_diff)

    # Save per-pattern results as CSVs
    df_pos.to_csv(output_dir / f"{safe_base}_term_corr_positive.csv", index=False)
    df_neg.to_csv(output_dir / f"{safe_base}_term_corr_negative.csv", index=False)
    df_diff.to_csv(output_dir / f"{safe_base}_term_corr_difference.csv", index=False)

    # Store for combined multi-panel tables (generated in plotting scripts)
    key = pattern.split('_', 1)[1] if '_' in pattern else pattern
    all_pos[key] = df_pos
    all_neg[key] = df_neg
    all_diff[key] = df_diff

# Done
log_script_end(logger)
