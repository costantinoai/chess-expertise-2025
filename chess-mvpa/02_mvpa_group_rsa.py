#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVPA Group Analysis — RSA Correlations (ROI-level)

METHODS
=======

Rationale
---------
Representational similarity analysis (RSA) tests whether neural
dissimilarity patterns correlate with theoretical model dissimilarities. To
test whether chess expertise modulates neural encoding of stimulus features,
we performed group-level statistical tests on subject-level RSA correlation
coefficients across regions of interest (ROIs).

Data
----
Subject-level RSA correlation coefficients were computed by correlating
neural representational dissimilarity matrices (RDMs) with theoretical model
RDMs. Neural RDMs were constructed from trial-wise beta estimates within each
of 22 bilateral cortical regions (Glasser multimodal parcellation). Subject-
level correlations (Spearman or Pearson) are stored as TSV files in
BIDS/derivatives/mvpa-rsa/, with one file per subject following BIDS-like
naming: sub-XX_space-MNI152NLin2009cAsym_roi-glasser_rdm.tsv.

Participants: N=40 (20 experts, 20 novices).
Stimuli: 40 chess board positions (20 check, 20 non-check).
ROIs: 22 bilateral cortical regions.

For RSA correlations, chance level = 0 (null hypothesis: no correlation
between neural and model RDMs).

Group-Level Statistical Testing
--------------------------------
Subject-level correlations were grouped by expertise (experts vs novices) for
each ROI and each model RDM target. Three statistical tests were conducted per
ROI:

1. **Experts vs Chance (zero)**: One-sample one-tailed t-test (greater) testing
   whether expert mean correlation is greater than zero.
   Null hypothesis: μ_expert ≤ 0.
   Implementation: scipy.stats.ttest_1samp with alternative='greater'.

2. **Novices vs Chance (zero)**: One-sample one-tailed t-test (greater) testing
   whether novice mean correlation is greater than zero.
   Null hypothesis: μ_novice ≤ 0.
   Implementation: scipy.stats.ttest_1samp with alternative='greater'.

3. **Experts vs Novices**: Welch two-sample two-tailed t-test comparing expert
   and novice mean correlations. Welch's t-test does not assume equal
   variances.
   Null hypothesis: μ_expert = μ_novice.
   Implementation: scipy.stats.ttest_ind with equal_var=False.

One-sample tests use one-tailed p-values (greater) as we test for positive
correlations with model RDMs; the group comparison (experts vs novices) uses a
two-tailed p-value.

False Discovery Rate (FDR) Correction
--------------------------------------
To control for multiple comparisons across the 22 ROIs, we applied the
Benjamini-Hochberg FDR correction separately within each model target and each
test type (expert vs zero, novice vs zero, expert vs novice).

Implementation: statsmodels.stats.multitest.multipletests with method='fdr_bh'
and alpha=0.05.

FDR correction was performed independently for each test type because each
addresses a distinct hypothesis family. ROIs were considered significant if
their FDR-corrected q-value was below 0.05.

Statistical Assumptions and Limitations
----------------------------------------
- **Normality**: t-tests assume normally distributed correlation coefficients
  within each group and ROI. With n=20 per group, the central limit theorem
  provides robustness to moderate deviations.
- **Independence**: Subject-level correlations are assumed independent.
- **Equal variances**: Welch's t-test relaxes the equal variance assumption.
- **Spatial dependence**: ROIs are anatomically adjacent and functionally
  connected, violating independence. FDR correction partially accounts for
  this by controlling the expected proportion of false positives.
- **Fisher z-transformation**: Correlation coefficients were not Fisher
  z-transformed for group-level testing. This is standard practice when sample
  correlations are not strongly skewed.

Outputs
-------
All results are saved to results/mvpa_group/:
- <target>_experts_vs_chance.csv: Expert vs zero statistics per ROI
- <target>_novices_vs_chance.csv: Novice vs zero statistics per ROI
- <target>_experts_vs_novices.csv: Group comparison statistics per ROI
- mvpa_group_stats.pkl: Complete results dictionary (for plotting scripts)
- 02_mvpa_group_rsa.py: Copy of this script
"""

import os
import sys
from pathlib import Path
import pickle

# Add parent (repo root) to sys.path for 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = Path(__file__).parent

from common import CONFIG, setup_or_reuse_analysis_dir, log_script_end
from common.bids_utils import (
    get_participants_with_expertise,
    load_roi_metadata,
)
from common.neuro_utils import get_roi_names_and_colors
from common.report_utils import write_group_stats_outputs

from modules.mvpa_io import (
    find_subject_tsvs,
    build_group_dataframe,
)
from modules.mvpa_group import (
    compute_per_roi_group_comparison,
    compute_per_roi_vs_chance_tests,
    split_data_by_target_and_group,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# No configuration needed - MVPA RSA data location is defined in constants.py


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

results_dir, logger, _ = setup_or_reuse_analysis_dir(
    __file__, analysis_name="mvpa_group"
)

# Locate subject-level RSA correlation results. At the subject level, neural RDMs
# were constructed from trial-wise beta patterns within each ROI, then correlated
# (Spearman or Pearson) with theoretical model RDMs. This script tests whether
# those correlations differ significantly from zero and between expertise groups.
rsa_dir = CONFIG["BIDS_MVPA_RSA"]
if not rsa_dir.exists():
    raise FileNotFoundError(f"Missing MVPA RSA directory: {rsa_dir}")
logger.info(f"Using MVPA RSA source: {rsa_dir}")

# Load participant list with expertise labels for group assignment
participants, (n_exp, n_nov) = get_participants_with_expertise(
    participants_file=CONFIG["BIDS_PARTICIPANTS"], bids_root=CONFIG["BIDS_ROOT"]
)
logger.info(f"Participants: {n_exp} experts, {n_nov} novices")

# Load ROI metadata for labeling and plotting
roi_info = load_roi_metadata(CONFIG["ROI_GLASSER_22"])
default_roi_names, _ = get_roi_names_and_colors(CONFIG["ROI_GLASSER_22"])

artifact_index = {}

# Find all subject-level TSV files containing correlation coefficients (one per subject)
files = find_subject_tsvs(rsa_dir)
logger.info(f"Found {len(files)} subject TSVs in {rsa_dir.name}")

# Load and consolidate into a single DataFrame: subject, expert, target, and one column per ROI.
# Each ROI column contains the correlation coefficient (r) between neural and model RDM.
df = build_group_dataframe(files, participants, default_roi_names)
roi_names = [c for c in df.columns if c not in ["participant_id", "expert", "target"]]

# For RSA correlations, chance level is 0 (null hypothesis: no correlation).
# Unlike decoding, there's no stimulus-dependent chance level; correlations can be
# positive, negative, or zero, so we test against 0.
chance_level = float(CONFIG.get('CHANCE_LEVEL_RSA', 0.0))

# For each target and each ROI, perform three statistical tests:
# 1. Experts vs zero: one-sample two-tailed t-test (µ_expert ≠ 0)
# 2. Novices vs zero: one-sample two-tailed t-test (µ_novice ≠ 0)
# 3. Experts vs novices: Welch two-sample two-tailed t-test
# Apply FDR correction across ROIs within each test type and target.
targets = sorted(df['target'].dropna().unique())
method_results = {}

for tgt in targets:
    # Extract expert and novice data for this target
    expert_data, novice_data = split_data_by_target_and_group(df, tgt, roi_names)

    # Perform between-group comparison (expert vs novice) with descriptives
    group_comparison = compute_per_roi_group_comparison(
        expert_data=expert_data,
        novice_data=novice_data,
        roi_names=roi_names,
        alpha=CONFIG['ALPHA_FDR'],
        confidence_level=0.95,
    )

    # Perform within-group vs-chance tests with descriptives
    expert_vs_chance, novice_vs_chance = compute_per_roi_vs_chance_tests(
        expert_data=expert_data,
        novice_data=novice_data,
        roi_names=roi_names,
        chance_level=chance_level,
        alpha=CONFIG['ALPHA_FDR'],
        alternative='greater',
        confidence_level=0.95,
    )

    # Package results for output functions
    method_results[tgt] = {
        'welch_expert_vs_novice': group_comparison['test_results'],
        'experts_vs_chance': expert_vs_chance['test_results'],
        'novices_vs_chance': novice_vs_chance['test_results'],
        'chance': chance_level,
        'experts_desc': group_comparison['expert_desc'],
        'novices_desc': group_comparison['novice_desc'],
    }

# Save results as human-readable CSVs and pickle files for plotting
for tgt, blocks in method_results.items():
    write_group_stats_outputs(results_dir, "rsa_corr", tgt, blocks)

# Merge with existing artifact index if present (single unified folder)
artifact_index_path = results_dir / "mvpa_group_stats.pkl"
if artifact_index_path.exists():
    try:
        with open(artifact_index_path, "rb") as f:
            prev = pickle.load(f)
    except Exception:
        prev = {}
else:
    prev = {}
prev["rsa_corr"] = method_results
with open(artifact_index_path, "wb") as f:
    pickle.dump(prev, f)

logger.info("Saved group statistics artifacts (RSA)")
log_script_end(logger)
logger.info(f"All outputs saved to: {results_dir}")
