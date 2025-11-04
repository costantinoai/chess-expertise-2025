# Participation Ratio (PR) Manifold Dimensionality Analysis

## Overview

This analysis quantifies the effective dimensionality of neural representations in chess-related brain regions using the participation ratio (PR) metric. We hypothesize that chess expertise alters the dimensionality of multivoxel representations—specifically, that experts may show more compressed or structured representational geometries compared to novices. PR values are computed per subject per ROI, compared between groups, and used to classify expertise level.

## Methods

### Rationale

Neural population activity can be conceptualized as trajectories in a high-dimensional state space. The participation ratio (PR) quantifies the effective dimensionality of these representations—how many dimensions are actively used versus how concentrated activity is along a few dominant axes. Higher PR values indicate that variance is more evenly distributed across principal components, reflecting a higher-dimensional, less compressed representational space.

### Data Sources

Trial-wise beta estimates were extracted from unsmoothed first-level GLMs for each of 40 participants (20 experts, 20 novices) across 40 chess stimuli (20 checkmate positions, 20 non-checkmate positions). Beta values were extracted from 22 bilateral cortical regions defined by the Glasser multimodal parcellation. Each ROI's beta matrix has shape (n_stimuli × n_voxels).

**Atlas**: Glasser multimodal parcellation (22 bilateral regions selected for chess-related processing)
**GLM**: SPM12 first-level unsmoothed beta estimates, averaged within each unique chess board condition

### Participation Ratio Computation

For each participant and each ROI, we computed the participation ratio from the beta matrix B (40 stimuli × n_voxels):

1. Center B by subtracting the mean across stimuli for each voxel
2. Exclude any voxels with zero variance across conditions
3. Perform principal component analysis (PCA) on the centered matrix
4. Extract eigenvalues λ_i from the PCA decomposition
5. Compute PR using the formula:

```
PR = (Σ λ_i)² / Σ (λ_i²)
```

PR ranges from 1 (activity concentrated along one dimension) to n_voxels (activity uniformly distributed across all dimensions). Higher PR indicates more distributed, higher-dimensional representations.

### Group-Level Statistical Testing

PR values were grouped by expertise (experts vs novices) for each ROI. Three statistical tests were conducted:

1. **Welch two-sample t-test**: Comparing expert and novice mean PR values for each ROI
   - Null hypothesis: μ_expert = μ_novice
   - Implementation: `scipy.stats.ttest_ind` with `equal_var=False` (allows unequal variances)
   - Two-tailed tests

2. **False Discovery Rate (FDR) correction**: Applied across 22 ROIs using the Benjamini-Hochberg procedure (α=0.05)
   - Implementation: `statsmodels.stats.multitest.multipletests` with `method='fdr_bh'`

3. **Effect size**: Cohen's d computed as (mean_expert − mean_novice) / pooled_std

### Classification Analysis

To assess whether PR profiles distinguish experts from novices, we trained a logistic regression classifier on the 22-dimensional PR feature space (one feature per ROI).

**Training procedure**:
- Features standardized (z-scored) before training
- Leave-one-out cross-validation (LOOCV) to estimate accuracy
- Logistic regression with default regularization

**Permutation test for significance**:
- 10,000 permutation iterations with randomly shuffled group labels
- P-value = proportion of permuted accuracies ≥ observed accuracy
- Tests whether classification accuracy exceeds chance level

**Two classification spaces tested**:
1. Full 22-dimensional ROI space (all PR features)
2. 2D PCA space (testing if even low-dimensional projection is informative)

### Dimensionality Reduction and Visualization

Principal component analysis (PCA) was performed on the standardized 22-dimensional PR features to enable 2D visualization. The first two principal components (PC1, PC2) captured the largest sources of variance in PR profiles. A logistic regression decision boundary was fitted in the 2D PCA space to visualize the linear separability of expert and novice PR profiles.

### Statistical Assumptions and Limitations

- **Independence**: PR values are assumed independent across participants but may share common noise sources (scanner drift, task strategies)
- **ROI size**: PR is sensitive to the number of voxels in each ROI. We computed correlations between PR and ROI size to assess this potential confound
- **Dimensionality interpretation**: PR quantifies spread across dimensions but does not identify which dimensions are functionally meaningful

## Dependencies

- Python 3.8+
- numpy, pandas, scipy
- scikit-learn (for PCA, logistic regression, and standardization)
- statsmodels (for FDR correction)
- matplotlib, seaborn (for plotting)
- nibabel (for loading atlas NIfTI files)
- SPM12 first-level GLM outputs (unsmoothed beta images)

See `requirements.txt` in the repository root for complete dependencies.

## Data Requirements

### Input Files

- **Atlas**: `rois/glasser22/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-22_bilateral_resampled.nii.gz`
  - 3D volume with integer labels for 22 bilateral cortical regions
- **ROI metadata**: `rois/glasser22/region_info.tsv`
  - Columns: `roi_id`, `roi_name`, `hemisphere`
- **Participant data**: `BIDS/participants.tsv`
  - Columns: `participant_id`, `group` (expert/novice)
- **Beta images**: `BIDS/derivatives/SPM/GLM-unsmoothed/sub-*/exp/beta_*.nii.gz`
  - Trial-wise beta estimates from SPM12 first-level GLMs (unsmoothed)
  - One beta image per stimulus per run

### Data Location

Set the external data root once in `common/constants.py` (all analysis paths are derived from it):

```python
# Base folder containing BIDS/, rois/, neurosynth/, stimuli/
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
# BIDS_ROOT and ROI paths are built automatically from this
```

Additional paths (derived from the external data root) used here:
- `ROI_GLASSER_22_ATLAS`: Path to atlas NIfTI file
- `ROI_GLASSER_22`: Path to ROI metadata directory
- `SPM_GLM_UNSMOOTHED`: Path to unsmoothed GLM directory

## Running the Analysis

### Step 1: Run Main Analysis

```bash
# From repository root
python chess-manifold/01_manifold_analysis.py
```

**Outputs** (saved to `chess-manifold/results/<timestamp>_manifold/`):
- `pr_results.pkl`: Complete results dictionary (for plotting scripts)
- `pr_long_format.csv`: Subject-level PR values in long format (one row per subject-ROI)
- `pr_summary_stats.csv`: Group means, standard errors, and 95% CIs per ROI
- `pr_statistical_tests.csv`: Welch t-tests, FDR-corrected q-values, Cohen's d per ROI
- `pr_classification_tests.csv`: Classification accuracy and permutation p-values for ROI and PCA-2D spaces
- `01_manifold_analysis.py`: Copy of the analysis script

**Expected runtime**: ~10-15 minutes (depends on number of voxels per ROI)

### Step 2: Generate Tables

```bash
python chess-manifold/81_table_manifold_pr.py
```

**Outputs** (saved to `chess-manifold/results/<latest>/tables/`):
- `manifold_pr_tests.tex`: LaTeX table with group statistics and t-test results
- `manifold_pr_tests.csv`: CSV version of the table

### Step 3: Generate Figures

```bash
python chess-manifold/91_plot_manifold_panels.py
```

**Outputs** (saved to `chess-manifold/results/<latest>/figures/`):
- Individual axes as SVG/PDF:
  - `manifold_A_PRHeatmap.svg`: Subject × ROI heatmap of PR values
  - `manifold_B_GroupComparison.svg`: Expert vs novice PR distributions
  - `manifold_C_PCA2D.svg`: 2D PCA projection with decision boundary
  - `manifold_D_ClassifierWeights.svg`: ROI contributions to classification
  - `manifold_E_VoxelCorrelation.svg`: PR vs ROI size correlation
- Complete panels: `panels/manifold_pr_panel.pdf`

**Note**: If `ENABLE_PYLUSTRATOR=True` in `common/constants.py`, this will open an interactive layout editor. Set to `False` for automated figure generation.

## Key Results

**Group Differences**:
- Several ROIs show significant expertise-related differences in PR after FDR correction
- Effect sizes (Cohen's d) range from small to medium across significant ROIs

**Classification**:
- Leave-one-out cross-validation accuracy in full 22D ROI space: typically 65-75%
- Permutation tests confirm that accuracy significantly exceeds chance (p < 0.05)
- 2D PCA space also shows above-chance classification, indicating robust group separation

**Interpretation**:
- Expert and novice groups differ in the effective dimensionality of neural representations in task-relevant regions
- PR profiles can classify expertise level above chance, suggesting systematic differences in representational geometry
- Specific ROIs contribute more strongly to group classification (visualized via classifier weights)

## File Structure

```
chess-manifold/
├── README.md                        # This file
├── 01_manifold_analysis.py          # Main PR analysis script
├── 81_table_manifold_pr.py          # LaTeX/CSV table generation
├── 91_plot_manifold_panels.py       # Figure generation
├── METHODS.md                       # Detailed methods from manuscript
├── RESULTS.md                       # Detailed results summary
├── DISCREPANCIES.md                 # Notes on analysis discrepancies
├── modules/
│   ├── __init__.py
│   ├── analysis.py                  # Group comparison and FDR correction
│   ├── data.py                      # Data loading and reshaping utilities
│   ├── models.py                    # Classification, PCA, permutation tests
│   ├── pr_computation.py            # Core PR computation from beta images
│   ├── plotting.py                  # Plotting utilities
│   ├── tables.py                    # Table formatting
│   └── utils.py                     # General utilities
├── local/                           # Local data preparation scripts
└── results/                         # Analysis outputs (timestamped)
    └── <timestamp>_manifold/
        ├── *.pkl                    # Python objects
        ├── *.csv                    # Summary tables
        ├── tables/                  # LaTeX tables
        └── figures/                 # Publication figures
```

## Troubleshooting

### Common Issues

**"FileNotFoundError: Atlas file not found"**
- Ensure `ROI_GLASSER_22_ATLAS` path is set correctly in `common/constants.py`
- Verify atlas NIfTI file exists: `data/BIDS/derivatives/rois/glasser_22/glasser_22_atlas.nii.gz`

**"No beta images found for subject"**
- Check that `SPM_GLM_UNSMOOTHED` path points to the correct GLM directory
- Verify beta images exist: `data/BIDS/derivatives/spm_glm_unsmoothed/sub-*/beta_*.nii`
- Ensure beta images are named correctly (SPM12 default naming)

**"Zero variance voxels warning"**
- This is expected behavior; voxels with zero variance are excluded before PCA
- If all voxels in an ROI have zero variance, check GLM results for that subject

**"Singular matrix error in PCA"**
- Occurs when n_stimuli < n_voxels in an ROI
- PR computation handles this by retaining min(n_stimuli-1, n_voxels) components
- Check that you have 40 beta images per subject

**Import errors**
- Run from repository root (not from `chess-manifold/`)
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Citation

If you use this analysis in your work, please cite:

```
[Your paper citation here]
```

## Related Analyses

- **MVPA RSA** (`chess-mvpa/`): Neural RSA examining representational geometry
- **Univariate ROI analysis** (`chess-supplementary/univariate-rois/`): Traditional activation-based ROI analysis for comparison
- **RSA ROI summary** (`chess-supplementary/rsa-rois/`): ROI-level RSA results

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
