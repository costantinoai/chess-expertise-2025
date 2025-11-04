# Univariate ROI Summary (Glasser-180)

## Overview

This supplementary analysis summarizes subject-level univariate contrast maps within the 180 bilateral Glasser cortical ROIs and performs expert vs. novice group comparisons per ROI. Two first-level contrasts are analyzed: (1) Checkmate > Non-checkmate, and (2) All > Rest. Each ROI is bilateral, averaging left and right hemisphere voxels.

## Methods

### Rationale

While whole-brain SPM analyses provide voxelwise statistical maps, ROI-level summaries facilitate anatomical interpretation and comparison with multivariate analyses.

### Data Sources

**Participants**: N=40 (20 experts, 20 novices)
**Atlas**: Glasser-180 bilateral volumetric atlas (MNI152NLin2009cAsym space)
**Input**: Subject-level first-level SPM contrast images (smoothed 4mm FWHM):
- con_0001: Checkmate > Non-checkmate
- con_0002: All > Rest

### Procedure

1. Load Glasser-180 bilateral atlas and ROI metadata (180 ROIs)
2. For each subject and contrast:
   - Load volumetric contrast image from SPM first-level analysis
   - Extract mean contrast value across voxels within each of 180 bilateral ROIs using NiftiLabelsMasker
3. For each contrast:
   - Form expert and novice matrices (subjects × 180 ROIs)
   - Run Welch's t-tests per ROI comparing expert vs novice means
   - Apply Benjamini-Hochberg FDR correction across 180 tests (α=0.05)
   - Compute per-group descriptive means and 95% CIs per ROI

### Statistical Tests

- **Welch two-sample t-test** (unequal variances) per ROI
- **FDR correction** (Benjamini-Hochberg) at α=0.05 across 180 ROIs
- **95% CIs** for group differences

## Dependencies

- Python 3.8+
- numpy, pandas, scipy
- nilearn (NiftiLabelsMasker for ROI extraction)
- statsmodels (for FDR correction)

See `requirements.txt` in the repository root for complete dependencies.

## Data Requirements

### Input Files

- **SPM contrast images**: `BIDS/derivatives/SPM/GLM-smooth4/sub-*/exp/con_*.nii.gz`
- **Atlas**: `rois/glasser180/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-180_bilateral_resampled.nii.gz`
- **ROI metadata**: `rois/glasser180/region_info.tsv`
- **Participant data**: `BIDS/participants.tsv`

### Data Location

Set paths in `common/constants.py` (derived from `_EXTERNAL_DATA_ROOT`):

```python
SPM_GLM_SMOOTH4 = BIDS_ROOT / "derivatives" / "SPM" / "GLM-smooth4"
ROI_GLASSER_180_ATLAS = ROI_GLASSER_180
```

## Running the Analysis

### Step 1: Run ROI Summary Analysis

```bash
# From repository root
python chess-supplementary/univariate-rois/01_univariate_roi_summary.py
```

**Outputs** (saved to `chess-supplementary/univariate-rois/results/<timestamp>_univariate_rois/`):
- `univ_subject_roi_means_{contrast}.tsv`: Subject × ROI tables per contrast
- `univ_group_stats.pkl`: Per-contrast Welch statistics and descriptives
- `01_univariate_roi_summary.py`: Copy of the analysis script

**Expected runtime**: ~2-5 minutes

## Key Results

**Significant ROIs**: ROIs surviving FDR correction show reliable expertise-related differences in univariate activations
**Spatial patterns**: Identify anatomical regions where experts and novices differ in task-related activations

## File Structure

```
chess-supplementary/univariate-rois/
├── README.md                              # This file
├── 01_univariate_roi_summary.py           # Main ROI summary analysis
├── DISCREPANCIES.md                       # Notes on analysis discrepancies
├── modules/
│   ├── __init__.py
│   └── io.py                              # Contrast map loading utilities
└── results/                               # Analysis outputs (timestamped)
    └── <timestamp>_univariate_rois/
        ├── *.tsv                          # Subject × ROI tables
        └── *.pkl                          # Statistical results
```

## Troubleshooting

**"FileNotFoundError: SPM contrast images not found"**
- Run SPM first-level GLM analysis first
- Verify GLM directory and contrast image naming (con_0001.nii, con_0002.nii)

**"FileNotFoundError: Atlas not found"**
- Verify Glasser-180 atlas path in `common/constants.py`

## Citation

If you use this analysis in your work, please cite:

```
[Your paper citation here]
```

## Related Analyses

- **SPM first-level GLM**: Source of contrast images
- **RSA ROI summary** (`chess-supplementary/rsa-rois/`): Parallel analysis for RSA searchlight maps
- **Manifold PR analysis** (`chess-manifold/`): Uses similar ROI-based approach

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
