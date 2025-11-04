# RSA Searchlight ROI Summary (Glasser-180)

## Overview

This supplementary analysis summarizes whole-brain RSA searchlight correlation maps within the 180 bilateral Glasser cortical ROIs and performs expert vs. novice group comparisons per ROI. Three model RDMs are analyzed: Visual Similarity, Strategy, and Checkmate. Each ROI is bilateral, averaging left and right hemisphere voxels.

## Methods

### Rationale

While searchlight RSA provides whole-brain voxelwise maps, ROI-level summaries facilitate anatomical interpretation and comparison with other analyses.

### Data Sources

**Participants**: N=40 (20 experts, 20 novices)
**Atlas**: Glasser-180 bilateral volumetric atlas (MNI152NLin2009cAsym space)
**Input**: Subject-level RSA searchlight correlation maps for three model RDMs:
- Visual Similarity
- Strategy
- Checkmate

### Procedure

1. Load Glasser-180 bilateral atlas and ROI metadata (180 ROIs)
2. For each subject and RSA target:
   - Load volumetric r-map from searchlight analysis
   - Extract mean correlation across voxels within each of 180 bilateral ROIs using NiftiLabelsMasker
3. For each target:
   - Form expert and novice matrices (subjects × 180 ROIs)
   - Run Welch's t-tests per ROI comparing expert vs novice means
   - Apply Benjamini-Hochberg FDR correction across 180 tests (α=0.05)
   - Compute per-group descriptive means and 95% CIs per ROI

### Statistical Tests

- **Welch two-sample t-test** (unequal variances) per ROI
- **FDR correction** (Benjamini-Hochberg) at α=0.05 across 180 ROIs
- **95% CIs** for group differences

## Data Requirements

### Input Files

- **RSA searchlight maps**: `BIDS/derivatives/rsa_searchlight/sub-*/sub-*_desc-searchlight_<target>_stat-r_map.nii.gz`
- **Atlas**: `rois/glasser180/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-180_bilateral_resampled.nii.gz`
- **ROI metadata**: `rois/glasser180/region_info.tsv`
- **Participant data**: `BIDS/participants.tsv`

### Data Location

Set paths in `common/constants.py`:

```python
BIDS_RSA_SEARCHLIGHT = BIDS_ROOT / "derivatives" / "rsa_searchlight"
ROI_GLASSER_180_ATLAS = ROI_GLASSER_180 / "tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-180_bilateral_resampled.nii.gz"
```

## Running the Analysis

### Step 1: Run ROI Summary Analysis

```bash
# From repository root
python chess-supplementary/rsa-rois/01_rsa_roi_summary.py
```

**Outputs** (saved to `chess-supplementary/rsa-rois/results/rsa_rois/`):
- `rsa_subject_roi_means_{target}.tsv`: Subject × ROI tables per target
- `rsa_group_stats.pkl`: Per-target Welch statistics and descriptives
- `01_rsa_roi_summary.py`: Copy of the analysis script

**Expected runtime**: ~5-10 minutes

## Key Results

**Significant ROIs**: ROIs surviving FDR correction show reliable expertise-related differences in RSA correlations
**Spatial patterns**: Identify anatomical regions where model RDMs differentially predict neural geometry in experts vs novices

## File Structure

```
chess-supplementary/rsa-rois/
├── README.md                              # This file
├── 01_rsa_roi_summary.py                  # Main ROI summary analysis
├── DISCREPANCIES.md                       # Notes on analysis discrepancies
├── modules/
│   ├── __init__.py
│   └── io.py                              # RSA map loading utilities
└── results/
    └── rsa_rois/
        ├── *.tsv                          # Subject × ROI tables
        └── *.pkl                          # Statistical results
```
