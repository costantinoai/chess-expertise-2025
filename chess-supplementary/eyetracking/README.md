# Eye-Tracking Decoding Analysis

## Overview

This analysis tests whether eye-tracking time-series features discriminate chess experts from novices during the fMRI task. Two feature sets are evaluated independently: (1) two-dimensional gaze coordinates (x, y), and (2) displacement from screen center. Linear support vector machine (SVM) classification is performed with stratified group k-fold cross-validation.

## Methods

### Rationale

Eye movements reflect cognitive processes during chess perception and problem-solving. If experts and novices employ different visual strategies, their gaze patterns should be systematically different and decodable from eye-tracking features.

### Data Sources

**Participants**: N=40 (20 experts, 20 novices), 357 total runs
**Data**: Eye-tracking TSV files in BIDS derivatives format
**Features**:
- **xy**: Two-dimensional gaze coordinates (x_coordinate, y_coordinate)
- **displacement**: Distance from screen center at each timepoint

### Feature Preparation

1. Load all eye-tracking TSV files and corresponding JSON metadata
2. For each feature set (xy, displacement):
   - Truncate runs to common length (minimum number of timepoints across all runs)
   - Flatten each run into a fixed-length feature vector
   - Result: Matrix of runs × features

### Classification Procedure

**Classifier**: Linear SVM with standardization preprocessing
**Cross-validation**: StratifiedGroupKFold (k=20 folds, group=subject)
- Stratification ensures balanced expert/novice representation in each fold
- Grouping ensures all runs from the same subject stay together (prevents data leakage)

**Metrics**:
- Fold accuracies, mean accuracy, balanced accuracy, F1 score
- ROC curve and AUC
- 95% Confidence interval via Student's t-distribution

**Statistical test**: One-sample t-test testing whether mean fold accuracy differs from chance (0.5)

## Dependencies

- Python 3.8+
- numpy, pandas, scipy
- scikit-learn (for SVM, cross-validation, preprocessing)
- matplotlib, seaborn (for plotting)

See `requirements.txt` in the repository root for complete dependencies.

## Data Requirements

### Input Files

- **Eye-tracking data**: `data/BIDS/derivatives/eyetracking/sub-*/func/*_eyetrack.tsv`
- **Metadata**: `data/BIDS/derivatives/eyetracking/sub-*/func/*_eyetrack.json`
- **Participant data**: `data/BIDS/participants.tsv`

### Data Location

Set the external data root once in `common/constants.py` (all analysis paths are derived from it):

```python
# Base folder containing BIDS/, rois/, neurosynth/, stimuli/
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
# BIDS_EYETRACK is then derived as part of CONFIG
```

## Running the Analysis

### Step 1: Run Decoding Analysis

```bash
# From repository root
python chess-supplementary/eyetracking/01_eye_decoding.py
```

**Outputs** (saved to `chess-supplementary/eyetracking/results/<timestamp>_eyetracking_decoding/`):
- `results_xy.json`: Metrics and predictions for xy features
- `results_displacement.json`: Metrics and predictions for displacement features
- `fold_accuracies_xy.csv`: Per-fold accuracies for xy
- `fold_accuracies_displacement.csv`: Per-fold accuracies for displacement
- `01_eye_decoding.py`: Copy of the analysis script

**Expected runtime**: ~2-5 minutes

## Key Results

Classification accuracies indicate whether gaze features systematically differ between experts and novices. Significant above-chance accuracy would suggest different visual strategies.

## File Structure

```
chess-supplementary/eyetracking/
├── README.md                              # This file
├── 01_eye_decoding.py                     # Main decoding analysis
├── DISCREPANCIES.md                       # Notes on analysis discrepancies
├── modules/
│   ├── __init__.py
│   └── io.py                              # Eye-tracking data loading
└── results/                               # Analysis outputs (timestamped)
    └── <timestamp>_eyetracking_decoding/
        ├── *.json                         # Results dictionaries
        └── *.csv                          # Fold accuracies
```

