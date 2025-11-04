# RDM Intercorrelation Analysis

## Overview

This analysis quantifies the relationships among three theoretical model representational dissimilarity matrices (RDMs) derived from chess stimulus features. Through pairwise correlations, partial correlations, and hierarchical variance partitioning, we assess shared and unique variance explained by each model RDM. Understanding RDM intercorrelations is critical for interpreting which aspects of stimulus structure drive behavioral and neural similarity patterns.

## Methods

### Rationale

Model RDMs may share representational structure if they capture overlapping aspects of stimulus organization. Quantifying these intercorrelations helps determine whether behavioral/neural RDMs reflect independent dimensions or correlated features.

### Model RDMs

Three theoretical RDMs derived from stimulus annotations (N=40 stimuli, 780 unique pairwise dissimilarities per RDM):

1. **Check Status**: Binary categorical RDM (0 if same status, 1 if different)
   - Balanced design: 20 checkmate, 20 non-checkmate

2. **Strategy**: Binary categorical RDM (0 if same strategy, 1 if different)
   - Five strategy categories (queen-rook, supported attacks, minor piece nets, bishop-driven, one-move mates)

3. **Visual Similarity**: Binary categorical RDM (0 if same visual cluster, 1 if different)
   - Four visual clusters based on piece configuration patterns

### Statistical Analyses

**Pairwise RDM Correlations**:
- Spearman rank correlations between all pairs of model RDMs
- Uses vectorized upper-triangle entries (780 pairwise dissimilarities per RDM, excluding diagonal)
- Quantifies overlap in representational structure

**Partial Correlations**:
- For each target-predictor pair, compute Spearman partial correlation controlling for all remaining model RDMs
- Isolates unique relationship after accounting for shared variance with other models
- Procedure:
  1. Regress target RDM on covariate RDMs (OLS), extract residuals
  2. Regress predictor RDM on same covariates, extract residuals
  3. Compute Spearman correlation between residual vectors
  4. P-values from t-distribution with df = n_observations − n_covariates − 2

**Variance Partitioning**:
- Hierarchical decomposition of variance (R²) explained by each predictor
- Components: unique variance per predictor, shared variance, unexplained variance
- Method:
  1. Fit full model: target ~ all predictors (OLS)
  2. Compute R²_full
  3. For each predictor k:
     - Fit reduced model: target ~ all predictors except k
     - Unique variance for k: R²_unique(k) = R²_full − R²_reduced(k)
  4. Shared variance: R²_shared = R²_full − Σ R²_unique
  5. Unexplained variance: 1 − R²_full

All regressions use z-scored predictors and targets for comparable coefficients.

## Dependencies

- Python 3.8+
- numpy, pandas, scipy
- statsmodels (for partial correlations)
- matplotlib, seaborn (for plotting)

See `requirements.txt` in the repository root for complete dependencies.

## Data Requirements

### Input Files

- **Stimulus metadata**: `stimuli/stimuli.tsv`
  - Required columns: `stim_id`, `check`, `strategy`, `visual`

### Data Location

Set the external data root once in `common/constants.py` (all analysis paths are derived from it):

```python
# Base folder containing BIDS/, rois/, neurosynth/, stimuli/
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
```

## Running the Analysis

### Step 1: Run RDM Intercorrelation Analysis

```bash
# From repository root
python chess-supplementary/rdm-intercorrelation/01_rdm_intercorrelation.py
```

**Outputs** (saved to `chess-supplementary/rdm-intercorrelation/results/<timestamp>_rdm_intercorr/`):
- `pairwise_correlations.csv`: Spearman correlations between all RDM pairs
- `partial_correlations.csv`: Partial correlations controlling for other RDMs
- `variance_partitioning.csv`: Unique, shared, and unexplained variance per target
- `rdm_intercorr_results.pkl`: Complete results dictionary
- `01_rdm_intercorrelation.py`: Copy of the analysis script

**Expected runtime**: ~30 seconds

### Step 2: Generate Figures

```bash
python chess-supplementary/rdm-intercorrelation/91_plot_rdm_intercorr.py
```

**Outputs** (saved to `chess-supplementary/rdm-intercorrelation/results/<latest>/figures/`):
- `rdm_intercorr_panel.pdf`: Combined visualization showing pairwise correlations, partial correlations, and variance partitioning

## Key Results

**Pairwise correlations**: Quantify overlap between model RDMs
**Partial correlations**: Reveal unique relationships after controlling for other models
**Variance partitioning**: Decompose explained variance into unique and shared components

**Interpretation**: Low intercorrelations suggest orthogonal dimensions; high shared variance indicates correlated features.

## File Structure

```
chess-supplementary/rdm-intercorrelation/
├── README.md                              # This file
├── 01_rdm_intercorrelation.py             # Main analysis
├── 91_plot_rdm_intercorr.py               # Figure generation
├── DISCREPANCIES.md                       # Notes on analysis discrepancies
├── modules/
│   ├── __init__.py
│   └── rdm_utils.py                       # RDM correlation and partitioning utilities
└── results/                               # Analysis outputs (timestamped)
    └── <timestamp>_rdm_intercorr/
        ├── *.csv                          # Statistical results
        ├── *.pkl                          # Python objects
        └── figures/                       # Publication figures
```

