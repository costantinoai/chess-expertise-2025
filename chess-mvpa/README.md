# Multi-Voxel Pattern Analysis (MVPA): RSA and Decoding

## Overview

This analysis examines neural representations of chess positions using two complementary multivariate approaches: representational similarity analysis (RSA) and support vector machine (SVM) decoding. RSA tests whether neural dissimilarity patterns correlate with theoretical model dissimilarities, while decoding assesses whether spatial patterns of brain activity encode task-relevant categorical information. Both analyses are performed on 22 bilateral cortical regions to test whether chess expertise modulates neural encoding of chess positions.

## Methods

### Rationale

Multivariate pattern analysis (MVPA) moves beyond traditional univariate approaches by examining distributed patterns of activity across voxels. RSA quantifies the geometry of neural representations by comparing neural dissimilarity matrices (RDMs) with theoretical model RDMs. Decoding assesses the linear separability of neural patterns corresponding to different stimulus categories. Together, these approaches provide complementary views of how chess expertise shapes neural representations.

### Data Sources

**Participants**: N=40 (20 experts, 20 novices)
**Stimuli**: 40 chess board positions (20 checkmate, 20 non-checkmate)
**ROIs**: 22 bilateral cortical regions from the Glasser multimodal parcellation
**Input data**: Unsmoothed trial-wise beta estimates from SPM12 first-level GLMs

### Model RDMs

Three theoretical model RDMs were used as prediction targets:

1. **Checkmate status**: Binary RDM (0 if same status, 1 if different)
2. **Strategy type**: Categorical RDM based on chess strategies
3. **Visual similarity**: Perceptual feature-based dissimilarity

These models reflect high-level conceptual dimensions (checkmate, strategy) and lower-level perceptual features (visual similarity).

### Subject-Level RSA (MATLAB/CoSMoMVPA)

**Implementation**: CoSMoMVPA toolbox in MATLAB R2022b

**Neural RDM construction**:
1. Extract multivoxel activity patterns from each ROI for all 40 conditions
2. Average beta estimates across runs to increase reliability
3. Apply voxel-wise mean centering
4. Compute pairwise correlation distances (1 - r) between all condition pairs
5. Result: 40×40 neural RDM per subject per ROI

**RSA correlation**:
- Neural RDM correlated with each model RDM using Pearson correlation
- Function: `cosmo_target_dsm_corr_measure` in CoSMoMVPA
- Output: One correlation value per subject, ROI, and model RDM

### Subject-Level Decoding (MATLAB/CoSMoMVPA)

**Implementation**: CoSMoMVPA toolbox in MATLAB R2022b/R2024b

**Classification procedure**:
1. Extract multivoxel patterns from each ROI for all 40 conditions
2. Construct classification targets from categorical regressors (checkmate, strategy, visual)
3. Partition data using leave-one-run-out cross-validation
4. Balance folds to ensure equal class representation (`cosmo_balance_partitions`)
5. Train linear SVM classifier (`cosmo_classify_svm`) on each fold
6. Compute mean classification accuracy across folds

**Chance levels**:
- Binary classification (e.g., checkmate vs non-checkmate): 0.5
- Multi-class classification: 1 / n_classes

**ROI exclusion**: ROIs with fewer than 10 usable voxels were excluded to ensure minimum feature dimensionality.

### Whole-Brain Searchlight RSA (MATLAB/CoSMoMVPA)

To extend analyses beyond predefined ROIs, whole-brain searchlight RSA was conducted:

**Searchlight parameters**:
- Spherical neighborhoods with radius = 3 voxels (6 mm)
- At each voxel, local RDM computed from surrounding multivoxel pattern
- Local RDM compared (Pearson correlation) to each model RDM
- Output: One RSA map per subject and model

### Group-Level Statistical Testing (Python)

Subject-level RSA correlations and decoding accuracies were aggregated and tested at the group level.

**Three statistical tests per ROI**:

1. **Experts vs Chance (zero for RSA, theoretical chance for decoding)**:
   - One-sample one-tailed t-test (greater)
   - Null hypothesis: μ_expert ≤ chance
   - Implementation: `scipy.stats.ttest_1samp` with `alternative='greater'`

2. **Novices vs Chance**:
   - One-sample one-tailed t-test (greater)
   - Null hypothesis: μ_novice ≤ chance
   - Implementation: `scipy.stats.ttest_1samp` with `alternative='greater'`

3. **Experts vs Novices**:
   - Welch two-sample two-tailed t-test
   - Null hypothesis: μ_expert = μ_novice
   - Implementation: `scipy.stats.ttest_ind` with `equal_var=False` (allows unequal variances)

**False Discovery Rate (FDR) Correction**:
- Applied across 22 ROIs using Benjamini-Hochberg procedure (α=0.05)
- Implementation: `statsmodels.stats.multitest.multipletests` with `method='fdr_bh'`
- Performed independently for each test type and each model target
- ROIs considered significant if FDR-corrected q-value < 0.05

### Statistical Assumptions and Limitations

- **Normality**: t-tests assume normally distributed values within each group and ROI. With n=20 per group, the central limit theorem provides robustness to moderate deviations
- **Independence**: Subject-level values are assumed independent. Scanner drift and shared task strategies may introduce correlated noise
- **Equal variances**: Welch's t-test relaxes the equal variance assumption for group comparisons
- **Spatial dependence**: ROIs are anatomically adjacent and functionally connected, violating independence. FDR correction partially accounts for this by controlling the expected proportion of false positives
- **Fisher z-transformation**: RSA correlation coefficients were not Fisher z-transformed for group-level testing, following standard practice when sample correlations are not strongly skewed

## Dependencies

**MATLAB**:
- MATLAB R2022b or later
- CoSMoMVPA toolbox (https://github.com/CoSMoMVPA/CoSMoMVPA)
- SPM12 (for loading NIfTI files and GLM outputs)

**Python**:
- Python 3.8+
- numpy, pandas, scipy
- statsmodels (for FDR correction)
- matplotlib, seaborn (for plotting)
- nibabel (for loading NIfTI files)

See `requirements.txt` in the repository root for complete Python dependencies.

## Data Requirements

### Input Files

**For subject-level analysis (MATLAB)**:
- **SPM GLM outputs**: `BIDS/derivatives/SPM/GLM-unsmoothed/sub-*/exp/SPM.mat`
  - Trial-wise beta estimates (unsmoothed)
  - One beta image per condition per run
- **Atlas**: `rois/glasser22/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-22_bilateral_resampled.nii.gz`
- **ROI metadata**: `rois/glasser22/region_info.tsv`
  - Columns: `index`, `name`, `hemisphere`

**For group-level analysis (Python)**:
- **RSA correlations**: `BIDS/derivatives/mvpa-rsa/sub-*/sub-*_space-MNI152NLin2009cAsym_roi-glasser_rdm.tsv`
- **Decoding accuracies**: `BIDS/derivatives/mvpa-decoding/sub-*/sub-*_space-MNI152NLin2009cAsym_roi-glasser_accuracy.tsv`
- **Participant data**: `BIDS/participants.tsv`
  - Columns: `participant_id`, `group` (expert/novice)
- **Stimulus metadata**: `stimuli/stimuli.tsv`
  - Required for deriving chance levels for decoding targets

### Data Location

Set the external data root once in `common/constants.py` (all analysis paths are derived from it):

```python
# Base folder containing BIDS/, rois/, neurosynth/, stimuli/
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
# BIDS_ROOT and derivative paths (e.g., mvpa-rsa, mvpa-decoding) are built from this
```

Key derived paths used here (from `CONFIG`):
- `BIDS_MVPA_RSA`: Path to RSA correlation results
- `BIDS_MVPA_DECODING`: Path to decoding accuracy results
- `SPM_GLM_UNSMOOTHED`: Path to unsmoothed GLM directory

**MATLAB paths** can be overridden using environment variables:
- `CHESS_BIDS_DERIVATIVES`: BIDS derivatives root
- `CHESS_ROI_ATLAS_22`: Path to atlas NIfTI
- `CHESS_ROI_TSV_22`: Path to ROI metadata TSV

## Running the Analysis

### Step 1: Subject-Level MVPA (MATLAB)

```matlab
% From MATLAB prompt, cd to chess-mvpa/ directory
cd /path/to/chess-expertise-2025/chess-mvpa/

% Run ROI-based RSA and decoding
01_roi_mvpa_main.m

% Run whole-brain searchlight RSA (optional, computationally intensive)
04_searchlight_rsa.m
```

**Outputs**:
- `BIDS/derivatives/mvpa-rsa/sub-*/sub-*_space-MNI152NLin2009cAsym_roi-glasser_rdm.tsv`
  - Subject-level RSA correlations (one row per model target, one column per ROI)
- `BIDS/derivatives/mvpa-decoding/sub-*/sub-*_space-MNI152NLin2009cAsym_roi-glasser_accuracy.tsv`
  - Subject-level decoding accuracies (one row per classification target, one column per ROI)

**Expected runtime**:
- ROI-based analysis: ~2-5 minutes per subject (total ~2-4 hours for 40 subjects)
- Searchlight analysis: ~30-60 minutes per subject (highly variable, parallelizable)

### Step 2: Group-Level RSA Analysis (Python)

```bash
# From repository root
python chess-mvpa/02_mvpa_group_rsa.py
```

**Outputs** (saved to `chess-mvpa/results/<timestamp>_mvpa_group_rsa/`):
- `<target>_experts_vs_chance.csv`: Expert vs zero statistics per ROI
- `<target>_novices_vs_chance.csv`: Novice vs zero statistics per ROI
- `<target>_experts_vs_novices.csv`: Group comparison statistics per ROI
- `mvpa_group_stats.pkl`: Complete results dictionary (for plotting scripts)
- `02_mvpa_group_rsa.py`: Copy of the analysis script

**Expected runtime**: ~30 seconds

### Step 3: Group-Level Decoding Analysis (Python)

```bash
python chess-mvpa/03_mvpa_group_decoding.py
```

**Outputs** (saved to `chess-mvpa/results/<timestamp>_mvpa_group_decoding/`):
- `<target>_experts_vs_chance.csv`: Expert vs chance statistics per ROI
- `<target>_novices_vs_chance.csv`: Novice vs chance statistics per ROI
- `<target>_experts_vs_novices.csv`: Group comparison statistics per ROI
- `mvpa_group_stats.pkl`: Complete results dictionary (for plotting scripts)
- `03_mvpa_group_decoding.py`: Copy of the analysis script

**Expected runtime**: ~30 seconds

### Step 4: Generate Tables

```bash
# RSA tables
python chess-mvpa/81_table_mvpa_rsa.py

# Decoding tables
python chess-mvpa/82_table_mvpa_decoding.py
```

**Outputs** (saved to `chess-mvpa/results/<latest>/tables/`):
- `mvpa_rsa_<target>.tex`: LaTeX tables for each model target
- `mvpa_rsa_<target>.csv`: CSV tables for each model target
- `mvpa_decoding_<target>.tex`: LaTeX tables for each classification target
- `mvpa_decoding_<target>.csv`: CSV tables for each classification target

### Step 5: Generate Figures

```bash
# RSA figures
python chess-mvpa/92_plot_mvpa_rsa.py

# Decoding figures
python chess-mvpa/93_plot_mvpa_decoding.py
```

**Outputs** (saved to `chess-mvpa/results/<latest>/figures/`):
- Individual axes as SVG/PDF: `mvpa_rsa_<target>_<panel>.svg`, etc.
- Complete panels: `panels/mvpa_rsa_panel.pdf`, `panels/mvpa_decoding_panel.pdf`

**Note**: If `ENABLE_PYLUSTRATOR=True` in `common/constants.py`, this will open an interactive layout editor. Set to `False` for automated figure generation.

## Key Results

### RSA Results

**Strategy dimension**:
- Experts show robust positive correlations across parietal, frontal, and temporal regions
- Novices show weaker or no significant correlations
- Group differences suggest expertise modulates strategy-related representational geometry

**Checkmate dimension**:
- Both groups show positive correlations in high-level associative cortices
- Experts show stronger effects in parietal and prefrontal regions
- Pattern consistent with goal-related processing

**Visual similarity**:
- Limited effects in both groups
- Some correlations in posterior visual regions for experts
- Suggests high-level categorical dimensions dominate over low-level visual features

### Decoding Results

**Strategy decoding**:
- Experts: Robust above-chance accuracy in superior parietal lobule, inferior parietal lobule, dorsolateral prefrontal cortex, premotor cortex, and dorsal stream visual areas
- Novices: Weaker or absent effects
- Spatial profile indicates abstract relational processing in frontoparietal network

**Checkmate decoding**:
- High accuracy in dorsolateral prefrontal cortex, inferior parietal cortex, superior parietal cortex
- Additional effects in lateral temporal cortex, premotor areas, posterior cingulate cortex
- Goal-related distinctions reliably encoded in higher-order associative cortices

**Visual similarity decoding**:
- Limited and non-significant effects in most regions
- Low accuracy in early visual cortex
- Confirms visual features alone do not strongly drive neural pattern separability

### Convergence Between RSA and Decoding

- Both approaches identify similar frontoparietal networks for strategy and checkmate processing
- RSA reveals representational geometry; decoding reveals linear separability
- Complementary methods provide converging evidence for expertise-related neural differences

## File Structure

```
chess-mvpa/
├── README.md                          # This file
├── 01_roi_mvpa_main.m                 # MATLAB: Subject-level ROI RSA and decoding
├── 02_mvpa_group_rsa.py               # Python: Group-level RSA statistics
├── 03_mvpa_group_decoding.py          # Python: Group-level decoding statistics
├── 04_searchlight_rsa.m               # MATLAB: Whole-brain searchlight RSA
├── 81_table_mvpa_rsa.py               # LaTeX/CSV table generation (RSA)
├── 82_table_mvpa_decoding.py          # LaTeX/CSV table generation (decoding)
├── 92_plot_mvpa_rsa.py                # Figure generation (RSA)
├── 93_plot_mvpa_decoding.py           # Figure generation (decoding)
├── METHODS.md                         # Detailed methods from manuscript
├── RESULTS.md                         # Detailed results summary
├── DISCREPANCIES.md                   # Notes on analysis discrepancies
├── modules/
│   ├── __init__.py
│   ├── mvpa_io.py                     # Loading subject-level MVPA results
│   ├── mvpa_group.py                  # Group-level statistical tests
│   └── plotting.py                    # Plotting utilities
├── local/                             # Local data preparation scripts
└── results/                           # Analysis outputs (timestamped)
    ├── <timestamp>_mvpa_group_rsa/
    │   ├── *.csv                      # Statistical results per target
    │   ├── *.pkl                      # Python objects
    │   ├── tables/                    # LaTeX tables
    │   └── figures/                   # Publication figures
    └── <timestamp>_mvpa_group_decoding/
        ├── *.csv                      # Statistical results per target
        ├── *.pkl                      # Python objects
        ├── tables/                    # LaTeX tables
        └── figures/                   # Publication figures
```

## Troubleshooting

### Common Issues

**MATLAB: "CoSMoMVPA not found"**
- Add CoSMoMVPA to MATLAB path: `addpath(genpath('/path/to/CoSMoMVPA'))`
- Or set `COSMOMVPA_PATH` environment variable

**MATLAB: "SPM.mat not found"**
- Verify GLM directory structure: `BIDS/derivatives/SPM/GLM-unsmoothed/sub-*/exp/SPM.mat`
- Check path in script line 41: `glmRoot` variable

**MATLAB: "ROI atlas not found"**
- Verify atlas exists: `data/BIDS/derivatives/rois/glasser22/*.nii`
- Override with environment variable `CHESS_ROI_ATLAS_22`

**Python: "No MVPA results found"**
- Run MATLAB subject-level analysis first (step 1)
- Verify TSV files exist in `BIDS/derivatives/mvpa-rsa/` and `mvpa-decoding/`
- Check paths in `common/constants.py`: `BIDS_MVPA_RSA` and `BIDS_MVPA_DECODING`

**Python: "Missing participant metadata"**
- Ensure `BIDS/participants.tsv` exists with `participant_id` and `group` columns

**MATLAB: Empty dataset warning**
- Check that beta images exist in SPM directory
- Verify SPM.mat contains condition labels matching expected format

**Import errors (Python)**
- Run from repository root (not from `chess-mvpa/`)
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Citation

If you use this analysis in your work, please cite:

```
[Your paper citation here]
```

## Related Analyses

- **Behavioral RSA** (`chess-behavioral/`): Behavioral similarity judgments using same model RDMs
- **RSA ROI summary** (`chess-supplementary/rsa-rois/`): Detailed ROI-level RSA results and visualization
- **MVPA finer resolution** (`chess-supplementary/mvpa-finer/`): RSA and decoding with finer categorical distinctions (checkmate boards only)
- **RDM intercorrelation** (`chess-supplementary/rdm-intercorrelation/`): Orthogonality analysis of model RDMs
- **Univariate ROI analysis** (`chess-supplementary/univariate-rois/`): Traditional activation-based analysis for comparison

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
