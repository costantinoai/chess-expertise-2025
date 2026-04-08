# Neurosynth Meta-Analytic Correlation Analysis

## Overview

This analysis places observed expertise-related brain activation differences in a broader functional context by correlating our group-level statistical maps with large-scale functional networks derived from automated meta-analysis. Specifically, we correlate (1) univariate GLM t-maps (Experts > Novices) and (2) RSA searchlight group contrast maps with term-based association maps from the Neurosynth database. This data-driven approach identifies which cognitive functions are preferentially associated with regions showing expertise effects.

## Required bundles

- `01_univariate_neurosynth.py` reads SPM smoothed group t-maps from `derivatives/fmriprep_spm-smoothed/group/` → needs **A** (core) + **D** (spm).
- `02_rsa_neurosynth.py` reads per-subject searchlight RSA maps from `derivatives/fmriprep_spm-unsmoothed_searchlight-rsa/` → needs **A** (core) + **E** (analyses).
- `81/82` (tables) and `91/92` (figures) only consume the outputs of 01/02 from the repo `results/` tree (no extra bundle).

## Methods

### Rationale

Rather than relying solely on anatomical labels or reverse inference to interpret functional significance, we leverage the Neurosynth database—a large-scale automated meta-analysis platform containing >10,000 fMRI studies. By correlating our group-level statistical maps with Neurosynth term maps, we can identify which cognitive functions (e.g., "working memory", "visual attention") are most strongly associated with regions showing expertise-related differences.

### Neurosynth Term Maps

Seven cognitive terms were selected *a priori* for their relevance to chess expertise and cognitive expertise more broadly:

1. **Working memory**: Proxy for frontoparietal multiple-demand network; reflects efficient information storage and manipulation in expert performance
2. **Memory retrieval**: Expert-specific mechanisms for efficient information access
3. **Navigation**: Spatial systems processing complex board configurations
4. **Language**: Verbal and propositional reasoning processes, thought to be more prominent in novices
5. **Face recognition**: Given prior evidence implicating fusiform cortex and FFA in expertise-related visual processing
6. **Early visual processing**: Assessment of reliance on low-level visual features
7. **Object recognition**: Shape-based processing and perceptual expertise in fine-grained discrimination

**Term map preparation**:
- Downloaded z-scored association test maps from Neurosynth database
- Resampled to resolution of group-level statistical images (nearest-neighbor interpolation)
- Restricted to gray matter (ICBM152 2009c probabilistic gray matter template thresholded at >0.5)
- Gray matter mask includes both cortical and subcortical regions

File locations (under external data root):
- `BIDS/sourcedata/atlases/neurosynth/terms/1_working memory.nii.gz`
- `BIDS/sourcedata/atlases/neurosynth/terms/2_navigation.nii.gz`
- `BIDS/sourcedata/atlases/neurosynth/terms/3_memory retrieval.nii.gz`
- `BIDS/sourcedata/atlases/neurosynth/terms/4_language network.nii.gz`
- `BIDS/sourcedata/atlases/neurosynth/terms/5_object recognition.nii.gz`
- `BIDS/sourcedata/atlases/neurosynth/terms/6_face recognition.nii.gz`
- `BIDS/sourcedata/atlases/neurosynth/terms/7_early visual.nii.gz`

### Univariate Analysis: T-Map Correlations

**Input data**:
- Group-level t-maps from SPM12 second-level GLMs
- Contrasts: "All boards > Baseline" or "Checkmate > Non-checkmate" (first-level), "Experts > Novices" (second-level)
- N=40 participants (20 experts, 20 novices)
- Degrees of freedom: 38 (n_subjects − 2 for two-sample t-test)
- Smoothing: 4mm FWHM Gaussian kernel

Paths (under external data root):
- `BIDS/derivatives/fmriprep_spm-smoothed/group/spmT_exp_gt_nonexp_all_gt_rest.nii.gz`
- `BIDS/derivatives/fmriprep_spm-smoothed/group/spmT_exp_gt_nonexp_check_gt_nocheck.nii.gz`

**T-to-Z conversion**:
Group-level t-maps were converted to signed two-tailed z-scores while preserving the sign (direction) of the effect:

1. Compute two-tailed p-value from t-statistic using t-distribution (df=38)
2. Convert p-value to two-tailed z-score using inverse cumulative standard normal distribution
3. Assign sign of original t-statistic to z-score

This standardizes effect magnitudes across contrasts and allows comparison with Neurosynth z-score maps.

**Sign-specific map splitting**:
- Z+ (positive z-values, zeros elsewhere): Regions with stronger expert activations
- Z− (absolute negative z-values, zeros elsewhere): Regions with stronger novice activations

This allows separate functional interpretation of expertise-enhanced vs expertise-reduced regions.

**Term correlation**:
- For each sign-specific map (Z+, Z−), compute spatial Pearson correlation with each Neurosynth term map
- Only non-zero voxels in z-map included (focus on regions with group differences)
- Compute correlation difference (r_pos − r_neg) to identify terms differentially associated with expert-enhanced vs expert-reduced regions
- Positive differences indicate terms more strongly associated with regions where experts show higher activations

### RSA Searchlight Analysis: Group Contrast Map Correlations

**Input data**:
- Subject-level RSA searchlight maps (correlation coefficients at each voxel)
- Models: Checkmate status, strategy type, visual similarity
- Searchlight radius: 3 voxels (6mm)
- N=40 participants (20 experts, 20 novices)

**Second-level group analysis**:
1. **Fisher z-transformation**: Subject-level correlation maps Fisher z-transformed (z = arctanh(r)) to stabilize variance for group-level inference
2. **Group GLM**: Fit second-level GLM with two regressors:
   - Intercept (group mean)
   - Group contrast (experts = +1, novices = −1)
   - Implementation: `nilearn.glm.second_level.SecondLevelModel`
3. **Group contrast map**: Extract z-score map for Experts > Novices contrast
   - Positive z-values: Voxels where experts show stronger RSA correlations
   - Negative z-values: Voxels where novices show stronger correlations
4. **Sign-specific maps**: Split into Z+ and Z− as above

Paths (under external data root):
- `BIDS/derivatives/fmriprep_spm-unsmoothed_searchlight-rsa/sub-*/sub-*_space-MNI152NLin2009cAsym_desc-checkmate_stat-r_searchlight.nii.gz`
- `BIDS/derivatives/fmriprep_spm-unsmoothed_searchlight-rsa/sub-*/sub-*_space-MNI152NLin2009cAsym_desc-strategy_stat-r_searchlight.nii.gz`
- `BIDS/derivatives/fmriprep_spm-unsmoothed_searchlight-rsa/sub-*/sub-*_space-MNI152NLin2009cAsym_desc-visualSimilarity_stat-r_searchlight.nii.gz`

**Term correlation**:
- Same procedure as univariate analysis
- Correlate Z+ and Z− maps with Neurosynth term maps
- Compute correlation differences (r_pos − r_neg)

## Data Requirements

### Input Files

**For univariate analysis**:
- **Group t-maps**: `BIDS/derivatives/fmriprep_spm-smoothed/group/spmT_exp-gt-nonexp_*.nii.gz`
  - SPM12 second-level GLM outputs
  - Contrasts: Experts > Novices for various first-level contrasts
- **Neurosynth term maps**: `BIDS/sourcedata/atlases/neurosynth/terms/*.nii.gz`
  - Z-scored association test maps for cognitive terms
  - Downloaded from https://neurosynth.org/

**For RSA analysis**:
- **Subject-level searchlight maps**: `BIDS/derivatives/fmriprep_spm-unsmoothed_searchlight-rsa/sub-*/sub-*_space-MNI152NLin2009cAsym_desc-<pattern>_stat-r_searchlight.nii.gz`
  - Correlation coefficient maps from whole-brain RSA searchlight
  - Models: checkmate, strategy, visualSimilarity
- **Participant data**: `BIDS/participants.tsv`
  - Columns: `participant_id`, `group` (expert/novice)
- **Neurosynth term maps**: Same as above

## Running the Analysis

### Step 1: Univariate Neurosynth Correlation

```bash
# From repository root
python chess-neurosynth/01_univariate_neurosynth.py
```

**Outputs** (saved to `results/neurosynth/data/`):
- `zmap_<contrast>.nii.gz`: Signed z-score maps (converted from t-maps)
- `<contrast>_term_corr_positive.csv`: Z+ term correlations (term, r)
- `<contrast>_term_corr_negative.csv`: Z− term correlations (term, r)
- `<contrast>_term_corr_difference.csv`: Correlation differences (term, r_pos, r_neg, r_diff)
- `01_univariate_neurosynth.py`: Copy of the analysis script

**Expected runtime**: ~2-3 minutes per contrast

### Step 2: RSA Searchlight Neurosynth Correlation

```bash
python chess-neurosynth/02_rsa_neurosynth.py
```

**Outputs** (saved to `results/neurosynth/data/`):
- `zmap_<pattern>.nii.gz`: Group z-score maps (Experts > Novices) for each RSA model
- `<pattern>_term_corr_positive.csv`: Z+ term correlations
- `<pattern>_term_corr_negative.csv`: Z− term correlations
- `<pattern>_term_corr_difference.csv`: Correlation differences
- `02_rsa_neurosynth.py`: Copy of the analysis script

**Expected runtime**: ~5-10 minutes (includes second-level GLM fitting)

### Step 3: Generate Tables

```bash
# Univariate tables
python chess-neurosynth/81_table_neurosynth_univariate.py

# RSA tables
python chess-neurosynth/82_table_neurosynth_rsa.py
```

**Outputs**:
- Univariate tables → `results/neurosynth/tables/`
  - `neurosynth_univariate_summary.tex`: Combined LaTeX summary table
  - `neurosynth_univariate_summary.csv`: Combined CSV summary table
  - `neurosynth_univariate_full_stats.csv`: Full statistics with CIs and p-values
- RSA tables → `results/neurosynth/tables/`
  - `neurosynth_rsa_summary.tex`: Combined LaTeX summary table
  - `neurosynth_rsa_summary.csv`: Combined CSV summary table
  - `neurosynth_rsa_full_stats.csv`: Full statistics with CIs and p-values

### Step 4: Generate Figures

```bash
# Univariate figures
python chess-neurosynth/91_plot_neurosynth_univariate.py

# RSA figures
python chess-neurosynth/92_plot_neurosynth_rsa.py
```

**Outputs**:
- Univariate figures → `results/neurosynth/figures/`
  - Individual axes as SVG: `neurosynth_univariate__*.svg`
  - Complete panel PDF: `panels/neurosynth_univariate_panel.pdf`
- RSA figures → `results/neurosynth/figures/`
  - Individual axes as SVG: `neurosynth_rsa__*.svg`
  - Complete panel PDF: `panels/neurosynth_rsa_panel.pdf`

**Note**: If `ENABLE_PYLUSTRATOR=True` in `common/constants.py`, this will open an interactive layout editor. Set to `False` for automated figure generation.

## Key Results

### Univariate Correlations

**Positive (Expert > Novice) regions**:
- Strong positive correlations with "working memory" and "memory retrieval" terms
- Moderate correlations with "navigation" and "object recognition"
- Weak correlations with "early visual" processing

**Negative (Novice > Expert) regions**:
- Stronger correlations with "language" processing
- Suggests novices rely more on verbal/propositional reasoning

**Interpretation**: Expert-enhanced regions overlap with frontoparietal multiple-demand network and spatial processing systems, consistent with efficient information storage and spatial reasoning. Novice-enhanced regions overlap with language networks, consistent with verbal reasoning strategies.

### RSA Searchlight Correlations

**Strategy model**:
- Positive correlations with "working memory" and "navigation" terms in expert-enhanced regions
- Reflects abstract relational processing in frontoparietal and spatial networks

**Checkmate model**:
- Strong correlations with "working memory" and "memory retrieval"
- Goal-related processing associated with executive function networks

**Visual similarity model**:
- Limited correlations with any terms
- Confirms low-level visual features do not strongly drive expertise differences

### Convergence with ROI-Based Results

- Meta-analytic correlations validate anatomical interpretations from ROI-based analyses
- Expert-enhanced regions consistently map onto frontoparietal and spatial networks
- Novice-enhanced regions consistently map onto language/semantic networks

## File Structure

```
chess-neurosynth/
├── README.md                             # This file
├── 01_univariate_neurosynth.py           # Univariate GLM t-map correlations
├── 02_rsa_neurosynth.py                  # RSA searchlight group contrast correlations
├── 81_table_neurosynth_univariate.py     # LaTeX/CSV table generation (univariate)
├── 82_table_neurosynth_rsa.py            # LaTeX/CSV table generation (RSA)
├── 91_plot_neurosynth_univariate.py      # Figure generation (univariate)
├── 92_plot_neurosynth_rsa.py             # Figure generation (RSA)
├── METHODS.md                            # Detailed methods from manuscript
├── RESULTS.md                            # Detailed results summary
├── DISCREPANCIES.md                      # Notes on analysis discrepancies
└── analyses/neurosynth/                  # Shared analysis modules (in repo root analyses/ package)
    ├── __init__.py
    ├── io_utils.py                       # Loading term maps and group t-maps
    ├── maps_utils.py                     # T-to-Z conversion, map splitting, correlations
    ├── glm_utils.py                      # Second-level GLM design matrix
    ├── plot_utils.py                     # Plotting utilities
    └── tables.py                         # Table formatting

results/neurosynth/                       # Unified results tree (not committed)
├── data/                                 # zmap_*.nii.gz, *_term_corr_*.csv
├── tables/                               # LaTeX tables
└── figures/                              # Publication figures
```

The `results/` tree is distributed as a release artifact (`chess-bids_F_code-results.zip`) and via the RDR repo; it is not tracked in git. Use `from common import results_for; results_for('neurosynth', 'data')` as the idiomatic accessor.
