# Neurosynth Meta-Analytic Correlation Analysis

## Overview

This analysis places observed expertise-related brain activation differences in a broader functional context by correlating our group-level statistical maps with large-scale functional networks derived from automated meta-analysis. Specifically, we correlate (1) univariate GLM t-maps (Experts > Novices) and (2) RSA searchlight group contrast maps with term-based association maps from the Neurosynth database. This data-driven approach identifies which cognitive functions are preferentially associated with regions showing expertise effects.

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
- `neurosynth/terms/1_working memory.nii.gz`
- `neurosynth/terms/2_navigation.nii.gz`
- `neurosynth/terms/3_memory retrieval.nii.gz`
- `neurosynth/terms/4_language network.nii.gz`
- `neurosynth/terms/5_object recognition.nii.gz`
- `neurosynth/terms/6_face recognition.nii.gz`
- `neurosynth/terms/7_early visual.nii.gz`

### Univariate Analysis: T-Map Correlations

**Input data**:
- Group-level t-maps from SPM12 second-level GLMs
- Contrasts: "All boards > Baseline" or "Checkmate > Non-checkmate" (first-level), "Experts > Novices" (second-level)
- N=40 participants (20 experts, 20 novices)
- Degrees of freedom: 38 (n_subjects − 2 for two-sample t-test)
- Smoothing: 4mm FWHM Gaussian kernel

Paths (under external data root):
- `BIDS/derivatives/SPM/GLM-smooth4/group/spmT_exp_gt_nonexp_all_gt_rest.nii.gz`
- `BIDS/derivatives/SPM/GLM-smooth4/group/spmT_exp_gt_nonexp_check_gt_nocheck.nii.gz`

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
- `BIDS/derivatives/rsa_searchlight/sub-*/sub-*_desc-searchlight_checkmate_stat-r_map.nii.gz`
- `BIDS/derivatives/rsa_searchlight/sub-*/sub-*_desc-searchlight_strategy_stat-r_map.nii.gz`
- `BIDS/derivatives/rsa_searchlight/sub-*/sub-*_desc-searchlight_visual_similarity_stat-r_map.nii.gz`

**Term correlation**:
- Same procedure as univariate analysis
- Correlate Z+ and Z− maps with Neurosynth term maps
- Compute correlation differences (r_pos − r_neg)

## Data Requirements

### Input Files

**For univariate analysis**:
- **Group t-maps**: `data/BIDS/derivatives/spm-glm/smooth4/group/spmT_*.nii`
  - SPM12 second-level GLM outputs
  - Contrasts: Experts > Novices for various first-level contrasts
- **Neurosynth term maps**: `data/neurosynth_terms/*.nii.gz`
  - Z-scored association test maps for cognitive terms
  - Downloaded from https://neurosynth.org/

**For RSA analysis**:
- **Subject-level searchlight maps**: `data/BIDS/derivatives/rsa-searchlight/sub-*/sub-*_space-MNI152NLin2009cAsym_model-<pattern>_rsa.nii.gz`
  - Correlation coefficient maps from whole-brain RSA searchlight
  - Models: checkmate, strategy, visual_similarity
- **Participant data**: `data/BIDS/participants.tsv`
  - Columns: `participant_id`, `group` (expert/novice)
- **Neurosynth term maps**: Same as above

## Running the Analysis

### Step 1: Univariate Neurosynth Correlation

```bash
# From repository root
python chess-neurosynth/01_univariate_neurosynth.py
```

**Outputs** (saved to `chess-neurosynth/results/neurosynth_univariate/`):
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

**Outputs** (saved to `chess-neurosynth/results/neurosynth_rsa/`):
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
- Univariate tables → `chess-neurosynth/results/neurosynth_univariate/tables/`
  - `neurosynth_univariate_<contrast>.tex`: LaTeX tables
  - `neurosynth_univariate_<contrast>.csv`: CSV tables
- RSA tables → `chess-neurosynth/results/neurosynth_rsa/tables/`
  - `neurosynth_rsa_<pattern>.tex`: LaTeX tables
  - `neurosynth_rsa_<pattern>.csv`: CSV tables

### Step 4: Generate Figures

```bash
# Univariate figures
python chess-neurosynth/91_plot_neurosynth_univariate.py

# RSA figures
python chess-neurosynth/92_plot_neurosynth_rsa.py
```

**Outputs**:
- Univariate figures → `chess-neurosynth/results/neurosynth_univariate/figures/`
  - Individual axes as SVG/PDF: `neurosynth_univariate_<contrast>_<panel>.svg`, etc.
  - Complete panel: `panels/neurosynth_univariate_panel.pdf`
- RSA figures → `chess-neurosynth/results/neurosynth_rsa/figures/`
  - Individual axes as SVG/PDF
  - Complete panel: `panels/neurosynth_rsa_panel.pdf`

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
├── modules/
│   ├── __init__.py
│   ├── io_utils.py                       # Loading term maps and group t-maps
│   ├── maps_utils.py                     # T-to-Z conversion, map splitting, correlations
│   ├── glm_utils.py                      # Second-level GLM design matrix
│   └── plotting.py                       # Plotting utilities
├── local/                                # Local data preparation scripts
└── results/                              # Analysis outputs (timestamped)
    ├── <timestamp>_neurosynth_univariate/
    │   ├── zmap_*.nii.gz                 # Z-score maps
    │   ├── *_term_corr_*.csv             # Correlation results
    │   ├── tables/                       # LaTeX tables
    │   └── figures/                      # Publication figures
    └── <timestamp>_neurosynth_rsa/
        ├── zmap_*.nii.gz                 # Group contrast z-maps
        ├── *_term_corr_*.csv             # Correlation results
        ├── tables/                       # LaTeX tables
        └── figures/                      # Publication figures
```
