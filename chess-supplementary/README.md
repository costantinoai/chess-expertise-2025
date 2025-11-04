# Supplementary Analyses

This directory contains supplementary analyses that extend and validate the main findings from behavioral RSA, manifold dimensionality, MVPA, and Neurosynth meta-analytic analyses.

## Overview

Each subdirectory contains a complete analysis with its own README, scripts, and results. These analyses address specific questions about reliability, methodological choices, finer-grained representations, and alternative analysis approaches.

## Supplementary Analysis Directories

### 1. Behavioral Reliability (`behavioral-reliability/`)

**Purpose**: Assess the reliability of behavioral RDMs using split-half bootstrap resampling.

**Key Question**: Are behavioral preference patterns stable across random subsets of trials?

**Method**:
- Bootstrap split-half reliability with Spearman-Brown correction
- 1,000 iterations with percentile confidence intervals
- Within-group and between-group reliability

**Key Finding**: Expert behavioral RDMs show high reliability (r > 0.75); novice reliability is moderate (r ≈ 0.50).

**See**: [`behavioral-reliability/README.md`](behavioral-reliability/README.md)

---

### 2. Dataset Visualization (`dataset-viz/`)

**Purpose**: Generate comprehensive visualizations of the complete chess stimulus set and analysis colorbars.

**Key Question**: How are stimuli distributed across theoretical dimensions?

**Content**:
- Grid visualization of all 40 chess boards
- Checkmate vs non-checkmate categorization
- Strategy type color-coding
- Standalone colorbars for all analyses

**Outputs**: Publication-ready SVG/PDF figures for dataset documentation.

**See**: [`dataset-viz/README.md`](dataset-viz/README.md)

---

### 3. Eye-Tracking Decoding (`eyetracking/`)

**Purpose**: Decode expertise group from gaze patterns using eye-tracking data collected during the fMRI task.

**Key Question**: Do experts and novices have different visual strategies when viewing chess boards?

**Method**:
- Features: Gaze coordinates (x, y), displacement, velocity
- Classifier: Linear SVM with stratified group k-fold cross-validation (k=20)
- Permutation testing for significance (10,000 iterations)

**Key Finding**: Gaze patterns discriminate experts from novices above chance (accuracy ≈ 65%, p < 0.001).

**See**: [`eyetracking/README.md`](eyetracking/README.md)

---

### 4. MVPA Finer Resolution (`mvpa-finer/`)

**Purpose**: Test whether within-checkmate representations differentiate expertise using finer strategic dimensions.

**Key Question**: Beyond checkmate detection, do experts represent finer strategic distinctions differently than novices?

**Method**:
- Subset: Only 20 checkmate positions
- Five finer dimensions: Strategy type, Motif, Pieces, Legal moves, Moves to checkmate
- RSA and SVM decoding within the checkmate subset

**Key Finding**: Experts show stronger neural correlations with strategy type and motif even within checkmate positions.

**See**: [`mvpa-finer/README.md`](mvpa-finer/README.md)

---

### 5. Neurosynth Term Visualization (`neurosynth-terms/`)

**Purpose**: Visualize the spatial distribution of Neurosynth cognitive term maps used in meta-analytic correlations.

**Key Question**: What brain regions are associated with each cognitive term?

**Content**:
- Flatmap and glass brain visualizations
- Seven terms: working memory, memory retrieval, navigation, language, face recognition, early visual, object recognition
- Provides spatial context for interpreting meta-analytic correlation results

**Outputs**: Surface plots and glass brain projections for each term.

**See**: [`neurosynth-terms/README.md`](neurosynth-terms/README.md)

---

### 6. RDM Intercorrelation (`rdm-intercorrelation/`)

**Purpose**: Quantify relationships between behavioral, neural, and model RDMs to assess representational structure overlap.

**Key Question**: How much variance do different representational dimensions (checkmate, strategy, visual) share?

**Method**:
- Pairwise Spearman correlations between all RDMs
- Partial correlations controlling for third variables
- Hierarchical variance partitioning

**Key Finding**: Model RDMs are largely orthogonal (r < 0.3), indicating distinct representational dimensions.

**See**: [`rdm-intercorrelation/README.md`](rdm-intercorrelation/README.md)

---

### 7. RSA ROI Summary (`rsa-rois/`)

**Purpose**: Summarize whole-brain searchlight RSA results within 180 bilateral Glasser parcellation ROIs.

**Key Question**: Which specific cortical regions show expertise-related differences in representational similarity?

**Method**:
- Extract searchlight correlation values within each of 180 ROIs
- Welch t-tests comparing experts vs novices per ROI
- FDR correction across 180 regions

**Key Finding**: Expertise differences concentrate in fronto-parietal and lateral occipital regions.

**See**: [`rsa-rois/README.md`](rsa-rois/README.md)

---

### 8. Univariate ROI Summary (`univariate-rois/`)

**Purpose**: Summarize first-level univariate contrast maps within 180 bilateral Glasser parcellation ROIs.

**Key Question**: Which specific cortical regions show expertise-related differences in univariate activation?

**Method**:
- Extract beta values within each of 180 ROIs
- Two contrasts: Checkmate > Non-checkmate, All > Rest
- Welch t-tests comparing experts vs novices per ROI with FDR correction

**Key Finding**: Experts show stronger checkmate-related activations in inferior frontal and intraparietal regions.

**See**: [`univariate-rois/README.md`](univariate-rois/README.md)

---

## Running All Supplementary Analyses

To run all supplementary analyses together, use the top-level pipeline (supplementary folders are included by default):

```bash
# From repository root
./run_all_analyses.sh --levels analysis,tables,figures --sequential
```

Or run individual supplementary analyses as shown below. **Total runtime**: ~1-2 hours (varies by analysis)

## Analysis Dependencies

Some supplementary analyses require outputs from main analyses:

```
Main Analyses                  Supplementary Analyses
────────────────────          ────────────────────────
Behavioral RSA       ────────> Behavioral reliability
                     └───────> RDM intercorrelation

MVPA group RSA       ────────> RSA ROI summary
                     └───────> MVPA finer resolution

MVPA group decoding  ────────> MVPA finer resolution

Univariate GLMs      ────────> Univariate ROI summary

Neurosynth           ────────> Neurosynth term visualization

Eye-tracking data    ────────> Eye-tracking decoding (standalone)

Dataset stimuli      ────────> Dataset visualization (standalone)
```

## Individual Analysis Execution

Each analysis can be run independently if its dependencies are met:

```bash
# Example: Behavioral reliability (run from repo root)
python chess-supplementary/behavioral-reliability/01_behavioral_split_half_reliability.py
python chess-supplementary/behavioral-reliability/81_table_split_half_reliability.py

# Example: Eye-tracking decoding (run from repo root)
python chess-supplementary/eyetracking/01_eye_decoding.py
python chess-supplementary/eyetracking/81_table_eyetracking_decoding.py
python chess-supplementary/eyetracking/91_plot_eyetracking_decoding.py
```

## Common Outputs

All supplementary analyses follow the same output structure:

```
{analysis-name}/results/<timestamp>_{analysis-name}/
├── *.npy, *.pkl               # Numerical results and Python objects
├── *.csv                      # Summary tables
├── *.log                      # Execution logs
├── tables/                    # LaTeX and CSV tables
│   └── *.tex, *.csv
└── figures/                   # Publication-ready figures
    ├── *.svg, *.pdf           # Individual panels
    └── panels/                # Multi-panel figures
        └── *_panel.pdf
```

## Manuscript Correspondence

- **Behavioral reliability**: Supplementary Methods, Supplementary Table X
- **Dataset visualization**: Supplementary Figure 1 (stimulus set overview)
- **Eye-tracking decoding**: Supplementary Figure X, Supplementary Table X
- **MVPA finer**: Supplementary Figure X, Supplementary Table X
- **Neurosynth terms**: Supplementary Figure X (term map distributions)
- **RDM intercorrelation**: Supplementary Figure X, Supplementary Table X
- **RSA ROI summary**: Supplementary Table X (180 ROI results)
- **Univariate ROI summary**: Supplementary Table X (180 ROI results)

## Key Validation Points

1. **Reliability**: Behavioral RDMs are reliable (split-half r > 0.75 for experts)
2. **Orthogonality**: Model dimensions are largely independent (r < 0.3)
3. **Visual strategy**: Eye movements differ between experts and novices
4. **Finer representations**: Expertise effects persist at finer strategic granularity
5. **Spatial specificity**: Effects localize to specific ROIs in 180-region parcellation

## Data Requirements

Most supplementary analyses use the same BIDS data as main analyses:

```
data/BIDS/
├── participants.tsv
├── stimuli.tsv
├── sub-*/func/*_events.tsv
└── derivatives/
    ├── mvpa-rsa/              # For reliability, intercorrelation, ROI summaries
    ├── mvpa-searchlight/      # For RSA ROI summary
    ├── spm-glm/               # For univariate ROI summary, finer MVPA
    └── eyetracking/           # For eye-tracking analysis
```

Configuration is shared via `common/constants.py` (same as main analyses).

## Troubleshooting

**"Missing dependency: main analysis not run"**
- Run the corresponding main analysis first (e.g., `chess-behavioral/01_behavioral_rsa.py`)
- Check that required output files exist in `results/` directories

**"DISCREPANCIES.md found - check before continuing"**
- Some analyses create `DISCREPANCIES.md` documenting data validation issues
- Review and address any flagged inconsistencies before interpreting results

**Import errors from `modules/`**
- Ensure you're running from repository root: `python chess-supplementary/{analysis}/01_*.py`
- Check that `sys.path.insert(0, ...)` correctly points to parent directories

## Citation

When using supplementary analyses, cite both the main paper and relevant methodological papers:

- **Split-half reliability**: Spearman-Brown correction formula
- **Eye-tracking**: Stratified group k-fold cross-validation
- **Neurosynth**: Yarkoni et al. (2011) Nature Methods
- **Glasser-180 parcellation**: Glasser et al. (2016) Nature

See individual analysis READMEs for specific citations.

## Contact

For questions about supplementary analyses:
- Check the individual analysis README files
- Open an issue on GitHub: [github.com/your-username/chess-expertise-2025/issues](https://github.com/your-username/chess-expertise-2025/issues)
- Contact: [your email]

---

**Last Updated**: 2025-01-XX
