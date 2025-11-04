# Supplementary Analyses

This directory contains supplementary analyses that extend and validate the main findings from behavioral RSA, manifold dimensionality, MVPA, and Neurosynth meta-analytic analyses.

## Overview

Each subdirectory contains a complete analysis with its own README, scripts, and results. These analyses address specific questions about reliability, methodological choices, finer-grained representations, and alternative analysis approaches.

## Supplementary Analysis Directories

### 1. Behavioral Reliability (`behavioral-reliability/`)

**Purpose**: Assess the reliability of behavioral RDMs using split-half bootstrap resampling.

**Key Question**: Are behavioral preference patterns stable across subjects (within and between expertise groups)?

**Method**:
- Bootstrap split-half reliability with Spearman-Brown correction
- 1,000 iterations with percentile confidence intervals
- Within-group and between-group reliability

**Key Finding**: Expert behavioral RDMs show higher reliability than novice.

**See**: [`behavioral-reliability/README.md`](behavioral-reliability/README.md)

---

### 2. Dataset Visualization (`dataset-viz/`)

**Purpose**: Generate comprehensive visualizations of the complete chess stimulus set and analysis colorbars.

**Content**:
- Grid visualization of all 40 chess boards

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

**Key Finding**: Gaze patterns do not discriminate experts from novices.

**See**: [`eyetracking/README.md`](eyetracking/README.md)

---

### 4. MVPA Finer Resolution (`mvpa-finer/`)

**Purpose**: Test whether within-checkmate representations differentiate expertise using finer strategic dimensions.

**Key Question**: Beyond checkmate detection, do experts represent finer strategic distinctions differently than novices?

**Method**:
- Subset: Only 20 checkmate positions
- Five finer dimensions: Strategy type, Motif, Pieces, Legal moves, Moves to checkmate
- RSA and SVM decoding within the checkmate subset

**Key Finding**: Experts encode task-relevant details even within checkmate positions (finer distinctions).

**See**: [`mvpa-finer/README.md`](mvpa-finer/README.md)

---

### 5. Neurosynth Term Visualization (`neurosynth-terms/`)

**Purpose**: Visualize the spatial distribution of Neurosynth cognitive term maps used in meta-analytic correlations.

**Content**:
- Flatmap and glass brain visualizations
- Seven terms: working memory, memory retrieval, navigation, language, face recognition, early visual, object recognition
- Provides spatial context for interpreting meta-analytic correlation results

**Outputs**: Surface plots and glass brain projections for each term.

**See**: [`neurosynth-terms/README.md`](neurosynth-terms/README.md)

---

### 6. RDM Intercorrelation (`rdm-intercorrelation/`)

**Purpose**: Quantify relationships between model RDMs to assess representational structure overlap.

**Key Question**: How much variance do different representational dimensions (checkmate, strategy, visual) share?

**Method**:
- Pairwise Spearman correlations between all RDMs
- Partial correlations controlling for third variables
- Hierarchical variance partitioning

**Key Finding**: Model RDMs are largely orthogonal, indicating distinct representational dimensions.

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

**See**: [`univariate-rois/README.md`](univariate-rois/README.md)

---

## Running All Supplementary Analyses

To run all supplementary analyses together, use the top-level pipeline (supplementary folders are included by default):

```bash
# From repository root
./run_all_analyses.sh --levels analysis,tables,figures --sequential
```

