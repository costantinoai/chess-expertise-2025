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

### 2. Eye-Tracking Decoding (`eyetracking/`)

**Purpose**: Decode expertise group from gaze patterns using eye-tracking data collected during the fMRI task.

**Key Question**: Do experts and novices have different visual strategies when viewing chess boards?

**Method**:
- Features: Gaze coordinates (x, y) and displacement from screen center
- Classifier: Linear SVM with stratified group k-fold cross-validation (k=20, grouped by subject)
- Inference: One-sample t-test on subject-level out-of-fold accuracies vs 0.5; pooled run accuracy reported descriptively

**Key Finding**: Gaze patterns do not discriminate experts from novices.

**See**: [`eyetracking/README.md`](eyetracking/README.md)

---

### 3. MVPA Finer Resolution (`mvpa-finer/`)

**Purpose**: Test whether within-checkmate representations differentiate expertise using finer strategic dimensions.

**Key Question**: Beyond checkmate detection, do experts represent finer strategic distinctions differently than novices?

**Method**:
- Subset: Only 20 checkmate positions
- Five finer dimensions: Strategy type, Motif, Pieces, Legal moves, Moves to checkmate
- RSA and SVM decoding within the checkmate subset

**Key Finding**: Experts encode task-relevant details even within checkmate positions (finer distinctions).

**See**: [`mvpa-finer/README.md`](mvpa-finer/README.md)

---

### 4. Neurosynth Term Visualization (`neurosynth-terms/`)

**Purpose**: Visualize the spatial distribution of Neurosynth cognitive term maps used in meta-analytic correlations.

**Content**:
- Flatmap and glass brain visualizations
- Seven terms: working memory, memory retrieval, navigation, language, face recognition, early visual, object recognition
- Provides spatial context for interpreting meta-analytic correlation results

**Outputs**: Surface plots and glass brain projections for each term.

**See**: [`neurosynth-terms/README.md`](neurosynth-terms/README.md)

---

### 5. RDM Intercorrelation (`rdm-intercorrelation/`)

**Purpose**: Quantify relationships between model RDMs to assess representational structure overlap.

**Key Question**: How much variance do different representational dimensions (checkmate, strategy, visual) share?

**Method**:
- Pairwise Spearman correlations between all RDMs
- Partial correlations controlling for third variables
- Hierarchical variance partitioning

**Key Finding**: Model RDMs are largely orthogonal, indicating distinct representational dimensions.

**See**: [`rdm-intercorrelation/README.md`](rdm-intercorrelation/README.md)

---

### 6. RSA ROI Summary (`rsa-rois/`)

**Purpose**: Summarize whole-brain searchlight RSA results within 180 bilateral Glasser parcellation ROIs.

**Key Question**: Which specific cortical regions show expertise-related differences in representational similarity?

**Method**:
- Extract searchlight correlation values within each of 180 ROIs
- Welch t-tests comparing experts vs novices per ROI
- FDR correction across 180 regions

**Key Finding**: Expertise differences concentrate in fronto-parietal and lateral occipital regions.

**See**: [`rsa-rois/README.md`](rsa-rois/README.md)

---

### 7. Univariate ROI Summary (`univariate-rois/`)

**Purpose**: Summarize first-level univariate contrast maps within 180 bilateral Glasser parcellation ROIs.

**Key Question**: Which specific cortical regions show expertise-related differences in univariate activation?

**Method**:
- Extract beta values within each of 180 ROIs
- Two contrasts: Checkmate > Non-checkmate, All > Rest
- Welch t-tests comparing experts vs novices per ROI with FDR correction

**See**: [`univariate-rois/README.md`](univariate-rois/README.md)

---

### 8. Task Engagement and Board Preference (`task-engagement/`)

**Purpose**: Characterise how experts and novices engage with the fMRI 1-back preference task, and identify which objective board features drive selection preferences.

**Key Questions**: Do novices comply with the task? Do experts and novices base their preferences on different stimulus properties?

**Method**:
- Four engagement diagnostics: response rate, checkmate preference, transitivity, C-NC within-pair correlation
- Board preference feature drivers: 16 board-level + 4 image-level features correlated with selection frequency (FDR-corrected)

**Key Finding**: Experts prefer boards based on checkmate status (r=0.87); novices prefer visually complex boards with more officer pieces (r=0.59).

**See**: [`task-engagement/README.md`](task-engagement/README.md)

---

### 9. Skill Gradient (`skill-gradient/`)

**Purpose**: Test whether neural measures scale continuously with chess skill rather than differing only as a binary group split.

**Key Question**: Is there a continuous expertise gradient in neural representations?

**Method**:
- Elo rating correlated with RSA, decoding, and participation ratio across ROIs (experts only, FDR-corrected)
- Familiarisation accuracy correlated with neural metrics (all participants and experts only)

**Key Finding**: Elo correlates with checkmate decoding in visual and parietal ROIs; familiarisation accuracy correlates with decoding across broader networks.

**See**: [`skill-gradient/README.md`](skill-gradient/README.md)

---

### 10. Subcortical ROI Analysis (`subcortical-rois/`)

**Purpose**: Extend the cortical MVPA pipeline to subcortical structures using the CAB-NP atlas (Ji et al., 2019).

**Key Question**: Do expertise-related representational effects extend to subcortical regions (hippocampus, caudate, thalamus, cerebellum)?

**Method**:
- 9 bilateral subcortical ROIs from CAB-NP atlas
- Identical SVM decoding and RSA pipeline as cortical analysis
- FDR correction across 9 ROIs

**Key Finding**: No FDR-significant subcortical effects; expertise-related representational reorganisation is predominantly cortical.

**See**: [`subcortical-rois/README.md`](subcortical-rois/README.md)

---

## Running All Supplementary Analyses

To run every group/table/plot script in the supplementary tree, use the top-level pipeline. With no arguments, `run_all_analyses.sh` runs at the `group` level (the Python layer that regenerates `derivatives/group-results/supplementary/<name>/{data,tables,figures}/`):

```bash
# From repository root
./run_all_analyses.sh # default: group level
./run_all_analyses.sh group # same as above, explicit
```

Use `./run_all_analyses.sh subject-level` to re-run the MATLAB subject scripts that write into `BIDS/derivatives/fmriprep_spm-unsmoothed_{rsa,decoding,...}/`; do not do this unless you have the time budget for the full MATLAB re-run.

Outputs land in the unified repo tree at `derivatives/group-results/supplementary/<name>/{data,tables,figures}/`. The `results/` tree contains **only group-level aggregates** (GDPR-compliant); per-subject data lives in `BIDS/derivatives/`.

### Naming convention

Scripts follow a subject/group split numbering scheme:
- **0x** = subject-level scripts (write per-subject data to `BIDS/derivatives/`)
- **1x** = group-level scripts (read from `derivatives/`, write group-level aggregates to `derivatives/group-results/`)
- **8x** = table scripts (read from `derivatives/group-results/`, write formatted tables)
- **9x** = figure scripts (read from `derivatives/group-results/`, write rendered figures)

Within each pipeline, related subject and group scripts share the same unit digit (e.g., `01` and `11` form a pair).

