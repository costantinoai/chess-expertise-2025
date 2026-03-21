# Skill Gradient Analysis

## Overview

This analysis tests whether neural measures (RSA model fit, decoding accuracy,
participation ratio) scale continuously with chess skill, rather than differing
only as a binary expert-vs-novice group split. Two complementary skill proxies
are used:

1. **Elo rating** (general chess strength): Correlated with neural metrics
   within the expert group (n=20, Elo 1751--2269).
2. **Familiarisation accuracy** (stimulus-specific competence): Move accuracy
   on the 20 checkmate boards used in fMRI, correlated with neural metrics
   across all participants (n=38) and within experts only.

## Methods

### Rationale

A reviewer requested evidence that neural differences reflect a continuous
skill gradient, not just an artefact of the binary group partition. If neural
representations truly track chess expertise, they should correlate with
continuous skill measures within and across groups.

### Data Sources

**Participants**: N=40 (20 experts, 20 novices)
**Skill proxies**:
- Elo rating from BIDS `participants.tsv` (experts only, n=20)
- Familiarisation accuracy from pre-scan checkmate detection task (all participants, n=38)

**Neural measures**:
- RSA model fit (checkmate, strategy) from BIDS `derivatives/mvpa-rsa/`
- SVM decoding accuracy (checkmate, strategy) from BIDS `derivatives/mvpa-decoding/`
- Participation ratio (PR) from `chess-manifold/results/manifold/pr_results.pkl`

### Correlation Procedure

For each skill proxy, Pearson and Spearman correlations are computed between
per-subject skill scores and per-subject neural measures (mean across 22
coarsened bilateral Glasser ROI groups). Correlations are also computed per
ROI. FDR correction (Benjamini-Hochberg) is applied across ROIs within each
measure.

**Note on pooled vs expert-only correlations**: Pooled (all participants)
correlations include between-group variance, so strong effects may partly
reflect the group split. Expert-only correlations isolate the within-group
skill gradient.

### Elo Correlations (Experts Only)

- Elo x Participation Ratio (PR): 22 ROIs + mean
- Elo x RSA model fit (checkmate, strategy): 22 ROIs + mean per model
- Elo x Decoding accuracy (checkmate, strategy): 22 ROIs + mean per target

### Familiarisation Correlations (All Participants + Experts Only)

- Move accuracy x PR, RSA (checkmate, strategy, visual similarity),
  Decoding (checkmate, strategy)
- Tested in both the full sample (n=38) and within experts (n=19)
- FDR correction applied across neural metrics within each sample

## Dependencies

- Python 3.9+ with packages: numpy, pandas, scipy, matplotlib
- Common utilities from `common/` (stats_utils, logging_utils, script_utils, plotting)

## Data Requirements

### Input Files

- **Participant metadata**: `BIDS_ROOT/participants.tsv` (Elo in `rating` column)
- **Participation ratio**: `chess-manifold/results/manifold/pr_results.pkl`
- **RSA per-subject results**: `BIDS_ROOT/derivatives/mvpa-rsa/sub-XX/*.tsv`
- **Decoding per-subject results**: `BIDS_ROOT/derivatives/mvpa-decoding/sub-XX/*.tsv`
- **Familiarisation accuracy**: `chess-supplementary/task-engagement/results/familiarisation_accuracy/familiarisation_subject_accuracy.csv`

### Data Location

Configure the external data root in `common/constants.py`:

```python
# Base folder containing BIDS/, rois/, neurosynth/, stimuli/
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
```

## Running the Analysis

### Step 1: Run skill gradient correlations

```bash
cd chess-supplementary/skill-gradient
python 01_skill_gradient.py
```

Computes Elo and familiarisation correlations with all neural metrics (RSA, decoding, PR) at the mean and per-ROI level, with FDR correction.

### Step 2: Generate skill gradient figures

```bash
python 91_plot_skill_gradient.py
```

Produces the combined 3x3 panel: Elo correlations (row 1, experts only) and familiarisation correlations (rows 2--3, all participants and experts only).

## Key Results

### Elo Correlations

No ROI-level correlations survive FDR correction for any measure. The
strongest uncorrected effects:

- **RSA checkmate (mean)**: r=0.465, p=0.039 -- higher Elo associated with
  stronger checkmate model fit (experts only)
- **RSA strategy (mean)**: r=0.366, p=0.113

PR and decoding show no reliable Elo gradient (all p>0.25 at the mean level).

### Familiarisation Correlations

**All participants (n=38)**: Several strong correlations with move accuracy,
reflecting both within- and between-group variance:

- Decoding checkmate: r=0.709, p<0.001
- RSA strategy: r=0.588, p<0.001
- Decoding strategy: r=0.541, p<0.001
- RSA checkmate: r=0.441, p=0.006
- PR: r=-0.394, p=0.014

**Experts only (n=19)**: No correlation reaches significance, consistent with
range restriction within the expert group.

## File Structure

```
chess-supplementary/skill-gradient/
├── 01_skill_gradient.py               # Main analysis: Elo and familiarisation correlations with neural metrics
├── 91_plot_skill_gradient.py          # Combined 3x3 figure: Elo (row 1) + familiarisation (rows 2-3)
├── README.md
└── results/
    └── skill_gradient/
        ├── elo_pr_correlations.csv
        ├── elo_rsa_correlations.csv
        ├── elo_decoding_correlations.csv
        ├── elo_correlations_all.csv
        ├── familiarisation_subject_enriched.csv
        ├── familiarisation_neural_correlations.csv
        └── figures/
            ├── skill_gradient_panel.pdf
            ├── elo_correlations_panel.pdf
            ├── familiarisation_correlations_panel.pdf
            └── familiarisation_correlations_experts_panel.pdf
```
