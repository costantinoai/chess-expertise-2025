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

## Motivation

A reviewer requested evidence that neural differences reflect a continuous
skill gradient, not just an artefact of the binary group partition. If neural
representations truly track chess expertise, they should correlate with
continuous skill measures within and across groups.

## Methods

For each skill proxy, Pearson and Spearman correlations are computed between
per-subject skill scores and per-subject neural measures (mean across 22
coarsened bilateral Glasser ROI groups). Correlations are also computed per
ROI. FDR correction (Benjamini-Hochberg) is applied across ROIs within each
measure.

**Note on pooled vs expert-only correlations**: Pooled (all participants)
correlations include between-group variance, so strong effects may partly
reflect the group split. Expert-only correlations isolate the within-group
skill gradient.

### Elo correlations (experts only)

- Elo x Participation Ratio (PR): 22 ROIs + mean
- Elo x RSA model fit (checkmate, strategy): 22 ROIs + mean per model
- Elo x Decoding accuracy (checkmate, strategy): 22 ROIs + mean per target

### Familiarisation correlations (all participants + experts only)

- Move accuracy x PR, RSA (checkmate, strategy, visual similarity),
  Decoding (checkmate, strategy)
- Tested in both the full sample (n=38) and within experts (n=19)
- FDR correction applied across neural metrics within each sample

## Key Results

### Elo correlations

No ROI-level correlations survive FDR correction for any measure. The
strongest uncorrected effects:

- **RSA checkmate (mean)**: r=0.465, p=0.039 -- higher Elo associated with
  stronger checkmate model fit (experts only)
- **RSA strategy (mean)**: r=0.366, p=0.113

PR and decoding show no reliable Elo gradient (all p>0.25 at the mean level).

### Familiarisation correlations

**All participants (n=38)**: Several strong correlations with move accuracy,
reflecting both within- and between-group variance:

- Decoding checkmate: r=0.709, p<0.001
- RSA strategy: r=0.588, p<0.001
- Decoding strategy: r=0.541, p<0.001
- RSA checkmate: r=0.441, p=0.006
- PR: r=-0.394, p=0.014

**Experts only (n=19)**: No correlation reaches significance, consistent with
range restriction within the expert group.

## Data

- **Participants**: BIDS `participants.tsv` (Elo in `rating` column)
- **PR**: `chess-manifold/results/manifold/pr_results.pkl`
- **RSA**: BIDS `derivatives/mvpa-rsa/` per-subject TSVs
- **Decoding**: BIDS `derivatives/mvpa-decoding/` per-subject TSVs
- **Familiarisation**: `chess-supplementary/task-engagement/results/familiarisation_accuracy/familiarisation_subject_accuracy.csv`

## Scripts

| Script | Purpose |
|--------|---------|
| `01_skill_gradient.py` | Main analysis: Elo and familiarisation correlations with neural metrics |
| `91_plot_skill_gradient.py` | Combined 3x3 figure: Elo (row 1) + familiarisation (rows 2-3) |

## Output Files

### Results (in `results/skill_gradient/`)

| File | Contents |
|------|----------|
| `elo_pr_correlations.csv` | Elo x PR per ROI + mean |
| `elo_rsa_correlations.csv` | Elo x RSA (checkmate, strategy) per ROI + mean |
| `elo_decoding_correlations.csv` | Elo x decoding per ROI + mean |
| `elo_correlations_all.csv` | Combined Elo results (all measures) |
| `familiarisation_subject_enriched.csv` | Per-subject familiarisation accuracy enriched with neural metrics |
| `familiarisation_neural_correlations.csv` | Familiarisation accuracy x neural metric correlations |

### Figures (in `results/skill_gradient/figures/`)

| File | Contents |
|------|----------|
| `skill_gradient_panel.pdf` | Combined 3x3 panel: Elo correlations (row 1, experts) + familiarisation correlations (rows 2-3, all participants) |
| `elo_correlations_panel.pdf` | Elo-only correlation panel |
| `familiarisation_correlations_panel.pdf` | Familiarisation correlations (all participants) |
| `familiarisation_correlations_experts_panel.pdf` | Familiarisation correlations (experts only) |
