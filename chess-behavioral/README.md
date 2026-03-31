# Behavioral Representational Similarity Analysis (RSA)

## Overview

This analysis examines behavioral similarity judgments from a 1-back preference task performed during fMRI scanning. Participants (20 experts, 20 novices) indicated which of two consecutively presented chess boards they preferred. We test whether behavioral preferences correlate with theoretical models of chess board similarity.

## Methods

### Task and Data

During fMRI scanning, participants viewed 40 chess board stimuli (20 checkmate positions, 20 non-checkmate positions) presented sequentially in a 1-back task. On each trial, participants indicated which of the two most recent boards they preferred by button press. Preference responses were recorded in BIDS-compliant `events.tsv` files.

### Behavioral RDM Construction

Pairwise preferences were aggregated across participants within each expertise group. For each unique stimulus pair (i, j), we counted the number of times stimulus i was preferred over j and vice versa. The behavioral representational dissimilarity matrix (RDM) was computed as the absolute difference between these counts:

```
RDM[i,j] = |count(i > j) - count(j > i)|
```

Higher values indicate greater inconsistency in preferences (stimuli are dissimilar in how they are perceived). Lower values indicate consistent preferences (stimuli are treated similarly).

### Count-Normalized RDM (primary analysis)

Because the 1-back task compares only consecutive boards, different stimulus pairs are compared different numbers of times depending on the random presentation sequence. Pairs compared more often accumulate higher raw counts, inflating their RDM values independently of preference consistency. To control for this exposure confound, the primary analysis uses count-normalized RDMs:

```
RDM_norm[i,j] = |count(i > j) - count(j > i)| / (count(i > j) + count(j > i))
```

Values range from 0 (perfectly tied preferences) to 1 (perfectly consistent preference direction). Pairs with zero comparisons are set to 0. The same normalization is applied to the directional preference matrix (DSM).

The unnormalized (raw count) RDM is also computed and reported as a supplementary panel for comparison. The normalization does not change the qualitative pattern of results but provides a more interpretable dissimilarity metric that is not confounded with pair exposure frequency.

### Model RDMs

Three theoretical model RDMs were constructed:

1. **Checkmate status**: Binary RDM (0 if same status, 1 if different)
2. **Strategy type**: Categorical RDM based on 5 chess strategies
3. **Visual similarity**: Perceptual feature-based dissimilarity

### Statistical Analysis

Behavioral RDMs were correlated with each model RDM using Pearson correlation. Significance was assessed via bootstrap resampling (10,000 iterations; `pingouin`). False discovery rate (FDR) correction was applied separately within each group (family size = 3 models per group) using the Benjamini–Hochberg procedure (α=0.05).

Separate analyses were conducted for experts and novices to test whether expertise modulates the cognitive dimensions underlying preference judgments.

### Visualization

Multidimensional scaling (MDS) was used to project the 40×40 RDM into 2D space for visualization, preserving pairwise dissimilarities as closely as possible.

## Dependencies

- Python 3.8+
- numpy, pandas, scipy
- scikit-learn (for MDS)
- pingouin (for bootstrap correlations)
- matplotlib, seaborn (for plotting)
- BIDS-compliant event files in `data/BIDS/{participant_id}/func/`

See `requirements.txt` in the repository root for complete dependencies.

## Data Requirements

### Input Files

- **Participant data**: `BIDS/participants.tsv` (columns: `participant_id`, `group`)
- **Event files**: `BIDS/sub-*/func/sub-*_task-exp_run-*_events.tsv`
  - Required columns: `stim_id`, `preference` (`current_preferred`, `previous_preferred`, or `n/a`)
- **Stimulus metadata**: `stimuli/stimuli.tsv`
  - Required columns: `stim_id`, `check`, `strategy`, `visual`

### Data Location

Set the external data root once in `common/constants.py` (all analysis paths are derived from it):

```python
# Base folder containing BIDS/ (all data lives inside BIDS/)
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
# BIDS_ROOT, ROI, and other paths are built from this automatically
```

Expected structure under this folder:

```
/path/to/manuscript-data/
└── BIDS/                      # Main dataset (BIDS-compliant)
    ├── participants.tsv
    ├── stimuli/               # Stimulus metadata
    ├── sub-*/
    └── derivatives/           # Per-subject derivatives + atlases
```

## Running the Analysis

### Step 1: Run Main Analysis

```bash
# From repository root (recommended)
python chess-behavioral/01_behavioral_rsa.py
```

**Outputs** (saved to `chess-behavioral/results/behavioral_rsa/`):
- `expert_behavioral_rdm.npy`: Expert group RDM (40×40)
- `novice_behavioral_rdm.npy`: Novice group RDM (40×40)
- `expert_directional_dsm.npy`: Expert directional preference matrix (40×40)
- `novice_directional_dsm.npy`: Novice directional preference matrix (40×40)
- `expert_mds_coords.npy`: Expert MDS 2D coordinates (40×2)
- `novice_mds_coords.npy`: Novice MDS 2D coordinates (40×2)
- `correlation_results.pkl`: RSA correlation statistics with FDR correction
- `correlation_summary.csv`: Human-readable summary table

**Expected runtime**: ~2-3 minutes

### Step 2: Generate Tables

```bash
python chess-behavioral/81_table_behavioral_correlations.py
```

**Outputs** (saved to `chess-behavioral/results/behavioral_rsa/tables/`):
- `behavioral_rsa_correlations.tex`: LaTeX table
- `behavioral_rsa_correlations.csv`: CSV table

### Step 3: Generate Figures

```bash
python chess-behavioral/91_plot_behavioral_panels.py
```

**Outputs** (saved to `chess-behavioral/results/behavioral_rsa/figures/`):
- Individual axes as SVG/PDF: `behavioral_A1_RDM_Experts.svg`, etc.
- Complete panels: `panels/behavioral_rsa_panel.pdf`
- Normalized RDM panel: `panels/behavioral_rsa_normalized_panel.pdf`

**Note**: If `ENABLE_PYLUSTRATOR=True` in `common/constants.py`, this will open an interactive layout editor. Set to `False` for automated figure generation.

## Key Results

**Experts**: Behavioral preferences correlate significantly with:
- Checkmate status (Pearson r = 0.73, pFDR < 0.001)
- Strategy type (Pearson r = 0.25, pFDR < 0.001)
- Visual similarity (Pearson r = −0.12, pFDR = 0.001)

Values are from the count-normalized RDM (primary analysis). Raw-count RDM correlations are similar (r = 0.70, 0.24, −0.10 respectively).

**Novices**: No significant correlations with any model RDM (all pFDR > 0.14).

The count-normalized analysis produces comparable results, confirming that the model fits are not driven by differential pair exposure.

This demonstrates that chess expertise shapes behavioral similarity judgments along task-relevant dimensions (checkmate status, strategic content) but not low-level visual features.

## File Structure

```
chess-behavioral/
├── README.md                              # This file
├── 01_behavioral_rsa.py                   # Main analysis script
├── 81_table_behavioral_correlations.py    # LaTeX/CSV table generation
├── 91_plot_behavioral_panels.py           # Figure generation
├── modules/
│   ├── data_loading.py                    # BIDS data loaders
│   └── rdm_utils.py                       # RDM computation and RSA
├── local/                                 # Data preparation (gitignored)
│   ├── convert_mat_to_bids_events_v3.py   # MATLAB to BIDS conversion (authoritative)
│   └── participants_descriptive_stats.py  # Demographic statistics
└── results/
    └── behavioral_rsa/
        ├── *.npy                          # Numerical results
        ├── *.pkl                          # Python objects
        ├── *.csv                          # Summary tables
        ├── tables/                        # LaTeX tables
        └── figures/                       # Publication figures
```

