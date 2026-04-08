# Behavioral Representational Similarity Analysis (RSA)

## Overview

This analysis examines behavioral similarity judgments from a 1-back preference task performed during fMRI scanning. Participants (20 experts, 20 novices) indicated which of two consecutively presented chess boards they preferred. We test whether behavioral preferences correlate with theoretical models of chess board similarity.

## Required bundles

- `01_behavioral_rsa.py` needs **A** (core: BIDS events, stimuli, participants).
- `81_table_behavioral_correlations.py` and `91_plot_behavioral_panels.py` only consume the outputs of 01 from the repo `results/` tree (no bundle download required beyond A).

In a future commit `01_behavioral_rsa.py` will be split into a subject-level stage that writes per-subject RDMs to `derivatives/behavioral-rsa/` (bundle E) and a group-level stage that aggregates them into the repo results tree.

## Data flow

```mermaid
flowchart LR
  classDef in fill:#cfe9ff,stroke:#0366d6,color:#000
  classDef out fill:#fff5b1,stroke:#b08800,color:#000
  classDef sc fill:#d1f5d3,stroke:#1a7f37,color:#000
  classDef rl fill:#eee,stroke:#888,stroke-dasharray:3 3,color:#333

  PT[participants.tsv]:::in
  ST[stimuli/]:::in
  EV["sub-*/func/ (events)"]:::in

  B01["01_behavioral_rsa_subject.py"]:::sc
  B02["02_behavioral_rsa_group.py"]:::sc
  B81["81_table_behavioral_correlations.py"]:::sc
  B91["91_plot_behavioral_panels.py"]:::sc
  BRSA[derivatives/behavioral-rsa/]:::out
  DATA["results/behavioral/data/"]:::rl
  TABLES["results/behavioral/tables/"]:::rl
  FIGURES["results/behavioral/figures/"]:::rl

  EV --> B01
  PT --> B01
  ST --> B01
  B01 --> BRSA

  BRSA --> B02
  PT --> B02
  B02 --> DATA

  DATA --> B81 --> TABLES
  PT --> B81
  DATA --> B91 --> FIGURES
  PT --> B91
```

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

**Outputs** (saved to `results/behavioral/data/`):
- `expert_behavioral_rdm.npy`: Expert group RDM (40×40)
- `novice_behavioral_rdm.npy`: Novice group RDM (40×40)
- `expert_directional_dsm.npy`: Expert directional preference matrix (40×40)
- `novice_directional_dsm.npy`: Novice directional preference matrix (40×40)
- `expert_mds_coords.npy`: Expert MDS 2D coordinates (40×2)
- `novice_mds_coords.npy`: Novice MDS 2D coordinates (40×2)
- `correlation_results.pkl`: RSA correlation statistics with FDR correction
- `correlation_summary.csv`: Human-readable summary table

Note: in a future commit this script will be split into `01_*_subject.py` (writes per-subject derivatives to BIDS) and `02_*_group.py` (reads those derivatives and writes to the repo `results/` tree).

**Expected runtime**: ~2-3 minutes

### Step 2: Generate Tables

```bash
python chess-behavioral/81_table_behavioral_correlations.py
```

**Outputs** (saved to `results/behavioral/tables/`):
- `behavioral_rsa_correlations.tex`: LaTeX table
- `behavioral_rsa_correlations.csv`: CSV table

### Step 3: Generate Figures

```bash
python chess-behavioral/91_plot_behavioral_panels.py
```

**Outputs** (saved to `results/behavioral/figures/`):
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
└── analyses/behavioral/                   # Shared analysis modules (in repo root analyses/ package)
    ├── data_loading.py                    # BIDS data loaders
    └── rdm_utils.py                       # RDM computation and RSA

results/behavioral/                        # Unified results tree (not committed)
├── data/                                  # *.npy, *.pkl, *.csv numerical results
├── tables/                                # LaTeX tables
└── figures/                               # Publication figures
```

The `results/` tree is distributed as a release artifact (`chess-bids_F_code-results.zip`) and via the RDR repo; it is not tracked in git. Use `from common import results_for; results_for('behavioral', 'data')` as the idiomatic accessor.

