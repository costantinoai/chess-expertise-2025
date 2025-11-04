# MVPA Finer Resolution Analysis: Checkmate Boards Only

## Overview

This supplementary analysis extends the main MVPA analysis by performing RSA and SVM decoding with finer categorical distinctions using only the 20 checkmate chess boards. Five additional dimensions are analyzed: Strategy (within checkmate), Motif, Total number of pieces, Total legal moves, and Number of moves to checkmate. This finer analysis assesses what information can be decoded within the checkmate class, offering insights into how strategic information is structured in expert and novice representations.

## Methods

### Rationale

The main MVPA analysis uses all 40 boards. This supplementary analysis focuses on the 20 checkmate boards to examine finer-grained categorical distinctions that are only defined for checkmate positions. This allows testing whether neural representations differentiate between checkmate subtypes based on tactical features, piece counts, and move complexity.

### Data Sources

**Participants**: N=40 (20 experts, 20 novices)
**Stimuli**: 20 checkmate chess boards (subset of main 40-board dataset)
**ROIs**: 22 bilateral cortical regions (Glasser parcellation)

### Categorical Dimensions (Checkmate Boards Only)

1. **Strategy**: Same as main analysis, but using only 20 checkmate boards
2. **Motif**: Tactical motif characterizing the checkmate sequence (e.g., fork, pin, skewer, back-rank mate)
3. **Total pieces**: Number of pieces on the board
4. **Legal moves**: Total number of available legal moves
5. **Moves to checkmate**: Number of white moves required to reach checkmate

### Subject-Level Analysis (MATLAB/CoSMoMVPA)

**RSA**: Compute neural RDMs from 20 checkmate boards, correlate with model RDMs
**Decoding**: Train linear SVM to classify checkmate boards by each categorical dimension
**Procedure**: Same as main MVPA analysis (`chess-mvpa/`), but restricted to 20 checkmate boards

### Group-Level Analysis (Python)

Same statistical framework as main MVPA:
- Welch t-tests per ROI (experts vs novices)
- One-sample t-tests vs chance/zero
- Benjamini-Hochberg FDR correction across 22 ROIs

## Dependencies

**MATLAB**:
- MATLAB R2024b or later
- CoSMoMVPA toolbox
- SPM12

**Python**:
- Python 3.8+
- numpy, pandas, scipy
- statsmodels (for FDR correction)
- matplotlib, seaborn (for plotting)

See `requirements.txt` in the repository root for complete dependencies.

## Data Requirements

### Input Files

Same as main MVPA analysis (`chess-mvpa/`):
- **SPM GLM outputs**: `BIDS/derivatives/SPM/GLM-unsmoothed/sub-*/exp/`
- **Atlas**: `rois/glasser22/tpl-...desc-22_bilateral_resampled.nii.gz`
- **Participant data**: `BIDS/participants.tsv`
- **Stimulus metadata**: `stimuli/stimuli.tsv` (with checkmate-specific columns)

## Running the Analysis

### Step 1: Subject-Level MVPA (MATLAB)

```matlab
% From MATLAB, cd to chess-supplementary/mvpa-finer/
01_roi_decoding_fine.m
```

**Outputs**: Subject-level TSV files in BIDS derivatives (same structure as main MVPA)

**Expected runtime**: ~2-5 minutes per subject

### Step 2: Group-Level RSA Analysis (Python)

```bash
# From repository root
python chess-supplementary/mvpa-finer/02_mvpa_finer_group_rsa.py
```

**Outputs** (saved to `chess-supplementary/mvpa-finer/results/<timestamp>_mvpa_finer_rsa/`):
- Statistical results per target (CSV files)
- `mvpa_group_stats.pkl`: Complete results dictionary

**Expected runtime**: ~30 seconds

### Step 3: Group-Level Decoding Analysis (Python)

```bash
python chess-supplementary/mvpa-finer/03_mvpa_finer_group_decoding.py
```

**Outputs** (saved to `chess-supplementary/mvpa-finer/results/<timestamp>_mvpa_finer_decoding/`):
- Statistical results per target (CSV files)
- `mvpa_group_stats.pkl`: Complete results dictionary

**Expected runtime**: ~30 seconds

### Step 4: Generate Tables and Figures

```bash
# Tables
python chess-supplementary/mvpa-finer/81_table_mvpa_finer_rsa.py
python chess-supplementary/mvpa-finer/82_table_mvpa_finer_decoding.py

# Figures
python chess-supplementary/mvpa-finer/92_plot_mvpa_finer_panel.py
```

## Key Results

**Finer distinctions**: Tests whether neural representations within checkmate boards differentiate between tactical motifs, piece counts, and move complexity
**Expert advantages**: Identifies which finer dimensions show stronger decoding/RSA in experts vs novices
**Strategic structure**: Reveals how strategic information is organized within the checkmate category

## File Structure

```
chess-supplementary/mvpa-finer/
├── README.md                              # This file
├── 01_roi_decoding_fine.m                 # MATLAB: Subject-level RSA and decoding
├── 02_mvpa_finer_group_rsa.py             # Python: Group-level RSA statistics
├── 03_mvpa_finer_group_decoding.py        # Python: Group-level decoding statistics
├── 81_table_mvpa_finer_rsa.py             # LaTeX/CSV table generation (RSA)
├── 82_table_mvpa_finer_decoding.py        # LaTeX/CSV table generation (decoding)
├── 92_plot_mvpa_finer_panel.py            # Figure generation
├── METHODS.md                             # Detailed methods from manuscript
├── DISCREPANCIES.md                       # Notes on analysis discrepancies
└── results/                               # Analysis outputs (timestamped)
    ├── <timestamp>_mvpa_finer_rsa/
    └── <timestamp>_mvpa_finer_decoding/
```

## Troubleshooting

**"Insufficient data for finer categories"**
- Verify stimulus metadata includes checkmate-specific columns (motif, n_pieces, n_moves, etc.)
- Check that 20 checkmate boards are correctly labeled

**Other issues**: See main MVPA troubleshooting (`chess-mvpa/README.md`)

## Citation

If you use this analysis in your work, please cite:

```
[Your paper citation here]
```

## Related Analyses

- **MVPA main** (`chess-mvpa/`): Main RSA and decoding analysis using all 40 boards
- **RDM intercorrelation** (`chess-supplementary/rdm-intercorrelation/`): Orthogonality of model RDMs

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
