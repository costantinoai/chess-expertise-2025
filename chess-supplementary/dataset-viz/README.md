# Dataset Visualization: Chess Stimuli and Colorbars

## Overview

This supplementary analysis provides visualization utilities for the chess stimulus dataset and generates standalone colorbars for all analyses. It includes (1) comprehensive visualization of all 40 chess board stimuli with metadata annotations, and (2) publication-ready colorbar figures extracted from actual data ranges to ensure consistency across all visualizations.

## Methods

### Stimulus Set Visualization

**Purpose**: Document and visualize the complete stimulus set used in the experiment.

**Stimulus dimensions**:
1. **Checkmate status (C/NC)**: Checkmate boards depict forced mate sequences resolvable in ≤4 moves; Non-checkmate counterparts have minimal interventions (typically one pawn repositioned) to neutralize the mating sequence while maintaining visual similarity
2. **Strategy type (SY)**: 5 tactical categories reflecting piece involvement and tactical features:
   - Queen-rook combinations (SY1, SY6): Coordinated linear tactics
   - Supported attacks with minor pieces (SY2, SY7): Knights/bishops supporting queen mate
   - Minor piece mating nets (SY3, SY8): Spatial confinement by knights and bishops
   - Bishop-driven forcing moves (SY4, SY9): Long-range control
   - One-move checkmates (SY5, SY10): Minimal calculation baselines
3. **Visual pairing (P)**: Stimuli matched for perceptual similarity while holding relational structure constant

**Metadata recorded**:
- Total number of pieces
- Count of legal moves
- Side of board occupied by defending king (left=0, right=1)
- Dominant tactical motif
- Number of moves to checkmate (where applicable)

### Colorbar Generation

**Purpose**: Create standalone colorbars for supplementary materials that match the actual data ranges and colormaps used in each analysis.

**Colorbars produced**:

1. **Behavioral Directional Matrices** (vertical)
   - Type: Diverging, symmetric around 0 (blue-white-red)
   - Label: 'Directional Preference'
   - Data source: Expert and novice directional dissimilarity matrices

2. **Positive 0-0.1 Range** (horizontal)
   - Type: Sequential (RdPu colormap)
   - Label: 'r' (correlation coefficient)
   - Range: Fixed 0.0 to 0.1

3. **Neurosynth Univariate Surfaces** (vertical)
   - Type: Diverging, symmetric around 0 (blue-white-red)
   - Label: 'z-score'
   - Data source: Neurosynth univariate z-maps projected to surface

4. **Neurosynth RSA Surfaces** (horizontal)
   - Type: Diverging, symmetric around 0 (blue-white-red)
   - Label: 'z-score'
   - Data source: Neurosynth RSA searchlight z-maps projected to surface

5. **Manifold PR Profiles Matrix** (vertical)
   - Type: Sequential (mako colormap)
   - Label: 'PR' (Participation Ratio)
   - Data source: Subject×ROI PR matrix

6. **Manifold PCA Components** (horizontal)
   - Type: Diverging, symmetric around 0 (blue-white-red)
   - Label: 'Loading'
   - Data source: PCA component loadings

## Dependencies

- Python 3.8+
- numpy, pandas
- matplotlib, seaborn
- nilearn (for surface projection of neurosynth z-maps)
- Common plotting utilities from `common/plotting_utils.py`

See `requirements.txt` in the repository root for complete dependencies.

## Data Requirements

### Input Files

**For stimulus visualization**:
- **Stimulus metadata**: `stimuli/stimuli.tsv`
  - Columns: `stim_id`, `check`, `strategy`, `visual`, `fen`, `n_pieces`, `n_moves`, etc.
- **Chess board images**: `stimuli/images/*.png`
  - One PNG image per stimulus board

**For colorbar generation**:
- **Behavioral results**: `chess-behavioral/results/*_behavioral_rsa/expert_directional_dsm.npy`
- **Neurosynth univariate**: `chess-neurosynth/results/*_neurosynth_univariate/zmap_*.nii.gz`
- **Neurosynth RSA**: `chess-neurosynth/results/*_neurosynth_rsa/zmap_*.nii.gz`
- **Manifold results**: `chess-manifold/results/*_manifold/pr_results.pkl`

### Data Location

Set the external data root once in `common/constants.py` (all analysis paths are derived from it):

```python
# Base folder containing BIDS/, rois/, neurosynth/, stimuli/
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
```

## Running the Analysis

### Step 1: Generate Chess Board Grid Visualization

```bash
# From repository root
python chess-supplementary/dataset-viz/91_plot_dataset_viz.py
```

**Outputs** (saved to `chess-supplementary/dataset-viz/results/<timestamp>_dataset_viz/figures/`):
- `chess_stimuli_grid.pdf`: Complete grid of all 40 chess boards with labels
- `chess_stimuli_grid.svg`: SVG version

**Expected runtime**: ~30 seconds

### Step 2: Generate Standalone Colorbars

```bash
python chess-supplementary/dataset-viz/91_plot_colorbars.py
```

**Outputs** (saved to `chess-supplementary/dataset-viz/results/<timestamp>_colorbars/figures/`):
- `colorbar_behavioral_directional_vertical.svg` (and .pdf)
- `colorbar_0_to_01_horizontal.svg` (and .pdf)
- `colorbar_neurosynth_univariate_vertical.svg` (and .pdf)
- `colorbar_neurosynth_rsa_horizontal.svg` (and .pdf)
- `colorbar_manifold_pr_profiles_vertical.svg` (and .pdf)
- `colorbar_manifold_pca_components_horizontal.svg` (and .pdf)

**Expected runtime**: ~1-2 minutes

## Key Results

**Stimulus set characteristics**:
- N=40 chess boards (20 checkmate, 20 non-checkmate)
- 5 strategy types with 4 exemplars each (2 checkmate, 2 non-checkmate per strategy)
- 20 visual pairs (matched for perceptual similarity)
- Systematic variation in piece count, legal moves, and tactical motifs

**Colorbars**:
- All colorbars extracted from actual data ranges to ensure consistency
- Both horizontal and vertical orientations provided for flexible figure composition
- Sequential and diverging colormaps selected based on data type

## File Structure

```
chess-supplementary/dataset-viz/
├── README.md                          # This file
├── 91_plot_dataset_viz.py             # Chess board grid visualization
├── 91_plot_colorbars.py               # Standalone colorbar generation
├── METHODS.md                         # Detailed methods from manuscript
├── DISCREPANCIES.md                   # Notes on analysis discrepancies
├── modules/
│   ├── __init__.py
│   └── plotting.py                    # Plotting utilities
└── results/                           # Analysis outputs (timestamped)
    ├── <timestamp>_dataset_viz/
    │   └── figures/                   # Chess board grid figures
    └── <timestamp>_colorbars/
        └── figures/                   # Standalone colorbar figures
```

## Troubleshooting

### Common Issues

**"FileNotFoundError: Stimulus images not found"**
- Verify chess board images exist: `data/BIDS/stimuli/images/*.png`
- Check external data root in `common/constants.py`

**"FileNotFoundError: Analysis results not found"**
- Run prerequisite analyses before generating colorbars:
  - `chess-behavioral/01_behavioral_rsa.py`
  - `chess-neurosynth/01_univariate_neurosynth.py`
  - `chess-neurosynth/02_rsa_neurosynth.py`
  - `chess-manifold/01_manifold_analysis.py`

**"Missing stimulus metadata"**
- Ensure `data/BIDS/stimuli.tsv` exists with required columns
- Check FEN strings are valid

**Import errors**
- Run from repository root (not from `chess-supplementary/dataset-viz/`)
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Citation

If you use this analysis in your work, please cite:

```
[Your paper citation here]
```

## Related Analyses

- All main analyses (colorbar references)
- **Neurosynth term visualization** (`chess-supplementary/neurosynth-terms/`): Individual term map visualization

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
