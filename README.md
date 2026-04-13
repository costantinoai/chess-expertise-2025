# Neural and Behavioral Representations of Chess Expertise

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9–3.12](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](https://www.python.org/downloads/)
[![BIDS](https://img.shields.io/badge/BIDS-compliant-green.svg)](https://bids.neuroimaging.io/)

This repository contains the complete analysis code for our study investigating how chess expertise shapes neural and behavioral representations of chess board positions. We combine behavioral preference judgments, fMRI multi-voxel pattern analysis (MVPA), univariate analyses, and meta-analytic correlations to characterize expertise-related differences in representational structure.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/costantinoai/chess-expertise-2025.git
cd chess-expertise-2025
conda env create -f environment.yml && conda activate chess-expertise
pip install -e .

# 2. Download data from RDR (doi.org/10.48804/VVCEWP)
#    For most users: bundles A (core, 20 MB) + E (derivatives, 184 MB)

# 3. Extract into a folder and point the repo at it
export CHESS_DATA_ROOT=/path/to/extracted/data   # parent of BIDS/

# 4. Reproduce all group stats, tables, and figures
./run_all_analyses.sh group
```

That's it. All outputs go to `BIDS/derivatives/`. No source files need editing — `CHESS_DATA_ROOT` is the only configuration.

## Overview

**Participants**: 40 adults (20 expert chess players, 20 novices)
**Task**: 1-back preference judgment task with 40 chess board stimuli (20 checkmate, 20 non-checkmate positions)
**Neuroimaging**: 3T fMRI, whole-brain coverage, 2mm isotropic resolution
**Behavioral**: Pairwise preference judgments during scanning

### Key Findings

- **Behavioral**: Expert preferences correlate with checkmate status (r=0.73) and strategy (r=0.25) using count-normalized RDMs; novices show no model correlations
- **Neural (MVPA)**: Expertise modulates neural similarity structure in fronto-parietal regions
- **Manifold**: Experts show higher-dimensional representations in task-relevant cortical regions
- **Meta-analytic**: Expertise differences localize to brain networks associated with working memory and spatial navigation

## Installation

### Prerequisites

- **Python 3.9–3.12** (tested on 3.11)
- **MATLAB R2019b or higher** (optional, only for subject-level GLM and MVPA. Group stats are computed in Python)
- **SPM12** (optional, for first-level GLM. This can be skipped: result from the GLM are provided in the RDR KU Leuven companion data repository.)

### Clone Repository

```bash
git clone https://github.com/costantinoai/chess-expertise-2025.git
cd chess-expertise-2025
```

### Install Python Dependencies

We recommend using a virtual environment:

```bash
# Create virtual environment
conda env create -f environment.yml
conda activate chess-expertise

# Install the project packages (common + analyses) in editable mode
pip install -e .

# Option B: Use pip in virtualenv
pip install -r requirements.txt && pip install -e .
```

### LaTeX (optional, for compiling tables)

If you want to validate/compile the generated LaTeX tables locally with `pdflatex`,
install the following TeX packages:

Debian/Ubuntu:

```bash
sudo apt -y install texlive-science texlive-latex-extra texlive-latex-recommended texlive-fonts-recommended
```

Quick checks:

```bash
pdflatex --version
kpsewhich booktabs.sty && echo OK || echo MISSING
kpsewhich multirow.sty && echo OK || echo MISSING
kpsewhich siunitx.sty && echo OK || echo MISSING
```

See docs/TABLES.md for details on the table style policy and validation.

### Install MATLAB Dependencies (Optional)

If running subject-level GLM and/or MVPA analyses:

1. Install [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
2. Install [CoSMoMVPA](http://www.cosmomvpa.org/)
3. Add both to your MATLAB path

**Important**: MATLAB does not allow running scripts whose filenames start with a number. The MATLAB scripts in this repository are named with numeric prefixes (e.g., `01_roi_mvpa_subject.m`) for ordering consistency with the Python scripts. To run them, either:

- Rename the file by removing the numeric prefix (e.g., `01_roi_mvpa_subject.m` → `roi_mvpa_subject.m`), or
- Call them via `run('01_roi_mvpa_subject.m')` from the MATLAB command window

## Data Setup

The BIDS dataset is on the KU Leuven Research Data Repository:
**[doi.org/10.48804/VVCEWP](https://doi.org/10.48804/VVCEWP)**.
Most users need only bundles **A** (core, 20 MB) and **E** (derivatives, 184 MB).
See the RDR README for the full bundle table, extraction steps, and per-analysis
dependency chart.

After extracting, set one environment variable:

```bash
# Linux / macOS
export CHESS_DATA_ROOT=/path/to/manuscript-data   # the parent of BIDS/

# Windows (PowerShell)
$env:CHESS_DATA_ROOT = "C:\path\to\manuscript-data"
```

All paths are derived automatically from `CHESS_DATA_ROOT` — no source files to edit. The expected layout is:

```
$CHESS_DATA_ROOT/
└── BIDS/
    ├── participants.tsv
    ├── stimuli/
    ├── sourcedata/atlases/{glasser22, glasser180, cab-np, neurosynth}/
    ├── sub-01/ ... sub-44/
    └── derivatives/
        ├── fmriprep/                            # bundle C
        ├── fmriprep_spm-{un,}smoothed/          # bundle D
        ├── fmriprep_spm-unsmoothed_{rsa,decoding,searchlight-rsa,manifold,...}/
        ├── behavioral-rsa/, bidsmreye/, task-engagement/, skill-gradient/, ...
        └── group-results/                       # group-level stats, tables, figures
            ├── behavioral/{data,tables,figures}/
            ├── manifold/, mvpa/, neurosynth/
            └── supplementary/<analysis>/
```

**Platform notes:**
- **Linux**: Works out of the box.
- **macOS**: The pipeline script needs bash 4+ (`brew install bash`), or run scripts individually with Python.
- **Windows**: Run Python scripts directly or use WSL/Git Bash for the pipeline script.

## Running Analyses

You have two options to run the code. Either you run the single python scripts one by one (in which case, make sure to cd into the analysis folder first, e.g., `cd chess-manifold` --> `conda activate chess-expertise` --> `python 01_....py`) or you can run all the scripts at once (this can be run directly from the repo root, see below).

### Option A: Automated Pipeline (bash)

Run all analyses with one command. This script also disables pylustrator during the run and backs up existing results.

```bash
# From repository root
./run_all_analyses.sh --levels analysis,tables,figures
```

Key options:
- `--levels analysis,tables,figures` Select which stages to run
- `--sequential` Run jobs sequentially (stream logs to console)
- `--python /path/to/python` Use explicit Python interpreter (overrides conda env)

Environment note:
- The script looks for a conda environment by name. Set `CONDA_ENV` at the top of `run_all_analyses.sh` (default is `ml`) to match your environment name (e.g., `chess-expertise`), or pass `--python $(which python)` to use the currently activated environment.

This will execute analyses in the following order:
1. Behavioral RSA
2. Manifold dimensionality analysis
3. MVPA group analyses
4. Neurosynth meta-analytic correlations
5. All supplementary analyses

**Total runtime**: ~30 minutes (depends on system and chosen options)

### Option B: Run Individually (Python)

Each analysis directory has detailed instructions in its README. Basic workflow:

```bash
# Always run from the analysis folder
cd chess-behavioral
conda activate <your-new-env-name>
python 01_behavioral_rsa_subject.py # 0x: Per-subject → derivatives/
python 11_behavioral_rsa_group.py # 1x: Group aggregate → derivatives/group-results/
python 81_table_behavioral_correlations.py # 8x: Tables (from derivatives/group-results/)
python 91_plot_behavioral_panels.py # 9x: Figures (from derivatives/group-results/)
```

### Script numbering convention

| Prefix | Level | Reads from | Writes to |
|--------|-------|------------|-----------|
| **0x** | Subject | BIDS raw / SPM derivatives | `BIDS/derivatives/` (per-subject data) |
| **1x** | Group | `BIDS/derivatives/` | `BIDS/derivatives/group-results/` (group stats) |
| **8x** | Tables | `derivatives/group-results/` | `derivatives/group-results/` (formatted tables) |
| **9x** | Figures | `derivatives/group-results/` + derivatives | `derivatives/group-results/` (rendered figures) |

All outputs — subject-level and group-level — live under `BIDS/derivatives/`.
Group-level stats, tables, and figures are written to `derivatives/group-results/`.
Bundle E contains everything needed to reproduce all results from the code repo.

Subject-level stages (0x scripts, MATLAB pipelines under `chess-mvpa/`, etc.)
should be re-run before the corresponding group stages if the derivatives
themselves are being rebuilt.

See individual analysis READMEs for details:
- [`chess-behavioral/README.md`](chess-behavioral/README.md) - Behavioral RSA
- [`chess-manifold/README.md`](chess-manifold/README.md) - Participation ratio analysis
- [`chess-mvpa/README.md`](chess-mvpa/README.md) - MVPA (RSA and decoding)
- [`chess-neurosynth/README.md`](chess-neurosynth/README.md) - Meta-analytic correlations
- [`chess-supplementary/*/README.md`](chess-supplementary/) - Supplementary analyses

## Analysis Overview

### Main Analyses

1. **Behavioral RSA** (`chess-behavioral/`)
 - Pairwise preference judgments → behavioral RDMs
 - Correlation with theoretical models (checkmate, strategy, visual)
 - Expert vs novice comparison

2. **Manifold Dimensionality** (`chess-manifold/`)
 - Participation ratio (PR) quantifies effective dimensionality
 - 22 bilateral cortical ROIs
 - Expert vs novice classification and group comparisons

3. **MVPA** (`chess-mvpa/`)
 - **RSA**: Neural RDM correlation with model RDMs (22 ROIs)
 - **Decoding**: SVM classification of checkmate/strategy (22 ROIs)
 - Expert vs novice comparisons with FDR correction

4. **Neurosynth Meta-Analysis** (`chess-neurosynth/`)
 - **Univariate**: GLM t-maps correlated with cognitive term maps
 - **RSA**: Searchlight contrast maps correlated with cognitive terms
 - Seven terms: working memory, retrieval, navigation, language, face, early visual, object

### Supplementary Analyses

See [`chess-supplementary/README.md`](chess-supplementary/README.md) for details on:
- Split-half reliability of behavioral RDMs
- Eye-tracking visual strategy differences
- Finer-grained MVPA within checkmate positions
- RDM intercorrelation and variance partitioning
- ROI-level summaries for 180 bilateral Glasser regions
- Dataset and colorbar visualization
- Task engagement diagnostics and board preference feature drivers
- Skill gradient analysis (Elo and familiarisation correlations)
- Subcortical ROI analysis (CAB-NP atlas, 9 bilateral ROIs)

## Reproducibility

### What to Expect

**Byte-identical outputs:**
- All publication tables (`derivatives/group-results/*/tables/*.tex`)
- Behavioral analysis data files (all `.npy`, `.pkl`, `.csv`)
- Within-group t-tests (`*_vs_chance.csv`)
- Behavioral reliability, eyetracking, RDM intercorrelation, and task engagement CSVs

**Expected floating-point variation (10th--16th decimal place):**
- Between-group t-tests (`*_experts_vs_novices.csv`): CI bounds vary at ~1e-10 due to BLAS/LAPACK non-determinism
- Manifold PR statistics (`pr_*.csv`, `pr_results.pkl`)
- Neurosynth correlation CSVs (NIfTI resampling operations)
- Skill-gradient permutation-based statistics

These variations do **not** affect significance decisions, publication tables, or any scientific conclusions. The pipeline sets `OMP_NUM_THREADS=1` to minimise floating-point non-determinism.

**Non-deterministic outputs (expected to differ):**
- SVG/PDF figures (matplotlib rendering IDs and timestamps)
- Script copies in `derivatives/group-results/` directories (provenance snapshots)

### Environment

The pipeline was validated on Ubuntu 24.04 with Python 3.11.6 (numpy 1.26.4, scipy 1.15.1, pandas 2.3.1, nilearn 0.11.1, scikit-learn 1.7.1). See `environment.yml` for the full specification.

## Citation

If you use this code or data in your research, please cite:

> Costantino, A.I. et al. (2026). Low-Dimensional and Optimised Representations of High-Level Information in the Expert Brain. *Under review*.

- **Code** (this repository, any version): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19392282.svg)](https://doi.org/10.5281/zenodo.19392282) — the concept DOI always resolves to the latest tagged release.
- **Data** (BIDS dataset on KU Leuven RDR): [10.48804/VVCEWP](https://doi.org/10.48804/VVCEWP)

The code repository contains every analysis pipeline needed to reproduce the
paper. Group-level statistics, manuscript tables, and publication figures live
under `derivatives/group-results/<analysis>/{data,tables,figures}/` in the
RDR dataset (bundle E). Raw MRI, preprocessed data, and subject-level
analysis derivatives also live in the RDR repository.

## Acknowledgments

- Participants who contributed data to this study
- [Human Connectome Project](https://www.humanconnectome.org/) for the Glasser parcellation
- [Neurosynth](https://neurosynth.org/) for meta-analytic term maps
- [fMRIPrep](https://fmriprep.org/) developers for preprocessing pipeline
- [CoSMoMVPA](http://www.cosmomvpa.org/) developers for MVPA tools
