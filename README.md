# Neural and Behavioral Representations of Chess Expertise

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9–3.12](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](https://www.python.org/downloads/)
[![BIDS](https://img.shields.io/badge/BIDS-compliant-green.svg)](https://bids.neuroimaging.io/)

This repository contains the complete analysis code for our study investigating how chess expertise shapes neural and behavioral representations of chess board positions. We combine behavioral preference judgments, fMRI multi-voxel pattern analysis (MVPA), univariate analyses, and meta-analytic correlations to characterize expertise-related differences in representational structure.

## Overview

**Participants**: 40 adults (20 expert chess players, 20 novices)
**Task**: 1-back preference judgment task with 40 chess board stimuli (20 checkmate, 20 non-checkmate positions)
**Neuroimaging**: 3T fMRI, whole-brain coverage, 2mm isotropic resolution
**Behavioral**: Pairwise preference judgments during scanning
**Eye-tracking**: Gaze coordinates during chess board viewing (supplementary)

### Key Findings

- **Behavioral**: Expert preferences correlate with checkmate status (r=0.49) and strategy (r=0.20); novices show no model correlations
- **Neural (MVPA)**: Expertise modulates neural similarity structure in fronto-parietal regions
- **Manifold**: Experts show higher-dimensional representations in task-relevant cortical regions
- **Meta-analytic**: Expertise differences localize to brain networks associated with working memory and spatial navigation

## Table of Contents

- [Installation](#installation)
- [Data Download and Setup](#data-download-and-setup)
- [Folder Setup](#folder-setup)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running Analyses](#running-analyses)
- [Outputs](#outputs)
- [Analysis Overview](#analysis-overview)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- **Python 3.9–3.12** (tested on 3.11)
- **MATLAB R2019b or higher** (optional, only for subject-level GLM and MVPA. Group stats are computed in Python)
- **SPM12** (optional, for first-level GLM preprocessing)

### Clone Repository

```bash
git clone https://github.com/your-username/chess-expertise-2025.git
cd chess-expertise-2025
```

### Install Python Dependencies

We recommend using a virtual environment:

```bash
# Create virtual environment
conda env create -f environment.yml
conda activate chess-expertise

# Option B: Use pip in virtualenv
pip install -r requirements.txt
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
kpsewhich siunitx.sty  && echo OK || echo MISSING
```

See docs/TABLES.md for details on the table style policy and validation.

### Install MATLAB Dependencies (Optional)

If running subject-level GLM and/or MVPA analyses:

1. Install [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
2. Install [CoSMoMVPA](http://www.cosmomvpa.org/)
3. Add both to your MATLAB path

## Data Download and Setup

### Download Data

**Download link**: [PLACEHOLDER - Data will be available on OpenNeuro/OSF upon publication/request]

The (minimal) dataset is organized according to the Brain Imaging Data Structure (BIDS) specification.

### Data Organization

After downloading, extract the data and organize as follows (see also Folder Setup below):

```
/path/to/your/data/
└── BIDS/
    ├── participants.tsv                    # Participant metadata
    ├── participants.json                   # Column descriptions
    ├── stimuli.tsv                         # Stimulus metadata
    ├── dataset_description.json            # Dataset info
    ├── sub-01/
    │   ├── anat/                           # Anatomical scans
    │   │   └── sub-01_T1w.nii.gz
    │   └── func/                           # Functional scans
    │       ├── sub-01_task-exp_run-1_bold.nii.gz
    │       ├── sub-01_task-exp_run-1_events.tsv
    │       └── ...
    ├── sub-02/
    │   └── ...
    └── derivatives/                        # Preprocessed/analyzed data
        ├── fmriprep/                       # fMRIPrep outputs (preprocessed data)
        ├── spm-glm/                        # SPM first-level GLMs
        │   ├── unsmoothed/                 # Unsmoothed beta images (for MVPA)
        │   └── smooth4/                    # Smoothed (4mm) beta images (for univariate)
        ├── mvpa-rsa/                       # Subject-level RSA results
        ├── mvpa-searchlight/               # Searchlight maps
        └── eyetracking/                    # Eye-tracking data
```

## Folder Setup

All analyses read their inputs from a **single external data root**. Configure this once and all paths are derived from it.

1) Choose a folder on your system to hold all inputs:

```
/path/to/manuscript-data/
├── BIDS/                  # BIDS dataset (raw + derivatives)
│   ├── participants.tsv
│   └── derivatives/
├── rois/                  # Atlases and ROI metadata
├── neurosynth/            # Term maps for meta-analytic correlations
└── stimuli/               # Stimulus metadata (e.g., stimuli.tsv)
```

2) Point the repository to this folder by editing `common/constants.py`:

```python
# Base folder containing BIDS/, rois/, neurosynth/, stimuli/
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
# Do not edit BIDS_ROOT directly — it is derived from the external root
```

## Expected Inputs

The repository expects the following files and layout under `_EXTERNAL_DATA_ROOT`. Adjust the root path in `common/constants.py` if your data lives elsewhere.

Top level
```
<EXTERNAL_DATA_ROOT>/
├── BIDS/
├── rois/
├── neurosynth/
└── stimuli/
```

BIDS (raw + derivatives)
```
BIDS/
├── dataset_description.json
├── participants.json
├── participants.tsv                     # participant_id, group
├── sub-01/func/
│   ├── sub-01_task-exp_run-1_bold.nii.gz
│   ├── sub-01_task-exp_run-1_events.tsv
│   └── ... (runs 1–6 per subject)
└── derivatives/
    ├── SPM/
    │   ├── GLM-unsmoothed/sub-*/exp/    # beta_*.nii.gz, SPM.mat, mask.nii.gz
    │   └── GLM-smooth4/
    │       └── group/
    │           ├── spmT_exp_gt_nonexp_all_gt_rest.nii.gz
    │           └── spmT_exp_gt_nonexp_check_gt_nocheck.nii.gz
    ├── mvpa-rsa/sub-*/sub-*_space-MNI152NLin2009cAsym_roi-glasser_rdm.tsv
    ├── mvpa-decoding/sub-*/sub-*_space-MNI152NLin2009cAsym_roi-glasser_accuracy.tsv
    ├── rsa_searchlight/sub-*/
    │   ├── sub-*_desc-searchlight_checkmate_stat-r_map.nii.gz
    │   ├── sub-*_desc-searchlight_strategy_stat-r_map.nii.gz
    │   └── sub-*_desc-searchlight_visual_similarity_stat-r_map.nii.gz
    └── eye-tracking/sub-*/func/
        ├── sub-*_task-exp_run-1_space-MNI152NLin2009cAsym_desc-1to6_eyetrack.tsv
        └── sub-*_task-exp_run-1_space-MNI152NLin2009cAsym_desc-1to6_eyetrack.json
```

ROIs and parcellations
```
rois/
├── glasser22/
│   ├── tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-22_bilateral_resampled.nii.gz
│   └── region_info.tsv
├── glasser180/
│   ├── tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-180_bilateral_resampled.nii.gz
│   └── region_info.tsv
└── glasser180-surface/
    ├── lh.HCPMMP1.annot
    └── rh.HCPMMP1.annot
```

Neurosynth term maps
```
neurosynth/terms/
├── 1_working memory.nii.gz
├── 2_navigation.nii.gz
├── 3_memory retrieval.nii.gz
├── 4_language network.nii.gz
├── 5_object recognition.nii.gz
├── 6_face recognition.nii.gz
└── 7_early visual.nii.gz
```

Stimuli metadata and images
```
stimuli/
├── stimuli.tsv                           # stim_id, check, strategy, visual, ...
└── images/*.png                          # 40 chess board images (C* and NC* files)
```

## Running Analyses

You have two options to run the code. Either you run the single python scripts one by one (in which case, make sure to cd into the analysis folder first, e.g., `cd chess-manifold` --> `conda activate ml` --> `python 01_....py`) or you can run all the scripts at once (this can be run directly from the repo root, see below).

### Option A: Automated Pipeline (bash)

Run all analyses with one command. This script also disables pylustrator during the run and backs up existing results.

```bash
# From repository root
./run_all_analyses.sh --levels analysis,tables,figures
```

Key options:
- `--levels analysis,tables,figures`  Select which stages to run
- `--sequential`                      Run jobs sequentially (stream logs to console)
- `--python /path/to/python`          Use explicit Python interpreter (overrides conda env)

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
python 01_behavioral_rsa.py                 # Main analysis (~2 min)
python 81_table_behavioral_correlations.py  # Tables (~10 sec)
python 91_plot_behavioral_panels.py         # Figures (~30 sec)
```

See individual analysis READMEs for details:
- [`chess-behavioral/README.md`](chess-behavioral/README.md) - Behavioral RSA
- [`chess-manifold/README.md`](chess-manifold/README.md) - Participation ratio analysis
- [`chess-mvpa/README.md`](chess-mvpa/README.md) - MVPA (RSA and decoding)
- [`chess-neurosynth/README.md`](chess-neurosynth/README.md) - Meta-analytic correlations
- [`chess-supplementary/*/README.md`](chess-supplementary/) - Supplementary analyses

## Outputs

All analyses save artefacts to the results directory within their respective folders:

```
chess-{analysis}/results/{analysis_name}/
├── *.npy                   # Numerical arrays (RDMs, coordinates, etc.)
├── *.pkl                   # Python objects (results dictionaries)
├── *.csv                   # Summary tables
├── *.log                   # Execution logs
├── {script_name}.py        # Copy of analysis script (for reproducibility)
├── tables/                 # LaTeX and CSV tables
│   └── *.tex, *.csv
└── figures/                # Publication-ready figures
    ├── *.svg, *.pdf        # Individual panels
    └── panels/             # Multi-panel figures
        └── *_panel.pdf
```

Additionally, publication-ready PDFs and LaTeX tables are copied to a consolidated bundle under `results-bundle/` for easy sharing:

```
results-bundle/
├── figures/    # Final panels (PDF)
└── tables/     # Final tables (LaTeX)
```

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

## Citation

If you use this code or data in your research, please cite:

[PLACEHOLDER]

## Acknowledgments

- Participants who contributed data to this study
- [Human Connectome Project](https://www.humanconnectome.org/) for the Glasser parcellation
- [Neurosynth](https://neurosynth.org/) for meta-analytic term maps
- [fMRIPrep](https://fmriprep.org/) developers for preprocessing pipeline
- [CoSMoMVPA](http://www.cosmomvpa.org/) developers for MVPA tools
