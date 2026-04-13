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

- **Behavioral**: Expert preferences correlate with checkmate status (r=0.73) and strategy (r=0.25) using count-normalized RDMs; novices show no model correlations
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
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

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

## Data Download and Setup

### Download Data

The BIDS dataset is deposited in the KU Leuven Research Data Repository (RDR):
**DOI: [10.48804/VVCEWP](https://doi.org/10.48804/VVCEWP)**. The repository
entry hosts raw MRI, preprocessed data, all subject-level analysis derivatives
and a mirror of this repo's group-level `derivatives/group-results/` snapshot, split into six
layered bundles so you only have to download what your workflow needs:

| Bundle | Approx. size | Contents | When to download |
|:--:|---|---|---|
| **A** `core` | ~20 MB | Sidecars, `participants.tsv`, `stimuli/`, `sourcedata/atlases/` | **Mandatory** for every user. |
| **B** `raw` | ~39 GB | `sub-*/anat/`, `sub-*/func/*_bold.nii.gz`, `sub-*/beh/` | Re-running fMRIPrep from scratch. |
| **C** `fmriprep` | ~187 GB | `derivatives/fmriprep/` | Re-running the SPM first-level GLM. |
| **D** `spm` | ~30 GB | `derivatives/fmriprep_spm-{un,}smoothed/` (subject betas + group contrasts) | Re-running any subject-level MVPA / manifold / searchlight; also neurosynth univariate and univariate-rois. |
| **E** `analyses` | ~260 MB | `derivatives/fmriprep_spm-unsmoothed_{rsa,decoding,searchlight-rsa,manifold,rsa-run-matched,rsa-subcortical,decoding-subcortical}/`, `derivatives/behavioral-rsa/`, `derivatives/bidsmreye/` | **Most users** — enough to regenerate every group stat, table, and figure together with this code repo. |

The typical **"reproduce the paper's group stats and figures"** path is bundles
**A + E** (~370 MB total), pointing `CHESS_DATA_ROOT` at the extracted BIDS
root. See the dataset-side `README` on RDR for the exact extraction steps and
the full dependency chain (`B → fmriprep → C → SPM → D → subject scripts → E`).

The dataset is organised according to the Brain Imaging Data Structure (BIDS)
specification (v1.10.0).

### Data Organization

After downloading, extract the data. The dataset follows the BIDS specification. All data -- including stimuli, atlases, and analysis derivatives -- lives under `BIDS/`. See [Folder Setup](#folder-setup) and [Expected Inputs](#expected-inputs) for the full directory tree.

## Folder Setup

All analyses read their inputs from a **single external data root**. Configure this once and all paths are derived from it.

1) Choose a folder on your system to hold the BIDS dataset:

```
/path/to/manuscript-data/
└── BIDS/ # BIDS dataset (raw + derivatives)
 ├── participants.tsv
 ├── stimuli/ # Stimulus images and metadata
 ├── sourcedata/atlases/ # Reference atlases (Glasser, CAB-NP, Neurosynth)
 └── derivatives/
 ├── fmriprep/ # fMRIPrep outputs
 ├── fmriprep_spm-unsmoothed/ # SPM first-level GLM on unsmoothed BOLD
 ├── fmriprep_spm-smoothed/ # SPM first-level GLM on 4 mm smoothed BOLD
 ├── fmriprep_spm-unsmoothed_rsa/ # ROI RSA (Glasser-22)
 ├── fmriprep_spm-unsmoothed_decoding/ # ROI decoding (Glasser-22)
 ├── fmriprep_spm-unsmoothed_searchlight-rsa/ # Whole-brain searchlight RSA
 ├── fmriprep_spm-unsmoothed_manifold/ # Per-ROI neural Participation Ratio
 ├── fmriprep_spm-unsmoothed_rsa-run-matched/ # Run-matched ROI RSA (supplementary)
 ├── fmriprep_spm-unsmoothed_rsa-subcortical/ # Subcortical ROI RSA (CAB-NP)
 ├── fmriprep_spm-unsmoothed_decoding-subcortical/ # Subcortical ROI decoding (CAB-NP)
 ├── behavioral-rsa/ # Per-subject behavioural preference RDMs
 └── bidsmreye/ # BidsMReye gaze estimates
```

2) Point the repository to your local data folder by setting the `CHESS_DATA_ROOT`
environment variable:

```bash
export CHESS_DATA_ROOT=/path/to/manuscript-data
```

Both Python (`common/constants.py`) and MATLAB (`common/chess_config.m`) read this
variable automatically. All other paths (BIDS root, derivatives, atlases, etc.) are
derived from this single root. No source files need to be edited.

## Expected Inputs

The repository expects the following layout under `_EXTERNAL_DATA_ROOT`. Everything lives inside `BIDS/` (including stimuli, atlases, and neurosynth term maps under `derivatives/atlases/`).

```
<EXTERNAL_DATA_ROOT>/
└── BIDS/
 ├── dataset_description.json
 ├── participants.tsv # participant_id, age, sex, group, rating
 ├── participants.json
 ├── README
 ├── task-exp_bold.json # TaskName via BIDS inheritance
 ├── task-familiarisation_beh.json # Beh sidecar via BIDS inheritance
 ├── stimuli/
 │ ├── stimuli.tsv # Stimulus metadata (stim_id, check, strategy, ...)
 │ └── *.png # 40 chess board images
 ├── sourcedata/atlases/ # Reference atlases (never consumed
 │ ├── glasser22/ # as derivatives — they are the input
 │ ├── glasser180/ # side of the analysis pipeline).
 │ ├── glasser180-surface/ # Surface parcellation (.annot)
 │ ├── cab-np/ # Subcortical ROIs (CAB-NP)
 │ └── neurosynth/terms/ # Meta-analytic term maps
 ├── sub-01/
 │ ├── anat/sub-01_T1w.nii.gz
 │ ├── func/sub-01_task-exp_run-{1..6}_bold.nii.gz
 │ ├── func/sub-01_task-exp_run-{1..6}_events.tsv
 │ └── beh/sub-01_task-familiarisation_beh.tsv
 ├── sub-02/ ... sub-44/
 └── derivatives/
 ├── fmriprep/ # fMRIPrep preprocessed data
 ├── fmriprep_spm-unsmoothed/sub-*/exp/ # Unsmoothed SPM betas (for MVPA)
 ├── fmriprep_spm-smoothed/ # Smoothed SPM first + group levels
 │ ├── sub-*/exp/ # Smoothed SPM betas
 │ └── group/ # Group contrasts (used by neurosynth)
 ├── fmriprep_spm-unsmoothed_rsa/sub-*/ # ROI RSA Pearson-r TSVs (Glasser-22)
 ├── fmriprep_spm-unsmoothed_decoding/sub-*/ # ROI SVM accuracy TSVs (Glasser-22)
 ├── fmriprep_spm-unsmoothed_searchlight-rsa/sub-*/ # Whole-brain searchlight r-maps
 ├── fmriprep_spm-unsmoothed_manifold/sub-*/ # Per-ROI Participation Ratio TSVs
 ├── fmriprep_spm-unsmoothed_rsa-run-matched/sub-*/
 ├── fmriprep_spm-unsmoothed_rsa-subcortical/sub-*/
 ├── fmriprep_spm-unsmoothed_decoding-subcortical/sub-*/
 ├── behavioral-rsa/sub-*/ # Per-subject preference RDMs
 └── bidsmreye/sub-*/ # Gaze-position estimates
```

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

## Outputs

Every Python analysis writes group-level outputs into `BIDS/derivatives/group-results/`. Each analysis owns one subfolder with four fixed buckets:

```
BIDS/derivatives/group-results/
├── behavioral/
│   ├── data/       # Numerical aggregates (CSV, TSV, JSON, NPY, PKL)
│   ├── tables/     # Formatted tables (LaTeX, CSV)
│   ├── figures/    # Rendered figures (PDF, PNG, SVG)
│   └── logs/       # Script copies and execution logs
├── manifold/{data,tables,figures,logs}/
├── mvpa/{data,tables,figures,logs}/
├── neurosynth/{data,tables,figures,logs}/
└── supplementary/
    ├── behavioral-reliability/{data,tables,figures,logs}/
    ├── eyetracking/{data,tables,figures,logs}/
    ├── mvpa-finer/{data,tables,figures,logs}/
    ├── neurosynth-terms/{figures,logs}/
    ├── rdm-intercorrelation/{data,tables,figures,logs}/
    ├── rsa-rois/{data,tables,figures,logs}/
    ├── run-matching/{data,tables,logs}/
    ├── skill-gradient/{data,figures,logs}/
    ├── subcortical-rois/{data,figures,logs}/
    ├── task-engagement/{data,figures,logs}/
    └── univariate-rois/{data,tables,figures,logs}/
```

This tree is regenerated by running `./run_all_analyses.sh group` (every 1x/8x/9x script). It lives inside the BIDS derivatives tree and ships as part of bundle E.

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
