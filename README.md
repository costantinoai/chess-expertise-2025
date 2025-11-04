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
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Option A: Use conda/mamba (recommended)
mamba env create -f environment.yml   # or: conda env create -f environment.yml
conda activate chess-expertise

# Option B: Use pip in virtualenv
pip install -r requirements.txt
```

### LaTeX (optional, for compiling tables)

If you want to validate/compile the generated LaTeX tables locally with `pdflatex`,
install the following TeX packages:

Debian/Ubuntu:

```bash
sudo apt -y install texlive-science \
                 texlive-latex-extra texlive-latex-recommended texlive-fonts-recommended
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

If running subject-level MVPA analyses:

1. Install [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
2. Install [CoSMoMVPA](http://www.cosmomvpa.org/)
3. Add both to your MATLAB path

## Data Download and Setup

### Download Data

**Download link**: [PLACEHOLDER - Data will be available on OpenNeuro/OSF]

The dataset is organized according to the Brain Imaging Data Structure (BIDS) specification.

**Dataset size**: ~XX GB (raw fMRI data + derivatives)

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

### Alternative: Use Symbolic Links

If you have the data stored elsewhere, create a symbolic link:

```bash
# Linux/Mac
ln -s /path/to/your/BIDS/data /path/to/chess-expertise-2025/data/BIDS

# Windows (run as Administrator)
mklink /D "C:\path\to\chess-expertise-2025\data\BIDS" "D:\path\to\your\BIDS\data"
```

## Folder Setup

All analyses read their inputs from a single external data root. Configure this once and all paths are derived from it.

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

Note: do not rely on environment variables or implicit defaults. All paths are explicit and validated at runtime.

Default for this repo: `_EXTERNAL_DATA_ROOT` is already set to `/media/costantino_ai/eik-T9/manuscript-data`. If your data lives elsewhere, update this path.

## Expected Inputs

The repository expects the following files and layout under `_EXTERNAL_DATA_ROOT` (current default: `/media/costantino_ai/eik-T9/manuscript-data`). Adjust the root path in `common/constants.py` if your data lives elsewhere.

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

If your file names differ from these patterns, let me know and I’ll align the loaders accordingly.

## Configuration

### Set Data Paths

Edit `common/constants.py` to point to your data location:

```python
# ============================================================================
# *** USER CONFIGURATION: Update this one path for your system ***
# ============================================================================

# Base folder containing BIDS/, rois/, neurosynth/, stimuli/
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
# All other paths (BIDS_ROOT, ROI_* paths, Neurosynth, etc.) are derived from this.
```

### Configure Analysis Parameters

Key parameters are defined in `common/constants.py`:

```python
# Statistical parameters
ALPHA = 0.05                    # Significance threshold
ALPHA_FDR = 0.05               # FDR correction threshold
RANDOM_SEED = 42               # For reproducibility

# Model definitions
MODEL_COLUMNS = ['check', 'strategy', 'visual']  # RSA model RDMs
MODEL_ORDER = ['check', 'strategy', 'visual']    # Display order

# Visualization
ENABLE_PYLUSTRATOR = False     # Set True for interactive figure layout
```

### Verify Configuration

Quick sanity check that imports and configuration load correctly:

```bash
python - << 'PY'
from common import CONFIG
print('CONFIG loaded')
print('BIDS_ROOT =', CONFIG['BIDS_ROOT'])
print('ROIs dir  =', CONFIG['ROI_ROOT'])
PY
```

## Running Analyses

Always run commands from the repository root so that `common/` imports resolve correctly.

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

**Total runtime**: ~2-3 hours (depends on system)

### Option B: Run Individually (Python)

Each analysis directory has detailed instructions in its README. Basic workflow:

```bash
# Always run from the repository root
python chess-behavioral/01_behavioral_rsa.py                 # Main analysis (~2 min)
python chess-behavioral/81_table_behavioral_correlations.py  # Tables (~10 sec)
python chess-behavioral/91_plot_behavioral_panels.py         # Figures (~30 sec)
```

See individual analysis READMEs for details:
- [`chess-behavioral/README.md`](chess-behavioral/README.md) - Behavioral RSA
- [`chess-manifold/README.md`](chess-manifold/README.md) - Participation ratio analysis
- [`chess-mvpa/README.md`](chess-mvpa/README.md) - MVPA (RSA and decoding)
- [`chess-neurosynth/README.md`](chess-neurosynth/README.md) - Meta-analytic correlations
- [`chess-supplementary/*/README.md`](chess-supplementary/) - Supplementary analyses

### Analysis Dependencies

Some analyses depend on outputs from others:

```
Behavioral RSA (01_behavioral_rsa.py)
    └─> Behavioral reliability (supplementary)
    └─> RDM intercorrelation (supplementary)

MVPA group RSA (02_mvpa_group_rsa.py)
    └─> RSA ROI summary (supplementary)

Manifold analysis (01_manifold_analysis.py)
    └─> (standalone, no dependencies)

Neurosynth analyses (01_univariate_neurosynth.py, 02_rsa_neurosynth.py)
    └─> Neurosynth term visualization (supplementary)
```

## Outputs

All analyses save results to timestamped directories within their respective folders:

```
chess-{analysis}/results/<timestamp>_{analysis_name}/
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

### Main Results Files

Key results for manuscript:

**Behavioral**:
- `chess-behavioral/results/latest/correlation_results.pkl`
- `chess-behavioral/results/latest/tables/behavioral_rsa_correlations.tex`
- `chess-behavioral/results/latest/figures/panels/behavioral_rsa_panel.pdf`

**Manifold**:
- `chess-manifold/results/latest/pr_results.pkl`
- `chess-manifold/results/latest/tables/manifold_pr_results.tex`
- `chess-manifold/results/latest/figures/panels/manifold_bars_panel.pdf`

**MVPA**:
- `chess-mvpa/results/latest_mvpa_group_rsa/mvpa_group_stats.pkl`
- `chess-mvpa/results/latest_mvpa_group_decoding/mvpa_group_stats.pkl`
- `chess-mvpa/results/latest/tables/mvpa_rsa_*.tex`
- `chess-mvpa/results/latest/figures/panels/*.pdf`

**Neurosynth**:
- `chess-neurosynth/results/latest_neurosynth_univariate/*_term_corr_*.csv`
- `chess-neurosynth/results/latest_neurosynth_rsa/*_term_corr_*.csv`
- `chess-neurosynth/results/latest/tables/*.tex`
- `chess-neurosynth/results/latest/figures/panels/*.pdf`

## Project Structure

```
chess-expertise-2025/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── run_all_analyses.sh               # Run all analyses sequentially
│
├── common/                            # Shared utilities
│   ├── constants.py                   # Configuration and paths ⚙️
│   ├── bids_utils.py                  # BIDS data loading
│   ├── rsa_utils.py                   # RSA computations
│   ├── stats_utils.py                 # Statistical functions
│   ├── group_stats.py                 # Group-level statistics
│   ├── neuro_utils.py                 # Neuroimaging utilities
│   ├── io_utils.py                    # File I/O helpers
│   ├── logging_utils.py               # Logging setup
│   ├── report_utils.py                # Table/report generation
│   ├── formatters.py                  # Data formatting
│   ├── spm_utils.py                   # SPM integration
│   └── plotting/                      # Plotting utilities
│       ├── bars.py                    # Bar plots
│       ├── heatmaps.py                # Heatmaps and RDMs
│       ├── scatter.py                 # Scatter plots
│       ├── surfaces.py                # Brain surface plots
│       ├── colors.py                  # Color palettes
│       ├── legends.py                 # Legend utilities
│       ├── helpers.py                 # Plotting helpers
│       └── style.py                   # Nature-compliant styling
│
├── chess-behavioral/                  # Behavioral RSA analysis
│   ├── README.md                      # Analysis documentation
│   ├── 01_behavioral_rsa.py           # Main analysis
│   ├── 81_table_*.py                  # Table generation
│   ├── 91_plot_*.py                   # Figure generation
│   ├── modules/                       # Analysis-specific modules
│   └── results/                       # Analysis outputs
│
├── chess-manifold/                    # Participation ratio analysis
│   ├── README.md
│   ├── 01_manifold_analysis.py        # Main analysis
│   ├── 81_table_*.py
│   ├── 91_plot_*.py
│   ├── modules/
│   └── results/
│
├── chess-mvpa/                        # MVPA: RSA and decoding
│   ├── README.md
│   ├── 02_mvpa_group_rsa.py           # Group-level RSA
│   ├── 03_mvpa_group_decoding.py      # Group-level decoding
│   ├── 81_table_*.py
│   ├── 92_plot_*.py, 93_plot_*.py
│   ├── modules/
│   └── results/
│
├── chess-neurosynth/                  # Meta-analytic correlations
│   ├── README.md
│   ├── 01_univariate_neurosynth.py    # GLM t-map correlations
│   ├── 02_rsa_neurosynth.py           # RSA map correlations
│   ├── 81_table_*.py, 82_table_*.py
│   ├── 91_plot_*.py, 92_plot_*.py
│   ├── modules/
│   └── results/
│
├── chess-supplementary/               # Supplementary analyses
│   ├── behavioral-reliability/        # Split-half reliability
│   ├── dataset-viz/                   # Stimulus visualization
│   ├── eyetracking/                   # Eye-tracking decoding
│   ├── mvpa-finer/                    # Finer MVPA resolution
│   ├── neurosynth-terms/              # Term map visualization
│   ├── rdm-intercorrelation/          # RDM intercorrelations
│   ├── rsa-rois/                      # RSA ROI summary
│   └── univariate-rois/               # Univariate ROI summary
│
├── scripts/                           # Utility scripts
│   ├── 00_generate_api_reference.py   # API reference (Markdown)
│   ├── 01_generate_file_io_index.py   # File I/O index (HTML)
│   ├── 02_generate_loader_patterns.py
│   ├── 03_manuscript_tools.py         # LaTeX/Methods tools
│   └── generate_docs.py               # Wrapper to open docs
│
└── data/                              # Data directory (not in repo)
    └── BIDS/                          # BIDS-compliant dataset
        ├── sub-01/, sub-02/, ...      # Subject directories
        ├── participants.tsv           # Participant info
        ├── stimuli.tsv                # Stimulus info
        └── derivatives/               # Preprocessed data
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

## Dependencies

### Python Packages

Core dependencies (see `requirements.txt` for complete list):

- **Scientific computing**: numpy, scipy, pandas
- **Statistics**: statsmodels, pingouin
- **Machine learning**: scikit-learn
- **Neuroimaging**: nibabel, nilearn
- **Visualization**: matplotlib, seaborn, pylustrator (optional)
- **Data formats**: h5py, openpyxl

### MATLAB Packages (Optional)

For subject-level MVPA preprocessing:

- MATLAB R2019b or higher
- SPM12 (Statistical Parametric Mapping)
- CoSMoMVPA (Cosmological Multi-Voxel Pattern Analysis toolbox)

### System Requirements

- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: ~100 GB for full dataset and derivatives
- **OS**: Linux, macOS, or Windows 10/11
- **CPU**: Multi-core recommended for searchlight analyses

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'common'"**

Solution: Always run scripts from the repository root directory:

```bash
# Correct (run from repo root)
cd /path/to/chess-expertise-2025
python chess-behavioral/01_behavioral_rsa.py

# Incorrect (may fail due to imports)
cd chess-behavioral
python 01_behavioral_rsa.py
```

**2. "FileNotFoundError: BIDS directory not found"**

Solution: Verify `BIDS_ROOT` in `common/constants.py` points to your data:

```python
BIDS_ROOT = Path("/correct/path/to/BIDS")
```

Run configuration check:

```bash
python scripts/00_verify_config.py
```

**3. "No participants found" or "No event files found"**

Solution: Ensure BIDS structure is correct:

```bash
data/BIDS/
├── participants.tsv          # Required
├── sub-01/                   # Subject directories
│   └── func/
│       ├── sub-01_task-exp_run-1_bold.nii.gz
│       └── sub-01_task-exp_run-1_events.tsv
```

**4. Import errors or missing dependencies**

Solution: Reinstall dependencies:

```bash
pip install --upgrade -r requirements.txt
```

**5. MATLAB/CoSMoMVPA errors**

Solution: Subject-level MVPA (MATLAB) is optional. Python group-level analyses can run using pre-computed derivatives in `data/BIDS/derivatives/mvpa-rsa/`.

**6. Figure generation fails with Pylustrator errors**

Solution: Disable interactive mode:

```python
# In common/constants.py
ENABLE_PYLUSTRATOR = False
```

### Getting Help

If you encounter issues:

1. Check the analysis-specific README in the relevant directory
2. Search existing [GitHub Issues](https://github.com/your-username/chess-expertise-2025/issues)
3. Open a new issue with:
   - Error message (full traceback)
   - Script being run
   - Python version and OS
   - Minimal reproducible example

## Contributing

We welcome contributions! Please follow these guidelines:

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write comprehensive docstrings (NumPy style)
- Add inline comments for complex logic

### Testing

Before submitting changes:

```bash
# Verify all imports work
python -c "import common; import common.plotting"

# Run configuration check
python scripts/00_verify_config.py

# Test a simple analysis
python chess-behavioral/01_behavioral_rsa.py
```

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Description"`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request with detailed description

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{YourName2025ChessExpertise,
  title={Neural and Behavioral Representations of Chess Expertise},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025},
  volume={XX},
  pages={XXX-XXX},
  doi={10.XXXX/XXXXXX}
}
```

### Dataset Citation

```bibtex
@dataset{YourName2025ChessData,
  title={Chess Expertise fMRI Dataset},
  author={Your Name and Collaborators},
  year={2025},
  publisher={OpenNeuro/OSF},
  doi={10.XXXX/XXXXXX},
  url={https://openneuro.org/datasets/dsXXXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **SPM12**: GNU General Public License v2.0
- **CoSMoMVPA**: MIT License
- **Neurosynth**: BSD 2-Clause License
- **Glasser Parcellation**: Human Connectome Project Open Access Data Use Terms

## Acknowledgments

- Participants who contributed data to this study
- [Human Connectome Project](https://www.humanconnectome.org/) for the Glasser parcellation
- [Neurosynth](https://neurosynth.org/) for meta-analytic term maps
- [fMRIPrep](https://fmriprep.org/) developers for preprocessing pipeline
- [CoSMoMVPA](http://www.cosmomvpa.org/) developers for MVPA tools

## Contact

**Principal Investigator**: [Name] ([email])
**Lead Analyst**: [Name] ([email])
**GitHub Issues**: [https://github.com/your-username/chess-expertise-2025/issues](https://github.com/your-username/chess-expertise-2025/issues)

---

**Repository Status**: ✅ Complete (as of 2025-01-XX)
**Last Updated**: 2025-01-XX
**Version**: 1.0.0
