"""
Central repository for all shared constants, colors, ROI definitions, and paths.

All constants are exported via the CONFIG dictionary, which is the single source
of truth for all configuration values used across analyses.

Usage
-----
>>> from common import CONFIG
>>> print(CONFIG['RANDOM_SEED'])
42
>>> print(CONFIG['BIDS_ROOT'])
PosixPath('/path/to/data/BIDS')

Note: All dataset paths are set here in constants and should be
configured explicitly. Environment overrides are not used.
"""

from pathlib import Path

# ============================================================================
# Repository Paths (computed first for use in CONFIG)
# ============================================================================
_REPO_ROOT = Path(__file__).parent.parent  # Root of this git repository
_DATA_DIR = _REPO_ROOT / "data"  # Local data assets (repo-internal, not primary dataset)

# ============================================================================
# External Manuscript Data Root
# ============================================================================
# IMPORTANT: Users must configure this path to their local manuscript-data location
# See README for setup instructions
_EXTERNAL_DATA_ROOT = Path("/media/costantino_ai/eik-T9/manuscript-data")

# ============================================================================
# Intermediate Path Construction (private - build CONFIG paths from these)
# ============================================================================

# ROI base directories
_ROI_ROOT = _EXTERNAL_DATA_ROOT / "rois"
_ROI_GLASSER_22_DIR = _ROI_ROOT / "glasser22"
_ROI_GLASSER_180_DIR = _ROI_ROOT / "glasser180"
_ROI_GLASSER_180_SURFACE_DIR = _ROI_ROOT / "glasser180-surface"

# BIDS base directories
_BIDS_ROOT = _EXTERNAL_DATA_ROOT / "BIDS"
_BIDS_DERIVATIVES = _BIDS_ROOT / "derivatives"

# Neurosynth base directory
_NEUROSYNTH_ROOT = _EXTERNAL_DATA_ROOT / "neurosynth"

# ============================================================================
# CONFIG Dictionary - All Constants in One Place
# ============================================================================
# This dictionary contains ALL constants, making it easy to:
# 1. Pass entire config to setup_analysis()
# 2. Log all configuration values
# 3. Access config programmatically
# 4. Serialize config for reproducibility

CONFIG = {
    # ========================================================================
    # Repository Structure
    # ========================================================================
    'REPO_ROOT': _REPO_ROOT,              # Root of this git repository
    'DATA_DIR': _DATA_DIR,                # Local data assets (repo-internal)
    # Note: Results directories are analysis-specific and created via
    # logging_utils.setup_analysis() under each package's results/ folder.

    # ========================================================================
    # External Manuscript Data (Inputs)
    # ========================================================================
    # All analysis inputs live in this external directory
    'EXTERNAL_DATA_ROOT': _EXTERNAL_DATA_ROOT,  # Base directory for all inputs

    # --- ROI Definitions ---
    'ROI_ROOT': _ROI_ROOT,                                      # Base ROI directory
    'ROI_GLASSER_22': _ROI_GLASSER_22_DIR,                      # 22 bilateral cortical ROIs
    'ROI_GLASSER_180': _ROI_GLASSER_180_DIR,                    # 180 bilateral cortical ROIs
    'ROI_GLASSER_22_ATLAS': _ROI_GLASSER_22_DIR / "tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-22_bilateral_resampled.nii",    # Volume atlas (NIfTI)
    'ROI_GLASSER_180_ATLAS': _ROI_GLASSER_180_DIR / "tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-180_bilateral_resampled.nii",  # Volume atlas (NIfTI)
    'ROI_GLASSER_180_SURFACE': _ROI_GLASSER_180_SURFACE_DIR,    # Surface parcellation directory
    'ROI_GLASSER_180_ANNOT_L': _ROI_GLASSER_180_SURFACE_DIR / "lh.HCPMMP1.annot",  # Left hemisphere annotation
    'ROI_GLASSER_180_ANNOT_R': _ROI_GLASSER_180_SURFACE_DIR / "rh.HCPMMP1.annot",  # Right hemisphere annotation

    # --- BIDS Structure ---
    'BIDS_ROOT': _BIDS_ROOT,                                    # Main BIDS dataset root
    'BIDS_DERIVATIVES': _BIDS_DERIVATIVES,                      # Preprocessed/derived data
    'BIDS_PARTICIPANTS': _BIDS_ROOT / "participants.tsv",       # Subject metadata TSV
    'STIMULI_FILE': _EXTERNAL_DATA_ROOT / "stimuli" / "stimuli.tsv",  # Chess board stimulus metadata

    # --- Preprocessing Outputs ---
    'BIDS_FMRIPREP': _BIDS_DERIVATIVES / "fmriprep",            # fMRIPrep outputs
    # Canonical SPM GLM derivatives root (single source of truth)
    'SPM_GLM_DIR': _BIDS_DERIVATIVES / "SPM",
    'SPM_GLM_UNSMOOTHED': _BIDS_DERIVATIVES / "SPM" / "GLM-unsmoothed",  # Unsmoothed GLM results
    'SPM_GLM_SMOOTH4': _BIDS_DERIVATIVES / "SPM" / "GLM-smooth4",        # 4mm smoothed GLM results

    # --- Analysis Derivatives ---
    'BIDS_MVPA_RSA': _BIDS_DERIVATIVES / "mvpa-rsa",           # MVPA RSA results (ROI-level)
    'BIDS_MVPA_DECODING': _BIDS_DERIVATIVES / "mvpa-decoding",  # MVPA decoding results (ROI-level)
    'BIDS_RSA_SEARCHLIGHT': _BIDS_DERIVATIVES / "rsa_searchlight",  # RSA searchlight results
    'BIDS_BEHAVIORAL': _BIDS_DERIVATIVES / "chess-behavioural",  # Behavioral task data
    'BIDS_EYETRACK': _BIDS_DERIVATIVES / "eye-tracking",        # Eye-tracking derivatives

    # --- External Resources ---
    'NEUROSYNTH_TERMS_DIR': _NEUROSYNTH_ROOT / "terms",         # Neurosynth term maps

    # ========================================================================
    # Analysis Parameters
    # ========================================================================
    'RANDOM_SEED': 42,                                          # Reproducibility seed
    'CHANCE_LEVEL_RSA': 0.0,                                    # RSA null hypothesis (r=0)

    # Note: Dataset info (n_subjects, n_experts, n_novices) is derived dynamically
    # from BIDS participants.tsv - do not hardcode counts in CONFIG

    # ========================================================================
    # Model RDMs
    # ========================================================================
    'MODEL_RDM_NAMES': ['Checkmate', 'Strategy', 'Visual_Similarity'],  # Model RDM names
    'MODEL_COLUMNS': ['check', 'visual', 'strategy'],          # Column names in stimuli.tsv
    'MODEL_ORDER': ["visual", "strategy", "check"],            # Display order in plots
    'MODEL_LABELS': {                                           # Pretty labels for plots/tables
        "visual": "Visual Similarity",
        "strategy": "Strategy",
        "check": "Checkmate",
    },
    'MODEL_LABELS_PRETTY': ["Visual Similarity", "Strategy", "Checkmate"],  # Ordered list

    # ========================================================================
    # Statistical Parameters
    # ========================================================================
    'ALPHA': 0.05,                                              # Significance threshold
    'ALPHA_FDR': 0.05,                                          # FDR-corrected threshold
    'SEARCHLIGHT_RADIUS_VOXELS': 3,                             # Searchlight sphere (voxels)
    'SEARCHLIGHT_RADIUS_MM': 6,                                 # Searchlight sphere (mm)
    'CV_FOLDS': 5,                                              # Cross-validation folds

    # ========================================================================
    # Display/Plotting Configuration
    # ========================================================================
    'ENABLE_PYLUSTRATOR': False,                                # Enable pylustrator for interactive plot editing (dev: True, production: False)
    'NEUROSYNTH_TERM_ORDER': [                                  # Term ordering for plots
        'working memory',
        'navigation',
        'memory retrieval',
        'language network',
        'object recognition',
        'face recognition',
        'early visual',
    ],

    # ========================================================================
    # MVPA Configuration
    # ========================================================================
    'MVPA_PATTERN_CORTICES': '*_glasser_cortices_bilateral',   # Subject-level MVPA file pattern
    'MVPA_PATTERN_CM_ONLY': '*_glasser_regions_bilateral_fine',  # Fine-grained (checkmate-only stimuli)

    'MVPA_TARGETS': {                                           # Target variable registry
        # Main targets
        'checkmate': {'display': 'Checkmate', 'type': 'categorical', 'set': 'main'},
        'visual_similarity': {'display': 'Visual Similarity', 'type': 'categorical', 'set': 'main'},
        'strategy': {'display': 'Strategy', 'type': 'categorical', 'set': 'main'},
        # Fine targets (checkmate-only and related)
        'strategy_half': {'display': 'Strategy (Half)', 'type': 'categorical', 'set': 'fine'},
        'check_n_half': {'display': 'Check N (Half)', 'type': 'categorical', 'set': 'fine'},
        'total_pieces': {'display': 'Total Pieces', 'type': 'ordinal', 'set': 'fine'},
        'total_pieces_half': {'display': 'Total Pieces (Half)', 'type': 'ordinal', 'set': 'fine'},
        'legal_moves': {'display': 'Legal Moves', 'type': 'ordinal', 'set': 'fine'},
        'legal_moves_half': {'display': 'Legal Moves (Half)', 'type': 'ordinal', 'set': 'fine'},
        'motif_half': {'display': 'Motif (Half)', 'type': 'categorical', 'set': 'fine'},
        'side_half': {'display': 'Side (Half)', 'type': 'categorical', 'set': 'fine'},
        'stimuli': {'display': 'Stimuli', 'type': 'categorical', 'set': 'fine'},
        'stimuli_half': {'display': 'Stimuli (Half)', 'type': 'categorical', 'set': 'fine'},
    },

    # Default SVM chance mapping for common targets
    'MVPA_SVM_CHANCE_DEFAULTS': {
        'checkmate': 1.0/2,
        'strategy': 1.0/10,
        'visual_similarity': 1.0/20,
    },
}

__all__ = [
    'CONFIG',
]
