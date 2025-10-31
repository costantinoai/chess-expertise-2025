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
# Repository Paths (Must be computed first for use in CONFIG)
# ============================================================================
_REPO_ROOT = Path(__file__).parent.parent  # Assumes constants.py is in common/
# Repository results root (outputs stay repo-local)
_DATA_DIR = _REPO_ROOT / "data"  # reserved for local assets; not the primary dataset

# External manuscript data root (ALL inputs live here)
_EXTERNAL_DATA_ROOT = Path("/media/costantino_ai/eik-T9/manuscript-data")

# ============================================================================
# CONFIG Dictionary - All Constants in One Place
# ============================================================================
# This dictionary contains ALL constants, making it easy to:
# 1. Pass entire config to setup_analysis()
# 2. Log all configuration values
# 3. Access config programmatically
# 4. Serialize config for reproducibility

CONFIG = {
    # Repository paths
    'REPO_ROOT': _REPO_ROOT,
    'DATA_DIR': _DATA_DIR,
    # Results directories are analysis-specific and created via
    # logging_utils.setup_analysis() under each package's results/ folder.

    # ROI paths (external)
    'ROI_ROOT': _EXTERNAL_DATA_ROOT / "rois",
    'ROI_GLASSER_22': _EXTERNAL_DATA_ROOT / "rois" / "glasser22",
    'ROI_GLASSER_180': _EXTERNAL_DATA_ROOT / "rois" / "glasser180",
    'ROI_GLASSER_22_ATLAS': _EXTERNAL_DATA_ROOT / "rois" / "glasser22" / "tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-22_bilateral_resampled.nii",
    'ROI_GLASSER_180_ATLAS': _EXTERNAL_DATA_ROOT / "rois" / "glasser180" / "tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-180_bilateral_resampled.nii",
    # Surface (fsaverage) annotation files for Glasser parcellation
    'ROI_GLASSER_180_SURFACE': _EXTERNAL_DATA_ROOT / "rois" / "glasser180-surface",
    'ROI_GLASSER_180_ANNOT_L': _EXTERNAL_DATA_ROOT / "rois" / "glasser180-surface" / "lh.HCPMMP1.annot",
    'ROI_GLASSER_180_ANNOT_R': _EXTERNAL_DATA_ROOT / "rois" / "glasser180-surface" / "rh.HCPMMP1.annot",

    # BIDS paths (external root)
    'BIDS_ROOT':  _EXTERNAL_DATA_ROOT / "BIDS",
    'BIDS_DERIVATIVES':  _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives",
    'BIDS_PARTICIPANTS':  _EXTERNAL_DATA_ROOT / "BIDS" / "participants.tsv",
    'STIMULI_FILE': _EXTERNAL_DATA_ROOT / "stimuli" / "stimuli.tsv",  # External stimulus metadata

    # BIDS derivatives (external folder structure)
    'BIDS_FMRIPREP':  _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives" / "fmriprep",
    'BIDS_GLM_UNSMOOTHED':  _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives" / "SPM" / "GLM-unsmoothed",
    'BIDS_GLM_SMOOTHED_4MM':  _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives" / "SPM" / "GLM-smooth4",
    'BIDS_GLM_SMOOTHED_6MM':  _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives" / "SPM" / "GLM-smooth6",
    'BIDS_MVPA':  _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives" / "mvpa",
    'BIDS_RSA_SEARCHLIGHT':  _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives" / "rsa_searchlight",
    'BIDS_BEHAVIORAL':  _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives" / "chess-behavioural",
    # Eye-tracking derivatives (canonical location)
    'BIDS_EYETRACK':      _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives" / "eye-tracking",

    # Neurosynth / SPM group maps
    'NEUROSYNTH_TERMS_DIR': _EXTERNAL_DATA_ROOT / "neurosynth" / "terms",
    'BIDS_SPM_GROUP':  _EXTERNAL_DATA_ROOT / "BIDS" / "derivatives" / "SPM",

    # Analysis parameters
    'RANDOM_SEED': 42,
    'CHANCE_LEVEL_RSA': 0.0,  # For RSA correlation tests, chance level is 0 (no correlation)

    # Dataset info (derived dynamically from BIDS ground truth; do not hardcode counts)

    # Model info
    'MODEL_RDM_NAMES': ['Checkmate', 'Strategy', 'Visual_Similarity'],
    'MODEL_COLUMNS': ['check', 'visual', 'strategy'],
    # Display order and labels for models
    'MODEL_ORDER': ["visual", "strategy", "check"],
    'MODEL_LABELS': {
        "visual": "Visual Similarity",
        "strategy": "Strategy",
        "check": "Checkmate",
    },
    'MODEL_LABELS_PRETTY': ["Visual Similarity", "Strategy", "Checkmate"],

    # Statistical parameters
    'ALPHA': 0.05,
    'ALPHA_FDR': 0.05,  # FDR-corrected threshold
    'SEARCHLIGHT_RADIUS_VOXELS': 3,  # voxels
    'SEARCHLIGHT_RADIUS_MM': 6,  # mm
    'CV_FOLDS': 5,  # 5-fold cross-validation

    # Neurosynth plotting order
    'NEUROSYNTH_TERM_ORDER': [
        'working memory',
        'navigation',
        'memory retrieval',
        'language network',
        'object recognition',
        'face recognition',
        'early visual',
    ],

    # MVPA directory pattern(s)
    'MVPA_PATTERN_CORTICES': '*_glasser_cortices_bilateral',
    # Fine dimensions (checkmate-only stimuli) — still Glasser-22 atlas
    'MVPA_PATTERN_CM_ONLY': '*_glasser_regions_bilateral_fine',

    # MVPA targets registry (canonical keys → metadata)
    'MVPA_TARGETS': {
        # Main targets
        'checkmate': {'display': 'Checkmate', 'type': 'categorical', 'set': 'main'},
        'visualStimuli': {'display': 'Visual Stimuli', 'type': 'categorical', 'set': 'main'},
        'categories': {'display': 'Categories', 'type': 'categorical', 'set': 'main'},
        # Fine targets (checkmate-only and related)
        'categories_half': {'display': 'Categories (Half)', 'type': 'categorical', 'set': 'fine'},
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
        'categories': 1.0/10,
        'visualStimuli': 1.0/20,
    },
}

__all__ = [
    'CONFIG',
]
