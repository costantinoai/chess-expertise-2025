"""
BIDS utilities for path handling, participant loading, and data validation.

This module provides functions for working with BIDS-formatted datasets,
including loading participant information, finding files, and validating paths.

Dependencies
------------
- pandas: For loading participant TSV files
- pathlib: For path operations
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import warnings

from .constants import CONFIG


def load_participants_tsv(tsv_path=None):
    """
    Load participants.tsv file from BIDS dataset.

    Parameters
    ----------
    tsv_path : str or Path, optional
        Path to participants.tsv file. If None, uses BIDS_PARTICIPANTS from constants.

    Returns
    -------
    pd.DataFrame
        DataFrame with participant information, including:
        - participant_id: Subject ID (e.g., 'sub-01')
        - group: 'expert' or 'novice'
        - Additional columns as defined in dataset

    Raises
    ------
    FileNotFoundError
        If participants.tsv file not found

    Example
    -------
    >>> df = load_participants_tsv()
    >>> print(df.head())
    >>> expert_ids = df[df['group'] == 'expert']['participant_id'].tolist()
    """
    if tsv_path is None:
        tsv_path = CONFIG['BIDS_PARTICIPANTS']

    tsv_path = Path(tsv_path)

    if not tsv_path.exists():
        raise FileNotFoundError(f"Participants file not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep='\t')

    return df


def get_subject_list(group='all', tsv_path=None):
    """
    Get list of subject IDs from participants.tsv.

    Parameters
    ----------
    group : str, default='all'
        Which group to return: 'expert', 'novice', or 'all'
    tsv_path : str or Path, optional
        Path to participants.tsv file. If None, uses BIDS_PARTICIPANTS.

    Returns
    -------
    list of str
        Subject IDs (e.g., ['sub-01', 'sub-02', ...])

    Raises
    ------
    ValueError
        If group is not 'all', 'expert', or 'novice'

    Example
    -------
    >>> experts = get_subject_list('expert')
    >>> novices = get_subject_list('novice')
    >>> all_subjects = get_subject_list('all')
    """
    df = load_participants_tsv(tsv_path)

    if group == 'all':
        return df['participant_id'].tolist()
    elif group == 'expert':
        return df[df['group'] == 'expert']['participant_id'].tolist()
    elif group == 'novice':
        return df[df['group'] == 'novice']['participant_id'].tolist()
    else:
        raise ValueError(f"Unknown group: {group}. Use 'all', 'expert', or 'novice'")


def get_subject_info(subject_id, tsv_path=None):
    """
    Get information for a specific subject.

    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'sub-01')
    tsv_path : str or Path, optional
        Path to participants.tsv file

    Returns
    -------
    pd.Series or None
        Subject information, or None if not found

    Example
    -------
    >>> info = get_subject_info('sub-01')
    >>> print(f"Subject {info['participant_id']} is in group {info['group']}")
    """
    df = load_participants_tsv(tsv_path)

    subject_row = df[df['participant_id'] == subject_id]

    if len(subject_row) == 0:
        warnings.warn(f"Subject {subject_id} not found in participants.tsv")
        return None

    return subject_row.iloc[0]


def find_beta_images(subject_id, bids_deriv_path=None, pattern='beta_*.nii'):
    """
    Find beta images for a subject in GLM derivatives.

    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'sub-01')
    bids_deriv_path : str or Path, optional
        Path to BIDS derivatives folder. If None, uses BIDS_GLM_UNSMOOTHED.
    pattern : str, default='beta_*.nii'
        Glob pattern for beta images

    Returns
    -------
    list of Path
        Sorted list of beta image paths

    Raises
    ------
    FileNotFoundError
        If subject directory not found
    ValueError
        If no beta images found

    Example
    -------
    >>> beta_paths = find_beta_images('sub-01')
    >>> print(f"Found {len(beta_paths)} beta images")
    """
    if bids_deriv_path is None:
        bids_deriv_path = CONFIG['BIDS_GLM_UNSMOOTHED']

    subject_dir = Path(bids_deriv_path) / subject_id

    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    # Find beta files
    beta_files = sorted(subject_dir.glob(pattern))

    if len(beta_files) == 0:
        raise ValueError(f"No beta images matching '{pattern}' found in {subject_dir}")

    return beta_files


def find_contrast_images(subject_id, bids_deriv_path=None, pattern='con_*.nii'):
    """
    Find contrast images for a subject in GLM derivatives.

    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'sub-01')
    bids_deriv_path : str or Path, optional
        Path to BIDS derivatives folder. If None, uses BIDS_GLM_UNSMOOTHED.
    pattern : str, default='con_*.nii'
        Glob pattern for contrast images

    Returns
    -------
    list of Path
        Sorted list of contrast image paths

    Example
    -------
    >>> con_paths = find_contrast_images('sub-01')
    """
    if bids_deriv_path is None:
        bids_deriv_path = CONFIG['BIDS_GLM_UNSMOOTHED']

    subject_dir = Path(bids_deriv_path) / subject_id

    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    # Find contrast files
    contrast_files = sorted(subject_dir.glob(pattern))

    return contrast_files


def validate_subject_data(subject_id, check_glm=True, check_mvpa=False, check_rsa=False):
    """
    Validate that required data exists for a subject.

    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'sub-01')
    check_glm : bool, default=True
        Check for GLM beta images
    check_mvpa : bool, default=False
        Check for MVPA results
    check_rsa : bool, default=False
        Check for RSA searchlight results

    Returns
    -------
    dict
        Dictionary with validation results:
        - 'valid': bool, True if all checks pass
        - 'missing': list of missing data types
        - 'messages': list of warning messages

    Example
    -------
    >>> result = validate_subject_data('sub-01', check_glm=True)
    >>> if not result['valid']:
    ...     print(f"Missing data: {result['missing']}")
    """
    result = {
        'valid': True,
        'missing': [],
        'messages': []
    }

    # Check if subject exists in participants.tsv
    info = get_subject_info(subject_id)
    if info is None:
        result['valid'] = False
        result['missing'].append('participants_tsv')
        result['messages'].append(f"Subject {subject_id} not in participants.tsv")

    # Check GLM data
    if check_glm:
        try:
            beta_files = find_beta_images(subject_id)
            if len(beta_files) == 0:
                result['valid'] = False
                result['missing'].append('glm_betas')
                result['messages'].append(f"No GLM beta images for {subject_id}")
        except (FileNotFoundError, ValueError) as e:
            result['valid'] = False
            result['missing'].append('glm_betas')
            result['messages'].append(str(e))

    # Check MVPA data
    if check_mvpa:
        mvpa_dir = Path(CONFIG['BIDS_MVPA']) / subject_id
        if not mvpa_dir.exists():
            result['valid'] = False
            result['missing'].append('mvpa_results')
            result['messages'].append(f"No MVPA results for {subject_id}")

    # Check RSA searchlight data
    if check_rsa:
        rsa_dir = Path(CONFIG['BIDS_RSA_SEARCHLIGHT']) / subject_id
        if not rsa_dir.exists():
            result['valid'] = False
            result['missing'].append('rsa_searchlight')
            result['messages'].append(f"No RSA searchlight results for {subject_id}")

    return result


def validate_all_subjects(group='all', **kwargs):
    """
    Validate data for all subjects in a group.

    Parameters
    ----------
    group : str, default='all'
        Which group to validate: 'expert', 'novice', or 'all'
    **kwargs
        Additional arguments passed to validate_subject_data

    Returns
    -------
    dict
        Dictionary mapping subject IDs to validation results

    Example
    -------
    >>> results = validate_all_subjects('expert', check_glm=True, check_mvpa=True)
    >>> valid_subjects = [sid for sid, res in results.items() if res['valid']]
    >>> print(f"{len(valid_subjects)} subjects have complete data")
    """
    subject_ids = get_subject_list(group)

    results = {}
    for subject_id in subject_ids:
        results[subject_id] = validate_subject_data(subject_id, **kwargs)

    return results


def get_group_summary(tsv_path=None):
    """
    Get summary statistics for participant groups.

    Parameters
    ----------
    tsv_path : str or Path, optional
        Path to participants.tsv file

    Returns
    -------
    dict
        Summary statistics:
        - 'n_total': Total number of subjects
        - 'n_expert': Number of experts
        - 'n_novice': Number of novices
        - 'expert_ids': List of expert IDs
        - 'novice_ids': List of novice IDs

    Example
    -------
    >>> summary = get_group_summary()
    >>> print(f"Total: {summary['n_total']} ({summary['n_expert']} experts, {summary['n_novice']} novices)")
    """
    df = load_participants_tsv(tsv_path)

    expert_ids = df[df['group'] == 'expert']['participant_id'].tolist()
    novice_ids = df[df['group'] == 'novice']['participant_id'].tolist()

    return {
        'n_total': len(df),
        'n_expert': len(expert_ids),
        'n_novice': len(novice_ids),
        'expert_ids': expert_ids,
        'novice_ids': novice_ids,
    }


def load_roi_metadata(roi_dir: Path) -> pd.DataFrame:
    """
    Load ROI metadata from ROI directory, supporting both legacy and new formats.

    Supported files (first found is used):
    - roi_labels.tsv with columns: roi_id, roi_name, family, color
    - region_info.tsv with columns: ROI_idx, roi_name, pretty_name, [family/group], color

    Returns a DataFrame standardized to columns: roi_id, roi_name, [pretty_name], family, color.
    The pretty_name column is included if available in the source file.

    Parameters
    ----------
    roi_dir : Path
        Path to ROI directory containing metadata TSV

    Returns
    -------
    pd.DataFrame
        Standardized metadata with columns: roi_id, roi_name, [pretty_name], family, color

    Raises
    ------
    FileNotFoundError
        If neither roi_labels.tsv nor region_info.tsv is found
    """
    roi_dir = Path(roi_dir)
    roi_labels_path = roi_dir / "roi_labels.tsv"
    region_info_path = roi_dir / "region_info.tsv"

    if roi_labels_path.exists():
        df = pd.read_csv(roi_labels_path, sep='\t')
        # Validate required columns
        required_cols = ['roi_id', 'roi_name', 'family', 'color']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {roi_labels_path}: {missing}\n"
                f"Expected: {required_cols}, Found: {list(df.columns)}"
            )
        # Include pretty_name if available
        cols_to_return = required_cols.copy()
        if 'pretty_name' in df.columns:
            cols_to_return.insert(2, 'pretty_name')  # Insert after roi_name
        return df[cols_to_return]

    if region_info_path.exists():
        df_raw = pd.read_csv(region_info_path, sep='\t')
        # Build case-insensitive column resolver
        name_map = {c.lower(): c for c in df_raw.columns}
        def has(col):
            return col in name_map
        def get(col):
            return name_map[col]

        # Map to standardized columns with case-insensitive matching
        df = pd.DataFrame()
        if has('index'):
            df['roi_id'] = df_raw[get('index')]
        elif has('roi_id'):
            df['roi_id'] = df_raw[get('roi_id')]
        elif has('roi_idx'):
            df['roi_id'] = df_raw[get('roi_idx')]
        else:
            raise ValueError(f"Missing ROI id column in {region_info_path} (expected 'index', 'roi_id' or 'ROI_idx')")

        if has('name'):
            df['roi_name'] = df_raw[get('name')]
        elif has('roi_name'):
            df['roi_name'] = df_raw[get('roi_name')]
        else:
            raise ValueError(f"Missing ROI name column in {region_info_path} (expected 'name' or 'roi_name')")

        if has('color'):
            df['color'] = df_raw[get('color')]
        else:
            # Optional; fill with default color if absent
            df['color'] = '#4C78A8'

        if has('family'):
            df['family'] = df_raw[get('family')]
        elif has('group'):
            df['family'] = df_raw[get('group')]
        else:
            df['family'] = 'ROI'

        # Include pretty_name if available
        if has('pretty_name'):
            df['pretty_name'] = df_raw[get('pretty_name')]
            return df[['roi_id', 'roi_name', 'pretty_name', 'family', 'color']]
        else:
            return df[['roi_id', 'roi_name', 'family', 'color']]

    raise FileNotFoundError(
        f"No ROI metadata found under {roi_dir}. Expected roi_labels.tsv or region_info.tsv"
    )


def get_roi_names_dict(roi_metadata: pd.DataFrame) -> Dict[int, str]:
    """
    Extract ROI ID to name mapping from metadata DataFrame.

    Parameters
    ----------
    roi_metadata : pd.DataFrame
        ROI metadata from load_roi_metadata()

    Returns
    -------
    dict
        Mapping from roi_id (int) to roi_name (str)

    Example
    -------
    >>> roi_metadata = load_roi_metadata(ROI_ROOT / "glasser_22_coarse")
    >>> roi_names = get_roi_names_dict(roi_metadata)
    >>> print(roi_names[1])  # "Primary Visual"
    """
    return dict(zip(roi_metadata['roi_id'], roi_metadata['roi_name']))


def get_roi_colors_dict(roi_metadata: pd.DataFrame) -> Dict[int, str]:
    """
    Extract ROI ID to color mapping from metadata DataFrame.

    Parameters
    ----------
    roi_metadata : pd.DataFrame
        ROI metadata from load_roi_metadata()

    Returns
    -------
    dict
        Mapping from roi_id (int) to color hex code (str)

    Example
    -------
    >>> roi_metadata = load_roi_metadata(ROI_ROOT / "glasser_22_coarse")
    >>> roi_colors = get_roi_colors_dict(roi_metadata)
    >>> print(roi_colors[1])  # "#a6cee3"
    """
    return dict(zip(roi_metadata['roi_id'], roi_metadata['color']))


def get_roi_family_colors(roi_metadata: pd.DataFrame) -> Dict[str, str]:
    """
    Extract unique family to color mapping from metadata DataFrame.

    Parameters
    ----------
    roi_metadata : pd.DataFrame
        ROI metadata from load_roi_metadata()

    Returns
    -------
    dict
        Mapping from family name (str) to color hex code (str)

    Example
    -------
    >>> roi_metadata = load_roi_metadata(ROI_ROOT / "glasser_22_coarse")
    >>> family_colors = get_roi_family_colors(roi_metadata)
    >>> print(family_colors["Early Visual"])  # "#a6cee3"
    """
    # Get unique family-color pairs
    family_color_pairs = roi_metadata[['family', 'color']].drop_duplicates()
    return dict(zip(family_color_pairs['family'], family_color_pairs['color']))


def get_participants_with_expertise(participants_file=None, bids_root=None):
    """
    Load participant list with expertise labels for behavioral/neural analyses.
    
    This is a convenience wrapper that loads participants.tsv and returns
    the data in a format ready for analysis loops.
    
    Parameters
    ----------
    participants_file : str or Path, optional
        Path to participants.tsv. If None, uses CONFIG['BIDS_PARTICIPANTS']
    bids_root : Path, optional
        BIDS root directory (not used, kept for backward compatibility)
    
    Returns
    -------
    participants_list : list of tuple
        List of (subject_id, is_expert) tuples
        Example: [('sub-01', True), ('sub-02', False), ...]
    counts : tuple of int
        (n_experts, n_novices)
    
    Example
    -------
    >>> participants, (n_exp, n_nov) = get_participants_with_expertise()
    >>> print(f"Loaded {n_exp} experts and {n_nov} novices")
    >>> for subject_id, is_expert in participants:
    ...     print(f"Processing {subject_id} (expert={is_expert})")
    """
    if participants_file is None:
        participants_file = CONFIG['BIDS_PARTICIPANTS']
    
    df = load_participants_tsv(participants_file)
    
    # Create list of (subject_id, is_expert) tuples
    participants_list = [
        (row['participant_id'], row['group'] == 'expert')
        for _, row in df.iterrows()
    ]
    
    # Count experts and novices
    n_experts = (df['group'] == 'expert').sum()
    n_novices = (df['group'] == 'novice').sum()
    
    return participants_list, (n_experts, n_novices)


def load_stimulus_metadata(stimuli_file=None, return_all: bool = False):
    """
    Load stimulus metadata for model RDM construction.
    
    This function loads the theoretical model dimensions (checkmate, strategy,
    visual similarity) for each chess board stimulus from the BIDS ground truth.
    
    Parameters
    ----------
    stimuli_file : str or Path, optional
        Path to stimuli.tsv file. If None, uses CONFIG['STIMULI_FILE']
    return_all : bool, default=False
        If True, return all available columns after filtering and renaming
        (no restriction to ['stim_id', 'check', 'visual', 'strategy']).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['stim_id', 'check', 'visual', 'strategy']
        - stim_id: Stimulus ID (1-40)
        - check: Checkmate status ('checkmate' or 'non_checkmate')
        - visual: Visual similarity category (1-20)
        - strategy: Strategy type category (1-5)
    
    Notes
    -----
    - Loads from data/stimuli/stimuli.tsv (BIDS ground truth)
    - Determines number of boards dynamically from the file (no hardcoded counts)
    - Renames 'check_status' to 'check' for compatibility
    - Returns only relevant columns for model RDM construction
    
    Example
    -------
    >>> stim_df = load_stimulus_metadata()
    >>> print(f"Loaded metadata for {len(stim_df)} stimuli")
    >>> print(stim_df.columns)
    Index(['stim_id', 'check', 'visual', 'strategy'], dtype='object')
    """
    if stimuli_file is None:
        stimuli_file = CONFIG['STIMULI_FILE']
    
    stimuli_file = Path(stimuli_file)
    
    if not stimuli_file.exists():
        raise FileNotFoundError(f"Stimulus metadata file not found: {stimuli_file}")
    
    # Load from TSV (BIDS ground truth)
    df = pd.read_csv(stimuli_file, sep='\t')
    
    # Filter to valid stim_id rows (positive integers) using dynamic range
    # Determine maximum stim_id present to avoid hardcoded board counts
    if 'stim_id' not in df.columns:
        raise ValueError("Stimulus metadata must contain a 'stim_id' column")

    # Keep only finite, positive stim_id values
    df_filtered = df[pd.to_numeric(df['stim_id'], errors='coerce').notna()].copy()
    df_filtered['stim_id'] = df_filtered['stim_id'].astype(int)
    df_filtered = df_filtered[df_filtered['stim_id'] >= 1].reset_index(drop=True)
    
    # Rename check_status to check for compatibility
    if 'check_status' in df_filtered.columns:
        df_filtered = df_filtered.rename(columns={'check_status': 'check'})
    
    if return_all:
        return df_filtered

    # Return only relevant columns
    relevant_cols = ["stim_id", "check", "visual", "strategy"]
    available_cols = [col for col in relevant_cols if col in df_filtered.columns]
    return df_filtered[available_cols]


def validate_bids_paths():
    """
    Validate that all required BIDS paths exist.

    This function checks that the minimum required BIDS directories and files
    exist for running analyses.

    Returns
    -------
    bool
        True if all paths exist

    Raises
    ------
    FileNotFoundError
        If any required path does not exist

    Notes
    -----
    - Only checks critical paths (BIDS root, derivatives, participants.tsv)
    - Does not check analysis-specific paths (MVPA, RSA, etc.)

    Example
    -------
    >>> from common.bids_utils import validate_bids_paths
    >>> try:
    ...     validate_bids_paths()
    ...     print("All BIDS paths valid")
    ... except FileNotFoundError as e:
    ...     print(f"Missing paths: {e}")
    """
    required_paths = [
        CONFIG['BIDS_ROOT'],
        CONFIG['BIDS_DERIVATIVES'],
        CONFIG['BIDS_PARTICIPANTS'],
        CONFIG['BIDS_GLM_UNSMOOTHED'],
    ]

    missing_paths = [p for p in required_paths if not p.exists()]

    if missing_paths:
        raise FileNotFoundError(
            f"Missing required BIDS paths:\n" +
            "\n".join(f"  - {p}" for p in missing_paths)
        )

    return True


__all__ = [
    'load_participants_tsv',
    'get_subject_list',
    'get_subject_info',
    'find_beta_images',
    'find_contrast_images',
    'validate_subject_data',
    'validate_all_subjects',
    'get_group_summary',
    'load_roi_metadata',
    'get_participants_with_expertise',
    'load_stimulus_metadata',
    'validate_bids_paths',
    'merge_group_labels',
    'to_sub_id',
    'from_sub_id',
    'derive_target_chance_from_stimuli',
]


def merge_group_labels(df: pd.DataFrame, participants_df: pd.DataFrame, subject_col: str = 'subject_id') -> pd.DataFrame:
    """
    Attach 'group' labels to a DataFrame using participants.tsv.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a subject column (default 'subject_id') with 'sub-XX' strings.
    participants_df : pd.DataFrame
        Output of load_participants_tsv(); must include 'participant_id' and 'group'.
    subject_col : str
        Name of the subject column in df.

    Returns
    -------
    pd.DataFrame
        df with 'participant_id' and 'group' columns merged.
    """
    return df.merge(
        participants_df[['participant_id', 'group']],
        left_on=subject_col,
        right_on='participant_id',
        how='left'
    )


def to_sub_id(x: str | int) -> str:
    """
    Convert a numeric subject code or string 'XX' to BIDS-style 'sub-XX' (zero-padded to 2 digits).
    """
    s = str(x).strip()
    if s.startswith('sub-'):
        return s
    # Keep original width if >=2, else pad to 2
    if s.isdigit() and len(s) < 2:
        s = s.zfill(2)
    return f"sub-{s}"


def from_sub_id(sub: str) -> str:
    """
    Extract numeric portion 'XX' from BIDS-style 'sub-XX'. If already 'XX', return as-is.
    """
    sub = str(sub)
    if sub.startswith('sub-'):
        return sub.split('sub-')[-1]
    return sub


def derive_target_chance_from_stimuli(
    targets: list[str],
    stimuli_df: pd.DataFrame | None = None,
) -> dict:
    """
    Derive per-target chance levels from stimuli metadata.

    For each target in `targets`, computes chance as 1 / n_unique where
    n_unique is the number of distinct labels in the corresponding
    column of stimuli.tsv. Requires that each target name matches a column
    in the stimuli DataFrame.

    Parameters
    ----------
    targets : list[str]
        List of target column names to derive chance for.
    stimuli_df : pd.DataFrame, optional
        If provided, use this DataFrame; otherwise loads via
        load_stimulus_metadata(return_all=True).

    Returns
    -------
    dict
        Mapping target -> chance level (float).

    Raises
    ------
    ValueError
        If a target column is missing or has <2 unique values.
    """
    if stimuli_df is None:
        stimuli_df = load_stimulus_metadata(return_all=True)

    chance_map: dict[str, float] = {}
    for t in targets:
        # Support alias: 'stimuli' → 'stim_id' (explicit mapping)
        col = t
        if t == 'stimuli':
            col = 'stim_id'

        if col in stimuli_df.columns:
            n_unique = int(pd.Series(stimuli_df[col]).dropna().nunique())
            if n_unique < 2:
                raise ValueError(f"Target '{t}' has <2 unique labels in stimuli.tsv (column '{col}')")
            chance_map[t] = 1.0 / float(n_unique)
            continue

        # Explicit rule: any *_half target is a binary split (chance = 0.5)
        if t.endswith('_half'):
            chance_map[t] = 0.5
            continue

        # Configured defaults (e.g., categories, visualStimuli, checkmate)
        defaults = CONFIG.get('MVPA_SVM_CHANCE_DEFAULTS', {})
        if t in defaults:
            chance_map[t] = float(defaults[t])
            continue

        raise ValueError(
            f"No chance level available for target '{t}'. Not found in stimuli.tsv (or alias) and no configured default."
        )
    return chance_map
