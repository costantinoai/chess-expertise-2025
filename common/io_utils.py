"""
Input/output utilities for loading and saving analysis results.

This module provides centralized functions for finding and managing results
directories, copying scripts, and handling file I/O across all analyses.

Functions are designed to work with the standardized timestamped results
directory structure: results/<YYYYMMDD-HHMMSS>_<analysis_name>/
"""

from pathlib import Path
from typing import Optional, List
import shutil
import logging


def find_latest_results_directory(
    results_base: Path,
    pattern: str = "*",
    specific_name: Optional[str] = None,
    create_subdirs: Optional[List[str]] = None,
    require_exists: bool = True,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Find or validate a results directory and optionally create subdirectories.

    This function handles all common results directory use cases:
    1. Find the most recent timestamped directory matching a pattern
    2. Validate a specific named directory exists
    3. Create standard subdirectories (e.g., 'figures', 'tables')

    This keeps analysis scripts clean by handling all directory logic in one place.

    Parameters
    ----------
    results_base : Path
        Base directory containing timestamped result folders
    pattern : str, default='*'
        Glob pattern to match directories (e.g., '*_behavioral_rsa')
        Only used if specific_name is None
    specific_name : str, optional
        Name of a specific results directory to use instead of finding latest
        If provided, returns results_base / specific_name
    create_subdirs : list of str, optional
        Subdirectories to create within the results directory
        Example: ['figures', 'tables']
    require_exists : bool, default=True
        If True, raises FileNotFoundError if directory not found
    verbose : bool, default=True
        If True, prints which directory is being used

    Returns
    -------
    Path
        Path to the results directory

    Raises
    ------
    FileNotFoundError
        If directory not found and require_exists=True

    Notes
    -----
    - Directories should be named with timestamp prefix: YYYYMMDD-HHMMSS_name
    - Sorting is alphanumeric, so timestamp format ensures correct ordering
    - Subdirectories are created with parents=True, exist_ok=True
    - This replaces the old pattern of manually checking, creating subdirs, etc.

    Example
    -------
    >>> # Use case 1: Find latest directory and create subdirs
    >>> results_dir = find_latest_results_directory(
    ...     Path('results'),
    ...     pattern='*_behavioral_rsa',
    ...     create_subdirs=['figures', 'tables']
    ... )
    >>> # Prints: "Using most recent results directory: 20251018-125050_behavioral_rsa"
    >>> # Creates: results/.../figures/ and results/.../tables/
    >>> # Returns: Path('results/20251018-125050_behavioral_rsa')
    >>>
    >>> # Use case 2: Use specific directory
    >>> results_dir = find_latest_results_directory(
    ...     Path('results'),
    ...     specific_name='20251018-125050_behavioral_rsa',
    ...     create_subdirs=['figures', 'tables']
    ... )
    >>> # Prints: "Using specified results directory: 20251018-125050_behavioral_rsa"
    >>>
    >>> # Use case 3: Just find latest, no subdirs
    >>> results_dir = find_latest_results_directory(
    ...     Path('results'),
    ...     pattern='*_mvpa'
    ... )
    """
    results_base = Path(results_base)

    # === Case 1: Specific directory name provided ===
    if specific_name is not None:
        results_dir = results_base / specific_name
        if not results_dir.exists():
            if require_exists:
                raise FileNotFoundError(f"Results directory not found: {results_dir}")
            else:
                results_dir.mkdir(parents=True, exist_ok=True)
                if verbose:
                    msg = f"Created results directory: {results_dir.name}"
                    (logger.info(msg) if logger else print(msg))
        else:
            if verbose:
                msg = f"Using specified results directory: {results_dir.name}"
                (logger.info(msg) if logger else print(msg))

    # === Case 2: Find latest matching directory ===
    else:
        if not results_base.exists():
            if require_exists:
                raise FileNotFoundError(f"Results base directory not found: {results_base}")
            results_base.mkdir(parents=True, exist_ok=True)

        # Find all matching directories
        matching_dirs = sorted(
            [d for d in results_base.glob(pattern) if d.is_dir()],
            key=lambda x: x.name,
            reverse=True  # Most recent first
        )

        if len(matching_dirs) == 0:
            if require_exists:
                raise FileNotFoundError(
                    f"No matching results directories found in {results_base} with pattern '{pattern}'"
                )
            return None

        results_dir = matching_dirs[0]

        if verbose:
            msg = f"Using most recent results directory: {results_dir.name}"
            (logger.info(msg) if logger else print(msg))

    # === Create subdirectories if requested ===
    if create_subdirs is not None:
        for subdir_name in create_subdirs:
            subdir = results_dir / subdir_name
            subdir.mkdir(parents=True, exist_ok=True)

    return results_dir


__all__ = [
    'find_latest_results_directory',
    'copy_script_to_results',
    'get_all_results_directories',
    'validate_results_directory',
    'resolve_latest_dir',
    'sanitize_matlab_varnames',
    'pick_first_present',
    'find_nifti_files',
    'split_by_group',
    'discover_files_by_group',
    'find_subject_tsvs',
]


def copy_script_to_results(
    script_path: Path,
    results_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Copy the executing script to the results directory for reproducibility.

    This ensures that the exact code used to generate results is preserved
    alongside the outputs.

    Parameters
    ----------
    script_path : Path
        Path to the script file to copy (typically __file__)
    results_dir : Path
        Results directory where the script copy will be saved
    logger : logging.Logger, optional
        Logger instance for logging the copy operation

    Returns
    -------
    Path
        Path to the copied script file

    Notes
    -----
    - If the script file doesn't exist, a warning is logged but no error is raised
    - Preserves the original filename
    - Overwrites existing copies without warning

    Example
    -------
    >>> from pathlib import Path
    >>> # In your analysis script:
    >>> script_copy = copy_script_to_results(
    ...     Path(__file__),
    ...     output_dir,
    ...     logger
    ... )
    """
    script_path = Path(script_path)
    results_dir = Path(results_dir)

    if not script_path.exists():
        msg = f"Script file not found: {script_path}"
        if logger:
            logger.warning(msg)
        else:
            import warnings
            warnings.warn(msg)
        return None

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Copy script with original filename
    dest_path = results_dir / script_path.name
    shutil.copy(script_path, dest_path)

    if logger:
        logger.info(f"Copied script to: {dest_path}")

    return dest_path


def get_all_results_directories(
    results_base: Path,
    pattern: str = "*"
) -> List[Path]:
    """
    Get all timestamped results directories matching a pattern, sorted by date.

    Parameters
    ----------
    results_base : Path
        Base directory containing results subdirectories
    pattern : str, default="*"
        Glob pattern to filter directories

    Returns
    -------
    list of Path
        List of matching results directories, sorted from newest to oldest

    Example
    -------
    >>> # Get all behavioral RSA results
    >>> all_results = get_all_results_directories(
    ...     Path("chess-behavioral/results"),
    ...     pattern="*_behavioral_rsa"
    ... )
    >>> for result_dir in all_results:
    ...     print(result_dir.name)
    """
    results_base = Path(results_base)

    if not results_base.exists():
        return []

    matching_dirs = sorted(
        [d for d in results_base.glob(pattern) if d.is_dir()],
        key=lambda x: x.name,
        reverse=True  # Newest first
    )

    return matching_dirs


def validate_results_directory(
    results_dir: Path,
    required_files: Optional[List[str]] = None
) -> bool:
    """
    Validate that a results directory exists and contains required files.

    Parameters
    ----------
    results_dir : Path
        Results directory to validate
    required_files : list of str, optional
        List of filenames that must exist in the results directory
        If None, only checks that the directory exists

    Returns
    -------
    bool
        True if valid, False otherwise

    Raises
    ------
    FileNotFoundError
        If results_dir doesn't exist or required files are missing

    Example
    -------
    >>> # Validate behavioral RSA results
    >>> is_valid = validate_results_directory(
    ...     results_dir,
    ...     required_files=[
    ...         "expert_behavioral_rdm.npy",
    ...         "novice_behavioral_rdm.npy",
    ...         "correlation_results.pkl"
    ...     ]
    ... )
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Path is not a directory: {results_dir}")

    if required_files:
        missing_files = []
        for filename in required_files:
            filepath = results_dir / filename
            if not filepath.exists():
                missing_files.append(filename)

        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {results_dir}:\n" +
                "\n".join(f"  - {f}" for f in missing_files)
            )

    return True


def resolve_latest_dir(
    base_dir: Path,
    pattern: str,
    specific_name: Optional[str] = None,
) -> Path:
    """
    Resolve a directory under base_dir by either specific name or latest match.

    Parameters
    ----------
    base_dir : Path
        Base directory to search
    pattern : str
        Glob pattern (e.g., '*_glasser_regions_bilateral')
    specific_name : str, optional
        If provided, returns base_dir / specific_name

    Returns
    -------
    Path
        Resolved directory path
    """
    base_dir = Path(base_dir)
    if specific_name is not None:
        path = base_dir / specific_name
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        return path
    matches = sorted([d for d in base_dir.glob(pattern) if d.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No directories matching '{pattern}' under {base_dir}")
    return matches[-1]


def sanitize_matlab_varnames(names: List[str]) -> List[str]:
    """
    Emulate MATLAB makeValidName with ReplacementStyle='delete'.

    - Remove invalid characters (keep A-Z, a-z, 0-9, and underscore)
    - If the first character is not a letter, prepend 'x'
    - If resulting name is empty, return 'x'
    """
    sanitized = []
    for n in names:
        s = ''.join(ch for ch in str(n) if (ch.isalnum() or ch == '_'))
        if not s or not s[0].isalpha():
            s = 'x' + s
        sanitized.append(s)
    return sanitized


def pick_first_present(obj, candidates: List[str]) -> Optional[str]:
    """
    Return the first candidate present in a DataFrame's columns or an iterable of names.
    """
    cols: List[str]
    if hasattr(obj, 'columns'):
        cols = list(obj.columns)
    else:
        cols = list(obj)
    for c in candidates:
        if c in cols:
            return c
    return None


# =============================================================================
# Generalized file discovery helpers (consolidated)
# =============================================================================

def find_nifti_files(data_dir: Path | str, pattern: str | None = None) -> List[Path]:
    """
    Recursively find .nii or .nii.gz files under `data_dir` optionally matching a substring.
    """
    root = Path(data_dir)
    matches: List[Path] = []
    for p in root.rglob('*'):
        if p.is_file() and (p.suffix in {'.nii', '.gz'} or p.name.endswith('.nii.gz')):
            if pattern is None or (pattern in p.name):
                matches.append(p)
    return sorted(matches)


def _normalize_sub_id(s: str) -> str:
    s = str(s)
    return s if s.startswith('sub-') else f"sub-{int(s):02d}" if s.isdigit() else f"sub-{s}"


def split_by_group(
    files: List[Path],
    expert_ids: List[str],
    novice_ids: List[str],
) -> tuple[List[Path], List[Path]]:
    """
    Split file paths into experts and novices based on subject IDs.
    Matches 'sub-XX' substrings within filenames.
    """
    exp: List[Path] = []
    nov: List[Path] = []
    exp_tags = {_normalize_sub_id(sid) for sid in expert_ids}
    nov_tags = {_normalize_sub_id(sid) for sid in novice_ids}
    for f in files:
        full = str(f)
        if any(tag in full for tag in exp_tags):
            exp.append(f)
        elif any(tag in full for tag in nov_tags):
            nov.append(f)
    return exp, nov


def discover_files_by_group(
    base_dir: Path | str,
    filename_substring: str,
    expert_ids: List[str],
    novice_ids: List[str],
) -> tuple[List[Path], List[Path]]:
    """
    Convenience wrapper: find NIfTIs matching a substring and split by group.
    """
    files = find_nifti_files(base_dir, pattern=filename_substring)
    return split_by_group(files, expert_ids, novice_ids)


def find_subject_tsvs(method_dir: Path) -> List[Path]:
    """
    Find subject-level TSV files under a method directory of the form:
        method_dir/sub-XX/*.tsv
    Returns one TSV per subject (first in lexical order when multiple found).
    """
    logger = logging.getLogger(__name__)
    tsvs: List[Path] = []
    method_dir = Path(method_dir)
    for sub_dir in sorted(method_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        candidates = sorted(sub_dir.glob("*.tsv"))
        if not candidates:
            continue
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple TSVs found under {sub_dir} — explicit disambiguation required: {', '.join(p.name for p in candidates)}"
            )
        tsvs.append(candidates[0])
    return tsvs
