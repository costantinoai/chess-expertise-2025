"""
Neurosynth IO utilities (local to chess-neurosynth analyses).

Responsibilities
----------------
- File discovery for NIfTI maps under a root directory
- Splitting file lists by expertise group using subject IDs
- Loading Neurosynth term maps into a {term: path} mapping

Notes
-----
These functions are specific to the neurosynth analysis and are intentionally
kept local to this package. Cross-analysis utilities (participants, etc.) are
provided by common.bids_utils and common.constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import pandas as pd
from common import CONFIG
# Avoid re-exporting common helpers; import them where needed at call sites


## find_nifti_files and split_by_group are imported from common.io_utils


def load_term_maps(map_dir: str | Path) -> Dict[str, Path]:
    """
    Load Neurosynth term maps into a dictionary mapping term -> filepath.

    This function scans a directory for NIfTI files representing Neurosynth term
    association maps and creates a lookup dictionary. Filenames are normalized
    to standardized term names for easy access.

    Parameters
    ----------
    map_dir : str or Path
        Directory containing Neurosynth term map NIfTI files

    Returns
    -------
    dict
        Mapping from normalized term name (str) to file path (Path)

    Notes
    -----
    Filename normalization rules:
    - Convert to lowercase
    - Replace underscores with spaces
    - Strip numeric prefixes (e.g., "1 working memory" -> "working memory")
    - Remove .nii.gz extension

    Examples
    --------
    >>> term_maps = load_term_maps('/path/to/neurosynth/terms')
    >>> print(term_maps.keys())
    dict_keys(['working memory', 'episodic memory', 'attention', ...])
    >>> # Access a specific term map
    >>> wm_path = term_maps['working memory']
    >>> print(wm_path)
    PosixPath('/path/to/neurosynth/terms/working_memory.nii.gz')

    See Also
    --------
    extract_run_label : Generate human-friendly labels from T-map filenames
    reorder_by_term : Sort correlation results by canonical term order
    """
    map_dir = Path(map_dir)
    out: Dict[str, Path] = {}
    for fname in sorted(map_dir.iterdir()):
        if fname.is_file() and (fname.suffix in {'.nii.gz', '.gz'} or fname.name.endswith('.nii.gz')):
            term = fname.stem.replace('.nii','').replace('.gz','').replace('_', ' ').lower()
            # Strip optional numeric prefix like "1 working memory" → "working memory"
            parts = term.split(' ', 1)
            if len(parts) == 2 and parts[0].isdigit():
                term = parts[1]
            out[term] = fname
    return out


def extract_run_label(path: Path) -> str:
    """
    Return a human-friendly label from a T-map filename.

    Converts SPM T-map filenames to readable contrast labels by replacing
    underscore patterns with mathematical symbols.

    Parameters
    ----------
    path : Path
        Path to SPM T-map file (e.g., 'spmT_experts_gt_novices.nii.gz')

    Returns
    -------
    str
        Human-readable label (e.g., 'spmT_experts > novices')

    Examples
    --------
    >>> from pathlib import Path
    >>> label = extract_run_label(Path('spmT_experts_gt_novices.nii.gz'))
    >>> print(label)
    spmT_experts > novices
    """
    # Handle .nii.gz files properly by removing both extensions
    name = Path(path).name
    if name.endswith('.nii.gz'):
        name = name[:-7]  # Remove .nii.gz
    elif name.endswith('.nii'):
        name = name[:-4]  # Remove .nii
    return name.replace('_gt_', ' > ')


def reorder_by_term(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder correlation results to a canonical Neurosynth term order.

    Applies a standardized ordering to Neurosynth correlation results for
    consistent presentation across figures and tables. Uses CONFIG to define
    the canonical order.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'term' column containing Neurosynth term names

    Returns
    -------
    pd.DataFrame
        Reordered DataFrame with terms sorted by CONFIG['NEUROSYNTH_TERM_ORDER']
        If 'term' column is missing or CONFIG order not defined, returns
        unchanged DataFrame

    Notes
    -----
    - Term matching is case-insensitive
    - Terms not in CONFIG order are appended at the end
    - Uses pandas categorical ordering for robust sorting

    Examples
    --------
    >>> # Assuming CONFIG['NEUROSYNTH_TERM_ORDER'] = ['memory', 'attention', 'visual']
    >>> df = pd.DataFrame({'term': ['visual', 'memory', 'attention'], 'r': [0.5, 0.7, 0.6]})
    >>> df_ordered = reorder_by_term(df)
    >>> print(df_ordered['term'].tolist())
    ['memory', 'attention', 'visual']
    """
    order = CONFIG.get('NEUROSYNTH_TERM_ORDER', [])
    if 'term' not in df.columns or not order:
        return df
    out = df.copy()
    out['term'] = out['term'].str.lower()
    cat = pd.Categorical(out['term'], categories=order, ordered=True)
    out['__ord'] = cat
    out = out.sort_values('__ord').drop(columns='__ord')
    return out


def find_group_tmaps(group_dir: Path) -> List[Path]:
    """
    Find group-level SPM T-maps under a specific group directory.

    Scans a directory for SPM T-map files produced by second-level analyses.
    Returns sorted paths for reproducible ordering.

    Parameters
    ----------
    group_dir : Path
        Directory containing group-level SPM outputs (e.g., results/experts/)

    Returns
    -------
    list of Path
        Sorted list of files matching pattern 'spmT_*.nii.gz'

    Examples
    --------
    >>> from pathlib import Path
    >>> group_dir = Path('derivatives/neurosynth-rsa/group/experts')
    >>> tmaps = find_group_tmaps(group_dir)
    >>> print([t.name for t in tmaps])
    ['spmT_0001.nii.gz', 'spmT_0002.nii.gz', ...]
    """
    group_dir = Path(group_dir)
    files = sorted(group_dir.glob('spmT_*.nii.gz'))
    return files
