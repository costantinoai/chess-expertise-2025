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
from typing import Iterable, Tuple, Dict, List
import pandas as pd
from common import CONFIG


def _normalize_sub_id(s: str) -> str:
    s = str(s)
    return s if s.startswith('sub-') else f"sub-{int(s):02d}" if s.isdigit() else f"sub-{s}"


def find_nifti_files(data_dir: str | Path, pattern: str | None = None) -> List[Path]:
    """
    Recursively find .nii or .nii.gz files under `data_dir` optionally matching a substring.

    Parameters
    ----------
    data_dir : str or Path
    pattern : str or None
        If provided, only filenames containing this substring are returned.

    Returns
    -------
    list[Path]
        Sorted list of file paths.
    """
    root = Path(data_dir)
    matches: List[Path] = []
    for p in root.rglob('*'):
        if p.is_file() and (p.suffix in {'.nii', '.gz'} or p.name.endswith('.nii.gz')):
            if pattern is None or (pattern in p.name):
                matches.append(p)
    return sorted(matches)


def split_by_group(
    files: Iterable[Path],
    expert_ids: Iterable[str],
    novice_ids: Iterable[str],
) -> Tuple[List[Path], List[Path]]:
    """
    Split file paths into experts and novices based on subject IDs.

    Matching is done by checking for the substring 'sub-XX' within the filename.

    Parameters
    ----------
    files : iterable of Path
    expert_ids, novice_ids : iterable of str
        Subject identifiers; can be '03', 'sub-03', etc.

    Returns
    -------
    (exp_files, nov_files) : tuple of lists of Path
    """
    exp: List[Path] = []
    nov: List[Path] = []

    exp_tags = { _normalize_sub_id(sid) for sid in expert_ids }
    nov_tags = { _normalize_sub_id(sid) for sid in novice_ids }

    for f in files:
        full = str(f)
        if any(tag in full for tag in exp_tags):
            exp.append(f)
        elif any(tag in full for tag in nov_tags):
            nov.append(f)
    return exp, nov


def load_term_maps(map_dir: str | Path) -> Dict[str, Path]:
    """
    Load Neurosynth term maps into a dictionary mapping term -> filepath.

    Filenames are converted to lowercase terms with underscores replaced by spaces.
    Example: 'working_memory.nii.gz' -> 'working memory'
    """
    map_dir = Path(map_dir)
    out: Dict[str, Path] = {}
    for fname in sorted(map_dir.iterdir()):
        if fname.is_file() and (fname.suffix in {'.nii', '.gz'} or fname.name.endswith('.nii.gz')):
            term = fname.stem.replace('_', ' ').lower()
            # Strip optional numeric prefix like "1 working memory" → "working memory"
            parts = term.split(' ', 1)
            if len(parts) == 2 and parts[0].isdigit():
                term = parts[1]
            out[term] = fname
    return out


def extract_run_label(path: Path) -> str:
    """Return a human-friendly label from a T-map filename.

    Replaces legacy `_gt_` with ` > ` for readability and strips suffix.
    """
    return Path(path).stem.replace('_gt_', ' > ')


def reorder_by_term(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder correlation results to a canonical Neurosynth term order.

    Uses CONFIG['NEUROSYNTH_TERM_ORDER'] when present; otherwise no-op.
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

    Returns a sorted list of files matching 'spmT_*.nii[.gz]'.
    """
    group_dir = Path(group_dir)
    files = sorted(group_dir.glob('spmT_*.nii')) + sorted(group_dir.glob('spmT_*.nii.gz'))
    return files
