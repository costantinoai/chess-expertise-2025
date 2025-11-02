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

    Filenames are converted to lowercase terms with underscores replaced by spaces.
    Example: 'working_memory.nii.gz' -> 'working memory'
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
    files = sorted(group_dir.glob('spmT_*.nii.gz'))
    return files
