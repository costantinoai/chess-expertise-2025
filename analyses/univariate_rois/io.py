from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# Canonical univariate contrasts (file stems → display labels)
UNIV_CONTRASTS: Dict[str, str] = {
    'con_0001': 'Checkmate > Non-checkmate',
    'con_0002': 'All > Rest',
}


def _candidates_for_contrast(subject_id: str, con_code: str, base_dir: Path) -> List[Path]:
    """
    Build strict candidate paths for a subject contrast file.

    Tries '.nii.gz', and with/without 'exp/' subfolder.
    Exactly one existing file must be found; otherwise raises.
    """
    con_file_gz = f"{con_code}.nii.gz"
    subject_dir = Path(base_dir) / subject_id

    pats = [
        subject_dir / 'exp' / con_file_gz,
        subject_dir / con_file_gz,
    ]
    return [p for p in pats if p.exists()]


def find_subject_contrast_path(subject_id: str, con_code: str, base_dir: Path) -> Path:
    """
    Resolve a single contrast image for a subject or raise on ambiguity/missing.

    Parameters
    ----------
    subject_id : str
        BIDS subject id, e.g., 'sub-01'.
    con_code : str
        Contrast code without extension, e.g., 'con_0001'.
    base_dir : Path
        Root directory of first-level SPM contrasts (smoothed).

    Returns
    -------
    Path
        Resolved file path.
    """
    cands = _candidates_for_contrast(subject_id, con_code, base_dir)
    if len(cands) == 0:
        raise FileNotFoundError(
            f"Missing contrast for {subject_id} {con_code} under {base_dir}"
        )
    if len(cands) > 1:
        raise RuntimeError(
            f"Multiple contrast matches for {subject_id} {con_code}: {cands}"
        )
    return cands[0]

