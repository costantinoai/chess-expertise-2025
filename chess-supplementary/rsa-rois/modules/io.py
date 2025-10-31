from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# Canonical RSA targets (keys → display labels)
RSA_TARGETS: Dict[str, str] = {
    'visualSimilarity': 'Visual Similarity',
    'strategy': 'Strategy',
    'checkmate': 'Checkmate',
}


def _candidates_for_target(subject_id: str, target_key: str, base_dir: Path) -> List[Path]:
    """
    Build strict candidate paths for a subject RSA r-map.

    Files are expected like: sub-XX/sub-XX_desc-searchlight_<target>_stat-r_map.(nii|nii.gz)
    """
    subject_dir = Path(base_dir) / subject_id
    stem = f"{subject_id}_desc-searchlight_{target_key}_stat-r_map"
    pats = [
        subject_dir / f"{stem}.nii.gz",
        subject_dir / f"{stem}.nii",
    ]
    return [p for p in pats if p.exists()]


def find_subject_rsa_path(subject_id: str, target_key: str, base_dir: Path) -> Path:
    """
    Resolve a single RSA r-map for a subject or raise on ambiguity/missing.

    Parameters
    ----------
    subject_id : str
        BIDS subject id, e.g., 'sub-01'.
    target_key : str
        One of RSA_TARGETS keys: 'visualSimilarity', 'strategy', 'checkmate'.
    base_dir : Path
        Root directory of RSA searchlight derivatives.
    """
    cands = _candidates_for_target(subject_id, target_key, base_dir)
    if len(cands) == 0:
        raise FileNotFoundError(
            f"Missing RSA r-map for {subject_id} {target_key} under {base_dir}"
        )
    if len(cands) > 1:
        raise RuntimeError(
            f"Multiple RSA r-map matches for {subject_id} {target_key}: {cands}"
        )
    return cands[0]

