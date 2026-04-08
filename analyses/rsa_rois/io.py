from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# Canonical RSA targets (keys → display labels). The dict KEY is the
# label used in repo-side code; the matching BIDS ``desc-`` entity may
# differ because BIDS entity values cannot contain underscores
# (so ``visual_similarity`` becomes ``visualSimilarity`` in filenames).
RSA_TARGETS: Dict[str, str] = {
    'visual_similarity': 'Visual Similarity',
    'strategy': 'Strategy',
    'checkmate': 'Checkmate',
}

# Map an internal target key onto the BIDS ``desc-`` entity value used
# inside the per-subject searchlight filenames.
_BIDS_DESC_FOR_TARGET: Dict[str, str] = {
    'visual_similarity': 'visualSimilarity',
    'strategy':          'strategy',
    'checkmate':         'checkmate',
}


def _candidates_for_target(subject_id: str, target_key: str, base_dir: Path) -> List[Path]:
    """
    Build strict candidate paths for a subject RSA r-map.

    Files live at:
    ``sub-XX/sub-XX_space-MNI152NLin2009cAsym_desc-<bids_desc>_stat-r_searchlight.nii.gz``
    """
    subject_dir = Path(base_dir) / subject_id
    bids_desc = _BIDS_DESC_FOR_TARGET[target_key]
    stem = (
        f"{subject_id}_space-MNI152NLin2009cAsym"
        f"_desc-{bids_desc}_stat-r_searchlight"
    )
    pats = [
        subject_dir / f"{stem}.nii.gz",
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
        One of RSA_TARGETS keys: 'visual_similarity', 'strategy', 'checkmate'.
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

