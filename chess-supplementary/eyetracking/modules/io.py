"""
Eyetracking IO helpers for supplementary decoding analysis.

Responsibilities
----------------
- Discover eyetracking TSV files under CONFIG['BIDS_EYETRACK']
- Load per-run TSVs and associated JSON metadata
- Assemble a long-format dataframe with columns: subject, run, expert, ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
import json
import re
import pandas as pd
import numpy as np

from common import CONFIG
from common.bids_utils import get_participants_with_expertise


def _parse_subject_and_run(stem: str) -> Tuple[str, str]:
    """Parse subject id and run number from a BIDS-like filename stem.

    Example: 'sub-01_task-rest_run-3_desc-1to6_eyetrack' → ('sub-01', '3')
    """
    parts = stem.split('_')
    sub = next((p for p in parts if p.startswith('sub-')), '')
    run = next((p for p in parts if p.startswith('run-')), '')
    run_num = run.split('-')[1] if '-' in run else ''
    return sub, run_num


def find_eyetracking_tsvs(root: Path | str) -> List[Path]:
    """
    Find eyetracking TSV files below the given root.

    Looks for files under sub-*/func/ matching '*eyetrack.tsv' to accommodate
    legacy naming. Enforces that each file has an adjacent .json metadata file.
    """
    root = Path(root)
    files = sorted(root.glob('sub-*/func/*eyetrack.tsv'))
    out = []
    for f in files:
        if not f.is_file():
            continue
        if not f.with_suffix('.json').exists():
            raise FileNotFoundError(f"Missing JSON metadata for {f}")
        out.append(f)
    return out


def load_eyetracking_dataframe(root: Path | str) -> pd.DataFrame:
    """
    Load all eyetracking TSVs and metadata into a single DataFrame.

    Adds subject, run, and expert columns.
    Participants are read from CONFIG['BIDS_PARTICIPANTS'].
    """
    root = Path(root)
    files = find_eyetracking_tsvs(root)
    participants, _ = get_participants_with_expertise(
        participants_file=CONFIG['BIDS_PARTICIPANTS'], bids_root=CONFIG['BIDS_ROOT']
    )
    expert_lookup = {sid: is_exp for sid, is_exp in participants}

    frames: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f, sep='\t')
        sub, run = _parse_subject_and_run(f.stem)
        if not sub:
            # Fallback parse from path (sub-XX directory)
            m = re.search(r"sub-[0-9]+", str(f))
            sub = m.group(0) if m else ''
        df['subject'] = sub
        df['run'] = run
        df['expert'] = bool(expert_lookup.get(sub, False))

        meta_path = f.with_suffix('.json')
        with meta_path.open('r') as fh:
            meta = json.load(fh)
        # attach selected scalar metadata columns (skip nested)
        for k, v in meta.items():
            if isinstance(v, (str, int, float)):
                # Avoid collisions with required columns
                if k in df.columns:
                    continue
                df[k] = v
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=['subject', 'run', 'expert'])
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out
