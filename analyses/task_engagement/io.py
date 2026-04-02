"""
I/O utilities for familiarisation task analysis.

Loads BIDS-format familiarisation behavioural TSVs and participant metadata.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from common import CONFIG
from common.bids_utils import load_participants_tsv


# Participants without familiarisation data (no Pavlovia record found)
MISSING_SUBJECTS = {'sub-07', 'sub-39'}


def load_familiarisation_data() -> pd.DataFrame:
    """
    Load all per-subject familiarisation TSVs and merge with participant metadata.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns from the BIDS TSVs plus
        'participant_id' and 'group' (expert/novice).
    """
    participants = load_participants_tsv()
    bids_root = Path(CONFIG['BIDS_ROOT'])

    rows = []
    for _, p in participants.iterrows():
        sub_id = p['participant_id']
        if sub_id in MISSING_SUBJECTS:
            continue

        tsv_path = bids_root / sub_id / 'beh' / f'{sub_id}_task-familiarisation_beh.tsv'
        if not tsv_path.exists():
            continue

        df = pd.read_csv(tsv_path, sep='\t')
        df['participant_id'] = sub_id
        df['group'] = p['group']
        rows.append(df)

    if not rows:
        raise FileNotFoundError("No familiarisation TSVs found in BIDS directory")

    return pd.concat(rows, ignore_index=True)
