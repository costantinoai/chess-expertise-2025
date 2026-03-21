"""
Local helpers shared by the skill-gradient analysis and plotting scripts.
"""

from pathlib import Path

import pandas as pd


def load_bids_tsvs(bids_dir, pattern='*.tsv'):
    """
    Load one TSV per subject from a BIDS-derivatives directory.

    Parameters
    ----------
    bids_dir : str or Path
        Directory containing one subdirectory per subject.
    pattern : str, default='*.tsv'
        Glob used to select the TSV file inside each subject directory.

    Returns
    -------
    pd.DataFrame
        Long-format table with subject ID, target label, and all TSV columns.
    """
    rows = []
    for sub_dir in sorted(Path(bids_dir).iterdir()):
        if not sub_dir.is_dir():
            continue
        sub_id = sub_dir.name
        tsv_files = list(sub_dir.glob(pattern))
        if not tsv_files:
            continue
        df = pd.read_csv(tsv_files[0], sep='\t', index_col=0)
        for target in df.index:
            row = {'subject': sub_id, 'target': target}
            row.update(df.loc[target].to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def compute_subject_mean_pr(pr_long: pd.DataFrame, subject_ids=None) -> pd.DataFrame:
    """
    Compute mean participation ratio per subject.

    Parameters
    ----------
    pr_long : pd.DataFrame
        Long-format PR data with columns 'subject_id' and 'PR'.
    subject_ids : collection or None, default=None
        Optional subset of subject IDs to retain before averaging.

    Returns
    -------
    pd.DataFrame
        Two columns: 'participant_id' and 'mean_pr'.
    """
    pr_subset = pr_long
    if subject_ids is not None:
        pr_subset = pr_long[pr_long['subject_id'].isin(subject_ids)]

    subject_pr = pr_subset.groupby('subject_id', as_index=False)['PR'].mean()
    return subject_pr.rename(columns={'subject_id': 'participant_id', 'PR': 'mean_pr'})


__all__ = ['load_bids_tsvs', 'compute_subject_mean_pr']
