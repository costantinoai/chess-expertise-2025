"""
RDM and DSM computation utilities for behavioral analysis.

This module provides behavioral-specific functions for converting trial data
to pairwise comparisons and computing behavioral RDMs/DSMs.

Key Concepts
------------
RDM (Representational Dissimilarity Matrix):
    Symmetric matrix capturing how dissimilar pairs of stimuli are based on
    behavioral preferences. Higher values indicate greater dissimilarity.

DSM (Directional Similarity/Dissimilarity Matrix):
    Antisymmetric matrix capturing directional preferences. Positive values
    indicate stimulus i is preferred over j; negative values indicate the opposite.

Workflow
--------
1. Load trial data from BIDS events files (via data_loading.py)
2. Convert trials to pairwise comparisons (create_pairwise_df)
3. Compute behavioral RDM (compute_symmetric_rdm) or DSM (compute_directional_dsm)
4. Correlate with model RDMs (correlate_with_all_models)

For general RSA functions (model RDM creation, correlation), use common.rsa_utils.

Notes for Academic Users
------------------------
- All RDM computations preserve the analytical logic from the original implementation
- Button mappings account for counterbalancing across participants
- Functions handle missing data gracefully (trials with no preference are skipped)
- Results are deterministic given the same input data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List


# Import shared RSA functions from common
from common.rsa_utils import (
    correlate_rdm_with_models,
)


def _pairwise_counts_series(pairwise_df: pd.DataFrame) -> pd.Series:
    """
    Normalize raw or aggregated pairwise preferences to a count series.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise data with 'better' and 'worse' columns and an optional 'count'
        column.

    Returns
    -------
    pd.Series
        MultiIndex series keyed by (better, worse) with comparison counts.
    """
    if pairwise_df.empty:
        return pd.Series(dtype=float)

    if "count" in pairwise_df.columns:
        return pairwise_df.set_index(["better", "worse"])["count"].sort_index()

    return pairwise_df.groupby(["better", "worse"]).size().sort_index()


def _build_count_matrix(
    pairwise_df: pd.DataFrame,
    dtype=float,
    n_stimuli: int | None = None,
) -> np.ndarray:
    """
    Build a directed count matrix from raw or aggregated pairwise preferences.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise data with 'better' and 'worse' columns and an optional 'count'
        column.
    dtype : data-type, default=float
        Output dtype for the matrix.
    n_stimuli : int or None, default=None
        Matrix size. If None, infer from the largest stimulus ID present.

    Returns
    -------
    np.ndarray
        Square matrix where entry (i, j) contains the count of times stimulus i
        was preferred over stimulus j.
    """
    counts = _pairwise_counts_series(pairwise_df)
    if counts.empty:
        size = 0 if n_stimuli is None else int(n_stimuli)
        return np.zeros((size, size), dtype=dtype)

    inferred_n = int(max(max(pair) for pair in counts.index))
    if n_stimuli is None:
        n_stimuli = inferred_n
    elif inferred_n > int(n_stimuli):
        raise ValueError(
            f"pairwise_df contains stimulus ID {inferred_n}, which exceeds n_stimuli={n_stimuli}"
        )

    count_matrix = np.zeros((int(n_stimuli), int(n_stimuli)), dtype=dtype)
    pairs = np.asarray(counts.index.tolist(), dtype=int)
    count_matrix[pairs[:, 0] - 1, pairs[:, 1] - 1] = counts.to_numpy(dtype=dtype)
    return count_matrix


def create_pairwise_df(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert trial-level data to pairwise comparison DataFrame.

    In the 1-back task, participants see two boards in sequence and indicate
    which they prefer. This function extracts those pairwise preferences from
    the trial data by reading the 'preference' column from BIDS events files.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial-level data with columns:
        - stim_id: Stimulus ID (1-40)
        - preference: 'current_preferred', 'previous_preferred', or 'n/a'
        - sub_id: Subject identifier
    aggregate : bool, default=False
        If True, aggregate pairwise counts across all comparisons (sum counts for each pair).
        If False, return raw pairwise comparisons (one row per trial).

    Returns
    -------
    pd.DataFrame
        If aggregate=False:
            Pairwise comparison data with columns:
            - better: ID of preferred stimulus
            - worse: ID of non-preferred stimulus
            - sub_id: Subject identifier
        If aggregate=True:
            Aggregated pairwise counts with columns:
            - better: ID of preferred stimulus
            - worse: ID of non-preferred stimulus
            - count: Number of times this pair was observed

    Notes
    -----
    - Reads 'preference' column directly from BIDS events.tsv files (GROUND TRUTH)
    - Preference has already been corrected for button mapping counterbalancing
    - Skips trials with preference='n/a' (no response or timeout)
    - IMPORTANT: Does NOT recalculate preference from button codes - that's already
      done in the BIDS conversion (convert_mat_to_bids_events.py)

    Example
    -------
    >>> trial_df = load_participant_trial_data('sub-01', True, BIDS_ROOT)
    >>> pairwise_df = create_pairwise_df(trial_df, aggregate=False)
    >>> print(f"Extracted {len(pairwise_df)} pairwise comparisons")
    >>>
    >>> # For group analysis, aggregate across participants
    >>> pairwise_agg = create_pairwise_df(trial_df, aggregate=True)
    >>> print(f"Aggregated to {len(pairwise_agg)} unique pairs")
    """
    comparisons = []
    skipped_counts = {'no_preference': 0, 'cross_run': 0}

    # === Process each trial to extract pairwise preferences ===
    # Start from second trial (index 1) because we need the previous stimulus
    # The 1-back task shows stimuli sequentially, and participants compare current vs previous
    for i in range(1, len(trial_df)):
        # Get preference from BIDS events file (already counterbalancing-corrected)
        preference = trial_df.iloc[i].get("preference", "n/a")

        # Skip trials where participant expressed no preference
        # NOTE: Skipping n/a is mathematically equivalent to treating it as "0 preference" (tie).
        # For RDM[i,j] = |count(i>j) - count(j>i)|, a tie contributes equally to both
        # directions, so the absolute difference remains the same whether we skip or count as 0.5 each.
        if preference == "n/a":
            skipped_counts['no_preference'] += 1
            continue

        # CRITICAL: Check if this crosses a run boundary
        # Don't compare stimuli from different runs (they were never shown consecutively)
        if 'run' in trial_df.columns:
            current_run = trial_df.iloc[i]['run']
            previous_run = trial_df.iloc[i - 1]['run']
            if current_run != previous_run:
                skipped_counts['cross_run'] += 1
                continue  # Skip cross-run comparisons

        # Get stimulus IDs for current and previous trial
        current_stim = int(trial_df.iloc[i]["stim_id"])
        previous_stim = int(trial_df.iloc[i - 1]["stim_id"])

        # === Determine which stimulus was preferred ===
        # Preference column is already corrected for button mapping counterbalancing
        if preference == "current_preferred":
            # Current stimulus preferred over previous
            better, worse = current_stim, previous_stim
        elif preference == "previous_preferred":
            # Previous stimulus preferred over current
            better, worse = previous_stim, current_stim
        else:
            # Should not happen (we already filtered n/a), but skip just in case
            continue

        # Store pairwise comparison
        comparisons.append(
            {
                "better": better,  # ID of preferred stimulus
                "worse": worse,  # ID of non-preferred stimulus
                "sub_id": trial_df.iloc[i]["sub_id"],  # Subject who made this comparison
            }
        )

    pairwise_df = pd.DataFrame(comparisons)

    return pairwise_df

def compute_symmetric_rdm(pairwise_df: pd.DataFrame) -> np.ndarray:
    """
    Compute symmetric representational dissimilarity matrix (RDM) from pairwise data.

    The symmetric RDM represents how dissimilar each pair of stimuli is based on
    behavioral preferences. It is computed as the absolute difference between
    the number of times stimulus i was preferred over j and vice versa.

    RDM[i,j] = |count(i preferred over j) - count(j preferred over i)|

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison data with 'better' and 'worse' columns.
        Optionally includes 'count' column if aggregated.

    Returns
    -------
    np.ndarray
        Symmetric RDM matrix (n_stimuli × n_stimuli)
        - Larger values = greater dissimilarity
        - Diagonal should be zeros
        - Matrix is symmetric: RDM[i,j] = RDM[j,i]

    Notes
    -----
    - Assumes stimulus IDs range from 1 to n
    - Creates n×n matrix where n is the maximum stimulus ID
    - Returns integer-valued matrix
    - Handles both raw pairwise data and aggregated data (with 'count' column)

    Example
    -------
    >>> pairwise_df = create_pairwise_df(trial_df)
    >>> rdm = compute_symmetric_rdm(pairwise_df)
    >>> print(f"RDM shape: {rdm.shape}")
    >>> print(f"RDM range: [{rdm.min()}, {rdm.max()}]")
    """
    count_matrix = _build_count_matrix(pairwise_df, dtype=int)

    # Compute symmetric RDM as absolute difference
    # RDM[i,j] = |count(i>j) - count(j>i)|
    rdm = np.abs(count_matrix - count_matrix.T)

    return rdm


def compute_normalized_rdm(pairwise_df: pd.DataFrame) -> np.ndarray:
    """
    Compute count-normalized symmetric RDM from pairwise data.

    Because the 1-back task compares only consecutive boards, different
    stimulus pairs are compared different numbers of times depending on
    the randomized presentation sequence. To control for this exposure
    confound, the normalized RDM divides each entry by the total number
    of comparisons for that pair:

    RDM_norm[i,j] = |count(i>j) - count(j>i)| / (count(i>j) + count(j>i))

    Values range from 0 (perfectly tied) to 1 (perfectly consistent).
    Pairs with zero comparisons are set to 0.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison data with 'better' and 'worse' columns.
        Optionally includes 'count' column if aggregated.

    Returns
    -------
    np.ndarray
        Count-normalized symmetric RDM matrix (n_stimuli x n_stimuli)
    """
    count_matrix = _build_count_matrix(pairwise_df, dtype=float)

    # Numerator: |count(i>j) - count(j>i)|
    numerator = np.abs(count_matrix - count_matrix.T)
    # Denominator: count(i>j) + count(j>i)
    denominator = count_matrix + count_matrix.T
    # Avoid division by zero for pairs never compared
    with np.errstate(divide='ignore', invalid='ignore'):
        rdm_norm = np.where(denominator > 0, numerator / denominator, 0.0)

    return rdm_norm


def compute_directional_dsm(pairwise_df: pd.DataFrame) -> np.ndarray:
    """
    Compute directional dissimilarity/preference matrix (DSM) from pairwise data.

    The directional DSM captures asymmetric preferences. Positive values indicate
    stimulus i is preferred over j; negative values indicate j is preferred over i.

    DSM[i,j] = count(i preferred over j) - count(j preferred over i)

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison data with 'better' and 'worse' columns.
        Optionally includes 'count' column if aggregated.

    Returns
    -------
    np.ndarray
        Directional DSM matrix (n_stimuli × n_stimuli)
        - Positive DSM[i,j]: stimulus i preferred over j
        - Negative DSM[i,j]: stimulus j preferred over i
        - Matrix is antisymmetric: DSM[i,j] = -DSM[j,i]
        - Diagonal should be zeros

    Notes
    -----
    - Unlike symmetric RDM, DSM preserves preference direction
    - Useful for understanding directional biases in preferences
    - Can be visualized with diverging colormap (blue/purple)
    - Handles both raw pairwise data and aggregated data (with 'count' column)

    Example
    -------
    >>> pairwise_df = create_pairwise_df(trial_df)
    >>> dsm = compute_directional_dsm(pairwise_df)
    >>> print(f"DSM range: [{dsm.min()}, {dsm.max()}]")
    >>> print(f"Is antisymmetric: {np.allclose(dsm, -dsm.T)}")
    """
    count_matrix = _build_count_matrix(pairwise_df, dtype=int)

    # Compute directional DSM as signed difference
    # DSM[i,j] = count(i>j) - count(j>i)
    dsm = count_matrix - count_matrix.T

    return dsm


def compute_normalized_dsm(pairwise_df: pd.DataFrame) -> np.ndarray:
    """
    Compute count-normalized directional DSM from pairwise data.

    Same exposure-confound correction as ``compute_normalized_rdm``: divide
    each cell by the total number of comparisons made for that pair, so the
    resulting score is the *fraction* of comparisons that went one way
    rather than a raw count.

    DSM_norm[i,j] = (count(i>j) - count(j>i)) / (count(i>j) + count(j>i))

    Values lie in [-1, +1]:
        +1  every comparison preferred i over j
         0  perfect tie (or pair never compared)
        -1  every comparison preferred j over i

    By construction ``|DSM_norm[i,j]| == RDM_norm[i,j]``, so DSM_norm and
    RDM_norm share the same dynamic range and can be plotted on a single
    diverging colorbar.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison data with 'better' and 'worse' columns.
        Optionally includes 'count' column if aggregated.

    Returns
    -------
    np.ndarray
        Count-normalized antisymmetric DSM matrix (n_stimuli x n_stimuli)
    """
    count_matrix = _build_count_matrix(pairwise_df, dtype=float)

    # Numerator: count(i>j) - count(j>i)  (signed difference)
    numerator = count_matrix - count_matrix.T
    # Denominator: count(i>j) + count(j>i)  (total comparisons for the pair)
    denominator = count_matrix + count_matrix.T
    # Pairs never compared map to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        dsm_norm = np.where(denominator > 0, numerator / denominator, 0.0)

    return dsm_norm


def aggregate_pairwise_counts(pairwise_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate raw pairwise comparisons into counts per (better, worse) pair.

    Parameters
    ----------
    pairwise_dfs : list of pd.DataFrame
        List of raw pairwise DataFrames (each with columns 'better', 'worse').

    Returns
    -------
    pd.DataFrame
        Aggregated table with columns: better, worse, count

    Notes
    -----
    - This is analysis-specific (behavioral pairwise preferences), so it
      lives in chess-behavioral rather than common to avoid over-generalizing.
    """
    if len(pairwise_dfs) == 0:
        return pd.DataFrame(columns=["better", "worse", "count"])  # empty

    return (
        pd.concat(pairwise_dfs, ignore_index=True)
        .groupby(["better", "worse"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["better", "worse"]).reset_index(drop=True)
    )


def correlate_with_all_models(
    behavioral_rdm: np.ndarray,
    category_df: pd.DataFrame,
    model_columns: List[str] = ["check", "visual", "strategy"],
) -> Tuple[List[Tuple[str, float, float, float, float]], List[np.ndarray]]:
    """
    Correlate behavioral RDM with multiple model RDMs.

    Delegates model creation and correlation to common.rsa_utils.correlate_rdm_with_models
    to ensure a single implementation is shared across analyses (DRY).

    Parameters
    ----------
    behavioral_rdm : np.ndarray
        Behavioral RDM matrix (n_stimuli × n_stimuli)
    category_df : pd.DataFrame
        DataFrame with 'stim_id' and model columns (e.g., 'check', 'visual', 'strategy')
    model_columns : list of str, default=['check', 'visual', 'strategy']
        Column names to use as models (treated as categorical)

    Returns
    -------
    results : list of tuple
        List of (column_name, r, p, ci_lower, ci_upper) for each model
    model_rdms : list of np.ndarray
        List of model RDM matrices (for plotting), in the same order as results
    """
    # Build features dictionary from the selected columns
    features = {col: category_df[col].values for col in model_columns}

    # Use common helper (categorical models)
    results, model_rdms_dict = correlate_rdm_with_models(
        rdm=behavioral_rdm,
        model_features=features,
        categorical_features=model_columns,
        continuous_features=[],
    )

    # Preserve order of model_columns for model RDMs list
    model_rdms = [model_rdms_dict[col] for col in model_columns]

    return results, model_rdms


def normalize_matrix_by_frequency(pairwise_df, matrix):
    """
    Normalize a dissimilarity matrix by total comparison count per pair.

    Each cell is divided by the total number of comparisons for that pair,
    removing the exposure confound where pairs compared more often accumulate
    higher raw dissimilarity values.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison data with columns 'better', 'worse', and optionally
        'count'. If 'count' is absent, each row is treated as a single comparison.
    matrix : np.ndarray
        Square dissimilarity matrix (n_stimuli x n_stimuli) with raw count-based
        values. Stimulus IDs are assumed 1-indexed (matrix[0,0] corresponds to
        stimulus 1 vs stimulus 1).

    Returns
    -------
    np.ndarray
        Normalized matrix where each cell is divided by the total number of
        comparisons for that pair. Values range from 0 (tied) to 1 (perfectly
        consistent). Pairs with zero comparisons are set to 0.
    """
    n = matrix.shape[0]
    count_mat = _build_count_matrix(pairwise_df, dtype=float, n_stimuli=n)
    total = count_mat + count_mat.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(total > 0, matrix / total, 0.0)


__all__ = [
    'create_pairwise_df',
    'compute_symmetric_rdm',
    'compute_normalized_rdm',
    'compute_directional_dsm',
    'compute_normalized_dsm',
    'aggregate_pairwise_counts',
    'correlate_with_all_models',
    'normalize_matrix_by_frequency',
]
