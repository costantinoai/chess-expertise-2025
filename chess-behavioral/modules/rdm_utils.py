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
import sys
from pathlib import Path
from typing import Tuple, List

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import shared RSA functions from common
from common.rsa_utils import (
    create_model_rdm,
    correlate_rdms,
    correlate_rdm_with_models,
)


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
    # Check if data is already aggregated (has 'count' column)
    if "count" in pairwise_df.columns:
        # Data is aggregated - use the count column
        counts_dict = {}
        for _, row in pairwise_df.iterrows():
            counts_dict[(row["better"], row["worse"])] = row["count"]
    else:
        # Data is not aggregated - count occurrences
        counts = pairwise_df.groupby(["better", "worse"]).size()
        counts_dict = counts.to_dict()

    # Get all unique stimulus IDs
    all_stimuli = sorted(set(pairwise_df["better"]).union(set(pairwise_df["worse"])))
    n_stimuli = int(max(all_stimuli))  # Assuming IDs from 1 to n

    # Initialize count matrix (0-indexed, so subtract 1 from stimulus IDs)
    count_matrix = np.zeros((n_stimuli, n_stimuli), dtype=int)

    # Fill count matrix from pairwise counts
    for (i, j), count in counts_dict.items():
        count_matrix[int(i) - 1, int(j) - 1] = count

    # Compute symmetric RDM as absolute difference
    # RDM[i,j] = |count(i>j) - count(j>i)|
    rdm = np.abs(count_matrix - count_matrix.T)

    return rdm


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
    # Check if data is already aggregated (has 'count' column)
    if "count" in pairwise_df.columns:
        # Data is aggregated - use the count column
        counts_dict = {}
        for _, row in pairwise_df.iterrows():
            counts_dict[(row["better"], row["worse"])] = row["count"]
    else:
        # Data is not aggregated - count occurrences
        counts = pairwise_df.groupby(["better", "worse"]).size()
        counts_dict = counts.to_dict()

    # Get all unique stimulus IDs
    all_stimuli = sorted(set(pairwise_df["better"]).union(set(pairwise_df["worse"])))
    n_stimuli = max(all_stimuli)

    # Initialize count matrix
    count_matrix = np.zeros((n_stimuli, n_stimuli), dtype=int)

    # Fill count matrix
    for (i, j), count in counts_dict.items():
        count_matrix[i - 1, j - 1] = count

    # Compute directional DSM as signed difference
    # DSM[i,j] = count(i>j) - count(j>i)
    dsm = count_matrix - count_matrix.T

    return dsm


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
