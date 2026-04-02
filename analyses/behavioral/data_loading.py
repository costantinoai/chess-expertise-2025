"""
Data loading utilities for behavioral RSA analysis using BIDS format.

This module provides functions for loading trial data from BIDS-compliant
_events.tsv files in the func/ directories. All data is loaded from BIDS
ground truth files (participants.tsv, events.tsv, stimuli.tsv) rather than
legacy MATLAB or Excel files.

Key Functions
-------------
load_participant_trial_data : Load all trial data for a single participant
load_trial_data_from_events_tsv : Load data from a single BIDS events file

Note: Participant lists and stimulus metadata are centralized in
common.bids_utils. Use:
- common.bids_utils.get_participants_with_expertise()
- common.bids_utils.load_stimulus_metadata()

Notes for Academic Users
------------------------
- All functions use BIDS-compliant file structures and naming conventions
- Missing data is handled gracefully with warnings (no silent failures)
- Returned DataFrames have consistent column names for downstream analyses
- All file paths should point to BIDS directories, not legacy data formats
"""

import pandas as pd
import warnings
from pathlib import Path
from typing import List, Optional


def load_trial_data_from_events_tsv(events_file: Path) -> Optional[pd.DataFrame]:
    """
    Load trial data from a BIDS _events.tsv file.

    Parameters
    ----------
    events_file : Path
        Path to _events.tsv file

    Returns
    -------
    pd.DataFrame or None
        Trial data as DataFrame, or None if file contains no valid trials

    Notes
    -----
    - BIDS events.tsv files have columns: onset, duration, trial_type, stim_id, response, response_time, button_mapping
    - Returns None if file is missing or has no valid responses

    Example
    -------
    >>> trial_df = load_trial_data_from_events_tsv(Path('sub-01/func/sub-01_task-exp_run-1_events.tsv'))
    """
    events_file = Path(events_file)

    if not events_file.exists():
        warnings.warn(f"Events file not found: {events_file}")
        return None

    try:
        df = pd.read_csv(events_file, sep='\t')
    except Exception as e:
        warnings.warn(f"Failed to load {events_file}: {e}")
        return None

    if len(df) == 0:
        warnings.warn(f"Empty events file: {events_file}")
        return None

    # Check for valid responses (preference column should have non-n/a values)
    if 'preference' in df.columns:
        valid_prefs = df[df["preference"] != "n/a"]
        if len(valid_prefs) == 0:
            warnings.warn(f"No valid preferences in {events_file}")
            return None

    return df


def load_participant_trial_data(
    subject_id: str,
    is_expert: bool,
    bids_root: Path,
    required_columns: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Load and concatenate all experimental run data for a single participant from BIDS structure.

    This function searches for all _events.tsv files in the participant's func/
    directory for task-exp runs, and concatenates them into a single DataFrame.

    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'sub-01')
    is_expert : bool
        Whether participant is an expert (not used in loading, kept for compatibility)
    bids_root : Path
        Root BIDS directory containing subject folders
    required_columns : list of str, optional
        Column names to ensure are present (not enforced, for compatibility)

    Returns
    -------
    pd.DataFrame or None
        Concatenated trial data for all experimental runs, or None if no valid data

        Key columns in returned DataFrame:
        - stim_id: Stimulus ID (1-40)
        - preference: 'current_preferred', 'previous_preferred', or 'n/a' (GROUND TRUTH)
        - sub_id: Subject number
        - run: Run number
        - run_trial_n: Trial number within run
        - onset / stim_onset_real: Stimulus onset time

    Notes
    -----
    - Looks for _events.tsv files in {bids_root}/{subject_id}/func/
    - Only processes files matching pattern: sub-XX_task-exp_run-N_events.tsv
    - Skips runs with no valid preferences (preference != 'n/a')
    - Returns None if no valid trials found across all runs
    - IMPORTANT: Reads 'preference' column directly from BIDS events.tsv (GROUND TRUTH)
    - Does NOT recalculate preference from button codes - already done in BIDS conversion
    - Adds 'run' column extracted from filename
    - Renames 'onset' to 'stim_onset_real' for compatibility
    - Adds 'run_trial_n' column (trial number within run, 1-indexed)
    - Adds 'sub_id' column (subject number without 'sub-' prefix)

    Example
    -------
    >>> df = process_single_participant('sub-01', True, Path('data/BIDS'))
    >>> print(f"Loaded {len(df)} trials")
    """
    bids_root = Path(bids_root)

    # === Step 1: Locate participant's functional data directory ===
    # BIDS structure: <bids_root>/<subject_id>/func/
    participant_func_dir = bids_root / subject_id / "func"

    if not participant_func_dir.exists():
        warnings.warn(f"Participant func directory not found: {participant_func_dir}")
        return None

    # === Step 2: Find all experimental run event files ===
    # Pattern: sub-XX_task-exp_run-N_events.tsv
    # Each participant has multiple runs (typically 4-6), each with a separate events file
    events_files = sorted(participant_func_dir.glob(f"{subject_id}_task-exp_run-*_events.tsv"))

    if len(events_files) == 0:
        warnings.warn(f"No task-exp events.tsv files found for {subject_id}")
        return None

    # === Step 3: Initialize storage for concatenated trial data ===
    all_trials = pd.DataFrame()

    # === Step 4: Load and concatenate data from each run ===
    # Process each events file (one per run)
    for events_file in events_files:
        # Extract run number from filename: sub-XX_task-exp_run-N_events.tsv
        filename = events_file.stem
        parts = filename.split("_")
        run_part = [p for p in parts if p.startswith("run-")]

        if not run_part:
            warnings.warn(f"Could not parse run number from: {filename}")
            continue

        run_num = int(run_part[0].replace("run-", ""))

        # === Step 5: Load trial data from this run's events file ===
        run_df = load_trial_data_from_events_tsv(events_file)

        if run_df is None:
            # Skip this run if loading failed (warnings already issued)
            continue

        # === Step 6: Add metadata columns for this run ===
        # These columns identify which run each trial came from

        # Run number (extracted from filename)
        run_df['run'] = float(run_num)

        # Trial number within this run (1-indexed)
        run_df['run_trial_n'] = range(1, len(run_df) + 1)

        # Subject ID as numeric value (for compatibility with legacy code)
        sub_num = int(subject_id.replace("sub-", ""))
        run_df['sub_id'] = float(sub_num)

        # === Step 7: Rename BIDS columns to match analysis code expectations ===
        # BIDS uses 'onset', but analysis code expects 'stim_onset_real'
        if 'onset' in run_df.columns:
            run_df['stim_onset_real'] = run_df['onset']

        # Add expected onset column (use actual onset as placeholder)
        if 'stim_onset_expected' not in run_df.columns:
            run_df['stim_onset_expected'] = run_df['onset']

        # === Step 8: Keep 'preference' column from BIDS events file ===
        # IMPORTANT: The 'preference' column is the GROUND TRUTH from BIDS events.tsv
        # It has already been corrected for button mapping counterbalancing in
        # convert_mat_to_bids_events.py - we do NOT recalculate it from response codes
        #
        # Preference values: 'current_preferred', 'previous_preferred', 'n/a'
        # These are used directly in modules/rdm_utils.py::create_pairwise_df()
        #
        # The 'preference' column should already exist in run_df from load_trial_data_from_events_tsv()
        # No conversion needed - just keep it as-is

        # === Step 9: Concatenate this run's data with previous runs ===
        all_trials = pd.concat([all_trials, run_df], ignore_index=True)

    # === Step 10: Validate that we loaded valid data ===
    if len(all_trials) == 0:
        warnings.warn(f"No trials loaded for {subject_id}")
        return None

    # Check for valid preferences (at least one valid preference required)
    if "preference" in all_trials.columns:
        valid_preferences = all_trials[all_trials["preference"] != "n/a"]
        if len(valid_preferences) == 0:
            warnings.warn(f"No valid preferences across all runs for {subject_id}")
            return None
    else:
        warnings.warn(f"No 'preference' column found for {subject_id}")
        return None

    return all_trials


## Note on stimuli metadata
# Stimulus categories are provided via common.bids_utils.load_stimulus_metadata().
# Keeping this module focused on per-participant trial event loading avoids
# duplication and makes sharing easy.
