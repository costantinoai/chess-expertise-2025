#!/usr/bin/env python3
"""
Script utilities for standardized setup across analysis, table, and plotting scripts.

This module centralizes the repeated boilerplate found in 81_/82_ (tables) and
91_/92_/93_ (plotting) scripts:
 - Add repo root to sys.path for 'common' imports
 - Locate the latest results directory matching a pattern
 - Set up logging in the results directory
 - Create standard output subdirectories (e.g., tables/, figures/)

Strict behavior: no silent fallbacks. If require_results=True and a results
directory is not found, an informative exception is raised.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

from .logging_utils import setup_analysis_in_dir, setup_analysis
from .io_utils import find_latest_results_directory


def setup_script(
    script_file: str,
    results_pattern: str,
    output_subdirs: Optional[List[str]] = None,
    log_name: Optional[str] = None,
    require_results: bool = True,
    verbose: bool = True,
) -> Tuple[Path, logging.Logger, Dict[str, Path]]:
    """
    Standard setup for scripts that READ from existing results.

    Parameters
    ----------
    script_file : str
        Pass __file__ from the calling script.
    results_pattern : str
        Exact results directory name under 'results/', e.g., 'behavioral_rsa'.
    output_subdirs : list of str, optional
        Subdirectories to create in the results directory (e.g., ['tables'], ['figures']).
    log_name : str, optional
        Name of the log file to create inside the results directory. Defaults to '<script>.log'.
    require_results : bool, default=True
        If True, raise if the results directory is not found.
    verbose : bool, default=True
        Whether to print the resolved results directory.

    Returns
    -------
    results_dir : Path
        The resolved results directory.
    logger : logging.Logger
        Configured logger writing inside the results directory.
    output_dirs : dict[str, Path]
        Mapping of requested output subdirectories to their Paths (created if missing).
    """
    # Resolve paths
    script_path = Path(script_file).resolve()
    results_base = script_path.parent / 'results'

    results_dir = find_latest_results_directory(
        results_base,
        pattern=results_pattern,
        create_subdirs=output_subdirs or [],
        require_exists=require_results,
        verbose=verbose,
    )

    # Configure logging within the results directory
    if log_name is None:
        log_name = f"{script_path.stem}.log"

    extra = {"RESULTS_DIR": str(results_dir)}
    _, _, logger = setup_analysis_in_dir(
        results_dir=results_dir,
        script_file=str(script_path),
        extra_config=extra,
        suppress_warnings=True,
        log_name=log_name,
    )

    # Prepare output subdirectories mapping
    output_dirs: Dict[str, Path] = {}
    if output_subdirs:
        for sub in output_subdirs:
            out_p = results_dir / sub
            out_p.mkdir(parents=True, exist_ok=True)
            output_dirs[sub] = out_p

    return results_dir, logger, output_dirs


# Removed unused setup_analysis_script helper to avoid stale API surface


def setup_or_reuse_analysis_dir(
    script_file: str,
    analysis_name: str,
    output_subdirs: Optional[List[str]] = None,
    log_name: Optional[str] = None,
) -> Tuple[Path, logging.Logger, Dict[str, Path]]:
    """
    Create a timestamped results dir for an analysis, or reuse the latest one.

    - If a directory named '{analysis_name}' exists under 'results/', reuse it
      and initialize logging in-place.
    - Otherwise, create a new directory via setup_analysis (no timestamp).

    This is useful when multiple scripts contribute artifacts to a single
    unified results folder (e.g., RSA + SVM combined into 'mvpa_group').
    """
    script_path = Path(script_file).resolve()
    results_base = script_path.parent / 'results'

    target_dir = results_base / analysis_name
    if target_dir.exists():
        # Initialize logging in the existing directory
        if log_name is None:
            log_name = f"{script_path.stem}.log"
        extra = {"RESULTS_DIR": str(target_dir)}
        _, _, logger = setup_analysis_in_dir(
            results_dir=target_dir,
            script_file=str(script_path),
            extra_config=extra,
            suppress_warnings=True,
            log_name=log_name,
        )
        out_dirs: Dict[str, Path] = {}
        if output_subdirs:
            for sub in output_subdirs:
                p = target_dir / sub
                p.mkdir(parents=True, exist_ok=True)
                out_dirs[sub] = p
        return target_dir, logger, out_dirs
    else:
        # Create new
        from . import CONFIG
        config, new_dir, logger = setup_analysis(
            analysis_name=analysis_name,
            results_base=results_base,
            script_file=str(script_path),
            extra_config=CONFIG,
        )
        out_dirs: Dict[str, Path] = {}
        if output_subdirs:
            for sub in output_subdirs:
                p = new_dir / sub
                p.mkdir(parents=True, exist_ok=True)
                out_dirs[sub] = p
        return new_dir, logger, out_dirs


__all__ = [
    'setup_script',
    'setup_or_reuse_analysis_dir',
]
