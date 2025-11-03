"""
Logging utilities for consistent logging across all analyses.

This module provides a centralized logging setup to ensure consistent log
formatting, file output, and console output across all scripts.

Usage
-----
>>> from common.logging_utils import setup_logging, setup_analysis
>>> logger = setup_logging(output_dir / "analysis.log")
>>> logger.info("Starting analysis...")

Or use the all-in-one setup:
>>> from common.logging_utils import setup_analysis
>>> config, output_dir, logger = setup_analysis("behavioral_rsa", Path("results"))
"""

import logging
import sys
import os
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from .io_utils import copy_script_to_results


def _get_log_level_from_env():
    """
    Get logging level from CHESS_LOG_LEVEL environment variable.

    Returns
    -------
    int
        logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR
        Defaults to logging.INFO if not set or invalid

    Examples
    --------
    export CHESS_LOG_LEVEL=DEBUG    # Verbose config dumps
    export CHESS_LOG_LEVEL=INFO     # Normal operation (default)
    export CHESS_LOG_LEVEL=WARNING  # Quiet mode
    """
    level_str = os.environ.get('CHESS_LOG_LEVEL', 'INFO').upper()
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    return level_map.get(level_str, logging.INFO)


def log_dataset_summary(logger: logging.Logger) -> None:
    """
    Log dynamic dataset summaries derived from BIDS ground truth.

    Logs participant counts (total/experts/novices) and number of stimuli boards.
    Safe to call in any analysis setup; guards all failures.
    """
    try:
        from .bids_utils import get_group_summary, load_stimulus_metadata
        summary = get_group_summary()
        logger.info(
            f"Participants: {summary['n_total']} total "
            f"({summary['n_expert']} experts, {summary['n_novice']} novices)"
        )
        try:
            stim_df = load_stimulus_metadata()
            n_stim = int(stim_df['stim_id'].nunique()) if 'stim_id' in stim_df.columns else len(stim_df)
            logger.info(f"Stimuli: {n_stim} boards")
        except Exception as e:
            logger.warning(f"Could not compute stimuli summary: {e}")
    except Exception as e:
        logger.warning(f"Could not compute participant summary: {e}")


def setup_logging(log_file=None, level=None, console=True):
    """
    Set up logging with consistent formatting for file and console output.

    Parameters
    ----------
    log_file : str or Path, optional
        Path to log file. If None, only console logging is used.
    level : int, optional
        Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
        If None, reads from CHESS_LOG_LEVEL environment variable (default: INFO)
    console : bool, default=True
        Whether to also log to console (in addition to file)

    Returns
    -------
    logging.Logger
        Configured logger instance

    Notes
    -----
    - Log format: "YYYY-MM-DD HH:MM:SS - LEVEL - message"
    - Creates parent directories for log_file if they don't exist
    - If log_file is provided, logs to both file and console (unless console=False)
    - Logger name is set to 'chess_analysis'
    - Respects CHESS_LOG_LEVEL environment variable (DEBUG, INFO, WARNING, ERROR)

    Environment Variables
    ---------------------
    CHESS_LOG_LEVEL : str
        Set logging verbosity: DEBUG (verbose), INFO (default), WARNING, ERROR

    Example
    -------
    >>> from pathlib import Path
    >>> from common.logging_utils import setup_logging
    >>>
    >>> output_dir = Path("results/20250117-162630_my_analysis")
    >>> output_dir.mkdir(parents=True, exist_ok=True)
    >>> logger = setup_logging(output_dir / "analysis.log")
    >>> logger.info("Analysis started")
    >>> logger.debug("Debug information")  # Only shown if CHESS_LOG_LEVEL=DEBUG
    >>> logger.warning("Warning message")
    >>> logger.error("Error message")
    """
    # Use environment variable if level not explicitly provided
    if level is None:
        level = _get_log_level_from_env()

    # Create logger
    logger = logging.getLogger('chess_analysis')
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add file handler if log_file is provided
    if log_file is not None:
        log_file = Path(log_file)
        # Create parent directories if they don't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    return logger


def log_script_start(logger, script_path, config_dict=None):
    """
    Log the start of a script with configuration information.

    Logs concise header at INFO level. Full configuration is logged at DEBUG level.
    Use CHESS_LOG_LEVEL=DEBUG to see complete config dumps.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    script_path : str or Path
        Path to the script being run
    config_dict : dict, optional
        Dictionary of configuration parameters to log

    Returns
    -------
    None

    Example
    -------
    >>> logger = setup_logging(output_dir / "analysis.log")
    >>> config = {'RANDOM_SEED': 42, 'N_FOLDS': 5}
    >>> log_script_start(logger, __file__, config)
    """
    # Concise header at INFO level
    logger.info("=" * 80)
    logger.info(f"{Path(script_path).name}")
    logger.info("=" * 80)

    # Full configuration dump at DEBUG level only
    if config_dict is not None:
        # Extract key parameters for INFO-level summary
        seed = config_dict.get('RANDOM_SEED', 'N/A')
        logger.debug(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.debug(f"Random seed: {seed}")
        logger.debug("-" * 80)
        logger.debug("Full configuration:")
        for key, value in config_dict.items():
            logger.debug(f"  {key}: {value}")
        logger.debug("-" * 80)


def log_script_end(logger):
    """
    Log the end of a script.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance

    Returns
    -------
    None

    Example
    -------
    >>> logger = setup_logging(output_dir / "analysis.log")
    >>> # ... run analysis ...
    >>> log_script_end(logger)
    """
    logger.info("=" * 80)
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def setup_analysis(
    analysis_name: str,
    results_base: Path,
    script_file: str,
    extra_config: dict = None,
    suppress_warnings: bool = True
) -> tuple:
    """
    All-in-one setup for analysis scripts.

    This function handles all common initialization tasks:
    1. Creates or reuses a stable output directory results/<analysis_name>
    2. Sets up logging (file + console)
    3. Suppresses warnings
    4. Sets random seed
    5. Copies script to output directory
    6. Builds complete configuration dictionary
    7. Logs script start with configuration

    Parameters
    ----------
    analysis_name : str
        Name of the analysis (e.g., "behavioral_rsa", "mvpa_checkmate")
        Used for directory naming: {timestamp}_{analysis_name}
    results_base : Path
        Base directory for results (e.g., Path("results"))
        Will be created if it doesn't exist
    script_file : str
        Path to the script being run (use __file__)
    extra_config : dict, optional
        Additional configuration parameters specific to this analysis
        Will be merged with standard constants
    suppress_warnings : bool, default=True
        Whether to suppress FutureWarning and UserWarning

    Returns
    -------
    config : dict
        Complete configuration dictionary (all constants + extra_config)
    output_dir : Path
        Timestamped output directory path
    logger : logging.Logger
        Configured logger instance

    Notes
    -----
    - Creates directory: {results_base}/{analysis_name}/ (no timestamp)
    - Sets random seed from RANDOM_SEED constant
    - Logs all configuration parameters from constants.py
    - Suppresses warnings by default (FutureWarning, UserWarning)
    - Copies the script file to output directory for reproducibility

    Example
    -------
    >>> from pathlib import Path
    >>> from common.logging_utils import setup_analysis
    >>> from common import RANDOM_SEED
    >>>
    >>> # Minimal usage
    >>> config, output_dir, logger = setup_analysis(
    ...     analysis_name="behavioral_rsa",
    ...     results_base=Path("results"),
    ...     script_file=__file__
    ... )
    >>> logger.info("Starting analysis...")
    >>>
    >>> # With extra configuration
    >>> extra = {'MODEL_COLUMNS': ['check', 'visual', 'strategy']}
    >>> config, output_dir, logger = setup_analysis(
    ...     analysis_name="behavioral_rsa",
    ...     results_base=Path("results"),
    ...     script_file=__file__,
    ...     extra_config=extra
    ... )
    """
    # Import CONFIG dictionary from constants
    from .constants import CONFIG

    # 1. Create timestamped output directory
    results_base = Path(results_base)
    results_base.mkdir(parents=True, exist_ok=True)

    # Stable directory without timestamp; overwrite on reruns is allowed
    output_dir = results_base / analysis_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Set up logging
    logger = setup_logging(output_dir / "analysis.log")

    # 3. Suppress warnings if requested
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    # 4. Set random seed
    np.random.seed(CONFIG['RANDOM_SEED'])

    # 5. Copy script to output directory (centralized helper)
    script_path = Path(script_file)
    copy_script_to_results(script_path, output_dir, logger)

    # 6. Build configuration dictionary from CONFIG
    # Start with a copy of the global CONFIG dictionary
    config = {}

    # Convert Path objects to strings for better logging/serialization
    for key, value in CONFIG.items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value

    # Add output directory
    config['OUTPUT_DIR'] = str(output_dir)

    # Merge with extra configuration (analysis-specific parameters)
    if extra_config is not None:
        config.update(extra_config)

    # 7. Log script start and dataset summary
    log_script_start(logger, script_file, config)
    log_dataset_summary(logger)

    return config, output_dir, logger


def setup_analysis_in_dir(
    results_dir: Path,
    script_file: str,
    extra_config: dict = None,
    suppress_warnings: bool = True,
    log_name: str = "analysis.log",
) -> tuple:
    """
    Set up logging and config for an analysis that writes into an existing directory.

    Does not create a timestamped folder; instead reuses the provided results_dir.
    Mirrors setup_analysis steps (2–7) to keep initialization centralized (DRY).

    Parameters
    ----------
    results_dir : Path
        Existing results directory to write logs and artifacts to
    script_file : str
        Path to the script being run (use __file__)
    extra_config : dict, optional
        Additional configuration parameters specific to this analysis
    suppress_warnings : bool, default=True
        Whether to suppress FutureWarning and UserWarning
    log_name : str, default="analysis.log"
        Log file name to create inside results_dir

    Returns
    -------
    config : dict
        Complete configuration dictionary (all constants + extra_config)
    output_dir : Path
        The same results_dir provided (for API symmetry)
    logger : logging.Logger
        Configured logger instance
    """
    from .constants import CONFIG

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logger = setup_logging(results_dir / log_name)

    # Warnings
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    # Seed
    np.random.seed(CONFIG['RANDOM_SEED'])

    # Copy script
    script_path = Path(script_file)
    copy_script_to_results(script_path, results_dir, logger)

    # Build config
    config = {}
    for key, value in CONFIG.items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    config['OUTPUT_DIR'] = str(results_dir)
    if extra_config is not None:
        config.update(extra_config)

    # Log start and dataset summary
    log_script_start(logger, script_file, config)
    log_dataset_summary(logger)

    return config, results_dir, logger


__all__ = [
    'setup_logging',
    'log_script_start',
    'log_script_end',
    'setup_analysis',
    'setup_analysis_in_dir',
    'log_dataset_summary',
]
