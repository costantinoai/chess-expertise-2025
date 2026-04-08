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


def _resolve_unified_analysis_folder(script_file: str) -> str | None:
    """Map a script path to its unified-tree analysis folder name.

    Returns e.g. ``'behavioral'`` for ``chess-behavioral/foo.py`` or
    ``'supplementary/eyetracking'`` for
    ``chess-supplementary/eyetracking/foo.py``. Returns ``None`` for
    scripts outside the chess-*/ or chess-supplementary/ layout.
    """
    script_path = Path(script_file).resolve()
    parent = script_path.parent
    grandparent = parent.parent
    if grandparent.name == "chess-supplementary":
        return f"supplementary/{parent.name}"
    if parent.name.startswith("chess-"):
        return parent.name[len("chess-"):]
    return None


def setup_analysis(
    analysis_name: str,
    results_base: Path,
    script_file: str,
    extra_config: dict = None,
    suppress_warnings: bool = True
) -> tuple:
    """
    All-in-one setup for analysis scripts.

    Unified-tree behaviour
    ----------------------
    If ``script_file`` lives inside a ``chess-<name>/`` or
    ``chess-supplementary/<name>/`` folder (i.e. one of the project's
    analysis directories), this helper ignores ``results_base`` and
    instead routes all outputs into the canonical

        <repo>/results/<analysis>/data/   (returned as output_dir)
        <repo>/results/<analysis>/logs/<analysis_name>/   (log file + script copy)

    so scripts that write their numerical outputs via ``output_dir / 'foo.csv'``
    land under ``results/<analysis>/data/`` automatically, while the log
    file and script-copy artefacts stay inside ``results/<analysis>/logs/``.

    For callers outside the chess-* layout (e.g. ``analyses/`` or
    ``common/``), the legacy behaviour is preserved: ``results_base`` is
    honoured literally and ``output_dir`` is ``results_base / analysis_name``.

    Parameters
    ----------
    analysis_name : str
        Name of the analysis stage (e.g., "behavioral_rsa",
        "01_behavioral_rsa_subject"). Used only as the log-subdirectory
        label under ``results/<analysis>/logs/``.
    results_base : Path
        Legacy parameter. Kept for backwards compatibility but ignored
        when the script resolves to a chess-* analysis folder. Most new
        call sites can pass any dummy path here.
    script_file : str
        Path to the script being run (use ``__file__``).
    extra_config : dict, optional
        Additional configuration parameters specific to this analysis;
        merged with standard constants.
    suppress_warnings : bool, default=True
        Whether to suppress FutureWarning and UserWarning.

    Returns
    -------
    config : dict
        Complete configuration dictionary (all constants + extra_config
        + DATA_DIR / TABLES_DIR / FIGURES_DIR / ANALYSIS_FOLDER when the
        unified-tree branch is taken).
    output_dir : Path
        For unified-tree scripts, the ``results/<analysis>/data/`` path
        (safe to use as the default sink for ``.csv``, ``.npy``, ``.pkl``
        etc.). For legacy scripts, ``results_base / analysis_name``.
    logger : logging.Logger
        Configured logger writing into the analysis log directory.
    """
    # Import CONFIG dictionary from constants
    from .constants import CONFIG

    # 0. Ensure the script's parent directory is on sys.path so that
    #    analysis-local ``modules/`` packages are importable without
    #    manual sys.path manipulation in every script.
    script_parent = str(Path(script_file).resolve().parent)
    if script_parent not in sys.path:
        sys.path.insert(0, script_parent)

    analysis_folder = _resolve_unified_analysis_folder(script_file)

    if analysis_folder is not None:
        # Unified-tree branch: ignore results_base; route everything into
        # <repo>/results/<analysis_folder>/{data,tables,figures,logs}/.
        results_root = Path(CONFIG["RESULTS_ROOT"])
        data_dir = results_root / analysis_folder / "data"
        tables_dir = results_root / analysis_folder / "tables"
        figures_dir = results_root / analysis_folder / "figures"
        logs_dir = results_root / analysis_folder / "logs" / analysis_name
        for p in (data_dir, tables_dir, figures_dir, logs_dir):
            p.mkdir(parents=True, exist_ok=True)
        output_dir = data_dir
        log_file = logs_dir / "analysis.log"
        script_copy_target = logs_dir
    else:
        # Legacy branch: honour results_base literally.
        results_base = Path(results_base)
        results_base.mkdir(parents=True, exist_ok=True)
        output_dir = results_base / analysis_name
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = tables_dir = figures_dir = output_dir
        logs_dir = output_dir
        log_file = output_dir / "analysis.log"
        script_copy_target = output_dir

    # 2. Set up logging
    logger = setup_logging(log_file)

    # 3. Suppress warnings if requested
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    # 4. Set random seed
    # Note: random seeding is handled by individual scripts via
    # np.random.default_rng(CONFIG['RANDOM_SEED']) for isolated reproducibility.

    # 5. Copy script to the log/setup directory (keeps data/tables/figures
    #    clean of script-copy noise when running under the unified tree).
    script_path = Path(script_file)
    copy_script_to_results(script_path, script_copy_target, logger)

    # 6. Build configuration dictionary from CONFIG
    # Start with a copy of the global CONFIG dictionary
    config = {}

    # Convert Path objects to strings for better logging/serialization
    for key, value in CONFIG.items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value

    # Add output/logs directories
    config['OUTPUT_DIR'] = str(output_dir)
    config['DATA_DIR'] = str(data_dir)
    config['TABLES_DIR'] = str(tables_dir)
    config['FIGURES_DIR'] = str(figures_dir)
    config['LOGS_DIR'] = str(logs_dir)
    if analysis_folder is not None:
        config['ANALYSIS_FOLDER'] = analysis_folder

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

    # Ensure the script's parent directory is on sys.path for local modules
    script_parent = str(Path(script_file).resolve().parent)
    if script_parent not in sys.path:
        sys.path.insert(0, script_parent)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logger = setup_logging(results_dir / log_name)

    # Warnings
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    # Seed
    # Note: random seeding is handled by individual scripts via
    # np.random.default_rng(CONFIG['RANDOM_SEED']) for isolated reproducibility.

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
