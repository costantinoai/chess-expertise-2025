#!/usr/bin/env python3
"""
Script utilities for standardized setup across analysis, table, and plotting
scripts.

Everything here now resolves paths against the unified results/ tree at the
repo root (``results/<analysis>/{data,tables,figures}/``). Scripts inside
``chess-<name>/`` automatically map to ``results/<name>/``; scripts inside
``chess-supplementary/<name>/`` map to ``results/supplementary/<name>/``.
The log directory for each script is placed under
``results/<analysis>/logs/<analysis_name>/`` so the data/tables/figures
buckets stay clean of log noise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logging_utils import setup_analysis, setup_analysis_in_dir


# ---------------------------------------------------------------------------
# Analysis-folder resolution
# ---------------------------------------------------------------------------
def _resolve_analysis_folder(script_file: str) -> str:
    """Derive the unified-tree analysis folder for a script path.

    Rules (deterministic, based on directory names):
      - ``chess-<name>/*.py`` -> ``<name>``
      - ``chess-supplementary/<name>/*.py`` -> ``supplementary/<name>``
      - otherwise -> fall back to the script's immediate parent folder name.
    """
    script_path = Path(script_file).resolve()
    parent = script_path.parent
    grandparent = parent.parent

    if grandparent.name == "chess-supplementary":
        return f"supplementary/{parent.name}"
    if parent.name.startswith("chess-"):
        return parent.name[len("chess-"):]
    return parent.name


def _results_root() -> Path:
    """Return the repo-level unified results/ root from CONFIG."""
    from .constants import CONFIG

    return Path(CONFIG["RESULTS_ROOT"])


def _analysis_dirs(
    analysis_folder: str,
) -> Tuple[Path, Path, Path]:
    """Return (data_dir, tables_dir, figures_dir) for an analysis folder.

    All three directories are created on disk if missing so callers can
    write into them directly without guarding on existence.
    """
    root = _results_root() / analysis_folder
    data_dir = root / "data"
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    for p in (data_dir, tables_dir, figures_dir):
        p.mkdir(parents=True, exist_ok=True)
    return data_dir, tables_dir, figures_dir


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def setup_script(
    script_file: str,
    results_pattern: str,
    output_subdirs: Optional[List[str]] = None,
    log_name: Optional[str] = None,
    require_results: bool = True,
    verbose: bool = True,
) -> Tuple[Path, logging.Logger, Dict[str, Path]]:
    """Standard setup for table/plot scripts that READ from the unified tree.

    Parameters
    ----------
    script_file : str
        Pass ``__file__`` from the calling script.
    results_pattern : str
        Historical name for the analysis output bucket (e.g.
        ``'behavioral_rsa'``). Retained for API compatibility but no longer
        used for path construction -- the unified tree routes to
        ``results/<analysis>/data/`` directly.
    output_subdirs : list of str, optional
        Subdirectories to expose to the caller. Each entry must be one of
        ``'data'``, ``'tables'``, or ``'figures'``. Values are resolved to
        the unified tree (``results/<analysis>/<entry>/``) rather than
        subfolders of a per-script results dir.
    log_name : str, optional
        Log file name. Defaults to ``<script_stem>.log``.
    require_results : bool, default True
        If True and the data directory does not yet exist, raise. The
        helper creates ``data/``, ``tables/``, ``figures/`` up front, so
        this flag only fails when the caller explicitly expects pre-existing
        data produced by an earlier stage (e.g. an 81_table reading an 02_
        group's output). The check is therefore on the *presence of data*,
        not the bare directory.
    verbose : bool, default True
        If True, print the resolved data directory.

    Returns
    -------
    data_dir : Path
        The ``results/<analysis>/data/`` directory (used for reading inputs).
    logger : logging.Logger
        Configured logger writing into the analysis log directory.
    output_dirs : dict[str, Path]
        Mapping from each requested subdir name in ``output_subdirs`` to its
        resolved path under ``results/<analysis>/``. The keys are exactly
        what the caller requested (``'data'``, ``'tables'``, or
        ``'figures'``), so existing ``output_dirs['tables']`` call sites
        continue to work unchanged.
    """
    analysis_folder = _resolve_analysis_folder(script_file)
    data_dir, tables_dir, figures_dir = _analysis_dirs(analysis_folder)

    # Strict behaviour: if the caller expects pre-existing data and the
    # data folder is empty, surface a clear error.
    if require_results and not any(data_dir.iterdir()):
        raise FileNotFoundError(
            f"No data found under {data_dir}. Run the upstream 0X_*_group "
            f"script for '{analysis_folder}' first."
        )

    script_path = Path(script_file).resolve()
    logs_dir = _results_root() / analysis_folder / "logs" / script_path.stem
    logs_dir.mkdir(parents=True, exist_ok=True)

    if log_name is None:
        log_name = f"{script_path.stem}.log"

    _, _, logger = setup_analysis_in_dir(
        results_dir=logs_dir,
        script_file=str(script_path),
        extra_config={
            "RESULTS_DIR": str(data_dir),
            "DATA_DIR": str(data_dir),
            "TABLES_DIR": str(tables_dir),
            "FIGURES_DIR": str(figures_dir),
            "ANALYSIS_FOLDER": analysis_folder,
        },
        suppress_warnings=True,
        log_name=log_name,
    )

    if verbose:
        logger.info(f"Analysis folder: {analysis_folder}")
        logger.info(f"Data dir:        {data_dir}")
        logger.info(f"Tables dir:      {tables_dir}")
        logger.info(f"Figures dir:     {figures_dir}")

    output_dirs: Dict[str, Path] = {}
    kind_map = {"data": data_dir, "tables": tables_dir, "figures": figures_dir}
    for sub in output_subdirs or []:
        if sub not in kind_map:
            raise ValueError(
                f"setup_script output_subdirs must be one of {sorted(kind_map)}, "
                f"got {sub!r}"
            )
        output_dirs[sub] = kind_map[sub]

    return data_dir, logger, output_dirs


def setup_or_reuse_analysis_dir(
    script_file: str,
    analysis_name: str,
    output_subdirs: Optional[List[str]] = None,
    log_name: Optional[str] = None,
) -> Tuple[Path, logging.Logger, Dict[str, Path]]:
    """Compute-script entry point: returns the unified data/ path.

    Parameters
    ----------
    script_file : str
        Pass ``__file__`` from the calling script.
    analysis_name : str
        Historical analysis-name label used for the log subdirectory
        (e.g. ``'behavioral_rsa'``). Does not affect data output location.
    output_subdirs : list of str, optional
        Additional buckets from ``{'data', 'tables', 'figures'}`` to expose
        to the caller (see ``setup_script``).
    log_name : str, optional
        Log file name. Defaults to ``<analysis_name>.log``.

    Returns
    -------
    data_dir : Path
        The ``results/<analysis>/data/`` directory where the script should
        write its primary outputs.
    logger : logging.Logger
        Configured logger writing into the analysis log directory.
    out_dirs : dict[str, Path]
        Same semantics as ``setup_script``.
    """
    script_path = Path(script_file).resolve()
    analysis_folder = _resolve_analysis_folder(script_file)
    data_dir, tables_dir, figures_dir = _analysis_dirs(analysis_folder)

    logs_dir = _results_root() / analysis_folder / "logs" / analysis_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    if log_name is None:
        log_name = f"{analysis_name}.log"

    _, _, logger = setup_analysis_in_dir(
        results_dir=logs_dir,
        script_file=str(script_path),
        extra_config={
            "RESULTS_DIR": str(data_dir),
            "DATA_DIR": str(data_dir),
            "TABLES_DIR": str(tables_dir),
            "FIGURES_DIR": str(figures_dir),
            "ANALYSIS_FOLDER": analysis_folder,
        },
        suppress_warnings=True,
        log_name=log_name,
    )

    out_dirs: Dict[str, Path] = {}
    kind_map = {"data": data_dir, "tables": tables_dir, "figures": figures_dir}
    for sub in output_subdirs or []:
        if sub not in kind_map:
            raise ValueError(
                f"setup_or_reuse_analysis_dir output_subdirs must be one of "
                f"{sorted(kind_map)}, got {sub!r}"
            )
        out_dirs[sub] = kind_map[sub]

    return data_dir, logger, out_dirs


__all__ = [
    "setup_script",
    "setup_or_reuse_analysis_dir",
]
