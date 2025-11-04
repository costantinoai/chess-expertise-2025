#!/usr/bin/env python3
"""
Stimuli Table — LaTeX Table Generation
=======================================

This script generates a publication-ready LaTeX table listing all stimulus boards
with their FEN (Forsyth-Edwards Notation) strings, providing a complete reference
for the experimental stimuli used in the study.

The table maps stimulus IDs (S01, S02, ..., S40) to their corresponding FEN
notation, which uniquely defines each chess board position.

Inputs
------
- stimuli.tsv: Main stimuli metadata file from CONFIG['STIMULI_FILE']
  Contains columns: stim_id, fen, and other metadata

Outputs
-------
- tables/stimuli.tex: LaTeX table with stimulus ID → FEN mappings
- Manuscript copy: stimuli.tex in final_results/tables/

Usage
-----
python chess-supplementary/dataset-viz/81_table_stimuli.py

Note: This script doesn't require analysis results - it reads directly from the
stimuli.tsv file in the manuscript data folder.
"""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.tables import generate_styled_table
from common import CONFIG

# ============================================================================
# Configuration & Setup
# ============================================================================

# Create a results directory for this script
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results' / 'stimuli_table'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
tables_dir = RESULTS_DIR / 'tables'
tables_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
extra = {"RESULTS_DIR": str(RESULTS_DIR)}
config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config=extra,
    suppress_warnings=True,
    log_name='table_stimuli.log',
)

# ============================================================================
# Load Stimuli Data
# ============================================================================

logger.info("Loading stimuli data...")

stimuli_file = Path(CONFIG['STIMULI_FILE'])
if not stimuli_file.exists():
    logger.error(f"Stimuli file not found: {stimuli_file}")
    raise FileNotFoundError(f"Could not find stimuli.tsv at {stimuli_file}")

df_stimuli = pd.read_csv(stimuli_file, sep='\t')
logger.info(f"Loaded {len(df_stimuli)} stimuli")

# ============================================================================
# Build LaTeX Table (centralized generator)
# ============================================================================

logger.info("Building stimuli table...")

from pandas import DataFrame
df_stimuli = df_stimuli.sort_values('stim_id').reset_index(drop=True)
rows = []
for _, row in df_stimuli.iterrows():
    stim_label = f"S{int(row['stim_id']):02d}"
    fen_tex = f"\\texttt{{{row['fen']}}}"
    rows.append({'Stimulus ID': stim_label, 'FEN Notation': fen_tex})
df_out = DataFrame(rows)

latex_path = tables_dir / 'stimuli.tex'
generate_styled_table(
    df=df_out,
    output_path=latex_path,
    caption=('Complete listing of all 40 experimental stimuli with their '
             'Forsyth-Edwards Notation (FEN) strings. Each stimulus ID (S01--S40) '
             'corresponds to a unique chess board position used in the study.'),
    label='tab:stimuli',
    column_format='ll',
    logger=logger,
    manuscript_name='stimuli.tex',
)

# Also save a simple CSV for reference
csv_data = []
for _, row in df_stimuli.iterrows():
    stim_id = int(row['stim_id'])
    csv_data.append({
        'Stimulus_ID': f'S{stim_id:02d}',
        'FEN': row['fen']
    })

df_csv = pd.DataFrame(csv_data)
csv_path = tables_dir / 'stimuli.csv'
df_csv.to_csv(csv_path, index=False)
logger.info(f"CSV table saved to: {csv_path}")

# ============================================================================
# Finish
# ============================================================================

log_script_end(logger)
