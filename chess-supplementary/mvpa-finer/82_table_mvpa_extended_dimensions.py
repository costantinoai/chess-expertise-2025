#!/usr/bin/env python3
"""MVPA Extended Dimensions — Manuscript Table (Finer-grained dimensions, 22 ROIs)"""
import sys
from pathlib import Path
import pandas as pd
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common import CONFIG
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.io_utils import find_latest_results_directory
from common.bids_utils import load_roi_metadata
from common.report_utils import save_table_with_manuscript_copy
from common.formatters import format_ci, format_p_cell, shorten_roi_name

# Find RSA and decoding results directories
try:
    rsa_dir = find_latest_results_directory(Path(__file__).parent / 'results', pattern='*_rsa', require_exists=True, verbose=True)
    dec_dir = find_latest_results_directory(Path(__file__).parent / 'results', pattern='*_decoding', require_exists=True, verbose=True)
except Exception as e:
    logger = None
    # If setup_analysis_in_dir below depends on rsa_dir, create a minimal logger via setup once we can
    print(f"[WARN] Skipping MVPA Extended Dimensions tables: {e}")
    sys.exit(0)

# Use RSA dir for logging/output
_, _, logger = setup_analysis_in_dir(rsa_dir, script_file=__file__,
                                      extra_config={"RESULTS_DIR": str(rsa_dir)},
                                      suppress_warnings=True, log_name='tables_mvpa_extended.log')
tables_dir = rsa_dir.parent / 'tables'
tables_dir.mkdir(exist_ok=True)

# Load data
logger.info("Loading MVPA RSA and decoding statistics...")
rsa_pkl = rsa_dir / 'mvpa_group_stats.pkl'
dec_pkl = dec_dir / 'mvpa_group_stats.pkl'
if not rsa_pkl.exists() or not dec_pkl.exists():
    print(f"[WARN] Skipping MVPA Extended Dimensions tables: missing results files {rsa_pkl} or {dec_pkl}")
    sys.exit(0)
with open(rsa_pkl, 'rb') as f:
    rsa_stats = pickle.load(f)
with open(dec_pkl, 'rb') as f:
    dec_stats = pickle.load(f)

# Load ROI metadata (22 bilateral regions)
roi_dir = CONFIG['ROI_GLASSER_22']
if not Path(roi_dir).exists():
    print(f"[WARN] Skipping MVPA Extended Dimensions tables: ROI directory not found: {roi_dir}")
    sys.exit(0)
roi_info = load_roi_metadata(roi_dir)
logger.info(f"Loaded {len(roi_info)} ROIs")

# Define finer-grained dimensions (checkmate boards only)
finer_dims = ['strategy_half', 'check_n_half', 'legal_moves_half', 'motif_half', 'total_pieces_half']
dim_labels = ['Strategy', 'Moves to Checkmate', 'Legal Moves', 'Motifs', 'Total Pieces']

logger.info("Building data matrices for RSA and decoding (single tables)...")
from common.tables import generate_styled_table, build_c_only_colspec

def build_df_for(kind: str) -> pd.DataFrame:
    rows = []
    source = rsa_stats if kind == 'rsa' else dec_stats
    # Top-level keys differ between RSA and decoding pickles
    key = 'rsa_corr' if kind == 'rsa' else 'svm'
    for _, roi_row in roi_info.iterrows():
        roi_name = shorten_roi_name(roi_row['pretty_name'])
        row_out = {'ROI': roi_name}
        for dim, lab in zip(finer_dims, dim_labels):
            if key in source and dim in source[key]:
                welch = source[key][dim]['welch_expert_vs_novice']
                roi_data = welch[welch['ROI_Label'] == roi_row['roi_id']]
                if len(roi_data) > 0:
                    r = roi_data.iloc[0]
                    delta = float(r['mean_diff'])
                    ci_l, ci_h = float(r['ci95_low']), float(r['ci95_high'])
                    p = r['p_val_fdr']
                    # Put metric first for short headers under multicolumn: Δ, 95% CI, pFDR
                    row_out[f'Δ_{lab}'] = delta
                    row_out[f'95% CI_{lab}'] = format_ci(ci_l, ci_h, precision=3, latex=False, use_numrange=True)
                    row_out[f'pFDR_{lab}'] = format_p_cell(p)
                else:
                    row_out[f'Δ_{lab}'] = float('nan')
                    row_out[f'95% CI_{lab}'] = '{--}{--}'
                    row_out[f'pFDR_{lab}'] = '--'
            else:
                row_out[f'Δ_{lab}'] = float('nan')
                row_out[f'95% CI_{lab}'] = '{--}{--}'
                row_out[f'pFDR_{lab}'] = '--'
        rows.append(row_out)
    return pd.DataFrame(rows)

# Build full matrices
df_rsa_all = build_df_for('rsa')
df_dec_all = build_df_for('dec')

multicol_all = {lab: [f'Δ_{lab}', f'95% CI_{lab}', f'pFDR_{lab}'] for lab in dim_labels}

# Render RSA and Decoding tables as one file with two tables
rsa_tex_path = generate_styled_table(
    df=df_rsa_all,
    output_path=tables_dir / 'mvpa_extended_dimensions_rsa.tex',
    caption='RSA — expert–novice difference across five finer regressors.',
    label='supptab:mvpa_ext_rsa',
    multicolumn_headers=multicol_all,
    column_format=build_c_only_colspec(df_rsa_all, multicolumn_headers=multicol_all),
    logger=logger,
    manuscript_name=None,
)

dec_tex_path = generate_styled_table(
    df=df_dec_all,
    output_path=tables_dir / 'mvpa_extended_dimensions_decoding.tex',
    caption='Decoding — expert–novice difference across five finer regressors.',
    label='supptab:mvpa_ext_dec',
    multicolumn_headers=multicol_all,
    column_format=build_c_only_colspec(df_dec_all, multicolumn_headers=multicol_all),
    logger=logger,
    manuscript_name=None,
)

# Combine into the single manuscript file (backward-compatible)
combined_tex = rsa_tex_path.read_text() + "\n\n" + dec_tex_path.read_text()
save_table_with_manuscript_copy(
    combined_tex,
    tables_dir / 'mvpa_extended_dimensions.tex',
    manuscript_name='mvpa_extended_dimensions.tex',
    logger=logger
)

# Remove intermediate single-table files to keep only the final combined file
try:
    (tables_dir / 'mvpa_extended_dimensions_rsa.tex').unlink(missing_ok=True)
    (tables_dir / 'mvpa_extended_dimensions_decoding.tex').unlink(missing_ok=True)
    logger.info("Removed intermediate RSA/Decoding .tex files (kept combined only)")
except Exception:
    pass

logger.info("="*80)
logger.info("MVPA Extended Dimensions Tables Complete (single file with two tables)")
logger.info("="*80)
log_script_end(logger)
