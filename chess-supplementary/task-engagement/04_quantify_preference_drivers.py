"""
Quantify Visual and Structural Drivers of Board Preference
============================================================

Extracts objective board-level features (from FEN) and image-level features
(from stimulus PNGs) and tests which features predict selection frequency
in experts vs novices using Spearman correlations and multiple regression.

METHODS
-------

Board-Level Features (from FEN via python-chess)
-------------------------------------------------
1. piece_count: Total number of pieces on the board
2. density: piece_count / 64 (proportion of occupied squares)
3. white_material / black_material: Summed piece values (P=1,N=3,B=3,R=5,Q=9)
4. material_imbalance: |white_material - black_material|
5. center_occupation: Number of pieces in the central 16 squares (c3-f6)
6. center_density: center_occupation / 16
7. pawn_count: Total pawns
8. officer_count: Total non-pawn, non-king pieces (N, B, R, Q)
9. piece_spread: Standard deviation of piece positions (spatial dispersion)
10. is_checkmate: Binary (checkmate board = 1)

Image-Level Features (from PNG pixel analysis)
-----------------------------------------------
11. image_entropy: Shannon entropy of grayscale pixel histogram (visual complexity)
12. edge_density: Proportion of edge pixels (Sobel filter, visual busyness)
13. luminance_mean: Mean pixel intensity (brightness)
14. luminance_std: Pixel intensity SD (contrast)

Statistical Analysis
--------------------
- Spearman correlations between each feature and preference frequency, per group
- FDR correction across features within each group
- Stepwise comparison: which features explain unique variance?

Outputs
-------
- feature_matrix.csv: All 40 boards x all features + preference per group
- feature_correlations_full.csv: Spearman r, p, pFDR for each feature x group
- figures/panels/preference_drivers_panel.pdf
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import chess

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.plotting import apply_nature_rc
from common.bids_utils import load_stimulus_metadata

# =============================================================================
# 1. SETUP
# =============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="novice_diagnostics",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)
FIGURES_DIR = out_dir / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
(FIGURES_DIR / "panels").mkdir(exist_ok=True)

STIMULI_DIR = CONFIG['EXTERNAL_DATA_ROOT'] / "stimuli"

# =============================================================================
# 2. LOAD DATA
# =============================================================================

logger.info("Loading data...")

pref_path = (CONFIG['REPO_ROOT'] / "chess-supplementary" / "task-engagement" /
             "results" / "novice_diagnostics" / "board_preference_group.csv")
pref_df = pd.read_csv(pref_path)
stim_df = load_stimulus_metadata(return_all=True)

# =============================================================================
# 3. EXTRACT BOARD-LEVEL FEATURES FROM FEN
# =============================================================================

logger.info("Extracting board-level features from FEN...")

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

# Central squares: c3-f6 (ranks 2-5, files 2-5 in 0-indexed)
CENTER_SQUARES = [chess.square(f, r) for f in range(2, 6) for r in range(2, 6)]

def extract_fen_features(fen):
    """Extract objective board features from a FEN string."""
    board = chess.Board(fen)
    pm = board.piece_map()

    white = [p for p in pm.values() if p.color == chess.WHITE]
    black = [p for p in pm.values() if p.color == chess.BLACK]

    piece_count = len(pm)
    density = piece_count / 64.0
    w_mat = sum(PIECE_VALUES[p.piece_type] for p in white)
    b_mat = sum(PIECE_VALUES[p.piece_type] for p in black)
    mat_imbalance = abs(w_mat - b_mat)
    total_material = w_mat + b_mat

    center_occ = sum(1 for sq in CENTER_SQUARES if sq in pm)
    center_density = center_occ / len(CENTER_SQUARES)

    pawn_count = sum(1 for p in pm.values() if p.piece_type == chess.PAWN)
    officer_count = sum(1 for p in pm.values()
                        if p.piece_type not in (chess.PAWN, chess.KING))

    # Spatial spread: SD of piece positions (file and rank)
    if piece_count > 1:
        files = [chess.square_file(sq) for sq in pm.keys()]
        ranks = [chess.square_rank(sq) for sq in pm.keys()]
        spread = np.std(files) + np.std(ranks)
    else:
        spread = 0.0

    return {
        'piece_count': piece_count,
        'density': density,
        'white_material': w_mat,
        'black_material': b_mat,
        'total_material': total_material,
        'material_imbalance': mat_imbalance,
        'center_occupation': center_occ,
        'center_density': center_density,
        'pawn_count': pawn_count,
        'officer_count': officer_count,
        'piece_spread': spread,
    }

fen_features = []
for _, row in stim_df.iterrows():
    feats = extract_fen_features(row['fen'])
    feats['stim_id'] = row['stim_id']
    check_col = 'check' if 'check' in stim_df.columns else 'check_status'
    feats['is_checkmate'] = int(row[check_col] == 'checkmate')
    fen_features.append(feats)

feat_df = pd.DataFrame(fen_features)
logger.info(f"Extracted {len(feat_df.columns) - 1} board features for {len(feat_df)} boards")

# =============================================================================
# 4. EXTRACT IMAGE-LEVEL FEATURES
# =============================================================================

logger.info("Extracting image-level features from PNGs...")

from scipy.ndimage import sobel

def extract_image_features(img_path):
    """Extract objective visual features from a board image."""
    img = plt.imread(str(img_path))

    # Convert to grayscale if RGB
    if img.ndim == 3:
        gray = np.mean(img[:, :, :3], axis=2)
    else:
        gray = img.copy()

    # Normalize to 0-255 range if float
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)

    # Shannon entropy of pixel histogram
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    entropy = -np.sum(probs * np.log2(probs))

    # Edge density (Sobel)
    edges_x = sobel(gray.astype(float), axis=0)
    edges_y = sobel(gray.astype(float), axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    edge_threshold = np.percentile(edge_mag, 90)
    edge_density = np.mean(edge_mag > edge_threshold)

    # Luminance stats
    luminance_mean = np.mean(gray)
    luminance_std = np.std(gray)

    return {
        'image_entropy': entropy,
        'edge_density': edge_density,
        'luminance_mean': luminance_mean,
        'luminance_std': luminance_std,
    }

img_features = []
for _, row in stim_df.iterrows():
    img_path = STIMULI_DIR / row['filename']
    if img_path.exists():
        feats = extract_image_features(img_path)
    else:
        logger.warning(f"Image not found: {img_path}")
        feats = {k: np.nan for k in ['image_entropy', 'edge_density',
                                       'luminance_mean', 'luminance_std']}
    feats['stim_id'] = row['stim_id']
    img_features.append(feats)

img_df = pd.DataFrame(img_features)
logger.info(f"Extracted {len(img_df.columns) - 1} image features")

# Merge all features
all_feat = feat_df.merge(img_df, on='stim_id')

# =============================================================================
# 5. ADD PREFERENCE FREQUENCIES PER GROUP
# =============================================================================

for group in ['expert', 'novice']:
    gdf = pref_df[pref_df['group'] == group]
    board_freqs = {}
    for _, row in gdf.iterrows():
        board_freqs[int(row['c_stim_id'])] = row['c_freq']
        board_freqs[int(row['nc_stim_id'])] = row['nc_freq']
    all_feat[f'pref_{group}'] = all_feat['stim_id'].map(board_freqs)

all_feat.to_csv(out_dir / "feature_matrix.csv", index=False)
logger.info(f"Saved feature matrix: {out_dir / 'feature_matrix.csv'}")

# =============================================================================
# 6. CORRELATIONS: FEATURES vs PREFERENCE
# =============================================================================

logger.info("\nFeature-preference Spearman correlations:")

feature_cols = [c for c in all_feat.columns
                if c not in ('stim_id', 'pref_expert', 'pref_novice')]

corr_rows = []
for group in ['expert', 'novice']:
    pref_col = f'pref_{group}'
    rs = []
    ps = []
    names = []

    for feat in feature_cols:
        valid = all_feat.dropna(subset=[feat, pref_col])
        if len(valid) < 5:
            continue
        r, p = stats.spearmanr(valid[feat], valid[pref_col])
        rs.append(r)
        ps.append(p)
        names.append(feat)

    # FDR correction
    reject, p_fdr, _, _ = multipletests(ps, alpha=0.05, method='fdr_bh')

    for feat, r, p, pf, sig in zip(names, rs, ps, p_fdr, reject):
        corr_rows.append({
            'group': group, 'feature': feat,
            'spearman_r': round(r, 4), 'p_value': round(p, 6),
            'p_fdr': round(pf, 6), 'significant_fdr': bool(sig),
        })
        star = " ***" if sig else ""
        logger.info(f"  {group:7s} | {feat:25s} | r={r:+.3f} | p={p:.4f} | pFDR={pf:.4f}{star}")

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(out_dir / "feature_correlations_full.csv", index=False)

# =============================================================================
# 7. FIGURE: PREFERENCE DRIVERS PANEL
# =============================================================================

logger.info("\nGenerating figure panel...")
apply_nature_rc()

# Select features for plotting (most interpretable)
plot_features = [
    ('is_checkmate', 'Checkmate status'),
    ('piece_count', 'Piece count'),
    ('total_material', 'Total material'),
    ('center_density', 'Center density'),
    ('piece_spread', 'Piece spread'),
    ('image_entropy', 'Image entropy'),
    ('edge_density', 'Edge density'),
    ('luminance_std', 'Contrast (lum. SD)'),
]

n_feat = len(plot_features)
fig, axes = plt.subplots(2, n_feat, figsize=(n_feat * 2.2, 7),
                          gridspec_kw={'hspace': 0.45, 'wspace': 0.35})

expert_color = '#E74C3C'
novice_color = '#3498DB'

for g_idx, (group, group_color, group_label) in enumerate([
    ('expert', expert_color, 'Experts'),
    ('novice', novice_color, 'Novices'),
]):
    pref_col = f'pref_{group}'

    for f_idx, (feat_name, feat_label) in enumerate(plot_features):
        ax = axes[g_idx, f_idx]
        valid = all_feat.dropna(subset=[feat_name, pref_col])

        # Color by check status
        cm_mask = valid['is_checkmate'] == 1
        ax.scatter(valid.loc[cm_mask, feat_name], valid.loc[cm_mask, pref_col],
                   c='#E74C3C', s=12, alpha=0.6, edgecolors='white', linewidths=0.2,
                   zorder=3, label='C')
        ax.scatter(valid.loc[~cm_mask, feat_name], valid.loc[~cm_mask, pref_col],
                   c='#3498DB', s=12, alpha=0.6, edgecolors='white', linewidths=0.2,
                   zorder=3, label='NC')

        # Regression line
        r, p = stats.spearmanr(valid[feat_name], valid[pref_col])
        z = np.polyfit(valid[feat_name], valid[pref_col], 1)
        x_line = np.linspace(valid[feat_name].min(), valid[feat_name].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), '--', color='gray', linewidth=0.6)

        # Get FDR p
        fdr_row = corr_df[(corr_df['group'] == group) & (corr_df['feature'] == feat_name)]
        p_fdr = fdr_row['p_fdr'].values[0] if len(fdr_row) > 0 else p
        sig_marker = '*' if p_fdr < 0.05 else ''

        ax.set_xlabel(feat_label, fontsize=5.5)
        if f_idx == 0:
            ax.set_ylabel(f'{group_label}\nSelection freq.', fontsize=6)
        ax.set_title(f'r={r:.2f}{sig_marker}', fontsize=6,
                     color='red' if p_fdr < 0.05 else 'black')
        ax.tick_params(labelsize=5)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.3, alpha=0.4)

        if g_idx == 0 and f_idx == n_feat - 1:
            ax.legend(fontsize=4, loc='best', framealpha=0.4)

# Suptitle
fig.suptitle('Feature Drivers of Board Preference', fontsize=9, weight='bold', y=0.98)

panel_path = FIGURES_DIR / 'panels' / 'preference_drivers_panel.pdf'
fig.savefig(panel_path, dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'preference_drivers_panel.svg', dpi=300, bbox_inches='tight')
plt.close(fig)
logger.info(f"Saved: {panel_path}")

# =============================================================================
# 8. SUMMARY
# =============================================================================

logger.info("\n" + "="*70)
logger.info("SUMMARY OF SIGNIFICANT PREFERENCE DRIVERS (FDR < 0.05)")
logger.info("="*70)
sig = corr_df[corr_df['significant_fdr']]
if len(sig) == 0:
    logger.info("  No features survive FDR correction.")
else:
    for _, row in sig.iterrows():
        logger.info(f"  {row['group']:7s} | {row['feature']:25s} | r={row['spearman_r']:+.3f} | pFDR={row['p_fdr']:.4f}")

logger.info("\nTop uncorrected correlations (p < 0.05):")
nom_sig = corr_df[corr_df['p_value'] < 0.05].sort_values('p_value')
for _, row in nom_sig.iterrows():
    logger.info(f"  {row['group']:7s} | {row['feature']:25s} | r={row['spearman_r']:+.3f} | p={row['p_value']:.4f}")

log_script_end(logger)
