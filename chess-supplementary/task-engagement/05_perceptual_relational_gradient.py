"""
Perceptual-to-Relational Feature Gradient of Board Preference
==============================================================

METHODS
-------

Rationale
---------
Diagnostics 4--5 established that expert preferences are driven by checkmate
status while novice preferences correlate with visual complexity metrics.
This analysis quantifies the structure of this dissociation by testing 8
board features spanning a gradient from purely perceptual to deeply relational,
computing both bivariate and partial correlations to isolate unique contributions,
and decomposing explained variance into perceptual, structural, and
strategic-relational blocks.

Data
----
Board selection frequencies from the 1-back preference task (same as
Diagnostics 4--5). Eight features extracted per board from FEN (via
python-chess) and stimulus PNG images.

Features (ordered perceptual to relational):
  1. Image entropy       -- Shannon entropy of grayscale pixel histogram
  2. Edge density        -- Proportion of edge pixels (Sobel filter, 90th pct)
  3. Piece count         -- Total pieces on board
  4. Officer count       -- Non-pawn, non-king pieces (N+B+R+Q)
  5. Center occupation   -- Pieces in central 4x4 squares (c3-f6)
  6. King advantage      -- Opponent king exposure minus own king exposure
                            (positive = opponent's king more threatened;
                            participants always play white)
  7. Attack advantage    -- White attack coverage minus black attack coverage
                            (positive = white controls more squares)
  8. Checkmate status    -- Binary: checkmate position or not

Procedure
---------
1. Extract all 8 features for each of the 40 boards.
2. Compute Spearman correlations between each feature and the board's mean
   selection frequency per group. Apply Benjamini-Hochberg FDR correction
   across the 8 features within each group (alpha = 0.05).
3. Compute partial Spearman correlations: for each feature, residualise both
   the feature and the preference on all other 7 features (OLS), then
   correlate the residuals. This isolates the unique predictive contribution
   of each feature beyond shared variance.
4. Hierarchical variance partitioning: enter features in three ordered blocks
   (Perceptual, Structural, Strategic-Relational) into a cumulative linear
   regression. Delta-R-squared per block quantifies how much unique variance
   each category adds beyond what was already explained.

Statistical Tests
-----------------
- Spearman rank correlation (bivariate and partial)
- Benjamini-Hochberg FDR correction (alpha = 0.05, family = 8 features per group)
- Hierarchical OLS regression for variance partitioning

Outputs
-------
- gradient_feature_matrix.csv: 40 boards x 8 features + preference per group
- gradient_bivariate_correlations.csv: Spearman r, p, pFDR per feature x group
- gradient_partial_correlations.csv: Partial r, p, pFDR per feature x group
- gradient_variance_partitioning.csv: Delta-R2 per block per group
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import sobel
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
import chess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end
from common.bids_utils import load_stimulus_metadata

# =============================================================================
# 1. SETUP
# =============================================================================

config, out_dir, logger = setup_analysis(
    analysis_name="novice_diagnostics",
    results_base=Path(__file__).parent / "results",
    script_file=__file__,
)

STIMULI_DIR = CONFIG['EXTERNAL_DATA_ROOT'] / "stimuli"

# The 8 features in perceptual → relational order
# "Advantage" features use (white - black) or (black_exposure - white_exposure)
# since participants always play as white
FEATURE_ORDER = [
    ('image_entropy',      'Image entropy'),
    ('edge_density',       'Edge density'),
    ('piece_count',        'Piece count'),
    ('officer_count',      'Officer count'),
    ('center_occupation',  'Center occupation'),
    ('king_advantage',     'King advantage'),
    ('attack_advantage',   'Attack advantage'),
    ('is_checkmate',       'Checkmate status'),
]

FEATURE_KEYS = [fk for fk, _ in FEATURE_ORDER]

# Variance partitioning blocks
VP_BLOCKS = {
    'Perceptual': ['image_entropy', 'edge_density'],
    'Structural': ['piece_count', 'officer_count', 'center_occupation'],
    'Strategic-Relational': ['king_advantage', 'attack_advantage', 'is_checkmate'],
}

# =============================================================================
# 2. LOAD DATA
# =============================================================================

logger.info("Loading data...")

pref_path = out_dir / "board_preference_group.csv"
pref_df = pd.read_csv(pref_path)
stim_df = load_stimulus_metadata(return_all=True)
ck = 'check' if 'check' in stim_df.columns else 'check_status'

# =============================================================================
# 3. EXTRACT FEATURES
# =============================================================================

logger.info("Extracting 8 features (perceptual → relational)...")

CENTER_SQUARES = [chess.square(f, r) for f in range(2, 6) for r in range(2, 6)]


def extract_board_features(fen, check_val):
    """Extract 6 board-level features from FEN via python-chess."""
    board = chess.Board(fen)
    pm = board.piece_map()

    piece_count = len(pm)
    officer_count = sum(1 for p in pm.values()
                        if p.piece_type not in (chess.PAWN, chess.KING))
    center_occ = sum(1 for sq in CENTER_SQUARES if sq in pm)

    # King advantage: opponent (black) king exposure minus own (white) king exposure
    # Positive = opponent's king is more threatened (good for the player)
    # Participants always play as white in this task
    king_exp = {}
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            king_exp[color] = 0
            continue
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        adj = [chess.square(kf + df, kr + dr)
               for df in [-1, 0, 1] for dr in [-1, 0, 1]
               if not (df == 0 and dr == 0)
               and 0 <= kf + df <= 7 and 0 <= kr + dr <= 7]
        opp = not color
        attacked = sum(1 for sq in adj if board.is_attacked_by(opp, sq))
        king_exp[color] = attacked / max(len(adj), 1)
    king_advantage = king_exp[chess.BLACK] - king_exp[chess.WHITE]

    # Attack advantage: white attack coverage minus black (positive = white dominates)
    w_att = sum(1 for sq in chess.SQUARES if board.is_attacked_by(chess.WHITE, sq))
    b_att = sum(1 for sq in chess.SQUARES if board.is_attacked_by(chess.BLACK, sq))
    attack_advantage = w_att - b_att

    return {
        'piece_count': piece_count,
        'officer_count': officer_count,
        'center_occupation': center_occ,
        'king_advantage': king_advantage,
        'attack_advantage': attack_advantage,
        'is_checkmate': int(check_val == 'checkmate'),
    }


def extract_image_features(img_path):
    """Extract 2 image-level features from stimulus PNG."""
    img = plt.imread(str(img_path))
    if img.ndim == 3:
        gray = np.mean(img[:, :, :3], axis=2)
    else:
        gray = img.copy()
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)

    # Shannon entropy of pixel histogram
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    entropy = -np.sum(probs * np.log2(probs))

    # Edge density via Sobel filter
    edges_x = sobel(gray.astype(float), axis=0)
    edges_y = sobel(gray.astype(float), axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    edge_density = np.mean(edge_mag > np.percentile(edge_mag, 90))

    return {'image_entropy': entropy, 'edge_density': edge_density}


rows = []
for _, r in stim_df.iterrows():
    feats = extract_board_features(r['fen'], r[ck])
    img_path = STIMULI_DIR / r['filename']
    if img_path.exists():
        feats.update(extract_image_features(img_path))
    else:
        feats['image_entropy'] = np.nan
        feats['edge_density'] = np.nan
    feats['stim_id'] = r['stim_id']
    rows.append(feats)

feat_df = pd.DataFrame(rows)

# Add preference frequencies per group
for group in ['expert', 'novice']:
    gdf = pref_df[pref_df['group'] == group]
    freqs = {}
    for _, r in gdf.iterrows():
        freqs[int(r['c_stim_id'])] = r['c_freq']
        freqs[int(r['nc_stim_id'])] = r['nc_freq']
    feat_df[f'pref_{group}'] = feat_df['stim_id'].map(freqs)

feat_df.to_csv(out_dir / "gradient_feature_matrix.csv", index=False)
logger.info(f"Saved feature matrix: {len(feat_df)} boards x {len(FEATURE_KEYS)} features")

# =============================================================================
# 4. PERMUTATION-BASED BIVARIATE CORRELATIONS WITH BOOTSTRAP CIs
# =============================================================================

N_PERM = 10000
N_BOOT = 10000
RNG = np.random.default_rng(config['RANDOM_SEED'])

logger.info(f"\nBivariate Spearman correlations (permutation p, n_perm={N_PERM}; "
            f"bootstrap 95% CI, n_boot={N_BOOT}):")


def permutation_spearman(x, y, n_perm, rng):
    """Spearman r with permutation-based two-tailed p-value."""
    r_obs = stats.spearmanr(x, y).statistic
    count = 0
    for _ in range(n_perm):
        r_perm = stats.spearmanr(rng.permutation(x), y).statistic
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)  # +1 for the observed value
    return r_obs, p_perm


def bootstrap_ci_spearman(x, y, n_boot, rng, ci=0.95):
    """Bootstrap 95% CI for Spearman r."""
    n = len(x)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rs[i] = stats.spearmanr(x[idx], y[idx]).statistic
    alpha = (1 - ci) / 2
    return float(np.nanpercentile(rs, 100 * alpha)), float(np.nanpercentile(rs, 100 * (1 - alpha)))


corr_rows = []
for group in ['expert', 'novice']:
    pref_col = f'pref_{group}'
    rs, ps, names = [], [], []
    ci_lows, ci_highs = [], []

    for fk, _ in FEATURE_ORDER:
        valid = feat_df.dropna(subset=[fk, pref_col])
        if len(valid) < 5 or valid[fk].nunique() < 2:
            continue
        xv = valid[fk].values
        yv = valid[pref_col].values

        r, p_perm = permutation_spearman(xv, yv, N_PERM, RNG)
        ci_lo, ci_hi = bootstrap_ci_spearman(xv, yv, N_BOOT, RNG)

        rs.append(r); ps.append(p_perm); names.append(fk)
        ci_lows.append(ci_lo); ci_highs.append(ci_hi)

    reject, p_fdr, _, _ = multipletests(ps, alpha=0.05, method='fdr_bh')
    for fk, r, p, pf, sig, clo, chi in zip(names, rs, ps, p_fdr, reject, ci_lows, ci_highs):
        corr_rows.append({'group': group, 'feature': fk,
                          'spearman_r': round(r, 4),
                          'p_perm': round(p, 6), 'p_fdr': round(pf, 6),
                          'ci_low': round(clo, 4), 'ci_high': round(chi, 4),
                          'significant_fdr': bool(sig)})
        star = " *" if sig else ""
        logger.info(f"  {group:7s} | {fk:20s} | r={r:+.3f} [{clo:+.3f}, {chi:+.3f}] | "
                    f"p_perm={p:.4f} | pFDR={pf:.4f}{star}")

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(out_dir / "gradient_bivariate_correlations.csv", index=False)

# =============================================================================
# 5. PARTIAL CORRELATIONS (permutation p + bootstrap CI)
# =============================================================================

logger.info(f"\nPartial correlations (permutation p, n_perm={N_PERM}; "
            f"bootstrap 95% CI, n_boot={N_BOOT}):")

partial_rows = []
for group in ['expert', 'novice']:
    pref_col = f'pref_{group}'
    valid = feat_df[FEATURE_KEYS + [pref_col]].dropna()
    X = valid[FEATURE_KEYS].values
    y = valid[pref_col].values
    n = len(y)

    for i, (fk, fl) in enumerate(FEATURE_ORDER):
        others = [j for j in range(len(FEATURE_KEYS)) if j != i]
        X_others = X[:, others]

        # Residualise feature and outcome on all other features
        res_x = X[:, i] - LinearRegression().fit(X_others, X[:, i]).predict(X_others)
        res_y = y - LinearRegression().fit(X_others, y).predict(X_others)

        # Observed partial r
        r_obs = stats.spearmanr(res_x, res_y).statistic

        # Permutation p-value: shuffle residualised feature, recompute
        count = 0
        for _ in range(N_PERM):
            r_perm = stats.spearmanr(RNG.permutation(res_x), res_y).statistic
            if abs(r_perm) >= abs(r_obs):
                count += 1
        p_perm = (count + 1) / (N_PERM + 1)

        # Bootstrap CI on partial r
        rs_boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = RNG.integers(0, n, size=n)
            X_b, y_b = X[idx], y[idx]
            X_oth_b = X_b[:, others]
            rx_b = X_b[:, i] - LinearRegression().fit(X_oth_b, X_b[:, i]).predict(X_oth_b)
            ry_b = y_b - LinearRegression().fit(X_oth_b, y_b).predict(X_oth_b)
            rs_boot[b] = stats.spearmanr(rx_b, ry_b).statistic
        ci_lo = float(np.nanpercentile(rs_boot, 2.5))
        ci_hi = float(np.nanpercentile(rs_boot, 97.5))

        partial_rows.append({'group': group, 'feature': fk,
                             'partial_r': round(r_obs, 4),
                             'partial_p_perm': round(p_perm, 6),
                             'ci_low': round(ci_lo, 4),
                             'ci_high': round(ci_hi, 4)})

partial_df = pd.DataFrame(partial_rows)

# FDR within group on permutation p-values
for group in ['expert', 'novice']:
    mask = partial_df['group'] == group
    _, p_fdr, _, _ = multipletests(partial_df.loc[mask, 'partial_p_perm'].values,
                                    alpha=0.05, method='fdr_bh')
    partial_df.loc[mask, 'partial_p_fdr'] = p_fdr
    partial_df.loc[mask, 'significant_fdr'] = p_fdr < 0.05

for _, row in partial_df.iterrows():
    star = " *" if row.get('significant_fdr', False) else ""
    ci_str = f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
    logger.info(f"  {row['group']:7s} | {row['feature']:20s} | "
                f"partial r={row['partial_r']:+.3f} {ci_str} | "
                f"p_perm={row['partial_p_perm']:.4f} | pFDR={row.get('partial_p_fdr', row['partial_p_perm']):.4f}{star}")

partial_df.to_csv(out_dir / "gradient_partial_correlations.csv", index=False)

# =============================================================================
# 6. VARIANCE PARTITIONING
# =============================================================================

logger.info("\nVariance partitioning (hierarchical regression):")

vp_rows = []
for group in ['expert', 'novice']:
    pref_col = f'pref_{group}'
    valid = feat_df[FEATURE_KEYS + [pref_col]].dropna()
    y = valid[pref_col].values

    cumulative_feats = []
    prev_r2 = 0.0
    for block_name, block_feats in VP_BLOCKS.items():
        cumulative_feats.extend(block_feats)
        X_block = valid[cumulative_feats].values
        r2 = LinearRegression().fit(X_block, y).score(X_block, y)
        delta_r2 = r2 - prev_r2
        vp_rows.append({'group': group, 'block': block_name,
                        'cumulative_r2': round(r2, 4),
                        'delta_r2': round(delta_r2, 4),
                        'features': ', '.join(block_feats)})
        logger.info(f"  {group:7s} | + {block_name:25s} | "
                    f"cumR2={r2:.3f} | deltaR2={delta_r2:.3f}")
        prev_r2 = r2

vp_df = pd.DataFrame(vp_rows)
vp_df.to_csv(out_dir / "gradient_variance_partitioning.csv", index=False)

# =============================================================================
# 7. SUMMARY
# =============================================================================

logger.info("\n" + "=" * 70)
logger.info("SUMMARY")
logger.info("=" * 70)

logger.info("\nFDR-significant bivariate:")
for _, row in corr_df[corr_df['significant_fdr']].sort_values(['group', 'p_fdr']).iterrows():
    logger.info(f"  {row['group']:7s} | {row['feature']:20s} | r={row['spearman_r']:+.3f}")

logger.info("\nFDR-significant partial:")
sig_p = partial_df[partial_df.get('significant_fdr', False) == True]
if len(sig_p) == 0:
    logger.info("  None survive FDR after partialling.")
else:
    for _, row in sig_p.sort_values(['group', 'partial_p_fdr']).iterrows():
        logger.info(f"  {row['group']:7s} | {row['feature']:20s} | partial r={row['partial_r']:+.3f}")

logger.info("\nVariance partitioning (deltaR2):")
for _, row in vp_df.iterrows():
    logger.info(f"  {row['group']:7s} | {row['block']:25s} | deltaR2={row['delta_r2']:.3f}")

log_script_end(logger)
