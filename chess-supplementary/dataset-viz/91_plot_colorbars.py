"""
Generate all required colorbars for the supplementary materials.

This script extracts vmin/vmax from the actual data and generates colorbars for:
1. Behavioral directional matrices (vertical, symmetric)
2. Positive only 0-0.1 (horizontal)
3. Neurosynth univariate surfaces (vertical, symmetric)
4. Neurosynth RSA surfaces (horizontal, symmetric)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make repo root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from common import CONFIG, apply_nature_rc, save_figure, CMAP_BRAIN, create_standalone_colorbar
from common.logging_utils import setup_analysis
from common.io_utils import find_latest_results_directory

# =============================================================================
# Setup
# =============================================================================

analysis_name = "colorbars"
script_dir = Path(__file__).parent
results_base = script_dir / "results"
config, output_dir, logger = setup_analysis(
    analysis_name=analysis_name,
    results_base=results_base,
    script_file=__file__,
)

figures_dir = output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

apply_nature_rc()


# =============================================================================
# 1. Behavioral directional matrices (vertical, symmetric around 0)
# =============================================================================

logger.info("Loading behavioral directional matrices...")

behavioral_results_dir = find_latest_results_directory(
    Path(__file__).parent.parent.parent / "chess-behavioral" / "results",
    pattern="*_behavioral_rsa",
    require_exists=True,
    verbose=False,
)

expert_dsm = np.load(behavioral_results_dir / "expert_directional_dsm.npy")
novice_dsm = np.load(behavioral_results_dir / "novice_directional_dsm.npy")

global_max_abs = max(
    np.abs(expert_dsm).max(),
    np.abs(novice_dsm).max()
)
behavioral_vmin = -global_max_abs
behavioral_vmax = global_max_abs

logger.info(f"Behavioral vmin={behavioral_vmin:.4f}, vmax={behavioral_vmax:.4f}")

fig_behavioral = create_standalone_colorbar(
    cmap=CMAP_BRAIN,
    vmin=behavioral_vmin,
    vmax=behavioral_vmax,
    orientation='vertical',
    label='Directional Preference',
)
save_figure(fig_behavioral, figures_dir / "colorbar_behavioral_directional_vertical")
logger.info("✓ Saved behavioral directional colorbar (vertical)")


# =============================================================================
# 2. Positive only 0-0.1 (horizontal, brain positive part)
# =============================================================================

logger.info("Generating 0-0.1 colorbar...")

import matplotlib.pyplot as plt

fig_0_01 = create_standalone_colorbar(
    cmap=plt.cm.RdPu,
    vmin=0.0,
    vmax=0.1,
    orientation='horizontal',
    label='r',
)
save_figure(fig_0_01, figures_dir / "colorbar_0_to_01_horizontal")
logger.info("✓ Saved 0-0.1 colorbar (horizontal)")


# =============================================================================
# 3. Neurosynth univariate (vertical, symmetric around 0)
# =============================================================================

logger.info("Loading neurosynth univariate zmaps...")

neurosynth_univ_results_dir = find_latest_results_directory(
    Path(__file__).parent.parent.parent / "chess-neurosynth" / "results",
    pattern="*_neurosynth_univariate",
    require_exists=True,
    verbose=False,
)

try:
    from nilearn import image, datasets, surface
    fsavg = datasets.fetch_surf_fsaverage('fsaverage')

    def _absmax_surface(zimg):
        texl = surface.vol_to_surf(zimg, fsavg.pial_left)
        texr = surface.vol_to_surf(zimg, fsavg.pial_right)
        return float(np.nanmax(np.abs(np.concatenate([texl, texr]))))

    vmax_univ = 0.0
    for stem in ['spmT_exp-gt-nonexp_all-gt-rest', 'spmT_exp-gt-nonexp_check-gt-nocheck']:
        zmap_file = neurosynth_univ_results_dir / f"zmap_{stem}.nii.gz"
        if zmap_file.exists():
            z_img = image.load_img(str(zmap_file))
            vmax_univ = max(vmax_univ, _absmax_surface(z_img))
        else:
            logger.warning(f"File not found: {zmap_file}")
    vmin_univ = -vmax_univ

except Exception as e:
    logger.warning(f"Could not compute from surface, using volume fallback: {e}")
    # Fallback: compute from volume values
    def _absmax_volume(zimg):
        arr = zimg.get_fdata()
        return float(np.nanmax(np.abs(arr)))

    vmax_univ = 0.0
    for stem in ['spmT_exp-gt-nonexp_all-gt-rest', 'spmT_exp-gt-nonexp_check-gt-nocheck']:
        zmap_file = neurosynth_univ_results_dir / f"zmap_{stem}.nii.gz"
        if zmap_file.exists():
            z_img = image.load_img(str(zmap_file))
            vmax_univ = max(vmax_univ, _absmax_volume(z_img))
        else:
            logger.warning(f"File not found: {zmap_file}")
    vmin_univ = -vmax_univ

logger.info(f"Neurosynth univariate vmin={vmin_univ:.4f}, vmax={vmax_univ:.4f}")

fig_neurosynth_univ = create_standalone_colorbar(
    cmap=CMAP_BRAIN,
    vmin=vmin_univ,
    vmax=vmax_univ,
    orientation='vertical',
    label='z-score',
)
save_figure(fig_neurosynth_univ, figures_dir / "colorbar_neurosynth_univariate_vertical")
logger.info("✓ Saved neurosynth univariate colorbar (vertical)")


# =============================================================================
# 4. Neurosynth RSA (horizontal, symmetric around 0)
# =============================================================================

logger.info("Loading neurosynth RSA zmaps...")

neurosynth_rsa_results_dir = find_latest_results_directory(
    Path(__file__).parent.parent.parent / "chess-neurosynth" / "results",
    pattern="*_neurosynth_rsa",
    require_exists=True,
    verbose=False,
)

PATTERNS = [
    ("searchlight_checkmate", "Checkmate"),
    ("searchlight_strategy", "Strategy"),
    ("searchlight_visualSimilarity", "Visual Similarity"),
]

try:
    vmax_rsa = 0.0
    for stem, _pretty in PATTERNS:
        zmap_file = neurosynth_rsa_results_dir / f"zmap_{stem}.nii.gz"
        if zmap_file.exists():
            z_img = image.load_img(str(zmap_file))
            vmax_rsa = max(vmax_rsa, _absmax_surface(z_img))
        else:
            logger.warning(f"File not found: {zmap_file}")
    vmin_rsa = -vmax_rsa

except Exception as e:
    logger.warning(f"Could not compute from surface, using volume fallback: {e}")
    vmax_rsa = 0.0
    for stem, _pretty in PATTERNS:
        zmap_file = neurosynth_rsa_results_dir / f"zmap_{stem}.nii.gz"
        if zmap_file.exists():
            z_img = image.load_img(str(zmap_file))
            vmax_rsa = max(vmax_rsa, _absmax_volume(z_img))
        else:
            logger.warning(f"File not found: {zmap_file}")
    vmin_rsa = -vmax_rsa

logger.info(f"Neurosynth RSA vmin={vmin_rsa:.4f}, vmax={vmax_rsa:.4f}")

fig_neurosynth_rsa = create_standalone_colorbar(
    cmap=CMAP_BRAIN,
    vmin=vmin_rsa,
    vmax=vmax_rsa,
    orientation='horizontal',
    label='z-score',
)
save_figure(fig_neurosynth_rsa, figures_dir / "colorbar_neurosynth_rsa_horizontal")
logger.info("✓ Saved neurosynth RSA colorbar (horizontal)")


# =============================================================================
# 5. Manifold - PR profiles matrix (vertical, mako colormap)
# =============================================================================

logger.info("Loading manifold PR profiles matrix...")

import pickle

manifold_results_dir = find_latest_results_directory(
    Path(__file__).parent.parent.parent / "chess-manifold" / "results",
    pattern="*_manifold",
    require_exists=True,
    verbose=False,
)

with open(manifold_results_dir / "pr_results.pkl", "rb") as f:
    manifold_results = pickle.load(f)

pr_matrix = manifold_results['pr_matrix']['matrix']
pr_vmin = float(pr_matrix.min())
pr_vmax = float(pr_matrix.max())

logger.info(f"Manifold PR profiles vmin={pr_vmin:.4f}, vmax={pr_vmax:.4f}")

fig_pr_profiles = create_standalone_colorbar(
    cmap='mako',
    vmin=pr_vmin,
    vmax=pr_vmax,
    orientation='vertical',
    label='PR',
)
save_figure(fig_pr_profiles, figures_dir / "colorbar_manifold_pr_profiles_vertical")
logger.info("✓ Saved manifold PR profiles colorbar (vertical)")


# =============================================================================
# 6. Manifold - PCA components contributions (horizontal, brain cmap)
# =============================================================================

logger.info("Loading manifold PCA components...")

pca_loadings = manifold_results['pca2d']['components']
pca_max_abs = float(np.abs(pca_loadings).max())
pca_vmin = -pca_max_abs
pca_vmax = pca_max_abs

logger.info(f"Manifold PCA components vmin={pca_vmin:.4f}, vmax={pca_vmax:.4f}")

fig_pca_components = create_standalone_colorbar(
    cmap=CMAP_BRAIN,
    vmin=pca_vmin,
    vmax=pca_vmax,
    orientation='horizontal',
    label='Loading',
)
save_figure(fig_pca_components, figures_dir / "colorbar_manifold_pca_components_horizontal")
logger.info("✓ Saved manifold PCA components colorbar (horizontal)")

logger.info("\n" + "="*80)
logger.info("All colorbars generated successfully!")
logger.info("="*80)
