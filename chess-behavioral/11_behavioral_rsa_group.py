#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioral RSA -- group stage
=============================

Reads per-subject pairwise preference counts from the ``behavioral-rsa``
BIDS derivative (produced by ``01_behavioral_rsa_subject.py``), separates
subjects by expertise group from ``participants.tsv``, sums counts per
(better, worse) pair within each group, and computes:

- raw-count behavioral RDM (reference)
- count-normalized behavioral RDM (primary for RSA correlations)
- directional DSM (for visualization)
- Pearson correlations with each model RDM (checkmate / strategy / visual
  similarity) + 95% CIs + raw and FDR-corrected p-values
- 2D MDS embedding of each group RDM

All outputs land under the unified repo results tree at
``results/behavioral/data/``.

The group-level statistics are mathematically equivalent to aggregating
the raw per-trial rows in one pass: summing already-aggregated
per-subject counts and re-normalising yields the same group RDMs as
walking the events TSVs subject-by-subject.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

from common import (
    CONFIG,
    setup_analysis,
    log_script_end,
    get_participants_with_expertise,
    load_stimulus_metadata,
    MODEL_LABELS,
)
from common.stats_utils import apply_fdr_correction
from common.report_utils import format_correlation_summary
from analyses.behavioral.rdm_utils import (
    compute_symmetric_rdm,
    compute_normalized_rdm,
    compute_directional_dsm,
    compute_normalized_dsm,
    correlate_with_all_models,
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
config, _, logger = setup_analysis(
    analysis_name="02_behavioral_rsa_group",
    results_base=CONFIG["RESULTS_ROOT"] / "behavioral" / "logs",
    script_file=__file__,
)

BEHAVIORAL_RSA_ROOT: Path = CONFIG["BIDS_BEHAVIORAL_RSA"]
OUTPUT_DIR: Path = CONFIG["RESULTS_ROOT"] / "behavioral" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_FILE_SUFFIX = "_desc-preference_pairs.tsv"


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------
def load_subject_pairs(subject_id: str) -> pd.DataFrame | None:
    """Load per-subject pair-count TSV or return None if missing."""
    path = BEHAVIORAL_RSA_ROOT / subject_id / f"{subject_id}{SUBJECT_FILE_SUFFIX}"
    if not path.is_file():
        logger.warning(f"  {subject_id}: no TSV at {path}")
        return None
    df = pd.read_csv(path, sep="\t")
    return df[["better", "worse", "count"]]


def aggregate_group(pair_dfs: list) -> pd.DataFrame:
    """Sum per-subject pair counts into a single group count table."""
    if not pair_dfs:
        return pd.DataFrame(columns=["better", "worse", "count"])
    stacked = pd.concat(pair_dfs, ignore_index=True)
    return (
        stacked.groupby(["better", "worse"], as_index=False)["count"]
        .sum()
        .sort_values(["better", "worse"])
        .reset_index(drop=True)
    )


def analyze_group(
    group_pairs: pd.DataFrame,
    category_df: pd.DataFrame,
    expertise_label: str,
):
    """Compute RDMs, DSMs (raw + normalized), and model correlations for one group."""
    logger.info(f"Analyzing {expertise_label} group...")
    logger.info(f"  {len(group_pairs)} unique pairwise comparisons")

    group_rdm_raw = compute_symmetric_rdm(group_pairs)
    logger.info("  Computed raw-count behavioral RDM")

    group_rdm = compute_normalized_rdm(group_pairs)
    logger.info("  Computed count-normalized behavioral RDM (primary)")

    group_dsm_raw = compute_directional_dsm(group_pairs)
    logger.info("  Computed raw-count directional DSM")

    group_dsm = compute_normalized_dsm(group_pairs)
    logger.info("  Computed count-normalized directional DSM (primary)")

    correlation_results, model_rdms = correlate_with_all_models(
        group_rdm, category_df, CONFIG["MODEL_COLUMNS"]
    )

    logger.info(f"\n  Count-Normalized Correlation Results ({expertise_label}):")
    for col, r, p, ci_l, ci_u in correlation_results:
        logger.info(
            f"    {col}: r={r:.3f}, p={p:.3e}, 95%CI=[{ci_l:.3f}, {ci_u:.3f}]"
        )

    return (
        group_rdm,
        group_rdm_raw,
        group_dsm,
        group_dsm_raw,
        correlation_results,
        model_rdms,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
logger.info("Loading participant information...")
participants_list, (n_experts, n_novices) = get_participants_with_expertise()
logger.info(f"Loaded {n_experts} experts and {n_novices} novices")

logger.info(f"Reading per-subject pairs from {BEHAVIORAL_RSA_ROOT}...")
experts_pairs = []
novices_pairs = []
for subject_id, is_expert in participants_list:
    df = load_subject_pairs(subject_id)
    if df is None:
        continue
    (experts_pairs if is_expert else novices_pairs).append(df)

logger.info(
    f"Loaded pairs from {len(experts_pairs)} experts and {len(novices_pairs)} novices"
)

logger.info("Aggregating pairwise counts within each group...")
expert_pairwise = aggregate_group(experts_pairs)
novice_pairwise = aggregate_group(novices_pairs)
logger.info(f"  Experts: {len(expert_pairwise)} unique stimulus pairs")
logger.info(f"  Novices: {len(novice_pairwise)} unique stimulus pairs")

logger.info("Loading stimulus categories for model RDMs...")
category_df = load_stimulus_metadata()
logger.info(f"Loaded categories for {len(category_df)} stimuli")

(
    expert_rdm,
    expert_rdm_raw,
    expert_dsm,
    expert_dsm_raw,
    expert_correlations,
    expert_model_rdms,
) = analyze_group(expert_pairwise, category_df, "Experts")
(
    novice_rdm,
    novice_rdm_raw,
    novice_dsm,
    novice_dsm_raw,
    novice_correlations,
    novice_model_rdms,
) = analyze_group(novice_pairwise, category_df, "Novices")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
logger.info(f"Saving results under {OUTPUT_DIR}...")

# Behavioral RDMs (symmetric, |count(i>j) - count(j>i)| / total)
np.save(OUTPUT_DIR / "expert_behavioral_rdm.npy", expert_rdm)            # normalized (primary)
np.save(OUTPUT_DIR / "novice_behavioral_rdm.npy", novice_rdm)            # normalized (primary)
np.save(OUTPUT_DIR / "expert_behavioral_rdm_raw.npy", expert_rdm_raw)    # raw signed counts
np.save(OUTPUT_DIR / "novice_behavioral_rdm_raw.npy", novice_rdm_raw)    # raw signed counts

# Directional DSMs (antisymmetric, (count(i>j) - count(j>i)) / total)
np.save(OUTPUT_DIR / "expert_directional_dsm.npy", expert_dsm)           # normalized (primary)
np.save(OUTPUT_DIR / "novice_directional_dsm.npy", novice_dsm)           # normalized (primary)
np.save(OUTPUT_DIR / "expert_directional_dsm_raw.npy", expert_dsm_raw)   # raw signed counts
np.save(OUTPUT_DIR / "novice_directional_dsm_raw.npy", novice_dsm_raw)   # raw signed counts
logger.info("  Saved behavioral RDMs and directional DSMs (normalized + raw)")

logger.info("Computing 2D MDS embeddings for Experts and Novices...")
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=CONFIG["RANDOM_SEED"],
)
expert_mds_coords = mds.fit_transform(expert_rdm)
novice_mds_coords = mds.fit_transform(novice_rdm)
np.save(OUTPUT_DIR / "expert_mds_coords.npy", expert_mds_coords)
np.save(OUTPUT_DIR / "novice_mds_coords.npy", novice_mds_coords)
logger.info("  Saved MDS coordinates for both groups")

pairwise_data = {"expert_pairwise": expert_pairwise, "novice_pairwise": novice_pairwise}
with open(OUTPUT_DIR / "pairwise_data.pkl", "wb") as f:
    pickle.dump(pairwise_data, f)
logger.info("  Saved group-aggregated pairwise data")

model_rdms_dict = {
    col: rdm
    for (col, _, _, _, _), rdm in zip(expert_correlations, expert_model_rdms)
}

exp_p_raw = np.array([p for (_, _, p, _, _) in expert_correlations], dtype=float)
nov_p_raw = np.array([p for (_, _, p, _, _) in novice_correlations], dtype=float)
_, exp_p_fdr = apply_fdr_correction(exp_p_raw, alpha=CONFIG.get("ALPHA_FDR", 0.05))
_, nov_p_fdr = apply_fdr_correction(nov_p_raw, alpha=CONFIG.get("ALPHA_FDR", 0.05))

expert_p_fdr_map = {
    name: float(pf) for (name, _, _, _, _), pf in zip(expert_correlations, exp_p_fdr)
}
novice_p_fdr_map = {
    name: float(pf) for (name, _, _, _, _), pf in zip(novice_correlations, nov_p_fdr)
}

correlation_results = {
    "expert": expert_correlations,
    "novice": novice_correlations,
    "expert_p_fdr": expert_p_fdr_map,
    "novice_p_fdr": novice_p_fdr_map,
    "model_columns": CONFIG["MODEL_COLUMNS"],
    "alpha_fdr": CONFIG.get("ALPHA_FDR", 0.05),
    "fdr_method": "fdr_bh",
}
with open(OUTPUT_DIR / "model_rdms.pkl", "wb") as f:
    pickle.dump(model_rdms_dict, f)
with open(OUTPUT_DIR / "correlation_results.pkl", "wb") as f:
    pickle.dump(correlation_results, f)
logger.info("  Saved model RDMs and correlation results")

summary_df = format_correlation_summary(
    expert_correlations,
    novice_correlations,
    model_columns=CONFIG["MODEL_COLUMNS"],
    model_labels_map=MODEL_LABELS,
    exp_p_fdr=expert_p_fdr_map,
    nov_p_fdr=novice_p_fdr_map,
)
summary_df.to_csv(OUTPUT_DIR / "correlation_summary.csv", index=False)
logger.info("  Saved summary table")

logger.info("\n" + "=" * 80)
logger.info("BEHAVIORAL RDM CORRELATIONS (Experts vs. Novices)")
logger.info("=" * 80)
logger.info("\n" + summary_df.to_string(index=False))
logger.info("=" * 80 + "\n")

log_script_end(logger)
logger.info(f"All outputs saved under: {OUTPUT_DIR}")
