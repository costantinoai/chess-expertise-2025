# Chess Expertise 2025 — Analysis Mapping (Authoritative)

Document purpose: map every analysis in the repo to the exact scripts that produce results, tables, and plots, capturing current conventions and strict data/compute policies. This reflects the refactored codebase as of the latest changes.

Manuscript: "Lower-Dimensional, Optimized Representations of High‑Level Information in Chess Experts"

Last Updated: 2025-10-29

---

## Repository Structure (current)

```
chess-expertise-2025/
├── common/                      # Shared utilities (single source of truth)
│   ├── __init__.py              # Re-exports unified plotting + helpers
│   ├── constants.py             # All config: paths, model labels/order
│   ├── logging_utils.py         # setup_analysis, setup_analysis_in_dir
│   ├── io_utils.py              # results discovery, TSV/NIfTI finders
│   ├── bids_utils.py            # BIDS I/O, ROI metadata, chance derivation
│   ├── neuro_utils.py           # NIfTI/atlas, Fisher z, load_atlas
│   ├── stats_utils.py           # Welch t, FDR, bootstrap corr (pingouin)
│   ├── report_utils.py          # LaTeX helpers, correlation tables
│   └── plotting/                # Nature-compliant plotting primitives
│       ├── bars.py, heatmaps.py, scatter.py, surfaces.py, style.py, ...
│
├── chess-behavioral/
│   ├── 01_behavioral_rsa.py
│   ├── 81_table_behavioral_correlations.py
│   ├── 91_plot_behavioral_panels.py
│   └── modules/ (data_loading.py, rdm_utils.py)
│
├── chess-manifold/
│   ├── 01_manifold_analysis.py
│   ├── 81_table_manifold_pr.py
│   ├── 91_plot_manifold_panels.py
│   └── modules/ (data.py, models.py, analysis.py, tables.py)
│
├── chess-mvpa/
│   ├── 01_roi_mvpa_main.m       # Subject-level MVPA (MATLAB)
│   ├── 02_mvpa_group_rsa.py
│   ├── 03_mvpa_group_decoding.py
│   ├── 81_table_mvpa_rsa.py
│   ├── 82_table_mvpa_decoding.py
│   ├── 92_plot_mvpa_rsa.py
│   ├── 93_plot_mvpa_decoding.py
│   └── modules/ (mvpa_io.py, mvpa_group.py, mvpa_plot_utils.py)
│
├── chess-neurosynth/
│   ├── 01_univariate_neurosynth.py
│   ├── 02_rsa_neurosynth.py
│   ├── 81_table_neurosynth_univariate.py
│   ├── 82_table_neurosynth_rsa.py
│   ├── 91_plot_neurosynth_univariate.py
│   ├── 92_plot_neurosynth_rsa.py
│   └── modules/ (io_utils.py, maps_utils.py, glm_utils.py)
│
├── chess-supplementary/         # Placeholder for supplementary analyses
└── old-implementation/          # Read-only reference
```

---

## Complete Analysis Inventory (scripts that actually run)

### 1) Behavioral RSA
- Folder: `chess-behavioral/`
- Analysis: `01_behavioral_rsa.py` — builds group RDM/DSM from BIDS events and correlates with models.
- Tables: `81_table_behavioral_correlations.py` — multicolumn LaTeX for Experts/Novices (r, 95% CI, p_FDR).
- Plots: `91_plot_behavioral_panels.py` — panels for RDMs, DSMs, 2D embeddings, correlations.
- Outputs: RDM/DSM `.npy`, 2D MDS `.npy`, `model_rdms.pkl`, `correlation_results.pkl`, CSV+LaTeX tables, figures.

---

### 2) Neurosynth — Univariate
- Folder: `chess-neurosynth/`
- Analysis: `01_univariate_neurosynth.py` — group GLM T→Z, split Z+/Z−, bootstrap corr vs 7 term maps, FDR, CSV.
- Tables: `81_table_neurosynth_univariate.py` — POS/NEG multicolumn and DIFF (Δr with CI).
- Plots: `91_plot_neurosynth_univariate.py` — glass brains, surfaces, bars.

---

### 3) Neurosynth — RSA (Searchlight groups)
- Folder: `chess-neurosynth/`
- Analysis: `02_rsa_neurosynth.py` — group z-score map (Experts>Novices) per RSA pattern; split Z+/Z−; bootstrap corr vs terms.
- Tables: `82_table_neurosynth_rsa.py` — POS/NEG multicolumn and DIFF.
- Plots: `92_plot_neurosynth_rsa.py` — glass brains, surfaces, bars.

---

### 4) MVPA — Group RSA (ROI-level)
- Folder: `chess-mvpa/`
- Analysis: `02_mvpa_group_rsa.py` — aggregates subject TSVs from `mvpa/rsa_corr/`, per-ROI stats and FDR, writes artifacts.
- Tables: `81_table_mvpa_rsa.py` — Experts/Novices means (CI), Δ and p_FDR per ROI.
- Plots: `92_plot_mvpa_rsa.py` — ROI bar panels.

---

### 5) MVPA — Group Decoding (ROI-level)
- Folder: `chess-mvpa/`
- Analysis: `03_mvpa_group_decoding.py` — aggregates `mvpa/svm/`, derives chance per target via stimuli.tsv or explicit rules, per-ROI stats.
- Tables: `82_table_mvpa_decoding.py` — report (acc − chance) with CI and p_FDR.
- Plots: `93_plot_mvpa_decoding.py` — ROI bar panels.

---

### 6) Manifold — Participation Ratio (PR)
- Folder: `chess-manifold/`
- Analysis: `01_manifold_analysis.py` — compute per-ROI PR per subject, group summaries, Welch+FDR, PCA2D, classification tests, voxel-size checks.
- Tables: `81_table_manifold_pr.py` — multicolumn table for Expert/Novice PR and Δ.
- Plots: `91_plot_manifold_panels.py` — PR bars, differences, PCA loadings/projection, voxel-size correlations.

---

## Configuration & Policies (enforced)

- Single source of truth: `common/constants.py` — all paths, model labels/order, MVPA targets and defaults.
- External data root: `CONFIG['BIDS_ROOT']` etc. point to `/media/.../manuscript-data`. No environment overrides.
- Hard dependencies: `pingouin`, `statsmodels`, `nibabel`, `matplotlib`, `seaborn`, `plotly` (+ `kaleido` for static images, or set `SKIP_PLOTLY_STATIC_EXPORT=1`). Missing deps raise ImportError; no fallbacks.
- Stats: `stats_utils.correlate_vectors_bootstrap` requires `pingouin.corr` with bootstrap; FDR via `statsmodels.multipletests`; no silent NaNs. Effect sizes via `pingouin.compute_effsize`.
- I/O strictness:
  - `common/io_utils.find_subject_tsvs` requires exactly one TSV per subject directory; multiple files is an error.
  - ROI TSV metadata standardized via `common/bids_utils.load_roi_metadata` → `roi_id`, `roi_name`, `pretty_name`, `family`, `color`.
- MVPA chance levels: `common/bids_utils.derive_target_chance_from_stimuli` derives chance from stimuli.tsv; alias `stimuli→stim_id`; any `*_half` is 0.5; otherwise uses explicit defaults in `CONFIG['MVPA_SVM_CHANCE_DEFAULTS']`.
- Logging: all analyses call `setup_analysis(...)` (or `setup_analysis_in_dir(...)` for table/plot scripts), which logs configuration, dataset summary, and copies the script into the results directory.
- Results structure: Timestamped `results/YYYYMMDD-HHMMSS_<analysis>` with subdirs like `figures/` and `tables/` as needed. Table scripts are `81_*.py`; plot scripts are `91+_*.py`.

---

## Input → Output Flow (high level)

```
Raw BIDS
  ↓
fMRIPrep → derivatives/fmriprep/
  ↓
GLM (SPM) → derivatives/SPM/GLM-smooth{4,6}/
  ↓             ↓                ↓
MVPA (MATLAB)  RSA searchlight   PR
  ↓             ↓                ↓
Group (ROI)     Group z (E>N)    Group stats + PCA + CLS
  ↓             ↓                ↓
Tables/Plots    Neurosynth uni/RSA   Tables/Plots
```

---

## Notes & Guarantees

- This document is the single source of truth for where to run analyses, generate tables (81/82), and render plots (91–93). It reflects current code and strict policies (no silent fallbacks, hard deps, error-on-ambiguity).
- ROI metadata are standardized to `roi_id` across analyses; joins should always use that column.
- MVPA TSV discovery enforces exactly one file per subject per method.
- All scripts log start/end and key results via `common.logging_utils` and copy the executing script into the results directory for provenance.

