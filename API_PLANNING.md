# API Index — Function Registry (Current)

Tracks reusable APIs across the codebase for discoverability and DRY. Reflects the refactor with centralized `common/` and per‑package `modules/`.

Last Updated: 2025-10-29

---

## Common Utilities (`common/`)

### constants.py
- CONFIG: all paths and parameters (BIDS roots, ROI paths, model labels/order, MVPA targets, chance defaults, ALPHA).
- Re-exported model label mappings and plotting primitives in `common/__init__.py`.

### plotting/ (re-exported via `common`)
- apply_nature_rc, figure_size, auto_bar_figure_size
- plot_grouped_bars_with_ci, plot_grouped_bars_on_ax, plot_counts_on_ax
- plot_rdm, plot_matrix_on_ax, add_rdm_category_bars, add_roi_color_legend
- plot_2d_embedding, plot_2d_embedding_on_ax
- plot_flat_pair, plot_flat_hemisphere (static export requires kaleido unless `SKIP_PLOTLY_STATIC_EXPORT=1`)

### logging_utils.py
- setup_analysis, setup_analysis_in_dir, setup_logging, log_script_start, log_script_end, log_dataset_summary

### neuro_utils.py
- load_nifti, load_atlas, fisher_z_transform, basic map utilities used by neurosynth and manifold.

### bids_utils.py
- load_participants_tsv, get_subject_list, get_group_summary, get_participants_with_expertise, get_subject_info
- load_stimulus_metadata (canonical columns: stim_id, check, visual, strategy)
- load_roi_metadata (standardizes metadata to roi_id + pretty_name/family/color)
- derive_target_chance_from_stimuli (alias and half-target rules; uses CONFIG defaults otherwise)
### stats_utils.py
- welch_ttest, compute_group_mean_and_ci, apply_fdr_correction, compute_cohens_d
- correlate_vectors_bootstrap (hard‑requires pingouin; validates CI columns)
- per_roi_* utilities used by group analyses
### report_utils.py
- generate_latex_table (multicolumn support), create_correlation_table, save_results_metadata, create_figure_summary
### io_utils.py
- find_latest_results_directory, get_all_results_directories, validate_results_directory
- find_nifti_files, split_by_group, discover_files_by_group
- find_subject_tsvs (strict: exactly one TSV per subject)

---

## Behavioral (`chess-behavioral/modules`)

### data_loading.py
- load_trial_data_from_events_tsv, load_participant_trial_data

### rdm_utils.py
- create_pairwise_df, aggregate_pairwise_counts
- compute_symmetric_rdm, compute_directional_dsm
- correlate_with_all_models (delegates to common.rsa_utils)

---

## Manifold (`chess-manifold/modules`)

### data.py
- load_atlas_and_metadata, pivot_pr_long_to_subject_roi, correlate_pr_with_roi_size

### analysis.py
- summarize_pr_by_group, compare_groups_welch_fdr

### models.py
- train_logreg_on_pr, compute_pca_2d, compute_2d_decision_boundary, evaluate_classification_significance

### tables.py
- generate_pr_results_table

---

## MVPA (`chess-mvpa/modules`)

### mvpa_io.py
- load_subject_tsv, build_group_dataframe (sanitizes MATLAB ROI names; strict overlap policy)

### mvpa_group.py
- compute_group_stats_for_method (ROI‑wise desc stats, Welch tests, FDR)

### mvpa_plot_utils.py
- helpers for standardized panels

---

## Neurosynth (`chess-neurosynth/modules`)

### io_utils.py
- load_term_maps, find_group_tmaps, extract_run_label, reorder_by_term

### glm_utils.py
- build_design_matrix (second‑level design used by nilearn)

### maps_utils.py
- t_to_two_tailed_z, split_zmap_by_sign, compute_all_zmap_correlations (bootstrap + FDR)

---

## Notes

- Centralization is intentional: extend `common/` instead of re‑adding ad‑hoc helpers.
- No alias imports or thin wrappers. Import public names directly.
- Strict error policy applies to all loaders and stats; warnings are not used as control flow.

---

## Behavioral Analysis (`chess-behavioral/`)

### `modules/data_loading.py`
| Function | Purpose | Notes |
|----------|---------|-------|
| `load_trial_data_from_events_tsv(events_file)` | Load BIDS events.tsv for one run | Behavioral-specific |
| `load_participant_trial_data(subject_id, is_expert, bids_root)` | Concatenate all runs | Behavioral-specific |
| Note | Participants list & stimuli metadata centralized in `common.bids_utils` | — |

### `modules/rdm_utils.py`
| Function | Purpose | Notes |
|----------|---------|-------|
| `create_pairwise_df(trial_df)` | Convert trials to pairwise comparisons | Behavioral-specific |
| `compute_symmetric_rdm(pairwise_df)` | Compute symmetric RDM | **CANDIDATE** - might be used in neural RSA |
| `compute_directional_dsm(pairwise_df)` | Compute directional dissimilarity matrix | Behavioral-specific |
| `correlate_with_all_models(behavioral_rdm, category_df, model_columns)` | Batch correlate via `common.rsa_utils` | Centralized |
| `aggregate_pairwise_counts(list_of_dfs)` | Aggregate raw pairwise rows to counts | Behavioral-specific |

### `modules/plotting.py`
| Function | Purpose | Notes |
|----------|---------|-------|
| `plot_mds_embedding(dissimilarity_matrix, title, expertise_label, output_path)` | Plot MDS 2D embedding | Behavioral-specific |
| `plot_choice_frequency(pairwise_df, expertise_label, output_path)` | Plot stimulus selection frequency | Behavioral-specific |
| `plot_model_correlations(expert_results, novice_results, output_path)` | Plot expert vs novice correlations | **CANDIDATE** - might use in neural RSA |
| Note | Use `common.plotting_utils.plot_rdm` for all RDM/DSM heatmaps | Centralized |

---

## RSA Analysis (`chess-rsa/`)

**Status**: Not yet implemented

**Expected Functions** (to check for duplication):
- `load_searchlight_results()` - Load RSA searchlight maps
- `extract_roi_rsa()` - Extract RSA values from ROI
- `compute_rsa_second_level()` - Group-level RSA statistics
- `plot_rsa_brain_map()` - Visualize RSA results on brain

**Potential Duplicates**:
- RDM correlation functions → Use `common/rsa_utils.py`
- Model RDM creation → Use `common/rsa_utils.py`
- RDM heatmap plotting → Use `common/plotting_utils.plot_rdm`

---

## MVPA Analysis (`chess-mvpa/`)

**Status**: Not yet implemented

**Expected Functions**:
- `run_svm_classification()` - SVM classification wrapper
- `extract_mvpa_accuracy()` - Extract accuracy from CoSMoMVPA results
- `compute_mvpa_group_stats()` - Group-level statistics
- `plot_decoding_accuracy()` - Visualize decoding results

**Potential Duplicates**:
- ROI data extraction → Use `common/neuro_utils.py`
- Beta image loading → Use `common/neuro_utils.py`
- Group statistics → May need new common function

---

## Manifold Analysis (`chess-manifold/`)

**Status**: Not yet implemented

**Expected Functions**:
- `compute_participation_ratio()` - Calculate PR from beta images
- `perform_pca_on_pr()` - PCA on PR profiles
- `classify_expertise_from_pr()` - Logistic regression on PR
- `plot_pr_comparison()` - Visualize PR differences

**Potential Duplicates**:
- ROI data extraction → Use `common/neuro_utils.py`
- Beta image loading → Use `common/neuro_utils.py`
- Expert vs novice plotting → May adapt behavioral plotting

---

## Neurosynth Analysis (`chess-neurosynth/`)

**Status**: Not yet implemented

**Expected Functions**:
- `load_neurosynth_maps()` - Load Neurosynth term maps
- `correlate_brain_maps()` - Spatial correlation between maps
- `compute_searchlight_rsa()` - Whole-brain searchlight RSA (MATLAB)
- `plot_neurosynth_correlations()` - Visualize meta-analytic results

**Potential Duplicates**:
- Brain map correlation → Related to RDM correlation
- Searchlight RSA → May reuse correlation functions

---

## Dataset Visualization (`chess-dataset-vis/`)

**Status**: Not yet implemented

**Expected Functions**:
- `visualize_chess_stimuli()` - Display chess board images
- `plot_theoretical_rdms()` - Visualize model RDMs
- `create_stimulus_grid()` - Arrange stimuli in grid

**Potential Duplicates**:
- RDM plotting → Use `common/plotting_utils.plot_rdm`
- Model RDM creation → Use `common/rsa_utils.py`

---

## GLM Analysis (`chess-glm/`)

**Status**: Not yet implemented (mostly MATLAB)

**Expected Functions** (Python helpers):
- `check_glm_quality()` - QC for GLM results
- `extract_contrast_values()` - Extract contrast values from images

**Potential Duplicates**:
- NIfTI loading → Use `common/neuro_utils.py`
- Contrast image finding → Use `common/bids_utils.py`

---

## Supplementary Analyses (`supplementary/`)

### `univariate-rois/`
**Expected Functions**:
- `extract_univariate_roi_values()` - Extract mean activation per ROI
- `compute_univariate_group_stats()` - T-tests, FDR correction

**Potential Duplicates**:
- ROI extraction → Use `common/neuro_utils.py`
- Contrast loading → Use `common/bids_utils.py`

### `eyetracking/`
**Expected Functions**:
- `load_bidsmreye_data()` - Load estimated gaze coordinates
- `classify_from_eyetrack()` - SVM classification from gaze

**Potential Duplicates**:
- Classification → May share logic with MVPA

---

## Notes

- **Decision Rule**: If a function is used (or will be used) in 2+ analyses, move to `common/`
- **Naming Convention**: Common functions should be general (e.g., `correlate_rdms()` not `correlate_behavioral_rdms()`)
