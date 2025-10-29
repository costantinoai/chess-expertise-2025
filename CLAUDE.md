# CLAUDE.md

Practical rules for this repo: strict dependencies, no silent fallbacks, DRY, and consistent logging, tables, and plots. Keep this short and actionable.

## Core principles

- DRY by default: centralize shared logic in `common/` and use per‑package `modules/` only for domain specifics.
- Separate analysis and outputs: 01… analysis saves artifacts; 81/82 build LaTeX tables; 91–93 produce figures.
- Reproducibility: every run creates `results/YYYYMMDD-HHMMSS_<analysis>`; `setup_analysis*` logs config, copies the script, and sets the seed.
- Strictness: no thin wrappers, no alias imports, no warnings-as-control‑flow, no hidden fallbacks. Errors should surface.

## Hard requirements and policies

- Dependencies: `pingouin`, `statsmodels`, `nibabel`, `matplotlib`, `seaborn`, `plotly`; `kaleido` for static Plotly (or set `SKIP_PLOTLY_STATIC_EXPORT=1`). Missing deps must raise ImportError.
- Stats: `common.stats_utils` is the single entry-point. Bootstrap correlations must use `pingouin.corr` with CIs; FDR via `statsmodels.multipletests`. No silent NaNs.
- I/O: `common.io_utils.find_subject_tsvs` must find exactly one TSV per subject per method. Multiple files → RuntimeError.
- ROI metadata: always obtain via `common.bids_utils.load_roi_metadata(...)` and join on `roi_id`. Do not assume legacy column names.
- MVPA chance levels: use `common.bids_utils.derive_target_chance_from_stimuli`. Alias `stimuli→stim_id`; any `*_half` target is binary (0.5); otherwise use explicit defaults in `CONFIG['MVPA_SVM_CHANCE_DEFAULTS']`. Missing mapping → ValueError.
- Logging: Always use `setup_analysis(...)` or `setup_analysis_in_dir(...)`. Log start/end, dataset summary, and key results; copy script to results dir.

## File and folder conventions

- Analysis scripts: `01_*`. Table builders: `81_*`. Plots: `91_*` and above.
- All scripts start by enabling imports from repo root and call `setup_analysis*`.
- Figures and tables live inside the corresponding analysis results dir under `figures/` or `tables/`.

## Plotting standards

- Use primitives from `common.plotting` (re-exported at `common`): bars, heatmaps, scatter, surfaces. Call `apply_nature_rc()` before plotting.
- Use color palettes from `common` (e.g., `COLORS_EXPERT_NOVICE`). Do not hard-code.
- Save vector formats; keep labels, ticks, and spines consistent. Avoid one‑off styles in scripts; extend primitives if a new need recurs.

## Minimal template

```python
from pathlib import Path
from common import CONFIG
from common.logging_utils import setup_analysis, log_script_end

config, out_dir, logger = setup_analysis(
    analysis_name="my_analysis",
    results_base=Path("results"),
    script_file=__file__,
)

# … run analysis …

log_script_end(logger)
```

## What to avoid

- try/except that returns NaN or falls back to alternative logic silently.
- Aliasing imports (e.g., `from pingouin import corr as _pg_corr`). Import by their public names.
- Duplicated utilities inside analysis folders that already exist in `common/`.
- Changing analytical logic for the sake of convenience. Keep methods identical; improve structure only.

## Data and configuration

- Paths and parameters come from `common.constants.CONFIG`. Do not mirror values with module‑level copies; access `CONFIG[...]` directly.
- External data root is configured in `constants.py` and points to `/media/.../manuscript-data`. Environment overrides are not used.

## Run order (quick reference)

- Behavioral: 01 → 81 → 91
- Neurosynth: 01/02 → 81/82 → 91/92
- MVPA: 02/03 → 81/82 → 92/93 (subject‑level produced by MATLAB beforehand)
- Manifold: 01 → 81 → 91

That’s it. If a pattern repeats, promote it to `common/` and update callers.

