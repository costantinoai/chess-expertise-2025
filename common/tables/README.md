# Common Table Styling (siunitx + booktabs)

This folder centralizes LaTeX table styling primitives so all tables in the
repository share the same look and formatting policy (DRY, no thin wrappers).

## Goals
- Single source of truth for LaTeX table generation.
- Decimal-aligned numeric columns using `siunitx` `S`.
- Booktabs rules; avoid `\resizebox` to prevent scaling artifacts.
- Consistent APA/Nature formatting: 3 decimals for estimates, 95% CI as
  `[low, high]`, p-values in cells as "<.001" or ".XYZ" (no leading zero).

## Primitives
- `generate_styled_table(df, output_path, caption, label, multicolumn_headers=None, column_format=None, logger=None, manuscript_name=None)`
  - Wraps `common.report_utils.generate_latex_table` with repository defaults.
  - Infers a `column_format` when not provided via `infer_column_format()`.
- `infer_column_format(df, multicolumn_headers=None, first_col_align='l')`
  - First column uses `l`; numeric columns use `S`; text columns use `c`.
  - If two or more multicolumn groups are provided, inserts a `|` separator
    between the first and second group for readability.

## Typical Usage
```
from common.tables import generate_styled_table
from common.formatters import format_ci, format_p_cell

# Build a DataFrame with a text label and numeric + text columns
rows = [
    {"ROI": "MT+ Complex", "Δr_Visual": 0.123, "95% CI_Visual": format_ci(0.050, 0.180, latex=False),
     "p_Visual": format_p_cell(0.012), "pFDR_Visual": format_p_cell(0.030)},
]
df = pd.DataFrame(rows)

multicolumn = {
    "Visual Similarity": ["Δr_Visual", "95% CI_Visual", "p_Visual", "pFDR_Visual"],
}

tex_path = generate_styled_table(
    df=df,
    output_path=Path("results/tables/example.tex"),
    caption="Example table",
    label="tab:example",
    multicolumn_headers=multicolumn,
)
```

## Formatting Helpers
- `format_ci(low, high, precision=3, latex=False)` → `'[low, high]'`
- `format_p_cell(p)` → `'<.001'` or `'.XYZ'` (3 decimals, no leading zero)
- `shorten_roi_name(name)` → consistent ROI label shortening across tables

## Notes
- Ensure your manuscript preamble includes:
  - `\usepackage{booktabs}`
  - `\usepackage{siunitx}` (configured as needed)
- Keep numeric values as floats in DataFrames to benefit from `S` alignment.
- For special-case tables, keep complex domain logic in the analysis module
  (e.g., `analysis/modules/tables.py`) but call these primitives for rendering.

