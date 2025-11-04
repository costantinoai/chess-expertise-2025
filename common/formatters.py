"""
Formatting helpers shared across reporting and plotting.

Contains plain-text and LaTeX-friendly formatters to avoid duplication
between stats, reporting, and plotting modules.

This module also centralizes table cell formatting policies to keep
LaTeX tables consistent repository‑wide (APA/Nature style):
- Estimates to 3 decimals by default
- 95% CI shown as "[low, high]" with matching precision
- p-values as "<.001" or ".XYZ" (no leading zero) when used in a p column
  (headers carry the math markup, cells remain plain text)
"""



def format_pvalue_plain(p: float, threshold: float = 0.001) -> str:
    """Format p-value for plain text display (no LaTeX)."""
    if p is None:
        raise ValueError("p-value is None")
    if p < threshold:
        return f'< {threshold}'
    return f'{p:.3f}'


def format_pvalue_latex(p: float, threshold: float = 0.001) -> str:
    """Format p-value for LaTeX tables (wrapped in $...$)."""
    if p is None:
        raise ValueError("p-value is None")
    if p < threshold:
        return r"$<.001$"
    # APA style: no leading zero
    return f"$.{int(round(p * 1000)):03d}$"


def format_ci(ci_lower: float, ci_upper: float, precision: int = 3, latex: bool = True, use_numrange: bool = False) -> str:
    """
    Format confidence interval as [low, high] or \\numrange{low}{high}.

    Parameters
    ----------
    ci_lower : float
        Lower bound of CI
    ci_upper : float
        Upper bound of CI
    precision : int
        Number of decimal places
    latex : bool
        If True and use_numrange=False, wrap in $...$
    use_numrange : bool
        If True, use \\numrange{low}{high} for siunitx compatibility (recommended for LaTeX tables)

    Returns
    -------
    str
        Formatted CI string
    """
    fmt = f"{{:.{precision}f}}"
    if use_numrange:
        # Use \numrange for siunitx compatibility (no wrapper needed)
        return f"\\numrange{{{fmt.format(ci_lower)}}}{{{fmt.format(ci_upper)}}}"
    else:
        # Traditional bracket notation
        content = f"[{fmt.format(ci_lower)}, {fmt.format(ci_upper)}]"
        return f"${content}$" if latex else content


def significance_stars(p_value: float) -> str:
    """Map p-value to significance stars for plots."""
    if p_value is None:
        return ''
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''


def format_p_cell(p: float, threshold: float = 0.001, decimals: int = 3, leading_zero: bool = False) -> str:
    """
    Format p-value for table cells (non-math), following APA style.

    Returns "<.001" if p < threshold, else ".XYZ" (or "0.XXX" if leading_zero=True).
    If p is NaN/None, returns "--".
    """
    try:
        if p is None:
            return "--"
        # Handle NaN
        if p != p:  # NaN check
            return "--"
        if p < threshold:
            return "<.001"
        val = f"{p:.{decimals}f}"
        return val if leading_zero else val[1:] if val.startswith('0') else val
    except Exception:
        return "--"


def shorten_roi_name(name: str) -> str:
    """
    Apply consistent shortening rules to long ROI names for tables.

    Centralizes the ad-hoc replacements previously scattered across scripts.
    """
    if name is None:
        return name
    out = name.replace('\n', ' ')
    replacements = {
        'Paracentral Lobular and Mid Cingulate': 'Paracentral Lob. and Mid Cing.',
        'Paracentral Lobule and Mid Cingulate': 'Paracentral Lob. and Mid Cing.',
        'Insular and Frontal Opercular': 'Insular and Frontal Operc.',
        'Temporo-Parieto-Occipital Junction': 'Temporo-Parieto-Occipital J.',
        'Temporo-Parieto Occipital Junction': 'Temporo-Parieto-Occipital J.',
        'Anterior Cingulate and Medial PFC': 'Anterior Cing. and Medial PFC',
        'MT+ Complex': 'MT+ Complex Visual',
    }
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out
