"""
Formatting helpers shared across reporting and plotting.

Contains plain-text and LaTeX-friendly formatters to avoid duplication
between stats, reporting, and plotting modules.
"""

from typing import Optional


def format_pvalue_plain(p: float, threshold: float = 0.001) -> str:
    """Format p-value for plain text display (no LaTeX)."""
    if p is None:
        return 'NaN'
    try:
        if p < threshold:
            return f'< {threshold}'
        return f'{p:.3f}'
    except Exception:
        return 'NaN'


def format_pvalue_latex(p: float, threshold: float = 0.001) -> str:
    """Format p-value for LaTeX tables (wrapped in $...$)."""
    if p is None:
        return r"$\mathrm{NaN}$"
    try:
        if p < threshold:
            return rf"$< {threshold}$"
        return f"${p:.3f}$"
    except Exception:
        return r"$\mathrm{NaN}$"


def format_ci(ci_lower: float, ci_upper: float, precision: int = 3, latex: bool = True) -> str:
    """Format confidence interval as [low, high]; wrap in $...$ if latex=True."""
    fmt = f"{{:.{precision}f}}"
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

