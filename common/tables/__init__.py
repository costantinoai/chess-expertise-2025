"""
Common table styling and helpers for LaTeX output.

Central place to keep DRY styling primitives for all table scripts.
"""

from .style import (
    DEFAULT_DECIMALS,
    infer_column_format,
    generate_styled_table,
    build_c_only_colspec,
)

__all__ = [
    'DEFAULT_DECIMALS',
    'infer_column_format',
    'generate_styled_table',
    'build_c_only_colspec',
]
