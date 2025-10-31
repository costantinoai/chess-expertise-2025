"""
Utilities specific to the RDM intercorrelation supplementary analysis.

Modules under this package are intentionally lightweight wrappers that keep the
top-level scripts declarative while re-using shared logic across tables and
figures.
"""

from .plotting import (
    plot_correlation_bars,
    plot_variance_partition_bars,
)

__all__ = [
    'plot_correlation_bars',
    'plot_variance_partition_bars',
]
