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


def pretty_model_label(model_key: str) -> str:
    """
    Convert model key to pretty label for display.

    Parameters
    ----------
    model_key : str
        Model key (e.g., 'visual', 'strategy', 'check')

    Returns
    -------
    str
        Pretty label (e.g., 'Visual Similarity', 'Strategy', 'Checkmate')
    """
    from common import CONFIG
    return CONFIG['MODEL_LABELS'].get(model_key, model_key.title())


__all__ = [
    'plot_correlation_bars',
    'plot_variance_partition_bars',
    'pretty_model_label',
]
