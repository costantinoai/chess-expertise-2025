"""
Chess-Manifold Analysis Modules

This package contains analysis-specific utilities for manifold dimensionality
analysis using participation ratio (PR).

The participation ratio quantifies the effective dimensionality of neural
population activity. This package provides functions to compute PR, compare
groups, and visualize results.

Modules
-------
pr_computation : Core PR computation from GLM beta weights
analysis : High-level analysis orchestration (group stats, comparisons)
data : Load and transform atlas, ROI, and participant data
models : Machine learning (classification, PCA) on PR features
plotting : Manifold-specific plots (heatmaps, PCA, etc.)
tables : LaTeX table generation for publication
"""

__all__ = [
    'pr_computation',
    'analysis',
    'data',
    'models',
    'plotting',
    'tables',
]
