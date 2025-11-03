"""
Representational Similarity Analysis (RSA) utilities.

This module provides shared functions for RSA analyses, including:
- Creating model RDMs from stimulus features
- Correlating RDMs with bootstrap confidence intervals
- Computing RDM similarity metrics

These functions are used across behavioral RSA, neural RSA, and meta-analytic
analyses, ensuring consistent methodology.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from .stats_utils import correlate_vectors_bootstrap


def create_model_rdm(
    category_values: np.ndarray,
    is_categorical: bool = True
) -> np.ndarray:
    """
    Create a model RDM from stimulus category or feature values.

    For categorical variables (e.g., checkmate vs non-checkmate), the RDM is
    binary: 0 if same category, 1 if different. For continuous variables,
    the RDM is the absolute difference between values.

    Parameters
    ----------
    category_values : np.ndarray
        1D array of category labels or feature values for each stimulus
    is_categorical : bool, default=True
        If True, treats values as categorical (0/1 RDM)
        If False, treats as continuous (absolute difference RDM)

    Returns
    -------
    np.ndarray
        Model RDM matrix (n_stimuli × n_stimuli)
        - Categorical: 0 if same category, 1 if different
        - Continuous: absolute difference between values
        - Symmetric matrix with zeros on diagonal

    Notes
    -----
    - Model RDMs represent theoretical predictions about stimulus similarity
    - Used to test whether behavioral or neural RDMs reflect specific dimensions
    - Categorical is most common (e.g., checkmate status, strategy type)

    Example
    -------
    >>> # Categorical model (e.g., checkmate status)
    >>> checkmate = np.array([1, 1, 0, 0, 1])  # 1=checkmate, 0=non-checkmate
    >>> rdm_check = create_model_rdm(checkmate, is_categorical=True)
    >>> print(rdm_check)
    [[0 0 1 1 0]
     [0 0 1 1 0]
     [1 1 0 0 1]
     [1 1 0 0 1]
     [0 0 1 1 0]]
    >>>
    >>> # Continuous model (e.g., number of pieces)
    >>> n_pieces = np.array([10, 12, 8, 15, 9])
    >>> rdm_pieces = create_model_rdm(n_pieces, is_categorical=False)
    """
    # Ensure 1D array
    vals = np.asarray(category_values).ravel()

    if is_categorical:
        # Binary RDM: 0 if same category, 1 if different
        rdm = (vals[:, None] != vals[None, :]).astype(int)
    else:
        # Continuous RDM: absolute difference
        rdm = np.abs(vals[:, None] - vals[None, :])

    return rdm


def correlate_rdms(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    method: str = 'pearson',
    n_bootstrap: int = 10000
) -> Tuple[float, float, float, float]:
    """
    Correlate two RDMs using bootstrapped confidence intervals.

    This function computes the correlation between lower triangles of the two
    RDMs (excluding diagonal) and provides bootstrapped 95% confidence intervals.
    This is the standard approach in RSA.

    Parameters
    ----------
    rdm1 : np.ndarray
        First RDM matrix
    rdm2 : np.ndarray
        Second RDM matrix (same size as rdm1)
    method : str, default='pearson'
        Correlation method: 'pearson' or 'spearman'
    n_bootstrap : int, default=10000
        Number of bootstrap samples for CI estimation

    Returns
    -------
    r_value : float
        Correlation coefficient
    p_value : float
        Two-sided p-value
    ci_lower : float
        Lower bound of 95% CI
    ci_upper : float
        Upper bound of 95% CI

    Notes
    -----
    - Uses pingouin's correlation with bootstrap CI
    - Only correlates lower triangle (k=-1) to avoid redundant comparisons
    - Both RDMs should be same size and have matching stimulus order
    - Standard method across behavioral RSA, neural RSA, and meta-analysis

    Example
    -------
    >>> behavioral_rdm = compute_symmetric_rdm(pairwise_df)
    >>> model_rdm = create_model_rdm(checkmate_labels, is_categorical=True)
    >>> r, p, ci_l, ci_u = correlate_rdms(behavioral_rdm, model_rdm)
    >>> print(f"r = {r:.3f}, p = {p:.3e}, 95% CI = [{ci_l:.3f}, {ci_u:.3f}]")
    r = 0.490, p = 1.23e-05, 95% CI = [0.320, 0.650]
    """
    # Ensure RDMs are same size
    if rdm1.shape != rdm2.shape:
        raise ValueError(f"RDMs must be same size. Got {rdm1.shape} and {rdm2.shape}")

    # Extract lower triangles (excluding diagonal)
    n = rdm1.shape[0]
    tri_indices = np.tril_indices(n, k=-1)

    rdm1_vector = rdm1[tri_indices]
    rdm2_vector = rdm2[tri_indices]

    # Compute correlation with bootstrapped CI using unified helper
    r_value, p_value, ci_lower, ci_upper = correlate_vectors_bootstrap(
        rdm1_vector,
        rdm2_vector,
        method=method,
        n_bootstraps=n_bootstrap,
        alternative='two-sided',
    )

    return float(r_value), float(p_value), float(ci_lower), float(ci_upper)


def correlate_rdm_with_models(
    rdm: np.ndarray,
    model_features: dict,
    categorical_features: List[str] = None,
    continuous_features: List[str] = None
) -> Tuple[List[Tuple[str, float, float, float, float]], dict]:
    """
    Correlate an RDM with multiple model RDMs.

    This function creates model RDMs from stimulus features and correlates
    each with the input RDM. Returns both correlation results and the model RDMs.

    Parameters
    ----------
    rdm : np.ndarray
        RDM to correlate with models (n_stimuli × n_stimuli)
    model_features : dict
        Dictionary mapping feature names to arrays of feature values
        Example: {'checkmate': [1,1,0,0], 'strategy': [1,2,1,3]}
    categorical_features : list of str, optional
        Names of categorical features (will create binary RDMs)
    continuous_features : list of str, optional
        Names of continuous features (will create distance RDMs)

    Returns
    -------
    results : list of tuple
        List of (feature_name, r, p, ci_lower, ci_upper) for each model
    model_rdms : dict
        Dictionary mapping feature names to model RDM matrices

    Notes
    -----
    - If neither categorical_features nor continuous_features specified,
      treats all as categorical by default
    - Truncates input RDM to match number of stimuli in model_features
    - Uses correlate_rdms() for consistent correlation methodology

    Example
    -------
    >>> features = {
    ...     'checkmate': np.array([1, 1, 0, 0, 1]),
    ...     'strategy': np.array([1, 2, 1, 3, 2]),
    ...     'n_pieces': np.array([10, 12, 8, 15, 9])
    ... }
    >>> results, models = correlate_rdm_with_models(
    ...     my_rdm,
    ...     features,
    ...     categorical_features=['checkmate', 'strategy'],
    ...     continuous_features=['n_pieces']
    ... )
    >>> for name, r, p, ci_l, ci_u in results:
    ...     print(f"{name}: r={r:.3f}, p={p:.3e}")
    """
    # Determine which features to use
    if categorical_features is None and continuous_features is None:
        # Default: treat all as categorical
        categorical_features = list(model_features.keys())
        continuous_features = []
    elif categorical_features is None:
        categorical_features = []
    elif continuous_features is None:
        continuous_features = []

    # Get number of stimuli from first feature
    first_feature = list(model_features.values())[0]
    n_stimuli = len(first_feature)

    # Truncate RDM to match number of stimuli
    rdm_truncated = rdm[:n_stimuli, :n_stimuli]

    results = []
    model_rdms = {}

    # Process categorical features
    for feature_name in categorical_features:
        if feature_name not in model_features:
            raise ValueError(f"Feature '{feature_name}' not found in model_features")

        feature_values = model_features[feature_name]

        # Create model RDM
        model_rdm = create_model_rdm(feature_values, is_categorical=True)
        model_rdms[feature_name] = model_rdm

        # Correlate with input RDM
        r, p, ci_l, ci_u = correlate_rdms(rdm_truncated, model_rdm)
        results.append((feature_name, r, p, ci_l, ci_u))

    # Process continuous features
    for feature_name in continuous_features:
        if feature_name not in model_features:
            raise ValueError(f"Feature '{feature_name}' not found in model_features")

        feature_values = model_features[feature_name]

        # Create model RDM
        model_rdm = create_model_rdm(feature_values, is_categorical=False)
        model_rdms[feature_name] = model_rdm

        # Correlate with input RDM
        r, p, ci_l, ci_u = correlate_rdms(rdm_truncated, model_rdm)
        results.append((feature_name, r, p, ci_l, ci_u))

    return results, model_rdms


def compute_pairwise_rdm_correlations(
    rdm_dict: Dict[str, np.ndarray],
    method: str = 'spearman'
) -> pd.DataFrame:
    """
    Compute pairwise correlations between multiple RDMs.

    Parameters
    ----------
    rdm_dict : dict[str, np.ndarray]
        Mapping from RDM name to square matrix (same size for all entries)
    method : str, default='spearman'
        Correlation method passed to ``pandas.DataFrame.corr``

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix indexed by RDM names
    """
    if not rdm_dict:
        raise ValueError("rdm_dict must contain at least one RDM")

    names = list(rdm_dict.keys())
    first_shape = rdm_dict[names[0]].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError(f"RDM '{names[0]}' is not a square matrix")

    n_stim = first_shape[0]
    tri_indices = np.triu_indices(n_stim, k=1)

    vectors = {}
    for name, rdm in rdm_dict.items():
        if rdm.shape != first_shape:
            raise ValueError(f"RDM '{name}' shape {rdm.shape} does not match {first_shape}")
        vectors[name] = rdm[tri_indices]

    df = pd.DataFrame(vectors)
    return df.corr(method=method)


def compute_pairwise_rdm_reliability(rdms_list: List[np.ndarray], method: str = 'spearman') -> Tuple[float, float]:
    """
    Compute split-half reliability of RDMs.

    This function computes the average correlation between all pairs of RDMs
    in the list, which estimates the internal consistency/reliability of the
    measurements.

    Parameters
    ----------
    rdms_list : list of np.ndarray
        List of RDM matrices (all same size)
    method : str, default='spearman'
        Correlation method: 'pearson' or 'spearman'

    Returns
    -------
    mean_r : float
        Mean correlation across all pairs
    std_r : float
        Standard deviation of correlations

    Notes
    -----
    - Used for split-half reliability analysis
    - Higher values indicate more consistent representations across individuals
    - Typically computed within each group (experts, novices) separately

    Example
    -------
    >>> # List of individual subject RDMs
    >>> expert_rdms = [rdm1, rdm2, rdm3, ...]
    >>> mean_r, std_r = compute_pairwise_rdm_reliability(expert_rdms)
    >>> print(f"Expert reliability: {mean_r:.3f} ± {std_r:.3f}")
    """
    n_rdms = len(rdms_list)

    if n_rdms < 2:
        raise ValueError("Need at least 2 RDMs to compute reliability")

    # Extract lower triangles from all RDMs
    n_stim = rdms_list[0].shape[0]
    tri_indices = np.tril_indices(n_stim, k=-1)

    rdm_vectors = [rdm[tri_indices] for rdm in rdms_list]

    # Compute all pairwise correlations
    correlations = []
    for i in range(n_rdms):
        for j in range(i + 1, n_rdms):
            if method == 'pearson':
                r = np.corrcoef(rdm_vectors[i], rdm_vectors[j])[0, 1]
            elif method == 'spearman':
                from scipy.stats import spearmanr
                r, _ = spearmanr(rdm_vectors[i], rdm_vectors[j])
            else:
                raise ValueError(f"Unknown method: {method}")

            correlations.append(r)

    mean_r = np.mean(correlations)
    std_r = np.std(correlations)

    return mean_r, std_r


__all__ = [
    'create_model_rdm',
    'correlate_rdms',
    'correlate_rdm_with_models',
    'compute_pairwise_rdm_correlations',
    'compute_pairwise_rdm_reliability',
]
