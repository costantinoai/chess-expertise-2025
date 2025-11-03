"""
Statistical utilities for group comparisons and multiple testing correction.

This module provides reusable statistical functions for neuroimaging analyses,
including Welch t-tests, FDR correction, effect sizes, and confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import ttest_ind, ttest_1samp
from statsmodels.stats.multitest import multipletests
from pingouin import compute_effsize
import pingouin as pg


def welch_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence_level: float = 0.95,
    equal_var: bool = False
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Perform independent samples t-test with confidence interval for the difference.

    Parameters
    ----------
    group1 : np.ndarray
        First group data (e.g., experts)
    group2 : np.ndarray
        Second group data (e.g., novices)
    confidence_level : float, default=0.95
        Confidence level for the CI (0.95 = 95% CI)
    equal_var : bool, default=False
        If False, perform Welch's t-test (unequal variances, more robust).
        If True, perform standard t-test (equal variances, matches old implementation).

    Returns
    -------
    mean1 : float
        Mean of group1
    mean2 : float
        Mean of group2
    mean_diff : float
        Difference in means (group1 - group2)
    ci_low : float
        Lower bound of CI for the difference
    ci_high : float
        Upper bound of CI for the difference
    t_stat : float
        T-statistic
    p_value : float
        Two-tailed p-value

    Notes
    -----
    Returns NaN values if either group has < 2 valid observations.

    Example
    -------
    >>> expert_vals = np.array([5.2, 6.1, 5.8, 6.3])
    >>> novice_vals = np.array([4.1, 3.9, 4.5, 4.2])
    >>> m1, m2, diff, ci_low, ci_high, t, p = welch_ttest(expert_vals, novice_vals)
    >>> # Or with equal variances assumed:
    >>> m1, m2, diff, ci_low, ci_high, t, p = welch_ttest(expert_vals, novice_vals, equal_var=True)
    """
    # Remove NaNs
    g1 = np.asarray(group1)[~np.isnan(group1)]
    g2 = np.asarray(group2)[~np.isnan(group2)]

    # Check sufficient data
    if g1.size < 2 or g2.size < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # Compute means
    mean1 = float(np.mean(g1))
    mean2 = float(np.mean(g2))
    mean_diff = mean1 - mean2

    # T-test with configurable variance assumption
    result = ttest_ind(g1, g2, equal_var=equal_var, nan_policy='omit')
    t_stat = float(result.statistic)
    p_value = float(result.pvalue)

    # Get confidence interval for the difference
    ci = result.confidence_interval(confidence_level=confidence_level)
    ci_low = float(ci.low)
    ci_high = float(ci.high)

    return (mean1, mean2, mean_diff, ci_low, ci_high, t_stat, p_value)


def compute_group_mean_and_ci(
    data: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval for a single group.

    Uses one-sample t-test against zero to compute CI.

    Parameters
    ----------
    data : np.ndarray
        Group data
    confidence_level : float, default=0.95
        Confidence level for the CI

    Returns
    -------
    mean : float
        Mean of the data
    ci_low : float
        Lower bound of CI
    ci_high : float
        Upper bound of CI

    Example
    -------
    >>> data = np.array([5.1, 5.3, 4.9, 5.2, 5.0])
    >>> mean, ci_low, ci_high = compute_group_mean_and_ci(data)
    """
    # Remove NaNs
    d = np.asarray(data)[~np.isnan(data)]

    if d.size == 0:
        return (np.nan, np.nan, np.nan)

    mean = float(np.mean(d))

    if d.size < 2:
        # Can't compute CI with < 2 samples
        return (mean, np.nan, np.nan)

    # Use one-sample t-test against 0 to get CI
    result = ttest_1samp(d, popmean=0)
    ci = result.confidence_interval(confidence_level=confidence_level)

    return (mean, float(ci.low), float(ci.high))


def compute_mean_ci_and_ttest_vs_value(
    data: np.ndarray,
    popmean: float = 0.0,
    alternative: str = 'two-sided',
    confidence_level: float = 0.95,
):
    """
    Compute mean, CI, and one-sample t-test vs a value.

    Parameters
    ----------
    data : array-like
        Sample values (NaNs ignored)
    popmean : float, default=0.0
        Hypothesized mean for one-sample t-test
    alternative : {'two-sided','greater','less'}, default='two-sided'
        Alternative hypothesis direction
    confidence_level : float, default=0.95
        Confidence level for the mean CI (based on t critical value)

    Returns
    -------
    (mean, ci_low, ci_high, t_stat, p_val)

    Notes
    -----
    - CI computed around the sample mean (not around the difference), using t critical value
    - p-value adjusted for one-tailed alternatives from the two-sided p-value
    """
    from scipy.stats import t

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    n = x.size
    mean = float(np.mean(x))
    if n < 2:
        # Not enough samples to compute CI or t-test
        return (mean, np.nan, np.nan, np.nan, np.nan)

    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(t.ppf(0.5 + confidence_level / 2.0, df=n - 1))
    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se

    ttest_result = ttest_1samp(x, popmean=popmean)
    t_stat = float(ttest_result.statistic)
    p_two = float(ttest_result.pvalue)

    if alternative == 'two-sided':
        p_val = p_two
    else:
        half = p_two / 2.0
        if alternative == 'greater':
            p_val = half if t_stat > 0 else (1.0 - half)
        elif alternative == 'less':
            p_val = half if t_stat < 0 else (1.0 - half)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")

    return (mean, float(ci_low), float(ci_high), t_stat, float(p_val))


def binomial_test_accuracy(
    successes: int,
    n_trials: int,
    p_null: float = 0.5,
    alternative: str = 'two-sided',
    confidence_level: float = 0.95,
    ci_method: str = 'wilson',
):
    """
    Exact binomial test for accuracy vs a null proportion, with CI.

    Parameters
    ----------
    successes : int
        Number of correct predictions (successes)
    n_trials : int
        Total number of predictions (trials)
    p_null : float, default=0.5
        Null hypothesis success probability (chance level)
    alternative : {'two-sided','greater','less'}, default='two-sided'
        Alternative hypothesis direction
    confidence_level : float, default=0.95
        Confidence level for binomial proportion CI
    ci_method : {'wilson','exact','jeffreys'}, default='wilson'
        Method for proportion CI (Wilson recommended for accuracy proportions)

    Returns
    -------
    (accuracy, ci_low, ci_high, p_value)

    Notes
    -----
    - Uses scipy.stats.binomtest for the exact test and CI computation.
    - Wilson CI is recommended for binomial proportions due to good coverage.
    """
    import numpy as np
    from scipy.stats import binomtest

    if n_trials <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    if successes < 0 or successes > n_trials:
        raise ValueError("successes must be in [0, n_trials]")

    res = binomtest(k=successes, n=n_trials, p=p_null, alternative=alternative)
    ci = res.proportion_ci(confidence_level=confidence_level, method=ci_method)
    accuracy = successes / n_trials
    return (float(accuracy), float(ci.low), float(ci.high), float(res.pvalue))


def binomial_test_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_null: float = 0.5,
    alternative: str = 'two-sided',
    confidence_level: float = 0.95,
    ci_method: str = 'wilson',
):
    """
    Perform an exact binomial test on pooled out-of-sample predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0/1 or False/True)
    y_pred : array-like
        Predicted binary labels (0/1 or False/True)
    p_null : float, default=0.5
        Null hypothesis success probability
    alternative : {'two-sided','greater','less'}, default='two-sided'
        Alternative hypothesis direction
    confidence_level : float, default=0.95
        Confidence level for binomial proportion CI
    ci_method : {'wilson','exact','jeffreys'}, default='wilson'
        Method for proportion CI

    Returns
    -------
    (accuracy, ci_low, ci_high, p_value, successes, n_trials)

    Notes
    -----
    - Treats each pooled out-of-fold prediction as a Bernoulli trial.
    - For cross-validated decoding, this complements fold-wise t-tests.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    successes = int(np.sum(y_true == y_pred))
    n_trials = int(y_true.size)
    acc, lo, hi, p = binomial_test_accuracy(
        successes, n_trials, p_null=p_null, alternative=alternative,
        confidence_level=confidence_level, ci_method=ci_method
    )
    return (acc, lo, hi, p, successes, n_trials)


def apply_fdr_correction(
    pvalues: np.ndarray,
    alpha: float = 0.05,
    method: str = 'fdr_bh'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply FDR correction to p-values.

    Uses Benjamini-Hochberg procedure by default.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of p-values
    alpha : float, default=0.05
        Family-wise error rate
    method : str, default='fdr_bh'
        Method for multiple testing correction
        Options: 'fdr_bh' (Benjamini-Hochberg), 'fdr_by' (Benjamini-Yekutieli),
                 'bonferroni', 'holm', etc.

    Returns
    -------
    reject : np.ndarray (bool)
        Boolean array indicating which hypotheses are rejected
    pvalues_corrected : np.ndarray
        FDR-corrected p-values

    Example
    -------
    >>> pvals = np.array([0.001, 0.04, 0.03, 0.5, 0.08])
    >>> reject, pvals_fdr = apply_fdr_correction(pvals, alpha=0.05)
    """
    # Handle NaNs by replacing with 1.0 (non-significant)
    pvals = np.asarray(pvalues).copy()
    nan_mask = np.isnan(pvals)
    pvals[nan_mask] = 1.0

    # Apply correction (requires statsmodels)
    reject, pvals_corrected, _, _ = multipletests(
        pvals,
        alpha=alpha,
        method=method
    )

    # Restore NaNs
    pvals_corrected[nan_mask] = np.nan

    return reject, pvals_corrected


def compute_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray
) -> float:
    """
    Compute Cohen's d effect size for two independent groups.

    Parameters
    ----------
    group1 : np.ndarray
        First group data
    group2 : np.ndarray
        Second group data

    Returns
    -------
    d : float
        Cohen's d effect size

    Notes
    -----
    Uses pooled standard deviation.
    Returns NaN if either group has < 2 observations.

    Example
    -------
    >>> g1 = np.array([5.2, 6.1, 5.8, 6.3])
    >>> g2 = np.array([4.1, 3.9, 4.5, 4.2])
    >>> d = compute_cohens_d(g1, g2)
    """
    # Remove NaNs
    g1 = np.asarray(group1)[~np.isnan(group1)]
    g2 = np.asarray(group2)[~np.isnan(group2)]

    if g1.size < 2 or g2.size < 2:
        return np.nan

    # Pingouin compute_effsize (required dependency)
    d = compute_effsize(g1, g2, eftype='cohen', paired=False)
    return float(d)


def correlate_vectors_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson',
    n_bootstraps: int = 10000,
    alternative: str = 'two-sided'
) -> Tuple[float, float, float, float]:
    """
    Bootstrap correlation between two 1D arrays with 95% CI.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of equal length (NaNs handled by Pingouin).
    method : {'pearson','spearman'}, default 'pearson'
    n_bootstraps : int, default 10000
        Number of bootstrap samples (only used for Pearson; Spearman does not support bootstrap in pingouin)
    alternative : str, default 'two-sided'

    Returns
    -------
    r : float, p : float, ci_low : float, ci_high : float
        For Spearman, ci_low and ci_high are np.nan (pingouin does not support bootstrap CIs for Spearman)

    Notes
    -----
    Pingouin's pg.corr() does not support bootstrap parameter for Spearman correlations.
    When method='spearman', this function computes correlation and p-value but returns np.nan for CIs.
    """
    if pg is None:
        raise ImportError(
            "correlate_vectors_bootstrap requires 'pingouin'. Install it to compute bootstrap CIs."
        )

    # Spearman does not support bootstrap in pingouin - compute without bootstrap
    if method == 'spearman':
        corr_result = pg.corr(
            x=x,
            y=y,
            method=method,
            alternative=alternative,
        )
        # Validate expected columns strictly (no silent fallbacks)
        required_cols = {'r', 'p-val'}
        missing = required_cols - set(corr_result.columns)
        if missing:
            raise RuntimeError(
                f"pingouin.corr missing expected columns: {sorted(missing)}; got columns={list(corr_result.columns)}"
            )
        r = float(corr_result['r'].iloc[0])
        p = float(corr_result['p-val'].iloc[0])
        # No bootstrap CIs available for Spearman in pingouin
        return r, p, np.nan, np.nan

    # Pearson with bootstrap CIs
    corr_result = pg.corr(
        x=x,
        y=y,
        method=method,
        bootstraps=n_bootstraps,
        confidence=0.95,
        method_ci='percentile',
        alternative=alternative,
    )
    # Validate expected columns strictly (no silent fallbacks)
    required_cols = {'r', 'p-val', 'CI95%'}
    missing = required_cols - set(corr_result.columns)
    if missing:
        raise RuntimeError(
            f"pingouin.corr missing expected columns: {sorted(missing)}; got columns={list(corr_result.columns)}"
        )
    r = float(corr_result['r'].iloc[0])
    p = float(corr_result['p-val'].iloc[0])
    v = corr_result['CI95%'].iloc[0]
    if not (hasattr(v, '__len__') and len(v) == 2):
        raise RuntimeError("pingouin.corr returned CI95% not parseable as (low, high)")
    try:
        ci_low, ci_high = float(v[0]), float(v[1])
    except Exception as e:
        raise RuntimeError(f"Failed to parse CI95% values: {v}") from e
    return r, p, ci_low, ci_high




def partial_correlation_rdms(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    covariate_rdms: list,
    method: str = 'spearman'
) -> dict:
    """
    Compute partial correlation between two RDMs controlling for covariate RDMs.

    Uses pingouin.partial_corr on flattened RDM vectors (upper triangles).

    Parameters
    ----------
    rdm1, rdm2 : np.ndarray
        RDMs to correlate (2D square matrices)
    covariate_rdms : list of np.ndarray
        Covariate RDMs to control for
    method : str, default='spearman'
        'spearman' or 'pearson'

    Returns
    -------
    dict : 'r' (correlation), 'p' (p-value), 'dof' (degrees of freedom)
    """
    if rdm1.shape != rdm2.shape or rdm1.shape[0] != rdm1.shape[1]:
        raise ValueError("RDMs must be square matrices of the same shape")

    for cov_rdm in covariate_rdms:
        if cov_rdm.shape != rdm1.shape:
            raise ValueError("All covariate RDMs must have the same shape")

    if method not in ['spearman', 'pearson']:
        raise ValueError(f"Method must be 'spearman' or 'pearson', got: {method}")

    # Vectorize RDMs by extracting upper triangles (unique pairwise dissimilarities).
    # For a 40×40 RDM, this gives 780 observations. We treat these as independent
    # observations for correlation analysis (standard RSA practice, though entries
    # sharing stimuli are technically dependent).
    triu_indices = np.triu_indices(rdm1.shape[0], k=1)
    data_dict = {
        'x': rdm1[triu_indices],
        'y': rdm2[triu_indices]
    }
    for i, cov_rdm in enumerate(covariate_rdms):
        data_dict[f'cov{i}'] = cov_rdm[triu_indices]

    df = pd.DataFrame(data_dict)
    covar_cols = [f'cov{i}' for i in range(len(covariate_rdms))]

    # Compute partial correlation using pingouin. This implements the standard
    # partial correlation procedure: regress x on covariates (extract residuals),
    # regress y on covariates (extract residuals), then correlate the two residual
    # vectors. This isolates the unique relationship between x and y after removing
    # shared variance with covariates.
    result = pg.partial_corr(
        data=df,
        x='x',
        y='y',
        covar=covar_cols if len(covar_cols) > 1 else covar_cols[0],
        method=method
    )

    return {
        'r': float(result['r'].iloc[0]),
        'p': float(result['p-val'].iloc[0]),
        'dof': int(result['dof'].iloc[0]) if 'dof' in result.columns else np.nan
    }


def variance_partitioning_rdms(
    target_rdm: np.ndarray,
    predictor_rdms_dict: dict
) -> pd.DataFrame:
    """
    Decompose target RDM variance using nested linear regression (matching old implementation).

    Parameters
    ----------
    target_rdm : np.ndarray
        Target RDM (2D square matrix)
    predictor_rdms_dict : dict
        Dictionary mapping predictor names to RDM arrays

    Returns
    -------
    pd.DataFrame : Single-row DataFrame with variance components
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    if not predictor_rdms_dict:
        raise ValueError("predictor_rdms_dict cannot be empty")
    if target_rdm.shape[0] != target_rdm.shape[1]:
        raise ValueError("target_rdm must be a square matrix")

    for name, rdm in predictor_rdms_dict.items():
        if rdm.shape != target_rdm.shape:
            raise ValueError(f"Predictor RDM '{name}' shape doesn't match target")

    # Vectorize RDMs to 1D arrays (upper triangles). Treat each pairwise dissimilarity
    # as an observation. For 40×40 RDMs, we get 780 observations per RDM.
    triu_indices = np.triu_indices(target_rdm.shape[0], k=1)
    y = target_rdm[triu_indices]

    predictor_names = list(predictor_rdms_dict.keys())
    X = np.column_stack([
        predictor_rdms_dict[name][triu_indices]
        for name in predictor_names
    ])

    # Fit full regression model: target ~ all predictors
    # This gives total variance explained when all predictors are included together.
    full_model = LinearRegression().fit(X, y)
    r2_full = r2_score(y, full_model.predict(X))

    # Hierarchical variance decomposition: compute unique contribution of each predictor
    # by comparing full model R² vs reduced model R² (model without that predictor).
    # Unique variance for predictor k = R²_full - R²_reduced(without k)
    # This measures how much variance is uniquely explained by predictor k that cannot
    # be explained by the other predictors.
    unique_variances = {}
    single_r2s = {}

    for i, pred_name in enumerate(predictor_names):
        reduced_indices = [j for j in range(len(predictor_names)) if j != i]

        if reduced_indices:
            # Fit reduced model without predictor i
            X_reduced = X[:, reduced_indices]
            reduced_model = LinearRegression().fit(X_reduced, y)
            r2_reduced = r2_score(y, reduced_model.predict(X_reduced))
        else:
            # If only one predictor, reduced model explains zero variance
            r2_reduced = 0.0

        unique_variances[pred_name] = max(r2_full - r2_reduced, 0.0)

        # Also fit single-predictor model to report marginal R² (predictor alone)
        X_single = X[:, i:i+1]
        single_model = LinearRegression().fit(X_single, y)
        single_r2s[pred_name] = r2_score(y, single_model.predict(X_single))

    # Shared variance: variance explained jointly by multiple predictors that cannot
    # be attributed to any single predictor uniquely. Computed as:
    # R²_shared = R²_full - sum(unique variances)
    total_unique = sum(unique_variances.values())
    shared = max(r2_full - total_unique, 0.0)

    # Residual variance: unexplained variance after accounting for all predictors
    residual = max(1.0 - r2_full, 0.0)

    result = {'r2_full': r2_full, 'shared': shared, 'residual': residual}
    for pred_name, unique_r2 in unique_variances.items():
        result[f'unique_{pred_name}'] = unique_r2
    for pred_name, single_r2 in single_r2s.items():
        result[f'{pred_name}_only_r2'] = single_r2

    return pd.DataFrame([result])


__all__ = [
    'welch_ttest',
    'compute_group_mean_and_ci',
    'compute_mean_ci_and_ttest_vs_value',
    'binomial_test_accuracy',
    'binomial_test_from_predictions',
    'apply_fdr_correction',
    'compute_cohens_d',
    'correlate_vectors_bootstrap',
    'partial_correlation_rdms',
    'variance_partitioning_rdms',
    'ci_to_errorbar_format',
    'per_roi_welch_and_fdr',
    'per_roi_one_sample_vs_value',
]


def ci_to_errorbar_format(
    means: np.ndarray,
    ci_lows: np.ndarray,
    ci_highs: np.ndarray
) -> np.ndarray:
    """
    Convert mean and CIs to matplotlib errorbar format.

    Matplotlib's errorbar expects [[lower_errors], [upper_errors]]
    where errors are distances from the mean.

    Parameters
    ----------
    means : np.ndarray
        Mean values
    ci_lows : np.ndarray
        Lower bounds of CIs
    ci_highs : np.ndarray
        Upper bounds of CIs

    Returns
    -------
    yerr : np.ndarray
        Shape (2, n) array for matplotlib errorbar

    Example
    -------
    >>> means = np.array([5.0, 6.0])
    >>> ci_lows = np.array([4.5, 5.3])
    >>> ci_highs = np.array([5.5, 6.7])
    >>> yerr = ci_to_errorbar_format(means, ci_lows, ci_highs)
    >>> # yerr = [[0.5, 0.7], [0.5, 0.7]]
    """
    lower_errors = means - ci_lows
    upper_errors = ci_highs - means

    # Handle NaNs
    lower_errors = np.where(np.isnan(lower_errors), 0, lower_errors)
    upper_errors = np.where(np.isnan(upper_errors), 0, upper_errors)

    return np.array([lower_errors, upper_errors])


 


def per_roi_welch_and_fdr(
    expert_vals: np.ndarray,
    novice_vals: np.ndarray,
    roi_labels: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = False
) -> pd.DataFrame:
    """
    Run t-tests per ROI with FDR correction and effect sizes.

    For each ROI, performs:
    - Independent samples t-test comparing expert vs. novice groups
    - Cohen's d effect size computation
    - Benjamini-Hochberg FDR correction across all ROIs

    Parameters
    ----------
    expert_vals : np.ndarray
        Expert group data, shape (n_experts, n_rois)
    novice_vals : np.ndarray
        Novice group data, shape (n_novices, n_rois)
    roi_labels : np.ndarray
        1D array of ROI labels corresponding to columns
    alpha : float, default=0.05
        FDR alpha level for significance testing
    equal_var : bool, default=False
        If False, use Welch's t-test (unequal variances, more robust).
        If True, use standard t-test (equal variances, matches old implementation).

    Returns
    -------
    pd.DataFrame
        Statistical results with columns:
        - ROI_Label: ROI identifier
        - t_stat: T-statistic from Welch test
        - p_val: Raw p-value (two-tailed)
        - dof: Degrees of freedom
        - cohen_d: Cohen's d effect size
        - mean_diff: Mean difference (expert - novice)
        - ci95_low: Lower bound of 95% CI for difference
        - ci95_high: Upper bound of 95% CI for difference
        - p_val_fdr: FDR-corrected p-value
        - significant: Boolean, p_val < alpha
        - significant_fdr: Boolean, p_val_fdr < alpha

    Notes
    -----
    - ROIs with insufficient data (<2 valid subjects per group) will have NaN values
    - FDR correction uses Benjamini-Hochberg procedure
    - Cohen's d is computed using pooled standard deviation for independent samples

    Example
    -------
    >>> # Expert and novice PR values for 22 ROIs
    >>> expert_pr = np.random.randn(20, 22) + 0.5  # 20 experts
    >>> novice_pr = np.random.randn(24, 22)         # 24 novices
    >>> roi_labels = np.arange(1, 23)
    >>>
    >>> results = per_roi_welch_and_fdr(expert_pr, novice_pr, roi_labels)
    >>> print(f"Significant ROIs (FDR): {results['significant_fdr'].sum()}")
    """
    results = []

    # Run Welch t-test for each ROI
    for roi_idx, roi_label in enumerate(roi_labels):
        expert_roi = expert_vals[:, roi_idx]
        novice_roi = novice_vals[:, roi_idx]

        # Remove NaNs for this ROI
        expert_clean = expert_roi[~np.isnan(expert_roi)]
        novice_clean = novice_roi[~np.isnan(novice_roi)]

        # Check sufficient data
        if expert_clean.size < 2 or novice_clean.size < 2:
            # Insufficient data - fill with NaNs
            results.append({
                'ROI_Label': int(roi_label),
                't_stat': np.nan,
                'p_val': np.nan,
                'dof': np.nan,
                'cohen_d': np.nan,
                'cohen_d_ci_low': np.nan,
                'cohen_d_ci_high': np.nan,
                'mean_diff': np.nan,
                'ci95_low': np.nan,
                'ci95_high': np.nan,
            })
            continue

        # T-test with configurable variance assumption
        mean1, mean2, mean_diff, ci_low, ci_high, t_stat, p_val = welch_ttest(
            expert_clean, novice_clean, confidence_level=0.95, equal_var=equal_var
        )

        # Degrees of freedom
        result_obj = ttest_ind(expert_clean, novice_clean, equal_var=equal_var)
        if hasattr(result_obj, 'df'):
            dof = result_obj.df
        else:
            # Compute df manually
            n1, n2 = expert_clean.size, novice_clean.size
            if equal_var:
                # Standard t-test: n1 + n2 - 2
                dof = n1 + n2 - 2
            else:
                # Welch-Satterthwaite approximation
                v1, v2 = np.var(expert_clean, ddof=1), np.var(novice_clean, ddof=1)
                num = (v1/n1 + v2/n2) ** 2
                den = ((v1**2)/((n1**2)*(n1-1))) + ((v2**2)/((n2**2)*(n2-1)))
                dof = num/den if den > 0 else np.nan

        # Cohen's d (pingouin)
        cohen_d = compute_effsize(expert_clean, novice_clean, eftype='cohen')
        # Approximate 95% CI for Cohen's d (Hedges & Olkin variance)
        n1, n2 = expert_clean.size, novice_clean.size
        denom = max(n1 + n2 - 2, 1)
        var_d = ((n1 + n2) / (n1 * n2)) + (cohen_d ** 2) / (2 * denom)
        se_d = np.sqrt(var_d) if var_d >= 0 else np.nan
        z_crit = 1.96
        d_ci_low = float(cohen_d - z_crit * se_d) if np.isfinite(se_d) else np.nan
        d_ci_high = float(cohen_d + z_crit * se_d) if np.isfinite(se_d) else np.nan

        results.append({
            'ROI_Label': int(roi_label),
            't_stat': t_stat,
            'p_val': p_val,
            'dof': float(dof),
            'cohen_d': float(cohen_d),
            'cohen_d_ci_low': d_ci_low,
            'cohen_d_ci_high': d_ci_high,
            'mean_diff': mean_diff,
            'ci95_low': ci_low,
            'ci95_high': ci_high,
        })

    # Convert to DataFrame
    stats_df = pd.DataFrame(results)

    # Apply FDR correction
    pvals = stats_df['p_val'].values
    valid_pvals = ~np.isnan(pvals)

    if np.sum(valid_pvals) > 0:
        # Only correct valid p-values
        reject, pvals_corrected = apply_fdr_correction(pvals[valid_pvals], alpha=alpha)

        # Initialize FDR columns with NaN
        stats_df['p_val_fdr'] = np.nan
        stats_df['significant'] = False
        stats_df['significant_fdr'] = False

        # Fill in corrected values
        stats_df.loc[valid_pvals, 'p_val_fdr'] = pvals_corrected
        stats_df.loc[valid_pvals, 'significant'] = pvals[valid_pvals] < alpha
        stats_df.loc[valid_pvals, 'significant_fdr'] = reject
    else:
        # No valid p-values
        stats_df['p_val_fdr'] = np.nan
        stats_df['significant'] = False
        stats_df['significant_fdr'] = False

    return stats_df


def per_roi_one_sample_vs_value(
    group_values: np.ndarray,
    roi_names: list,
    value: float,
    alpha: float = 0.05,
    alternative: str = 'greater',
) -> pd.DataFrame:
    """
    One-sample tests per ROI vs a constant value, with FDR correction.

    Parameters
    ----------
    group_values : np.ndarray
        Shape (n_subjects, n_rois)
    roi_names : list of str
        ROI display names (length n_rois)
    value : float
        Hypothesized mean (e.g., chance level)
    alpha : float, default=0.05
        FDR alpha
    alternative : {'greater','less','two-sided'}
        Tail for the one-sample test

    Returns
    -------
    pd.DataFrame
        Columns: ROI_Name, mean, sem, t_stat, p_val, p_val_fdr, significant, significant_fdr
    """
    # Compute means and SEMs (NaN-safe)
    means = np.nanmean(group_values, axis=0)
    sem = np.nanstd(group_values, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(group_values), axis=0))

    # Run one-sample tests per ROI
    t_stats = np.full(group_values.shape[1], np.nan, dtype=float)
    p_two_sided = np.full(group_values.shape[1], np.nan, dtype=float)

    for i in range(group_values.shape[1]):
        x = group_values[:, i]
        x = x[~np.isnan(x)]
        if x.size < 2:
            continue
        ttest_result = ttest_1samp(x, popmean=value)
        t_stats[i] = float(ttest_result.statistic)
        p_two_sided[i] = float(ttest_result.pvalue)

    # Tail adjustment
    if alternative == 'two-sided':
        p_vals = p_two_sided
    else:
        half = p_two_sided / 2.0
        if alternative == 'greater':
            p_vals = np.where(t_stats > 0, half, 1.0 - half)
        elif alternative == 'less':
            p_vals = np.where(t_stats < 0, half, 1.0 - half)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")

    # FDR
    valid = ~np.isnan(p_vals)
    p_fdr = np.full_like(p_vals, np.nan)
    reject = np.zeros_like(valid, dtype=bool)
    if np.any(valid):
        rej, p_corr = apply_fdr_correction(p_vals[valid], alpha=alpha)
        p_fdr[valid] = p_corr
        reject[valid] = rej

    df = pd.DataFrame({
        'ROI_Name': roi_names,
        'mean': means,
        'sem': sem,
        't_stat': t_stats,
        'p_val': p_vals,
        'p_val_fdr': p_fdr,
        'significant': (p_vals < alpha),
        'significant_fdr': reject,
    })
    return df
