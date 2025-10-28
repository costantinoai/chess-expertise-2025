"""
Statistical utilities for group comparisons and multiple testing correction.

This module provides reusable statistical functions for neuroimaging analyses,
including Welch t-tests, FDR correction, effect sizes, and confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.stats import ttest_ind, ttest_1samp
try:
    from statsmodels.stats.multitest import multipletests as _smm_multipletests
except Exception:
    _smm_multipletests = None
try:
    from pingouin import compute_effsize as _pg_compute_effsize
    import pingouin as pg
except Exception:
    _pg_compute_effsize = None
    class _PGStub:
        def corr(self, *a, **k):
            return pd.DataFrame({'r': [np.nan], 'p-val': [np.nan], 'CI95%': [(np.nan, np.nan)]})
    pg = _PGStub()
from .formatters import format_pvalue_plain as _format_pvalue_plain


def welch_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Perform Welch's t-test (unequal variances) with confidence interval for the difference.

    Parameters
    ----------
    group1 : np.ndarray
        First group data (e.g., experts)
    group2 : np.ndarray
        Second group data (e.g., novices)
    confidence_level : float, default=0.95
        Confidence level for the CI (0.95 = 95% CI)

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

    # Welch t-test (equal_var=False)
    result = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
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

    # Apply correction
    if _smm_multipletests is not None:
        reject, pvals_corrected, _, _ = _smm_multipletests(
            pvals,
            alpha=alpha,
            method=method
        )
    else:
        # Simple BH implementation
        order = np.argsort(pvals)
        ranked = np.empty_like(pvals)
        ranked[order] = np.arange(1, len(pvals) + 1)
        pvals_corrected = pvals * (len(pvals) / ranked)
        # Ensure monotonicity
        pvals_corrected[order] = np.minimum.accumulate(pvals_corrected[order][::-1])[::-1]
        reject = pvals_corrected < alpha

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

    # Use pingouin for standardized calculation
    if _pg_compute_effsize is not None:
        d = _pg_compute_effsize(g1, g2, eftype='cohen', paired=False)
    else:
        # Manual Cohen's d (unbiased not needed here)
        n1, n2 = g1.size, g2.size
        s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = (np.mean(g1) - np.mean(g2)) / sp if sp > 0 else np.nan
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
    alternative : str, default 'two-sided'

    Returns
    -------
    r : float, p : float, ci_low : float, ci_high : float
    """
    res = pg.corr(
        x=x,
        y=y,
        method=method,
        bootstraps=n_bootstraps,
        confidence=0.95,
        method_ci='percentile',
        alternative=alternative,
    )
    r = float(res['r'].iloc[0]) if 'r' in res.columns else float('nan')
    p = float(res['p-val'].iloc[0]) if 'p-val' in res.columns else float('nan')
    ci_val = res.get('CI95%')
    ci_low = float('nan')
    ci_high = float('nan')
    if ci_val is not None:
        v = ci_val.iloc[0]
        try:
            # Expect a tuple/list-like of length 2
            if hasattr(v, '__len__') and len(v) == 2:
                ci_low, ci_high = float(v[0]), float(v[1])
        except Exception:
            ci_low = float('nan')
            ci_high = float('nan')
    return r, p, ci_low, ci_high


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """
    Format p-value for display in tables or plots.

    Parameters
    ----------
    p : float
        P-value to format
    threshold : float, default=0.001
        Threshold below which to display as "< threshold"

    Returns
    -------
    formatted : str
        Formatted p-value string

    Example
    -------
    >>> format_pvalue(0.0001)
    '< 0.001'
    >>> format_pvalue(0.0234)
    '0.023'
    >>> format_pvalue(0.456)
    '0.456'
    """
    # Delegate to centralized formatter for consistency
    try:
        return _format_pvalue_plain(float(p), threshold=threshold)
    except Exception:
        return 'NaN'


__all__ = [
    'welch_ttest',
    'compute_group_mean_and_ci',
    'apply_fdr_correction',
    'compute_cohens_d',
    'format_pvalue',
    'correlate_vectors_bootstrap',
]


def run_roi_group_comparison(
    group1_values: np.ndarray,
    group2_values: np.ndarray,
    roi_labels: np.ndarray,
    alpha_fdr: float = 0.05,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Run group comparisons across multiple ROIs with FDR correction.

    Performs Welch t-tests for each ROI and applies FDR correction across ROIs.

    Parameters
    ----------
    group1_values : np.ndarray
        Shape (n_subjects_group1, n_rois)
    group2_values : np.ndarray
        Shape (n_subjects_group2, n_rois)
    roi_labels : np.ndarray
        ROI labels (length n_rois)
    alpha_fdr : float, default=0.05
        FDR significance level
    confidence_level : float, default=0.95
        Confidence level for CIs

    Returns
    -------
    results : pd.DataFrame
        Columns: ROI_Label, mean_group1, mean_group2, mean_diff,
                 ci_low, ci_high, t_stat, p_val, p_val_fdr,
                 cohen_d, significant, significant_fdr

    Example
    -------
    >>> expert_pr = np.random.rand(20, 22)  # 20 experts, 22 ROIs
    >>> novice_pr = np.random.rand(20, 22)  # 20 novices, 22 ROIs
    >>> roi_labels = np.arange(1, 23)
    >>> results = run_roi_group_comparison(expert_pr, novice_pr, roi_labels)
    """
    n_rois = len(roi_labels)

    # Initialize storage
    results = []

    for i, roi in enumerate(roi_labels):
        g1 = group1_values[:, i]
        g2 = group2_values[:, i]

        # Welch t-test with CI
        m1, m2, diff, ci_low, ci_high, t_stat, p_val = welch_ttest(
            g1, g2, confidence_level=confidence_level
        )

        # Effect size
        d = compute_cohens_d(g1, g2)

        results.append({
            'ROI_Label': int(roi),
            'mean_group1': m1,
            'mean_group2': m2,
            'mean_diff': diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            't_stat': t_stat,
            'p_val': p_val,
            'cohen_d': d
        })

    df = pd.DataFrame(results)

    # Apply FDR correction
    reject, pvals_fdr = apply_fdr_correction(df['p_val'].values, alpha=alpha_fdr)
    df['p_val_fdr'] = pvals_fdr
    df['significant'] = df['p_val'] < 0.05
    df['significant_fdr'] = reject

    return df


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


def compute_classification_chance_level(n_classes: int) -> float:
    """
    Compute chance level for multi-class classification.

    For balanced classification tasks, chance level is 1 / n_classes.
    This represents the expected accuracy if a classifier were to guess
    randomly with uniform probability across all classes.

    Parameters
    ----------
    n_classes : int
        Number of classes in the classification task
        Must be >= 2

    Returns
    -------
    float
        Chance level accuracy (between 0 and 1)

    Raises
    ------
    ValueError
        If n_classes < 2

    Notes
    -----
    - Assumes balanced classes (equal number of trials per class)
    - All MVPA/decoding analyses in this project use balanced classes via resampling
    - Returned value is accuracy (e.g., 0.5 for binary, 0.25 for 4-class)
    - This is the correct way to compute chance level (NOT hardcoded)

    Examples
    --------
    >>> # Binary classification (e.g., checkmate vs non-checkmate)
    >>> chance = compute_classification_chance_level(2)
    >>> print(f"Binary chance level: {chance:.2f}")  # 0.50

    >>> # 5-class classification (e.g., chess strategies)
    >>> chance = compute_classification_chance_level(5)
    >>> print(f"5-class chance level: {chance:.2f}")  # 0.20

    >>> # Use in significance testing
    >>> svm_accuracy = 0.65
    >>> chance = compute_classification_chance_level(n_classes=2)
    >>> if svm_accuracy > chance:
    ...     print(f"Above chance ({chance:.2f})")
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")

    return 1.0 / n_classes


def per_roi_welch_and_fdr(
    expert_vals: np.ndarray,
    novice_vals: np.ndarray,
    roi_labels: np.ndarray,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run Welch t-tests per ROI with FDR correction and effect sizes.

    For each ROI, performs:
    - Welch's t-test (unequal variances) comparing expert vs. novice groups
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
                'mean_diff': np.nan,
                'ci95_low': np.nan,
                'ci95_high': np.nan,
            })
            continue

        # Welch t-test
        mean1, mean2, mean_diff, ci_low, ci_high, t_stat, p_val = welch_ttest(
            expert_clean, novice_clean, confidence_level=0.95
        )

        # Degrees of freedom (Welch-Satterthwaite)
        result_obj = ttest_ind(expert_clean, novice_clean, equal_var=False)
        try:
            dof = result_obj.df
        except Exception:
            # Compute Welch-Satterthwaite df manually if SciPy object lacks .df
            n1, n2 = expert_clean.size, novice_clean.size
            v1, v2 = np.var(expert_clean, ddof=1), np.var(novice_clean, ddof=1)
            num = (v1/n1 + v2/n2) ** 2
            den = ((v1**2)/((n1**2)*(n1-1))) + ((v2**2)/((n2**2)*(n2-1)))
            dof = num/den if den > 0 else np.nan

        # Cohen's d
        if _pg_compute_effsize is not None:
            cohen_d = _pg_compute_effsize(expert_clean, novice_clean, eftype='cohen')
        else:
            n1, n2 = expert_clean.size, novice_clean.size
            s1, s2 = np.var(expert_clean, ddof=1), np.var(novice_clean, ddof=1)
            sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            cohen_d = (np.mean(expert_clean) - np.mean(novice_clean)) / sp if sp > 0 else np.nan

        results.append({
            'ROI_Label': int(roi_label),
            't_stat': t_stat,
            'p_val': p_val,
            'dof': float(dof),
            'cohen_d': float(cohen_d),
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
        res = ttest_1samp(x, popmean=value)
        t_stats[i] = float(res.statistic)
        p_two_sided[i] = float(res.pvalue)

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
