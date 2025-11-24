//! Python bindings for datascienceutils
//!
//! This module provides PyO3 bindings to expose the Rust datascienceutils library to Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array1;

// Import Rust modules
use datascienceutils_core::stats;
use datascienceutils_core::outliers;
use datascienceutils_core::sampling;
use datascienceutils_core::utils;

/// Convert Rust error to Python exception
fn to_py_err(err: datascienceutils_core::error::DsuError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

// ============================================================================
// Statistics Functions
// ============================================================================

/// Perform chi-square test of independence
///
/// Args:
///     observed: 2D array of observed frequencies
///
/// Returns:
///     Tuple of (chi_square, p_value, degrees_of_freedom)
#[pyfunction]
fn chi2_test_independence(
    py: Python,
    observed: PyReadonlyArray2<f64>,
) -> PyResult<(f64, f64, usize)> {
    let observed = observed.as_array();
    stats::chi2_test_independence(&observed.to_owned())
        .map_err(to_py_err)
}

/// Perform chi-square goodness of fit test
///
/// Args:
///     observed: Observed frequencies
///     expected: Expected frequencies
///     ddof: Delta degrees of freedom
///
/// Returns:
///     Tuple of (chi_square, p_value)
#[pyfunction]
fn chi2_test(
    py: Python,
    observed: PyReadonlyArray1<f64>,
    expected: PyReadonlyArray1<f64>,
    ddof: usize,
) -> PyResult<(f64, f64)> {
    let obs = observed.as_array();
    let exp = expected.as_array();
    stats::chi2_test(&obs, &exp, ddof)
        .map_err(to_py_err)
}

/// Check normality using Anderson-Darling test
///
/// Args:
///     data: Data to test
///
/// Returns:
///     Tuple of (statistic, critical_values, significance_levels)
#[pyfunction]
fn check_normality(
    py: Python,
    data: PyReadonlyArray1<f64>,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let data = data.as_array();
    stats::check_normality(&data)
        .map_err(to_py_err)
}

/// Calculate Pearson correlation coefficient
///
/// Args:
///     x: First variable
///     y: Second variable
///
/// Returns:
///     Correlation coefficient (-1 to 1)
#[pyfunction]
fn pearson_correlation(
    py: Python,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let x = x.as_array();
    let y = y.as_array();
    stats::pearson_correlation(&x, &y)
        .map_err(to_py_err)
}

/// Calculate Spearman rank correlation coefficient
#[pyfunction]
fn spearman_correlation(
    py: Python,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let x = x.as_array();
    let y = y.as_array();
    stats::spearman_correlation(&x, &y)
        .map_err(to_py_err)
}

/// Calculate Kendall tau correlation coefficient
#[pyfunction]
fn kendall_correlation(
    py: Python,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let x = x.as_array();
    let y = y.as_array();
    stats::kendall_correlation(&x, &y)
        .map_err(to_py_err)
}

/// Perform Kolmogorov-Smirnov test
///
/// Args:
///     data: Sample data
///     distribution: Distribution name ("normal")
///
/// Returns:
///     Tuple of (ks_statistic, p_value)
#[pyfunction]
fn ks_test(
    py: Python,
    data: PyReadonlyArray1<f64>,
    distribution: &str,
) -> PyResult<(f64, f64)> {
    let data = data.as_array();
    stats::ks_test(&data, distribution)
        .map_err(to_py_err)
}

/// Perform one-way ANOVA
///
/// Args:
///     groups: List of arrays, one per group
///
/// Returns:
///     Tuple of (f_statistic, p_value)
#[pyfunction]
fn anova_oneway(
    py: Python,
    groups: Vec<PyReadonlyArray1<f64>>,
) -> PyResult<(f64, f64)> {
    let group_views: Vec<_> = groups.iter()
        .map(|g| g.as_array())
        .collect();
    
    stats::anova_oneway(&group_views)
        .map_err(to_py_err)
}

// ============================================================================
// Outlier Detection Functions
// ============================================================================

/// Detect outliers using sigma deviation method
///
/// Args:
///     data: Input data
///     n_sigma: Number of standard deviations (default: 3.0)
///
/// Returns:
///     Tuple of (outlier_indices, lower_bound, upper_bound)
#[pyfunction]
fn detect_outliers_sigma(
    py: Python,
    data: PyReadonlyArray1<f64>,
    n_sigma: f64,
) -> PyResult<(Vec<usize>, f64, f64)> {
    let data = data.as_array();
    outliers::detect_outliers_sigma(&data, n_sigma)
        .map_err(to_py_err)
}

/// Detect outliers using IQR method
///
/// Args:
///     data: Input data
///     k: IQR multiplier (default: 1.5)
///
/// Returns:
///     Tuple of (outlier_indices, lower_bound, upper_bound)
#[pyfunction]
fn detect_outliers_iqr(
    py: Python,
    data: PyReadonlyArray1<f64>,
    k: f64,
) -> PyResult<(Vec<usize>, f64, f64)> {
    let data = data.as_array();
    outliers::detect_outliers_iqr(&data, k)
        .map_err(to_py_err)
}

/// Detect outliers using Z-score method
///
/// Args:
///     data: Input data
///     threshold: Z-score threshold (default: 3.0)
///
/// Returns:
///     List of outlier indices
#[pyfunction]
fn detect_outliers_zscore(
    py: Python,
    data: PyReadonlyArray1<f64>,
    threshold: f64,
) -> PyResult<Vec<usize>> {
    let data = data.as_array();
    outliers::detect_outliers_zscore(&data, threshold)
        .map_err(to_py_err)
}

/// Detect outliers using modified Z-score method
///
/// Args:
///     data: Input data
///     threshold: Modified Z-score threshold (default: 3.5)
///
/// Returns:
///     List of outlier indices
#[pyfunction]
fn detect_outliers_modified_zscore(
    py: Python,
    data: PyReadonlyArray1<f64>,
    threshold: f64,
) -> PyResult<Vec<usize>> {
    let data = data.as_array();
    outliers::detect_outliers_modified_zscore(&data, threshold)
        .map_err(to_py_err)
}

/// Cap outliers using percentile method
///
/// Args:
///     data: Input data
///     lower_percentile: Lower percentile (e.g., 5.0)
///     upper_percentile: Upper percentile (e.g., 95.0)
///
/// Returns:
///     Data with outliers capped
#[pyfunction]
fn cap_outliers_percentile<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    lower_percentile: f64,
    upper_percentile: f64,
) -> PyResult<&'py PyArray1<f64>> {
    let data = data.as_array();
    let result = outliers::cap_outliers_percentile(&data, lower_percentile, upper_percentile)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_array(py, &result))
}

/// Remove outliers from data
///
/// Args:
///     data: Input data
///     outlier_indices: Indices of outliers to remove
///
/// Returns:
///     Data with outliers removed
#[pyfunction]
fn remove_outliers<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    outlier_indices: Vec<usize>,
) -> PyResult<&'py PyArray1<f64>> {
    let data = data.as_array();
    let result = outliers::remove_outliers(&data, &outlier_indices)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_array(py, &result))
}

// ============================================================================
// Sampling Functions
// ============================================================================

/// Sample from normal distribution
///
/// Args:
///     mean: Mean of the distribution
///     std_dev: Standard deviation
///     n: Number of samples
///
/// Returns:
///     Array of samples
#[pyfunction]
fn sample_normal<'py>(
    py: Python<'py>,
    mean: f64,
    std_dev: f64,
    n: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let result = sampling::sample_normal(mean, std_dev, n)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_array(py, &result))
}

/// Sample from uniform distribution
///
/// Args:
///     low: Lower bound
///     high: Upper bound
///     n: Number of samples
///
/// Returns:
///     Array of samples
#[pyfunction]
fn sample_uniform<'py>(
    py: Python<'py>,
    low: f64,
    high: f64,
    n: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let result = sampling::sample_uniform(low, high, n)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_array(py, &result))
}

/// Bootstrap sampling
///
/// Args:
///     data: Input data
///     n_samples: Number of bootstrap samples
///
/// Returns:
///     List of bootstrap samples
#[pyfunction]
fn bootstrap_sample<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    n_samples: usize,
) -> PyResult<Vec<&'py PyArray1<f64>>> {
    let data = data.as_array();
    let results = sampling::bootstrap_sample(&data, n_samples)
        .map_err(to_py_err)?;
    
    Ok(results.iter()
        .map(|arr| PyArray1::from_array(py, arr))
        .collect())
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculate Bayesian blocks for optimal histogram binning
///
/// Args:
///     data: Input data
///
/// Returns:
///     Array of bin edges
#[pyfunction]
fn bayesian_blocks<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let data = data.as_array();
    let result = utils::bayesian_blocks(&data)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_array(py, &result))
}

/// Calculate fractal dimension using box-counting method
///
/// Args:
///     pointlist: Array of points
///     boxlevel: Number of divisions
///
/// Returns:
///     Fractal dimension
#[pyfunction]
fn fractaldim(
    py: Python,
    pointlist: PyReadonlyArray2<f64>,
    boxlevel: usize,
) -> PyResult<f64> {
    let pointlist = pointlist.as_array();
    utils::fractaldim(&pointlist.to_owned(), boxlevel)
        .map_err(to_py_err)
}

// ============================================================================
// Module Definition
// ============================================================================

/// Python module initialization
#[pymodule]
fn datascienceutils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Statistics functions
    m.add_function(wrap_pyfunction!(chi2_test_independence, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_test, m)?)?;
    m.add_function(wrap_pyfunction!(check_normality, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(spearman_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(kendall_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(ks_test, m)?)?;
    m.add_function(wrap_pyfunction!(anova_oneway, m)?)?;
    
    // Outlier detection functions
    m.add_function(wrap_pyfunction!(detect_outliers_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(detect_outliers_iqr, m)?)?;
    m.add_function(wrap_pyfunction!(detect_outliers_zscore, m)?)?;
    m.add_function(wrap_pyfunction!(detect_outliers_modified_zscore, m)?)?;
    m.add_function(wrap_pyfunction!(cap_outliers_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(remove_outliers, m)?)?;
    
    // Sampling functions
    m.add_function(wrap_pyfunction!(sample_normal, m)?)?;
    m.add_function(wrap_pyfunction!(sample_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_sample, m)?)?;
    
    // Utility functions
    m.add_function(wrap_pyfunction!(bayesian_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(fractaldim, m)?)?;
    
    Ok(())
}
