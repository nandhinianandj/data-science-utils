//! Python bindings for datascienceutils
//!
//! This module provides PyO3 bindings to expose the Rust datascienceutils library to Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
// Causal Analysis Functions
// ============================================================================

#[cfg(feature = "causal-analysis")]
use datascienceutils_core::analyze::causal;

/// Causal Graph for representing causal relationships
#[cfg(feature = "causal-analysis")]
#[pyclass]
struct CausalGraph {
    inner: causal::CausalGraph,
}

#[cfg(feature = "causal-analysis")]
#[pymethods]
impl CausalGraph {
    /// Create a new empty causal graph
    #[new]
    fn new() -> Self {
        CausalGraph {
            inner: causal::CausalGraph::new(),
        }
    }

    /// Add a node to the graph
    ///
    /// Args:
    ///     name: Name of the node
    fn add_node(&mut self, name: &str) -> PyResult<()> {
        self.inner.add_node(name).map_err(to_py_err)
    }

    /// Add a directed edge from parent to child
    ///
    /// Args:
    ///     parent: Parent node name
    ///     child: Child node name
    fn add_edge(&mut self, parent: &str, child: &str) -> PyResult<()> {
        self.inner.add_edge(parent, child).map_err(to_py_err)
    }

    /// Add a weighted directed edge
    ///
    /// Args:
    ///     parent: Parent node name
    ///     child: Child node name
    ///     weight: Edge weight
    fn add_edge_weighted(&mut self, parent: &str, child: &str, weight: f64) -> PyResult<()> {
        self.inner.add_edge_weighted(parent, child, weight).map_err(to_py_err)
    }

    /// Freeze an edge (prevent modification)
    ///
    /// Args:
    ///     parent: Parent node name
    ///     child: Child node name
    fn freeze_edge(&mut self, parent: &str, child: &str) -> PyResult<()> {
        self.inner.freeze_edge(parent, child).map_err(to_py_err)
    }

    /// Freeze a subgraph (all edges involving these nodes)
    ///
    /// Args:
    ///     nodes: List of node names
    fn freeze_subgraph(&mut self, nodes: Vec<&str>) {
        self.inner.freeze_subgraph(&nodes)
    }

    /// Get number of nodes
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Get number of edges
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Export graph to DOT format (Graphviz)
    fn to_dot(&self) -> String {
        self.inner.to_dot()
    }

    /// Export graph to DOT file
    ///
    /// Args:
    ///     path: File path to save DOT file
    fn save_dot(&self, path: &str) -> PyResult<()> {
        self.inner.save_dot(path).map_err(to_py_err)
    }

    /// Get parents of a node
    ///
    /// Args:
    ///     node: Node name
    ///
    /// Returns:
    ///     List of parent node names
    fn parents(&self, node: &str) -> Vec<String> {
        self.inner.parents(node)
    }

    /// Get children of a node
    ///
    /// Args:
    ///     node: Node name
    ///
    /// Returns:
    ///     List of child node names
    fn children(&self, node: &str) -> Vec<String> {
        self.inner.children(node)
    }
}

/// Estimate Average Treatment Effect (ATE)
///
/// Args:
///     confounders: Confounding variables (n_samples x n_features)
///     treatment: Treatment assignment (0 or 1)
///     outcome: Outcome variable
///
/// Returns:
///     Estimated average treatment effect
#[cfg(feature = "causal-analysis")]
#[pyfunction]
fn estimate_ate(
    py: Python,
    confounders: PyReadonlyArray2<f64>,
    treatment: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let confounders = confounders.as_array().to_owned();
    let treatment = treatment.as_array().to_owned();
    let outcome = outcome.as_array().to_owned();
    
    causal::estimate_ate(&confounders, &treatment, &outcome)
        .map_err(to_py_err)
}

/// Propensity score matching
///
/// Args:
///     confounders: Confounding variables
///     treatment: Treatment assignment
///     outcome: Outcome variable
///
/// Returns:
///     ATE estimated using propensity score matching
#[cfg(feature = "causal-analysis")]
#[pyfunction]
fn propensity_score_matching(
    py: Python,
    confounders: PyReadonlyArray2<f64>,
    treatment: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let confounders = confounders.as_array().to_owned();
    let treatment = treatment.as_array().to_owned();
    let outcome = outcome.as_array().to_owned();
    
    causal::propensity_score_matching(&confounders, &treatment, &outcome)
        .map_err(to_py_err)
}

/// Instrumental variable estimation
///
/// Args:
///     treatment: Treatment variable
///     outcome: Outcome variable
///     instrument: Instrumental variable
///
/// Returns:
///     IV estimate of treatment effect
#[cfg(feature = "causal-analysis")]
#[pyfunction]
fn instrumental_variable(
    py: Python,
    treatment: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>,
    instrument: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let treatment = treatment.as_array().to_owned();
    let outcome = outcome.as_array().to_owned();
    let instrument = instrument.as_array().to_owned();
    
    causal::instrumental_variable(&treatment, &outcome, &instrument)
        .map_err(to_py_err)
}

/// Difference-in-differences estimation
///
/// Args:
///     group: Group indicator (0=control, 1=treatment)
///     time: Time indicator (0=pre, 1=post)
///     outcome: Outcome variable
///
/// Returns:
///     DiD estimate
#[cfg(feature = "causal-analysis")]
#[pyfunction]
fn diff_in_diff(
    py: Python,
    group: PyReadonlyArray1<f64>,
    time: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let group = group.as_array().to_owned();
    let time = time.as_array().to_owned();
    let outcome = outcome.as_array().to_owned();
    
    causal::diff_in_diff(&group, &time, &outcome)
        .map_err(to_py_err)
}

/// Regression Discontinuity Design (RDD)
///
/// Args:
///     running_var: Running variable (e.g., test score)
///     outcome: Outcome variable
///     cutoff: Threshold for treatment assignment
///     bandwidth: Optional bandwidth for local regression (None = auto-select)
///
/// Returns:
///     RDD estimate of treatment effect
#[cfg(feature = "causal-analysis")]
#[pyfunction]
#[pyo3(signature = (running_var, outcome, cutoff, bandwidth=None))]
fn regression_discontinuity(
    _py: Python,
    running_var: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>,
    cutoff: f64,
    bandwidth: Option<f64>,
) -> PyResult<f64> {
    let running_var = running_var.as_array().to_owned();
    let outcome = outcome.as_array().to_owned();
    
    causal::regression_discontinuity(&running_var, &outcome, cutoff, bandwidth)
        .map_err(to_py_err)
}

/// Synthetic Control Method
///
/// Args:
///     treated_pre: Pre-treatment outcomes for treated unit
///     treated_post: Post-treatment outcomes for treated unit
///     control_pre: Pre-treatment outcomes for control units (n_controls x n_periods)
///     control_post: Post-treatment outcomes for control units
///
/// Returns:
///     Estimated treatment effect
#[cfg(feature = "causal-analysis")]
#[pyfunction]
fn synthetic_control(
    py: Python,
    treated_pre: PyReadonlyArray1<f64>,
    treated_post: PyReadonlyArray1<f64>,
    control_pre: PyReadonlyArray2<f64>,
    control_post: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let treated_pre = treated_pre.as_array().to_owned();
    let treated_post = treated_post.as_array().to_owned();
    let control_pre = control_pre.as_array().to_owned();
    let control_post = control_post.as_array().to_owned();
    
    causal::synthetic_control(&treated_pre, &treated_post, &control_pre, &control_post)
        .map_err(to_py_err)
}

/// Mediation Analysis
///
/// Args:
///     treatment: Treatment variable
///     mediator: Mediator variable
///     outcome: Outcome variable
///
/// Returns:
///     Tuple of (total_effect, direct_effect, indirect_effect)
#[cfg(feature = "causal-analysis")]
#[pyfunction]
fn mediation_analysis(
    py: Python,
    treatment: PyReadonlyArray1<f64>,
    mediator: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>,
) -> PyResult<(f64, f64, f64)> {
    let treatment = treatment.as_array().to_owned();
    let mediator = mediator.as_array().to_owned();
    let outcome = outcome.as_array().to_owned();
    
    causal::mediation_analysis(&treatment, &mediator, &outcome)
        .map_err(to_py_err)
}

/// Conditional Average Treatment Effect (CATE)
///
/// Args:
///     confounders: Confounding variables (n_samples x n_features)
///     treatment: Treatment assignment
///     outcome: Outcome variable
///     subgroup_idx: Index of confounder to stratify by (default: 0)
///
/// Returns:
///     List of tuples (subgroup_value, cate_estimate)
#[cfg(feature = "causal-analysis")]
#[pyfunction]
#[pyo3(signature = (confounders, treatment, outcome, subgroup_idx=0))]
fn conditional_ate(
    _py: Python,
    confounders: PyReadonlyArray2<f64>,
    treatment: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>,
    subgroup_idx: usize,
) -> PyResult<Vec<(f64, f64)>> {
    let confounders = confounders.as_array().to_owned();
    let treatment = treatment.as_array().to_owned();
    let outcome = outcome.as_array().to_owned();
    
    let result = causal::conditional_ate(&confounders, &treatment, &outcome, subgroup_idx)
        .map_err(to_py_err)?;
    Ok(result)
}

// ============================================================================
// Clustering Functions
// ============================================================================

#[cfg(feature = "clustering-hdbscan")]
use datascienceutils_core::cluster::hdbscan_wrapper;

/// HDBSCAN Clustering Result
#[cfg(feature = "clustering-hdbscan")]
#[pyclass]
struct HdbscanResult {
    #[pyo3(get)]
    labels: Vec<i32>,
    #[pyo3(get)]
    probabilities: Vec<f64>,
    #[pyo3(get)]
    outlier_scores: Vec<f64>,
    #[pyo3(get)]
    n_clusters: usize,
}

/// Perform HDBSCAN clustering
///
/// Args:
///     data: Input data (n_samples x n_features)
///     min_cluster_size: Minimum number of samples in a cluster
///     min_samples: Minimum number of samples in a neighborhood (optional)
///
/// Returns:
///     HdbscanResult object
#[cfg(feature = "clustering-hdbscan")]
#[pyfunction]
#[pyo3(signature = (data, min_cluster_size, min_samples=None))]
fn hdbscan_cluster(
    py: Python,
    data: PyReadonlyArray2<f64>,
    min_cluster_size: usize,
    min_samples: Option<usize>,
) -> PyResult<HdbscanResult> {
    let data = data.as_array();
    
    let result = hdbscan_wrapper::hdbscan_cluster(data, min_cluster_size, min_samples)
        .map_err(to_py_err)?;
        
    Ok(HdbscanResult {
        labels: result.labels,
        probabilities: result.probabilities,
        outlier_scores: result.outlier_scores,
        n_clusters: result.n_clusters,
    })
}

// ============================================================================
// Bayesian Inference Functions
// ============================================================================

use datascienceutils_core::analyze::bayesian;

/// Spectral Clustering Result
#[pyclass]
struct SpectralResult {
    #[pyo3(get)]
    labels: Vec<usize>,
}

/// Perform Spectral Clustering
///
/// Args:
///     data: Input data (n_samples x n_features)
///     n_clusters: Number of clusters
///     gamma: RBF kernel coefficient (default: 1.0)
///     seed: Random seed (default: 42)
///
/// Returns:
///     SpectralResult object
#[pyfunction]
#[pyo3(signature = (data, n_clusters, gamma=1.0, seed=42))]
fn spectral_cluster(
    py: Python,
    data: PyReadonlyArray2<f64>,
    n_clusters: usize,
    gamma: f64,
    seed: u64,
) -> PyResult<SpectralResult> {
    use datascienceutils_core::cluster::spectral;
    let data = data.as_array();
    let result = spectral::spectral_cluster(data, n_clusters, gamma, seed)
        .map_err(to_py_err)?;
    Ok(SpectralResult {
        labels: result.labels,
    })
}

/// Bayesian A/B Test Result
#[pyclass]
struct ABTestResult {
    #[pyo3(get)]
    prob_treatment_better: f64,
    #[pyo3(get)]
    expected_lift: f64,
    #[pyo3(get)]
    credible_interval: (f64, f64),
}

/// Perform Bayesian A/B testing
///
/// Args:
///     control: Control group data (0s and 1s)
///     treatment: Treatment group data (0s and 1s)
///
/// Returns:
///     ABTestResult object
#[pyfunction]
fn bayesian_ab_test(
    py: Python,
    control: PyReadonlyArray1<f64>,
    treatment: PyReadonlyArray1<f64>,
) -> PyResult<ABTestResult> {
    let control = control.as_array().to_owned();
    let treatment = treatment.as_array().to_owned();
    
    let result = bayesian::bayesian_ab_test(control.as_slice().unwrap(), treatment.as_slice().unwrap())
        .map_err(to_py_err)?;
        
    Ok(ABTestResult {
        prob_treatment_better: result.prob_treatment_better,
        expected_lift: result.expected_lift,
        credible_interval: result.credible_interval,
    })
}

/// Run MCMC sampling using Metropolis-Hastings
///
/// Args:
///     log_posterior: Python function that takes a 1D array and returns log probability
///     initial: Initial parameter values
///     proposal_std: Standard deviation for proposal distribution
///     num_samples: Number of samples to draw
///
/// Returns:
///     2D array of samples (num_samples x num_params)
#[pyfunction]
fn mcmc_sample<'py>(
    py: Python<'py>,
    log_posterior: PyObject,
    initial: PyReadonlyArray1<'py, f64>,
    proposal_std: f64,
    num_samples: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let initial = initial.as_array().to_owned();
    
    // Wrap Python function
    let log_post_wrapper = |x: &Array1<f64>| -> f64 {
        Python::with_gil(|py| {
            let py_x = PyArray1::from_array(py, x);
            let args = (py_x,);
            log_posterior.call1(py, args)
                .and_then(|res| res.extract(py))
                .unwrap_or(-f64::INFINITY) // Return -inf on error to reject sample
        })
    };
    
    let sampler = bayesian::MCMCSampler::MetropolisHastings { proposal_std };
    
    let samples = bayesian::mcmc_sample(log_post_wrapper, initial, sampler, num_samples)
        .map_err(to_py_err)?;
        
    Ok(PyArray2::from_array(py, &samples))
}

/// Compute R-hat convergence diagnostic
///
/// Args:
///     samples: MCMC samples (n_samples x n_params)
///
/// Returns:
///     R-hat value
#[pyfunction]
fn compute_rhat(
    py: Python,
    samples: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let samples = samples.as_array();
    Ok(bayesian::compute_rhat(&samples.to_owned()))
}

/// Compute Effective Sample Size (ESS)
///
/// Args:
///     samples: MCMC samples (n_samples x n_params)
///
/// Returns:
///     ESS value
#[pyfunction]
fn effective_sample_size(
    py: Python,
    samples: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let samples = samples.as_array();
    Ok(bayesian::effective_sample_size(&samples.to_owned()))
}

// ============================================================================
// Clustering & Predictive Functions (New)
// ============================================================================

/// K-Means Clustering Result
#[pyclass]
struct KMeansResult {
    #[pyo3(get)]
    labels: Vec<usize>,
    #[pyo3(get)]
    centroids: Vec<Vec<f64>>,
    #[pyo3(get)]
    inertia: f64,
}

/// Perform K-Means clustering
///
/// Args:
///     data: Input data (n_samples x n_features)
///     n_clusters: Number of clusters
///     max_iter: Maximum iterations (default: 300)
///     tolerance: Convergence tolerance (default: 1e-4)
///     seed: Random seed (default: 42)
///
/// Returns:
///     KMeansResult object
#[pyfunction]
#[pyo3(signature = (data, n_clusters, max_iter=300, tolerance=1e-4, seed=42))]
fn kmeans_cluster(
    py: Python,
    data: PyReadonlyArray2<f64>,
    n_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    seed: u64,
) -> PyResult<KMeansResult> {
    use datascienceutils_core::cluster::kmeans;
    let data = data.as_array();
    let result = kmeans::kmeans_cluster(data, n_clusters, max_iter, tolerance, seed)
        .map_err(to_py_err)?;
    Ok(KMeansResult {
        labels: result.labels,
        centroids: result.centroids,
        inertia: result.inertia,
    })
}

/// KNN Classification Result
#[pyclass]
struct KNNResult {
    #[pyo3(get)]
    predictions: Vec<f64>,
    #[pyo3(get)]
    accuracy: Option<f64>,
}

/// Perform KNN classification
///
/// Args:
///     train_data: Training features
///     train_target: Training labels
///     test_data: Test features
///     k: Number of neighbors (default: 5)
///     weight: Weight function "uniform" or "distance" (default: "uniform")
///     algorithm: Search algorithm "linear", "kd_tree", "cover_tree", "ball_tree" (default: "linear")
///
/// Returns:
///     KNNResult object
#[pyfunction]
#[pyo3(signature = (train_data, train_target, test_data, k=5, weight="uniform", algorithm="linear"))]
fn knn_classify(
    py: Python,
    train_data: PyReadonlyArray2<f64>,
    train_target: PyReadonlyArray1<f64>,
    test_data: PyReadonlyArray2<f64>,
    k: usize,
    weight: &str,
    algorithm: &str,
) -> PyResult<KNNResult> {
    use datascienceutils_core::predictive::knn;
    let train_data = train_data.as_array();
    let train_target = train_target.as_array();
    let test_data = test_data.as_array();
    
    let result = knn::knn_classify(train_data, train_target, test_data, k, weight, algorithm)
        .map_err(to_py_err)?;
        
    Ok(KNNResult {
        predictions: result.predictions,
        accuracy: result.accuracy,
    })
}

/// K-Medians Clustering Result
#[pyclass]
struct KMediansResult {
    #[pyo3(get)]
    labels: Vec<usize>,
    #[pyo3(get)]
    centroids: Vec<Vec<f64>>,
    #[pyo3(get)]
    cost: f64,
}

/// Perform K-Medians clustering
///
/// Args:
///     data: Input data (n_samples x n_features)
///     n_clusters: Number of clusters
///     max_iter: Maximum iterations (default: 300)
///     tolerance: Convergence tolerance (default: 1e-4)
///     seed: Random seed (default: 42)
///
/// Returns:
///     KMediansResult object
#[pyfunction]
#[pyo3(signature = (data, n_clusters, max_iter=300, tolerance=1e-4, seed=42))]
fn kmedians_cluster(
    py: Python,
    data: PyReadonlyArray2<f64>,
    n_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    seed: u64,
) -> PyResult<KMediansResult> {
    use datascienceutils_core::cluster::kmedians;
    let data = data.as_array();
    let result = kmedians::kmedians_cluster(data, n_clusters, max_iter, tolerance, seed)
        .map_err(to_py_err)?;
    Ok(KMediansResult {
        labels: result.labels,
        centroids: result.centroids,
        cost: result.cost,
    })
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
    
    // Causal analysis functions (if feature enabled)
    #[cfg(feature = "causal-analysis")]
    {
        m.add_class::<CausalGraph>()?;
        m.add_function(wrap_pyfunction!(estimate_ate, m)?)?;
        m.add_function(wrap_pyfunction!(propensity_score_matching, m)?)?;
        m.add_function(wrap_pyfunction!(instrumental_variable, m)?)?;
        m.add_function(wrap_pyfunction!(diff_in_diff, m)?)?;
        m.add_function(wrap_pyfunction!(regression_discontinuity, m)?)?;
        m.add_function(wrap_pyfunction!(synthetic_control, m)?)?;
        m.add_function(wrap_pyfunction!(mediation_analysis, m)?)?;
        m.add_function(wrap_pyfunction!(conditional_ate, m)?)?;
        m.add_function(wrap_pyfunction!(conditional_ate, m)?)?;
    }
    
    // Bayesian inference functions
    m.add_class::<ABTestResult>()?;
    m.add_function(wrap_pyfunction!(bayesian_ab_test, m)?)?;
    m.add_function(wrap_pyfunction!(mcmc_sample, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rhat, m)?)?;
    m.add_function(wrap_pyfunction!(effective_sample_size, m)?)?;
    
    // Clustering functions
    #[cfg(feature = "clustering-hdbscan")]
    {
        m.add_class::<HdbscanResult>()?;
        m.add_function(wrap_pyfunction!(hdbscan_cluster, m)?)?;
    }
    
    // K-Means
    m.add_class::<KMeansResult>()?;
    m.add_function(wrap_pyfunction!(kmeans_cluster, m)?)?;
    
    // Spectral Clustering
    m.add_class::<SpectralResult>()?;
    m.add_function(wrap_pyfunction!(spectral_cluster, m)?)?;
    
    // KNN
    m.add_class::<KNNResult>()?;
    m.add_function(wrap_pyfunction!(knn_classify, m)?)?;
    
    // K-Medians
    m.add_class::<KMediansResult>()?;
    m.add_function(wrap_pyfunction!(kmedians_cluster, m)?)?;
    
    Ok(())
}
