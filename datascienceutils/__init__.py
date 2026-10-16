"""
datascienceutils: A high-performance Python toolkit for data exploration,
statistical modeling, causal inference, and machine learning, powered by Rust.
"""

# Version definition
__version__ = "0.1.0"

# Import Python submodules
from . import analyze
from . import dimensionAnalyze
from . import plotter
from . import timeSeriesUtils
from . import statsutils
from . import sklearnUtils
from . import sampling
from . import outliers
from . import dists
from . import clusteringModels
from . import datautils
from . import modelValidators
from . import predictiveModels
from . import utils

# Try to import Rust-accelerated core functionality if compiled
try:
    from ._core import (
        # Causal analysis
        CausalGraph,
        estimate_ate,
        propensity_score_matching,
        instrumental_variable,
        diff_in_diff,
        regression_discontinuity,
        synthetic_control,
        mediation_analysis,
        conditional_ate,
        # Bayesian inference
        ABTestResult,
        bayesian_ab_test,
        mcmc_sample,
        compute_rhat,
        effective_sample_size,
        # Clustering
        HdbscanResult,
        hdbscan_cluster,
        KMeansResult,
        kmeans_cluster,
        SpectralResult,
        spectral_cluster,
        KNNResult,
        knn_classify,
        KMediansResult,
        kmedians_cluster,
        # Statistics
        chi2_test_independence,
        chi2_test,
        check_normality,
        pearson_correlation,
        spearman_correlation,
        kendall_correlation,
        ks_test,
        anova_oneway,
        # Outlier detection
        detect_outliers_sigma,
        detect_outliers_iqr,
        detect_outliers_zscore,
        detect_outliers_modified_zscore,
        cap_outliers_percentile,
        remove_outliers,
        # Sampling
        sample_normal,
        sample_uniform,
        bootstrap_sample,
        # Utilities
        bayesian_blocks,
        fractaldim,
    )
    _RUST_AVAILABLE = True
    _RUST_SYMBOLS = [
        "CausalGraph", "estimate_ate", "propensity_score_matching", "instrumental_variable",
        "diff_in_diff", "regression_discontinuity", "synthetic_control", "mediation_analysis", "conditional_ate",
        "ABTestResult", "bayesian_ab_test", "mcmc_sample", "compute_rhat", "effective_sample_size",
        "HdbscanResult", "hdbscan_cluster", "KMeansResult", "kmeans_cluster",
        "SpectralResult", "spectral_cluster", "KNNResult", "knn_classify", "KMediansResult", "kmedians_cluster",
        "chi2_test_independence", "chi2_test", "check_normality", "pearson_correlation", "spearman_correlation",
        "kendall_correlation", "ks_test", "anova_oneway",
        "detect_outliers_sigma", "detect_outliers_iqr", "detect_outliers_zscore", "detect_outliers_modified_zscore",
        "cap_outliers_percentile", "remove_outliers",
        "sample_normal", "sample_uniform", "bootstrap_sample",
        "bayesian_blocks", "fractaldim",
    ]
except ImportError:
    _RUST_AVAILABLE = False
    _RUST_SYMBOLS = []

__all__ = [
    "analyze",
    "dimensionAnalyze",
    "plotter",
    "timeSeriesUtils",
    "statsutils",
    "sklearnUtils",
    "sampling",
    "outliers",
    "dists",
    "clusteringModels",
    "datautils",
    "modelValidators",
    "predictiveModels",
    "utils",
    "__version__",
    "_RUST_AVAILABLE",
] + _RUST_SYMBOLS
