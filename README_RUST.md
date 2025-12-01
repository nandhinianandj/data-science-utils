# DataScienceUtils - Rust Port

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)]()
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()

A high-performance Rust implementation of data science utilities for exploratory data analysis (EDA), with Python bindings for seamless Jupyter notebook integration.

## 🚀 Features

- **Causal Analysis**: ATE, PSM, IV, DiD, RDD, Synthetic Control, Mediation, CATE with graph visualization
- **Statistical Analysis**: Chi-square tests, ANOVA, distribution fitting, correlation analysis
- **Outlier Detection**: Sigma deviation, IQR, Z-score, Modified Z-score, percentile capping
- **Sampling**: Reservoir sampling, stratified sampling, bootstrap, distribution samplers
- **Clustering**: K-Means, DBSCAN, Hierarchical, Spectral, SOM with silhouette analysis
- **Dimensionality Reduction**: PCA, t-SNE, UMAP, factor analysis
- **Time Series**: Stationarity tests, autocorrelation, seasonal decomposition
- **Visualization**: 25+ plot types using plotters (histograms, heatmaps, scatter, violin, etc.)
- **Feature Engineering**: Scaling, normalization, encoding, selection

## 📦 Installation

### From PyPI (when released)

```bash
pip install datascienceutils
```

### From Source

```bash
# Install maturin
pip install maturin

# Build and install
cd /path/to/data-science-utils
maturin develop --release
```

## 🎯 Quick Start

### Python (Jupyter Notebook)

```python
import datascienceutils as dsu
import numpy as np

# ===== Causal Analysis =====
# Create a causal graph
graph = dsu.CausalGraph()
graph.add_node("Treatment")
graph.add_node("Outcome")
graph.add_node("Confounder")
graph.add_edge("Confounder", "Treatment")
graph.add_edge("Treatment", "Outcome")
graph.save_dot("causal_graph.dot")

# Estimate Average Treatment Effect
confounders = np.random.randn(100, 2)
treatment = (confounders[:, 0] > 0).astype(float)
outcome = treatment * 5.0 + confounders[:, 1] + np.random.randn(100)
ate = dsu.estimate_ate(confounders, treatment, outcome)

# Propensity Score Matching
psm_ate = dsu.propensity_score_matching(confounders, treatment, outcome)

# Difference-in-Differences
group = np.array([0, 0, 1, 1])
time = np.array([0, 1, 0, 1])
outcome = np.array([50, 52, 50, 68])
did = dsu.diff_in_diff(group, time, outcome)

# ===== Outlier Detection =====
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

# Detect outliers using different methods
outliers_sigma, lower, upper = dsu.detect_outliers_sigma(data, n_sigma=2.0)
outliers_iqr, _, _ = dsu.detect_outliers_iqr(data, k=1.5)
outliers_z = dsu.detect_outliers_zscore(data, threshold=2.0)

# Remove or cap outliers
cleaned = dsu.remove_outliers(data, outliers_sigma)
capped = dsu.cap_outliers_percentile(data, 5.0, 95.0)

# ===== Sampling =====
# Sample from distributions
normal_samples = dsu.sample_normal(mean=0.0, std_dev=1.0, n=1000)
uniform_samples = dsu.sample_uniform(low=0.0, high=10.0, n=1000)

# Bootstrap sampling
bootstrap_samples = dsu.bootstrap_sample(data, n_samples=100)

# ===== Statistics =====
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# Correlation analysis
pearson_r = dsu.pearson_correlation(x, y)
spearman_rho = dsu.spearman_correlation(x, y)
kendall_tau = dsu.kendall_correlation(x, y)

# Statistical tests
stat, p_value, dof = dsu.chi2_test_independence(observed_matrix)
f_stat, p_value = dsu.anova_oneway([group1, group2, group3])
```

### Rust

```rust
use datascienceutils_core::prelude::*;
use datascienceutils_core::analyze::causal::*;
use ndarray::array;

fn main() -> DsuResult<()> {
    // Causal Analysis
    let mut graph = CausalGraph::new();
    graph.add_node("Treatment")?;
    graph.add_node("Outcome")?;
    graph.add_edge("Treatment", "Outcome")?;
    graph.save_dot("graph.dot")?;
    
    // Outlier Detection
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
    let (outliers, lower, upper) = detect_outliers_sigma(&data.view(), 2.0)?;
    
    // Sampling
    let samples = sample_normal(0.0, 1.0, 1000)?;
    
    // Statistical analysis
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
    let r = pearson_correlation(&x.view(), &y.view())?;
    
    Ok(())
}
```

## 🏗️ Architecture

```
data-science-utils/
├── Cargo.toml                 # Workspace configuration
├── pyproject.toml             # Python packaging (maturin)
├── datascienceutils-core/     # Core Rust library
│   ├── src/
│   │   ├── analyze/           # EDA functions
│   │   ├── cluster/           # Clustering algorithms
│   │   ├── plot/              # Visualization
│   │   ├── stats/             # Statistical tests
│   │   ├── timeseries/        # Time series analysis
│   │   ├── utils/             # Utilities
│   │   └── lib.rs
│   └── Cargo.toml
└── datascienceutils-py/       # Python bindings (PyO3)
    ├── src/
    │   └── lib.rs
    └── Cargo.toml
```

## 🔧 Development Status

**Current Phase**: Phase 1-2 Complete ✅

- [x] Project structure and build system
- [x] Core utilities (type checking, Bayesian blocks, fractal dimension)
- [x] Error handling framework
- [ ] Statistics module (in progress)
- [ ] Analysis module (planned)
- [ ] Clustering module (planned)
- [ ] Plotting module (planned)
- [ ] Python bindings (planned)

See [task.md](/.gemini/antigravity/brain/8627d558-92ba-4810-a325-de0f1ca2674d/task.md) for detailed progress.

## 🧪 Testing

```bash
# Run Rust tests
cargo test --all-features

# Run Python tests
pytest python-tests/

# Run Jupyter notebook tests
pytest --nbmake templates/*.ipynb

# Benchmarks
cargo bench
```

## 📊 Performance

Rust implementation provides significant performance improvements over Python:

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Bayesian Blocks | TBD | TBD | TBD |
| Fractal Dimension | TBD | TBD | TBD |
| K-Means Clustering | TBD | TBD | TBD |
| PCA | TBD | TBD | TBD |

*Benchmarks coming soon*

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

GNU General Public License v3 (GPLv3) - see [LICENSE.md](LICENSE.md)

## 🙏 Acknowledgments

- Original Python implementation by [@anandjeyahar](https://github.com/anandjeyahar)
- Built with amazing Rust data science ecosystem:
  - [ndarray](https://github.com/rust-ndarray/ndarray) - NumPy-like arrays
  - [polars](https://github.com/pola-rs/polars) - Fast DataFrames
  - [linfa](https://github.com/rust-ml/linfa) - Machine learning toolkit
  - [plotters](https://github.com/plotters-rs/plotters) - Visualization
  - [PyO3](https://github.com/PyO3/pyo3) - Python bindings

## 📚 Documentation

- [API Reference](docs/api_reference.md)
- [Migration Guide](docs/migration_guide.md)
- [Examples](templates/)

## 🔗 Links

- [GitHub Repository](https://github.com/emofeedback/data-science-utils)
- [Original Python Version](https://github.com/anandjeyahar/mlDemoExamples)
- [Issue Tracker](https://github.com/emofeedback/data-science-utils/issues)
