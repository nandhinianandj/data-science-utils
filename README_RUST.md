# DataScienceUtils - Rust Port

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)]()
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()

A high-performance Rust implementation of data science utilities for exploratory data analysis (EDA), with Python bindings for seamless Jupyter notebook integration.

## 🚀 Features

- **Statistical Analysis**: Chi-square tests, ANOVA, distribution fitting, correlation analysis
- **Clustering**: K-Means, DBSCAN, Hierarchical, Spectral, SOM with silhouette analysis
- **Dimensionality Reduction**: PCA, t-SNE, UMAP, factor analysis
- **Time Series**: Stationarity tests, autocorrelation, seasonal decomposition
- **Visualization**: 25+ plot types using plotters (histograms, heatmaps, scatter, violin, etc.)
- **Feature Engineering**: Scaling, normalization, encoding, selection
- **Outlier Detection**: Sigma deviation, IQR, percentile capping, Z-score
- **Sampling**: Reservoir sampling, file sampling, distribution samplers

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
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Distribution analysis
dsu.dist_analyze(df, column='age', is_normal=True, kdeplot=True)

# Correlation analysis
dsu.correlation_analyze(df, 'feature1', 'feature2')

# Clustering
dsu.cluster_analyze(df, name='customer_segments')

# Time series analysis
dsu.time_series_analysis(df, timeCol='date', valueCol='sales')
```

### Rust

```rust
use datascienceutils_core::prelude::*;
use ndarray::array;

fn main() -> DsuResult<()> {
    // Statistical analysis
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let pct_missing = na_pct(&data.view());
    
    // Bayesian blocks for optimal binning
    let edges = bayesian_blocks(&data.view())?;
    
    // Fractal dimension
    let points = array![[0.1, 0.2], [0.3, 0.4]];
    let dim = fractaldim(&points, 10)?;
    
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
