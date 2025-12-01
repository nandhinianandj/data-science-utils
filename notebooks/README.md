# Data Science Utils - Jupyter Notebooks

This directory contains demonstration notebooks for the datascienceutils library.

## Available Notebooks

### 1. [Causal Analysis Demo](05_causal_analysis_demo.ipynb)
**Status**: ✅ Complete (Rust kernel)

Demonstrates causal inference capabilities:
- Average Treatment Effect (ATE) estimation
- Propensity Score Matching (PSM)
- Instrumental Variables (IV)
- Difference-in-Differences (DiD)
- Causal graph creation and manipulation

**Requirements**: evcxr Jupyter kernel for Rust

---

### 2. [Outlier Detection Demo](06_outliers_detection_demo.ipynb)
**Status**: ✅ Complete (Python kernel)

Comprehensive outlier detection methods:
- **Sigma Deviation Method** - Standard deviation-based detection
- **IQR Method** - Interquartile range method
- **Z-Score Method** - Standardized score method
- **Modified Z-Score** - MAD-based robust method
- **Percentile Capping** - Cap values at percentiles
- **Outlier Removal** - Remove detected outliers

**Features**:
- Visual comparisons of all methods
- Before/after visualizations
- Method recommendations

**Requirements**: Python 3.8+, matplotlib, numpy, datascienceutils

---

### 3. [Sampling Methods Demo](07_sampling_methods_demo.ipynb)
**Status**: ✅ Complete (Python kernel)

Sampling and resampling techniques:
- **Normal Distribution Sampling** - Gaussian samples
- **Uniform Distribution Sampling** - Uniform samples
- **Bootstrap Sampling** - Resampling with replacement
- **Confidence Intervals** - Bootstrap-based CI estimation

**Features**:
- Distribution comparisons
- Bootstrap for mean, median, std dev
- Visual confidence intervals
- Statistical inference examples

**Requirements**: Python 3.8+, matplotlib, numpy, datascienceutils

---

### 4. [Causal Inference Basics](09_causal_inference_basics.ipynb)
**Status**: ✅ Complete (Python kernel)

Basic causal inference methods:
- **Average Treatment Effect (ATE)** - Regression-based estimation
- **Propensity Score Matching (PSM)** - Matching treated/control units
- **Instrumental Variables (IV)** - Two-stage least squares
- **Difference-in-Differences (DiD)** - Panel data analysis

**Features**:
- Synthetic data examples for each method
- Step-by-step explanations
- Comprehensive visualizations
- Method comparison and interpretation

**Requirements**: Python 3.8+, matplotlib, numpy, datascienceutils

---

### 5. [Advanced Causal Estimators](10_advanced_causal_estimators.ipynb)
**Status**: ✅ Complete (Python kernel)

Advanced causal inference techniques:
- **Regression Discontinuity Design (RDD)** - Sharp cutoff analysis
- **Synthetic Control Method** - Comparative case studies
- **Mediation Analysis** - Direct and indirect effects
- **Conditional Average Treatment Effect (CATE)** - Heterogeneous treatment effects

**Features**:
- Real-world inspired scenarios
- Before/after visualizations
- Effect decomposition
- Subgroup analysis

**Requirements**: Python 3.8+, matplotlib, numpy, datascienceutils

---

### 6. [Causal Graph Operations](11_causal_graph_operations.ipynb)
**Status**: ✅ Complete (Python kernel)

Causal graph construction and manipulation:
- Creating and querying causal graphs
- Adding weighted edges
- Freezing edges/subgraphs (domain knowledge)
- DOT export for Graphviz visualization
- Common causal structures (confounding, mediation, collider, IV)

**Features**:
- Multiple graph examples
- Graph visualization with Graphviz
- Best practices for graph construction
- Real-world complex examples

**Requirements**: Python 3.8+, datascienceutils
**Optional**: graphviz (for rendering DOT files)

---

## Setup Instructions

### Python Notebooks (06, 07, 09, 10, 11)

1. **Install datascienceutils**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install maturin
pip install maturin

# Build and install datascienceutils
cd /path/to/data-science-utils
maturin develop --release
```

2. **Install Jupyter and dependencies**:
```bash
pip install jupyter matplotlib numpy
```

3. **Run notebook**:
```bash
# Run Jupyter in the notebooks directory
jupyter notebook notebooks/

# Or open a specific notebook
jupyter notebook notebooks/09_causal_inference_basics.ipynb
```

### Rust Notebook (05)

1. **Install evcxr Jupyter kernel**:
```bash
cargo install evcxr_jupyter
evcxr_jupyter --install
```

2. **Run notebook**:
```bash
jupyter notebook notebooks/05_causal_analysis_demo.ipynb
```

## Quick Start

```python
# Example: Outlier Detection
import datascienceutils as dsu
import numpy as np

data = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
outliers, lower, upper = dsu.detect_outliers_iqr(data, k=1.5)
print(f"Outliers at indices: {outliers}")
```

```python
# Example: Bootstrap Sampling
samples = dsu.bootstrap_sample(data, n_samples=1000)
means = [s.mean() for s in samples]
ci = (np.percentile(means, 2.5), np.percentile(means, 97.5))
print(f"95% CI: {ci}")
```
