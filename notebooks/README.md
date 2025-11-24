# DataScienceUtils - Jupyter Notebooks

This directory contains Jupyter notebooks demonstrating the usage of the Rust-powered DataScienceUtils library.

## Notebooks

### 1. Statistics Tests (`01_statistics_tests.ipynb`)
Demonstrates statistical analysis functions:
- Chi-square tests (independence & goodness of fit)
- Normality testing (Anderson-Darling)
- Correlation analysis (Pearson, Spearman, Kendall)
- Distribution fitting (KS test)
- ANOVA (one-way)

### 2. Outlier Detection (`02_outlier_detection.ipynb`)
Shows outlier detection methods:
- Sigma deviation method
- IQR (Interquartile Range) method
- Z-score method
- Modified Z-score (MAD-based)
- Percentile capping
- Outlier removal

### 3. Sampling & Utilities (`03_sampling_utilities.ipynb`)
Covers sampling algorithms and utilities:
- Distribution sampling (normal, uniform)
- Bootstrap sampling
- Bayesian blocks (optimal binning)
- Fractal dimension calculation

---

## Rust Data Analysis Notebooks

The following notebooks are from the [rust-data-analysis](https://github.com/wiseaidev/rust-data-analysis) repository and demonstrate advanced data analysis techniques using Rust:

### 1. Iris Data Analysis (`1-iris-data-analysis-rust.ipynb`)
Comprehensive analysis of the Iris dataset:
- DataFrame operations with Polars
- Data visualization with Plotters
- Machine learning with SmartCore
- Array operations with ndarray
- **Dependencies**: polars, ndarray, plotters, smartcore

### 2. Ndarray Tutorial (`2-ndarray-tutorial.ipynb`)
Deep dive into ndarray and linear algebra:
- Array creation and manipulation
- Linear algebra operations (eigenvalues, SVD, determinants)
- Matrix operations (inverse, solve, trace)
- Random array generation
- **Dependencies**: ndarray, ndarray-linalg, ndarray-rand

### 3. Polars Tutorial Part 1 (`3-polars-tutorial-part-1.ipynb`)
Introduction to Polars DataFrames:
- DataFrame creation and I/O
- Data selection and filtering
- Date/time operations with chrono
- Lazy evaluation
- **Dependencies**: polars, chrono

### 4. Polars Tutorial Part 2 (`4-polars-tutorial-part-2.ipynb`)
Advanced Polars operations:
- Complex transformations
- Either type for error handling
- Advanced aggregations
- Window functions
- **Dependencies**: polars, either

### 5. Probability Theory Tutorial (`5-probability-theory-tutorial.ipynb`)
Statistical distributions and probability:
- Probability distributions (normal, binomial, etc.)
- Statistical measures
- Visualization with Plotters
- Distribution fitting
- **Dependencies**: statrs, plotters

### 6. Plotters Tutorial Part 1 (`6-plotters-tutorial-part-1.ipynb`)
Data visualization with Plotters:
- Line plots and scatter plots
- Error bars and histograms
- 3D plotting
- Custom styling
- **Dependencies**: plotters, ndarray, ndarray-rand

### 7. Calculus Tutorial Part 1 (`7-calculus-tutorial-part-1.ipynb`)
Numerical calculus operations:
- Derivatives and integrals
- Numerical methods
- Function analysis
- **Dependencies**: Standard library only

## Running the Notebooks

### Setup

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Navigate to the `notebooks/` directory and open any notebook.

### Requirements

All required packages are installed in the virtual environment:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- jupyter
- datascienceutils (Rust-powered)

## Performance

The Rust implementation provides significant performance improvements over pure Python:
- Faster statistical computations
- Efficient outlier detection
- Quick sampling operations
- Optimized numerical algorithms

## Examples

Each notebook includes:
- Clear explanations
- Working code examples
- Visualizations
- Performance comparisons
- Real-world use cases

## Notes

- All functions are drop-in replacements for their Python equivalents
- The API is designed to be familiar to Python data scientists
- Rust's type safety ensures robust error handling
- No GIL limitations for parallel processing
