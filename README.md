# data-science-utils

A comprehensive Rust library for data science, machine learning, and statistical analysis.

## Features

### 🧮 Core Data Science
- **Data Analysis**: Statistical functions, correlation, hypothesis testing
- **Data Manipulation**: DataFrame operations, feature engineering
- **Visualization**: Plotting utilities with Plotters
- **Clustering**: K-means, DBSCAN, hierarchical clustering
- **Sampling**: Stratified, reservoir, bootstrap sampling
- **Outlier Detection**: IQR, Z-score, isolation forest methods

### 🧠 Neural Networks (NEW!)
- **Transfer Learning**: Fine-tune pre-trained models with easy wrappers
- **Edge Deployment**: Quantize models for Raspberry Pi, ARM Cortex-M, NVIDIA Jetson
- **Model Architectures**: MLP, CNN, RNN/LSTM configurations
- **ONNX Support**: Load and run pre-trained models
- **Quantization**: INT8 (4x), INT4 (8x), Float16 (2x) compression
- **Burn Framework**: Pure Rust deep learning (v0.20)

### 📊 Advanced Analytics
- **Time Series**: Forecasting, trend analysis
- **Linear Algebra**: Matrix operations, decompositions
- **Predictive Modeling**: Integration with SmartCore

## Installation

```toml
[dependencies]
datascienceutils-core = "0.1"

# With neural networks
datascienceutils-core = { version = "0.1", features = ["neural-networks", "quantization"] }
```

## Quick Examples

### Neural Network Transfer Learning
```rust
use datascienceutils_core::nn::{fine_tune, TransferConfig, optimize_for_device, EdgeDevice};

// Fine-tune a pre-trained model
let config = TransferConfig::fine_tune(0.0001, 10);
let model = fine_tune("resnet50.onnx", config)?;

// Optimize for Raspberry Pi deployment
let optimized = optimize_for_device("model.onnx", EdgeDevice::RaspberryPi)?;
optimized.export_tflite("model.tflite")?;
```

### Data Analysis
```rust
use datascienceutils_core::stats::correlation::pearson_correlation;
use datascienceutils_core::cluster::kmeans;

// Statistical analysis
let corr = pearson_correlation(&x, &y)?;

// Clustering
let (labels, centroids) = kmeans(&data, 3, 100)?;
```

## Documentation

- **Notebooks**: See [notebooks/](notebooks/) for Jupyter examples
- **API Docs**: Run `cargo doc --open`
- **Examples**: See [examples/](https://github.com/anandjeyahar/mlDemoExamples)

## Python Bindings

Python bindings available via `datascienceutils-py` package.

## License

See LICENSE file for details.
