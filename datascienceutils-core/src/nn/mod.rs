//! # Neural Network Module
//!
//! This module provides neural network training and inference capabilities using the Burn framework.
//!
//! ## Features
//!
//! - **High-level training API**: Simple functions for common tasks
//! - **Pre-built architectures**: MLP, CNN, RNN/LSTM
//! - **ONNX support**: Load pre-trained models
//! - **Model quantization**: Optimize models for edge deployment
//! - **Backend agnostic**: CPU, GPU, or WebAssembly
//!
//! ## Usage
//!
//! Enable the `neural-networks` feature in your `Cargo.toml`:
//!
//! ```toml
//! datascienceutils-core = { version = "0.1", features = ["neural-networks"] }
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use datascienceutils_core::nn::quick::train_classifier;
//! use ndarray::array;
//!
//! // Quick training with sensible defaults
//! let x_train = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
//! let y_train = array![0, 1, 1, 0];  // XOR problem
//!
//! let model = train_classifier(x_train, y_train, None, None)?;
//! let predictions = model.predict(x_test)?;
//! ```

#[cfg(feature = "neural-networks")]
pub mod models;

#[cfg(feature = "neural-networks")]
pub mod trainer;

#[cfg(feature = "neural-networks")]
pub mod optimizers;

#[cfg(feature = "neural-networks")]
pub mod losses;

#[cfg(feature = "neural-networks")]
pub mod quick;

#[cfg(all(feature = "neural-networks", feature = "burn-import"))]
pub mod onnx;

#[cfg(feature = "quantization")]
pub mod quantization;

// Re-export commonly used types
#[cfg(feature = "neural-networks")]
pub use trainer::Trainer;

#[cfg(feature = "neural-networks")]
pub use optimizers::OptimizerConfig;

#[cfg(feature = "neural-networks")]
pub use losses::LossFunction;

#[cfg(feature = "quantization")]
pub use quantization::{QuantizationConfig, QuantizationMethod, QuantizedModel};
