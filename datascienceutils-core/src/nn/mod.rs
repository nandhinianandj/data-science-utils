//! # Neural Network Module
//!
//! Comprehensive neural network training, inference, and deployment capabilities.
//!
//! ## Features
//!
//! - **Transfer Learning**: Easy wrappers for fine-tuning pre-trained models
//! - **Model Quantization**: Optimize models for edge deployment
//! - **Pre-built Architectures**: MLP, CNN, RNN/LSTM
//! - **ONNX Support**: Load and run pre-trained models
//! - **Edge Computing**: Device-specific optimizations
//!
//! ## Usage
//!
//! ```toml
//! datascienceutils-core = { version = "0.1", features = ["neural-networks"] }
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

#[cfg(feature = "neural-networks")]
pub mod onnx;

#[cfg(feature = "neural-networks")]
pub mod transfer;

#[cfg(feature = "quantization")]
pub mod quantization;

// Re-export commonly used types
#[cfg(feature = "neural-networks")]
pub use optimizers::OptimizerConfig;

#[cfg(feature = "neural-networks")]
pub use losses::LossFunction;

#[cfg(feature = "neural-networks")]
pub use models::mlp::{Activation, MLPBuilder, MLPConfig};

#[cfg(feature = "neural-networks")]
pub use onnx::ONNXModel;

#[cfg(feature = "neural-networks")]
pub use transfer::{
    fine_tune, feature_extractor, load_pretrained, 
    PretrainedModel, TransferConfig
};

#[cfg(feature = "quantization")]
pub use quantization::{
    optimize_for_device, quantize_model, EdgeDevice, 
    QuantizationConfig, QuantizationMethod, QuantizedModel, QuantizationStats
};
