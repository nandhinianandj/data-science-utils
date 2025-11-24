//! Model quantization for edge deployment
//!
//! Utilities for quantizing neural network models for deployment on edge devices.

use crate::error::{DsuError, DsuResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Quantization method
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// 8-bit integer quantization (4x size reduction)
    INT8,
    /// 4-bit integer quantization (8x size reduction)
    INT4,
    /// 16-bit float quantization (2x size reduction)
    Float16,
    /// Dynamic range quantization (weights only)
    DynamicRange,
}

impl QuantizationMethod {
    /// Get the expected compression ratio
    pub fn compression_ratio(&self) -> f64 {
        match self {
            Self::INT8 => 4.0,
            Self::INT4 => 8.0,
            Self::Float16 => 2.0,
            Self::DynamicRange => 4.0,
        }
    }

    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::INT8 => "INT8",
            Self::INT4 => "INT4",
            Self::Float16 => "Float16",
            Self::DynamicRange => "Dynamic Range",
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method
    pub method: QuantizationMethod,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Calibration data size (for PTQ)
    pub calibration_samples: Option<usize>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            method: QuantizationMethod::INT8,
            symmetric: true,
            calibration_samples: Some(100),
        }
    }
}

impl QuantizationConfig {
    /// Create INT8 quantization config
    pub fn int8() -> Self {
        Self {
            method: QuantizationMethod::INT8,
            symmetric: true,
            calibration_samples: Some(100),
        }
    }

    /// Create INT4 quantization config (maximum compression)
    pub fn int4() -> Self {
        Self {
            method: QuantizationMethod::INT4,
            symmetric: true,
            calibration_samples: Some(200),
        }
    }

    /// Create Float16 quantization config (balanced)
    pub fn float16() -> Self {
        Self {
            method: QuantizationMethod::Float16,
            symmetric: true,
            calibration_samples: None,
        }
    }

    /// Create dynamic range quantization config
    pub fn dynamic_range() -> Self {
        Self {
            method: QuantizationMethod::DynamicRange,
            symmetric: false,
            calibration_samples: None,
        }
    }
}

/// Quantized model for edge deployment
#[derive(Debug)]
pub struct QuantizedModel {
    /// Original model size in bytes
    original_size: usize,
    /// Quantized model size in bytes
    quantized_size: usize,
    /// Quantization method used
    method: QuantizationMethod,
}

impl QuantizedModel {
    /// Get model size in bytes
    pub fn size_bytes(&self) -> usize {
        self.quantized_size
    }

    /// Get original model size in MB
    pub fn original_size_mb(&self) -> f64 {
        self.original_size as f64 / (1024.0 * 1024.0)
    }

    /// Get quantized model size in MB
    pub fn quantized_size_mb(&self) -> f64 {
        self.quantized_size as f64 / (1024.0 * 1024.0)
    }

    /// Get compression ratio achieved
    pub fn compression_ratio(&self) -> f64 {
        self.original_size as f64 / self.quantized_size as f64
    }

    /// Get quantization statistics
    pub fn stats(&self) -> QuantizationStats {
        QuantizationStats {
            original_size_mb: self.original_size_mb(),
            quantized_size_mb: self.quantized_size_mb(),
            compression_ratio: self.compression_ratio(),
            method: self.method,
        }
    }

    /// Export to TFLite format for mobile deployment
    pub fn export_tflite<P: AsRef<Path>>(&self, _path: P) -> DsuResult<()> {
        Err(DsuError::NotImplemented(
            "TFLite export not yet implemented".to_string(),
        ))
    }

    /// Export to ONNX quantized format
    pub fn export_onnx<P: AsRef<Path>>(&self, _path: P) -> DsuResult<()> {
        Err(DsuError::NotImplemented(
            "ONNX export not yet implemented".to_string(),
        ))
    }
}

/// Quantization statistics
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Original model size in MB
    pub original_size_mb: f64,
    /// Quantized model size in MB
    pub quantized_size_mb: f64,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Quantization method used
    pub method: QuantizationMethod,
}

/// Quantize a model for edge deployment
///
/// # Arguments
/// * `model_path` - Path to the model to quantize
/// * `config` - Quantization configuration
///
/// # Example
/// ```rust,ignore
/// use datascienceutils_core::nn::quantization::{quantize_model, QuantizationConfig};
///
/// let config = QuantizationConfig::int8();
/// let quantized = quantize_model("model.onnx", config)?;
/// println!("Compression: {}x", quantized.compression_ratio());
/// ```
pub fn quantize_model<P: AsRef<Path>>(
    _model_path: P,
    _config: QuantizationConfig,
) -> DsuResult<QuantizedModel> {
    // TODO: Implement with ONNX Runtime quantization
    Err(DsuError::NotImplemented(
        "Model quantization not yet implemented".to_string(),
    ))
}

/// Edge device types for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDevice {
    /// ARM Cortex-M (microcontrollers)
    CortexM,
    /// ARM Cortex-A (mobile/embedded)
    CortexA,
    /// RISC-V processors
    RiscV,
    /// x86 edge servers
    X86,
    /// Raspberry Pi
    RaspberryPi,
    /// NVIDIA Jetson
    Jetson,
}

impl EdgeDevice {
    /// Get recommended quantization method for this device
    pub fn recommended_quantization(&self) -> QuantizationMethod {
        match self {
            Self::CortexM => QuantizationMethod::INT4,  // Most constrained
            Self::CortexA | Self::RaspberryPi => QuantizationMethod::INT8,
            Self::RiscV => QuantizationMethod::INT8,
            Self::X86 | Self::Jetson => QuantizationMethod::Float16,
        }
    }

    /// Get device name
    pub fn name(&self) -> &'static str {
        match self {
            Self::CortexM => "ARM Cortex-M",
            Self::CortexA => "ARM Cortex-A",
            Self::RiscV => "RISC-V",
            Self::X86 => "x86",
            Self::RaspberryPi => "Raspberry Pi",
            Self::Jetson => "NVIDIA Jetson",
        }
    }
}

/// Optimize model for specific edge device
///
/// # Arguments
/// * `model_path` - Path to the model
/// * `device` - Target edge device
///
/// # Example
/// ```rust,ignore
/// use datascienceutils_core::nn::quantization::{optimize_for_device, EdgeDevice};
///
/// let optimized = optimize_for_device("model.onnx", EdgeDevice::RaspberryPi)?;
/// ```
pub fn optimize_for_device<P: AsRef<Path>>(
    model_path: P,
    device: EdgeDevice,
) -> DsuResult<QuantizedModel> {
    let method = device.recommended_quantization();
    let config = match method {
        QuantizationMethod::INT8 => QuantizationConfig::int8(),
        QuantizationMethod::INT4 => QuantizationConfig::int4(),
        QuantizationMethod::Float16 => QuantizationConfig::float16(),
        QuantizationMethod::DynamicRange => QuantizationConfig::dynamic_range(),
    };
    
    quantize_model(model_path, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_method_compression() {
        assert_eq!(QuantizationMethod::INT8.compression_ratio(), 4.0);
        assert_eq!(QuantizationMethod::INT4.compression_ratio(), 8.0);
        assert_eq!(QuantizationMethod::Float16.compression_ratio(), 2.0);
    }

    #[test]
    fn test_quantization_config_int8() {
        let config = QuantizationConfig::int8();
        assert_eq!(config.method, QuantizationMethod::INT8);
        assert!(config.symmetric);
    }

    #[test]
    fn test_quantization_config_int4() {
        let config = QuantizationConfig::int4();
        assert_eq!(config.method, QuantizationMethod::INT4);
    }

    #[test]
    fn test_edge_device_recommendations() {
        assert_eq!(
            EdgeDevice::CortexM.recommended_quantization(),
            QuantizationMethod::INT4
        );
        assert_eq!(
            EdgeDevice::RaspberryPi.recommended_quantization(),
            QuantizationMethod::INT8
        );
        assert_eq!(
            EdgeDevice::Jetson.recommended_quantization(),
            QuantizationMethod::Float16
        );
    }

    #[test]
    fn test_edge_device_names() {
        assert_eq!(EdgeDevice::RaspberryPi.name(), "Raspberry Pi");
        assert_eq!(EdgeDevice::Jetson.name(), "NVIDIA Jetson");
    }
}
