//! Error types for the datascienceutils library

use thiserror::Error;

/// Result type alias for datascienceutils operations
pub type DsuResult<T> = Result<T, DsuError>;

/// Main error type for datascienceutils operations
#[derive(Error, Debug)]
pub enum DsuError {
    /// Error during numerical computation
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Error during statistical analysis
    #[error("Statistical error: {0}")]
    StatisticalError(String),

    /// Error during clustering
    #[error("Clustering error: {0}")]
    ClusteringError(String),

    /// Error during prediction/classification
    #[error("Prediction error: {0}")]
    PredictionError(String),

    /// Error during plotting
    #[error("Plotting error: {0}")]
    PlottingError(String),

    /// Error during data processing
    #[error("Data processing error: {0}")]
    DataError(String),

    /// Invalid input parameters
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Empty data error
    #[error("Empty data provided")]
    EmptyData,

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Feature not implemented
    #[error("Feature not yet implemented: {0}")]
    NotImplemented(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for DsuError {
    fn from(err: anyhow::Error) -> Self {
        DsuError::Other(err.to_string())
    }
}

impl From<polars::error::PolarsError> for DsuError {
    fn from(err: polars::error::PolarsError) -> Self {
        DsuError::DataError(err.to_string())
    }
}
