//! # DataScienceUtils Core
//!
//! A comprehensive Rust library for data science utilities including:
//! - Statistical analysis and hypothesis testing
//! - Clustering algorithms and analysis
//! - Dimensionality reduction (PCA, t-SNE, UMAP)
//! - Linear algebra operations (eigenvalues, SVD, determinants, inverse, solve)
//! - Time series analysis
//! - Data visualization
//! - Feature engineering and preprocessing
//!
//! This library is designed to be a high-performance replacement for common
//! Python data science workflows, with particular focus on exploratory data analysis (EDA).

pub mod analyze;
pub mod cluster;
pub mod dataframe;
pub mod error;
pub mod features;
pub mod linalg;

#[cfg(feature = "neural-networks")]
pub mod nn;

pub mod outliers;
pub mod plot;
pub mod predictive;
pub mod sampling;
pub mod sklearn_utils;
pub mod stats;
pub mod timeseries;
pub mod utils;

// Re-export commonly used types
pub use dataframe::DsuDataFrame;
pub use error::{DsuError, DsuResult};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
