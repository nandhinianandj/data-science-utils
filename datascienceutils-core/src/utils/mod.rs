//! Utility functions for data science operations
//!
//! This module provides various utility functions including:
//! - Type checking for numeric data
//! - Missing value analysis
//! - Data chunking and iteration
//! - Bayesian blocks algorithm
//! - Fractal dimension calculation
//! - Memoization utilities

use ndarray::ArrayView1;
use polars::prelude::*;

pub mod bayesian;
pub mod fractal;
pub mod memoization;
pub mod types;

pub use bayesian::bayesian_blocks;
pub use fractal::fractaldim;
pub use memoization::{memoized_fibonacci, Memoize};
pub use types::*;

/// Calculate the percentage of missing values in a series
///
/// # Arguments
/// * `series` - Array of values (NaN represents missing)
///
/// # Returns
/// Percentage of NaN values (0.0 to 100.0)
pub fn na_pct(series: &ArrayView1<f64>) -> f64 {
    if series.is_empty() {
        return 0.0;
    }
    
    let na_count = series.iter().filter(|x| x.is_nan()).count();
    (na_count as f64 / series.len() as f64) * 100.0
}

/// Split an iterator into chunks of specified size
///
/// # Arguments
/// * `data` - Vector to chunk
/// * `size` - Size of each chunk
///
/// # Returns
/// Vector of chunks
pub fn chunks<T: Clone>(data: &[T], size: usize) -> Vec<Vec<T>> {
    data.chunks(size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Round up a floating point number to the nearest integer
///
/// # Arguments
/// * `x` - Number to round up
///
/// # Returns
/// Rounded up integer value
pub fn roundup(x: f64) -> i64 {
    x.ceil() as i64
}

/// Create a timestamp from a datetime-like value
///
/// # Arguments
/// * `datetime_obj` - Unix timestamp or similar
///
/// # Returns
/// Formatted timestamp
pub fn timestamp(datetime_obj: i64) -> String {
    // Simple implementation - can be enhanced with chrono
    format!("{}", datetime_obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_na_pct() {
        let data = array![1.0, 2.0, f64::NAN, 4.0, f64::NAN];
        let pct = na_pct(&data.view());
        assert!((pct - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_na_pct_empty() {
        let data = array![];
        let pct = na_pct(&data.view());
        assert_eq!(pct, 0.0);
    }

    #[test]
    fn test_chunks() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let result = chunks(&data, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![1, 2, 3]);
        assert_eq!(result[2], vec![7, 8, 9]);
    }

    #[test]
    fn test_roundup() {
        assert_eq!(roundup(3.1), 4);
        assert_eq!(roundup(3.0), 3);
        assert_eq!(roundup(-2.5), -2);
    }
}
