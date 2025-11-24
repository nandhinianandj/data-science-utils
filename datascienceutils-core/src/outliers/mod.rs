//! Outlier detection and handling module
//!
//! Provides methods for detecting and handling outliers in data:
//! - Sigma deviation method
//! - IQR (Interquartile Range) method
//! - Percentile capping
//! - Z-score method
//! - Modified Z-score method

use crate::error::{DsuError, DsuResult};
use ndarray::{Array1, ArrayView1};

/// Detect outliers using sigma deviation method
///
/// # Arguments
/// * `data` - Input data
/// * `n_sigma` - Number of standard deviations (default: 3.0)
///
/// # Returns
/// Tuple of (outlier indices, lower bound, upper bound)
pub fn detect_outliers_sigma(
    data: &ArrayView1<f64>,
    n_sigma: f64,
) -> DsuResult<(Vec<usize>, f64, f64)> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let n = data.len() as f64;
    let mean = data.sum() / n;
    let variance = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let lower_bound = mean - n_sigma * std_dev;
    let upper_bound = mean + n_sigma * std_dev;

    let outliers: Vec<usize> = data.iter()
        .enumerate()
        .filter(|(_, &x)| x < lower_bound || x > upper_bound)
        .map(|(i, _)| i)
        .collect();

    Ok((outliers, lower_bound, upper_bound))
}

/// Detect outliers using IQR (Interquartile Range) method
///
/// # Arguments
/// * `data` - Input data
/// * `k` - IQR multiplier (default: 1.5)
///
/// # Returns
/// Tuple of (outlier indices, lower bound, upper bound)
pub fn detect_outliers_iqr(
    data: &ArrayView1<f64>,
    k: f64,
) -> DsuResult<(Vec<usize>, f64, f64)> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1 = percentile(&sorted, 25.0);
    let q3 = percentile(&sorted, 75.0);
    let iqr = q3 - q1;

    let lower_bound = q1 - k * iqr;
    let upper_bound = q3 + k * iqr;

    let outliers: Vec<usize> = data.iter()
        .enumerate()
        .filter(|(_, &x)| x < lower_bound || x > upper_bound)
        .map(|(i, _)| i)
        .collect();

    Ok((outliers, lower_bound, upper_bound))
}

/// Detect outliers using Z-score method
///
/// # Arguments
/// * `data` - Input data
/// * `threshold` - Z-score threshold (default: 3.0)
///
/// # Returns
/// Vector of outlier indices
pub fn detect_outliers_zscore(
    data: &ArrayView1<f64>,
    threshold: f64,
) -> DsuResult<Vec<usize>> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let n = data.len() as f64;
    let mean = data.sum() / n;
    let variance = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return Ok(Vec::new());
    }

    let outliers: Vec<usize> = data.iter()
        .enumerate()
        .filter(|(_, &x)| ((x - mean) / std_dev).abs() > threshold)
        .map(|(i, _)| i)
        .collect();

    Ok(outliers)
}

/// Detect outliers using modified Z-score method (MAD-based)
///
/// # Arguments
/// * `data` - Input data
/// * `threshold` - Modified Z-score threshold (default: 3.5)
///
/// # Returns
/// Vector of outlier indices
pub fn detect_outliers_modified_zscore(
    data: &ArrayView1<f64>,
    threshold: f64,
) -> DsuResult<Vec<usize>> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let median = percentile(&sorted, 50.0);
    
    // Calculate MAD (Median Absolute Deviation)
    let mut deviations: Vec<f64> = data.iter()
        .map(|&x| (x - median).abs())
        .collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mad = percentile(&deviations, 50.0);
    
    if mad == 0.0 {
        return Ok(Vec::new());
    }

    // Modified Z-score = 0.6745 * (x - median) / MAD
    let outliers: Vec<usize> = data.iter()
        .enumerate()
        .filter(|(_, &x)| {
            let modified_z = 0.6745 * (x - median).abs() / mad;
            modified_z > threshold
        })
        .map(|(i, _)| i)
        .collect();

    Ok(outliers)
}

/// Cap outliers using percentile method
///
/// # Arguments
/// * `data` - Input data
/// * `lower_percentile` - Lower percentile (e.g., 5.0)
/// * `upper_percentile` - Upper percentile (e.g., 95.0)
///
/// # Returns
/// Data with outliers capped
pub fn cap_outliers_percentile(
    data: &ArrayView1<f64>,
    lower_percentile: f64,
    upper_percentile: f64,
) -> DsuResult<Array1<f64>> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    if lower_percentile >= upper_percentile {
        return Err(DsuError::InvalidParameter(
            "Lower percentile must be less than upper percentile".to_string(),
        ));
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_bound = percentile(&sorted, lower_percentile);
    let upper_bound = percentile(&sorted, upper_percentile);

    let capped: Vec<f64> = data.iter()
        .map(|&x| {
            if x < lower_bound {
                lower_bound
            } else if x > upper_bound {
                upper_bound
            } else {
                x
            }
        })
        .collect();

    Ok(Array1::from(capped))
}

/// Remove outliers from data
///
/// # Arguments
/// * `data` - Input data
/// * `outlier_indices` - Indices of outliers to remove
///
/// # Returns
/// Data with outliers removed
pub fn remove_outliers(
    data: &ArrayView1<f64>,
    outlier_indices: &[usize],
) -> DsuResult<Array1<f64>> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let filtered: Vec<f64> = data.iter()
        .enumerate()
        .filter(|(i, _)| !outlier_indices.contains(i))
        .map(|(_, &x)| x)
        .collect();

    Ok(Array1::from(filtered))
}

/// Calculate percentile of sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let n = sorted_data.len();
    let index = (p / 100.0) * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_detect_outliers_sigma() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is outlier
        let (outliers, lower, upper) = detect_outliers_sigma(&data.view(), 2.0).unwrap();
        
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0], 5);
        assert!(lower < 1.0);
        assert!(upper < 100.0);
    }

    #[test]
    fn test_detect_outliers_iqr() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let (outliers, _, _) = detect_outliers_iqr(&data.view(), 1.5).unwrap();
        
        assert!(outliers.len() > 0);
        assert!(outliers.contains(&5));
    }

    #[test]
    fn test_detect_outliers_zscore() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = detect_outliers_zscore(&data.view(), 2.0).unwrap();
        
        assert!(outliers.len() > 0);
        assert!(outliers.contains(&5));
    }

    #[test]
    fn test_detect_outliers_modified_zscore() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = detect_outliers_modified_zscore(&data.view(), 3.5).unwrap();
        
        assert!(outliers.len() > 0);
    }

    #[test]
    fn test_cap_outliers_percentile() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let capped = cap_outliers_percentile(&data.view(), 10.0, 90.0).unwrap();
        
        assert!(capped[5] < 100.0); // Last value should be capped
        assert_eq!(capped.len(), data.len());
    }

    #[test]
    fn test_remove_outliers() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outlier_indices = vec![5];
        let filtered = remove_outliers(&data.view(), &outlier_indices).unwrap();
        
        assert_eq!(filtered.len(), 5);
        assert!(!filtered.iter().any(|&x| x == 100.0));
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert_eq!(percentile(&data, 100.0), 5.0);
    }

    #[test]
    fn test_empty_data() {
        let data = array![];
        let result = detect_outliers_sigma(&data.view(), 3.0);
        assert!(result.is_err());
    }
}
