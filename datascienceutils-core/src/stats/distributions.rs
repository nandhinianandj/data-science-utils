//! Distribution analysis and fitting

use crate::error::{DsuError, DsuResult};
use ndarray::ArrayView1;
use statrs::distribution::{ContinuousCDF, Normal};

/// Kolmogorov-Smirnov test for distribution similarity
///
/// # Arguments
/// * `data` - Sample data
/// * `distribution` - Distribution name ("normal", "uniform", etc.)
///
/// # Returns
/// Tuple of (KS statistic, p-value approximation)
pub fn ks_test(data: &ArrayView1<f64>, distribution: &str) -> DsuResult<(f64, f64)> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted_data.len() as f64;

    // Calculate empirical CDF vs theoretical CDF
    let ks_stat = match distribution {
        "normal" => {
            // Fit normal distribution to data
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (data.len() - 1) as f64;
            let std_dev = variance.sqrt();
            
            let normal = Normal::new(mean, std_dev)
                .map_err(|e| DsuError::StatisticalError(e.to_string()))?;
            
            // Calculate KS statistic
            let mut max_diff: f64 = 0.0;
            for (i, &x) in sorted_data.iter().enumerate() {
                let empirical_cdf = (i + 1) as f64 / n;
                let theoretical_cdf = normal.cdf(x);
                let diff = (empirical_cdf - theoretical_cdf).abs();
                max_diff = max_diff.max(diff);
            }
            max_diff
        }
        _ => {
            return Err(DsuError::InvalidParameter(
                format!("Unknown distribution: {}", distribution),
            ));
        }
    };

    // Approximate p-value using Kolmogorov distribution
    // This is a simplified approximation
    let p_value = (-2.0 * n * ks_stat * ks_stat).exp();

    Ok((ks_stat, p_value.max(0.0).min(1.0)))
}

/// Check which distribution best fits the data
///
/// # Arguments
/// * `data` - Sample data
/// * `distributions` - List of distribution names to test
///
/// # Returns
/// Vector of (distribution name, KS statistic, p-value) sorted by best fit
pub fn check_distribution(
    data: &ArrayView1<f64>,
    distributions: &[&str],
) -> DsuResult<Vec<(String, f64, f64)>> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let mut results = Vec::new();

    for &dist in distributions {
        match ks_test(data, dist) {
            Ok((ks_stat, p_value)) => {
                results.push((dist.to_string(), ks_stat, p_value));
            }
            Err(_) => continue,
        }
    }

    // Sort by p-value (descending) - higher p-value means better fit
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ks_test_normal() {
        // Generate approximately normal data
        let data = array![
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0,
            -1.8, -1.2, -0.8, 0.2, 0.8, 1.2, 1.8
        ];
        
        let result = ks_test(&data.view(), "normal");
        assert!(result.is_ok());
        
        let (ks_stat, p_value) = result.unwrap();
        assert!(ks_stat >= 0.0 && ks_stat <= 1.0);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    fn test_check_distribution() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let distributions = vec!["normal"];
        
        let result = check_distribution(&data.view(), &distributions);
        assert!(result.is_ok());
        
        let results = result.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "normal");
    }

    #[test]
    fn test_ks_test_unknown_distribution() {
        let data = array![1.0, 2.0, 3.0];
        let result = ks_test(&data.view(), "unknown");
        assert!(result.is_err());
    }
}
