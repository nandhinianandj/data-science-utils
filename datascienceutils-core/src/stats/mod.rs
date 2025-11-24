//! Statistical analysis module
//!
//! Provides statistical tests and analysis functions including:
//! - Chi-square tests
//! - Normality tests
//! - Distribution similarity tests
//! - ANOVA
//! - Correlation analysis

use crate::error::{DsuError, DsuResult};
use ndarray::{Array2, ArrayView1};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

pub mod correlation;
pub mod distributions;
pub mod hypothesis_tests;

pub use correlation::*;
pub use distributions::*;
pub use hypothesis_tests::*;

/// Perform chi-square test of independence between two categorical variables
///
/// # Arguments
/// * `observed` - 2D array of observed frequencies
///
/// # Returns
/// Tuple of (chi-square statistic, p-value, degrees of freedom)
pub fn chi2_test_independence(observed: &Array2<f64>) -> DsuResult<(f64, f64, usize)> {
    let (rows, cols) = observed.dim();
    
    if rows < 2 || cols < 2 {
        return Err(DsuError::InvalidParameter(
            "Contingency table must have at least 2 rows and 2 columns".to_string(),
        ));
    }

    // Calculate row and column totals
    let row_totals: Vec<f64> = (0..rows)
        .map(|i| observed.row(i).sum())
        .collect();
    
    let col_totals: Vec<f64> = (0..cols)
        .map(|j| observed.column(j).sum())
        .collect();
    
    let total: f64 = row_totals.iter().sum();
    
    if total == 0.0 {
        return Err(DsuError::StatisticalError(
            "Total count is zero".to_string(),
        ));
    }

    // Calculate expected frequencies and chi-square statistic
    let mut chi_square = 0.0;
    
    for i in 0..rows {
        for j in 0..cols {
            let expected = (row_totals[i] * col_totals[j]) / total;
            if expected > 0.0 {
                let diff = observed[[i, j]] - expected;
                chi_square += (diff * diff) / expected;
            }
        }
    }

    // Degrees of freedom
    let dof = (rows - 1) * (cols - 1);
    
    // Calculate p-value
    let chi_dist = ChiSquared::new(dof as f64)
        .map_err(|e| DsuError::StatisticalError(e.to_string()))?;
    let p_value = 1.0 - chi_dist.cdf(chi_square);

    Ok((chi_square, p_value, dof))
}

/// Perform chi-square goodness of fit test
///
/// # Arguments
/// * `observed` - Observed frequencies
/// * `expected` - Expected frequencies
/// * `ddof` - Delta degrees of freedom (default: 0)
///
/// # Returns
/// Tuple of (chi-square statistic, p-value)
pub fn chi2_test(
    observed: &ArrayView1<f64>,
    expected: &ArrayView1<f64>,
    ddof: usize,
) -> DsuResult<(f64, f64)> {
    if observed.len() != expected.len() {
        return Err(DsuError::DimensionMismatch {
            expected: expected.len(),
            actual: observed.len(),
        });
    }

    if observed.is_empty() {
        return Err(DsuError::EmptyData);
    }

    // Calculate chi-square statistic
    let mut chi_square = 0.0;
    
    for i in 0..observed.len() {
        if expected[i] <= 0.0 {
            return Err(DsuError::StatisticalError(
                "Expected frequencies must be positive".to_string(),
            ));
        }
        
        let diff = observed[i] - expected[i];
        chi_square += (diff * diff) / expected[i];
    }

    // Degrees of freedom
    let dof = observed.len().saturating_sub(1 + ddof);
    
    if dof == 0 {
        return Err(DsuError::StatisticalError(
            "Insufficient degrees of freedom".to_string(),
        ));
    }

    // Calculate p-value
    let chi_dist = ChiSquared::new(dof as f64)
        .map_err(|e| DsuError::StatisticalError(e.to_string()))?;
    let p_value = 1.0 - chi_dist.cdf(chi_square);

    Ok((chi_square, p_value))
}

/// Check normality using Anderson-Darling test
///
/// # Arguments
/// * `data` - Data to test for normality
///
/// # Returns
/// Tuple of (test statistic, critical values, significance levels)
pub fn check_normality(data: &ArrayView1<f64>) -> DsuResult<(f64, Vec<f64>, Vec<f64>)> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let n = data.len() as f64;
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate mean and std dev
    let mean = sorted_data.iter().sum::<f64>() / n;
    let variance = sorted_data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return Err(DsuError::StatisticalError(
            "Standard deviation is zero".to_string(),
        ));
    }

    // Standardize data
    let normal = Normal::new(mean, std_dev)
        .map_err(|e| DsuError::StatisticalError(e.to_string()))?;

    // Calculate Anderson-Darling statistic
    let mut ad_stat = 0.0;
    
    for (i, &x) in sorted_data.iter().enumerate() {
        let phi = normal.cdf(x);
        
        // Avoid log(0) and log(1)
        if phi > 1e-10 && phi < (1.0 - 1e-10) {
            let i_f64 = (i + 1) as f64;
            let n_minus_i = (n - i_f64) as usize;
            let phi_complement = normal.cdf(sorted_data[n_minus_i]);
            
            if phi_complement > 1e-10 && phi_complement < (1.0 - 1e-10) {
                ad_stat += (2.0 * i_f64 - 1.0) * (phi.ln() + (1.0 - phi_complement).ln());
            }
        }
    }
    
    ad_stat = -n - ad_stat / n;
    ad_stat = ad_stat.abs(); // Ensure positive

    // Critical values for different significance levels
    let significance_levels = vec![0.15, 0.10, 0.05, 0.025, 0.01];
    let critical_values = vec![1.621, 1.933, 2.492, 3.070, 3.857];

    Ok((ad_stat, critical_values, significance_levels))
}

/// Calculate Pearson correlation coefficient
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
/// Pearson correlation coefficient (-1 to 1)
pub fn pearson_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> DsuResult<f64> {
    if x.len() != y.len() {
        return Err(DsuError::DimensionMismatch {
            expected: x.len(),
            actual: y.len(),
        });
    }

    if x.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let n = x.len() as f64;
    
    // Calculate means
    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;

    // Calculate correlation
    let mut numerator = 0.0;
    let mut sum_x_sq = 0.0;
    let mut sum_y_sq = 0.0;

    for i in 0..x.len() {
        let x_diff = x[i] - mean_x;
        let y_diff = y[i] - mean_y;
        
        numerator += x_diff * y_diff;
        sum_x_sq += x_diff * x_diff;
        sum_y_sq += y_diff * y_diff;
    }

    let denominator = (sum_x_sq * sum_y_sq).sqrt();
    
    if denominator == 0.0 {
        return Err(DsuError::StatisticalError(
            "Cannot calculate correlation: zero variance".to_string(),
        ));
    }

    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_chi2_test_independence() {
        // Example: Gender vs Preference
        let observed = array![
            [10.0, 20.0, 30.0],
            [15.0, 25.0, 35.0],
        ];
        
        let result = chi2_test_independence(&observed);
        assert!(result.is_ok());
        
        let (chi_sq, p_value, dof) = result.unwrap();
        assert!(chi_sq >= 0.0);
        assert!(p_value >= 0.0 && p_value <= 1.0);
        assert_eq!(dof, 2);
    }

    #[test]
    fn test_chi2_test() {
        let observed = array![10.0, 20.0, 30.0, 40.0];
        let expected = array![15.0, 15.0, 35.0, 35.0];
        
        let result = chi2_test(&observed.view(), &expected.view(), 0);
        assert!(result.is_ok());
        
        let (chi_sq, p_value) = result.unwrap();
        assert!(chi_sq >= 0.0);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    fn test_check_normality() {
        // Generate approximately normal data
        let data = array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0
        ];
        
        let result = check_normality(&data.view());
        assert!(result.is_ok());
        
        let (stat, critical_vals, sig_levels) = result.unwrap();
        assert!(stat >= 0.0);
        assert_eq!(critical_vals.len(), 5);
        assert_eq!(sig_levels.len(), 5);
    }

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = pearson_correlation(&x.view(), &y.view()).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
        
        // Perfect negative correlation
        let y_neg = array![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = pearson_correlation(&x.view(), &y_neg.view()).unwrap();
        assert!((corr_neg + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi2_empty_data() {
        let observed = array![];
        let expected = array![];
        
        let result = chi2_test(&observed.view(), &expected.view(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_pearson_dimension_mismatch() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];
        
        let result = pearson_correlation(&x.view(), &y.view());
        assert!(result.is_err());
    }
}
