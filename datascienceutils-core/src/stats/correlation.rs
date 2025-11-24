//! Correlation analysis functions

use crate::error::{DsuError, DsuResult};
use ndarray::ArrayView1;

/// Calculate Spearman rank correlation coefficient
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
/// Spearman correlation coefficient (-1 to 1)
pub fn spearman_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> DsuResult<f64> {
    if x.len() != y.len() {
        return Err(DsuError::DimensionMismatch {
            expected: x.len(),
            actual: y.len(),
        });
    }

    if x.is_empty() {
        return Err(DsuError::EmptyData);
    }

    // Convert to ranks
    let x_ranks = rank_data(x);
    let y_ranks = rank_data(y);

    // Calculate Pearson correlation on ranks
    super::pearson_correlation(&x_ranks.view(), &y_ranks.view())
}

/// Convert data to ranks
fn rank_data(data: &ArrayView1<f64>) -> ndarray::Array1<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    // Sort by value
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    // Assign ranks
    let mut ranks = vec![0.0; n];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = (rank + 1) as f64;
    }
    
    ndarray::Array1::from(ranks)
}

/// Calculate Kendall tau correlation coefficient
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
/// Kendall tau correlation coefficient (-1 to 1)
pub fn kendall_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> DsuResult<f64> {
    if x.len() != y.len() {
        return Err(DsuError::DimensionMismatch {
            expected: x.len(),
            actual: y.len(),
        });
    }

    if x.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let n = x.len();
    let mut concordant = 0;
    let mut discordant = 0;

    // Count concordant and discordant pairs
    for i in 0..n {
        for j in (i + 1)..n {
            let x_diff = x[j] - x[i];
            let y_diff = y[j] - y[i];
            
            if x_diff.signum() == y_diff.signum() {
                concordant += 1;
            } else if x_diff.signum() != 0.0 && y_diff.signum() != 0.0 {
                discordant += 1;
            }
        }
    }

    let total_pairs = (n * (n - 1)) / 2;
    
    if total_pairs == 0 {
        return Ok(0.0);
    }

    Ok((concordant as f64 - discordant as f64) / total_pairs as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_spearman_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = spearman_correlation(&x.view(), &y.view()).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let corr = kendall_correlation(&x.view(), &y.view()).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank_data() {
        let data = array![3.0, 1.0, 2.0, 5.0, 4.0];
        let ranks = rank_data(&data.view());
        
        assert_eq!(ranks[0], 3.0); // 3.0 is rank 3
        assert_eq!(ranks[1], 1.0); // 1.0 is rank 1
        assert_eq!(ranks[2], 2.0); // 2.0 is rank 2
        assert_eq!(ranks[3], 5.0); // 5.0 is rank 5
        assert_eq!(ranks[4], 4.0); // 4.0 is rank 4
    }
}
