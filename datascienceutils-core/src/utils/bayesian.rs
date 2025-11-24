//! Bayesian blocks algorithm for optimal histogram binning
//!
//! Based on the algorithm outlined in Scargle et al. 2012
//! http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

use crate::error::{DsuError, DsuResult};
use ndarray::{Array1, ArrayView1};

/// Compute optimal histogram bins using Bayesian Blocks algorithm
///
/// This is an incomplete implementation that may fail for some datasets.
/// Alternative fitness functions and prior forms can be found in the paper.
///
/// # Arguments
/// * `t` - Array of data points to bin
///
/// # Returns
/// Array of bin edges
///
/// # Example
/// ```
/// use ndarray::array;
/// use datascienceutils_core::utils::bayesian_blocks;
///
/// let data = array![1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
/// let edges = bayesian_blocks(&data.view()).unwrap();
/// ```
pub fn bayesian_blocks(t: &ArrayView1<f64>) -> DsuResult<Array1<f64>> {
    if t.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let n = t.len();
    
    // Sort the data
    let mut sorted_t = t.to_vec();
    sorted_t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Initialize arrays
    let mut best = vec![0.0; n];
    let mut last = vec![0; n];
    
    // Prior on number of bins (4 - gamma)
    let ncp_prior = 4.0 - (73.53 * (n as f64).powf(0.4)).ln();
    
    // Fitness function (simple version)
    for r in 0..n {
        let mut max_fitness = f64::NEG_INFINITY;
        let mut max_idx = 0;
        
        for k in 0..=r {
            let n_k = (r - k + 1) as f64;
            
            // Simple fitness: log-likelihood - penalty
            let fitness = n_k * n_k.ln() - ncp_prior;
            let total_fitness = if k > 0 { best[k - 1] + fitness } else { fitness };
            
            if total_fitness > max_fitness {
                max_fitness = total_fitness;
                max_idx = k;
            }
        }
        
        best[r] = max_fitness;
        last[r] = max_idx;
    }
    
    // Reconstruct bin edges
    let mut edges = Vec::new();
    let mut idx = n - 1;
    
    loop {
        edges.push(sorted_t[idx]);
        if last[idx] == 0 {
            edges.push(sorted_t[0]);
            break;
        }
        idx = last[idx] - 1;
    }
    
    edges.reverse();
    Ok(Array1::from(edges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bayesian_blocks() {
        let data = array![1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0];
        let result = bayesian_blocks(&data.view());
        assert!(result.is_ok());
        
        let edges = result.unwrap();
        assert!(edges.len() >= 2); // At least start and end
    }

    #[test]
    fn test_bayesian_blocks_empty() {
        let data = array![];
        let result = bayesian_blocks(&data.view());
        assert!(result.is_err());
    }
}
