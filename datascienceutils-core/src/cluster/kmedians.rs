use ndarray::{Array2, ArrayView2};
use crate::error::{DsuError, DsuResult};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// K-Medians clustering result
#[derive(Debug, Clone)]
pub struct KMediansResult {
    /// Cluster labels
    pub labels: Vec<usize>,
    /// Centroids (medians)
    pub centroids: Vec<Vec<f64>>,
    /// Cost (sum of Manhattan distances)
    pub cost: f64,
}

/// Perform K-Medians clustering
///
/// # Arguments
/// * `data` - Input data (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Tolerance for convergence
/// * `seed` - Random seed
///
/// # Returns
/// * `KMediansResult` struct
pub fn kmedians_cluster(
    data: ArrayView2<f64>,
    n_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    seed: u64,
) -> DsuResult<KMediansResult> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    
    if n_samples < n_clusters {
        return Err(DsuError::ClusteringError("n_samples must be >= n_clusters".to_string()));
    }
    
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Initialize centroids randomly from data points
    let mut centroids = Array2::<f64>::zeros((n_clusters, n_features));
    let mut indices: Vec<usize> = (0..n_samples).collect();
    // Fisher-Yates shuffle for random selection
    for i in (1..n_samples).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    
    for k in 0..n_clusters {
        centroids.row_mut(k).assign(&data.row(indices[k]));
    }
    
    let mut labels = vec![0; n_samples];
    let mut cost = f64::MAX;
    
    for _iter in 0..max_iter {
        let mut new_cost = 0.0;
        let mut changed = false;
        
        // Assignment step
        for i in 0..n_samples {
            let point = data.row(i);
            let mut min_dist = f64::MAX;
            let mut best_cluster = 0;
            
            for k in 0..n_clusters {
                // Manhattan distance (L1 norm)
                let centroid = centroids.row(k);
                let dist = (&point - &centroid).mapv(|v| v.abs()).sum();
                
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            
            if labels[i] != best_cluster {
                changed = true;
                labels[i] = best_cluster;
            }
            new_cost += min_dist;
        }
        
        // Check convergence
        if !changed || (cost - new_cost).abs() < tolerance {
            cost = new_cost;
            break;
        }
        cost = new_cost;
        
        // Update step (calculate medians)
        for k in 0..n_clusters {
            // Collect points assigned to cluster k
            let mut cluster_points = Vec::new();
            for i in 0..n_samples {
                if labels[i] == k {
                    cluster_points.push(i);
                }
            }
            
            if cluster_points.is_empty() {
                // Handle empty cluster: re-initialize to a random point
                // For simplicity, just pick a random point from data
                let idx = rng.gen_range(0..n_samples);
                centroids.row_mut(k).assign(&data.row(idx));
                continue;
            }
            
            // Calculate component-wise median
            for j in 0..n_features {
                let mut feature_vals: Vec<f64> = cluster_points.iter()
                    .map(|&idx| data[[idx, j]])
                    .collect();
                
                // Sort to find median
                // Using quick select or just sort
                feature_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = feature_vals.len() / 2;
                let median = if feature_vals.len() % 2 == 0 {
                    (feature_vals[mid - 1] + feature_vals[mid]) / 2.0
                } else {
                    feature_vals[mid]
                };
                
                centroids[[k, j]] = median;
            }
        }
    }
    
    let centroids_vec: Vec<Vec<f64>> = centroids.outer_iter()
        .map(|row| row.to_vec())
        .collect();
        
    Ok(KMediansResult {
        labels,
        centroids: centroids_vec,
        cost,
    })
}
