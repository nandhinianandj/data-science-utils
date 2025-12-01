use ndarray::ArrayView2;
use crate::error::{DsuError, DsuResult};
use linfa::prelude::*;
use linfa_clustering::KMeans;


/// K-Means clustering result
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster labels
    pub labels: Vec<usize>,
    /// Centroids
    pub centroids: Vec<Vec<f64>>,
    /// Inertia (sum of squared distances to closest centroid)
    pub inertia: f64,
}

/// Perform K-Means clustering
///
/// # Arguments
/// * `data` - Input data (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Tolerance for convergence
/// * `seed` - Random seed
///
/// # Returns
/// * `KMeansResult` struct
pub fn kmeans_cluster(
    data: ArrayView2<f64>,
    n_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    _seed: u64,
) -> DsuResult<KMeansResult> {
    // Create linfa dataset
    let dataset = DatasetBase::from(data.to_owned());
    
    // Configure K-Means
    // Note: Seed is ignored in this version due to linfa version compatibility issues
    let model = KMeans::params(n_clusters)
        .max_n_iterations(max_iter as u64)
        .tolerance(tolerance)
        .fit(&dataset)
        .map_err(|e| DsuError::ClusteringError(format!("K-Means error: {:?}", e)))?;
        
    let labels = model.predict(&dataset);
    let centroids = model.centroids().outer_iter().map(|row| row.to_vec()).collect();
    
    // Calculate inertia (approximation or manual calculation if not exposed)
    // linfa KMeans doesn't seem to expose inertia directly in the struct easily in all versions,
    // but we can calculate it.
    // Inertia = sum(dist(x, centroid(x))^2)
    
    let mut inertia = 0.0;
    let centroids_arr = model.centroids();
    
    for (i, x) in data.outer_iter().enumerate() {
        let label = labels[i];
        let centroid = centroids_arr.row(label);
        let dist_sq = (&x - &centroid).mapv(|v| v.powi(2)).sum();
        inertia += dist_sq;
    }
    
    Ok(KMeansResult {
        labels: labels.to_vec(),
        centroids,
        inertia,
    })
}
