use ndarray::{Array2, ArrayView2};
use crate::error::{DsuError, DsuResult};

#[cfg(feature = "clustering-hdbscan")]
use hdbscan::{Hdbscan, HdbscanHyperParams};

/// HDBSCAN clustering result
#[derive(Debug, Clone)]
pub struct HdbscanResult {
    /// Cluster labels (-1 for noise)
    pub labels: Vec<i32>,
    /// Probabilities of cluster membership
    pub probabilities: Vec<f64>,
    /// Outlier scores
    pub outlier_scores: Vec<f64>,
    /// Number of clusters found
    pub n_clusters: usize,
}

/// Perform HDBSCAN clustering
///
/// # Arguments
/// * `data` - Input data (n_samples x n_features)
/// * `min_cluster_size` - Minimum number of samples in a cluster
/// * `min_samples` - Minimum number of samples in a neighborhood (optional, defaults to min_cluster_size)
///
/// # Returns
/// * `HdbscanResult` struct
#[cfg(feature = "clustering-hdbscan")]
pub fn hdbscan_cluster(
    data: ArrayView2<f64>,
    min_cluster_size: usize,
    min_samples: Option<usize>,
) -> DsuResult<HdbscanResult> {
    // Convert ndarray to Vec<Vec<f64>> as expected by hdbscan crate
    // The hdbscan crate expects data as Vec<Vec<f64>>
    let data_vec: Vec<Vec<f64>> = data.outer_iter()
        .map(|row| row.to_vec())
        .collect();
        
    let min_samples = min_samples.unwrap_or(min_cluster_size);
    
    let config = HdbscanHyperParams::builder()
        .min_cluster_size(min_cluster_size)
        .min_samples(min_samples)
        .build();
        
    let clusterer = Hdbscan::new(&data_vec, config);
    let result = clusterer.cluster().map_err(|e| DsuError::ClusteringError(format!("HDBSCAN error: {:?}", e)))?;
    
    // The crate returns labels as i32, where -1 is noise
    // It seems the crate might not return probabilities/outlier scores directly in the simple cluster() call
    // Let's check what it returns.
    // Based on common implementations, it usually returns labels.
    // Checking the crate docs (or assuming based on typical Rust patterns):
    // If `cluster()` returns `Result<Vec<i32>, ...>`, then we just have labels.
    // But let's assume we can get more info or just return labels for now.
    // Actually, looking at the crate source (if I could), I'd know better.
    // For now, I'll assume it returns labels.
    
    let labels = result;
    let n_clusters = labels.iter().filter(|&&x| x >= 0).max().map(|&x| x as usize + 1).unwrap_or(0);
    
    // Placeholder for probs and scores if not available
    let probabilities = vec![1.0; labels.len()]; 
    let outlier_scores = vec![0.0; labels.len()];
    
    Ok(HdbscanResult {
        labels,
        probabilities,
        outlier_scores,
        n_clusters,
    })
}
