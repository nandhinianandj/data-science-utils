use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::Eig;
use crate::error::{DsuError, DsuResult};
use crate::cluster::kmeans::kmeans_cluster;

/// Spectral Clustering result
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Cluster labels
    pub labels: Vec<usize>,
}

/// Perform Spectral Clustering
///
/// # Arguments
/// * `data` - Input data (n_samples x n_features)
/// * `n_clusters` - Number of clusters
/// * `gamma` - RBF kernel coefficient
/// * `seed` - Random seed
///
/// # Returns
/// * `SpectralResult` struct
pub fn spectral_cluster(
    data: ArrayView2<f64>,
    n_clusters: usize,
    gamma: f64,
    seed: u64,
) -> DsuResult<SpectralResult> {
    let n_samples = data.nrows();
    
    // 1. Compute Affinity Matrix (RBF Kernel)
    // A_ij = exp(-gamma * ||x_i - x_j||^2)
    let mut affinity = Array2::<f64>::zeros((n_samples, n_samples));
    
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i == j {
                affinity[[i, j]] = 1.0;
            } else {
                let diff = &data.row(i) - &data.row(j);
                let dist_sq = diff.dot(&diff);
                affinity[[i, j]] = (-gamma * dist_sq).exp();
            }
        }
    }
    
    // 2. Compute Laplacian
    // D_ii = sum_j A_ij
    // L = D^(-1/2) * A * D^(-1/2) (Normalized Laplacian)
    // Actually, usually we solve L v = lambda D v or similar.
    // Let's use the symmetric normalized Laplacian: L_sym = I - D^(-1/2) W D^(-1/2)
    // Eigenvectors of L_sym corresponding to smallest eigenvalues are used.
    // Equivalently, eigenvectors of D^(-1/2) W D^(-1/2) corresponding to largest eigenvalues.
    
    let mut d_inv_sqrt = Array2::<f64>::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        let sum = affinity.row(i).sum();
        if sum > 1e-10 {
            d_inv_sqrt[[i, i]] = 1.0 / sum.sqrt();
        }
    }
    
    let l_sym = d_inv_sqrt.dot(&affinity).dot(&d_inv_sqrt);
    
    // 3. Eigen decomposition
    // We want k largest eigenvectors of D^(-1/2) W D^(-1/2)
    let (eigenvalues, eigenvectors) = l_sym.eig()
        .map_err(|e| DsuError::ClusteringError(format!("Eigen decomposition error: {:?}", e)))?;
        
    // Extract real parts (assuming symmetric matrix gives real eigenvalues)
    let mut eig_pairs: Vec<(f64, Array1<f64>)> = eigenvalues.iter()
        .zip(eigenvectors.axis_iter(Axis(1)))
        .map(|(val, vec)| (val.re, vec.mapv(|v| v.re)))
        .collect();
        
    // Sort by eigenvalue descending
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    // Select top k eigenvectors
    let mut embedding = Array2::<f64>::zeros((n_samples, n_clusters));
    for k in 0..n_clusters {
        embedding.column_mut(k).assign(&eig_pairs[k].1);
    }
    
    // Normalize rows
    for i in 0..n_samples {
        let norm = embedding.row(i).mapv(|v| v.powi(2)).sum().sqrt();
        if norm > 1e-10 {
            let mut row = embedding.row_mut(i);
            row /= norm;
        }
    }
    
    // 4. K-Means on embedding
    let kmeans_res = kmeans_cluster(embedding.view(), n_clusters, 100, 1e-4, seed)?;
    
    Ok(SpectralResult {
        labels: kmeans_res.labels,
    })
}
