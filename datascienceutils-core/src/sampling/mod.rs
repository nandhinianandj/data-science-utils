//! Sampling algorithms and utilities
//!
//! Provides various sampling methods:
//! - Reservoir sampling
//! - Random sampling
//! - Stratified sampling
//! - File sampling

use crate::error::{DsuError, DsuResult};
use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Perform reservoir sampling on a stream of data
///
/// # Arguments
/// * `stream` - Iterator over data
/// * `k` - Number of samples to collect
///
/// # Returns
/// Vector of k samples
pub fn reservoir_sample<T: Clone>(stream: impl Iterator<Item = T>, k: usize) -> Vec<T> {
    let mut reservoir = Vec::with_capacity(k);
    let mut rng = thread_rng();

    for (i, item) in stream.enumerate() {
        if i < k {
            reservoir.push(item);
        } else {
            let j = rng.gen_range(0..=i);
            if j < k {
                reservoir[j] = item;
            }
        }
    }

    reservoir
}

/// Perform random sampling without replacement
///
/// # Arguments
/// * `data` - Input data
/// * `n` - Number of samples
///
/// # Returns
/// Sampled data
pub fn random_sample<T: Clone>(data: &[T], n: usize) -> DsuResult<Vec<T>> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    if n > data.len() {
        return Err(DsuError::InvalidParameter(
            format!("Sample size {} exceeds data size {}", n, data.len()),
        ));
    }

    let mut rng = thread_rng();
    let indices: Vec<usize> = (0..data.len()).collect();
    let sampled_indices = indices.choose_multiple(&mut rng, n);
    
    Ok(sampled_indices.map(|&i| data[i].clone()).collect())
}

/// Perform random sampling with replacement
///
/// # Arguments
/// * `data` - Input data
/// * `n` - Number of samples
///
/// # Returns
/// Sampled data (may contain duplicates)
pub fn random_sample_with_replacement<T: Clone>(data: &[T], n: usize) -> DsuResult<Vec<T>> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let mut rng = thread_rng();
    let samples: Vec<T> = (0..n)
        .map(|_| {
            let idx = rng.gen_range(0..data.len());
            data[idx].clone()
        })
        .collect();

    Ok(samples)
}

/// Perform stratified sampling
///
/// # Arguments
/// * `data` - Input data
/// * `labels` - Stratification labels
/// * `n` - Total number of samples
///
/// # Returns
/// Sampled indices
pub fn stratified_sample(
    data_size: usize,
    labels: &[usize],
    n: usize,
) -> DsuResult<Vec<usize>> {
    if labels.is_empty() {
        return Err(DsuError::EmptyData);
    }

    if labels.len() != data_size {
        return Err(DsuError::DimensionMismatch {
            expected: data_size,
            actual: labels.len(),
        });
    }

    // Count samples per stratum
    let mut strata: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        strata.entry(label).or_insert_with(Vec::new).push(i);
    }

    let n_strata = strata.len();
    let samples_per_stratum = n / n_strata;
    let remainder = n % n_strata;

    let mut rng = thread_rng();
    let mut sampled_indices = Vec::new();

    for (i, (_, indices)) in strata.iter().enumerate() {
        let n_samples = if i < remainder {
            samples_per_stratum + 1
        } else {
            samples_per_stratum
        };

        let n_samples = n_samples.min(indices.len());
        let samples = indices.choose_multiple(&mut rng, n_samples);
        sampled_indices.extend(samples.copied());
    }

    Ok(sampled_indices)
}

/// Sample from a normal distribution
///
/// # Arguments
/// * `mean` - Mean of the distribution
/// * `std_dev` - Standard deviation
/// * `n` - Number of samples
///
/// # Returns
/// Array of samples
pub fn sample_normal(mean: f64, std_dev: f64, n: usize) -> DsuResult<Array1<f64>> {
    if std_dev <= 0.0 {
        return Err(DsuError::InvalidParameter(
            "Standard deviation must be positive".to_string(),
        ));
    }

    let mut rng = thread_rng();
    let normal = rand_distr::Normal::new(mean, std_dev)
        .map_err(|e| DsuError::StatisticalError(e.to_string()))?;

    let samples: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    Ok(Array1::from(samples))
}

/// Sample from a uniform distribution
///
/// # Arguments
/// * `low` - Lower bound
/// * `high` - Upper bound
/// * `n` - Number of samples
///
/// # Returns
/// Array of samples
pub fn sample_uniform(low: f64, high: f64, n: usize) -> DsuResult<Array1<f64>> {
    if low >= high {
        return Err(DsuError::InvalidParameter(
            "Lower bound must be less than upper bound".to_string(),
        ));
    }

    let mut rng = thread_rng();
    let uniform = rand_distr::Uniform::new(low, high);

    let samples: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();
    Ok(Array1::from(samples))
}

/// Bootstrap sampling
///
/// # Arguments
/// * `data` - Input data
/// * `n_samples` - Number of bootstrap samples
///
/// # Returns
/// Vector of bootstrap samples
pub fn bootstrap_sample(data: &ArrayView1<f64>, n_samples: usize) -> DsuResult<Vec<Array1<f64>>> {
    if data.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let data_vec = data.to_vec();
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let sample = random_sample_with_replacement(&data_vec, data.len())?;
        samples.push(Array1::from(sample));
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_reservoir_sample() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sample = reservoir_sample(data.into_iter(), 5);
        
        assert_eq!(sample.len(), 5);
    }

    #[test]
    fn test_random_sample() {
        let data = vec![1, 2, 3, 4, 5];
        let sample = random_sample(&data, 3).unwrap();
        
        assert_eq!(sample.len(), 3);
        for &x in &sample {
            assert!(data.contains(&x));
        }
    }

    #[test]
    fn test_random_sample_with_replacement() {
        let data = vec![1, 2, 3];
        let sample = random_sample_with_replacement(&data, 10).unwrap();
        
        assert_eq!(sample.len(), 10);
    }

    #[test]
    fn test_stratified_sample() {
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let sample = stratified_sample(9, &labels, 6).unwrap();
        
        assert_eq!(sample.len(), 6);
    }

    #[test]
    fn test_sample_normal() {
        let samples = sample_normal(0.0, 1.0, 100).unwrap();
        
        assert_eq!(samples.len(), 100);
        
        // Check that mean is approximately 0
        let mean = samples.sum() / samples.len() as f64;
        assert!(mean.abs() < 0.5);
    }

    #[test]
    fn test_sample_uniform() {
        let samples = sample_uniform(0.0, 10.0, 100).unwrap();
        
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| x >= 0.0 && x < 10.0));
    }

    #[test]
    fn test_bootstrap_sample() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let samples = bootstrap_sample(&data.view(), 10).unwrap();
        
        assert_eq!(samples.len(), 10);
        for sample in samples {
            assert_eq!(sample.len(), 5);
        }
    }

    #[test]
    fn test_random_sample_too_large() {
        let data = vec![1, 2, 3];
        let result = random_sample(&data, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_normal_invalid_std() {
        let result = sample_normal(0.0, -1.0, 10);
        assert!(result.is_err());
    }
}
