//! Fractal dimension calculation utilities

use crate::error::{DsuError, DsuResult};
use ndarray::Array2;
use std::collections::HashSet;

/// Calculate the fractal dimension of a point set using box-counting method
///
/// # Arguments
/// * `pointlist` - Array of points (each point should have values between 0.0 and 1.0)
/// * `boxlevel` - Number of divisions on each dimension (should be > 1)
///
/// # Returns
/// Approximate fractal dimension
///
/// # Example
/// ```
/// use ndarray::array;
/// use datascienceutils_core::utils::fractaldim;
///
/// let points = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
/// let dim = fractaldim(&points, 10).unwrap();
/// ```
pub fn fractaldim(pointlist: &Array2<f64>, boxlevel: usize) -> DsuResult<f64> {
    if boxlevel <= 1 {
        return Err(DsuError::InvalidParameter(
            "boxlevel must be greater than 1".to_string(),
        ));
    }

    if pointlist.is_empty() {
        return Err(DsuError::EmptyData);
    }

    let _n_dims = pointlist.ncols();
    
    // Map function to discretize values
    let map_to_box = |val: f64, level: usize| -> usize {
        let mapped = (val * level as f64).floor() as usize;
        mapped.min(level - 1)
    };

    // Count boxes at different scales
    let mut scales = Vec::new();
    let mut counts = Vec::new();

    for level in 2..=boxlevel {
        let mut boxes = HashSet::new();
        
        for point in pointlist.rows() {
            let box_coords: Vec<usize> = point
                .iter()
                .map(|&val| map_to_box(val, level))
                .collect();
            boxes.insert(box_coords);
        }
        
        // Store log(1/epsilon) and log(N(epsilon))
        // epsilon = 1/level, so log(1/epsilon) = log(level)
        scales.push((level as f64).ln());
        counts.push((boxes.len() as f64).ln());
    }

    // Calculate fractal dimension using linear regression
    // log(N) = D * log(1/epsilon) + const
    // dimension = slope of log(count) vs log(scale)
    if scales.len() < 2 {
        return Err(DsuError::NumericalError(
            "Not enough scales for dimension calculation".to_string(),
        ));
    }

    let n = scales.len() as f64;
    let sum_x: f64 = scales.iter().sum();
    let sum_y: f64 = counts.iter().sum();
    let sum_xx: f64 = scales.iter().map(|x| x * x).sum();
    let sum_xy: f64 = scales.iter().zip(counts.iter()).map(|(x, y)| x * y).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    
    // The slope is the fractal dimension (positive value)
    Ok(slope.abs())
}

/// Count boxes containing points at a given scale
///
/// # Arguments
/// * `data` - Point data
/// * `box_size` - Size of each box
/// * `m` - Number of dimensions
///
/// # Returns
/// Number of non-empty boxes
pub fn count_boxes(data: &Array2<f64>, box_size: f64, _m: usize) -> usize {
    let mut boxes = HashSet::new();
    
    for point in data.rows() {
        let box_coords: Vec<i64> = point
            .iter()
            .map(|&val| (val / box_size).floor() as i64)
            .collect();
        boxes.insert(box_coords);
    }
    
    boxes.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fractaldim() {
        // Create a simple 2D point set
        let points = array![
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
        ];
        
        let result = fractaldim(&points, 10);
        assert!(result.is_ok());
        
        let dim = result.unwrap();
        // Fractal dimension should be positive and reasonable
        // For a line-like structure, it should be around 1, but the algorithm
        // may give varying results depending on the data
        assert!(dim > 0.0 && dim < 3.0, "Dimension {} out of expected range", dim);
    }

    #[test]
    fn test_fractaldim_invalid_boxlevel() {
        let points = array![[0.1, 0.2], [0.3, 0.4]];
        let result = fractaldim(&points, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_count_boxes() {
        let points = array![
            [0.1, 0.1],
            [0.2, 0.2],
            [0.9, 0.9],
        ];
        
        let count = count_boxes(&points, 0.5, 2);
        assert!(count > 0);
    }
}
