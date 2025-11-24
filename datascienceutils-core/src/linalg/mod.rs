//! # Linear Algebra Module
//!
//! This module provides linear algebra operations using ndarray-linalg.
//! It includes functionality for:
//! - Eigenvalue decomposition
//! - Singular Value Decomposition (SVD)
//! - Matrix determinants
//! - Matrix inverse
//! - Solving linear systems
//! - Trace operations
//!
//! # Examples
//!
//! ```rust
//! use datascienceutils_core::linalg::*;
//! use ndarray::array;
//!
//! // Compute eigenvalues
//! let matrix = array![[1.0, 2.0], [3.0, 4.0]];
//! // let eigenvalues = matrix.eig().unwrap();
//! ```

// Re-export ndarray-linalg functionality
pub use ndarray_linalg::*;

use crate::error::{DsuError, DsuResult};
use ndarray::{Array1, Array2};

/// Compute the eigenvalues and eigenvectors of a square matrix
///
/// # Arguments
/// * `matrix` - A square matrix
///
/// # Returns
/// A tuple of (eigenvalues, eigenvectors)
///
/// # Example
/// ```rust,ignore
/// use ndarray::array;
/// use datascienceutils_core::linalg::compute_eigenvalues;
///
/// let matrix = array![[4.0, -2.0], [1.0, 1.0]];
/// let (eigenvalues, eigenvectors) = compute_eigenvalues(&matrix).unwrap();
/// ```
pub fn compute_eigenvalues(
    matrix: &Array2<f64>,
) -> DsuResult<(Array1<ndarray_linalg::types::c64>, Array2<ndarray_linalg::types::c64>)> {
    use ndarray_linalg::Eig;
    
    matrix
        .eig()
        .map_err(|e| DsuError::NumericalError(format!("Eigenvalue computation failed: {}", e)))
}

/// Compute the Singular Value Decomposition (SVD) of a matrix
///
/// # Arguments
/// * `matrix` - Input matrix
///
/// # Returns
/// A tuple of (U, singular_values, Vt) where A = U * Σ * Vt
///
/// # Example
/// ```rust,ignore
/// use ndarray::array;
/// use datascienceutils_core::linalg::compute_svd;
///
/// let matrix = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let (u, s, vt) = compute_svd(&matrix).unwrap();
/// ```
pub fn compute_svd(
    matrix: &Array2<f64>,
) -> DsuResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    use ndarray_linalg::SVD;
    
    matrix
        .svd(true, true)
        .map(|(u, s, vt)| (u.unwrap(), s, vt.unwrap()))
        .map_err(|e| DsuError::NumericalError(format!("SVD computation failed: {}", e)))
}

/// Compute the determinant of a square matrix
///
/// # Arguments
/// * `matrix` - A square matrix
///
/// # Returns
/// The determinant value
///
/// # Example
/// ```rust,ignore
/// use ndarray::array;
/// use datascienceutils_core::linalg::compute_determinant;
///
/// let matrix = array![[1.0, 2.0], [3.0, 4.0]];
/// let det = compute_determinant(&matrix).unwrap();
/// ```
pub fn compute_determinant(matrix: &Array2<f64>) -> DsuResult<f64> {
    use ndarray_linalg::Determinant;
    
    matrix
        .det()
        .map_err(|e| DsuError::NumericalError(format!("Determinant computation failed: {}", e)))
}

/// Compute the inverse of a square matrix
///
/// # Arguments
/// * `matrix` - A square invertible matrix
///
/// # Returns
/// The inverse matrix
///
/// # Example
/// ```rust,ignore
/// use ndarray::array;
/// use datascienceutils_core::linalg::compute_inverse;
///
/// let matrix = array![[1.0, 2.0], [3.0, 4.0]];
/// let inv = compute_inverse(&matrix).unwrap();
/// ```
pub fn compute_inverse(matrix: &Array2<f64>) -> DsuResult<Array2<f64>> {
    use ndarray_linalg::Inverse;
    
    matrix
        .inv()
        .map_err(|e| DsuError::NumericalError(format!("Matrix inversion failed: {}", e)))
}

/// Solve a linear system Ax = b
///
/// # Arguments
/// * `a` - Coefficient matrix A
/// * `b` - Right-hand side vector or matrix b
///
/// # Returns
/// Solution vector or matrix x
///
/// # Example
/// ```rust,ignore
/// use ndarray::array;
/// use datascienceutils_core::linalg::solve_linear_system;
///
/// let a = array![[3.0, 1.0], [1.0, 2.0]];
/// let b = array![9.0, 8.0];
/// let x = solve_linear_system(&a, &b).unwrap();
/// ```
pub fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> DsuResult<Array1<f64>> {
    use ndarray_linalg::Solve;
    
    a.solve(b)
        .map_err(|e| DsuError::NumericalError(format!("Linear system solve failed: {}", e)))
}

/// Compute the trace of a square matrix (sum of diagonal elements)
///
/// # Arguments
/// * `matrix` - A square matrix
///
/// # Returns
/// The trace value
///
/// # Example
/// ```rust,ignore
/// use ndarray::array;
/// use datascienceutils_core::linalg::compute_trace;
///
/// let matrix = array![[1.0, 2.0], [3.0, 4.0]];
/// let trace = compute_trace(&matrix).unwrap();
/// ```
pub fn compute_trace(matrix: &Array2<f64>) -> DsuResult<f64> {
    use ndarray_linalg::Trace;
    
    Ok(matrix.trace().map_err(|e| {
        DsuError::NumericalError(format!("Trace computation failed: {}", e))
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_determinant() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let det = compute_determinant(&matrix).unwrap();
        assert_abs_diff_eq!(det, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trace() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let trace = compute_trace(&matrix).unwrap();
        assert_abs_diff_eq!(trace, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse() {
        let matrix = array![[4.0, 7.0], [2.0, 6.0]];
        let inv = compute_inverse(&matrix).unwrap();
        
        // Check that A * A^-1 = I
        let identity = matrix.dot(&inv);
        assert_abs_diff_eq!(identity[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(identity[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(identity[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(identity[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_linear_system() {
        // Solve: 3x + y = 9, x + 2y = 8
        // Solution: x = 2, y = 3
        let a = array![[3.0, 1.0], [1.0, 2.0]];
        let b = array![9.0, 8.0];
        let x = solve_linear_system(&a, &b).unwrap();
        
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svd() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (u, s, vt) = compute_svd(&matrix).unwrap();
        
        // Check dimensions
        assert_eq!(u.shape(), &[3, 3]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), &[2, 2]);
        
        // Singular values should be positive and in descending order
        assert!(s[0] > s[1]);
        assert!(s[1] > 0.0);
    }
}
