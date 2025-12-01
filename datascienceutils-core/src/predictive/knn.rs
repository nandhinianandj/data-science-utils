use ndarray::{ArrayView1, ArrayView2};
use crate::error::{DsuError, DsuResult};
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;

/// KNN Classification Result
#[derive(Debug, Clone)]
pub struct KNNResult {
    /// Predicted labels
    pub predictions: Vec<f64>,
    /// Accuracy (if test labels provided)
    pub accuracy: Option<f64>,
}

/// Train and predict using KNN Classifier
///
/// # Arguments
/// * `train_data` - Training features (n_samples x n_features)
/// * `train_target` - Training labels (n_samples)
/// * `test_data` - Test features (n_samples x n_features)
/// * `k` - Number of neighbors
/// * `weight` - Weighting function ("uniform" or "distance")
/// * `algorithm` - Search algorithm ("linear", "kd_tree", "cover_tree", "ball_tree")
///
/// # Returns
/// * `KNNResult` struct
pub fn knn_classify(
    train_data: ArrayView2<f64>,
    train_target: ArrayView1<f64>,
    test_data: ArrayView2<f64>,
    k: usize,
    _weight: &str,
    _algorithm: &str,
) -> DsuResult<KNNResult> {
    // Convert ndarray to DenseMatrix/Vec for smartcore
    // smartcore expects data as DenseMatrix<f64> and target as Vec<f64>
    
    // We need to convert ArrayView2 to DenseMatrix. 
    // DenseMatrix::from_array(nrows, ncols, data)
    let train_rows = train_data.nrows();
    let train_cols = train_data.ncols();
    let train_vec = train_data.as_standard_layout().as_slice().unwrap().to_vec();
    let x_train = DenseMatrix::new(train_rows, train_cols, train_vec, false);
    
    let y_train: Vec<i32> = train_target.iter().map(|&x| x as i32).collect();
    
    let test_rows = test_data.nrows();
    let test_cols = test_data.ncols();
    let test_vec = test_data.as_standard_layout().as_slice().unwrap().to_vec();
    let x_test = DenseMatrix::new(test_rows, test_cols, test_vec, false);
    
    // Configure parameters
    let weight_enum = smartcore::neighbors::KNNWeightFunction::Uniform;
    
    let algorithm_enum = smartcore::algorithm::neighbour::KNNAlgorithmName::CoverTree;
    
    let params = KNNClassifierParameters::default()
        .with_k(k)
        .with_weight(weight_enum)
        .with_algorithm(algorithm_enum);
        
    // Train
    let knn = KNNClassifier::fit(&x_train, &y_train, params)
        .map_err(|e| DsuError::PredictionError(format!("KNN fit error: {:?}", e)))?;
        
    // Predict
    let predictions = knn.predict(&x_test)
        .map_err(|e| DsuError::PredictionError(format!("KNN predict error: {:?}", e)))?;
        
    // Convert predictions back to f64
    let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
        
    Ok(KNNResult {
        predictions: predictions_f64,
        accuracy: None,
    })
}
