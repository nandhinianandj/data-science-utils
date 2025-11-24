//! Loss functions for neural network training

use serde::{Deserialize, Serialize};

/// Loss function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LossFunction {
    /// Cross-entropy loss for classification
    CrossEntropy,
    /// Mean Squared Error for regression
    MSE,
    /// Mean Absolute Error for regression
    MAE,
    /// Binary cross-entropy for binary classification
    BinaryCrossEntropy,
}

impl LossFunction {
    /// Get a human-readable name for the loss function
    pub fn name(&self) -> &'static str {
        match self {
            Self::CrossEntropy => "Cross Entropy",
            Self::MSE => "Mean Squared Error",
            Self::MAE => "Mean Absolute Error",
            Self::BinaryCrossEntropy => "Binary Cross Entropy",
        }
    }

    /// Check if this is a classification loss
    pub fn is_classification(&self) -> bool {
        matches!(self, Self::CrossEntropy | Self::BinaryCrossEntropy)
    }

    /// Check if this is a regression loss
    pub fn is_regression(&self) -> bool {
        matches!(self, Self::MSE | Self::MAE)
    }
}

impl Default for LossFunction {
    fn default() -> Self {
        Self::CrossEntropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_function_names() {
        assert_eq!(LossFunction::CrossEntropy.name(), "Cross Entropy");
        assert_eq!(LossFunction::MSE.name(), "Mean Squared Error");
    }

    #[test]
    fn test_loss_function_types() {
        assert!(LossFunction::CrossEntropy.is_classification());
        assert!(!LossFunction::CrossEntropy.is_regression());
        
        assert!(LossFunction::MSE.is_regression());
        assert!(!LossFunction::MSE.is_classification());
    }

    #[test]
    fn test_default_loss() {
        assert_eq!(LossFunction::default(), LossFunction::CrossEntropy);
    }
}
