//! Optimizer configurations for neural network training

use serde::{Deserialize, Serialize};

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerConfig {
    /// Stochastic Gradient Descent with momentum
    SGD {
        /// Learning rate
        lr: f64,
        /// Momentum factor
        momentum: f64,
    },
    /// Adam optimizer
    Adam {
        /// Learning rate
        lr: f64,
        /// Beta parameters for moment estimation
        betas: (f64, f64),
    },
    /// AdamW optimizer with weight decay
    AdamW {
        /// Learning rate
        lr: f64,
        /// Weight decay factor
        weight_decay: f64,
    },
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::Adam {
            lr: 0.001,
            betas: (0.9, 0.999),
        }
    }
}

impl OptimizerConfig {
    /// Create SGD optimizer with default momentum
    pub fn sgd(lr: f64) -> Self {
        Self::SGD { lr, momentum: 0.9 }
    }

    /// Create Adam optimizer with default betas
    pub fn adam(lr: f64) -> Self {
        Self::Adam {
            lr,
            betas: (0.9, 0.999),
        }
    }

    /// Create AdamW optimizer
    pub fn adamw(lr: f64, weight_decay: f64) -> Self {
        Self::AdamW { lr, weight_decay }
    }

    /// Get the learning rate
    pub fn learning_rate(&self) -> f64 {
        match self {
            Self::SGD { lr, .. } => *lr,
            Self::Adam { lr, .. } => *lr,
            Self::AdamW { lr, .. } => *lr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_optimizer() {
        let opt = OptimizerConfig::default();
        assert_eq!(opt.learning_rate(), 0.001);
    }

    #[test]
    fn test_sgd_creation() {
        let opt = OptimizerConfig::sgd(0.01);
        assert_eq!(opt.learning_rate(), 0.01);
    }

    #[test]
    fn test_adam_creation() {
        let opt = OptimizerConfig::adam(0.001);
        match opt {
            OptimizerConfig::Adam { lr, betas } => {
                assert_eq!(lr, 0.001);
                assert_eq!(betas, (0.9, 0.999));
            }
            _ => panic!("Expected Adam optimizer"),
        }
    }
}
