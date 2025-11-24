//! Analysis module for exploratory data analysis

#[cfg(feature = "causal-analysis")]
pub mod causal;

#[cfg(feature = "bayesian-inference")]
pub mod bayesian;

// Re-export commonly used functions
#[cfg(feature = "causal-analysis")]
pub use causal::{
    estimate_ate, propensity_score, propensity_score_matching,
    instrumental_variable, diff_in_diff
};
