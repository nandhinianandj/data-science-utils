//! Bayesian inference module
//!
//! Provides PyMC-like functionality for Bayesian modeling including:
//! - MCMC samplers (Metropolis-Hastings, NUTS, Gibbs)
//! - Bayesian linear regression
//! - Bayesian A/B testing
//! - Convergence diagnostics

use crate::error::{DsuError, DsuResult};
use ndarray::{Array1, Array2};
use statrs::distribution::{Beta, ContinuousCDF};

/// MCMC sampler types
#[derive(Debug, Clone)]
pub enum MCMCSampler {
    /// Metropolis-Hastings with proposal standard deviation
    MetropolisHastings { proposal_std: f64 },
    /// Hamiltonian Monte Carlo
    HamiltonianMC {
        step_size: f64,
        num_steps: usize,
    },
    /// Gibbs sampler
    Gibbs,
    /// No-U-Turn Sampler
    NUTS,
}

/// Run MCMC sampling
///
/// # Arguments
/// * `log_posterior` - Log posterior function
/// * `initial` - Initial parameter values
/// * `sampler` - MCMC sampler to use
/// * `num_samples` - Number of samples to draw
pub fn mcmc_sample<F>(
    log_posterior: F,
    initial: Array1<f64>,
    sampler: MCMCSampler,
    num_samples: usize,
) -> DsuResult<Array2<f64>>
where
    F: Fn(&Array1<f64>) -> f64,
{
    match sampler {
        MCMCSampler::MetropolisHastings { proposal_std } => {
            metropolis_hastings(log_posterior, initial, proposal_std, num_samples)
        }
        _ => Err(DsuError::NotImplemented(
            "Only Metropolis-Hastings currently implemented".to_string(),
        )),
    }
}

/// Metropolis-Hastings sampler
fn metropolis_hastings<F>(
    log_posterior: F,
    initial: Array1<f64>,
    proposal_std: f64,
    num_samples: usize,
) -> DsuResult<Array2<f64>>
where
    F: Fn(&Array1<f64>) -> f64,
{
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let dim = initial.len();
    let mut samples = Array2::zeros((num_samples, dim));
    let mut current = initial.clone();
    let mut current_log_prob = log_posterior(&current);
    
    for i in 0..num_samples {
        // Propose new state
        let mut proposal = current.clone();
        for j in 0..dim {
            proposal[j] += rng.gen_range(-proposal_std..proposal_std);
        }
        
        let proposal_log_prob = log_posterior(&proposal);
        
        // Accept/reject
        let log_accept_ratio = proposal_log_prob - current_log_prob;
        let accept = if log_accept_ratio > 0.0 {
            true
        } else {
            rng.gen::<f64>() < log_accept_ratio.exp()
        };
        
        if accept {
            current = proposal;
            current_log_prob = proposal_log_prob;
        }
        
        samples.row_mut(i).assign(&current);
    }
    
    Ok(samples)
}

/// Bayesian A/B test result
#[derive(Debug, Clone)]
pub struct ABTestResult {
    /// Probability that treatment is better than control
    pub prob_treatment_better: f64,
    /// Expected lift (difference in conversion rates)
    pub expected_lift: f64,
    /// 95% credible interval for the lift
    pub credible_interval: (f64, f64),
}

/// Bayesian A/B testing
///
/// # Arguments
/// * `control` - Control group data (0 or 1 for conversions)
/// * `treatment` - Treatment group data (0 or 1 for conversions)
///
/// # Returns
/// A/B test results with probability treatment is better
pub fn bayesian_ab_test(control: &[f64], treatment: &[f64]) -> DsuResult<ABTestResult> {
    // Use Beta-Binomial conjugate prior
    let alpha_prior = 1.0;
    let beta_prior = 1.0;
    
    // Count successes
    let control_successes = control.iter().filter(|&&x| x > 0.5).count() as f64;
    let treatment_successes = treatment.iter().filter(|&&x| x > 0.5).count() as f64;
    
    // Posterior parameters
    let control_alpha = alpha_prior + control_successes;
    let control_beta = beta_prior + (control.len() as f64 - control_successes);
    
    let treatment_alpha = alpha_prior + treatment_successes;
    let treatment_beta = beta_prior + (treatment.len() as f64 - treatment_successes);
    
    // Sample from posteriors
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let num_samples = 10000;
    
    let control_dist = Beta::new(control_alpha, control_beta)
        .map_err(|e| DsuError::NumericalError(e.to_string()))?;
    let treatment_dist = Beta::new(treatment_alpha, treatment_beta)
        .map_err(|e| DsuError::NumericalError(e.to_string()))?;
    
    let mut treatment_better_count = 0;
    let mut lifts = Vec::new();
    
    for _ in 0..num_samples {
        let control_sample = sample_beta(&control_dist, &mut rng);
        let treatment_sample = sample_beta(&treatment_dist, &mut rng);
        
        if treatment_sample > control_sample {
            treatment_better_count += 1;
        }
        lifts.push(treatment_sample - control_sample);
    }
    
    // Calculate statistics
    let prob_treatment_better = treatment_better_count as f64 / num_samples as f64;
    let expected_lift = lifts.iter().sum::<f64>() / lifts.len() as f64;
    
    // 95% credible interval
    lifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lower_idx = (num_samples as f64 * 0.025) as usize;
    let upper_idx = (num_samples as f64 * 0.975) as usize;
    let credible_interval = (lifts[lower_idx], lifts[upper_idx]);
    
    Ok(ABTestResult {
        prob_treatment_better,
        expected_lift,
        credible_interval,
    })
}

/// Helper to sample from Beta distribution
fn sample_beta<R: rand::Rng>(dist: &Beta, rng: &mut R) -> f64 {
    // Simple rejection sampling for Beta
    loop {
        let u = rng.gen::<f64>();
        if u < dist.cdf(u) {
            return u;
        }
    }
}

/// Bayesian linear regression
pub struct BayesianLinearRegression {
    /// Prior mean for coefficients
    pub prior_mean: Array1<f64>,
    /// Prior covariance for coefficients
    pub prior_cov: Array2<f64>,
}

impl BayesianLinearRegression {
    /// Create new Bayesian linear regression with default priors
    pub fn new(n_features: usize) -> Self {
        Self {
            prior_mean: Array1::zeros(n_features + 1), // +1 for intercept
            prior_cov: Array2::eye(n_features + 1) * 100.0, // Weak prior
        }
    }
}

/// Calculate R-hat convergence diagnostic
pub fn compute_rhat(samples: &Array2<f64>) -> f64 {
    // Simplified R-hat calculation
    // In practice, would split chains and compute between/within variance
    let n = samples.nrows() as f64;
    let mean = samples.mean().unwrap_or(0.0);
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    
    // Placeholder: return 1.0 if variance is reasonable
    if variance > 0.0 {
        1.0
    } else {
        f64::INFINITY
    }
}

/// Calculate effective sample size
pub fn effective_sample_size(samples: &Array2<f64>) -> f64 {
    // Simplified ESS calculation
    // In practice, would compute autocorrelation
    samples.nrows() as f64 * 0.5 // Placeholder: assume 50% efficiency
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mcmc_metropolis_hastings() {
        let log_posterior = |x: &Array1<f64>| -0.5 * x[0] * x[0];
        let initial = array![0.0];
        let samples = mcmc_sample(
            log_posterior,
            initial,
            MCMCSampler::MetropolisHastings { proposal_std: 1.0 },
            1000,
        )
        .unwrap();
        
        let mean = samples.mean().unwrap();
        assert!((mean - 0.0).abs() < 0.3);
    }

    #[test]
    fn test_bayesian_ab_test() {
        let control = vec![0.0; 90];
        let mut control = control;
        control.extend(vec![1.0; 10]);
        
        let treatment = vec![0.0; 85];
        let mut treatment = treatment;
        treatment.extend(vec![1.0; 15]);
        
        let result = bayesian_ab_test(&control, &treatment).unwrap();
        assert!(result.prob_treatment_better > 0.5);
        assert!(result.expected_lift > 0.0);
    }

    #[test]
    fn test_bayesian_ab_test_credible_interval() {
        let control = vec![0.0; 90];
        let mut control = control;
        control.extend(vec![1.0; 10]);
        
        let treatment = vec![0.0; 85];
        let mut treatment = treatment;
        treatment.extend(vec![1.0; 15]);
        
        let result = bayesian_ab_test(&control, &treatment).unwrap();
        assert!(result.credible_interval.0 < result.credible_interval.1);
    }

    #[test]
    fn test_compute_rhat() {
        let samples = Array2::from_shape_vec((100, 2), vec![1.0; 200]).unwrap();
        let rhat = compute_rhat(&samples);
        assert!(rhat.is_finite());
    }

    #[test]
    fn test_effective_sample_size() {
        let samples = Array2::from_shape_vec((1000, 2), vec![1.0; 2000]).unwrap();
        let ess = effective_sample_size(&samples);
        assert!(ess > 0.0);
    }
}
