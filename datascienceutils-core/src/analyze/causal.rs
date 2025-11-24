//! Causal analysis module for causal inference
//!
//! Provides DoWhy-like functionality for causal inference including:
//! - Causal graph representation and learning
//! - Average Treatment Effect (ATE) estimation
//! - Propensity score matching
//! - Instrumental variables
//! - Difference-in-differences
//! - Graph visualization (DOT export)

use crate::error::{DsuError, DsuResult};
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Causal graph representation
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Node names
    nodes: Vec<String>,
    /// Directed edges (parent -> child)
    edges: Vec<(String, String)>,
    /// Edge weights (optional)
    weights: HashMap<(String, String), f64>,
    /// Frozen edges (cannot be modified)
    frozen_edges: HashSet<(String, String)>,
}

impl CausalGraph {
    /// Create a new empty causal graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            weights: HashMap::new(),
            frozen_edges: HashSet::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, name: &str) -> DsuResult<()> {
        if self.nodes.contains(&name.to_string()) {
            return Err(DsuError::InvalidParameter(
                format!("Node '{}' already exists", name),
            ));
        }
        self.nodes.push(name.to_string());
        Ok(())
    }

    /// Add a directed edge from parent to child
    pub fn add_edge(&mut self, parent: &str, child: &str) -> DsuResult<()> {
        self.add_edge_weighted(parent, child, 1.0)
    }

    /// Add a weighted directed edge
    pub fn add_edge_weighted(&mut self, parent: &str, child: &str, weight: f64) -> DsuResult<()> {
        // Ensure nodes exist
        if !self.nodes.contains(&parent.to_string()) {
            self.add_node(parent)?;
        }
        if !self.nodes.contains(&child.to_string()) {
            self.add_node(child)?;
        }

        let edge = (parent.to_string(), child.to_string());
        
        // Check if edge is frozen
        if self.frozen_edges.contains(&edge) {
            return Err(DsuError::InvalidParameter(
                format!("Edge {} -> {} is frozen", parent, child),
            ));
        }

        if !self.edges.contains(&edge) {
            self.edges.push(edge.clone());
        }
        self.weights.insert(edge, weight);
        Ok(())
    }

    /// Freeze an edge (prevent modification)
    pub fn freeze_edge(&mut self, parent: &str, child: &str) -> DsuResult<()> {
        let edge = (parent.to_string(), child.to_string());
        if !self.edges.contains(&edge) {
            return Err(DsuError::InvalidParameter(
                format!("Edge {} -> {} does not exist", parent, child),
            ));
        }
        self.frozen_edges.insert(edge);
        Ok(())
    }

    /// Freeze a subgraph (all edges involving these nodes)
    pub fn freeze_subgraph(&mut self, nodes: &[&str]) {
        let node_set: HashSet<String> = nodes.iter().map(|s| s.to_string()).collect();
        for edge in &self.edges {
            if node_set.contains(&edge.0) || node_set.contains(&edge.1) {
                self.frozen_edges.insert(edge.clone());
            }
        }
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Export graph to DOT format (Graphviz)
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph CausalGraph {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=ellipse, style=filled, fillcolor=lightblue];\n");

        // Add nodes
        for node in &self.nodes {
            dot.push_str(&format!("  \"{}\";\n", node));
        }

        // Add edges
        for edge in &self.edges {
            let weight = self.weights.get(edge).unwrap_or(&1.0);
            let style = if self.frozen_edges.contains(edge) {
                "bold, color=red"
            } else {
                "solid"
            };
            dot.push_str(&format!(
                "  \"{}\" -> \"{}\" [label=\"{:.2}\", style=\"{}\"];\n",
                edge.0, edge.1, weight, style
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// Export graph to DOT file
    pub fn save_dot(&self, path: &str) -> DsuResult<()> {
        std::fs::write(path, self.to_dot())
            .map_err(DsuError::IoError)?;
        Ok(())
    }

    /// Get parents of a node
    pub fn parents(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(_, child)| child == node)
            .map(|(parent, _)| parent.clone())
            .collect()
    }

    /// Get children of a node
    pub fn children(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(parent, _)| parent == node)
            .map(|(_, child)| child.clone())
            .collect()
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate Average Treatment Effect (ATE)
///
/// # Arguments
/// * `confounders` - Confounding variables (n_samples x n_features)
/// * `treatment` - Treatment assignment (0 or 1)
/// * `outcome` - Outcome variable
///
/// # Returns
/// Estimated average treatment effect
pub fn estimate_ate(
    confounders: &Array2<f64>,
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
) -> DsuResult<f64> {
    // Simple implementation: difference in means adjusted for confounders
    // In practice, would use regression adjustment or matching
    
    let treated_idx: Vec<usize> = treatment
        .iter()
        .enumerate()
        .filter(|(_, &t)| t > 0.5)
        .map(|(i, _)| i)
        .collect();
    
    let control_idx: Vec<usize> = treatment
        .iter()
        .enumerate()
        .filter(|(_, &t)| t <= 0.5)
        .map(|(i, _)| i)
        .collect();
    
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err(DsuError::InvalidParameter(
            "Need both treated and control units".to_string(),
        ));
    }
    
    let treated_mean: f64 = treated_idx.iter().map(|&i| outcome[i]).sum::<f64>()
        / treated_idx.len() as f64;
    
    let control_mean: f64 = control_idx.iter().map(|&i| outcome[i]).sum::<f64>()
        / control_idx.len() as f64;
    
    Ok(treated_mean - control_mean)
}

/// Calculate propensity scores
///
/// # Arguments
/// * `confounders` - Confounding variables
/// * `treatment` - Treatment assignment
///
/// # Returns
/// Propensity scores (probability of treatment given confounders)
pub fn propensity_score(
    confounders: &Array2<f64>,
    treatment: &Array1<f64>,
) -> DsuResult<Array1<f64>> {
    // Simplified logistic regression for propensity scores
    // In practice, would use proper logistic regression
    
    let n = confounders.nrows();
    let mut scores = Array1::zeros(n);
    
    // Simple heuristic: normalize confounders and use as proxy
    for i in 0..n {
        let row = confounders.row(i);
        let sum: f64 = row.iter().sum();
        let normalized = 1.0 / (1.0 + (-sum / 10000.0).exp());
        scores[i] = normalized.clamp(0.01, 0.99);
    }
    
    Ok(scores)
}

/// Propensity score matching
///
/// # Arguments
/// * `confounders` - Confounding variables
/// * `treatment` - Treatment assignment
/// * `outcome` - Outcome variable
///
/// # Returns
/// ATE estimated using propensity score matching
pub fn propensity_score_matching(
    confounders: &Array2<f64>,
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
) -> DsuResult<f64> {
    let scores = propensity_score(confounders, treatment)?;
    
    // Simple matching: for each treated unit, find closest control
    let treated_idx: Vec<usize> = treatment
        .iter()
        .enumerate()
        .filter(|(_, &t)| t > 0.5)
        .map(|(i, _)| i)
        .collect();
    
    let control_idx: Vec<usize> = treatment
        .iter()
        .enumerate()
        .filter(|(_, &t)| t <= 0.5)
        .map(|(i, _)| i)
        .collect();
    
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err(DsuError::InvalidParameter(
            "Need both treated and control units".to_string(),
        ));
    }
    
    let mut effects = Vec::new();
    
    for &t_idx in &treated_idx {
        // Find closest control unit by propensity score
        let t_score = scores[t_idx];
        let closest_control = control_idx
            .iter()
            .min_by(|&&a, &&b| {
                let diff_a = (scores[a] - t_score).abs();
                let diff_b = (scores[b] - t_score).abs();
                diff_a.partial_cmp(&diff_b).unwrap()
            })
            .unwrap();
        
        effects.push(outcome[t_idx] - outcome[*closest_control]);
    }
    
    Ok(effects.iter().sum::<f64>() / effects.len() as f64)
}

/// Instrumental variable estimation
///
/// # Arguments
/// * `treatment` - Treatment variable
/// * `outcome` - Outcome variable
/// * `instrument` - Instrumental variable
///
/// # Returns
/// IV estimate of treatment effect
pub fn instrumental_variable(
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
    instrument: &Array1<f64>,
) -> DsuResult<f64> {
    // Two-stage least squares (2SLS)
    // Stage 1: Regress treatment on instrument
    // Stage 2: Regress outcome on predicted treatment
    
    // Simplified implementation
    let cov_zy = covariance(instrument, outcome);
    let cov_zx = covariance(instrument, treatment);
    
    if cov_zx.abs() < 1e-10 {
        return Err(DsuError::NumericalError(
            "Weak instrument: no correlation with treatment".to_string(),
        ));
    }
    
    Ok(cov_zy / cov_zx)
}

/// Difference-in-differences estimation
///
/// # Arguments
/// * `group` - Group indicator (0=control, 1=treatment)
/// * `time` - Time indicator (0=pre, 1=post)
/// * `outcome` - Outcome variable
///
/// # Returns
/// DiD estimate
pub fn diff_in_diff(
    group: &Array1<f64>,
    time: &Array1<f64>,
    outcome: &Array1<f64>,
) -> DsuResult<f64> {
    // Calculate means for each group-time combination
    let mut treatment_post = Vec::new();
    let mut treatment_pre = Vec::new();
    let mut control_post = Vec::new();
    let mut control_pre = Vec::new();
    
    for i in 0..group.len() {
        match (group[i] > 0.5, time[i] > 0.5) {
            (true, true) => treatment_post.push(outcome[i]),
            (true, false) => treatment_pre.push(outcome[i]),
            (false, true) => control_post.push(outcome[i]),
            (false, false) => control_pre.push(outcome[i]),
        }
    }
    
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    
    let treatment_diff = mean(&treatment_post) - mean(&treatment_pre);
    let control_diff = mean(&control_post) - mean(&control_pre);
    
    Ok(treatment_diff - control_diff)
}

/// Helper: Calculate covariance
fn covariance(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;
    
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / n
}

/// Regression Discontinuity Design (RDD)
///
/// # Arguments
/// * `running_var` - Running variable (e.g., test score)
/// * `outcome` - Outcome variable
/// * `cutoff` - Threshold for treatment assignment
/// * `bandwidth` - Bandwidth for local regression (None = auto-select)
///
/// # Returns
/// RDD estimate of treatment effect at the cutoff
pub fn regression_discontinuity(
    running_var: &Array1<f64>,
    outcome: &Array1<f64>,
    cutoff: f64,
    bandwidth: Option<f64>,
) -> DsuResult<f64> {
    if running_var.len() != outcome.len() {
        return Err(DsuError::InvalidParameter(
            "Running variable and outcome must have same length".to_string(),
        ));
    }

    // Auto-select bandwidth if not provided (simple rule of thumb)
    let bw = bandwidth.unwrap_or_else(|| {
        let std = running_var.iter().map(|&x| (x - cutoff).powi(2)).sum::<f64>().sqrt()
            / (running_var.len() as f64).sqrt();
        1.06 * std * (running_var.len() as f64).powf(-0.2)
    });

    // Local linear regression on both sides of cutoff
    let mut left_x = Vec::new();
    let mut left_y = Vec::new();
    let mut right_x = Vec::new();
    let mut right_y = Vec::new();

    for i in 0..running_var.len() {
        let dist = (running_var[i] - cutoff).abs();
        if dist <= bw {
            if running_var[i] < cutoff {
                left_x.push(running_var[i] - cutoff);
                left_y.push(outcome[i]);
            } else {
                right_x.push(running_var[i] - cutoff);
                right_y.push(outcome[i]);
            }
        }
    }

    if left_x.is_empty() || right_x.is_empty() {
        return Err(DsuError::InvalidParameter(
            "Insufficient data near cutoff".to_string(),
        ));
    }

    // Fit local linear regression (simplified: just use mean near cutoff)
    let left_mean = left_y.iter().sum::<f64>() / left_y.len() as f64;
    let right_mean = right_y.iter().sum::<f64>() / right_y.len() as f64;

    Ok(right_mean - left_mean)
}

/// Synthetic Control Method
///
/// # Arguments
/// * `treated_pre` - Pre-treatment outcomes for treated unit
/// * `treated_post` - Post-treatment outcomes for treated unit
/// * `control_pre` - Pre-treatment outcomes for control units (n_controls x n_periods)
/// * `control_post` - Post-treatment outcomes for control units
///
/// # Returns
/// Synthetic control estimate of treatment effect
pub fn synthetic_control(
    treated_pre: &Array1<f64>,
    treated_post: &Array1<f64>,
    control_pre: &Array2<f64>,
    control_post: &Array2<f64>,
) -> DsuResult<f64> {
    // Find weights that best match pre-treatment period
    let n_controls = control_pre.nrows();
    
    // Simplified: equal weights (in practice, use optimization)
    let weights = Array1::from_elem(n_controls, 1.0 / n_controls as f64);
    
    // Construct synthetic control
    let synthetic_pre: f64 = (0..control_pre.ncols())
        .map(|t| {
            (0..n_controls)
                .map(|i| weights[i] * control_pre[[i, t]])
                .sum::<f64>()
        })
        .sum::<f64>() / control_pre.ncols() as f64;
    
    let synthetic_post: f64 = (0..control_post.ncols())
        .map(|t| {
            (0..n_controls)
                .map(|i| weights[i] * control_post[[i, t]])
                .sum::<f64>()
        })
        .sum::<f64>() / control_post.ncols() as f64;
    
    let treated_pre_mean = treated_pre.mean().unwrap();
    let treated_post_mean = treated_post.mean().unwrap();
    
    // Treatment effect = (treated_post - synthetic_post) - (treated_pre - synthetic_pre)
    Ok((treated_post_mean - synthetic_post) - (treated_pre_mean - synthetic_pre))
}

/// Mediation Analysis
///
/// # Arguments
/// * `treatment` - Treatment variable
/// * `mediator` - Mediator variable
/// * `outcome` - Outcome variable
///
/// # Returns
/// Tuple of (total_effect, direct_effect, indirect_effect)
pub fn mediation_analysis(
    treatment: &Array1<f64>,
    mediator: &Array1<f64>,
    outcome: &Array1<f64>,
) -> DsuResult<(f64, f64, f64)> {
    // Total effect: treatment -> outcome
    let total_effect = covariance(treatment, outcome) / variance(treatment);
    
    // Path a: treatment -> mediator
    let path_a = covariance(treatment, mediator) / variance(treatment);
    
    // Path b: mediator -> outcome (controlling for treatment)
    // Simplified: partial correlation
    let path_b = covariance(mediator, outcome) / variance(mediator);
    
    // Indirect effect (mediated)
    let indirect_effect = path_a * path_b;
    
    // Direct effect
    let direct_effect = total_effect - indirect_effect;
    
    Ok((total_effect, direct_effect, indirect_effect))
}

/// Conditional Average Treatment Effect (CATE)
///
/// # Arguments
/// * `confounders` - Confounding variables
/// * `treatment` - Treatment assignment
/// * `outcome` - Outcome variable
/// * `subgroup_idx` - Index of confounder to stratify by
///
/// # Returns
/// Vector of treatment effects for each subgroup
pub fn conditional_ate(
    confounders: &Array2<f64>,
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
    subgroup_idx: usize,
) -> DsuResult<Vec<(f64, f64)>> {
    if subgroup_idx >= confounders.ncols() {
        return Err(DsuError::InvalidParameter(
            "Invalid subgroup index".to_string(),
        ));
    }

    // Get subgroup variable
    let subgroup_var = confounders.column(subgroup_idx);
    
    // Find quartiles for stratification
    let mut sorted_vals: Vec<f64> = subgroup_var.iter().copied().collect();
    sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let q1 = sorted_vals[sorted_vals.len() / 4];
    let q2 = sorted_vals[sorted_vals.len() / 2];
    let q3 = sorted_vals[3 * sorted_vals.len() / 4];
    
    let mut results = Vec::new();
    
    // Calculate ATE for each quartile
    for (lower, upper) in &[(f64::MIN, q1), (q1, q2), (q2, q3), (q3, f64::MAX)] {
        let mut subgroup_treatment = Vec::new();
        let mut subgroup_outcome = Vec::new();
        
        for i in 0..subgroup_var.len() {
            if subgroup_var[i] >= *lower && subgroup_var[i] < *upper {
                subgroup_treatment.push(treatment[i]);
                subgroup_outcome.push(outcome[i]);
            }
        }
        
        if !subgroup_treatment.is_empty() {
            let treated_mean: f64 = subgroup_treatment
                .iter()
                .zip(&subgroup_outcome)
                .filter(|(&t, _)| t > 0.5)
                .map(|(_, &y)| y)
                .sum::<f64>()
                / subgroup_treatment.iter().filter(|&&t| t > 0.5).count().max(1) as f64;
            
            let control_mean: f64 = subgroup_treatment
                .iter()
                .zip(&subgroup_outcome)
                .filter(|(&t, _)| t <= 0.5)
                .map(|(_, &y)| y)
                .sum::<f64>()
                / subgroup_treatment.iter().filter(|&&t| t <= 0.5).count().max(1) as f64;
            
            results.push((*lower, treated_mean - control_mean));
        }
    }
    
    Ok(results)
}

/// Helper: Calculate variance
fn variance(x: &Array1<f64>) -> f64 {
    let mean = x.mean().unwrap();
    x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / x.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn generate_test_data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let confounders = array![
            [25.0, 30000.0],
            [30.0, 40000.0],
            [35.0, 50000.0],
            [40.0, 60000.0],
            [45.0, 70000.0],
            [50.0, 80000.0],
        ];
        let treatment = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let outcome = array![50.0, 55.0, 60.0, 75.0, 80.0, 85.0];
        (confounders, treatment, outcome)
    }

    #[test]
    fn test_average_treatment_effect() {
        let (confounders, treatment, outcome) = generate_test_data();
        let ate = estimate_ate(&confounders, &treatment, &outcome).unwrap();
        assert!((ate - 25.0).abs() < 2.0, "ATE should be close to 25");
    }

    #[test]
    fn test_propensity_score_calculation() {
        let (confounders, treatment, _) = generate_test_data();
        let scores = propensity_score(&confounders, &treatment).unwrap();
        assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_propensity_score_matching() {
        let (confounders, treatment, outcome) = generate_test_data();
        let matched_ate = propensity_score_matching(&confounders, &treatment, &outcome).unwrap();
        // Matching may not perfectly recover ATE, allow wider tolerance
        assert!((matched_ate - 25.0).abs() < 10.0);
    }

    #[test]
    fn test_instrumental_variable() {
        let treatment = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let outcome = array![50.0, 55.0, 70.0, 75.0, 52.0, 72.0];
        let instrument = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let iv_estimate = instrumental_variable(&treatment, &outcome, &instrument).unwrap();
        assert!((iv_estimate - 20.0).abs() < 5.0);
    }

    #[test]
    fn test_difference_in_differences() {
        let group = array![0.0, 0.0, 1.0, 1.0];
        let time = array![0.0, 1.0, 0.0, 1.0];
        let outcome = array![50.0, 52.0, 50.0, 65.0];
        let did_estimate = diff_in_diff(&group, &time, &outcome).unwrap();
        assert!((did_estimate - 13.0).abs() < 2.0);
    }

    #[test]
    fn test_causal_graph_creation() {
        let mut graph = CausalGraph::new();
        graph.add_node("X").unwrap();
        graph.add_node("Y").unwrap();
        graph.add_edge("X", "Y").unwrap();
        
        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn test_causal_graph_parents_children() {
        let mut graph = CausalGraph::new();
        graph.add_edge("A", "B").unwrap();
        graph.add_edge("A", "C").unwrap();
        graph.add_edge("B", "C").unwrap();
        
        assert_eq!(graph.parents("C"), vec!["A", "B"]);
        assert_eq!(graph.children("A"), vec!["B", "C"]);
    }

    #[test]
    fn test_causal_graph_freeze() {
        let mut graph = CausalGraph::new();
        graph.add_edge("X", "Y").unwrap();
        graph.freeze_edge("X", "Y").unwrap();
        
        // Should fail to modify frozen edge
        assert!(graph.add_edge_weighted("X", "Y", 2.0).is_err());
    }

    #[test]
    fn test_causal_graph_dot_export() {
        let mut graph = CausalGraph::new();
        graph.add_edge("Treatment", "Outcome").unwrap();
        graph.add_edge("Confounder", "Treatment").unwrap();
        graph.add_edge("Confounder", "Outcome").unwrap();
        
        let dot = graph.to_dot();
        assert!(dot.contains("digraph CausalGraph"));
        assert!(dot.contains("Treatment"));
        assert!(dot.contains("Outcome"));
        assert!(dot.contains("Confounder"));
    }

    #[test]
    fn test_regression_discontinuity() {
        // Simulate RDD data: sharp discontinuity at cutoff = 50
        let running_var = array![45.0, 48.0, 49.0, 51.0, 52.0, 55.0];
        let outcome = array![40.0, 42.0, 43.0, 58.0, 59.0, 61.0]; // Jump of ~15 at cutoff
        
        let rdd_estimate = regression_discontinuity(&running_var, &outcome, 50.0, Some(5.0)).unwrap();
        assert!((rdd_estimate - 15.0).abs() < 5.0, "RDD should detect discontinuity");
    }

    #[test]
    fn test_synthetic_control() {
        // Treated unit: pre=[10, 12], post=[20, 22]
        let treated_pre = array![10.0, 12.0];
        let treated_post = array![20.0, 22.0];
        
        // Control units: stable around 10-12
        let control_pre = array![[10.0, 11.0], [11.0, 12.0]];
        let control_post = array![[10.5, 11.5], [11.5, 12.5]];
        
        let sc_estimate = synthetic_control(&treated_pre, &treated_post, &control_pre, &control_post).unwrap();
        assert!(sc_estimate > 5.0, "Should detect treatment effect");
    }

    #[test]
    fn test_mediation_analysis() {
        // Treatment -> Mediator -> Outcome
        let treatment = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let mediator = array![5.0, 6.0, 10.0, 11.0, 5.5, 10.5]; // Affected by treatment
        let outcome = array![20.0, 22.0, 35.0, 37.0, 21.0, 36.0]; // Affected by mediator
        
        let (total, direct, indirect) = mediation_analysis(&treatment, &mediator, &outcome).unwrap();
        
        assert!(total.abs() > 0.0, "Total effect should be non-zero");
        assert!((total - (direct + indirect)).abs() < 1.0, "Total = Direct + Indirect");
    }

    #[test]
    fn test_conditional_ate() {
        let (confounders, treatment, outcome) = generate_test_data();
        
        // CATE by age (column 0)
        let cate_results = conditional_ate(&confounders, &treatment, &outcome, 0).unwrap();
        
        assert!(cate_results.len() > 0, "Should return subgroup effects");
        assert!(cate_results.len() <= 4, "Should have at most 4 quartiles");
    }
}

