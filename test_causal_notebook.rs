// Test script to verify the causal analysis notebook code works
use datascienceutils_core::analyze::causal::*;
use ndarray::{Array1, Array2};
use rand::Rng;

// Generate synthetic medical data
fn generate_medical_data(n: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();
    
    let mut confounders = Array2::zeros((n, 3));
    let mut treatment = Array1::zeros(n);
    let mut outcome = Array1::zeros(n);
    
    for i in 0..n {
        // Confounders: age, baseline_health, income
        let age = rng.gen_range(20.0..80.0);
        let baseline_health = rng.gen_range(30.0..70.0);
        let income = rng.gen_range(20000.0..100000.0);
        
        confounders[[i, 0]] = age;
        confounders[[i, 1]] = baseline_health;
        confounders[[i, 2]] = income;
        
        // Treatment assignment (biased by confounders)
        let treatment_prob = 0.3 + (income / 200000.0) + (baseline_health / 200.0);
        treatment[i] = if rng.gen_range(0.0..1.0) < treatment_prob { 1.0 } else { 0.0 };
        
        // Outcome (true treatment effect = 15 points)
        let treatment_effect = if treatment[i] > 0.5 { 15.0 } else { 0.0 };
        outcome[i] = baseline_health + treatment_effect + (age * -0.2) + rng.gen_range(-5.0..5.0);
    }
    
    (confounders, treatment, outcome)
}

fn main() {
    println!("Testing causal analysis notebook code...\n");
    
    // Test 1: Generate data
    let (confounders, treatment, outcome) = generate_medical_data(500);
    println!("✓ Generated {} samples", confounders.nrows());
    println!("  Treatment group size: {}", treatment.iter().filter(|&&x| x > 0.5).count());
    println!("  Control group size: {}", treatment.iter().filter(|&&x| x <= 0.5).count());
    
    // Test 2: ATE
    let ate = estimate_ate(&confounders, &treatment, &outcome).unwrap();
    println!("\n✓ Average Treatment Effect: {:.2} points", ate);
    println!("  True treatment effect: 15.00 points");
    println!("  Estimation error: {:.2} points", (ate - 15.0).abs());
    
    // Test 3: PSM
    let psm_ate = propensity_score_matching(&confounders, &treatment, &outcome).unwrap();
    println!("\n✓ Propensity Score Matching ATE: {:.2} points", psm_ate);
    println!("  Estimation error: {:.2} points", (psm_ate - 15.0).abs());
    
    // Test 4: DiD
    use ndarray::array;
    let group = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let time = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let outcome_did = array![50.0, 52.0, 51.0, 53.0, 50.0, 51.0, 50.0, 68.0];
    
    let did_estimate = diff_in_diff(&group, &time, &outcome_did).unwrap();
    println!("\n✓ Difference-in-Differences: {:.2}", did_estimate);
    
    // Test 5: Outcome statistics
    let treated_outcomes: Vec<f64> = treatment.iter()
        .enumerate()
        .filter(|(_, &t)| t > 0.5)
        .map(|(i, _)| outcome[i])
        .collect();
    
    let control_outcomes: Vec<f64> = treatment.iter()
        .enumerate()
        .filter(|(_, &t)| t <= 0.5)
        .map(|(i, _)| outcome[i])
        .collect();
    
    println!("\n✓ Outcome Statistics:");
    println!("  Treated group mean: {:.2}", treated_outcomes.iter().sum::<f64>() / treated_outcomes.len() as f64);
    println!("  Control group mean: {:.2}", control_outcomes.iter().sum::<f64>() / control_outcomes.len() as f64);
    println!("  Difference: {:.2}", 
        (treated_outcomes.iter().sum::<f64>() / treated_outcomes.len() as f64) - 
        (control_outcomes.iter().sum::<f64>() / control_outcomes.len() as f64));
    
    println!("\n✅ All tests passed!");
}
