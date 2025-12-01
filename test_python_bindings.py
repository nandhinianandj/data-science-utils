"""
Python Bindings Test for datascienceutils

This script demonstrates the usage of the Python bindings for the datascienceutils library.
It tests causal analysis, outlier detection, and sampling functions.

Note: This requires the package to be built with maturin:
    maturin develop --release
"""

import numpy as np

# Import the datascienceutils module
try:
    import datascienceutils as dsu
    print(f"✓ Successfully imported datascienceutils v{dsu.__version__}")
except ImportError as e:
    print(f"✗ Failed to import datascienceutils: {e}")
    print("  Please build the package with: maturin develop --release")
    exit(1)

print("\n" + "="*60)
print("CAUSAL ANALYSIS TESTS")
print("="*60)

# Test 1: CausalGraph
print("\n1. Testing CausalGraph...")
try:
    graph = dsu.CausalGraph()
    graph.add_node("Treatment")
    graph.add_node("Outcome")
    graph.add_node("Confounder")
    graph.add_edge("Confounder", "Treatment")
    graph.add_edge("Confounder", "Outcome")
    graph.add_edge("Treatment", "Outcome")
    
    print(f"   Nodes: {graph.num_nodes()}")
    print(f"   Edges: {graph.num_edges()}")
    print(f"   Parents of Outcome: {graph.parents('Outcome')}")
    print(f"   Children of Confounder: {graph.children('Confounder')}")
    print("   ✓ CausalGraph works!")
except Exception as e:
    print(f"   ✗ CausalGraph failed: {e}")

# Test 2: Average Treatment Effect
print("\n2. Testing ATE estimation...")
try:
    np.random.seed(42)
    n = 100
    confounders = np.random.randn(n, 2)
    treatment = (confounders[:, 0] + np.random.randn(n) > 0).astype(float)
    outcome = treatment * 5.0 + confounders[:, 1] + np.random.randn(n)
    
    ate = dsu.estimate_ate(confounders, treatment, outcome)
    print(f"   Estimated ATE: {ate:.2f}")
    print(f"   True effect: 5.00")
    print("   ✓ ATE estimation works!")
except Exception as e:
    print(f"   ✗ ATE estimation failed: {e}")

# Test 3: Difference-in-Differences
print("\n3. Testing DiD...")
try:
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    time = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    outcome = np.array([50, 52, 51, 53, 50, 51, 50, 68])
    
    did = dsu.diff_in_diff(group, time, outcome)
    print(f"   DiD estimate: {did:.2f}")
    print("   ✓ DiD works!")
except Exception as e:
    print(f"   ✗ DiD failed: {e}")

print("\n" + "="*60)
print("OUTLIER DETECTION TESTS")
print("="*60)

# Test 4: Outlier Detection
print("\n4. Testing outlier detection...")
try:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    
    # Sigma method
    outliers, lower, upper = dsu.detect_outliers_sigma(data, 2.0)
    print(f"   Sigma method: {len(outliers)} outliers at indices {outliers}")
    
    # IQR method
    outliers, lower, upper = dsu.detect_outliers_iqr(data, 1.5)
    print(f"   IQR method: {len(outliers)} outliers at indices {outliers}")
    
    # Z-score method
    outliers = dsu.detect_outliers_zscore(data, 2.0)
    print(f"   Z-score method: {len(outliers)} outliers at indices {outliers}")
    
    print("   ✓ Outlier detection works!")
except Exception as e:
    print(f"   ✗ Outlier detection failed: {e}")

# Test 5: Outlier Removal
print("\n5. Testing outlier removal...")
try:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    outlier_indices = [5]
    
    cleaned = dsu.remove_outliers(data, outlier_indices)
    print(f"   Original data: {data}")
    print(f"   Cleaned data: {cleaned}")
    print("   ✓ Outlier removal works!")
except Exception as e:
    print(f"   ✗ Outlier removal failed: {e}")

print("\n" + "="*60)
print("SAMPLING TESTS")
print("="*60)

# Test 6: Distribution Sampling
print("\n6. Testing distribution sampling...")
try:
    # Normal distribution
    samples = dsu.sample_normal(0.0, 1.0, 100)
    print(f"   Normal samples: mean={samples.mean():.2f}, std={samples.std():.2f}")
    
    # Uniform distribution
    samples = dsu.sample_uniform(0.0, 10.0, 100)
    print(f"   Uniform samples: min={samples.min():.2f}, max={samples.max():.2f}")
    
    print("   ✓ Distribution sampling works!")
except Exception as e:
    print(f"   ✗ Distribution sampling failed: {e}")

# Test 7: Bootstrap Sampling
print("\n7. Testing bootstrap sampling...")
try:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bootstrap_samples = dsu.bootstrap_sample(data, 10)
    
    print(f"   Generated {len(bootstrap_samples)} bootstrap samples")
    print(f"   Each sample size: {len(bootstrap_samples[0])}")
    print("   ✓ Bootstrap sampling works!")
except Exception as e:
    print(f"   ✗ Bootstrap sampling failed: {e}")

print("\n" + "="*60)
print("STATISTICS TESTS")
print("="*60)

# Test 8: Correlation
print("\n8. Testing correlation...")
try:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    r = dsu.pearson_correlation(x, y)
    print(f"   Pearson correlation: {r:.4f}")
    
    rho = dsu.spearman_correlation(x, y)
    print(f"   Spearman correlation: {rho:.4f}")
    
    tau = dsu.kendall_correlation(x, y)
    print(f"   Kendall tau: {tau:.4f}")
    
    print("   ✓ Correlation works!")
except Exception as e:
    print(f"   ✗ Correlation failed: {e}")

print("\n" + "="*60)
print("ADVANCED CAUSAL ESTIMATORS TESTS")
print("="*60)

# Test 9: Regression Discontinuity Design
print("\n9. Testing RDD...")
try:
    np.random.seed(42)
    n = 200
    cutoff = 70.0
    running_var = np.random.uniform(40, 100, n)
    treatment_rdd = (running_var >= cutoff).astype(float)
    true_effect_rdd = 15.0
    outcome_rdd = 30 + 0.5 * running_var + true_effect_rdd * treatment_rdd + np.random.randn(n) * 5
    
    rdd_estimate = dsu.regression_discontinuity(running_var, outcome_rdd, cutoff)
    print(f"   RDD estimate: {rdd_estimate:.2f}")
    print(f"   True effect: {true_effect_rdd:.2f}")
    print("   ✓ RDD works!")
except Exception as e:
    print(f"   ✗ RDD failed: {e}")

# Test 10: Synthetic Control
print("\n10. Testing Synthetic Control...")
try:
    treated_pre = np.array([100, 102, 105, 103, 107, 110, 108, 112, 115, 113])
    treated_post = np.array([117, 110, 112, 115, 113])
    control_pre = np.array([
        [98, 100, 103, 101, 105, 108, 106, 110, 113, 111],
        [102, 104, 107, 105, 109, 112, 110, 114, 117, 115],
        [95, 97, 100, 98, 102, 105, 103, 107, 110, 108],
        [105, 107, 110, 108, 112, 115, 113, 117, 120, 118],
        [100, 102, 105, 103, 107, 110, 108, 112, 115, 113],
    ])
    control_post = np.array([
        [115, 117, 119, 121, 123],
        [119, 121, 123, 125, 127],
        [112, 114, 116, 118, 120],
        [122, 124, 126, 128, 130],
        [117, 119, 121, 123, 125],
    ])
    
    sc_estimate = dsu.synthetic_control(treated_pre, treated_post, control_pre, control_post)
    print(f"   Synthetic Control estimate: {sc_estimate:.2f}")
    print("   ✓ Synthetic Control works!")
except Exception as e:
    print(f"   ✗ Synthetic Control failed: {e}")

# Test 11: Mediation Analysis
print("\n11. Testing Mediation Analysis...")
try:
    np.random.seed(42)
    n = 400
    treatment_med = np.random.binomial(1, 0.5, n).astype(float)
    mediator = 50 + 5 * treatment_med + np.random.randn(n) * 3
    outcome_med = 10 + 3 * treatment_med + 0.4 * mediator + np.random.randn(n) * 2
    
    total_eff, direct_eff, indirect_eff = dsu.mediation_analysis(treatment_med, mediator, outcome_med)
    print(f"   Total effect: {total_eff:.2f}")
    print(f"   Direct effect: {direct_eff:.2f}")
    print(f"   Indirect effect: {indirect_eff:.2f}")
    print(f"   Proportion mediated: {100 * indirect_eff / total_eff:.1f}%")
    print("   ✓ Mediation Analysis works!")
except Exception as e:
    print(f"   ✗ Mediation Analysis failed: {e}")

# Test 12: Conditional ATE (CATE)
print("\n12. Testing CATE...")
try:
    np.random.seed(42)
    n = 300
    covariates = np.random.randn(n, 2)
    covariates[:, 0] = covariates[:, 0] * 15 + 50
    covariates[:, 1] = covariates[:, 1] * 10 + 70
    treatment_cate = np.random.binomial(1, 0.5, n).astype(float)
    true_cate = 10 - 0.2 * (covariates[:, 0] - 50) + 0.1 * (covariates[:, 1] - 70)
    outcome_cate = 50 + 0.3 * covariates[:, 0] + 0.5 * covariates[:, 1] + true_cate * treatment_cate + np.random.randn(n) * 5
    
    cate_estimates = dsu.conditional_ate(treatment_cate, outcome_cate, covariates)
    print(f"   CATE mean: {cate_estimates.mean():.2f}")
    print(f"   CATE std: {cate_estimates.std():.2f}")
    print(f"   CATE range: [{cate_estimates.min():.2f}, {cate_estimates.max():.2f}]")
    print("   ✓ CATE works!")
except Exception as e:
    print(f"   ✗ CATE failed: {e}")

print("\n" + "="*60)
print("CAUSAL GRAPH TESTS")
print("="*60)

# Test 13: Enhanced Graph Operations
print("\n13. Testing enhanced graph operations...")
try:
    graph2 = dsu.CausalGraph()
    
    # Add nodes
    graph2.add_node("Age")
    graph2.add_node("Education")
    graph2.add_node("Income")
    graph2.add_node("Health")
    
    # Add weighted edges
    graph2.add_edge_weighted("Age", "Health", 0.6)
    graph2.add_edge_weighted("Education", "Income", 0.8)
    graph2.add_edge_weighted("Age", "Income", 0.3)
    
    # Freeze an edge
    graph2.freeze_edge("Education", "Income")
    
    # Freeze a subgraph
    graph2.freeze_subgraph(["Age", "Health"])
    
    print(f"   Nodes: {graph2.num_nodes()}")
    print(f"   Edges: {graph2.num_edges()}")
    print(f"   Parents of Income: {graph2.parents('Income')}")
    print(f"   Children of Age: {graph2.children('Age')}")
    
    # Export to DOT
    dot_str = graph2.to_dot()
    print(f"   DOT export length: {len(dot_str)} characters")
    
    # Save to file
    graph2.save_dot("test_graph.dot")
    print("   ✓ Enhanced graph operations work!")
except Exception as e:
    print(f"   ✗ Enhanced graph operations failed: {e}")

print("\n" + "="*60)
print("ALL TESTS COMPLETED!")
print("="*60)

