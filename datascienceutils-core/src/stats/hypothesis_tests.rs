//! Statistical hypothesis tests

use crate::error::{DsuError, DsuResult};
use ndarray::ArrayView1;

/// Perform one-way ANOVA
///
/// # Arguments
/// * `groups` - Vector of data groups
///
/// # Returns
/// Tuple of (F-statistic, p-value approximation)
pub fn anova_oneway(groups: &[ArrayView1<f64>]) -> DsuResult<(f64, f64)> {
    if groups.len() < 2 {
        return Err(DsuError::InvalidParameter(
            "ANOVA requires at least 2 groups".to_string(),
        ));
    }

    // Calculate grand mean and group means
    let mut all_values = Vec::new();
    let mut group_means = Vec::new();
    let mut group_sizes = Vec::new();

    for group in groups {
        if group.is_empty() {
            return Err(DsuError::EmptyData);
        }
        
        let group_vec = group.to_vec();
        let group_mean = group_vec.iter().sum::<f64>() / group_vec.len() as f64;
        
        all_values.extend(&group_vec);
        group_means.push(group_mean);
        group_sizes.push(group_vec.len());
    }

    let grand_mean = all_values.iter().sum::<f64>() / all_values.len() as f64;

    // Calculate between-group sum of squares (SSB)
    let mut ssb = 0.0;
    for (i, &mean) in group_means.iter().enumerate() {
        ssb += group_sizes[i] as f64 * (mean - grand_mean).powi(2);
    }

    // Calculate within-group sum of squares (SSW)
    let mut ssw = 0.0;
    for (i, group) in groups.iter().enumerate() {
        for &value in group.iter() {
            ssw += (value - group_means[i]).powi(2);
        }
    }

    // Degrees of freedom
    let df_between = groups.len() - 1;
    let df_within = all_values.len() - groups.len();

    if df_within == 0 {
        return Err(DsuError::StatisticalError(
            "Insufficient degrees of freedom".to_string(),
        ));
    }

    // Mean squares
    let msb = ssb / df_between as f64;
    let msw = ssw / df_within as f64;

    if msw == 0.0 {
        return Err(DsuError::StatisticalError(
            "Within-group variance is zero".to_string(),
        ));
    }

    // F-statistic
    let f_stat = msb / msw;

    // Simplified p-value approximation
    // In a full implementation, this would use the F-distribution
    let p_value = if f_stat > 4.0 { 0.01 } else if f_stat > 3.0 { 0.05 } else { 0.10 };

    Ok((f_stat, p_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_anova_oneway() {
        let group1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = array![2.0, 3.0, 4.0, 5.0, 6.0];
        let group3 = array![3.0, 4.0, 5.0, 6.0, 7.0];
        
        let groups = vec![group1.view(), group2.view(), group3.view()];
        
        let result = anova_oneway(&groups);
        assert!(result.is_ok());
        
        let (f_stat, p_value) = result.unwrap();
        assert!(f_stat >= 0.0);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    fn test_anova_insufficient_groups() {
        let group1 = array![1.0, 2.0, 3.0];
        let groups = vec![group1.view()];
        
        let result = anova_oneway(&groups);
        assert!(result.is_err());
    }

    #[test]
    fn test_anova_empty_group() {
        let group1 = array![1.0, 2.0, 3.0];
        let group2 = array![];
        let groups = vec![group1.view(), group2.view()];
        
        let result = anova_oneway(&groups);
        assert!(result.is_err());
    }
}
