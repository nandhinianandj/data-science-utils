//! Type checking utilities for data analysis

use polars::prelude::*;

/// Check if a Polars Series contains numeric data
///
/// # Arguments
/// * `series` - Polars Series to check
///
/// # Returns
/// true if the series contains numeric data (integers or floats)
pub fn is_numeric(series: &Series) -> bool {
    matches!(
        series.dtype(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Check if a Polars Series contains float data
///
/// # Arguments
/// * `series` - Polars Series to check
///
/// # Returns
/// true if the series contains float data
pub fn is_float(series: &Series) -> bool {
    matches!(series.dtype(), DataType::Float32 | DataType::Float64)
}

/// Check if a Polars Series contains integer data
///
/// # Arguments
/// * `series` - Polars Series to check
///
/// # Returns
/// true if the series contains integer data
pub fn is_integer(series: &Series) -> bool {
    matches!(
        series.dtype(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

/// Check if a value is of a specific numeric type
///
/// # Arguments
/// * `series` - Polars Series to check
/// * `base_type` - Type to check against ("float", "integer", "numeric")
///
/// # Returns
/// true if the series matches the base type
pub fn is_type(series: &Series, base_type: &str) -> bool {
    match base_type {
        "float" => is_float(series),
        "integer" => is_integer(series),
        "numeric" => is_numeric(series),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_numeric() {
        let s_int = Series::new("test", &[1i32, 2, 3]);
        let s_float = Series::new("test", &[1.0f64, 2.0, 3.0]);
        let s_str = Series::new("test", &["a", "b", "c"]);

        assert!(is_numeric(&s_int));
        assert!(is_numeric(&s_float));
        assert!(!is_numeric(&s_str));
    }

    #[test]
    fn test_is_float() {
        let s_int = Series::new("test", &[1i32, 2, 3]);
        let s_float = Series::new("test", &[1.0f64, 2.0, 3.0]);

        assert!(!is_float(&s_int));
        assert!(is_float(&s_float));
    }

    #[test]
    fn test_is_integer() {
        let s_int = Series::new("test", &[1i32, 2, 3]);
        let s_float = Series::new("test", &[1.0f64, 2.0, 3.0]);

        assert!(is_integer(&s_int));
        assert!(!is_integer(&s_float));
    }

    #[test]
    fn test_is_type() {
        let s_int = Series::new("test", &[1i32, 2, 3]);
        let s_float = Series::new("test", &[1.0f64, 2.0, 3.0]);

        assert!(is_type(&s_int, "integer"));
        assert!(is_type(&s_int, "numeric"));
        assert!(!is_type(&s_int, "float"));

        assert!(is_type(&s_float, "float"));
        assert!(is_type(&s_float, "numeric"));
        assert!(!is_type(&s_float, "integer"));
    }
}
