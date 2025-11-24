//! DataFrame wrapper and utilities
//!
//! Provides a wrapper around Polars DataFrames with additional utilities
//! for data science workflows and Python interoperability.

use crate::error::{DsuError, DsuResult};
use polars::prelude::*;

/// Wrapper around Polars DataFrame with additional utilities
pub struct DsuDataFrame {
    df: DataFrame,
}

impl DsuDataFrame {
    /// Create a new DsuDataFrame from a Polars DataFrame
    pub fn new(df: DataFrame) -> Self {
        Self { df }
    }

    /// Get a reference to the underlying Polars DataFrame
    pub fn inner(&self) -> &DataFrame {
        &self.df
    }

    /// Get a mutable reference to the underlying Polars DataFrame
    pub fn inner_mut(&mut self) -> &mut DataFrame {
        &mut self.df
    }

    /// Consume self and return the underlying DataFrame
    pub fn into_inner(self) -> DataFrame {
        self.df
    }

    /// Read a CSV file into a DsuDataFrame
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    ///
    /// # Returns
    /// A new DsuDataFrame
    pub fn read_csv<P: Into<std::path::PathBuf>>(path: P) -> DsuResult<Self> {
        let df = CsvReader::from_path(path.into())
            .map_err(|e| DsuError::DataError(e.to_string()))?
            .finish()
            .map_err(|e| DsuError::DataError(e.to_string()))?;
        
        Ok(Self::new(df))
    }

    /// Write the DataFrame to a CSV file
    ///
    /// # Arguments
    /// * `path` - Path to write the CSV file
    pub fn write_csv<P: Into<std::path::PathBuf>>(&mut self, path: P) -> DsuResult<()> {
        let file = std::fs::File::create(path.into())
            .map_err(|e| DsuError::IoError(e))?;
        
        CsvWriter::new(file)
            .finish(&mut self.df)
            .map_err(|e| DsuError::DataError(e.to_string()))?;
        
        Ok(())
    }

    /// Read a Parquet file into a DsuDataFrame
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    pub fn read_parquet<P: Into<std::path::PathBuf>>(path: P) -> DsuResult<Self> {
        let file = std::fs::File::open(path.into())
            .map_err(|e| DsuError::IoError(e))?;
        
        let df = ParquetReader::new(file)
            .finish()
            .map_err(|e| DsuError::DataError(e.to_string()))?;
        
        Ok(Self::new(df))
    }

    /// Write the DataFrame to a Parquet file
    ///
    /// # Arguments
    /// * `path` - Path to write the Parquet file
    pub fn write_parquet<P: Into<std::path::PathBuf>>(&mut self, path: P) -> DsuResult<()> {
        let file = std::fs::File::create(path.into())
            .map_err(|e| DsuError::IoError(e))?;
        
        ParquetWriter::new(file)
            .finish(&mut self.df)
            .map_err(|e| DsuError::DataError(e.to_string()))?;
        
        Ok(())
    }

    /// Get the shape of the DataFrame (rows, columns)
    pub fn shape(&self) -> (usize, usize) {
        self.df.shape()
    }

    /// Get the number of rows
    pub fn height(&self) -> usize {
        self.df.height()
    }

    /// Get the number of columns
    pub fn width(&self) -> usize {
        self.df.width()
    }

    /// Get column names
    pub fn columns(&self) -> Vec<&str> {
        self.df.get_column_names()
    }

    /// Select specific columns
    ///
    /// # Arguments
    /// * `columns` - List of column names to select
    pub fn select(&self, columns: &[&str]) -> DsuResult<Self> {
        let df = self.df
            .select(columns)
            .map_err(|e| DsuError::DataError(e.to_string()))?;
        
        Ok(Self::new(df))
    }

    /// Filter rows based on a boolean mask
    ///
    /// # Arguments
    /// * `mask` - Boolean series for filtering
    pub fn filter(&self, mask: &Series) -> DsuResult<Self> {
        let mask_bool = mask.bool()
            .map_err(|e| DsuError::DataError(e.to_string()))?;
        
        let df = self.df
            .filter(mask_bool)
            .map_err(|e| DsuError::DataError(e.to_string()))?;
        
        Ok(Self::new(df))
    }

    /// Get a column by name
    ///
    /// # Arguments
    /// * `name` - Column name
    pub fn column(&self, name: &str) -> DsuResult<&Series> {
        self.df
            .column(name)
            .map_err(|e| DsuError::DataError(e.to_string()))
    }

    /// Get the first n rows
    pub fn head(&self, n: usize) -> Self {
        Self::new(self.df.head(Some(n)))
    }

    /// Get the last n rows
    pub fn tail(&self, n: usize) -> Self {
        Self::new(self.df.tail(Some(n)))
    }

    /// Drop null values
    pub fn drop_nulls(&self, subset: Option<Vec<String>>) -> DsuResult<Self> {
        let df = self.df
            .drop_nulls(subset.as_deref())
            .map_err(|e| DsuError::DataError(e.to_string()))?;
        
        Ok(Self::new(df))
    }

    /// Fill null values with a strategy
    pub fn fill_null(&self, strategy: FillNullStrategy) -> DsuResult<Self> {
        let mut df = self.df.clone();
        let columns: Vec<_> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        
        for col_name in columns {
            let col = df.column(&col_name)
                .map_err(|e| DsuError::DataError(e.to_string()))?;
            let filled = col.fill_null(strategy.clone())
                .map_err(|e| DsuError::DataError(e.to_string()))?;
            df.replace(&col_name, filled)
                .map_err(|e| DsuError::DataError(e.to_string()))?;
        }
        
        Ok(Self::new(df))
    }

    /// Group by columns
    pub fn group_by(&self, by: Vec<&str>) -> DsuResult<GroupBy> {
        self.df
            .group_by(by)
            .map_err(|e| DsuError::DataError(e.to_string()))
    }

    /// Sort by columns
    pub fn sort(&self, by: Vec<&str>, descending: Vec<bool>) -> DsuResult<Self> {
        let df = self.df
            .sort(by, descending, false)
            .map_err(|e| DsuError::DataError(e.to_string()))?;
        
        Ok(Self::new(df))
    }
}

impl From<DataFrame> for DsuDataFrame {
    fn from(df: DataFrame) -> Self {
        Self::new(df)
    }
}

impl From<DsuDataFrame> for DataFrame {
    fn from(dsu_df: DsuDataFrame) -> Self {
        dsu_df.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    #[test]
    fn test_dataframe_creation() {
        let df = df! {
            "a" => &[1, 2, 3],
            "b" => &[4, 5, 6],
        }.unwrap();
        
        let dsu_df = DsuDataFrame::new(df);
        assert_eq!(dsu_df.shape(), (3, 2));
        assert_eq!(dsu_df.height(), 3);
        assert_eq!(dsu_df.width(), 2);
    }

    #[test]
    fn test_dataframe_select() {
        let df = df! {
            "a" => &[1, 2, 3],
            "b" => &[4, 5, 6],
            "c" => &[7, 8, 9],
        }.unwrap();
        
        let dsu_df = DsuDataFrame::new(df);
        let selected = dsu_df.select(&["a", "c"]).unwrap();
        
        assert_eq!(selected.width(), 2);
        assert_eq!(selected.columns(), vec!["a", "c"]);
    }

    #[test]
    fn test_dataframe_head_tail() {
        let df = df! {
            "a" => &[1, 2, 3, 4, 5],
        }.unwrap();
        
        let dsu_df = DsuDataFrame::new(df);
        
        let head = dsu_df.head(2);
        assert_eq!(head.height(), 2);
        
        let tail = dsu_df.tail(2);
        assert_eq!(tail.height(), 2);
    }

    #[test]
    fn test_dataframe_column() {
        let df = df! {
            "a" => &[1, 2, 3],
            "b" => &[4, 5, 6],
        }.unwrap();
        
        let dsu_df = DsuDataFrame::new(df);
        let col = dsu_df.column("a").unwrap();
        
        assert_eq!(col.name(), "a");
        assert_eq!(col.len(), 3);
    }
}
