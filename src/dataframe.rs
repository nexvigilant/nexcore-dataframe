//! DataFrame: the core columnar data structure.

use crate::column::{Column, DataType};
use crate::error::DataFrameError;
use crate::scalar::Scalar;
use crate::schema::Schema;

/// A columnar data structure. Each column is a named, typed array.
/// All columns must have the same length.
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<Column>,
}

impl DataFrame {
    /// Create a new DataFrame from columns.
    /// All columns must have the same length. Empty vec produces an empty DataFrame.
    pub fn new(columns: Vec<Column>) -> Result<Self, DataFrameError> {
        if columns.is_empty() {
            return Ok(Self { columns });
        }
        // columns is non-empty (checked above), so index 0 and slice [1..] are always valid
        #[allow(
            clippy::indexing_slicing,
            reason = "columns is non-empty (checked by is_empty() guard above); index 0 and slice [1..] are always in bounds"
        )]
        let expected = columns[0].len();
        #[allow(
            clippy::indexing_slicing,
            reason = "columns is non-empty; slice [1..] on a non-empty Vec is always valid (may produce empty slice)"
        )]
        for col in &columns[1..] {
            if col.len() != expected {
                return Err(DataFrameError::LengthMismatch {
                    expected,
                    actual: col.len(),
                });
            }
        }
        Ok(Self { columns })
    }

    /// Create an empty DataFrame (no columns, no rows).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// Number of rows.
    #[must_use]
    pub fn height(&self) -> usize {
        self.columns.first().map_or(0, |c| c.len())
    }

    /// Number of columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Whether the DataFrame has no columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Get a column by name.
    pub fn column(&self, name: &str) -> Result<&Column, DataFrameError> {
        self.columns
            .iter()
            .find(|c| c.name() == name)
            .ok_or_else(|| DataFrameError::ColumnNotFound(name.to_string()))
    }

    /// Get all column names.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name()).collect()
    }

    /// Get all columns as a slice.
    #[must_use]
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Get the schema (column names and types).
    #[must_use]
    pub fn schema(&self) -> Schema {
        Schema::new(
            self.columns
                .iter()
                .map(|c| (c.name().to_string(), c.dtype()))
                .collect(),
        )
    }

    /// Get a row as a vector of Scalars.
    pub fn row(&self, index: usize) -> Option<Vec<Scalar>> {
        if index >= self.height() {
            return None;
        }
        Some(
            self.columns
                .iter()
                .map(|c| c.get(index).unwrap_or(Scalar::Null))
                .collect(),
        )
    }

    /// Add or replace a column. If a column with the same name exists, it is replaced.
    pub fn with_column(&self, col: Column) -> Result<Self, DataFrameError> {
        if !self.is_empty() && col.len() != self.height() {
            return Err(DataFrameError::LengthMismatch {
                expected: self.height(),
                actual: col.len(),
            });
        }
        let mut columns: Vec<Column> = self
            .columns
            .iter()
            .filter(|c| c.name() != col.name())
            .cloned()
            .collect();
        columns.push(col);
        Ok(Self { columns })
    }

    /// Internal: build DataFrame from pre-validated columns (same length guaranteed).
    pub(crate) fn from_columns_unchecked(columns: Vec<Column>) -> Self {
        Self { columns }
    }
}

impl std::fmt::Display for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Header
        let names: Vec<&str> = self.columns.iter().map(|c| c.name()).collect();
        writeln!(f, "{}", names.join("\t"))?;
        // Rows (max 20)
        let max_rows = self.height().min(20);
        for i in 0..max_rows {
            let vals: Vec<String> = self
                .columns
                .iter()
                .map(|c| c.get(i).map_or("null".to_string(), |s| s.to_string()))
                .collect();
            writeln!(f, "{}", vals.join("\t"))?;
        }
        if self.height() > max_rows {
            // max_rows = height.min(20) <= height, so subtraction cannot underflow
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "max_rows = self.height().min(20) so max_rows <= self.height(); subtraction cannot underflow"
            )]
            writeln!(f, "... ({} more rows)", self.height() - max_rows)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        let df = DataFrame::new(vec![
            Column::from_strs("name", &["a", "b"]),
            Column::from_i64s("val", vec![1, 2]),
        ]);
        assert!(df.is_ok());
        let df = df.unwrap_or_else(|_| unreachable!());
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 2);
    }

    #[test]
    fn new_length_mismatch() {
        let df = DataFrame::new(vec![
            Column::from_strs("a", &["x", "y"]),
            Column::from_i64s("b", vec![1]),
        ]);
        assert!(df.is_err());
    }

    #[test]
    fn empty_dataframe() {
        let df = DataFrame::empty();
        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 0);
        assert!(df.is_empty());
    }

    #[test]
    fn column_lookup() {
        let df = DataFrame::new(vec![
            Column::from_strs("name", &["a"]),
            Column::from_i64s("val", vec![1]),
        ])
        .unwrap_or_else(|_| unreachable!());
        assert!(df.column("name").is_ok());
        assert!(df.column("missing").is_err());
    }

    #[test]
    fn schema_extraction() {
        let df = DataFrame::new(vec![
            Column::from_strs("s", &["x"]),
            Column::from_f64s("f", vec![1.0]),
        ])
        .unwrap_or_else(|_| unreachable!());
        let s = df.schema();
        assert_eq!(s.len(), 2);
        assert_eq!(s.dtype("s"), Some(DataType::Utf8));
        assert_eq!(s.dtype("f"), Some(DataType::Float64));
    }

    #[test]
    fn with_column_add() {
        let df = DataFrame::new(vec![Column::from_i64s("a", vec![1, 2])])
            .unwrap_or_else(|_| unreachable!());
        let df2 = df
            .with_column(Column::from_i64s("b", vec![3, 4]))
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(df2.width(), 2);
    }

    #[test]
    fn with_column_replace() {
        let df = DataFrame::new(vec![Column::from_i64s("a", vec![1, 2])])
            .unwrap_or_else(|_| unreachable!());
        let df2 = df
            .with_column(Column::from_i64s("a", vec![10, 20]))
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(df2.width(), 1);
        assert_eq!(
            df2.column("a").unwrap_or_else(|_| unreachable!()).get(0),
            Some(Scalar::Int64(10))
        );
    }

    #[test]
    fn with_column_length_mismatch() {
        let df = DataFrame::new(vec![Column::from_i64s("a", vec![1, 2])])
            .unwrap_or_else(|_| unreachable!());
        assert!(df.with_column(Column::from_i64s("b", vec![1])).is_err());
    }

    #[test]
    fn row_access() {
        let df = DataFrame::new(vec![
            Column::from_strs("name", &["alice"]),
            Column::from_i64s("age", vec![30]),
        ])
        .unwrap_or_else(|_| unreachable!());
        let r = df.row(0);
        assert!(r.is_some());
        let r = r.unwrap_or_else(|| unreachable!());
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], Scalar::String("alice".to_string()));
        assert_eq!(r[1], Scalar::Int64(30));
        assert!(df.row(1).is_none());
    }
}
