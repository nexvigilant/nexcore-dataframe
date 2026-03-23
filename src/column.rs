//! Column: a named, typed array — the fundamental storage unit.

use crate::error::DataFrameError;
use crate::scalar::Scalar;

/// Type enumeration for column data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DataType {
    Bool,
    Int64,
    UInt64,
    Float64,
    Utf8,
}

/// Type-safe column storage. Each variant holds nullable values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ColumnData {
    Bool(Vec<Option<bool>>),
    Int64(Vec<Option<i64>>),
    UInt64(Vec<Option<u64>>),
    Float64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
}

impl ColumnData {
    /// Number of elements (including nulls).
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::UInt64(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::String(v) => v.len(),
        }
    }

    /// Whether the column has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the data type.
    #[must_use]
    pub fn dtype(&self) -> DataType {
        match self {
            Self::Bool(_) => DataType::Bool,
            Self::Int64(_) => DataType::Int64,
            Self::UInt64(_) => DataType::UInt64,
            Self::Float64(_) => DataType::Float64,
            Self::String(_) => DataType::Utf8,
        }
    }

    /// Get value at index as Scalar.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Scalar> {
        match self {
            Self::Bool(v) => v.get(index).map(|o| match o {
                Some(b) => Scalar::Bool(*b),
                None => Scalar::Null,
            }),
            Self::Int64(v) => v.get(index).map(|o| match o {
                Some(n) => Scalar::Int64(*n),
                None => Scalar::Null,
            }),
            Self::UInt64(v) => v.get(index).map(|o| match o {
                Some(n) => Scalar::UInt64(*n),
                None => Scalar::Null,
            }),
            Self::Float64(v) => v.get(index).map(|o| match o {
                Some(n) => Scalar::Float64(*n),
                None => Scalar::Null,
            }),
            Self::String(v) => v.get(index).map(|o| match o {
                Some(s) => Scalar::String(s.clone()),
                None => Scalar::Null,
            }),
        }
    }

    /// Count of non-null values.
    #[must_use]
    pub fn non_null_count(&self) -> usize {
        match self {
            Self::Bool(v) => v.iter().filter(|o| o.is_some()).count(),
            Self::Int64(v) => v.iter().filter(|o| o.is_some()).count(),
            Self::UInt64(v) => v.iter().filter(|o| o.is_some()).count(),
            Self::Float64(v) => v.iter().filter(|o| o.is_some()).count(),
            Self::String(v) => v.iter().filter(|o| o.is_some()).count(),
        }
    }

    /// Collect rows at specified indices into a new ColumnData.
    ///
    /// # Panics
    ///
    /// Panics if any index in `indices` is out of bounds. Callers must ensure all
    /// indices are valid (e.g., from `filter` or `sort` which derive indices from
    /// `0..self.len()`).
    pub fn take(&self, indices: &[usize]) -> Self {
        #[allow(
            clippy::indexing_slicing,
            reason = "indices are always derived from 0..len() in sort/filter/group_by — bounds are structurally guaranteed by callers"
        )]
        match self {
            Self::Bool(v) => Self::Bool(indices.iter().map(|&i| v[i]).collect()),
            Self::Int64(v) => Self::Int64(indices.iter().map(|&i| v[i]).collect()),
            Self::UInt64(v) => Self::UInt64(indices.iter().map(|&i| v[i]).collect()),
            Self::Float64(v) => Self::Float64(indices.iter().map(|&i| v[i]).collect()),
            Self::String(v) => Self::String(indices.iter().map(|&i| v[i].clone()).collect()),
        }
    }

    /// Collect rows at specified optional indices into a new ColumnData.
    ///
    /// `Some(i)` takes the value at index `i`; `None` produces a null value.
    /// Used by join operations where one side may have no matching row.
    pub fn take_optional(&self, indices: &[Option<usize>]) -> Self {
        #[allow(
            clippy::indexing_slicing,
            reason = "Some(i) indices are derived from 0..len() in join probe — bounds are structurally guaranteed by callers"
        )]
        match self {
            Self::Bool(v) => Self::Bool(indices.iter().map(|opt| opt.and_then(|i| v[i])).collect()),
            Self::Int64(v) => {
                Self::Int64(indices.iter().map(|opt| opt.and_then(|i| v[i])).collect())
            }
            Self::UInt64(v) => {
                Self::UInt64(indices.iter().map(|opt| opt.and_then(|i| v[i])).collect())
            }
            Self::Float64(v) => {
                Self::Float64(indices.iter().map(|opt| opt.and_then(|i| v[i])).collect())
            }
            Self::String(v) => Self::String(
                indices
                    .iter()
                    .map(|opt| opt.and_then(|i| v[i].clone()))
                    .collect(),
            ),
        }
    }
}

/// A single named column with homogeneous type.
#[derive(Debug, Clone)]
pub struct Column {
    name: String,
    data: ColumnData,
}

impl Column {
    // =========================================================================
    // Nullable constructors
    // =========================================================================

    /// Create a boolean column with nullable values.
    pub fn new_bool(name: impl Into<String>, data: Vec<Option<bool>>) -> Self {
        Self {
            name: name.into(),
            data: ColumnData::Bool(data),
        }
    }

    /// Create an i64 column with nullable values.
    pub fn new_i64(name: impl Into<String>, data: Vec<Option<i64>>) -> Self {
        Self {
            name: name.into(),
            data: ColumnData::Int64(data),
        }
    }

    /// Create a u64 column with nullable values.
    pub fn new_u64(name: impl Into<String>, data: Vec<Option<u64>>) -> Self {
        Self {
            name: name.into(),
            data: ColumnData::UInt64(data),
        }
    }

    /// Create an f64 column with nullable values.
    pub fn new_f64(name: impl Into<String>, data: Vec<Option<f64>>) -> Self {
        Self {
            name: name.into(),
            data: ColumnData::Float64(data),
        }
    }

    /// Create a string column with nullable values.
    pub fn new_string(name: impl Into<String>, data: Vec<Option<String>>) -> Self {
        Self {
            name: name.into(),
            data: ColumnData::String(data),
        }
    }

    // =========================================================================
    // Convenience constructors (non-nullable)
    // =========================================================================

    /// Create a boolean column from non-nullable values.
    pub fn from_bools(name: impl Into<String>, data: Vec<bool>) -> Self {
        Self::new_bool(name, data.into_iter().map(Some).collect())
    }

    /// Create an i64 column from non-nullable values.
    pub fn from_i64s(name: impl Into<String>, data: Vec<i64>) -> Self {
        Self::new_i64(name, data.into_iter().map(Some).collect())
    }

    /// Create a u64 column from non-nullable values.
    pub fn from_u64s(name: impl Into<String>, data: Vec<u64>) -> Self {
        Self::new_u64(name, data.into_iter().map(Some).collect())
    }

    /// Create an f64 column from non-nullable values.
    pub fn from_f64s(name: impl Into<String>, data: Vec<f64>) -> Self {
        Self::new_f64(name, data.into_iter().map(Some).collect())
    }

    /// Create a string column from owned strings (non-nullable).
    pub fn from_strings(name: impl Into<String>, data: Vec<String>) -> Self {
        Self::new_string(name, data.into_iter().map(Some).collect())
    }

    /// Create a string column from string slices (non-nullable).
    pub fn from_strs(name: impl Into<String>, data: &[&str]) -> Self {
        Self::new_string(name, data.iter().map(|s| Some((*s).to_string())).collect())
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Column name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Data type of this column.
    #[must_use]
    pub fn dtype(&self) -> DataType {
        self.data.dtype()
    }

    /// Number of elements (including nulls).
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the column is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Count of non-null values.
    #[must_use]
    pub fn non_null_count(&self) -> usize {
        self.data.non_null_count()
    }

    /// Count of null values.
    #[must_use]
    pub fn null_count(&self) -> usize {
        // non_null_count() <= len() is an invariant: every non-null is also in len()
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "non_null_count() is always <= len() by construction — both count the same Vec elements"
        )]
        {
            self.len() - self.non_null_count()
        }
    }

    /// Get the underlying data reference.
    #[must_use]
    pub fn data(&self) -> &ColumnData {
        &self.data
    }

    /// Get value at index as Scalar. Returns None if index out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Scalar> {
        self.data.get(index)
    }

    /// Rename this column (returns a new column).
    #[must_use]
    pub fn rename(&self, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data: self.data.clone(),
        }
    }

    /// Take rows at specified indices.
    pub fn take(&self, indices: &[usize]) -> Self {
        Self {
            name: self.name.clone(),
            data: self.data.take(indices),
        }
    }

    /// Take rows at optional indices. `None` produces null values.
    /// Used by join operations where one side may have no matching row.
    pub fn take_optional(&self, indices: &[Option<usize>]) -> Self {
        Self {
            name: self.name.clone(),
            data: self.data.take_optional(indices),
        }
    }

    // =========================================================================
    // Typed iterators
    // =========================================================================

    /// Iterate as string references. Returns Err if column is not Utf8.
    pub fn as_str_iter(&self) -> Result<impl Iterator<Item = Option<&str>>, DataFrameError> {
        match &self.data {
            ColumnData::String(v) => Ok(v.iter().map(|o| o.as_deref())),
            ColumnData::Bool(_)
            | ColumnData::Int64(_)
            | ColumnData::UInt64(_)
            | ColumnData::Float64(_) => Err(DataFrameError::TypeMismatch {
                column: self.name.clone(),
                expected: DataType::Utf8,
                actual: self.dtype(),
            }),
        }
    }

    /// Iterate as i64 values. Returns Err if column is not Int64.
    pub fn as_i64_iter(&self) -> Result<impl Iterator<Item = Option<i64>> + '_, DataFrameError> {
        match &self.data {
            ColumnData::Int64(v) => Ok(v.iter().copied()),
            ColumnData::Bool(_)
            | ColumnData::UInt64(_)
            | ColumnData::Float64(_)
            | ColumnData::String(_) => Err(DataFrameError::TypeMismatch {
                column: self.name.clone(),
                expected: DataType::Int64,
                actual: self.dtype(),
            }),
        }
    }

    /// Iterate as u64 values. Returns Err if column is not UInt64.
    pub fn as_u64_iter(&self) -> Result<impl Iterator<Item = Option<u64>> + '_, DataFrameError> {
        match &self.data {
            ColumnData::UInt64(v) => Ok(v.iter().copied()),
            ColumnData::Bool(_)
            | ColumnData::Int64(_)
            | ColumnData::Float64(_)
            | ColumnData::String(_) => Err(DataFrameError::TypeMismatch {
                column: self.name.clone(),
                expected: DataType::UInt64,
                actual: self.dtype(),
            }),
        }
    }

    /// Iterate as f64 values. Returns Err if column is not Float64.
    pub fn as_f64_iter(&self) -> Result<impl Iterator<Item = Option<f64>> + '_, DataFrameError> {
        match &self.data {
            ColumnData::Float64(v) => Ok(v.iter().copied()),
            ColumnData::Bool(_)
            | ColumnData::Int64(_)
            | ColumnData::UInt64(_)
            | ColumnData::String(_) => Err(DataFrameError::TypeMismatch {
                column: self.name.clone(),
                expected: DataType::Float64,
                actual: self.dtype(),
            }),
        }
    }

    /// Iterate as bool values. Returns Err if column is not Bool.
    pub fn as_bool_iter(&self) -> Result<impl Iterator<Item = Option<bool>> + '_, DataFrameError> {
        match &self.data {
            ColumnData::Bool(v) => Ok(v.iter().copied()),
            ColumnData::Int64(_)
            | ColumnData::UInt64(_)
            | ColumnData::Float64(_)
            | ColumnData::String(_) => Err(DataFrameError::TypeMismatch {
                column: self.name.clone(),
                expected: DataType::Bool,
                actual: self.dtype(),
            }),
        }
    }

    /// Get a string value at index. Returns Err if not Utf8 column.
    pub fn get_str(&self, index: usize) -> Result<Option<&str>, DataFrameError> {
        match &self.data {
            ColumnData::String(v) => match v.get(index) {
                Some(o) => Ok(o.as_deref()),
                None => Err(DataFrameError::IndexOutOfBounds {
                    index,
                    length: v.len(),
                }),
            },
            ColumnData::Bool(_)
            | ColumnData::Int64(_)
            | ColumnData::UInt64(_)
            | ColumnData::Float64(_) => Err(DataFrameError::TypeMismatch {
                column: self.name.clone(),
                expected: DataType::Utf8,
                actual: self.dtype(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_strs_construction() {
        let c = Column::from_strs("names", &["alice", "bob", "carol"]);
        assert_eq!(c.name(), "names");
        assert_eq!(c.dtype(), DataType::Utf8);
        assert_eq!(c.len(), 3);
        assert_eq!(c.non_null_count(), 3);
        assert_eq!(c.null_count(), 0);
    }

    #[test]
    fn nullable_column() {
        let c = Column::new_i64("x", vec![Some(1), None, Some(3)]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.non_null_count(), 2);
        assert_eq!(c.null_count(), 1);
        assert_eq!(c.get(0), Some(Scalar::Int64(1)));
        assert_eq!(c.get(1), Some(Scalar::Null));
        assert_eq!(c.get(3), None);
    }

    #[test]
    fn typed_iterators() {
        let c = Column::from_i64s("nums", vec![10, 20, 30]);
        let vals: Vec<_> = c.as_i64_iter().unwrap_or_else(|_| unreachable!()).collect();
        assert_eq!(vals, vec![Some(10), Some(20), Some(30)]);

        // Type mismatch
        assert!(c.as_str_iter().is_err());
    }

    #[test]
    fn take_indices() {
        let c = Column::from_strs("x", &["a", "b", "c", "d"]);
        let taken = c.take(&[0, 2, 3]);
        assert_eq!(taken.len(), 3);
        assert_eq!(taken.get_str(0).unwrap_or(None), Some("a"));
        assert_eq!(taken.get_str(1).unwrap_or(None), Some("c"));
        assert_eq!(taken.get_str(2).unwrap_or(None), Some("d"));
    }

    #[test]
    fn rename_column() {
        let c = Column::from_i64s("old", vec![1, 2]);
        let c2 = c.rename("new");
        assert_eq!(c2.name(), "new");
        assert_eq!(c2.len(), 2);
    }

    #[test]
    fn take_optional_indices() {
        let c = Column::from_strs("x", &["a", "b", "c"]);
        let taken = c.take_optional(&[Some(0), None, Some(2)]);
        assert_eq!(taken.len(), 3);
        assert_eq!(taken.get_str(0).unwrap_or(None), Some("a"));
        assert_eq!(taken.get_str(1).unwrap_or(None), None);
        assert_eq!(taken.get_str(2).unwrap_or(None), Some("c"));

        // Numeric types
        let n = Column::from_i64s("n", vec![10, 20, 30]);
        let taken = n.take_optional(&[None, Some(1), Some(2)]);
        assert_eq!(taken.get(0), Some(Scalar::Null));
        assert_eq!(taken.get(1), Some(Scalar::Int64(20)));
        assert_eq!(taken.get(2), Some(Scalar::Int64(30)));
    }

    #[test]
    fn all_data_types_construct() {
        let b = Column::from_bools("b", vec![true, false]);
        assert_eq!(b.dtype(), DataType::Bool);
        let i = Column::from_i64s("i", vec![1, 2]);
        assert_eq!(i.dtype(), DataType::Int64);
        let u = Column::from_u64s("u", vec![1, 2]);
        assert_eq!(u.dtype(), DataType::UInt64);
        let f = Column::from_f64s("f", vec![1.0, 2.0]);
        assert_eq!(f.dtype(), DataType::Float64);
        let s = Column::from_strings("s", vec!["a".into(), "b".into()]);
        assert_eq!(s.dtype(), DataType::Utf8);
    }
}
