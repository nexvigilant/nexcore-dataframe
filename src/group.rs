//! GroupBy: hash-based grouping with aggregation.
//!
//! Supports the polars pattern: `df.lazy().group_by([cols]).agg([exprs])`
//! via `df.group_by(&["col1", "col2"])?.agg(&[Agg::Sum("val"), Agg::Count])`

// HashMap used for group-key accumulation during group_by. O(1) lookup is essential
// for large DataFrames; BTreeMap O(log n) is not appropriate here.
// Output row order is explicitly unspecified — callers sort if order matters.
#[allow(
    clippy::disallowed_types,
    reason = "HashMap needed for O(1) group-key accumulation; output row order is explicitly unspecified"
)]
use std::collections::HashMap;

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::error::DataFrameError;
use crate::scalar::Scalar;

/// Aggregation operation to apply within each group.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Agg {
    /// Sum of a numeric column.
    Sum(String),
    /// Mean of a numeric column.
    Mean(String),
    /// Minimum value.
    Min(String),
    /// Maximum value.
    Max(String),
    /// Count of rows in each group (no column needed).
    Count,
    /// First value of a column.
    First(String),
    /// Last value of a column.
    Last(String),
    /// Count of unique values.
    NUnique(String),
}

/// Intermediate grouping result. Call `.agg()` to produce a DataFrame.
#[derive(Debug)]
pub struct GroupBy<'a> {
    df: &'a DataFrame,
    group_cols: Vec<String>,
    /// Maps group key → row indices belonging to that group.
    #[allow(
        clippy::disallowed_types,
        reason = "HashMap for O(1) group-key lookup; see module-level allow"
    )]
    groups: HashMap<Vec<String>, Vec<usize>>,
}

impl GroupBy<'_> {
    /// Number of unique groups.
    #[must_use]
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// Apply aggregations and produce a result DataFrame.
    ///
    /// The result has one row per group. Group key columns come first,
    /// followed by one column per aggregation.
    pub fn agg(&self, aggs: &[Agg]) -> Result<DataFrame, DataFrameError> {
        let n_groups = self.groups.len();

        // Pre-allocate group key columns
        let mut key_vecs: Vec<Vec<Option<String>>> = self
            .group_cols
            .iter()
            .map(|_| Vec::with_capacity(n_groups))
            .collect();

        // Pre-allocate agg result columns (as Scalar vecs)
        let mut agg_results: Vec<Vec<Scalar>> =
            aggs.iter().map(|_| Vec::with_capacity(n_groups)).collect();

        // HashMap iteration: row order is arbitrary but consistent within this call.
        // The output DataFrame rows are not required to be ordered.
        #[allow(
            clippy::iter_over_hash_type,
            reason = "HashMap iteration builds parallel group rows; output row order is explicitly unspecified — callers sort if order matters"
        )]
        // Process each group
        for (key, indices) in &self.groups {
            // Fill key columns
            for (i, val) in key.iter().enumerate() {
                // i < key.len() == group_cols.len() == key_vecs.len() by construction
                #[allow(
                    clippy::indexing_slicing,
                    reason = "i iterates over key positions; key.len() == group_cols.len() == key_vecs.len() by GroupBy construction"
                )]
                key_vecs[i].push(Some(val.clone()));
            }

            // Compute each aggregation on the group's rows
            for (agg_idx, agg) in aggs.iter().enumerate() {
                let result = self.compute_agg(agg, indices)?;
                // agg_idx < aggs.len() == agg_results.len() by construction
                #[allow(
                    clippy::indexing_slicing,
                    reason = "agg_idx < aggs.len() == agg_results.len(); index is valid by parallel iteration"
                )]
                agg_results[agg_idx].push(result);
            }
        }

        // Build columns — i < group_cols.len() == key_vecs.len()
        #[allow(
            clippy::indexing_slicing,
            reason = "i iterates over 0..group_cols.len(); key_vecs has the same length by construction"
        )]
        let mut columns: Vec<Column> = key_vecs
            .into_iter()
            .enumerate()
            .map(|(i, data)| Column::new_string(self.group_cols[i].clone(), data))
            .collect();

        // Convert agg result scalars to typed columns
        // agg_idx < aggs.len() == agg_results.len()
        #[allow(
            clippy::indexing_slicing,
            reason = "agg_idx < aggs.len() == agg_results.len(); parallel zip ensures valid index"
        )]
        for (agg_idx, agg) in aggs.iter().enumerate() {
            let col_name = agg_column_name(agg);
            let col = scalars_to_column(&col_name, &agg_results[agg_idx]);
            columns.push(col);
        }

        DataFrame::new(columns)
    }

    /// Compute a single aggregation over the rows at `indices`.
    fn compute_agg(&self, agg: &Agg, indices: &[usize]) -> Result<Scalar, DataFrameError> {
        match agg {
            Agg::Count => {
                // indices.len() fits u64: slice length is bounded by usize which is <= u64 on all targets
                #[allow(
                    clippy::as_conversions,
                    reason = "usize→u64: on all supported platforms usize <= 64 bits, so this cast is lossless"
                )]
                Ok(Scalar::UInt64(indices.len() as u64))
            }
            Agg::Sum(col_name) => {
                let col = self.df.column(col_name)?;
                let sub = col.take(indices);
                Ok(sub.sum())
            }
            Agg::Mean(col_name) => {
                let col = self.df.column(col_name)?;
                let sub = col.take(indices);
                Ok(sub.mean())
            }
            Agg::Min(col_name) => {
                let col = self.df.column(col_name)?;
                let sub = col.take(indices);
                Ok(sub.min())
            }
            Agg::Max(col_name) => {
                let col = self.df.column(col_name)?;
                let sub = col.take(indices);
                Ok(sub.max())
            }
            Agg::First(col_name) => {
                let col = self.df.column(col_name)?;
                let sub = col.take(indices);
                Ok(sub.first())
            }
            Agg::Last(col_name) => {
                let col = self.df.column(col_name)?;
                let sub = col.take(indices);
                Ok(sub.last())
            }
            Agg::NUnique(col_name) => {
                let col = self.df.column(col_name)?;
                let sub = col.take(indices);
                // usize→u64: lossless on all platforms where usize <= 64 bits
                #[allow(
                    clippy::as_conversions,
                    reason = "usize→u64: n_unique() returns a Vec-length bounded by usize; lossless on all supported 32/64-bit platforms"
                )]
                Ok(Scalar::UInt64(sub.n_unique() as u64))
            }
        }
    }
}

impl DataFrame {
    /// Group the DataFrame by specified columns. Returns a `GroupBy` that
    /// can be aggregated.
    pub fn group_by(&self, cols: &[&str]) -> Result<GroupBy<'_>, DataFrameError> {
        // Validate all group columns exist
        for name in cols {
            self.column(name)?;
        }

        let group_cols: Vec<String> = cols.iter().map(|s| (*s).to_string()).collect();
        #[allow(
            clippy::disallowed_types,
            reason = "HashMap::new() for group accumulation; see module-level allow"
        )]
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for row_idx in 0..self.height() {
            let key: Vec<String> = cols
                .iter()
                .map(|name| {
                    self.column(name)
                        .ok()
                        .and_then(|col| col.get(row_idx))
                        .map_or_else(|| "null".to_string(), |s| s.to_string())
                })
                .collect();
            groups.entry(key).or_default().push(row_idx);
        }

        Ok(GroupBy {
            df: self,
            group_cols,
            groups,
        })
    }
}

/// Generate a descriptive column name for an aggregation.
fn agg_column_name(agg: &Agg) -> String {
    match agg {
        Agg::Sum(c) => format!("{c}_sum"),
        Agg::Mean(c) => format!("{c}_mean"),
        Agg::Min(c) => format!("{c}_min"),
        Agg::Max(c) => format!("{c}_max"),
        Agg::Count => "count".to_string(),
        Agg::First(c) => format!("{c}_first"),
        Agg::Last(c) => format!("{c}_last"),
        Agg::NUnique(c) => format!("{c}_nunique"),
    }
}

/// Convert a vec of mixed Scalars into a Column, inferring the best type.
fn scalars_to_column(name: &str, scalars: &[Scalar]) -> Column {
    // Determine dominant type (first non-null)
    let first_non_null = scalars.iter().find(|s| !s.is_null());

    match first_non_null {
        Some(Scalar::Int64(_)) => {
            let data: Vec<Option<i64>> = scalars
                .iter()
                .map(|s| match s {
                    Scalar::Int64(v) => Some(*v),
                    Scalar::Null
                    | Scalar::Bool(_)
                    | Scalar::UInt64(_)
                    | Scalar::Float64(_)
                    | Scalar::String(_) => None,
                })
                .collect();
            Column::new_i64(name, data)
        }
        Some(Scalar::UInt64(_)) => {
            let data: Vec<Option<u64>> = scalars
                .iter()
                .map(|s| match s {
                    Scalar::UInt64(v) => Some(*v),
                    Scalar::Null
                    | Scalar::Bool(_)
                    | Scalar::Int64(_)
                    | Scalar::Float64(_)
                    | Scalar::String(_) => None,
                })
                .collect();
            Column::new_u64(name, data)
        }
        Some(Scalar::Float64(_)) => {
            let data: Vec<Option<f64>> = scalars.iter().map(|s| s.as_f64()).collect();
            Column::new_f64(name, data)
        }
        Some(Scalar::Bool(_)) => {
            let data: Vec<Option<bool>> = scalars
                .iter()
                .map(|s| match s {
                    Scalar::Bool(v) => Some(*v),
                    Scalar::Null
                    | Scalar::Int64(_)
                    | Scalar::UInt64(_)
                    | Scalar::Float64(_)
                    | Scalar::String(_) => None,
                })
                .collect();
            Column::new_bool(name, data)
        }
        Some(Scalar::String(_)) | None => {
            let data: Vec<Option<String>> = scalars
                .iter()
                .map(|s| match s {
                    Scalar::String(v) => Some(v.clone()),
                    Scalar::Null => None,
                    other @ (Scalar::Bool(_)
                    | Scalar::Int64(_)
                    | Scalar::UInt64(_)
                    | Scalar::Float64(_)) => Some(other.to_string()),
                })
                .collect();
            Column::new_string(name, data)
        }
        Some(Scalar::Null) => {
            // All nulls — default to string column
            let data: Vec<Option<String>> = scalars.iter().map(|_| None).collect();
            Column::new_string(name, data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_by_count() {
        let df = DataFrame::new(vec![
            Column::from_strs("drug", &["asp", "met", "asp", "met", "asp"]),
            Column::from_i64s("val", vec![1, 2, 3, 4, 5]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let gb = df.group_by(&["drug"]).unwrap_or_else(|_| unreachable!());
        assert_eq!(gb.n_groups(), 2);

        let result = gb.agg(&[Agg::Count]).unwrap_or_else(|_| unreachable!());
        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 2); // drug + count
    }

    #[test]
    fn group_by_sum() {
        let df = DataFrame::new(vec![
            Column::from_strs("cat", &["a", "b", "a"]),
            Column::from_i64s("val", vec![10, 20, 30]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = df
            .group_by(&["cat"])
            .unwrap_or_else(|_| unreachable!())
            .agg(&[Agg::Sum("val".into())])
            .unwrap_or_else(|_| unreachable!());

        assert_eq!(result.height(), 2);
        // Find the "a" group and verify sum = 40
        for i in 0..result.height() {
            let cat = result
                .column("cat")
                .unwrap_or_else(|_| unreachable!())
                .get(i);
            let val = result
                .column("val_sum")
                .unwrap_or_else(|_| unreachable!())
                .get(i);
            if cat == Some(Scalar::String("a".into())) {
                assert_eq!(val, Some(Scalar::Int64(40)));
            }
        }
    }

    #[test]
    fn group_by_multiple_aggs() {
        let df = DataFrame::new(vec![
            Column::from_strs("g", &["x", "y", "x"]),
            Column::from_i64s("n", vec![1, 2, 3]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = df
            .group_by(&["g"])
            .unwrap_or_else(|_| unreachable!())
            .agg(&[
                Agg::Count,
                Agg::Sum("n".into()),
                Agg::Min("n".into()),
                Agg::Max("n".into()),
            ])
            .unwrap_or_else(|_| unreachable!());

        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 5); // g + count + n_sum + n_min + n_max
    }

    #[test]
    fn group_by_missing_column() {
        let df = DataFrame::new(vec![Column::from_i64s("x", vec![1])])
            .unwrap_or_else(|_| unreachable!());
        assert!(df.group_by(&["missing"]).is_err());
    }

    #[test]
    fn group_by_multi_key() {
        let df = DataFrame::new(vec![
            Column::from_strs("drug", &["asp", "asp", "met", "asp"]),
            Column::from_strs("event", &["ha", "na", "ha", "ha"]),
            Column::from_i64s("n", vec![1, 1, 1, 1]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let gb = df
            .group_by(&["drug", "event"])
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(gb.n_groups(), 3); // asp+ha, asp+na, met+ha
    }

    #[test]
    fn group_by_first_last() {
        let df = DataFrame::new(vec![
            Column::from_strs("g", &["a", "a", "a"]),
            Column::from_i64s("v", vec![10, 20, 30]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = df
            .group_by(&["g"])
            .unwrap_or_else(|_| unreachable!())
            .agg(&[Agg::First("v".into()), Agg::Last("v".into())])
            .unwrap_or_else(|_| unreachable!());

        assert_eq!(result.height(), 1);
        assert_eq!(
            result
                .column("v_first")
                .unwrap_or_else(|_| unreachable!())
                .get(0),
            Some(Scalar::Int64(10))
        );
        assert_eq!(
            result
                .column("v_last")
                .unwrap_or_else(|_| unreachable!())
                .get(0),
            Some(Scalar::Int64(30))
        );
    }
}
