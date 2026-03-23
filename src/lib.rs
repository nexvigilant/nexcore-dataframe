//! # nexcore-dataframe
//!
//! Sovereign columnar DataFrame engine for NexCore.
//!
//! Zero unsafe code, zero external DataFrame dependency.
//! Replaces polars across the workspace, eliminating CRITICAL (fast-float segfault)
//! and HIGH (pyo3 buffer overflow) transitive vulnerabilities.
//!
//! ## Core types
//!
//! - [`DataFrame`] — columnar table with named, typed columns
//! - [`Column`] — named array of homogeneous nullable values
//! - [`Scalar`] — single typed value for comparisons and aggregation results
//! - [`Counter`] — optimized hash-based group-count (replaces polars group_by+count)
//! - [`Schema`] — column name → type mapping
//!
//! ## Operations
//!
//! - Filter: `df.filter(&mask)`, `df.filter_by("col", |v| pred)`
//! - Sort: `df.sort("col", descending)`, `df.head(n)`, `df.tail(n)`
//! - Select: `df.select(&["col1", "col2"])`, `df.drop_columns(&["col"])`
//! - Aggregate: `col.sum()`, `col.mean()`, `col.min()`, `col.max()`, `col.median()`
//! - GroupBy: `df.group_by(&["col"])?.agg(&[Agg::Sum("val".into())])`
//! - Join: `df.join(&other, &["key"], JoinType::Inner)` (inner/left/right/outer/semi/anti)
//! - I/O: `DataFrame::from_json(s)`, `df.to_json()`, `df.to_json_file(path)`

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

// Core types
pub mod column;
pub mod dataframe;
pub mod error;
pub mod scalar;
pub mod schema;

// Operations
pub mod agg;
pub mod counter;
pub mod filter;
pub mod group;
pub mod join;
pub mod select;
pub mod sort;

// I/O
pub mod io;

// Re-exports for ergonomic use
pub use column::{Column, ColumnData, DataType};
pub use counter::Counter;
pub use dataframe::DataFrame;
pub use error::DataFrameError;
pub use group::Agg;
pub use join::JoinType;
pub use scalar::Scalar;
pub use schema::Schema;

/// Prelude: import everything needed for typical DataFrame usage.
pub mod prelude {
    pub use crate::column::{Column, DataType};
    pub use crate::counter::Counter;
    pub use crate::dataframe::DataFrame;
    pub use crate::error::DataFrameError;
    pub use crate::group::Agg;
    pub use crate::join::JoinType;
    pub use crate::scalar::Scalar;
    pub use crate::schema::Schema;
}
