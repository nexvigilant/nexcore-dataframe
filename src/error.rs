//! DataFrame error types.

use std::fmt;

/// All errors produced by nexcore-dataframe operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum DataFrameError {
    /// Named column was not found in the DataFrame.
    ColumnNotFound(String),

    /// Column lengths don't match during DataFrame construction.
    LengthMismatch { expected: usize, actual: usize },

    /// Column has wrong type for the requested operation.
    TypeMismatch {
        column: String,
        expected: DataType,
        actual: DataType,
    },

    /// Operation requires a non-empty DataFrame.
    Empty,

    /// I/O error during read/write.
    Io(std::io::Error),

    /// JSON serialization/deserialization error.
    Json(serde_json::Error),

    /// Index out of bounds.
    IndexOutOfBounds { index: usize, length: usize },

    /// Join key column count mismatch between left and right tables.
    JoinKeyMismatch {
        left_count: usize,
        right_count: usize,
    },

    /// General error with message.
    Other(String),
}

use crate::DataType;

impl fmt::Display for DataFrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnNotFound(name) => write!(f, "column not found: '{name}'"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "column length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::TypeMismatch {
                column,
                expected,
                actual,
            } => write!(
                f,
                "type mismatch: column '{column}' is {actual:?}, expected {expected:?}"
            ),
            Self::Empty => write!(f, "empty dataframe"),
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Json(e) => write!(f, "json error: {e}"),
            Self::IndexOutOfBounds { index, length } => {
                write!(f, "index {index} out of bounds for length {length}")
            }
            Self::JoinKeyMismatch {
                left_count,
                right_count,
            } => write!(
                f,
                "join key count mismatch: left has {left_count}, right has {right_count}"
            ),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for DataFrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::ColumnNotFound(_)
            | Self::LengthMismatch { .. }
            | Self::TypeMismatch { .. }
            | Self::Empty
            | Self::IndexOutOfBounds { .. }
            | Self::JoinKeyMismatch { .. }
            | Self::Other(_) => None,
        }
    }
}

impl From<std::io::Error> for DataFrameError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for DataFrameError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
