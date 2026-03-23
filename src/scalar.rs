//! Scalar: a single typed value for heterogeneous operations.

use std::cmp::Ordering;
use std::fmt;

/// A single typed value used in aggregation results, comparisons, and literals.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Scalar {
    Null,
    Bool(bool),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    String(String),
}

impl Scalar {
    /// Returns true if this scalar is null.
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Try to extract as i64.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int64(v) => Some(*v),
            Self::UInt64(v) => i64::try_from(*v).ok(),
            Self::Null | Self::Bool(_) | Self::Float64(_) | Self::String(_) => None,
        }
    }

    /// Try to extract as u64.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::UInt64(v) => Some(*v),
            Self::Int64(v) => u64::try_from(*v).ok(),
            Self::Null | Self::Bool(_) | Self::Float64(_) | Self::String(_) => None,
        }
    }

    /// Try to extract as f64.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float64(v) => Some(*v),
            // i64→f64 and u64→f64 are widening casts; precision loss only for values
            // outside ±2^53 which are not meaningful in this DataFrame context
            #[allow(
                clippy::as_conversions,
                reason = "i64→f64 widening cast; no From<i64> for f64 in std, precision loss only beyond ±2^53"
            )]
            Self::Int64(v) => Some(*v as f64),
            #[allow(
                clippy::as_conversions,
                reason = "u64→f64 widening cast; no From<u64> for f64 in std, precision loss only beyond 2^53"
            )]
            Self::UInt64(v) => Some(*v as f64),
            Self::Null | Self::Bool(_) | Self::String(_) => None,
        }
    }

    /// Try to extract as bool.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::Null | Self::Int64(_) | Self::UInt64(_) | Self::Float64(_) | Self::String(_) => {
                None
            }
        }
    }

    /// Try to extract as string reference.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v.as_str()),
            Self::Null | Self::Bool(_) | Self::Int64(_) | Self::UInt64(_) | Self::Float64(_) => {
                None
            }
        }
    }

    /// Compare two scalars for ordering (nulls sort last).
    #[must_use]
    pub fn compare(&self, other: &Self) -> Ordering {
        // Cross-type numeric casts: i64→f64 and u64→f64 are widening; u64→i64 guarded
        // by sign check; all casts here are provably safe given the runtime guards.
        #[allow(
            clippy::as_conversions,
            reason = "cross-type numeric comparisons require widening casts; guards ensure safety (sign check for i64→u64, widening for int→f64)"
        )]
        match (self, other) {
            (Self::Null, Self::Null) => Ordering::Equal,
            (Self::Null, _) => Ordering::Greater,
            (_, Self::Null) => Ordering::Less,
            (Self::Bool(a), Self::Bool(b)) => a.cmp(b),
            (Self::Int64(a), Self::Int64(b)) => a.cmp(b),
            (Self::UInt64(a), Self::UInt64(b)) => a.cmp(b),
            (Self::Float64(a), Self::Float64(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Self::String(a), Self::String(b)) => a.cmp(b),
            // Cross-type: convert to f64 for numeric comparison
            (Self::Int64(a), Self::Float64(b)) => {
                (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (Self::Float64(a), Self::Int64(b)) => {
                a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
            }
            (Self::UInt64(a), Self::Float64(b)) => {
                (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (Self::Float64(a), Self::UInt64(b)) => {
                a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
            }
            (Self::Int64(a), Self::UInt64(b)) => {
                if *a < 0 {
                    Ordering::Less
                } else {
                    (*a as u64).cmp(b)
                }
            }
            (Self::UInt64(a), Self::Int64(b)) => {
                if *b < 0 {
                    Ordering::Greater
                } else {
                    a.cmp(&(*b as u64))
                }
            }
            // Non-comparable cross-type pairs (e.g. Bool vs String): arbitrary but consistent
            (
                Self::Bool(_),
                Self::Int64(_) | Self::UInt64(_) | Self::Float64(_) | Self::String(_),
            )
            | (
                Self::Int64(_) | Self::UInt64(_) | Self::Float64(_) | Self::String(_),
                Self::Bool(_),
            )
            | (Self::String(_), Self::Int64(_) | Self::UInt64(_) | Self::Float64(_))
            | (Self::Int64(_) | Self::UInt64(_) | Self::Float64(_), Self::String(_)) => {
                Ordering::Equal
            }
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Int64(v) => write!(f, "{v}"),
            Self::UInt64(v) => write!(f, "{v}"),
            Self::Float64(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "{v}"),
        }
    }
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Self::Int64(v)
    }
}
impl From<u64> for Scalar {
    fn from(v: u64) -> Self {
        Self::UInt64(v)
    }
}
impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Self::Float64(v)
    }
}
impl From<String> for Scalar {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}
impl From<&str> for Scalar {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_conversions() {
        assert_eq!(Scalar::from(42i64).as_i64(), Some(42));
        assert_eq!(Scalar::from(42u64).as_u64(), Some(42));
        assert_eq!(Scalar::from(3.14).as_f64(), Some(3.14));
        assert_eq!(Scalar::from(true).as_bool(), Some(true));
        assert_eq!(Scalar::from("hello").as_str(), Some("hello"));
        assert!(Scalar::Null.is_null());
    }

    #[test]
    fn scalar_cross_type_numeric() {
        assert_eq!(Scalar::from(42i64).as_f64(), Some(42.0));
        assert_eq!(Scalar::from(42u64).as_i64(), Some(42));
        assert_eq!(Scalar::from(42i64).as_u64(), Some(42));
    }

    #[test]
    fn scalar_ordering() {
        assert_eq!(
            Scalar::from(1i64).compare(&Scalar::from(2i64)),
            Ordering::Less
        );
        assert_eq!(
            Scalar::from(2i64).compare(&Scalar::from(1i64)),
            Ordering::Greater
        );
        assert_eq!(Scalar::Null.compare(&Scalar::from(1i64)), Ordering::Greater);
        assert_eq!(Scalar::from(1i64).compare(&Scalar::Null), Ordering::Less);
    }

    #[test]
    fn scalar_display() {
        assert_eq!(format!("{}", Scalar::Null), "null");
        assert_eq!(format!("{}", Scalar::from(42i64)), "42");
        assert_eq!(format!("{}", Scalar::from("hi")), "hi");
    }
}
