//! Scalar aggregation functions on columns.

use crate::column::{Column, ColumnData};
use crate::error::DataFrameError;
use crate::scalar::Scalar;

impl Column {
    /// Sum of non-null numeric values. Returns Null for empty/non-numeric columns.
    #[must_use]
    pub fn sum(&self) -> Scalar {
        match self.data() {
            ColumnData::Int64(v) => {
                let total: i64 = v.iter().filter_map(|o| *o).sum();
                Scalar::Int64(total)
            }
            ColumnData::UInt64(v) => {
                let total: u64 = v.iter().filter_map(|o| *o).sum();
                Scalar::UInt64(total)
            }
            ColumnData::Float64(v) => {
                let total: f64 = v.iter().filter_map(|o| *o).sum();
                Scalar::Float64(total)
            }
            ColumnData::Bool(_) | ColumnData::String(_) => Scalar::Null,
        }
    }

    /// Mean of non-null numeric values. Returns Null if no non-null values.
    #[must_use]
    pub fn mean(&self) -> Scalar {
        let count = self.non_null_count();
        if count == 0 {
            return Scalar::Null;
        }
        // count > 0 is verified above; int→f64 widening casts are safe for practical values
        #[allow(
            clippy::as_conversions,
            reason = "i64/u64→f64 widening cast for numeric mean; count→f64 safe as count <= usize::MAX << 2^53"
        )]
        match self.data() {
            ColumnData::Int64(v) => {
                let total: f64 = v.iter().filter_map(|o| o.map(|n| n as f64)).sum();
                Scalar::Float64(total / count as f64)
            }
            ColumnData::UInt64(v) => {
                let total: f64 = v.iter().filter_map(|o| o.map(|n| n as f64)).sum();
                Scalar::Float64(total / count as f64)
            }
            ColumnData::Float64(v) => {
                let total: f64 = v.iter().filter_map(|o| *o).sum();
                Scalar::Float64(total / count as f64)
            }
            ColumnData::Bool(_) | ColumnData::String(_) => Scalar::Null,
        }
    }

    /// Minimum non-null value. Returns Null if no non-null values.
    #[must_use]
    pub fn min(&self) -> Scalar {
        let mut result = Scalar::Null;
        for i in 0..self.len() {
            if let Some(val) = self.get(i) {
                if val.is_null() {
                    continue;
                }
                if result.is_null() || val.compare(&result) == std::cmp::Ordering::Less {
                    result = val;
                }
            }
        }
        result
    }

    /// Maximum non-null value. Returns Null if no non-null values.
    #[must_use]
    pub fn max(&self) -> Scalar {
        let mut result = Scalar::Null;
        for i in 0..self.len() {
            if let Some(val) = self.get(i) {
                if val.is_null() {
                    continue;
                }
                if result.is_null() || val.compare(&result) == std::cmp::Ordering::Greater {
                    result = val;
                }
            }
        }
        result
    }

    /// Median of non-null numeric values. Returns Null if no non-null values.
    #[must_use]
    pub fn median(&self) -> Scalar {
        // int→f64 widening casts are safe for practical column values
        #[allow(
            clippy::as_conversions,
            reason = "i64/u64→f64 widening cast for median computation; precision loss only beyond ±2^53"
        )]
        let mut vals: Vec<f64> = match self.data() {
            ColumnData::Int64(v) => v.iter().filter_map(|o| o.map(|n| n as f64)).collect(),
            ColumnData::UInt64(v) => v.iter().filter_map(|o| o.map(|n| n as f64)).collect(),
            ColumnData::Float64(v) => v.iter().filter_map(|o| *o).collect(),
            ColumnData::Bool(_) | ColumnData::String(_) => return Scalar::Null,
        };
        if vals.is_empty() {
            return Scalar::Null;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = vals.len() / 2;
        // vals is non-empty (checked above); mid = len/2 so mid < len
        // for even len: mid >= 1 (since len >= 2 when even and non-empty), mid-1 < mid < len
        // for odd len: mid < len
        #[allow(
            clippy::indexing_slicing,
            reason = "mid = len/2 so mid < len; for even len mid >= 1 since len >= 2"
        )]
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "mid = len/2 >= 1 when len is even and non-empty (len >= 2); subtraction cannot underflow"
        )]
        if vals.len() % 2 == 0 {
            Scalar::Float64((vals[mid - 1] + vals[mid]) / 2.0)
        } else {
            Scalar::Float64(vals[mid])
        }
    }

    /// Standard deviation (population) of non-null numeric values.
    #[must_use]
    pub fn std_dev(&self) -> Scalar {
        let mean = match self.mean() {
            Scalar::Float64(m) => m,
            Scalar::Null
            | Scalar::Bool(_)
            | Scalar::Int64(_)
            | Scalar::UInt64(_)
            | Scalar::String(_) => {
                return Scalar::Null;
            }
        };
        let count = self.non_null_count();
        if count == 0 {
            return Scalar::Null;
        }
        // int→f64 widening casts; count→f64 safe as count << 2^53
        #[allow(
            clippy::as_conversions,
            reason = "i64/u64→f64 widening cast for variance computation; count→f64 safe since Vec capacity is bounded by usize << 2^53"
        )]
        let variance: f64 = match self.data() {
            ColumnData::Int64(v) => {
                v.iter()
                    .filter_map(|o| o.map(|n| (n as f64 - mean).powi(2)))
                    .sum::<f64>()
                    / count as f64
            }
            ColumnData::UInt64(v) => {
                v.iter()
                    .filter_map(|o| o.map(|n| (n as f64 - mean).powi(2)))
                    .sum::<f64>()
                    / count as f64
            }
            ColumnData::Float64(v) => {
                v.iter()
                    .filter_map(|o| o.map(|n| (n - mean).powi(2)))
                    .sum::<f64>()
                    / count as f64
            }
            ColumnData::Bool(_) | ColumnData::String(_) => return Scalar::Null,
        };
        Scalar::Float64(variance.sqrt())
    }

    /// Count of unique non-null values.
    #[must_use]
    pub fn n_unique(&self) -> usize {
        // HashSet used purely for deduplication — returns only the count, not the set.
        // Order of the set never matters here; O(1) insert beats BTreeSet O(log n).
        #[allow(
            clippy::disallowed_types,
            reason = "HashSet used for O(1) deduplication; only the count is returned, set order is irrelevant"
        )]
        use std::collections::HashSet;
        #[allow(
            clippy::disallowed_types,
            reason = "HashSet::new() for n_unique deduplication; see inline allow above"
        )]
        let mut seen = HashSet::new();
        for i in 0..self.len() {
            if let Some(val) = self.get(i) {
                if !val.is_null() {
                    seen.insert(format!("{val}"));
                }
            }
        }
        seen.len()
    }

    /// First non-null value in the column.
    #[must_use]
    pub fn first(&self) -> Scalar {
        for i in 0..self.len() {
            if let Some(val) = self.get(i) {
                if !val.is_null() {
                    return val;
                }
            }
        }
        Scalar::Null
    }

    /// Last non-null value in the column.
    #[must_use]
    pub fn last(&self) -> Scalar {
        for i in (0..self.len()).rev() {
            if let Some(val) = self.get(i) {
                if !val.is_null() {
                    return val;
                }
            }
        }
        Scalar::Null
    }

    /// Get a quantile (0.0 to 1.0) from non-null numeric values.
    pub fn quantile(&self, q: f64) -> Result<Scalar, DataFrameError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(DataFrameError::Other(format!(
                "quantile must be between 0.0 and 1.0, got {q}"
            )));
        }
        // int→f64 widening casts are safe for practical column values
        #[allow(
            clippy::as_conversions,
            reason = "i64/u64→f64 widening for quantile; (len-1)→f64 safe since Vec len << 2^53; floor/ceil→usize: pos is in [0, len-1] so fits usize"
        )]
        let mut vals: Vec<f64> = match self.data() {
            ColumnData::Int64(v) => v.iter().filter_map(|o| o.map(|n| n as f64)).collect(),
            ColumnData::UInt64(v) => v.iter().filter_map(|o| o.map(|n| n as f64)).collect(),
            ColumnData::Float64(v) => v.iter().filter_map(|o| *o).collect(),
            ColumnData::Bool(_) | ColumnData::String(_) => return Ok(Scalar::Null),
        };
        if vals.is_empty() {
            return Ok(Scalar::Null);
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // vals.len() >= 1 (checked above); q in [0,1]; pos in [0, len-1]
        // lower = floor(pos) <= upper = ceil(pos) <= len-1, so both indices are valid
        // vals.len() >= 1, so vals.len() - 1 cannot underflow
        // All `as` casts here are safe: len-1 < 2^53 (Vec bound); pos in [0, len-1] so
        // floor/ceil fit usize; lower <= len-1 so lower fits f64 exactly.
        #[allow(
            clippy::as_conversions,
            clippy::arithmetic_side_effects,
            reason = "len-1 safe: vals non-empty (is_empty check); len-1→f64 exact (Vec << 2^53); \
                      floor/ceil→usize: pos in [0,len-1] fits usize; lower→f64 exact (lower <= len-1 << 2^53)"
        )]
        let pos = q * (vals.len() - 1) as f64;
        #[allow(
            clippy::as_conversions,
            reason = "f64→usize: pos.floor()/ceil() are in [0, vals.len()-1] which fits usize on all platforms"
        )]
        let lower = pos.floor() as usize;
        #[allow(
            clippy::as_conversions,
            reason = "f64→usize: pos.ceil() is in [0, vals.len()-1] which fits usize on all platforms"
        )]
        let upper = pos.ceil() as usize;
        #[allow(
            clippy::indexing_slicing,
            reason = "lower and upper are floor/ceil of q*(len-1) in [0,len-1]; both are valid indices into vals"
        )]
        if lower == upper {
            Ok(Scalar::Float64(vals[lower]))
        } else {
            #[allow(
                clippy::as_conversions,
                reason = "lower→f64: lower <= len-1 << 2^53, fits exactly"
            )]
            let frac = pos - lower as f64;
            Ok(Scalar::Float64(
                vals[lower] * (1.0 - frac) + vals[upper] * frac,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_i64() {
        let c = Column::from_i64s("x", vec![1, 2, 3]);
        assert_eq!(c.sum(), Scalar::Int64(6));
    }

    #[test]
    fn sum_f64() {
        let c = Column::from_f64s("x", vec![1.0, 2.5, 3.5]);
        assert_eq!(c.sum(), Scalar::Float64(7.0));
    }

    #[test]
    fn sum_with_nulls() {
        let c = Column::new_i64("x", vec![Some(10), None, Some(20)]);
        assert_eq!(c.sum(), Scalar::Int64(30));
    }

    #[test]
    fn sum_string_returns_null() {
        let c = Column::from_strs("x", &["a", "b"]);
        assert_eq!(c.sum(), Scalar::Null);
    }

    #[test]
    fn mean_i64() {
        let c = Column::from_i64s("x", vec![2, 4, 6]);
        assert_eq!(c.mean(), Scalar::Float64(4.0));
    }

    #[test]
    fn mean_empty() {
        let c = Column::new_i64("x", vec![]);
        assert_eq!(c.mean(), Scalar::Null);
    }

    #[test]
    fn min_max() {
        let c = Column::from_i64s("x", vec![3, 1, 4, 1, 5]);
        assert_eq!(c.min(), Scalar::Int64(1));
        assert_eq!(c.max(), Scalar::Int64(5));
    }

    #[test]
    fn min_max_with_nulls() {
        let c = Column::new_i64("x", vec![Some(3), None, Some(1)]);
        assert_eq!(c.min(), Scalar::Int64(1));
        assert_eq!(c.max(), Scalar::Int64(3));
    }

    #[test]
    fn median_odd() {
        let c = Column::from_i64s("x", vec![3, 1, 2]);
        assert_eq!(c.median(), Scalar::Float64(2.0));
    }

    #[test]
    fn median_even() {
        let c = Column::from_i64s("x", vec![1, 2, 3, 4]);
        assert_eq!(c.median(), Scalar::Float64(2.5));
    }

    #[test]
    fn std_dev_basic() {
        let c = Column::from_f64s("x", vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let sd = match c.std_dev() {
            Scalar::Float64(v) => v,
            _ => f64::NAN,
        };
        assert!((sd - 2.0).abs() < 0.01);
    }

    #[test]
    fn n_unique_basic() {
        let c = Column::from_strs("x", &["a", "b", "a", "c"]);
        assert_eq!(c.n_unique(), 3);
    }

    #[test]
    fn first_last() {
        let c = Column::new_i64("x", vec![None, Some(10), Some(20), None]);
        assert_eq!(c.first(), Scalar::Int64(10));
        assert_eq!(c.last(), Scalar::Int64(20));
    }

    #[test]
    fn quantile_basic() {
        let c = Column::from_i64s("x", vec![1, 2, 3, 4, 5]);
        let q50 = c.quantile(0.5);
        assert_eq!(q50.ok(), Some(Scalar::Float64(3.0)));
        let q0 = c.quantile(0.0);
        assert_eq!(q0.ok(), Some(Scalar::Float64(1.0)));
        let q100 = c.quantile(1.0);
        assert_eq!(q100.ok(), Some(Scalar::Float64(5.0)));
    }

    #[test]
    fn quantile_invalid() {
        let c = Column::from_i64s("x", vec![1, 2, 3]);
        assert!(c.quantile(1.5).is_err());
        assert!(c.quantile(-0.1).is_err());
    }
}
