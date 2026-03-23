//! Counter: optimized hash-based group-count for the FAERS pipeline.
//!
//! Replaces the pattern: `df.lazy().group_by([cols]).agg([col.count()])` with a
//! direct `HashMap<Vec<String>, u64>` accumulation. For 20-50M rows where the
//! only aggregation is count, this avoids building an intermediate DataFrame.

// Counter is purpose-built for O(1) key lookup on 20-50M FAERS rows.
// BTreeMap would increase insert/lookup from O(1) to O(log n) — unacceptable at this scale.
// Iteration order is explicitly unspecified: callers sort the resulting DataFrame if needed.
#[allow(
    clippy::disallowed_types,
    reason = "HashMap required for O(1) amortized insert/lookup at FAERS scale (20-50M rows); BTreeMap O(log n) cost is prohibitive here"
)]
use std::collections::HashMap;

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::error::DataFrameError;

/// Hash-based counter: accumulates `(key_tuple → count)` without building
/// an intermediate DataFrame. Purpose-built for FAERS drug×event counting.
#[derive(Debug, Clone)]
pub struct Counter {
    /// Column names that form the composite key.
    key_names: Vec<String>,
    /// Accumulated counts per unique key combination.
    #[allow(
        clippy::disallowed_types,
        reason = "HashMap required for O(1) amortized insert/lookup at FAERS scale (20-50M rows); BTreeMap O(log n) cost is prohibitive here"
    )]
    counts: HashMap<Vec<String>, u64>,
}

impl Counter {
    /// Create a new counter for the given key columns.
    #[must_use]
    pub fn new(key_names: Vec<String>) -> Self {
        Self {
            key_names,
            #[allow(
                clippy::disallowed_types,
                reason = "HashMap::new() for the counts field; see field-level allow"
            )]
            counts: HashMap::new(),
        }
    }

    /// Increment the count for a key combination.
    pub fn increment(&mut self, key: Vec<String>) {
        // count starts at 0 and is incremented once per call; overflow at u64::MAX (>1.8×10^19)
        // is not a realistic concern for any DataFrame workload
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "u64 counter incremented by 1; realistic row counts are far below u64::MAX"
        )]
        {
            *self.counts.entry(key).or_insert(0) += 1;
        }
    }

    /// Increment by a specific amount.
    pub fn increment_by(&mut self, key: Vec<String>, n: u64) {
        // Same reasoning: accumulating u64 counts from finite data cannot realistically overflow
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "u64 accumulator; sum of row counts bounded by total dataset size which is far below u64::MAX"
        )]
        {
            *self.counts.entry(key).or_insert(0) += n;
        }
    }

    /// Number of unique key combinations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether no counts have been accumulated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Get the count for a specific key combination.
    #[must_use]
    pub fn get(&self, key: &[String]) -> u64 {
        self.counts.get(key).copied().unwrap_or(0)
    }

    /// Iterate over all (key, count) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Vec<String>, &u64)> {
        self.counts.iter()
    }

    /// Total count across all keys.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Convert the counter into a DataFrame with key columns + a "count" column.
    ///
    /// Each unique key combination becomes a row. The last column is always "count".
    pub fn into_dataframe(self) -> Result<DataFrame, DataFrameError> {
        let n_keys = self.key_names.len();
        let n_rows = self.counts.len();

        // Pre-allocate column vecs
        let mut key_vecs: Vec<Vec<Option<String>>> =
            (0..n_keys).map(|_| Vec::with_capacity(n_rows)).collect();
        let mut count_vec: Vec<Option<u64>> = Vec::with_capacity(n_rows);

        // Iteration order of HashMap is unspecified but consistent within a single call;
        // the resulting DataFrame rows are not required to be in any particular order.
        #[allow(
            clippy::iter_over_hash_type,
            reason = "HashMap iteration builds parallel column vecs; output row order is explicitly unspecified — callers sort if order matters"
        )]
        for (key, count) in &self.counts {
            for (i, val) in key.iter().enumerate() {
                if i < n_keys {
                    // i < n_keys <= key_vecs.len() by construction above
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "i is bounded by n_keys = key_vecs.len(); the guard i < n_keys ensures the index is valid"
                    )]
                    key_vecs[i].push(Some(val.clone()));
                }
            }
            count_vec.push(Some(*count));
        }

        let mut columns: Vec<Column> = key_vecs
            .into_iter()
            .enumerate()
            .map(|(i, data)| {
                // i < n_keys = key_names.len() because key_vecs was built with n_keys elements
                #[allow(clippy::indexing_slicing, reason = "i iterates over 0..n_keys which equals key_names.len(); index is always valid")]
                Column::new_string(self.key_names[i].clone(), data)
            })
            .collect();
        columns.push(Column::new_u64("count", count_vec));

        DataFrame::new(columns)
    }

    /// Build a counter from a DataFrame by counting rows grouped by specified columns.
    pub fn from_dataframe(df: &DataFrame, group_cols: &[&str]) -> Result<Self, DataFrameError> {
        // Validate columns exist — propagate error if not found
        for name in group_cols {
            df.column(name)?;
        }

        let key_names: Vec<String> = group_cols.iter().map(|s| (*s).to_string()).collect();
        let mut counter = Self::new(key_names);

        for row_idx in 0..df.height() {
            let key: Vec<String> = group_cols
                .iter()
                .map(|name| {
                    df.column(name)
                        .ok()
                        .and_then(|col| col.get(row_idx))
                        .map_or_else(|| "null".to_string(), |s| s.to_string())
                })
                .collect();
            counter.increment(key);
        }

        Ok(counter)
    }

    /// Filter the counter, keeping only entries where count >= min_count.
    #[must_use]
    pub fn filter_min_count(&self, min_count: u64) -> Self {
        Self {
            key_names: self.key_names.clone(),
            counts: self
                .counts
                .iter()
                .filter(|&(_, &count)| count >= min_count)
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_basic() {
        let mut c = Counter::new(vec!["drug".into(), "event".into()]);
        c.increment(vec!["aspirin".into(), "headache".into()]);
        c.increment(vec!["aspirin".into(), "headache".into()]);
        c.increment(vec!["aspirin".into(), "nausea".into()]);

        assert_eq!(c.len(), 2);
        assert_eq!(c.get(&["aspirin".to_string(), "headache".to_string()]), 2);
        assert_eq!(c.get(&["aspirin".to_string(), "nausea".to_string()]), 1);
        assert_eq!(c.total(), 3);
    }

    #[test]
    fn counter_into_dataframe() {
        let mut c = Counter::new(vec!["drug".into()]);
        c.increment(vec!["asp".into()]);
        c.increment(vec!["asp".into()]);
        c.increment(vec!["met".into()]);

        let df = c.into_dataframe().unwrap_or_else(|_| unreachable!());
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 2); // drug + count
        assert!(df.column("drug").is_ok());
        assert!(df.column("count").is_ok());
    }

    #[test]
    fn counter_from_dataframe() {
        let df = DataFrame::new(vec![
            Column::from_strs("drug", &["asp", "met", "asp", "asp", "met"]),
            Column::from_strs("event", &["ha", "na", "ha", "di", "na"]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let c = Counter::from_dataframe(&df, &["drug", "event"]).unwrap_or_else(|_| unreachable!());
        assert_eq!(c.len(), 3); // asp+ha, met+na, asp+di
        assert_eq!(c.get(&["asp".to_string(), "ha".to_string()]), 2);
        assert_eq!(c.total(), 5);
    }

    #[test]
    fn counter_filter_min_count() {
        let mut c = Counter::new(vec!["x".into()]);
        c.increment(vec!["a".into()]);
        c.increment(vec!["b".into()]);
        c.increment(vec!["b".into()]);
        c.increment(vec!["b".into()]);

        let filtered = c.filter_min_count(2);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.get(&["b".to_string()]), 3);
    }

    #[test]
    fn counter_from_dataframe_missing_column() {
        let df = DataFrame::new(vec![Column::from_i64s("x", vec![1])])
            .unwrap_or_else(|_| unreachable!());
        assert!(Counter::from_dataframe(&df, &["missing"]).is_err());
    }

    #[test]
    fn counter_empty() {
        let c = Counter::new(vec!["k".into()]);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.total(), 0);
    }
}
