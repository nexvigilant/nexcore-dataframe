//! Join: hash-based join operations on DataFrames.
//!
//! Supports 6 join types: Inner, Left, Right, Outer, Semi, Anti.
//! Algorithm: hash join — build index on right table, probe from left.
//!
//! Primitive composition: μ(Mapping) + κ(Comparison) + ∂(Boundary) + ς(State)

// HashMap is essential for O(1) join-key lookup during the probe phase.
// Output row order follows left-table order for deterministic results.
#[allow(
    clippy::disallowed_types,
    reason = "HashMap needed for O(1) hash-join probe; output follows left-table order, which is deterministic"
)]
use std::collections::HashMap;

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::error::DataFrameError;
use crate::scalar::Scalar;

/// Null sentinel for key representation. Uses NUL bytes to avoid collision
/// with any legitimate string value (improvement over GroupBy's "null" literal).
const NULL_SENTINEL: &str = "\0NULL\0";

/// Join type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JoinType {
    /// Keep only rows that match in both tables.
    Inner,
    /// Keep all left rows; fill right with nulls where no match.
    Left,
    /// Keep all right rows; fill left with nulls where no match.
    Right,
    /// Keep all rows from both tables; fill with nulls where no match.
    Outer,
    /// Keep left rows that have at least one match in right (no right columns added).
    Semi,
    /// Keep left rows that have NO match in right (no right columns added).
    Anti,
}

impl DataFrame {
    /// Join two DataFrames on shared key column names.
    ///
    /// Equivalent to `self.join_on(other, on, on, how)` — both tables use
    /// the same column names as join keys.
    pub fn join(
        &self,
        other: &DataFrame,
        on: &[&str],
        how: JoinType,
    ) -> Result<DataFrame, DataFrameError> {
        self.join_on(other, on, on, how)
    }

    /// Join two DataFrames with potentially different key column names.
    ///
    /// `left_on` columns from `self`, `right_on` columns from `other`.
    /// Key columns must have the same count. Column name collisions in non-key
    /// columns are resolved with `_left` / `_right` suffixes.
    pub fn join_on(
        &self,
        other: &DataFrame,
        left_on: &[&str],
        right_on: &[&str],
        how: JoinType,
    ) -> Result<DataFrame, DataFrameError> {
        // Validate key counts match
        if left_on.len() != right_on.len() {
            return Err(DataFrameError::JoinKeyMismatch {
                left_count: left_on.len(),
                right_count: right_on.len(),
            });
        }
        if left_on.is_empty() {
            return Err(DataFrameError::Other(
                "join requires at least one key column".to_string(),
            ));
        }

        // Validate key columns exist
        for name in left_on {
            self.column(name)?;
        }
        for name in right_on {
            other.column(name)?;
        }

        // Build hash index on RIGHT table: key → Vec<row_index>
        #[allow(
            clippy::disallowed_types,
            reason = "HashMap for O(1) hash-join index; see module-level allow"
        )]
        let mut right_index: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        for row_idx in 0..other.height() {
            let key = extract_key(other, right_on, row_idx);
            // Null keys never match (SQL standard) — skip indexing them
            if key.iter().any(|k| k == NULL_SENTINEL) {
                continue;
            }
            right_index.entry(key).or_default().push(row_idx);
        }

        // Probe from LEFT table
        match how {
            JoinType::Semi => self.join_semi(left_on, &right_index),
            JoinType::Anti => self.join_anti(left_on, &right_index),
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Outer => {
                self.join_matching(other, left_on, right_on, how, &right_index)
            }
        }
    }

    /// Inner, Left, Right, Outer joins — produce combined columns.
    #[allow(
        clippy::disallowed_types,
        reason = "HashMap parameter from join_on caller"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "Internal method — 6 args needed for left/right context + index; not part of public API"
    )]
    fn join_matching(
        &self,
        other: &DataFrame,
        left_on: &[&str],
        right_on: &[&str],
        how: JoinType,
        right_index: &HashMap<Vec<String>, Vec<usize>>,
    ) -> Result<DataFrame, DataFrameError> {
        let mut left_indices: Vec<Option<usize>> = Vec::new();
        let mut right_indices: Vec<Option<usize>> = Vec::new();

        // Track which right rows were matched (for Right/Outer joins)
        let mut right_matched = vec![false; other.height()];

        // Probe left table against right index
        for left_row in 0..self.height() {
            let key = extract_key(self, left_on, left_row);

            // Null keys never match — but Left/Outer preserves the left row
            if key.iter().any(|k| k == NULL_SENTINEL) {
                match how {
                    JoinType::Left | JoinType::Outer => {
                        left_indices.push(Some(left_row));
                        right_indices.push(None);
                    }
                    JoinType::Inner | JoinType::Right | JoinType::Semi | JoinType::Anti => {}
                }
                continue;
            }

            match right_index.get(&key) {
                Some(matches) => {
                    for &right_row in matches {
                        left_indices.push(Some(left_row));
                        right_indices.push(Some(right_row));
                        // right_row < other.height() by construction (built from 0..other.height())
                        #[allow(
                            clippy::indexing_slicing,
                            reason = "right_row is from right_index which was built from 0..other.height()"
                        )]
                        {
                            right_matched[right_row] = true;
                        }
                    }
                }
                None => match how {
                    JoinType::Left | JoinType::Outer => {
                        left_indices.push(Some(left_row));
                        right_indices.push(None);
                    }
                    JoinType::Inner | JoinType::Right | JoinType::Semi | JoinType::Anti => {}
                },
            }
        }

        // Append unmatched right rows for Right/Outer joins
        if matches!(how, JoinType::Right | JoinType::Outer) {
            for (right_row, matched) in right_matched.iter().enumerate() {
                if !matched {
                    left_indices.push(None);
                    right_indices.push(Some(right_row));
                }
            }
        }

        // Assemble result columns
        assemble_columns(
            self,
            other,
            left_on,
            right_on,
            &left_indices,
            &right_indices,
        )
    }

    /// Semi join: keep left rows that have at least one match in right.
    #[allow(
        clippy::disallowed_types,
        reason = "HashMap parameter from join_on caller"
    )]
    fn join_semi(
        &self,
        left_on: &[&str],
        right_index: &HashMap<Vec<String>, Vec<usize>>,
    ) -> Result<DataFrame, DataFrameError> {
        let mut keep_indices: Vec<usize> = Vec::new();

        for left_row in 0..self.height() {
            let key = extract_key(self, left_on, left_row);
            if key.iter().any(|k| k == NULL_SENTINEL) {
                continue;
            }
            if right_index.contains_key(&key) {
                keep_indices.push(left_row);
            }
        }

        Ok(DataFrame::from_columns_unchecked(
            self.columns()
                .iter()
                .map(|c| c.take(&keep_indices))
                .collect(),
        ))
    }

    /// Anti join: keep left rows that have NO match in right.
    #[allow(
        clippy::disallowed_types,
        reason = "HashMap parameter from join_on caller"
    )]
    fn join_anti(
        &self,
        left_on: &[&str],
        right_index: &HashMap<Vec<String>, Vec<usize>>,
    ) -> Result<DataFrame, DataFrameError> {
        let mut keep_indices: Vec<usize> = Vec::new();

        for left_row in 0..self.height() {
            let key = extract_key(self, left_on, left_row);
            // Null keys never match → they are always "unmatched" → kept in anti-join
            if key.iter().any(|k| k == NULL_SENTINEL) {
                keep_indices.push(left_row);
                continue;
            }
            if !right_index.contains_key(&key) {
                keep_indices.push(left_row);
            }
        }

        Ok(DataFrame::from_columns_unchecked(
            self.columns()
                .iter()
                .map(|c| c.take(&keep_indices))
                .collect(),
        ))
    }
}

/// Extract the join key for a row as a Vec<String>.
/// Null values become the NULL_SENTINEL.
fn extract_key(df: &DataFrame, key_cols: &[&str], row_idx: usize) -> Vec<String> {
    key_cols
        .iter()
        .map(|name| {
            df.column(name)
                .ok()
                .and_then(|col| col.get(row_idx))
                .map_or_else(
                    || NULL_SENTINEL.to_string(),
                    |s| {
                        if s.is_null() {
                            NULL_SENTINEL.to_string()
                        } else {
                            s.to_string()
                        }
                    },
                )
        })
        .collect()
}

/// Assemble result columns from left/right index pairs.
/// Key columns come from whichever side has a value (left preferred).
/// Non-key columns are included from both sides with collision suffixes.
#[allow(
    clippy::too_many_arguments,
    reason = "Assembling columns requires both table refs, both key slices, and both index vecs — cannot reduce without wrapper struct"
)]
fn assemble_columns(
    left: &DataFrame,
    right: &DataFrame,
    left_on: &[&str],
    right_on: &[&str],
    left_indices: &[Option<usize>],
    right_indices: &[Option<usize>],
) -> Result<DataFrame, DataFrameError> {
    let mut result_columns: Vec<Column> = Vec::new();
    let left_names: Vec<&str> = left.column_names();
    let right_names: Vec<&str> = right.column_names();

    // Build sets for fast lookup
    let left_key_set: Vec<&str> = left_on.to_vec();
    let right_key_set: Vec<&str> = right_on.to_vec();

    // Determine name collisions between non-key columns
    let left_non_keys: Vec<&str> = left_names
        .iter()
        .filter(|n| !left_key_set.contains(n))
        .copied()
        .collect();
    let right_non_keys: Vec<&str> = right_names
        .iter()
        .filter(|n| !right_key_set.contains(n))
        .copied()
        .collect();

    let collisions: Vec<&str> = left_non_keys
        .iter()
        .filter(|n| right_non_keys.contains(n))
        .copied()
        .collect();

    // 1. Key columns — take from left where available, else from right
    for (i, &left_key_name) in left_on.iter().enumerate() {
        let left_col = left.column(left_key_name)?;
        // i < right_on.len() guaranteed by key count validation in join_on
        #[allow(
            clippy::indexing_slicing,
            reason = "i < left_on.len() == right_on.len() by JoinKeyMismatch validation in join_on"
        )]
        let right_key_name = right_on[i];
        let right_col = right.column(right_key_name)?;

        let merged = merge_key_column(left_col, right_col, left_indices, right_indices);
        result_columns.push(merged.rename(left_key_name));
    }

    // 2. Left non-key columns
    for &name in &left_non_keys {
        let col = left.column(name)?;
        let taken = col.take_optional(left_indices);
        if collisions.contains(&name) {
            result_columns.push(taken.rename(format!("{name}_left")));
        } else {
            result_columns.push(taken);
        }
    }

    // 3. Right non-key columns
    for &name in &right_non_keys {
        let col = right.column(name)?;
        let taken = col.take_optional(right_indices);
        if collisions.contains(&name) {
            result_columns.push(taken.rename(format!("{name}_right")));
        } else {
            result_columns.push(taken);
        }
    }

    Ok(DataFrame::from_columns_unchecked(result_columns))
}

/// Merge a key column from left and right: prefer left value, fall back to right.
fn merge_key_column(
    left_col: &Column,
    right_col: &Column,
    left_indices: &[Option<usize>],
    right_indices: &[Option<usize>],
) -> Column {
    // Key columns are always string-representable for hashing, but we want to
    // preserve the original type. Use the left column's type as canonical.
    let data = left_col.data();

    // Build element-by-element: prefer left value, fall back to right.
    let len = left_indices.len();
    match data {
        crate::column::ColumnData::Bool(_) => {
            let vals: Vec<Option<bool>> = (0..len)
                .map(|i| {
                    // i < left_indices.len() == right_indices.len() by zip construction
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "i iterates 0..len where len = left_indices.len()"
                    )]
                    match (left_indices[i], right_indices[i]) {
                        (Some(li), _) => left_col.get(li).and_then(|s| s.as_bool()),
                        (None, Some(ri)) => right_col.get(ri).and_then(|s| s.as_bool()),
                        (None, None) => None,
                    }
                })
                .collect();
            Column::new_bool(left_col.name(), vals)
        }
        crate::column::ColumnData::Int64(_) => {
            let vals: Vec<Option<i64>> = (0..len)
                .map(|i| {
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "i iterates 0..len where len = left_indices.len()"
                    )]
                    match (left_indices[i], right_indices[i]) {
                        (Some(li), _) => left_col.get(li).and_then(|s| s.as_i64()),
                        (None, Some(ri)) => right_col.get(ri).and_then(|s| s.as_i64()),
                        (None, None) => None,
                    }
                })
                .collect();
            Column::new_i64(left_col.name(), vals)
        }
        crate::column::ColumnData::UInt64(_) => {
            let vals: Vec<Option<u64>> = (0..len)
                .map(|i| {
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "i iterates 0..len where len = left_indices.len()"
                    )]
                    match (left_indices[i], right_indices[i]) {
                        (Some(li), _) => left_col.get(li).and_then(|s| s.as_u64()),
                        (None, Some(ri)) => right_col.get(ri).and_then(|s| s.as_u64()),
                        (None, None) => None,
                    }
                })
                .collect();
            Column::new_u64(left_col.name(), vals)
        }
        crate::column::ColumnData::Float64(_) => {
            let vals: Vec<Option<f64>> = (0..len)
                .map(|i| {
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "i iterates 0..len where len = left_indices.len()"
                    )]
                    match (left_indices[i], right_indices[i]) {
                        (Some(li), _) => left_col.get(li).and_then(|s| s.as_f64()),
                        (None, Some(ri)) => right_col.get(ri).and_then(|s| s.as_f64()),
                        (None, None) => None,
                    }
                })
                .collect();
            Column::new_f64(left_col.name(), vals)
        }
        crate::column::ColumnData::String(_) => {
            let vals: Vec<Option<String>> = (0..len)
                .map(|i| {
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "i iterates 0..len where len = left_indices.len()"
                    )]
                    match (left_indices[i], right_indices[i]) {
                        (Some(li), _) => left_col
                            .get(li)
                            .and_then(|s| s.as_str().map(|s| s.to_string())),
                        (None, Some(ri)) => right_col
                            .get(ri)
                            .and_then(|s| s.as_str().map(|s| s.to_string())),
                        (None, None) => None,
                    }
                })
                .collect();
            Column::new_string(left_col.name(), vals)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn drugs() -> DataFrame {
        DataFrame::new(vec![
            Column::from_strs("drug_id", &["D1", "D2", "D3", "D4"]),
            Column::from_strs(
                "drug_name",
                &["aspirin", "metformin", "ibuprofen", "lisinopril"],
            ),
        ])
        .unwrap_or_else(|_| unreachable!())
    }

    fn events() -> DataFrame {
        DataFrame::new(vec![
            Column::from_strs("drug_id", &["D1", "D1", "D2", "D5"]),
            Column::from_strs("event", &["headache", "nausea", "rash", "dizziness"]),
            Column::from_i64s("count", vec![10, 5, 3, 7]),
        ])
        .unwrap_or_else(|_| unreachable!())
    }

    // =========================================================================
    // Inner join
    // =========================================================================

    #[test]
    fn inner_join_basic() {
        let result = drugs()
            .join(&events(), &["drug_id"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        // D1 matches 2 events, D2 matches 1 = 3 rows
        assert_eq!(result.height(), 3);
        // drug_id + drug_name + event + count = 4 columns
        assert_eq!(result.width(), 4);
    }

    #[test]
    fn inner_join_no_matches() {
        let left = DataFrame::new(vec![
            Column::from_strs("k", &["a", "b"]),
            Column::from_i64s("v", vec![1, 2]),
        ])
        .unwrap_or_else(|_| unreachable!());
        let right = DataFrame::new(vec![
            Column::from_strs("k", &["c", "d"]),
            Column::from_i64s("w", vec![3, 4]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join(&right, &["k"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(result.height(), 0);
    }

    // =========================================================================
    // Left join
    // =========================================================================

    #[test]
    fn left_join_basic() {
        let result = drugs()
            .join(&events(), &["drug_id"], JoinType::Left)
            .unwrap_or_else(|_| unreachable!());
        // D1→2, D2→1, D3→1(null right), D4→1(null right) = 5 rows
        assert_eq!(result.height(), 5);
        assert_eq!(result.width(), 4);

        // Verify D3 has null event
        let mut found_d3 = false;
        for i in 0..result.height() {
            let id = result
                .column("drug_id")
                .unwrap_or_else(|_| unreachable!())
                .get(i);
            if id == Some(Scalar::String("D3".into())) {
                found_d3 = true;
                assert_eq!(
                    result
                        .column("event")
                        .unwrap_or_else(|_| unreachable!())
                        .get(i),
                    Some(Scalar::Null)
                );
            }
        }
        assert!(found_d3);
    }

    // =========================================================================
    // Right join
    // =========================================================================

    #[test]
    fn right_join_basic() {
        let result = drugs()
            .join(&events(), &["drug_id"], JoinType::Right)
            .unwrap_or_else(|_| unreachable!());
        // D1→2, D2→1, D5→1(null left) = 4 rows
        assert_eq!(result.height(), 4);

        // Verify D5 has null drug_name
        let mut found_d5 = false;
        for i in 0..result.height() {
            let id = result
                .column("drug_id")
                .unwrap_or_else(|_| unreachable!())
                .get(i);
            if id == Some(Scalar::String("D5".into())) {
                found_d5 = true;
                assert_eq!(
                    result
                        .column("drug_name")
                        .unwrap_or_else(|_| unreachable!())
                        .get(i),
                    Some(Scalar::Null)
                );
            }
        }
        assert!(found_d5);
    }

    // =========================================================================
    // Outer join
    // =========================================================================

    #[test]
    fn outer_join_basic() {
        let result = drugs()
            .join(&events(), &["drug_id"], JoinType::Outer)
            .unwrap_or_else(|_| unreachable!());
        // D1→2, D2→1, D3→1(null right), D4→1(null right), D5→1(null left) = 6
        assert_eq!(result.height(), 6);
        assert_eq!(result.width(), 4);
    }

    // =========================================================================
    // Semi join
    // =========================================================================

    #[test]
    fn semi_join_basic() {
        let result = drugs()
            .join(&events(), &["drug_id"], JoinType::Semi)
            .unwrap_or_else(|_| unreachable!());
        // D1 and D2 have matches → 2 rows
        assert_eq!(result.height(), 2);
        // Only left columns
        assert_eq!(result.width(), 2);
    }

    // =========================================================================
    // Anti join
    // =========================================================================

    #[test]
    fn anti_join_basic() {
        let result = drugs()
            .join(&events(), &["drug_id"], JoinType::Anti)
            .unwrap_or_else(|_| unreachable!());
        // D3 and D4 have no matches → 2 rows
        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 2);
    }

    // =========================================================================
    // Multi-key join
    // =========================================================================

    #[test]
    fn multi_key_join() {
        let left = DataFrame::new(vec![
            Column::from_strs("drug", &["asp", "asp", "met"]),
            Column::from_strs("event", &["ha", "na", "ha"]),
            Column::from_i64s("left_val", vec![1, 2, 3]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_strs("drug", &["asp", "met", "met"]),
            Column::from_strs("event", &["ha", "ha", "na"]),
            Column::from_i64s("right_val", vec![10, 20, 30]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join(&right, &["drug", "event"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        // asp+ha → 1 match, met+ha → 1 match = 2 rows
        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 4); // drug + event + left_val + right_val
    }

    // =========================================================================
    // Asymmetric keys (join_on)
    // =========================================================================

    #[test]
    fn join_on_different_key_names() {
        let left = DataFrame::new(vec![
            Column::from_strs("id", &["a", "b", "c"]),
            Column::from_i64s("val", vec![1, 2, 3]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_strs("key", &["b", "c", "d"]),
            Column::from_i64s("score", vec![10, 20, 30]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join_on(&right, &["id"], &["key"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(result.height(), 2); // b, c
        assert_eq!(result.width(), 3); // id + val + score
        // Key column uses left name "id"
        assert!(result.column("id").is_ok());
    }

    // =========================================================================
    // Null key handling
    // =========================================================================

    #[test]
    fn null_keys_never_match() {
        let left = DataFrame::new(vec![
            Column::new_string("k", vec![Some("a".into()), None, Some("c".into())]),
            Column::from_i64s("v", vec![1, 2, 3]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::new_string("k", vec![Some("a".into()), None]),
            Column::from_i64s("w", vec![10, 20]),
        ])
        .unwrap_or_else(|_| unreachable!());

        // Inner: null keys don't match → only "a" matches
        let inner = left
            .join(&right, &["k"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(inner.height(), 1);

        // Left: null-keyed left row preserved with null right
        let lj = left
            .join(&right, &["k"], JoinType::Left)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(lj.height(), 3); // a→1, null→1(null right), c→1(null right)
    }

    #[test]
    fn null_keys_kept_in_anti_join() {
        let left = DataFrame::new(vec![
            Column::new_string("k", vec![Some("a".into()), None, Some("c".into())]),
            Column::from_i64s("v", vec![1, 2, 3]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_strs("k", &["a"]),
            Column::from_i64s("w", vec![10]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join(&right, &["k"], JoinType::Anti)
            .unwrap_or_else(|_| unreachable!());
        // null and "c" don't match → 2 rows
        assert_eq!(result.height(), 2);
    }

    // =========================================================================
    // Name collision handling
    // =========================================================================

    #[test]
    fn name_collision_suffixes() {
        let left = DataFrame::new(vec![
            Column::from_strs("k", &["a", "b"]),
            Column::from_i64s("value", vec![1, 2]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_strs("k", &["a", "b"]),
            Column::from_i64s("value", vec![10, 20]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join(&right, &["k"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(result.height(), 2);
        assert!(result.column("value_left").is_ok());
        assert!(result.column("value_right").is_ok());
    }

    // =========================================================================
    // Empty DataFrames
    // =========================================================================

    #[test]
    fn join_empty_left() {
        let left = DataFrame::new(vec![
            Column::from_strs("k", &[]),
            Column::from_i64s("v", vec![]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_strs("k", &["a"]),
            Column::from_i64s("w", vec![1]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join(&right, &["k"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(result.height(), 0);

        let result = left
            .join(&right, &["k"], JoinType::Left)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(result.height(), 0);
    }

    #[test]
    fn join_empty_right() {
        let left = DataFrame::new(vec![
            Column::from_strs("k", &["a"]),
            Column::from_i64s("v", vec![1]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_strs("k", &[]),
            Column::from_i64s("w", vec![]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let inner = left
            .join(&right, &["k"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(inner.height(), 0);

        let lj = left
            .join(&right, &["k"], JoinType::Left)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(lj.height(), 1); // left row preserved
    }

    // =========================================================================
    // Error cases
    // =========================================================================

    #[test]
    fn error_key_mismatch() {
        let left = DataFrame::new(vec![
            Column::from_strs("a", &["x"]),
            Column::from_strs("b", &["y"]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right =
            DataFrame::new(vec![Column::from_strs("c", &["x"])]).unwrap_or_else(|_| unreachable!());

        let err = left.join_on(&right, &["a", "b"], &["c"], JoinType::Inner);
        assert!(err.is_err());
    }

    #[test]
    fn error_empty_keys() {
        let left =
            DataFrame::new(vec![Column::from_strs("a", &["x"])]).unwrap_or_else(|_| unreachable!());
        let right =
            DataFrame::new(vec![Column::from_strs("a", &["x"])]).unwrap_or_else(|_| unreachable!());

        let err = left.join(&right, &[], JoinType::Inner);
        assert!(err.is_err());
    }

    #[test]
    fn error_missing_column() {
        let left =
            DataFrame::new(vec![Column::from_strs("a", &["x"])]).unwrap_or_else(|_| unreachable!());
        let right =
            DataFrame::new(vec![Column::from_strs("b", &["x"])]).unwrap_or_else(|_| unreachable!());

        let err = left.join(&right, &["a"], JoinType::Inner);
        assert!(err.is_err()); // "a" not found in right
    }

    // =========================================================================
    // Type preservation
    // =========================================================================

    #[test]
    fn type_preservation_numeric_keys() {
        let left = DataFrame::new(vec![
            Column::from_i64s("id", vec![1, 2, 3]),
            Column::from_strs("name", &["a", "b", "c"]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_i64s("id", vec![2, 3, 4]),
            Column::from_f64s("score", vec![9.5, 8.0, 7.5]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join(&right, &["id"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(result.height(), 2); // id 2, 3

        // Verify key column preserved as Int64
        let id_col = result.column("id").unwrap_or_else(|_| unreachable!());
        assert_eq!(id_col.dtype(), crate::column::DataType::Int64);

        // Verify score preserved as Float64
        let score_col = result.column("score").unwrap_or_else(|_| unreachable!());
        assert_eq!(score_col.dtype(), crate::column::DataType::Float64);
    }

    #[test]
    fn many_to_many_join() {
        let left = DataFrame::new(vec![
            Column::from_strs("k", &["a", "a", "b"]),
            Column::from_i64s("lv", vec![1, 2, 3]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_strs("k", &["a", "a"]),
            Column::from_i64s("rv", vec![10, 20]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join(&right, &["k"], JoinType::Inner)
            .unwrap_or_else(|_| unreachable!());
        // 2 left "a" × 2 right "a" = 4 matches, b→0 = total 4
        assert_eq!(result.height(), 4);
    }

    #[test]
    fn semi_join_deduplicates() {
        // Semi join should produce at most one row per left row, even with many right matches
        let left = DataFrame::new(vec![
            Column::from_strs("k", &["a", "b"]),
            Column::from_i64s("v", vec![1, 2]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_strs("k", &["a", "a", "a"]),
            Column::from_i64s("w", vec![10, 20, 30]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join(&right, &["k"], JoinType::Semi)
            .unwrap_or_else(|_| unreachable!());
        // "a" appears once despite 3 right matches, "b" no match
        assert_eq!(result.height(), 1);
    }

    #[test]
    fn outer_join_preserves_all_keys() {
        let left = DataFrame::new(vec![
            Column::from_strs("k", &["a", "b"]),
            Column::from_i64s("v", vec![1, 2]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right = DataFrame::new(vec![
            Column::from_strs("k", &["b", "c"]),
            Column::from_i64s("w", vec![20, 30]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let result = left
            .join(&right, &["k"], JoinType::Outer)
            .unwrap_or_else(|_| unreachable!());
        // a(left only), b(both), c(right only) = 3
        assert_eq!(result.height(), 3);

        // All key values present
        let keys: Vec<Scalar> = (0..result.height())
            .filter_map(|i| result.column("k").ok().and_then(|c| c.get(i)))
            .collect();
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn right_join_symmetric_to_left() {
        // right_join(A, B) should have same rows as left_join(B, A) (different column order)
        let a = DataFrame::new(vec![
            Column::from_strs("k", &["x", "y"]),
            Column::from_i64s("a_val", vec![1, 2]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let b = DataFrame::new(vec![
            Column::from_strs("k", &["y", "z"]),
            Column::from_i64s("b_val", vec![10, 20]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let right_result = a
            .join(&b, &["k"], JoinType::Right)
            .unwrap_or_else(|_| unreachable!());
        let left_result = b
            .join(&a, &["k"], JoinType::Left)
            .unwrap_or_else(|_| unreachable!());

        assert_eq!(right_result.height(), left_result.height());
    }
}
