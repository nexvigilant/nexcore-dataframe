//! Filtering operations on DataFrames.

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::error::DataFrameError;
use crate::scalar::Scalar;

impl DataFrame {
    /// Filter rows by a boolean mask. Mask length must equal height.
    pub fn filter(&self, mask: &[bool]) -> Result<Self, DataFrameError> {
        if mask.len() != self.height() {
            return Err(DataFrameError::LengthMismatch {
                expected: self.height(),
                actual: mask.len(),
            });
        }
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|&(_, c)| *c)
            .map(|(i, _)| i)
            .collect();
        let columns: Vec<Column> = self.columns().iter().map(|c| c.take(&indices)).collect();
        Ok(Self::from_columns_unchecked(columns))
    }

    /// Filter rows by a predicate applied to a named column.
    /// Each row's value is extracted as a Scalar and passed to the predicate.
    pub fn filter_by(
        &self,
        column: &str,
        predicate: impl Fn(&Scalar) -> bool,
    ) -> Result<Self, DataFrameError> {
        let col = self.column(column)?;
        let mask: Vec<bool> = (0..col.len())
            .map(|i| {
                let val = col.get(i).unwrap_or(Scalar::Null);
                predicate(&val)
            })
            .collect();
        self.filter(&mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_by_mask() {
        let df = DataFrame::new(vec![
            Column::from_strs("name", &["a", "b", "c"]),
            Column::from_i64s("val", vec![1, 2, 3]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let filtered = df
            .filter(&[true, false, true])
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(filtered.height(), 2);
        assert_eq!(
            filtered
                .column("name")
                .unwrap_or_else(|_| unreachable!())
                .get(0),
            Some(Scalar::String("a".to_string()))
        );
        assert_eq!(
            filtered
                .column("name")
                .unwrap_or_else(|_| unreachable!())
                .get(1),
            Some(Scalar::String("c".to_string()))
        );
    }

    #[test]
    fn filter_mask_length_mismatch() {
        let df = DataFrame::new(vec![Column::from_i64s("x", vec![1, 2])])
            .unwrap_or_else(|_| unreachable!());
        assert!(df.filter(&[true]).is_err());
    }

    #[test]
    fn filter_by_predicate() {
        let df = DataFrame::new(vec![
            Column::from_strs("drug", &["ASP", "MET", "IBU"]),
            Column::from_i64s("n", vec![10, 2, 5]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let filtered = df
            .filter_by("n", |v| v.as_i64().is_some_and(|n| n >= 5))
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(filtered.height(), 2);
    }

    #[test]
    fn filter_all_false() {
        let df = DataFrame::new(vec![Column::from_i64s("x", vec![1, 2, 3])])
            .unwrap_or_else(|_| unreachable!());
        let filtered = df
            .filter(&[false, false, false])
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(filtered.height(), 0);
    }

    #[test]
    fn filter_by_column_not_found() {
        let df = DataFrame::new(vec![Column::from_i64s("x", vec![1])])
            .unwrap_or_else(|_| unreachable!());
        assert!(df.filter_by("missing", |_| true).is_err());
    }

    #[test]
    fn filter_by_neq_string() {
        let df = DataFrame::new(vec![
            Column::from_strs("direction", &["forward", "backward", "forward"]),
            Column::from_i64s("score", vec![10, 5, 15]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let filtered = df
            .filter_by("direction", |v| v.as_str() != Some("backward"))
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(filtered.height(), 2);
    }
}
