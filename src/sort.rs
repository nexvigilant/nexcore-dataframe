//! Sorting operations on DataFrames.

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::error::DataFrameError;
use crate::scalar::Scalar;

impl DataFrame {
    /// Sort by a single column. Nulls sort last.
    pub fn sort(&self, by: &str, descending: bool) -> Result<Self, DataFrameError> {
        let col = self.column(by)?;
        let mut indices: Vec<usize> = (0..self.height()).collect();

        indices.sort_by(|&a, &b| {
            let va = col.get(a).unwrap_or(Scalar::Null);
            let vb = col.get(b).unwrap_or(Scalar::Null);
            let ord = va.compare(&vb);
            if descending { ord.reverse() } else { ord }
        });

        let columns: Vec<Column> = self.columns().iter().map(|c| c.take(&indices)).collect();
        Ok(Self::from_columns_unchecked(columns))
    }

    /// Take the first n rows.
    #[must_use]
    pub fn head(&self, n: usize) -> Self {
        let take = n.min(self.height());
        let indices: Vec<usize> = (0..take).collect();
        let columns: Vec<Column> = self.columns().iter().map(|c| c.take(&indices)).collect();
        Self::from_columns_unchecked(columns)
    }

    /// Take the last n rows.
    #[must_use]
    pub fn tail(&self, n: usize) -> Self {
        let take = n.min(self.height());
        let start = self.height().saturating_sub(take);
        let indices: Vec<usize> = (start..self.height()).collect();
        let columns: Vec<Column> = self.columns().iter().map(|c| c.take(&indices)).collect();
        Self::from_columns_unchecked(columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_ascending() {
        let df = DataFrame::new(vec![
            Column::from_strs("name", &["c", "a", "b"]),
            Column::from_i64s("val", vec![3, 1, 2]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let sorted = df.sort("val", false).unwrap_or_else(|_| unreachable!());
        assert_eq!(
            sorted
                .column("name")
                .unwrap_or_else(|_| unreachable!())
                .get(0),
            Some(Scalar::String("a".into()))
        );
        assert_eq!(
            sorted
                .column("name")
                .unwrap_or_else(|_| unreachable!())
                .get(2),
            Some(Scalar::String("c".into()))
        );
    }

    #[test]
    fn sort_descending() {
        let df = DataFrame::new(vec![Column::from_i64s("x", vec![1, 3, 2])])
            .unwrap_or_else(|_| unreachable!());
        let sorted = df.sort("x", true).unwrap_or_else(|_| unreachable!());
        assert_eq!(
            sorted.column("x").unwrap_or_else(|_| unreachable!()).get(0),
            Some(Scalar::Int64(3))
        );
        assert_eq!(
            sorted.column("x").unwrap_or_else(|_| unreachable!()).get(2),
            Some(Scalar::Int64(1))
        );
    }

    #[test]
    fn sort_with_nulls() {
        let df = DataFrame::new(vec![Column::new_i64("x", vec![Some(2), None, Some(1)])])
            .unwrap_or_else(|_| unreachable!());
        let sorted = df.sort("x", false).unwrap_or_else(|_| unreachable!());
        assert_eq!(
            sorted.column("x").unwrap_or_else(|_| unreachable!()).get(0),
            Some(Scalar::Int64(1))
        );
        assert_eq!(
            sorted.column("x").unwrap_or_else(|_| unreachable!()).get(1),
            Some(Scalar::Int64(2))
        );
        assert_eq!(
            sorted.column("x").unwrap_or_else(|_| unreachable!()).get(2),
            Some(Scalar::Null)
        );
    }

    #[test]
    fn head_and_tail() {
        let df = DataFrame::new(vec![Column::from_i64s("x", vec![1, 2, 3, 4, 5])])
            .unwrap_or_else(|_| unreachable!());
        let h = df.head(3);
        assert_eq!(h.height(), 3);
        assert_eq!(
            h.column("x").unwrap_or_else(|_| unreachable!()).get(0),
            Some(Scalar::Int64(1))
        );

        let t = df.tail(2);
        assert_eq!(t.height(), 2);
        assert_eq!(
            t.column("x").unwrap_or_else(|_| unreachable!()).get(0),
            Some(Scalar::Int64(4))
        );
    }

    #[test]
    fn head_exceeds_length() {
        let df = DataFrame::new(vec![Column::from_i64s("x", vec![1, 2])])
            .unwrap_or_else(|_| unreachable!());
        let h = df.head(100);
        assert_eq!(h.height(), 2);
    }
}
