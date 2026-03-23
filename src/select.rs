//! Column selection operations on DataFrames.

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::error::DataFrameError;

impl DataFrame {
    /// Select a subset of columns by name. Returns an error if any column is not found.
    pub fn select(&self, names: &[&str]) -> Result<Self, DataFrameError> {
        let mut columns = Vec::with_capacity(names.len());
        for name in names {
            let col = self.column(name)?;
            columns.push(col.clone());
        }
        Ok(Self::from_columns_unchecked(columns))
    }

    /// Drop columns by name. Silently ignores names not found.
    #[must_use]
    pub fn drop_columns(&self, names: &[&str]) -> Self {
        let columns: Vec<Column> = self
            .columns()
            .iter()
            .filter(|c| !names.contains(&c.name()))
            .cloned()
            .collect();
        Self::from_columns_unchecked(columns)
    }

    /// Rename a column. Returns error if the column is not found.
    pub fn rename_column(&self, old: &str, new: &str) -> Result<Self, DataFrameError> {
        // Verify column exists — propagate error if not found
        self.column(old)?;
        let columns: Vec<Column> = self
            .columns()
            .iter()
            .map(|c| {
                if c.name() == old {
                    c.rename(new)
                } else {
                    c.clone()
                }
            })
            .collect();
        Ok(Self::from_columns_unchecked(columns))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Scalar;

    #[test]
    fn select_columns() {
        let df = DataFrame::new(vec![
            Column::from_i64s("a", vec![1, 2]),
            Column::from_i64s("b", vec![3, 4]),
            Column::from_i64s("c", vec![5, 6]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let selected = df.select(&["a", "c"]).unwrap_or_else(|_| unreachable!());
        assert_eq!(selected.width(), 2);
        assert_eq!(selected.column_names(), vec!["a", "c"]);
    }

    #[test]
    fn select_missing_column() {
        let df = DataFrame::new(vec![Column::from_i64s("a", vec![1])])
            .unwrap_or_else(|_| unreachable!());
        assert!(df.select(&["a", "missing"]).is_err());
    }

    #[test]
    fn drop_columns_basic() {
        let df = DataFrame::new(vec![
            Column::from_i64s("a", vec![1]),
            Column::from_i64s("b", vec![2]),
            Column::from_i64s("c", vec![3]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let dropped = df.drop_columns(&["b"]);
        assert_eq!(dropped.width(), 2);
        assert_eq!(dropped.column_names(), vec!["a", "c"]);
    }

    #[test]
    fn drop_columns_nonexistent_ignored() {
        let df = DataFrame::new(vec![Column::from_i64s("a", vec![1])])
            .unwrap_or_else(|_| unreachable!());
        let dropped = df.drop_columns(&["missing"]);
        assert_eq!(dropped.width(), 1);
    }

    #[test]
    fn rename_column_basic() {
        let df = DataFrame::new(vec![
            Column::from_i64s("old_name", vec![1, 2]),
            Column::from_i64s("keep", vec![3, 4]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let renamed = df
            .rename_column("old_name", "new_name")
            .unwrap_or_else(|_| unreachable!());
        assert!(renamed.column("new_name").is_ok());
        assert!(renamed.column("old_name").is_err());
        assert_eq!(
            renamed
                .column("new_name")
                .unwrap_or_else(|_| unreachable!())
                .get(0),
            Some(Scalar::Int64(1))
        );
    }

    #[test]
    fn rename_column_not_found() {
        let df = DataFrame::new(vec![Column::from_i64s("a", vec![1])])
            .unwrap_or_else(|_| unreachable!());
        assert!(df.rename_column("missing", "new").is_err());
    }
}
