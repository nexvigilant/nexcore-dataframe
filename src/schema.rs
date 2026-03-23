//! Schema: column name → DataType mapping.

use crate::DataType;

/// Describes the structure of a DataFrame: ordered list of (name, type) pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    fields: Vec<(String, DataType)>,
}

impl Schema {
    /// Create a new schema from field definitions.
    #[must_use]
    pub fn new(fields: Vec<(String, DataType)>) -> Self {
        Self { fields }
    }

    /// Number of columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the schema has no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Get field names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.fields.iter().map(|(name, _)| name.as_str())
    }

    /// Get the data type for a named column.
    #[must_use]
    pub fn dtype(&self, name: &str) -> Option<DataType> {
        self.fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, dt)| *dt)
    }

    /// Get all fields as slice.
    #[must_use]
    pub fn fields(&self) -> &[(String, DataType)] {
        &self.fields
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_basics() {
        let s = Schema::new(vec![
            ("a".into(), DataType::Int64),
            ("b".into(), DataType::Utf8),
        ]);
        assert_eq!(s.len(), 2);
        assert!(!s.is_empty());
        assert_eq!(s.dtype("a"), Some(DataType::Int64));
        assert_eq!(s.dtype("b"), Some(DataType::Utf8));
        assert_eq!(s.dtype("c"), None);
    }
}
