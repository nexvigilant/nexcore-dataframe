//! JSON I/O for DataFrames.
//!
//! Row-oriented JSON: each row is a JSON object with column names as keys.
//! This matches the format used by polars `JsonWriter` / `JsonReader`.

use std::io::{Read, Write};

use serde_json::Value;

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::error::DataFrameError;
use crate::scalar::Scalar;

impl DataFrame {
    /// Serialize the DataFrame to a JSON string (array of row objects).
    pub fn to_json(&self) -> Result<String, DataFrameError> {
        let rows = self.to_json_rows();
        let val = Value::Array(rows);
        serde_json::to_string_pretty(&val).map_err(DataFrameError::from)
    }

    /// Serialize the DataFrame to a writer.
    pub fn to_json_writer<W: Write>(&self, writer: W) -> Result<(), DataFrameError> {
        let rows = self.to_json_rows();
        let val = Value::Array(rows);
        serde_json::to_writer_pretty(writer, &val).map_err(DataFrameError::from)
    }

    /// Write to a JSON file at the given path.
    pub fn to_json_file(&self, path: &std::path::Path) -> Result<(), DataFrameError> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        self.to_json_writer(writer)
    }

    /// Deserialize a DataFrame from a JSON string (array of row objects).
    pub fn from_json(json: &str) -> Result<Self, DataFrameError> {
        let val: Value = serde_json::from_str(json)?;
        Self::from_json_value(&val)
    }

    /// Deserialize a DataFrame from a reader.
    pub fn from_json_reader<R: Read>(reader: R) -> Result<Self, DataFrameError> {
        let val: Value = serde_json::from_reader(reader)?;
        Self::from_json_value(&val)
    }

    /// Convert DataFrame rows to JSON Value objects.
    fn to_json_rows(&self) -> Vec<Value> {
        let mut rows = Vec::with_capacity(self.height());
        let names = self.column_names();

        for i in 0..self.height() {
            let mut map = serde_json::Map::new();
            for (col_idx, name) in names.iter().enumerate() {
                let val = self.columns().get(col_idx).and_then(|c| c.get(i));
                map.insert((*name).to_string(), scalar_to_json(val));
            }
            rows.push(Value::Object(map));
        }
        rows
    }

    /// Parse a JSON Value (expected array of objects) into a DataFrame.
    fn from_json_value(val: &Value) -> Result<Self, DataFrameError> {
        let arr = match val {
            Value::Array(a) => a,
            Value::Null
            | Value::Bool(_)
            | Value::Number(_)
            | Value::String(_)
            | Value::Object(_) => {
                return Err(DataFrameError::Other(
                    "expected JSON array of objects".to_string(),
                ));
            }
        };

        if arr.is_empty() {
            return Ok(Self::empty());
        }

        // Discover column names from the first object; arr is non-empty (checked above)
        #[allow(
            clippy::indexing_slicing,
            reason = "arr is non-empty (is_empty() guard above); index 0 is always valid"
        )]
        let first = match &arr[0] {
            Value::Object(m) => m,
            Value::Null
            | Value::Bool(_)
            | Value::Number(_)
            | Value::String(_)
            | Value::Array(_) => {
                return Err(DataFrameError::Other(
                    "expected JSON object as array element".to_string(),
                ));
            }
        };

        let col_names: Vec<String> = first.keys().cloned().collect();
        let n_rows = arr.len();

        // Collect raw values per column
        let mut raw_cols: Vec<Vec<Option<&Value>>> = col_names
            .iter()
            .map(|_| Vec::with_capacity(n_rows))
            .collect();

        for row_val in arr {
            let obj = match row_val {
                Value::Object(m) => m,
                Value::Null
                | Value::Bool(_)
                | Value::Number(_)
                | Value::String(_)
                | Value::Array(_) => {
                    return Err(DataFrameError::Other(
                        "expected JSON object as array element".to_string(),
                    ));
                }
            };
            for (col_idx, name) in col_names.iter().enumerate() {
                // col_idx < col_names.len() == raw_cols.len(); always valid
                #[allow(
                    clippy::indexing_slicing,
                    reason = "col_idx iterates 0..col_names.len() which equals raw_cols.len(); index is always in bounds"
                )]
                raw_cols[col_idx].push(obj.get(name));
            }
        }

        // Infer types and build columns
        let columns: Vec<Column> = col_names
            .into_iter()
            .zip(raw_cols)
            .map(|(name, vals)| infer_column(&name, &vals))
            .collect();

        DataFrame::new(columns)
    }
}

/// Convert a Scalar to a JSON Value.
fn scalar_to_json(val: Option<Scalar>) -> Value {
    match val {
        None | Some(Scalar::Null) => Value::Null,
        Some(Scalar::Bool(b)) => Value::Bool(b),
        Some(Scalar::Int64(n)) => Value::Number(n.into()),
        Some(Scalar::UInt64(n)) => Value::Number(n.into()),
        Some(Scalar::Float64(f)) => {
            serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number)
        }
        Some(Scalar::String(s)) => Value::String(s),
    }
}

/// Infer a Column's type from JSON values.
fn infer_column(name: &str, vals: &[Option<&Value>]) -> Column {
    // Find first non-null value to determine type
    let first_non_null = vals.iter().find_map(|v| match v {
        Some(Value::Null) | None => None,
        Some(inner) => Some(*inner),
    });

    match first_non_null {
        Some(Value::Bool(_)) => {
            let data: Vec<Option<bool>> = vals
                .iter()
                .map(|v| match v {
                    Some(Value::Bool(b)) => Some(*b),
                    Some(
                        Value::Null
                        | Value::Number(_)
                        | Value::String(_)
                        | Value::Array(_)
                        | Value::Object(_),
                    )
                    | None => None,
                })
                .collect();
            Column::new_bool(name, data)
        }
        Some(Value::Number(first_num)) => {
            // If any value is fractional, use f64
            let has_float = vals.iter().any(
                |v| matches!(v, Some(Value::Number(num)) if num.is_f64() && num.as_i64().is_none()),
            );
            if has_float {
                let data: Vec<Option<f64>> = vals
                    .iter()
                    .map(|v| match v {
                        Some(Value::Number(num)) => num.as_f64(),
                        Some(
                            Value::Null
                            | Value::Bool(_)
                            | Value::String(_)
                            | Value::Array(_)
                            | Value::Object(_),
                        )
                        | None => None,
                    })
                    .collect();
                Column::new_f64(name, data)
            } else if first_num.is_u64() && first_num.as_i64().is_none() {
                // Pure u64 (exceeds i64 range)
                let data: Vec<Option<u64>> = vals
                    .iter()
                    .map(|v| match v {
                        Some(Value::Number(num)) => num.as_u64(),
                        Some(
                            Value::Null
                            | Value::Bool(_)
                            | Value::String(_)
                            | Value::Array(_)
                            | Value::Object(_),
                        )
                        | None => None,
                    })
                    .collect();
                Column::new_u64(name, data)
            } else {
                let data: Vec<Option<i64>> = vals
                    .iter()
                    .map(|v| match v {
                        Some(Value::Number(num)) => num.as_i64(),
                        Some(
                            Value::Null
                            | Value::Bool(_)
                            | Value::String(_)
                            | Value::Array(_)
                            | Value::Object(_),
                        )
                        | None => None,
                    })
                    .collect();
                Column::new_i64(name, data)
            }
        }
        // String, Array, Object, or no non-null values → fall back to string column
        Some(Value::String(_) | Value::Array(_) | Value::Object(_) | Value::Null) | None => {
            let data: Vec<Option<String>> = vals
                .iter()
                .map(|v| match v {
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(Value::Null) | None => None,
                    Some(other) => Some(other.to_string()),
                })
                .collect();
            Column::new_string(name, data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_json_string() {
        let df = DataFrame::new(vec![
            Column::from_strs("name", &["alice", "bob"]),
            Column::from_i64s("age", vec![30, 25]),
        ])
        .unwrap_or_else(|_| unreachable!());

        let json = df.to_json().unwrap_or_else(|_| unreachable!());
        let df2 = DataFrame::from_json(&json).unwrap_or_else(|_| unreachable!());

        assert_eq!(df2.height(), 2);
        assert_eq!(df2.width(), 2);
    }

    #[test]
    fn from_json_mixed_types() {
        let json = r#"[
            {"x": 1, "y": "hello", "z": true},
            {"x": 2, "y": "world", "z": false}
        ]"#;
        let df = DataFrame::from_json(json).unwrap_or_else(|_| unreachable!());
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 3);
    }

    #[test]
    fn from_json_with_nulls() {
        let json = r#"[
            {"x": 1, "y": "a"},
            {"x": null, "y": "b"},
            {"x": 3, "y": null}
        ]"#;
        let df = DataFrame::from_json(json).unwrap_or_else(|_| unreachable!());
        assert_eq!(df.height(), 3);

        let x = df.column("x").unwrap_or_else(|_| unreachable!());
        assert_eq!(x.get(0), Some(Scalar::Int64(1)));
        assert_eq!(x.get(1), Some(Scalar::Null));
        assert_eq!(x.get(2), Some(Scalar::Int64(3)));
    }

    #[test]
    fn from_json_empty_array() {
        let df = DataFrame::from_json("[]").unwrap_or_else(|_| unreachable!());
        assert!(df.is_empty());
    }

    #[test]
    fn from_json_invalid() {
        assert!(DataFrame::from_json("not json").is_err());
        assert!(DataFrame::from_json("42").is_err());
    }

    #[test]
    fn from_json_floats() {
        let json = r#"[{"val": 1.5}, {"val": 2.5}]"#;
        let df = DataFrame::from_json(json).unwrap_or_else(|_| unreachable!());
        let col = df.column("val").unwrap_or_else(|_| unreachable!());
        assert_eq!(col.get(0), Some(Scalar::Float64(1.5)));
    }

    #[test]
    fn roundtrip_json_file() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| unreachable!());
        let path = dir.path().join("test.json");

        let df = DataFrame::new(vec![
            Column::from_strs("drug", &["asp", "met"]),
            Column::from_i64s("n", vec![100, 200]),
        ])
        .unwrap_or_else(|_| unreachable!());

        df.to_json_file(&path).unwrap_or_else(|_| unreachable!());

        let file = std::fs::File::open(&path).unwrap_or_else(|_| unreachable!());
        let reader = std::io::BufReader::new(file);
        let df2 = DataFrame::from_json_reader(reader).unwrap_or_else(|_| unreachable!());

        assert_eq!(df2.height(), 2);
    }
}
