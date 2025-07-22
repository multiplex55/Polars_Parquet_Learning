//! Example utilities for working with Parquet files using Polars.
//!
//! The functions here are kept separate from the GUI so they can also be
//! exercised from unit tests.  They demonstrate a very small slice of what
//! Polars offers when dealing with columnar data.

use anyhow::Result;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;

/// Example record used throughout the module.
///
/// In practice your schema may contain many more fields.  The struct is
/// `serde` serializable so we can easily convert between rows and Rust types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub id: i64,
    pub name: String,
}

/// Read a Parquet file into a [`DataFrame`].
///
/// Polars stores data in a *columnar* fashion which is ideal for analytics
/// workloads.  Each column is stored contiguously in memory resulting in
/// efficient vectorised operations.  Here we use a *lazy* scan to avoid
/// reading the entire file immediately and then materialise it with
/// [`collect`].
pub fn read_parquet_to_dataframe(path: &str) -> Result<DataFrame> {
    let lf = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
    // LazyFrames allow query optimisation before execution.  Once collected
    // we obtain an eager `DataFrame` for direct manipulation.
    Ok(lf.collect()?)
}

/// Convert the rows of a [`DataFrame`] into a vector of [`Record`] structs.
///
/// This uses typed accessors on each column followed by an iterator over the
/// values.  `serde` isn't strictly required for this simple case but deriving
/// `Serialize`/`Deserialize` on [`Record`] means it could easily be sent over
/// the network or written to other formats.
pub fn dataframe_to_records(df: &DataFrame) -> Result<Vec<Record>> {
    let id = df.column("id")?.i64()?;
    let name = df.column("name")?.str()?;

    let records: Result<Vec<Record>> = id
        .into_iter()
        .zip(name.into_iter())
        .map(|(opt_id, opt_name)| {
            Ok(Record {
                id: opt_id.ok_or_else(|| PolarsError::NoData("null id".into()))?,
                name: opt_name
                    .ok_or_else(|| PolarsError::NoData("null name".into()))?
                    .to_string(),
            })
        })
        .collect();
    records
}

/// Modify the provided records in place.
///
/// Any complex domain logic could be run here.  We simply append an
/// exclamation mark to demonstrate mutation.
pub fn modify_records(records: &mut [Record]) {
    for rec in records.iter_mut() {
        rec.name.push('!');
    }
}

/// Convert [`Record`]s back into a [`DataFrame`].
///
/// This showcases Polars' eager API where columns are constructed from
/// `Vec`s.  The same operations could also be expressed using the lazy API for
/// larger pipelines.
pub fn records_to_dataframe(records: &[Record]) -> Result<DataFrame> {
    let ids: Vec<i64> = records.iter().map(|r| r.id).collect();
    let names: Vec<String> = records.iter().map(|r| r.name.clone()).collect();
    let df = df!("id" => ids, "name" => names)?;
    Ok(df)
}

/// Write a [`DataFrame`] to a Parquet file.
///
/// The writer takes a reference to a file handle.  Because the data is already
/// materialised in memory this uses the eager API.  In a real application the
/// lazy API could be used to stream results directly to disk.
pub fn write_dataframe_to_parquet(df: &DataFrame, path: &str) -> Result<()> {
    let mut df = df.clone();
    let file = File::create(path)?;
    ParquetWriter::new(file).finish(&mut df)?;
    Ok(())
}

/// Write a [`DataFrame`] to a CSV file.
pub fn write_dataframe_to_csv(df: &DataFrame, path: &str) -> Result<()> {
    let mut df = df.clone();
    let file = File::create(path)?;
    CsvWriter::new(file).finish(&mut df)?;
    Ok(())
}

/// Write a [`DataFrame`] to a JSON file in line-delimited format.
pub fn write_dataframe_to_json(df: &DataFrame, path: &str) -> Result<()> {
    let mut df = df.clone();
    let file = File::create(path)?;
    JsonWriter::new(file).finish(&mut df)?;
    Ok(())
}

/// Convenience wrapper which writes the provided [`DataFrame`] to the given path.
///
/// This simply forwards to [`write_dataframe_to_parquet`].  It exists so the
/// GUI can create a `DataFrame` and immediately persist it using a single call.
pub fn create_and_write_parquet(df: &DataFrame, path: &str) -> Result<()> {
    write_dataframe_to_parquet(df, path)
}

/// Build a new [`DataFrame`] from a user provided schema and row data.
///
/// The `schema` slice defines column names and their [`DataType`].  `rows`
/// contains values for each row using [`AnyValue`] to allow mixed types.
pub fn create_dataframe(
    schema: &[(String, DataType)],
    rows: &[Vec<AnyValue>],
) -> Result<DataFrame> {
    use polars::prelude::{IntoColumn, Series};

    let mut cols: Vec<Column> = Vec::with_capacity(schema.len());
    for (idx, (name, dtype)) in schema.iter().enumerate() {
        // gather all values for this column
        let values: Vec<AnyValue> = rows
            .iter()
            .map(|r| r.get(idx).cloned().unwrap_or(AnyValue::Null))
            .collect();
        let s = Series::from_any_values_and_dtype(name.clone().into(), &values, dtype, true)?;
        cols.push(s.into_column());
    }
    Ok(DataFrame::new(cols)?)
}

/// Write the [`DataFrame`] grouped by `column` into separate Parquet files.
///
/// Each distinct value of `column` is written to `dir/<value>.parquet`.
pub fn write_partitioned(df: &DataFrame, column: &str, dir: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for part in df.partition_by([column], true)? {
        let series = part.column(column)?;
        let mut value = series.get(0)?.to_string();
        if matches!(series.dtype(), DataType::String) {
            value = value.trim_matches('"').replace(['/', '\\'], "_");
        }
        let file = format!("{}/{}.parquet", dir.trim_end_matches('/'), value);
        write_dataframe_to_parquet(&part, &file)?;
    }
    Ok(())
}

/// Read all Parquet files in a directory and concatenate them into a single [`DataFrame`].
pub fn read_partitions(dir: &str) -> Result<DataFrame> {
    let pattern = format!("{}/*.parquet", dir.trim_end_matches('/'));
    let lf = LazyFrame::scan_parquet(&pattern, ScanArgsParquet::default())?;
    Ok(lf.collect()?)
}

/// Read all Parquet files in a directory using a single scan.
///
/// Passing either a directory path or a glob pattern allows Polars to treat
/// every matching file as one logical dataset.  The combined result is
/// collected into an eager [`DataFrame`].
pub fn read_parquet_directory(dir: &str) -> Result<DataFrame> {
    let lf = LazyFrame::scan_parquet(dir, ScanArgsParquet::default())?;
    Ok(lf.collect()?)
}

/// Lazily read only a subset of columns from a Parquet file.
///
/// Providing a slice of column names allows Polars to skip all other data
/// during the scan which can significantly reduce IO for wide tables.  The
/// selected columns are then collected into an eager [`DataFrame`].
pub fn read_selected_columns(path: &str, columns: &[&str]) -> Result<DataFrame> {
    let lf = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?
        .select(columns.iter().map(|c| col(*c)).collect::<Vec<_>>());
    Ok(lf.collect()?)
}

/// Filter rows in a Parquet file by a prefix on the `name` column.
///
/// This demonstrates constructing an expression on a `LazyFrame` before
/// materialising the result.  Only rows beginning with the provided prefix are
/// returned.
pub fn filter_by_name_prefix(path: &str, prefix: &str) -> Result<DataFrame> {
    let lf = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
    Ok(lf
        .filter(col("name").str().starts_with(lit(prefix)))
        .collect()?)
}

/// Apply a simple expression string as a filter on a Parquet file.
///
/// The expression must be in the form `column op value` where `op` is one of
/// `==`, `!=`, `>`, `<`, `>=` or `<=`. Values may be numeric, boolean or
/// quoted strings. Returns an error if the expression cannot be parsed.
pub fn filter_with_expr(path: &str, expr: &str) -> Result<DataFrame> {
    let filter = parse_simple_expr(expr)?;
    let lf = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
    Ok(lf.filter(filter).collect()?)
}

fn parse_simple_expr(s: &str) -> Result<Expr> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 3 {
        anyhow::bail!("expression format should be: <column> <op> <value>");
    }
    let column = parts[0];
    let op = parts[1];
    let raw_value = parts[2];
    let value_expr = if let Ok(v) = raw_value.parse::<i64>() {
        lit(v)
    } else if let Ok(v) = raw_value.parse::<f64>() {
        lit(v)
    } else if raw_value.eq_ignore_ascii_case("true") || raw_value.eq_ignore_ascii_case("false") {
        lit(raw_value.parse::<bool>()?)
    } else if (raw_value.starts_with('"') && raw_value.ends_with('"'))
        || (raw_value.starts_with('\'') && raw_value.ends_with('\''))
    {
        let trimmed = &raw_value[1..raw_value.len() - 1];
        lit(trimmed)
    } else {
        lit(raw_value)
    };

    let col_expr = col(column);
    let expr = match op {
        "==" | "=" => col_expr.eq(value_expr),
        "!=" | "<>" => col_expr.neq(value_expr),
        ">" => col_expr.gt(value_expr),
        "<" => col_expr.lt(value_expr),
        ">=" => col_expr.gt_eq(value_expr),
        "<=" => col_expr.lt_eq(value_expr),
        _ => anyhow::bail!("unsupported operator"),
    };
    Ok(expr)
}

/// Retrieve low level metadata from a Parquet file using the `parquet` crate.
///
/// Accessing the metadata can be useful for quickly inspecting files without
/// loading the data into memory.  The returned structure exposes row group and
/// column information among other details.
pub fn read_parquet_metadata(path: &str) -> Result<parquet::file::metadata::ParquetMetaData> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    Ok(reader.metadata().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trip() -> Result<()> {
        let dir = tempdir()?;
        let input_path = dir.path().join("input.parquet");
        let output_path = dir.path().join("output.parquet");

        // Create a small DataFrame and write it as Parquet
        let df = df!("id" => &[1i64, 2], "name" => &["a", "b"])?;
        write_dataframe_to_parquet(&df, input_path.to_str().unwrap())?;

        // Read it back
        let df_read = read_parquet_to_dataframe(input_path.to_str().unwrap())?;
        let mut records = dataframe_to_records(&df_read)?;
        assert_eq!(records.len(), 2);

        // Modify then write again
        modify_records(&mut records);
        let out_df = records_to_dataframe(&records)?;
        write_dataframe_to_parquet(&out_df, output_path.to_str().unwrap())?;

        // Ensure written file has expected modifications
        let final_df = read_parquet_to_dataframe(output_path.to_str().unwrap())?;
        let final_records = dataframe_to_records(&final_df)?;
        assert_eq!(final_records[0].name, "a!");
        Ok(())
    }

    #[test]
    fn extra_examples() -> Result<()> {
        let dir = tempdir()?;
        let file = dir.path().join("data.parquet");

        let df = df!("id" => &[1i64, 2, 3], "name" => &["alice", "bob", "anne"])?;
        write_dataframe_to_parquet(&df, file.to_str().unwrap())?;

        // Reading only the "id" column
        let cols = read_selected_columns(file.to_str().unwrap(), &["id"])?;
        assert_eq!(cols.get_column_names(), vec!["id"]);

        // Filtering by prefix
        let filtered = filter_by_name_prefix(file.to_str().unwrap(), "a")?;
        assert_eq!(filtered.height(), 2);

        // Metadata should report a single row group
        let meta = read_parquet_metadata(file.to_str().unwrap())?;
        assert_eq!(meta.num_row_groups(), 1);

        Ok(())
    }

    #[test]
    fn partition_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let part_dir = dir.path().join("parts");

        let schema = vec![
            ("id".to_string(), DataType::Int64),
            ("name".to_string(), DataType::String),
        ];
        let rows = vec![
            vec![AnyValue::Int64(1), AnyValue::String("a".into())],
            vec![AnyValue::Int64(2), AnyValue::String("b".into())],
        ];

        let df = create_dataframe(&schema, &rows)?;
        write_partitioned(&df, "name", part_dir.to_str().unwrap())?;

        // ensure files were written for each unique key
        let mut files: Vec<_> = std::fs::read_dir(&part_dir)?
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        files.sort();
        assert_eq!(files, vec!["a.parquet", "b.parquet"]);

        let read = read_partitions(part_dir.to_str().unwrap())?;
        assert_eq!(read.shape(), df.shape());
        Ok(())
    }

    #[test]
    fn create_dataframe_correct() -> Result<()> {
        let schema = vec![
            ("id".to_string(), DataType::Int64),
            ("name".to_string(), DataType::String),
        ];
        let rows = vec![
            vec![AnyValue::Int64(1), AnyValue::String("a".into())],
            vec![AnyValue::Int64(2), AnyValue::String("b".into())],
        ];

        let df = create_dataframe(&schema, &rows)?;
        let names = df.get_column_names();
        assert_eq!(names, vec!["id", "name"]);
        assert_eq!(df.dtypes(), vec![DataType::Int64, DataType::String]);
        let ids: Vec<i64> = df.column("id")?.i64()?.into_no_null_iter().collect();
        let names_col: Vec<String> = df
            .column("name")?
            .str()?
            .into_no_null_iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(names_col, vec!["a".to_string(), "b".to_string()]);
        Ok(())
    }

    #[test]
    fn create_dataframe_floats_and_bools() -> Result<()> {
        let schema = vec![
            ("val".to_string(), DataType::Float64),
            ("flag".to_string(), DataType::Boolean),
        ];
        let rows = vec![
            vec![AnyValue::Float64(1.5), AnyValue::Boolean(true)],
            vec![AnyValue::Float64(2.5), AnyValue::Boolean(false)],
        ];

        let df = create_dataframe(&schema, &rows)?;
        assert_eq!(df.dtypes(), vec![DataType::Float64, DataType::Boolean]);
        let vals: Vec<f64> = df.column("val")?.f64()?.into_no_null_iter().collect();
        let flags: Vec<bool> = df.column("flag")?.bool()?.into_no_null_iter().collect();
        assert_eq!(vals, vec![1.5, 2.5]);
        assert_eq!(flags, vec![true, false]);
        Ok(())
    }

    #[test]
    fn scan_directory_multiple_files() -> Result<()> {
        let dir = tempdir()?;
        let f1 = dir.path().join("one.parquet");
        let f2 = dir.path().join("two.parquet");

        let df1 = df!("id" => &[1i64], "name" => &["a"])?;
        let df2 = df!("id" => &[2i64], "name" => &["b"])?;

        write_dataframe_to_parquet(&df1, f1.to_str().unwrap())?;
        write_dataframe_to_parquet(&df2, f2.to_str().unwrap())?;

        let read = read_parquet_directory(dir.path().to_str().unwrap())?;
        assert_eq!(read.height(), df1.height() + df2.height());
        Ok(())
    }

    #[test]
    fn filter_with_expr_basic() -> Result<()> {
        let dir = tempdir()?;
        let file = dir.path().join("data.parquet");

        let df = df!("id" => &[1i64, 2, 3], "val" => &[10i64, 20, 30])?;
        write_dataframe_to_parquet(&df, file.to_str().unwrap())?;

        let filtered = filter_with_expr(file.to_str().unwrap(), "id > 1")?;
        assert_eq!(filtered.height(), 2);
        Ok(())
    }

    #[test]
    fn write_csv_and_json() -> Result<()> {
        let dir = tempdir()?;
        let csv_path = dir.path().join("out.csv");
        let json_path = dir.path().join("out.json");

        let df = df!("id" => &[1i64, 2], "name" => &["a", "b"])?;
        write_dataframe_to_csv(&df, csv_path.to_str().unwrap())?;
        write_dataframe_to_json(&df, json_path.to_str().unwrap())?;

        assert!(csv_path.exists());
        assert!(json_path.exists());
        Ok(())
    }
}
