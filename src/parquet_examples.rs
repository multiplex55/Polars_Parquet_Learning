//! Example utilities for working with Parquet files using Polars.
//!
//! The functions here are kept separate from the GUI so they can also be
//! exercised from unit tests.  They demonstrate a very small slice of what
//! Polars offers when dealing with columnar data.

use anyhow::Result;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use shlex;
use std::fs::File;

/// Basic summary information for a [`DataFrame`].
#[derive(Debug, Clone)]
pub struct DataFrameSummary {
    /// Number of rows in the frame.
    pub rows: usize,
    /// Number of columns in the frame.
    pub columns: usize,
    /// Approximate memory usage of the frame in bytes.
    pub approx_bytes: usize,
    /// Simple statistics for each column.
    pub stats: DataFrame,
}

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

/// Write a [`DataFrame`] to a Parquet file using the chosen compression.
///
/// The writer takes a reference to a file handle.  Because the data is already
/// materialised in memory this uses the eager API.  In a real application the
/// lazy API could be used to stream results directly to disk.
pub fn write_dataframe_to_parquet(
    df: &mut DataFrame,
    path: &str,
    compression: parquet::basic::Compression,
) -> Result<()> {
    use parquet::basic::Compression as C;
    use polars::prelude::{BrotliLevel, GzipLevel, ParquetCompression as Pc, ZstdLevel};

    let pc = match compression {
        C::UNCOMPRESSED => Pc::Uncompressed,
        C::SNAPPY => Pc::Snappy,
        C::GZIP(level) => Pc::Gzip(Some(GzipLevel::try_new(level.compression_level() as u8)?)),
        C::LZO => Pc::Lzo,
        C::BROTLI(level) => Pc::Brotli(Some(BrotliLevel::try_new(level.compression_level())?)),
        C::LZ4 | C::LZ4_RAW => Pc::Lz4Raw,
        C::ZSTD(level) => Pc::Zstd(Some(ZstdLevel::try_new(level.compression_level() as i32)?)),
    };

    let file = File::create(path)?;
    ParquetWriter::new(file).with_compression(pc).finish(df)?;
    Ok(())
}

/// Write a [`DataFrame`] to a CSV file.
pub fn write_dataframe_to_csv(df: &mut DataFrame, path: &str) -> Result<()> {
    let file = File::create(path)?;
    CsvWriter::new(file).finish(df)?;
    Ok(())
}

/// Write a [`DataFrame`] to a JSON file in line-delimited format.
pub fn write_dataframe_to_json(df: &mut DataFrame, path: &str) -> Result<()> {
    let file = File::create(path)?;
    JsonWriter::new(file).finish(df)?;
    Ok(())
}

/// Example structs used to demonstrate Dremel encoded columns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Foo {
    pub a: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bar {
    pub x: bool,
}

/// Enum wrapping the different example structs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExampleItem {
    Foo(Foo),
    Bar(Bar),
}

impl ExampleItem {
    fn type_name(&self) -> &'static str {
        match self {
            ExampleItem::Foo(_) => "foo",
            ExampleItem::Bar(_) => "bar",
        }
    }

    fn from_parts(type_name: &str, json: &str) -> Result<Self> {
        Ok(match type_name {
            "foo" => ExampleItem::Foo(serde_json::from_str(json)?),
            "bar" => ExampleItem::Bar(serde_json::from_str(json)?),
            _ => anyhow::bail!("unknown type"),
        })
    }
}

/// Build a [`DataFrame`] containing list columns representing repeated structs.
///
/// The returned frame has two columns: `type` holding the struct name and
/// `value` containing a JSON representation. Each cell of these columns is a
/// list allowing an arbitrary number of structs per row.
pub fn create_dremel_dataframe(rows: &[Vec<ExampleItem>]) -> Result<DataFrame> {
    let mut type_lists: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    let mut value_lists: Vec<Vec<String>> = Vec::with_capacity(rows.len());

    for row in rows {
        let mut types = Vec::with_capacity(row.len());
        let mut values = Vec::with_capacity(row.len());
        for item in row {
            types.push(item.type_name().to_string());
            let json = match item {
                ExampleItem::Foo(f) => serde_json::to_string(f)?,
                ExampleItem::Bar(b) => serde_json::to_string(b)?,
            };
            values.push(json);
        }
        type_lists.push(types);
        value_lists.push(values);
    }

    use polars::prelude::ListChunked;

    let type_series_vec: Vec<Option<Series>> = type_lists
        .iter()
        .map(|v| Some(Series::new("".into(), v.as_slice())))
        .collect();
    let value_series_vec: Vec<Option<Series>> = value_lists
        .iter()
        .map(|v| Some(Series::new("".into(), v.as_slice())))
        .collect();

    let mut type_list = ListChunked::from_iter(type_series_vec).into_series();
    type_list.rename("type".into());
    let mut value_list = ListChunked::from_iter(value_series_vec).into_series();
    value_list.rename("value".into());
    Ok(DataFrame::new(vec![type_list.into(), value_list.into()])?)
}

/// Convert a [`DataFrame`] produced by [`create_dremel_dataframe`] back into
/// typed structs.
pub fn dataframe_to_items(df: &DataFrame) -> Result<Vec<Vec<ExampleItem>>> {
    let types = df.column("type")?.list()?;
    let values = df.column("value")?.list()?;

    let mut rows: Vec<Vec<ExampleItem>> = Vec::with_capacity(df.height());
    for (t_opt, v_opt) in types.into_iter().zip(values.into_iter()) {
        let t_series = t_opt.ok_or_else(|| PolarsError::NoData("null type".into()))?;
        let v_series = v_opt.ok_or_else(|| PolarsError::NoData("null value".into()))?;
        let t_iter = t_series.str()?;
        let v_iter = v_series.str()?;
        let row: Result<Vec<ExampleItem>> = t_iter
            .into_iter()
            .zip(v_iter.into_iter())
            .map(|(t, v)| ExampleItem::from_parts(t.unwrap(), v.unwrap()))
            .collect();
        rows.push(row?);
    }
    Ok(rows)
}

/// Convenience helper which writes the provided structs as a Dremel encoded
/// Parquet file.
pub fn write_dremel_parquet(rows: &[Vec<ExampleItem>], path: &str) -> Result<()> {
    let df = create_dremel_dataframe(rows)?;
    let mut df = df.clone();
    write_dataframe_to_parquet(&mut df, path, parquet::basic::Compression::SNAPPY)
}

/// Read a Dremel encoded Parquet file into typed structs.
pub fn read_dremel_parquet(path: &str) -> Result<Vec<Vec<ExampleItem>>> {
    let df = read_parquet_to_dataframe(path)?;
    dataframe_to_items(&df)
}

/// Summarise a [`DataFrame`] returning row/column counts and basic statistics.
pub fn summarize_dataframe(df: &DataFrame) -> Result<DataFrameSummary> {
    let mut names: Vec<String> = Vec::new();
    let mut nulls: Vec<i64> = Vec::new();
    let mut mins: Vec<Option<f64>> = Vec::new();
    let mut maxs: Vec<Option<f64>> = Vec::new();
    let mut means: Vec<Option<f64>> = Vec::new();

    for col in df.get_columns() {
        names.push(col.name().to_string());
        nulls.push(col.null_count() as i64);
        match col.dtype() {
            DataType::Float64 => {
                let ca = col.f64()?;
                mins.push(ca.min());
                maxs.push(ca.max());
                means.push(ca.mean());
            }
            DataType::Int64 => {
                let ca = col.i64()?;
                mins.push(ca.min().map(|v| v as f64));
                maxs.push(ca.max().map(|v| v as f64));
                means.push(ca.mean());
            }
            _ => {
                mins.push(None);
                maxs.push(None);
                means.push(None);
            }
        }
    }

    let stats = df!(
        "column" => names,
        "nulls" => nulls,
        "min" => mins,
        "max" => maxs,
        "mean" => means
    )?;

    Ok(DataFrameSummary {
        rows: df.height(),
        columns: df.width(),
        approx_bytes: df.estimated_size(),
        stats,
    })
}

/// Convenience wrapper which writes the provided [`DataFrame`] to the given path.
///
/// This simply forwards to [`write_dataframe_to_parquet`].  It exists so the
/// GUI can create a `DataFrame` and immediately persist it using a single call.
pub fn create_and_write_parquet(df: &DataFrame, path: &str) -> Result<()> {
    let mut df = df.clone();
    write_dataframe_to_parquet(&mut df, path, parquet::basic::Compression::SNAPPY)
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

/// Rename a column in the [`DataFrame`].
pub fn rename_column(df: &mut DataFrame, old: &str, new: &str) -> Result<()> {
    df.rename(old, new.into())?;
    Ok(())
}

/// Drop a column from the [`DataFrame`].
pub fn drop_column(df: &mut DataFrame, name: &str) -> Result<()> {
    df.drop_in_place(name)?;
    Ok(())
}

/// Reorder columns in the [`DataFrame`] according to the provided names.
pub fn reorder_columns(df: &mut DataFrame, order: &[String]) -> Result<()> {
    use polars::prelude::Column;
    let mut cols: Vec<Column> = Vec::with_capacity(order.len());
    for name in order {
        cols.push(df.column(name)?.clone());
    }
    *df = DataFrame::new(cols)?;
    Ok(())
}

/// Compute a correlation matrix for the provided numeric columns.
///
/// The returned [`DataFrame`] contains a `column` label column followed by
/// each input column containing the pairwise Pearson correlation coefficients.
/// Null values are converted to `NaN` before the calculations.
pub fn correlation_matrix(df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
    if columns.is_empty() {
        return Ok(DataFrame::default());
    }

    // Collect each column as `f64` values once so we can efficiently reuse them
    // when calculating all pairs.
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(columns.len());
    for &name in columns {
        let s = df.column(name)?;
        let s = s.cast(&DataType::Float64)?;
        let vals: Vec<f64> = s
            .f64()?
            .into_iter()
            .map(|o| o.unwrap_or(f64::NAN))
            .collect();
        data.push(vals);
    }

    // Compute correlation for every pair of columns.
    let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(columns.len());
    for i in 0..columns.len() {
        let mut row: Vec<f64> = Vec::with_capacity(columns.len());
        for j in 0..columns.len() {
            row.push(pearson_corr_slice(&data[i], &data[j]));
        }
        matrix.push(row);
    }

    // Assemble the result DataFrame with row/column labels.
    use polars::prelude::Column;
    let mut cols: Vec<Column> = Vec::with_capacity(columns.len() + 1);
    cols.push(Column::new("column".into(), columns));
    for (j, &name) in columns.iter().enumerate() {
        let col_vals: Vec<f64> = matrix.iter().map(|row| row[j]).collect();
        cols.push(Column::new(name.into(), col_vals));
    }

    Ok(DataFrame::new(cols)?)
}

/// Compute the Pearson correlation coefficient for two equal length slices.
fn pearson_corr_slice(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a == 0.0 || var_b == 0.0 {
        0.0
    } else {
        cov / (var_a.sqrt() * var_b.sqrt())
    }
}

/// Write the [`DataFrame`] grouped by one or more columns into separate Parquet files.
///
/// For a single column this writes `dir/<value>.parquet`. When multiple columns
/// are provided nested folders are created so each combination ends up at
/// `dir/<col1>/<col2>.parquet` and so on.
pub fn write_partitioned(df: &DataFrame, columns: &[&str], dir: &str) -> Result<()> {
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    std::fs::create_dir_all(dir)?;
    let mut written: HashSet<PathBuf> = HashSet::new();

    for part in df.partition_by(columns.iter().copied(), true)? {
        let mut path: PathBuf = Path::new(dir).to_path_buf();
        for &col in columns {
            let av = part.column(col)?.get(0)?;
            let mut value = match av {
                AnyValue::String(s) => s.to_string(),
                AnyValue::StringOwned(ref s) => s.to_string(),
                _ => av.to_string(),
            };
            value = value.replace(['/', '\\', ':', '*', '?', '<', '>', '|'], "_");
            path.push(value);
        }
        path.set_extension("parquet");

        // Ensure we don't overwrite files when sanitised names collide
        let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let mut unique_path = path.clone();
        let mut idx = 1;
        while written.contains(&unique_path) {
            unique_path = parent.join(format!("{}_{idx}", stem));
            unique_path.set_extension("parquet");
            idx += 1;
        }

        if let Some(p) = unique_path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let file = unique_path.to_string_lossy().to_string();
        let mut part = part;
        write_dataframe_to_parquet(&mut part, &file, parquet::basic::Compression::SNAPPY)?;
        written.insert(unique_path);
    }
    Ok(())
}

/// Read all Parquet files in a directory and concatenate them into a single [`DataFrame`].
pub fn read_partitions(dir: &str) -> Result<DataFrame> {
    let pattern = format!("{}/*.parquet", dir.trim_end_matches('/'));
    Ok(LazyFrame::scan_parquet(&pattern, ScanArgsParquet::default())?.collect()?)
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

/// Read a slice of rows from a Parquet file.
///
/// This performs a lazy scan and applies [`slice`] before collecting
/// the result.  `start` is the zero-based offset of the first row and
/// `len` is the number of rows to return.
pub fn read_parquet_slice(path: &str, start: i64, len: usize) -> Result<DataFrame> {
    let lf = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?.slice(start, len as u32);
    Ok(lf.collect()?)
}

/// Compute histogram counts for a slice of values.
///
/// Returns the bin counts along with the minimum value and bin step.
pub fn compute_histogram(
    values: &[f64],
    bins: usize,
    range: Option<(f64, f64)>,
) -> Result<(Vec<f64>, f64, f64)> {
    if bins == 0 {
        return Ok((Vec::new(), 0.0, 0.0));
    }
    if values.is_empty() {
        return Ok((vec![0.0; bins], 0.0, 0.0));
    }

    let (min, max) = match range {
        Some(r) => r,
        None => {
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        }
    };
    if min > max {
        anyhow::bail!("histogram range min is greater than max");
    }
    let step = (max - min) / bins as f64;
    let mut counts = vec![0f64; bins];
    if step == 0.0 {
        counts[0] = values.len() as f64;
    } else {
        for &v in values {
            let mut idx = ((v - min) / step).floor() as isize;
            if idx < 0 {
                idx = 0;
            }
            let mut i = idx as usize;
            if i >= bins {
                i = bins - 1;
            }
            counts[i] += 1.0;
        }
    }
    Ok((counts, min, step))
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

/// Quote a string value for use in an expression.
///
/// Escapes any embedded backslashes and double quotes and wraps the result
/// in double quotes so it can be parsed by [`shlex`].
pub fn quote_expr_value(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{}\"", escaped)
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

/// Join two [`DataFrame`]s on a key column using an inner join.
pub fn join_on_key(left: &DataFrame, right: &DataFrame, key: &str) -> Result<DataFrame> {
    Ok(left.join(right, [key], [key], JoinArgs::from(JoinType::Inner), None)?)
}

/// Group by `group` and compute the sum of `values`.
pub fn group_by_sum(df: &DataFrame, group: &str, values: &str) -> Result<DataFrame> {
    Ok(
        df.clone()
            .lazy()
            .group_by([col(group)])
            .agg([col(values).sum()])
            .collect()?,
    )
}

/// Pivot long-form data into a wide layout.
pub fn pivot_wider(df: &DataFrame, index: &str, columns: &str, values: &str) -> Result<DataFrame> {
    use polars::lazy::frame::pivot::pivot_stable;

    Ok(pivot_stable(
        df,
        [columns],
        Some([index]),
        Some([values]),
        true,
        Some(first()),
        None,
    )?)
}

fn parse_simple_expr(s: &str) -> Result<Expr> {
    let parts: Vec<String> = shlex::Shlex::new(s).collect();
    if parts.len() != 3 {
        anyhow::bail!("expression format should be: <column> <op> <value>");
    }
    let column = parts[0].as_str();
    let op = parts[1].as_str();
    let raw_value = parts[2].as_str();
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

/// Apply multiple simple expressions as filters on a Parquet file.
pub fn filter_with_exprs(path: &str, exprs: &[String]) -> Result<DataFrame> {
    let mut lf = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
    let mut contains: Vec<(String, String)> = Vec::new();
    for ex in exprs {
        let parts: Vec<String> = shlex::Shlex::new(ex).collect();
        if parts.len() == 3 && parts[1].eq_ignore_ascii_case("contains") {
            let val = parts[2].trim_matches(&['"', '\''][..]).to_string();
            contains.push((parts[0].clone(), val));
        } else {
            lf = lf.filter(parse_simple_expr(ex)?);
        }
    }
    let mut df = lf.collect()?;
    for (col, val) in contains {
        let series = df.column(&col)?;
        if !matches!(series.dtype(), DataType::String) {
            anyhow::bail!("'contains' only supported on string columns");
        }
        let mask = series.str()?.contains_literal(&val)?;
        df = df.filter(&mask)?;
    }
    Ok(df)
}

/// Apply multiple expressions as filters and return a slice of rows.
pub fn filter_slice(path: &str, exprs: &[String], start: i64, len: usize) -> Result<DataFrame> {
    let mut lf = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
    let mut contains: Vec<(String, String)> = Vec::new();
    for ex in exprs {
        let parts: Vec<String> = shlex::Shlex::new(ex).collect();
        if parts.len() == 3 && parts[1].eq_ignore_ascii_case("contains") {
            let val = parts[2].trim_matches(&['"', '\''][..]).to_string();
            contains.push((parts[0].clone(), val));
        } else {
            lf = lf.filter(parse_simple_expr(ex)?);
        }
    }
    let mut df = lf.collect()?;
    for (col, val) in contains {
        let series = df.column(&col)?;
        if !matches!(series.dtype(), DataType::String) {
            anyhow::bail!("'contains' only supported on string columns");
        }
        let mask = series.str()?.contains_literal(&val)?;
        df = df.filter(&mask)?;
    }
    Ok(df.slice(start, len))
}

/// Return the number of rows matching multiple expressions without collecting them all.
pub fn filter_count(path: &str, exprs: &[String]) -> Result<usize> {
    let mut lf = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
    for ex in exprs {
        let parts: Vec<String> = shlex::Shlex::new(ex).collect();
        if parts.len() == 3 && parts[1].eq_ignore_ascii_case("contains") {
            let val = parts[2].trim_matches(&['"', '\''][..]).to_string();
            lf = lf.filter(col(&parts[0]).str().contains_literal(lit(val)));
        } else {
            lf = lf.filter(parse_simple_expr(ex)?);
        }
    }
    let df = lf.select([len()]).collect()?;
    let n = df.column("len")?.u32()?.get(0).unwrap_or(0);
    Ok(n as usize)
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
        let mut df = df!("id" => &[1i64, 2], "name" => &["a", "b"])?;
        write_dataframe_to_parquet(
            &mut df,
            input_path.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;

        // Read it back
        let df_read = read_parquet_to_dataframe(input_path.to_str().unwrap())?;
        let mut records = dataframe_to_records(&df_read)?;
        assert_eq!(records.len(), 2);

        // Modify then write again
        modify_records(&mut records);
        let mut out_df = records_to_dataframe(&records)?;
        write_dataframe_to_parquet(
            &mut out_df,
            output_path.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;

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

        let mut df = df!("id" => &[1i64, 2, 3], "name" => &["alice", "bob", "anne"])?;
        write_dataframe_to_parquet(
            &mut df,
            file.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;

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
        write_partitioned(&df, &["name"], part_dir.to_str().unwrap())?;

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
    fn partition_round_trip_multi_cols() -> Result<()> {
        let dir = tempdir()?;
        let part_dir = dir.path().join("parts");

        let schema = vec![
            ("id".to_string(), DataType::Int64),
            ("name".to_string(), DataType::String),
            ("flag".to_string(), DataType::Boolean),
        ];
        let rows = vec![
            vec![
                AnyValue::Int64(1),
                AnyValue::String("a".into()),
                AnyValue::Boolean(true),
            ],
            vec![
                AnyValue::Int64(2),
                AnyValue::String("a".into()),
                AnyValue::Boolean(false),
            ],
        ];

        let df = create_dataframe(&schema, &rows)?;
        write_partitioned(&df, &["name", "flag"], part_dir.to_str().unwrap())?;

        let subdir = part_dir.join("a");
        let mut files: Vec<_> = std::fs::read_dir(&subdir)?
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        files.sort();
        assert_eq!(files, vec!["false.parquet", "true.parquet"]);

        let pattern = format!("{}/a/*.parquet", part_dir.to_str().unwrap());
        let read = read_parquet_directory(&pattern)?;
        assert_eq!(read.shape(), df.shape());
        Ok(())
    }

    #[test]
    fn partition_sanitizes_invalid_chars() -> Result<()> {
        let dir = tempdir()?;
        let part_dir = dir.path().join("parts");

        let schema = vec![
            ("id".to_string(), DataType::Int64),
            ("name".to_string(), DataType::String),
        ];
        let rows = vec![
            vec![AnyValue::Int64(1), AnyValue::String("a:b".into())],
            vec![AnyValue::Int64(2), AnyValue::String("c*d".into())],
            vec![AnyValue::Int64(3), AnyValue::String("e?f".into())],
            vec![AnyValue::Int64(4), AnyValue::String("<g>".into())],
            vec![AnyValue::Int64(5), AnyValue::String("h|i".into())],
            vec![AnyValue::Int64(6), AnyValue::String("a/b".into())],
        ];

        let df = create_dataframe(&schema, &rows)?;
        write_partitioned(&df, &["name"], part_dir.to_str().unwrap())?;

        let mut files: Vec<_> = std::fs::read_dir(&part_dir)?
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        files.sort();

        assert_eq!(
            files,
            vec![
                "_g_.parquet",
                "a_b.parquet",
                "a_b_1.parquet",
                "c_d.parquet",
                "e_f.parquet",
                "h_i.parquet",
            ]
        );
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
        let ids: Vec<i64> = df
            .column("id")?
            .i64()?
            .into_iter()
            .map(|o| o.unwrap())
            .collect();
        let names_col: Vec<String> = df
            .column("name")?
            .str()?
            .into_iter()
            .map(|o| o.unwrap().to_string())
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
        let vals: Vec<f64> = df
            .column("val")?
            .f64()?
            .into_iter()
            .map(|o| o.unwrap())
            .collect();
        let flags: Vec<bool> = df
            .column("flag")?
            .bool()?
            .into_iter()
            .map(|o| o.unwrap())
            .collect();
        assert_eq!(vals, vec![1.5, 2.5]);
        assert_eq!(flags, vec![true, false]);
        Ok(())
    }

    #[test]
    fn create_dataframe_temporal_types() -> Result<()> {
        use chrono::{NaiveDate, NaiveDateTime, NaiveTime, Timelike};

        let schema = vec![
            ("d".to_string(), DataType::Date),
            (
                "dt".to_string(),
                DataType::Datetime(TimeUnit::Microseconds, None),
            ),
            ("t".to_string(), DataType::Time),
        ];

        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let d1 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let d2 = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();

        let dt1 = NaiveDateTime::new(d1, NaiveTime::from_hms_opt(12, 0, 0).unwrap());
        let dt2 = NaiveDateTime::new(d2, NaiveTime::from_hms_opt(13, 0, 0).unwrap());

        let t1 = NaiveTime::from_hms_opt(6, 0, 0).unwrap();
        let t2 = NaiveTime::from_hms_opt(7, 0, 0).unwrap();

        let rows = vec![
            vec![
                AnyValue::Date((d1 - epoch).num_days() as i32),
                AnyValue::Datetime(
                    dt1.and_utc().timestamp_micros(),
                    TimeUnit::Microseconds,
                    None,
                ),
                AnyValue::Time((t1.num_seconds_from_midnight() as i64) * 1_000_000_000),
            ],
            vec![
                AnyValue::Date((d2 - epoch).num_days() as i32),
                AnyValue::Datetime(
                    dt2.and_utc().timestamp_micros(),
                    TimeUnit::Microseconds,
                    None,
                ),
                AnyValue::Time((t2.num_seconds_from_midnight() as i64) * 1_000_000_000),
            ],
        ];

        let df = create_dataframe(&schema, &rows)?;
        assert_eq!(
            df.dtypes(),
            vec![
                DataType::Date,
                DataType::Datetime(TimeUnit::Microseconds, None),
                DataType::Time,
            ]
        );
        let dates: Vec<i32> = df
            .column("d")?
            .date()?
            .into_iter()
            .map(|o| o.unwrap())
            .collect();
        assert_eq!(dates.len(), 2);
        Ok(())
    }

    #[test]
    fn scan_directory_multiple_files() -> Result<()> {
        let dir = tempdir()?;
        let f1 = dir.path().join("one.parquet");
        let f2 = dir.path().join("two.parquet");

        let mut df1 = df!("id" => &[1i64], "name" => &["a"])?;
        let mut df2 = df!("id" => &[2i64], "name" => &["b"])?;

        write_dataframe_to_parquet(
            &mut df1,
            f1.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;
        write_dataframe_to_parquet(
            &mut df2,
            f2.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;

        let read = read_parquet_directory(dir.path().to_str().unwrap())?;
        assert_eq!(read.height(), df1.height() + df2.height());
        Ok(())
    }

    #[test]
    fn filter_with_expr_basic() -> Result<()> {
        let dir = tempdir()?;
        let file = dir.path().join("data.parquet");

        let mut df = df!("id" => &[1i64, 2, 3], "val" => &[10i64, 20, 30])?;
        write_dataframe_to_parquet(
            &mut df,
            file.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;

        let filtered = filter_with_expr(file.to_str().unwrap(), "id > 1")?;
        assert_eq!(filtered.height(), 2);
        Ok(())
    }

    #[test]
    fn filter_with_expr_quoted_string() -> Result<()> {
        let dir = tempdir()?;
        let file = dir.path().join("data.parquet");

        let mut df = df!("name" => &["John Doe", "Jane"])?;
        write_dataframe_to_parquet(
            &mut df,
            file.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;

        let filtered = filter_with_expr(file.to_str().unwrap(), "name == \"John Doe\"")?;
        assert_eq!(filtered.height(), 1);
        Ok(())
    }

    #[test]
    fn filter_with_expr_escaped_quotes() -> Result<()> {
        let dir = tempdir()?;
        let file = dir.path().join("data.parquet");

        let mut df = df!("name" => ["Ann \"The Hammer\"", "Bob"])?;
        write_dataframe_to_parquet(
            &mut df,
            file.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;

        let expr = format!("name == {}", quote_expr_value("Ann \"The Hammer\""));
        let filtered = filter_with_exprs(file.to_str().unwrap(), &[expr])?;
        assert_eq!(filtered.height(), 1);
        Ok(())
    }

    #[test]
    fn write_csv_and_json() -> Result<()> {
        let dir = tempdir()?;
        let csv_path = dir.path().join("out.csv");
        let json_path = dir.path().join("out.json");

        let mut df = df!("id" => &[1i64, 2], "name" => &["a", "b"])?;
        write_dataframe_to_csv(&mut df, csv_path.to_str().unwrap())?;
        write_dataframe_to_json(&mut df, json_path.to_str().unwrap())?;

        assert!(csv_path.exists());
        assert!(json_path.exists());
        Ok(())
    }

    #[test]
    fn read_csv_file() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("data.csv");

        let mut df = df!("id" => &[1i64, 2], "name" => &["a", "b"])?;
        write_dataframe_to_csv(&mut df, path.to_str().unwrap())?;

        let read = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.to_path_buf()))?
            .finish()?;
        assert_eq!(read.shape(), df.shape());
        Ok(())
    }

    #[test]
    fn dremel_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("dremel.parquet");

        let rows = vec![
            vec![
                ExampleItem::Foo(Foo { a: 1 }),
                ExampleItem::Bar(Bar { x: true }),
            ],
            vec![ExampleItem::Foo(Foo { a: 2 })],
        ];
        write_dremel_parquet(&rows, path.to_str().unwrap())?;

        let read = read_dremel_parquet(path.to_str().unwrap())?;
        assert_eq!(rows, read);
        Ok(())
    }

    #[test]
    fn slice_reads_correct_rows() -> Result<()> {
        let dir = tempdir()?;
        let file = dir.path().join("data.parquet");

        let mut df = df!("id" => &[1i64, 2, 3, 4], "name" => &["a", "b", "c", "d"])?;
        write_dataframe_to_parquet(
            &mut df,
            file.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;

        let slice = read_parquet_slice(file.to_str().unwrap(), 1, 2)?;
        assert_eq!(slice.height(), 2);
        let ids: Vec<i64> = slice
            .column("id")?
            .i64()?
            .into_iter()
            .map(|o| o.unwrap())
            .collect();
        assert_eq!(ids, vec![2, 3]);
        Ok(())
    }

    #[test]
    fn contains_on_non_string_returns_error() -> Result<()> {
        let dir = tempdir()?;
        let file = dir.path().join("data.parquet");

        let mut df = df!("id" => &[1i64, 2, 3])?;
        write_dataframe_to_parquet(
            &mut df,
            file.to_str().unwrap(),
            parquet::basic::Compression::SNAPPY,
        )?;

        let exprs = vec!["id contains \"1\"".to_string()];
        let res = filter_with_exprs(file.to_str().unwrap(), &exprs);
        assert!(res.is_err());

        let res_slice = filter_slice(file.to_str().unwrap(), &exprs, 0, 1);
        assert!(res_slice.is_err());
        Ok(())
    }
}
