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
    let name = df.column("name")?.utf8()?;

    let records = id
        .into_iter()
        .zip(name.into_iter())
        .map(|(opt_id, opt_name)| Record {
            id: opt_id.ok_or_else(|| PolarsError::NoData("null id".into()))?,
            name: opt_name
                .ok_or_else(|| PolarsError::NoData("null name".into()))?
                .to_string(),
        })
        .collect();
    Ok(records)
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
    let file = File::create(path)?;
    ParquetWriter::new(file).finish(df)?;
    Ok(())
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
}
