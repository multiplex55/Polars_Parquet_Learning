use anyhow::Result;
use polars::prelude::*;
use tokio::task;

use crate::parquet_examples;

/// Result returned by background jobs.
pub enum JobResult {
    DataFrame(DataFrame),
    Unit,
}

/// Asynchronously read a Parquet file into a [`DataFrame`].
pub async fn read_dataframe(path: String) -> Result<JobResult> {
    let df =
        task::spawn_blocking(move || parquet_examples::read_parquet_to_dataframe(&path)).await??;
    Ok(JobResult::DataFrame(df))
}

/// Asynchronously read all Parquet files in a directory into a single [`DataFrame`].
pub async fn read_directory(path: String) -> Result<JobResult> {
    let df =
        task::spawn_blocking(move || parquet_examples::read_parquet_directory(&path)).await??;
    Ok(JobResult::DataFrame(df))
}

/// Asynchronously read a CSV file into a [`DataFrame`].
pub async fn read_csv(path: String) -> Result<JobResult> {
    let df = task::spawn_blocking(move || -> Result<DataFrame> {
        Ok(CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()?)
    })
    .await??;
    Ok(JobResult::DataFrame(df))
}

/// Asynchronously read a JSON file into a [`DataFrame`].
pub async fn read_json(path: String) -> Result<JobResult> {
    let df = task::spawn_blocking(move || -> Result<DataFrame> {
        let file = std::fs::File::open(&path)?;
        Ok(JsonReader::new(file).finish()?)
    })
    .await??;
    Ok(JobResult::DataFrame(df))
}

/// Asynchronously read a slice of rows from a Parquet file.
pub async fn read_dataframe_slice(path: String, start: i64, len: usize) -> Result<JobResult> {
    let df =
        task::spawn_blocking(move || parquet_examples::read_parquet_slice(&path, start, len))
            .await??;
    Ok(JobResult::DataFrame(df))
}

/// Asynchronously write a [`DataFrame`] to Parquet.
pub async fn write_dataframe(mut df: DataFrame, path: String) -> Result<JobResult> {
    task::spawn_blocking(move || parquet_examples::write_dataframe_to_parquet(&mut df, &path))
        .await??;
    Ok(JobResult::Unit)
}
