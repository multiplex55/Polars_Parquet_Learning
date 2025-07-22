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
    let df = task::spawn_blocking(move || parquet_examples::read_parquet_to_dataframe(&path))
        .await??;
    Ok(JobResult::DataFrame(df))
}

/// Asynchronously write a [`DataFrame`] to Parquet.
pub async fn write_dataframe(df: DataFrame, path: String) -> Result<JobResult> {
    task::spawn_blocking(move || parquet_examples::write_dataframe_to_parquet(&df, &path))
        .await??;
    Ok(JobResult::Unit)
}
