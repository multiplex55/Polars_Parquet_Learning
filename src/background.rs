use anyhow::Result;
use polars::prelude::*;
use std::sync::mpsc::Sender;
use tokio::{task, time};

use crate::parquet_examples;

/// Result returned by background jobs.
pub enum JobResult {
    DataFrame(DataFrame),
    Count(usize),
    Unit,
}

/// Messages sent from background jobs.
pub enum JobUpdate {
    Progress(f32),
    Done(JobResult),
}

async fn run_with_progress<T, F>(
    tx: Sender<anyhow::Result<JobUpdate>>,
    job: F,
    map: fn(T) -> JobResult,
) where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let handle = task::spawn_blocking(job);
    tokio::pin!(handle);
    let mut progress = 0.0f32;
    let _ = tx.send(Ok(JobUpdate::Progress(progress)));
    loop {
        tokio::select! {
            res = &mut handle => {
                match res {
                    Ok(Ok(val)) => {
                        let _ = tx.send(Ok(JobUpdate::Progress(1.0)));
                        let _ = tx.send(Ok(JobUpdate::Done(map(val))));
                    }
                    Ok(Err(e)) => { let _ = tx.send(Err(e)); }
                    Err(e) => { let _ = tx.send(Err(e.into())); }
                }
                break;
            }
            _ = time::sleep(time::Duration::from_millis(200)) => {
                progress = (progress + 0.05).min(0.95);
                let _ = tx.send(Ok(JobUpdate::Progress(progress)));
            }
        }
    }
}

/// Asynchronously read a Parquet file into a [`DataFrame`].
pub async fn read_dataframe(path: String, tx: Sender<anyhow::Result<JobUpdate>>) {
    run_with_progress(
        tx,
        move || parquet_examples::read_parquet_to_dataframe(&path),
        JobResult::DataFrame,
    )
    .await;
}

/// Asynchronously read all Parquet files in a directory into a single [`DataFrame`].
pub async fn read_directory(path: String, tx: Sender<anyhow::Result<JobUpdate>>) {
    run_with_progress(
        tx,
        move || parquet_examples::read_parquet_directory(&path),
        JobResult::DataFrame,
    )
    .await;
}

/// Asynchronously read a CSV file into a [`DataFrame`].
pub async fn read_csv(path: String, tx: Sender<anyhow::Result<JobUpdate>>) {
    run_with_progress(
        tx,
        move || {
            Ok(CsvReadOptions::default()
                .try_into_reader_with_file_path(Some(path.into()))?
                .finish()?)
        },
        JobResult::DataFrame,
    )
    .await;
}

/// Asynchronously read a JSON file into a [`DataFrame`].
pub async fn read_json(path: String, tx: Sender<anyhow::Result<JobUpdate>>) {
    run_with_progress(
        tx,
        move || {
            let file = std::fs::File::open(&path)?;
            Ok(JsonReader::new(file).finish()?)
        },
        JobResult::DataFrame,
    )
    .await;
}

/// Asynchronously read a slice of rows from a Parquet file.
pub async fn read_dataframe_slice(
    path: String,
    start: i64,
    len: usize,
    tx: Sender<anyhow::Result<JobUpdate>>,
) {
    run_with_progress(
        tx,
        move || parquet_examples::read_parquet_slice(&path, start, len),
        JobResult::DataFrame,
    )
    .await;
}

/// Asynchronously filter a Parquet file and return a slice of rows.
pub async fn read_filter_slice(
    path: String,
    exprs: Vec<String>,
    start: i64,
    len: usize,
    tx: Sender<anyhow::Result<JobUpdate>>,
) {
    run_with_progress(
        tx,
        move || parquet_examples::filter_slice(&path, &exprs, start, len),
        JobResult::DataFrame,
    )
    .await;
}

/// Asynchronously write a [`DataFrame`] to Parquet.
pub async fn write_dataframe(
    mut df: DataFrame,
    path: String,
    tx: Sender<anyhow::Result<JobUpdate>>,
) {
    run_with_progress(
        tx,
        move || {
            parquet_examples::write_dataframe_to_parquet(&mut df, &path)?;
            Ok(())
        },
        |_| JobResult::Unit,
    )
    .await;
}

/// Asynchronously filter a Parquet file by multiple expressions.
pub async fn filter_with_exprs(
    path: String,
    exprs: Vec<String>,
    tx: Sender<anyhow::Result<JobUpdate>>,
) {
    run_with_progress(
        tx,
        move || parquet_examples::filter_with_exprs(&path, &exprs),
        JobResult::DataFrame,
    )
    .await;
}

/// Asynchronously count rows matching multiple expressions.
pub async fn filter_count(path: String, exprs: Vec<String>, tx: Sender<anyhow::Result<JobUpdate>>) {
    run_with_progress(
        tx,
        move || parquet_examples::filter_count(&path, &exprs),
        JobResult::Count,
    )
    .await;
}
