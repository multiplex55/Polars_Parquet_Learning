use Polars_Parquet_Learning::{background, parquet_examples};
use polars::prelude::*;
use parquet::basic::Compression;
use std::sync::mpsc;
use tempfile::tempdir;

#[test]
fn count_filtered_rows() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");
    let ids: Vec<i64> = (0..1000).collect();
    let names: Vec<String> = ids
        .iter()
        .map(|i| {
            if i % 2 == 0 {
                "yes".into()
            } else {
                "no".into()
            }
        })
        .collect();
    let mut df = df!("id" => ids, "name" => names)?;
    parquet_examples::write_dataframe_to_parquet(
        &mut df,
        file.to_str().unwrap(),
        parquet::basic::Compression::SNAPPY,
    )?;

    let count =
        parquet_examples::filter_count(file.to_str().unwrap(), &["name == \"yes\"".to_string()])?;
    assert_eq!(count, 500);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let (tx, rx) = mpsc::channel();
    rt.block_on(background::filter_count(
        file.to_str().unwrap().to_string(),
        vec!["name == \"yes\"".to_string()],
        tx,
    ));
    let res = loop {
        if let Ok(msg) = rx.recv() {
            if let Ok(background::JobUpdate::Done(background::JobResult::Count(c))) = msg {
                break c;
            }
        }
    };
    assert_eq!(res, 500);
    Ok(())
}

#[test]
fn count_filtered_rows_with_quotes() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let mut df = df!("name" => ["Ann \"The Hammer\"", "Bob"])?;
    parquet_examples::write_dataframe_to_parquet(
        &mut df,
        file.to_str().unwrap(),
        Compression::SNAPPY,
    )?;

    let expr = format!(
        "name == {}",
        parquet_examples::quote_expr_value("Ann \"The Hammer\"")
    );
    let count = parquet_examples::filter_count(file.to_str().unwrap(), &[expr.clone()])?;
    assert_eq!(count, 1);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let (tx, rx) = mpsc::channel();
    rt.block_on(background::filter_count(
        file.to_str().unwrap().to_string(),
        vec![expr],
        tx,
    ));
    let res = loop {
        if let Ok(msg) = rx.recv() {
            if let Ok(background::JobUpdate::Done(background::JobResult::Count(c))) = msg {
                break c;
            }
        }
    };
    assert_eq!(res, 1);
    Ok(())
}
