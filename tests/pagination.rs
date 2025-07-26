use polars::prelude::*;
use tempfile::tempdir;
use Polars_Parquet_Learning::{parquet_examples, background};

#[test]
fn paginate_multiple_pages() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let ids: Vec<i64> = (0..120).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    let mut df = df!("id" => ids, "name" => names)?;
    parquet_examples::write_dataframe_to_parquet(&mut df, file.to_str().unwrap())?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let (tx1, rx1) = std::sync::mpsc::channel();
    rt.block_on(background::read_dataframe_slice(
        file.to_str().unwrap().to_string(),
        0,
        50,
        tx1,
    ));
    let res1 = loop {
        if let Ok(msg) = rx1.recv() {
            if let Ok(background::JobUpdate::Done(res)) = msg {
                break res;
            }
        }
    };
    let (tx2, rx2) = std::sync::mpsc::channel();
    rt.block_on(background::read_dataframe_slice(
        file.to_str().unwrap().to_string(),
        50,
        50,
        tx2,
    ));
    let res2 = loop {
        if let Ok(msg) = rx2.recv() {
            if let Ok(background::JobUpdate::Done(res)) = msg {
                break res;
            }
        }
    };
    if let background::JobResult::DataFrame(df1) = res1 {
        assert_eq!(df1.height(), 50);
    } else { panic!("unexpected result") }
    if let background::JobResult::DataFrame(df2) = res2 {
        assert_eq!(df2.height(), 50);
    } else { panic!("unexpected result") }
    Ok(())
}

#[test]
fn paginate_filtered_pages() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let ids: Vec<i64> = (0..100).collect();
    let names: Vec<String> = ids
        .iter()
        .map(|i| if i % 2 == 0 { "yes".to_string() } else { "no".to_string() })
        .collect();
    let mut df = df!("id" => ids, "name" => names)?;
    parquet_examples::write_dataframe_to_parquet(&mut df, file.to_str().unwrap())?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let exprs = vec!["name == \"yes\"".to_string()];

    let (tx1, rx1) = std::sync::mpsc::channel();
    rt.block_on(background::read_filter_slice(
        file.to_str().unwrap().to_string(),
        exprs.clone(),
        0,
        20,
        tx1,
    ));
    let res1 = loop {
        if let Ok(msg) = rx1.recv() {
            if let Ok(background::JobUpdate::Done(res)) = msg {
                break res;
            }
        }
    };
    let (tx2, rx2) = std::sync::mpsc::channel();
    rt.block_on(background::read_filter_slice(
        file.to_str().unwrap().to_string(),
        exprs.clone(),
        20,
        20,
        tx2,
    ));
    let res2 = loop {
        if let Ok(msg) = rx2.recv() {
            if let Ok(background::JobUpdate::Done(res)) = msg {
                break res;
            }
        }
    };
    if let background::JobResult::DataFrame(df1) = res1 {
        assert_eq!(df1.height(), 20);
    } else { panic!("unexpected result") }
    if let background::JobResult::DataFrame(df2) = res2 {
        assert_eq!(df2.height(), 20);
    } else { panic!("unexpected result") }
    Ok(())
}

#[test]
fn paginate_filtered_contains() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let ids: Vec<i64> = (0..120).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    let mut df = df!("id" => ids, "name" => names)?;
    parquet_examples::write_dataframe_to_parquet(&mut df, file.to_str().unwrap())?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let exprs = vec!["name contains \"1\"".to_string()];

    let (tx1, rx1) = std::sync::mpsc::channel();
    rt.block_on(background::read_filter_slice(
        file.to_str().unwrap().to_string(),
        exprs.clone(),
        0,
        30,
        tx1,
    ));
    let res1 = loop {
        if let Ok(msg) = rx1.recv() {
            if let Ok(background::JobUpdate::Done(res)) = msg {
                break res;
            }
        }
    };
    let (tx2, rx2) = std::sync::mpsc::channel();
    rt.block_on(background::read_filter_slice(
        file.to_str().unwrap().to_string(),
        exprs.clone(),
        30,
        30,
        tx2,
    ));
    let res2 = loop {
        if let Ok(msg) = rx2.recv() {
            if let Ok(background::JobUpdate::Done(res)) = msg {
                break res;
            }
        }
    };
    if let background::JobResult::DataFrame(df1) = res1 {
        assert_eq!(df1.height(), 30);
    } else { panic!("unexpected result") }
    if let background::JobResult::DataFrame(df2) = res2 {
        assert_eq!(df2.height(), 30);
    } else { panic!("unexpected result") }
    Ok(())
}
