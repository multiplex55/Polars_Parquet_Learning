use Polars_Parquet_Learning::{background, parquet_examples};
use polars::prelude::*;
use parquet::basic::Compression;
use tempfile::tempdir;

#[test]
fn paginate_multiple_pages() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let ids: Vec<i64> = (0..120).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    let mut df = df!("id" => ids, "name" => names)?;
    parquet_examples::write_dataframe_to_parquet(
        &mut df,
        file.to_str().unwrap(),
        parquet::basic::Compression::SNAPPY,
    )?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let (tx1, rx1) = std::sync::mpsc::channel();
    rt.block_on(background::read_dataframe_slice(
        file.to_str().unwrap().to_string(),
        0,
        40,
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
        40,
        40,
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
        assert_eq!(df1.height(), 40);
    } else {
        panic!("unexpected result")
    }
    if let background::JobResult::DataFrame(df2) = res2 {
        assert_eq!(df2.height(), 40);
    } else {
        panic!("unexpected result")
    }
    Ok(())
}

#[test]
fn paginate_filtered_pages() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let ids: Vec<i64> = (0..100).collect();
    let names: Vec<String> = ids
        .iter()
        .map(|i| {
            if i % 2 == 0 {
                "yes".to_string()
            } else {
                "no".to_string()
            }
        })
        .collect();
    let mut df = df!("id" => ids, "name" => names)?;
    parquet_examples::write_dataframe_to_parquet(
        &mut df,
        file.to_str().unwrap(),
        parquet::basic::Compression::SNAPPY,
    )?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let exprs = vec!["name == \"yes\"".to_string()];

    let (tx1, rx1) = std::sync::mpsc::channel();
    rt.block_on(background::read_filter_slice(
        file.to_str().unwrap().to_string(),
        exprs.clone(),
        0,
        15,
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
        15,
        15,
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
        assert_eq!(df1.height(), 15);
    } else {
        panic!("unexpected result")
    }
    if let background::JobResult::DataFrame(df2) = res2 {
        assert_eq!(df2.height(), 15);
    } else {
        panic!("unexpected result")
    }
    Ok(())
}

#[test]
fn paginate_filtered_contains() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let ids: Vec<i64> = (0..120).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    let mut df = df!("id" => ids, "name" => names)?;
    parquet_examples::write_dataframe_to_parquet(
        &mut df,
        file.to_str().unwrap(),
        parquet::basic::Compression::SNAPPY,
    )?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let exprs = vec!["name contains \"1\"".to_string()];

    let (tx1, rx1) = std::sync::mpsc::channel();
    rt.block_on(background::read_filter_slice(
        file.to_str().unwrap().to_string(),
        exprs.clone(),
        0,
        25,
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
        25,
        25,
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
        assert_eq!(df1.height(), 25);
    } else {
        panic!("unexpected result")
    }
    if let background::JobResult::DataFrame(df2) = res2 {
        assert_eq!(df2.height(), 14);
    } else {
        panic!("unexpected result")
    }
    Ok(())
}

#[test]
fn paginate_filtered_quotes() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let mut df = df!("name" => ["Ann \"The Hammer\"", "Bob"])?;
    parquet_examples::write_dataframe_to_parquet(
        &mut df,
        file.to_str().unwrap(),
        Compression::SNAPPY,
    )?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let exprs = vec![format!(
        "name == {}",
        parquet_examples::quote_expr_value("Ann \"The Hammer\"")
    )];

    let (tx, rx) = std::sync::mpsc::channel();
    rt.block_on(background::read_filter_slice(
        file.to_str().unwrap().to_string(),
        exprs.clone(),
        0,
        10,
        tx,
    ));
    let res = loop {
        if let Ok(msg) = rx.recv() {
            if let Ok(background::JobUpdate::Done(res)) = msg {
                break res;
            }
        }
    };
    if let background::JobResult::DataFrame(df) = res {
        assert_eq!(df.height(), 1);
    } else {
        panic!("unexpected result");
    }
    Ok(())
}

#[test]
fn prefetch_reuses_cache() -> anyhow::Result<()> {
    use std::collections::HashMap;

    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let ids: Vec<i64> = (0..50).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    let mut df = df!("id" => ids, "name" => names)?;
    parquet_examples::write_dataframe_to_parquet(
        &mut df,
        file.to_str().unwrap(),
        parquet::basic::Compression::SNAPPY,
    )?;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut cache: HashMap<usize, DataFrame> = HashMap::new();

    let (tx1, rx1) = std::sync::mpsc::channel();
    rt.block_on(background::read_dataframe_slice(
        file.to_str().unwrap().to_string(),
        0,
        20,
        tx1,
    ));
    let df1 = loop {
        if let Ok(msg) = rx1.recv() {
            if let Ok(background::JobUpdate::Done(background::JobResult::DataFrame(df))) = msg {
                break df;
            }
        }
    };
    cache.insert(0, df1);

    let (tx2, rx2) = std::sync::mpsc::channel();
    rt.block_on(background::read_dataframe_slice(
        file.to_str().unwrap().to_string(),
        20,
        20,
        tx2,
    ));
    let df2 = loop {
        if let Ok(msg) = rx2.recv() {
            if let Ok(background::JobUpdate::Done(background::JobResult::DataFrame(df))) = msg {
                break df;
            }
        }
    };
    cache.insert(20, df2.clone());

    if let Some(cached) = cache.get(&20) {
        assert_eq!(cached.height(), 20);
    } else {
        panic!("page not cached");
    }
    assert_eq!(cache.len(), 2);
    Ok(())
}
