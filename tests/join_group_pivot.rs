use polars_parquet_learning::parquet_examples::{join_on_key, group_by_sum, pivot_wider, rolling_mean, write_dataframe_to_parquet};
use polars::prelude::*;

#[test]
fn join_two_frames() -> anyhow::Result<()> {
    let df1 = df!("id" => &[1i64,2,3], "a" => &[10i64,20,30])?;
    let df2 = df!("id" => &[2i64,3,4], "b" => &[200i64,300,400])?;
    let joined = join_on_key(&df1, &df2, "id")?;
    assert_eq!(joined.height(), 2);
    let ids: Vec<i64> = joined.column("id")?.i64()?.into_no_null_iter().collect();
    assert_eq!(ids, vec![2,3]);
    Ok(())
}

#[test]
fn group_and_sum() -> anyhow::Result<()> {
    let df = df!("cat" => ["x","x","y"], "val" => [1i64,2,3])?;
    let grouped = group_by_sum(&df, "cat", "val")?;
    assert_eq!(grouped.height(), 2);
    let vals: Vec<i64> = grouped.column("val_sum")?.i64()?.into_no_null_iter().collect();
    let cats: Vec<&str> = grouped.column("cat")?.str()?.into_no_null_iter().collect();
    assert_eq!(cats.len(), 2);
    assert!(cats.contains(&"x"));
    assert!(cats.contains(&"y"));
    assert_eq!(vals.iter().sum::<i64>(), 6);
    Ok(())
}

#[test]
fn pivot_to_wide() -> anyhow::Result<()> {
    let df = df!(
        "id" => &[1i64,1,2],
        "var" => &["A","B","A"],
        "val" => &[10i64,20,30]
    )?;
    let wide = pivot_wider(&df, "id", "var", "val")?;
    assert_eq!(wide.height(), 2);
    assert!(wide.column("A").is_ok());
    assert!(wide.column("B").is_ok());
    Ok(())
}

#[test]
fn rolling_window_mean() -> anyhow::Result<()> {
    use parquet::basic::Compression;
    use tempfile::tempdir;

    let dir = tempdir()?;
    let file = dir.path().join("data.parquet");

    let mut df = df!("val" => &[1.0f64, 2.0, 3.0, 4.0, 5.0])?;
    write_dataframe_to_parquet(&mut df, file.to_str().unwrap(), Compression::SNAPPY)?;

    let rolled = rolling_mean(file.to_str().unwrap(), "val", 2)?;
    let vals: Vec<f64> = rolled
        .column("rolling_mean")?
        .f64()?
        .into_no_null_iter()
        .collect();
    assert_eq!(vals, vec![1.0, 1.5, 2.5, 3.5, 4.5]);
    Ok(())
}
