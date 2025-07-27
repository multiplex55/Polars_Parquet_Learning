use Polars_Parquet_Learning::parquet_examples::correlation_matrix;
use polars::prelude::*;

#[test]
fn matrix_basic() -> anyhow::Result<()> {
    let df = df!(
        "a" => &[1.0f64, 2.0, 3.0],
        "b" => &[1.0f64, 2.0, 3.0],
        "c" => &[3.0f64, 2.0, 1.0]
    )?;
    let corr = correlation_matrix(&df, &["a", "b", "c"])?;
    assert_eq!(corr.shape(), (3, 4));
    let labels: Vec<&str> = corr
        .column("column")?
        .str()?
        .into_iter()
        .map(|o| o.unwrap())
        .collect();
    assert_eq!(labels, vec!["a", "b", "c"]);
    let col_b: Vec<f64> = corr
        .column("b")?
        .f64()?
        .into_iter()
        .map(|o| o.unwrap())
        .collect();
    assert_eq!(col_b, vec![1.0, 1.0, -1.0]);
    Ok(())
}

#[test]
fn matrix_empty_columns() -> anyhow::Result<()> {
    let df = df!("x" => &[1.0f64, 2.0])?;
    let corr = correlation_matrix(&df, &[])?;
    assert_eq!(corr.shape(), (0, 0));
    Ok(())
}

#[test]
fn matrix_symmetry() -> anyhow::Result<()> {
    let df = df!(
        "a" => &[1.0f64, 2.0, 3.0],
        "b" => &[3.0f64, 2.0, 1.0]
    )?;
    let corr = correlation_matrix(&df, &["a", "b"])?;
    let ab = corr.column("b")?.f64()?.get(0).unwrap();
    let ba = corr.column("a")?.f64()?.get(1).unwrap();
    let aa = corr.column("a")?.f64()?.get(0).unwrap();
    let bb = corr.column("b")?.f64()?.get(1).unwrap();
    assert!((ab - ba).abs() < 1e-12);
    assert!((aa - 1.0).abs() < 1e-12);
    assert!((bb - 1.0).abs() < 1e-12);
    Ok(())
}

#[test]
fn matrix_with_nulls() -> anyhow::Result<()> {
    let df = df!(
        "a" => &[Some(1.0f64), None, Some(3.0)],
        "b" => &[Some(1.0f64), Some(2.0), None]
    )?;
    let corr = correlation_matrix(&df, &["a", "b"])?;
    assert_eq!(corr.shape(), (2, 3));
    let ab = corr.column("b")?.f64()?.get(0).unwrap();
    let ba = corr.column("a")?.f64()?.get(1).unwrap();
    assert!(ab.is_nan());
    assert!(ba.is_nan());
    Ok(())
}
