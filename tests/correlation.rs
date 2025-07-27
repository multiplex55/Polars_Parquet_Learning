use polars::prelude::*;
use Polars_Parquet_Learning::parquet_examples::correlation_matrix;

#[test]
fn matrix_basic() -> anyhow::Result<()> {
    let df = df!(
        "a" => &[1.0f64, 2.0, 3.0],
        "b" => &[1.0f64, 2.0, 3.0],
        "c" => &[3.0f64, 2.0, 1.0]
    )?;
    let corr = correlation_matrix(&df, &["a", "b", "c"])?;
    assert_eq!(corr.shape(), (3, 4));
    let labels: Vec<&str> = corr.column("column")?.str()?.into_no_null_iter().collect();
    assert_eq!(labels, vec!["a", "b", "c"]);
    let col_b: Vec<f64> = corr.column("b")?.f64()?.into_no_null_iter().collect();
    assert_eq!(col_b, vec![1.0, 1.0, -1.0]);
    Ok(())
}
