use polars::prelude::*;
use Polars_Parquet_Learning::parquet_examples::summarize_dataframe;

#[test]
fn summary_reports_size() -> anyhow::Result<()> {
    let df = df!("id" => &[1i64, 2, 3], "name" => &["a", "b", "c"])?;
    let summary = summarize_dataframe(&df)?;
    let est = df.estimated_size();
    let diff = if summary.approx_bytes > est {
        summary.approx_bytes - est
    } else {
        est - summary.approx_bytes
    };
    assert!(diff <= est / 10);
    Ok(())
}
