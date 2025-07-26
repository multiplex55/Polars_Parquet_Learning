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
    let res1 = rt.block_on(background::read_dataframe_slice(file.to_str().unwrap().to_string(), 0, 50))?;
    let res2 = rt.block_on(background::read_dataframe_slice(file.to_str().unwrap().to_string(), 50, 50))?;
    if let background::JobResult::DataFrame(df1) = res1 {
        assert_eq!(df1.height(), 50);
    } else { panic!("unexpected result") }
    if let background::JobResult::DataFrame(df2) = res2 {
        assert_eq!(df2.height(), 50);
    } else { panic!("unexpected result") }
    Ok(())
}
