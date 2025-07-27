use polars::prelude::*;
use tempfile::tempdir;
use polars_parquet_learning::xml_examples;

#[test]
fn read_parquet_with_nulls_returns_error() -> anyhow::Result<()> {
    let dir = tempdir()?;
    // create minimal tables with a null value in templates.name
    let mut templates = df!("id" => &[1u32], "name" => &[Option::<&str>::None])?;
    let mut messages = df!("id" => &[1u32], "template_id" => &[1u32])?;
    let mut repos = df!("id" => &[1u32], "path" => &["/tmp"])?;

    ParquetWriter::new(std::fs::File::create(dir.path().join("templates.parquet"))?)
        .finish(&mut templates)?;
    ParquetWriter::new(std::fs::File::create(dir.path().join("messages.parquet"))?)
        .finish(&mut messages)?;
    ParquetWriter::new(std::fs::File::create(dir.path().join("repositories.parquet"))?)
        .finish(&mut repos)?;

    let result = xml_examples::read_parquet_to_root(dir.path().to_str().unwrap());
    assert!(result.is_err());
    Ok(())
}
