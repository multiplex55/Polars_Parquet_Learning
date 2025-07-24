use polars::prelude::*;
use std::fs::File;
use std::process::Command;
use tempfile::tempdir;

#[test]
fn cli_read_runs() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("data.parquet");
    let mut df = df!("id" => &[1i64], "name" => &["a"]).unwrap();
    ParquetWriter::new(File::create(&file).unwrap())
        .finish(&mut df)
        .unwrap();

    let exe = env!("CARGO_BIN_EXE_Polars_Parquet_Learning");
    let status = Command::new(exe)
        .args(["read", file.to_str().unwrap()])
        .status()
        .expect("run");
    assert!(status.success());
}
