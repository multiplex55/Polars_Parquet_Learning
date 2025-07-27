use polars::prelude::*;
use std::fs::File;
use polars_parquet_learning::cli::{
    self, Cli, Commands, ReadArgs, WriteArgs, CompressionArg, XmlArgs,
};
use parquet::basic::{Compression, GzipLevel};
use tempfile::tempdir;

#[test]
fn cli_read_runs() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("data.parquet");
    let mut df = df!("id" => &[1i64], "name" => &["a"]).unwrap();
    ParquetWriter::new(File::create(&file).unwrap())
        .finish(&mut df)
        .unwrap();

    let cli = Cli {
        command: Commands::Read(ReadArgs {
            file: file.to_str().unwrap().to_string(),
        }),
    };
    cli::run(cli).unwrap();
}

#[test]
fn cli_xml_creates_files() {
    let dir = tempdir().unwrap();
    let xml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.xml");
    let out_dir = dir.path().join("out");
    let cli = Cli {
        command: Commands::Xml(XmlArgs {
            input: xml_path.to_string(),
            output_dir: out_dir.to_str().unwrap().to_string(),
            schema: true,
        }),
    };
    cli::run(cli).unwrap();
    assert!(out_dir.join("templates.parquet").exists());
    assert!(out_dir.join("messages.parquet").exists());
}

#[test]
fn cli_write_with_compression() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("in.parquet");
    let output = dir.path().join("out.parquet");
    let mut df = df!("id" => &[1i64], "name" => &["a"]).unwrap();
    ParquetWriter::new(File::create(&input).unwrap())
        .finish(&mut df)
        .unwrap();

    let cli = Cli {
        command: Commands::Write(WriteArgs {
            input: input.to_str().unwrap().to_string(),
            output: output.to_str().unwrap().to_string(),
            compression: CompressionArg::Gzip,
        }),
    };
    cli::run(cli).unwrap();
    let meta =
        polars_parquet_learning::parquet_examples::read_parquet_metadata(output.to_str().unwrap())
            .unwrap();
    assert_eq!(
        meta.row_group(0)
            .column(0)
            .compression(),
        parquet::basic::Compression::GZIP(parquet::basic::GzipLevel::default())
    );
}
