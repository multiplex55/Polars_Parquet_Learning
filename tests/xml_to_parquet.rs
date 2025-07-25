use polars::prelude::*;
use tempfile::tempdir;

use Polars_Parquet_Learning::xml_to_parquet::{flatten_to_tables, parse_xml, write_tables};

#[test]
fn xml_round_trip() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let xml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.xml");

    let root = parse_xml(xml_path)?;
    let tables = flatten_to_tables(&root)?;
    assert!(tables.contains_key("templates"));
    assert!(tables.contains_key("fields"));
    assert!(tables.contains_key("messages"));
    assert!(tables.contains_key("parts"));
    assert!(tables.contains_key("repositories"));

    let templates = &tables["templates"];
    assert_eq!(templates.column("id")?.u32()?.get(0), Some(1));

    let output_dir = dir.path().join("out");
    write_tables(&tables, output_dir.to_str().unwrap(), true)?;
    assert!(output_dir.join("templates.parquet").exists());
    assert!(output_dir.join("_schema.json").exists());
    Ok(())
}
