use polars::prelude::*;
use Polars_Parquet_Learning::xml_dynamic::{parse_any_xml, value_to_tables};

#[test]
fn parse_sample_xml() -> anyhow::Result<()> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.xml");
    let value = parse_any_xml(path)?;
    assert_eq!(value["templates"][0]["id"], 1);
    assert_eq!(value["templates"][0]["fields"][0]["name"], "f1");
    assert_eq!(value["messages"][0]["parts"][1]["content"], "b");
    assert_eq!(value["repositories"][0]["path"], "/tmp");
    Ok(())
}

#[test]
fn value_to_tables_matches_expectations() -> anyhow::Result<()> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.xml");
    let value = parse_any_xml(path)?;
    let tables = value_to_tables(&value)?;
    assert!(tables.contains_key("templates"));
    assert!(tables.contains_key("fields"));
    assert!(tables.contains_key("messages"));
    assert!(tables.contains_key("parts"));
    assert!(tables.contains_key("repositories"));

    let templates = &tables["templates"];
    assert_eq!(templates.column("id")?.u32()?.get(0), Some(1));
    assert_eq!(templates.column("name")?.str()?.get(0), Some("temp"));

    let fields = &tables["fields"];
    assert_eq!(fields.height(), 2);
    assert_eq!(fields.column("name")?.str()?.get(0), Some("f1"));
    assert_eq!(fields.column("value")?.str()?.get(1), Some("v2"));

    let messages = &tables["messages"];
    assert_eq!(messages.column("id")?.u32()?.get(0), Some(10));
    assert_eq!(messages.column("template_id")?.u32()?.get(0), Some(1));

    let parts = &tables["parts"];
    assert_eq!(parts.height(), 2);
    assert_eq!(parts.column("content")?.str()?.get(1), Some("b"));

    let repos = &tables["repositories"];
    assert_eq!(repos.column("id")?.u32()?.get(0), Some(100));
    assert_eq!(repos.column("path")?.str()?.get(0), Some("/tmp"));

    Ok(())
}
