use tempfile::tempdir;

use polars_parquet_learning::{xml_examples, xml_to_parquet};

#[test]
fn xml_examples_round_trip() -> anyhow::Result<()> {
    let root = xml_examples::parse_sample_xml()?;
    let tables = xml_examples::root_to_tables(&root)?;
    let dir = tempdir()?;
    xml_examples::write_tables_to_parquet(&tables, dir.path().to_str().unwrap())?;

    let round = xml_examples::read_parquet_to_root(dir.path().to_str().unwrap())?;
    assert_eq!(round, root);

    let round_tables = xml_examples::root_to_tables(&round)?;
    assert_eq!(tables.keys().count(), round_tables.keys().count());

    let xml = xml_examples::root_to_xml(&round)?;
    let root2: xml_to_parquet::Root = quick_xml::de::from_str(&xml)?;
    assert_eq!(root2.templates.len(), root.templates.len());
    Ok(())
}
