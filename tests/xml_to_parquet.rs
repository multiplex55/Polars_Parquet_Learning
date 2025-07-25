use polars::prelude::*;
use tempfile::tempdir;

use Polars_Parquet_Learning::xml_to_parquet::{flatten_to_tables, parse_xml, write_tables};

#[test]
fn xml_round_trip() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let xml_path = dir.path().join("sample.xml");
    std::fs::write(
        &xml_path,
        r#"<root>
    <template id='1' name='temp'>
        <field name='f1' value='v1'/>
    </template>
    <message id='10' template_id='1'>
        <part content='a'/>
        <part content='b'/>
    </message>
    <repository id='100' path='/tmp'/>
</root>"#,
    )?;

    let root = parse_xml(xml_path.to_str().unwrap())?;
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
