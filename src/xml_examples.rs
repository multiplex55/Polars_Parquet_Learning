use crate::{parquet_examples, xml_to_parquet};
use anyhow::Result;
use polars::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Parse the bundled `sample.xml` file into a [`Root`] struct.
pub fn parse_sample_xml() -> Result<xml_to_parquet::Root> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.xml");
    xml_to_parquet::parse_xml(path)
}

/// Convert the provided [`Root`] into normalised [`DataFrame`] tables.
pub fn root_to_tables(root: &xml_to_parquet::Root) -> Result<BTreeMap<&'static str, DataFrame>> {
    xml_to_parquet::flatten_to_tables(root)
}

/// Write the tables returned by [`root_to_tables`] to `output_dir`.
///
/// The files are named `<table>.parquet` and no schema file is emitted.
pub fn write_tables_to_parquet(tables: &BTreeMap<&str, DataFrame>, output_dir: &str) -> Result<()> {
    xml_to_parquet::write_tables(tables, output_dir, false)
}

/// Read previously written Parquet tables and rebuild the [`Root`] structure.
pub fn read_parquet_to_root(dir: &str) -> Result<xml_to_parquet::Root> {
    let p = Path::new(dir);
    let templates =
        parquet_examples::read_parquet_to_dataframe(p.join("templates.parquet").to_str().unwrap())?;
    let messages =
        parquet_examples::read_parquet_to_dataframe(p.join("messages.parquet").to_str().unwrap())?;
    let repositories = parquet_examples::read_parquet_to_dataframe(
        p.join("repositories.parquet").to_str().unwrap(),
    )?;

    let fields_path = p.join("fields.parquet");
    let fields = if fields_path.exists() {
        Some(parquet_examples::read_parquet_to_dataframe(
            fields_path.to_str().unwrap(),
        )?)
    } else {
        None
    };
    let parts_path = p.join("parts.parquet");
    let parts = if parts_path.exists() {
        Some(parquet_examples::read_parquet_to_dataframe(
            parts_path.to_str().unwrap(),
        )?)
    } else {
        None
    };

    // Build maps of foreign key -> rows
    let mut field_map: HashMap<u32, Vec<xml_to_parquet::Field>> = HashMap::new();
    if let Some(df) = &fields {
        let tids = df.column("template_id")?.u32()?;
        let names = df.column("name")?.str()?;
        let values = df.column("value")?.str()?;
        for i in 0..df.height() {
            let tid = tids.get(i).unwrap();
            let f = xml_to_parquet::Field {
                name: names.get(i).unwrap().to_string(),
                value: values.get(i).unwrap().to_string(),
            };
            field_map.entry(tid).or_default().push(f);
        }
    }

    let mut templates_vec = Vec::new();
    let ids = templates.column("id")?.u32()?;
    let names = templates.column("name")?.str()?;
    for i in 0..templates.height() {
        let id = ids.get(i).unwrap();
        let name = names.get(i).unwrap().to_string();
        let fields = field_map.remove(&id).unwrap_or_default();
        templates_vec.push(xml_to_parquet::Template { id, name, fields });
    }

    let mut part_map: HashMap<u32, Vec<xml_to_parquet::Part>> = HashMap::new();
    if let Some(df) = &parts {
        let mids = df.column("message_id")?.u32()?;
        let contents = df.column("content")?.str()?;
        for i in 0..df.height() {
            let mid = mids.get(i).unwrap();
            let p = xml_to_parquet::Part {
                content: contents.get(i).unwrap().to_string(),
            };
            part_map.entry(mid).or_default().push(p);
        }
    }

    let mut messages_vec = Vec::new();
    let msg_ids = messages.column("id")?.u32()?;
    let msg_tids = messages.column("template_id")?.u32()?;
    for i in 0..messages.height() {
        let id = msg_ids.get(i).unwrap();
        let template_id = msg_tids.get(i).unwrap();
        let parts = part_map.remove(&id).unwrap_or_default();
        messages_vec.push(xml_to_parquet::Message {
            id,
            template_id,
            parts,
        });
    }

    let mut repos_vec = Vec::new();
    let repo_ids = repositories.column("id")?.u32()?;
    let repo_paths = repositories.column("path")?.str()?;
    for i in 0..repositories.height() {
        let id = repo_ids.get(i).unwrap();
        let path = repo_paths.get(i).unwrap().to_string();
        repos_vec.push(xml_to_parquet::Repository { id, path });
    }

    Ok(xml_to_parquet::Root {
        templates: templates_vec,
        messages: messages_vec,
        repositories: repos_vec,
    })
}

/// Serialize the [`Root`] back into an XML string.
pub fn root_to_xml(root: &xml_to_parquet::Root) -> Result<String> {
    Ok(quick_xml::se::to_string(root)?)
}
