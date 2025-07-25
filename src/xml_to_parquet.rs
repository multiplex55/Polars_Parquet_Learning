use anyhow::Result;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize, Clone)]
pub struct Field {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@value")]
    pub value: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Template {
    #[serde(rename = "@id")]
    pub id: u32,
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "field", default)]
    pub fields: Vec<Field>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Part {
    #[serde(rename = "@content")]
    pub content: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Message {
    #[serde(rename = "@id")]
    pub id: u32,
    #[serde(rename = "@template_id")]
    pub template_id: u32,
    #[serde(rename = "part", default)]
    pub parts: Vec<Part>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Repository {
    #[serde(rename = "@id")]
    pub id: u32,
    #[serde(rename = "@path")]
    pub path: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename = "root")]
pub struct Root {
    #[serde(rename = "template", default)]
    pub templates: Vec<Template>,
    #[serde(rename = "message", default)]
    pub messages: Vec<Message>,
    #[serde(rename = "repository", default)]
    pub repositories: Vec<Repository>,
}

pub fn parse_xml(path: &str) -> Result<Root> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let root: Root = quick_xml::de::from_reader(reader)?;
    Ok(root)
}

pub fn flatten_to_tables(root: &Root) -> Result<BTreeMap<&'static str, DataFrame>> {
    let mut map = BTreeMap::new();

    // templates table
    let template_ids: Vec<u32> = root.templates.iter().map(|t| t.id).collect();
    let template_names: Vec<String> = root.templates.iter().map(|t| t.name.clone()).collect();
    let df_templates = df!("id" => template_ids, "name" => template_names)?;
    map.insert("templates", df_templates);

    // fields table
    let mut f_tid = Vec::new();
    let mut f_name = Vec::new();
    let mut f_value = Vec::new();
    for t in &root.templates {
        for f in &t.fields {
            f_tid.push(t.id);
            f_name.push(f.name.clone());
            f_value.push(f.value.clone());
        }
    }
    if !f_tid.is_empty() {
        let df_fields = df!("template_id" => f_tid, "name" => f_name, "value" => f_value)?;
        map.insert("fields", df_fields);
    }

    // messages table
    let msg_ids: Vec<u32> = root.messages.iter().map(|m| m.id).collect();
    let msg_tids: Vec<u32> = root.messages.iter().map(|m| m.template_id).collect();
    let df_messages = df!("id" => msg_ids, "template_id" => msg_tids)?;
    map.insert("messages", df_messages);

    // parts table
    let mut p_mid = Vec::new();
    let mut p_content = Vec::new();
    for m in &root.messages {
        for p in &m.parts {
            p_mid.push(m.id);
            p_content.push(p.content.clone());
        }
    }
    if !p_mid.is_empty() {
        let df_parts = df!("message_id" => p_mid, "content" => p_content)?;
        map.insert("parts", df_parts);
    }

    // repositories table
    let repo_ids: Vec<u32> = root.repositories.iter().map(|r| r.id).collect();
    let repo_paths: Vec<String> = root.repositories.iter().map(|r| r.path.clone()).collect();
    let df_repos = df!("id" => repo_ids, "path" => repo_paths)?;
    map.insert("repositories", df_repos);

    Ok(map)
}

#[derive(Serialize)]
struct ForeignKey {
    table: String,
    column: String,
    references: String,
}

#[derive(Serialize)]
struct ExportSchema {
    foreign_keys: Vec<ForeignKey>,
}

pub fn write_tables(tables: &BTreeMap<&str, DataFrame>, output_dir: &str, write_schema: bool) -> Result<()> {
    use std::fs::File;
    use std::path::Path;
    std::fs::create_dir_all(output_dir)?;

    let mut fks = Vec::new();
    if tables.contains_key("fields") {
        fks.push(ForeignKey {
            table: "fields".into(),
            column: "template_id".into(),
            references: "templates.id".into(),
        });
    }
    if tables.contains_key("parts") {
        fks.push(ForeignKey {
            table: "parts".into(),
            column: "message_id".into(),
            references: "messages.id".into(),
        });
    }

    for (name, df) in tables {
        let path = Path::new(output_dir).join(format!("{name}.parquet"));
        let file = File::create(path)?;
        let mut df = df.clone();
        ParquetWriter::new(file).finish(&mut df)?;
    }

    if write_schema && !fks.is_empty() {
        let schema = ExportSchema { foreign_keys: fks };
        let path = Path::new(output_dir).join("_schema.json");
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &schema)?;
    }
    Ok(())
}

pub fn xml_to_parquet(input: &str, output_dir: &str, write_schema: bool) -> Result<()> {
    let root = parse_xml(input)?;
    let tables = flatten_to_tables(&root)?;
    write_tables(&tables, output_dir, write_schema)
}

