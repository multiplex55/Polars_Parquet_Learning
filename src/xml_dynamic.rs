use crate::xml_to_parquet::{self, Root};
use anyhow::Result;
use polars::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;

/// Parse any XML file supported by [`xml_to_parquet`] and return a dynamic
/// [`serde_json::Value`] representation.
pub fn parse_any_xml(path: &str) -> Result<Value> {
    let root: Root = xml_to_parquet::parse_xml(path)?;
    Ok(serde_json::to_value(root)?)
}

/// Convert a [`serde_json::Value`] produced by [`parse_any_xml`] into
/// normalized [`DataFrame`] tables.
pub fn value_to_tables(value: &Value) -> Result<BTreeMap<String, DataFrame>> {
    let root: Root = serde_json::from_value(value.clone())?;
    let tables = xml_to_parquet::flatten_to_tables(&root)?;
    Ok(tables
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect())
}
