//! Simple egui application for interacting with Parquet files.

// Expose example functions for GUI callbacks or tests.
pub mod background;
pub mod cli;
pub mod parquet_examples;

use crate::search;
use crate::{xml_dynamic, xml_to_parquet};
use anyhow::Result;
use background::{JobResult, JobUpdate};
use clap::Parser;
use eframe::egui;
use egui_extras::{Column as TableColumn, TableBuilder};
#[cfg(feature = "plotting")]
use egui_plot::{BarChart, BoxElem, BoxPlot, Line, Plot, PlotPoints, Points};
use polars::prelude::SortMultipleOptions;
use polars::prelude::*;
use rfd::FileDialog;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;
use std::sync::mpsc;

/// Defines the user selected operation on the Parquet file.
#[derive(Debug, PartialEq)]
enum Operation {
    Read,
    Modify,
    Write,
    /// Write the DataFrame to CSV
    WriteCsv,
    /// Write the DataFrame to JSON
    WriteJson,
    /// Create a new DataFrame in memory
    Create,
    /// Partition the currently loaded DataFrame by a column
    Partition,
    /// Run a simple query against the file
    Query,
    /// Convert an XML file to Parquet tables
    Xml,
    /// Dynamically parse XML and map fields
    XmlDynamic,
    /// Compute correlation matrix for numeric columns
    Correlation,
}

impl Default for Operation {
    fn default() -> Self {
        Operation::Read
    }
}

#[cfg(feature = "plotting")]
#[derive(Debug, PartialEq)]
enum PlotType {
    Histogram,
    Line,
    Scatter,
    BoxPlot,
}

#[cfg(feature = "plotting")]
impl Default for PlotType {
    fn default() -> Self {
        PlotType::Histogram
    }
}

#[derive(Default, Clone, Copy, PartialEq)]
enum FilterOp {
    #[default]
    Equals,
    Contains,
}

#[derive(Default, Clone)]
struct ColumnFilter {
    op: FilterOp,
    value: String,
}

/// Main application state.
struct ParquetApp {
    /// Path to the Parquet file entered by the user.
    file_path: String,
    /// Path used when saving created or partitioned data
    save_path: String,
    /// The operation the user would like to perform.
    operation: Operation,
    /// DataFrame currently being edited/created
    edit_df: Option<polars::prelude::DataFrame>,
    /// Column definitions for creating a new DataFrame
    schema: Vec<(String, polars::prelude::DataType)>,
    /// Working rows for the DataFrame editor
    rows: Vec<Vec<String>>,
    /// Selected column for rename/reorder operations
    rename_column: String,
    /// New name when renaming a column
    new_name: String,
    /// Current order of columns
    column_order: Vec<String>,
    /// Temporary inputs for adding columns
    new_col_name: String,
    new_col_type: String,
    /// Selected column when partitioning via write mode
    partition_column: Option<String>,
    /// Selected columns when using the dedicated partition operation
    partition_columns: Vec<String>,
    /// Selected numeric columns for correlation matrix
    correlation_columns: Vec<String>,
    /// Query prefix when using the query operation
    query_prefix: String,
    /// Expression string when using the query operation
    query_expr: String,
    /// Per-column filters
    column_filters: std::collections::HashMap<String, ColumnFilter>,
    /// Status message shown to the user
    status: String,
    /// Number of rows to display from the current DataFrame
    display_rows: usize,
    /// Start index of the current page
    page_start: usize,
    /// Number of rows per page
    page_size: usize,
    /// Total rows in the source file
    total_rows: usize,
    /// Cached pages of data
    page_cache: std::collections::HashMap<usize, DataFrame>,
    /// Least recently used order for cached pages
    cache_order: std::collections::VecDeque<usize>,
    /// Start index of a page currently being prefetched
    prefetch_start: Option<usize>,
    #[cfg(feature = "plotting")]
    /// Selected column to plot
    plot_column: Option<String>,
    #[cfg(feature = "plotting")]
    /// Second column for scatter plots
    plot_y_column: Option<String>,
    #[cfg(feature = "plotting")]
    /// Type of plot to display
    plot_type: PlotType,
    #[cfg(feature = "plotting")]
    /// Number of bins for histogram plots
    hist_bins: usize,
    #[cfg(feature = "plotting")]
    /// Optional x-axis range for plots
    x_range: Option<(f64, f64)>,
    #[cfg(feature = "plotting")]
    /// Optional y-axis range for plots
    y_range: Option<(f64, f64)>,
    /// Tokio runtime for background tasks
    runtime: tokio::runtime::Runtime,
    /// Receives updates from background jobs
    result_rx: Option<std::sync::mpsc::Receiver<anyhow::Result<background::JobUpdate>>>,
    /// Metadata for the loaded Parquet file
    metadata: Option<parquet::file::metadata::ParquetMetaData>,
    /// Schema of the loaded DataFrame
    loaded_schema: Vec<(String, polars::prelude::DataType)>,
    /// Show or hide the schema information
    show_schema: bool,
    /// Indicates an operation is running
    busy: bool,
    /// Current progress value if known
    progress: Option<f32>,
    /// Treat the selected path as a directory when reading
    use_directory: bool,
    /// Write an additional _schema.json when converting XML
    xml_schema: bool,
    /// Parsed XML value when using XmlDynamic
    xml_value: Option<serde_json::Value>,
    /// DataFrames produced from the XML
    xml_tables: std::collections::BTreeMap<String, DataFrame>,
    /// Selected table names for conversion
    xml_selected: Vec<String>,
    /// Active table being edited
    xml_active_table: Option<String>,
    /// Current search text for highlighting cells
    search_text: String,
    /// Whether search is case-insensitive
    search_ignore_case: bool,
    /// Coordinates of matches in the current page
    search_matches: Vec<(usize, usize)>,
    /// Index of the active match
    search_index: usize,
}

impl Default for ParquetApp {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            save_path: String::new(),
            operation: Operation::default(),
            edit_df: None,
            schema: Vec::new(),
            rows: Vec::new(),
            rename_column: String::new(),
            new_name: String::new(),
            column_order: Vec::new(),
            new_col_name: String::new(),
            new_col_type: String::new(),
            partition_column: None,
            partition_columns: Vec::new(),
            correlation_columns: Vec::new(),
            query_prefix: String::new(),
            query_expr: String::new(),
            column_filters: std::collections::HashMap::new(),
            status: String::new(),
            display_rows: 5,
            page_start: 0,
            page_size: 100,
            total_rows: 0,
            page_cache: std::collections::HashMap::new(),
            cache_order: std::collections::VecDeque::new(),
            prefetch_start: None,
            #[cfg(feature = "plotting")]
            plot_column: None,
            #[cfg(feature = "plotting")]
            plot_y_column: None,
            #[cfg(feature = "plotting")]
            plot_type: PlotType::default(),
            #[cfg(feature = "plotting")]
            hist_bins: 10,
            #[cfg(feature = "plotting")]
            x_range: None,
            #[cfg(feature = "plotting")]
            y_range: None,
            runtime: tokio::runtime::Runtime::new().expect("runtime"),
            result_rx: None,
            metadata: None,
            loaded_schema: Vec::new(),
            show_schema: false,
            busy: false,
            progress: None,
            use_directory: false,
            xml_schema: false,
            xml_value: None,
            xml_tables: std::collections::BTreeMap::new(),
            xml_selected: Vec::new(),
            xml_active_table: None,
            search_text: String::new(),
            search_ignore_case: false,
            search_matches: Vec::new(),
            search_index: 0,
        }
    }
}

impl ParquetApp {
    /// Create a new [`ParquetApp`] instance.
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
}

fn parse_dtype(t: &str) -> anyhow::Result<polars::prelude::DataType> {
    use polars::prelude::{DataType, TimeUnit};
    match t.trim().to_lowercase().as_str() {
        "int" | "i64" => Ok(DataType::Int64),
        "str" | "string" => Ok(DataType::String),
        "float" | "f64" => Ok(DataType::Float64),
        "bool" | "boolean" => Ok(DataType::Boolean),
        "date" => Ok(DataType::Date),
        "datetime" | "timestamp" => Ok(DataType::Datetime(TimeUnit::Microseconds, None)),
        "time" => Ok(DataType::Time),
        _ => Err(anyhow::anyhow!("unsupported type")),
    }
}

fn build_dataframe(schema: &[(String, DataType)], rows: &[Vec<String>]) -> Result<DataFrame> {
    use polars::prelude::IntoColumn;
    use std::collections::HashSet;

    // Ensure column names are unique before building the DataFrame
    let mut seen: HashSet<&str> = HashSet::new();
    for (name, _) in schema {
        if !seen.insert(name) {
            return Err(anyhow::anyhow!("duplicate column name '{name}'"));
        }
    }

    let mut cols: Vec<Column> = Vec::new();
    for (idx, (name, dtype)) in schema.iter().enumerate() {
        match dtype {
            DataType::Int64 => {
                let data: Vec<i64> = rows
                    .iter()
                    .map(|r| -> Result<i64> {
                        let val = r
                            .get(idx)
                            .ok_or_else(|| anyhow::anyhow!("missing value in column {}", name))?;
                        val.parse::<i64>()
                            .map_err(|e| anyhow::anyhow!("failed to parse '{val}' as i64: {e}"))
                    })
                    .collect::<Result<_>>()?;
                cols.push(Series::new(name.as_str().into(), data).into_column());
            }
            DataType::Float64 => {
                let data: Vec<f64> = rows
                    .iter()
                    .map(|r| -> Result<f64> {
                        let val = r
                            .get(idx)
                            .ok_or_else(|| anyhow::anyhow!("missing value in column {}", name))?;
                        val.parse::<f64>()
                            .map_err(|e| anyhow::anyhow!("failed to parse '{val}' as f64: {e}"))
                    })
                    .collect::<Result<_>>()?;
                cols.push(Series::new(name.as_str().into(), data).into_column());
            }
            DataType::String => {
                let data: Vec<String> = rows
                    .iter()
                    .map(|r| r.get(idx).cloned().unwrap_or_default())
                    .collect();
                cols.push(Series::new(name.as_str().into(), data).into_column());
            }
            DataType::Boolean => {
                let data: Vec<bool> = rows
                    .iter()
                    .map(|r| -> Result<bool> {
                        let val = r
                            .get(idx)
                            .ok_or_else(|| anyhow::anyhow!("missing value in column {}", name))?;
                        match val.to_lowercase().as_str() {
                            "true" | "1" => Ok(true),
                            "false" | "0" => Ok(false),
                            _ => Err(anyhow::anyhow!("failed to parse '{val}' as bool")),
                        }
                    })
                    .collect::<Result<_>>()?;
                cols.push(Series::new(name.as_str().into(), data).into_column());
            }
            DataType::Date => {
                use chrono::NaiveDate;
                let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                let data: Vec<i32> = rows
                    .iter()
                    .map(|r| -> Result<i32> {
                        let val = r
                            .get(idx)
                            .ok_or_else(|| anyhow::anyhow!("missing value in column {}", name))?;
                        let d = NaiveDate::parse_from_str(val, "%Y-%m-%d")
                            .map_err(|e| anyhow::anyhow!("failed to parse '{val}' as date: {e}"))?;
                        Ok((d - epoch).num_days() as i32)
                    })
                    .collect::<Result<_>>()?;
                cols.push(Series::new(name.as_str().into(), data).into_column());
            }
            DataType::Datetime(_, _) => {
                use chrono::{DateTime, NaiveDateTime};
                let data: Vec<i64> = rows
                    .iter()
                    .map(|r| -> Result<i64> {
                        let val = r
                            .get(idx)
                            .ok_or_else(|| anyhow::anyhow!("missing value in column {}", name))?;
                        let ts = DateTime::parse_from_rfc3339(val)
                            .map(|dt| dt.timestamp_micros())
                            .or_else(|_| {
                                NaiveDateTime::parse_from_str(val, "%Y-%m-%d %H:%M:%S")
                                    .or_else(|_| {
                                        NaiveDateTime::parse_from_str(val, "%Y-%m-%dT%H:%M:%S")
                                    })
                                    .map(|dt| dt.timestamp_micros())
                            })
                            .map_err(|e| {
                                anyhow::anyhow!("failed to parse '{val}' as datetime: {e}")
                            })?;
                        Ok(ts)
                    })
                    .collect::<Result<_>>()?;
                cols.push(Series::new(name.as_str().into(), data).into_column());
            }
            DataType::Time => {
                use chrono::NaiveTime;
                use chrono::Timelike;
                let data: Vec<i64> = rows
                    .iter()
                    .map(|r| -> Result<i64> {
                        let val = r
                            .get(idx)
                            .ok_or_else(|| anyhow::anyhow!("missing value in column {}", name))?;
                        let t = NaiveTime::parse_from_str(val, "%H:%M:%S")
                            .map_err(|e| anyhow::anyhow!("failed to parse '{val}' as time: {e}"))?;
                        Ok((t.num_seconds_from_midnight() as i64) * 1_000_000_000
                            + t.nanosecond() as i64)
                    })
                    .collect::<Result<_>>()?;
                cols.push(Series::new(name.as_str().into(), data).into_column());
            }
            _ => {
                let data: Vec<String> = rows
                    .iter()
                    .map(|r| r.get(idx).cloned().unwrap_or_default())
                    .collect();
                cols.push(Series::new(name.as_str().into(), data).into_column());
            }
        }
    }
    DataFrame::new(cols).map_err(|e| e.into())
}

fn set_cell_value(df: &mut DataFrame, row: usize, col: usize, value: &str) -> anyhow::Result<()> {
    use polars::prelude::*;
    use polars::utils::IdxSize;
    let dtype = df.dtypes()[col].clone();
    df.try_apply_at_idx(col, |c| -> PolarsResult<Series> {
        let series = match dtype {
            DataType::Int64 => {
                let ca = c.i64()?;
                let val = value.parse::<i64>()?;
                ca.scatter_single(vec![row as IdxSize], Some(val))?
                    .into_series()
            }
            DataType::Float64 => {
                let ca = c.f64()?;
                let val = value.parse::<f64>()?;
                ca.scatter_single(vec![row as IdxSize], Some(val))?
                    .into_series()
            }
            DataType::Boolean => {
                let ca = c.bool()?;
                let val = matches!(value.to_lowercase().as_str(), "true" | "1");
                ca.scatter_single(vec![row as IdxSize], Some(val))?
                    .into_series()
            }
            DataType::String => {
                let ca = c.str()?;
                ca.scatter_single(vec![row as IdxSize], Some(value))?
                    .into_series()
            }
            DataType::Date => {
                use chrono::NaiveDate;
                let ca = c.date()?;
                let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                let d = NaiveDate::parse_from_str(value, "%Y-%m-%d")?;
                let days = (d - epoch).num_days() as i32;
                ca.scatter_single(vec![row as IdxSize], Some(days))?
                    .into_series()
            }
            DataType::Datetime(_, _) => {
                use chrono::{DateTime, NaiveDateTime};
                let ca = c.datetime()?;
                let ts = DateTime::parse_from_rfc3339(value)
                    .map(|dt| dt.timestamp_micros())
                    .or_else(|_| {
                        NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
                            .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S"))
                            .map(|dt| dt.timestamp_micros())
                    })?;
                ca.scatter_single(vec![row as IdxSize], Some(ts))?
                    .into_series()
            }
            DataType::Time => {
                use chrono::{NaiveTime, Timelike};
                let ca = c.time()?;
                let t = NaiveTime::parse_from_str(value, "%H:%M:%S")?;
                let ns =
                    (t.num_seconds_from_midnight() as i64) * 1_000_000_000 + t.nanosecond() as i64;
                ca.scatter_single(vec![row as IdxSize], Some(ns))?
                    .into_series()
            }
            _ => c.clone(),
        };
        Ok(series)
    })?;
    Ok(())
}

fn dataframe_to_rows(df: &DataFrame) -> Vec<Vec<String>> {
    let mut rows = Vec::with_capacity(df.height());
    for row_idx in 0..df.height() {
        let mut row = Vec::with_capacity(df.width());
        for col in df.get_columns() {
            let val = col.get(row_idx).map(|v| v.to_string()).unwrap_or_default();
            row.push(val);
        }
        rows.push(row);
    }
    rows
}

impl ParquetApp {
    /// Spawn a background task to read the current `file_path`.
    fn start_read(&mut self) {
        let path = self.file_path.clone();
        let use_dir = self.use_directory;
        self.page_start = 0;
        self.page_cache.clear();
        self.cache_order.clear();
        self.prefetch_start = None;
        let (tx, rx) = mpsc::channel();
        self.result_rx = Some(rx);
        self.busy = true;
        self.progress = Some(0.0);
        self.status = "Reading...".into();
        let page_size = self.page_size;
        if !use_dir {
            if let Ok(meta) = parquet_examples::read_parquet_metadata(&path) {
                self.total_rows = meta.file_metadata().num_rows() as usize;
                self.metadata = Some(meta);
            }
        }
        self.runtime.spawn(async move {
            if use_dir {
                background::read_directory(path, tx).await;
            } else {
                let ext = std::path::Path::new(&path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                match ext.as_str() {
                    "csv" => background::read_csv(path, tx).await,
                    "json" => background::read_json(path, tx).await,
                    _ => background::read_dataframe_slice(path, 0, page_size, tx).await,
                }
            }
        });
    }

    /// Load the current page from the Parquet file.
    fn load_page(&mut self) {
        if let Some(cached) = self.page_cache.get(&self.page_start) {
            self.edit_df = Some(cached.clone());
            self.refresh_dataframe_state(cached);
            let end = (self.page_start + cached.height()).min(self.total_rows);
            self.status = format!(
                "Rows {}-{} of {}",
                self.page_start + 1,
                end,
                self.total_rows
            );
            if let Some(pos) = self.cache_order.iter().position(|&p| p == self.page_start) {
                self.cache_order.remove(pos);
            }
            self.cache_order.push_back(self.page_start);
            return;
        }

        let path = self.file_path.clone();
        let start = self.page_start as i64;
        let len = self.page_size;
        let exprs: Vec<String> = self
            .column_filters
            .iter()
            .filter_map(|(name, f)| {
                if f.value.trim().is_empty() {
                    None
                } else {
                    let op = match f.op {
                        FilterOp::Equals => "==",
                        FilterOp::Contains => "contains",
                    };
                    Some(format!("{} {} \"{}\"", name, op, f.value))
                }
            })
            .collect();
        let (tx, rx) = mpsc::channel();
        self.result_rx = Some(rx);
        self.busy = true;
        self.prefetch_start = None;
        self.progress = Some(0.0);
        self.status = "Reading...".into();
        self.runtime.spawn(async move {
            if exprs.is_empty() {
                background::read_dataframe_slice(path, start, len, tx).await;
            } else {
                background::read_filter_slice(path, exprs, start, len, tx).await;
            }
        });
    }

    /// Apply all column filters and reload the file.
    fn apply_filters(&mut self) {
        let path = self.file_path.clone();
        let len = self.page_size;
        let start = self.page_start as i64;
        self.page_cache.clear();
        self.cache_order.clear();
        self.prefetch_start = None;
        let exprs: Vec<String> = self
            .column_filters
            .iter()
            .filter_map(|(name, f)| {
                if f.value.trim().is_empty() {
                    None
                } else {
                    let op = match f.op {
                        FilterOp::Equals => "==",
                        FilterOp::Contains => "contains",
                    };
                    Some(format!("{} {} \"{}\"", name, op, f.value))
                }
            })
            .collect();
        let (tx, rx) = mpsc::channel();
        self.result_rx = Some(rx);
        self.busy = true;
        self.progress = Some(0.0);
        self.page_start = 0;
        self.status = "Filtering...".into();
        self.total_rows = 0;
        self.runtime.spawn(async move {
            if exprs.is_empty() {
                background::read_dataframe_slice(path, start, len, tx).await;
            } else {
                let tx_clone = tx.clone();
                let path_clone = path.clone();
                let exprs_clone = exprs.clone();
                background::filter_count(path_clone, exprs_clone, tx_clone).await;
                background::read_filter_slice(path, exprs, start, len, tx).await;
            }
        });
    }

    /// Refresh internal state after the DataFrame has changed.
    fn refresh_dataframe_state(&mut self, df: &DataFrame) {
        self.loaded_schema = df
            .get_column_names()
            .into_iter()
            .zip(df.dtypes())
            .map(|(n, t)| (n.to_string(), t))
            .collect();
        self.column_filters = self
            .loaded_schema
            .iter()
            .map(|(n, _)| (n.clone(), ColumnFilter::default()))
            .collect();
        self.rows = dataframe_to_rows(df);
        self.column_order = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        self.correlation_columns = self
            .loaded_schema
            .iter()
            .filter(|(_, t)| matches!(t, DataType::Int64 | DataType::Float64))
            .map(|(n, _)| n.clone())
            .collect();
        self.update_search_matches();
    }

    fn update_search_matches(&mut self) {
        self.search_matches =
            search::find_matches(&self.rows, &self.search_text, self.search_ignore_case);
        if self.search_index >= self.search_matches.len() {
            self.search_index = 0;
        }
    }

    /// Display a serde_json `Value` recursively as a tree.
    fn show_json(ui: &mut egui::Ui, value: &Value) {
        match value {
            Value::Object(map) => {
                for (k, v) in map {
                    egui::CollapsingHeader::new(k).show(ui, |ui| Self::show_json(ui, v));
                }
            }
            Value::Array(arr) => {
                for (i, v) in arr.iter().enumerate() {
                    egui::CollapsingHeader::new(format!("[{i}]"))
                        .show(ui, |ui| Self::show_json(ui, v));
                }
            }
            other => {
                ui.label(other.to_string());
            }
        }
    }

    /// Show top-level fields with checkboxes for mapping.
    fn show_xml_mapping(ui: &mut egui::Ui, value: &Value, selected: &mut Vec<String>) {
        if let Value::Object(map) = value {
            for (k, v) in map {
                let mut sel = selected.contains(k);
                egui::CollapsingHeader::new(k).show(ui, |ui| {
                    if v.is_array() {
                        if ui.checkbox(&mut sel, "map").changed() {
                            if sel {
                                if !selected.contains(k) {
                                    selected.push(k.clone());
                                }
                            } else {
                                selected.retain(|s| s != k);
                            }
                        }
                    }
                    Self::show_json(ui, v);
                });
            }
        }
    }
}

impl eframe::App for ParquetApp {
    /// Called each frame to update the UI.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle files dropped onto the window
        let dropped: Vec<egui::DroppedFile> = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(file) = dropped.first() {
            if let Some(path) = &file.path {
                self.file_path = path.display().to_string();
                self.use_directory = Path::new(&self.file_path).is_dir();
                self.operation = Operation::Read;
                self.start_read();
            }
        }

        if let Some(df) = &mut self.edit_df {
            let mut sort_after: Option<String> = None;
            egui::SidePanel::right("preview_panel").show(ctx, |ui| {
                ui.heading("Preview");
                let end = (self.page_start + df.height()).min(self.total_rows);
                let approx = df.estimated_size();
                ui.label(format!(
                    "Rows {}-{} of {} (~{} bytes)",
                    self.page_start + 1,
                    end,
                    self.total_rows,
                    approx
                ));
                ui.horizontal(|ui| {
                    ui.label("Search:");
                    let changed = ui.text_edit_singleline(&mut self.search_text).changed();
                    let case_changed = ui
                        .checkbox(&mut self.search_ignore_case, "Ignore case")
                        .changed();
                    if changed || case_changed {
                        self.search_index = 0;
                        self.update_search_matches();
                    }
                    if !self.search_matches.is_empty() {
                        if ui.button("Previous").clicked() {
                            self.search_index =
                                search::prev_index(self.search_index, &self.search_matches);
                        }
                        if ui.button("Next").clicked() {
                            self.search_index =
                                search::next_index(self.search_index, &self.search_matches);
                        }
                        ui.label(format!(
                            "{}/{}",
                            self.search_index + 1,
                            self.search_matches.len()
                        ));
                    }
                });
                if self.total_rows > self.page_size {
                    ui.horizontal(|ui| {
                        if ui.button("Previous").clicked() && self.page_start >= self.page_size {
                            self.page_start -= self.page_size;
                            self.load_page();
                        }
                        if ui.button("Next").clicked()
                            && self.page_start + self.page_size < self.total_rows
                        {
                            self.page_start += self.page_size;
                            self.load_page();
                        }
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Page size:");
                    if ui
                        .add(egui::DragValue::new(&mut self.page_size).clamp_range(1..=1000))
                        .changed()
                    {
                        self.load_page();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Rows to display:");
                    if ui
                        .add(egui::DragValue::new(&mut self.display_rows).clamp_range(1..=1000))
                        .changed()
                    {
                        self.update_search_matches();
                    }
                    if ui.button("Toggle schema").clicked() {
                        self.show_schema = !self.show_schema;
                    }
                });

                if self.show_schema {
                    egui::Grid::new("schema_grid").striped(true).show(ui, |ui| {
                        ui.label("Column");
                        ui.label("Type");
                        ui.end_row();
                        for (name, dtype) in &self.loaded_schema {
                            ui.label(name);
                            ui.label(format!("{:?}", dtype));
                            ui.end_row();
                        }
                    });
                    ui.separator();
                }

                let head = df.head(Some(self.display_rows));
                let names: Vec<String> = head
                    .get_column_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                let mut table = TableBuilder::new(ui);
                for _ in &names {
                    table = table.column(TableColumn::auto());
                }
                table
                    .striped(true)
                    .header(20.0, |mut header| {
                        for name in &names {
                            header.col(|ui| {
                                if ui.button(name).clicked() {
                                    sort_after = Some(name.clone());
                                }
                            });
                        }
                    })
                    .body(|mut body| {
                        for row_idx in 0..head.height() {
                            body.row(18.0, |mut row| {
                                for (col_idx, col) in df.get_columns().iter().enumerate() {
                                    let dtype = col.dtype();
                                    let cell = &mut self.rows[row_idx][col_idx];
                                    let match_idx = self
                                        .search_matches
                                        .iter()
                                        .position(|&(r, c)| r == row_idx && c == col_idx);
                                    let mut show_cell = |ui: &mut egui::Ui| match dtype {
                                        DataType::Boolean => {
                                            let mut checked = matches!(cell.as_str(), "true" | "1");
                                            if ui.checkbox(&mut checked, "").changed() {
                                                *cell = checked.to_string();
                                                if let Err(e) =
                                                    set_cell_value(df, row_idx, col_idx, cell)
                                                {
                                                    self.status = format!("Edit failed: {e}");
                                                }
                                                self.update_search_matches();
                                            }
                                        }
                                        _ => {
                                            let resp = ui.text_edit_singleline(cell);
                                            if resp.lost_focus() && resp.changed() {
                                                if let Err(e) =
                                                    set_cell_value(df, row_idx, col_idx, cell)
                                                {
                                                    self.status = format!("Edit failed: {e}");
                                                }
                                                self.update_search_matches();
                                            }
                                        }
                                    };
                                    if let Some(idx) = match_idx {
                                        let color = if idx == self.search_index {
                                            egui::Color32::from_rgb(255, 200, 200)
                                        } else {
                                            egui::Color32::from_rgb(255, 255, 0)
                                        };
                                        egui::Frame::none().fill(color).show(ui, |ui| {
                                            show_cell(ui);
                                        });
                                    } else {
                                        show_cell(ui);
                                    }
                                }
                            });
                        }
                    });
                if ui.button("Save changes").clicked() {
                    let df = df.clone();
                    let file = self.file_path.clone();
                    let (tx, rx) = mpsc::channel();
                    self.result_rx = Some(rx);
                    self.busy = true;
                    self.progress = Some(0.0);
                    self.status = "Saving...".into();
                    self.runtime.spawn(async move {
                        background::write_dataframe(df, file, tx).await;
                    });
                }

                if let Ok(summary) = parquet_examples::summarize_dataframe(df) {
                    ui.separator();
                    ui.label("Statistics");
                    ui.label(format!("Rows: {}", summary.rows));
                    ui.label(format!("Columns: {}", summary.columns));
                    ui.label(format!("Approx size: {} bytes", summary.approx_bytes));

                    let stat_names: Vec<String> = summary
                        .stats
                        .get_column_names()
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    let mut table = TableBuilder::new(ui);
                    for _ in &stat_names {
                        table = table.column(TableColumn::auto());
                    }
                    table
                        .striped(true)
                        .header(18.0, |mut header| {
                            for name in &stat_names {
                                header.col(|ui| {
                                    ui.label(name);
                                });
                            }
                        })
                        .body(|mut body| {
                            for row_idx in 0..summary.stats.height() {
                                body.row(18.0, |mut row| {
                                    for col in summary.stats.get_columns() {
                                        let val = col
                                            .get(row_idx)
                                            .map(|v| v.to_string())
                                            .unwrap_or_default();
                                        row.col(|ui| {
                                            ui.label(val);
                                        });
                                    }
                                });
                            }
                        });
                }

                ui.separator();
                ui.label("Column operations");
                egui::ComboBox::from_label("Column")
                    .selected_text(if self.rename_column.is_empty() {
                        "Select".to_string()
                    } else {
                        self.rename_column.clone()
                    })
                    .show_ui(ui, |ui| {
                        for name in df.get_column_names_str() {
                            ui.selectable_value(&mut self.rename_column, name.to_string(), name);
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label("New name:");
                    ui.text_edit_singleline(&mut self.new_name);
                    if ui.button("Rename").clicked() {
                        if let Err(e) =
                            parquet_examples::rename_column(df, &self.rename_column, &self.new_name)
                        {
                            self.status = format!("Rename failed: {e}");
                        } else {
                            self.rename_column.clear();
                            self.new_name.clear();
                            self.refresh_dataframe_state(df);
                        }
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Up").clicked() {
                        if let Some(pos) = self
                            .column_order
                            .iter()
                            .position(|c| c == &self.rename_column)
                        {
                            if pos > 0 {
                                self.column_order.swap(pos, pos - 1);
                                if parquet_examples::reorder_columns(df, &self.column_order).is_ok()
                                {
                                    self.refresh_dataframe_state(df);
                                }
                            }
                        }
                    }
                    if ui.button("Down").clicked() {
                        if let Some(pos) = self
                            .column_order
                            .iter()
                            .position(|c| c == &self.rename_column)
                        {
                            if pos + 1 < self.column_order.len() {
                                self.column_order.swap(pos, pos + 1);
                                if parquet_examples::reorder_columns(df, &self.column_order).is_ok()
                                {
                                    self.refresh_dataframe_state(df);
                                }
                            }
                        }
                    }
                });
                ui.label("Drop columns:");
                let mut drops: Vec<String> = Vec::new();
                for name in df.get_column_names_str() {
                    let mut mark = false;
                    if ui.checkbox(&mut mark, name).clicked() && mark {
                        drops.push(name.to_string());
                    }
                }
                if !drops.is_empty() {
                    for d in &drops {
                        if parquet_examples::drop_column(df, d).is_ok() {
                            self.column_order.retain(|c| c != d);
                        }
                    }
                    self.refresh_dataframe_state(df);
                }

                #[cfg(feature = "plotting")]
                {
                    use polars::prelude::DataType;
                    ui.separator();
                    ui.label("Plot");
                    let numeric: Vec<String> = self
                        .loaded_schema
                        .iter()
                        .filter(|(_, t)| matches!(t, DataType::Int64 | DataType::Float64))
                        .map(|(n, _)| n.clone())
                        .collect();
                    egui::ComboBox::from_label("X Column")
                        .selected_text(
                            self.plot_column
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| "None".into()),
                        )
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.plot_column, None, "None");
                            for name in &numeric {
                                ui.selectable_value(
                                    &mut self.plot_column,
                                    Some(name.clone()),
                                    name,
                                );
                            }
                        });
                    if matches!(self.plot_type, PlotType::Scatter) {
                        egui::ComboBox::from_label("Y Column")
                            .selected_text(
                                self.plot_y_column
                                    .as_ref()
                                    .cloned()
                                    .unwrap_or_else(|| "None".into()),
                            )
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.plot_y_column, None, "None");
                                for name in &numeric {
                                    ui.selectable_value(
                                        &mut self.plot_y_column,
                                        Some(name.clone()),
                                        name,
                                    );
                                }
                            });
                    }
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.plot_type, PlotType::Histogram, "Histogram");
                        ui.radio_value(&mut self.plot_type, PlotType::Line, "Line");
                        ui.radio_value(&mut self.plot_type, PlotType::Scatter, "Scatter");
                        ui.radio_value(&mut self.plot_type, PlotType::BoxPlot, "Box");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Bins:");
                        ui.add(egui::DragValue::new(&mut self.hist_bins).clamp_range(1..=100));
                    });
                    ui.horizontal(|ui| {
                        ui.label("X range:");
                        let mut xmin = self.x_range.map(|r| r.0).unwrap_or(0.0);
                        let mut xmax = self.x_range.map(|r| r.1).unwrap_or(0.0);
                        ui.add(egui::DragValue::new(&mut xmin));
                        ui.add(egui::DragValue::new(&mut xmax));
                        if ui.button("Apply").clicked() {
                            self.x_range = Some((xmin, xmax));
                        }
                        if ui.button("Clear").clicked() {
                            self.x_range = None;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Y range:");
                        let mut ymin = self.y_range.map(|r| r.0).unwrap_or(0.0);
                        let mut ymax = self.y_range.map(|r| r.1).unwrap_or(0.0);
                        ui.add(egui::DragValue::new(&mut ymin));
                        ui.add(egui::DragValue::new(&mut ymax));
                        if ui.button("Apply").clicked() {
                            self.y_range = Some((ymin, ymax));
                        }
                        if ui.button("Clear").clicked() {
                            self.y_range = None;
                        }
                    });
                    if let Some(col) = &self.plot_column {
                        if let Ok(series) = df.column(col) {
                            let values: Vec<f64> = series
                                .f64()
                                .map(|ca| ca.into_no_null_iter().collect())
                                .or_else(|_| {
                                    series.i64().map(|ca| {
                                        ca.into_no_null_iter().map(|v| v as f64).collect()
                                    })
                                })
                                .unwrap_or_default();
                            if !values.is_empty() {
                                match self.plot_type {
                                    PlotType::Histogram => {
                                        let (counts, min, step) =
                                            parquet_examples::compute_histogram(
                                                &values,
                                                self.hist_bins,
                                                self.x_range,
                                            );
                                        let bars: Vec<_> = counts
                                            .iter()
                                            .enumerate()
                                            .map(|(i, c)| {
                                                egui_plot::Bar::new(
                                                    min + step * (i as f64 + 0.5),
                                                    *c,
                                                )
                                            })
                                            .collect();
                                        let mut plot = Plot::new("histogram");
                                        if let Some((xmin, xmax)) = self.x_range {
                                            plot = plot.default_x_bounds(xmin, xmax);
                                        }
                                        if let Some((ymin, ymax)) = self.y_range {
                                            plot = plot.default_y_bounds(ymin, ymax);
                                        }
                                        plot.show(ui, |plot_ui| {
                                            plot_ui.bar_chart(BarChart::new("", bars));
                                        });
                                    }
                                    PlotType::Line => {
                                        let points: PlotPoints = values
                                            .iter()
                                            .enumerate()
                                            .map(|(i, v)| [i as f64, *v])
                                            .collect();
                                        let mut plot = Plot::new("line");
                                        if let Some((xmin, xmax)) = self.x_range {
                                            plot = plot.default_x_bounds(xmin, xmax);
                                        }
                                        if let Some((ymin, ymax)) = self.y_range {
                                            plot = plot.default_y_bounds(ymin, ymax);
                                        }
                                        plot.show(ui, |plot_ui| {
                                            plot_ui.line(Line::new("", points));
                                        });
                                    }
                                    PlotType::Scatter => {
                                        if let Some(ycol) = &self.plot_y_column {
                                            if let Ok(ys) = df.column(ycol) {
                                                let y_vals: Vec<f64> = ys
                                                    .f64()
                                                    .map(|ca| ca.into_no_null_iter().collect())
                                                    .or_else(|_| {
                                                        ys.i64().map(|ca| {
                                                            ca.into_no_null_iter()
                                                                .map(|v| v as f64)
                                                                .collect()
                                                        })
                                                    })
                                                    .unwrap_or_default();
                                                let points: PlotPoints = values
                                                    .iter()
                                                    .cloned()
                                                    .zip(y_vals)
                                                    .map(|(x, y)| [x, y])
                                                    .collect();
                                                let mut plot = Plot::new("scatter");
                                                if let Some((xmin, xmax)) = self.x_range {
                                                    plot = plot.default_x_bounds(xmin, xmax);
                                                }
                                                if let Some((ymin, ymax)) = self.y_range {
                                                    plot = plot.default_y_bounds(ymin, ymax);
                                                }
                                                plot.show(ui, |plot_ui| {
                                                    plot_ui
                                                        .points(egui_plot::Points::new("", points));
                                                });
                                            }
                                        }
                                    }
                                    PlotType::BoxPlot => {
                                        use egui_plot::{BoxElem, BoxPlot};
                                        if !values.is_empty() {
                                            let mut sorted: Vec<f64> = values
                                                .iter()
                                                .cloned()
                                                .filter(|v| !v.is_nan())
                                                .collect();
                                            if sorted.is_empty() {
                                                continue;
                                            }
                                            sorted.sort_by(|a, b| {
                                                a.partial_cmp(b)
                                                    .unwrap_or(std::cmp::Ordering::Equal)
                                            });
                                            let q1 = sorted[(sorted.len() as f64 * 0.25) as usize];
                                            let q2 = sorted[(sorted.len() as f64 * 0.5) as usize];
                                            let q3 = sorted[(sorted.len() as f64 * 0.75) as usize];
                                            let min = *sorted.first().unwrap();
                                            let max = *sorted.last().unwrap();
                                            let spread =
                                                egui_plot::BoxSpread::new(min, q1, q2, q3, max);
                                            let elem = BoxElem::new(0.0, spread);
                                            let mut plot = Plot::new("boxplot");
                                            if let Some((xmin, xmax)) = self.x_range {
                                                plot = plot.default_x_bounds(xmin, xmax);
                                            }
                                            if let Some((ymin, ymax)) = self.y_range {
                                                plot = plot.default_y_bounds(ymin, ymax);
                                            }
                                            plot.show(ui, |plot_ui| {
                                                plot_ui.box_plot(BoxPlot::new("", vec![elem]));
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });

            if let Some(col) = sort_after {
                if let Ok(sorted) = df.sort([col.as_str()], SortMultipleOptions::default()) {
                    *df = sorted;
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Polars Parquet Learning");
            ui.label("Drag and drop a Parquet file to load it automatically.");

            if let Some(rx) = &self.result_rx {
                match rx.try_recv() {
                    Ok(res) => match res {
                        Ok(JobUpdate::Progress(p)) => {
                            self.progress = Some(p);
                        }
                        Ok(JobUpdate::Done(JobResult::Count(c))) => {
                            self.total_rows = c;
                        }
                        Ok(JobUpdate::Done(JobResult::DataFrame(df))) => {
                            let prefetch = !self.busy;
                            let start_idx = if prefetch {
                                self.prefetch_start.take().unwrap_or(self.page_start)
                            } else {
                                self.page_start
                            };
                            if !prefetch {
                                self.busy = false;
                                self.result_rx = None;
                                self.progress = None;
                                if self.status.starts_with("Filtering") {
                                    let end = (self.page_start + df.height()).min(self.total_rows);
                                    self.status = format!(
                                        "Rows {}-{} of {}",
                                        self.page_start + 1,
                                        end,
                                        self.total_rows
                                    );
                                } else if self.use_directory {
                                    self.total_rows = df.height();
                                    self.status = format!("Combined {} rows", df.height());
                                } else {
                                    let ext = Path::new(&self.file_path)
                                        .extension()
                                        .and_then(|e| e.to_str())
                                        .unwrap_or("")
                                        .to_ascii_lowercase();
                                    match ext.as_str() {
                                        "csv" => {
                                            self.total_rows = df.height();
                                            self.status =
                                                format!("Loaded {} rows from CSV", df.height());
                                        }
                                        "json" => {
                                            self.total_rows = df.height();
                                            self.status =
                                                format!("Loaded {} rows from JSON", df.height());
                                        }
                                        _ => {
                                            let end = (self.page_start + df.height())
                                                .min(self.total_rows);
                                            self.status = format!(
                                                "Rows {}-{} of {}",
                                                self.page_start + 1,
                                                end,
                                                self.total_rows
                                            );
                                        }
                                    }
                                }
                                self.refresh_dataframe_state(&df);
                                self.edit_df = Some(df.clone());
                            }

                            self.page_cache.insert(start_idx, df.clone());
                            if let Some(pos) = self.cache_order.iter().position(|&p| p == start_idx)
                            {
                                self.cache_order.remove(pos);
                            }
                            self.cache_order.push_back(start_idx);
                            while self.cache_order.len() > 3 {
                                if let Some(old) = self.cache_order.pop_front() {
                                    self.page_cache.remove(&old);
                                }
                            }

                            if !prefetch {
                                let next_start = self.page_start + self.page_size;
                                if next_start < self.total_rows
                                    && !self.page_cache.contains_key(&next_start)
                                {
                                    let path = self.file_path.clone();
                                    let len = self.page_size;
                                    let exprs: Vec<String> = self
                                        .column_filters
                                        .iter()
                                        .filter_map(|(name, f)| {
                                            if f.value.trim().is_empty() {
                                                None
                                            } else {
                                                let op = match f.op {
                                                    FilterOp::Equals => "==",
                                                    FilterOp::Contains => "contains",
                                                };
                                                Some(format!("{} {} \"{}\"", name, op, f.value))
                                            }
                                        })
                                        .collect();
                                    let (tx, rx) = mpsc::channel();
                                    self.result_rx = Some(rx);
                                    self.prefetch_start = Some(next_start);
                                    self.runtime.spawn(async move {
                                        if exprs.is_empty() {
                                            background::read_dataframe_slice(
                                                path,
                                                next_start as i64,
                                                len,
                                                tx,
                                            )
                                            .await;
                                        } else {
                                            background::read_filter_slice(
                                                path,
                                                exprs,
                                                next_start as i64,
                                                len,
                                                tx,
                                            )
                                            .await;
                                        }
                                    });
                                }
                            }
                        }
                        Ok(JobUpdate::Done(JobResult::Unit)) => {
                            self.busy = false;
                            self.result_rx = None;
                            self.progress = None;
                            self.status = "Done".into();
                        }
                        Err(e) => {
                            self.busy = false;
                            self.result_rx = None;
                            self.progress = None;
                            self.status = format!("Failed: {e}");
                        }
                    },
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.busy = false;
                        self.result_rx = None;
                        self.progress = None;
                        self.status = "Background task disconnected".into();
                    }
                }
            }

            // Input field for the Parquet file path
            ui.horizontal(|ui| {
                ui.label("File:");
                ui.text_edit_singleline(&mut self.file_path);
                if ui.button("...").clicked() {
                    let mut dialog = FileDialog::new();
                    match self.operation {
                        Operation::WriteCsv => {
                            dialog = dialog.add_filter("CSV", &["csv"]);
                        }
                        Operation::WriteJson => {
                            dialog = dialog.add_filter("JSON", &["json"]);
                        }
                        Operation::Xml | Operation::XmlDynamic => {
                            dialog = dialog.add_filter("XML", &["xml"]);
                        }
                        _ => {
                            if !self.use_directory {
                                dialog = dialog.add_filter("Parquet", &["parquet"]);
                                dialog = dialog.add_filter("CSV", &["csv"]);
                                dialog = dialog.add_filter("JSON", &["json"]);
                            }
                        }
                    }
                    let picked = if self.use_directory
                        && !matches!(self.operation, Operation::Xml | Operation::XmlDynamic)
                    {
                        dialog.pick_folder()
                    } else {
                        dialog.pick_file()
                    };
                    if let Some(path) = picked {
                        self.file_path = path.display().to_string();
                    }
                }
                ui.checkbox(&mut self.use_directory, "Directory");
            });

            // Radio buttons to pick an operation
            ui.horizontal(|ui| {
                ui.label("Operation:");
                ui.radio_value(&mut self.operation, Operation::Read, "Read");
                ui.radio_value(&mut self.operation, Operation::Modify, "Modify");
                ui.radio_value(&mut self.operation, Operation::Write, "Write");
                ui.radio_value(&mut self.operation, Operation::WriteCsv, "Write CSV");
                ui.radio_value(&mut self.operation, Operation::WriteJson, "Write JSON");
                ui.radio_value(&mut self.operation, Operation::Create, "Create");
                ui.radio_value(&mut self.operation, Operation::Partition, "Partition");
                ui.radio_value(&mut self.operation, Operation::Query, "Query");
                ui.radio_value(&mut self.operation, Operation::Correlation, "Correlation");
                ui.radio_value(&mut self.operation, Operation::Xml, "XML");
                ui.radio_value(&mut self.operation, Operation::XmlDynamic, "XML Dynamic");
            });

            ui.horizontal(|ui| {
                ui.label("Save:");
                ui.text_edit_singleline(&mut self.save_path);
                if ui.button("...").clicked() {
                    let mut dialog = FileDialog::new();
                    match self.operation {
                        Operation::WriteCsv => {
                            dialog = dialog.add_filter("CSV", &["csv"]);
                        }
                        Operation::WriteJson => {
                            dialog = dialog.add_filter("JSON", &["json"]);
                        }
                        Operation::Xml | Operation::XmlDynamic => {}
                        _ => {
                            dialog = dialog.add_filter("Parquet", &["parquet"]);
                        }
                    }
                    let picked = if matches!(self.operation, Operation::Xml | Operation::XmlDynamic)
                    {
                        dialog.pick_folder()
                    } else {
                        dialog.save_file()
                    };
                    if let Some(path) = picked {
                        self.save_path = path.display().to_string();
                    }
                }
            });

            match self.operation {
                Operation::Create => {
                    ui.horizontal(|ui| {
                        ui.label("New column:");
                        ui.text_edit_singleline(&mut self.new_col_name);
                        egui::ComboBox::from_id_source("col_type")
                            .selected_text(if self.new_col_type.is_empty() {
                                "Select".to_string()
                            } else {
                                self.new_col_type.clone()
                            })
                            .show_ui(ui, |ui| {
                                for t in ["int", "str", "float", "bool", "date", "datetime", "time"]
                                {
                                    ui.selectable_value(&mut self.new_col_type, t.to_string(), t);
                                }
                            });
                        if ui.button("Add").clicked() {
                            if let Ok(dtype) = parse_dtype(&self.new_col_type) {
                                self.schema.push((self.new_col_name.clone(), dtype));
                                for row in &mut self.rows {
                                    row.push(String::new());
                                }
                                self.new_col_name.clear();
                                self.new_col_type.clear();
                            }
                        }
                        if ui.button("Remove column").clicked() {
                            if !self.schema.is_empty() {
                                self.schema.pop();
                                for row in &mut self.rows {
                                    row.pop();
                                }
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        if ui.button("Add row").clicked() {
                            self.rows.push(vec![String::new(); self.schema.len()]);
                        }
                        if ui.button("Remove row").clicked() {
                            self.rows.pop();
                        }
                    });

                    egui::Grid::new("data_grid").show(ui, |ui| {
                        for (i, row) in self.rows.iter_mut().enumerate() {
                            for (j, _col) in self.schema.iter().enumerate() {
                                ui.text_edit_singleline(&mut row[j]);
                            }
                            ui.end_row();
                        }
                    });
                }
                Operation::Write => {
                    if let Some(df) = &self.edit_df {
                        egui::ComboBox::from_label("Partition column")
                            .selected_text(
                                self.partition_column
                                    .clone()
                                    .unwrap_or_else(|| "None".to_string()),
                            )
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.partition_column, None, "None");
                                for name in df.get_column_names_str() {
                                    ui.selectable_value(
                                        &mut self.partition_column,
                                        Some(name.to_string()),
                                        name,
                                    );
                                }
                            });
                    }
                }
                Operation::Partition => {
                    if let Some(df) = &self.edit_df {
                        ui.label("Columns:");
                        for name in df.get_column_names_str() {
                            let mut selected = self.partition_columns.contains(&name.to_string());
                            if ui.checkbox(&mut selected, name).changed() {
                                if selected {
                                    if !self.partition_columns.contains(&name.to_string()) {
                                        self.partition_columns.push(name.to_string());
                                    }
                                } else {
                                    self.partition_columns.retain(|c| c != name);
                                }
                            }
                        }
                    }
                }
                Operation::Query => {
                    ui.horizontal(|ui| {
                        ui.label("Prefix:");
                        ui.text_edit_singleline(&mut self.query_prefix);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Expr:");
                        ui.text_edit_singleline(&mut self.query_expr);
                    });
                    ui.separator();
                    ui.label("Column filters:");
                    let mut changed = false;
                    for (name, _) in &self.loaded_schema {
                        let filter = self.column_filters.entry(name.clone()).or_default();
                        ui.horizontal(|ui| {
                            ui.label(name);
                            egui::ComboBox::from_id_source(format!("op_{name}"))
                                .selected_text(match filter.op {
                                    FilterOp::Equals => "=",
                                    FilterOp::Contains => "contains",
                                })
                                .show_ui(ui, |ui| {
                                    changed |= ui
                                        .selectable_value(&mut filter.op, FilterOp::Equals, "=")
                                        .changed();
                                    changed |= ui
                                        .selectable_value(
                                            &mut filter.op,
                                            FilterOp::Contains,
                                            "contains",
                                        )
                                        .changed();
                                });
                            changed |= ui.text_edit_singleline(&mut filter.value).changed();
                        });
                    }
                    if changed && !self.busy {
                        self.apply_filters();
                    }
                }
                Operation::Correlation => {
                    if let Some(_df) = &self.edit_df {
                        ui.label("Columns:");
                        for (name, dtype) in &self.loaded_schema {
                            if matches!(dtype, DataType::Int64 | DataType::Float64) {
                                let mut selected = self.correlation_columns.contains(name);
                                if ui.checkbox(&mut selected, name).changed() {
                                    if selected {
                                        if !self.correlation_columns.contains(name) {
                                            self.correlation_columns.push(name.clone());
                                        }
                                    } else {
                                        self.correlation_columns.retain(|c| c != name);
                                    }
                                }
                            }
                        }
                    }
                }
                Operation::Xml => {
                    ui.checkbox(&mut self.xml_schema, "Write _schema.json");
                }
                Operation::XmlDynamic => {
                    if ui.button("Load XML").clicked() {
                        match xml_dynamic::parse_any_xml(&self.file_path) {
                            Ok(v) => {
                                self.xml_value = Some(v);
                                self.xml_tables.clear();
                                self.xml_selected.clear();
                                self.xml_active_table = None;
                                self.status.clear();
                            }
                            Err(e) => self.status = format!("Load failed: {e}"),
                        }
                    }
                    if let Some(val) = &self.xml_value {
                        Self::show_xml_mapping(ui, val, &mut self.xml_selected);
                        if !self.xml_tables.is_empty() {
                            let current = self
                                .xml_active_table
                                .clone()
                                .unwrap_or_else(|| "None".into());
                            egui::ComboBox::from_id_source("xml_table")
                                .selected_text(current)
                                .show_ui(ui, |ui| {
                                    for name in self.xml_tables.keys() {
                                        ui.selectable_value(
                                            &mut self.xml_active_table,
                                            Some(name.clone()),
                                            name,
                                        );
                                    }
                                });
                            if let Some(name) = &self.xml_active_table {
                                if let Some(df) = self.xml_tables.get(name) {
                                    self.refresh_dataframe_state(df);
                                    self.edit_df = Some(df.clone());
                                }
                            }
                        }
                    }
                }
                _ => {}
            }

            if let Some(df) = &self.edit_df {
                ui.separator();
                if let Some(meta) = &self.metadata {
                    ui.label("Metadata");
                    egui::Grid::new("meta_grid").striped(true).show(ui, |ui| {
                        ui.label(format!(
                            "File size: {} bytes",
                            meta.file_metadata().total_byte_size()
                        ));
                        ui.end_row();
                        ui.label(format!("Row groups: {}", meta.num_row_groups()));
                        ui.end_row();
                        for (i, rg) in meta.row_groups().iter().enumerate() {
                            ui.label(format!("Row group {i}: {} rows", rg.num_rows()));
                            ui.end_row();
                        }
                        ui.end_row();
                        ui.label("Columns:");
                        ui.end_row();
                        for col in meta.file_metadata().schema_descr().columns() {
                            ui.label(format!("{} ({:?})", col.name(), col.physical_type()));
                            ui.end_row();
                        }
                    });
                }
            }

            ui.separator();
            ui.label(&self.status);
            if self.busy {
                if let Some(p) = self.progress {
                    ui.add(egui::ProgressBar::new(p).show_percentage());
                } else {
                    ui.add(egui::Spinner::new());
                }
            }

            // Run the selected action
            let requires_df = matches!(
                self.operation,
                Operation::Write | Operation::Partition | Operation::Query | Operation::Correlation
            );
            let run_enabled = !self.busy && !(requires_df && self.edit_df.is_none());
            let run_clicked = ui
                .add_enabled(run_enabled, egui::Button::new("Run"))
                .clicked();
            if requires_df && self.edit_df.is_none() {
                ui.label("Load or create a DataFrame first.");
            }
            if run_clicked {
                match self.operation {
                    Operation::Read => {
                        self.start_read();
                    }
                    Operation::Modify => {
                        match parquet_examples::read_parquet_to_dataframe(&self.file_path) {
                            Ok(df) => {
                                if let Ok(mut rec) = parquet_examples::dataframe_to_records(&df) {
                                    parquet_examples::modify_records(&mut rec);
                                    match parquet_examples::records_to_dataframe(&rec) {
                                        Ok(df) => {
                                            self.status = "Modified records".into();
                                            self.refresh_dataframe_state(&df);
                                            self.edit_df = Some(df);
                                        }
                                        Err(e) => self.status = format!("Failed to convert: {e}"),
                                    }
                                }
                            }
                            Err(e) => self.status = format!("Failed to read: {e}"),
                        }
                    }
                    Operation::Write => {
                        if let Some(df) = &self.edit_df {
                            if let Some(col) = &self.partition_column {
                                let cols = [col.as_str()];
                                match parquet_examples::write_partitioned(
                                    df,
                                    &cols,
                                    &self.save_path,
                                ) {
                                    Ok(_) => {
                                        self.status =
                                            format!("Wrote partitions to {}", self.save_path)
                                    }
                                    Err(e) => self.status = format!("Write failed: {e}"),
                                }
                            } else {
                                let df = df.clone();
                                let file = self.file_path.clone();
                                let (tx, rx) = mpsc::channel();
                                self.result_rx = Some(rx);
                                self.busy = true;
                                self.progress = Some(0.0);
                                self.status = "Writing...".into();
                                self.runtime.spawn(async move {
                                    background::write_dataframe(df, file, tx).await;
                                });
                            }
                        } else {
                            self.status = "Load or create a DataFrame first.".into();
                        }
                    }
                    Operation::WriteCsv => {
                        if let Some(df) = &self.edit_df {
                            let mut df = df.clone();
                            match parquet_examples::write_dataframe_to_csv(&mut df, &self.file_path)
                            {
                                Ok(_) => self.status = format!("Wrote {}", self.file_path),
                                Err(e) => self.status = format!("Write failed: {e}"),
                            }
                        }
                    }
                    Operation::WriteJson => {
                        if let Some(df) = &self.edit_df {
                            let mut df = df.clone();
                            match parquet_examples::write_dataframe_to_json(
                                &mut df,
                                &self.file_path,
                            ) {
                                Ok(_) => self.status = format!("Wrote {}", self.file_path),
                                Err(e) => self.status = format!("Write failed: {e}"),
                            }
                        }
                    }
                    Operation::Create => match build_dataframe(&self.schema, &self.rows) {
                        Ok(df) => {
                            if !self.save_path.is_empty() {
                                match parquet_examples::create_and_write_parquet(
                                    &df,
                                    &self.save_path,
                                ) {
                                    Ok(_) => self.status = format!("Saved {}", self.save_path),
                                    Err(e) => self.status = format!("Save failed: {e}"),
                                }
                            }
                            self.refresh_dataframe_state(&df);
                            self.edit_df = Some(df);
                        }
                        Err(e) => self.status = format!("Create failed: {e}"),
                    },
                    Operation::Partition => {
                        if let Some(df) = &self.edit_df {
                            if !self.partition_columns.is_empty() {
                                let cols: Vec<&str> =
                                    self.partition_columns.iter().map(String::as_str).collect();
                                match parquet_examples::write_partitioned(
                                    df,
                                    &cols,
                                    &self.save_path,
                                ) {
                                    Ok(_) => {
                                        self.status =
                                            format!("Wrote partitions to {}", self.save_path)
                                    }
                                    Err(e) => self.status = format!("Partition failed: {e}"),
                                }
                            } else {
                                self.status = "Select one or more columns.".into();
                            }
                        } else {
                            self.status = "Load or create a DataFrame first.".into();
                        }
                    }
                    Operation::Query => {
                        if let Some(_) = &self.edit_df {
                            let res = if !self.query_expr.is_empty() {
                                parquet_examples::filter_with_expr(
                                    &self.file_path,
                                    &self.query_expr,
                                )
                            } else {
                                parquet_examples::filter_by_name_prefix(
                                    &self.file_path,
                                    &self.query_prefix,
                                )
                            };
                            match res {
                                Ok(df) => {
                                    self.status = format!("Query returned {} rows", df.height());
                                    self.refresh_dataframe_state(&df);
                                    self.edit_df = Some(df);
                                }
                                Err(e) => self.status = format!("Query failed: {e}"),
                            }
                        } else {
                            self.status = "Load or create a DataFrame first.".into();
                        }
                    }
                    Operation::Correlation => {
                        if let Some(df) = &self.edit_df {
                            if !self.correlation_columns.is_empty() {
                                let cols: Vec<&str> = self
                                    .correlation_columns
                                    .iter()
                                    .map(String::as_str)
                                    .collect();
                                match parquet_examples::correlation_matrix(df, &cols) {
                                    Ok(corr) => {
                                        self.status = "Computed correlation matrix".into();
                                        self.refresh_dataframe_state(&corr);
                                        self.edit_df = Some(corr);
                                    }
                                    Err(e) => self.status = format!("Correlation failed: {e}"),
                                }
                            } else {
                                self.status = "Select one or more columns.".into();
                            }
                        } else {
                            self.status = "Load or create a DataFrame first.".into();
                        }
                    }
                    Operation::Xml => {
                        match Polars_Parquet_Learning::xml_to_parquet::xml_to_parquet(
                            &self.file_path,
                            &self.save_path,
                            self.xml_schema,
                        ) {
                            Ok(_) => {
                                self.status = format!("Wrote Parquet tables to {}", self.save_path);
                            }
                            Err(e) => self.status = format!("XML conversion failed: {e}"),
                        }
                    }
                    Operation::XmlDynamic => {
                        if let Some(value) = &self.xml_value {
                            match xml_dynamic::value_to_tables(value) {
                                Ok(mut tables) => {
                                    if !self.xml_selected.is_empty() {
                                        tables.retain(|k, _| self.xml_selected.contains(k));
                                    }
                                    self.xml_tables = tables.clone();
                                    if !self.save_path.is_empty() {
                                        let mut map: BTreeMap<&str, DataFrame> = BTreeMap::new();
                                        for (k, v) in &mut tables {
                                            map.insert(k.as_str(), v.clone());
                                        }
                                        match xml_to_parquet::write_tables(
                                            &map,
                                            &self.save_path,
                                            self.xml_schema,
                                        ) {
                                            Ok(_) => {
                                                self.status = format!(
                                                    "Wrote Parquet tables to {}",
                                                    self.save_path
                                                );
                                            }
                                            Err(e) => {
                                                self.status = format!("XML conversion failed: {e}")
                                            }
                                        }
                                    } else {
                                        self.status = "Parsed XML".into();
                                    }
                                    if let Some(name) = self.xml_tables.keys().next().cloned() {
                                        self.xml_active_table = Some(name.clone());
                                        if let Some(df) = self.xml_tables.get(&name) {
                                            self.refresh_dataframe_state(df);
                                            self.edit_df = Some(df.clone());
                                        }
                                    }
                                }
                                Err(e) => self.status = format!("Parse failed: {e}"),
                            }
                        } else {
                            self.status = "Load an XML file first.".into();
                        }
                    }
                }
            }
        });
    }
}

/// Entry point which launches the GUI application through `eframe`.
fn main() -> eframe::Result<()> {
    if std::env::args().len() > 1 {
        let cli = cli::Cli::parse();
        if let Err(e) = cli::run(cli) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    // `eframe` sets up a native window and integrates the `egui` event loop.
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Polars Parquet Learning",
        options,
        Box::new(|cc| Ok::<Box<dyn eframe::App>, _>(Box::new(ParquetApp::new(cc)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dtype_handles_whitespace() {
        let dt = parse_dtype(" int ").unwrap();
        assert_eq!(dt, polars::prelude::DataType::Int64);
    }

    #[test]
    fn build_dataframe_returns_error_on_bad_int() {
        let schema = vec![("value".to_string(), DataType::Int64)];
        let rows = vec![vec!["abc".to_string()]];
        assert!(build_dataframe(&schema, &rows).is_err());
    }

    #[test]
    fn build_dataframe_returns_error_on_bad_bool() {
        let schema = vec![("flag".to_string(), DataType::Boolean)];
        let rows = vec![vec!["notabool".to_string()]];
        assert!(build_dataframe(&schema, &rows).is_err());
    }

    #[test]
    fn build_dataframe_returns_error_on_duplicate_names() {
        let schema = vec![
            ("id".to_string(), DataType::Int64),
            ("id".to_string(), DataType::Int64),
        ];
        let rows = vec![vec!["1".to_string(), "2".to_string()]];
        let err = build_dataframe(&schema, &rows).unwrap_err();
        assert!(err.to_string().contains("duplicate column name"));
    }

    #[test]
    fn dropped_sender_clears_busy() {
        let mut app = ParquetApp::default();
        let (_tx, rx) = mpsc::channel::<anyhow::Result<JobUpdate>>();
        app.result_rx = Some(rx);
        app.busy = true;

        if let Some(rx) = &app.result_rx {
            match rx.try_recv() {
                Ok(res) => {
                    app.busy = false;
                    app.result_rx = None;
                    match res {
                        Ok(JobUpdate::Progress(p)) => app.progress = Some(p),
                        Ok(JobUpdate::Done(JobResult::Count(c))) => app.total_rows = c,
                        Ok(JobUpdate::Done(JobResult::DataFrame(df))) => app.edit_df = Some(df),
                        Ok(JobUpdate::Done(JobResult::Unit)) => app.status = "Done".into(),
                        Err(e) => app.status = format!("Failed: {e}"),
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    app.busy = false;
                    app.result_rx = None;
                    app.status = "Background task disconnected".into();
                }
            }
        }

        assert!(!app.busy);
        assert!(app.result_rx.is_none());
        assert_eq!(app.status, "Background task disconnected");
    }

    #[test]
    fn build_dataframe_propagates_edits() {
        let schema = vec![
            ("id".to_string(), DataType::Int64),
            ("name".to_string(), DataType::String),
        ];
        let mut rows = vec![
            vec!["1".to_string(), "Alice".to_string()],
            vec!["2".to_string(), "Bob".to_string()],
        ];
        let df1 = build_dataframe(&schema, &rows).unwrap();
        assert_eq!(
            df1.column("name").unwrap().str().unwrap().get(1),
            Some("Bob")
        );
        rows[1][1] = "Charlie".to_string();
        let df2 = build_dataframe(&schema, &rows).unwrap();
        assert_eq!(
            df2.column("name").unwrap().str().unwrap().get(1),
            Some("Charlie")
        );
        rows[0][0] = "10".to_string();
        let df3 = build_dataframe(&schema, &rows).unwrap();
        assert_eq!(df3.column("id").unwrap().i64().unwrap().get(0), Some(10));
    }

    #[test]
    fn set_cell_value_updates_date() {
        let schema = vec![("when".to_string(), DataType::Date)];
        let rows = vec![vec!["2024-01-01".to_string()]];
        let mut df = build_dataframe(&schema, &rows).unwrap();
        set_cell_value(&mut df, 0, 0, "2024-06-15").unwrap();
        use chrono::NaiveDate;
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let nd = NaiveDate::parse_from_str("2024-06-15", "%Y-%m-%d").unwrap();
        let expected = (nd - epoch).num_days() as i32;
        assert_eq!(
            df.column("when").unwrap().date().unwrap().get(0),
            Some(expected)
        );
    }

    #[test]
    fn set_cell_value_updates_datetime() {
        let schema = vec![(
            "ts".to_string(),
            DataType::Datetime(TimeUnit::Microseconds, None),
        )];
        let rows = vec![vec!["2024-01-01T00:00:00".to_string()]];
        let mut df = build_dataframe(&schema, &rows).unwrap();
        set_cell_value(&mut df, 0, 0, "2024-01-01 12:34:56").unwrap();
        use chrono::NaiveDateTime;
        let ndt =
            NaiveDateTime::parse_from_str("2024-01-01 12:34:56", "%Y-%m-%d %H:%M:%S").unwrap();
        let expected = ndt.timestamp_micros();
        assert_eq!(
            df.column("ts").unwrap().datetime().unwrap().get(0),
            Some(expected)
        );
    }

    #[test]
    fn set_cell_value_updates_time() {
        let schema = vec![("t".to_string(), DataType::Time)];
        let rows = vec![vec!["01:02:03".to_string()]];
        let mut df = build_dataframe(&schema, &rows).unwrap();
        set_cell_value(&mut df, 0, 0, "04:05:06").unwrap();
        use chrono::{NaiveTime, Timelike};
        let nt = NaiveTime::parse_from_str("04:05:06", "%H:%M:%S").unwrap();
        let expected =
            (nt.num_seconds_from_midnight() as i64) * 1_000_000_000 + nt.nanosecond() as i64;
        assert_eq!(
            df.column("t").unwrap().time().unwrap().get(0),
            Some(expected)
        );
    }

    #[test]
    fn search_matches_across_all_rows() {
        let mut app = ParquetApp::default();
        app.display_rows = 2;
        app.rows = vec![
            vec!["a".to_string()],
            vec!["b".to_string()],
            vec!["target".to_string()],
        ];
        app.search_text = "target".to_string();
        app.update_search_matches();
        assert_eq!(app.search_matches, vec![(2, 0)]);
    }

    #[test]
    fn boxplot_handles_nan() {
        let values = vec![1.0, f64::NAN, 2.0, 3.0];
        let mut filtered: Vec<f64> = values.iter().cloned().filter(|v| !v.is_nan()).collect();
        filtered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = filtered.len();
        assert!(len > 0);
        let q1 = filtered[(len as f64 * 0.25) as usize];
        let q2 = filtered[(len as f64 * 0.5) as usize];
        let q3 = filtered[(len as f64 * 0.75) as usize];
        let min = *filtered.first().unwrap();
        let max = *filtered.last().unwrap();
        let _spread = egui_plot::BoxSpread::new(min, q1, q2, q3, max);
    }
}
