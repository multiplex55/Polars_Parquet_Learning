//! Simple egui application for interacting with Parquet files.

// Expose example functions for GUI callbacks or tests.
pub mod background;
pub mod cli;
pub mod parquet_examples;

use anyhow::Result;
use background::JobResult;
use clap::Parser;
use eframe::egui;
use egui_extras::{Column as TableColumn, TableBuilder};
#[cfg(feature = "plotting")]
use egui_plot::{BarChart, BoxElem, BoxPlot, Line, Plot, PlotPoints, Points};
use polars::prelude::SortMultipleOptions;
use polars::prelude::*;
use rfd::FileDialog;
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
    /// Temporary inputs for adding columns
    new_col_name: String,
    new_col_type: String,
    /// Selected column when partitioning via write mode
    partition_column: Option<String>,
    /// Selected columns when using the dedicated partition operation
    partition_columns: Vec<String>,
    /// Query prefix when using the query operation
    query_prefix: String,
    /// Expression string when using the query operation
    query_expr: String,
    /// Status message shown to the user
    status: String,
    /// Number of rows to display from the current DataFrame
    display_rows: usize,
    #[cfg(feature = "plotting")]
    /// Selected column to plot
    plot_column: Option<String>,
    #[cfg(feature = "plotting")]
    /// Second column for scatter plots
    plot_y_column: Option<String>,
    #[cfg(feature = "plotting")]
    /// Type of plot to display
    plot_type: PlotType,
    /// Tokio runtime for background tasks
    runtime: tokio::runtime::Runtime,
    /// Receives results from background jobs
    result_rx: Option<std::sync::mpsc::Receiver<anyhow::Result<background::JobResult>>>,
    /// Metadata for the loaded Parquet file
    metadata: Option<parquet::file::metadata::ParquetMetaData>,
    /// Schema of the loaded DataFrame
    loaded_schema: Vec<(String, polars::prelude::DataType)>,
    /// Show or hide the schema information
    show_schema: bool,
    /// Indicates an operation is running
    busy: bool,
    /// Treat the selected path as a directory when reading
    use_directory: bool,
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
            new_col_name: String::new(),
            new_col_type: String::new(),
            partition_column: None,
            partition_columns: Vec::new(),
            query_prefix: String::new(),
            query_expr: String::new(),
            status: String::new(),
            display_rows: 5,
            #[cfg(feature = "plotting")]
            plot_column: None,
            #[cfg(feature = "plotting")]
            plot_y_column: None,
            #[cfg(feature = "plotting")]
            plot_type: PlotType::default(),
            runtime: tokio::runtime::Runtime::new().expect("runtime"),
            result_rx: None,
            metadata: None,
            loaded_schema: Vec::new(),
            show_schema: false,
            busy: false,
            use_directory: false,
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
                        val
                            .parse::<i64>()
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
                                    .or_else(|_| NaiveDateTime::parse_from_str(val, "%Y-%m-%dT%H:%M:%S"))
                                    .map(|dt| dt.timestamp_micros())
                            })
                            .map_err(|e| anyhow::anyhow!("failed to parse '{val}' as datetime: {e}"))?;
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
                        Ok((t.num_seconds_from_midnight() as i64) * 1_000_000_000 + t.nanosecond() as i64)
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

impl ParquetApp {
    /// Spawn a background task to read the current `file_path`.
    fn start_read(&mut self) {
        let path = self.file_path.clone();
        let use_dir = self.use_directory;
        let (tx, rx) = mpsc::channel();
        self.result_rx = Some(rx);
        self.busy = true;
        self.status = "Reading...".into();
        self.runtime.spawn(async move {
            let res = if use_dir {
                background::read_directory(path).await
            } else {
                background::read_dataframe(path).await
            };
            let _ = tx.send(res);
        });
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
                ui.label(format!("Rows: {}", df.height()));
                ui.horizontal(|ui| {
                    ui.label("Rows to display:");
                    ui.add(egui::DragValue::new(&mut self.display_rows).clamp_range(1..=1000));
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
                                for col in head.get_columns() {
                                    let val =
                                        col.get(row_idx).map(|v| v.to_string()).unwrap_or_default();
                                    row.col(|ui| {
                                        ui.label(val);
                                    });
                                }
                            });
                        }
                    });

                if let Ok(summary) = parquet_examples::summarize_dataframe(df) {
                    ui.separator();
                    ui.label("Statistics");
                    ui.label(format!("Rows: {}", summary.rows));
                    ui.label(format!("Columns: {}", summary.columns));

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
                                        let min =
                                            values.iter().cloned().fold(f64::INFINITY, f64::min);
                                        let max = values
                                            .iter()
                                            .cloned()
                                            .fold(f64::NEG_INFINITY, f64::max);
                                        let bins = 10usize;
                                        let step = (max - min) / bins as f64;
                                        let mut counts = vec![0f64; bins];
                                        for v in values.iter() {
                                            let mut idx = ((v - min) / step).floor() as usize;
                                            if idx >= bins {
                                                idx = bins - 1;
                                            }
                                            counts[idx] += 1.0;
                                        }
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
                                        Plot::new("histogram").show(ui, |plot_ui| {
                                            plot_ui.bar_chart(BarChart::new(bars));
                                        });
                                    }
                                    PlotType::Line => {
                                        let points: PlotPoints = values
                                            .iter()
                                            .enumerate()
                                            .map(|(i, v)| [i as f64, *v])
                                            .collect();
                                        Plot::new("line").show(ui, |plot_ui| {
                                            plot_ui.line(Line::new(points));
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
                                                            ca.into_no_null_iter().map(|v| v as f64).collect()
                                                        })
                                                    })
                                                    .unwrap_or_default();
                                                let points: PlotPoints = values
                                                    .iter()
                                                    .cloned()
                                                    .zip(y_vals)
                                                    .map(|(x, y)| [x, y])
                                                    .collect();
                                                Plot::new("scatter").show(ui, |plot_ui| {
                                                    plot_ui.points(egui_plot::Points::new(points));
                                                });
                                            }
                                        }
                                    }
                                    PlotType::BoxPlot => {
                                        use egui_plot::{BoxElem, BoxPlot};
                                        if !values.is_empty() {
                                            let mut sorted = values.clone();
                                            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                            let q1 = sorted[(sorted.len() as f64 * 0.25) as usize];
                                            let q2 = sorted[(sorted.len() as f64 * 0.5) as usize];
                                            let q3 = sorted[(sorted.len() as f64 * 0.75) as usize];
                                            let min = *sorted.first().unwrap();
                                            let max = *sorted.last().unwrap();
                                            let elem = BoxElem::new(0.0, q1, q2, q3, min, max);
                                            Plot::new("boxplot").show(ui, |plot_ui| {
                                                plot_ui.box_plot(BoxPlot::new(vec![elem]));
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
                if let Ok(res) = rx.try_recv() {
                    self.busy = false;
                    self.result_rx = None;
                    match res {
                        Ok(JobResult::DataFrame(df)) => {
                            if self.use_directory {
                                self.status = format!("Combined {} rows", df.height());
                            } else {
                                self.status = format!("Loaded {} rows", df.height());
                                if let Ok(meta) =
                                    parquet_examples::read_parquet_metadata(&self.file_path)
                                {
                                    self.metadata = Some(meta);
                                }
                            }
                            self.loaded_schema = df
                                .get_column_names()
                                .into_iter()
                                .zip(df.dtypes())
                                .map(|(n, t)| (n.to_string(), t))
                                .collect();
                            self.edit_df = Some(df);
                        }
                        Ok(JobResult::Unit) => {
                            self.status = "Done".into();
                        }
                        Err(e) => self.status = format!("Failed: {e}"),
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
                        _ => {
                            if !self.use_directory {
                                dialog = dialog.add_filter("Parquet", &["parquet"]);
                            }
                        }
                    }
                    let picked = if self.use_directory {
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
                        _ => {
                            dialog = dialog.add_filter("Parquet", &["parquet"]);
                        }
                    }
                    if let Some(path) = dialog.save_file() {
                        self.save_path = path.display().to_string();
                    }
                }
            });

            match self.operation {
                Operation::Create => {
                    ui.horizontal(|ui| {
                        ui.label("New column:");
                        ui.text_edit_singleline(&mut self.new_col_name);
                        ui.text_edit_singleline(&mut self.new_col_type);
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
                }
                _ => {}
            }

            if let Some(df) = &self.edit_df {
                ui.separator();
                if let Some(meta) = &self.metadata {
                    ui.label("Metadata");
                    egui::Grid::new("meta_grid").striped(true).show(ui, |ui| {
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
                ui.add(egui::Spinner::new());
            }

            // Run the selected action
            let requires_df = matches!(
                self.operation,
                Operation::Write | Operation::Partition | Operation::Query
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
                                            self.loaded_schema = df
                                                .get_column_names()
                                                .into_iter()
                                                .zip(df.dtypes())
                                                .map(|(n, t)| (n.to_string(), t))
                                                .collect();
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
                                self.status = "Writing...".into();
                                self.runtime.spawn(async move {
                                    let res = background::write_dataframe(df, file).await;
                                    let _ = tx.send(res);
                                });
                            }
                        } else {
                            self.status = "Load or create a DataFrame first.".into();
                        }
                    }
                    Operation::WriteCsv => {
                        if let Some(df) = &self.edit_df {
                            match parquet_examples::write_dataframe_to_csv(df, &self.file_path) {
                                Ok(_) => self.status = format!("Wrote {}", self.file_path),
                                Err(e) => self.status = format!("Write failed: {e}"),
                            }
                        }
                    }
                    Operation::WriteJson => {
                        if let Some(df) = &self.edit_df {
                            match parquet_examples::write_dataframe_to_json(df, &self.file_path) {
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
                            self.loaded_schema = df
                                .get_column_names()
                                .into_iter()
                                .zip(df.dtypes())
                                .map(|(n, t)| (n.to_string(), t))
                                .collect();
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
                                    self.loaded_schema = df
                                        .get_column_names()
                                        .into_iter()
                                        .zip(df.dtypes())
                                        .map(|(n, t)| (n.to_string(), t))
                                        .collect();
                                    self.edit_df = Some(df);
                                }
                                Err(e) => self.status = format!("Query failed: {e}"),
                            }
                        } else {
                            self.status = "Load or create a DataFrame first.".into();
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
}
