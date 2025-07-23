//! Simple egui application for interacting with Parquet files.

// Expose example functions for GUI callbacks or tests.
pub mod background;
pub mod parquet_examples;

use anyhow::Result;
use background::JobResult;
use eframe::egui;
use polars::prelude::*;
use rfd::FileDialog;
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
    /// Selected column when partitioning
    partition_column: Option<String>,
    /// Query prefix when using the query operation
    query_prefix: String,
    /// Expression string when using the query operation
    query_expr: String,
    /// Status message shown to the user
    status: String,
    /// Number of rows to display from the current DataFrame
    display_rows: usize,
    /// Tokio runtime for background tasks
    runtime: tokio::runtime::Runtime,
    /// Receives results from background jobs
    result_rx: Option<std::sync::mpsc::Receiver<anyhow::Result<background::JobResult>>>,
    /// Metadata for the loaded Parquet file
    metadata: Option<parquet::file::metadata::ParquetMetaData>,
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
            query_prefix: String::new(),
            query_expr: String::new(),
            status: String::new(),
            display_rows: 5,
            runtime: tokio::runtime::Runtime::new().expect("runtime"),
            result_rx: None,
            metadata: None,
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
    use polars::prelude::DataType;
    match t.trim().to_lowercase().as_str() {
        "int" | "i64" => Ok(DataType::Int64),
        "str" | "string" => Ok(DataType::String),
        "float" | "f64" => Ok(DataType::Float64),
        "bool" | "boolean" => Ok(DataType::Boolean),
        _ => Err(anyhow::anyhow!("unsupported type")),
    }
}

fn build_dataframe(schema: &[(String, DataType)], rows: &[Vec<String>]) -> Result<DataFrame> {
    use polars::prelude::IntoColumn;
    let mut cols: Vec<Column> = Vec::new();
    for (idx, (name, dtype)) in schema.iter().enumerate() {
        match dtype {
            DataType::Int64 => {
                let data: Vec<i64> = rows
                    .iter()
                    .map(|r| r.get(idx).and_then(|s| s.parse::<i64>().ok()).unwrap_or(0))
                    .collect();
                cols.push(Series::new(name.as_str().into(), data).into_column());
            }
            DataType::Float64 => {
                let data: Vec<f64> = rows
                    .iter()
                    .map(|r| {
                        r.get(idx)
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0)
                    })
                    .collect();
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
                    .map(|r| {
                        r.get(idx)
                            .and_then(|s| match s.to_lowercase().as_str() {
                                "true" | "1" => Some(true),
                                "false" | "0" => Some(false),
                                _ => None,
                            })
                            .unwrap_or(false)
                    })
                    .collect();
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

impl eframe::App for ParquetApp {
    /// Called each frame to update the UI.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Polars Parquet Learning");

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
                        egui::ComboBox::from_label("Column")
                            .selected_text(self.partition_column.clone().unwrap_or_default())
                            .show_ui(ui, |ui| {
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
                ui.horizontal(|ui| {
                    ui.label("Rows to display:");
                    ui.add(egui::DragValue::new(&mut self.display_rows).clamp_range(1..=1000));
                });
                egui::ScrollArea::both().max_height(200.0).show(ui, |ui| {
                    egui::Grid::new("df_preview").striped(true).show(ui, |ui| {
                        for name in df.get_column_names() {
                            ui.label(name.to_string());
                        }
                        ui.end_row();
                        let rows = df.height().min(self.display_rows);
                        for i in 0..rows {
                            for s in df.get_columns() {
                                ui.label(s.get(i).map(|v| v.to_string()).unwrap_or_default());
                            }
                            ui.end_row();
                        }
                    });
                });
                if let Some(meta) = &self.metadata {
                    ui.separator();
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
                    Operation::Modify => {
                        match parquet_examples::read_parquet_to_dataframe(&self.file_path) {
                            Ok(df) => {
                                if let Ok(mut rec) = parquet_examples::dataframe_to_records(&df) {
                                    parquet_examples::modify_records(&mut rec);
                                    match parquet_examples::records_to_dataframe(&rec) {
                                        Ok(df) => {
                                            self.status = "Modified records".into();
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
                                match parquet_examples::write_partitioned(df, col, &self.save_path)
                                {
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
                            self.edit_df = Some(df);
                        }
                        Err(e) => self.status = format!("Create failed: {e}"),
                    },
                    Operation::Partition => {
                        if let (Some(df), Some(col)) = (&self.edit_df, &self.partition_column) {
                            match parquet_examples::write_partitioned(df, col, &self.save_path) {
                                Ok(_) => {
                                    self.status = format!("Wrote partitions to {}", self.save_path)
                                }
                                Err(e) => self.status = format!("Partition failed: {e}"),
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
}
