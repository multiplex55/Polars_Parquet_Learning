//! Simple egui application for interacting with Parquet files.

// Expose example functions for GUI callbacks or tests.
pub mod parquet_examples;

use anyhow::Result;
use eframe::egui;
use polars::prelude::*;

/// Defines the user selected operation on the Parquet file.
#[derive(Debug, PartialEq)]
enum Operation {
    Read,
    Modify,
    Write,
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
    match t.to_lowercase().as_str() {
        "int" | "i64" => Ok(DataType::Int64),
        "str" | "string" => Ok(DataType::String),
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
            DataType::String => {
                let data: Vec<String> = rows
                    .iter()
                    .map(|r| r.get(idx).cloned().unwrap_or_default())
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

            // Input field for the Parquet file path
            ui.horizontal(|ui| {
                ui.label("File:");
                ui.text_edit_singleline(&mut self.file_path);
            });

            // Radio buttons to pick an operation
            ui.horizontal(|ui| {
                ui.label("Operation:");
                ui.radio_value(&mut self.operation, Operation::Read, "Read");
                ui.radio_value(&mut self.operation, Operation::Modify, "Modify");
                ui.radio_value(&mut self.operation, Operation::Write, "Write");
                ui.radio_value(&mut self.operation, Operation::Create, "Create");
                ui.radio_value(&mut self.operation, Operation::Partition, "Partition");
                ui.radio_value(&mut self.operation, Operation::Query, "Query");
            });

            ui.horizontal(|ui| {
                ui.label("Save:");
                ui.text_edit_singleline(&mut self.save_path);
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
                }
                _ => {}
            }

            // Run the selected action
            if ui.button("Run").clicked() {
                match self.operation {
                    Operation::Read => {
                        if let Ok(df) = parquet_examples::read_parquet_to_dataframe(&self.file_path)
                        {
                            println!("Loaded {} rows", df.height());
                            self.edit_df = Some(df);
                        }
                    }
                    Operation::Modify => {
                        if let Ok(df) = parquet_examples::read_parquet_to_dataframe(&self.file_path)
                        {
                            if let Ok(mut rec) = parquet_examples::dataframe_to_records(&df) {
                                parquet_examples::modify_records(&mut rec);
                                if let Ok(df) = parquet_examples::records_to_dataframe(&rec) {
                                    self.edit_df = Some(df);
                                }
                            }
                        }
                    }
                    Operation::Write => {
                        if let Some(df) = &self.edit_df {
                            if let Some(col) = &self.partition_column {
                                if parquet_examples::write_partitioned(df, col, &self.save_path).is_ok() {
                                    println!("Wrote partitions to {}", self.save_path);
                                }
                            } else if parquet_examples::write_dataframe_to_parquet(df, &self.file_path).is_ok() {
                                println!("Wrote {}", self.file_path);
                            }
                        }
                    }
                    Operation::Create => {
                        if let Ok(df) = build_dataframe(&self.schema, &self.rows) {
                            if !self.save_path.is_empty() {
                                let _ = parquet_examples::create_and_write_parquet(
                                    &df,
                                    &self.save_path,
                                );
                            }
                            self.edit_df = Some(df);
                        }
                    }
                    Operation::Partition => {
                        if let (Some(df), Some(col)) = (&self.edit_df, &self.partition_column) {
                            let _ = parquet_examples::write_partitioned(df, col, &self.save_path);
                        }
                    }
                    Operation::Query => {
                        if let Ok(df) = parquet_examples::filter_by_name_prefix(
                            &self.file_path,
                            &self.query_prefix,
                        ) {
                            println!("Query returned {} rows", df.height());
                            self.edit_df = Some(df);
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
