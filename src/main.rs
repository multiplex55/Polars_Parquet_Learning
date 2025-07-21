//! Simple egui application for interacting with Parquet files.

// Expose example functions for GUI callbacks or tests.
pub mod parquet_examples;

use eframe::egui;

/// Defines the user selected operation on the Parquet file.
#[derive(Debug, PartialEq)]
enum Operation {
    Read,
    Modify,
    Write,
}

impl Default for Operation {
    fn default() -> Self {
        Operation::Read
    }
}

/// Main application state.
#[derive(Default)]
struct ParquetApp {
    /// Path to the Parquet file entered by the user.
    file_path: String,
    /// The operation the user would like to perform.
    operation: Operation,
}

impl ParquetApp {
    /// Create a new [`ParquetApp`] instance.
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
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
            });

            // Placeholder button to perform the chosen action
            if ui.button("Run").clicked() {
                // Actual Parquet logic would go here
                println!("Running {:?} on {}", self.operation, self.file_path);
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
