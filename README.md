# Polars Parquet Learning

Minimal playground demonstrating how to work with [Polars](https://pola.rs/) and
[Parquet](https://parquet.apache.org/) in Rust.  A tiny GUI built on top of
[egui](https://github.com/emilk/egui) allows selecting a file and performing a
basic read/modify/write cycle.

## Building and running

This project uses the Rust 2024 edition.  A recent stable toolchain is
recommended.  To compile everything and launch the GUI:

```bash
cargo run
```

For release builds use `cargo build --release` followed by the generated binary
in `target/release/Polars_Parquet_Learning`.

## Example workflow

1. Start the application with `cargo run`.
2. Enter the path to a Parquet file in the **File** field.
3. Choose one of the available operations.
   * **Read**: load an existing file into a `DataFrame` using the lazy API.
   * **Modify**: convert rows to typed `Record`s and append `!` to each name.
   * **Write**: write the current in-memory `DataFrame` back to the path in **File**.
   * **Create**: define a schema and rows in the UI then save to the **Save** path.
   * **Partition**: split the loaded `DataFrame` by the selected column and
     write partitions using the **Save** path as a prefix.
   * **Query**: filter the file by a name prefix.
4. Use the **Save** field to specify where newly created or partitioned data
   should be written.
5. Click **Run** to perform the selected action. Status is printed to stdout.

The helper functions live in [`src/parquet_examples.rs`](src/parquet_examples.rs)
and are covered by the `round_trip` unit test.

## Polars and Parquet notes

Polars stores data in columnar Arrow memory which is efficient for analytical
workloads.  Large datasets may require substantial RAM; prefer the lazy API when
possible.  Parquet enforces strict schemas so field names and types must match
when writing new files.  The `polars` crate already includes a mature Parquet
implementation so the lower level `parquet` dependency is optional here.

## GUI overview

The interface is implemented using
[eframe](https://docs.rs/eframe/latest/eframe/) and
[egui](https://docs.rs/egui/latest/egui/).  When launched a small window is
displayed containing a text field, radio buttons for the operation and a **Run**
button.  All heavy lifting is handled by Polars; the GUI simply wires the chosen
action to the example functions.

## Additional examples

The `parquet_examples` module includes several helper functions that showcase
common patterns:

* **`read_selected_columns`** – lazily read only a subset of columns from a
  Parquet file.
* **`filter_by_name_prefix`** – apply an expression to filter rows before they
  are collected into a `DataFrame`.
* **`read_parquet_metadata`** – inspect low level metadata with the `parquet`
  crate without loading the entire file.

See [`src/parquet_examples.rs`](src/parquet_examples.rs) for implementation
details and tests for each use case.

