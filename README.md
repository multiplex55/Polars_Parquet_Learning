# Polars Parquet Learning

Minimal playground demonstrating how to work with [Polars](https://pola.rs/) and
[Parquet](https://parquet.apache.org/) in Rust.  A tiny GUI built on top of
[egui](https://github.com/emilk/egui) allows selecting a file and performing a
basic read/modify/write cycle.

## Building and running

This project uses the Rust 2024 edition.  A recent stable toolchain is
recommended.  To compile everything and launch the GUI with plotting support
enabled by default:

```bash
cargo run
```

For release builds use `cargo build --release` followed by the generated binary
in `target/release/Polars_Parquet_Learning`.

## Example workflow

1. Start the application with `cargo run`.
2. Enter the path to a Parquet file in the **File** field, or drag a file onto the window.
3. Choose one of the available operations.
   * **Read**: load an existing file into a `DataFrame` using the lazy API.
   * **Modify**: convert rows to typed `Record`s and append `!` to each name.
   * **Write**: write the current in-memory `DataFrame` back to the path in **File**.
   * **Create**: define a schema and rows in the UI then save to the **Save** path.
   * **Partition**: split the loaded `DataFrame` by one or more columns and
     write each combination to nested folders under the **Save** path.
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
button. Files can also be dragged onto the window to automatically populate the
path. All heavy lifting is handled by Polars; the GUI simply wires the chosen
action to the example functions.

The application includes a simple chart viewer using
[`egui_plot`](https://crates.io/crates/egui_plot). A drop-down in the preview
panel lets you pick one or two numeric columns and choose between histogram,
line, scatter or box plots for that data.

A small statistics table summarising the loaded `DataFrame` is also shown in the
preview panel. It lists row/column counts along with the output of
`DataFrame::describe` for quick inspection.

## Additional examples

The `parquet_examples` module includes several helper functions that showcase
common patterns:

* **`read_selected_columns`** – lazily read only a subset of columns from a
  Parquet file.
* **`filter_by_name_prefix`** – apply an expression to filter rows before they
  are collected into a `DataFrame`.
* **`read_parquet_metadata`** – inspect low level metadata with the `parquet`
  crate without loading the entire file.
* **`read_partitions`** – load all Parquet files from a directory pattern with a single
  `scan_parquet` call.

See [`src/parquet_examples.rs`](src/parquet_examples.rs) for implementation
details and tests for each use case.

## Creating a DataFrame from scratch

Choose the **Create** mode in the GUI to build a brand new dataset. Add column
names and pick their types from a drop-down list, then enter one or more rows of
data. Providing a path in the **Save** field will write the result as a Parquet
file when **Run** is clicked. The helper function
[`create_dataframe`](src/parquet_examples.rs) shows how to accomplish the same
programmatically.

Allowed column types are `int`/`i64`, `str`/`string`, `float`/`f64`,
`bool`/`boolean`, `date`, `datetime` and `time`.

## Command line

Running the binary with additional arguments invokes a CLI instead of the GUI.
Each subcommand mirrors one of the operations available in the application.

```
cargo run -- <SUBCOMMAND> [OPTIONS]
```

Example reading a file:

```
cargo run -- read path/to/file.parquet
```

When writing files the `write` subcommand accepts an optional
`--compression` flag which defaults to `snappy`:

```
cargo run -- write input.parquet output.parquet --compression gzip
```

## Benchmarks

Running `cargo bench` builds a small benchmark comparing the old per-file
collection approach against using `scan_parquet` over a directory pattern.

## Converting XML to normalized Parquet tables

The CLI can also transform a small hierarchical XML format into multiple
normalized Parquet tables. The input must contain a `<root>` element with any
combination of `<template>`, `<message>` and `<repository>` children. Templates
may include nested `<field>` elements while messages can contain one or more
`<part>` nodes.

Use the following command to perform the conversion:

```
cargo run -- xml <input.xml> <output_dir> [--schema]
```

Each entity is written to `<output_dir>` as its own file (for example
`templates.parquet`, `fields.parquet`, `messages.parquet`, `parts.parquet` and
`repositories.parquet`). Foreign key columns follow the `<table>_id` naming
convention and an optional `_schema.json` lists these relationships when the
`--schema` flag is supplied.

Programmatic access is provided by the [`xml_to_parquet`](src/xml_to_parquet.rs)
module which exposes `parse_xml`, `flatten_to_tables`, `write_tables` and the
convenience `xml_to_parquet` function.

The GUI offers the same functionality via the **XML** mode. Select an input
file in the **File** field, choose an output directory in **Save** and decide
whether to write a `_schema.json` using the checkbox before clicking **Run**.

Additional helper functions in [`src/xml_examples.rs`](src/xml_examples.rs)
demonstrate a full round trip using the bundled
[`sample.xml`](tests/fixtures/sample.xml). They parse the XML into a `Root`,
convert it to `DataFrame`s, write them as Parquet files, read the tables back
into a `Root` and finally serialize it to an XML string again. See
[`tests/xml_round_trip.rs`](tests/xml_round_trip.rs) for usage.

