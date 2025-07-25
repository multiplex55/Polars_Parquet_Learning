use crate::parquet_examples;
use anyhow::Result;
use Polars_Parquet_Learning::xml_to_parquet;
use clap::{Args, Parser, Subcommand};

/// Top level command line arguments
#[derive(Parser)]
#[command(author, version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Read a Parquet file and print the resulting DataFrame
    Read(ReadArgs),
    /// Modify records by appending `!` to each name
    Modify(ModifyArgs),
    /// Write a Parquet file to a new location
    Write(WriteArgs),
    /// Create a small example DataFrame and save it
    Create(CreateArgs),
    /// Partition a file by a column
    Partition(PartitionArgs),
    /// Query rows by prefix or expression
    Query(QueryArgs),
    /// Convert an XML file to Parquet tables
    Xml(XmlArgs),
}

#[derive(Args)]
pub struct ReadArgs {
    /// File to read
    pub file: String,
}

#[derive(Args)]
pub struct ModifyArgs {
    /// File to modify
    pub file: String,
}

#[derive(Args)]
pub struct WriteArgs {
    /// Input file
    pub input: String,
    /// Output file
    pub output: String,
}

#[derive(Args)]
pub struct CreateArgs {
    /// Output Parquet file
    pub output: String,
}

#[derive(Args)]
pub struct PartitionArgs {
    /// Input file
    pub input: String,
    /// Column to partition by
    pub column: String,
    /// Output directory
    pub dir: String,
}

#[derive(Args)]
pub struct QueryArgs {
    /// Input file
    pub input: String,
    /// Filter by name prefix
    #[arg(long)]
    pub prefix: Option<String>,
    /// Filter with a simple expression
    #[arg(long)]
    pub expr: Option<String>,
}
#[derive(Args)]
pub struct XmlArgs {
    /// Input XML file
    pub input: String,
    /// Directory for Parquet tables
    pub output_dir: String,
    /// Write _schema.json
    #[arg(long)]
    pub schema: bool,
}


pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Read(args) => cmd_read(&args.file),
        Commands::Modify(args) => cmd_modify(&args.file),
        Commands::Write(args) => cmd_write(&args.input, &args.output),
        Commands::Create(args) => cmd_create(&args.output),
        Commands::Partition(args) => cmd_partition(&args.input, &args.column, &args.dir),
        Commands::Xml(args) => cmd_xml(&args),
        Commands::Query(args) => {
            cmd_query(&args.input, args.prefix.as_deref(), args.expr.as_deref())
        }
    }
}

fn cmd_read(file: &str) -> Result<()> {
    let df = parquet_examples::read_parquet_to_dataframe(file)?;
    println!("{}", df);
    Ok(())
}

fn cmd_modify(file: &str) -> Result<()> {
    let df = parquet_examples::read_parquet_to_dataframe(file)?;
    let mut records = parquet_examples::dataframe_to_records(&df)?;
    parquet_examples::modify_records(&mut records);
    let df = parquet_examples::records_to_dataframe(&records)?;
    println!("{}", df);
    Ok(())
}

fn cmd_write(input: &str, output: &str) -> Result<()> {
    let mut df = parquet_examples::read_parquet_to_dataframe(input)?;
    parquet_examples::write_dataframe_to_parquet(&mut df, output)?;
    println!("Wrote {output}");
    Ok(())
}

fn cmd_create(output: &str) -> Result<()> {
    use polars::prelude::*;
    let df = df!("id" => &[1i64, 2], "name" => &["a", "b"])?;
    parquet_examples::create_and_write_parquet(&df, output)?;
    println!("Wrote {output}");
    Ok(())
}

fn cmd_partition(input: &str, column: &str, dir: &str) -> Result<()> {
    let df = parquet_examples::read_parquet_to_dataframe(input)?;
    parquet_examples::write_partitioned(&df, &[column], dir)?;
    println!("Wrote partitions to {dir}");
    Ok(())
}
fn cmd_xml(args: &XmlArgs) -> Result<()> {
    xml_to_parquet::xml_to_parquet(&args.input, &args.output_dir, args.schema)?;
    println!("Wrote Parquet tables to {}", args.output_dir);
    Ok(())
}


fn cmd_query(input: &str, prefix: Option<&str>, expr: Option<&str>) -> Result<()> {
    let df = if let Some(expr) = expr {
        parquet_examples::filter_with_expr(input, expr)?
    } else {
        let p = prefix.unwrap_or("");
        parquet_examples::filter_by_name_prefix(input, p)?
    };
    println!("{}", df);
    Ok(())
}
