use polars_parquet_learning::parquet_examples::{drop_column, rename_column, reorder_columns};
use polars::prelude::*;

#[test]
fn column_operations() -> anyhow::Result<()> {
    let mut df = df!("a" => &[1i64,2], "b" => &["x","y"], "c" => &[true,false])?;
    rename_column(&mut df, "b", "name")?;
    assert!(df.column("name").is_ok());
    assert!(df.column("b").is_err());

    drop_column(&mut df, "c")?;
    assert_eq!(df.width(), 2);
    assert!(df.column("c").is_err());

    let order = vec!["name".to_string(), "a".to_string()];
    reorder_columns(&mut df, &order)?;
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(names, vec!["name".to_string(), "a".to_string()]);
    Ok(())
}
