use polars_parquet_learning::parquet_examples::{
    self, read_parquet_to_dataframe, write_example_data,
};
use tempfile::tempdir;

#[test]
fn writes_all_example_data() -> anyhow::Result<()> {
    let dir = tempdir()?;
    write_example_data(dir.path().to_str().unwrap())?;

    let subs = [
        "read",
        "modify",
        "write",
        "partition",
        "query",
        "xml",
        "xml_dynamic",
        "correlation",
    ];
    for s in &subs {
        assert!(dir.path().join(s).exists());
    }

    // read
    let df = read_parquet_to_dataframe(dir.path().join("read/data.parquet").to_str().unwrap())?;
    assert_eq!(df.get_column_names(), vec!["id", "name"]);

    // modify
    assert!(dir.path().join("modify/input.parquet").exists());

    // write
    assert!(dir.path().join("write/input.parquet").exists());

    // partition
    assert!(dir.path().join("partition/a.parquet").exists());
    assert!(dir.path().join("partition/b.parquet").exists());

    // query
    let df_q = read_parquet_to_dataframe(dir.path().join("query/data.parquet").to_str().unwrap())?;
    assert_eq!(df_q.height(), 3);

    // xml and xml_dynamic
    assert!(dir.path().join("xml/sample.xml").exists());
    assert!(dir.path().join("xml/templates.parquet").exists());
    assert!(dir.path().join("xml_dynamic/sample.xml").exists());
    assert!(dir.path().join("xml_dynamic/templates.parquet").exists());

    // correlation
    let df_c = read_parquet_to_dataframe(
        dir.path()
            .join("correlation/data.parquet")
            .to_str()
            .unwrap(),
    )?;
    assert_eq!(df_c.get_column_names(), vec!["a", "b", "c"]);

    Ok(())
}
