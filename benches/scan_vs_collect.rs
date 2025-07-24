use criterion::{criterion_group, criterion_main, Criterion};
use polars::prelude::*;
use tempfile::tempdir;
use polars_parquet_learning::parquet_examples;

fn make_dir() -> std::path::PathBuf {
    let dir = tempdir().unwrap();
    for i in 0..5 {
        let mut df = df!("id" => &[i as i64], "name" => &["a"]).unwrap();
        let path = dir.path().join(format!("p{i}.parquet"));
        parquet_examples::write_dataframe_to_parquet(&mut df, path.to_str().unwrap()).unwrap();
    }
    dir.into_path()
}

fn old_collect(dir: &str) -> DataFrame {
    let mut dfs = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|e| e.to_str()) == Some("parquet") {
            let lf = LazyFrame::scan_parquet(p.to_str().unwrap(), ScanArgsParquet::default()).unwrap();
            dfs.push(lf.collect().unwrap());
        }
    }
    polars::functions::concat_df(&dfs).unwrap()
}

fn bench(c: &mut Criterion) {
    let dir = make_dir();
    c.bench_function("old_collect", |b| b.iter(|| old_collect(dir.to_str().unwrap())));
    c.bench_function("scan_parquet", |b| b.iter(|| parquet_examples::read_partitions(dir.to_str().unwrap()).unwrap()));
}

criterion_group!(benches, bench);
criterion_main!(benches);
