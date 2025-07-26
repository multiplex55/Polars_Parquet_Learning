use Polars_Parquet_Learning::parquet_examples::compute_histogram;

#[test]
fn histogram_respects_bins() {
    let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let (counts, _min, _step) = compute_histogram(&values, 5, None);
    assert_eq!(counts.len(), 5);
    let total: f64 = counts.iter().sum();
    assert_eq!(total as usize, values.len());
}
