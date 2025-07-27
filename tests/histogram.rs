use polars_parquet_learning::parquet_examples::compute_histogram;

#[test]
fn histogram_respects_bins() {
    let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let (counts, _min, _step) = compute_histogram(&values, 5, None).unwrap();
    assert_eq!(counts.len(), 5);
    let total: f64 = counts.iter().sum();
    assert_eq!(total as usize, values.len());
}

#[test]
fn histogram_constant_values() {
    let values = vec![1.0, 1.0, 1.0, 1.0];
    let (counts, _min, _step) = compute_histogram(&values, 3, None).unwrap();
    assert_eq!(counts.len(), 3);
    assert_eq!(counts[0] as usize, values.len());
    assert!(counts[1..].iter().all(|&c| c == 0.0));
}

#[test]
fn histogram_invalid_range_error() {
    let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let result = compute_histogram(&values, 5, Some((5.0, 0.0)));
    assert!(result.is_err());
}
