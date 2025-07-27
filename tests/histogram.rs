use Polars_Parquet_Learning::parquet_examples::compute_histogram;

#[test]
fn histogram_respects_bins() {
    let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let (counts, _min, _step) = compute_histogram(&values, 5, None);
    assert_eq!(counts.len(), 5);
    let total: f64 = counts.iter().sum();
    assert_eq!(total as usize, values.len());
}

#[test]
fn histogram_constant_values() {
    let values = vec![1.0, 1.0, 1.0, 1.0];
    let (counts, _min, _step) = compute_histogram(&values, 3, None);
    assert_eq!(counts.len(), 3);
    assert_eq!(counts[0] as usize, values.len());
    assert!(counts[1..].iter().all(|&c| c == 0.0));
}

#[test]
fn histogram_swaps_invalid_range() {
    let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let expected = compute_histogram(&values, 5, Some((0.0, 5.0)));
    let reversed = compute_histogram(&values, 5, Some((5.0, 0.0)));
    assert_eq!(expected, reversed);
}
