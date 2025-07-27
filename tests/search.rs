use Polars_Parquet_Learning::search::{find_matches, next_index, prev_index};

#[test]
fn navigate_matches_cycles() {
    let rows = vec![
        vec!["apple".to_string(), "banana".to_string()],
        vec!["carrot".to_string(), "apple".to_string()],
    ];
    let matches = find_matches(&rows, "apple", false);
    assert_eq!(matches, vec![(0, 0), (1, 1)]);
    let mut idx = 0;
    idx = next_index(idx, &matches);
    assert_eq!(idx, 1);
    idx = next_index(idx, &matches);
    assert_eq!(idx, 0);
    idx = prev_index(idx, &matches);
    assert_eq!(idx, 1);
    idx = prev_index(idx, &matches);
    assert_eq!(idx, 0);
}

#[test]
fn navigation_with_no_matches() {
    let rows = vec![vec!["a".to_string()]];
    let matches = find_matches(&rows, "z", false);
    assert!(matches.is_empty());
    assert_eq!(next_index(0, &matches), 0);
    assert_eq!(prev_index(0, &matches), 0);
}

#[test]
fn case_insensitive_search() {
    let rows = vec![
        vec!["Apple".to_string(), "banana".to_string()],
        vec!["carrot".to_string(), "APPLE".to_string()],
    ];
    let matches = find_matches(&rows, "apple", true);
    assert_eq!(matches, vec![(0, 0), (1, 1)]);
    let no_matches = find_matches(&rows, "apple", false);
    assert!(no_matches.is_empty());
}
