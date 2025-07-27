/// Search utilities for table data.

/// Return all cell coordinates containing `query`.
pub fn find_matches(rows: &[Vec<String>], query: &str, ignore_case: bool) -> Vec<(usize, usize)> {
    if query.is_empty() {
        return Vec::new();
    }
    let query_cmp = if ignore_case {
        query.to_lowercase()
    } else {
        query.to_string()
    };
    let mut out = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        for (c, cell) in row.iter().enumerate() {
            let hay = if ignore_case { cell.to_lowercase() } else { cell.clone() };
            if hay.contains(&query_cmp) {
                out.push((r, c));
            }
        }
    }
    out
}

/// Advance to the next match index cycling to the start when at the end.
pub fn next_index(current: usize, matches: &[(usize, usize)]) -> usize {
    if matches.is_empty() {
        0
    } else {
        (current + 1) % matches.len()
    }
}

/// Move to the previous match index cycling to the end when at the start.
pub fn prev_index(current: usize, matches: &[(usize, usize)]) -> usize {
    if matches.is_empty() {
        0
    } else if current == 0 {
        matches.len() - 1
    } else {
        current - 1
    }
}
