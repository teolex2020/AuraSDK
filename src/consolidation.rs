//! MinHash deduplication & merge.
//!
//! Rewritten from aura-cognitive memory.py consolidate().

use std::collections::{HashMap, HashSet};
use crate::record::Record;
use crate::ngram::NGramIndex;
use crate::cognitive_store::CognitiveStore;
use crate::graph;

/// Hard merge threshold — no LLM needed.
pub const CONSOLIDATION_THRESHOLD: f32 = 0.85;
/// Soft merge threshold — LLM-assisted range.
pub const CONSOLIDATION_SOFT_THRESHOLD: f32 = 0.5;

/// Result of a consolidation run.
#[derive(Debug, Default)]
pub struct ConsolidationResult {
    pub merged: usize,
    pub checked: usize,
}

/// Run hard-merge consolidation (MinHash >= 0.85).
///
/// Finds duplicate pairs, keeps the higher-importance record,
/// merges tags/connections/strength from the other.
pub fn consolidate(
    records: &mut HashMap<String, Record>,
    ngram_index: &mut NGramIndex,
    tag_index: &mut HashMap<String, HashSet<String>>,
    aura_index: &mut HashMap<String, String>,
    store: &CognitiveStore,
) -> ConsolidationResult {
    let mut result = ConsolidationResult::default();

    // Build namespace lookup for O(1) filtering
    let ns_map: HashMap<&str, &str> = records.iter()
        .map(|(id, r)| (id.as_str(), r.namespace.as_str()))
        .collect();

    // Find similar pairs (global MinHash) and pre-filter to same-namespace only
    let all_pairs = ngram_index.find_similar_pairs(CONSOLIDATION_THRESHOLD);
    let pairs: Vec<_> = all_pairs.into_iter()
        .filter(|(id_a, id_b, _)| ns_map.get(id_a.as_str()) == ns_map.get(id_b.as_str()))
        .collect();
    result.checked = pairs.len();

    // Process pairs (avoid double-processing)
    let mut removed: HashSet<String> = HashSet::new();

    for (id_a, id_b, _sim) in &pairs {
        if removed.contains(id_a) || removed.contains(id_b) {
            continue;
        }

        let imp_a = records.get(id_a).map(|r| r.importance()).unwrap_or(0.0);
        let imp_b = records.get(id_b).map(|r| r.importance()).unwrap_or(0.0);

        let (keep_id, remove_id) = if imp_a >= imp_b {
            (id_a.clone(), id_b.clone())
        } else {
            (id_b.clone(), id_a.clone())
        };

        graph::merge_records(
            &keep_id,
            &remove_id,
            records,
            ngram_index,
            tag_index,
            aura_index,
            store,
        );

        removed.insert(remove_id);
        result.merged += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::Level;

    #[test]
    fn test_consolidation_threshold() {
        assert!(CONSOLIDATION_THRESHOLD > CONSOLIDATION_SOFT_THRESHOLD);
        assert!(CONSOLIDATION_THRESHOLD <= 1.0);
    }
}
