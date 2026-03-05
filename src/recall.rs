//! RRF Fusion recall pipeline — the KEY intellectual property.
//!
//! Rewritten from aura-cognitive recall.py.
//!
//! Pipeline:
//! Query → [3 parallel ranked lists] → RRF Fusion → Graph Walk → Causal Walk → Rank → Format

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::instrument;

use crate::record::Record;
use crate::levels::Level;
use crate::ngram::NGramIndex;
use crate::sdr::SDRInterpreter;
use crate::index::InvertedIndex;
use crate::storage::AuraStorage;
use crate::graph::SessionTracker;
use crate::record::DEFAULT_NAMESPACE;
use crate::trust::{self, TrustConfig};

/// RRF constant (higher = more weight to top ranks).
pub const RRF_K: usize = 60;

/// Check if a record belongs to one of the given namespaces.
#[inline]
fn in_namespace(rec: &Record, namespaces: &[&str]) -> bool {
    namespaces.contains(&rec.namespace.as_str())
}

/// Graph walk parameters.
pub const GRAPH_WALK_MAX_HOPS: usize = 2;
pub const GRAPH_WALK_DAMPING: f32 = 0.6;
pub const GRAPH_WALK_MIN_SCORE: f32 = 0.05;
pub const GRAPH_WALK_MAX_EXPANDED: usize = 30;

/// Maximum causal chain depth.
const CAUSAL_MAX_DEPTH: usize = 3;

/// Result of the recall pipeline.
pub struct RecallResult {
    /// Scored records: (score, Record).
    pub scored: Vec<(f32, Record)>,
    /// Timing breakdown in microseconds.
    pub timings: HashMap<String, u64>,
}

// ── Signal Collection ──

/// Collect SDR similarity results from aura-memory engine.
#[instrument(skip_all, fields(top_k))]
pub fn collect_sdr(
    sdr: &SDRInterpreter,
    index: &InvertedIndex,
    storage: &AuraStorage,
    aura_index: &HashMap<String, String>,
    records: &HashMap<String, Record>,
    query: &str,
    top_k: usize,
    namespaces: &[&str],
) -> Vec<(String, f32)> {
    // Generate query SDR
    let query_sdr = sdr.text_to_sdr(query, false);
    if query_sdr.is_empty() {
        return vec![];
    }

    // Search inverted index
    let candidates = index.search(&query_sdr, top_k * 2, 1);

    let mut results = Vec::new();
    let cache = storage.header_cache.read();

    for (aura_id, _overlap) in candidates {
        // Map aura_id → record_id
        let record_id = if let Some(rid) = aura_index.get(&aura_id) {
            rid.clone()
        } else {
            // Fallback: try direct match
            if records.contains_key(&aura_id) {
                aura_id.clone()
            } else {
                continue;
            }
        };

        match records.get(&record_id) {
            Some(rec) if in_namespace(rec, namespaces) => {}
            _ => continue,
        }

        // Compute Tanimoto similarity
        if let Some(header) = cache.get(&aura_id) {
            let score = sdr.tanimoto_sparse(&query_sdr, &header.sdr_indices);
            if score > 0.0 {
                results.push((record_id, score));
            }
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}

/// Collect N-gram fuzzy match results, filtered to namespace-scoped records.
#[instrument(skip_all, fields(top_k))]
pub fn collect_ngram(
    ngram_index: &NGramIndex,
    records: &HashMap<String, Record>,
    query: &str,
    top_k: usize,
    namespaces: &[&str],
) -> Vec<(String, f32)> {
    ngram_index
        .query(query, top_k * 4)
        .into_iter()
        .filter(|(_, rid)| {
            records.get(rid).map_or(false, |r| in_namespace(r, namespaces))
        })
        .take(top_k)
        .map(|(sim, rid)| (rid, sim))
        .collect()
}

/// Collect Tag Jaccard similarity results.
#[instrument(skip_all, fields(top_k))]
pub fn collect_tags(
    tag_index: &HashMap<String, HashSet<String>>,
    records: &HashMap<String, Record>,
    query: &str,
    top_k: usize,
    namespaces: &[&str],
) -> Vec<(String, f32)> {
    // Parse query words as potential tags
    let query_tags: HashSet<String> = query
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    if query_tags.is_empty() {
        return vec![];
    }

    // Collect candidates from tag index
    let mut candidates: HashMap<String, HashSet<String>> = HashMap::new();
    for qtag in &query_tags {
        if let Some(ids) = tag_index.get(qtag) {
            for id in ids {
                candidates
                    .entry(id.clone())
                    .or_default()
                    .insert(qtag.clone());
            }
        }
    }

    // Compute Jaccard for each candidate
    let mut results: Vec<(String, f32)> = candidates
        .into_iter()
        .filter_map(|(rid, matched_tags)| {
            let rec = records.get(&rid)?;
            if !in_namespace(rec, namespaces) {
                return None;
            }
            let rec_tags: HashSet<String> = rec.tags.iter().map(|t| t.to_lowercase()).collect();
            let union: HashSet<_> = query_tags.union(&rec_tags).collect();
            let intersection = matched_tags.len();
            if union.is_empty() {
                return None;
            }
            let jaccard = intersection as f32 / union.len() as f32;
            Some((rid, jaccard))
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}

// ── RRF Fusion ──

/// Reciprocal Rank Fusion — combines multiple ranked lists.
///
/// RRF score = Σ(1 / (k + rank_i)) for each list where record appears.
#[instrument(skip_all, fields(top_k))]
pub fn rrf_fuse(
    records: &HashMap<String, Record>,
    ranked_lists: &[Vec<(String, f32)>],
    min_strength: f32,
    top_k: usize,
    namespaces: &[&str],
) -> Vec<(f32, Record)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    let num_lists = ranked_lists.len();

    for list in ranked_lists {
        for (rank, (rid, _raw_score)) in list.iter().enumerate() {
            let rrf_score = 1.0 / (RRF_K as f32 + rank as f32 + 1.0);
            *scores.entry(rid.clone()).or_insert(0.0) += rrf_score;
        }
    }

    // Normalize
    let max_possible = num_lists as f32 / (RRF_K as f32 + 1.0);
    if max_possible > 0.0 {
        for score in scores.values_mut() {
            *score /= max_possible;
        }
    }

    // Filter and sort
    let mut results: Vec<(f32, Record)> = scores
        .into_iter()
        .filter_map(|(rid, score)| {
            let rec = records.get(&rid)?;
            if rec.strength >= min_strength && in_namespace(rec, namespaces) {
                Some((score, rec.clone()))
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}

// ── Graph Walk ──

/// Expand results via 2-hop graph walk with damping.
pub fn graph_walk(
    matched: &mut Vec<(f32, Record)>,
    records: &HashMap<String, Record>,
    min_strength: f32,
    namespaces: &[&str],
) {
    let mut matched_ids: HashSet<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();
    let mut expanded_count = 0;

    let mut frontier: Vec<(f32, String)> = matched
        .iter()
        .map(|(score, rec)| (*score, rec.id.clone()))
        .collect();

    for _hop in 0..GRAPH_WALK_MAX_HOPS {
        let mut next_frontier: Vec<(f32, String)> = Vec::new();

        for (parent_score, parent_id) in &frontier {
            if let Some(parent) = records.get(parent_id) {
                for (conn_id, conn_weight) in &parent.connections {
                    if matched_ids.contains(conn_id) {
                        continue;
                    }

                    let score = parent_score * conn_weight * GRAPH_WALK_DAMPING;
                    if score < GRAPH_WALK_MIN_SCORE {
                        continue;
                    }

                    next_frontier.push((score, conn_id.clone()));
                }
            }
        }

        // Deduplicate frontier (keep best score)
        let mut deduped: HashMap<String, f32> = HashMap::new();
        for (score, rid) in next_frontier {
            let entry = deduped.entry(rid).or_insert(0.0);
            if score > *entry {
                *entry = score;
            }
        }

        // Add to matched results
        let mut new_frontier = Vec::new();
        let mut sorted: Vec<_> = deduped.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rid, score) in sorted {
            if expanded_count >= GRAPH_WALK_MAX_EXPANDED {
                break;
            }
            if let Some(rec) = records.get(&rid) {
                if rec.strength >= min_strength && in_namespace(rec, namespaces) {
                    matched.push((score, rec.clone()));
                    matched_ids.insert(rid.clone());
                    new_frontier.push((score, rid));
                    expanded_count += 1;
                }
            }
        }

        frontier = new_frontier;
        if frontier.is_empty() {
            break;
        }
    }
}

/// Follow caused_by_id chains to discover causal context.
pub fn causal_walk(
    matched: &mut Vec<(f32, Record)>,
    records: &HashMap<String, Record>,
    min_strength: f32,
    namespaces: &[&str],
) {
    let mut matched_ids: HashSet<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();
    let mut additions = Vec::new();

    for (overlap, rec) in matched.iter() {
        let mut current = rec.clone();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(current.id.clone());

        for depth in 0..CAUSAL_MAX_DEPTH {
            let parent_id = match &current.caused_by_id {
                Some(id) => id.clone(),
                None => break,
            };

            if matched_ids.contains(&parent_id) || visited.contains(&parent_id) {
                break;
            }

            let parent = match records.get(&parent_id) {
                Some(p) if p.strength >= min_strength && in_namespace(p, namespaces) => p,
                _ => break,
            };

            visited.insert(parent_id.clone());
            let causal_score = overlap * 0.8 * 0.9f32.powi(depth as i32);
            additions.push((causal_score, parent.clone()));
            matched_ids.insert(parent_id);

            current = parent.clone();
        }
    }

    matched.extend(additions);
}

// ── Recency-weighted scoring ──

/// Apply trust-aware recency weighting and sort.
///
/// Uses `compute_effective_trust()` which factors in:
/// - Source authority (user > agent > autonomous)
/// - Recency boost (fresh records get +boost, decays over half_life)
/// - Base trust score from provenance
/// - Source type factor (recorded > retrieved > inferred > generated)
///
/// Final score = rrf_score × strength × effective_trust
pub fn apply_recency_scoring(
    matched: &mut Vec<(f32, Record)>,
    top_k: usize,
    trust_config: Option<&TrustConfig>,
) {
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let default_config = TrustConfig::default();
    let config = trust_config.unwrap_or(&default_config);

    for (score, rec) in matched.iter_mut() {
        let effective_trust = trust::compute_effective_trust(
            &rec.metadata, now_unix, config, &rec.source_type,
        );
        *score = *score * rec.strength * effective_trust;
    }

    matched.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    matched.truncate(top_k);
}

// ── Activation & Co-recall strengthening ──

/// Activate top records and strengthen co-recalled connections.
pub fn activate_and_strengthen(
    scored: &[(f32, Record)],
    records: &mut HashMap<String, Record>,
    session_tracker: &mut SessionTracker,
    session_id: Option<&str>,
) {
    let top_ids: Vec<String> = scored.iter().take(10).map(|(_, r)| r.id.clone()).collect();

    // Activate top-10
    for id in &top_ids {
        if let Some(rec) = records.get_mut(id) {
            rec.activate();
        }
    }

    // Strengthen co-recalled connections
    for i in 0..top_ids.len() {
        for j in (i + 1)..top_ids.len() {
            let id_a = &top_ids[i];
            let id_b = &top_ids[j];

            let current = records
                .get(id_a)
                .and_then(|r| r.connections.get(id_b).copied())
                .unwrap_or(0.0);

            let delta = 0.05 * (1.0 - current);
            let boosted = (current + delta).min(1.0);

            if let Some(rec_a) = records.get_mut(id_a) {
                rec_a.connections.insert(id_b.clone(), boosted);
            }
            if let Some(rec_b) = records.get_mut(id_b) {
                rec_b.connections.insert(id_a.clone(), boosted);
            }
        }
    }

    // Session tracking
    if let Some(sid) = session_id {
        session_tracker.track_activation(sid, &top_ids);
    }
}

// ── Preamble formatting ──

/// Token budget allocation per level (IDENTITY gets 25%).
const IDENTITY_BUDGET_RATIO: f32 = 0.25;
const TOKENS_PER_WORD: f32 = 1.3;

/// Format scored records into a token-budgeted preamble for LLM context.
pub fn format_preamble(
    scored: &[(f32, Record)],
    token_budget: usize,
    records: &HashMap<String, Record>,
) -> String {
    if scored.is_empty() {
        return String::new();
    }

    // Group by level
    let mut by_level: HashMap<Level, Vec<&(f32, Record)>> = HashMap::new();
    for item in scored {
        by_level.entry(item.1.level).or_default().push(item);
    }

    let mut output = String::from("=== COGNITIVE CONTEXT ===\n");

    let identity_budget = (token_budget as f32 * IDENTITY_BUDGET_RATIO).max(128.0) as usize;
    let remaining_budget = token_budget.saturating_sub(identity_budget);

    // Output order: IDENTITY → DOMAIN → DECISIONS → WORKING
    let level_order = [Level::Identity, Level::Domain, Level::Decisions, Level::Working];

    for level in &level_order {
        let budget = if *level == Level::Identity {
            identity_budget
        } else {
            remaining_budget / 3
        };

        if let Some(items) = by_level.get(level) {
            output.push_str(&format!("[{}]\n", level.name()));
            let mut level_tokens = 0;

            for (_score, rec) in items.iter() {
                let formatted = format_record(rec, records);
                let est_tokens = estimate_tokens(&formatted);

                if level_tokens + est_tokens > budget {
                    break;
                }

                output.push_str(&formatted);
                output.push('\n');
                level_tokens += est_tokens;
            }

            output.push('\n');
        }
    }

    output.push_str("=== END CONTEXT ===");
    output
}

fn format_record(rec: &Record, records: &HashMap<String, Record>) -> String {
    let tags_str = if rec.tags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", rec.tags.join(", "))
    };

    // Source type label for non-recorded data (epistemological provenance)
    let source_label = match rec.source_type.as_str() {
        "retrieved" => " [retrieved]",
        "inferred" => " [inferred]",
        "generated" => " [generated]",
        _ => "", // "recorded" is the default — no label needed
    };

    let mut base = match rec.content_type.as_str() {
        "code" => {
            let lang = rec.metadata.get("language").map(|s| s.as_str()).unwrap_or("");
            format!("  - [CODE]{}{}\n```{}\n{}\n```", source_label, tags_str, lang, rec.content)
        }
        "json" => {
            format!("  - [JSON]{}{}\n```json\n{}\n```", source_label, tags_str, rec.content)
        }
        _ => {
            format!("  - {}{}{}", rec.content, source_label, tags_str)
        }
    };

    // Append causal reasoning
    if let Some(ref caused_by) = rec.caused_by_id {
        if let Some(parent) = records.get(caused_by) {
            let preview: String = parent.content.chars().take(120).collect();
            base.push_str(&format!("\n    ^ because: {}", preview));
        }
    }

    base
}

fn estimate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    (words as f32 * TOKENS_PER_WORD) as usize
}

/// Full recall pipeline.
///
/// `embedding_ranked` is an optional 4th signal from pluggable embeddings.
/// When provided, it participates in RRF fusion alongside SDR, N-gram, and Tag Jaccard.
///
/// `trust_config` is used for recency boost + source authority scoring.
#[instrument(skip_all, fields(query, top_k, min_strength))]
pub fn recall_pipeline(
    query: &str,
    top_k: usize,
    min_strength: f32,
    expand_connections: bool,
    sdr: &SDRInterpreter,
    inverted_index: &InvertedIndex,
    storage: &AuraStorage,
    ngram_index: &NGramIndex,
    tag_index: &HashMap<String, HashSet<String>>,
    aura_index: &HashMap<String, String>,
    records: &HashMap<String, Record>,
    embedding_ranked: Option<Vec<(String, f32)>>,
    trust_config: Option<&TrustConfig>,
    namespaces: Option<&[&str]>,
) -> Vec<(f32, Record)> {
    let default_ns = [DEFAULT_NAMESPACE];
    let ns = namespaces.unwrap_or(&default_ns);

    // 1. Collect signals
    let sdr_ranked = collect_sdr(sdr, inverted_index, storage, aura_index, records, query, top_k, ns);
    let ngram_ranked = collect_ngram(ngram_index, records, query, top_k, ns);
    let tag_ranked = collect_tags(tag_index, records, query, top_k, ns);

    // 2. RRF Fuse
    let mut lists = Vec::new();
    if !sdr_ranked.is_empty() {
        lists.push(sdr_ranked);
    }
    if !ngram_ranked.is_empty() {
        lists.push(ngram_ranked);
    }
    if !tag_ranked.is_empty() {
        lists.push(tag_ranked);
    }
    // 4th signal: embedding similarity (optional)
    if let Some(emb) = embedding_ranked {
        if !emb.is_empty() {
            lists.push(emb);
        }
    }

    if lists.is_empty() {
        return vec![];
    }

    let mut matched = rrf_fuse(records, &lists, min_strength, top_k, ns);

    // 3. Graph expansion
    if expand_connections {
        graph_walk(&mut matched, records, min_strength, ns);
        causal_walk(&mut matched, records, min_strength, ns);
    }

    // 4. Trust-aware recency-weighted scoring
    apply_recency_scoring(&mut matched, top_k, trust_config);

    matched
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fuse_basic() {
        let mut records = HashMap::new();
        let r1 = Record::new("Apple pie recipe".into(), Level::Working);
        let r2 = Record::new("Banana smoothie".into(), Level::Working);
        let id1 = r1.id.clone();
        let id2 = r2.id.clone();
        records.insert(id1.clone(), r1);
        records.insert(id2.clone(), r2);

        let list1 = vec![(id1.clone(), 0.9), (id2.clone(), 0.5)];
        let list2 = vec![(id2.clone(), 0.8), (id1.clone(), 0.3)];

        let fused = rrf_fuse(&records, &[list1, list2], 0.0, 10, &["default"]);
        assert_eq!(fused.len(), 2);
        // Both appear in both lists, so scores should be non-zero
        assert!(fused[0].0 > 0.0);
    }

    #[test]
    fn test_format_preamble() {
        let mut records = HashMap::new();
        let r1 = Record::new("Teo loves Rust".into(), Level::Identity);
        let id1 = r1.id.clone();
        records.insert(id1.clone(), r1.clone());

        let scored = vec![(0.9, r1)];
        let preamble = format_preamble(&scored, 2048, &records);

        assert!(preamble.contains("COGNITIVE CONTEXT"));
        assert!(preamble.contains("[IDENTITY]"));
        assert!(preamble.contains("Teo loves Rust"));
    }
}
