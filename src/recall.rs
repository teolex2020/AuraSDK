//! RRF Fusion recall pipeline — the KEY intellectual property.
//!
//! Rewritten from aura-cognitive recall.py.
//!
//! Pipeline:
//! Query → [3 parallel ranked lists] → RRF Fusion → Graph Walk → Causal Walk → Rank → Format

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::instrument;

use crate::belief::{BeliefEngine, BeliefState};
use crate::causal::{CausalEngine, CausalState};
use crate::concept::{ConceptEngine, ConceptState};
use crate::graph::SessionTracker;
use crate::index::InvertedIndex;
use crate::levels::Level;
use crate::ngram::NGramIndex;
use crate::policy::{PolicyActionKind, PolicyEngine, PolicyState};
use crate::record::Record;
use crate::record::DEFAULT_NAMESPACE;
use crate::sdr::SDRInterpreter;
use crate::storage::AuraStorage;
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

#[derive(Debug, Clone, Default)]
pub struct SignalTrace {
    pub raw_score: f32,
    pub rank: usize,
    pub rrf_share: f32,
}

#[derive(Debug, Clone, Default)]
pub struct RecallScoreTrace {
    pub record_id: String,
    pub sdr: Option<SignalTrace>,
    pub ngram: Option<SignalTrace>,
    pub tags: Option<SignalTrace>,
    pub embedding: Option<SignalTrace>,
    pub rrf_score: f32,
    pub graph_score: f32,
    pub causal_score: f32,
    pub pre_trust_score: f32,
    pub trust_multiplier: f32,
    pub pre_rerank_score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct RecallTraceResult {
    pub scored: Vec<(f32, Record)>,
    pub traces: HashMap<String, RecallScoreTrace>,
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
            records
                .get(rid)
                .is_some_and(|r| in_namespace(r, namespaces))
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
    let query_tags: HashSet<String> = query.split_whitespace().map(|w| w.to_lowercase()).collect();

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

fn graph_walk_with_trace(
    matched: &mut Vec<(f32, Record)>,
    records: &HashMap<String, Record>,
    min_strength: f32,
    namespaces: &[&str],
    traces: &mut HashMap<String, RecallScoreTrace>,
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

        let mut deduped: HashMap<String, f32> = HashMap::new();
        for (score, rid) in next_frontier {
            let entry = deduped.entry(rid).or_insert(0.0);
            if score > *entry {
                *entry = score;
            }
        }

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
                    traces
                        .entry(rid.clone())
                        .or_insert_with(|| RecallScoreTrace {
                            record_id: rid.clone(),
                            ..RecallScoreTrace::default()
                        });
                    if let Some(trace) = traces.get_mut(&rid) {
                        trace.graph_score = score;
                    }
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

fn causal_walk_with_trace(
    matched: &mut Vec<(f32, Record)>,
    records: &HashMap<String, Record>,
    min_strength: f32,
    namespaces: &[&str],
    traces: &mut HashMap<String, RecallScoreTrace>,
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
            matched_ids.insert(parent_id.clone());
            traces
                .entry(parent_id.clone())
                .or_insert_with(|| RecallScoreTrace {
                    record_id: parent_id.clone(),
                    ..RecallScoreTrace::default()
                })
                .causal_score = causal_score;

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
        let effective_trust =
            trust::compute_effective_trust(&rec.metadata, now_unix, config, &rec.source_type);
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
    let level_order = [
        Level::Identity,
        Level::Domain,
        Level::Decisions,
        Level::Working,
    ];

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

    // Semantic role label (only shown for non-default types)
    let semantic_label = match rec.semantic_type.as_str() {
        "decision" => " {decision}",
        "preference" => " {preference}",
        "trend" => " {trend}",
        "serendipity" => " {serendipity}",
        "contradiction" => " {contradiction}",
        _ => "", // "fact" is the default — no label needed
    };

    let mut base = match rec.content_type.as_str() {
        "code" => {
            let lang = rec
                .metadata
                .get("language")
                .map(|s| s.as_str())
                .unwrap_or("");
            format!(
                "  - [CODE]{}{}{}\n```{}\n{}\n```",
                source_label, semantic_label, tags_str, lang, rec.content
            )
        }
        "json" => {
            format!(
                "  - [JSON]{}{}{}\n```json\n{}\n```",
                source_label, semantic_label, tags_str, rec.content
            )
        }
        _ => {
            format!(
                "  - {}{}{}{}",
                rec.content, source_label, semantic_label, tags_str
            )
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
    let sdr_ranked = collect_sdr(
        sdr,
        inverted_index,
        storage,
        aura_index,
        records,
        query,
        top_k,
        ns,
    );
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

pub fn recall_pipeline_with_trace(
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
) -> RecallTraceResult {
    let default_ns = [DEFAULT_NAMESPACE];
    let ns = namespaces.unwrap_or(&default_ns);

    let sdr_ranked = collect_sdr(
        sdr,
        inverted_index,
        storage,
        aura_index,
        records,
        query,
        top_k,
        ns,
    );
    let ngram_ranked = collect_ngram(ngram_index, records, query, top_k, ns);
    let tag_ranked = collect_tags(tag_index, records, query, top_k, ns);

    let mut named_lists: Vec<(&str, Vec<(String, f32)>)> = Vec::new();
    if !sdr_ranked.is_empty() {
        named_lists.push(("sdr", sdr_ranked));
    }
    if !ngram_ranked.is_empty() {
        named_lists.push(("ngram", ngram_ranked));
    }
    if !tag_ranked.is_empty() {
        named_lists.push(("tags", tag_ranked));
    }
    if let Some(emb) = embedding_ranked {
        if !emb.is_empty() {
            named_lists.push(("embedding", emb));
        }
    }

    if named_lists.is_empty() {
        return RecallTraceResult::default();
    }

    let num_lists = named_lists.len();
    let max_possible = num_lists as f32 / (RRF_K as f32 + 1.0);
    let mut traces: HashMap<String, RecallScoreTrace> = HashMap::new();
    for (name, list) in &named_lists {
        for (rank, (rid, raw_score)) in list.iter().enumerate() {
            let rrf_share = if max_possible > 0.0 {
                (1.0 / (RRF_K as f32 + rank as f32 + 1.0)) / max_possible
            } else {
                0.0
            };
            let trace = traces
                .entry(rid.clone())
                .or_insert_with(|| RecallScoreTrace {
                    record_id: rid.clone(),
                    ..RecallScoreTrace::default()
                });
            trace.rrf_score += rrf_share;
            let signal = SignalTrace {
                raw_score: *raw_score,
                rank,
                rrf_share,
            };
            match *name {
                "sdr" => trace.sdr = Some(signal),
                "ngram" => trace.ngram = Some(signal),
                "tags" => trace.tags = Some(signal),
                "embedding" => trace.embedding = Some(signal),
                _ => {}
            }
        }
    }

    let lists: Vec<Vec<(String, f32)>> = named_lists.into_iter().map(|(_, list)| list).collect();
    let mut matched = rrf_fuse(records, &lists, min_strength, top_k, ns);

    if expand_connections {
        graph_walk_with_trace(&mut matched, records, min_strength, ns, &mut traces);
        causal_walk_with_trace(&mut matched, records, min_strength, ns, &mut traces);
    }

    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let default_config = TrustConfig::default();
    let config = trust_config.unwrap_or(&default_config);
    for (score, rec) in matched.iter_mut() {
        let effective_trust =
            trust::compute_effective_trust(&rec.metadata, now_unix, config, &rec.source_type);
        let multiplier = rec.strength * effective_trust;
        let pre_trust_score = *score;
        *score = *score * multiplier;

        let trace = traces
            .entry(rec.id.clone())
            .or_insert_with(|| RecallScoreTrace {
                record_id: rec.id.clone(),
                ..RecallScoreTrace::default()
            });
        trace.pre_trust_score = pre_trust_score;
        trace.trust_multiplier = multiplier;
        trace.pre_rerank_score = *score;
    }

    matched.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    matched.truncate(top_k);

    RecallTraceResult {
        scored: matched,
        traces,
    }
}

// ── Belief Reranking (Phase 4 — Limited Influence Activation) ──
//
// Tri-state mode: Off (default), Shadow (observe-only), Limited (bounded rerank).
// Applied AFTER trust-aware recency scoring. Capped so baseline dominates.

/// Belief rerank operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefRerankMode {
    /// No belief influence on recall ranking. Default.
    Off = 0,
    /// Shadow mode: compute shadow scores for logging, do NOT alter ranking.
    Shadow = 1,
    /// Limited influence: apply bounded reranking (capped score delta + positional shift limit).
    Limited = 2,
}

impl BeliefRerankMode {
    /// Convert from u8 (for atomic storage). Invalid values → Off.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Shadow,
            2 => Self::Limited,
            _ => Self::Off,
        }
    }
}

/// Maximum belief rerank score effect: ±5% of original score.
const BELIEF_RERANK_CAP: f32 = 0.05;

/// Maximum positional shift allowed: ±2 positions in the ranking.
const BELIEF_RERANK_MAX_POS_SHIFT: usize = 2;

/// Minimum result count to apply limited reranking (avoid artificial movement).
const BELIEF_RERANK_MIN_RESULTS: usize = 4;

/// Maximum top_k for which limited reranking is applied.
const BELIEF_RERANK_MAX_TOP_K: usize = 20;

/// Phase 4 limited-influence multipliers.
const BELIEF_RERANK_RESOLVED: f32 = 1.05;
const BELIEF_RERANK_SINGLETON: f32 = 1.02;
const BELIEF_RERANK_UNRESOLVED: f32 = 0.97;

/// Report from limited reranking, capturing what changed.
#[derive(Debug, Clone)]
pub struct LimitedRerankReport {
    /// Whether limited reranking was actually applied (false if scope guards blocked).
    pub was_applied: bool,
    /// Reason reranking was skipped (empty if applied).
    pub skip_reason: String,
    /// Number of records whose position changed.
    pub records_moved: usize,
    /// Maximum upward positional shift observed.
    pub max_up_shift: usize,
    /// Maximum downward positional shift observed.
    pub max_down_shift: usize,
    /// Average belief multiplier across all records.
    pub avg_belief_multiplier: f32,
    /// Fraction of records that have belief membership.
    pub belief_coverage: f32,
    /// Top-k overlap: fraction of top-k records shared between baseline and reranked.
    pub top_k_overlap: f32,
    /// Latency of reranking in microseconds.
    pub rerank_latency_us: u64,
}

impl LimitedRerankReport {
    /// Create a "skipped" report.
    fn skipped(reason: &str) -> Self {
        Self {
            was_applied: false,
            skip_reason: reason.to_string(),
            records_moved: 0,
            max_up_shift: 0,
            max_down_shift: 0,
            avg_belief_multiplier: 1.0,
            belief_coverage: 0.0,
            top_k_overlap: 1.0,
            rerank_latency_us: 0,
        }
    }
}

/// Apply belief-aware reranking with Phase 4 guardrails.
///
/// Returns a report describing what happened. If scope guards prevent
/// reranking (too few results, no belief coverage, top_k too large),
/// the report indicates the skip reason and `matched` is unchanged.
///
/// Guardrails:
/// - Score delta capped at ±5% of original score
/// - Positional shift capped at ±2 positions
/// - Only applied when result count ≥ 4, top_k ≤ 20, belief_coverage > 0
pub fn apply_belief_rerank(
    matched: &mut Vec<(f32, Record)>,
    belief_engine: &BeliefEngine,
    top_k: usize,
) -> LimitedRerankReport {
    let start = std::time::Instant::now();
    let n = matched.len();

    // ── Scope guards ──

    if n < BELIEF_RERANK_MIN_RESULTS {
        return LimitedRerankReport::skipped("too few results");
    }

    if top_k > BELIEF_RERANK_MAX_TOP_K {
        return LimitedRerankReport::skipped("top_k exceeds limit");
    }

    // Check belief coverage before doing work
    let mut belief_count = 0usize;
    for (_, rec) in matched.iter() {
        if belief_engine.belief_for_record(&rec.id).is_some() {
            belief_count += 1;
        }
    }

    let belief_coverage = belief_count as f32 / n as f32;
    if belief_count == 0 {
        return LimitedRerankReport::skipped("no belief coverage");
    }

    // ── Phase 1: Score adjustment (capped) ──

    // Save baseline order for positional shift cap
    let baseline_ids: Vec<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();

    let mut multiplier_sum = 0.0f32;
    for (score, rec) in matched.iter_mut() {
        let multiplier = match belief_engine.belief_for_record(&rec.id) {
            Some(belief) => match belief.state {
                BeliefState::Resolved => BELIEF_RERANK_RESOLVED,
                BeliefState::Singleton => BELIEF_RERANK_SINGLETON,
                BeliefState::Unresolved => BELIEF_RERANK_UNRESOLVED,
                BeliefState::Empty => 1.0,
            },
            None => 1.0,
        };
        multiplier_sum += multiplier;

        let original = *score;
        let adjusted = original * multiplier;
        let max_delta = original * BELIEF_RERANK_CAP;
        *score = adjusted.clamp(original - max_delta, original + max_delta);
    }

    // ── Phase 2: Sort, then enforce positional shift cap ──

    matched.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Check if any record moved more than MAX_POS_SHIFT positions
    // If so, restore it closer to its original position by swapping
    let mut needs_fixup = true;
    let mut fixup_rounds = 0;
    while needs_fixup && fixup_rounds < n {
        needs_fixup = false;
        fixup_rounds += 1;
        for i in 0..matched.len() {
            let id = &matched[i].1.id;
            if let Some(orig_pos) = baseline_ids.iter().position(|x| x == id) {
                let shift = if i > orig_pos {
                    i - orig_pos
                } else {
                    orig_pos - i
                };
                if shift > BELIEF_RERANK_MAX_POS_SHIFT {
                    // Swap toward original position
                    let target = if i > orig_pos {
                        (i - 1).max(orig_pos)
                    } else {
                        (i + 1).min(orig_pos)
                    };
                    matched.swap(i, target);
                    needs_fixup = true;
                    break; // restart scan after swap
                }
            }
        }
    }

    // ── Phase 3: Compute report ──

    let final_ids: Vec<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();
    let mut records_moved = 0;
    let mut max_up: usize = 0;
    let mut max_down: usize = 0;

    for (orig_pos, id) in baseline_ids.iter().enumerate() {
        if let Some(new_pos) = final_ids.iter().position(|x| x == id) {
            if new_pos != orig_pos {
                records_moved += 1;
                if new_pos < orig_pos {
                    // promoted (moved up = lower index)
                    max_up = max_up.max(orig_pos - new_pos);
                } else {
                    max_down = max_down.max(new_pos - orig_pos);
                }
            }
        }
    }

    // Top-k overlap
    let effective_k = n.min(top_k);
    let baseline_top: HashSet<&str> = baseline_ids
        .iter()
        .take(effective_k)
        .map(|s| s.as_str())
        .collect();
    let final_top: HashSet<&str> = final_ids
        .iter()
        .take(effective_k)
        .map(|s| s.as_str())
        .collect();
    let overlap = if effective_k > 0 {
        baseline_top.intersection(&final_top).count() as f32 / effective_k as f32
    } else {
        1.0
    };

    let latency = start.elapsed().as_micros() as u64;

    LimitedRerankReport {
        was_applied: true,
        skip_reason: String::new(),
        records_moved,
        max_up_shift: max_up,
        max_down_shift: max_down,
        avg_belief_multiplier: if n > 0 {
            multiplier_sum / n as f32
        } else {
            1.0
        },
        belief_coverage,
        top_k_overlap: overlap,
        rerank_latency_us: latency,
    }
}

// ── Shadow Belief Scoring ──
//
// Phase 3 Candidate B: parallel shadow scoring based on belief state.
// Does NOT change actual recall ranking — produces comparison metrics only.

/// Shadow belief score for a single recalled record.
#[derive(Debug, Clone)]
pub struct ShadowBeliefScore {
    /// Record ID.
    pub record_id: String,
    /// Original recall score (from trust-aware pipeline).
    pub baseline_score: f32,
    /// Belief-adjusted shadow score (baseline × belief_multiplier).
    pub shadow_score: f32,
    /// Belief multiplier applied (1.0 if no belief membership).
    pub belief_multiplier: f32,
    /// Belief state of the record's belief (None if no belief membership).
    pub belief_state: Option<String>,
    /// Belief confidence (0.0 if no belief membership).
    pub belief_confidence: f32,
    /// Position in baseline ranking (0-based).
    pub baseline_rank: usize,
    /// Position in shadow ranking (0-based).
    pub shadow_rank: usize,
    /// Rank change: positive = promoted, negative = demoted.
    pub rank_delta: i32,
}

/// Comparison report: baseline vs shadow ranking.
#[derive(Debug, Clone)]
pub struct ShadowRecallReport {
    /// Per-record shadow scores.
    pub scores: Vec<ShadowBeliefScore>,
    /// Top-k overlap: fraction of top-k records shared between baseline and shadow.
    pub top_k_overlap: f32,
    /// Number of records promoted (moved up in shadow ranking).
    pub promoted_count: usize,
    /// Number of records demoted (moved down in shadow ranking).
    pub demoted_count: usize,
    /// Number of records with no rank change.
    pub unchanged_count: usize,
    /// Fraction of recalled records that have belief membership.
    pub belief_coverage: f32,
    /// Average belief multiplier across all records.
    pub avg_belief_multiplier: f32,
    /// Latency of shadow scoring in microseconds.
    pub shadow_latency_us: u64,
}

/// Belief state → score multiplier.
///
/// Resolved beliefs boost: the system is confident about the claim.
/// Singleton beliefs get a smaller boost: unchallenged but unverified.
/// Unresolved beliefs are penalized: competing hypotheses, uncertain.
/// No belief membership: neutral (1.0).
const RESOLVED_MULTIPLIER: f32 = 1.10;
const SINGLETON_MULTIPLIER: f32 = 1.05;
const UNRESOLVED_MULTIPLIER: f32 = 0.95;
const NO_BELIEF_MULTIPLIER: f32 = 1.00;

/// Compute shadow belief scores for a set of recall results.
///
/// `requested_top_k` is the caller's top-k so the overlap metric aligns
/// with the actual recall surface (capped to result count).
///
/// Returns the shadow report with per-record scores and aggregate metrics.
/// Does NOT modify the input — purely observational.
pub fn compute_shadow_belief_scores(
    baseline: &[(f32, Record)],
    belief_engine: &BeliefEngine,
    requested_top_k: usize,
) -> ShadowRecallReport {
    let start = std::time::Instant::now();

    let mut scores: Vec<ShadowBeliefScore> = Vec::with_capacity(baseline.len());
    let mut belief_member_count: usize = 0;
    let mut multiplier_sum: f32 = 0.0;

    // Phase 1: compute shadow scores
    for (baseline_rank, (base_score, rec)) in baseline.iter().enumerate() {
        let (multiplier, state_str, confidence) = match belief_engine.belief_for_record(&rec.id) {
            Some(belief) => {
                belief_member_count += 1;
                let m = match belief.state {
                    BeliefState::Resolved => RESOLVED_MULTIPLIER,
                    BeliefState::Singleton => SINGLETON_MULTIPLIER,
                    BeliefState::Unresolved => UNRESOLVED_MULTIPLIER,
                    BeliefState::Empty => NO_BELIEF_MULTIPLIER,
                };
                let state = match belief.state {
                    BeliefState::Resolved => "resolved",
                    BeliefState::Singleton => "singleton",
                    BeliefState::Unresolved => "unresolved",
                    BeliefState::Empty => "empty",
                };
                (m, Some(state.to_string()), belief.confidence)
            }
            None => (NO_BELIEF_MULTIPLIER, None, 0.0),
        };

        multiplier_sum += multiplier;

        scores.push(ShadowBeliefScore {
            record_id: rec.id.clone(),
            baseline_score: *base_score,
            shadow_score: base_score * multiplier,
            belief_multiplier: multiplier,
            belief_state: state_str,
            belief_confidence: confidence,
            baseline_rank,
            shadow_rank: 0, // computed in phase 2
            rank_delta: 0,  // computed in phase 2
        });
    }

    // Phase 2: compute shadow ranking (sort by shadow_score descending, stable)
    let mut shadow_order: Vec<usize> = (0..scores.len()).collect();
    shadow_order.sort_by(|&a, &b| {
        scores[b]
            .shadow_score
            .partial_cmp(&scores[a].shadow_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign shadow ranks
    for (shadow_rank, &original_idx) in shadow_order.iter().enumerate() {
        scores[original_idx].shadow_rank = shadow_rank;
        scores[original_idx].rank_delta =
            scores[original_idx].baseline_rank as i32 - shadow_rank as i32;
    }

    // Phase 3: compute aggregate metrics
    let n = scores.len();
    let top_k = n.min(requested_top_k);

    let baseline_top: HashSet<&str> = scores
        .iter()
        .filter(|s| s.baseline_rank < top_k)
        .map(|s| s.record_id.as_str())
        .collect();
    let shadow_top: HashSet<&str> = scores
        .iter()
        .filter(|s| s.shadow_rank < top_k)
        .map(|s| s.record_id.as_str())
        .collect();

    let overlap = if top_k > 0 {
        baseline_top.intersection(&shadow_top).count() as f32 / top_k as f32
    } else {
        1.0
    };

    let promoted = scores.iter().filter(|s| s.rank_delta > 0).count();
    let demoted = scores.iter().filter(|s| s.rank_delta < 0).count();
    let unchanged = scores.iter().filter(|s| s.rank_delta == 0).count();

    let belief_coverage = if n > 0 {
        belief_member_count as f32 / n as f32
    } else {
        0.0
    };
    let avg_multiplier = if n > 0 {
        multiplier_sum / n as f32
    } else {
        1.0
    };

    let latency = start.elapsed().as_micros() as u64;

    ShadowRecallReport {
        scores,
        top_k_overlap: overlap,
        promoted_count: promoted,
        demoted_count: demoted,
        unchanged_count: unchanged,
        belief_coverage,
        avg_belief_multiplier: avg_multiplier,
        shadow_latency_us: latency,
    }
}

// ── Concept Reranking (Phase 4 — Limited Influence, Concept-Weighted) ──
//
// Applied after belief reranking (if both enabled) and after trust-aware scoring.
// Uses concept membership as a signal: records that belong to a strong/stable
// concept cluster receive a small boost; those outside any concept are unaffected.
// Guardrails mirror belief reranking: ±4% score cap, ±2 positional shift.

/// Maximum concept rerank score effect: ±4% of original score.
const CONCEPT_RERANK_CAP: f32 = 0.04;

/// Maximum positional shift allowed: ±2 positions in the ranking.
const CONCEPT_RERANK_MAX_POS_SHIFT: usize = 2;

/// Minimum result count to apply limited concept reranking.
const CONCEPT_RERANK_MIN_RESULTS: usize = 4;

/// Maximum top_k for which concept reranking is applied.
const CONCEPT_RERANK_MAX_TOP_K: usize = 20;

/// Multiplier for records inside a Stable concept cluster.
const CONCEPT_RERANK_STABLE: f32 = 1.04;

/// Multiplier for records inside a strong Candidate concept (score ≥ 0.70).
const CONCEPT_RERANK_CANDIDATE: f32 = 1.02;

/// Report from limited concept reranking.
#[derive(Debug, Clone)]
pub struct LimitedConceptRerankReport {
    /// Whether concept reranking was actually applied (false if scope guards blocked).
    pub was_applied: bool,
    /// Reason reranking was skipped (empty if applied).
    pub skip_reason: String,
    /// Number of records whose position changed.
    pub records_moved: usize,
    /// Maximum upward positional shift observed.
    pub max_up_shift: usize,
    /// Maximum downward positional shift observed.
    pub max_down_shift: usize,
    /// Average concept multiplier across all records.
    pub avg_concept_multiplier: f32,
    /// Fraction of records that belong to at least one concept.
    pub concept_coverage: f32,
    /// Top-k overlap: fraction of top-k records shared between baseline and reranked.
    pub top_k_overlap: f32,
    /// Latency in microseconds.
    pub rerank_latency_us: u64,
}

impl LimitedConceptRerankReport {
    fn skipped(reason: &str) -> Self {
        Self {
            was_applied: false,
            skip_reason: reason.to_string(),
            records_moved: 0,
            max_up_shift: 0,
            max_down_shift: 0,
            avg_concept_multiplier: 1.0,
            concept_coverage: 0.0,
            top_k_overlap: 1.0,
            rerank_latency_us: 0,
        }
    }
}

/// Build an index: record_id → (best_multiplier, concept_state_label).
///
/// For each record, we find the best concept it belongs to (Stable beats Candidate).
fn build_concept_membership_index(concept_engine: &ConceptEngine) -> HashMap<String, f32> {
    let mut index: HashMap<String, f32> = HashMap::new();

    for concept in concept_engine.concepts.values() {
        let multiplier = match concept.state {
            ConceptState::Stable => CONCEPT_RERANK_STABLE,
            ConceptState::Candidate if concept.abstraction_score >= 0.70 => {
                CONCEPT_RERANK_CANDIDATE
            }
            _ => continue, // below threshold — skip
        };
        for rid in &concept.record_ids {
            let entry = index.entry(rid.clone()).or_insert(1.0f32);
            if multiplier > *entry {
                *entry = multiplier;
            }
        }
    }

    index
}

/// Apply concept-aware reranking with guardrails.
///
/// Mirrors `apply_belief_rerank`:
/// - Score delta capped at ±4% of original score
/// - Positional shift capped at ±2 positions
/// - Only applied when result count ≥ 4, top_k ≤ 20, concept_coverage > 0
pub fn apply_concept_rerank(
    matched: &mut Vec<(f32, Record)>,
    concept_engine: &ConceptEngine,
    top_k: usize,
) -> LimitedConceptRerankReport {
    let start = std::time::Instant::now();
    let n = matched.len();

    // ── Scope guards ──

    if n < CONCEPT_RERANK_MIN_RESULTS {
        return LimitedConceptRerankReport::skipped("too few results");
    }

    if top_k > CONCEPT_RERANK_MAX_TOP_K {
        return LimitedConceptRerankReport::skipped("top_k exceeds limit");
    }

    // Build membership index once
    let membership = build_concept_membership_index(concept_engine);

    // Check concept coverage
    let concept_count = matched
        .iter()
        .filter(|(_, r)| membership.contains_key(&r.id))
        .count();

    if concept_count == 0 {
        return LimitedConceptRerankReport::skipped("no concept coverage");
    }

    let concept_coverage = concept_count as f32 / n as f32;

    // ── Phase 1: Score adjustment (capped) ──

    let baseline_ids: Vec<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();

    let mut multiplier_sum = 0.0f32;
    for (score, rec) in matched.iter_mut() {
        let multiplier = *membership.get(&rec.id).unwrap_or(&1.0f32);
        multiplier_sum += multiplier;

        let original = *score;
        let adjusted = original * multiplier;
        let max_delta = original * CONCEPT_RERANK_CAP;
        *score = adjusted.clamp(original - max_delta, original + max_delta);
    }

    // ── Phase 2: Sort, then enforce positional shift cap ──

    matched.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut needs_fixup = true;
    let mut fixup_rounds = 0;
    while needs_fixup && fixup_rounds < n {
        needs_fixup = false;
        fixup_rounds += 1;
        for i in 0..matched.len() {
            let id = &matched[i].1.id;
            if let Some(orig_pos) = baseline_ids.iter().position(|x| x == id) {
                let shift = if i > orig_pos {
                    i - orig_pos
                } else {
                    orig_pos - i
                };
                if shift > CONCEPT_RERANK_MAX_POS_SHIFT {
                    let target = if i > orig_pos {
                        (i - 1).max(orig_pos)
                    } else {
                        (i + 1).min(orig_pos)
                    };
                    matched.swap(i, target);
                    needs_fixup = true;
                    break;
                }
            }
        }
    }

    // ── Phase 3: Compute report ──

    let final_ids: Vec<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();
    let mut records_moved = 0;
    let mut max_up: usize = 0;
    let mut max_down: usize = 0;

    for (orig_pos, id) in baseline_ids.iter().enumerate() {
        if let Some(new_pos) = final_ids.iter().position(|x| x == id) {
            if new_pos != orig_pos {
                records_moved += 1;
                if new_pos < orig_pos {
                    max_up = max_up.max(orig_pos - new_pos);
                } else {
                    max_down = max_down.max(new_pos - orig_pos);
                }
            }
        }
    }

    let effective_k = n.min(top_k);
    let baseline_top: HashSet<&str> = baseline_ids
        .iter()
        .take(effective_k)
        .map(|s| s.as_str())
        .collect();
    let final_top: HashSet<&str> = final_ids
        .iter()
        .take(effective_k)
        .map(|s| s.as_str())
        .collect();
    let overlap = if effective_k > 0 {
        baseline_top.intersection(&final_top).count() as f32 / effective_k as f32
    } else {
        1.0
    };

    let latency = start.elapsed().as_micros() as u64;

    LimitedConceptRerankReport {
        was_applied: true,
        skip_reason: String::new(),
        records_moved,
        max_up_shift: max_up,
        max_down_shift: max_down,
        avg_concept_multiplier: if n > 0 {
            multiplier_sum / n as f32
        } else {
            1.0
        },
        concept_coverage,
        top_k_overlap: overlap,
        rerank_latency_us: latency,
    }
}

// ── Causal Reranking (Phase 4c — Limited Influence, Causal-Pattern-Weighted) ──
//
// Applied after belief + concept reranking (if enabled), after trust-aware scoring.
// Uses causal pattern membership as a signal:
//   - Effect-side records in a Stable/strong-Candidate pattern → small boost (1.03)
//   - Cause-side records → smaller boost (1.01)
//   - No pattern membership → 1.0
// Guardrails: ±3% score cap, ±2 positional shift, same scope guards as belief/concept.

/// Maximum causal rerank score effect: ±3% of original score.
const CAUSAL_RERANK_CAP: f32 = 0.03;

/// Maximum positional shift allowed: ±2 positions in the ranking.
const CAUSAL_RERANK_MAX_POS_SHIFT: usize = 2;

/// Minimum result count to apply limited causal reranking.
const CAUSAL_RERANK_MIN_RESULTS: usize = 4;

/// Maximum top_k for which limited causal reranking is applied.
const CAUSAL_RERANK_MAX_TOP_K: usize = 20;

/// Multiplier for effect-side records in a Stable causal pattern.
const CAUSAL_RERANK_EFFECT_STABLE: f32 = 1.03;

/// Multiplier for effect-side records in a strong Candidate pattern (strength ≥ 0.65).
const CAUSAL_RERANK_EFFECT_CANDIDATE: f32 = 1.015;

/// Multiplier for cause-side records in a Stable causal pattern.
const CAUSAL_RERANK_CAUSE_STABLE: f32 = 1.01;

/// Report from limited causal reranking.
#[derive(Debug, Clone)]
pub struct LimitedCausalRerankReport {
    /// Whether causal reranking was actually applied (false if scope guards blocked).
    pub was_applied: bool,
    /// Reason reranking was skipped (empty if applied).
    pub skip_reason: String,
    /// Number of records whose position changed.
    pub records_moved: usize,
    /// Maximum upward positional shift observed.
    pub max_up_shift: usize,
    /// Maximum downward positional shift observed.
    pub max_down_shift: usize,
    /// Average causal multiplier across all records.
    pub avg_causal_multiplier: f32,
    /// Fraction of records that appear in at least one causal pattern.
    pub causal_coverage: f32,
    /// Top-k overlap: fraction of top-k records shared between baseline and reranked.
    pub top_k_overlap: f32,
    /// Latency in microseconds.
    pub rerank_latency_us: u64,
}

impl LimitedCausalRerankReport {
    fn skipped(reason: &str) -> Self {
        Self {
            was_applied: false,
            skip_reason: reason.to_string(),
            records_moved: 0,
            max_up_shift: 0,
            max_down_shift: 0,
            avg_causal_multiplier: 1.0,
            causal_coverage: 0.0,
            top_k_overlap: 1.0,
            rerank_latency_us: 0,
        }
    }
}

/// Build an index: record_id → best causal multiplier.
///
/// Effect-side membership in a strong pattern outweighs cause-side.
fn build_causal_membership_index(causal_engine: &CausalEngine) -> HashMap<String, f32> {
    let mut index: HashMap<String, f32> = HashMap::new();

    for pattern in causal_engine.patterns.values() {
        // Determine pattern quality
        let effect_multiplier = match pattern.state {
            CausalState::Stable => CAUSAL_RERANK_EFFECT_STABLE,
            CausalState::Candidate if pattern.causal_strength >= 0.65 => {
                CAUSAL_RERANK_EFFECT_CANDIDATE
            }
            _ => continue, // weak, rejected, or invalidated pattern — skip
        };
        let cause_multiplier = match pattern.state {
            CausalState::Stable => CAUSAL_RERANK_CAUSE_STABLE,
            _ => continue,
        };

        // Effect-side records get the effect multiplier
        for rid in &pattern.effect_record_ids {
            let entry = index.entry(rid.clone()).or_insert(1.0f32);
            if effect_multiplier > *entry {
                *entry = effect_multiplier;
            }
        }

        // Cause-side records get the cause multiplier (only if not already higher from effect)
        for rid in &pattern.cause_record_ids {
            let entry = index.entry(rid.clone()).or_insert(1.0f32);
            if cause_multiplier > *entry {
                *entry = cause_multiplier;
            }
        }
    }

    index
}

/// Apply causal-pattern-aware reranking with guardrails.
///
/// Mirrors `apply_belief_rerank` and `apply_concept_rerank`:
/// - Score delta capped at ±3% of original score
/// - Positional shift capped at ±2 positions
/// - Only applied when result count ≥ 4, top_k ≤ 20, causal_coverage > 0
pub fn apply_causal_rerank(
    matched: &mut Vec<(f32, Record)>,
    causal_engine: &CausalEngine,
    top_k: usize,
) -> LimitedCausalRerankReport {
    let start = std::time::Instant::now();
    let n = matched.len();

    // ── Scope guards ──

    if n < CAUSAL_RERANK_MIN_RESULTS {
        return LimitedCausalRerankReport::skipped("too few results");
    }

    if top_k > CAUSAL_RERANK_MAX_TOP_K {
        return LimitedCausalRerankReport::skipped("top_k exceeds limit");
    }

    let membership = build_causal_membership_index(causal_engine);

    let causal_count = matched
        .iter()
        .filter(|(_, r)| membership.contains_key(&r.id))
        .count();

    if causal_count == 0 {
        return LimitedCausalRerankReport::skipped("no causal coverage");
    }

    let causal_coverage = causal_count as f32 / n as f32;

    // ── Phase 1: Score adjustment (capped) ──

    let baseline_ids: Vec<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();

    let mut multiplier_sum = 0.0f32;
    for (score, rec) in matched.iter_mut() {
        let multiplier = *membership.get(&rec.id).unwrap_or(&1.0f32);
        multiplier_sum += multiplier;

        let original = *score;
        let adjusted = original * multiplier;
        let max_delta = original * CAUSAL_RERANK_CAP;
        *score = adjusted.clamp(original - max_delta, original + max_delta);
    }

    // ── Phase 2: Sort, then enforce positional shift cap ──

    matched.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut needs_fixup = true;
    let mut fixup_rounds = 0;
    while needs_fixup && fixup_rounds < n {
        needs_fixup = false;
        fixup_rounds += 1;
        for i in 0..matched.len() {
            let id = &matched[i].1.id;
            if let Some(orig_pos) = baseline_ids.iter().position(|x| x == id) {
                let shift = if i > orig_pos {
                    i - orig_pos
                } else {
                    orig_pos - i
                };
                if shift > CAUSAL_RERANK_MAX_POS_SHIFT {
                    let target = if i > orig_pos {
                        (i - 1).max(orig_pos)
                    } else {
                        (i + 1).min(orig_pos)
                    };
                    matched.swap(i, target);
                    needs_fixup = true;
                    break;
                }
            }
        }
    }

    // ── Phase 3: Compute report ──

    let final_ids: Vec<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();
    let mut records_moved = 0;
    let mut max_up: usize = 0;
    let mut max_down: usize = 0;

    for (orig_pos, id) in baseline_ids.iter().enumerate() {
        if let Some(new_pos) = final_ids.iter().position(|x| x == id) {
            if new_pos != orig_pos {
                records_moved += 1;
                if new_pos < orig_pos {
                    max_up = max_up.max(orig_pos - new_pos);
                } else {
                    max_down = max_down.max(new_pos - orig_pos);
                }
            }
        }
    }

    let effective_k = n.min(top_k);
    let baseline_top: HashSet<&str> = baseline_ids
        .iter()
        .take(effective_k)
        .map(|s| s.as_str())
        .collect();
    let final_top: HashSet<&str> = final_ids
        .iter()
        .take(effective_k)
        .map(|s| s.as_str())
        .collect();
    let overlap = if effective_k > 0 {
        baseline_top.intersection(&final_top).count() as f32 / effective_k as f32
    } else {
        1.0
    };

    let latency = start.elapsed().as_micros() as u64;

    LimitedCausalRerankReport {
        was_applied: true,
        skip_reason: String::new(),
        records_moved,
        max_up_shift: max_up,
        max_down_shift: max_down,
        avg_causal_multiplier: if n > 0 {
            multiplier_sum / n as f32
        } else {
            1.0
        },
        causal_coverage,
        top_k_overlap: overlap,
        rerank_latency_us: latency,
    }
}

// ── Policy Reranking (Phase 4d — Limited Influence, Policy-Hint-Weighted) ──
//
// Final shaping signal in the recall pipeline. Applied last (after belief,
// concept, causal). Uses policy hint action_kind + state as signal:
//   - Prefer/Recommend + Stable → small boost for supporting records (1.02)
//   - Prefer/Recommend + strong Candidate (strength ≥ 0.70) → smaller boost (1.01)
//   - Avoid + Stable → slight downrank (0.99)
//   - VerifyFirst/Warn → neutral (1.0)
// Guardrails: ±2% score cap, ±2 positional shift (tightest of all phases).

/// Maximum policy rerank score effect: ±2% of original score.
const POLICY_RERANK_CAP: f32 = 0.02;

/// Maximum positional shift: ±2 positions.
const POLICY_RERANK_MAX_POS_SHIFT: usize = 2;

/// Minimum result count to apply policy reranking.
const POLICY_RERANK_MIN_RESULTS: usize = 4;

/// Maximum top_k for which policy reranking is applied.
const POLICY_RERANK_MAX_TOP_K: usize = 20;

/// Multiplier for records supporting a Stable Prefer/Recommend hint.
const POLICY_RERANK_PREFER_STABLE: f32 = 1.02;

/// Multiplier for records supporting a strong Candidate Prefer/Recommend hint.
const POLICY_RERANK_PREFER_CANDIDATE: f32 = 1.01;

/// Multiplier for records supporting a Stable Avoid hint (slight downrank).
const POLICY_RERANK_AVOID_STABLE: f32 = 0.99;

/// Report from limited policy reranking.
#[derive(Debug, Clone)]
pub struct LimitedPolicyRerankReport {
    /// Whether policy reranking was actually applied.
    pub was_applied: bool,
    /// Reason reranking was skipped (empty if applied).
    pub skip_reason: String,
    /// Number of records whose position changed.
    pub records_moved: usize,
    /// Maximum upward positional shift observed.
    pub max_up_shift: usize,
    /// Maximum downward positional shift observed.
    pub max_down_shift: usize,
    /// Average policy multiplier across all records.
    pub avg_policy_multiplier: f32,
    /// Fraction of records covered by at least one policy hint.
    pub policy_coverage: f32,
    /// Top-k overlap: fraction of top-k records shared between baseline and reranked.
    pub top_k_overlap: f32,
    /// Latency in microseconds.
    pub rerank_latency_us: u64,
}

impl LimitedPolicyRerankReport {
    fn skipped(reason: &str) -> Self {
        Self {
            was_applied: false,
            skip_reason: reason.to_string(),
            records_moved: 0,
            max_up_shift: 0,
            max_down_shift: 0,
            avg_policy_multiplier: 1.0,
            policy_coverage: 0.0,
            top_k_overlap: 1.0,
            rerank_latency_us: 0,
        }
    }
}

/// Build an index: record_id → best policy multiplier.
fn build_policy_membership_index(policy_engine: &PolicyEngine) -> HashMap<String, f32> {
    let mut index: HashMap<String, f32> = HashMap::new();

    for hint in policy_engine.hints.values() {
        let multiplier = match (&hint.action_kind, &hint.state) {
            (PolicyActionKind::Prefer | PolicyActionKind::Recommend, PolicyState::Stable) => {
                POLICY_RERANK_PREFER_STABLE
            }
            (PolicyActionKind::Prefer | PolicyActionKind::Recommend, PolicyState::Candidate)
                if hint.policy_strength >= 0.70 =>
            {
                POLICY_RERANK_PREFER_CANDIDATE
            }
            (PolicyActionKind::Avoid, PolicyState::Stable) => POLICY_RERANK_AVOID_STABLE,
            _ => continue,
        };

        for rid in &hint.supporting_record_ids {
            let entry = index.entry(rid.clone()).or_insert(1.0f32);
            // For boosts, take the best (highest) multiplier.
            // For downranks (< 1.0), take the most aggressive (lowest).
            if multiplier >= 1.0 && multiplier > *entry {
                *entry = multiplier;
            } else if multiplier < 1.0 && multiplier < *entry {
                *entry = multiplier;
            }
        }
    }

    index
}

/// Apply policy-hint-aware reranking with guardrails.
///
/// Phase 4d — the final shaping signal:
/// - Score delta capped at ±2% of original score
/// - Positional shift capped at ±2 positions
/// - Only applied when result count ≥ 4, top_k ≤ 20, policy_coverage > 0
pub fn apply_policy_rerank(
    matched: &mut Vec<(f32, Record)>,
    policy_engine: &PolicyEngine,
    top_k: usize,
) -> LimitedPolicyRerankReport {
    let start = std::time::Instant::now();
    let n = matched.len();

    if n < POLICY_RERANK_MIN_RESULTS {
        return LimitedPolicyRerankReport::skipped("too few results");
    }
    if top_k > POLICY_RERANK_MAX_TOP_K {
        return LimitedPolicyRerankReport::skipped("top_k exceeds limit");
    }

    let membership = build_policy_membership_index(policy_engine);

    let policy_count = matched
        .iter()
        .filter(|(_, r)| membership.contains_key(&r.id))
        .count();

    if policy_count == 0 {
        return LimitedPolicyRerankReport::skipped("no policy coverage");
    }

    let policy_coverage = policy_count as f32 / n as f32;

    // ── Phase 1: Score adjustment (capped) ──

    let baseline_ids: Vec<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();
    let mut multiplier_sum = 0.0f32;

    for (score, rec) in matched.iter_mut() {
        let multiplier = *membership.get(&rec.id).unwrap_or(&1.0f32);
        multiplier_sum += multiplier;

        let original = *score;
        let adjusted = original * multiplier;
        let max_delta = original * POLICY_RERANK_CAP;
        *score = adjusted.clamp(original - max_delta, original + max_delta);
    }

    // ── Phase 2: Sort + positional shift cap ──

    matched.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut needs_fixup = true;
    let mut fixup_rounds = 0;
    while needs_fixup && fixup_rounds < n {
        needs_fixup = false;
        fixup_rounds += 1;
        for i in 0..matched.len() {
            let id = &matched[i].1.id;
            if let Some(orig_pos) = baseline_ids.iter().position(|x| x == id) {
                let shift = if i > orig_pos {
                    i - orig_pos
                } else {
                    orig_pos - i
                };
                if shift > POLICY_RERANK_MAX_POS_SHIFT {
                    let target = if i > orig_pos {
                        (i - 1).max(orig_pos)
                    } else {
                        (i + 1).min(orig_pos)
                    };
                    matched.swap(i, target);
                    needs_fixup = true;
                    break;
                }
            }
        }
    }

    // ── Phase 3: Report ──

    let final_ids: Vec<String> = matched.iter().map(|(_, r)| r.id.clone()).collect();
    let mut records_moved = 0;
    let mut max_up: usize = 0;
    let mut max_down: usize = 0;

    for (orig_pos, id) in baseline_ids.iter().enumerate() {
        if let Some(new_pos) = final_ids.iter().position(|x| x == id) {
            if new_pos != orig_pos {
                records_moved += 1;
                if new_pos < orig_pos {
                    max_up = max_up.max(orig_pos - new_pos);
                } else {
                    max_down = max_down.max(new_pos - orig_pos);
                }
            }
        }
    }

    let effective_k = n.min(top_k);
    let baseline_top: HashSet<&str> = baseline_ids
        .iter()
        .take(effective_k)
        .map(|s| s.as_str())
        .collect();
    let final_top: HashSet<&str> = final_ids
        .iter()
        .take(effective_k)
        .map(|s| s.as_str())
        .collect();
    let overlap = if effective_k > 0 {
        baseline_top.intersection(&final_top).count() as f32 / effective_k as f32
    } else {
        1.0
    };

    let latency = start.elapsed().as_micros() as u64;

    LimitedPolicyRerankReport {
        was_applied: true,
        skip_reason: String::new(),
        records_moved,
        max_up_shift: max_up,
        max_down_shift: max_down,
        avg_policy_multiplier: if n > 0 {
            multiplier_sum / n as f32
        } else {
            1.0
        },
        policy_coverage,
        top_k_overlap: overlap,
        rerank_latency_us: latency,
    }
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

    // ── Shadow Belief Scoring Tests ──

    use crate::belief::{Belief, Hypothesis};
    use crate::concept::{ConceptCandidate, ConceptEngine};

    /// Helper: build a BeliefEngine with specific beliefs and record→hypothesis mappings.
    fn make_belief_engine_with_records(
        entries: &[(&str, BeliefState, f32)], // (record_id, state, confidence)
    ) -> BeliefEngine {
        let mut engine = BeliefEngine::default();
        for (record_id, state, confidence) in entries {
            let mut belief = Belief::new(format!("claim_{}", record_id));
            belief.state = state.clone();
            belief.confidence = *confidence;
            let bid = belief.id.clone();

            let hyp = Hypothesis {
                id: Record::generate_id(),
                belief_id: bid.clone(),
                prototype_record_ids: vec![record_id.to_string()],
                score: 0.8,
                confidence: *confidence,
                support_mass: 1.0,
                conflict_mass: 0.0,
                recency: 1.0,
                consistency: 1.0,
            };

            let hid = hyp.id.clone();
            engine.hypotheses.insert(hid.clone(), hyp);
            engine.beliefs.insert(bid, belief);
            engine.record_index.insert(record_id.to_string(), hid);
        }
        engine
    }

    fn make_concept_engine_with_records(
        entries: &[(&str, ConceptState, f32)], // (record_id, state, abstraction_score)
    ) -> ConceptEngine {
        let mut engine = ConceptEngine::new();
        for (record_id, state, abstraction_score) in entries {
            let concept = ConceptCandidate {
                id: Record::generate_id(),
                key: format!("concept_{}", record_id),
                namespace: "default".to_string(),
                semantic_type: "fact".to_string(),
                belief_ids: vec![format!("belief_{}", record_id)],
                record_ids: vec![record_id.to_string()],
                core_terms: vec!["test".to_string()],
                shell_terms: vec![],
                tags: vec!["test".to_string()],
                support_mass: 1.0,
                confidence: 0.9,
                stability: 3.0,
                cohesion: 1.0,
                abstraction_score: *abstraction_score,
                state: state.clone(),
                last_updated: 1.0,
            };
            engine
                .key_index
                .insert(concept.key.clone(), concept.id.clone());
            engine.concepts.insert(concept.id.clone(), concept);
        }
        engine
    }

    #[test]
    fn test_shadow_empty_baseline() {
        let engine = BeliefEngine::default();
        let report = compute_shadow_belief_scores(&[], &engine, 10);
        assert!(report.scores.is_empty());
        assert_eq!(report.top_k_overlap, 1.0);
        assert_eq!(report.belief_coverage, 0.0);
    }

    #[test]
    fn test_shadow_no_belief_membership() {
        let r1 = Record::new("test".into(), Level::Working);
        let r2 = Record::new("test2".into(), Level::Working);
        let baseline = vec![(0.9, r1), (0.5, r2)];
        let engine = BeliefEngine::default();

        let report = compute_shadow_belief_scores(&baseline, &engine, 10);
        assert_eq!(report.scores.len(), 2);
        assert_eq!(report.belief_coverage, 0.0);
        assert_eq!(report.avg_belief_multiplier, 1.0);
        // No rank changes — all multipliers are 1.0
        assert_eq!(report.unchanged_count, 2);
        assert_eq!(report.promoted_count, 0);
        assert_eq!(report.demoted_count, 0);
    }

    #[test]
    fn test_shadow_resolved_promotes() {
        // Two records: r1 at 0.80 (resolved belief), r2 at 0.82 (no belief)
        // After shadow: r1 = 0.80×1.10 = 0.88, r2 = 0.82×1.00 = 0.82
        // Baseline order: r2, r1. Shadow order: r1, r2.
        let mut r1 = Record::new("resolved record".into(), Level::Working);
        r1.id = "r1".to_string();
        let mut r2 = Record::new("no belief record".into(), Level::Working);
        r2.id = "r2".to_string();

        let baseline = vec![(0.82, r2), (0.80, r1)];
        let engine = make_belief_engine_with_records(&[("r1", BeliefState::Resolved, 0.9)]);

        let report = compute_shadow_belief_scores(&baseline, &engine, 10);
        // r1 should be promoted from rank 1 to rank 0
        let r1_score = report.scores.iter().find(|s| s.record_id == "r1").unwrap();
        assert_eq!(r1_score.baseline_rank, 1);
        assert_eq!(r1_score.shadow_rank, 0);
        assert_eq!(r1_score.rank_delta, 1); // promoted
        assert!((r1_score.shadow_score - 0.88).abs() < 0.001);
        assert_eq!(report.promoted_count, 1);
        assert_eq!(report.demoted_count, 1);
    }

    #[test]
    fn test_shadow_unresolved_demotes() {
        // Two records: r1 at 0.80 (unresolved), r2 at 0.78 (no belief)
        // After shadow: r1 = 0.80×0.95 = 0.76, r2 = 0.78×1.00 = 0.78
        // Baseline order: r1, r2. Shadow order: r2, r1.
        let mut r1 = Record::new("unresolved record".into(), Level::Working);
        r1.id = "r1".to_string();
        let mut r2 = Record::new("no belief record".into(), Level::Working);
        r2.id = "r2".to_string();

        let baseline = vec![(0.80, r1), (0.78, r2)];
        let engine = make_belief_engine_with_records(&[("r1", BeliefState::Unresolved, 0.5)]);

        let report = compute_shadow_belief_scores(&baseline, &engine, 10);
        let r1_score = report.scores.iter().find(|s| s.record_id == "r1").unwrap();
        assert_eq!(r1_score.baseline_rank, 0);
        assert_eq!(r1_score.shadow_rank, 1);
        assert_eq!(r1_score.rank_delta, -1); // demoted
        assert!((r1_score.shadow_score - 0.76).abs() < 0.001);
    }

    #[test]
    fn test_shadow_singleton_small_boost() {
        let mut r1 = Record::new("singleton".into(), Level::Working);
        r1.id = "r1".to_string();
        let baseline = vec![(0.50, r1)];
        let engine = make_belief_engine_with_records(&[("r1", BeliefState::Singleton, 0.7)]);

        let report = compute_shadow_belief_scores(&baseline, &engine, 10);
        let s = &report.scores[0];
        assert!((s.belief_multiplier - SINGLETON_MULTIPLIER).abs() < 0.001);
        assert!((s.shadow_score - 0.525).abs() < 0.001);
        assert_eq!(s.belief_state.as_deref(), Some("singleton"));
    }

    #[test]
    fn test_shadow_belief_coverage() {
        let mut r1 = Record::new("a".into(), Level::Working);
        r1.id = "r1".to_string();
        let mut r2 = Record::new("b".into(), Level::Working);
        r2.id = "r2".to_string();
        let mut r3 = Record::new("c".into(), Level::Working);
        r3.id = "r3".to_string();

        let baseline = vec![(0.9, r1), (0.8, r2), (0.7, r3)];
        let engine = make_belief_engine_with_records(&[
            ("r1", BeliefState::Resolved, 0.9),
            ("r3", BeliefState::Singleton, 0.6),
        ]);

        let report = compute_shadow_belief_scores(&baseline, &engine, 10);
        // 2 out of 3 have belief membership
        assert!((report.belief_coverage - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_shadow_top_k_overlap() {
        // With close scores where belief changes ranking, top-k overlap < 1.0
        let mut records = Vec::new();
        for i in 0..5 {
            let mut r = Record::new(format!("rec_{}", i), Level::Working);
            r.id = format!("r{}", i);
            records.push(r);
        }

        // Baseline: r0(0.90) > r1(0.89) > r2(0.88) > r3(0.87) > r4(0.86)
        let baseline: Vec<(f32, Record)> = records
            .into_iter()
            .enumerate()
            .map(|(i, r)| (0.90 - i as f32 * 0.01, r))
            .collect();

        // r4 is resolved → 0.86×1.10=0.946, jumps to rank 0
        let engine = make_belief_engine_with_records(&[("r4", BeliefState::Resolved, 0.95)]);
        let report = compute_shadow_belief_scores(&baseline, &engine, 5);

        // r4 was rank 4, now should be rank 0
        let r4 = report.scores.iter().find(|s| s.record_id == "r4").unwrap();
        assert_eq!(r4.shadow_rank, 0);
        assert!(r4.rank_delta > 0);
        // top-k overlap should be 1.0 since all 5 are in top-5 regardless
        assert!((report.top_k_overlap - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_shadow_top_k_overlap_respects_requested_k() {
        // 5 records, r4 jumps from rank 4 to rank 0 via resolved belief.
        // With requested_top_k=3: baseline top-3 = {r0,r1,r2}, shadow top-3 = {r4,r0,r1}
        // Overlap = 2/3 (r0, r1 shared; r2 displaced by r4).
        let mut records = Vec::new();
        for i in 0..5 {
            let mut r = Record::new(format!("rec_{}", i), Level::Working);
            r.id = format!("r{}", i);
            records.push(r);
        }
        let baseline: Vec<(f32, Record)> = records
            .into_iter()
            .enumerate()
            .map(|(i, r)| (0.90 - i as f32 * 0.01, r))
            .collect();
        let engine = make_belief_engine_with_records(&[("r4", BeliefState::Resolved, 0.95)]);

        let report = compute_shadow_belief_scores(&baseline, &engine, 3);
        // r4 jumps into shadow top-3, displacing r2
        assert!(
            (report.top_k_overlap - 2.0 / 3.0).abs() < 0.01,
            "expected overlap ~0.67, got {}",
            report.top_k_overlap
        );
    }

    #[test]
    fn test_shadow_preserves_baseline_order() {
        // Shadow scoring must not mutate the input
        let mut r1 = Record::new("a".into(), Level::Working);
        r1.id = "r1".to_string();
        let mut r2 = Record::new("b".into(), Level::Working);
        r2.id = "r2".to_string();

        let baseline = vec![(0.9, r1.clone()), (0.5, r2.clone())];
        let engine = make_belief_engine_with_records(&[("r2", BeliefState::Resolved, 0.9)]);

        let report = compute_shadow_belief_scores(&baseline, &engine, 10);

        // baseline_rank should reflect original order
        assert_eq!(report.scores[0].record_id, "r1");
        assert_eq!(report.scores[0].baseline_rank, 0);
        assert_eq!(report.scores[1].record_id, "r2");
        assert_eq!(report.scores[1].baseline_rank, 1);

        // Original vec unchanged (we took &, so can't mutate)
        assert_eq!(baseline[0].1.id, "r1");
        assert_eq!(baseline[1].1.id, "r2");
    }

    // ── Belief Reranking Tests (Phase 4) ──

    #[test]
    fn test_rerank_no_beliefs_skipped() {
        // No belief coverage → scope guard skips reranking
        let mut matched: Vec<(f32, Record)> = (0..5)
            .map(|i| {
                let mut r = Record::new(format!("rec_{}", i), Level::Working);
                r.id = format!("r{}", i);
                (0.90 - i as f32 * 0.02, r)
            })
            .collect();

        let engine = BeliefEngine::default();
        let report = apply_belief_rerank(&mut matched, &engine, 10);

        assert!(!report.was_applied);
        assert_eq!(report.skip_reason, "no belief coverage");
        // Scores unchanged
        assert!((matched[0].0 - 0.90).abs() < 0.0001);
    }

    #[test]
    fn test_rerank_too_few_results_skipped() {
        // Only 2 results → below min threshold (4)
        let mut r1 = Record::new("a".into(), Level::Working);
        r1.id = "r1".to_string();
        let mut r2 = Record::new("b".into(), Level::Working);
        r2.id = "r2".to_string();

        let mut matched = vec![(0.90, r1), (0.80, r2)];
        let engine = make_belief_engine_with_records(&[("r1", BeliefState::Resolved, 0.9)]);
        let report = apply_belief_rerank(&mut matched, &engine, 10);

        assert!(!report.was_applied);
        assert_eq!(report.skip_reason, "too few results");
    }

    #[test]
    fn test_rerank_top_k_too_large_skipped() {
        let mut matched: Vec<(f32, Record)> = (0..5)
            .map(|i| {
                let mut r = Record::new(format!("rec_{}", i), Level::Working);
                r.id = format!("r{}", i);
                (0.90 - i as f32 * 0.02, r)
            })
            .collect();

        let engine = make_belief_engine_with_records(&[("r0", BeliefState::Resolved, 0.9)]);
        let report = apply_belief_rerank(&mut matched, &engine, 50);

        assert!(!report.was_applied);
        assert_eq!(report.skip_reason, "top_k exceeds limit");
    }

    #[test]
    fn test_rerank_resolved_boosts_within_cap() {
        // 5 records, r0 has resolved belief → boosted by 1.05
        let mut matched: Vec<(f32, Record)> = (0..5)
            .map(|i| {
                let mut r = Record::new(format!("rec_{}", i), Level::Working);
                r.id = format!("r{}", i);
                (0.50 - i as f32 * 0.01, r)
            })
            .collect();
        let engine = make_belief_engine_with_records(&[("r0", BeliefState::Resolved, 0.9)]);

        let report = apply_belief_rerank(&mut matched, &engine, 10);

        assert!(report.was_applied);
        // r0: 0.50 * 1.05 = 0.525, within 5% cap
        assert!(
            (matched[0].0 - 0.525).abs() < 0.001,
            "expected 0.525, got {}",
            matched[0].0
        );
    }

    #[test]
    fn test_rerank_unresolved_penalizes_within_cap() {
        let mut matched: Vec<(f32, Record)> = (0..5)
            .map(|i| {
                let mut r = Record::new(format!("rec_{}", i), Level::Working);
                r.id = format!("r{}", i);
                (0.50 - i as f32 * 0.01, r)
            })
            .collect();
        let engine = make_belief_engine_with_records(&[("r0", BeliefState::Unresolved, 0.5)]);

        let report = apply_belief_rerank(&mut matched, &engine, 10);

        assert!(report.was_applied);
        // r0: 0.50 * 0.97 = 0.485, within 5% cap
        let r0 = matched.iter().find(|(_, r)| r.id == "r0").unwrap();
        assert!((r0.0 - 0.485).abs() < 0.001, "expected 0.485, got {}", r0.0);
    }

    #[test]
    fn test_rerank_can_swap_close_scores() {
        // r0=0.500 (no belief), r1=0.497 (resolved → 0.497*1.05=0.5219)
        // Plus filler records to meet min count
        let mut r0 = Record::new("a".into(), Level::Working);
        r0.id = "r0".to_string();
        let mut r1 = Record::new("b".into(), Level::Working);
        r1.id = "r1".to_string();
        let mut r2 = Record::new("c".into(), Level::Working);
        r2.id = "r2".to_string();
        let mut r3 = Record::new("d".into(), Level::Working);
        r3.id = "r3".to_string();

        let mut matched = vec![(0.500, r0), (0.497, r1), (0.30, r2), (0.20, r3)];
        let engine = make_belief_engine_with_records(&[("r1", BeliefState::Resolved, 0.9)]);

        let report = apply_belief_rerank(&mut matched, &engine, 10);

        assert!(report.was_applied);
        assert_eq!(matched[0].1.id, "r1", "resolved record should be promoted");
        assert_eq!(matched[1].1.id, "r0");
        assert!(report.records_moved >= 2);
    }

    #[test]
    fn test_rerank_cannot_swap_distant_scores() {
        // r0=0.90 (no belief), r3=0.50 (resolved → 0.50*1.05=0.525)
        // 5% cap is too small to bridge 0.40 gap
        let mut r0 = Record::new("a".into(), Level::Working);
        r0.id = "r0".to_string();
        let mut r1 = Record::new("b".into(), Level::Working);
        r1.id = "r1".to_string();
        let mut r2 = Record::new("c".into(), Level::Working);
        r2.id = "r2".to_string();
        let mut r3 = Record::new("d".into(), Level::Working);
        r3.id = "r3".to_string();

        let mut matched = vec![(0.90, r0), (0.80, r1), (0.60, r2), (0.50, r3)];
        let engine = make_belief_engine_with_records(&[("r3", BeliefState::Resolved, 0.9)]);

        let report = apply_belief_rerank(&mut matched, &engine, 10);

        assert!(report.was_applied);
        assert_eq!(matched[0].1.id, "r0", "distant scores should not swap");
    }

    #[test]
    fn test_rerank_effect_bounded_by_cap() {
        // Verify the actual delta never exceeds BELIEF_RERANK_CAP (5%)
        let mut matched: Vec<(f32, Record)> = (0..5)
            .map(|i| {
                let mut r = Record::new(format!("rec_{}", i), Level::Working);
                r.id = format!("r{}", i);
                (0.80 - i as f32 * 0.05, r)
            })
            .collect();
        let engine = make_belief_engine_with_records(&[
            ("r0", BeliefState::Resolved, 0.95),
            ("r1", BeliefState::Unresolved, 0.5),
        ]);

        let original_scores: Vec<f32> = matched.iter().map(|(s, _)| *s).collect();
        let report = apply_belief_rerank(&mut matched, &engine, 10);

        assert!(report.was_applied);
        for (score, rec) in &matched {
            if let Some(orig) = original_scores
                .iter()
                .zip(["r0", "r1", "r2", "r3", "r4"].iter())
                .find(|(_, id)| **id == rec.id)
                .map(|(s, _)| *s)
            {
                let delta = (*score - orig).abs();
                let max_allowed = orig * BELIEF_RERANK_CAP;
                assert!(
                    delta <= max_allowed + 0.0001,
                    "record {} delta {} exceeds cap {}",
                    rec.id,
                    delta,
                    max_allowed
                );
            }
        }
    }

    #[test]
    fn test_rerank_positional_shift_bounded() {
        // 8 records, r7 (last) has resolved belief.
        // Even with boost, it should not move more than 2 positions up.
        let mut matched: Vec<(f32, Record)> = (0..8)
            .map(|i| {
                let mut r = Record::new(format!("rec_{}", i), Level::Working);
                r.id = format!("r{}", i);
                (0.80 - i as f32 * 0.001, r) // very close scores
            })
            .collect();

        let engine = make_belief_engine_with_records(&[("r7", BeliefState::Resolved, 0.95)]);
        let report = apply_belief_rerank(&mut matched, &engine, 10);

        assert!(report.was_applied);
        assert!(
            report.max_up_shift <= BELIEF_RERANK_MAX_POS_SHIFT,
            "max up shift {} exceeds limit {}",
            report.max_up_shift,
            BELIEF_RERANK_MAX_POS_SHIFT
        );
        assert!(
            report.max_down_shift <= BELIEF_RERANK_MAX_POS_SHIFT,
            "max down shift {} exceeds limit {}",
            report.max_down_shift,
            BELIEF_RERANK_MAX_POS_SHIFT
        );
    }

    #[test]
    fn test_rerank_report_metrics() {
        let mut matched: Vec<(f32, Record)> = (0..5)
            .map(|i| {
                let mut r = Record::new(format!("rec_{}", i), Level::Working);
                r.id = format!("r{}", i);
                (0.80 - i as f32 * 0.002, r) // close scores
            })
            .collect();

        let engine = make_belief_engine_with_records(&[
            ("r0", BeliefState::Resolved, 0.9),
            ("r3", BeliefState::Unresolved, 0.5),
        ]);

        let report = apply_belief_rerank(&mut matched, &engine, 5);

        assert!(report.was_applied);
        assert!((report.belief_coverage - 0.4).abs() < 0.01); // 2/5
        assert!(report.avg_belief_multiplier > 0.99);
        assert!(report.top_k_overlap >= 0.0 && report.top_k_overlap <= 1.0);
        assert!(report.rerank_latency_us < 10_000); // should be fast
    }

    #[test]
    fn test_concept_rerank_no_coverage_skipped() {
        let mut matched: Vec<(f32, Record)> = (0..5)
            .map(|i| {
                let mut r = Record::new(format!("rec_{}", i), Level::Working);
                r.id = format!("r{}", i);
                (0.90 - i as f32 * 0.02, r)
            })
            .collect();

        let engine = ConceptEngine::new();
        let report = apply_concept_rerank(&mut matched, &engine, 10);

        assert!(!report.was_applied);
        assert_eq!(report.skip_reason, "no concept coverage");
        assert!((matched[0].0 - 0.90).abs() < 0.0001);
    }

    #[test]
    fn test_concept_rerank_limited_is_active_and_bounded() {
        let mut r0 = Record::new("a".into(), Level::Working);
        r0.id = "r0".to_string();
        let mut r1 = Record::new("b".into(), Level::Working);
        r1.id = "r1".to_string();
        let mut r2 = Record::new("c".into(), Level::Working);
        r2.id = "r2".to_string();
        let mut r3 = Record::new("d".into(), Level::Working);
        r3.id = "r3".to_string();

        let mut matched = vec![(0.500, r0), (0.497, r1), (0.30, r2), (0.20, r3)];
        let engine = make_concept_engine_with_records(&[("r1", ConceptState::Stable, 0.90)]);

        let report = apply_concept_rerank(&mut matched, &engine, 10);

        assert!(report.was_applied);
        assert_eq!(
            matched[0].1.id, "r1",
            "stable concept member should be promoted"
        );
        assert_eq!(matched[1].1.id, "r0");

        let r1_score = matched.iter().find(|(_, r)| r.id == "r1").unwrap().0;
        assert!(
            (r1_score - 0.51688).abs() < 0.001,
            "expected bounded concept rerank score, got {}",
            r1_score
        );
        assert!(report.records_moved >= 2);
        assert!((report.concept_coverage - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_rerank_mode_enum() {
        assert_eq!(BeliefRerankMode::from_u8(0), BeliefRerankMode::Off);
        assert_eq!(BeliefRerankMode::from_u8(1), BeliefRerankMode::Shadow);
        assert_eq!(BeliefRerankMode::from_u8(2), BeliefRerankMode::Limited);
        assert_eq!(BeliefRerankMode::from_u8(255), BeliefRerankMode::Off);
    }
}
