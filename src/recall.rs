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
use crate::graph::SessionTracker;
use crate::index::InvertedIndex;
use crate::levels::Level;
use crate::ngram::NGramIndex;
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
    fn test_rerank_mode_enum() {
        assert_eq!(BeliefRerankMode::from_u8(0), BeliefRerankMode::Off);
        assert_eq!(BeliefRerankMode::from_u8(1), BeliefRerankMode::Shadow);
        assert_eq!(BeliefRerankMode::from_u8(2), BeliefRerankMode::Limited);
        assert_eq!(BeliefRerankMode::from_u8(255), BeliefRerankMode::Off);
    }
}
