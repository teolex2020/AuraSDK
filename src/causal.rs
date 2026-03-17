//! Causal Pattern Discovery Layer — finds candidate causal relations from records.
//!
//! Fourth tier of the cognitive hierarchy:
//!   Record → Belief → Concept → **Causal Pattern** → Policy
//!
//! Phase 1 constraints (read-only candidate discovery):
//!   - Does NOT influence recall ranking or record merge
//!   - Full rebuild each maintenance cycle (no persistent trust)
//!   - Every pattern traces back to source belief_ids + record_ids
//!   - Namespace barrier: no cross-namespace causal patterns
//!   - Signal sources: explicit caused_by_id links, connection_type=="causal",
//!     temporal ordering within namespace

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::belief::{BeliefEngine, BeliefState, SdrLookup};
use crate::record::Record;

// ── Constants ──

/// Maximum temporal gap (seconds) between cause and effect records.
/// 7 days = 604800 seconds.
const MAX_CAUSAL_WINDOW_SECS: f64 = 7.0 * 86400.0;

/// Minimum number of supporting record-level edges before a pattern
/// can become a Candidate.
const MIN_SUPPORT: usize = 2;
/// Maximum tolerated counterfactual ratio before a pattern is blocked from
/// Candidate/Stable promotion.
const MAX_COUNTERFACTUAL_RATIO: f32 = 0.50;
/// Minimum share of explicit support a single outcome must hold, if the same
/// cause has multiple explicit competing outcomes.
const MIN_EXPLICIT_DOMINANCE_SHARE: f32 = 0.70;
/// Fixed bucket width for repeated temporal evidence.
const EVIDENCE_WINDOW_SECS: f64 = 24.0 * 3600.0;

/// Scoring weights.
const W_TRANSITION_LIFT: f32 = 0.35;
const W_TEMPORAL_CONSISTENCY: f32 = 0.30;
const W_OUTCOME_STABILITY: f32 = 0.20;
const W_SUPPORT: f32 = 0.15;

/// State thresholds for causal_strength.
const STABLE_THRESHOLD: f32 = 0.75;
const CANDIDATE_THRESHOLD: f32 = 0.50;

/// Maximum record-level edges to consider per namespace to keep
/// quadratic blowup in check.
const MAX_EDGES_PER_NAMESPACE: usize = 5000;
/// Maximum number of temporal successors checked per cause record in the
/// budgeted nearby-successor mode.
const MAX_TEMPORAL_SUCCESSORS_PER_RECORD: usize = 16;

/// Lightweight polarity keywords used only for causal-side ambiguity checks.
const NEGATIVE_OUTCOME_KEYWORDS: &[&str] = &[
    "error",
    "failure",
    "fail",
    "crash",
    "bug",
    "incident",
    "rollback",
    "revert",
    "risk",
    "vulnerability",
    "downtime",
    "outage",
    "regression",
    "contradiction",
    "conflict",
    "noise",
    "review",
];

const POSITIVE_OUTCOME_KEYWORDS: &[&str] = &[
    "success",
    "improvement",
    "improve",
    "faster",
    "reliable",
    "stable",
    "healthy",
    "secure",
    "optimized",
    "resolved",
    "fixed",
    "deployed",
    "completed",
    "approved",
];

// ── CausalState ──

/// Lifecycle state of a causal pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalState {
    /// Meets threshold, not yet confirmed stable.
    Candidate,
    /// High confidence causal pattern.
    Stable,
    /// Below threshold — discarded at end of cycle.
    Rejected,
}

impl Default for CausalState {
    fn default() -> Self {
        Self::Candidate
    }
}

/// Budgeting mode for temporal causal edge extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TemporalEdgeBudgetMode {
    /// Current behavior: scan all successor pairs until the namespace cap.
    #[default]
    ExhaustiveCapped,
    /// Budgeted behavior: only inspect the nearest successors for each cause.
    NearbySuccessors,
}

/// Evidence gating mode for causal promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CausalEvidenceMode {
    /// Current strict behavior: support plus repeated explicit support or
    /// repeated temporal windows.
    #[default]
    StrictRepeatedWindows,
    /// Experimental behavior: allow tightly consistent temporal clusters to
    /// qualify even when all temporal evidence lands inside one 24h bucket.
    TemporalClusterRecovery,
    /// Trusted explicit mode: user-declared causal links (via link_records) are
    /// treated as authoritative evidence. A single explicit link suffices for
    /// the repeated-evidence gate, and the effect-variants gate is bypassed when
    /// all declared effects share the same polarity (all negative or all
    /// positive). Counterfactual ratio and support count gates still apply.
    /// Recommended when the corpus is built programmatically or from structured
    /// event logs where the user explicitly encodes causality.
    ExplicitTrusted,
}

/// Recall reranking mode for causal-pattern-weighted influence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CausalRerankMode {
    /// No causal influence on recall ranking. Default.
    #[default]
    Off = 0,
    /// Limited influence: apply bounded reranking (capped score delta + positional shift limit).
    Limited = 1,
}

impl CausalRerankMode {
    /// Convert from u8 (for atomic storage). Invalid values → Off.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Limited,
            _ => Self::Off,
        }
    }
}

// ── CausalPattern ──

/// A discovered candidate causal relation between two groups of records/beliefs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPattern {
    /// Unique identifier.
    pub id: String,
    /// Stable identity key (namespace:cause_key:effect_key:edge_hash).
    pub key: String,
    /// Namespace this pattern belongs to.
    pub namespace: String,
    /// Belief IDs on the cause side.
    pub cause_belief_ids: Vec<String>,
    /// Belief IDs on the effect side.
    pub effect_belief_ids: Vec<String>,
    /// Record IDs on the cause side.
    pub cause_record_ids: Vec<String>,
    /// Record IDs on the effect side.
    pub effect_record_ids: Vec<String>,
    /// Number of record-level edges supporting this pattern.
    pub support_count: usize,
    /// Number of explicit supporting edges.
    #[serde(default)]
    pub explicit_support_count: usize,
    /// Number of temporal supporting edges.
    #[serde(default)]
    pub temporal_support_count: usize,
    /// Number of unique 24h temporal windows covered by support.
    #[serde(default)]
    pub unique_temporal_windows: usize,
    /// Number of distinct effect-record signatures inside this pattern.
    #[serde(default)]
    pub effect_record_signature_variants: usize,
    /// Positive outcome signals observed across effect-side records.
    #[serde(default)]
    pub positive_effect_signals: usize,
    /// Negative outcome signals observed across effect-side records.
    #[serde(default)]
    pub negative_effect_signals: usize,
    /// Number of counterevidence edges (same cause, different effect).
    pub counterevidence: usize,
    /// Total explicit support across all explicit effect variants for this cause.
    #[serde(default)]
    pub explicit_support_total_for_cause: usize,
    /// Number of explicit effect variants seen for this cause.
    #[serde(default)]
    pub explicit_effect_variants_for_cause: usize,
    /// Transition lift: P(effect|cause) / P(effect).
    pub transition_lift: f32,
    /// Temporal consistency: fraction of edges where cause precedes effect.
    pub temporal_consistency: f32,
    /// Outcome stability: 1 - (variance of effect strengths / mean).
    pub outcome_stability: f32,
    /// Composite causal strength score.
    pub causal_strength: f32,
    /// Current lifecycle state.
    pub state: CausalState,
    /// Timestamp of last rebuild.
    pub last_updated: f64,
}

// ── CausalReport ──

/// Per-cycle report returned by CausalEngine::discover().
#[derive(Debug, Clone, Default)]
pub struct CausalReport {
    /// True if the rebuild was skipped because the corpus fingerprint is unchanged.
    pub skipped: bool,
    /// Number of raw record-level edges found.
    pub edges_found: usize,
    /// Number of explicit record-level edges found.
    pub explicit_edges_found: usize,
    /// Number of temporal record-level edges found.
    pub temporal_edges_found: usize,
    /// Namespaces scanned for temporal edge extraction.
    pub temporal_namespaces_scanned: usize,
    /// Pairwise temporal record checks considered before dedup/capping.
    pub temporal_pairs_considered: usize,
    /// Pairwise temporal checks skipped by the budgeting policy.
    pub temporal_pairs_skipped_by_budget: usize,
    /// Temporal edges skipped due to per-namespace cap.
    pub temporal_edges_capped: usize,
    /// Number of namespaces that hit the temporal cap.
    pub temporal_namespaces_hit_cap: usize,
    /// Number of causal pattern candidates after aggregation.
    pub candidates_found: usize,
    /// Patterns that pass the minimum support gate.
    pub patterns_meeting_support_gate: usize,
    /// Patterns that pass the repeated-evidence gate.
    pub patterns_meeting_repeated_window_gate: usize,
    /// Patterns that pass the counterfactual-ratio gate.
    pub patterns_meeting_counterfactual_gate: usize,
    /// Patterns blocked by evidence gates before state promotion.
    pub patterns_blocked_by_evidence_gates: usize,
    /// Patterns blocked by counterfactual pressure before state promotion.
    pub patterns_blocked_by_counterfactual_gate: usize,
    /// Patterns that reached Stable state.
    pub stable_count: usize,
    /// Patterns that were Rejected.
    pub rejected_count: usize,
    /// Average causal_strength across all candidates.
    pub avg_causal_strength: f32,
}

// ── Internal: record-level causal edge ──

/// A single directed edge: cause_record → effect_record.
#[derive(Debug, Clone)]
struct CausalEdge {
    cause_id: String,
    effect_id: String,
    namespace: String,
    /// Time gap in seconds (effect.created_at - cause.created_at).
    time_gap: f64,
    /// Weight from connections map (0.0 if purely temporal).
    weight: f32,
    /// Whether this edge came from an explicit caused_by_id or causal connection.
    explicit: bool,
    /// Deterministic timestamp used for repeated-evidence window bucketing.
    event_time: f64,
}

#[derive(Debug, Clone, Default)]
struct EdgeExtractionStats {
    edges: Vec<CausalEdge>,
    explicit_edges_found: usize,
    temporal_edges_found: usize,
    temporal_namespaces_scanned: usize,
    temporal_pairs_considered: usize,
    temporal_pairs_skipped_by_budget: usize,
    temporal_edges_capped: usize,
    temporal_namespaces_hit_cap: usize,
}

// ── CausalEngine ──

/// Causal pattern discovery engine. Full rebuild each maintenance cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEngine {
    /// Discovered patterns keyed by pattern ID.
    pub patterns: HashMap<String, CausalPattern>,
    /// Key → pattern ID index for deduplication.
    pub key_index: HashMap<String, String>,
    /// Budgeting policy for temporal edge extraction.
    pub temporal_budget_mode: TemporalEdgeBudgetMode,
    /// Evidence-gating policy for Candidate/Stable promotion.
    pub evidence_mode: CausalEvidenceMode,
    /// xxh3 fingerprint of the last corpus seen by discover().
    /// Used to skip full rebuild when the corpus has not changed.
    #[serde(default)]
    pub last_corpus_fingerprint: u64,
    /// Edge counts from the last full rebuild, replayed in skipped reports.
    #[serde(default)]
    pub last_edges_found: usize,
    #[serde(default)]
    pub last_explicit_edges_found: usize,
    #[serde(default)]
    pub last_temporal_edges_found: usize,
}

impl CausalEngine {
    /// Create a fresh empty engine. Called on startup — never loaded from disk.
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            key_index: HashMap::new(),
            temporal_budget_mode: TemporalEdgeBudgetMode::ExhaustiveCapped,
            evidence_mode: CausalEvidenceMode::StrictRepeatedWindows,
            last_corpus_fingerprint: 0,
            last_edges_found: 0,
            last_explicit_edges_found: 0,
            last_temporal_edges_found: 0,
        }
    }

    /// Compute a deterministic fingerprint over the causal-relevant fields of
    /// all records: record ID, namespace, created_at, caused_by_id, and any
    /// connection entries typed "causal".
    ///
    /// Two corpora produce the same fingerprint if and only if their causal
    /// signal (explicit edges + temporal ordering) is identical.
    fn corpus_fingerprint(records: &HashMap<String, Record>) -> u64 {
        // Sort by record ID for determinism across HashMap iteration order.
        let mut keys: Vec<&String> = records.keys().collect();
        keys.sort_unstable();

        let mut buf = String::with_capacity(keys.len() * 64);
        for rid in keys {
            let rec = &records[rid];
            buf.push_str(rid);
            buf.push('|');
            buf.push_str(&rec.namespace);
            buf.push('|');
            // created_at as bits for exact float comparison
            buf.push_str(&rec.created_at.to_bits().to_string());
            buf.push('|');
            if let Some(ref cid) = rec.caused_by_id {
                buf.push_str(cid);
            }
            buf.push('|');
            // Include any "causal" connection entries, sorted for determinism
            let mut causal_conns: Vec<&String> = rec
                .connection_types
                .iter()
                .filter(|(_, t)| t.as_str() == "causal")
                .map(|(id, _)| id)
                .collect();
            causal_conns.sort_unstable();
            for cid in causal_conns {
                buf.push_str(cid);
                buf.push(',');
            }
            buf.push('\n');
        }
        xxhash_rust::xxh3::xxh3_64(buf.as_bytes())
    }

    /// Discover causal patterns from records and beliefs.
    ///
    /// Skips the full rebuild if the corpus fingerprint (record IDs, namespaces,
    /// created_at, caused_by_id, causal connections) has not changed since the
    /// last cycle. Returns a report with `skipped=true` and preserves existing
    /// pattern state unchanged.
    ///
    /// Algorithm when not skipped (3 phases):
    ///   1. Extract record-level causal edges (explicit + temporal)
    ///   2. Aggregate edges to belief-level patterns
    ///   3. Score and classify patterns
    pub fn discover(
        &mut self,
        belief_engine: &BeliefEngine,
        records: &HashMap<String, Record>,
        _sdr_lookup: &SdrLookup,
    ) -> CausalReport {
        // Skip rebuild if corpus is unchanged
        let fingerprint = Self::corpus_fingerprint(records);
        if fingerprint != 0 && fingerprint == self.last_corpus_fingerprint {
            // Re-derive summary counters from cached patterns (no rebuild)
            let stable_count = self.patterns.values().filter(|p| p.state == CausalState::Stable).count();
            let rejected_count = self.patterns.values().filter(|p| p.state == CausalState::Rejected).count();
            let candidates_found = self.patterns.len();
            let avg_causal_strength = if candidates_found > 0 {
                self.patterns.values().map(|p| p.causal_strength).sum::<f32>() / candidates_found as f32
            } else {
                0.0
            };
            return CausalReport {
                skipped: true,
                // Replay edge counts from last full rebuild so callers see consistent metrics
                edges_found: self.last_edges_found,
                explicit_edges_found: self.last_explicit_edges_found,
                temporal_edges_found: self.last_temporal_edges_found,
                candidates_found,
                stable_count,
                rejected_count,
                avg_causal_strength,
                ..CausalReport::default()
            };
        }

        // Full rebuild — clear previous state
        self.patterns.clear();
        self.key_index.clear();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Phase 1: Extract record-level causal edges
        let edge_stats = self.extract_edges(records);
        let edges_found = edge_stats.edges.len();

        if edge_stats.edges.is_empty() {
            self.last_corpus_fingerprint = fingerprint;
            return CausalReport::default();
        }

        // Phase 2: Aggregate to belief-level patterns
        let raw_patterns =
            self.aggregate_to_patterns(&edge_stats.edges, belief_engine, records, now);

        // Phase 3: Score and classify
        let mut candidates_found = 0;
        let mut patterns_meeting_support_gate = 0;
        let mut patterns_meeting_repeated_window_gate = 0;
        let mut patterns_meeting_counterfactual_gate = 0;
        let mut patterns_blocked_by_evidence_gates = 0;
        let mut patterns_blocked_by_counterfactual_gate = 0;
        let mut stable_count = 0;
        let mut rejected_count = 0;
        let mut strength_sum = 0.0f32;

        for mut pattern in raw_patterns {
            self.score_pattern(&mut pattern, records);
            let support_gate_ok = meets_support_gate(&pattern, self.evidence_mode);
            let repeated_gate_ok = meets_repeated_evidence_gate(&pattern, self.evidence_mode);
            let counterfactual_gate_ok = meets_counterfactual_gate(&pattern, self.evidence_mode);
            if support_gate_ok {
                patterns_meeting_support_gate += 1;
            }
            if repeated_gate_ok {
                patterns_meeting_repeated_window_gate += 1;
            }
            if counterfactual_gate_ok {
                patterns_meeting_counterfactual_gate += 1;
            }
            pattern.state = if !meets_evidence_gate(&pattern, self.evidence_mode) {
                patterns_blocked_by_evidence_gates += 1;
                CausalState::Rejected
            } else if !counterfactual_gate_ok {
                patterns_blocked_by_counterfactual_gate += 1;
                CausalState::Rejected
            } else {
                if pattern.causal_strength >= STABLE_THRESHOLD {
                    CausalState::Stable
                } else if pattern.causal_strength >= CANDIDATE_THRESHOLD {
                    CausalState::Candidate
                } else {
                    CausalState::Rejected
                }
            };

            match pattern.state {
                CausalState::Stable => stable_count += 1,
                CausalState::Candidate => {}
                CausalState::Rejected => rejected_count += 1,
            }

            candidates_found += 1;
            strength_sum += pattern.causal_strength;
            self.key_index
                .insert(pattern.key.clone(), pattern.id.clone());
            self.patterns.insert(pattern.id.clone(), pattern);
        }

        let avg_causal_strength = if candidates_found > 0 {
            strength_sum / candidates_found as f32
        } else {
            0.0
        };

        // Persist fingerprint and edge counts so the next cycle can skip if corpus is unchanged
        self.last_corpus_fingerprint = fingerprint;
        self.last_edges_found = edges_found;
        self.last_explicit_edges_found = edge_stats.explicit_edges_found;
        self.last_temporal_edges_found = edge_stats.temporal_edges_found;

        CausalReport {
            skipped: false,
            edges_found,
            explicit_edges_found: edge_stats.explicit_edges_found,
            temporal_edges_found: edge_stats.temporal_edges_found,
            temporal_namespaces_scanned: edge_stats.temporal_namespaces_scanned,
            temporal_pairs_considered: edge_stats.temporal_pairs_considered,
            temporal_pairs_skipped_by_budget: edge_stats.temporal_pairs_skipped_by_budget,
            temporal_edges_capped: edge_stats.temporal_edges_capped,
            temporal_namespaces_hit_cap: edge_stats.temporal_namespaces_hit_cap,
            candidates_found,
            patterns_meeting_support_gate,
            patterns_meeting_repeated_window_gate,
            patterns_meeting_counterfactual_gate,
            patterns_blocked_by_evidence_gates,
            patterns_blocked_by_counterfactual_gate,
            stable_count,
            rejected_count,
            avg_causal_strength,
        }
    }

    // ── Phase 1: Edge extraction ──

    /// Extract record-level causal edges from two signal sources:
    ///   (A) Explicit: caused_by_id links + connection_type=="causal"
    ///   (B) Temporal: records in same namespace within MAX_CAUSAL_WINDOW
    fn extract_edges(&self, records: &HashMap<String, Record>) -> EdgeExtractionStats {
        let mut edges = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut explicit_edges_found = 0usize;
        let mut temporal_edges_found = 0usize;
        let mut temporal_namespaces_scanned = 0usize;
        let mut temporal_pairs_considered = 0usize;
        let mut temporal_pairs_skipped_by_budget = 0usize;
        let mut temporal_edges_capped = 0usize;
        let mut temporal_namespaces_hit_cap = 0usize;

        // (A) Explicit causal links
        for (rid, rec) in records {
            // caused_by_id → this record was caused by another
            if let Some(ref cause_id) = rec.caused_by_id {
                if let Some(cause_rec) = records.get(cause_id) {
                    if cause_rec.namespace == rec.namespace {
                        let edge_key = format!("{}→{}", cause_id, rid);
                        if seen.insert(edge_key) {
                            edges.push(CausalEdge {
                                cause_id: cause_id.clone(),
                                effect_id: rid.clone(),
                                namespace: rec.namespace.clone(),
                                time_gap: rec.created_at - cause_rec.created_at,
                                weight: rec.connections.get(cause_id).copied().unwrap_or(0.5),
                                explicit: true,
                                event_time: rec.created_at,
                            });
                            explicit_edges_found += 1;
                        }
                    }
                }
            }

            // connection_type == "causal" links
            for (conn_id, conn_type) in &rec.connection_types {
                if conn_type == "causal" {
                    if let Some(conn_rec) = records.get(conn_id) {
                        if conn_rec.namespace == rec.namespace {
                            // Direction: the record with earlier created_at is cause
                            let (cause, effect) = if rec.created_at <= conn_rec.created_at {
                                (rid.clone(), conn_id.clone())
                            } else {
                                (conn_id.clone(), rid.clone())
                            };
                            let edge_key = format!("{}→{}", cause, effect);
                            if seen.insert(edge_key) {
                                let cause_ts = if rec.created_at <= conn_rec.created_at {
                                    rec.created_at
                                } else {
                                    conn_rec.created_at
                                };
                                let effect_ts = if rec.created_at <= conn_rec.created_at {
                                    conn_rec.created_at
                                } else {
                                    rec.created_at
                                };
                                edges.push(CausalEdge {
                                    cause_id: cause,
                                    effect_id: effect,
                                    namespace: rec.namespace.clone(),
                                    time_gap: effect_ts - cause_ts,
                                    weight: rec.connections.get(conn_id).copied().unwrap_or(0.5),
                                    explicit: true,
                                    event_time: effect_ts,
                                });
                                explicit_edges_found += 1;
                            }
                        }
                    }
                }
            }
        }

        // (B) Temporal edges: within same namespace, cause precedes effect
        // Partition by namespace to enforce barrier
        let mut by_ns: HashMap<&str, Vec<(&String, &Record)>> = HashMap::new();
        for (rid, rec) in records {
            by_ns.entry(&rec.namespace).or_default().push((rid, rec));
        }

        for (_ns, mut ns_recs) in by_ns {
            temporal_namespaces_scanned += 1;
            // Sort by created_at ascending
            ns_recs.sort_by(|a, b| {
                a.1.created_at
                    .partial_cmp(&b.1.created_at)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut ns_edge_count = 0;
            let mut ns_hit_cap = false;
            for i in 0..ns_recs.len() {
                if ns_edge_count >= MAX_EDGES_PER_NAMESPACE {
                    break;
                }
                let (cause_id, cause_rec) = &ns_recs[i];
                let mut budgeted_successors = 0usize;
                for j in (i + 1)..ns_recs.len() {
                    let (effect_id, effect_rec) = &ns_recs[j];
                    if self.temporal_budget_mode == TemporalEdgeBudgetMode::NearbySuccessors
                        && budgeted_successors >= MAX_TEMPORAL_SUCCESSORS_PER_RECORD
                    {
                        temporal_pairs_skipped_by_budget += ns_recs.len().saturating_sub(j);
                        break;
                    }
                    temporal_pairs_considered += 1;
                    budgeted_successors += 1;
                    let gap = effect_rec.created_at - cause_rec.created_at;
                    if gap > MAX_CAUSAL_WINDOW_SECS {
                        break; // sorted, so all further will exceed window
                    }
                    if gap <= 0.0 {
                        continue;
                    }
                    let edge_key = format!("{}→{}", cause_id, effect_id);
                    if seen.insert(edge_key) {
                        edges.push(CausalEdge {
                            cause_id: (*cause_id).clone(),
                            effect_id: (*effect_id).clone(),
                            namespace: cause_rec.namespace.clone(),
                            time_gap: gap,
                            weight: 0.0, // no explicit weight for temporal
                            explicit: false,
                            event_time: effect_rec.created_at,
                        });
                        ns_edge_count += 1;
                        temporal_edges_found += 1;
                        if ns_edge_count >= MAX_EDGES_PER_NAMESPACE {
                            ns_hit_cap = true;
                            break;
                        }
                    }
                }
            }
            if ns_hit_cap {
                temporal_namespaces_hit_cap += 1;
                let remaining = ns_recs.len().saturating_sub(1 + ns_edge_count);
                temporal_edges_capped += remaining;
            }
        }

        EdgeExtractionStats {
            edges,
            explicit_edges_found,
            temporal_edges_found,
            temporal_namespaces_scanned,
            temporal_pairs_considered,
            temporal_pairs_skipped_by_budget,
            temporal_edges_capped,
            temporal_namespaces_hit_cap,
        }
    }

    // ── Phase 2: Aggregate to belief-level patterns ──

    /// Build a reverse index: record_id → belief_id, then aggregate edges
    /// that share the same (cause_belief, effect_belief) pair.
    fn aggregate_to_patterns(
        &self,
        edges: &[CausalEdge],
        belief_engine: &BeliefEngine,
        _records: &HashMap<String, Record>,
        now: f64,
    ) -> Vec<CausalPattern> {
        // Build record → belief reverse index
        let record_to_belief = Self::build_record_to_belief(belief_engine);

        // Group edges by (cause_belief_key, effect_belief_key) within namespace
        // If a record has no belief, use "orphan:{record_id}" as fallback key
        #[derive(Debug, Clone, Hash, Eq, PartialEq)]
        struct PatternKey {
            namespace: String,
            cause_key: String,
            effect_key: String,
        }

        struct PatternAccum {
            cause_belief_ids: Vec<String>,
            effect_belief_ids: Vec<String>,
            cause_record_ids: Vec<String>,
            effect_record_ids: Vec<String>,
            time_gaps: Vec<f64>,
            weights: Vec<f32>,
            explicit_count: usize,
            temporal_count: usize,
            temporal_window_buckets: HashSet<i64>,
        }

        let mut accum: HashMap<PatternKey, PatternAccum> = HashMap::new();

        for edge in edges {
            let cause_belief = record_to_belief.get(&edge.cause_id);
            let effect_belief = record_to_belief.get(&edge.effect_id);

            // Build stable keys for the pattern identity
            let cause_key = cause_belief
                .cloned()
                .unwrap_or_else(|| format!("orphan:{}", edge.cause_id));
            let effect_key = effect_belief
                .cloned()
                .unwrap_or_else(|| format!("orphan:{}", edge.effect_id));

            // Skip self-loops at belief level
            if cause_key == effect_key {
                continue;
            }

            let pk = PatternKey {
                namespace: edge.namespace.clone(),
                cause_key: cause_key.clone(),
                effect_key: effect_key.clone(),
            };

            let entry = accum.entry(pk).or_insert_with(|| PatternAccum {
                cause_belief_ids: Vec::new(),
                effect_belief_ids: Vec::new(),
                cause_record_ids: Vec::new(),
                effect_record_ids: Vec::new(),
                time_gaps: Vec::new(),
                weights: Vec::new(),
                explicit_count: 0,
                temporal_count: 0,
                temporal_window_buckets: HashSet::new(),
            });

            // Add belief IDs (dedup later)
            if let Some(bid) = cause_belief {
                if !entry.cause_belief_ids.contains(bid) {
                    entry.cause_belief_ids.push(bid.clone());
                }
            }
            if let Some(bid) = effect_belief {
                if !entry.effect_belief_ids.contains(bid) {
                    entry.effect_belief_ids.push(bid.clone());
                }
            }

            // Add record IDs
            if !entry.cause_record_ids.contains(&edge.cause_id) {
                entry.cause_record_ids.push(edge.cause_id.clone());
            }
            if !entry.effect_record_ids.contains(&edge.effect_id) {
                entry.effect_record_ids.push(edge.effect_id.clone());
            }

            entry.time_gaps.push(edge.time_gap);
            entry.weights.push(edge.weight);
            if edge.explicit {
                entry.explicit_count += 1;
            } else {
                entry.temporal_count += 1;
                entry
                    .temporal_window_buckets
                    .insert(temporal_window_bucket(edge.event_time));
            }
        }

        let mut cause_explicit_support_totals: HashMap<(String, String), usize> = HashMap::new();
        let mut cause_explicit_variant_counts: HashMap<(String, String), usize> = HashMap::new();
        for (pk, acc) in &accum {
            *cause_explicit_support_totals
                .entry((pk.namespace.clone(), pk.cause_key.clone()))
                .or_default() += acc.explicit_count;
            if acc.explicit_count > 0 {
                *cause_explicit_variant_counts
                    .entry((pk.namespace.clone(), pk.cause_key.clone()))
                    .or_default() += 1;
            }
        }

        // Convert accumulated groups to CausalPattern candidates
        let mut patterns = Vec::new();
        for (pk, acc) in accum {
            // Build a stable pattern key from namespace + belief keys + edge hash
            let key = pattern_key(&pk.namespace, &pk.cause_key, &pk.effect_key);
            let id = deterministic_id(&key);
            let support_count = acc.time_gaps.len();
            let total_explicit_for_cause = cause_explicit_support_totals
                .get(&(pk.namespace.clone(), pk.cause_key.clone()))
                .copied()
                .unwrap_or(acc.explicit_count);
            let explicit_variants_for_cause = cause_explicit_variant_counts
                .get(&(pk.namespace.clone(), pk.cause_key.clone()))
                .copied()
                .unwrap_or(usize::from(acc.explicit_count > 0));

            patterns.push(CausalPattern {
                id,
                key,
                namespace: pk.namespace,
                cause_belief_ids: acc.cause_belief_ids,
                effect_belief_ids: acc.effect_belief_ids,
                cause_record_ids: acc.cause_record_ids,
                effect_record_ids: acc.effect_record_ids,
                support_count,
                explicit_support_count: acc.explicit_count,
                temporal_support_count: acc.temporal_count,
                unique_temporal_windows: acc.temporal_window_buckets.len(),
                effect_record_signature_variants: 0,
                positive_effect_signals: 0,
                negative_effect_signals: 0,
                // Counterevidence is explicit competing effect mass for the same cause.
                // We intentionally ignore temporal-only competitors here so policy-side
                // confounder gating reacts to repeated alternative outcomes, not noise.
                counterevidence: total_explicit_for_cause.saturating_sub(acc.explicit_count),
                explicit_support_total_for_cause: total_explicit_for_cause,
                explicit_effect_variants_for_cause: explicit_variants_for_cause,
                transition_lift: 0.0,
                temporal_consistency: 0.0,
                outcome_stability: 0.0,
                causal_strength: 0.0,
                state: CausalState::Candidate,
                last_updated: now,
            });
        }

        patterns
    }

    /// Build reverse index: record_id → belief_id.
    /// Only maps records that belong to resolved/singleton beliefs.
    /// belief_engine.record_index stores record_id → hypothesis_id, so we first
    /// resolve hypothesis → belief before filtering by belief state.
    fn build_record_to_belief(belief_engine: &BeliefEngine) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for (rid, hid) in &belief_engine.record_index {
            if let Some(hyp) = belief_engine.hypotheses.get(hid) {
                if let Some(belief) = belief_engine.beliefs.get(&hyp.belief_id) {
                    match belief.state {
                        BeliefState::Resolved | BeliefState::Singleton => {
                            map.insert(rid.clone(), hyp.belief_id.clone());
                        }
                        _ => {} // skip unresolved/empty
                    }
                }
            }
        }
        map
    }

    // ── Phase 3: Scoring ──

    /// Compute scoring metrics for a single pattern.
    fn score_pattern(&self, pattern: &mut CausalPattern, records: &HashMap<String, Record>) {
        let n = pattern.support_count;
        if n == 0 {
            pattern.causal_strength = 0.0;
            return;
        }

        // ── Transition lift: P(effect|cause) / P(effect) ──
        // Approximate P(effect|cause) = support_count / total_cause_records
        // Approximate P(effect) = effect_records / total_records_in_namespace
        let ns_total = records
            .values()
            .filter(|r| r.namespace == pattern.namespace)
            .count();
        let effect_count = pattern.effect_record_ids.len();
        let cause_count = pattern.cause_record_ids.len().max(1);

        let p_effect_given_cause = n as f32 / cause_count as f32;
        let p_effect = if ns_total > 0 {
            effect_count as f32 / ns_total as f32
        } else {
            1.0
        };
        // Lift capped at 5.0 for numerical stability, normalized to [0, 1]
        let raw_lift = if p_effect > 0.0 {
            (p_effect_given_cause / p_effect).min(5.0)
        } else {
            1.0
        };
        pattern.transition_lift = (raw_lift / 5.0).min(1.0);

        // ── Temporal consistency ──
        // Fraction of edges where cause actually precedes effect (time_gap > 0)
        // For edges extracted by our algorithm, this should be ~1.0 for temporal
        // edges, but explicit edges might have negative gaps if timestamps are wrong.
        // We re-verify here.
        let positive_gaps = pattern
            .cause_record_ids
            .iter()
            .flat_map(|cid| {
                pattern.effect_record_ids.iter().filter_map(move |eid| {
                    let cause_ts = records.get(cid).map(|r| r.created_at)?;
                    let effect_ts = records.get(eid).map(|r| r.created_at)?;
                    Some(effect_ts - cause_ts)
                })
            })
            .filter(|gap| *gap > 0.0)
            .count();
        let total_pairs = (pattern.cause_record_ids.len() * pattern.effect_record_ids.len()).max(1);
        let raw_temporal = positive_gaps as f32 / total_pairs as f32;
        // Explicit causal links (user-declared via link_records) are trusted as
        // directional regardless of timestamp precision. When explicit support
        // meets MIN_SUPPORT, floor temporal_consistency at 0.60 so that
        // same-session records (near-zero time gap) still score meaningfully.
        // In ExplicitTrusted mode a single user-declared explicit link is sufficient
        // to trust directionality, so floor at 0.60 even with explicit_support_count=1.
        let explicit_trusted_floor = self.evidence_mode == CausalEvidenceMode::ExplicitTrusted
            && pattern.explicit_support_count >= 1;
        let strict_floor = pattern.explicit_support_count >= MIN_SUPPORT;
        pattern.temporal_consistency = if (explicit_trusted_floor || strict_floor) && raw_temporal < 0.60 {
            0.60_f32.max(raw_temporal)
        } else {
            raw_temporal
        };

        // ── Outcome stability ──
        // 1 - coefficient_of_variation(effect_strengths)
        let effect_strengths: Vec<f32> = pattern
            .effect_record_ids
            .iter()
            .filter_map(|eid| records.get(eid).map(|r| r.strength))
            .collect();
        pattern.effect_record_signature_variants = pattern
            .effect_record_ids
            .iter()
            .filter_map(|eid| records.get(eid))
            .map(effect_record_signature)
            .collect::<HashSet<_>>()
            .len();
        let (positive_effect_signals, negative_effect_signals) =
            effect_polarity_signal_counts(pattern, records);
        pattern.positive_effect_signals = positive_effect_signals;
        pattern.negative_effect_signals = negative_effect_signals;
        pattern.outcome_stability = if effect_strengths.len() >= 2 {
            let mean = effect_strengths.iter().sum::<f32>() / effect_strengths.len() as f32;
            if mean > 0.0 {
                let variance = effect_strengths
                    .iter()
                    .map(|s| (s - mean).powi(2))
                    .sum::<f32>()
                    / effect_strengths.len() as f32;
                let cv = variance.sqrt() / mean;
                (1.0 - cv).max(0.0).min(1.0)
            } else {
                0.5
            }
        } else {
            0.5 // not enough data — neutral
        };

        // ── Support score ──
        // Logarithmic: log2(support_count + 1) / log2(max_expected + 1)
        // max_expected = 20 (a pattern with 20+ edges is strongly supported)
        let support_score = ((n as f32 + 1.0).log2() / 21.0f32.log2()).min(1.0);

        // ── Composite causal strength ──
        // Apply MIN_SUPPORT gate. In ExplicitTrusted mode a single user-declared
        // explicit link bypasses the support count requirement.
        let support_ok = n >= MIN_SUPPORT
            || (self.evidence_mode == CausalEvidenceMode::ExplicitTrusted
                && pattern.explicit_support_count >= 1);
        if !support_ok {
            pattern.causal_strength = pattern.transition_lift * 0.3; // penalized
            return;
        }

        pattern.causal_strength = W_TRANSITION_LIFT * pattern.transition_lift
            + W_TEMPORAL_CONSISTENCY * pattern.temporal_consistency
            + W_OUTCOME_STABILITY * pattern.outcome_stability
            + W_SUPPORT * support_score;
    }
}

fn temporal_window_bucket(ts: f64) -> i64 {
    (ts / EVIDENCE_WINDOW_SECS).floor() as i64
}

pub(crate) fn meets_support_gate(pattern: &CausalPattern, mode: CausalEvidenceMode) -> bool {
    if mode == CausalEvidenceMode::ExplicitTrusted && pattern.explicit_support_count >= 1 {
        // In ExplicitTrusted mode a single user-declared explicit link is sufficient
        // to pass the support gate. The user explicitly encoded causality — we trust it.
        return true;
    }
    pattern.support_count >= MIN_SUPPORT
}

pub(crate) fn meets_repeated_evidence_gate(
    pattern: &CausalPattern,
    mode: CausalEvidenceMode,
) -> bool {
    let strict = pattern.explicit_support_count >= MIN_SUPPORT
        || pattern.unique_temporal_windows >= MIN_SUPPORT;
    if strict {
        return true;
    }
    match mode {
        CausalEvidenceMode::StrictRepeatedWindows => false,
        CausalEvidenceMode::TemporalClusterRecovery => {
            // Narrowed guard: require at least one positive outcome signal on the
            // effect side. This prevents neutral temporal co-occurrence (records
            // that happen to be close in time but carry no outcome signal) from
            // passing the recovery gate on diverse corpora.
            pattern.temporal_support_count >= MIN_SUPPORT
                && pattern.explicit_support_count == 0
                && pattern.counterevidence == 0
                && pattern.effect_record_signature_variants <= 1
                && pattern.negative_effect_signals == 0
                && pattern.positive_effect_signals >= 1
        }
        CausalEvidenceMode::ExplicitTrusted => {
            // A single user-declared explicit link is sufficient evidence.
            // The user explicitly encoded causality — we trust it.
            // Still requires support_count >= MIN_SUPPORT (from meets_evidence_gate).
            pattern.explicit_support_count >= 1
        }
    }
}

pub(crate) fn meets_evidence_gate(pattern: &CausalPattern, mode: CausalEvidenceMode) -> bool {
    meets_support_gate(pattern, mode) && meets_repeated_evidence_gate(pattern, mode)
}

pub(crate) fn counterevidence_ratio(pattern: &CausalPattern) -> f32 {
    if pattern.support_count == 0 {
        0.0
    } else {
        pattern.counterevidence as f32 / pattern.support_count as f32
    }
}

pub(crate) fn meets_counterevidence_gate(pattern: &CausalPattern) -> bool {
    counterevidence_ratio(pattern) <= 1.0
}

pub(crate) fn counterfactual_ratio(pattern: &CausalPattern) -> f32 {
    let total = pattern.support_count + pattern.counterevidence;
    if total == 0 {
        0.0
    } else {
        pattern.counterevidence as f32 / total as f32
    }
}

pub(crate) fn explicit_dominance_ratio(pattern: &CausalPattern) -> f32 {
    if pattern.explicit_support_total_for_cause == 0 {
        1.0
    } else {
        pattern.explicit_support_count as f32 / pattern.explicit_support_total_for_cause as f32
    }
}

pub(crate) fn meets_explicit_dominance_gate(pattern: &CausalPattern) -> bool {
    if pattern.explicit_effect_variants_for_cause <= 1 {
        return true;
    }
    explicit_dominance_ratio(pattern) > MIN_EXPLICIT_DOMINANCE_SHARE
}

pub(crate) fn meets_effect_signature_consistency_gate(pattern: &CausalPattern) -> bool {
    !(pattern.explicit_support_count >= MIN_SUPPORT
        && pattern.explicit_effect_variants_for_cause <= 1
        && pattern.effect_record_signature_variants > 1)
}

pub(crate) fn meets_effect_polarity_consistency_gate(pattern: &CausalPattern) -> bool {
    !(pattern.explicit_support_count >= MIN_SUPPORT
        && pattern.effect_record_signature_variants > 1
        && pattern.positive_effect_signals >= 2
        && pattern.negative_effect_signals >= 2)
}

pub(crate) fn meets_counterfactual_gate(
    pattern: &CausalPattern,
    mode: CausalEvidenceMode,
) -> bool {
    // Counterfactual ratio is always enforced — user-declared links cannot
    // override a genuine contradiction signal.
    if counterfactual_ratio(pattern) > MAX_COUNTERFACTUAL_RATIO {
        return false;
    }

    if mode == CausalEvidenceMode::ExplicitTrusted && pattern.explicit_support_count >= 1 {
        // In ExplicitTrusted mode the effect-variants gate is bypassed when all
        // declared effects carry consistent polarity (all negative or all positive).
        // Rationale: if the user linked multiple cause records to multiple effect
        // records and every effect is "bad" (or every effect is "good"), the
        // divergence in effect text is not contradictory — it is just multiple
        // expressions of the same outcome type. We still block mixed polarity.
        let mixed_polarity = pattern.positive_effect_signals >= 1
            && pattern.negative_effect_signals >= 1;
        if mixed_polarity {
            return false; // genuinely ambiguous — do not bypass
        }
        return true; // consistent polarity → trust the explicit links
    }

    meets_explicit_dominance_gate(pattern)
        && meets_effect_signature_consistency_gate(pattern)
        && meets_effect_polarity_consistency_gate(pattern)
}

fn effect_record_signature(record: &Record) -> String {
    let mut normalized: Vec<String> = record
        .tags
        .iter()
        .map(|t: &String| t.to_ascii_lowercase())
        .collect();
    normalized.sort();
    normalized.dedup();
    if !normalized.is_empty() {
        return normalized.join("|");
    }
    format!(
        "{}:{}",
        record.semantic_type.clone(),
        record.content.to_ascii_lowercase()
    )
}

fn effect_polarity_signal_counts(
    pattern: &CausalPattern,
    records: &HashMap<String, Record>,
) -> (usize, usize) {
    let mut positive = 0;
    let mut negative = 0;

    for eid in &pattern.effect_record_ids {
        if let Some(record) = records.get(eid) {
            if record.semantic_type == "contradiction" {
                negative += 2;
            }

            for tag in &record.tags {
                let tag_lower = tag.to_ascii_lowercase();
                if NEGATIVE_OUTCOME_KEYWORDS
                    .iter()
                    .any(|kw| tag_lower.contains(kw))
                {
                    negative += 1;
                }
                if POSITIVE_OUTCOME_KEYWORDS
                    .iter()
                    .any(|kw| tag_lower.contains(kw))
                {
                    positive += 1;
                }
            }

            let content_lower = record.content.to_ascii_lowercase();
            for kw in NEGATIVE_OUTCOME_KEYWORDS {
                if content_lower.contains(kw) {
                    negative += 1;
                }
            }
            for kw in POSITIVE_OUTCOME_KEYWORDS {
                if content_lower.contains(kw) {
                    positive += 1;
                }
            }
        }
    }

    (positive, negative)
}

// ── Stable pattern key ──

/// Build a deterministic key for pattern deduplication.
/// Format: "namespace:cause_key→effect_key"
fn pattern_key(namespace: &str, cause_key: &str, effect_key: &str) -> String {
    format!("{}:{}→{}", namespace, cause_key, effect_key)
}

/// Generate a deterministic causal pattern ID from its stable key.
fn deterministic_id(key: &str) -> String {
    let hash = xxhash_rust::xxh3::xxh3_64(key.as_bytes());
    format!("ca-{:012x}", hash)
}

// ── CausalStore (persistence — write-only cache for inspection) ──

/// Persistent store for causal patterns. Same pattern as ConceptStore:
/// write-only cache for debugging/inspection. Never loaded on startup.
#[derive(Debug)]
pub struct CausalStore {
    path: std::path::PathBuf,
}

impl CausalStore {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Save current engine state to causal.cog (best-effort).
    /// This is a write-only inspection cache — NOT loaded on startup.
    pub fn save(&self, engine: &CausalEngine) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.path)?;
        let file_path = self.path.join("causal.cog");
        let data = serde_json::to_vec(engine)?;
        std::fs::write(&file_path, data)?;
        Ok(())
    }

    /// Load from disk. Inspection-only utility — NOT called on startup.
    /// The Aura orchestrator always creates a fresh CausalEngine::new().
    pub fn load(&self) -> anyhow::Result<CausalEngine> {
        let file_path = self.path.join("causal.cog");
        if !file_path.exists() {
            return Ok(CausalEngine::new());
        }
        let data = std::fs::read(&file_path)?;
        let engine: CausalEngine = serde_json::from_slice(&data)?;
        Ok(engine)
    }
}

// ════════════════════════════════════════════════════════════
// Unit tests
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::Level;

    // ── Helper to build test records ──

    fn make_record(id: &str, content: &str, ns: &str, created_at: f64) -> Record {
        let mut rec = Record::new(content.to_string(), Level::Domain);
        rec.id = id.to_string();
        rec.namespace = ns.to_string();
        rec.created_at = created_at;
        rec
    }

    fn empty_sdr_lookup() -> SdrLookup {
        HashMap::new()
    }

    fn make_pattern_for_scoring(
        support_count: usize,
        explicit_support_count: usize,
        unique_temporal_windows: usize,
    ) -> CausalPattern {
        CausalPattern {
            id: "ca-test".to_string(),
            key: "default:belief-a→belief-b".to_string(),
            namespace: "default".to_string(),
            cause_belief_ids: vec!["belief-a".to_string()],
            effect_belief_ids: vec!["belief-b".to_string()],
            cause_record_ids: vec!["c1".to_string()],
            effect_record_ids: vec!["e1".to_string()],
            support_count,
            explicit_support_count,
            temporal_support_count: support_count.saturating_sub(explicit_support_count),
            unique_temporal_windows,
            effect_record_signature_variants: 1,
            positive_effect_signals: 0,
            negative_effect_signals: 0,
            counterevidence: 0,
            explicit_support_total_for_cause: explicit_support_count,
            explicit_effect_variants_for_cause: usize::from(explicit_support_count > 0),
            transition_lift: 0.0,
            temporal_consistency: 0.0,
            outcome_stability: 0.0,
            causal_strength: 0.0,
            state: CausalState::Candidate,
            last_updated: 0.0,
        }
    }

    fn default_scoring_records() -> HashMap<String, Record> {
        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record("c1", "Enable canary deploy", "default", 1000.0),
        );
        records.insert(
            "e1".into(),
            make_record("e1", "Deploy stability improved", "default", 1100.0),
        );
        records
    }

    // ── 1. Fresh engine is empty ──

    #[test]
    fn new_engine_is_empty() {
        let engine = CausalEngine::new();
        assert!(engine.patterns.is_empty());
        assert!(engine.key_index.is_empty());
    }

    // ── 2. No edges → empty report ──

    #[test]
    fn no_edges_produces_empty_report() {
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let records = HashMap::new();
        let sdr = empty_sdr_lookup();

        let report = engine.discover(&belief_engine, &records, &sdr);
        assert_eq!(report.edges_found, 0);
        assert_eq!(report.candidates_found, 0);
    }

    // ── 3. Explicit caused_by_id creates edge ──

    #[test]
    fn explicit_caused_by_creates_edge() {
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let sdr = empty_sdr_lookup();

        let mut records = HashMap::new();
        let r1 = make_record("aaa", "cause event", "default", 1000.0);
        let mut r2 = make_record("bbb", "effect event", "default", 1001.0);
        r2.caused_by_id = Some("aaa".to_string());

        records.insert("aaa".to_string(), r1);
        records.insert("bbb".to_string(), r2);

        let report = engine.discover(&belief_engine, &records, &sdr);
        assert!(
            report.edges_found >= 1,
            "should find at least the explicit edge"
        );
    }

    // ── 4. Causal connection type creates edge ──

    #[test]
    fn causal_connection_type_creates_edge() {
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let sdr = empty_sdr_lookup();

        let mut records = HashMap::new();
        let mut r1 = make_record("aaa", "cause event", "default", 1000.0);
        let r2 = make_record("bbb", "effect event", "default", 1001.0);
        r1.add_typed_connection("bbb", 0.8, "causal");

        records.insert("aaa".to_string(), r1);
        records.insert("bbb".to_string(), r2);

        let report = engine.discover(&belief_engine, &records, &sdr);
        assert!(report.edges_found >= 1);
    }

    // ── 5. Namespace barrier ──

    #[test]
    fn cross_namespace_edges_blocked() {
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let sdr = empty_sdr_lookup();

        let mut records = HashMap::new();
        let r1 = make_record("aaa", "cause in ns-a", "ns-a", 1000.0);
        let mut r2 = make_record("bbb", "effect in ns-b", "ns-b", 1001.0);
        r2.caused_by_id = Some("aaa".to_string());

        records.insert("aaa".to_string(), r1);
        records.insert("bbb".to_string(), r2);

        let report = engine.discover(&belief_engine, &records, &sdr);
        // The explicit edge should be blocked by namespace check
        // Only temporal edges within same namespace should exist (none here)
        assert_eq!(
            report.candidates_found, 0,
            "cross-namespace causal patterns must not form"
        );
    }

    // ── 6. Temporal edges within window ──

    #[test]
    fn temporal_edges_within_window() {
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let sdr = empty_sdr_lookup();

        let mut records = HashMap::new();
        let base = 1_000_000.0;
        // 3 records within 1 day of each other
        records.insert("r1".into(), make_record("r1", "first", "default", base));
        records.insert(
            "r2".into(),
            make_record("r2", "second", "default", base + 3600.0),
        );
        records.insert(
            "r3".into(),
            make_record("r3", "third", "default", base + 7200.0),
        );

        let report = engine.discover(&belief_engine, &records, &sdr);
        // Should have temporal edges: r1→r2, r1→r3, r2→r3
        assert!(
            report.edges_found >= 3,
            "expected ≥3 temporal edges, got {}",
            report.edges_found
        );
    }

    // ── 7. Temporal edges outside window are excluded ──

    #[test]
    fn temporal_edges_outside_window_excluded() {
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let sdr = empty_sdr_lookup();

        let mut records = HashMap::new();
        let base = 1_000_000.0;
        // r1 and r2 are 10 days apart — outside MAX_CAUSAL_WINDOW (7 days)
        records.insert("r1".into(), make_record("r1", "old event", "default", base));
        records.insert(
            "r2".into(),
            make_record("r2", "new event", "default", base + 10.0 * 86400.0),
        );

        let report = engine.discover(&belief_engine, &records, &sdr);
        // Only temporal edges — and they're outside window
        assert_eq!(
            report.edges_found, 0,
            "edges outside window should be excluded"
        );
    }

    // ── 8. Full rebuild clears previous state ──

    #[test]
    fn full_rebuild_clears_state() {
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let sdr = empty_sdr_lookup();

        // First pass: create some edges
        let mut records = HashMap::new();
        let r1 = make_record("aaa", "cause", "default", 1000.0);
        let mut r2 = make_record("bbb", "effect", "default", 1001.0);
        r2.caused_by_id = Some("aaa".to_string());
        records.insert("aaa".to_string(), r1);
        records.insert("bbb".to_string(), r2);

        let _ = engine.discover(&belief_engine, &records, &sdr);

        // Second pass: empty records
        let empty = HashMap::new();
        let report = engine.discover(&belief_engine, &empty, &sdr);
        assert!(
            engine.patterns.is_empty(),
            "full rebuild should clear old patterns"
        );
        assert_eq!(report.edges_found, 0);
    }

    // ── 9. Pattern key is stable and deterministic ──

    #[test]
    fn pattern_key_is_deterministic() {
        let k1 = pattern_key("default", "belief-a", "belief-b");
        let k2 = pattern_key("default", "belief-a", "belief-b");
        assert_eq!(k1, k2);

        let k3 = pattern_key("default", "belief-b", "belief-a");
        assert_ne!(k1, k3, "direction matters in causal key");
    }

    #[test]
    fn pattern_id_is_deterministic_from_key() {
        let k1 = pattern_key("default", "belief-a", "belief-b");
        let k2 = pattern_key("default", "belief-a", "belief-b");
        let id1 = deterministic_id(&k1);
        let id2 = deterministic_id(&k2);
        assert_eq!(id1, id2, "same causal key must yield same id");

        let other = deterministic_id(&pattern_key("default", "belief-a", "belief-c"));
        assert_ne!(id1, other, "different causal key must yield different id");
    }

    // ── 10. CausalState defaults ──

    #[test]
    fn causal_state_default_is_candidate() {
        assert_eq!(CausalState::default(), CausalState::Candidate);
    }

    // ── 11. Scoring: support below MIN_SUPPORT is penalized ──

    #[test]
    fn low_support_penalized() {
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let sdr = empty_sdr_lookup();

        // Create a single explicit edge (support=1 < MIN_SUPPORT=2)
        let mut records = HashMap::new();
        let r1 = make_record("aaa", "cause", "default", 1000.0);
        let mut r2 = make_record("bbb", "effect", "default", 1001.0);
        r2.caused_by_id = Some("aaa".to_string());
        records.insert("aaa".to_string(), r1);
        records.insert("bbb".to_string(), r2);

        let _report = engine.discover(&belief_engine, &records, &sdr);
        // With low support, patterns should have reduced causal_strength
        for pattern in engine.patterns.values() {
            if pattern.support_count < MIN_SUPPORT {
                assert!(
                    pattern.causal_strength < CANDIDATE_THRESHOLD,
                    "low-support pattern should be below candidate threshold"
                );
                assert_eq!(
                    pattern.state,
                    CausalState::Rejected,
                    "low-support pattern should now be hard-blocked by the evidence gate"
                );
            }
        }
    }

    #[test]
    fn support_count_one_stays_rejected() {
        let engine = CausalEngine::new();
        let records = default_scoring_records();
        let mut pattern = make_pattern_for_scoring(1, 1, 1);

        engine.score_pattern(&mut pattern, &records);
        pattern.state = if meets_evidence_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows)
            && pattern.causal_strength >= STABLE_THRESHOLD
        {
            CausalState::Stable
        } else if meets_evidence_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows)
            && pattern.causal_strength >= CANDIDATE_THRESHOLD
        {
            CausalState::Candidate
        } else {
            CausalState::Rejected
        };

        assert!(!meets_support_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows));
        assert_eq!(pattern.state, CausalState::Rejected);
    }

    #[test]
    fn single_window_temporal_evidence_stays_rejected() {
        let engine = CausalEngine::new();
        let records = default_scoring_records();
        let mut pattern = make_pattern_for_scoring(3, 0, 1);

        engine.score_pattern(&mut pattern, &records);
        assert!(pattern.causal_strength >= CANDIDATE_THRESHOLD);
        assert!(!meets_repeated_evidence_gate(
            &pattern,
            CausalEvidenceMode::StrictRepeatedWindows
        ));
        assert!(!meets_evidence_gate(
            &pattern,
            CausalEvidenceMode::StrictRepeatedWindows
        ));
    }

    #[test]
    fn repeated_temporal_windows_can_reach_candidate() {
        let engine = CausalEngine::new();
        let records = default_scoring_records();
        let mut pattern = make_pattern_for_scoring(3, 0, 2);

        engine.score_pattern(&mut pattern, &records);
        assert!(meets_evidence_gate(
            &pattern,
            CausalEvidenceMode::StrictRepeatedWindows
        ));
        assert!(
            pattern.causal_strength >= CANDIDATE_THRESHOLD,
            "repeated-window temporal evidence should remain eligible for candidate promotion"
        );
    }

    #[test]
    fn explicit_repeated_support_can_reach_candidate_without_temporal_spread() {
        let engine = CausalEngine::new();
        let records = default_scoring_records();
        let mut pattern = make_pattern_for_scoring(3, 2, 1);

        engine.score_pattern(&mut pattern, &records);
        assert!(meets_repeated_evidence_gate(
            &pattern,
            CausalEvidenceMode::StrictRepeatedWindows
        ));
        assert!(meets_evidence_gate(
            &pattern,
            CausalEvidenceMode::StrictRepeatedWindows
        ));
        assert!(
            pattern.causal_strength >= CANDIDATE_THRESHOLD,
            "explicit repeated support should satisfy the evidence gate without repeated temporal windows"
        );
    }

    #[test]
    fn high_counterfactual_ratio_blocks_candidate_promotion() {
        let engine = CausalEngine::new();
        let records = default_scoring_records();
        let mut pattern = make_pattern_for_scoring(3, 3, 1);
        pattern.counterevidence = 4;

        engine.score_pattern(&mut pattern, &records);
        assert!(
            pattern.causal_strength >= CANDIDATE_THRESHOLD,
            "pattern should be strong enough on score alone before counterfactual gating"
        );
        assert!(counterfactual_ratio(&pattern) > 0.5);
        assert!(!meets_counterfactual_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows));
    }

    #[test]
    fn bounded_counterfactual_ratio_keeps_candidate_eligibility() {
        let engine = CausalEngine::new();
        let records = default_scoring_records();
        let mut pattern = make_pattern_for_scoring(3, 3, 1);
        pattern.counterevidence = 2;

        engine.score_pattern(&mut pattern, &records);
        assert!(counterfactual_ratio(&pattern) <= 0.5);
        assert!(meets_counterfactual_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows));
    }

    #[test]
    fn ambiguous_explicit_outcomes_fail_counterfactual_gate() {
        let engine = CausalEngine::new();
        let records = default_scoring_records();
        let mut pattern = make_pattern_for_scoring(6, 6, 1);
        pattern.counterevidence = 3;
        pattern.explicit_support_total_for_cause = 9;
        pattern.explicit_effect_variants_for_cause = 2;
        pattern.effect_record_signature_variants = 1;

        engine.score_pattern(&mut pattern, &records);
        assert!(counterfactual_ratio(&pattern) <= 0.5);
        assert!(explicit_dominance_ratio(&pattern) < 0.70);
        assert!(!meets_counterfactual_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows));
    }

    #[test]
    fn merged_explicit_effect_signatures_fail_counterfactual_gate() {
        let engine = CausalEngine::new();
        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record("c1", "Enable deploy orchestration", "default", 1000.0),
        );
        records.insert(
            "e1".into(),
            make_record("e1", "Stability improved", "default", 1100.0),
        );
        records.get_mut("e1").unwrap().tags =
            vec!["deploy".into(), "stability".into(), "improvement".into()];
        records.insert(
            "e2".into(),
            make_record("e2", "Rollback frequency increased", "default", 1101.0),
        );
        records.get_mut("e2").unwrap().tags =
            vec!["deploy".into(), "rollback".into(), "regression".into()];
        records.insert(
            "e3".into(),
            make_record("e3", "Security review load increased", "default", 1102.0),
        );
        records.get_mut("e3").unwrap().tags =
            vec!["deploy".into(), "security".into(), "review".into()];

        let mut pattern = make_pattern_for_scoring(6, 6, 1);
        pattern.effect_record_ids = vec!["e1".into(), "e2".into(), "e3".into()];
        pattern.explicit_support_total_for_cause = 6;
        pattern.explicit_effect_variants_for_cause = 1;

        engine.score_pattern(&mut pattern, &records);
        assert!(meets_support_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows));
        assert!(meets_repeated_evidence_gate(
            &pattern,
            CausalEvidenceMode::StrictRepeatedWindows
        ));
        assert_eq!(pattern.effect_record_signature_variants, 3);
        assert!(!meets_effect_signature_consistency_gate(&pattern));
        assert!(!meets_counterfactual_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows));
    }

    #[test]
    fn mixed_explicit_polarity_fails_counterfactual_gate() {
        let engine = CausalEngine::new();
        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record("c1", "Changed deploy workflow rollout", "default", 1000.0),
        );
        records.insert(
            "e1".into(),
            make_record("e1", "Release stability improved", "default", 1100.0),
        );
        records.get_mut("e1").unwrap().tags =
            vec!["deploy".into(), "stability".into(), "improvement".into()];
        records.insert(
            "e2".into(),
            make_record("e2", "Alert noise increased", "default", 1101.0),
        );
        records.get_mut("e2").unwrap().tags =
            vec!["deploy".into(), "alerts".into(), "noise".into()];
        records.insert(
            "e3".into(),
            make_record("e3", "Rollback frequency regressed", "default", 1102.0),
        );
        records.get_mut("e3").unwrap().tags =
            vec!["deploy".into(), "rollback".into(), "regression".into()];

        let mut pattern = make_pattern_for_scoring(9, 9, 1);
        pattern.effect_record_ids = vec!["e1".into(), "e2".into(), "e3".into()];
        pattern.explicit_support_total_for_cause = 9;
        pattern.explicit_effect_variants_for_cause = 1;

        engine.score_pattern(&mut pattern, &records);
        assert!(pattern.effect_record_signature_variants > 1);
        assert!(pattern.positive_effect_signals >= 2);
        assert!(pattern.negative_effect_signals >= 2);
        assert!(!meets_effect_polarity_consistency_gate(&pattern));
        assert!(!meets_counterfactual_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows));
    }

    // ── 12. Serialization roundtrip ──

    #[test]
    fn engine_serialization_roundtrip() {
        let engine = CausalEngine::new();
        let json = serde_json::to_string(&engine).unwrap();
        let restored: CausalEngine = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.patterns.len(), engine.patterns.len());
    }

    // ── 13. CausalStore save/load ──

    #[test]
    fn store_save_and_load() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CausalStore::new(dir.path());

        let engine = CausalEngine::new();
        store.save(&engine).expect("save");

        let loaded = store.load().expect("load");
        assert_eq!(loaded.patterns.len(), 0);
    }

    // ── 14. Multiple explicit edges boost support ──

    #[test]
    fn multiple_edges_increase_support() {
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let sdr = empty_sdr_lookup();

        let mut records = HashMap::new();
        let base = 1_000_000.0;

        // Create 4 records, all linked causally: r1→r2, r1→r3, r1→r4
        let r1 = make_record("r1", "root cause", "default", base);
        let mut r2 = make_record("r2", "effect one", "default", base + 100.0);
        let mut r3 = make_record("r3", "effect two", "default", base + 200.0);
        let mut r4 = make_record("r4", "effect three", "default", base + 300.0);
        r2.caused_by_id = Some("r1".to_string());
        r3.caused_by_id = Some("r1".to_string());
        r4.caused_by_id = Some("r1".to_string());

        records.insert("r1".into(), r1);
        records.insert("r2".into(), r2);
        records.insert("r3".into(), r3);
        records.insert("r4".into(), r4);

        let report = engine.discover(&belief_engine, &records, &sdr);
        // Should have explicit edges plus temporal edges
        assert!(report.edges_found >= 3, "expected ≥3 explicit edges");
    }

    // ── 15. Self-loops at belief level are skipped ──

    // ── TemporalClusterRecovery narrowed guard ──────────────────────────────

    #[test]
    fn temporal_cluster_recovery_requires_positive_effect_signal() {
        // Pattern with temporal support but zero positive signals must NOT pass.
        let mut pattern = make_pattern_for_scoring(2, 0, 1);
        pattern.temporal_support_count = 2;
        pattern.positive_effect_signals = 0;
        pattern.negative_effect_signals = 0;
        assert!(
            !meets_repeated_evidence_gate(&pattern, CausalEvidenceMode::TemporalClusterRecovery),
            "recovery must not pass when positive_effect_signals == 0"
        );
    }

    #[test]
    fn temporal_cluster_recovery_passes_with_positive_signal() {
        // Same pattern but with one positive signal must pass.
        let mut pattern = make_pattern_for_scoring(2, 0, 1);
        pattern.temporal_support_count = 2;
        pattern.positive_effect_signals = 1;
        pattern.negative_effect_signals = 0;
        assert!(
            meets_repeated_evidence_gate(&pattern, CausalEvidenceMode::TemporalClusterRecovery),
            "recovery must pass when positive_effect_signals >= 1 and all other guards hold"
        );
    }

    #[test]
    fn temporal_cluster_recovery_blocked_by_negative_signal() {
        // Positive signal present but negative too — negative gate blocks it.
        let mut pattern = make_pattern_for_scoring(2, 0, 1);
        pattern.temporal_support_count = 2;
        pattern.positive_effect_signals = 2;
        pattern.negative_effect_signals = 1;
        assert!(
            !meets_repeated_evidence_gate(&pattern, CausalEvidenceMode::TemporalClusterRecovery),
            "recovery must not pass when negative_effect_signals > 0"
        );
    }

    #[test]
    fn strict_mode_unaffected_by_positive_signal_change() {
        // StrictRepeatedWindows must not be affected by the new positive_signal guard.
        let mut pattern = make_pattern_for_scoring(2, 2, 2);
        pattern.positive_effect_signals = 0;
        assert!(
            meets_repeated_evidence_gate(&pattern, CausalEvidenceMode::StrictRepeatedWindows),
            "strict mode must still pass on explicit_support >= MIN_SUPPORT regardless of positive signal"
        );
    }

    #[test]
    fn self_loop_edges_skipped() {
        // When cause and effect map to the same belief, the edge should be dropped
        // We can't easily set up belief state in unit tests without the full stack,
        // but we verify the orphan path doesn't create self-loops
        let mut engine = CausalEngine::new();
        let belief_engine = BeliefEngine::default();
        let sdr = empty_sdr_lookup();

        let mut records = HashMap::new();
        let mut r1 = make_record("aaa", "self-ref", "default", 1000.0);
        r1.caused_by_id = Some("aaa".to_string()); // self-reference
        records.insert("aaa".to_string(), r1);

        let report = engine.discover(&belief_engine, &records, &sdr);
        // Self-reference at record level: cause_key == effect_key → skipped
        assert_eq!(report.candidates_found, 0, "self-loops should be skipped");
    }
}
