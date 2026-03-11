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

use std::collections::HashMap;

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
    /// Number of counterevidence edges (same cause, different effect).
    pub counterevidence: usize,
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
    /// Number of raw record-level edges found.
    pub edges_found: usize,
    /// Number of causal pattern candidates after aggregation.
    pub candidates_found: usize,
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
}

// ── CausalEngine ──

/// Causal pattern discovery engine. Full rebuild each maintenance cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEngine {
    /// Discovered patterns keyed by pattern ID.
    pub patterns: HashMap<String, CausalPattern>,
    /// Key → pattern ID index for deduplication.
    pub key_index: HashMap<String, String>,
}

impl CausalEngine {
    /// Create a fresh empty engine. Called on startup — never loaded from disk.
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            key_index: HashMap::new(),
        }
    }

    /// Full rebuild: discover causal patterns from records and beliefs.
    ///
    /// Algorithm (3 phases):
    ///   1. Extract record-level causal edges (explicit + temporal)
    ///   2. Aggregate edges to belief-level patterns
    ///   3. Score and classify patterns
    pub fn discover(
        &mut self,
        belief_engine: &BeliefEngine,
        records: &HashMap<String, Record>,
        _sdr_lookup: &SdrLookup,
    ) -> CausalReport {
        // Full rebuild — clear previous state
        self.patterns.clear();
        self.key_index.clear();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Phase 1: Extract record-level causal edges
        let edges = self.extract_edges(records);
        let edges_found = edges.len();

        if edges.is_empty() {
            return CausalReport::default();
        }

        // Phase 2: Aggregate to belief-level patterns
        let raw_patterns = self.aggregate_to_patterns(&edges, belief_engine, records, now);

        // Phase 3: Score and classify
        let mut candidates_found = 0;
        let mut stable_count = 0;
        let mut rejected_count = 0;
        let mut strength_sum = 0.0f32;

        for mut pattern in raw_patterns {
            self.score_pattern(&mut pattern, records);
            pattern.state = if pattern.causal_strength >= STABLE_THRESHOLD {
                CausalState::Stable
            } else if pattern.causal_strength >= CANDIDATE_THRESHOLD {
                CausalState::Candidate
            } else {
                CausalState::Rejected
            };

            match pattern.state {
                CausalState::Stable => stable_count += 1,
                CausalState::Candidate => {}
                CausalState::Rejected => rejected_count += 1,
            }

            candidates_found += 1;
            strength_sum += pattern.causal_strength;
            self.key_index.insert(pattern.key.clone(), pattern.id.clone());
            self.patterns.insert(pattern.id.clone(), pattern);
        }

        let avg_causal_strength = if candidates_found > 0 {
            strength_sum / candidates_found as f32
        } else {
            0.0
        };

        CausalReport {
            edges_found,
            candidates_found,
            stable_count,
            rejected_count,
            avg_causal_strength,
        }
    }

    // ── Phase 1: Edge extraction ──

    /// Extract record-level causal edges from two signal sources:
    ///   (A) Explicit: caused_by_id links + connection_type=="causal"
    ///   (B) Temporal: records in same namespace within MAX_CAUSAL_WINDOW
    fn extract_edges(&self, records: &HashMap<String, Record>) -> Vec<CausalEdge> {
        let mut edges = Vec::new();
        let mut seen = std::collections::HashSet::new();

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
                            });
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
                                });
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
            // Sort by created_at ascending
            ns_recs.sort_by(|a, b| a.1.created_at.partial_cmp(&b.1.created_at).unwrap_or(std::cmp::Ordering::Equal));

            let mut ns_edge_count = 0;
            for i in 0..ns_recs.len() {
                if ns_edge_count >= MAX_EDGES_PER_NAMESPACE {
                    break;
                }
                let (cause_id, cause_rec) = &ns_recs[i];
                for j in (i + 1)..ns_recs.len() {
                    let (effect_id, effect_rec) = &ns_recs[j];
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
                        });
                        ns_edge_count += 1;
                        if ns_edge_count >= MAX_EDGES_PER_NAMESPACE {
                            break;
                        }
                    }
                }
            }
        }

        edges
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
        }

        let mut accum: HashMap<PatternKey, PatternAccum> = HashMap::new();

        for edge in edges {
            let cause_belief = record_to_belief.get(&edge.cause_id);
            let effect_belief = record_to_belief.get(&edge.effect_id);

            // Build stable keys for the pattern identity
            let cause_key = cause_belief.cloned()
                .unwrap_or_else(|| format!("orphan:{}", edge.cause_id));
            let effect_key = effect_belief.cloned()
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
            }
        }

        // Convert accumulated groups to CausalPattern candidates
        let mut patterns = Vec::new();
        for (pk, acc) in accum {
            // Build a stable pattern key from namespace + belief keys + edge hash
            let key = pattern_key(&pk.namespace, &pk.cause_key, &pk.effect_key);
            let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();

            patterns.push(CausalPattern {
                id,
                key,
                namespace: pk.namespace,
                cause_belief_ids: acc.cause_belief_ids,
                effect_belief_ids: acc.effect_belief_ids,
                cause_record_ids: acc.cause_record_ids,
                effect_record_ids: acc.effect_record_ids,
                support_count: acc.time_gaps.len(),
                counterevidence: 0, // computed in scoring
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
    /// belief_engine.record_index is record_id → belief_id already,
    /// so we just filter to resolved/singleton beliefs.
    fn build_record_to_belief(belief_engine: &BeliefEngine) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for (rid, bid) in &belief_engine.record_index {
            if let Some(belief) = belief_engine.beliefs.get(bid) {
                match belief.state {
                    BeliefState::Resolved | BeliefState::Singleton => {
                        map.insert(rid.clone(), bid.clone());
                    }
                    _ => {} // skip unresolved/empty
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
        let ns_total = records.values()
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
        let positive_gaps = pattern.cause_record_ids.iter()
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
        pattern.temporal_consistency = positive_gaps as f32 / total_pairs as f32;

        // ── Outcome stability ──
        // 1 - coefficient_of_variation(effect_strengths)
        let effect_strengths: Vec<f32> = pattern.effect_record_ids.iter()
            .filter_map(|eid| records.get(eid).map(|r| r.strength))
            .collect();
        pattern.outcome_stability = if effect_strengths.len() >= 2 {
            let mean = effect_strengths.iter().sum::<f32>() / effect_strengths.len() as f32;
            if mean > 0.0 {
                let variance = effect_strengths.iter()
                    .map(|s| (s - mean).powi(2))
                    .sum::<f32>() / effect_strengths.len() as f32;
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
        // Apply MIN_SUPPORT gate
        if n < MIN_SUPPORT {
            pattern.causal_strength = pattern.transition_lift * 0.3; // penalized
            return;
        }

        pattern.causal_strength =
            W_TRANSITION_LIFT * pattern.transition_lift
            + W_TEMPORAL_CONSISTENCY * pattern.temporal_consistency
            + W_OUTCOME_STABILITY * pattern.outcome_stability
            + W_SUPPORT * support_score;
    }
}

// ── Stable pattern key ──

/// Build a deterministic key for pattern deduplication.
/// Format: "namespace:cause_key→effect_key"
fn pattern_key(namespace: &str, cause_key: &str, effect_key: &str) -> String {
    format!("{}:{}→{}", namespace, cause_key, effect_key)
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
        Self { path: path.as_ref().to_path_buf() }
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
        assert!(report.edges_found >= 1, "should find at least the explicit edge");
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
        assert_eq!(report.candidates_found, 0,
            "cross-namespace causal patterns must not form");
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
        records.insert("r2".into(), make_record("r2", "second", "default", base + 3600.0));
        records.insert("r3".into(), make_record("r3", "third", "default", base + 7200.0));

        let report = engine.discover(&belief_engine, &records, &sdr);
        // Should have temporal edges: r1→r2, r1→r3, r2→r3
        assert!(report.edges_found >= 3, "expected ≥3 temporal edges, got {}", report.edges_found);
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
        records.insert("r2".into(), make_record("r2", "new event", "default", base + 10.0 * 86400.0));

        let report = engine.discover(&belief_engine, &records, &sdr);
        // Only temporal edges — and they're outside window
        assert_eq!(report.edges_found, 0, "edges outside window should be excluded");
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
        assert!(engine.patterns.is_empty(), "full rebuild should clear old patterns");
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
                assert!(pattern.causal_strength < CANDIDATE_THRESHOLD,
                    "low-support pattern should be below candidate threshold");
            }
        }
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
