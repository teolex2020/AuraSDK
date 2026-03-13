//! Living Memory — autonomous 8-phase maintenance engine.
//!
//! Rewritten from background_brain.py (generic phases only).
//! Agent-specific features (Telegram, file cleanup, knowledge sync) are NOT included.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::RwLock;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::levels::Level;
use crate::record::Record;
use crate::trust::TagTaxonomy;

// ── Archival Rule ──

/// Configurable archival rule for a tag category.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArchivalRule {
    /// Tag to match.
    pub tag: String,
    /// Maximum age in days before deletion.
    pub max_age_days: u32,
    /// Keep at least this many most-recent records.
    pub keep_recent: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl ArchivalRule {
    #[new]
    fn py_new(tag: String, max_age_days: u32, keep_recent: usize) -> Self {
        Self {
            tag,
            max_age_days,
            keep_recent,
        }
    }
}

/// Completed archival rule — only deletes completed/done items.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct CompletedArchivalRule {
    pub tag: String,
    pub max_age_days: u32,
}

#[cfg(feature = "python")]
#[pymethods]
impl CompletedArchivalRule {
    #[new]
    fn py_new(tag: String, max_age_days: u32) -> Self {
        Self { tag, max_age_days }
    }
}

// ── Maintenance Config ──

/// User-configurable maintenance settings.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct MaintenanceConfig {
    pub decay_enabled: bool,
    pub reflect_enabled: bool,
    pub insights_enabled: bool,
    pub consolidation_enabled: bool,
    pub synthesis_enabled: bool,
    pub archival_enabled: bool,
    /// Run level fix every Nth cycle.
    pub level_fix_interval: u64,
    /// Max clusters per consolidation run.
    pub max_clusters_per_run: usize,
    /// Configurable archival rules.
    pub archival_rules: Vec<ArchivalRule>,
    /// Completed-item archival rules.
    pub completed_archival_rules: Vec<CompletedArchivalRule>,
    /// Tag used for scheduled tasks (default: "scheduled-task").
    pub task_tag: String,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            decay_enabled: true,
            reflect_enabled: true,
            insights_enabled: true,
            consolidation_enabled: true,
            synthesis_enabled: true,
            archival_enabled: true,
            level_fix_interval: 10,
            max_clusters_per_run: 3,
            archival_rules: vec![
                ArchivalRule {
                    tag: "web-search-cache".into(),
                    max_age_days: 1,
                    keep_recent: 0,
                },
                ArchivalRule {
                    tag: "autonomous-outcome".into(),
                    max_age_days: 7,
                    keep_recent: 50,
                },
                ArchivalRule {
                    tag: "session-summary".into(),
                    max_age_days: 14,
                    keep_recent: 20,
                },
                ArchivalRule {
                    tag: "proactive-session".into(),
                    max_age_days: 7,
                    keep_recent: 20,
                },
                ArchivalRule {
                    tag: "action-plan".into(),
                    max_age_days: 14,
                    keep_recent: 10,
                },
                ArchivalRule {
                    tag: "session-reflection".into(),
                    max_age_days: 30,
                    keep_recent: 50,
                },
                ArchivalRule {
                    tag: "research-finding".into(),
                    max_age_days: 30,
                    keep_recent: 100,
                },
                ArchivalRule {
                    tag: "consolidated-meta".into(),
                    max_age_days: 90,
                    keep_recent: 200,
                },
                ArchivalRule {
                    tag: "research-project".into(),
                    max_age_days: 90,
                    keep_recent: 50,
                },
                ArchivalRule {
                    tag: "feedback-signal".into(),
                    max_age_days: 14,
                    keep_recent: 50,
                },
            ],
            completed_archival_rules: vec![
                CompletedArchivalRule {
                    tag: "todo-item".into(),
                    max_age_days: 30,
                },
                CompletedArchivalRule {
                    tag: "scheduled-task".into(),
                    max_age_days: 30,
                },
            ],
            task_tag: "scheduled-task".into(),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MaintenanceConfig {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

// ── Maintenance Report ──

/// Decay phase report.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct DecayReport {
    pub decayed: usize,
    pub archived: usize,
}

/// Reflect phase report.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct ReflectReport {
    pub promoted: usize,
    pub archived: usize,
}

/// Consolidation phase report.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct ConsolidationReport {
    pub native_merged: usize,
    pub clusters_found: usize,
    pub meta_created: usize,
}

/// Epistemic phase report — support/conflict propagation from local graph structure.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct EpistemicPhaseReport {
    /// Number of records whose epistemic state changed this cycle.
    pub updated_records: usize,
    /// Sum of confirming neighbor counts across all live records.
    pub total_support_links: usize,
    /// Sum of conflicting neighbor counts across all live records.
    pub total_conflict_links: usize,
    /// Number of records with materially non-zero volatility after the update.
    pub volatile_records: usize,
}

/// Belief phase report — epistemic layer stats from a single maintenance cycle.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct BeliefPhaseReport {
    /// Number of new beliefs created this cycle.
    pub beliefs_created: usize,
    /// Number of stale beliefs pruned.
    pub beliefs_pruned: usize,
    /// Number of belief revisions (winner changed).
    pub revisions: usize,
    /// Number of beliefs with a clear winner.
    pub resolved: usize,
    /// Number of beliefs with no clear winner.
    pub unresolved: usize,
    /// Total active beliefs after this cycle.
    pub total_beliefs: usize,
    /// Total active hypotheses after this cycle.
    pub total_hypotheses: usize,
    /// Churn rate = revisions / max(total_beliefs, 1).
    /// Values > 0.10 on stable data indicate belief layer instability.
    pub churn_rate: f32,
}

/// Concept phase report — concept discovery stats from a single maintenance cycle.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct ConceptPhaseReport {
    /// Number of eligible belief seeds found.
    pub seeds_found: usize,
    /// Total concept candidates discovered.
    pub candidates_found: usize,
    /// Candidates that reached Stable state.
    pub stable_count: usize,
    /// Candidates that were Rejected.
    pub rejected_count: usize,
    /// Average abstraction score across all candidates.
    pub avg_abstraction_score: f32,
    // ── Centroid diagnostics ──
    /// Non-empty centroids built.
    pub centroids_built: usize,
    /// Partitions with >= 2 seeds.
    pub partitions_with_multiple_seeds: usize,
    /// Sizes of partitions with >= 2 seeds.
    pub multi_seed_partition_sizes: Vec<usize>,
    /// Sizes of clusters emitted by union-find clustering.
    pub cluster_sizes: Vec<usize>,
    /// Number of clusters with >= 2 beliefs.
    pub clusters_with_multiple_beliefs: usize,
    /// Largest cluster size observed this cycle.
    pub largest_cluster_size: usize,
    /// Total pairwise centroid comparisons.
    pub pairwise_comparisons: usize,
    /// Pairs above similarity threshold.
    pub pairwise_above_threshold: usize,
    /// Min pairwise Tanimoto.
    pub tanimoto_min: f32,
    /// Max pairwise Tanimoto.
    pub tanimoto_max: f32,
    /// Avg pairwise Tanimoto.
    pub tanimoto_avg: f32,
    /// Avg centroid size (bits).
    pub avg_centroid_size: f32,
}

/// Causal phase report — causal pattern discovery stats from a single maintenance cycle.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct CausalPhaseReport {
    /// Number of raw record-level causal edges found.
    pub edges_found: usize,
    /// Number of explicit record-level causal edges found.
    pub explicit_edges_found: usize,
    /// Number of temporal record-level causal edges found.
    pub temporal_edges_found: usize,
    /// Namespaces scanned for temporal edge extraction.
    pub temporal_namespaces_scanned: usize,
    /// Pairwise temporal checks considered.
    pub temporal_pairs_considered: usize,
    /// Pairwise temporal checks skipped by the budgeting policy.
    pub temporal_pairs_skipped_by_budget: usize,
    /// Temporal edges skipped due to cap pressure.
    pub temporal_edges_capped: usize,
    /// Namespaces that hit the temporal edge cap.
    pub temporal_namespaces_hit_cap: usize,
    /// Total causal pattern candidates after aggregation.
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

/// Policy phase report — policy hint discovery stats from a single maintenance cycle.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct PolicyPhaseReport {
    /// Number of causal seeds considered.
    pub seeds_found: usize,
    /// Total policy hints after generation.
    pub hints_found: usize,
    /// Hints that reached Stable state.
    pub stable_hints: usize,
    /// Hints suppressed by conflict.
    pub suppressed_hints: usize,
    /// Hints that were Rejected.
    pub rejected_hints: usize,
    /// Average policy_strength across all hints.
    pub avg_policy_strength: f32,
}

/// Per-phase timing in milliseconds.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct PhaseTimings {
    /// Phase 0: Level fix (ms).
    pub level_fix_ms: f64,
    /// Phase 1: Decay (ms).
    pub decay_ms: f64,
    /// Phase 2: Reflect (ms).
    pub reflect_ms: f64,
    /// Phase 2.5: Epistemic update (ms).
    pub epistemic_ms: f64,
    /// Phase 3: Insights (ms).
    pub insights_ms: f64,
    /// SDR lookup build (ms) — shared by belief/concept/causal.
    pub sdr_build_ms: f64,
    /// Phase 3.5: Belief update (ms).
    pub belief_ms: f64,
    /// Phase 3.7: Concept discovery (ms).
    pub concept_ms: f64,
    /// Phase 3.8: Causal discovery (ms).
    pub causal_ms: f64,
    /// Phase 3.9: Policy discovery (ms).
    pub policy_ms: f64,
    /// Phase 4: Consolidation (ms).
    pub consolidation_ms: f64,
    /// Phase 5: Cross-connections (ms).
    pub cross_connections_ms: f64,
    /// Phase 6+7: Tasks + archival (ms).
    pub tasks_archival_ms: f64,
    /// Total cycle time (ms).
    pub total_ms: f64,
}

/// Per-layer identity stability counters.
/// Tracks how many entities survived, appeared, or disappeared between cycles.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct LayerStability {
    // ── Belief layer ──
    /// Belief IDs retained from previous cycle.
    pub belief_retained: usize,
    /// New belief IDs this cycle.
    pub belief_new: usize,
    /// Belief IDs dropped since previous cycle.
    pub belief_dropped: usize,
    /// Belief identity churn rate = (new + dropped) / max(total, 1).
    pub belief_churn: f32,

    // ── Concept layer ──
    pub concept_retained: usize,
    pub concept_new: usize,
    pub concept_dropped: usize,
    pub concept_churn: f32,

    // ── Causal layer ──
    pub causal_retained: usize,
    pub causal_new: usize,
    pub causal_dropped: usize,
    pub causal_churn: f32,

    // ── Policy layer ──
    pub policy_retained: usize,
    pub policy_new: usize,
    pub policy_dropped: usize,
    pub policy_churn: f32,
}

/// Maintenance hot-spot accounting for scalability visibility.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct MaintenanceHotspots {
    /// Records present at cycle start.
    pub records_before_cycle: usize,
    /// Records present after archival/consolidation finishes.
    pub records_after_cycle: usize,
    /// Number of records cloned into the belief/concept/causal snapshot.
    pub belief_snapshot_records: usize,
    /// Total source text bytes processed during SDR rebuild.
    pub sdr_source_bytes: usize,
    /// Number of SDR vectors built this cycle.
    pub sdr_vectors_built: usize,
    /// Number of SDR vectors computed from scratch this cycle.
    pub sdr_vectors_computed: usize,
    /// Number of SDR vectors reused from cache this cycle.
    pub sdr_vectors_reused: usize,
    /// Active beliefs after the belief phase.
    pub belief_total_beliefs: usize,
    /// Active hypotheses after the belief phase.
    pub belief_total_hypotheses: usize,
    /// Concept pairwise centroid comparisons this cycle.
    pub concept_pairwise_comparisons: usize,
    /// Concept partitions with multiple seeds.
    pub concept_partitions_with_multiple_seeds: usize,
    /// Causal record-level edges evaluated this cycle.
    pub causal_edges_found: usize,
    /// Explicit causal edges found this cycle.
    pub causal_explicit_edges_found: usize,
    /// Temporal causal edges found this cycle.
    pub causal_temporal_edges_found: usize,
    /// Namespaces scanned for temporal causal extraction.
    pub causal_temporal_namespaces_scanned: usize,
    /// Pairwise temporal causal checks considered.
    pub causal_temporal_pairs_considered: usize,
    /// Pairwise temporal causal checks skipped by budget.
    pub causal_temporal_pairs_skipped_by_budget: usize,
    /// Temporal causal edges suppressed by per-namespace cap.
    pub causal_temporal_edges_capped: usize,
    /// Namespaces that hit the temporal causal cap.
    pub causal_temporal_namespaces_hit_cap: usize,
    /// Policy seeds considered this cycle.
    pub policy_seeds_found: usize,
    /// Cross-connections discovered this cycle.
    pub cross_connections_found: usize,
    /// Scheduled task reminders emitted this cycle.
    pub task_reminders_found: usize,
    /// Name of the dominant timing phase in this cycle.
    pub dominant_phase: String,
    /// Time spent in the dominant phase.
    pub dominant_phase_ms: f64,
    /// Share of total cycle time spent in the dominant phase.
    pub dominant_phase_share: f64,
}

/// Runtime telemetry for the bounded concept inspection surface.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct ConceptSurfaceTelemetry {
    /// Current runtime surface mode: Off, Inspect, or Limited.
    pub mode: String,
    /// Number of surfaced concepts currently eligible for inspection.
    pub surfaced_concepts_available: usize,
    /// Number of namespaces represented in the current surfaced concept set.
    pub surfaced_namespaces: usize,
    /// Global surfaced-concept API calls observed since the previous maintenance cycle.
    pub global_calls_since_last_cycle: u64,
    /// Namespace-scoped surfaced-concept API calls observed since the previous maintenance cycle.
    pub namespace_calls_since_last_cycle: u64,
    /// Per-record annotation API calls observed since the previous maintenance cycle.
    pub record_calls_since_last_cycle: u64,
    /// Total surfaced concepts returned across all surface APIs since the previous maintenance cycle.
    pub concepts_returned_since_last_cycle: u64,
    /// Total per-record concept annotations returned since the previous maintenance cycle.
    pub record_annotations_returned_since_last_cycle: u64,
}

/// Full maintenance cycle report.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone)]
pub struct MaintenanceReport {
    pub timestamp: String,
    pub decay: DecayReport,
    pub reflect: ReflectReport,
    pub epistemic: EpistemicPhaseReport,
    pub insights_found: usize,
    pub belief: BeliefPhaseReport,
    pub concept: ConceptPhaseReport,
    pub causal: CausalPhaseReport,
    pub policy: PolicyPhaseReport,
    pub consolidation: ConsolidationReport,
    pub cross_connections: usize,
    pub task_reminders: Vec<String>,
    pub records_archived: usize,
    pub total_records: usize,
    /// Per-phase timing breakdown.
    pub timings: PhaseTimings,
    /// Cross-cycle identity stability.
    pub stability: LayerStability,
    /// Audit/telemetry for the bounded concept inspection surface.
    pub concept_surface: ConceptSurfaceTelemetry,
    /// Scalability-oriented load and hot-spot accounting.
    pub hotspots: MaintenanceHotspots,
}

// ── Phase Implementations ──

/// Phase 0: Level fix — downgrade records incorrectly at IDENTITY.
pub fn fix_memory_levels(
    records: &mut HashMap<String, Record>,
    taxonomy: &TagTaxonomy,
) -> HashMap<String, usize> {
    let mut stats = HashMap::new();
    stats.insert("total_identity".into(), 0);
    stats.insert("downgraded".into(), 0);
    stats.insert("kept".into(), 0);

    let identity_ids: Vec<String> = records
        .values()
        .filter(|r| r.level == Level::Identity)
        .map(|r| r.id.clone())
        .collect();

    *stats.get_mut("total_identity").unwrap() = identity_ids.len();

    for id in identity_ids {
        let rec = match records.get_mut(&id) {
            Some(r) => r,
            None => continue,
        };

        let rec_tags: HashSet<&str> = rec.tags.iter().map(|s| s.as_str()).collect();

        // Keep if has identity-specific tags
        if taxonomy
            .identity_tags
            .iter()
            .any(|t| rec_tags.contains(t.as_str()))
        {
            *stats.get_mut("kept").unwrap() += 1;
            continue;
        }

        // Downgrade if has non-identity tags
        if taxonomy
            .non_identity_tags
            .iter()
            .any(|t| rec_tags.contains(t.as_str()))
        {
            rec.level = Level::Domain;
            *stats.get_mut("downgraded").unwrap() += 1;
            continue;
        }

        // High activation = earned IDENTITY, keep
        if rec.activation_count >= 20 && rec.strength >= 0.9 {
            *stats.get_mut("kept").unwrap() += 1;
            continue;
        }

        // Default: downgrade
        rec.level = Level::Domain;
        *stats.get_mut("downgraded").unwrap() += 1;
    }

    stats
}

/// Phase 2.5: Epistemic update — derive support/conflict mass from local structure.
///
/// Conservative heuristics:
/// - supporting neighbors: same namespace, semantically aligned, and connected or strongly tag-overlapping
/// - conflicting neighbors: explicit contradiction markers, conflict-like relations, or WORKING/IDENTITY tag clashes
pub fn update_epistemic_state(records: &mut HashMap<String, Record>) -> EpistemicPhaseReport {
    let live_ids: Vec<String> = records
        .values()
        .filter(|r| r.is_alive())
        .map(|r| r.id.clone())
        .collect();

    let mut tag_groups: HashMap<String, Vec<String>> = HashMap::new();
    for rec in records.values().filter(|r| r.is_alive()) {
        for tag in &rec.tags {
            tag_groups
                .entry(tag.clone())
                .or_default()
                .push(rec.id.clone());
        }
    }

    let mut updates: Vec<(String, u32, u32)> = Vec::with_capacity(live_ids.len());
    let mut report = EpistemicPhaseReport::default();

    for rid in live_ids {
        let Some(rec) = records.get(&rid) else {
            continue;
        };

        let mut neighbors: HashSet<String> = rec.connections.keys().cloned().collect();
        for tag in &rec.tags {
            if let Some(ids) = tag_groups.get(tag) {
                for nid in ids {
                    if nid != &rid {
                        neighbors.insert(nid.clone());
                    }
                }
            }
        }

        let namespace = rec.namespace.clone();
        let level = rec.level;
        let semantic_type = rec.semantic_type.clone();
        let rec_tags: HashSet<&str> = rec.tags.iter().map(|t| t.as_str()).collect();

        let mut confirming = 0u32;
        let mut conflicting = 0u32;

        for nid in neighbors {
            let Some(other) = records.get(&nid) else {
                continue;
            };
            if !other.is_alive() || other.namespace != namespace {
                continue;
            }

            let other_tags: HashSet<&str> = other.tags.iter().map(|t| t.as_str()).collect();
            let shared_tags = rec_tags.intersection(&other_tags).count();
            let relation = rec
                .connection_type(&nid)
                .or_else(|| other.connection_type(&rid));
            let connection_weight = rec
                .connections
                .get(&nid)
                .copied()
                .or_else(|| other.connections.get(&rid).copied())
                .unwrap_or(0.0);
            let connected = connection_weight >= 0.10 || relation.is_some();

            if !connected && shared_tags == 0 {
                continue;
            }

            let explicit_conflict_relation =
                relation.is_some_and(|rel| rel.contains("conflict") || rel.contains("contradict"));
            // Contradiction propagation requires strong evidence:
            // at least 2 shared tags. Auto-connections alone are too noisy
            // (a record sharing 1 tag like "safety" shouldn't be pulled
            // into conflict with an unrelated contradiction).
            let contradiction_pair = (semantic_type == "contradiction"
                || other.semantic_type == "contradiction")
                && shared_tags >= 2;
            let level_conflict = shared_tags > 0
                && matches!(
                    (level, other.level),
                    (Level::Working, Level::Identity) | (Level::Identity, Level::Working)
                );

            if explicit_conflict_relation || contradiction_pair || level_conflict {
                conflicting += 1;
                continue;
            }

            let reinforcing_relation =
                matches!(relation, Some("causal" | "associative" | "coactivation"));
            let shared_semantic = semantic_type == other.semantic_type;
            if (shared_semantic && shared_tags > 0)
                || (reinforcing_relation && connected)
                || shared_tags >= 2
            {
                confirming += 1;
            }
        }

        report.total_support_links += confirming as usize;
        report.total_conflict_links += conflicting as usize;
        updates.push((rid, confirming, conflicting));
    }

    for (rid, confirming, conflicting) in updates {
        if let Some(rec) = records.get_mut(&rid) {
            let prev_support = rec.support_mass;
            let prev_conflict = rec.conflict_mass;
            let prev_volatility = rec.volatility;
            rec.update_epistemic_signals(confirming, conflicting);
            if prev_support != rec.support_mass
                || prev_conflict != rec.conflict_mass
                || (prev_volatility - rec.volatility).abs() > f32::EPSILON
            {
                report.updated_records += 1;
            }
            if rec.volatility >= 0.05 {
                report.volatile_records += 1;
            }
        }
    }

    report
}

/// Phase 2: Guarded reflect — prevents overpromotion.
///
/// Saves original levels, runs reflect, restores violations:
/// 1. WORKING records are NEVER promoted
/// 2. Records can't reach IDENTITY without 20+ activations AND 0.9+ strength
pub fn guarded_reflect(
    records: &mut HashMap<String, Record>,
    _taxonomy: &TagTaxonomy,
) -> ReflectReport {
    // Save original levels for non-IDENTITY records
    let original_levels: HashMap<String, Level> = records
        .iter()
        .filter(|(_, r)| r.level != Level::Identity)
        .map(|(id, r)| (id.clone(), r.level))
        .collect();

    // Run reflect logic: promote candidates
    let mut promoted: usize = 0;
    let promotable: Vec<String> = records
        .values()
        .filter(|r| r.can_promote())
        .map(|r| r.id.clone())
        .collect();

    for id in &promotable {
        if let Some(rec) = records.get_mut(id) {
            if rec.promote() {
                promoted += 1;
            }
        }
    }

    // Archive dead
    let dead: Vec<String> = records
        .values()
        .filter(|r| !r.is_alive())
        .map(|r| r.id.clone())
        .collect();
    let archived = dead.len();
    for id in &dead {
        records.remove(id);
    }

    // Restore overpromotions
    let mut restored: usize = 0;
    for (rec_id, orig_level) in &original_levels {
        let rec = match records.get_mut(rec_id) {
            Some(r) => r,
            None => continue,
        };

        let should_restore =
            // Rule 1: WORKING must stay WORKING
            (*orig_level == Level::Working && rec.level != Level::Working)
            // Rule 2: No IDENTITY without threshold
            || (rec.level == Level::Identity
                && (rec.activation_count < 20 || rec.strength < 0.9));

        if should_restore {
            rec.level = *orig_level;
            restored += 1;
        }
    }

    if restored > 0 {
        tracing::info!(restored, "Overpromotion guard: restored records");
    }

    ReflectReport {
        promoted: promoted.saturating_sub(restored),
        archived,
    }
}

/// Truncate a string to at most `max_bytes` on a valid UTF-8 char boundary.
fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Phase 5: Knowledge synthesis — 2-hop graph walk for cross-connections.
pub fn discover_cross_connections(
    records: &HashMap<String, Record>,
    max_discoveries: usize,
) -> Vec<String> {
    let mut discoveries = Vec::new();

    // Sample connected records
    let sample: Vec<&Record> = records
        .values()
        .filter(|r| !r.connections.is_empty() && r.is_alive())
        .take(10)
        .collect();

    for rec in sample {
        // 1-hop neighbors
        for neighbor_id in rec.connections.keys() {
            if let Some(neighbor) = records.get(neighbor_id) {
                // Namespace guard: skip cross-namespace connections
                if neighbor.namespace != rec.namespace {
                    continue;
                }
                // 2-hop: neighbor's connections
                for hop2_id in neighbor.connections.keys() {
                    if hop2_id != &rec.id
                        && !rec.connections.contains_key(hop2_id)
                        && discoveries.len() < max_discoveries
                    {
                        if let Some(hop2) = records.get(hop2_id) {
                            // Namespace guard: skip cross-namespace 2-hop
                            if hop2.namespace != rec.namespace {
                                continue;
                            }
                            discoveries.push(format!(
                                "{} ← {} → {} (indirect connection)",
                                truncate_utf8(&rec.content, 50),
                                truncate_utf8(&neighbor.content, 30),
                                truncate_utf8(&hop2.content, 50),
                            ));
                        }
                    }
                }
            }
        }

        if discoveries.len() >= max_discoveries {
            break;
        }
    }

    discoveries
}

/// Phase 6: Scheduled task check — find tasks due today or tomorrow.
pub fn check_scheduled_tasks(records: &HashMap<String, Record>, task_tag: &str) -> Vec<String> {
    let now = chrono::Utc::now();
    let tomorrow = now + chrono::Duration::days(1);
    let mut reminders = Vec::new();

    for rec in records.values() {
        if !rec.tags.contains(&task_tag.to_string()) {
            continue;
        }

        let status = rec.metadata.get("status").map(|s| s.as_str()).unwrap_or("");
        if status != "active" {
            continue;
        }

        let due_str = match rec.metadata.get("due_date") {
            Some(s) => s,
            None => continue,
        };

        let due_date = match chrono::DateTime::parse_from_rfc3339(due_str) {
            Ok(dt) => dt,
            Err(_) => {
                // Try ISO date without timezone
                match chrono::NaiveDate::parse_from_str(due_str, "%Y-%m-%d") {
                    Ok(d) => d.and_hms_opt(0, 0, 0).unwrap().and_utc().fixed_offset(),
                    Err(_) => continue,
                }
            }
        };

        let due_naive = due_date.date_naive();
        let now_naive = now.date_naive();
        let tomorrow_naive = tomorrow.date_naive();

        if due_naive > tomorrow_naive {
            continue;
        }

        let description = rec.metadata.get("description").unwrap_or(&rec.content);

        if due_naive == now_naive {
            reminders.push(format!("Due today: {}", description));
        } else if due_naive == tomorrow_naive {
            reminders.push(format!("Due tomorrow: {}", description));
        } else if due_naive < now_naive {
            reminders.push(format!("Overdue: {} (was due {})", description, due_naive));
        }
    }

    reminders
}

/// Phase 7: Archive old transient records.
pub fn archive_old_records(
    records: &mut HashMap<String, Record>,
    config: &MaintenanceConfig,
    taxonomy: &TagTaxonomy,
) -> usize {
    let now = chrono::Utc::now();
    let mut total_archived = 0;

    // Strategy 1: Age-based deletion
    for rule in &config.archival_rules {
        let mut matching: Vec<(String, String)> = records
            .values()
            .filter(|r| r.tags.contains(&rule.tag) && r.is_alive())
            .map(|r| {
                let ts = r
                    .metadata
                    .get("timestamp")
                    .or_else(|| r.metadata.get("created_at"))
                    .cloned()
                    .unwrap_or_default();
                (r.id.clone(), ts)
            })
            .collect();

        if matching.len() <= rule.keep_recent {
            continue;
        }

        // Sort by timestamp descending (newest first)
        matching.sort_by(|a, b| b.1.cmp(&a.1));

        let cutoff = (now - chrono::Duration::days(rule.max_age_days as i64)).to_rfc3339();

        // Skip keep_recent newest records
        let candidates = &matching[rule.keep_recent..];
        for (id, ts) in candidates {
            if ts.is_empty() || ts.as_str() < cutoff.as_str() {
                // Check archive protection
                if let Some(rec) = records.get(id) {
                    if !crate::guards::is_archive_protected(&rec.tags, taxonomy) {
                        records.remove(id);
                        total_archived += 1;
                    }
                }
            }
        }
    }

    // Strategy 2: Completion-based deletion
    for rule in &config.completed_archival_rules {
        let cutoff = (now - chrono::Duration::days(rule.max_age_days as i64)).to_rfc3339();

        let to_delete: Vec<String> = records
            .values()
            .filter(|r| {
                r.tags.contains(&rule.tag)
                    && matches!(
                        r.metadata.get("status").map(|s| s.as_str()),
                        Some("completed" | "done" | "cancelled" | "archived")
                    )
            })
            .filter(|r| {
                let completed_at = r
                    .metadata
                    .get("completed_at")
                    .or_else(|| r.metadata.get("timestamp"))
                    .or_else(|| r.metadata.get("created_at"))
                    .map(|s| s.as_str())
                    .unwrap_or("");
                completed_at.is_empty() || completed_at < cutoff.as_str()
            })
            .map(|r| r.id.clone())
            .collect();

        for id in &to_delete {
            records.remove(id);
            total_archived += 1;
        }
    }

    total_archived
}

// ── Background Brain Controller ──

/// BackgroundBrain — spawns daemon thread for periodic maintenance.
pub struct BackgroundBrain {
    /// Background loop running flag.
    running: Arc<AtomicBool>,
    /// Background thread handle.
    thread: Option<JoinHandle<()>>,
    /// Cycle count (for periodic tasks).
    pub cycle_count: AtomicU64,
    /// Transient insights from last cycle.
    pub last_insights: RwLock<Vec<String>>,
    /// Transient cross-connections from last cycle.
    pub last_cross_connections: RwLock<Vec<String>>,
}

impl BackgroundBrain {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            thread: None,
            cycle_count: AtomicU64::new(0),
            last_insights: RwLock::new(Vec::new()),
            last_cross_connections: RwLock::new(Vec::new()),
        }
    }

    /// Is the background loop currently running?
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Stop the background loop.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }

    /// Get current cycle count.
    pub fn cycles(&self) -> u64 {
        self.cycle_count.load(Ordering::Relaxed)
    }
}

impl Default for BackgroundBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BackgroundBrain {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_memory_levels() {
        let taxonomy = TagTaxonomy::default();
        let mut records = HashMap::new();

        // Record with non-identity tag at IDENTITY → should be downgraded
        let mut r1 = Record::new("session stuff".into(), Level::Identity);
        r1.tags.push("session-summary".into());
        records.insert(r1.id.clone(), r1);

        // Record with identity tag at IDENTITY → should stay
        let mut r2 = Record::new("user profile".into(), Level::Identity);
        r2.tags.push("user-profile".into());
        records.insert(r2.id.clone(), r2);

        let stats = fix_memory_levels(&mut records, &taxonomy);
        assert_eq!(*stats.get("downgraded").unwrap(), 1);
        assert_eq!(*stats.get("kept").unwrap(), 1);
    }

    #[test]
    fn test_guarded_reflect_prevents_working_promotion() {
        let taxonomy = TagTaxonomy::default();
        let mut records = HashMap::new();

        let mut r = Record::new("temporary thought".into(), Level::Working);
        r.activation_count = 10;
        r.strength = 0.9;
        let rid = r.id.clone();
        records.insert(rid.clone(), r);

        let _report = guarded_reflect(&mut records, &taxonomy);

        // WORKING should NOT have been promoted
        if let Some(rec) = records.get(&rid) {
            assert_eq!(rec.level, Level::Working);
        }
    }

    #[test]
    fn test_archival_rules() {
        let config = MaintenanceConfig::default();
        let taxonomy = TagTaxonomy::default();
        let mut records = HashMap::new();

        // Create an old cache record
        let mut r = Record::new("cached result".into(), Level::Working);
        r.tags.push("web-search-cache".into());
        r.metadata
            .insert("timestamp".into(), "2020-01-01T00:00:00Z".into());
        records.insert(r.id.clone(), r);

        let archived = archive_old_records(&mut records, &config, &taxonomy);
        assert_eq!(archived, 1);
        assert!(records.is_empty());
    }

    #[test]
    fn test_default_maintenance_config() {
        let config = MaintenanceConfig::default();
        assert!(config.decay_enabled);
        assert!(config.reflect_enabled);
        assert_eq!(config.level_fix_interval, 10);
        assert!(!config.archival_rules.is_empty());
    }

    #[test]
    fn test_update_epistemic_state_support_and_conflict() {
        let mut records = HashMap::new();

        let mut r1 = Record::new("Deploy to staging before production".into(), Level::Domain);
        r1.tags = vec!["deploy".into(), "safety".into()];
        r1.semantic_type = "decision".into();

        let mut r2 = Record::new("Always use staging for safe deploys".into(), Level::Domain);
        r2.tags = vec!["deploy".into(), "safety".into()];
        r2.semantic_type = "decision".into();
        r1.add_typed_connection(&r2.id, 0.7, "coactivation");
        r2.add_typed_connection(&r1.id, 0.7, "coactivation");

        let mut r3 = Record::new("Skip staging for production deploys".into(), Level::Working);
        r3.tags = vec!["deploy".into(), "safety".into()];
        r3.semantic_type = "contradiction".into();

        let id1 = r1.id.clone();
        let id2 = r2.id.clone();
        let id3 = r3.id.clone();
        records.insert(id1.clone(), r1);
        records.insert(id2.clone(), r2);
        records.insert(id3.clone(), r3);

        let report = update_epistemic_state(&mut records);

        assert!(report.updated_records >= 3);
        assert!(report.total_support_links >= 2);
        assert!(report.total_conflict_links >= 2);
        assert!(records.get(&id1).unwrap().support_mass >= 1);
        assert!(records.get(&id1).unwrap().conflict_mass >= 1);
        assert!(records.get(&id3).unwrap().conflict_mass >= 1);
    }

    #[test]
    fn test_update_epistemic_state_tracks_volatility_on_change() {
        let mut records = HashMap::new();

        let mut r1 = Record::new(
            "User prefers dark mode in the editor".into(),
            Level::Identity,
        );
        r1.tags = vec!["ui".into(), "theme".into()];
        r1.semantic_type = "preference".into();

        let mut r2 = Record::new("Dark theme is used for coding".into(), Level::Working);
        r2.tags = vec!["ui".into(), "theme".into()];
        r2.semantic_type = "preference".into();

        let id1 = r1.id.clone();
        let id2 = r2.id.clone();
        records.insert(id1.clone(), r1);
        records.insert(id2.clone(), r2);

        let _ = update_epistemic_state(&mut records);
        let first_vol = records.get(&id1).unwrap().volatility;

        let mut r3 = Record::new(
            "User rejects dark mode in the editor".into(),
            Level::Working,
        );
        r3.tags = vec!["ui".into(), "theme".into()];
        r3.semantic_type = "contradiction".into();
        let id3 = r3.id.clone();
        records.insert(id3, r3);

        let report = update_epistemic_state(&mut records);
        let second_vol = records.get(&id1).unwrap().volatility;

        assert!(report.total_conflict_links > 0);
        assert!(second_vol > first_vol);
    }

    #[test]
    fn test_truncate_utf8() {
        // ASCII — exact boundary
        assert_eq!(truncate_utf8("hello world", 5), "hello");
        // Cyrillic (2 bytes per char) — cut inside char
        let cyrillic = "Обговорено"; // 10 chars × 2 bytes = 20 bytes
        assert_eq!(truncate_utf8(cyrillic, 5), "Об"); // 5 → backs up to 4 (2 full chars)
                                                      // Short string — returned as-is
        assert_eq!(truncate_utf8("hi", 50), "hi");
        // Empty
        assert_eq!(truncate_utf8("", 10), "");
    }

    #[test]
    fn test_discover_cross_connections_cyrillic() {
        // Regression: content with Cyrillic must not panic on truncation
        let mut records = HashMap::new();

        let long_ukr = "Обговорено та заплановано триденне SEO-дослідження для проєкту. До пам'яті збережено графік завдань.";
        let mut r1 = Record::new(long_ukr.into(), Level::Domain);
        let mut r2 = Record::new(
            "Зв'язаний запис із кириличним текстом номер два".into(),
            Level::Domain,
        );
        let mut r3 = Record::new(
            "Третій запис — ще один кириличний текст для тесту".into(),
            Level::Domain,
        );

        let id1 = r1.id.clone();
        let id2 = r2.id.clone();
        let id3 = r3.id.clone();

        // r1 → r2 → r3 (2-hop)
        r1.connections.insert(id2.clone(), 1.0);
        r2.connections.insert(id1.clone(), 1.0);
        r2.connections.insert(id3.clone(), 1.0);
        r3.connections.insert(id2.clone(), 1.0);

        records.insert(id1, r1);
        records.insert(id2, r2);
        records.insert(id3, r3);

        // Should NOT panic — this was the production crash
        let discoveries = discover_cross_connections(&records, 5);
        assert!(!discoveries.is_empty());
        for d in &discoveries {
            assert!(d.contains("indirect connection"));
        }
    }
}
