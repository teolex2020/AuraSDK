//! MaintenanceService — internal orchestration helpers for maintenance cycles.
//!
//! This starts the extraction of `Aura::run_maintenance()` into a dedicated
//! service layer without changing the phase logic itself.

use std::sync::atomic::Ordering;

use parking_lot::RwLock;
use rayon;
use std::collections::{HashMap, HashSet};

use crate::background_brain;
use crate::belief::{BeliefEngine, BeliefStore, SdrLookup};
use crate::causal::{CausalEngine, CausalStore};
use crate::cognitive_store::CognitiveStore;
use crate::concept::{self, ConceptEngine, ConceptStore, ConceptSurfaceMode};
use crate::consolidation;
use crate::insights;
use crate::ngram::NGramIndex;
use crate::policy::{PolicyEngine, PolicyStore};
use crate::record::Record;
use crate::sdr::SDRInterpreter;
use crate::trust::TagTaxonomy;

pub(crate) struct InitialMaintenancePhaseResult {
    pub(crate) total_records: usize,
    pub(crate) decay: background_brain::DecayReport,
    pub(crate) reflect: background_brain::ReflectReport,
    pub(crate) epistemic: background_brain::EpistemicPhaseReport,
    pub(crate) insights_found: usize,
}

pub(crate) struct DiscoveryPhaseResult {
    pub(crate) belief: background_brain::BeliefPhaseReport,
    pub(crate) concept: background_brain::ConceptPhaseReport,
    pub(crate) causal: background_brain::CausalPhaseReport,
    pub(crate) policy: background_brain::PolicyPhaseReport,
    pub(crate) feedback: background_brain::FeedbackAuditReport,
}

pub(crate) struct PostDiscoveryPhaseResult {
    pub(crate) consolidation: background_brain::ConsolidationReport,
    pub(crate) cross_connections: usize,
    pub(crate) task_reminders: Vec<String>,
    pub(crate) records_archived: usize,
}

pub(crate) struct ConceptSurfaceCounters<'a> {
    pub(crate) global_calls: &'a std::sync::atomic::AtomicU64,
    pub(crate) namespace_calls: &'a std::sync::atomic::AtomicU64,
    pub(crate) record_calls: &'a std::sync::atomic::AtomicU64,
    pub(crate) concepts_returned: &'a std::sync::atomic::AtomicU64,
    pub(crate) record_annotations_returned: &'a std::sync::atomic::AtomicU64,
}

pub(crate) struct MaintenanceService;

impl MaintenanceService {
    pub(crate) const MAINTENANCE_TREND_LIMIT: usize = 32;
    pub(crate) const REFLECTION_SUMMARY_LIMIT: usize = 16;
    const REFLECTION_FINDING_LIMIT: usize = 6;
    const REFLECTION_KIND_LIMIT: usize = 12;
    const REFLECTION_NAMESPACE_LIMIT: usize = 8;

    pub(crate) fn run_initial_phases(
        records: &mut HashMap<String, Record>,
        config: &background_brain::MaintenanceConfig,
        taxonomy: &TagTaxonomy,
        cognitive_store: &CognitiveStore,
        cycle: u64,
        timings: &mut background_brain::PhaseTimings,
        hotspots: &mut background_brain::MaintenanceHotspots,
    ) -> InitialMaintenancePhaseResult {
        let total_records = records.len();
        hotspots.records_before_cycle = total_records;

        let t0 = std::time::Instant::now();
        if cycle % config.level_fix_interval == 0 {
            background_brain::fix_memory_levels(records, taxonomy);
        }
        timings.level_fix_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let decay = if config.decay_enabled {
            let mut decayed = 0;
            let mut to_archive = Vec::new();

            for rec in records.values_mut() {
                rec.apply_decay();
                decayed += 1;
                if !rec.is_alive() {
                    to_archive.push(rec.id.clone());
                }
            }

            for rec in records.values_mut() {
                let weak: Vec<String> = rec
                    .connections
                    .iter()
                    .filter(|(_, w)| **w < 0.05)
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in &weak {
                    rec.connections.remove(id);
                    rec.connection_types.remove(id);
                }
                for w in rec.connections.values_mut() {
                    *w *= 0.99;
                }
            }

            let archived = to_archive.len();
            for id in &to_archive {
                records.remove(id);
                let _ = cognitive_store.append_delete(id);
            }

            background_brain::DecayReport { decayed, archived }
        } else {
            background_brain::DecayReport::default()
        };
        timings.decay_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = std::time::Instant::now();
        let reflect = if config.reflect_enabled {
            background_brain::guarded_reflect(records, taxonomy)
        } else {
            background_brain::ReflectReport::default()
        };
        timings.reflect_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let t25 = std::time::Instant::now();
        let epistemic = background_brain::update_epistemic_state(records);
        timings.epistemic_ms = t25.elapsed().as_secs_f64() * 1000.0;

        let t3 = std::time::Instant::now();
        let insights_found = if config.insights_enabled {
            insights::detect_all(records).len()
        } else {
            0
        };
        timings.insights_ms = t3.elapsed().as_secs_f64() * 1000.0;

        InitialMaintenancePhaseResult {
            total_records,
            decay,
            reflect,
            epistemic,
            insights_found,
        }
    }

    pub(crate) fn build_sdr_lookup(
        sdr: &SDRInterpreter,
        sdr_lookup_cache: &RwLock<HashMap<String, Vec<u16>>>,
        belief_snapshot: &HashMap<String, Record>,
        timings: &mut background_brain::PhaseTimings,
        hotspots: &mut background_brain::MaintenanceHotspots,
    ) -> SdrLookup {
        hotspots.belief_snapshot_records = belief_snapshot.len();
        hotspots.sdr_source_bytes = belief_snapshot.values().map(|rec| rec.content.len()).sum();

        let t_sdr = std::time::Instant::now();
        let mut computed = 0usize;
        let mut reused = 0usize;
        let sdr_lookup = {
            let mut cache = sdr_lookup_cache.write();
            cache.retain(|rid, _| belief_snapshot.contains_key(rid));

            let mut lookup = HashMap::with_capacity(belief_snapshot.len());
            for (rid, rec) in belief_snapshot {
                if let Some(existing) = cache.get(rid) {
                    lookup.insert(rid.clone(), existing.clone());
                    reused += 1;
                } else {
                    let sdr_vec = sdr.text_to_sdr(&rec.content, false);
                    cache.insert(rid.clone(), sdr_vec.clone());
                    lookup.insert(rid.clone(), sdr_vec);
                    computed += 1;
                }
            }
            lookup
        };

        hotspots.sdr_vectors_built = sdr_lookup.len();
        hotspots.sdr_vectors_computed = computed;
        hotspots.sdr_vectors_reused = reused;
        timings.sdr_build_ms = t_sdr.elapsed().as_secs_f64() * 1000.0;

        sdr_lookup
    }

    pub(crate) fn compute_layer_stability(
        belief_eng: &BeliefEngine,
        concept_eng: &ConceptEngine,
        causal_eng: &CausalEngine,
        policy_eng: &PolicyEngine,
        prev_belief_keys: &RwLock<HashSet<String>>,
        prev_concept_keys: &RwLock<HashSet<String>>,
        prev_causal_keys: &RwLock<HashSet<String>>,
        prev_policy_keys: &RwLock<HashSet<String>>,
    ) -> background_brain::LayerStability {
        let cur_belief: HashSet<String> = belief_eng.key_index.keys().cloned().collect();
        let cur_concept: HashSet<String> = concept_eng.key_index.keys().cloned().collect();
        let cur_causal: HashSet<String> = causal_eng.key_index.keys().cloned().collect();
        let cur_policy: HashSet<String> = policy_eng.key_index.keys().cloned().collect();

        let prev_b = prev_belief_keys.read();
        let prev_c = prev_concept_keys.read();
        let prev_ca = prev_causal_keys.read();
        let prev_p = prev_policy_keys.read();

        let b_retained = cur_belief.intersection(&prev_b).count();
        let b_new = cur_belief.len() - b_retained;
        let b_dropped = prev_b.len() - b_retained;
        let b_total = cur_belief.len().max(1);

        let c_retained = cur_concept.intersection(&prev_c).count();
        let c_new = cur_concept.len() - c_retained;
        let c_dropped = prev_c.len() - c_retained;
        let c_total = cur_concept.len().max(1);

        let ca_retained = cur_causal.intersection(&prev_ca).count();
        let ca_new = cur_causal.len() - ca_retained;
        let ca_dropped = prev_ca.len() - ca_retained;
        let ca_total = cur_causal.len().max(1);

        let p_retained = cur_policy.intersection(&prev_p).count();
        let p_new = cur_policy.len() - p_retained;
        let p_dropped = prev_p.len() - p_retained;
        let p_total = cur_policy.len().max(1);

        drop(prev_b);
        drop(prev_c);
        drop(prev_ca);
        drop(prev_p);

        *prev_belief_keys.write() = cur_belief;
        *prev_concept_keys.write() = cur_concept;
        *prev_causal_keys.write() = cur_causal;
        *prev_policy_keys.write() = cur_policy;

        background_brain::LayerStability {
            belief_retained: b_retained,
            belief_new: b_new,
            belief_dropped: b_dropped,
            belief_churn: (b_new + b_dropped) as f32 / b_total as f32,
            concept_retained: c_retained,
            concept_new: c_new,
            concept_dropped: c_dropped,
            concept_churn: (c_new + c_dropped) as f32 / c_total as f32,
            causal_retained: ca_retained,
            causal_new: ca_new,
            causal_dropped: ca_dropped,
            causal_churn: (ca_new + ca_dropped) as f32 / ca_total as f32,
            policy_retained: p_retained,
            policy_new: p_new,
            policy_dropped: p_dropped,
            policy_churn: (p_new + p_dropped) as f32 / p_total as f32,
        }
    }

    pub(crate) fn run_discovery_phases(
        belief_engine: &RwLock<BeliefEngine>,
        belief_store: &BeliefStore,
        concept_engine: &RwLock<ConceptEngine>,
        concept_store: &ConceptStore,
        causal_engine: &RwLock<CausalEngine>,
        causal_store: &CausalStore,
        policy_engine: &RwLock<PolicyEngine>,
        policy_store: &PolicyStore,
        belief_snapshot: &HashMap<String, Record>,
        sdr_lookup: &SdrLookup,
        timings: &mut background_brain::PhaseTimings,
        hotspots: &mut background_brain::MaintenanceHotspots,
    ) -> DiscoveryPhaseResult {
        let t35 = std::time::Instant::now();
        let belief = {
            let mut engine = belief_engine.write();
            let br = engine.update_with_sdr(belief_snapshot, sdr_lookup);
            let _ = belief_store.save(&engine);
            background_brain::BeliefPhaseReport {
                beliefs_created: br.beliefs_created,
                beliefs_pruned: br.beliefs_pruned,
                revisions: br.revisions,
                resolved: br.resolved,
                unresolved: br.unresolved,
                total_beliefs: br.total_beliefs,
                total_hypotheses: br.total_hypotheses,
                churn_rate: br.churn_rate,
            }
        };
        hotspots.belief_total_beliefs = belief.total_beliefs;
        hotspots.belief_total_hypotheses = belief.total_hypotheses;
        timings.belief_ms = t35.elapsed().as_secs_f64() * 1000.0;

        // Phases 3.7 (concept) and 3.8 (causal) are independent — both only need a
        // read lock on belief_engine and write locks on their own engines.  Run them
        // in parallel via rayon::join to reduce wall-clock maintenance latency.
        // Timing is measured per branch; the longer branch dominates the wall clock.
        let t37_38 = std::time::Instant::now();
        let (concept_result, causal_result) = rayon::join(
            || {
                let t = std::time::Instant::now();
                let engine = belief_engine.read();
                let mut concept_eng = concept_engine.write();
                let cr = concept_eng.discover(&engine, belief_snapshot, sdr_lookup);
                let _ = concept_store.save(&concept_eng);
                let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
                let report = background_brain::ConceptPhaseReport {
                    seeds_found: cr.seeds_found,
                    candidates_found: cr.candidates_found,
                    stable_count: cr.stable_count,
                    rejected_count: cr.rejected_count,
                    avg_abstraction_score: cr.avg_abstraction_score,
                    centroids_built: cr.centroids_built,
                    partitions_with_multiple_seeds: cr.partitions_with_multiple_seeds,
                    multi_seed_partition_sizes: cr.multi_seed_partition_sizes,
                    cluster_sizes: cr.cluster_sizes,
                    clusters_with_multiple_beliefs: cr.clusters_with_multiple_beliefs,
                    largest_cluster_size: cr.largest_cluster_size,
                    pairwise_comparisons: cr.pairwise_comparisons,
                    pairwise_above_threshold: cr.pairwise_above_threshold,
                    tanimoto_min: cr.tanimoto_min,
                    tanimoto_max: cr.tanimoto_max,
                    tanimoto_avg: cr.tanimoto_avg,
                    avg_centroid_size: cr.avg_centroid_size,
                    seeds_capped: cr.seeds_capped,
                };
                (report, elapsed_ms)
            },
            || {
                let t = std::time::Instant::now();
                let engine = belief_engine.read();
                let mut causal_eng = causal_engine.write();
                let cr = causal_eng.discover(&engine, belief_snapshot, sdr_lookup);
                let _ = causal_store.save(&causal_eng);
                let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
                let report = background_brain::CausalPhaseReport {
                    skipped: cr.skipped,
                    edges_found: cr.edges_found,
                    explicit_edges_found: cr.explicit_edges_found,
                    temporal_edges_found: cr.temporal_edges_found,
                    temporal_namespaces_scanned: cr.temporal_namespaces_scanned,
                    temporal_pairs_considered: cr.temporal_pairs_considered,
                    temporal_pairs_skipped_by_budget: cr.temporal_pairs_skipped_by_budget,
                    temporal_edges_capped: cr.temporal_edges_capped,
                    temporal_namespaces_hit_cap: cr.temporal_namespaces_hit_cap,
                    candidates_found: cr.candidates_found,
                    patterns_meeting_support_gate: cr.patterns_meeting_support_gate,
                    patterns_meeting_repeated_window_gate: cr.patterns_meeting_repeated_window_gate,
                    patterns_meeting_counterfactual_gate: cr.patterns_meeting_counterfactual_gate,
                    patterns_blocked_by_evidence_gates: cr.patterns_blocked_by_evidence_gates,
                    patterns_blocked_by_counterfactual_gate: cr.patterns_blocked_by_counterfactual_gate,
                    stable_count: cr.stable_count,
                    rejected_count: cr.rejected_count,
                    avg_causal_strength: cr.avg_causal_strength,
                };
                (report, elapsed_ms)
            },
        );
        let (concept, concept_ms) = concept_result;
        let (causal, causal_ms) = causal_result;
        hotspots.concept_pairwise_comparisons = concept.pairwise_comparisons;
        hotspots.concept_partitions_with_multiple_seeds = concept.partitions_with_multiple_seeds;
        timings.concept_ms = concept_ms;
        hotspots.causal_edges_found = causal.edges_found;
        hotspots.causal_explicit_edges_found = causal.explicit_edges_found;
        hotspots.causal_temporal_edges_found = causal.temporal_edges_found;
        hotspots.causal_temporal_namespaces_scanned = causal.temporal_namespaces_scanned;
        hotspots.causal_temporal_pairs_considered = causal.temporal_pairs_considered;
        hotspots.causal_temporal_pairs_skipped_by_budget = causal.temporal_pairs_skipped_by_budget;
        hotspots.causal_temporal_edges_capped = causal.temporal_edges_capped;
        hotspots.causal_temporal_namespaces_hit_cap = causal.temporal_namespaces_hit_cap;
        timings.causal_ms = causal_ms;
        // Wall-clock time for the parallel pair is the longer of the two branches.
        let _ = t37_38; // individual branch timings are used instead

        let t39 = std::time::Instant::now();
        let policy = {
            let causal_eng = causal_engine.read();
            let concept_eng = concept_engine.read();
            let belief_eng = belief_engine.read();
            let mut policy_eng = policy_engine.write();
            let pr = policy_eng.discover(&causal_eng, &concept_eng, &belief_eng, belief_snapshot);
            let _ = policy_store.save(&policy_eng);
            background_brain::PolicyPhaseReport {
                seeds_found: pr.seeds_found,
                hints_found: pr.hints_found,
                stable_hints: pr.stable_hints,
                suppressed_hints: pr.suppressed_hints,
                rejected_hints: pr.rejected_hints,
                avg_policy_strength: pr.avg_policy_strength,
            }
        };
        hotspots.policy_seeds_found = policy.seeds_found;
        timings.policy_ms = t39.elapsed().as_secs_f64() * 1000.0;

        let feedback = {
            let causal_eng = causal_engine.read();
            let policy_eng = policy_engine.read();
            let mut belief_eng = belief_engine.write();
            let feedback = belief_eng.apply_layer_feedback(&causal_eng, &policy_eng);
            let _ = belief_store.save(&belief_eng);
            background_brain::FeedbackAuditReport {
                beliefs_touched: feedback.beliefs_touched,
                beliefs_boosted: feedback.beliefs_boosted,
                beliefs_dampened: feedback.beliefs_dampened,
                net_confidence_delta: feedback.net_confidence_delta,
                net_volatility_delta: feedback.net_volatility_delta,
                entries: feedback
                    .entries
                    .into_iter()
                    .map(|entry| background_brain::FeedbackAuditEntry {
                        belief_id: entry.belief_id,
                        source_kind: entry.source_kind,
                        source_id: entry.source_id,
                        reason: entry.reason,
                        delta_requested: entry.delta_requested,
                        delta_applied: entry.delta_applied,
                        confidence_before: entry.confidence_before,
                        confidence_after: entry.confidence_after,
                        volatility_before: entry.volatility_before,
                        volatility_after: entry.volatility_after,
                        volatility_delta_applied: entry.volatility_delta_applied,
                        stability_before: entry.stability_before,
                        stability_after: entry.stability_after,
                        stability_delta_applied: entry.stability_delta_applied,
                    })
                    .collect(),
            }
        };

        DiscoveryPhaseResult {
            belief,
            concept,
            causal,
            policy,
            feedback,
        }
    }

    pub(crate) fn run_post_discovery_phases(
        records_lock: &RwLock<HashMap<String, Record>>,
        ngram_index: &RwLock<NGramIndex>,
        tag_index: &RwLock<HashMap<String, HashSet<String>>>,
        aura_index: &RwLock<HashMap<String, String>>,
        cognitive_store: &CognitiveStore,
        background: Option<&background_brain::BackgroundBrain>,
        config: &background_brain::MaintenanceConfig,
        taxonomy: &TagTaxonomy,
        timings: &mut background_brain::PhaseTimings,
        hotspots: &mut background_brain::MaintenanceHotspots,
    ) -> PostDiscoveryPhaseResult {
        let t4 = std::time::Instant::now();
        let consolidation = if config.consolidation_enabled {
            let mut records = records_lock.write();
            let mut ngram = ngram_index.write();
            let mut tag_idx = tag_index.write();
            let mut aura_idx = aura_index.write();

            let result = consolidation::consolidate(
                &mut records,
                &mut ngram,
                &mut tag_idx,
                &mut aura_idx,
                cognitive_store,
            );

            background_brain::ConsolidationReport {
                native_merged: result.merged,
                clusters_found: 0,
                meta_created: 0,
            }
        } else {
            background_brain::ConsolidationReport::default()
        };
        timings.consolidation_ms = t4.elapsed().as_secs_f64() * 1000.0;

        let mut records = records_lock.write();

        let t5 = std::time::Instant::now();
        let cross_connections = if config.synthesis_enabled {
            let discoveries = background_brain::discover_cross_connections(&records, 3);
            let count = discoveries.len();
            if let Some(bg) = background {
                *bg.last_cross_connections.write() = discoveries;
            }
            count
        } else {
            0
        };
        hotspots.cross_connections_found = cross_connections;
        timings.cross_connections_ms = t5.elapsed().as_secs_f64() * 1000.0;

        let t67 = std::time::Instant::now();
        let task_reminders = background_brain::check_scheduled_tasks(&records, &config.task_tag);
        hotspots.task_reminders_found = task_reminders.len();

        let records_archived = if config.archival_enabled {
            background_brain::archive_old_records(&mut records, config, taxonomy)
        } else {
            0
        };
        hotspots.records_after_cycle = records.len();
        timings.tasks_archival_ms = t67.elapsed().as_secs_f64() * 1000.0;

        PostDiscoveryPhaseResult {
            consolidation,
            cross_connections,
            task_reminders,
            records_archived,
        }
    }

    pub(crate) fn finalize_telemetry(
        timings: &background_brain::PhaseTimings,
        hotspots: &mut background_brain::MaintenanceHotspots,
        concept_surface_mode: ConceptSurfaceMode,
        concept_engine: &RwLock<ConceptEngine>,
        counters: ConceptSurfaceCounters<'_>,
    ) -> background_brain::ConceptSurfaceTelemetry {
        let phase_candidates = [
            ("level_fix", timings.level_fix_ms),
            ("decay", timings.decay_ms),
            ("reflect", timings.reflect_ms),
            ("epistemic", timings.epistemic_ms),
            ("insights", timings.insights_ms),
            ("sdr_build", timings.sdr_build_ms),
            ("belief", timings.belief_ms),
            ("concept", timings.concept_ms),
            ("causal", timings.causal_ms),
            ("policy", timings.policy_ms),
            ("consolidation", timings.consolidation_ms),
            ("cross_connections", timings.cross_connections_ms),
            ("tasks_archival", timings.tasks_archival_ms),
        ];
        if let Some((name, ms)) = phase_candidates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            hotspots.dominant_phase = (*name).to_string();
            hotspots.dominant_phase_ms = *ms;
            hotspots.dominant_phase_share = if timings.total_ms > 0.0 {
                *ms / timings.total_ms
            } else {
                0.0
            };
        }

        let (surfaced_concepts_available, surfaced_namespaces) = if concept_surface_mode
            == ConceptSurfaceMode::Inspect
            || concept_surface_mode == ConceptSurfaceMode::Limited
        {
            let concept_eng = concept_engine.read();
            let surfaced = concept::surface_concepts(&concept_eng, None);
            let namespaces: HashSet<String> =
                surfaced.iter().map(|c| c.namespace.clone()).collect();
            (surfaced.len(), namespaces.len())
        } else {
            (0, 0)
        };

        background_brain::ConceptSurfaceTelemetry {
            mode: format!("{concept_surface_mode:?}"),
            surfaced_concepts_available,
            surfaced_namespaces,
            global_calls_since_last_cycle: counters.global_calls.swap(0, Ordering::Relaxed),
            namespace_calls_since_last_cycle: counters.namespace_calls.swap(0, Ordering::Relaxed),
            record_calls_since_last_cycle: counters.record_calls.swap(0, Ordering::Relaxed),
            concepts_returned_since_last_cycle: counters
                .concepts_returned
                .swap(0, Ordering::Relaxed),
            record_annotations_returned_since_last_cycle: counters
                .record_annotations_returned
                .swap(0, Ordering::Relaxed),
        }
    }

    pub(crate) fn build_trend_snapshot(
        timestamp: String,
        total_records: usize,
        records_archived: usize,
        insights_found: usize,
        epistemic: &background_brain::EpistemicPhaseReport,
        belief: &background_brain::BeliefPhaseReport,
        causal: &background_brain::CausalPhaseReport,
        policy: &background_brain::PolicyPhaseReport,
        feedback: &background_brain::FeedbackAuditReport,
        timings: &background_brain::PhaseTimings,
        hotspots: &background_brain::MaintenanceHotspots,
        cumulative_corrections: usize,
        previous_cumulative_corrections: usize,
    ) -> background_brain::MaintenanceTrendSnapshot {
        let causal_total = causal.stable_count + causal.rejected_count;
        let policy_total = policy.stable_hints + policy.suppressed_hints + policy.rejected_hints;

        background_brain::MaintenanceTrendSnapshot {
            timestamp,
            total_records,
            records_archived,
            insights_found,
            volatile_records: epistemic.volatile_records,
            belief_churn: belief.churn_rate,
            causal_rejection_rate: if causal_total > 0 {
                causal.rejected_count as f32 / causal_total as f32
            } else {
                0.0
            },
            policy_suppression_rate: if policy_total > 0 {
                policy.suppressed_hints as f32 / policy_total as f32
            } else {
                0.0
            },
            feedback_beliefs_touched: feedback.beliefs_touched,
            feedback_net_confidence_delta: feedback.net_confidence_delta,
            feedback_net_volatility_delta: feedback.net_volatility_delta,
            correction_events: cumulative_corrections
                .saturating_sub(previous_cumulative_corrections),
            cumulative_corrections,
            cycle_time_ms: timings.total_ms,
            dominant_phase: hotspots.dominant_phase.clone(),
        }
    }

    pub(crate) fn push_trend_snapshot(
        history: &mut Vec<background_brain::MaintenanceTrendSnapshot>,
        snapshot: background_brain::MaintenanceTrendSnapshot,
    ) {
        history.push(snapshot);
        if history.len() > Self::MAINTENANCE_TREND_LIMIT {
            let overflow = history.len() - Self::MAINTENANCE_TREND_LIMIT;
            history.drain(0..overflow);
        }
    }

    pub(crate) fn summarize_trends(
        history: &[background_brain::MaintenanceTrendSnapshot],
    ) -> background_brain::MaintenanceTrendSummary {
        let count = history.len();
        if count == 0 {
            return background_brain::MaintenanceTrendSummary::default();
        }

        let sum_belief_churn: f32 = history.iter().map(|s| s.belief_churn).sum();
        let sum_causal_rejection_rate: f32 = history.iter().map(|s| s.causal_rejection_rate).sum();
        let sum_policy_suppression_rate: f32 =
            history.iter().map(|s| s.policy_suppression_rate).sum();
        let sum_cycle_time_ms: f64 = history.iter().map(|s| s.cycle_time_ms).sum();
        let total_corrections_in_window: usize = history.iter().map(|s| s.correction_events).sum();
        let latest_dominant_phase = history
            .last()
            .map(|snapshot| snapshot.dominant_phase.clone())
            .unwrap_or_default();

        background_brain::MaintenanceTrendSummary {
            snapshot_count: count,
            recent: history.to_vec(),
            avg_belief_churn: sum_belief_churn / count as f32,
            avg_causal_rejection_rate: sum_causal_rejection_rate / count as f32,
            avg_policy_suppression_rate: sum_policy_suppression_rate / count as f32,
            avg_cycle_time_ms: sum_cycle_time_ms / count as f64,
            avg_correction_events: total_corrections_in_window as f32 / count as f32,
            total_corrections_in_window,
            latest_dominant_phase,
        }
    }

    pub(crate) fn build_reflection_summary(
        timestamp: String,
        records: &HashMap<String, Record>,
        task_tag: &str,
        contradiction_clusters: &[crate::epistemic_runtime::ContradictionCluster],
        trend_summary: &background_brain::MaintenanceTrendSummary,
        hotspots: &background_brain::MaintenanceHotspots,
    ) -> background_brain::ReflectionSummary {
        let mut findings = Vec::new();
        let blocker_findings;
        let contradiction_findings;
        let mut trend_findings = 0usize;
        let today = chrono::Utc::now().date_naive();

        let mut blocker_candidates = records
            .values()
            .filter(|rec| rec.tags.iter().any(|tag| tag == task_tag))
            .filter(|rec| {
                rec.metadata
                    .get("status")
                    .map(|value| value.as_str())
                    .unwrap_or_default()
                    == "active"
            })
            .filter_map(|rec| {
                let due = rec.metadata.get("due_date")?;
                let due = chrono::DateTime::parse_from_rfc3339(due)
                    .ok()?
                    .with_timezone(&chrono::Utc);
                if due.date_naive() > today {
                    return None;
                }
                let overdue_days = (today - due.date_naive()).num_days().max(0) as f32;
                let preview = if rec.content.chars().count() > 56 {
                    format!("{}...", rec.content.chars().take(56).collect::<String>())
                } else {
                    rec.content.clone()
                };
                Some(background_brain::ReflectionFinding {
                    kind: "repeated_blocker".into(),
                    namespace: rec.namespace.clone(),
                    title: format!("Overdue task remains active: {preview}"),
                    detail: format!(
                        "Task is overdue by {} day(s), salience {:.2}, strength {:.2}.",
                        overdue_days as i64, rec.salience, rec.strength
                    ),
                    related_ids: vec![rec.id.clone()],
                    score: overdue_days + rec.salience + (1.0 - rec.strength).max(0.0),
                    severity: if overdue_days >= 3.0 || rec.salience >= 0.70 {
                        "high".into()
                    } else if overdue_days >= 1.0 {
                        "medium".into()
                    } else {
                        "low".into()
                    },
                })
            })
            .collect::<Vec<_>>();
        blocker_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        blocker_candidates.truncate(2);
        blocker_findings = blocker_candidates.len();
        findings.extend(blocker_candidates);

        let mut contradiction_candidates = contradiction_clusters
            .iter()
            .filter(|cluster| {
                cluster.unresolved_belief_count > 0 || cluster.total_conflict_mass > 0.0
            })
            .map(|cluster| background_brain::ReflectionFinding {
                kind: "unresolved_contradiction".into(),
                namespace: cluster.namespace.clone(),
                title: format!(
                    "Unresolved contradiction corridor with {} beliefs",
                    cluster.belief_ids.len()
                ),
                detail: format!(
                    "Avg volatility {:.2}, conflict mass {:.2}, shared tags: {}.",
                    cluster.avg_volatility,
                    cluster.total_conflict_mass,
                    if cluster.shared_tags.is_empty() {
                        "none".into()
                    } else {
                        cluster.shared_tags.join(", ")
                    }
                ),
                related_ids: cluster.belief_ids.clone(),
                score: cluster.avg_volatility + cluster.total_conflict_mass.min(1.5),
                severity: if cluster.avg_volatility >= 0.45 || cluster.total_conflict_mass >= 1.0 {
                    "high".into()
                } else if cluster.avg_volatility >= 0.20 {
                    "medium".into()
                } else {
                    "low".into()
                },
            })
            .collect::<Vec<_>>();
        contradiction_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        contradiction_candidates.truncate(2);
        contradiction_findings = contradiction_candidates.len();
        findings.extend(contradiction_candidates);

        let trend_direction = if trend_summary.recent.len() < 2 {
            "insufficient_data"
        } else {
            let first = &trend_summary.recent[0];
            let last = trend_summary
                .recent
                .last()
                .expect("recent has at least 2 items");
            let first_pressure = first.volatile_records as f32
                + first.correction_events as f32
                + first.policy_suppression_rate * 10.0
                + first.causal_rejection_rate * 10.0;
            let last_pressure = last.volatile_records as f32
                + last.correction_events as f32
                + last.policy_suppression_rate * 10.0
                + last.causal_rejection_rate * 10.0;
            let delta = last_pressure - first_pressure;
            if delta > 1.0 {
                "worsening"
            } else if delta < -1.0 {
                "improving"
            } else {
                "stable"
            }
        };

        if trend_direction == "worsening"
            || trend_summary.avg_belief_churn >= 0.10
            || trend_summary.avg_policy_suppression_rate >= 0.25
        {
            trend_findings = 1;
            findings.push(background_brain::ReflectionFinding {
                kind: "trend_tension".into(),
                namespace: crate::record::DEFAULT_NAMESPACE.into(),
                title: format!(
                    "Maintenance trend is {} with dominant phase {}",
                    trend_direction, hotspots.dominant_phase
                ),
                detail: format!(
                    "Avg belief churn {:.2}, policy suppression {:.2}, cycle time {:.0}ms.",
                    trend_summary.avg_belief_churn,
                    trend_summary.avg_policy_suppression_rate,
                    trend_summary.avg_cycle_time_ms
                ),
                related_ids: Vec::new(),
                score: trend_summary.avg_belief_churn
                    + trend_summary.avg_policy_suppression_rate
                    + if trend_direction == "worsening" {
                        0.5
                    } else {
                        0.0
                    },
                severity: if trend_direction == "worsening" {
                    "high".into()
                } else {
                    "medium".into()
                },
            });
        }

        findings.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total_findings = findings.len();
        let capped = total_findings > Self::REFLECTION_FINDING_LIMIT;
        findings.truncate(Self::REFLECTION_FINDING_LIMIT);

        let digest = if findings.is_empty() {
            "No significant reflection findings this cycle.".into()
        } else {
            format!(
                "{} reflection finding(s): {}",
                total_findings,
                findings
                    .iter()
                    .take(2)
                    .map(|finding| finding.title.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        };

        background_brain::ReflectionSummary {
            timestamp,
            digest,
            dominant_phase: hotspots.dominant_phase.clone(),
            report: background_brain::ReflectionJobReport {
                jobs_run: 3,
                blocker_findings,
                contradiction_findings,
                trend_findings,
                total_findings,
                capped,
            },
            findings,
        }
    }

    pub(crate) fn push_reflection_summary(
        history: &mut Vec<background_brain::ReflectionSummary>,
        summary: background_brain::ReflectionSummary,
    ) {
        history.push(summary);
        if history.len() > Self::REFLECTION_SUMMARY_LIMIT {
            let overflow = history.len() - Self::REFLECTION_SUMMARY_LIMIT;
            history.drain(0..overflow);
        }
    }

    pub(crate) fn summarize_reflections(
        history: &[background_brain::ReflectionSummary],
    ) -> background_brain::ReflectionDigest {
        if history.is_empty() {
            return background_brain::ReflectionDigest::default();
        }

        let mut kind_counts: HashMap<String, (usize, usize, f32)> = HashMap::new();
        let mut namespace_counts: HashMap<String, usize> = HashMap::new();
        let mut top_findings = Vec::new();
        let mut total_findings = 0usize;
        let mut high_severity_findings = 0usize;

        for summary in history {
            for finding in &summary.findings {
                total_findings += 1;
                if finding.severity == "high" {
                    high_severity_findings += 1;
                }
                let entry = kind_counts
                    .entry(finding.kind.clone())
                    .or_insert((0, 0, 0.0));
                entry.0 += 1;
                if finding.severity == "high" {
                    entry.1 += 1;
                }
                entry.2 += finding.score;
                *namespace_counts
                    .entry(finding.namespace.clone())
                    .or_insert(0) += 1;
                top_findings.push(finding.clone());
            }
        }

        let mut kinds = kind_counts
            .into_iter()
            .map(
                |(kind, (count, high, score_sum))| background_brain::ReflectionKindSummary {
                    kind,
                    count,
                    high_severity_count: high,
                    avg_score: if count > 0 {
                        score_sum / count as f32
                    } else {
                        0.0
                    },
                },
            )
            .collect::<Vec<_>>();
        kinds.sort_by(|a, b| {
            b.high_severity_count
                .cmp(&a.high_severity_count)
                .then_with(|| b.count.cmp(&a.count))
                .then_with(|| {
                    b.avg_score
                        .partial_cmp(&a.avg_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        kinds.truncate(Self::REFLECTION_KIND_LIMIT);

        let mut namespaces = namespace_counts.into_iter().collect::<Vec<_>>();
        namespaces.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        top_findings.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.title.cmp(&b.title))
        });
        top_findings.truncate(Self::REFLECTION_FINDING_LIMIT);

        let latest = history.last().expect("history is non-empty");
        background_brain::ReflectionDigest {
            summary_count: history.len(),
            total_findings,
            high_severity_findings,
            latest_timestamp: latest.timestamp.clone(),
            latest_dominant_phase: latest.dominant_phase.clone(),
            kinds,
            namespaces: namespaces
                .into_iter()
                .take(Self::REFLECTION_NAMESPACE_LIMIT)
                .map(|(namespace, _)| namespace)
                .collect(),
            top_findings,
        }
    }
}
