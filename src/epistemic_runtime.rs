//! Read-only cognitive inspection surface extracted from `Aura`.
//!
//! Keeps belief/concept/causal/policy inspection logic separate from storage,
//! recall, and maintenance orchestration.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLockReadGuard;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::belief::{Belief, BeliefEngine, BeliefState};
use crate::causal::{CausalEngine, CausalPattern, CausalState};
use crate::concept::{
    self, ConceptCandidate, ConceptEngine, ConceptState, ConceptSurfaceMode, SurfacedConcept,
};
use crate::policy::{self, PolicyEngine, PolicyHint, PolicyState, SurfacedPolicyHint};
use crate::record::Record;

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BeliefVolatilityBands {
    pub low: usize,
    pub medium: usize,
    pub high: usize,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BeliefInstabilitySummary {
    pub total_beliefs: usize,
    pub resolved: usize,
    pub unresolved: usize,
    pub singleton: usize,
    pub empty: usize,
    pub contradiction_cluster_count: usize,
    pub high_volatility_count: usize,
    pub low_stability_count: usize,
    pub avg_volatility: f32,
    pub avg_stability: f32,
    pub volatility_bands: BeliefVolatilityBands,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ContradictionCluster {
    pub id: String,
    pub namespace: String,
    pub belief_ids: Vec<String>,
    pub belief_keys: Vec<String>,
    pub record_ids: Vec<String>,
    pub shared_tags: Vec<String>,
    pub unresolved_belief_count: usize,
    pub high_volatility_belief_count: usize,
    pub avg_volatility: f32,
    pub avg_stability: f32,
    pub total_conflict_mass: f32,
    pub max_conflict_mass: f32,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PolicyActionSummary {
    pub action_kind: String,
    pub total_hints: usize,
    pub stable_hints: usize,
    pub candidate_hints: usize,
    pub suppressed_hints: usize,
    pub rejected_hints: usize,
    pub avg_policy_strength: f32,
    pub avg_risk_score: f32,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PolicyDomainSummary {
    pub namespace: String,
    pub domain: String,
    pub total_hints: usize,
    pub active_hints: usize,
    pub stable_hints: usize,
    pub candidate_hints: usize,
    pub suppressed_hints: usize,
    pub rejected_hints: usize,
    pub avg_policy_strength: f32,
    pub avg_risk_score: f32,
    pub advisory_pressure: f32,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PolicyPressureArea {
    pub namespace: String,
    pub domain: String,
    pub advisory_pressure: f32,
    pub active_hints: usize,
    pub suppressed_hints: usize,
    pub rejected_hints: usize,
    pub strongest_hint_id: String,
    pub strongest_action_kind: String,
    pub strongest_policy_strength: f32,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PolicyLifecycleSummary {
    pub total_hints: usize,
    pub active_hints: usize,
    pub stable_hints: usize,
    pub candidate_hints: usize,
    pub suppressed_hints: usize,
    pub rejected_hints: usize,
    pub avg_policy_strength: f32,
    pub avg_risk_score: f32,
    pub action_summaries: Vec<PolicyActionSummary>,
    pub domain_summaries: Vec<PolicyDomainSummary>,
}

pub struct EpistemicRuntime<'a> {
    records: RwLockReadGuard<'a, HashMap<String, Record>>,
    belief_engine: RwLockReadGuard<'a, BeliefEngine>,
    concept_engine: RwLockReadGuard<'a, ConceptEngine>,
    causal_engine: RwLockReadGuard<'a, CausalEngine>,
    policy_engine: RwLockReadGuard<'a, PolicyEngine>,
    concept_surface_mode: ConceptSurfaceMode,
    concept_surface_global_calls: &'a AtomicU64,
    concept_surface_namespace_calls: &'a AtomicU64,
    concept_surface_record_calls: &'a AtomicU64,
    concept_surface_results_returned: &'a AtomicU64,
    concept_surface_record_results_returned: &'a AtomicU64,
}

impl<'a> EpistemicRuntime<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        records: RwLockReadGuard<'a, HashMap<String, Record>>,
        belief_engine: RwLockReadGuard<'a, BeliefEngine>,
        concept_engine: RwLockReadGuard<'a, ConceptEngine>,
        causal_engine: RwLockReadGuard<'a, CausalEngine>,
        policy_engine: RwLockReadGuard<'a, PolicyEngine>,
        concept_surface_mode: ConceptSurfaceMode,
        concept_surface_global_calls: &'a AtomicU64,
        concept_surface_namespace_calls: &'a AtomicU64,
        concept_surface_record_calls: &'a AtomicU64,
        concept_surface_results_returned: &'a AtomicU64,
        concept_surface_record_results_returned: &'a AtomicU64,
    ) -> Self {
        Self {
            records,
            belief_engine,
            concept_engine,
            causal_engine,
            policy_engine,
            concept_surface_mode,
            concept_surface_global_calls,
            concept_surface_namespace_calls,
            concept_surface_record_calls,
            concept_surface_results_returned,
            concept_surface_record_results_returned,
        }
    }

    pub fn get_beliefs(&self, state_filter: Option<&str>) -> Vec<Belief> {
        self.belief_engine
            .beliefs
            .values()
            .filter(|belief| match state_filter {
                Some("resolved") => belief.state == BeliefState::Resolved,
                Some("unresolved") => belief.state == BeliefState::Unresolved,
                Some("singleton") => belief.state == BeliefState::Singleton,
                Some("empty") => belief.state == BeliefState::Empty,
                _ => true,
            })
            .cloned()
            .collect()
    }

    pub fn get_belief_for_record(&self, record_id: &str) -> Option<Belief> {
        self.belief_engine.belief_for_record(record_id).cloned()
    }

    pub fn get_high_volatility_beliefs(
        &self,
        min_volatility: Option<f32>,
        limit: Option<usize>,
    ) -> Vec<Belief> {
        let threshold = min_volatility.unwrap_or(0.20).clamp(0.0, 1.0);
        let max = limit.unwrap_or(20).min(100);
        let mut beliefs: Vec<Belief> = self
            .belief_engine
            .beliefs
            .values()
            .filter(|belief| belief.volatility >= threshold)
            .cloned()
            .collect();
        beliefs.sort_by(|a, b| {
            b.volatility
                .partial_cmp(&a.volatility)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.stability
                        .partial_cmp(&b.stability)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        beliefs.truncate(max);
        beliefs
    }

    pub fn get_low_stability_beliefs(
        &self,
        max_stability: Option<f32>,
        limit: Option<usize>,
    ) -> Vec<Belief> {
        let threshold = max_stability.unwrap_or(1.0).max(0.0);
        let max = limit.unwrap_or(20).min(100);
        let mut beliefs: Vec<Belief> = self
            .belief_engine
            .beliefs
            .values()
            .filter(|belief| belief.stability <= threshold)
            .cloned()
            .collect();
        beliefs.sort_by(|a, b| {
            a.stability
                .partial_cmp(&b.stability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.volatility
                        .partial_cmp(&a.volatility)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        beliefs.truncate(max);
        beliefs
    }

    pub fn get_belief_instability_summary(&self) -> BeliefInstabilitySummary {
        let total = self.belief_engine.beliefs.len();
        if total == 0 {
            return BeliefInstabilitySummary::default();
        }

        let mut summary = BeliefInstabilitySummary {
            total_beliefs: total,
            ..BeliefInstabilitySummary::default()
        };

        for belief in self.belief_engine.beliefs.values() {
            match belief.state {
                BeliefState::Resolved => summary.resolved += 1,
                BeliefState::Unresolved => summary.unresolved += 1,
                BeliefState::Singleton => summary.singleton += 1,
                BeliefState::Empty => summary.empty += 1,
            }

            summary.avg_volatility += belief.volatility;
            summary.avg_stability += belief.stability;

            if belief.volatility >= 0.20 {
                summary.high_volatility_count += 1;
                summary.volatility_bands.high += 1;
            } else if belief.volatility >= 0.05 {
                summary.volatility_bands.medium += 1;
            } else {
                summary.volatility_bands.low += 1;
            }

            if belief.stability <= 1.0 {
                summary.low_stability_count += 1;
            }
        }

        summary.avg_volatility /= total as f32;
        summary.avg_stability /= total as f32;
        summary.contradiction_cluster_count = self.get_contradiction_clusters(None, None).len();
        summary
    }

    pub fn get_contradiction_clusters(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<ContradictionCluster> {
        let max = limit.unwrap_or(20).min(100);

        #[derive(Clone)]
        struct BeliefConflictNode {
            belief: Belief,
            record_ids: Vec<String>,
            tags: HashSet<String>,
            namespace: String,
        }

        let mut nodes = Vec::new();
        for belief in self.belief_engine.beliefs.values() {
            let belief_namespace = belief
                .key
                .split(':')
                .next()
                .unwrap_or(crate::record::DEFAULT_NAMESPACE)
                .to_string();
            if namespace.is_some_and(|expected| expected != belief_namespace) {
                continue;
            }

            if belief.state != BeliefState::Unresolved
                && belief.volatility < 0.20
                && belief.stability > 1.0
                && belief.conflict_mass <= 0.0
            {
                continue;
            }

            let mut record_ids = belief
                .hypothesis_ids
                .iter()
                .filter_map(|hid| self.belief_engine.hypotheses.get(hid))
                .flat_map(|hyp| hyp.prototype_record_ids.iter().cloned())
                .collect::<Vec<_>>();
            record_ids.sort();
            record_ids.dedup();

            let tags = record_ids
                .iter()
                .filter_map(|rid| self.records.get(rid))
                .flat_map(|record| record.tags.iter().cloned())
                .collect::<HashSet<_>>();

            nodes.push(BeliefConflictNode {
                belief: belief.clone(),
                record_ids,
                tags,
                namespace: belief_namespace,
            });
        }

        let mut visited = vec![false; nodes.len()];
        let mut clusters = Vec::new();

        for idx in 0..nodes.len() {
            if visited[idx] {
                continue;
            }

            let mut stack = vec![idx];
            let mut component = Vec::new();
            visited[idx] = true;

            while let Some(current) = stack.pop() {
                component.push(current);
                for next in 0..nodes.len() {
                    if visited[next] || nodes[current].namespace != nodes[next].namespace {
                        continue;
                    }
                    let share_records = nodes[current]
                        .record_ids
                        .iter()
                        .any(|rid| nodes[next].record_ids.contains(rid));
                    let share_tags = nodes[current]
                        .tags
                        .iter()
                        .any(|tag| nodes[next].tags.contains(tag));
                    if share_records || share_tags {
                        visited[next] = true;
                        stack.push(next);
                    }
                }
            }

            let namespace = nodes[component[0]].namespace.clone();
            let mut belief_ids = component
                .iter()
                .map(|index| nodes[*index].belief.id.clone())
                .collect::<Vec<_>>();
            belief_ids.sort();

            let belief_keys = component
                .iter()
                .map(|index| nodes[*index].belief.key.clone())
                .collect::<Vec<_>>();

            let mut record_ids = component
                .iter()
                .flat_map(|index| nodes[*index].record_ids.iter().cloned())
                .collect::<Vec<_>>();
            record_ids.sort();
            record_ids.dedup();

            let mut tag_counts = BTreeMap::<String, usize>::new();
            for index in &component {
                for tag in &nodes[*index].tags {
                    *tag_counts.entry(tag.clone()).or_insert(0) += 1;
                }
            }
            let mut shared_tags = tag_counts
                .iter()
                .filter(|(_, count)| **count >= 2)
                .map(|(tag, _)| tag.clone())
                .collect::<Vec<_>>();
            if shared_tags.is_empty() {
                shared_tags = tag_counts.keys().take(3).cloned().collect();
            }

            let unresolved_belief_count = component
                .iter()
                .filter(|index| nodes[**index].belief.state == BeliefState::Unresolved)
                .count();
            let high_volatility_belief_count = component
                .iter()
                .filter(|index| nodes[**index].belief.volatility >= 0.20)
                .count();
            let total_conflict_mass = component
                .iter()
                .map(|index| nodes[*index].belief.conflict_mass)
                .sum::<f32>();
            let max_conflict_mass = component
                .iter()
                .map(|index| nodes[*index].belief.conflict_mass)
                .fold(0.0_f32, f32::max);
            let avg_volatility = component
                .iter()
                .map(|index| nodes[*index].belief.volatility)
                .sum::<f32>()
                / component.len() as f32;
            let avg_stability = component
                .iter()
                .map(|index| nodes[*index].belief.stability)
                .sum::<f32>()
                / component.len() as f32;

            let cluster_key = format!("{}\0{}", namespace, belief_ids.join("\0"));
            let id = format!(
                "{:012x}",
                xxhash_rust::xxh3::xxh3_64(cluster_key.as_bytes())
            );

            clusters.push(ContradictionCluster {
                id,
                namespace,
                belief_ids,
                belief_keys,
                record_ids,
                shared_tags,
                unresolved_belief_count,
                high_volatility_belief_count,
                avg_volatility,
                avg_stability,
                total_conflict_mass,
                max_conflict_mass,
            });
        }

        clusters.sort_by(|a, b| {
            b.avg_volatility
                .partial_cmp(&a.avg_volatility)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.total_conflict_mass
                        .partial_cmp(&a.total_conflict_mass)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| b.belief_ids.len().cmp(&a.belief_ids.len()))
        });
        clusters.truncate(max);
        clusters
    }

    pub fn get_concepts(&self, state_filter: Option<&str>) -> Vec<ConceptCandidate> {
        self.concept_engine
            .concepts
            .values()
            .filter(|concept| match state_filter {
                Some("stable") => concept.state == ConceptState::Stable,
                Some("candidate") => concept.state == ConceptState::Candidate,
                Some("rejected") => concept.state == ConceptState::Rejected,
                _ => true,
            })
            .cloned()
            .collect()
    }

    pub fn get_surfaced_concepts(&self, limit: Option<usize>) -> Vec<SurfacedConcept> {
        if !self.concept_surface_enabled() {
            return Vec::new();
        }
        let surfaced = concept::surface_concepts(&self.concept_engine, limit);
        self.track_global_concept_surface_call(surfaced.len());
        surfaced
    }

    pub fn get_surfaced_concepts_for_namespace(
        &self,
        namespace: &str,
        limit: Option<usize>,
    ) -> Vec<SurfacedConcept> {
        if !self.concept_surface_enabled() {
            return Vec::new();
        }
        let surfaced =
            concept::surface_concepts_filtered(&self.concept_engine, limit, Some(namespace));
        self.track_namespace_concept_surface_call(surfaced.len());
        surfaced
    }

    pub fn get_surfaced_concepts_for_record(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Vec<SurfacedConcept> {
        if !self.concept_surface_enabled() {
            return Vec::new();
        }
        let max = limit.unwrap_or(3).min(3);
        let surfaced: Vec<_> = concept::surface_concepts(&self.concept_engine, None)
            .into_iter()
            .filter(|concept| concept.record_ids.iter().any(|rid| rid == record_id))
            .take(max)
            .collect();
        self.track_record_concept_surface_call(surfaced.len());
        surfaced
    }

    pub fn get_causal_patterns(&self, state_filter: Option<&str>) -> Vec<CausalPattern> {
        self.causal_engine
            .patterns
            .values()
            .filter(|pattern| match state_filter {
                Some("stable") => pattern.state == CausalState::Stable,
                Some("candidate") => pattern.state == CausalState::Candidate,
                Some("rejected") => pattern.state == CausalState::Rejected,
                _ => true,
            })
            .cloned()
            .collect()
    }

    pub fn get_policy_hints(&self, state_filter: Option<&str>) -> Vec<PolicyHint> {
        self.policy_engine
            .hints
            .values()
            .filter(|hint| match state_filter {
                Some("stable") => hint.state == PolicyState::Stable,
                Some("candidate") => hint.state == PolicyState::Candidate,
                Some("suppressed") => hint.state == PolicyState::Suppressed,
                Some("rejected") => hint.state == PolicyState::Rejected,
                _ => true,
            })
            .cloned()
            .collect()
    }

    pub fn get_suppressed_policy_hints(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<PolicyHint> {
        self.collect_policy_hints_by_state(PolicyState::Suppressed, namespace, limit)
    }

    pub fn get_rejected_policy_hints(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<PolicyHint> {
        self.collect_policy_hints_by_state(PolicyState::Rejected, namespace, limit)
    }

    pub fn get_policy_lifecycle_summary(
        &self,
        action_limit: Option<usize>,
        domain_limit: Option<usize>,
    ) -> PolicyLifecycleSummary {
        #[derive(Default)]
        struct ActionAccumulator {
            total_hints: usize,
            stable_hints: usize,
            candidate_hints: usize,
            suppressed_hints: usize,
            rejected_hints: usize,
            total_policy_strength: f32,
            total_risk_score: f32,
        }

        #[derive(Default)]
        struct DomainAccumulator {
            total_hints: usize,
            stable_hints: usize,
            candidate_hints: usize,
            suppressed_hints: usize,
            rejected_hints: usize,
            total_policy_strength: f32,
            total_risk_score: f32,
            advisory_pressure: f32,
        }

        let max_actions = action_limit.unwrap_or(8).min(16);
        let max_domains = domain_limit.unwrap_or(12).min(32);
        let mut summary = PolicyLifecycleSummary::default();
        let mut action_map: BTreeMap<&'static str, ActionAccumulator> = BTreeMap::new();
        let mut domain_map: BTreeMap<(String, String), DomainAccumulator> = BTreeMap::new();

        for hint in self.policy_engine.hints.values() {
            summary.total_hints += 1;
            summary.avg_policy_strength += hint.policy_strength;
            summary.avg_risk_score += hint.risk_score;

            if matches!(hint.state, PolicyState::Stable | PolicyState::Candidate) {
                summary.active_hints += 1;
            }

            match hint.state {
                PolicyState::Stable => summary.stable_hints += 1,
                PolicyState::Candidate => summary.candidate_hints += 1,
                PolicyState::Suppressed => summary.suppressed_hints += 1,
                PolicyState::Rejected => summary.rejected_hints += 1,
            }

            let action_entry = action_map
                .entry(policy_action_kind_name(hint.action_kind))
                .or_default();
            action_entry.total_hints += 1;
            action_entry.total_policy_strength += hint.policy_strength;
            action_entry.total_risk_score += hint.risk_score;
            match hint.state {
                PolicyState::Stable => action_entry.stable_hints += 1,
                PolicyState::Candidate => action_entry.candidate_hints += 1,
                PolicyState::Suppressed => action_entry.suppressed_hints += 1,
                PolicyState::Rejected => action_entry.rejected_hints += 1,
            }

            let domain_entry = domain_map
                .entry((hint.namespace.clone(), hint.domain.clone()))
                .or_default();
            domain_entry.total_hints += 1;
            domain_entry.total_policy_strength += hint.policy_strength;
            domain_entry.total_risk_score += hint.risk_score;
            if matches!(hint.state, PolicyState::Stable | PolicyState::Candidate) {
                domain_entry.advisory_pressure +=
                    hint.policy_strength * policy_action_pressure_weight(hint.action_kind);
            }
            match hint.state {
                PolicyState::Stable => domain_entry.stable_hints += 1,
                PolicyState::Candidate => domain_entry.candidate_hints += 1,
                PolicyState::Suppressed => domain_entry.suppressed_hints += 1,
                PolicyState::Rejected => domain_entry.rejected_hints += 1,
            }
        }

        if summary.total_hints > 0 {
            let denom = summary.total_hints as f32;
            summary.avg_policy_strength /= denom;
            summary.avg_risk_score /= denom;
        }

        let mut action_summaries: Vec<_> = action_map
            .into_iter()
            .map(|(action_kind, acc)| PolicyActionSummary {
                action_kind: action_kind.to_string(),
                total_hints: acc.total_hints,
                stable_hints: acc.stable_hints,
                candidate_hints: acc.candidate_hints,
                suppressed_hints: acc.suppressed_hints,
                rejected_hints: acc.rejected_hints,
                avg_policy_strength: if acc.total_hints > 0 {
                    acc.total_policy_strength / acc.total_hints as f32
                } else {
                    0.0
                },
                avg_risk_score: if acc.total_hints > 0 {
                    acc.total_risk_score / acc.total_hints as f32
                } else {
                    0.0
                },
            })
            .collect();
        action_summaries.sort_by(|a, b| {
            b.total_hints
                .cmp(&a.total_hints)
                .then_with(|| {
                    b.avg_policy_strength
                        .partial_cmp(&a.avg_policy_strength)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.action_kind.cmp(&b.action_kind))
        });
        action_summaries.truncate(max_actions);
        summary.action_summaries = action_summaries;

        let mut domain_summaries: Vec<_> = domain_map
            .into_iter()
            .map(|((namespace, domain), acc)| PolicyDomainSummary {
                namespace,
                domain,
                total_hints: acc.total_hints,
                active_hints: acc.stable_hints + acc.candidate_hints,
                stable_hints: acc.stable_hints,
                candidate_hints: acc.candidate_hints,
                suppressed_hints: acc.suppressed_hints,
                rejected_hints: acc.rejected_hints,
                avg_policy_strength: if acc.total_hints > 0 {
                    acc.total_policy_strength / acc.total_hints as f32
                } else {
                    0.0
                },
                avg_risk_score: if acc.total_hints > 0 {
                    acc.total_risk_score / acc.total_hints as f32
                } else {
                    0.0
                },
                advisory_pressure: acc.advisory_pressure,
            })
            .collect();
        domain_summaries.sort_by(|a, b| {
            b.advisory_pressure
                .partial_cmp(&a.advisory_pressure)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.active_hints.cmp(&a.active_hints))
                .then_with(|| a.namespace.cmp(&b.namespace))
                .then_with(|| a.domain.cmp(&b.domain))
        });
        domain_summaries.truncate(max_domains);
        summary.domain_summaries = domain_summaries;

        summary
    }

    pub fn get_policy_pressure_report(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<PolicyPressureArea> {
        #[derive(Default)]
        struct PressureAccumulator {
            advisory_pressure: f32,
            active_hints: usize,
            suppressed_hints: usize,
            rejected_hints: usize,
            strongest_hint_id: String,
            strongest_action_kind: String,
            strongest_policy_strength: f32,
        }

        let max = limit.unwrap_or(10).min(25);
        let mut report: BTreeMap<(String, String), PressureAccumulator> = BTreeMap::new();

        for hint in self.policy_engine.hints.values() {
            if namespace.is_some_and(|value| hint.namespace != value) {
                continue;
            }
            let entry = report
                .entry((hint.namespace.clone(), hint.domain.clone()))
                .or_default();
            match hint.state {
                PolicyState::Stable | PolicyState::Candidate => {
                    entry.active_hints += 1;
                    entry.advisory_pressure +=
                        hint.policy_strength * policy_action_pressure_weight(hint.action_kind);
                    if hint.policy_strength > entry.strongest_policy_strength
                        || (hint.policy_strength == entry.strongest_policy_strength
                            && hint.id < entry.strongest_hint_id)
                    {
                        entry.strongest_hint_id = hint.id.clone();
                        entry.strongest_action_kind =
                            policy_action_kind_name(hint.action_kind).to_string();
                        entry.strongest_policy_strength = hint.policy_strength;
                    }
                }
                PolicyState::Suppressed => entry.suppressed_hints += 1,
                PolicyState::Rejected => entry.rejected_hints += 1,
            }
        }

        let mut areas: Vec<_> = report
            .into_iter()
            .filter_map(|((namespace, domain), acc)| {
                if acc.active_hints == 0 && acc.suppressed_hints == 0 && acc.rejected_hints == 0 {
                    return None;
                }
                Some(PolicyPressureArea {
                    namespace,
                    domain,
                    advisory_pressure: acc.advisory_pressure,
                    active_hints: acc.active_hints,
                    suppressed_hints: acc.suppressed_hints,
                    rejected_hints: acc.rejected_hints,
                    strongest_hint_id: acc.strongest_hint_id,
                    strongest_action_kind: acc.strongest_action_kind,
                    strongest_policy_strength: acc.strongest_policy_strength,
                })
            })
            .collect();
        areas.sort_by(|a, b| {
            b.advisory_pressure
                .partial_cmp(&a.advisory_pressure)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.active_hints.cmp(&a.active_hints))
                .then_with(|| a.namespace.cmp(&b.namespace))
                .then_with(|| a.domain.cmp(&b.domain))
        });
        areas.truncate(max);
        areas
    }

    pub fn get_surfaced_policy_hints(&self, limit: Option<usize>) -> Vec<SurfacedPolicyHint> {
        policy::surface_policy_hints(&self.policy_engine, limit)
    }

    pub fn get_surfaced_policy_hints_for_namespace(
        &self,
        namespace: &str,
        limit: Option<usize>,
    ) -> Vec<SurfacedPolicyHint> {
        policy::surface_policy_hints_filtered(&self.policy_engine, limit, Some(namespace))
    }

    fn concept_surface_enabled(&self) -> bool {
        self.concept_surface_mode == ConceptSurfaceMode::Inspect
            || self.concept_surface_mode == ConceptSurfaceMode::Limited
    }

    fn track_global_concept_surface_call(&self, returned: usize) {
        self.concept_surface_global_calls
            .fetch_add(1, Ordering::Relaxed);
        self.concept_surface_results_returned
            .fetch_add(returned as u64, Ordering::Relaxed);
    }

    fn track_namespace_concept_surface_call(&self, returned: usize) {
        self.concept_surface_namespace_calls
            .fetch_add(1, Ordering::Relaxed);
        self.concept_surface_results_returned
            .fetch_add(returned as u64, Ordering::Relaxed);
    }

    fn track_record_concept_surface_call(&self, returned: usize) {
        self.concept_surface_record_calls
            .fetch_add(1, Ordering::Relaxed);
        self.concept_surface_results_returned
            .fetch_add(returned as u64, Ordering::Relaxed);
        self.concept_surface_record_results_returned
            .fetch_add(returned as u64, Ordering::Relaxed);
    }

    fn collect_policy_hints_by_state(
        &self,
        state: PolicyState,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<PolicyHint> {
        let max = limit.unwrap_or(20).min(100);
        let mut hints: Vec<_> = self
            .policy_engine
            .hints
            .values()
            .filter(|hint| hint.state == state)
            .filter(|hint| namespace.is_none_or(|value| hint.namespace == value))
            .cloned()
            .collect();
        hints.sort_by(|a, b| {
            b.policy_strength
                .partial_cmp(&a.policy_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.key.cmp(&b.key))
        });
        hints.truncate(max);
        hints
    }
}

fn policy_action_kind_name(action_kind: crate::policy::PolicyActionKind) -> &'static str {
    match action_kind {
        crate::policy::PolicyActionKind::Prefer => "prefer",
        crate::policy::PolicyActionKind::Recommend => "recommend",
        crate::policy::PolicyActionKind::VerifyFirst => "verify_first",
        crate::policy::PolicyActionKind::Avoid => "avoid",
        crate::policy::PolicyActionKind::Warn => "warn",
    }
}

fn policy_action_pressure_weight(action_kind: crate::policy::PolicyActionKind) -> f32 {
    match action_kind {
        crate::policy::PolicyActionKind::Avoid => 1.30,
        crate::policy::PolicyActionKind::Warn => 1.15,
        crate::policy::PolicyActionKind::VerifyFirst => 1.00,
        crate::policy::PolicyActionKind::Recommend => 0.85,
        crate::policy::PolicyActionKind::Prefer => 0.75,
    }
}
