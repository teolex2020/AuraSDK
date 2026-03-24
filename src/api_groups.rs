//! Public API grouping facades over `Aura`.
//!
//! These facades do not add new behavior. They expose the existing `Aura`
//! surface in clearer conceptual families so callers can navigate the API
//! without scanning the full monolithic method list.

use std::collections::HashMap;

use anyhow::Result;

use crate::aura::{
    Aura, ContradictionReviewCandidate, CorrectionLogEntry, CorrectionReviewCandidate,
    CrossNamespaceDigest, CrossNamespaceDigestOptions, ExplainabilityBundle, MemoryHealthDigest,
    NamespaceGovernanceStatus, ProvenanceChain, RecallExplanation, RecallExplanationItem,
    SalienceSummary, SuggestedCorrection,
};
use crate::background_brain::{
    MaintenanceTrendSnapshot, MaintenanceTrendSummary, ReflectionDigest, ReflectionSummary,
};
use crate::belief::Belief;
use crate::causal::CausalPattern;
use crate::concept::{ConceptCandidate, SurfacedConcept};
use crate::epistemic_runtime::{
    BeliefInstabilitySummary, ContradictionCluster, PolicyLifecycleSummary, PolicyPressureArea,
};
use crate::levels::Level;
use crate::policy::{PolicyHint, SurfacedPolicyHint};
use crate::record::Record;

#[derive(Clone, Copy)]
pub struct MemoryApi<'a> {
    aura: &'a Aura,
}

#[derive(Clone, Copy)]
pub struct ExplainabilityApi<'a> {
    aura: &'a Aura,
}

#[derive(Clone, Copy)]
pub struct CorrectionApi<'a> {
    aura: &'a Aura,
}

#[derive(Clone, Copy)]
pub struct AnalyticsApi<'a> {
    aura: &'a Aura,
}

#[derive(Clone, Copy)]
pub struct OperatorApi<'a> {
    aura: &'a Aura,
}

impl<'a> MemoryApi<'a> {
    pub(crate) fn new(aura: &'a Aura) -> Self {
        Self { aura }
    }

    pub fn store(
        &self,
        content: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        pin: Option<bool>,
        content_type: Option<&str>,
        source_type: Option<&str>,
        metadata: Option<HashMap<String, String>>,
        deduplicate: Option<bool>,
        caused_by_id: Option<&str>,
        namespace: Option<&str>,
        semantic_type: Option<&str>,
    ) -> Result<Record> {
        self.aura.store(
            content,
            level,
            tags,
            pin,
            content_type,
            source_type,
            metadata,
            deduplicate,
            caused_by_id,
            namespace,
            semantic_type,
        )
    }

    pub fn recall(
        &self,
        query: &str,
        token_budget: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespaces: Option<&[&str]>,
    ) -> Result<String> {
        self.aura.recall(
            query,
            token_budget,
            min_strength,
            expand_connections,
            session_id,
            namespaces,
        )
    }

    pub fn search(
        &self,
        query: Option<&str>,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        limit: Option<usize>,
        content_type: Option<&str>,
        source_type: Option<&str>,
        namespaces: Option<&[&str]>,
        semantic_type: Option<&str>,
    ) -> Vec<Record> {
        self.aura.search(
            query,
            level,
            tags,
            limit,
            content_type,
            source_type,
            namespaces,
            semantic_type,
        )
    }

    pub fn get(&self, record_id: &str) -> Option<Record> {
        self.aura.get(record_id)
    }

    pub fn mark_record_salience(
        &self,
        record_id: &str,
        salience: f32,
        reason: Option<&str>,
    ) -> Result<Option<Record>> {
        self.aura.mark_record_salience(record_id, salience, reason)
    }

    pub fn update(
        &self,
        record_id: &str,
        content: Option<&str>,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        strength: Option<f32>,
        metadata: Option<HashMap<String, String>>,
        source_type: Option<&str>,
    ) -> Result<Option<Record>> {
        self.aura.update(
            record_id,
            content,
            level,
            tags,
            strength,
            metadata,
            source_type,
        )
    }
}

impl<'a> ExplainabilityApi<'a> {
    pub(crate) fn new(aura: &'a Aura) -> Self {
        Self { aura }
    }

    pub fn explain_recall(
        &self,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        namespaces: Option<&[&str]>,
    ) -> RecallExplanation {
        self.aura
            .explain_recall(query, top_k, min_strength, expand_connections, namespaces)
    }

    pub fn explain_record(&self, record_id: &str) -> Option<RecallExplanationItem> {
        self.aura.explain_record(record_id)
    }

    pub fn provenance_chain(&self, record_id: &str) -> Option<ProvenanceChain> {
        self.aura.provenance_chain(record_id)
    }

    pub fn explainability_bundle(&self, record_id: &str) -> Option<ExplainabilityBundle> {
        self.aura.explainability_bundle(record_id)
    }

    pub fn get_latest_reflection_digest(&self) -> Option<ReflectionSummary> {
        self.aura.get_latest_reflection_digest()
    }

    pub fn get_reflection_digest(&self, limit: Option<usize>) -> ReflectionDigest {
        self.aura.get_reflection_digest(limit)
    }
}

impl<'a> CorrectionApi<'a> {
    pub(crate) fn new(aura: &'a Aura) -> Self {
        Self { aura }
    }

    pub fn deprecate_belief_with_reason(&self, belief_id: &str, reason: &str) -> Result<bool> {
        self.aura.deprecate_belief_with_reason(belief_id, reason)
    }

    pub fn invalidate_causal_pattern_with_reason(
        &self,
        pattern_id: &str,
        reason: &str,
    ) -> Result<bool> {
        self.aura
            .invalidate_causal_pattern_with_reason(pattern_id, reason)
    }

    pub fn retract_policy_hint_with_reason(&self, hint_id: &str, reason: &str) -> Result<bool> {
        self.aura.retract_policy_hint_with_reason(hint_id, reason)
    }

    pub fn get_correction_log(&self) -> Vec<CorrectionLogEntry> {
        self.aura.get_correction_log()
    }

    pub fn get_correction_log_for_target(
        &self,
        target_kind: &str,
        target_id: &str,
    ) -> Vec<CorrectionLogEntry> {
        self.aura
            .get_correction_log_for_target(target_kind, target_id)
    }

    pub fn get_correction_review_queue(
        &self,
        limit: Option<usize>,
    ) -> Vec<CorrectionReviewCandidate> {
        self.aura.get_correction_review_queue(limit)
    }

    pub fn get_contradiction_review_queue(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<ContradictionReviewCandidate> {
        self.aura.get_contradiction_review_queue(namespace, limit)
    }

    pub fn get_suggested_corrections(&self, limit: Option<usize>) -> Vec<SuggestedCorrection> {
        self.aura.get_suggested_corrections(limit)
    }

    pub fn get_namespace_governance_status(
        &self,
        namespaces: Option<&[&str]>,
    ) -> Vec<NamespaceGovernanceStatus> {
        self.aura
            .get_namespace_governance_status_filtered(namespaces)
    }
}

impl<'a> AnalyticsApi<'a> {
    pub(crate) fn new(aura: &'a Aura) -> Self {
        Self { aura }
    }

    pub fn cross_namespace_digest(&self) -> CrossNamespaceDigest {
        self.aura.cross_namespace_digest()
    }

    pub fn cross_namespace_digest_filtered(
        &self,
        namespaces: Option<&[&str]>,
        top_concepts_limit: Option<usize>,
    ) -> CrossNamespaceDigest {
        self.aura
            .cross_namespace_digest_filtered(namespaces, top_concepts_limit)
    }

    pub fn cross_namespace_digest_with_options(
        &self,
        namespaces: Option<&[&str]>,
        options: CrossNamespaceDigestOptions,
    ) -> CrossNamespaceDigest {
        self.aura
            .cross_namespace_digest_with_options(namespaces, options)
    }

    pub fn get_maintenance_trend_history(&self) -> Vec<MaintenanceTrendSnapshot> {
        self.aura.get_maintenance_trend_history()
    }

    pub fn get_maintenance_trend_summary(&self) -> MaintenanceTrendSummary {
        self.aura.get_maintenance_trend_summary()
    }

    pub fn get_reflection_summaries(&self, limit: Option<usize>) -> Vec<ReflectionSummary> {
        self.aura.get_reflection_summaries(limit)
    }

    pub fn get_reflection_digest(&self, limit: Option<usize>) -> ReflectionDigest {
        self.aura.get_reflection_digest(limit)
    }

    pub fn get_analytics(&self) -> (HashMap<String, usize>, usize, f64, f64) {
        self.aura.get_analytics()
    }
}

impl<'a> OperatorApi<'a> {
    pub(crate) fn new(aura: &'a Aura) -> Self {
        Self { aura }
    }

    pub fn get_beliefs(&self, state_filter: Option<&str>) -> Vec<Belief> {
        self.aura.get_beliefs(state_filter)
    }

    pub fn get_high_volatility_beliefs(
        &self,
        min_volatility: Option<f32>,
        limit: Option<usize>,
    ) -> Vec<Belief> {
        self.aura.get_high_volatility_beliefs(min_volatility, limit)
    }

    pub fn get_low_stability_beliefs(
        &self,
        max_stability: Option<f32>,
        limit: Option<usize>,
    ) -> Vec<Belief> {
        self.aura.get_low_stability_beliefs(max_stability, limit)
    }

    pub fn get_belief_instability_summary(&self) -> BeliefInstabilitySummary {
        self.aura.get_belief_instability_summary()
    }

    pub fn get_contradiction_clusters(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<ContradictionCluster> {
        self.aura.get_contradiction_clusters(namespace, limit)
    }

    pub fn get_recently_corrected_beliefs(&self, limit: Option<usize>) -> Vec<Belief> {
        self.aura.get_recently_corrected_beliefs(limit)
    }

    pub fn get_high_salience_records(
        &self,
        min_salience: Option<f32>,
        limit: Option<usize>,
    ) -> Vec<Record> {
        self.aura.get_high_salience_records(min_salience, limit)
    }

    pub fn get_salience_summary(&self) -> SalienceSummary {
        self.aura.get_salience_summary()
    }

    pub fn get_concepts(&self, state_filter: Option<&str>) -> Vec<ConceptCandidate> {
        self.aura.get_concepts(state_filter)
    }

    pub fn get_surfaced_concepts(&self, limit: Option<usize>) -> Vec<SurfacedConcept> {
        self.aura.get_surfaced_concepts(limit)
    }

    pub fn get_causal_patterns(&self, state_filter: Option<&str>) -> Vec<CausalPattern> {
        self.aura.get_causal_patterns(state_filter)
    }

    pub fn get_policy_hints(&self, state_filter: Option<&str>) -> Vec<PolicyHint> {
        self.aura.get_policy_hints(state_filter)
    }

    pub fn get_suppressed_policy_hints(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<PolicyHint> {
        self.aura.get_suppressed_policy_hints(namespace, limit)
    }

    pub fn get_rejected_policy_hints(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<PolicyHint> {
        self.aura.get_rejected_policy_hints(namespace, limit)
    }

    pub fn get_policy_lifecycle_summary(
        &self,
        action_limit: Option<usize>,
        domain_limit: Option<usize>,
    ) -> PolicyLifecycleSummary {
        self.aura
            .get_policy_lifecycle_summary(action_limit, domain_limit)
    }

    pub fn get_policy_pressure_report(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<PolicyPressureArea> {
        self.aura.get_policy_pressure_report(namespace, limit)
    }

    pub fn get_memory_health_digest(&self, limit: Option<usize>) -> MemoryHealthDigest {
        self.aura.get_memory_health_digest(limit)
    }

    pub fn get_latest_reflection_digest(&self) -> Option<ReflectionSummary> {
        self.aura.get_latest_reflection_digest()
    }

    pub fn get_reflection_digest(&self, limit: Option<usize>) -> ReflectionDigest {
        self.aura.get_reflection_digest(limit)
    }

    pub fn get_correction_review_queue(
        &self,
        limit: Option<usize>,
    ) -> Vec<CorrectionReviewCandidate> {
        self.aura.get_correction_review_queue(limit)
    }

    pub fn get_suggested_corrections(&self, limit: Option<usize>) -> Vec<SuggestedCorrection> {
        self.aura.get_suggested_corrections(limit)
    }

    pub fn get_surfaced_policy_hints(&self, limit: Option<usize>) -> Vec<SurfacedPolicyHint> {
        self.aura.get_surfaced_policy_hints(limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_api_facades_match_aura_surface() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let record = aura.memory_api().store(
            "Grouped API smoke record",
            Some(Level::Working),
            Some(vec!["grouped-api".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;

        let fetched = aura.memory_api().get(&record.id);
        assert_eq!(
            fetched.as_ref().map(|item| item.id.as_str()),
            Some(record.id.as_str())
        );

        let search = aura.memory_api().search(
            Some("Grouped API smoke"),
            None,
            None,
            Some(5),
            None,
            None,
            None,
            None,
        );
        assert_eq!(search.len(), 1);

        let explanation = aura.explainability_api().explain_record(&record.id);
        assert_eq!(
            explanation.as_ref().map(|item| item.record_id.as_str()),
            Some(record.id.as_str())
        );

        let digest = aura.analytics_api().cross_namespace_digest();
        assert_eq!(digest.namespaces.len(), 1);

        let lifecycle = aura
            .operator_api()
            .get_policy_lifecycle_summary(Some(5), Some(5));
        assert_eq!(lifecycle.total_hints, 0);

        let corrections = aura.correction_api().get_correction_log();
        assert!(corrections.is_empty());

        Ok(())
    }
}
