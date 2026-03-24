//! RecallService — internal orchestration layer for recall execution.
//!
//! Keeps recall pipeline orchestration separate from the public `Aura` facade
//! without changing the retrieval primitives in `recall.rs`.

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::audit::AuditLog;
use crate::belief::BeliefEngine;
use crate::cache::{RecallCache, StructuredRecallCache};
use crate::causal::{CausalEngine, CausalRerankMode};
use crate::concept::{ConceptEngine, ConceptSurfaceMode};
use crate::graph::SessionTracker;
use crate::index::InvertedIndex;
use crate::ngram::NGramIndex;
use crate::policy::{PolicyEngine, PolicyRerankMode};
use crate::recall;
use crate::record::Record;
use crate::sdr::SDRInterpreter;
use crate::storage::AuraStorage;
use crate::trust::TrustConfig;

pub(crate) struct RecallPipelineView<'a> {
    pub(crate) sdr: &'a SDRInterpreter,
    pub(crate) index: &'a InvertedIndex,
    pub(crate) storage: &'a AuraStorage,
    pub(crate) ngram: &'a NGramIndex,
    pub(crate) tag_index: &'a HashMap<String, HashSet<String>>,
    pub(crate) aura_index: &'a HashMap<String, String>,
    pub(crate) records: &'a HashMap<String, Record>,
    pub(crate) embedding_ranked: Option<Vec<(String, f32)>>,
    pub(crate) trust_config: Option<&'a TrustConfig>,
}

pub(crate) struct RecallRerankView<'a> {
    pub(crate) belief_engine: &'a BeliefEngine,
    pub(crate) concept_engine: &'a ConceptEngine,
    pub(crate) causal_engine: &'a CausalEngine,
    pub(crate) policy_engine: &'a PolicyEngine,
    pub(crate) belief_mode: recall::BeliefRerankMode,
    pub(crate) concept_mode: ConceptSurfaceMode,
    pub(crate) causal_mode: CausalRerankMode,
    pub(crate) policy_mode: PolicyRerankMode,
}

pub(crate) struct RecallService;

impl RecallService {
    pub(crate) fn text_cache_key(query: &str, namespaces: Option<&[&str]>) -> String {
        let default_ns = [crate::record::DEFAULT_NAMESPACE];
        let ns_list = namespaces.unwrap_or(&default_ns);
        let mut sorted_ns: Vec<&str> = ns_list.to_vec();
        sorted_ns.sort_unstable();
        format!("{}|ns={:?}", query, sorted_ns)
    }

    pub(crate) fn raw(
        view: RecallPipelineView<'_>,
        query: &str,
        top_k: usize,
        min_strength: f32,
        expand_connections: bool,
        namespaces: Option<&[&str]>,
    ) -> Vec<(f32, Record)> {
        recall::recall_pipeline(
            query,
            top_k,
            min_strength,
            expand_connections,
            view.sdr,
            view.index,
            view.storage,
            view.ngram,
            view.tag_index,
            view.aura_index,
            view.records,
            view.embedding_ranked,
            view.trust_config,
            namespaces,
        )
    }

    pub(crate) fn raw_with_trace(
        view: RecallPipelineView<'_>,
        query: &str,
        top_k: usize,
        min_strength: f32,
        expand_connections: bool,
        namespaces: Option<&[&str]>,
    ) -> recall::RecallTraceResult {
        recall::recall_pipeline_with_trace(
            query,
            top_k,
            min_strength,
            expand_connections,
            view.sdr,
            view.index,
            view.storage,
            view.ngram,
            view.tag_index,
            view.aura_index,
            view.records,
            view.embedding_ranked,
            view.trust_config,
            namespaces,
        )
    }

    pub(crate) fn apply_bounded_reranking(
        scored: &mut Vec<(f32, Record)>,
        top_k: usize,
        view: RecallRerankView<'_>,
    ) {
        if view.belief_mode == recall::BeliefRerankMode::Limited {
            let _report = recall::apply_belief_rerank(scored, view.belief_engine, top_k);
        }

        if view.concept_mode == ConceptSurfaceMode::Limited {
            let _report = recall::apply_concept_rerank(scored, view.concept_engine, top_k);
        }

        if view.causal_mode == CausalRerankMode::Limited {
            let _report = recall::apply_causal_rerank(scored, view.causal_engine, top_k);
        }

        if view.policy_mode == PolicyRerankMode::Limited {
            let _report = recall::apply_policy_rerank(scored, view.policy_engine, top_k);
        }
    }

    pub(crate) fn finalize(
        scored: &[(f32, Record)],
        query: &str,
        session_id: Option<&str>,
        records: &mut HashMap<String, Record>,
        tracker: &mut SessionTracker,
        audit_log: Option<&AuditLog>,
    ) {
        recall::activate_and_strengthen(scored, records, tracker, session_id);
        if let Some(log) = audit_log {
            let _ = log.log_retrieve(query, scored.len());
        }
    }

    pub(crate) fn recall_formatted<F, G>(
        cache: &RecallCache,
        query: &str,
        token_budget: usize,
        namespaces: Option<&[&str]>,
        run_core: F,
        format_preamble: G,
    ) -> Result<String>
    where
        F: FnOnce() -> Result<Vec<(f32, Record)>>,
        G: FnOnce(&[(f32, Record)]) -> String,
    {
        let cache_key = Self::text_cache_key(query, namespaces);
        if let Some(cached) = cache.get(&cache_key) {
            return Ok(cached);
        }

        let scored = run_core()?;
        let _ = token_budget;
        let preamble = format_preamble(&scored);
        cache.put(&cache_key, preamble.clone());
        Ok(preamble)
    }

    pub(crate) fn recall_structured_cached<F>(
        cache: &StructuredRecallCache,
        query: &str,
        top_k: usize,
        min_strength: f32,
        namespaces: Option<&[&str]>,
        run_core: F,
    ) -> Result<Vec<(f32, Record)>>
    where
        F: FnOnce() -> Result<Vec<(f32, Record)>>,
    {
        if let Some(cached) = cache.get(query, top_k, min_strength, namespaces) {
            return Ok(cached);
        }

        let scored = run_core()?;
        cache.put(query, top_k, min_strength, namespaces, scored.clone());
        Ok(scored)
    }

    pub(crate) fn shadow_report(
        scored: &[(f32, Record)],
        belief_engine: &BeliefEngine,
        top_k: usize,
    ) -> recall::ShadowRecallReport {
        recall::compute_shadow_belief_scores(scored, belief_engine, top_k)
    }

    pub(crate) fn rerank_report(
        scored: &mut Vec<(f32, Record)>,
        belief_engine: &BeliefEngine,
        top_k: usize,
    ) -> recall::LimitedRerankReport {
        recall::apply_belief_rerank(scored, belief_engine, top_k)
    }

    pub(crate) fn recall_temporal(
        view: RecallPipelineView<'_>,
        query: &str,
        top_k: usize,
        min_strength: f32,
        expand_connections: bool,
        namespaces: Option<&[&str]>,
    ) -> Vec<(f32, Record)> {
        Self::raw(
            view,
            query,
            top_k,
            min_strength,
            expand_connections,
            namespaces,
        )
    }
}
