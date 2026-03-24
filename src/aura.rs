//! Aura — Unified Cognitive Memory orchestrator.
//!
//! This is the SINGLE entry point. Replaces both `AuraMemory` (Rust)
//! and `CognitiveMemory` (Python).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::instrument;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::api_groups::{AnalyticsApi, CorrectionApi, ExplainabilityApi, MemoryApi, OperatorApi};
use crate::audit::AuditLog;
use crate::aura_state::{AuraConfigState, AuraRuntimeState};
use crate::canonical::CanonicalProjector;
use crate::cognitive_store::CognitiveStore;
use crate::consolidation;
use crate::cortex::ActiveCortex;
use crate::crypto::EncryptionKey;
use crate::embedding::EmbeddingStore;
use crate::epistemic_runtime::EpistemicRuntime;
use crate::graph::SessionTracker;
use crate::index::InvertedIndex;
use crate::insights;
use crate::levels::Level;
use crate::maintenance_service::MaintenanceService;
use crate::ngram::NGramIndex;
use crate::recall;
use crate::recall_service::{RecallPipelineView, RecallRerankView, RecallService};
use crate::record::Record;
use crate::sdr::SDRInterpreter;
use crate::semantic_learner::SemanticLearnerEngine;
use crate::storage::AuraStorage;
use crate::synonym::SynonymRing;

// SDK Wrapper modules
use crate::background_brain::{self, BackgroundBrain, MaintenanceConfig, MaintenanceReport};
use crate::belief::{BeliefEngine, BeliefStore, CoarseKeyMode, SdrLookup};
use crate::causal::{
    CausalEngine, CausalEvidenceMode, CausalRerankMode, CausalStore, TemporalEdgeBudgetMode,
};
use crate::circuit_breaker::CircuitBreakerConfig;
use crate::concept::{
    ConceptEngine, ConceptPartitionMode, ConceptSeedMode, ConceptSimilarityMode, ConceptStore,
    ConceptSurfaceMode,
};
use crate::guards;
use crate::identity::{self, AgentPersona};
use crate::persistence_contract::{PersistenceManifest, PERSISTENCE_MANIFEST_FILE};
use crate::policy::{PolicyEngine, PolicyRerankMode, PolicyStore};
use crate::relation::{
    self, EntityDigest, EntityGraphDigest, EntityGraphNeighbor, EntityRelationEdge,
    FamilyGraphSnapshot, FamilyRelationMember, PersonDigest, ProjectDigest, ProjectGraphSnapshot,
    ProjectStatusSnapshot, ProjectTimelineEntry, ProjectTimelineSnapshot, RelationDigest,
    RelationEdge, StructuralRelation,
};
use crate::research::{ResearchEngine, ResearchProject};
use crate::startup_validation::{StartupValidationEvent, StartupValidationReport};
use crate::storage::StoredRecord;
use crate::trust::{self, TagTaxonomy, TrustConfig};

/// Maximum content size (100KB).
const MAX_CONTENT_SIZE: usize = 100 * 1024;
const MAINTENANCE_TRENDS_FILE: &str = "maintenance_trends.json";
const REFLECTION_SUMMARIES_FILE: &str = "reflection_summaries.json";
/// Maximum tags per record.
const MAX_TAGS: usize = 50;
const STRUCTURAL_RELATION_DEFAULT_LIMIT: usize = 32;
const ENTITY_RELATION_PROMOTION_MIN_WEIGHT: f32 = 0.8;
/// Surprise threshold — below this similarity, info is considered novel.
const SURPRISE_THRESHOLD: f32 = 0.2;
const RECORD_SALIENCE_REASON_KEY: &str = "salience_reason";
const RECORD_SALIENCE_MARKED_AT_KEY: &str = "salience_marked_at";
const CONTRADICTION_REVIEW_PRIORITY_MAX: f32 = 10.0;

/// Unified cognitive memory for AI agents.
#[cfg_attr(feature = "python", pyclass)]
pub struct Aura {
    // ── From aura-memory (already Rust) ──
    sdr: SDRInterpreter,
    storage: Arc<AuraStorage>,
    index: Arc<InvertedIndex>,
    cortex: Arc<ActiveCortex>,

    // ── From aura-cognitive (rewritten to Rust) ──
    records: RwLock<HashMap<String, Record>>,
    cognitive_store: CognitiveStore,
    ngram_index: RwLock<NGramIndex>,
    tag_index: RwLock<HashMap<String, HashSet<String>>>,
    synonym_ring: RwLock<SynonymRing>,
    session_tracker: RwLock<SessionTracker>,
    #[allow(dead_code)]
    learner: RwLock<Option<SemanticLearnerEngine>>,

    // ── Shared ──
    #[allow(dead_code)]
    canonical: RwLock<Option<CanonicalProjector>>,
    encryption_key: Option<EncryptionKey>,
    audit_log: Option<AuditLog>,

    // ── Bridge: aura_id → record_id ──
    aura_index: RwLock<HashMap<String, String>>,

    // ── SDK Wrapper Layer ──
    research_engine: ResearchEngine,
    config: AuraConfigState,
    runtime: AuraRuntimeState,

    // ── Epistemic Belief Layer ──
    belief_engine: RwLock<BeliefEngine>,
    belief_store: BeliefStore,

    // ── Concept Discovery Layer ──
    concept_engine: RwLock<ConceptEngine>,
    concept_store: ConceptStore,

    // ── Causal Pattern Discovery Layer ──
    causal_engine: RwLock<CausalEngine>,
    causal_store: CausalStore,

    // ── Policy Hint Layer ──
    policy_engine: RwLock<PolicyEngine>,
    policy_store: PolicyStore,

    // ── Cross-cycle stability tracking ──
    prev_belief_keys: RwLock<HashSet<String>>,
    prev_concept_keys: RwLock<HashSet<String>>,
    prev_causal_keys: RwLock<HashSet<String>>,
    prev_policy_keys: RwLock<HashSet<String>>,

    // ── Optional Embedding Support ──
    embedding_store: EmbeddingStore,
    #[cfg(feature = "python")]
    embedding_fn: RwLock<Option<PyObject>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallBeliefExplanation {
    pub id: String,
    pub state: String,
    pub confidence: f32,
    pub support_mass: f32,
    pub conflict_mass: f32,
    pub stability: f32,
    pub volatility: f32,
    pub has_unresolved_evidence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallConceptExplanation {
    pub id: String,
    pub key: String,
    pub state: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallCausalExplanation {
    pub id: String,
    pub key: String,
    pub state: String,
    pub causal_strength: f32,
    pub invalidation_reason: Option<String>,
    pub invalidated_at: Option<f64>,
    pub corrections: Vec<CorrectionLogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallPolicyExplanation {
    pub id: String,
    pub key: String,
    pub state: String,
    pub action_kind: String,
    pub policy_strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallSignalScore {
    pub raw_score: f32,
    pub rank: usize,
    pub rrf_share: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallTraceScore {
    pub sdr: Option<RecallSignalScore>,
    pub ngram: Option<RecallSignalScore>,
    pub tags: Option<RecallSignalScore>,
    pub embedding: Option<RecallSignalScore>,
    pub rrf_score: f32,
    pub graph_score: f32,
    pub causal_score: f32,
    pub pre_trust_score: f32,
    pub trust_multiplier: f32,
    pub pre_rerank_score: f32,
    pub rerank_delta: f32,
    pub final_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HonestAnswerSupport {
    pub significance_phrase: Option<String>,
    pub uncertainty_phrase: Option<String>,
    pub contradiction_phrase: Option<String>,
    pub reflection_phrase: Option<String>,
    pub recommended_framing: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallExplanationItem {
    pub rank: usize,
    pub record_id: String,
    pub score: f32,
    pub namespace: String,
    pub salience: f32,
    pub salience_reason: Option<String>,
    pub salience_explanation: Option<String>,
    pub content_preview: String,
    pub because_record_id: Option<String>,
    pub because_preview: Option<String>,
    pub belief: Option<RecallBeliefExplanation>,
    pub has_unresolved_evidence: bool,
    pub honesty_note: Option<String>,
    pub contradiction_dependency: bool,
    pub reflection_references: Vec<String>,
    pub answer_support: HonestAnswerSupport,
    pub concepts: Vec<RecallConceptExplanation>,
    pub causal_patterns: Vec<RecallCausalExplanation>,
    pub policy_hints: Vec<RecallPolicyExplanation>,
    pub trace: RecallTraceScore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallExplanation {
    pub query: String,
    pub top_k: usize,
    pub result_count: usize,
    pub latency_ms: f64,
    pub belief_rerank_mode: String,
    pub concept_surface_mode: String,
    pub causal_rerank_mode: String,
    pub policy_rerank_mode: String,
    pub items: Vec<RecallExplanationItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossNamespaceConceptSummary {
    pub concept_id: String,
    pub key: String,
    pub confidence: f32,
    pub state: String,
    pub record_count: usize,
    pub belief_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossNamespaceBeliefStateSummary {
    pub resolved: usize,
    pub unresolved: usize,
    pub singleton: usize,
    pub empty: usize,
    pub high_volatility_count: usize,
    pub avg_volatility: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossNamespaceNamespaceDigest {
    pub namespace: String,
    pub record_count: usize,
    pub concept_count: usize,
    pub stable_concept_count: usize,
    pub top_concepts: Vec<CrossNamespaceConceptSummary>,
    pub concept_signatures: Vec<String>,
    pub tags: Vec<String>,
    pub structural_relation_types: Vec<String>,
    pub causal_signatures: Vec<String>,
    pub belief_state_summary: Option<CrossNamespaceBeliefStateSummary>,
    pub correction_count: Option<usize>,
    pub correction_density: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossNamespacePairDigest {
    pub namespace_a: String,
    pub namespace_b: String,
    pub shared_concept_signatures: Vec<String>,
    pub concept_signature_similarity: f32,
    pub shared_tags: Vec<String>,
    pub tag_jaccard: f32,
    pub shared_structural_relation_types: Vec<String>,
    pub structural_similarity: f32,
    pub shared_causal_signatures: Vec<String>,
    pub causal_signature_similarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossNamespaceDigestOptions {
    pub min_record_count: usize,
    pub top_concepts_limit: usize,
    pub pairwise_similarity_threshold: f32,
    pub compact_summary: bool,
    pub include_concepts: bool,
    pub include_tags: bool,
    pub include_structural: bool,
    pub include_causal: bool,
    pub include_belief_states: bool,
    pub include_corrections: bool,
}

impl Default for CrossNamespaceDigestOptions {
    fn default() -> Self {
        Self {
            min_record_count: 1,
            top_concepts_limit: 5,
            pairwise_similarity_threshold: 0.0,
            compact_summary: false,
            include_concepts: true,
            include_tags: true,
            include_structural: true,
            include_causal: true,
            include_belief_states: true,
            include_corrections: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossNamespaceDigest {
    pub namespace_count: usize,
    pub latency_ms: f64,
    pub compact_summary: bool,
    pub included_dimensions: Vec<String>,
    pub namespaces: Vec<CrossNamespaceNamespaceDigest>,
    pub pairs: Vec<CrossNamespacePairDigest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChain {
    pub record_id: String,
    pub namespace: String,
    pub content_preview: String,
    pub build_latency_ms: f64,
    pub because_record_id: Option<String>,
    pub because_preview: Option<String>,
    pub belief: Option<RecallBeliefExplanation>,
    pub concepts: Vec<RecallConceptExplanation>,
    pub causal_patterns: Vec<RecallCausalExplanation>,
    pub policy_hints: Vec<RecallPolicyExplanation>,
    pub steps: Vec<String>,
    pub narrative: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainabilityBundle {
    pub record_id: String,
    pub explanation: RecallExplanationItem,
    pub provenance: ProvenanceChain,
    pub record_corrections: Vec<CorrectionLogEntry>,
    pub belief_corrections: Vec<CorrectionLogEntry>,
    pub causal_corrections: Vec<CorrectionLogEntry>,
    pub policy_corrections: Vec<CorrectionLogEntry>,
    pub belief_instability: crate::epistemic_runtime::BeliefInstabilitySummary,
    pub reflection_digest: background_brain::ReflectionDigest,
    pub related_reflection_findings: Vec<background_brain::ReflectionFinding>,
    pub maintenance_trends: background_brain::MaintenanceTrendSummary,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SalienceBands {
    pub low: usize,
    pub medium: usize,
    pub high: usize,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SalienceSummary {
    pub total_records: usize,
    pub high_salience_count: usize,
    pub avg_salience: f32,
    pub max_salience: f32,
    pub bands: SalienceBands,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionLogEntry {
    pub timestamp: u64,
    pub time_iso: String,
    pub target_kind: String,
    pub target_id: String,
    pub operation: String,
    pub reason: String,
    pub session_id: String,
}

impl CorrectionLogEntry {
    fn matches_target(&self, target_kind: &str, target_id: &str) -> bool {
        self.target_kind == target_kind && self.target_id == target_id
    }
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperatorReviewIssue {
    pub kind: String,
    pub target_id: String,
    pub namespace: String,
    pub title: String,
    pub score: f32,
    pub severity: String,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryHealthDigest {
    pub total_records: usize,
    pub startup_has_recovery_warnings: bool,
    pub high_salience_record_count: usize,
    pub avg_salience: f32,
    pub max_salience: f32,
    pub reflection_summary_count: usize,
    pub reflection_high_severity_findings: usize,
    pub contradiction_cluster_count: usize,
    pub high_volatility_belief_count: usize,
    pub low_stability_belief_count: usize,
    pub recent_correction_count: usize,
    pub suppressed_policy_hint_count: usize,
    pub rejected_policy_hint_count: usize,
    pub policy_pressure_area_count: usize,
    pub maintenance_trend_direction: String,
    pub latest_dominant_phase: String,
    pub top_issues: Vec<OperatorReviewIssue>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrectionReviewCandidate {
    pub timestamp: u64,
    pub time_iso: String,
    pub target_kind: String,
    pub target_id: String,
    pub operation: String,
    pub reason: String,
    pub session_id: String,
    pub namespace: String,
    pub title: String,
    pub repeat_count: usize,
    pub dependent_causal_patterns: usize,
    pub dependent_policy_hints: usize,
    pub downstream_impact: usize,
    pub priority_score: f32,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionReviewCandidate {
    pub cluster_id: String,
    pub namespace: String,
    pub title: String,
    pub belief_ids: Vec<String>,
    pub belief_keys: Vec<String>,
    pub record_ids: Vec<String>,
    pub shared_tags: Vec<String>,
    pub unresolved_belief_count: usize,
    pub high_volatility_belief_count: usize,
    pub dependent_causal_patterns: usize,
    pub dependent_policy_hints: usize,
    pub downstream_impact: usize,
    pub total_conflict_mass: f32,
    pub avg_volatility: f32,
    pub avg_stability: f32,
    pub priority_score: f32,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedCorrection {
    pub target_kind: String,
    pub target_id: String,
    pub namespace: String,
    pub reason_kind: String,
    pub suggested_action: String,
    pub reason_detail: String,
    pub priority_score: f32,
    pub severity: String,
    pub supporting_record_id: Option<String>,
    pub provenance: Option<ProvenanceChain>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedCorrectionsReport {
    pub scan_latency_ms: f64,
    pub entries: Vec<SuggestedCorrection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceGovernanceStatus {
    pub namespace: String,
    pub record_count: usize,
    pub belief_count: usize,
    pub correction_count: usize,
    pub correction_density: f32,
    pub high_volatility_belief_count: usize,
    pub low_stability_belief_count: usize,
    pub instability_score: f32,
    pub instability_level: String,
    pub policy_pressure_area_count: usize,
    pub suggested_correction_count: usize,
    pub last_maintenance_cycle: Option<String>,
    pub latest_dominant_phase: String,
}

impl Aura {
    /// Create a new Aura instance at the given path.
    pub fn open(path: &str) -> Result<Self> {
        Self::open_with_password(path, None)
    }

    /// Create a new Aura instance with optional encryption.
    pub fn open_with_password(path: &str, password: Option<&str>) -> Result<Self> {
        let path_buf = PathBuf::from(path);
        std::fs::create_dir_all(&path_buf)?;

        // Initialize aura-memory components
        let encryption_key = if let Some(pwd) = password {
            let salt = crate::crypto::generate_salt();
            Some(EncryptionKey::from_password(pwd, &salt)?)
        } else {
            None
        };

        let storage = Arc::new(if let Some(ref key) = encryption_key {
            AuraStorage::with_encryption(&path_buf, Some(key.clone()))?
        } else {
            AuraStorage::new(&path_buf)?
        });

        let index_path = path_buf.join("index");
        let index = Arc::new(InvertedIndex::new(&index_path));
        let _ = index.load();

        let cortex = Arc::new(ActiveCortex::new());
        let sdr = SDRInterpreter::default();

        // Initialize cognitive components
        let cognitive_store = CognitiveStore::new(&path_buf)?;
        let mut startup_events = Vec::new();
        let mut loaded_records = cognitive_store.load_all().with_context(|| {
            format!(
                "failed to load persisted records from {}",
                path_buf.display()
            )
        })?;
        startup_events.push(startup_event(
            "records",
            path_buf.display().to_string(),
            "loaded",
            Some(format!("loaded {} records", loaded_records.len())),
            false,
        ));

        // Fix legacy records: confidence defaults to 0.90 on deserialization,
        // but should match the stored source_type for non-"recorded" records.
        let mut migrated_ids: Vec<String> = Vec::new();
        for rec in loaded_records.values_mut() {
            let expected = Record::default_confidence_for_source(&rec.source_type);
            // Only fix if confidence is at the default 0.90 and source_type disagrees
            if (rec.confidence - 0.90).abs() < 0.001 && (expected - 0.90).abs() > 0.001 {
                rec.confidence = expected;
                migrated_ids.push(rec.id.clone());
            }
        }
        // Persist only the migrated records
        if !migrated_ids.is_empty() {
            for id in &migrated_ids {
                if let Some(rec) = loaded_records.get(id) {
                    let _ = cognitive_store.append_update(rec);
                }
            }
            tracing::info!(
                count = migrated_ids.len(),
                "Migrated legacy record confidence values"
            );
        }

        // Build indexes from loaded records
        let mut ngram_index = NGramIndex::new(None, None);
        let mut tag_index: HashMap<String, HashSet<String>> = HashMap::new();
        let mut aura_index: HashMap<String, String> = HashMap::new();

        for rec in loaded_records.values() {
            ngram_index.add(&rec.id, &rec.content);

            for tag in &rec.tags {
                tag_index
                    .entry(tag.clone())
                    .or_default()
                    .insert(rec.id.clone());
            }

            if let Some(ref aura_id) = rec.aura_id {
                aura_index.insert(aura_id.clone(), rec.id.clone());
            }
        }

        // Audit log
        let audit_log = AuditLog::new(&path_buf).ok();

        // Epistemic belief layer
        let belief_store = BeliefStore::new(&path_buf);
        let belief_path = path_buf.join("beliefs.cog");
        let belief_engine =
            load_belief_engine_with_validation(&belief_store, &belief_path, &mut startup_events);

        // Concept discovery layer
        // Concepts are derived state but are persisted to concepts.cog so that
        // the last completed maintenance cycle's output is available immediately
        // on re-open (e.g. for inspection, reranking). If the file is missing
        // or corrupt we fall back to an empty engine — maintenance rebuilds it.
        let concept_store = ConceptStore::new(&path_buf);
        let concept_path = path_buf.join("concepts.cog");
        let concept_engine =
            load_concept_engine_with_validation(&concept_store, &concept_path, &mut startup_events);

        // Causal pattern discovery layer
        // Causal patterns are persisted runtime state and are loaded on startup.
        let causal_store = CausalStore::new(&path_buf);
        let causal_path = path_buf.join("causal.cog");
        let causal_engine =
            load_causal_engine_with_validation(&causal_store, &causal_path, &mut startup_events);

        // Policy hint layer
        // Policy hints are persisted runtime state and are loaded on startup.
        let policy_store = PolicyStore::new(&path_buf);
        let policy_path = path_buf.join("policies.cog");
        let policy_engine =
            load_policy_engine_with_validation(&policy_store, &policy_path, &mut startup_events);

        let persistence_manifest =
            load_persistence_manifest_with_validation(&path_buf, &mut startup_events);

        let runtime = AuraRuntimeState::new();
        let maintenance_trends =
            load_maintenance_trends_with_validation(&path_buf, &mut startup_events);
        let reflection_summaries =
            load_reflection_summaries_with_validation(&path_buf, &mut startup_events);
        *runtime.maintenance_trends.write() = maintenance_trends;
        *runtime.reflection_summaries.write() = reflection_summaries;
        *runtime.persistence_manifest.write() = persistence_manifest;
        *runtime.startup_validation.write() = finalize_startup_validation_report(startup_events);

        Ok(Self {
            sdr,
            storage,
            index,
            cortex,
            records: RwLock::new(loaded_records),
            cognitive_store,
            ngram_index: RwLock::new(ngram_index),
            tag_index: RwLock::new(tag_index),
            synonym_ring: RwLock::new(SynonymRing::new()),
            session_tracker: RwLock::new(SessionTracker::new()),
            learner: RwLock::new(None),
            canonical: RwLock::new(None),
            encryption_key,
            audit_log,
            aura_index: RwLock::new(aura_index),
            research_engine: ResearchEngine::new(),
            config: AuraConfigState::new(path_buf.clone()),
            runtime,
            // Epistemic belief layer
            belief_engine: RwLock::new(belief_engine),
            belief_store,
            // Concept discovery layer
            concept_engine: RwLock::new(concept_engine),
            concept_store,
            // Causal pattern discovery layer
            causal_engine: RwLock::new(causal_engine),
            causal_store,
            // Policy hint layer
            policy_engine: RwLock::new(policy_engine),
            policy_store,
            // Cross-cycle stability tracking
            prev_belief_keys: RwLock::new(HashSet::new()),
            prev_concept_keys: RwLock::new(HashSet::new()),
            prev_causal_keys: RwLock::new(HashSet::new()),
            prev_policy_keys: RwLock::new(HashSet::new()),
            // Optional embedding support
            embedding_store: EmbeddingStore::new(),
            #[cfg(feature = "python")]
            embedding_fn: RwLock::new(None),
        })
    }

    // ── Core Operations ──

    /// Grouped memory-operation facade over the existing `Aura` API.
    pub fn memory_api(&self) -> MemoryApi<'_> {
        MemoryApi::new(self)
    }

    /// Grouped explainability facade over the existing `Aura` API.
    pub fn explainability_api(&self) -> ExplainabilityApi<'_> {
        ExplainabilityApi::new(self)
    }

    /// Grouped correction and audit-log facade over the existing `Aura` API.
    pub fn correction_api(&self) -> CorrectionApi<'_> {
        CorrectionApi::new(self)
    }

    /// Grouped analytics facade over the existing `Aura` API.
    pub fn analytics_api(&self) -> AnalyticsApi<'_> {
        AnalyticsApi::new(self)
    }

    /// Grouped operator/runtime-inspection facade over the existing `Aura` API.
    pub fn operator_api(&self) -> OperatorApi<'_> {
        OperatorApi::new(self)
    }

    /// Store a memory with automatic guards (provenance, auto-protect, dedup).
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
        self.store_with_channel(
            content,
            level,
            tags,
            pin,
            content_type,
            source_type,
            metadata,
            deduplicate,
            caused_by_id,
            None,
            None,
            namespace,
            semantic_type,
        )
    }

    /// Store with explicit channel for provenance stamping.
    /// `auto_promote`: if Some(false), disables surprise-based level promotion.
    #[instrument(skip(self, content, metadata), fields(level, namespace, tag_count))]
    pub fn store_with_channel(
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
        channel: Option<&str>,
        auto_promote: Option<bool>,
        namespace: Option<&str>,
        semantic_type: Option<&str>,
    ) -> Result<Record> {
        // Validation
        if content.is_empty() {
            return Err(anyhow::anyhow!("Content cannot be empty"));
        }
        if content.len() > MAX_CONTENT_SIZE {
            return Err(anyhow::anyhow!("Content exceeds maximum size of 100KB"));
        }

        let level = level.unwrap_or(Level::Working);
        let mut tags = tags.unwrap_or_default();
        if tags.len() > MAX_TAGS {
            return Err(anyhow::anyhow!("Maximum {} tags allowed", MAX_TAGS));
        }

        let pin = pin.unwrap_or(false);
        let content_type = content_type.unwrap_or("text");
        let source_type = source_type.unwrap_or(crate::record::DEFAULT_SOURCE_TYPE);
        crate::record::Record::validate_source_type(source_type).map_err(|e| anyhow::anyhow!(e))?;
        let deduplicate = deduplicate.unwrap_or(true);
        let semantic_type = semantic_type.unwrap_or(crate::record::DEFAULT_SEMANTIC_TYPE);
        crate::record::Record::validate_semantic_type(semantic_type)
            .map_err(|e| anyhow::anyhow!(e))?;

        // ── Namespace resolution & validation ──
        let ns = namespace.unwrap_or(crate::record::DEFAULT_NAMESPACE);
        crate::record::Record::validate_namespace(ns).map_err(|e| anyhow::anyhow!(e))?;

        // ── Guard: Auto-protect tags (detect sensitive content) ──
        guards::auto_protect_tags(content, &mut tags);

        // ── Guard: Sensitive tag check ──
        let taxonomy = self.config.taxonomy.read();
        let guard_result = guards::apply_store_guard(content, &tags, channel, &taxonomy);

        // Deduplication check
        if deduplicate && content_type == "text" && content.len() >= 20 {
            let ngram = self.ngram_index.read();
            let matches = ngram.query(content, 1);
            if let Some((sim, existing_id)) = matches.first() {
                if *sim >= 0.85 {
                    // Strong match — activate existing record instead (only within same namespace)
                    let mut records = self.records.write();
                    if let Some(existing) = records.get_mut(existing_id) {
                        if existing.namespace == ns {
                            existing.activate();
                            // Merge tags
                            for tag in &tags {
                                if !existing.tags.contains(tag) {
                                    existing.tags.push(tag.clone());
                                }
                            }
                            self.cognitive_store.append_update(existing)?;
                            // Invalidate recall cache on write
                            self.runtime.recall_cache.clear();
                            self.runtime.structured_recall_cache.clear();
                            return Ok(existing.clone());
                        }
                    }
                }
            }
        }

        // Surprise detection (skipped when auto_promote=false)
        let mut effective_level = level;
        if auto_promote.unwrap_or(true) {
            let records = self.records.read();
            if records.len() >= 5 {
                let ngram = self.ngram_index.read();
                let matches = ngram.query(content, 1);
                let best_sim = matches.first().map(|(s, _)| *s).unwrap_or(0.0);
                if best_sim < SURPRISE_THRESHOLD {
                    // Novel information — promote
                    if let Some(promoted) = effective_level.promote() {
                        effective_level = promoted;
                    }
                }
            }
        }

        // Create record
        let mut rec = Record::new(content.to_string(), effective_level);
        rec.tags = tags;
        rec.content_type = content_type.to_string();
        rec.source_type = source_type.to_string();
        // Recompute confidence from actual source_type (Record::new() defaults to "recorded")
        rec.confidence = Record::default_confidence_for_source(source_type);
        if let Some(meta) = metadata {
            rec.metadata = meta;
        }
        if let Some(parent_id) = caused_by_id {
            rec.caused_by_id = Some(parent_id.to_string());
        }
        rec.namespace = ns.to_string();
        rec.semantic_type = semantic_type.to_string();

        // ── Guard: Stamp provenance ──
        {
            let trust_config = self.config.trust_config.read();
            trust::stamp_provenance(
                &mut rec.metadata,
                channel,
                &rec.tags,
                &taxonomy,
                &trust_config,
            );
        }

        // ── Guard: Apply guard result metadata ──
        for (key, value) in &guard_result.extra_metadata {
            rec.metadata
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
        for extra_tag in &guard_result.extra_tags {
            if !rec.tags.contains(extra_tag) {
                rec.tags.push(extra_tag.clone());
            }
        }

        // Drop taxonomy lock before acquiring other locks
        drop(taxonomy);

        // SDR processing
        let is_identity = effective_level.is_identity_sdr();
        let sdr_indices = self.sdr.text_to_sdr(content, is_identity);
        self.index.add(&rec.id, &sdr_indices);

        // Store in aura storage
        let stored_record = crate::storage::StoredRecord {
            id: rec.id.clone(),
            dna: effective_level.to_dna().to_string(),
            timestamp: rec.created_at,
            intensity: rec.strength,
            stability: if pin { 100.0 } else { 1.0 },
            decay_velocity: 0.0,
            entropy: 0.0,
            sdr_indices: sdr_indices.clone(),
            text: content.to_string(),
            offset: 0,
        };
        self.storage.append(&stored_record)?;
        rec.aura_id = Some(rec.id.clone()); // In unified SDK, aura_id == record_id

        // Index in ngram
        {
            let mut ngram = self.ngram_index.write();
            ngram.add(&rec.id, content);
        }

        // Index tags
        {
            let mut tag_idx = self.tag_index.write();
            for tag in &rec.tags {
                tag_idx
                    .entry(tag.clone())
                    .or_default()
                    .insert(rec.id.clone());
            }
        }

        // Update aura_index
        {
            let mut ai = self.aura_index.write();
            ai.insert(rec.id.clone(), rec.id.clone());
        }

        // Causal link
        if let Some(parent_id) = caused_by_id {
            let mut records = self.records.write();
            if let Some(parent) = records.get_mut(parent_id) {
                parent.add_typed_connection(&rec.id, 0.7, "causal");
            }
            rec.add_typed_connection(parent_id, 0.7, "causal");
        }

        // Auto-connect by tags
        {
            let mut records = self.records.write();
            let tag_idx = self.tag_index.read();
            crate::graph::auto_connect(&mut rec, &tag_idx, &mut records);
        }

        // Persist
        self.cognitive_store.append_store(&rec)?;

        // Add to records
        {
            let mut records = self.records.write();
            records.insert(rec.id.clone(), rec.clone());
        }
        {
            let mut sdr_cache = self.runtime.sdr_lookup_cache.write();
            sdr_cache.insert(rec.id.clone(), self.sdr.text_to_sdr(content, false));
        }

        // Cortex insert for anchors
        if pin || is_identity {
            let sdr_u32: Vec<u32> = sdr_indices.iter().map(|&i| i as u32).collect();
            let payload = crate::cortex::ReflexPayload::new(
                content.to_string(),
                rec.strength,
                None,
                0, // doc_id not used in cognitive mode
            );
            self.cortex.insert(&sdr_u32, payload);
        }

        // Compute embedding if embedding_fn is set (Python only)
        #[cfg(feature = "python")]
        {
            let embedding_fn = self.embedding_fn.read();
            if let Some(ref py_fn) = *embedding_fn {
                let emb: Option<Vec<f32>> = Python::with_gil(|py| {
                    let result = py_fn.call1(py, (content,)).ok()?;
                    result.extract::<Vec<f32>>(py).ok()
                });
                if let Some(embedding) = emb {
                    self.embedding_store.insert(&rec.id, embedding);
                }
            }
        }

        // Audit
        if let Some(ref log) = self.audit_log {
            let _ = log.log_store(&rec.id, content);
        }

        self.refresh_deterministic_relations_for_namespace(&rec.namespace)?;

        // Invalidate recall cache on write
        self.runtime.clear_recall_caches();

        Ok(rec)
    }

    /// Recall memories (formatted string for LLM context).
    /// Uses in-memory cache — repeated queries return instantly.
    pub fn recall(
        &self,
        query: &str,
        token_budget: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespaces: Option<&[&str]>,
    ) -> Result<String> {
        let budget = token_budget.unwrap_or(2048);
        RecallService::recall_formatted(
            &self.runtime.recall_cache,
            query,
            budget,
            namespaces,
            || {
                self.recall_core(
                    query,
                    20,
                    min_strength.unwrap_or(0.1),
                    expand_connections.unwrap_or(true),
                    session_id,
                    namespaces,
                )
            },
            |scored| {
                let records = self.records.read();
                recall::format_preamble(scored, budget, &records)
            },
        )
    }

    /// Recall structured (raw results with trust scoring).
    #[instrument(skip(self), fields(top_k, min_strength))]
    pub fn recall_structured(
        &self,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespaces: Option<&[&str]>,
    ) -> Result<Vec<(f32, Record)>> {
        let top = top_k.unwrap_or(20);
        let min_str = min_strength.unwrap_or(0.1);

        RecallService::recall_structured_cached(
            &self.runtime.structured_recall_cache,
            query,
            top,
            min_str,
            namespaces,
            || {
                self.recall_core(
                    query,
                    top,
                    min_str,
                    expand_connections.unwrap_or(true),
                    session_id,
                    namespaces,
                )
            },
        )
    }

    /// Temporal recall: recall only from records created at or before a given timestamp.
    ///
    /// Answers the question: "What did the agent know at time X?"
    /// The pipeline is identical to `recall_structured`, but the record set is
    /// pre-filtered by `created_at <= timestamp` before scoring.
    pub fn recall_at(
        &self,
        query: &str,
        timestamp: f64,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespaces: Option<&[&str]>,
    ) -> Result<Vec<(f32, Record)>> {
        let records = self.records.read();
        let time_records = Self::records_before_timestamp(&records, timestamp);

        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();

        let top = top_k.unwrap_or(20);
        let embedding_ranked = self.collect_embedding_signal(query, top);
        let trust_config = self.config.trust_config.read();

        let scored = RecallService::recall_temporal(
            RecallPipelineView {
                sdr: &self.sdr,
                index: &self.index,
                storage: &self.storage,
                ngram: &ngram,
                tag_index: &tag_idx,
                aura_index: &aura_idx,
                records: &time_records,
                embedding_ranked,
                trust_config: Some(&trust_config),
            },
            query,
            top,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            namespaces,
        );

        drop(records);
        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        self.recall_finalize(&scored, query, session_id);

        Ok(scored)
    }

    /// Return the access/strength timeline for a single record.
    ///
    /// Returns a snapshot with: creation time, last activation, current strength,
    /// activation count, age in days, and days since last activation.
    pub fn history(&self, record_id: &str) -> Result<HashMap<String, String>> {
        let records = self.records.read();
        let rec = records
            .get(record_id)
            .ok_or_else(|| anyhow::anyhow!("Record not found: {}", record_id))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let mut info = HashMap::new();
        info.insert("id".into(), rec.id.clone());
        info.insert("content".into(), rec.content.clone());
        info.insert("level".into(), rec.level.name().to_string());
        info.insert("strength".into(), format!("{:.4}", rec.strength));
        info.insert("activation_count".into(), rec.activation_count.to_string());
        info.insert("created_at".into(), format!("{:.3}", rec.created_at));
        info.insert(
            "last_activated".into(),
            format!("{:.3}", rec.last_activated),
        );
        info.insert(
            "age_days".into(),
            format!("{:.2}", (now - rec.created_at) / 86400.0),
        );
        info.insert(
            "days_since_activation".into(),
            format!("{:.2}", (now - rec.last_activated) / 86400.0),
        );
        info.insert("namespace".into(), rec.namespace.clone());
        info.insert("source_type".into(), rec.source_type.clone());
        info.insert("tags".into(), rec.tags.join(", "));
        info.insert("salience".into(), format!("{:.4}", rec.salience));
        if let Some(reason) = rec.metadata.get(RECORD_SALIENCE_REASON_KEY) {
            info.insert("salience_reason".into(), reason.clone());
        }

        // Include connection count
        info.insert("connections".into(), rec.connections.len().to_string());

        Ok(info)
    }

    /// Unified recall: recall_core (RRF) + substring fallback + failure records in fewer lock passes.
    ///
    /// Combines what Python previously did in 3 separate calls:
    /// 1. recall_structured (RRF semantic)
    /// 2. search (substring fallback)
    /// 3. search with tags=["outcome-failure"]
    ///
    /// Stage 1 runs recall_core as-is (it needs write lock for activation).
    /// Stages 2+3 are merged into a single read lock pass.
    #[instrument(skip(self), fields(top_k, include_failures))]
    pub fn recall_full(
        &self,
        query: &str,
        top_k: Option<usize>,
        include_failures: Option<bool>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespaces: Option<&[&str]>,
    ) -> Result<Vec<(f32, Record)>> {
        let top_k = top_k.unwrap_or(20);
        let include_failures = include_failures.unwrap_or(true);
        let min_strength_v = min_strength.unwrap_or(0.1);
        let default_ns = [crate::record::DEFAULT_NAMESPACE];
        let ns_list = namespaces.unwrap_or(&default_ns);

        // Stage 1: RRF pipeline (has its own lock cycle — read + write for activation)
        let mut scored = self.recall_core(
            query,
            top_k,
            min_strength_v,
            expand_connections.unwrap_or(true),
            session_id,
            namespaces,
        )?;

        // Stage 2+3: Merge substring matches + failure records in ONE read lock
        {
            let records = self.records.read();
            let seen_ids: std::collections::HashSet<String> =
                scored.iter().map(|(_, r)| r.id.clone()).collect();
            let query_lower = query.to_lowercase();

            for rec in records.values() {
                if seen_ids.contains(&rec.id) {
                    continue;
                }
                if !ns_list.contains(&rec.namespace.as_str()) {
                    continue;
                }
                if rec.strength < min_strength_v {
                    continue;
                }

                let content_lower = rec.content.to_lowercase();
                let matches_query = content_lower.contains(&query_lower);

                // Substring match (stage 2)
                if matches_query {
                    let is_failure = rec.tags.contains(&"outcome-failure".to_string());
                    let score = if is_failure { 0.8 } else { 0.6 };
                    scored.push((score, rec.clone()));
                    continue;
                }

                // Failure-only match: tag "outcome-failure" but content didn't substring-match
                if include_failures && rec.tags.contains(&"outcome-failure".to_string()) {
                    let query_words: Vec<&str> = query_lower
                        .split_whitespace()
                        .filter(|w| w.len() > 3)
                        .collect();
                    if query_words.iter().any(|w| content_lower.contains(w)) {
                        scored.push((0.8, rec.clone()));
                    }
                }
            }
        }

        // Re-sort by score desc, truncate
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k + 15);

        Ok(scored)
    }

    /// Filter records to one or more namespaces.
    ///
    /// - `None` → only "default" namespace
    /// - `Some(&["default"])` → only "default"
    /// - `Some(&["default", "sandbox"])` → records from either namespace
    #[allow(dead_code)]
    fn records_for_namespace(
        records: &HashMap<String, Record>,
        namespaces: Option<&[&str]>,
    ) -> HashMap<String, Record> {
        let default_ns = [crate::record::DEFAULT_NAMESPACE];
        let ns_list = namespaces.unwrap_or(&default_ns);
        records
            .iter()
            .filter(|(_, r)| ns_list.contains(&r.namespace.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Filter records to those created at or before a given timestamp.
    fn records_before_timestamp(
        records: &HashMap<String, Record>,
        timestamp: f64,
    ) -> HashMap<String, Record> {
        records
            .iter()
            .filter(|(_, r)| r.created_at <= timestamp)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn record_to_stored_record(rec: &Record) -> StoredRecord {
        StoredRecord {
            id: rec.id.clone(),
            dna: rec.level.to_dna().to_string(),
            timestamp: rec.created_at,
            intensity: rec.strength,
            stability: if rec.level == Level::Identity {
                100.0
            } else {
                1.0
            },
            decay_velocity: 0.0,
            entropy: 0.0,
            sdr_indices: Vec::new(),
            text: rec.content.clone(),
            offset: 0,
        }
    }

    fn ingest_batch_with_pin(&self, texts: Vec<String>, pin: bool) -> Result<usize> {
        if texts.is_empty() {
            return Ok(0);
        }

        let mut ids = Vec::with_capacity(texts.len());
        for text in texts {
            let rec = self.store(
                &text,
                if pin { Some(Level::Identity) } else { None },
                None,
                Some(pin),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )?;
            ids.push(rec.id);
        }

        for pair in ids.windows(2) {
            if let [from_id, to_id] = pair {
                self.storage.set_next_id(from_id, to_id);
            }
        }

        Ok(ids.len())
    }

    /// Raw recall pipeline: signals → RRF → graph walk → trust scoring.
    /// Does NOT apply belief reranking. Does NOT activate/strengthen records.
    /// Used as the clean baseline for diagnostic APIs.
    fn recall_raw(
        &self,
        query: &str,
        top_k: usize,
        min_strength: f32,
        expand_connections: bool,
        namespaces: Option<&[&str]>,
    ) -> Vec<(f32, Record)> {
        let records = self.records.read();
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();
        let embedding_ranked = self.collect_embedding_signal(query, top_k);
        let trust_config = self.config.trust_config.read();

        RecallService::raw(
            RecallPipelineView {
                sdr: &self.sdr,
                index: &self.index,
                storage: &self.storage,
                ngram: &ngram,
                tag_index: &tag_idx,
                aura_index: &aura_idx,
                records: &records,
                embedding_ranked,
                trust_config: Some(&trust_config),
            },
            query,
            top_k,
            min_strength,
            expand_connections,
            namespaces,
        )
    }

    /// Post-recall side effects: activate records and log audit.
    fn recall_finalize(&self, scored: &[(f32, Record)], query: &str, session_id: Option<&str>) {
        let mut records = self.records.write();
        let mut tracker = self.session_tracker.write();
        RecallService::finalize(
            scored,
            query,
            session_id,
            &mut records,
            &mut tracker,
            self.audit_log.as_ref(),
        );
    }

    /// Core recall pipeline: raw baseline + optional belief reranking + side effects.
    #[instrument(skip(self), fields(top_k, min_strength))]
    fn recall_core(
        &self,
        query: &str,
        top_k: usize,
        min_strength: f32,
        expand_connections: bool,
        session_id: Option<&str>,
        namespaces: Option<&[&str]>,
    ) -> Result<Vec<(f32, Record)>> {
        let mut scored =
            self.recall_raw(query, top_k, min_strength, expand_connections, namespaces);

        let belief_eng = self.belief_engine.read();
        let concept_eng = self.concept_engine.read();
        let causal_eng = self.causal_engine.read();
        let policy_eng = self.policy_engine.read();
        let rerank_mode = recall::BeliefRerankMode::from_u8(
            self.runtime
                .belief_rerank_mode
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        let concept_mode = self.get_concept_surface_mode();
        let causal_rerank = CausalRerankMode::from_u8(
            self.runtime
                .causal_rerank_mode
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        let policy_rerank = PolicyRerankMode::from_u8(
            self.runtime
                .policy_rerank_mode
                .load(std::sync::atomic::Ordering::Relaxed),
        );

        RecallService::apply_bounded_reranking(
            &mut scored,
            top_k,
            RecallRerankView {
                belief_engine: &belief_eng,
                concept_engine: &concept_eng,
                causal_engine: &causal_eng,
                policy_engine: &policy_eng,
                belief_mode: rerank_mode,
                concept_mode,
                causal_mode: causal_rerank,
                policy_mode: policy_rerank,
            },
        );

        self.recall_finalize(&scored, query, session_id);
        Ok(scored)
    }

    /// Recall with parallel shadow belief scoring.
    ///
    /// Returns (baseline_results, shadow_report). The baseline is the raw
    /// pipeline output (steps 1-4) WITHOUT belief reranking, regardless of
    /// the current mode setting. Shadow scoring is purely observational.
    pub fn recall_structured_with_shadow(
        &self,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespaces: Option<&[&str]>,
    ) -> Result<(Vec<(f32, Record)>, recall::ShadowRecallReport)> {
        let top = top_k.unwrap_or(20);
        let min_str = min_strength.unwrap_or(0.1);

        let scored = self.recall_raw(
            query,
            top,
            min_str,
            expand_connections.unwrap_or(true),
            namespaces,
        );

        // Shadow scoring on raw baseline
        let belief_eng = self.belief_engine.read();
        let shadow_report = RecallService::shadow_report(&scored, &belief_eng, top);

        self.recall_finalize(&scored, query, session_id);
        Ok((scored, shadow_report))
    }

    /// Recall with limited reranking and a diagnostic report.
    ///
    /// Applies a single pass of limited belief reranking on the raw baseline
    /// (regardless of mode setting) and returns a `LimitedRerankReport`.
    /// The raw baseline is used to avoid double-reranking when mode is Limited.
    pub fn recall_structured_with_rerank_report(
        &self,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespaces: Option<&[&str]>,
    ) -> Result<(Vec<(f32, Record)>, recall::LimitedRerankReport)> {
        let top = top_k.unwrap_or(20);
        let min_str = min_strength.unwrap_or(0.1);

        let mut scored = self.recall_raw(
            query,
            top,
            min_str,
            expand_connections.unwrap_or(true),
            namespaces,
        );

        let belief_eng = self.belief_engine.read();
        let report = RecallService::rerank_report(&mut scored, &belief_eng, top);

        self.recall_finalize(&scored, query, session_id);
        Ok((scored, report))
    }

    /// Explain recall results using persisted provenance across belief,
    /// concept, causal, and policy layers.
    ///
    /// This is inspection-only: it mirrors the current bounded reranking path
    /// but does not activate records or mutate runtime state.
    pub fn explain_recall(
        &self,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        namespaces: Option<&[&str]>,
    ) -> RecallExplanation {
        let started = std::time::Instant::now();
        let top = top_k.unwrap_or(20);
        let min_str = min_strength.unwrap_or(0.1);
        let records = self.records.read();
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();
        let embedding_ranked = self.collect_embedding_signal(query, top);
        let trust_config = self.config.trust_config.read();
        let traced = RecallService::raw_with_trace(
            RecallPipelineView {
                sdr: &self.sdr,
                index: &self.index,
                storage: &self.storage,
                ngram: &ngram,
                tag_index: &tag_idx,
                aura_index: &aura_idx,
                records: &records,
                embedding_ranked,
                trust_config: Some(&trust_config),
            },
            query,
            top,
            min_str,
            expand_connections.unwrap_or(true),
            namespaces,
        );
        let crate::recall::RecallTraceResult { mut scored, traces } = traced;

        let belief_eng = self.belief_engine.read();
        let concept_eng = self.concept_engine.read();
        let causal_eng = self.causal_engine.read();
        let policy_eng = self.policy_engine.read();
        let correction_log = self.get_correction_log();
        let reflection_summaries = self.get_reflection_summaries(Some(8));
        let belief_mode = self.get_belief_rerank_mode();
        let concept_mode = self.get_concept_surface_mode();
        let causal_mode = self.get_causal_rerank_mode();
        let policy_mode = self.get_policy_rerank_mode();

        RecallService::apply_bounded_reranking(
            &mut scored,
            top,
            RecallRerankView {
                belief_engine: &belief_eng,
                concept_engine: &concept_eng,
                causal_engine: &causal_eng,
                policy_engine: &policy_eng,
                belief_mode,
                concept_mode,
                causal_mode,
                policy_mode,
            },
        );

        let items = scored
            .iter()
            .enumerate()
            .map(|(idx, (score, rec))| {
                build_recall_explanation_item(
                    idx + 1,
                    *score,
                    rec,
                    &records,
                    &belief_eng,
                    &concept_eng,
                    &causal_eng,
                    &policy_eng,
                    &correction_log,
                    &reflection_summaries,
                    traces.get(&rec.id),
                )
            })
            .collect();

        RecallExplanation {
            query: query.to_string(),
            top_k: top,
            result_count: scored.len(),
            latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            belief_rerank_mode: format!("{belief_mode:?}").to_lowercase(),
            concept_surface_mode: format!("{concept_mode:?}").to_lowercase(),
            causal_rerank_mode: format!("{causal_mode:?}").to_lowercase(),
            policy_rerank_mode: format!("{policy_mode:?}").to_lowercase(),
            items,
        }
    }

    /// Explain a single record using current persisted provenance across
    /// belief, concept, causal, and policy layers.
    ///
    /// This is inspection-only and does not depend on recall ranking.
    pub fn explain_record(&self, record_id: &str) -> Option<RecallExplanationItem> {
        let records = self.records.read();
        let rec = records.get(record_id)?;
        let belief_eng = self.belief_engine.read();
        let concept_eng = self.concept_engine.read();
        let causal_eng = self.causal_engine.read();
        let policy_eng = self.policy_engine.read();
        let correction_log = self.get_correction_log();
        let reflection_summaries = self.get_reflection_summaries(Some(8));

        Some(build_recall_explanation_item(
            1,
            rec.strength,
            rec,
            &records,
            &belief_eng,
            &concept_eng,
            &causal_eng,
            &policy_eng,
            &correction_log,
            &reflection_summaries,
            None,
        ))
    }

    /// Build a deterministic provenance chain and narrative for a single record.
    ///
    /// This is a read-only inspection API built on top of the same persisted
    /// provenance used by `explain_record()`.
    pub fn provenance_chain(&self, record_id: &str) -> Option<ProvenanceChain> {
        let started = std::time::Instant::now();
        let item = self.explain_record(record_id)?;
        Some(build_provenance_chain(
            &item,
            started.elapsed().as_secs_f64() * 1000.0,
        ))
    }

    /// Build a single bounded explainability bundle for one record.
    ///
    /// This combines the current direct explanation surface, provenance
    /// narrative, relevant correction excerpts, and compact runtime summaries
    /// into one inspect object suitable for UI/debugging.
    pub fn explainability_bundle(&self, record_id: &str) -> Option<ExplainabilityBundle> {
        let explanation = self.explain_record(record_id)?;
        let provenance_started = std::time::Instant::now();
        let provenance = build_provenance_chain(
            &explanation,
            provenance_started.elapsed().as_secs_f64() * 1000.0,
        );
        let record_corrections = self.get_correction_log_for_target("record", record_id);
        let belief_corrections = explanation
            .belief
            .as_ref()
            .map(|belief| self.get_correction_log_for_target("belief", &belief.id))
            .unwrap_or_default();

        let mut causal_corrections = Vec::new();
        let mut seen_causal = HashSet::new();
        for pattern in &explanation.causal_patterns {
            if seen_causal.insert(pattern.id.clone()) {
                causal_corrections
                    .extend(self.get_correction_log_for_target("causal_pattern", &pattern.id));
            }
        }

        let mut policy_corrections = Vec::new();
        let mut seen_policy = HashSet::new();
        for hint in &explanation.policy_hints {
            if seen_policy.insert(hint.id.clone()) {
                policy_corrections
                    .extend(self.get_correction_log_for_target("policy_hint", &hint.id));
            }
        }
        let reflection_digest = self.get_reflection_digest(Some(8));
        let related_reflection_findings = self
            .get_reflection_summaries(Some(8))
            .into_iter()
            .flat_map(|summary| summary.findings.into_iter())
            .filter(|finding| {
                finding.related_ids.iter().any(|id| id == record_id)
                    || finding.namespace == explanation.namespace
            })
            .collect::<Vec<_>>();

        Some(ExplainabilityBundle {
            record_id: record_id.to_string(),
            explanation,
            provenance,
            record_corrections,
            belief_corrections,
            causal_corrections,
            policy_corrections,
            belief_instability: self.get_belief_instability_summary(),
            reflection_digest,
            related_reflection_findings,
            maintenance_trends: self.get_maintenance_trend_summary(),
        })
    }

    /// Build a read-only bounded analytics digest across namespaces.
    ///
    /// This does not mutate runtime state and does not bypass namespace
    /// isolation in recall. It is intended for inspection and dashboards.
    pub fn cross_namespace_digest(&self) -> CrossNamespaceDigest {
        self.cross_namespace_digest_with_options(None, CrossNamespaceDigestOptions::default())
    }

    /// Build a read-only bounded analytics digest across namespaces with
    /// optional namespace filtering and top-concept truncation.
    pub fn cross_namespace_digest_filtered(
        &self,
        namespaces: Option<&[&str]>,
        top_concepts_limit: Option<usize>,
    ) -> CrossNamespaceDigest {
        let options = CrossNamespaceDigestOptions {
            top_concepts_limit: top_concepts_limit.unwrap_or(5).clamp(1, 10),
            ..CrossNamespaceDigestOptions::default()
        };
        self.cross_namespace_digest_with_options(namespaces, options)
    }

    /// Build a read-only bounded analytics digest across namespaces with
    /// richer operator-facing filtering and summary controls.
    pub fn cross_namespace_digest_with_options(
        &self,
        namespaces: Option<&[&str]>,
        options: CrossNamespaceDigestOptions,
    ) -> CrossNamespaceDigest {
        let started = std::time::Instant::now();
        let records = self.records.read();
        let concept_engine = self.concept_engine.read();
        let causal_engine = self.causal_engine.read();
        let top_concepts_limit = options.top_concepts_limit.clamp(1, 10);
        let correction_log = if options.include_corrections {
            Some(self.get_correction_log())
        } else {
            None
        };

        let mut namespaces: Vec<String> = records
            .values()
            .map(|record| record.namespace.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|namespace| {
                namespaces
                    .as_ref()
                    .is_none_or(|allowed| allowed.contains(&namespace.as_str()))
            })
            .filter(|namespace| {
                records
                    .values()
                    .filter(|record| &record.namespace == namespace)
                    .count()
                    >= options.min_record_count
            })
            .collect();
        namespaces.sort();

        let namespace_digests: Vec<CrossNamespaceNamespaceDigest> = namespaces
            .iter()
            .map(|namespace| {
                let namespace_records: Vec<&Record> = records
                    .values()
                    .filter(|record| &record.namespace == namespace)
                    .collect();
                let mut tags: Vec<String> = if options.include_tags && !options.compact_summary {
                    namespace_records
                        .iter()
                        .flat_map(|record| record.tags.iter().map(|tag| tag.to_lowercase()))
                        .collect::<HashSet<_>>()
                        .into_iter()
                        .collect()
                } else {
                    Vec::new()
                };
                tags.sort();

                let mut structural_relation_types: Vec<String> =
                    if options.include_structural && !options.compact_summary {
                        namespace_records
                            .iter()
                            .flat_map(|record| {
                                record
                                    .connection_types
                                    .values()
                                    .filter(|relation_type| {
                                        relation::is_structural_relation_type(relation_type)
                                    })
                                    .cloned()
                            })
                            .collect::<HashSet<_>>()
                            .into_iter()
                            .collect()
                    } else {
                        Vec::new()
                    };
                structural_relation_types.sort();

                let mut concepts: Vec<_> = concept_engine
                    .concepts
                    .values()
                    .filter(|concept| &concept.namespace == namespace)
                    .collect();
                concepts.sort_by(|a, b| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.key.cmp(&b.key))
                });
                let top_concepts = if options.include_concepts && !options.compact_summary {
                    concepts
                        .iter()
                        .take(top_concepts_limit)
                        .map(|concept| CrossNamespaceConceptSummary {
                            concept_id: concept.id.clone(),
                            key: concept.key.clone(),
                            confidence: concept.confidence,
                            state: format!("{:?}", concept.state).to_lowercase(),
                            record_count: concept.record_ids.len(),
                            belief_count: concept.belief_ids.len(),
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let stable_concept_count = concepts
                    .iter()
                    .filter(|concept| concept.state == crate::concept::ConceptState::Stable)
                    .count();
                let mut concept_signatures: Vec<String> =
                    if options.include_concepts && !options.compact_summary {
                        concepts
                            .iter()
                            .map(|concept| canonical_concept_signature(concept))
                            .collect::<HashSet<_>>()
                            .into_iter()
                            .collect()
                    } else {
                        Vec::new()
                    };
                concept_signatures.sort();

                let mut causal_signatures: Vec<String> =
                    if options.include_causal && !options.compact_summary {
                        causal_engine
                            .patterns
                            .values()
                            .filter(|pattern| &pattern.namespace == namespace)
                            .map(|pattern| canonical_causal_signature(pattern, &records))
                            .collect::<HashSet<_>>()
                            .into_iter()
                            .collect()
                    } else {
                        Vec::new()
                    };
                causal_signatures.sort();

                let belief_state_summary = if options.include_belief_states {
                    let beliefs: Vec<_> = self
                        .belief_engine
                        .read()
                        .beliefs
                        .values()
                        .filter(|belief| belief.key.starts_with(&format!("{namespace}:")))
                        .cloned()
                        .collect();
                    let total = beliefs.len();
                    let avg_volatility = if total > 0 {
                        beliefs.iter().map(|belief| belief.volatility).sum::<f32>() / total as f32
                    } else {
                        0.0
                    };
                    Some(CrossNamespaceBeliefStateSummary {
                        resolved: beliefs
                            .iter()
                            .filter(|belief| belief.state == crate::belief::BeliefState::Resolved)
                            .count(),
                        unresolved: beliefs
                            .iter()
                            .filter(|belief| belief.state == crate::belief::BeliefState::Unresolved)
                            .count(),
                        singleton: beliefs
                            .iter()
                            .filter(|belief| belief.state == crate::belief::BeliefState::Singleton)
                            .count(),
                        empty: beliefs
                            .iter()
                            .filter(|belief| belief.state == crate::belief::BeliefState::Empty)
                            .count(),
                        high_volatility_count: beliefs
                            .iter()
                            .filter(|belief| belief.volatility >= 0.20)
                            .count(),
                        avg_volatility,
                    })
                } else {
                    None
                };

                let correction_count = correction_log.as_ref().map(|entries| {
                    entries
                        .iter()
                        .filter(|entry| {
                            entry.target_id.starts_with(&format!("{namespace}:"))
                                || entry
                                    .target_id
                                    .to_lowercase()
                                    .contains(&format!("{namespace}-"))
                        })
                        .count()
                });
                let correction_density = correction_count.map(|count| {
                    if namespace_records.is_empty() {
                        0.0
                    } else {
                        count as f32 / namespace_records.len() as f32
                    }
                });

                CrossNamespaceNamespaceDigest {
                    namespace: namespace.clone(),
                    record_count: namespace_records.len(),
                    concept_count: concepts.len(),
                    stable_concept_count,
                    top_concepts,
                    concept_signatures,
                    tags,
                    structural_relation_types,
                    causal_signatures,
                    belief_state_summary,
                    correction_count,
                    correction_density,
                }
            })
            .collect();

        let mut pairs = Vec::new();
        for left in 0..namespace_digests.len() {
            for right in (left + 1)..namespace_digests.len() {
                let a = &namespace_digests[left];
                let b = &namespace_digests[right];
                let concept_a: HashSet<&str> =
                    a.concept_signatures.iter().map(String::as_str).collect();
                let concept_b: HashSet<&str> =
                    b.concept_signatures.iter().map(String::as_str).collect();
                let mut shared_concept_signatures: Vec<String> =
                    if options.include_concepts && !options.compact_summary {
                        concept_a
                            .intersection(&concept_b)
                            .map(|item| (*item).to_string())
                            .collect()
                    } else {
                        Vec::new()
                    };
                shared_concept_signatures.sort();

                let tag_a: HashSet<&str> = a.tags.iter().map(String::as_str).collect();
                let tag_b: HashSet<&str> = b.tags.iter().map(String::as_str).collect();
                let mut shared_tags: Vec<String> =
                    if options.include_tags && !options.compact_summary {
                        tag_a
                            .intersection(&tag_b)
                            .map(|tag| (*tag).to_string())
                            .collect()
                    } else {
                        Vec::new()
                    };
                shared_tags.sort();

                let struct_a: HashSet<&str> = a
                    .structural_relation_types
                    .iter()
                    .map(String::as_str)
                    .collect();
                let struct_b: HashSet<&str> = b
                    .structural_relation_types
                    .iter()
                    .map(String::as_str)
                    .collect();
                let mut shared_structural_relation_types: Vec<String> =
                    if options.include_structural && !options.compact_summary {
                        struct_a
                            .intersection(&struct_b)
                            .map(|item| (*item).to_string())
                            .collect()
                    } else {
                        Vec::new()
                    };
                shared_structural_relation_types.sort();

                let sig_a: HashSet<&str> = a.causal_signatures.iter().map(String::as_str).collect();
                let sig_b: HashSet<&str> = b.causal_signatures.iter().map(String::as_str).collect();
                let mut shared_causal_signatures: Vec<String> =
                    if options.include_causal && !options.compact_summary {
                        sig_a
                            .intersection(&sig_b)
                            .map(|item| (*item).to_string())
                            .collect()
                    } else {
                        Vec::new()
                    };
                shared_causal_signatures.sort();

                let concept_similarity = jaccard_similarity(&concept_a, &concept_b);
                let tag_similarity = jaccard_similarity(&tag_a, &tag_b);
                let structural_similarity = jaccard_similarity(&struct_a, &struct_b);
                let causal_similarity = jaccard_similarity(&sig_a, &sig_b);
                let max_similarity = concept_similarity
                    .max(tag_similarity)
                    .max(structural_similarity)
                    .max(causal_similarity);
                if max_similarity < options.pairwise_similarity_threshold {
                    continue;
                }

                pairs.push(CrossNamespacePairDigest {
                    namespace_a: a.namespace.clone(),
                    namespace_b: b.namespace.clone(),
                    shared_concept_signatures,
                    concept_signature_similarity: concept_similarity,
                    shared_tags,
                    tag_jaccard: tag_similarity,
                    shared_structural_relation_types,
                    structural_similarity,
                    shared_causal_signatures,
                    causal_signature_similarity: causal_similarity,
                });
            }
        }

        let mut included_dimensions = Vec::new();
        if options.include_concepts {
            included_dimensions.push("concepts".to_string());
        }
        if options.include_tags {
            included_dimensions.push("tags".to_string());
        }
        if options.include_structural {
            included_dimensions.push("structural".to_string());
        }
        if options.include_causal {
            included_dimensions.push("causal".to_string());
        }
        if options.include_belief_states {
            included_dimensions.push("belief_states".to_string());
        }
        if options.include_corrections {
            included_dimensions.push("corrections".to_string());
        }

        CrossNamespaceDigest {
            namespace_count: namespace_digests.len(),
            latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            compact_summary: options.compact_summary,
            included_dimensions,
            namespaces: namespace_digests,
            pairs,
        }
    }

    /// Search with filters.
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
        let records = self.records.read();
        let limit = limit.unwrap_or(20);
        let default_ns = [crate::record::DEFAULT_NAMESPACE];
        let ns_list = namespaces.unwrap_or(&default_ns);

        let mut results: Vec<Record> = records
            .values()
            .filter(|r| {
                if !ns_list.contains(&r.namespace.as_str()) {
                    return false;
                }
                if let Some(l) = level {
                    if r.level != l {
                        return false;
                    }
                }
                if let Some(ref t) = tags {
                    if !t.iter().any(|tag| r.tags.contains(tag)) {
                        return false;
                    }
                }
                if let Some(ct) = content_type {
                    if r.content_type != ct {
                        return false;
                    }
                }
                if let Some(st) = source_type {
                    if r.source_type != st {
                        return false;
                    }
                }
                if let Some(sem) = semantic_type {
                    if r.semantic_type != sem {
                        return false;
                    }
                }
                if let Some(q) = query {
                    if !r.content.to_lowercase().contains(&q.to_lowercase()) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        results.sort_by(|a, b| {
            b.importance()
                .partial_cmp(&a.importance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }

    /// Get a single record by ID.
    pub fn get(&self, record_id: &str) -> Option<Record> {
        self.records.read().get(record_id).cloned()
    }

    /// Return records with elevated salience, highest salience first.
    pub fn get_high_salience_records(
        &self,
        min_salience: Option<f32>,
        limit: Option<usize>,
    ) -> Vec<Record> {
        let threshold = min_salience.unwrap_or(0.50).clamp(0.0, 1.0);
        let max = limit.unwrap_or(20).min(100);
        let mut records: Vec<Record> = self
            .records
            .read()
            .values()
            .filter(|rec| rec.salience >= threshold)
            .cloned()
            .collect();
        records.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.importance()
                        .partial_cmp(&a.importance())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        records.truncate(max);
        records
    }

    /// Return a bounded summary of current record salience distribution.
    pub fn get_salience_summary(&self) -> SalienceSummary {
        let records = self.records.read();
        let total = records.len();
        if total == 0 {
            return SalienceSummary::default();
        }

        let mut summary = SalienceSummary {
            total_records: total,
            ..SalienceSummary::default()
        };

        for rec in records.values() {
            summary.avg_salience += rec.salience;
            summary.max_salience = summary.max_salience.max(rec.salience);

            if rec.salience >= 0.70 {
                summary.high_salience_count += 1;
                summary.bands.high += 1;
            } else if rec.salience >= 0.30 {
                summary.bands.medium += 1;
            } else {
                summary.bands.low += 1;
            }
        }

        summary.avg_salience /= total as f32;
        summary
    }

    /// Mark a record with bounded manual salience and optional reason metadata.
    pub fn mark_record_salience(
        &self,
        record_id: &str,
        salience: f32,
        reason: Option<&str>,
    ) -> Result<Option<Record>> {
        let mut records = self.records.write();
        let rec = match records.get_mut(record_id) {
            Some(r) => r,
            None => return Ok(None),
        };

        rec.salience = salience.clamp(0.0, 1.0);
        match reason.map(str::trim).filter(|value| !value.is_empty()) {
            Some(value) => {
                rec.metadata
                    .insert(RECORD_SALIENCE_REASON_KEY.into(), value.to_string());
            }
            None => {
                rec.metadata.remove(RECORD_SALIENCE_REASON_KEY);
            }
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        rec.metadata
            .insert(RECORD_SALIENCE_MARKED_AT_KEY.into(), format!("{now:.3}"));

        self.cognitive_store.append_update(rec)?;
        let updated = rec.clone();
        drop(records);

        self.runtime.clear_recall_caches();
        Ok(Some(updated))
    }

    /// Update a record.
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
        if let Some(st) = source_type {
            crate::record::Record::validate_source_type(st).map_err(|e| anyhow::anyhow!(e))?;
        }
        let mut records = self.records.write();
        let rec = match records.get_mut(record_id) {
            Some(r) => r,
            None => return Ok(None),
        };

        if let Some(c) = content {
            rec.content = c.to_string();
            // Re-index
            let mut ngram = self.ngram_index.write();
            ngram.remove(record_id);
            ngram.add(record_id, c);
        }
        if let Some(l) = level {
            rec.level = l;
        }
        if let Some(t) = tags {
            rec.tags = t;
        }
        if let Some(s) = strength {
            rec.strength = s.clamp(0.0, 1.0);
        }
        if let Some(m) = metadata {
            rec.metadata = m;
        }
        if let Some(st) = source_type {
            rec.source_type = st.to_string();
        }

        let namespace = rec.namespace.clone();
        self.cognitive_store.append_update(rec)?;
        let updated = rec.clone();
        drop(records);

        self.refresh_deterministic_relations_for_namespace(&namespace)?;

        // Invalidate recall cache on write
        self.runtime.clear_recall_caches();

        Ok(Some(updated))
    }

    /// Delete a record.
    pub fn delete(&self, record_id: &str) -> Result<bool> {
        let mut records = self.records.write();
        if !records.contains_key(record_id) {
            return Ok(false);
        }

        let mut ngram = self.ngram_index.write();
        let mut tag_idx = self.tag_index.write();
        let mut aura_idx = self.aura_index.write();

        crate::graph::remove_record(
            record_id,
            &mut records,
            &mut ngram,
            &mut tag_idx,
            &mut aura_idx,
            &self.cognitive_store,
        );

        self.storage.delete(record_id);
        self.index.remove(record_id);
        self.embedding_store.remove(record_id);
        self.runtime.sdr_lookup_cache.write().remove(record_id);

        // Invalidate recall cache on write
        self.runtime.clear_recall_caches();

        Ok(true)
    }

    /// Connect two records with optional relationship type.
    ///
    /// Relationship types (inspired by molecular reasoning bonds):
    /// - `"causal"` — A caused/led to B (deep reasoning, covalent-like)
    /// - `"reflective"` — B validates/corrects A (self-reflection, hydrogen-bond-like)
    /// - `"associative"` — A and B are thematically related (exploration, van der Waals-like)
    /// - `"coactivation"` — A and B were recalled together in a session
    /// - Any custom string
    pub fn connect(
        &self,
        id_a: &str,
        id_b: &str,
        weight: Option<f32>,
        relationship: Option<&str>,
    ) -> Result<()> {
        let weight = weight.unwrap_or(0.5);
        let mut records = self.records.write();

        // Namespace guard: prevent cross-namespace connections
        let ns_a = records.get(id_a).map(|r| r.namespace.clone());
        let ns_b = records.get(id_b).map(|r| r.namespace.clone());
        match (&ns_a, &ns_b) {
            (Some(a), Some(b)) if a != b => {
                return Err(anyhow::anyhow!(
                    "Cannot connect records across namespaces ('{}' vs '{}')",
                    a,
                    b
                ));
            }
            (None, _) => return Err(anyhow::anyhow!("Record {} not found", id_a)),
            (_, None) => return Err(anyhow::anyhow!("Record {} not found", id_b)),
            _ => {}
        }

        if let Some(a) = records.get_mut(id_a) {
            if let Some(rel) = relationship {
                a.add_typed_connection(id_b, weight, rel);
            } else {
                a.add_connection(id_b, weight);
            }
        }

        if let Some(b) = records.get_mut(id_b) {
            if let Some(rel) = relationship {
                b.add_typed_connection(id_a, weight, rel);
            } else {
                b.add_connection(id_a, weight);
            }
        }

        Ok(())
    }

    // ── Maintenance Operations ──

    /// Apply decay to all records.
    #[instrument(skip(self))]
    pub fn decay(&self) -> Result<(usize, usize)> {
        let mut records = self.records.write();
        let mut decayed = 0;
        let mut to_archive = Vec::new();

        for rec in records.values_mut() {
            rec.apply_decay();
            decayed += 1;

            if !rec.is_alive() {
                to_archive.push(rec.id.clone());
            }
        }

        // Also decay connections
        for rec in records.values_mut() {
            let weak_conns: Vec<String> = rec
                .connections
                .iter()
                .filter(|(_, w)| **w < 0.05)
                .map(|(id, _)| id.clone())
                .collect();

            for id in &weak_conns {
                rec.connections.remove(id);
                rec.connection_types.remove(id);
            }

            for w in rec.connections.values_mut() {
                *w *= 0.99;
            }
        }

        // Archive dead records
        let archived = to_archive.len();
        for id in &to_archive {
            records.remove(id);
            let _ = self.cognitive_store.append_delete(id);
        }

        // Compact if many dead entries
        if archived > 100 {
            let _ = self.cognitive_store.compact(&records);
        }

        Ok((decayed, archived))
    }

    /// Consolidate duplicates.
    #[instrument(skip(self))]
    pub fn consolidate(&self) -> Result<HashMap<String, usize>> {
        let mut records = self.records.write();
        let mut ngram = self.ngram_index.write();
        let mut tag_idx = self.tag_index.write();
        let mut aura_idx = self.aura_index.write();

        let result = consolidation::consolidate(
            &mut records,
            &mut ngram,
            &mut tag_idx,
            &mut aura_idx,
            &self.cognitive_store,
        );

        let mut stats = HashMap::new();
        stats.insert("merged".to_string(), result.merged);
        stats.insert("checked".to_string(), result.checked);
        Ok(stats)
    }

    /// Reflect — promote, archive, detect conflicts.
    #[instrument(skip(self))]
    pub fn reflect(&self) -> Result<HashMap<String, usize>> {
        let mut records = self.records.write();
        let mut promoted = 0;

        // Promote frequently used
        let promotable: Vec<String> = records
            .values()
            .filter(|r| r.can_promote())
            .map(|r| r.id.clone())
            .collect();

        for id in &promotable {
            if let Some(rec) = records.get_mut(id) {
                if rec.promote() {
                    promoted += 1;
                    let _ = self.cognitive_store.append_update(rec);
                }
            }
        }

        // Semantic-aware promotion removed: Level decay rates already encode importance.
        // Standard promotion threshold applies uniformly (activation_count >= 5, strength >= 0.7).
        let semantic_promotable: Vec<String> = vec![];

        for id in &semantic_promotable {
            if let Some(rec) = records.get_mut(id) {
                if rec.promote() {
                    promoted += 1;
                    let _ = self.cognitive_store.append_update(rec);
                }
            }
        }

        // Contextual hub promotion (10+ connections, avg weight >= 0.4)
        let hub_promotable: Vec<String> = records
            .values()
            .filter(|r| {
                r.connections.len() >= 10
                    && r.strength >= 0.5
                    && r.level < Level::Identity
                    && r.connections.values().sum::<f32>() / r.connections.len() as f32 >= 0.4
            })
            .map(|r| r.id.clone())
            .collect();

        for id in &hub_promotable {
            if let Some(rec) = records.get_mut(id) {
                if rec.promote() {
                    promoted += 1;
                    let _ = self.cognitive_store.append_update(rec);
                }
            }
        }

        // Archive dead records — uniform threshold regardless of semantic_type.
        // Level decay rates (Identity=0.99 .. Working=0.80) already protect important records.
        let dead: Vec<String> = records
            .values()
            .filter(|r| !r.is_alive()) // strength < 0.05
            .map(|r| r.id.clone())
            .collect();

        let archived = dead.len();
        for id in &dead {
            records.remove(id);
            let _ = self.cognitive_store.append_delete(id);
        }

        let mut stats = HashMap::new();
        stats.insert("promoted".to_string(), promoted);
        stats.insert("archived".to_string(), archived);
        Ok(stats)
    }

    /// Get insights (pattern detection).
    pub fn insights(&self) -> Vec<insights::Insight> {
        let records = self.records.read();
        insights::detect_all(&records)
    }

    /// Run only Phase 2 (cross-domain) detectors.
    pub fn insights_cross_domain(&self) -> Vec<insights::Insight> {
        let records = self.records.read();
        insights::detect_phase2(&records)
    }

    /// End a session (co-activation strengthening).
    pub fn end_session(&self, session_id: &str) -> Result<HashMap<String, usize>> {
        let mut records = self.records.write();
        let mut tracker = self.session_tracker.write();
        Ok(tracker.end_session(session_id, &mut records))
    }

    /// Get statistics.
    pub fn stats(&self) -> HashMap<String, usize> {
        let records = self.records.read();
        let mut stats = HashMap::new();

        stats.insert("total_records".into(), records.len());
        stats.insert(
            "working".into(),
            records
                .values()
                .filter(|r| r.level == Level::Working)
                .count(),
        );
        stats.insert(
            "decisions".into(),
            records
                .values()
                .filter(|r| r.level == Level::Decisions)
                .count(),
        );
        stats.insert(
            "domain".into(),
            records
                .values()
                .filter(|r| r.level == Level::Domain)
                .count(),
        );
        stats.insert(
            "identity".into(),
            records
                .values()
                .filter(|r| r.level == Level::Identity)
                .count(),
        );
        stats.insert(
            "total_connections".into(),
            records.values().map(|r| r.connections.len()).sum(),
        );
        stats.insert("total_tags".into(), self.tag_index.read().len());

        stats
    }

    /// Count records, optionally filtered by level.
    pub fn count(&self, level: Option<Level>) -> usize {
        let records = self.records.read();
        match level {
            Some(l) => records.values().filter(|r| r.level == l).count(),
            None => records.len(),
        }
    }

    // ── Two-Tier API (Cognitive / Core) ──

    /// Recall from the cognitive tier only (WORKING + DECISIONS).
    ///
    /// When `query` is provided, runs the full RRF Fusion pipeline (SDR + MinHash +
    /// Tag Jaccard + optional embeddings) and then filters results to cognitive-tier
    /// records. This gives the same ranking quality as `recall_structured()`.
    ///
    /// When `query` is None, returns all cognitive records sorted by importance.
    pub fn recall_cognitive(
        &self,
        query: Option<&str>,
        limit: Option<usize>,
        namespaces: Option<&[&str]>,
    ) -> Vec<Record> {
        let limit = limit.unwrap_or(20);
        let default_ns = [crate::record::DEFAULT_NAMESPACE];
        let ns_list = namespaces.unwrap_or(&default_ns);

        if let Some(q) = query {
            // RRF pipeline → filter to cognitive tier
            // Request more from pipeline to compensate for tier filtering
            let pipeline_limit = limit * 3;
            if let Ok(scored) = self.recall_core(q, pipeline_limit, 0.1, true, None, namespaces) {
                let results: Vec<Record> = scored
                    .into_iter()
                    .filter(|(_, r)| r.level.is_cognitive())
                    .take(limit)
                    .map(|(_, r)| r)
                    .collect();
                return results;
            }
        }

        // No query or pipeline error → list all cognitive by importance
        let records = self.records.read();
        let mut results: Vec<Record> = records
            .values()
            .filter(|r| r.level.is_cognitive() && ns_list.contains(&r.namespace.as_str()))
            .cloned()
            .collect();

        results.sort_by(|a, b| {
            b.importance()
                .partial_cmp(&a.importance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }

    /// Recall from the core tier only (DOMAIN + IDENTITY).
    ///
    /// When `query` is provided, runs the full RRF Fusion pipeline (SDR + MinHash +
    /// Tag Jaccard + optional embeddings) and then filters results to core-tier
    /// records. This gives the same ranking quality as `recall_structured()`.
    ///
    /// When `query` is None, returns all core records sorted by importance.
    pub fn recall_core_tier(
        &self,
        query: Option<&str>,
        limit: Option<usize>,
        namespaces: Option<&[&str]>,
    ) -> Vec<Record> {
        let limit = limit.unwrap_or(20);
        let default_ns = [crate::record::DEFAULT_NAMESPACE];
        let ns_list = namespaces.unwrap_or(&default_ns);

        if let Some(q) = query {
            // RRF pipeline → filter to core tier
            let pipeline_limit = limit * 3;
            if let Ok(scored) = self.recall_core(q, pipeline_limit, 0.1, true, None, namespaces) {
                let results: Vec<Record> = scored
                    .into_iter()
                    .filter(|(_, r)| r.level.is_core())
                    .take(limit)
                    .map(|(_, r)| r)
                    .collect();
                return results;
            }
        }

        // No query or pipeline error → list all core by importance
        let records = self.records.read();
        let mut results: Vec<Record> = records
            .values()
            .filter(|r| r.level.is_core() && ns_list.contains(&r.namespace.as_str()))
            .cloned()
            .collect();

        results.sort_by(|a, b| {
            b.importance()
                .partial_cmp(&a.importance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }

    /// Get memory statistics broken down by tier.
    ///
    /// Returns a structured breakdown:
    /// - `cognitive_total`: WORKING + DECISIONS count
    /// - `cognitive_working`: WORKING count
    /// - `cognitive_decisions`: DECISIONS count
    /// - `core_total`: DOMAIN + IDENTITY count
    /// - `core_domain`: DOMAIN count
    /// - `core_identity`: IDENTITY count
    /// - `total`: all records
    pub fn tier_stats(&self) -> HashMap<String, usize> {
        let records = self.records.read();

        let working = records
            .values()
            .filter(|r| r.level == Level::Working)
            .count();
        let decisions = records
            .values()
            .filter(|r| r.level == Level::Decisions)
            .count();
        let domain = records
            .values()
            .filter(|r| r.level == Level::Domain)
            .count();
        let identity = records
            .values()
            .filter(|r| r.level == Level::Identity)
            .count();

        let mut stats = HashMap::new();
        stats.insert("cognitive_total".into(), working + decisions);
        stats.insert("cognitive_working".into(), working);
        stats.insert("cognitive_decisions".into(), decisions);
        stats.insert("core_total".into(), domain + identity);
        stats.insert("core_domain".into(), domain);
        stats.insert("core_identity".into(), identity);
        stats.insert("total".into(), working + decisions + domain + identity);
        stats
    }

    /// Find cognitive records that are candidates for promotion to core.
    ///
    /// A record qualifies when:
    /// - It's in the cognitive tier (WORKING or DECISIONS)
    /// - activation_count >= `min_activations` (default 5)
    /// - strength >= `min_strength` (default 0.7)
    ///
    /// These are records that started as ephemeral but proved important
    /// through repeated recall — they should graduate to permanent memory.
    pub fn promotion_candidates(
        &self,
        min_activations: Option<u32>,
        min_strength: Option<f32>,
    ) -> Vec<Record> {
        let records = self.records.read();
        let min_act = min_activations.unwrap_or(5);
        let min_str = min_strength.unwrap_or(0.7);

        let mut candidates: Vec<Record> = records
            .values()
            .filter(|r| {
                r.level.is_cognitive() && r.activation_count >= min_act && r.strength >= min_str
            })
            .cloned()
            .collect();

        candidates.sort_by(|a, b| {
            b.activation_count.cmp(&a.activation_count).then_with(|| {
                b.strength
                    .partial_cmp(&a.strength)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        candidates
    }

    /// Promote a record to the next cognitive level.
    ///
    /// WORKING → DECISIONS → DOMAIN → IDENTITY.
    /// Returns the new level, or None if already at IDENTITY or record not found.
    pub fn promote_record(&self, record_id: &str) -> Option<Level> {
        let mut records = self.records.write();
        if let Some(rec) = records.get_mut(record_id) {
            if rec.promote() {
                // Persist the change
                let _ = self.cognitive_store.append_update(rec);
                self.runtime.recall_cache.clear();
                self.runtime.structured_recall_cache.clear();
                Some(rec.level)
            } else {
                None
            }
        } else {
            None
        }
    }

    // ── Optional Embedding Support ──

    /// Store an embedding vector for a record.
    /// Used when embeddings are computed externally (e.g., via an LLM API).
    pub fn store_embedding(&self, record_id: &str, embedding: Vec<f32>) {
        self.embedding_store.insert(record_id, embedding);
    }

    /// Remove an embedding for a record.
    pub fn remove_embedding(&self, record_id: &str) {
        self.embedding_store.remove(record_id);
    }

    /// Check if embedding support is active (any embeddings stored).
    pub fn has_embeddings(&self) -> bool {
        self.embedding_store.is_active()
    }

    /// Collect embedding similarity signal for recall pipeline.
    /// Returns None if no embeddings are stored or no query embedding is available.
    #[allow(unused_variables)]
    fn collect_embedding_signal(&self, _query: &str, top_k: usize) -> Option<Vec<(String, f32)>> {
        if !self.embedding_store.is_active() {
            return None;
        }

        // In pure Rust mode, the user must provide query embeddings via
        // recall_with_embedding(). In Python mode, the embedding_fn callback
        // is used. This method is a no-op without explicit query embeddings.
        #[cfg(feature = "python")]
        {
            let embedding_fn = self.embedding_fn.read();
            if let Some(ref py_fn) = *embedding_fn {
                let result: Option<Vec<f32>> = Python::with_gil(|py| {
                    let result = py_fn.call1(py, (_query,)).ok()?;
                    result.extract::<Vec<f32>>(py).ok()
                });
                if let Some(query_emb) = result {
                    return Some(self.embedding_store.query(&query_emb, top_k));
                }
            }
        }

        None
    }

    /// Recall with explicit query embedding (Rust API).
    /// Uses the embedding as a 4th RRF signal alongside SDR, N-gram, and Tag Jaccard.
    pub fn recall_with_embedding(
        &self,
        query: &str,
        query_embedding: &[f32],
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> Result<Vec<(f32, Record)>> {
        let top_k = top_k.unwrap_or(20);
        let records = self.records.read();
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();

        // Get embedding signal
        let embedding_ranked = Some(self.embedding_store.query(query_embedding, top_k));
        let trust_config = self.config.trust_config.read();

        let scored = recall::recall_pipeline(
            query,
            top_k,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            &self.sdr,
            &self.index,
            &self.storage,
            &ngram,
            &tag_idx,
            &aura_idx,
            &records,
            embedding_ranked,
            Some(&trust_config),
            None, // default namespace
        );

        drop(records);
        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        {
            let mut records = self.records.write();
            let mut tracker = self.session_tracker.write();
            recall::activate_and_strengthen(&scored, &mut records, &mut tracker, session_id);
        }

        Ok(scored)
    }

    // ── SDK Wrapper: Taxonomy & Trust ──

    /// Set tag taxonomy (configurable tag classification).
    pub fn set_taxonomy(&self, taxonomy: TagTaxonomy) {
        *self.config.taxonomy.write() = taxonomy;
    }

    /// Get current tag taxonomy.
    pub fn get_taxonomy(&self) -> TagTaxonomy {
        self.config.taxonomy.read().clone()
    }

    /// Set trust configuration.
    pub fn set_trust_config(&self, config: TrustConfig) {
        *self.config.trust_config.write() = config;
    }

    /// Get current trust configuration.
    pub fn get_trust_config(&self) -> TrustConfig {
        self.config.trust_config.read().clone()
    }

    /// Get credibility score for a URL.
    pub fn get_credibility(&self, url: &str) -> f32 {
        self.research_engine.get_credibility(url)
    }

    /// Set credibility override for a domain.
    pub fn set_credibility_override(&self, domain: &str, score: f32) {
        self.research_engine.set_credibility_override(domain, score);
    }

    // ── SDK Wrapper: Background Brain ──

    /// Configure maintenance settings.
    pub fn configure_maintenance(&self, config: MaintenanceConfig) {
        *self.config.maintenance_config.write() = config;
    }

    /// Get current maintenance configuration.
    pub fn get_maintenance_config(&self) -> MaintenanceConfig {
        self.config.maintenance_config.read().clone()
    }

    /// Run a single maintenance cycle across all maintenance phases.
    #[instrument(skip(self))]
    pub fn run_maintenance(&self) -> MaintenanceReport {
        use std::time::Instant;

        let cycle_start = Instant::now();
        let mut timings = background_brain::PhaseTimings::default();
        let mut hotspots = background_brain::MaintenanceHotspots::default();

        let config = self.config.maintenance_config.read().clone();
        let taxonomy = self.config.taxonomy.read().clone();

        // ── Phase 3.6: Experience integration ────────────────────────────────
        // Done BEFORE acquiring records.write() to avoid a deadlock:
        // apply_experience_internal() → store_with_channel() → records.write(),
        // which would re-enter the lock that run_maintenance() holds for its
        // remaining phases.  Injecting here means the fresh records are visible
        // to all subsequent phases (3.5 → 3.9) in the snapshot taken below.
        //
        // This is the ONLY place where experience captures produce mutations.
        // capture_experience() is extraction-only; apply_experience_internal()
        // is called here (not in capture_experience()) to avoid double-injection.
        //
        // Guards enforced here (Phase 3.1):
        //   - source_type = "generated"  (max confidence 0.70)
        //   - Level::Working max (no Identity promotion)
        //   - Deduplication via store_with_channel(deduplicate=true)
        //
        // Bug 2 fix: risk throttling applied here before any injection.
        // Bug 3 fix: Full mode uses operator-supplied custom policy.
        let experience_injected = {
            let pending = self.drain_experience_queue();
            let mut injected = 0usize;
            if !pending.is_empty() {
                // ── Bug 2 fix: risk throttling ──
                // Compute current risk ONCE per cycle and apply throttling to
                // every capture's effective policy before any injection.
                let risk_assessment = self.get_plasticity_risk();
                let mode = self.runtime.plasticity_mode();

                for capture in &pending {
                    let (base_policy, policy_name) = self.resolve_plasticity_policy(mode);
                    let effective_policy = risk_assessment.apply_throttling(base_policy);
                    let applied = self.apply_experience_internal(
                        &capture.raw_events,
                        &effective_policy,
                        &capture.source,
                        Some(capture.session_id.as_str()),
                        &capture.prompt_hash,
                        policy_name,
                    );
                    injected += applied.new_records_stored;
                } // end for capture
            }
            injected
        };

        let mut records = self.records.write();
        // Get cycle count from background brain or use 0
        let cycle = {
            let bg = self.runtime.background.read();
            bg.as_ref().map_or(0, |b| b.cycles())
        };
        let initial = MaintenanceService::run_initial_phases(
            &mut records,
            &config,
            &taxonomy,
            &self.cognitive_store,
            cycle,
            &mut timings,
            &mut hotspots,
        );
        let total_records = initial.total_records;
        let decay = initial.decay;
        let reflect = initial.reflect;
        let epistemic = initial.epistemic;
        let insights_found = initial.insights_found;

        // Phase 3.5: Belief update (read-only — builds beliefs, does not affect recall)
        // Take a read-only snapshot of record refs to avoid holding write lock during
        // belief computation and disk persistence.
        let belief_snapshot: HashMap<String, Record> = records
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        // Release records lock before belief work (Phase 4 will re-acquire)
        drop(records);

        // Build SDR lookup for content-aware claim grouping.
        // IMPORTANT: always use is_identity=false (general bit range) so
        // records at different levels (Domain vs Decisions) can still be
        // compared. The stored SDR uses level-dependent bit ranges which
        // would give Tanimoto ≈ 0 across range boundaries.
        // Shared by belief phase (3.5) and concept phase (3.7).
        let sdr_lookup: SdrLookup = MaintenanceService::build_sdr_lookup(
            &self.sdr,
            &self.runtime.sdr_lookup_cache,
            &belief_snapshot,
            &mut timings,
            &mut hotspots,
        );

        let discovery = MaintenanceService::run_discovery_phases(
            &self.belief_engine,
            &self.belief_store,
            &self.concept_engine,
            &self.concept_store,
            &self.causal_engine,
            &self.causal_store,
            &self.policy_engine,
            &self.policy_store,
            &belief_snapshot,
            &sdr_lookup,
            &mut timings,
            &mut hotspots,
        );
        let belief_phase_report = discovery.belief;
        let concept_phase_report = discovery.concept;
        let causal_phase_report = discovery.causal;
        let policy_phase_report = discovery.policy;
        let feedback_phase_report = discovery.feedback;

        // ── Compute cross-cycle identity stability ──
        let stability = {
            let belief_eng = self.belief_engine.read();
            let concept_eng = self.concept_engine.read();
            let causal_eng = self.causal_engine.read();
            let policy_eng = self.policy_engine.read();

            MaintenanceService::compute_layer_stability(
                &belief_eng,
                &concept_eng,
                &causal_eng,
                &policy_eng,
                &self.prev_belief_keys,
                &self.prev_concept_keys,
                &self.prev_causal_keys,
                &self.prev_policy_keys,
            )
        };

        // Phase 4: Consolidation (fast pass only — no LLM)
        let background = self.runtime.background.read();
        let post_discovery = MaintenanceService::run_post_discovery_phases(
            &self.records,
            &self.ngram_index,
            &self.tag_index,
            &self.aura_index,
            &self.cognitive_store,
            background.as_ref(),
            &config,
            &taxonomy,
            &mut timings,
            &mut hotspots,
        );
        drop(background);
        let consolidation_report = post_discovery.consolidation;
        let cross_connections = post_discovery.cross_connections;
        let task_reminders = post_discovery.task_reminders;
        let records_archived = post_discovery.records_archived;

        // Persist changes
        let _ = self.flush();

        // Invalidate cache after maintenance
        self.runtime.clear_recall_caches();

        timings.total_ms = cycle_start.elapsed().as_secs_f64() * 1000.0;
        let concept_surface = MaintenanceService::finalize_telemetry(
            &timings,
            &mut hotspots,
            self.get_concept_surface_mode(),
            &self.concept_engine,
            self.runtime.concept_surface_counters(),
        );
        let timestamp = chrono::Utc::now().to_rfc3339();
        let cumulative_corrections = self.get_correction_log().len();
        let trend_summary = {
            let mut history = self.runtime.maintenance_trends.write();
            let previous_cumulative_corrections = history
                .last()
                .map(|snapshot| snapshot.cumulative_corrections)
                .unwrap_or(cumulative_corrections);
            let snapshot = MaintenanceService::build_trend_snapshot(
                timestamp.clone(),
                total_records,
                records_archived,
                insights_found,
                &epistemic,
                &belief_phase_report,
                &causal_phase_report,
                &policy_phase_report,
                &feedback_phase_report,
                &timings,
                &hotspots,
                cumulative_corrections,
                previous_cumulative_corrections,
            );
            MaintenanceService::push_trend_snapshot(&mut history, snapshot);
            let _ = save_maintenance_trends(&self.config.path, &history);
            MaintenanceService::summarize_trends(&history)
        };
        let reflection = {
            let contradiction_clusters = self.get_contradiction_clusters(None, Some(4));
            let records = self.records.read();
            let summary = MaintenanceService::build_reflection_summary(
                timestamp.clone(),
                &records,
                &config.task_tag,
                &contradiction_clusters,
                &trend_summary,
                &hotspots,
            );
            drop(records);

            let mut history = self.runtime.reflection_summaries.write();
            MaintenanceService::push_reflection_summary(&mut history, summary.clone());
            let _ = save_reflection_summaries(&self.config.path, &history);
            summary
        };

        MaintenanceReport {
            timestamp,
            decay,
            reflect,
            epistemic,
            insights_found,
            belief: belief_phase_report,
            concept: concept_phase_report,
            causal: causal_phase_report,
            policy: policy_phase_report,
            feedback: feedback_phase_report,
            consolidation: consolidation_report,
            cross_connections,
            task_reminders,
            records_archived,
            total_records,
            experience_injected,
            timings,
            stability,
            concept_surface,
            reflection,
            trend_summary,
            hotspots,
        }
    }

    /// Return the bounded persisted maintenance trend history.
    pub fn get_maintenance_trend_history(&self) -> Vec<background_brain::MaintenanceTrendSnapshot> {
        self.runtime.maintenance_trends.read().clone()
    }

    /// Return a compact summary over the bounded persisted maintenance trend history.
    pub fn get_maintenance_trend_summary(&self) -> background_brain::MaintenanceTrendSummary {
        let history = self.runtime.maintenance_trends.read();
        MaintenanceService::summarize_trends(&history)
    }

    /// Return the bounded persisted reflection history.
    pub fn get_reflection_summaries(
        &self,
        limit: Option<usize>,
    ) -> Vec<background_brain::ReflectionSummary> {
        let max = limit.unwrap_or(8).clamp(1, 32);
        let history = self.runtime.reflection_summaries.read();
        history.iter().rev().take(max).cloned().collect()
    }

    /// Return the latest persisted reflection digest, if any.
    pub fn get_latest_reflection_digest(&self) -> Option<background_brain::ReflectionSummary> {
        self.runtime.reflection_summaries.read().last().cloned()
    }

    /// Return a bounded aggregated digest across recent reflection summaries.
    pub fn get_reflection_digest(
        &self,
        limit: Option<usize>,
    ) -> background_brain::ReflectionDigest {
        let max = limit.unwrap_or(8).clamp(1, 32);
        let history = self.runtime.reflection_summaries.read();
        let start = history.len().saturating_sub(max);
        MaintenanceService::summarize_reflections(&history[start..])
    }

    /// Return the startup validation and recovery report for the current runtime.
    pub fn get_startup_validation_report(&self) -> StartupValidationReport {
        self.runtime.startup_validation.read().clone()
    }

    /// Return the current persistence manifest describing versioned persisted surfaces.
    pub fn get_persistence_manifest(&self) -> PersistenceManifest {
        self.runtime.persistence_manifest.read().clone()
    }

    /// Return a compact operator-facing digest of current memory health and pressure hotspots.
    pub fn get_memory_health_digest(&self, limit: Option<usize>) -> MemoryHealthDigest {
        let max = limit.unwrap_or(8).min(20);
        let total_records = self.records.read().len();
        let startup = self.get_startup_validation_report();
        let salience = self.get_salience_summary();
        let instability = self.get_belief_instability_summary();
        let reflection = self.get_reflection_digest(Some(max));
        let contradiction_clusters = self.get_contradiction_clusters(None, Some(max));
        let lifecycle = self.get_policy_lifecycle_summary(Some(max), Some(max));
        let pressure = self.get_policy_pressure_report(None, Some(max));
        let trend_summary = self.get_maintenance_trend_summary();
        let mut corrections = self.get_correction_log();
        corrections.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        corrections.truncate(max);

        let trend_direction = derive_maintenance_trend_direction(&trend_summary);
        let mut issues = Vec::new();

        for belief in self.get_high_volatility_beliefs(Some(0.20), Some(max)) {
            issues.push(OperatorReviewIssue {
                kind: "belief_instability".into(),
                target_id: belief.id.clone(),
                namespace: belief
                    .key
                    .split(':')
                    .next()
                    .unwrap_or(crate::record::DEFAULT_NAMESPACE)
                    .to_string(),
                title: format!("High-volatility belief {}", belief.key),
                score: belief.volatility,
                severity: issue_severity(belief.volatility, 0.45, 0.25),
            });
        }

        for cluster in contradiction_clusters.iter().take(max) {
            issues.push(OperatorReviewIssue {
                kind: "contradiction_cluster".into(),
                target_id: cluster.id.clone(),
                namespace: cluster.namespace.clone(),
                title: format!(
                    "Contradiction cluster with {} beliefs",
                    cluster.belief_ids.len()
                ),
                score: cluster.avg_volatility + cluster.total_conflict_mass.min(1.0),
                severity: issue_severity(
                    cluster.avg_volatility + cluster.total_conflict_mass.min(1.0),
                    1.2,
                    0.6,
                ),
            });
        }

        for finding in reflection.top_findings.iter().take(max) {
            issues.push(OperatorReviewIssue {
                kind: "reflection_finding".into(),
                target_id: finding.related_ids.first().cloned().unwrap_or_default(),
                namespace: finding.namespace.clone(),
                title: finding.title.clone(),
                score: finding.score,
                severity: finding.severity.clone(),
            });
        }

        for rec in self.get_high_salience_records(Some(0.70), Some(max)) {
            issues.push(OperatorReviewIssue {
                kind: "high_salience_record".into(),
                target_id: rec.id.clone(),
                namespace: rec.namespace.clone(),
                title: format!("High-salience record {}", preview_text(&rec.content, 48)),
                score: rec.salience,
                severity: issue_severity(rec.salience, 0.90, 0.70),
            });
        }

        for area in pressure.iter().take(max) {
            issues.push(OperatorReviewIssue {
                kind: "policy_pressure".into(),
                target_id: area.strongest_hint_id.clone(),
                namespace: area.namespace.clone(),
                title: format!("Policy pressure in {}:{}", area.namespace, area.domain),
                score: area.advisory_pressure,
                severity: issue_severity(area.advisory_pressure, 1.2, 0.7),
            });
        }

        for entry in corrections.iter().take(max) {
            issues.push(OperatorReviewIssue {
                kind: "recent_correction".into(),
                target_id: entry.target_id.clone(),
                namespace: infer_namespace_from_correction_target(entry)
                    .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                title: format!("Recent {} {}", entry.operation, entry.target_kind),
                score: 1.0,
                severity: "medium".into(),
            });
        }

        issues.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.target_id.cmp(&b.target_id))
        });
        issues.truncate(max);

        MemoryHealthDigest {
            total_records,
            startup_has_recovery_warnings: startup.has_recovery_warnings,
            high_salience_record_count: salience.high_salience_count,
            avg_salience: salience.avg_salience,
            max_salience: salience.max_salience,
            reflection_summary_count: reflection.summary_count,
            reflection_high_severity_findings: reflection.high_severity_findings,
            contradiction_cluster_count: contradiction_clusters.len(),
            high_volatility_belief_count: instability.high_volatility_count,
            low_stability_belief_count: instability.low_stability_count,
            recent_correction_count: corrections.len(),
            suppressed_policy_hint_count: lifecycle.suppressed_hints,
            rejected_policy_hint_count: lifecycle.rejected_hints,
            policy_pressure_area_count: pressure.len(),
            maintenance_trend_direction: trend_direction,
            latest_dominant_phase: trend_summary.latest_dominant_phase,
            top_issues: issues,
        }
    }

    /// Return recent correction candidates sorted for operator review by downstream impact,
    /// repeated correction pressure, and recency. Advisory only — does not apply any changes.
    pub fn get_correction_review_queue(
        &self,
        limit: Option<usize>,
    ) -> Vec<CorrectionReviewCandidate> {
        let max = limit.unwrap_or(10).min(50);
        let mut corrections = self.get_correction_log();
        if corrections.is_empty() {
            return Vec::new();
        }
        corrections.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        let repeat_counts = corrections.iter().fold(
            std::collections::HashMap::<(String, String), usize>::new(),
            |mut acc, entry| {
                *acc.entry((entry.target_kind.clone(), entry.target_id.clone()))
                    .or_insert(0) += 1;
                acc
            },
        );
        let beliefs = self.get_beliefs(None);
        let causal_patterns = self.get_causal_patterns(None);
        let policy_hints = self.get_policy_hints(None);
        let records = self.records.read();

        let belief_namespaces = beliefs
            .iter()
            .map(|belief| {
                (
                    belief.id.clone(),
                    belief
                        .key
                        .split(':')
                        .next()
                        .unwrap_or(crate::record::DEFAULT_NAMESPACE)
                        .to_string(),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        let causal_namespaces = causal_patterns
            .iter()
            .map(|pattern| (pattern.id.clone(), pattern.namespace.clone()))
            .collect::<std::collections::HashMap<_, _>>();
        let policy_namespaces = policy_hints
            .iter()
            .map(|hint| (hint.id.clone(), hint.namespace.clone()))
            .collect::<std::collections::HashMap<_, _>>();
        let total = corrections.len().max(1) as f32;

        let mut queue = corrections
            .into_iter()
            .enumerate()
            .map(|(idx, entry)| {
                let dependent_causal_patterns = match entry.target_kind.as_str() {
                    "belief" => causal_patterns
                        .iter()
                        .filter(|pattern| {
                            matches!(
                                pattern.state,
                                crate::causal::CausalState::Candidate
                                    | crate::causal::CausalState::Stable
                            ) && (pattern.cause_belief_ids.contains(&entry.target_id)
                                || pattern.effect_belief_ids.contains(&entry.target_id))
                        })
                        .count(),
                    "record" => causal_patterns
                        .iter()
                        .filter(|pattern| {
                            matches!(
                                pattern.state,
                                crate::causal::CausalState::Candidate
                                    | crate::causal::CausalState::Stable
                            ) && (pattern.cause_record_ids.contains(&entry.target_id)
                                || pattern.effect_record_ids.contains(&entry.target_id))
                        })
                        .count(),
                    _ => 0,
                };
                let dependent_policy_hints = match entry.target_kind.as_str() {
                    "belief" => policy_hints
                        .iter()
                        .filter(|hint| {
                            hint.state != crate::policy::PolicyState::Rejected
                                && hint.trigger_belief_ids.contains(&entry.target_id)
                        })
                        .count(),
                    "causal_pattern" => policy_hints
                        .iter()
                        .filter(|hint| {
                            hint.state != crate::policy::PolicyState::Rejected
                                && hint.trigger_causal_ids.contains(&entry.target_id)
                        })
                        .count(),
                    "record" => policy_hints
                        .iter()
                        .filter(|hint| {
                            hint.state != crate::policy::PolicyState::Rejected
                                && (hint.supporting_record_ids.contains(&entry.target_id)
                                    || hint.cause_record_ids.contains(&entry.target_id))
                        })
                        .count(),
                    _ => 0,
                };
                let downstream_impact = dependent_causal_patterns + dependent_policy_hints;
                let repeat_count = repeat_counts
                    .get(&(entry.target_kind.clone(), entry.target_id.clone()))
                    .copied()
                    .unwrap_or(1);
                let recency_score = (1.0 - (idx as f32 / total) * 0.85).clamp(0.15, 1.0);
                let priority_score = downstream_impact as f32 * 1.6
                    + repeat_count.saturating_sub(1) as f32 * 0.9
                    + recency_score;
                let namespace = match entry.target_kind.as_str() {
                    "belief" => belief_namespaces
                        .get(&entry.target_id)
                        .cloned()
                        .or_else(|| infer_namespace_from_correction_target(&entry))
                        .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                    "causal_pattern" => causal_namespaces
                        .get(&entry.target_id)
                        .cloned()
                        .or_else(|| infer_namespace_from_correction_target(&entry))
                        .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                    "policy_hint" => policy_namespaces
                        .get(&entry.target_id)
                        .cloned()
                        .or_else(|| infer_namespace_from_correction_target(&entry))
                        .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                    "record" => records
                        .get(&entry.target_id)
                        .map(|record| record.namespace.clone())
                        .or_else(|| infer_namespace_from_correction_target(&entry))
                        .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                    _ => infer_namespace_from_correction_target(&entry)
                        .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                };
                let title = format!(
                    "Review {} {} ({})",
                    entry.operation, entry.target_kind, entry.reason
                );

                CorrectionReviewCandidate {
                    timestamp: entry.timestamp,
                    time_iso: entry.time_iso,
                    target_kind: entry.target_kind,
                    target_id: entry.target_id,
                    operation: entry.operation,
                    reason: entry.reason,
                    session_id: entry.session_id,
                    namespace,
                    title,
                    repeat_count,
                    dependent_causal_patterns,
                    dependent_policy_hints,
                    downstream_impact,
                    priority_score,
                    severity: issue_severity(priority_score, 5.0, 2.5),
                }
            })
            .collect::<Vec<_>>();

        queue.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.timestamp.cmp(&a.timestamp))
                .then_with(|| a.target_kind.cmp(&b.target_kind))
                .then_with(|| a.target_id.cmp(&b.target_id))
        });
        queue.truncate(max);
        queue
    }

    /// Return contradiction clusters prioritized for operator review by volatility,
    /// conflict mass, unresolved breadth, and downstream causal/policy impact.
    pub fn get_contradiction_review_queue(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<ContradictionReviewCandidate> {
        let max = limit.unwrap_or(10).min(50);
        let clusters = self.get_contradiction_clusters(namespace, Some(max * 3));
        if clusters.is_empty() {
            return Vec::new();
        }

        let causal_patterns = self.get_causal_patterns(None);
        let policy_hints = self.get_policy_hints(None);

        let mut queue = clusters
            .into_iter()
            .map(|cluster| {
                let dependent_causal_patterns = causal_patterns
                    .iter()
                    .filter(|pattern| {
                        matches!(
                            pattern.state,
                            crate::causal::CausalState::Candidate
                                | crate::causal::CausalState::Stable
                        ) && cluster.belief_ids.iter().any(|belief_id| {
                            pattern.cause_belief_ids.contains(belief_id)
                                || pattern.effect_belief_ids.contains(belief_id)
                        })
                    })
                    .count();
                let dependent_policy_hints = policy_hints
                    .iter()
                    .filter(|hint| {
                        hint.state != crate::policy::PolicyState::Rejected
                            && cluster
                                .belief_ids
                                .iter()
                                .any(|belief_id| hint.trigger_belief_ids.contains(belief_id))
                    })
                    .count();
                let downstream_impact = dependent_causal_patterns + dependent_policy_hints;

                let priority_score = (cluster.avg_volatility * 4.0
                    + cluster.total_conflict_mass.min(2.0)
                    + cluster.unresolved_belief_count as f32 * 0.8
                    + cluster.high_volatility_belief_count as f32 * 0.6
                    + downstream_impact as f32 * 0.9
                    + (1.5 - cluster.avg_stability.min(1.5)))
                .min(CONTRADICTION_REVIEW_PRIORITY_MAX);

                let title = format!(
                    "Review contradiction cluster in {} ({} beliefs, {} records)",
                    cluster.namespace,
                    cluster.belief_ids.len(),
                    cluster.record_ids.len()
                );

                ContradictionReviewCandidate {
                    cluster_id: cluster.id,
                    namespace: cluster.namespace,
                    title,
                    belief_ids: cluster.belief_ids,
                    belief_keys: cluster.belief_keys,
                    record_ids: cluster.record_ids,
                    shared_tags: cluster.shared_tags,
                    unresolved_belief_count: cluster.unresolved_belief_count,
                    high_volatility_belief_count: cluster.high_volatility_belief_count,
                    dependent_causal_patterns,
                    dependent_policy_hints,
                    downstream_impact,
                    total_conflict_mass: cluster.total_conflict_mass,
                    avg_volatility: cluster.avg_volatility,
                    avg_stability: cluster.avg_stability,
                    priority_score,
                    severity: issue_severity(priority_score, 5.0, 2.5),
                }
            })
            .collect::<Vec<_>>();

        queue.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.downstream_impact.cmp(&a.downstream_impact))
                .then_with(|| b.unresolved_belief_count.cmp(&a.unresolved_belief_count))
                .then_with(|| a.cluster_id.cmp(&b.cluster_id))
        });
        queue.truncate(max);
        queue
    }

    /// Return bounded suggested corrections without auto-applying them.
    /// This advisory surface combines instability, lifecycle state, and review pressure.
    pub fn get_suggested_corrections(&self, limit: Option<usize>) -> Vec<SuggestedCorrection> {
        self.get_suggested_corrections_report(limit).entries
    }

    /// Return bounded suggested corrections together with scan latency.
    pub fn get_suggested_corrections_report(
        &self,
        limit: Option<usize>,
    ) -> SuggestedCorrectionsReport {
        let started = std::time::Instant::now();
        let max = limit.unwrap_or(10).min(50);
        let review_queue = self.get_correction_review_queue(Some(max.saturating_mul(3).max(8)));
        let review_lookup = review_queue
            .into_iter()
            .map(|item| ((item.target_kind.clone(), item.target_id.clone()), item))
            .collect::<std::collections::HashMap<_, _>>();
        let belief_engine = self.belief_engine.read();
        let causal_engine = self.causal_engine.read();
        let policy_engine = self.policy_engine.read();
        let mut suggestions =
            std::collections::HashMap::<(String, String), SuggestedCorrection>::new();

        let mut upsert = |candidate: SuggestedCorrection| {
            let key = (candidate.target_kind.clone(), candidate.target_id.clone());
            match suggestions.get(&key) {
                Some(existing) if existing.priority_score >= candidate.priority_score => {}
                _ => {
                    suggestions.insert(key, candidate);
                }
            }
        };

        for belief in belief_engine
            .beliefs
            .values()
            .filter(|belief| belief.volatility >= 0.20)
        {
            let representative_record_id = belief
                .winner_id
                .as_ref()
                .and_then(|winner_id| belief_engine.hypotheses.get(winner_id))
                .and_then(|hyp| hyp.prototype_record_ids.first())
                .cloned();
            let review = review_lookup.get(&("belief".to_string(), belief.id.clone()));
            let review_boost = review.map(|item| item.priority_score * 0.25).unwrap_or(0.0);
            let score = belief.volatility * 4.0 + review_boost;
            let namespace = belief
                .key
                .split(':')
                .next()
                .unwrap_or(crate::record::DEFAULT_NAMESPACE)
                .to_string();
            upsert(SuggestedCorrection {
                target_kind: "belief".into(),
                target_id: belief.id.clone(),
                namespace,
                reason_kind: "HighVolatility".into(),
                suggested_action: "Deprecate".into(),
                reason_detail: format!(
                    "belief volatility {:.2} exceeded bounded review threshold",
                    belief.volatility
                ),
                priority_score: score,
                severity: issue_severity(score, 3.8, 2.2),
                supporting_record_id: representative_record_id.clone(),
                provenance: representative_record_id
                    .as_deref()
                    .and_then(|record_id| self.provenance_chain(record_id)),
            });
        }

        for belief in belief_engine
            .beliefs
            .values()
            .filter(|belief| belief.stability <= 1.0)
        {
            let causal_pressure = causal_engine
                .patterns
                .values()
                .filter(|pattern| {
                    matches!(
                        pattern.state,
                        crate::causal::CausalState::Candidate | crate::causal::CausalState::Stable
                    ) && (pattern.cause_belief_ids.contains(&belief.id)
                        || pattern.effect_belief_ids.contains(&belief.id))
                })
                .count();
            if causal_pressure == 0 {
                continue;
            }
            let representative_record_id = belief
                .winner_id
                .as_ref()
                .and_then(|winner_id| belief_engine.hypotheses.get(winner_id))
                .and_then(|hyp| hyp.prototype_record_ids.first())
                .cloned();
            let review = review_lookup.get(&("belief".to_string(), belief.id.clone()));
            let review_boost = review.map(|item| item.priority_score * 0.2).unwrap_or(0.0);
            let score = (2.0 - belief.stability).max(0.0) * 1.5
                + causal_pressure as f32 * 1.1
                + review_boost;
            let namespace = belief
                .key
                .split(':')
                .next()
                .unwrap_or(crate::record::DEFAULT_NAMESPACE)
                .to_string();
            upsert(SuggestedCorrection {
                target_kind: "belief".into(),
                target_id: belief.id.clone(),
                namespace,
                reason_kind: "LowStabilityWithCausalPressure".into(),
                suggested_action: "Deprecate".into(),
                reason_detail: format!(
                    "belief stability {:.2} is low while {} causal patterns still depend on it",
                    belief.stability, causal_pressure
                ),
                priority_score: score,
                severity: issue_severity(score, 3.8, 2.2),
                supporting_record_id: representative_record_id.clone(),
                provenance: representative_record_id
                    .as_deref()
                    .and_then(|record_id| self.provenance_chain(record_id)),
            });
        }

        for pattern in causal_engine.patterns.values().filter(|pattern| {
            matches!(pattern.state, crate::causal::CausalState::Rejected)
                || (pattern.counterevidence > 0
                    && !matches!(pattern.state, crate::causal::CausalState::Invalidated))
        }) {
            let representative_record_id = pattern
                .cause_record_ids
                .first()
                .cloned()
                .or_else(|| pattern.effect_record_ids.first().cloned());
            let review = review_lookup.get(&("causal_pattern".to_string(), pattern.id.clone()));
            let review_boost = review.map(|item| item.priority_score * 0.25).unwrap_or(0.0);
            let score = pattern.counterevidence as f32 * 1.4
                + (1.0 - pattern.causal_strength).max(0.0)
                + review_boost;
            upsert(SuggestedCorrection {
                target_kind: "causal_pattern".into(),
                target_id: pattern.id.clone(),
                namespace: pattern.namespace.clone(),
                reason_kind: "InvalidationProne".into(),
                suggested_action: "Invalidate".into(),
                reason_detail: format!(
                    "causal pattern has {} counterevidence edges with strength {:.2}",
                    pattern.counterevidence, pattern.causal_strength
                ),
                priority_score: score,
                severity: issue_severity(score, 3.8, 2.2),
                supporting_record_id: representative_record_id.clone(),
                provenance: representative_record_id
                    .as_deref()
                    .and_then(|record_id| self.provenance_chain(record_id)),
            });
        }

        for hint in policy_engine.hints.values().filter(|hint| {
            matches!(
                hint.state,
                crate::policy::PolicyState::Suppressed | crate::policy::PolicyState::Rejected
            )
        }) {
            let representative_record_id = hint
                .supporting_record_ids
                .first()
                .cloned()
                .or_else(|| hint.cause_record_ids.first().cloned());
            let review = review_lookup.get(&("policy_hint".to_string(), hint.id.clone()));
            let repeat_count = review.map(|item| item.repeat_count).unwrap_or(1);
            let repeat_weight = repeat_count.saturating_sub(1) as f32 * 1.1;
            let state_weight = match hint.state {
                crate::policy::PolicyState::Rejected => 1.8,
                crate::policy::PolicyState::Suppressed => 1.2,
                _ => 0.0,
            };
            let score = state_weight + hint.policy_strength * 1.5 + repeat_weight;
            upsert(SuggestedCorrection {
                target_kind: "policy_hint".into(),
                target_id: hint.id.clone(),
                namespace: hint.namespace.clone(),
                reason_kind: "RepeatedSuppression".into(),
                suggested_action: "Retract".into(),
                reason_detail: format!(
                    "policy hint is {:?} with strength {:.2} and repeat pressure {}",
                    hint.state, hint.policy_strength, repeat_count
                )
                .to_lowercase(),
                priority_score: score,
                severity: issue_severity(score, 3.8, 2.2),
                supporting_record_id: representative_record_id.clone(),
                provenance: representative_record_id
                    .as_deref()
                    .and_then(|record_id| self.provenance_chain(record_id)),
            });
        }

        let mut items = suggestions.into_values().collect::<Vec<_>>();
        items.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.target_kind.cmp(&b.target_kind))
                .then_with(|| a.target_id.cmp(&b.target_id))
        });
        items.truncate(max);
        SuggestedCorrectionsReport {
            scan_latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            entries: items,
        }
    }

    /// Return read-only governance status per namespace.
    /// This preserves recall isolation and only exposes bounded operator summaries.
    pub fn get_namespace_governance_status(&self) -> Vec<NamespaceGovernanceStatus> {
        self.get_namespace_governance_status_filtered(None)
    }

    /// Return read-only governance status for all or a filtered subset of namespaces.
    pub fn get_namespace_governance_status_filtered(
        &self,
        namespaces: Option<&[&str]>,
    ) -> Vec<NamespaceGovernanceStatus> {
        let namespace_filter = namespaces.map(|items| {
            items
                .iter()
                .map(|item| (*item).to_string())
                .collect::<std::collections::HashSet<_>>()
        });
        let records = self.records.read();
        let beliefs = self.get_beliefs(None);
        let corrections = self.get_correction_log();
        let suggested = self.get_suggested_corrections(Some(64));
        let trend_history = self.get_maintenance_trend_history();
        let last_cycle = trend_history
            .last()
            .map(|snapshot| snapshot.timestamp.clone());
        let latest_dominant_phase = trend_history
            .last()
            .map(|snapshot| snapshot.dominant_phase.clone())
            .unwrap_or_else(|| "none".to_string());

        let mut namespace_list = records
            .values()
            .map(|record| record.namespace.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        namespace_list.sort();
        if let Some(filter) = &namespace_filter {
            namespace_list.retain(|namespace| filter.contains(namespace));
        }

        let mut correction_counts = std::collections::HashMap::<String, usize>::new();
        let belief_namespaces = beliefs
            .iter()
            .map(|belief| {
                (
                    belief.id.clone(),
                    belief
                        .key
                        .split(':')
                        .next()
                        .unwrap_or(crate::record::DEFAULT_NAMESPACE)
                        .to_string(),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        let causal_namespaces = self
            .get_causal_patterns(None)
            .into_iter()
            .map(|pattern| (pattern.id, pattern.namespace))
            .collect::<std::collections::HashMap<_, _>>();
        let policy_namespaces = self
            .get_policy_hints(None)
            .into_iter()
            .map(|hint| (hint.id, hint.namespace))
            .collect::<std::collections::HashMap<_, _>>();
        for entry in corrections {
            let namespace = match entry.target_kind.as_str() {
                "belief" => belief_namespaces
                    .get(&entry.target_id)
                    .cloned()
                    .or_else(|| infer_namespace_from_correction_target(&entry))
                    .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                "causal_pattern" => causal_namespaces
                    .get(&entry.target_id)
                    .cloned()
                    .or_else(|| infer_namespace_from_correction_target(&entry))
                    .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                "policy_hint" => policy_namespaces
                    .get(&entry.target_id)
                    .cloned()
                    .or_else(|| infer_namespace_from_correction_target(&entry))
                    .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                "record" => records
                    .get(&entry.target_id)
                    .map(|record| record.namespace.clone())
                    .or_else(|| infer_namespace_from_correction_target(&entry))
                    .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
                _ => infer_namespace_from_correction_target(&entry)
                    .unwrap_or_else(|| crate::record::DEFAULT_NAMESPACE.to_string()),
            };
            *correction_counts.entry(namespace).or_insert(0) += 1;
        }

        let mut statuses = namespace_list
            .into_iter()
            .map(|namespace| {
                let record_count = records
                    .values()
                    .filter(|record| record.namespace == namespace)
                    .count();
                let namespace_beliefs = beliefs
                    .iter()
                    .filter(|belief| belief.key.starts_with(&format!("{namespace}:")))
                    .collect::<Vec<_>>();
                let belief_count = namespace_beliefs.len();
                let high_volatility_belief_count = namespace_beliefs
                    .iter()
                    .filter(|belief| belief.volatility >= 0.20)
                    .count();
                let low_stability_belief_count = namespace_beliefs
                    .iter()
                    .filter(|belief| belief.stability <= 1.0)
                    .count();
                let correction_count = correction_counts.get(&namespace).copied().unwrap_or(0);
                let correction_density = if record_count == 0 {
                    0.0
                } else {
                    correction_count as f32 / record_count as f32
                };
                let policy_pressure_area_count = self
                    .get_policy_pressure_report(Some(&namespace), Some(64))
                    .len();
                let suggested_correction_count = suggested
                    .iter()
                    .filter(|item| item.namespace == namespace)
                    .count();
                let instability_score = high_volatility_belief_count as f32 * 1.5
                    + low_stability_belief_count as f32 * 1.2
                    + correction_density * 3.0
                    + policy_pressure_area_count as f32 * 0.8;
                let instability_level = issue_severity(instability_score, 4.0, 1.8);

                NamespaceGovernanceStatus {
                    namespace,
                    record_count,
                    belief_count,
                    correction_count,
                    correction_density,
                    high_volatility_belief_count,
                    low_stability_belief_count,
                    instability_score,
                    instability_level,
                    policy_pressure_area_count,
                    suggested_correction_count,
                    last_maintenance_cycle: last_cycle.clone(),
                    latest_dominant_phase: latest_dominant_phase.clone(),
                }
            })
            .collect::<Vec<_>>();

        statuses.sort_by(|a, b| {
            b.instability_score
                .partial_cmp(&a.instability_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.namespace.cmp(&b.namespace))
        });
        statuses
    }

    /// Start the autonomous experience loop (Phase 2.2).
    ///
    /// The loop drains the experience queue and runs `run_maintenance()`
    /// automatically on the given interval.  Requires `Arc<Self>` so the
    /// thread can hold a reference without a lifetime.
    ///
    /// Returns an `ExperienceLoopHandle` — drop or call `.stop()` to shut
    /// the thread down gracefully.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::sync::Arc;
    /// use aura::background_brain::ExperienceLoopConfig;
    ///
    /// let aura = Arc::new(Aura::open("/tmp/my_aura").unwrap());
    /// aura.set_plasticity_mode(aura::experience::PlasticityMode::Limited);
    ///
    /// let handle = aura.clone().start_experience_loop(ExperienceLoopConfig {
    ///     interval_secs: 30,
    ///     ..Default::default()
    /// });
    ///
    /// // Captures are now processed automatically every 30 seconds.
    /// handle.stop();
    /// ```
    pub fn start_experience_loop(
        self: std::sync::Arc<Self>,
        config: crate::background_brain::ExperienceLoopConfig,
    ) -> crate::background_brain::ExperienceLoopHandle {
        crate::background_brain::start_experience_loop(self, config)
    }

    /// Start background maintenance loop (daemon thread).
    pub fn start_background(&self, interval_secs: Option<u64>) {
        let _interval = interval_secs.unwrap_or(120);
        let mut bg = self.runtime.background.write();
        if bg.as_ref().is_some_and(|b| b.is_running()) {
            return; // Already running
        }
        // Create BackgroundBrain controller (actual thread spawning
        // requires Arc<Self> which we can't get from &self — the CLI
        // wrapper handles the loop externally via run_maintenance())
        *bg = Some(BackgroundBrain::new());
    }

    /// Stop background maintenance loop.
    pub fn stop_background(&self) {
        let mut bg = self.runtime.background.write();
        if let Some(ref mut brain) = *bg {
            brain.stop();
        }
        *bg = None;
    }

    /// Check if background maintenance is running.
    pub fn is_background_running(&self) -> bool {
        let bg = self.runtime.background.read();
        bg.as_ref().is_some_and(|b| b.is_running())
    }

    // ── Inspection Helpers (observability) ──

    /// Build a read-only epistemic inspection facade over the current
    /// belief/concept/causal/policy runtime state.
    pub fn epistemic_runtime(&self) -> EpistemicRuntime<'_> {
        EpistemicRuntime::new(
            self.records.read(),
            self.belief_engine.read(),
            self.concept_engine.read(),
            self.causal_engine.read(),
            self.policy_engine.read(),
            self.get_concept_surface_mode(),
            &self.runtime.concept_surface_global_calls,
            &self.runtime.concept_surface_namespace_calls,
            &self.runtime.concept_surface_record_calls,
            &self.runtime.concept_surface_results_returned,
            &self.runtime.concept_surface_record_results_returned,
        )
    }

    /// Return a snapshot of all current beliefs (cloned).
    /// Optional filter by state: "resolved", "unresolved", "singleton", "empty".
    pub fn get_beliefs(&self, state_filter: Option<&str>) -> Vec<crate::belief::Belief> {
        self.epistemic_runtime().get_beliefs(state_filter)
    }

    /// Return the belief that currently owns a record, if any.
    pub fn get_belief_for_record(&self, record_id: &str) -> Option<crate::belief::Belief> {
        self.epistemic_runtime().get_belief_for_record(record_id)
    }

    /// Return beliefs with elevated volatility, highest volatility first.
    pub fn get_high_volatility_beliefs(
        &self,
        min_volatility: Option<f32>,
        limit: Option<usize>,
    ) -> Vec<crate::belief::Belief> {
        self.epistemic_runtime()
            .get_high_volatility_beliefs(min_volatility, limit)
    }

    /// Return beliefs with low stability, lowest stability first.
    pub fn get_low_stability_beliefs(
        &self,
        max_stability: Option<f32>,
        limit: Option<usize>,
    ) -> Vec<crate::belief::Belief> {
        self.epistemic_runtime()
            .get_low_stability_beliefs(max_stability, limit)
    }

    /// Return a compact instability summary over the current belief layer.
    pub fn get_belief_instability_summary(
        &self,
    ) -> crate::epistemic_runtime::BeliefInstabilitySummary {
        self.epistemic_runtime().get_belief_instability_summary()
    }

    /// Return deterministic contradiction clusters derived from unstable belief groups.
    pub fn get_contradiction_clusters(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<crate::epistemic_runtime::ContradictionCluster> {
        self.epistemic_runtime()
            .get_contradiction_clusters(namespace, limit)
    }

    /// Return beliefs that were explicitly corrected most recently.
    pub fn get_recently_corrected_beliefs(
        &self,
        limit: Option<usize>,
    ) -> Vec<crate::belief::Belief> {
        let max = limit.unwrap_or(20).min(100);
        let belief_eng = self.belief_engine.read();
        let mut corrections: Vec<_> = self
            .get_correction_log()
            .into_iter()
            .filter(|entry| entry.target_kind == "belief")
            .collect();
        corrections.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        corrections.reverse();

        let mut seen = HashSet::new();
        let mut beliefs = Vec::new();
        for entry in corrections {
            if !seen.insert(entry.target_id.clone()) {
                continue;
            }
            if let Some(belief) = belief_eng.beliefs.get(&entry.target_id) {
                beliefs.push(belief.clone());
            }
            if beliefs.len() >= max {
                break;
            }
        }
        beliefs
    }

    /// Return a snapshot of all current concepts (cloned).
    /// Optional filter by state: "stable", "candidate", "rejected".
    pub fn get_concepts(
        &self,
        state_filter: Option<&str>,
    ) -> Vec<crate::concept::ConceptCandidate> {
        self.epistemic_runtime().get_concepts(state_filter)
    }

    /// Return surfaced concepts for external inspection.
    /// Returns bounded, sorted, provenance-checked concepts suitable for public consumption.
    /// This is inspection-only — surfaced concepts do not affect recall, compression, or behavior.
    pub fn get_surfaced_concepts(
        &self,
        limit: Option<usize>,
    ) -> Vec<crate::concept::SurfacedConcept> {
        self.epistemic_runtime().get_surfaced_concepts(limit)
    }

    /// Return surfaced concepts for a specific namespace.
    pub fn get_surfaced_concepts_for_namespace(
        &self,
        namespace: &str,
        limit: Option<usize>,
    ) -> Vec<crate::concept::SurfacedConcept> {
        self.epistemic_runtime()
            .get_surfaced_concepts_for_namespace(namespace, limit)
    }

    /// Return surfaced concepts that contain the given record ID.
    ///
    /// This remains bounded and inspection-only, and follows the same runtime
    /// rollout gate as other surfaced concept APIs.
    pub fn get_surfaced_concepts_for_record(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Vec<crate::concept::SurfacedConcept> {
        self.epistemic_runtime()
            .get_surfaced_concepts_for_record(record_id, limit)
    }

    fn upsert_structural_connection(
        rec: &mut Record,
        other_id: &str,
        weight: f32,
        relation_type: &str,
    ) -> bool {
        let current_weight = rec.connections.get(other_id).copied();
        let current_type = rec.connection_type(other_id);
        if current_weight == Some(weight) && current_type == Some(relation_type) {
            return false;
        }
        rec.add_typed_connection(other_id, weight, relation_type);
        true
    }

    fn build_relation_edges(
        &self,
        record_id: Option<&str>,
        limit: Option<usize>,
        structural_only: bool,
    ) -> Vec<RelationEdge> {
        let max = limit.unwrap_or(STRUCTURAL_RELATION_DEFAULT_LIMIT);
        let records = self.records.read();
        let mut seen_pairs = HashSet::new();
        let mut edges = Vec::new();

        let iter: Box<dyn Iterator<Item = &Record>> = match record_id {
            Some(record_id) => match records.get(record_id) {
                Some(rec) => Box::new(std::iter::once(rec)),
                None => Box::new(std::iter::empty()),
            },
            None => Box::new(records.values()),
        };

        for rec in iter {
            for (other_id, relation_type) in &rec.connection_types {
                if structural_only && !relation::is_structural_relation_type(relation_type) {
                    continue;
                }
                let Some(other) = records.get(other_id) else {
                    continue;
                };

                let has_reverse = other
                    .connection_type(&rec.id)
                    .map(|rev_type| rev_type == relation_type)
                    .unwrap_or(false);
                let rec_is_profile = rec.tags.iter().any(|tag| tag == identity::PROFILE_TAG);
                let other_is_profile = other.tags.iter().any(|tag| tag == identity::PROFILE_TAG);
                let rec_is_project = rec.tags.iter().any(|tag| tag == "research-project");
                let other_is_project = other.tags.iter().any(|tag| tag == "research-project");
                let is_record_scoped = record_id.is_some();
                let (source_id, target_id, dedupe_key) = if relation_type
                    == relation::PROJECT_MEMBERSHIP_RELATION
                    && rec_is_project
                    && !other_is_project
                {
                    (
                        rec.id.clone(),
                        other.id.clone(),
                        (rec.id.clone(), other.id.clone(), relation_type.clone()),
                    )
                } else if relation_type == relation::PROJECT_MEMBERSHIP_RELATION
                    && other_is_project
                    && !rec_is_project
                {
                    (
                        other.id.clone(),
                        rec.id.clone(),
                        (other.id.clone(), rec.id.clone(), relation_type.clone()),
                    )
                } else if relation::is_family_relation_type(relation_type)
                    && rec_is_profile
                    && !other_is_profile
                {
                    (
                        rec.id.clone(),
                        other.id.clone(),
                        (rec.id.clone(), other.id.clone(), relation_type.clone()),
                    )
                } else if relation::is_family_relation_type(relation_type)
                    && other_is_profile
                    && !rec_is_profile
                {
                    (
                        other.id.clone(),
                        rec.id.clone(),
                        (other.id.clone(), rec.id.clone(), relation_type.clone()),
                    )
                } else if is_record_scoped {
                    (
                        rec.id.clone(),
                        other.id.clone(),
                        (rec.id.clone(), other.id.clone(), relation_type.clone()),
                    )
                } else if has_reverse {
                    if rec.id <= other.id {
                        (
                            rec.id.clone(),
                            other.id.clone(),
                            (rec.id.clone(), other.id.clone(), relation_type.clone()),
                        )
                    } else {
                        (
                            other.id.clone(),
                            rec.id.clone(),
                            (other.id.clone(), rec.id.clone(), relation_type.clone()),
                        )
                    }
                } else {
                    (
                        rec.id.clone(),
                        other.id.clone(),
                        (rec.id.clone(), other.id.clone(), relation_type.clone()),
                    )
                };

                if !seen_pairs.insert(dedupe_key) {
                    continue;
                }

                edges.push(RelationEdge {
                    source_record_id: source_id,
                    target_record_id: target_id,
                    relation_type: relation_type.clone(),
                    weight: rec.connections.get(other_id).copied().unwrap_or(0.0),
                    namespace: rec.namespace.clone(),
                    structural: relation::is_structural_relation_type(relation_type),
                });

                if edges.len() >= max {
                    return edges;
                }
            }
        }

        edges
    }

    fn refresh_family_relations_for_namespace(&self, namespace: &str) -> Result<usize> {
        let mut changed_records: HashMap<String, Record> = HashMap::new();

        {
            let mut records = self.records.write();
            let Some(profile_id) = records
                .values()
                .find(|rec| {
                    rec.namespace == namespace
                        && rec.tags.iter().any(|tag| tag == identity::PROFILE_TAG)
                })
                .map(|rec| rec.id.clone())
            else {
                return Ok(0);
            };

            let relation_targets: Vec<(String, String)> = records
                .values()
                .filter(|rec| rec.namespace == namespace && rec.id != profile_id)
                .filter_map(|rec| {
                    rec.metadata
                        .get("family_relation")
                        .and_then(|relation_type| {
                            relation::is_family_relation_type(relation_type)
                                .then_some(relation_type.clone())
                        })
                        .or_else(|| {
                            relation::detect_family_relation(&rec.content)
                                .map(|relation_type| relation_type.to_string())
                        })
                        .map(|relation_type| (rec.id.clone(), relation_type))
                })
                .collect();

            for (record_id, relation_type) in relation_targets {
                if let Some(profile) = records.get_mut(&profile_id) {
                    if Self::upsert_structural_connection(
                        profile,
                        &record_id,
                        relation::STRUCTURAL_FAMILY_WEIGHT,
                        &relation_type,
                    ) {
                        changed_records.insert(profile.id.clone(), profile.clone());
                    }
                }

                if let Some(target) = records.get_mut(&record_id) {
                    if Self::upsert_structural_connection(
                        target,
                        &profile_id,
                        relation::STRUCTURAL_FAMILY_WEIGHT,
                        &relation_type,
                    ) {
                        changed_records.insert(target.id.clone(), target.clone());
                    }
                }
            }
        }

        if changed_records.is_empty() {
            return Ok(0);
        }

        for rec in changed_records.values() {
            self.cognitive_store.append_update(rec)?;
        }
        self.runtime.clear_recall_caches();

        Ok(changed_records.len())
    }

    fn find_profile_record(&self, namespace: Option<&str>) -> Option<Record> {
        let records = self.records.read();
        records
            .values()
            .filter(|rec| rec.tags.iter().any(|tag| tag == identity::PROFILE_TAG))
            .find(|rec| namespace.map(|ns| rec.namespace == ns).unwrap_or(true))
            .cloned()
    }

    fn normalize_entity_component(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        let mut last_sep = false;
        for ch in value.chars().flat_map(char::to_lowercase) {
            if ch.is_ascii_alphanumeric() {
                out.push(ch);
                last_sep = false;
            } else if !last_sep {
                out.push('-');
                last_sep = true;
            }
        }
        out.trim_matches('-').to_string()
    }

    fn derive_family_entity_id(
        relation_type: &str,
        metadata: &HashMap<String, String>,
        content: &str,
    ) -> Option<String> {
        if let Some(entity_id) = metadata.get("entity_id") {
            if !entity_id.trim().is_empty() {
                return Some(entity_id.clone());
            }
        }
        let seed = metadata
            .get("name")
            .map(|s| s.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(content);
        let normalized = Self::normalize_entity_component(seed);
        if normalized.is_empty() {
            None
        } else {
            Some(format!("person:{}:{}", relation_type, normalized))
        }
    }

    fn refresh_project_membership_relations_for_namespace(&self, namespace: &str) -> Result<usize> {
        let mut changed_records: HashMap<String, Record> = HashMap::new();

        {
            let mut records = self.records.write();
            let project_anchors: HashMap<String, String> = records
                .values()
                .filter(|rec| {
                    rec.namespace == namespace
                        && rec.tags.iter().any(|tag| tag == "research-project")
                })
                .filter_map(|rec| {
                    rec.metadata
                        .get("project_id")
                        .cloned()
                        .map(|project_id| (project_id, rec.id.clone()))
                })
                .collect();

            if project_anchors.is_empty() {
                return Ok(0);
            }

            let relation_targets: Vec<(String, String)> = records
                .values()
                .filter(|rec| rec.namespace == namespace)
                .filter_map(|rec| {
                    if rec.tags.iter().any(|tag| tag == "research-project") {
                        return None;
                    }
                    rec.metadata.get("project_id").and_then(|project_id| {
                        project_anchors
                            .get(project_id)
                            .map(|anchor_id| (rec.id.clone(), anchor_id.clone()))
                    })
                })
                .collect();

            for (report_id, project_id) in relation_targets {
                if let Some(project) = records.get_mut(&project_id) {
                    if Self::upsert_structural_connection(
                        project,
                        &report_id,
                        relation::STRUCTURAL_PROJECT_WEIGHT,
                        relation::PROJECT_MEMBERSHIP_RELATION,
                    ) {
                        changed_records.insert(project.id.clone(), project.clone());
                    }
                }

                if let Some(report) = records.get_mut(&report_id) {
                    if Self::upsert_structural_connection(
                        report,
                        &project_id,
                        relation::STRUCTURAL_PROJECT_WEIGHT,
                        relation::PROJECT_MEMBERSHIP_RELATION,
                    ) {
                        changed_records.insert(report.id.clone(), report.clone());
                    }
                }
            }
        }

        if changed_records.is_empty() {
            return Ok(0);
        }

        for rec in changed_records.values() {
            self.cognitive_store.append_update(rec)?;
        }
        self.runtime.clear_recall_caches();

        Ok(changed_records.len())
    }

    fn refresh_deterministic_relations_for_namespace(&self, namespace: &str) -> Result<usize> {
        let mut changed = 0;
        changed += self.refresh_family_relations_for_namespace(namespace)?;
        changed += self.refresh_project_membership_relations_for_namespace(namespace)?;
        Ok(changed)
    }

    fn ensure_research_project_anchor(&self, project: &ResearchProject) -> Result<Record> {
        let metadata = HashMap::from([
            ("project_id".to_string(), project.id.clone()),
            ("entity_id".to_string(), format!("project:{}", project.id)),
            ("project_topic".to_string(), project.topic.clone()),
            ("project_depth".to_string(), project.depth.clone()),
            (
                "project_status".to_string(),
                match project.status {
                    crate::research::ResearchStatus::Active => "active",
                    crate::research::ResearchStatus::Completed => "completed",
                    crate::research::ResearchStatus::Cancelled => "cancelled",
                }
                .to_string(),
            ),
            ("project_created_at".to_string(), project.created_at.clone()),
        ]);
        let content = format!(
            "Research Project: {}\nDepth: {}\nStatus: {}\nProject ID: {}",
            project.topic,
            project.depth,
            metadata.get("project_status").cloned().unwrap_or_default(),
            project.id
        );

        let existing_id = {
            let records = self.records.read();
            records
                .values()
                .find(|rec| {
                    rec.tags.iter().any(|tag| tag == "research-project")
                        && rec
                            .metadata
                            .get("project_id")
                            .map(|value| value == &project.id)
                            .unwrap_or(false)
                })
                .map(|rec| rec.id.clone())
        };

        if let Some(record_id) = existing_id {
            self.update(
                &record_id,
                Some(&content),
                None,
                None,
                None,
                Some(metadata),
                None,
            )?
            .ok_or_else(|| anyhow::anyhow!("Research project anchor disappeared"))
        } else {
            self.store(
                &content,
                Some(Level::Domain),
                Some(vec!["research-project".into()]),
                Some(false),
                None,
                Some("recorded"),
                Some(metadata),
                Some(false),
                None,
                None,
                Some("fact"),
            )
        }
    }

    fn ensure_project_anchor_by_id(&self, project_id: &str) -> Result<Record> {
        let existing_id = {
            let records = self.records.read();
            records
                .values()
                .find(|rec| {
                    rec.tags.iter().any(|tag| tag == "research-project")
                        && rec
                            .metadata
                            .get("project_id")
                            .map(|value| value == project_id)
                            .unwrap_or(false)
                })
                .map(|rec| rec.id.clone())
        };

        if let Some(record_id) = existing_id {
            return self
                .get(&record_id)
                .ok_or_else(|| anyhow::anyhow!("Project anchor disappeared"));
        }

        let project = self
            .research_engine
            .get_project(project_id)
            .ok_or_else(|| anyhow::anyhow!("Project {} not found", project_id))?;
        self.ensure_research_project_anchor(&project)
    }

    fn build_project_graph_snapshot(
        &self,
        project_anchor: &Record,
    ) -> Option<ProjectGraphSnapshot> {
        let project_id = project_anchor.metadata.get("project_id")?.clone();
        let project_topic = project_anchor
            .metadata
            .get("project_topic")
            .cloned()
            .unwrap_or_default();
        let records = self.records.read();
        let mut member_ids = Vec::new();
        let mut member_tags: HashMap<String, usize> = HashMap::new();

        for rec in records.values() {
            if rec.id == project_anchor.id {
                continue;
            }
            if rec.metadata.get("project_id") != Some(&project_id) {
                continue;
            }
            member_ids.push(rec.id.clone());
            for tag in &rec.tags {
                *member_tags.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        member_ids.sort();

        Some(ProjectGraphSnapshot {
            project_id,
            project_record_id: project_anchor.id.clone(),
            project_topic,
            namespace: project_anchor.namespace.clone(),
            relation_count: member_ids.len(),
            member_record_ids: member_ids,
            member_tags,
        })
    }

    fn collect_project_records(
        &self,
        project_id: &str,
    ) -> Option<(Record, HashMap<String, Record>)> {
        let project_anchor = self.ensure_project_anchor_by_id(project_id).ok()?;
        let records = self.records.read();
        let mut scoped_records = HashMap::new();
        scoped_records.insert(project_anchor.id.clone(), project_anchor.clone());

        for rec in records.values() {
            if rec.id == project_anchor.id {
                continue;
            }
            if rec.metadata.get("project_id") == Some(&project_id.to_string()) {
                scoped_records.insert(rec.id.clone(), rec.clone());
            }
        }

        Some((project_anchor, scoped_records))
    }

    fn record_marked_completed(rec: &Record) -> bool {
        rec.metadata
            .get("completed")
            .map(|v| matches!(v.as_str(), "true" | "1" | "yes"))
            .unwrap_or(false)
            || rec
                .metadata
                .get("status")
                .map(|v| matches!(v.as_str(), "done" | "completed" | "closed"))
                .unwrap_or(false)
    }

    fn parse_due_date_for_timeline(due_str: &str) -> Option<chrono::DateTime<chrono::FixedOffset>> {
        chrono::DateTime::parse_from_rfc3339(due_str)
            .ok()
            .or_else(|| {
                chrono::NaiveDate::parse_from_str(due_str, "%Y-%m-%d")
                    .ok()
                    .and_then(|d| d.and_hms_opt(0, 0, 0))
                    .map(|dt| dt.and_utc().fixed_offset())
            })
    }

    /// Return deterministic structural relations built from typed record connections.
    pub fn get_structural_relations(&self, limit: Option<usize>) -> Vec<StructuralRelation> {
        self.build_relation_edges(None, limit, true)
            .into_iter()
            .map(|edge| StructuralRelation {
                source_record_id: edge.source_record_id,
                target_record_id: edge.target_record_id,
                relation_type: edge.relation_type,
                weight: edge.weight,
                namespace: edge.namespace,
                evidence_record_ids: Vec::new(),
            })
            .collect()
    }

    /// Return deterministic structural relations touching a specific record.
    pub fn get_structural_relations_for_record(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Vec<StructuralRelation> {
        self.build_relation_edges(Some(record_id), limit, true)
            .into_iter()
            .map(|edge| StructuralRelation {
                source_record_id: edge.source_record_id,
                target_record_id: edge.target_record_id,
                relation_type: edge.relation_type,
                weight: edge.weight,
                namespace: edge.namespace,
                evidence_record_ids: Vec::new(),
            })
            .collect()
    }

    /// Return all explicit typed relation edges.
    pub fn get_relations(&self, limit: Option<usize>) -> Vec<RelationEdge> {
        self.build_relation_edges(None, limit, false)
    }

    /// Return explicit typed relation edges touching one record.
    pub fn get_relations_for_record(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Vec<RelationEdge> {
        self.build_relation_edges(Some(record_id), limit, false)
    }

    /// Return a bounded digest of direct typed relations for one record.
    pub fn get_relation_digest(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Option<RelationDigest> {
        let anchor = {
            let records = self.records.read();
            records.get(record_id)?.clone()
        };

        let edges = self.get_relations_for_record(record_id, limit);
        let mut relation_types = HashMap::new();
        let mut linked_record_ids = Vec::new();
        let mut structural_relations = 0;
        let mut non_structural_relations = 0;

        for edge in &edges {
            *relation_types
                .entry(edge.relation_type.clone())
                .or_insert(0) += 1;
            if edge.structural {
                structural_relations += 1;
            } else {
                non_structural_relations += 1;
            }

            let other_id = if edge.source_record_id == record_id {
                edge.target_record_id.clone()
            } else {
                edge.source_record_id.clone()
            };
            if !linked_record_ids.contains(&other_id) {
                linked_record_ids.push(other_id);
            }
        }

        Some(RelationDigest {
            anchor_record_id: anchor.id,
            namespace: anchor.namespace,
            anchor_tags: anchor.tags,
            anchor_content: anchor.content,
            relation_count: edges.len(),
            structural_relations,
            non_structural_relations,
            relation_types,
            linked_record_ids,
            edges,
        })
    }

    fn collect_entity_records(&self, entity_id: &str) -> Option<(String, HashMap<String, Record>)> {
        let records = self.records.read();
        let mut namespace = None;
        let mut scoped = HashMap::new();

        for rec in records.values() {
            if rec.metadata.get("entity_id") != Some(&entity_id.to_string()) {
                continue;
            }
            if namespace.is_none() {
                namespace = Some(rec.namespace.clone());
            }
            if namespace.as_deref() != Some(rec.namespace.as_str()) {
                continue;
            }
            scoped.insert(rec.id.clone(), rec.clone());
        }

        Some((namespace?, scoped)).filter(|(_, scoped)| !scoped.is_empty())
    }

    fn select_entity_anchor_record(&self, entity_id: &str) -> Option<Record> {
        let (_, scoped) = self.collect_entity_records(entity_id)?;
        let mut records: Vec<Record> = scoped.into_values().collect();
        records.sort_by(|a, b| {
            let a_project = a.tags.iter().any(|tag| tag == "research-project");
            let b_project = b.tags.iter().any(|tag| tag == "research-project");
            let a_profile = a.tags.iter().any(|tag| tag == identity::PROFILE_TAG);
            let b_profile = b.tags.iter().any(|tag| tag == identity::PROFILE_TAG);
            b_project
                .cmp(&a_project)
                .then_with(|| b_profile.cmp(&a_profile))
                .then_with(|| b.level.value().cmp(&a.level.value()))
                .then_with(|| a.created_at.total_cmp(&b.created_at))
                .then_with(|| a.id.cmp(&b.id))
        });
        records.into_iter().next()
    }

    fn promote_record_link_to_entity_anchors(
        &self,
        source_id: &str,
        target_id: &str,
        relation_type: &str,
        weight: f32,
    ) -> Result<usize> {
        if weight < ENTITY_RELATION_PROMOTION_MIN_WEIGHT {
            return Ok(0);
        }

        let (source_entity_id, target_entity_id) = {
            let records = self.records.read();
            let Some(source) = records.get(source_id) else {
                return Ok(0);
            };
            let Some(target) = records.get(target_id) else {
                return Ok(0);
            };
            let Some(source_entity_id) = source.metadata.get("entity_id").cloned() else {
                return Ok(0);
            };
            let Some(target_entity_id) = target.metadata.get("entity_id").cloned() else {
                return Ok(0);
            };
            if source_entity_id == target_entity_id {
                return Ok(0);
            }
            (source_entity_id, target_entity_id)
        };

        let Some(source_anchor) = self.select_entity_anchor_record(&source_entity_id) else {
            return Ok(0);
        };
        let Some(target_anchor) = self.select_entity_anchor_record(&target_entity_id) else {
            return Ok(0);
        };

        let mut changed_records: Vec<Record> = Vec::new();
        {
            let mut records = self.records.write();
            if let Some(source) = records.get_mut(&source_anchor.id) {
                if Self::upsert_structural_connection(
                    source,
                    &target_anchor.id,
                    weight,
                    relation_type,
                ) {
                    changed_records.push(source.clone());
                }
            }
            if let Some(target) = records.get_mut(&target_anchor.id) {
                if Self::upsert_structural_connection(
                    target,
                    &source_anchor.id,
                    weight,
                    relation_type,
                ) {
                    changed_records.push(target.clone());
                }
            }
        }

        for rec in &changed_records {
            self.cognitive_store.append_update(rec)?;
        }
        if !changed_records.is_empty() {
            self.runtime.recall_cache.clear();
            self.runtime.structured_recall_cache.clear();
        }

        Ok(changed_records.len())
    }

    /// Return an aggregate digest for one local entity id.
    pub fn get_entity_digest(&self, entity_id: &str) -> Option<EntityDigest> {
        let (namespace, scoped) = self.collect_entity_records(entity_id)?;
        let mut tags = HashMap::new();
        let mut levels = HashMap::new();
        let mut record_ids = Vec::new();
        let mut relation_count = 0;

        for rec in scoped.values() {
            record_ids.push(rec.id.clone());
            relation_count += rec.connection_types.len();
            for tag in &rec.tags {
                *tags.entry(tag.clone()).or_insert(0) += 1;
            }
            *levels.entry(rec.level.name().to_string()).or_insert(0) += 1;
        }

        record_ids.sort();

        Some(EntityDigest {
            entity_id: entity_id.to_string(),
            namespace,
            record_ids,
            relation_count,
            tags,
            levels,
        })
    }

    /// Return an aggregate digest inferred from a record's `entity_id`.
    pub fn get_entity_digest_for_record(&self, record_id: &str) -> Option<EntityDigest> {
        let entity_id = {
            let records = self.records.read();
            records.get(record_id)?.metadata.get("entity_id")?.clone()
        };
        self.get_entity_digest(&entity_id)
    }

    /// Return a bounded entity graph snapshot for one local entity.
    pub fn get_entity_graph_digest(
        &self,
        entity_id: &str,
        limit: Option<usize>,
    ) -> Option<EntityGraphDigest> {
        let entity = self.get_entity_digest(entity_id)?;
        let anchor = self.select_entity_anchor_record(entity_id)?;
        let edges = self.get_entity_relations(entity_id, limit);
        let mut relation_types = HashMap::new();
        let mut neighbors = Vec::new();

        for edge in &edges {
            *relation_types
                .entry(edge.relation_type.clone())
                .or_insert(0) += 1;
        }

        let mut grouped: HashMap<String, Vec<&EntityRelationEdge>> = HashMap::new();
        for edge in &edges {
            grouped
                .entry(edge.target_entity_id.clone())
                .or_default()
                .push(edge);
        }

        for (neighbor_entity_id, neighbor_edges) in grouped {
            let Some(neighbor_digest) = self.get_entity_digest(&neighbor_entity_id) else {
                continue;
            };
            let Some(neighbor_anchor) = self.select_entity_anchor_record(&neighbor_entity_id)
            else {
                continue;
            };
            let mut neighbor_relation_types = HashMap::new();
            let mut strongest_weight: f32 = 0.0;
            for edge in &neighbor_edges {
                *neighbor_relation_types
                    .entry(edge.relation_type.clone())
                    .or_insert(0) += 1;
                strongest_weight = strongest_weight.max(edge.weight);
            }
            neighbors.push(EntityGraphNeighbor {
                entity_id: neighbor_entity_id,
                anchor_record_id: neighbor_anchor.id,
                record_count: neighbor_digest.record_ids.len(),
                relation_count: neighbor_edges.len(),
                relation_types: neighbor_relation_types,
                strongest_weight,
            });
        }

        neighbors.sort_by(|a, b| {
            b.strongest_weight
                .total_cmp(&a.strongest_weight)
                .then_with(|| a.entity_id.cmp(&b.entity_id))
        });

        Some(EntityGraphDigest {
            entity,
            anchor_record_id: anchor.id,
            neighbor_count: neighbors.len(),
            relation_types,
            neighbors,
            edges,
        })
    }

    /// Return top-N neighbor `EntityGraphDigest` objects for one entity.
    ///
    /// Neighbors are taken from the entity's own `EntityGraphDigest` (already sorted by
    /// `strongest_weight` desc, then `entity_id`), optionally filtered by a relation-type
    /// prefix, then truncated to `top_n`.  For each surviving neighbor entity its own
    /// `EntityGraphDigest` is built and returned in the same order.
    ///
    /// Parameters
    /// - `entity_id`           — anchor entity
    /// - `top_n`               — max neighbors to return (default: all)
    /// - `min_weight`          — drop neighbors whose `strongest_weight` is below this
    /// - `relation_type_filter`— keep only neighbors connected by a relation whose type
    ///                           starts with this prefix (e.g. `"supports"`)
    /// - `edge_limit`          — forwarded to `get_entity_graph_digest` / `get_entity_relations`
    pub fn get_entity_graph_neighbors(
        &self,
        entity_id: &str,
        top_n: Option<usize>,
        min_weight: Option<f32>,
        relation_type_filter: Option<&str>,
        edge_limit: Option<usize>,
    ) -> Vec<EntityGraphDigest> {
        let Some(anchor_digest) = self.get_entity_graph_digest(entity_id, edge_limit) else {
            return Vec::new();
        };
        let min_w = min_weight.unwrap_or(0.0);
        let mut neighbors: Vec<&EntityGraphNeighbor> = anchor_digest
            .neighbors
            .iter()
            .filter(|n| n.strongest_weight >= min_w)
            .filter(|n| {
                relation_type_filter
                    .map(|prefix| n.relation_types.keys().any(|rt| rt.starts_with(prefix)))
                    .unwrap_or(true)
            })
            .collect();
        if let Some(k) = top_n {
            neighbors.truncate(k);
        }
        neighbors
            .into_iter()
            .filter_map(|n| self.get_entity_graph_digest(&n.entity_id, edge_limit))
            .collect()
    }

    /// Return a bounded entity graph snapshot inferred from a record's `entity_id`.
    pub fn get_entity_graph_digest_for_record(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Option<EntityGraphDigest> {
        let entity_id = {
            let records = self.records.read();
            records.get(record_id)?.metadata.get("entity_id")?.clone()
        };
        self.get_entity_graph_digest(&entity_id, limit)
    }

    /// Link two local entities by connecting their deterministic anchor records.
    pub fn link_entities(
        &self,
        source_entity_id: &str,
        target_entity_id: &str,
        relation_type: &str,
        weight: Option<f32>,
    ) -> Result<EntityRelationEdge> {
        if source_entity_id == target_entity_id {
            anyhow::bail!("Cannot link entity to itself");
        }
        let source = self
            .select_entity_anchor_record(source_entity_id)
            .ok_or_else(|| anyhow::anyhow!("Source entity {} not found", source_entity_id))?;
        let target = self
            .select_entity_anchor_record(target_entity_id)
            .ok_or_else(|| anyhow::anyhow!("Target entity {} not found", target_entity_id))?;
        let edge = self.link_records(&source.id, &target.id, relation_type, weight)?;
        Ok(EntityRelationEdge {
            source_entity_id: source_entity_id.to_string(),
            target_entity_id: target_entity_id.to_string(),
            relation_type: edge.relation_type,
            weight: edge.weight,
            namespace: edge.namespace,
            source_record_id: source.id,
            target_record_id: target.id,
        })
    }

    /// Return direct entity-to-entity edges for one local entity.
    pub fn get_entity_relations(
        &self,
        entity_id: &str,
        limit: Option<usize>,
    ) -> Vec<EntityRelationEdge> {
        let Some((namespace, scoped)) = self.collect_entity_records(entity_id) else {
            return Vec::new();
        };

        let records = self.records.read();
        let max = limit.unwrap_or(STRUCTURAL_RELATION_DEFAULT_LIMIT);
        let source_anchor_id = self
            .select_entity_anchor_record(entity_id)
            .map(|rec| rec.id)
            .unwrap_or_default();
        let mut aggregated: HashMap<(String, String), EntityRelationEdge> = HashMap::new();

        for rec in scoped.values() {
            for (other_id, relation_type) in &rec.connection_types {
                let Some(other) = records.get(other_id) else {
                    continue;
                };
                let Some(other_entity_id) = other.metadata.get("entity_id").cloned() else {
                    continue;
                };
                if other_entity_id == entity_id {
                    continue;
                }
                let target_anchor_id = self
                    .select_entity_anchor_record(&other_entity_id)
                    .map(|rec| rec.id)
                    .unwrap_or_default();
                let key = (other_entity_id.clone(), relation_type.clone());
                let candidate = EntityRelationEdge {
                    source_entity_id: entity_id.to_string(),
                    target_entity_id: other_entity_id,
                    relation_type: relation_type.clone(),
                    weight: rec.connections.get(other_id).copied().unwrap_or(0.0),
                    namespace: namespace.clone(),
                    source_record_id: rec.id.clone(),
                    target_record_id: other.id.clone(),
                };
                match aggregated.get(&key) {
                    Some(existing)
                        if existing.weight > candidate.weight
                            || (existing.weight == candidate.weight
                                && existing.source_record_id == source_anchor_id
                                && existing.target_record_id == target_anchor_id) => {}
                    _ => {
                        aggregated.insert(key, candidate);
                    }
                }
            }
        }

        let mut edges: Vec<EntityRelationEdge> = aggregated.into_values().collect();
        edges.sort_by(|a, b| {
            b.weight
                .total_cmp(&a.weight)
                .then_with(|| a.target_entity_id.cmp(&b.target_entity_id))
                .then_with(|| a.relation_type.cmp(&b.relation_type))
        });
        edges.truncate(max);
        edges
    }

    /// Create a deterministic explicit typed relation between two existing records.
    pub fn link_records(
        &self,
        source_id: &str,
        target_id: &str,
        relation_type: &str,
        weight: Option<f32>,
    ) -> Result<RelationEdge> {
        if source_id == target_id {
            anyhow::bail!("Cannot link record to itself");
        }
        if relation_type.trim().is_empty() {
            anyhow::bail!("Relation type must not be empty");
        }

        let clamped_weight = weight.unwrap_or(0.8).clamp(0.0, 1.0);
        let mut changed_records: Vec<Record> = Vec::new();
        let namespace = {
            let records = self.records.read();
            let source = records
                .get(source_id)
                .ok_or_else(|| anyhow::anyhow!("Source record {} not found", source_id))?;
            let target = records
                .get(target_id)
                .ok_or_else(|| anyhow::anyhow!("Target record {} not found", target_id))?;
            if source.namespace != target.namespace {
                anyhow::bail!(
                    "Cannot link records across namespaces: {} vs {}",
                    source.namespace,
                    target.namespace
                );
            }
            source.namespace.clone()
        };

        {
            let mut records = self.records.write();
            if let Some(source) = records.get_mut(source_id) {
                if Self::upsert_structural_connection(
                    source,
                    target_id,
                    clamped_weight,
                    relation_type,
                ) {
                    changed_records.push(source.clone());
                }
            }
            if let Some(target) = records.get_mut(target_id) {
                if Self::upsert_structural_connection(
                    target,
                    source_id,
                    clamped_weight,
                    relation_type,
                ) {
                    changed_records.push(target.clone());
                }
            }
        }

        for rec in &changed_records {
            self.cognitive_store.append_update(rec)?;
        }
        self.promote_record_link_to_entity_anchors(
            source_id,
            target_id,
            relation_type,
            clamped_weight,
        )?;
        if !changed_records.is_empty() {
            self.runtime.recall_cache.clear();
            self.runtime.structured_recall_cache.clear();
        }

        Ok(RelationEdge {
            source_record_id: source_id.to_string(),
            target_record_id: target_id.to_string(),
            relation_type: relation_type.to_string(),
            weight: clamped_weight,
            namespace,
            structural: relation::is_structural_relation_type(relation_type),
        })
    }

    /// Recall only within one direct typed-relation corridor.
    pub fn recall_relation_context(
        &self,
        record_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<(f32, Record)>> {
        let digest = self
            .get_relation_digest(record_id, limit)
            .ok_or_else(|| anyhow::anyhow!("Relation digest not found for {}", record_id))?;

        let scoped_records = {
            let records = self.records.read();
            let mut scoped = HashMap::new();
            if let Some(anchor) = records.get(&digest.anchor_record_id) {
                scoped.insert(anchor.id.clone(), anchor.clone());
            }
            for linked_id in &digest.linked_record_ids {
                if let Some(rec) = records.get(linked_id) {
                    scoped.insert(rec.id.clone(), rec.clone());
                }
            }
            scoped
        };

        let top = top_k.unwrap_or(10);
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();
        let trust_config = self.config.trust_config.read();
        let embedding_ranked = self.collect_embedding_signal(query, top);
        let ns = [digest.namespace.as_str()];

        let scored = recall::recall_pipeline(
            query,
            top,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            &self.sdr,
            &self.index,
            &self.storage,
            &ngram,
            &tag_idx,
            &aura_idx,
            &scoped_records,
            embedding_ranked,
            Some(&trust_config),
            Some(&ns),
        );

        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        {
            let mut records = self.records.write();
            let mut tracker = self.session_tracker.write();
            recall::activate_and_strengthen(&scored, &mut records, &mut tracker, session_id);
        }

        if let Some(ref log) = self.audit_log {
            let _ = log.log_retrieve(query, scored.len());
        }

        Ok(scored)
    }

    /// Recall within one entity plus its direct linked entity neighbors.
    pub fn recall_entity_graph_context(
        &self,
        entity_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<(f32, Record)>> {
        let (namespace, mut scoped_records) = self
            .collect_entity_records(entity_id)
            .ok_or_else(|| anyhow::anyhow!("Entity {} not found", entity_id))?;

        for edge in self.get_entity_relations(entity_id, limit) {
            if let Some((_, related_records)) = self.collect_entity_records(&edge.target_entity_id)
            {
                for (id, rec) in related_records {
                    scoped_records.insert(id, rec);
                }
            }
        }

        let top = top_k.unwrap_or(10);
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();
        let trust_config = self.config.trust_config.read();
        let embedding_ranked = self.collect_embedding_signal(query, top);
        let ns = [namespace.as_str()];

        let scored = recall::recall_pipeline(
            query,
            top,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            &self.sdr,
            &self.index,
            &self.storage,
            &ngram,
            &tag_idx,
            &aura_idx,
            &scoped_records,
            embedding_ranked,
            Some(&trust_config),
            Some(&ns),
        );

        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        {
            let mut records = self.records.write();
            let mut tracker = self.session_tracker.write();
            recall::activate_and_strengthen(&scored, &mut records, &mut tracker, session_id);
        }

        if let Some(ref log) = self.audit_log {
            let _ = log.log_retrieve(query, scored.len());
        }

        Ok(scored)
    }

    /// Recall only within one local entity corridor.
    pub fn recall_entity_context(
        &self,
        entity_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> Result<Vec<(f32, Record)>> {
        let (namespace, scoped_records) = self
            .collect_entity_records(entity_id)
            .ok_or_else(|| anyhow::anyhow!("Entity {} not found", entity_id))?;

        let top = top_k.unwrap_or(10);
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();
        let trust_config = self.config.trust_config.read();
        let embedding_ranked = self.collect_embedding_signal(query, top);
        let ns = [namespace.as_str()];

        let scored = recall::recall_pipeline(
            query,
            top,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            &self.sdr,
            &self.index,
            &self.storage,
            &ngram,
            &tag_idx,
            &aura_idx,
            &scoped_records,
            embedding_ranked,
            Some(&trust_config),
            Some(&ns),
        );

        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        {
            let mut records = self.records.write();
            let mut tracker = self.session_tracker.write();
            recall::activate_and_strengthen(&scored, &mut records, &mut tracker, session_id);
        }

        if let Some(ref log) = self.audit_log {
            let _ = log.log_retrieve(query, scored.len());
        }

        Ok(scored)
    }

    /// Recall only within an entity inferred from one record.
    pub fn recall_entity_context_for_record(
        &self,
        record_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> Result<Vec<(f32, Record)>> {
        let entity_id = {
            let records = self.records.read();
            records
                .get(record_id)
                .and_then(|rec| rec.metadata.get("entity_id"))
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Record {} has no entity_id", record_id))?
        };
        self.recall_entity_context(
            &entity_id,
            query,
            top_k,
            min_strength,
            expand_connections,
            session_id,
        )
    }

    fn build_family_graph_snapshot(&self, profile: &Record) -> FamilyGraphSnapshot {
        let records = self.records.read();
        let mut relation_types = HashMap::new();
        let mut members = Vec::new();

        for (other_id, relation_type) in &profile.connection_types {
            if !relation_type.starts_with("family.") {
                continue;
            }
            let Some(other) = records.get(other_id) else {
                continue;
            };
            *relation_types.entry(relation_type.clone()).or_insert(0) += 1;
            members.push(FamilyRelationMember {
                record_id: other.id.clone(),
                relation_type: relation_type.clone(),
                weight: profile.connections.get(other_id).copied().unwrap_or(0.0),
                tags: other.tags.clone(),
                content: other.content.clone(),
            });
        }

        members.sort_by(|a, b| {
            a.relation_type
                .cmp(&b.relation_type)
                .then_with(|| a.record_id.cmp(&b.record_id))
        });

        FamilyGraphSnapshot {
            namespace: profile.namespace.clone(),
            profile_record_id: profile.id.clone(),
            relation_count: members.len(),
            relation_types,
            members,
        }
    }

    /// Return a deterministic self/family graph snapshot for one namespace.
    pub fn get_family_graph(&self, namespace: Option<&str>) -> Option<FamilyGraphSnapshot> {
        let profile = {
            let records = self.records.read();
            records
                .values()
                .filter(|rec| rec.tags.iter().any(|tag| tag == identity::PROFILE_TAG))
                .find(|rec| namespace.map(|ns| rec.namespace == ns).unwrap_or(true))
                .cloned()
        }?;
        Some(self.build_family_graph_snapshot(&profile))
    }

    /// Return a deterministic self/family graph inferred from a record namespace.
    pub fn get_family_graph_for_record(&self, record_id: &str) -> Option<FamilyGraphSnapshot> {
        let namespace = {
            let records = self.records.read();
            records.get(record_id)?.namespace.clone()
        };
        self.get_family_graph(Some(&namespace))
    }

    /// Return a deterministic digest for one linked family/person record.
    pub fn get_person_digest(&self, record_id: &str) -> Option<PersonDigest> {
        let family_graph = self.get_family_graph_for_record(record_id)?;
        let member = family_graph
            .members
            .iter()
            .find(|member| member.record_id == record_id)?;
        Some(PersonDigest {
            namespace: family_graph.namespace,
            profile_record_id: family_graph.profile_record_id,
            person_record_id: member.record_id.clone(),
            relation_type: member.relation_type.clone(),
            weight: member.weight,
            person_tags: member.tags.clone(),
            person_content: member.content.clone(),
        })
    }

    /// Recall only within one deterministic self/person corridor.
    pub fn recall_person_context(
        &self,
        record_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> Result<Vec<(f32, Record)>> {
        let digest = self
            .get_person_digest(record_id)
            .ok_or_else(|| anyhow::anyhow!("Person digest not found for {}", record_id))?;

        let person_records = {
            let records = self.records.read();
            let mut scoped = HashMap::new();
            if let Some(profile) = records.get(&digest.profile_record_id) {
                scoped.insert(profile.id.clone(), profile.clone());
            }
            if let Some(person) = records.get(&digest.person_record_id) {
                scoped.insert(person.id.clone(), person.clone());
            }
            scoped
        };

        let top = top_k.unwrap_or(10);
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();
        let trust_config = self.config.trust_config.read();
        let embedding_ranked = self.collect_embedding_signal(query, top);
        let ns = [digest.namespace.as_str()];

        let scored = recall::recall_pipeline(
            query,
            top,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            &self.sdr,
            &self.index,
            &self.storage,
            &ngram,
            &tag_idx,
            &aura_idx,
            &person_records,
            embedding_ranked,
            Some(&trust_config),
            Some(&ns),
        );

        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        {
            let mut records = self.records.write();
            let mut tracker = self.session_tracker.write();
            recall::activate_and_strengthen(&scored, &mut records, &mut tracker, session_id);
        }

        if let Some(ref log) = self.audit_log {
            let _ = log.log_retrieve(query, scored.len());
        }

        Ok(scored)
    }

    /// Return a project graph snapshot for an explicit project id.
    pub fn get_project_graph(&self, project_id: &str) -> Option<ProjectGraphSnapshot> {
        let project_anchor = self.ensure_project_anchor_by_id(project_id).ok()?;
        self.build_project_graph_snapshot(&project_anchor)
    }

    /// Return a project graph snapshot inferred from a project-scoped record.
    pub fn get_project_graph_for_record(&self, record_id: &str) -> Option<ProjectGraphSnapshot> {
        let project_id = {
            let records = self.records.read();
            records.get(record_id)?.metadata.get("project_id")?.clone()
        };
        self.get_project_graph(&project_id)
    }

    /// Return deterministic status counters for one project graph.
    pub fn get_project_status(&self, project_id: &str) -> Option<ProjectStatusSnapshot> {
        let (project_anchor, project_records) = self.collect_project_records(project_id)?;
        let mut reports = 0;
        let mut scheduled_tasks = 0;
        let mut open_tasks = 0;
        let mut completed_tasks = 0;
        let mut todos = 0;
        let mut open_todos = 0;
        let mut completed_todos = 0;
        let mut notes = 0;
        let mut due_tasks = 0;
        let mut high_priority_todos = 0;

        for rec in project_records.values() {
            if rec.id == project_anchor.id {
                continue;
            }
            if rec.tags.iter().any(|tag| tag == "research-report") {
                reports += 1;
            }
            if rec.tags.iter().any(|tag| tag == "scheduled-task") {
                scheduled_tasks += 1;
                if Self::record_marked_completed(rec) {
                    completed_tasks += 1;
                } else {
                    open_tasks += 1;
                }
                if rec.metadata.contains_key("due_date") {
                    due_tasks += 1;
                }
            }
            if rec.tags.iter().any(|tag| tag == "todo-item") {
                todos += 1;
                if Self::record_marked_completed(rec) {
                    completed_todos += 1;
                } else {
                    open_todos += 1;
                }
                if rec
                    .metadata
                    .get("priority")
                    .map(|v| v.eq_ignore_ascii_case("high"))
                    .unwrap_or(false)
                {
                    high_priority_todos += 1;
                }
            }
            if rec.tags.iter().any(|tag| tag == "project-note") {
                notes += 1;
            }
        }

        Some(ProjectStatusSnapshot {
            project_id: project_anchor.metadata.get("project_id")?.clone(),
            project_record_id: project_anchor.id.clone(),
            project_topic: project_anchor
                .metadata
                .get("project_topic")
                .cloned()
                .unwrap_or_default(),
            namespace: project_anchor.namespace.clone(),
            project_status: project_anchor
                .metadata
                .get("project_status")
                .cloned()
                .unwrap_or_else(|| "unknown".into()),
            total_members: project_records.len().saturating_sub(1),
            reports,
            scheduled_tasks,
            open_tasks,
            completed_tasks,
            todos,
            open_todos,
            completed_todos,
            notes,
            due_tasks,
            high_priority_todos,
        })
    }

    /// Return project status counters inferred from any project-scoped record.
    pub fn get_project_status_for_record(&self, record_id: &str) -> Option<ProjectStatusSnapshot> {
        let project_id = {
            let records = self.records.read();
            records.get(record_id)?.metadata.get("project_id")?.clone()
        };
        self.get_project_status(&project_id)
    }

    /// Return deterministic timeline rows for one project graph.
    pub fn get_project_timeline(&self, project_id: &str) -> Option<ProjectTimelineSnapshot> {
        let (project_anchor, project_records) = self.collect_project_records(project_id)?;
        let now = chrono::Utc::now().date_naive();
        let tomorrow = (chrono::Utc::now() + chrono::Duration::days(1)).date_naive();
        let mut entries = Vec::new();
        let mut overdue_entries = 0;
        let mut upcoming_entries = 0;
        let mut open_entries = 0;

        for rec in project_records.values() {
            if rec.id == project_anchor.id {
                continue;
            }

            let kind = if rec.tags.iter().any(|tag| tag == "scheduled-task") {
                "scheduled-task"
            } else if rec.tags.iter().any(|tag| tag == "todo-item") {
                "todo-item"
            } else if rec.tags.iter().any(|tag| tag == "project-note") {
                "project-note"
            } else if rec.tags.iter().any(|tag| tag == "research-report") {
                "research-report"
            } else {
                "project-record"
            }
            .to_string();

            let status = rec.metadata.get("status").cloned().unwrap_or_else(|| {
                if Self::record_marked_completed(rec) {
                    "completed".into()
                } else {
                    "open".into()
                }
            });

            let due_date = rec.metadata.get("due_date").cloned();
            let overdue = due_date
                .as_deref()
                .and_then(Self::parse_due_date_for_timeline)
                .map(|dt| dt.date_naive() < now && !Self::record_marked_completed(rec))
                .unwrap_or(false);
            let upcoming = due_date
                .as_deref()
                .and_then(Self::parse_due_date_for_timeline)
                .map(|dt| {
                    let day = dt.date_naive();
                    day >= now && day <= tomorrow && !Self::record_marked_completed(rec)
                })
                .unwrap_or(false);

            if overdue {
                overdue_entries += 1;
            }
            if upcoming {
                upcoming_entries += 1;
            }
            if !Self::record_marked_completed(rec) {
                open_entries += 1;
            }

            entries.push(ProjectTimelineEntry {
                record_id: rec.id.clone(),
                content: rec.content.clone(),
                kind,
                status,
                created_at: rec.created_at,
                due_date,
                overdue,
                tags: rec.tags.clone(),
            });
        }

        entries.sort_by(|a, b| {
            let a_due = a
                .due_date
                .as_deref()
                .and_then(Self::parse_due_date_for_timeline)
                .map(|dt| dt.timestamp())
                .unwrap_or(i64::MAX);
            let b_due = b
                .due_date
                .as_deref()
                .and_then(Self::parse_due_date_for_timeline)
                .map(|dt| dt.timestamp())
                .unwrap_or(i64::MAX);
            a_due.cmp(&b_due).then_with(|| {
                a.created_at
                    .partial_cmp(&b.created_at)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        Some(ProjectTimelineSnapshot {
            project_id: project_anchor.metadata.get("project_id")?.clone(),
            project_record_id: project_anchor.id.clone(),
            project_topic: project_anchor
                .metadata
                .get("project_topic")
                .cloned()
                .unwrap_or_default(),
            namespace: project_anchor.namespace.clone(),
            total_entries: entries.len(),
            overdue_entries,
            upcoming_entries,
            open_entries,
            entries,
        })
    }

    /// Return timeline rows inferred from a project-scoped record.
    pub fn get_project_timeline_for_record(
        &self,
        record_id: &str,
    ) -> Option<ProjectTimelineSnapshot> {
        let project_id = {
            let records = self.records.read();
            records.get(record_id)?.metadata.get("project_id")?.clone()
        };
        self.get_project_timeline(&project_id)
    }

    /// Return a combined deterministic project digest for one project.
    pub fn get_project_digest(&self, project_id: &str) -> Option<ProjectDigest> {
        let graph = self.get_project_graph(project_id)?;
        let status = self.get_project_status(project_id)?;
        let timeline = self.get_project_timeline(project_id)?;
        Some(ProjectDigest {
            graph,
            status,
            timeline,
        })
    }

    /// Return a combined deterministic project digest inferred from a project-scoped record.
    pub fn get_project_digest_for_record(&self, record_id: &str) -> Option<ProjectDigest> {
        let project_id = {
            let records = self.records.read();
            records.get(record_id)?.metadata.get("project_id")?.clone()
        };
        self.get_project_digest(&project_id)
    }

    /// Recall only within one deterministic self/family corridor.
    pub fn recall_family_context(
        &self,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespace: Option<&str>,
    ) -> Result<Vec<(f32, Record)>> {
        let family_graph = self
            .get_family_graph(namespace)
            .ok_or_else(|| anyhow::anyhow!("Family graph not found"))?;

        let family_records = {
            let records = self.records.read();
            let mut scoped = HashMap::new();
            if let Some(profile) = records.get(&family_graph.profile_record_id) {
                scoped.insert(profile.id.clone(), profile.clone());
            }
            for member in &family_graph.members {
                if let Some(rec) = records.get(&member.record_id) {
                    scoped.insert(rec.id.clone(), rec.clone());
                }
            }
            scoped
        };

        let top = top_k.unwrap_or(10);
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();
        let trust_config = self.config.trust_config.read();
        let embedding_ranked = self.collect_embedding_signal(query, top);
        let ns = [family_graph.namespace.as_str()];

        let scored = recall::recall_pipeline(
            query,
            top,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            &self.sdr,
            &self.index,
            &self.storage,
            &ngram,
            &tag_idx,
            &aura_idx,
            &family_records,
            embedding_ranked,
            Some(&trust_config),
            Some(&ns),
        );

        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        {
            let mut records = self.records.write();
            let mut tracker = self.session_tracker.write();
            recall::activate_and_strengthen(&scored, &mut records, &mut tracker, session_id);
        }

        if let Some(ref log) = self.audit_log {
            let _ = log.log_retrieve(query, scored.len());
        }

        Ok(scored)
    }

    /// Recall only within one explicit project graph.
    pub fn recall_project_context(
        &self,
        project_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> Result<Vec<(f32, Record)>> {
        let (project_anchor, project_records) = self
            .collect_project_records(project_id)
            .ok_or_else(|| anyhow::anyhow!("Project {} not found", project_id))?;

        let top = top_k.unwrap_or(10);
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();
        let trust_config = self.config.trust_config.read();
        let embedding_ranked = self.collect_embedding_signal(query, top);
        let ns = [project_anchor.namespace.as_str()];

        let scored = recall::recall_pipeline(
            query,
            top,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            &self.sdr,
            &self.index,
            &self.storage,
            &ngram,
            &tag_idx,
            &aura_idx,
            &project_records,
            embedding_ranked,
            Some(&trust_config),
            Some(&ns),
        );

        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        {
            let mut records = self.records.write();
            let mut tracker = self.session_tracker.write();
            recall::activate_and_strengthen(&scored, &mut records, &mut tracker, session_id);
        }

        if let Some(ref log) = self.audit_log {
            let _ = log.log_retrieve(query, scored.len());
        }

        Ok(scored)
    }

    /// Return a snapshot of all current causal patterns (cloned).
    /// Optional filter by state: "stable", "candidate", "rejected".
    pub fn get_causal_patterns(
        &self,
        state_filter: Option<&str>,
    ) -> Vec<crate::causal::CausalPattern> {
        self.epistemic_runtime().get_causal_patterns(state_filter)
    }

    /// Return a snapshot of all current policy hints (cloned).
    /// Optional filter by state: "stable", "candidate", "suppressed", "rejected".
    pub fn get_policy_hints(&self, state_filter: Option<&str>) -> Vec<crate::policy::PolicyHint> {
        self.epistemic_runtime().get_policy_hints(state_filter)
    }

    /// Return suppressed policy hints, strongest first.
    pub fn get_suppressed_policy_hints(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<crate::policy::PolicyHint> {
        self.epistemic_runtime()
            .get_suppressed_policy_hints(namespace, limit)
    }

    /// Return rejected policy hints, strongest first.
    pub fn get_rejected_policy_hints(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<crate::policy::PolicyHint> {
        self.epistemic_runtime()
            .get_rejected_policy_hints(namespace, limit)
    }

    /// Return a compact lifecycle summary for the current policy layer.
    pub fn get_policy_lifecycle_summary(
        &self,
        action_limit: Option<usize>,
        domain_limit: Option<usize>,
    ) -> crate::epistemic_runtime::PolicyLifecycleSummary {
        self.epistemic_runtime()
            .get_policy_lifecycle_summary(action_limit, domain_limit)
    }

    /// Return the strongest advisory-pressure areas across policy domains.
    pub fn get_policy_pressure_report(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<crate::epistemic_runtime::PolicyPressureArea> {
        self.epistemic_runtime()
            .get_policy_pressure_report(namespace, limit)
    }

    /// Soft-deprecate a belief so it no longer acts as a confident winner.
    pub fn deprecate_belief(&self, belief_id: &str) -> Result<bool> {
        self.deprecate_belief_with_reason(belief_id, "manual_deprecation")
    }

    /// Soft-deprecate a belief and persist the correction reason.
    pub fn deprecate_belief_with_reason(&self, belief_id: &str, reason: &str) -> Result<bool> {
        let changed = {
            let mut engine = self.belief_engine.write();
            let changed = engine.deprecate_belief(belief_id);
            if changed {
                self.belief_store.save(&engine)?;
            }
            changed
        };
        if changed {
            self.runtime.recall_cache.clear();
            self.runtime.structured_recall_cache.clear();
            if let Some(ref log) = self.audit_log {
                let _ = log.log_correction("belief", belief_id, "deprecate", reason);
            }
        }
        Ok(changed)
    }

    /// Invalidate a single causal pattern while preserving its tombstone.
    pub fn invalidate_causal_pattern(&self, pattern_id: &str) -> Result<bool> {
        self.invalidate_causal_pattern_with_reason(pattern_id, "manual_invalidation")
    }

    /// Invalidate a single causal pattern while preserving its tombstone and reason.
    pub fn invalidate_causal_pattern_with_reason(
        &self,
        pattern_id: &str,
        reason: &str,
    ) -> Result<bool> {
        let changed = {
            let mut engine = self.causal_engine.write();
            let changed = engine.invalidate_pattern(pattern_id, reason);
            if changed {
                self.causal_store.save(&engine)?;
            }
            changed
        };
        if changed {
            self.runtime.recall_cache.clear();
            self.runtime.structured_recall_cache.clear();
            if let Some(ref log) = self.audit_log {
                let _ = log.log_correction("causal_pattern", pattern_id, "invalidate", reason);
            }
        }
        Ok(changed)
    }

    /// Legacy compatibility alias: causal retraction now preserves an invalidated tombstone.
    pub fn retract_causal_pattern(&self, pattern_id: &str) -> Result<bool> {
        self.invalidate_causal_pattern_with_reason(pattern_id, "manual_retraction")
    }

    /// Legacy compatibility alias: causal retraction now preserves an invalidated tombstone.
    pub fn retract_causal_pattern_with_reason(
        &self,
        pattern_id: &str,
        reason: &str,
    ) -> Result<bool> {
        self.invalidate_causal_pattern_with_reason(pattern_id, reason)
    }

    /// Retract a single policy hint from persisted runtime state.
    pub fn retract_policy_hint(&self, hint_id: &str) -> Result<bool> {
        self.retract_policy_hint_with_reason(hint_id, "manual_retraction")
    }

    /// Retract a single policy hint from persisted runtime state with a reason.
    pub fn retract_policy_hint_with_reason(&self, hint_id: &str, reason: &str) -> Result<bool> {
        let changed = {
            let mut engine = self.policy_engine.write();
            let changed = engine.retract_hint(hint_id);
            if changed {
                self.policy_store.save(&engine)?;
            }
            changed
        };
        if changed {
            self.runtime.recall_cache.clear();
            self.runtime.structured_recall_cache.clear();
            if let Some(ref log) = self.audit_log {
                let _ = log.log_correction("policy_hint", hint_id, "retract", reason);
            }
        }
        Ok(changed)
    }

    /// Return persisted correction log entries, newest last.
    pub fn get_correction_log(&self) -> Vec<CorrectionLogEntry> {
        let Some(log) = &self.audit_log else {
            return Vec::new();
        };

        log.read_all()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|entry| match entry.action {
                crate::audit::AuditAction::Correction {
                    target_kind,
                    target_id,
                    operation,
                    reason,
                } => Some(CorrectionLogEntry {
                    timestamp: entry.timestamp,
                    time_iso: entry.time_iso,
                    target_kind,
                    target_id,
                    operation,
                    reason,
                    session_id: entry.session_id,
                }),
                _ => None,
            })
            .collect()
    }

    /// Return persisted correction log entries for a specific target, newest last.
    pub fn get_correction_log_for_target(
        &self,
        target_kind: &str,
        target_id: &str,
    ) -> Vec<CorrectionLogEntry> {
        self.get_correction_log()
            .into_iter()
            .filter(|entry| entry.matches_target(target_kind, target_id))
            .collect()
    }

    // ── Surfaced Policy Output ──

    /// Return filtered, sorted, bounded advisory policy hints suitable for
    /// external consumption. Inspection-only — does not affect recall or behavior.
    ///
    /// Only surfaces Stable hints and strong Candidates (policy_strength >= 0.70,
    /// confidence >= 0.55) with complete provenance. Bounded to 10 global,
    /// 3 per domain.
    pub fn get_surfaced_policy_hints(
        &self,
        limit: Option<usize>,
    ) -> Vec<crate::policy::SurfacedPolicyHint> {
        self.epistemic_runtime().get_surfaced_policy_hints(limit)
    }

    /// Return surfaced hints for a specific namespace.
    pub fn get_surfaced_policy_hints_for_namespace(
        &self,
        namespace: &str,
        limit: Option<usize>,
    ) -> Vec<crate::policy::SurfacedPolicyHint> {
        self.epistemic_runtime()
            .get_surfaced_policy_hints_for_namespace(namespace, limit)
    }

    // ── Belief Reranking Config (Phase 4) ──

    /// Set belief reranking mode.
    ///
    /// - `Off`: no belief influence on ranking (default)
    /// - `Shadow`: compute shadow scores for logging, do not alter ranking
    /// - `Limited`: apply bounded reranking (±5% score cap, ±2 position cap)
    pub fn set_belief_rerank_mode(&self, mode: recall::BeliefRerankMode) {
        self.runtime
            .belief_rerank_mode
            .store(mode as u8, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current belief reranking mode.
    pub fn get_belief_rerank_mode(&self) -> recall::BeliefRerankMode {
        recall::BeliefRerankMode::from_u8(
            self.runtime
                .belief_rerank_mode
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Convenience: enable limited belief reranking.
    pub fn set_belief_rerank_enabled(&self, enabled: bool) {
        let mode = if enabled {
            recall::BeliefRerankMode::Limited
        } else {
            recall::BeliefRerankMode::Off
        };
        self.set_belief_rerank_mode(mode);
    }

    /// Convenience: check if belief reranking is actively influencing ranking.
    pub fn is_belief_rerank_enabled(&self) -> bool {
        self.get_belief_rerank_mode() == recall::BeliefRerankMode::Limited
    }

    /// Set the coarse key mode for belief grouping.
    /// Takes effect on the next `run_maintenance()` call.
    pub fn set_belief_coarse_key_mode(&self, mode: CoarseKeyMode) {
        let mut engine = self.belief_engine.write();
        engine.coarse_key_mode = mode;
    }

    /// Get current coarse key mode.
    pub fn get_belief_coarse_key_mode(&self) -> CoarseKeyMode {
        let engine = self.belief_engine.read();
        engine.coarse_key_mode
    }

    /// Override the SDR subclustering similarity threshold.
    /// Pass `None` to restore default (0.15).
    pub fn set_belief_similarity_threshold(&self, threshold: Option<f32>) {
        let mut engine = self.belief_engine.write();
        engine.claim_similarity_override = threshold;
    }

    /// Set the concept seed selection mode (Standard or Relaxed).
    /// Takes effect on the next `run_maintenance()` call.
    pub fn set_concept_seed_mode(&self, mode: ConceptSeedMode) {
        let mut engine = self.concept_engine.write();
        engine.seed_mode = mode;
    }

    /// Get current concept seed mode.
    pub fn get_concept_seed_mode(&self) -> ConceptSeedMode {
        let engine = self.concept_engine.read();
        engine.seed_mode
    }

    /// Set the concept similarity mode (SdrTanimoto or CanonicalFeature).
    /// Takes effect on the next `run_maintenance()` call.
    pub fn set_concept_similarity_mode(&self, mode: ConceptSimilarityMode) {
        let mut engine = self.concept_engine.write();
        engine.similarity_mode = mode;
    }

    /// Get current concept similarity mode.
    pub fn get_concept_similarity_mode(&self) -> ConceptSimilarityMode {
        let engine = self.concept_engine.read();
        engine.similarity_mode
    }

    /// Set the concept partition mode (Standard or NamespaceOnly).
    /// Takes effect on the next `run_maintenance()` call.
    pub fn set_concept_partition_mode(&self, mode: ConceptPartitionMode) {
        let mut engine = self.concept_engine.write();
        engine.partition_mode = mode;
    }

    /// Get current concept partition mode.
    pub fn get_concept_partition_mode(&self) -> ConceptPartitionMode {
        let engine = self.concept_engine.read();
        engine.partition_mode
    }

    /// Set the concept union relaxation mode.
    /// Takes effect on the next `run_maintenance()` call.
    pub fn set_concept_union_mode(&self, mode: crate::concept::ConceptUnionMode) {
        let mut engine = self.concept_engine.write();
        engine.union_mode = mode;
    }

    /// Get current concept union relaxation mode.
    pub fn get_concept_union_mode(&self) -> crate::concept::ConceptUnionMode {
        let engine = self.concept_engine.read();
        engine.union_mode
    }

    /// Set the concept rollout mode for surfaced output and bounded reranking.
    ///
    /// - `Off`: no surfaced concept output and no concept reranking (default)
    /// - `Inspect`: bounded inspection-only surfaced concepts
    /// - `Limited`: surfaced concepts plus bounded concept-aware reranking
    pub fn set_concept_surface_mode(&self, mode: ConceptSurfaceMode) {
        self.runtime
            .concept_surface_mode
            .store(mode as u8, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current concept rollout mode for surfaced output and bounded reranking.
    pub fn get_concept_surface_mode(&self) -> ConceptSurfaceMode {
        self.runtime.concept_surface_mode()
    }

    /// Set the temporal causal edge budgeting mode.
    /// Takes effect on the next `run_maintenance()` call.
    pub fn set_causal_temporal_budget_mode(&self, mode: TemporalEdgeBudgetMode) {
        let mut engine = self.causal_engine.write();
        engine.temporal_budget_mode = mode;
    }

    /// Get current temporal causal edge budgeting mode.
    pub fn get_causal_temporal_budget_mode(&self) -> TemporalEdgeBudgetMode {
        let engine = self.causal_engine.read();
        engine.temporal_budget_mode
    }

    /// Set the causal evidence gating mode.
    /// Takes effect on the next `run_maintenance()` call.
    pub fn set_causal_evidence_mode(&self, mode: CausalEvidenceMode) {
        let mut engine = self.causal_engine.write();
        engine.evidence_mode = mode;
    }

    /// Get current causal evidence gating mode.
    pub fn get_causal_evidence_mode(&self) -> CausalEvidenceMode {
        let engine = self.causal_engine.read();
        engine.evidence_mode
    }

    /// Set policy-hint recall reranking mode.
    ///
    /// - `Off`: no policy influence on recall ranking (default)
    /// - `Limited`: bounded reranking (±2% score cap, ±2 positional shift)
    pub fn set_policy_rerank_mode(&self, mode: PolicyRerankMode) {
        self.runtime
            .policy_rerank_mode
            .store(mode as u8, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current policy recall reranking mode.
    pub fn get_policy_rerank_mode(&self) -> PolicyRerankMode {
        PolicyRerankMode::from_u8(
            self.runtime
                .policy_rerank_mode
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Enable all four cognitive recall reranking signals simultaneously.
    ///
    /// Sets every phase to `Limited` mode:
    /// - Phase 4a: Belief reranking (±5% score cap, ±2 positional shift)
    /// - Phase 4b: Concept grouping surface (±4% cap, ±2 shift)
    /// - Phase 4c: Causal pattern reranking (±3% cap, ±2 shift)
    /// - Phase 4d: Policy hint reranking (±2% cap, ±2 shift)
    ///
    /// Equivalent to calling `set_*_mode(Limited)` on all four phases.
    /// Use `disable_full_cognitive_stack()` to revert to the Off baseline.
    pub fn enable_full_cognitive_stack(&self) {
        use crate::causal::CausalRerankMode;
        use crate::concept::ConceptSurfaceMode;
        use crate::recall::BeliefRerankMode;
        self.set_belief_rerank_mode(BeliefRerankMode::Limited);
        self.set_concept_surface_mode(ConceptSurfaceMode::Limited);
        self.set_causal_rerank_mode(CausalRerankMode::Limited);
        self.set_policy_rerank_mode(PolicyRerankMode::Limited);
        // ExplicitTrusted: user-declared causal links are authoritative.
        // This is the correct default for programmatic use — the strict mode
        // was designed for long-running corpora where patterns emerge naturally
        // over many maintenance cycles. ExplicitTrusted is better for all cases
        // where users explicitly encode causality via link_records().
        self.set_causal_evidence_mode(CausalEvidenceMode::ExplicitTrusted);
    }

    /// Disable all four cognitive recall reranking signals simultaneously.
    ///
    /// Resets every phase to `Off` mode (the default). Raw RRF ranking is used
    /// with no cognitive shaping. Counterpart to `enable_full_cognitive_stack()`.
    pub fn disable_full_cognitive_stack(&self) {
        use crate::causal::CausalRerankMode;
        use crate::concept::ConceptSurfaceMode;
        use crate::recall::BeliefRerankMode;
        self.set_belief_rerank_mode(BeliefRerankMode::Off);
        self.set_concept_surface_mode(ConceptSurfaceMode::Off);
        self.set_causal_rerank_mode(CausalRerankMode::Off);
        self.set_policy_rerank_mode(PolicyRerankMode::Off);
    }

    /// Set causal-pattern recall reranking mode.
    ///
    /// - `Off`: no causal influence on recall ranking (default)
    /// - `Limited`: bounded reranking (±3% score cap, ±2 positional shift)
    pub fn set_causal_rerank_mode(&self, mode: CausalRerankMode) {
        self.runtime
            .causal_rerank_mode
            .store(mode as u8, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current causal recall reranking mode.
    pub fn get_causal_rerank_mode(&self) -> CausalRerankMode {
        CausalRerankMode::from_u8(
            self.runtime
                .causal_rerank_mode
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    // ── SDK Wrapper: Research Orchestrator ──

    /// Start a new research project.
    pub fn start_research(&self, topic: &str, depth: Option<&str>) -> ResearchProject {
        self.research_engine.start_research(topic, depth)
    }

    /// Add a research finding.
    pub fn add_research_finding(
        &self,
        project_id: &str,
        query: &str,
        result: &str,
        url: Option<&str>,
    ) -> Result<()> {
        self.research_engine
            .add_finding(project_id, query, result, url)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Complete a research project and store as a record.
    pub fn complete_research(&self, project_id: &str, synthesis: Option<String>) -> Result<Record> {
        let project = self
            .research_engine
            .complete_research(project_id, synthesis)
            .map_err(|e| anyhow::anyhow!(e))?;
        let _project_anchor = self.ensure_research_project_anchor(&project)?;

        // Build content from project
        let content = if let Some(ref syn) = project.synthesis {
            format!("Research: {}\n\n{}", project.topic, syn)
        } else {
            let findings: Vec<String> = project
                .findings
                .iter()
                .map(|f| format!("- {}: {}", f.query, f.result))
                .collect();
            format!(
                "Research: {}\n\nFindings:\n{}",
                project.topic,
                findings.join("\n")
            )
        };

        let report_metadata = HashMap::from([
            ("project_id".to_string(), project.id.clone()),
            ("entity_id".to_string(), format!("project:{}", project.id)),
            ("project_topic".to_string(), project.topic.clone()),
            ("project_depth".to_string(), project.depth.clone()),
            (
                "project_status".to_string(),
                match project.status {
                    crate::research::ResearchStatus::Active => "active",
                    crate::research::ResearchStatus::Completed => "completed",
                    crate::research::ResearchStatus::Cancelled => "cancelled",
                }
                .to_string(),
            ),
            (
                "finding_count".to_string(),
                project.findings.len().to_string(),
            ),
        ]);

        // Store as a record
        let rec = self.store(
            &content,
            Some(Level::Domain),
            Some(vec!["research-report".into()]),
            None,
            None,
            Some("retrieved"),
            Some(report_metadata),
            Some(false), // Don't dedup research
            None,
            None,
            Some("fact"),
        )?;

        Ok(rec)
    }

    /// Store a scheduled task that is explicitly scoped to a project.
    pub fn store_project_task(
        &self,
        project_id: &str,
        content: &str,
        due_date: Option<&str>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Record> {
        let project_anchor = self.ensure_project_anchor_by_id(project_id)?;
        let mut task_metadata = metadata.unwrap_or_default();
        task_metadata.insert("project_id".into(), project_id.to_string());
        task_metadata.insert("entity_id".into(), format!("project:{}", project_id));
        task_metadata.insert("project_anchor_id".into(), project_anchor.id.clone());
        if let Some(due) = due_date {
            task_metadata.insert("due_date".into(), due.to_string());
        }

        self.store(
            content,
            Some(Level::Working),
            Some(vec!["scheduled-task".into()]),
            Some(false),
            None,
            Some("recorded"),
            Some(task_metadata),
            Some(false),
            None,
            Some(&project_anchor.namespace),
            Some("decision"),
        )
    }

    /// Store a todo item that is explicitly scoped to a project.
    pub fn store_project_todo(
        &self,
        project_id: &str,
        content: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Record> {
        let project_anchor = self.ensure_project_anchor_by_id(project_id)?;
        let mut todo_metadata = metadata.unwrap_or_default();
        todo_metadata.insert("project_id".into(), project_id.to_string());
        todo_metadata.insert("entity_id".into(), format!("project:{}", project_id));
        todo_metadata.insert("project_anchor_id".into(), project_anchor.id.clone());

        self.store(
            content,
            Some(Level::Working),
            Some(vec!["todo-item".into()]),
            Some(false),
            None,
            Some("recorded"),
            Some(todo_metadata),
            Some(false),
            None,
            Some(&project_anchor.namespace),
            Some("decision"),
        )
    }

    /// Store a project-scoped note/fact with explicit project linkage.
    pub fn store_project_note(
        &self,
        project_id: &str,
        content: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Record> {
        let project_anchor = self.ensure_project_anchor_by_id(project_id)?;
        let mut note_metadata = metadata.unwrap_or_default();
        note_metadata.insert("project_id".into(), project_id.to_string());
        note_metadata.insert("entity_id".into(), format!("project:{}", project_id));
        note_metadata.insert("project_anchor_id".into(), project_anchor.id.clone());

        self.store(
            content,
            Some(Level::Domain),
            Some(vec!["project-note".into()]),
            Some(false),
            None,
            Some("recorded"),
            Some(note_metadata),
            Some(false),
            None,
            Some(&project_anchor.namespace),
            Some("fact"),
        )
    }

    /// Store a family/person record with an explicit deterministic relation type.
    pub fn store_family_person(
        &self,
        relation_type: &str,
        content: &str,
        metadata: Option<HashMap<String, String>>,
        namespace: Option<&str>,
    ) -> Result<Record> {
        if !relation::is_family_relation_type(relation_type) {
            anyhow::bail!("Invalid family relation type: {}", relation_type);
        }

        let profile = self.find_profile_record(namespace);
        let mut person_metadata = metadata.unwrap_or_default();
        person_metadata.insert("family_relation".into(), relation_type.to_string());
        if let Some(entity_id) =
            Self::derive_family_entity_id(relation_type, &person_metadata, content)
        {
            person_metadata.insert("entity_id".into(), entity_id);
        }
        if let Some(profile) = &profile {
            person_metadata.insert("profile_record_id".into(), profile.id.clone());
        }

        let target_namespace = profile
            .as_ref()
            .map(|profile| profile.namespace.as_str())
            .or(namespace);

        self.store(
            content,
            Some(Level::Identity),
            Some(vec!["family".into()]),
            Some(false),
            None,
            Some("recorded"),
            Some(person_metadata),
            Some(false),
            None,
            target_namespace,
            Some("fact"),
        )
    }

    /// Get active research projects.
    pub fn active_research(&self) -> Vec<ResearchProject> {
        self.research_engine.active_projects()
    }

    // ── SDK Wrapper: Identity ──

    /// Store user profile at IDENTITY level.
    pub fn store_user_profile(&self, fields: HashMap<String, String>) -> Result<Record> {
        let content = identity::format_profile_content(&fields);

        // Check for existing profile
        let existing = self.search(
            None,
            Some(Level::Identity),
            Some(vec![identity::PROFILE_TAG.into()]),
            Some(1),
            None,
            None,
            None,
            None,
        );

        if let Some(existing_rec) = existing.first() {
            // Merge new fields into existing profile metadata
            let mut merged = existing_rec.metadata.clone();
            for (k, v) in fields {
                merged.insert(k, v);
            }
            let content = identity::format_profile_content(&merged);
            self.update(
                &existing_rec.id,
                Some(&content),
                None,
                None,
                None,
                Some(merged),
                None,
            )?;
            return self
                .get(&existing_rec.id)
                .ok_or_else(|| anyhow::anyhow!("Profile record disappeared"));
        }

        // Create new profile
        self.store(
            &content,
            Some(Level::Identity),
            Some(vec![identity::PROFILE_TAG.into()]),
            Some(true), // Pin
            None,
            None,
            Some(fields),
            Some(false), // Don't dedup
            None,
            None,
            None,
        )
    }

    /// Get user profile (returns metadata fields or None).
    pub fn get_user_profile(&self) -> Option<HashMap<String, String>> {
        let results = self.search(
            None,
            Some(Level::Identity),
            Some(vec![identity::PROFILE_TAG.into()]),
            Some(1),
            None,
            None,
            None,
            None,
        );
        results.first().map(|r| r.metadata.clone())
    }

    /// Set agent persona.
    pub fn set_persona(&self, persona: AgentPersona) -> Result<Record> {
        let content = identity::persona_to_instruction(&persona);
        let mut metadata = HashMap::new();
        if let Ok(json) = serde_json::to_string(&persona) {
            metadata.insert("persona_json".into(), json);
        }

        // Check for existing persona
        let existing = self.search(
            None,
            Some(Level::Identity),
            Some(vec![identity::PERSONA_TAG.into()]),
            Some(1),
            None,
            None,
            None,
            None,
        );

        if let Some(existing_rec) = existing.first() {
            self.update(
                &existing_rec.id,
                Some(&content),
                None,
                None,
                None,
                Some(metadata),
                None,
            )?;
            return self
                .get(&existing_rec.id)
                .ok_or_else(|| anyhow::anyhow!("Persona record disappeared"));
        }

        self.store(
            &content,
            Some(Level::Identity),
            Some(vec![identity::PERSONA_TAG.into()]),
            Some(true),
            None,
            None,
            Some(metadata),
            Some(false),
            None,
            None,
            Some("preference"),
        )
    }

    /// Get agent persona.
    pub fn get_persona(&self) -> Option<AgentPersona> {
        let results = self.search(
            None,
            Some(Level::Identity),
            Some(vec![identity::PERSONA_TAG.into()]),
            Some(1),
            None,
            None,
            None,
            None,
        );
        results.first().and_then(|r| {
            r.metadata
                .get("persona_json")
                .and_then(|json| serde_json::from_str(json).ok())
        })
    }

    // ── Multimodal Memory Stubs ──

    /// Store an image reference with its description.
    ///
    /// This stores the textual description as a standard memory record with
    /// `content_type=image` and the source path in metadata. When an embedding
    /// function is set, the description is embedded for semantic search.
    ///
    /// Actual image processing (CLIP, OCR, etc.) is left to the caller —
    /// pass the results as `description`.
    pub fn store_image(
        &self,
        path: &str,
        description: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        namespace: Option<&str>,
    ) -> Result<Record> {
        let mut metadata = HashMap::new();
        metadata.insert("source_path".into(), path.to_string());
        metadata.insert("media_type".into(), "image".into());

        let mut all_tags = tags.unwrap_or_default();
        if !all_tags.iter().any(|t| t == "image") {
            all_tags.push("image".into());
        }

        self.store_with_channel(
            description,
            level,
            Some(all_tags),
            None,
            Some("image"),
            Some("recorded"),
            Some(metadata),
            None,
            None,
            None,
            None,
            namespace,
            None,
        )
    }

    /// Store an audio transcript with provenance metadata.
    ///
    /// The transcript text is stored as a standard memory record with
    /// `content_type=audio_transcript` and the source path in metadata.
    pub fn store_audio_transcript(
        &self,
        transcript: &str,
        source_path: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        namespace: Option<&str>,
    ) -> Result<Record> {
        let mut metadata = HashMap::new();
        metadata.insert("source_path".into(), source_path.to_string());
        metadata.insert("media_type".into(), "audio".into());

        let mut all_tags = tags.unwrap_or_default();
        if !all_tags.iter().any(|t| t == "audio") {
            all_tags.push("audio".into());
        }

        self.store_with_channel(
            transcript,
            level,
            Some(all_tags),
            None,
            Some("audio_transcript"),
            Some("recorded"),
            Some(metadata),
            None,
            None,
            None,
            None,
            namespace,
            None,
        )
    }

    // ── SDK Wrapper: Circuit Breaker ──

    /// Record a tool failure.
    pub fn record_tool_failure(&self, tool_name: &str) {
        self.config.circuit_breaker.record_failure(tool_name);
    }

    /// Record a tool success.
    pub fn record_tool_success(&self, tool_name: &str) {
        self.config.circuit_breaker.record_success(tool_name);
    }

    /// Check if a tool is available (circuit closed).
    pub fn is_tool_available(&self, tool_name: &str) -> bool {
        self.config.circuit_breaker.is_available(tool_name)
    }

    /// Get health report for all tracked tools.
    pub fn tool_health(&self) -> HashMap<String, String> {
        self.config.circuit_breaker.health_report()
    }

    /// Configure circuit breaker.
    pub fn configure_circuit_breaker(&self, config: CircuitBreakerConfig) {
        // Circuit breaker doesn't support reconfiguration at runtime
        // (it's created once). This is a no-op for now.
        // Users should configure before opening Aura.
        let _ = config;
    }

    // ── Persistence ──

    /// Close and flush everything. Runs final maintenance cycle.
    pub fn close(&self) -> Result<()> {
        // Stop background if running
        self.stop_background();

        self.flush()?;
        let _ = self.index.save();
        self.cognitive_store.close()?;
        self.storage.close()?;
        Ok(())
    }

    /// Flush pending writes.
    pub fn flush(&self) -> Result<()> {
        self.cognitive_store.flush()?;
        self.storage.flush()?;
        Ok(())
    }

    // ── Phase 6: Adaptive Recall (Feedback) ──

    /// Provide feedback on a recalled record.
    ///
    /// Positive feedback boosts the record's strength and lowers its decay rate
    /// (via activation). Negative feedback weakens the record.
    /// The feedback is tracked in metadata for analytics.
    #[instrument(skip(self))]
    pub fn feedback(&self, record_id: &str, useful: bool) -> Result<bool> {
        let mut records = self.records.write();
        let rec = match records.get_mut(record_id) {
            Some(r) => r,
            None => return Ok(false),
        };

        // Track feedback counts in metadata
        let pos_key = "feedback_positive";
        let neg_key = "feedback_negative";
        let last_key = "feedback_last";

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        rec.metadata.insert(last_key.into(), format!("{:.3}", now));

        if useful {
            let prev: u32 = rec
                .metadata
                .get(pos_key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            rec.metadata.insert(pos_key.into(), (prev + 1).to_string());
            // Positive: boost strength (like an activation, but weaker)
            rec.strength = (rec.strength + 0.1).min(1.0);
        } else {
            let prev: u32 = rec
                .metadata
                .get(neg_key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            rec.metadata.insert(neg_key.into(), (prev + 1).to_string());
            // Negative: weaken strength
            rec.strength = (rec.strength - 0.15).max(0.0);
        }

        // Persist change
        self.cognitive_store.append_update(rec)?;
        self.runtime.clear_recall_caches();

        Ok(true)
    }

    /// Get feedback stats for a record.
    ///
    /// Returns (positive_count, negative_count, net_score).
    pub fn feedback_stats(&self, record_id: &str) -> Option<(u32, u32, i32)> {
        let records = self.records.read();
        let rec = records.get(record_id)?;

        let pos: u32 = rec
            .metadata
            .get("feedback_positive")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let neg: u32 = rec
            .metadata
            .get("feedback_negative")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        Some((pos, neg, pos as i32 - neg as i32))
    }

    // ── Phase 6: Semantic Versioning (Supersede) ──

    /// Supersede an old record with new content.
    ///
    /// The old record is marked with `superseded_by` in metadata and its
    /// strength is halved. A new record is created with a causal link to
    /// the old one. Recall prefers the new version automatically because
    /// the old record's weakened strength pushes it down in rankings.
    #[instrument(skip(self, new_content))]
    pub fn supersede(
        &self,
        old_id: &str,
        new_content: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        namespace: Option<&str>,
    ) -> Result<Record> {
        // Validate old record exists
        {
            let mut records = self.records.write();
            let old_rec = records
                .get_mut(old_id)
                .ok_or_else(|| anyhow::anyhow!("Record '{}' not found", old_id))?;

            // Mark as superseded
            old_rec
                .metadata
                .insert("superseded_by".into(), "pending".into());
            old_rec.strength *= 0.5; // Halve strength — still findable but de-ranked
            self.cognitive_store.append_update(old_rec)?;
        }

        // Determine level and tags from old record if not provided
        let (effective_level, effective_tags, effective_ns) = {
            let records = self.records.read();
            let old_rec = records.get(old_id).unwrap();
            let l = level.unwrap_or(old_rec.level);
            let t = tags.unwrap_or_else(|| old_rec.tags.clone());
            let n = namespace.unwrap_or(&old_rec.namespace).to_string();
            (l, t, n)
        };

        // Store new record with causal link to old
        let new_rec = self.store_with_channel(
            new_content,
            Some(effective_level),
            Some(effective_tags),
            None,
            None,
            None,
            None,
            Some(false), // Don't deduplicate against old version
            Some(old_id),
            None,
            None,
            Some(&effective_ns),
            None,
        )?;

        // Update old record's superseded_by with actual new ID
        {
            let mut records = self.records.write();
            if let Some(old_rec) = records.get_mut(old_id) {
                old_rec
                    .metadata
                    .insert("superseded_by".into(), new_rec.id.clone());
                let _ = self.cognitive_store.append_update(old_rec);
            }
        }

        self.runtime.clear_recall_caches();
        Ok(new_rec)
    }

    /// Check if a record has been superseded.
    ///
    /// Returns `Some(new_record_id)` if superseded, `None` otherwise.
    pub fn superseded_by(&self, record_id: &str) -> Option<String> {
        let records = self.records.read();
        records
            .get(record_id)
            .and_then(|r| r.metadata.get("superseded_by"))
            .filter(|v| !v.is_empty() && *v != "pending")
            .cloned()
    }

    /// Get the full version chain for a record.
    ///
    /// Follows `superseded_by` links forward and `caused_by_id` links backward,
    /// returning all versions from oldest to newest.
    pub fn version_chain(&self, record_id: &str) -> Vec<Record> {
        let records = self.records.read();

        // Walk backward to find the oldest version
        let mut oldest_id = record_id.to_string();
        let mut visited = HashSet::new();
        visited.insert(oldest_id.clone());

        loop {
            if let Some(rec) = records.get(&oldest_id) {
                if let Some(ref parent) = rec.caused_by_id {
                    // Only follow if parent has superseded_by pointing forward in chain
                    if let Some(parent_rec) = records.get(parent.as_str()) {
                        if parent_rec.metadata.contains_key("superseded_by")
                            && !visited.contains(parent)
                        {
                            visited.insert(parent.clone());
                            oldest_id = parent.clone();
                            continue;
                        }
                    }
                }
            }
            break;
        }

        // Walk forward collecting all versions
        let mut chain = Vec::new();
        let mut current_id = oldest_id;
        let mut visited_fwd = HashSet::new();

        loop {
            if visited_fwd.contains(&current_id) {
                break;
            }
            visited_fwd.insert(current_id.clone());

            if let Some(rec) = records.get(&current_id) {
                chain.push(rec.clone());
                if let Some(next_id) = rec.metadata.get("superseded_by") {
                    if !next_id.is_empty() && next_id != "pending" {
                        current_id = next_id.clone();
                        continue;
                    }
                }
            }
            break;
        }

        chain
    }

    // ── Phase 6: Memory Snapshots & Rollback ──

    /// Create a named snapshot of the current memory state.
    ///
    /// The snapshot is a JSON export stored in the brain directory as
    /// `<brain_path>_snapshot_<label>.json`.
    #[instrument(skip(self))]
    pub fn snapshot(&self, label: &str) -> Result<String> {
        if label.is_empty() || label.len() > 64 {
            return Err(anyhow::anyhow!("Label must be 1-64 characters"));
        }
        // Only allow safe chars in label (alphanumeric, dash, underscore)
        if !label
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(anyhow::anyhow!(
                "Label must contain only alphanumeric, dash, or underscore characters"
            ));
        }

        // Flush pending writes
        self.flush()?;

        let records = self.records.read();
        let recs: Vec<&Record> = records.values().collect();
        let json = serde_json::to_string(&recs)?;

        let snap_path = self.snapshot_path(label);
        std::fs::write(&snap_path, json)?;

        Ok(snap_path.to_string_lossy().to_string())
    }

    /// Rollback memory to a previously saved snapshot.
    ///
    /// This replaces all current records with those from the snapshot.
    /// The current state is NOT saved before rollback — call `snapshot()`
    /// first if you want to preserve it.
    #[instrument(skip(self))]
    pub fn rollback(&self, label: &str) -> Result<usize> {
        let snap_path = self.snapshot_path(label);
        if !snap_path.exists() {
            return Err(anyhow::anyhow!("Snapshot '{}' not found", label));
        }

        let json = std::fs::read_to_string(&snap_path)?;
        let imported: Vec<Record> = serde_json::from_str(&json)?;
        let count = imported.len();

        // Replace all records
        let mut records = self.records.write();
        let mut ngram = self.ngram_index.write();
        let mut tag_idx = self.tag_index.write();
        let mut aura_idx = self.aura_index.write();

        // Clear existing indices
        records.clear();
        *ngram = NGramIndex::new(None, None);
        tag_idx.clear();
        aura_idx.clear();

        // Re-import
        for rec in imported {
            ngram.add(&rec.id, &rec.content);
            for tag in &rec.tags {
                tag_idx
                    .entry(tag.clone())
                    .or_default()
                    .insert(rec.id.clone());
            }
            if let Some(ref aid) = rec.aura_id {
                aura_idx.insert(aid.clone(), rec.id.clone());
            }
            records.insert(rec.id.clone(), rec);
        }

        self.runtime.clear_recall_caches();
        Ok(count)
    }

    /// Compare two snapshots, returning added, removed, and modified record IDs.
    pub fn diff(&self, label_a: &str, label_b: &str) -> Result<HashMap<String, Vec<String>>> {
        let load_snap = |label: &str| -> Result<HashMap<String, Record>> {
            let path = self.snapshot_path(label);
            if !path.exists() {
                return Err(anyhow::anyhow!("Snapshot '{}' not found", label));
            }
            let json = std::fs::read_to_string(&path)?;
            let recs: Vec<Record> = serde_json::from_str(&json)?;
            Ok(recs.into_iter().map(|r| (r.id.clone(), r)).collect())
        };

        let snap_a = load_snap(label_a)?;
        let snap_b = load_snap(label_b)?;

        let keys_a: HashSet<&String> = snap_a.keys().collect();
        let keys_b: HashSet<&String> = snap_b.keys().collect();

        let added: Vec<String> = keys_b.difference(&keys_a).map(|s| (*s).clone()).collect();
        let removed: Vec<String> = keys_a.difference(&keys_b).map(|s| (*s).clone()).collect();
        let modified: Vec<String> = keys_a
            .intersection(&keys_b)
            .filter(|id| {
                let a = &snap_a[**id];
                let b = &snap_b[**id];
                a.content != b.content || a.strength != b.strength || a.level != b.level
            })
            .map(|s| (*s).clone())
            .collect();

        let mut result = HashMap::new();
        result.insert("added".into(), added);
        result.insert("removed".into(), removed);
        result.insert("modified".into(), modified);
        Ok(result)
    }

    /// List available snapshot labels.
    pub fn list_snapshots(&self) -> Vec<String> {
        let dir = self.config.path.parent().unwrap_or(Path::new("."));
        let prefix = format!(
            "{}_snapshot_",
            self.config
                .path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        );

        let mut labels = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&prefix) && name.ends_with(".json") {
                    let label = name
                        .strip_prefix(&prefix)
                        .and_then(|s| s.strip_suffix(".json"))
                        .map(|s| s.to_string());
                    if let Some(l) = label {
                        labels.push(l);
                    }
                }
            }
        }
        labels.sort();
        labels
    }

    /// Helper: build snapshot file path.
    fn snapshot_path(&self, label: &str) -> PathBuf {
        let stem = self
            .config
            .path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let dir = self.config.path.parent().unwrap_or(Path::new("."));
        dir.join(format!("{}_snapshot_{}.json", stem, label))
    }

    // ── Phase 6: Agent-to-Agent Memory Sharing Protocol ──

    /// Export a portable memory fragment based on a query.
    ///
    /// Returns a JSON string containing matching records with provenance
    /// metadata stamped for sharing. The recipient can import this via
    /// `import_context()`.
    #[instrument(skip(self))]
    pub fn export_context(
        &self,
        query: &str,
        top_k: Option<usize>,
        namespaces: Option<&[&str]>,
    ) -> Result<String> {
        let results = self.recall_structured(query, top_k, None, None, None, namespaces)?;

        // Build portable fragment with provenance
        let fragment: Vec<serde_json::Value> = results
            .iter()
            .map(|(score, rec)| {
                let mut meta = rec.metadata.clone();
                meta.insert("shared_score".into(), format!("{:.4}", score));
                meta.insert(
                    "shared_from".into(),
                    self.config.path.to_string_lossy().to_string(),
                );

                serde_json::json!({
                    "id": rec.id,
                    "content": rec.content,
                    "level": rec.level.name(),
                    "strength": rec.strength,
                    "tags": rec.tags,
                    "created_at": rec.created_at,
                    "source_type": rec.source_type,
                    "content_type": rec.content_type,
                    "metadata": meta,
                    "namespace": rec.namespace,
                })
            })
            .collect();

        let envelope = serde_json::json!({
            "version": "1.0",
            "format": "aura_context",
            "query": query,
            "record_count": fragment.len(),
            "records": fragment,
        });

        Ok(serde_json::to_string_pretty(&envelope)?)
    }

    /// Import a portable memory fragment from another agent.
    ///
    /// Records are imported with `source_type=retrieved` and tagged with
    /// `shared` to distinguish them from locally created memories.
    #[instrument(skip(self, fragment_json))]
    /// Strength is reduced to 0.5x to require local validation.
    pub fn import_context(&self, fragment_json: &str) -> Result<usize> {
        let envelope: serde_json::Value = serde_json::from_str(fragment_json)?;

        let format = envelope
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if format != "aura_context" {
            return Err(anyhow::anyhow!(
                "Unknown format: '{}'. Expected 'aura_context'",
                format
            ));
        }

        let records_val = envelope
            .get("records")
            .ok_or_else(|| anyhow::anyhow!("Missing 'records' field"))?;
        let records_arr = records_val
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("'records' must be an array"))?;

        let mut imported = 0;
        for item in records_arr {
            let content = item
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("Record missing 'content'"))?;

            let level_str = item
                .get("level")
                .and_then(|v| v.as_str())
                .unwrap_or("Working");
            let level = match level_str.to_uppercase().as_str() {
                "WORKING" => Level::Working,
                "DECISIONS" => Level::Decisions,
                "DOMAIN" => Level::Domain,
                "IDENTITY" => Level::Identity,
                _ => Level::Working,
            };

            let mut tags: Vec<String> = item
                .get("tags")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            if !tags.contains(&"shared".to_string()) {
                tags.push("shared".into());
            }

            let mut metadata: HashMap<String, String> = item
                .get("metadata")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            metadata.insert("trust_external".into(), "true".into());

            // Store with reduced strength (source_type=retrieved for external data)
            let rec = self.store_with_channel(
                content,
                Some(level),
                Some(tags),
                None,
                None,
                Some("retrieved"),
                Some(metadata),
                Some(true), // Deduplicate against existing memories
                None,
                None,
                Some(false), // Don't auto-promote external memories
                None,
                None,
            )?;

            // Reduce strength for imported memories (needs local validation)
            {
                let mut records = self.records.write();
                if let Some(r) = records.get_mut(&rec.id) {
                    r.strength *= 0.5;
                    let _ = self.cognitive_store.append_update(r);
                }
            }

            imported += 1;
        }

        self.runtime.clear_recall_caches();
        Ok(imported)
    }

    /// Export all records as JSON.
    pub fn export_json(&self) -> Result<String> {
        let records = self.records.read();
        let recs: Vec<&Record> = records.values().collect();
        Ok(serde_json::to_string_pretty(&recs)?)
    }

    /// Import records from JSON.
    pub fn import_json(&self, json_str: &str) -> Result<usize> {
        let imported: Vec<Record> = serde_json::from_str(json_str)?;
        let count = imported.len();

        let mut records = self.records.write();
        let mut ngram = self.ngram_index.write();
        let mut tag_idx = self.tag_index.write();

        for rec in imported {
            ngram.add(&rec.id, &rec.content);
            for tag in &rec.tags {
                tag_idx
                    .entry(tag.clone())
                    .or_default()
                    .insert(rec.id.clone());
            }
            self.cognitive_store.append_store(&rec)?;
            records.insert(rec.id.clone(), rec);
        }

        // Invalidate recall cache
        self.runtime.clear_recall_caches();

        Ok(count)
    }

    // ── SDR-specific (from aura-memory, for power users) ──

    /// Process text via SDR engine.
    pub fn process(&self, text: &str, pin: Option<bool>) -> Result<String> {
        let pin = pin.unwrap_or(false);
        let result = self.store(
            text,
            if pin { Some(Level::Identity) } else { None },
            None,
            Some(pin),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        Ok(format!(
            "Stored record {} (level={})",
            result.id, result.level
        ))
    }

    /// Legacy-compatible delete helper for server mode.
    pub fn delete_synapse(&self, id: &str) -> bool {
        self.delete(id).unwrap_or(false)
    }

    /// Legacy-compatible full retrieval for server mode.
    pub fn retrieve_full(&self, raw_query: &str, top_k: usize) -> Result<Vec<(StoredRecord, f32)>> {
        let results = self.recall_full(
            raw_query,
            Some(top_k),
            Some(true),
            Some(0.1),
            Some(true),
            None,
            None,
        )?;

        Ok(results
            .into_iter()
            .map(|(score, rec)| (Self::record_to_stored_record(&rec), score))
            .collect())
    }

    /// Batch delete helper for server mode.
    pub fn batch_delete(&self, ids: &[String]) -> usize {
        ids.iter()
            .filter(|id| self.delete(id).ok() == Some(true))
            .count()
    }

    /// Paginated memory listing for server mode.
    pub fn list_memories(
        &self,
        offset: usize,
        limit: usize,
        filter_dna: Option<&str>,
    ) -> (Vec<StoredRecord>, usize) {
        let cache = self.storage.header_cache.read();

        let mut entries: Vec<_> = cache
            .values()
            .filter(|h| match filter_dna {
                Some(dna) if dna == "phantom" => h.dna == "phantom",
                Some(dna) if dna != "all" => h.dna == dna,
                _ => h.dna != "phantom",
            })
            .collect();

        let total = entries.len();
        entries.sort_by(|a, b| {
            b.timestamp()
                .partial_cmp(&a.timestamp())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let records = entries
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|h| StoredRecord {
                id: h.id.clone(),
                dna: h.dna.clone(),
                timestamp: h.timestamp(),
                intensity: h.intensity(),
                stability: h.stability(),
                decay_velocity: h.decay_velocity(),
                entropy: h.entropy(),
                sdr_indices: h.sdr_indices.clone(),
                text: h.text.clone(),
                offset: 0,
            })
            .collect();

        (records, total)
    }

    /// Analytics view for server mode.
    pub fn get_analytics(&self) -> (HashMap<String, usize>, usize, f64, f64) {
        let cache = self.storage.header_cache.read();
        let mut by_dna = HashMap::new();
        let mut oldest = f64::MAX;
        let mut newest = f64::MIN;

        for header in cache.values() {
            *by_dna.entry(header.dna.clone()).or_insert(0) += 1;
            let ts = header.timestamp();
            if ts < oldest {
                oldest = ts;
            }
            if ts > newest {
                newest = ts;
            }
        }

        let total = cache.len();
        if total == 0 {
            oldest = 0.0;
            newest = 0.0;
        }

        (by_dna, total, oldest, newest)
    }

    /// Count phantom records imported via SDR exchange.
    pub fn phantom_count(&self) -> usize {
        self.storage.phantom_count()
    }

    /// Batch ingest with temporal links, compatible with server mode.
    pub fn ingest_batch(&self, texts: Vec<String>) -> Result<usize> {
        self.ingest_batch_with_pin(texts, false)
    }

    /// Batch ingest with pinned identity-level records.
    pub fn ingest_batch_pinned(&self, texts: Vec<String>) -> Result<usize> {
        self.ingest_batch_with_pin(texts, true)
    }

    /// O(1) sequence prediction based on temporal links.
    pub fn retrieve_prediction(&self, current_id: &str) -> Result<Option<StoredRecord>> {
        Ok(self
            .storage
            .get_prediction(current_id)
            .map(|next| StoredRecord {
                id: next.id.clone(),
                dna: next.dna.clone(),
                timestamp: next.timestamp(),
                intensity: next.intensity(),
                stability: next.stability(),
                decay_velocity: next.decay_velocity(),
                entropy: next.entropy(),
                sdr_indices: next.sdr_indices.clone(),
                text: next.text.clone(),
                offset: 0,
            }))
    }

    /// Surprise metric: 1 - Tanimoto(predicted, actual).
    pub fn surprise(&self, predicted_id: &str, actual_text: &str) -> Result<f32> {
        let actual_sdr = self.sdr.text_to_sdr(actual_text, false);
        if let Some(predicted) = self.storage.get_header(predicted_id) {
            let similarity = self
                .sdr
                .tanimoto_sparse(&predicted.sdr_indices, &actual_sdr);
            Ok(1.0 - similarity)
        } else {
            Ok(1.0)
        }
    }

    /// Retrieve top-k via SDR similarity only.
    pub fn retrieve(&self, query: &str, top_k: Option<usize>) -> Result<Vec<String>> {
        let top_k = top_k.unwrap_or(5);
        let records = self.records.read();
        let aura_idx = self.aura_index.read();

        let sdr_results = recall::collect_sdr(
            &self.sdr,
            &self.index,
            &self.storage,
            &aura_idx,
            &records,
            query,
            top_k,
            &[crate::record::DEFAULT_NAMESPACE],
        );

        Ok(sdr_results
            .into_iter()
            .filter_map(|(rid, _)| records.get(&rid).map(|r| r.content.clone()))
            .collect())
    }

    // ── Encryption & Security ──

    /// Check if encryption is enabled.
    pub fn is_encrypted(&self) -> bool {
        self.encryption_key.is_some()
    }

    /// Load synonyms from file.
    pub fn load_synonyms(&self, path: &str) -> Result<usize> {
        let mut ring = self.synonym_ring.write();
        ring.load_toml(Path::new(path))
    }

    /// Check if synonyms are loaded.
    pub fn has_synonyms(&self) -> bool {
        !self.synonym_ring.read().is_empty()
    }

    // ── Namespace Operations ──

    /// List all distinct namespaces present in the brain.
    pub fn list_namespaces(&self) -> Vec<String> {
        let records = self.records.read();
        let mut ns_set: std::collections::HashSet<String> =
            records.values().map(|r| r.namespace.clone()).collect();
        ns_set.insert(crate::record::DEFAULT_NAMESPACE.to_string());
        let mut sorted: Vec<String> = ns_set.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Move a record to a different namespace.
    ///
    /// Prunes connections that would become cross-namespace after the move.
    pub fn move_record(&self, record_id: &str, new_namespace: &str) -> Option<Record> {
        if crate::record::Record::validate_namespace(new_namespace).is_err() {
            return None;
        }
        let mut records = self.records.write();

        // 1. Collect outgoing connection keys and old namespace (immutable access)
        let rec = records.get(record_id)?;
        let old_namespace = rec.namespace.clone();
        let outgoing_keys: Vec<String> = rec.connections.keys().cloned().collect();

        // 2. Move the record
        let rec = records.get_mut(record_id)?;
        rec.namespace = new_namespace.to_string();

        // 3. Determine which outgoing connections are now cross-namespace
        let cross_ns_ids: Vec<String> = outgoing_keys
            .into_iter()
            .filter(|cid| {
                records
                    .get(cid.as_str())
                    .map(|r| r.namespace != new_namespace)
                    .unwrap_or(false)
            })
            .collect();

        // 4. Prune cross-namespace outgoing connections
        //    (re-borrow after immutable filter above)
        let rec = records.get_mut(record_id).unwrap();
        for cid in &cross_ns_ids {
            rec.connections.remove(cid);
        }

        // 5. Prune incoming connections from old-namespace records pointing to this one
        let peers_to_clean: Vec<String> = records
            .iter()
            .filter(|(id, r)| {
                *id != record_id
                    && r.namespace == old_namespace
                    && r.connections.contains_key(record_id)
            })
            .map(|(id, _)| id.clone())
            .collect();
        for pid in &peers_to_clean {
            if let Some(peer) = records.get_mut(pid.as_str()) {
                peer.connections.remove(record_id);
            }
        }

        let _ = self
            .cognitive_store
            .append_update(records.get(record_id).unwrap());
        self.runtime.clear_recall_caches();
        Some(records.get(record_id).unwrap().clone())
    }

    /// Get record counts per namespace.
    pub fn namespace_stats(&self) -> HashMap<String, usize> {
        let records = self.records.read();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for rec in records.values() {
            *counts.entry(rec.namespace.clone()).or_insert(0) += 1;
        }
        counts
    }
}

// ── PyO3 Bindings ──

/// Extract namespaces from a Python argument that can be str, list[str], or None.
///
/// - `None` → `None` (will default to `["default"]` in Rust methods)
/// - `"sandbox"` → `Some(vec!["sandbox"])`
/// - `["default", "sandbox"]` → `Some(vec!["default", "sandbox"])`
#[cfg(feature = "python")]
fn extract_namespaces(
    ns: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
) -> PyResult<Option<Vec<String>>> {
    use pyo3::prelude::*;
    match ns {
        None => Ok(None),
        Some(obj) => {
            // Try extracting as a single string first
            if let Ok(s) = obj.extract::<String>() {
                return Ok(Some(vec![s]));
            }
            // Try extracting as a list of strings
            if let Ok(list) = obj.extract::<Vec<String>>() {
                return Ok(Some(list));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "namespace must be a str, list[str], or None",
            ))
        }
    }
}

impl Drop for Aura {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

// ── v5: Autonomous Cognitive Plasticity ─────────────────────────────────────

impl Aura {
    // ── Plasticity mode control ──────────────────────────────────────────────

    /// Get the current plasticity mode.
    ///
    /// Default: `PlasticityMode::Off` — the system never changes silently.
    pub fn get_plasticity_mode(&self) -> crate::experience::PlasticityMode {
        self.runtime.plasticity_mode()
    }

    /// Set the plasticity mode.
    ///
    /// `Off`     — capture_experience() is a no-op.
    /// `Observe` — events extracted and logged, never applied.
    /// `Limited` — applied with Default PlasticityPolicy.
    /// `Full`    — applied with operator-supplied PlasticityPolicy.
    pub fn set_plasticity_mode(&self, mode: crate::experience::PlasticityMode) {
        self.runtime
            .plasticity_mode
            .store(mode as u8, std::sync::atomic::Ordering::Relaxed);
    }

    // ── Capture ──────────────────────────────────────────────────────────────

    /// Observe a model response and extract structured experience events.
    ///
    /// **Contract**: this method ONLY extracts and enriches events — it does NOT
    /// mutate any state. Mutations happen exclusively in maintenance phase 3.6
    /// after the caller enqueues the capture via `ingest_experience_batch()`.
    ///
    /// - In `PlasticityMode::Off` — returns an empty capture immediately (no-op).
    /// - In `PlasticityMode::Observe` — extracts events, marks report as observe-only.
    /// - In `PlasticityMode::Limited` — extracts + enriches; mutations deferred to maintenance.
    /// - In `PlasticityMode::Full` — same as Limited; operator policy applied in maintenance.
    ///
    /// Call `ingest_experience_batch()` to enqueue for the next maintenance cycle.
    pub fn capture_experience(
        &self,
        prompt: &str,
        retrieved_context_ids: &[String],
        model_response: &str,
        session_id: Option<&str>,
        source: crate::experience::ExperienceSource,
    ) -> Result<crate::experience::ExperienceCapture> {
        use crate::experience::{
            extract_experience_events_heuristic, hash_prompt, now_secs, ExperienceCapture,
            PlasticityMode, PlasticityReport,
        };

        let mode = self.runtime.plasticity_mode();

        // Fast path: Off mode — no-op, zero events
        if mode == PlasticityMode::Off {
            return Ok(ExperienceCapture {
                session_id: session_id.unwrap_or("").to_string(),
                timestamp: now_secs(),
                prompt_hash: hash_prompt(prompt),
                response_summary: model_response.chars().take(200).collect(),
                context_record_ids: retrieved_context_ids.to_vec(),
                source,
                raw_events: vec![],
                plasticity_report: PlasticityReport::default(),
            });
        }

        // Step 1: heuristic extraction (Variant A)
        let mut raw_events = extract_experience_events_heuristic(model_response);

        // Step 2: SDR similarity pass (Variant B) — upgrade Claims to Confirmations
        // or Contradictions by comparing against existing records.
        // NOTE: read-only operation — does not mutate any record.
        raw_events = self.enrich_events_with_sdr(raw_events, &source);

        let (base_policy, _) = self.resolve_plasticity_policy(mode);
        let policy = self.get_plasticity_risk().apply_throttling(base_policy);
        let plasticity_report = self.preview_experience_report(&raw_events, &policy, &source);

        Ok(ExperienceCapture {
            session_id: session_id.unwrap_or("").to_string(),
            timestamp: now_secs(),
            prompt_hash: hash_prompt(prompt),
            response_summary: model_response.chars().take(200).collect(),
            context_record_ids: retrieved_context_ids.to_vec(),
            source,
            raw_events,
            plasticity_report,
        })
    }

    /// Namespace-aware variant of `capture_experience()`.
    ///
    /// If `namespace` is frozen via `freeze_namespace_plasticity()`, new record
    /// creation is suppressed (Confirmation/Contradiction events still apply).
    /// In all other respects behaves identically to `capture_experience()`.
    pub fn capture_experience_in_namespace(
        &self,
        prompt: &str,
        retrieved_context_ids: &[String],
        model_response: &str,
        session_id: Option<&str>,
        source: crate::experience::ExperienceSource,
        namespace: &str,
    ) -> Result<crate::experience::ExperienceCapture> {
        use crate::experience::{
            extract_experience_events_heuristic, hash_prompt, now_secs, ExperienceCapture,
            PlasticityMode, PlasticityReport,
        };

        let mode = self.runtime.plasticity_mode();

        if mode == PlasticityMode::Off {
            return Ok(ExperienceCapture {
                session_id: session_id.unwrap_or("").to_string(),
                timestamp: now_secs(),
                prompt_hash: hash_prompt(prompt),
                response_summary: model_response.chars().take(200).collect(),
                context_record_ids: retrieved_context_ids.to_vec(),
                source,
                raw_events: vec![],
                plasticity_report: PlasticityReport::default(),
            });
        }

        let mut raw_events = extract_experience_events_heuristic(model_response);
        raw_events = self.enrich_events_with_sdr(raw_events, &source);

        // Phase 4.3: record the freeze flag in the capture metadata so phase 3.6
        // knows not to inject records for this namespace.
        // No mutations happen here — deferred to maintenance.
        let namespace_frozen = self.is_namespace_plasticity_frozen(namespace);
        if namespace_frozen {
            tracing::debug!(
                namespace,
                "capture_experience_in_namespace: namespace frozen — new records suppressed in maintenance"
            );
        }

        let (base_policy, _) = self.resolve_plasticity_policy(mode);
        let mut policy = self.get_plasticity_risk().apply_throttling(base_policy);
        if namespace_frozen {
            policy.allow_new_records = false;
            policy.max_new_records_per_call = 0;
        }
        let plasticity_report = self.preview_experience_report(&raw_events, &policy, &source);

        Ok(ExperienceCapture {
            session_id: session_id.unwrap_or("").to_string(),
            timestamp: now_secs(),
            prompt_hash: hash_prompt(prompt),
            response_summary: model_response.chars().take(200).collect(),
            context_record_ids: retrieved_context_ids.to_vec(),
            source,
            raw_events,
            plasticity_report,
        })
    }

    /// Enrich heuristic events with SDR similarity (Variant B).
    ///
    /// - Claims with Tanimoto ≥ CONFIRMATION_TANIMOTO_THRESHOLD → upgraded to Confirmation.
    /// - Claims with CONTRADICTION_TANIMOTO_THRESHOLD ≤ Tanimoto < CONFIRMATION_TANIMOTO_THRESHOLD
    ///   + contradiction keyword → upgraded to Contradiction.
    fn enrich_events_with_sdr(
        &self,
        events: Vec<crate::experience::ExperienceEvent>,
        source: &crate::experience::ExperienceSource,
    ) -> Vec<crate::experience::ExperienceEvent> {
        use crate::experience::{
            ConflictSeverity, ExperienceEvent, CONFIRMATION_STRENGTH_DELTA,
            CONFIRMATION_TANIMOTO_THRESHOLD, CONTRADICTION_TANIMOTO_THRESHOLD,
            CONTRADICTION_VOLATILITY_DELTA,
        };

        // Only enrich ModelInference — WorldFact/HumanStatement go in as-is
        if *source != crate::experience::ExperienceSource::ModelInference {
            return events;
        }

        let records = self.records.read();
        let belief_engine = self.belief_engine.read();

        events
            .into_iter()
            .map(|event| {
                let claim_text = match &event {
                    ExperienceEvent::Claim { text, .. } => text.clone(),
                    _ => return event, // non-Claim events pass through
                };

                let claim_sdr = self.sdr.text_to_sdr_lowered(&claim_text, false);

                // Use the cached SDR lookup to avoid recomputing per-record SDRs.
                let sdr_cache = self.runtime.sdr_lookup_cache.read();

                // Find the closest existing record by SDR Tanimoto
                let best = records
                    .values()
                    .filter_map(|rec| {
                        let rec_sdr = if let Some(cached) = sdr_cache.get(&rec.id) {
                            cached.clone()
                        } else {
                            self.sdr.text_to_sdr_lowered(&rec.content, false)
                        };
                        let sim = self.sdr.tanimoto_sparse(&claim_sdr, &rec_sdr);
                        if sim >= CONTRADICTION_TANIMOTO_THRESHOLD {
                            Some((sim, rec))
                        } else {
                            None
                        }
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let Some((sim, closest_rec)) = best else {
                    return event; // no similar record — keep as Claim
                };

                // Check if closest record belongs to a stable belief
                let belief_id = belief_engine
                    .belief_for_record(&closest_rec.id)
                    .map(|b| b.id.clone());

                if sim >= CONFIRMATION_TANIMOTO_THRESHOLD {
                    // Strong overlap → Confirmation
                    if let Some(bid) = belief_id {
                        return ExperienceEvent::Confirmation {
                            belief_id: bid,
                            strength_delta: CONFIRMATION_STRENGTH_DELTA,
                        };
                    }
                }

                // Moderate overlap + contradiction keyword → Contradiction
                let is_recorded =
                    matches!(closest_rec.source_type.as_str(), "recorded" | "retrieved");
                let lower = claim_text.to_lowercase();
                let has_contradiction_keyword = [
                    "but",
                    "however",
                    "actually",
                    "incorrect",
                    "wrong",
                    "not true",
                    "але",
                    "однак",
                    "проте",
                    "насправді",
                    "навпаки",
                    "не є",
                    "хибно",
                ]
                .iter()
                .any(|kw| lower.contains(kw));

                if has_contradiction_keyword {
                    if let Some(bid) = belief_id {
                        return ExperienceEvent::Contradiction {
                            belief_id: bid,
                            volatility_delta: CONTRADICTION_VOLATILITY_DELTA,
                            severity: if is_recorded {
                                ConflictSeverity::Strong
                            } else {
                                ConflictSeverity::Weak
                            },
                        };
                    }
                }

                event // keep as Claim if no upgrade triggered
            })
            .collect()
    }

    fn resolve_plasticity_policy(
        &self,
        mode: crate::experience::PlasticityMode,
    ) -> (crate::experience::PlasticityPolicy, &'static str) {
        match mode {
            crate::experience::PlasticityMode::Observe => (
                crate::experience::PlasticityPolicy::observe_only(),
                "observe",
            ),
            crate::experience::PlasticityMode::Full => (
                self.runtime.custom_plasticity_policy.read().clone(),
                "custom",
            ),
            crate::experience::PlasticityMode::Limited => {
                (crate::experience::PlasticityPolicy::default(), "default")
            }
            crate::experience::PlasticityMode::Off => {
                (crate::experience::PlasticityPolicy::observe_only(), "off")
            }
        }
    }

    fn preview_experience_report(
        &self,
        events: &[crate::experience::ExperienceEvent],
        policy: &crate::experience::PlasticityPolicy,
        source: &crate::experience::ExperienceSource,
    ) -> crate::experience::PlasticityReport {
        use crate::experience::{apply_contradiction_asymmetry, ConflictSeverity, ExperienceEvent};

        let mut report = crate::experience::PlasticityReport::default();
        let is_model_inference = *source == crate::experience::ExperienceSource::ModelInference;
        let belief_engine = self.belief_engine.read();

        for event in events {
            report.events_processed += 1;
            match event {
                ExperienceEvent::Claim {
                    text, certainty, ..
                } => {
                    if !policy.allow_new_records {
                        report.events_skipped += 1;
                        report.skipped_reasons.push("policy:no_new_records".into());
                        continue;
                    }
                    if report.new_records_stored >= policy.max_new_records_per_call as usize {
                        report.events_skipped += 1;
                        report
                            .skipped_reasons
                            .push("policy:max_records_reached".into());
                        continue;
                    }
                    if *certainty < policy.min_claim_certainty {
                        report.events_skipped += 1;
                        report
                            .skipped_reasons
                            .push(format!("policy:certainty_too_low:{:?}", certainty));
                        continue;
                    }

                    let contradicts_recorded = self.claim_contradicts_recorded(text);
                    let (allow_store, _) = apply_contradiction_asymmetry(
                        certainty,
                        contradicts_recorded && is_model_inference,
                    );
                    if !allow_store {
                        report.events_skipped += 1;
                        report.hallucination_alerts += 1;
                        report
                            .skipped_reasons
                            .push("guard:contradicts_recorded_belief".into());
                        continue;
                    }

                    report.new_records_stored += 1;
                }
                ExperienceEvent::Confirmation { belief_id, .. } => {
                    if !policy.allow_belief_reinforcement {
                        report.events_skipped += 1;
                        report
                            .skipped_reasons
                            .push("policy:no_belief_reinforcement".into());
                        continue;
                    }
                    if belief_engine.beliefs.contains_key(belief_id) {
                        report.beliefs_reinforced += 1;
                    }
                }
                ExperienceEvent::Contradiction {
                    belief_id,
                    severity,
                    ..
                } => {
                    if !policy.allow_volatility_increase {
                        report.events_skipped += 1;
                        report
                            .skipped_reasons
                            .push("policy:no_volatility_increase".into());
                        continue;
                    }
                    if is_model_inference && *severity == ConflictSeverity::Strong {
                        report.hallucination_alerts += 1;
                    }
                    if belief_engine.beliefs.contains_key(belief_id) {
                        let affected = belief_engine
                            .hypotheses
                            .values()
                            .filter(|h| h.belief_id == *belief_id)
                            .map(|h| h.prototype_record_ids.len())
                            .sum::<usize>();
                        report.volatility_increases += affected.max(1);
                    }
                }
                ExperienceEvent::Commitment { .. } | ExperienceEvent::UncertaintyMarker { .. } => {
                    report.events_skipped += 1;
                    report
                        .skipped_reasons
                        .push("event:commitment_or_uncertainty_logged".into());
                }
            }
        }

        report
    }

    /// Apply experience events to the cognitive substrate.
    ///
    /// Enforces all Phase 3.1 anti-hallucination guards.
    /// Returns a PlasticityReport describing what was mutated.
    fn apply_experience_internal(
        &self,
        events: &[crate::experience::ExperienceEvent],
        policy: &crate::experience::PlasticityPolicy,
        source: &crate::experience::ExperienceSource,
        session_id: Option<&str>,
        prompt_hash: &str,
        policy_name: &str,
    ) -> crate::experience::PlasticityReport {
        use crate::experience::{
            apply_confidence_ceiling, apply_contradiction_asymmetry, now_secs, ConflictSeverity,
            ExperienceEvent, PlasticityAuditEntry, PlasticityReport, HALLUCINATION_ALERT_THRESHOLD,
        };

        let mut report = PlasticityReport::default();
        let sid = session_id.unwrap_or("").to_string();
        let is_model_inference = *source == crate::experience::ExperienceSource::ModelInference;

        for event in events {
            report.events_processed += 1;

            match event {
                // ── New Claim → store as generated record ─────────────────
                ExperienceEvent::Claim {
                    text,
                    tags,
                    semantic_type,
                    certainty,
                } => {
                    if !policy.allow_new_records {
                        report.events_skipped += 1;
                        report.skipped_reasons.push("policy:no_new_records".into());
                        continue;
                    }
                    if report.new_records_stored >= policy.max_new_records_per_call as usize {
                        report.events_skipped += 1;
                        report
                            .skipped_reasons
                            .push("policy:max_records_reached".into());
                        continue;
                    }
                    if *certainty < policy.min_claim_certainty {
                        report.events_skipped += 1;
                        report
                            .skipped_reasons
                            .push(format!("policy:certainty_too_low:{:?}", certainty));
                        continue;
                    }

                    // Guard: check contradiction asymmetry before storing
                    let contradicts_recorded = self.claim_contradicts_recorded(text);
                    let (allow_store, _volatility) = apply_contradiction_asymmetry(
                        certainty,
                        contradicts_recorded && is_model_inference,
                    );

                    if !allow_store {
                        report.events_skipped += 1;
                        report.hallucination_alerts += 1;
                        report
                            .skipped_reasons
                            .push("guard:contradicts_recorded_belief".into());
                        continue;
                    }

                    // Store the claim
                    let source_type = source.to_source_type();
                    let mut base_conf = certainty.base_confidence();
                    if is_model_inference {
                        base_conf = apply_confidence_ceiling(base_conf);
                    }

                    let store_result = self.store_with_channel(
                        text,
                        Some(crate::levels::Level::Working),
                        Some(tags.clone()),
                        Some(false), // never pin
                        Some("text"),
                        Some(source_type),
                        Some({
                            let mut meta = std::collections::HashMap::new();
                            meta.insert(
                                "experience_source".to_string(),
                                source.metadata_value().to_string(),
                            );
                            meta.insert("session_id".to_string(), sid.clone());
                            meta
                        }),
                        Some(true), // deduplicate
                        None,       // no caused_by
                        Some("experience"),
                        Some(false), // auto_promote=false for generated
                        None,        // namespace from session context
                        Some(semantic_type.as_str()),
                    );

                    match store_result {
                        Ok(rec) => {
                            // Override confidence to the certainty-based value
                            // (store_with_channel sets it from source_type default)
                            if let Ok(mut records) =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    self.records.write()
                                }))
                            {
                                if let Some(r) = records.get_mut(&rec.id) {
                                    r.confidence = base_conf;
                                }
                            }

                            report.new_records_stored += 1;
                            self.write_plasticity_audit(PlasticityAuditEntry {
                                timestamp: now_secs(),
                                session_id: sid.clone(),
                                event_kind: "new_record".to_string(),
                                target_id: rec.id.clone(),
                                source_prompt_hash: prompt_hash.to_string(),
                                confidence_before: None,
                                confidence_after: Some(base_conf),
                                policy_name: policy_name.to_string(),
                            });
                        }
                        Err(_) => {
                            report.events_skipped += 1;
                            report.skipped_reasons.push("store_failed".into());
                        }
                    }
                }

                // ── Confirmation → nudge belief confidence ────────────────
                ExperienceEvent::Confirmation {
                    belief_id,
                    strength_delta,
                } => {
                    if !policy.allow_belief_reinforcement {
                        report.events_skipped += 1;
                        report
                            .skipped_reasons
                            .push("policy:no_belief_reinforcement".into());
                        continue;
                    }
                    let mut engine = self.belief_engine.write();
                    if let Some(belief) = engine.beliefs.get_mut(belief_id) {
                        let before = belief.confidence;
                        belief.confidence = (belief.confidence + strength_delta).min(1.0);
                        report.beliefs_reinforced += 1;
                        let after = belief.confidence;
                        drop(engine);
                        self.write_plasticity_audit(PlasticityAuditEntry {
                            timestamp: now_secs(),
                            session_id: sid.clone(),
                            event_kind: "belief_reinforced".to_string(),
                            target_id: belief_id.clone(),
                            source_prompt_hash: prompt_hash.to_string(),
                            confidence_before: Some(before),
                            confidence_after: Some(after),
                            policy_name: policy_name.to_string(),
                        });
                    }
                }

                // ── Contradiction → raise volatility on affected records ──
                ExperienceEvent::Contradiction {
                    belief_id,
                    volatility_delta,
                    severity,
                } => {
                    if !policy.allow_volatility_increase {
                        report.events_skipped += 1;
                        report
                            .skipped_reasons
                            .push("policy:no_volatility_increase".into());
                        continue;
                    }

                    // Guard 4: recorded wins over generated
                    if is_model_inference && *severity == ConflictSeverity::Strong {
                        report.hallucination_alerts += 1;
                    }

                    let engine = self.belief_engine.read();
                    let record_ids: Vec<String> = engine
                        .beliefs
                        .get(belief_id)
                        .map(|_b| {
                            engine
                                .hypotheses
                                .values()
                                .filter(|h| h.belief_id == *belief_id)
                                .flat_map(|h| h.prototype_record_ids.iter().cloned())
                                .collect()
                        })
                        .unwrap_or_default();
                    drop(engine);

                    let mut records = self.records.write();
                    for rid in &record_ids {
                        if let Some(rec) = records.get_mut(rid) {
                            rec.volatility = (rec.volatility + volatility_delta).min(1.0);
                            report.volatility_increases += 1;
                        }
                    }
                    drop(records);

                    self.write_plasticity_audit(PlasticityAuditEntry {
                        timestamp: now_secs(),
                        session_id: sid.clone(),
                        event_kind: "volatility_raised".to_string(),
                        target_id: belief_id.clone(),
                        source_prompt_hash: prompt_hash.to_string(),
                        confidence_before: None,
                        confidence_after: None,
                        policy_name: policy_name.to_string(),
                    });
                }

                // ── Other events: log only ────────────────────────────────
                ExperienceEvent::Commitment { .. } | ExperienceEvent::UncertaintyMarker { .. } => {
                    // Logged in raw_events; no mutation in default policy
                    report.events_skipped += 1;
                    report
                        .skipped_reasons
                        .push("event:commitment_or_uncertainty_logged".into());
                }
            }
        }

        // Hallucination alert threshold check
        if report.hallucination_alerts >= HALLUCINATION_ALERT_THRESHOLD {
            tracing::warn!(
                session_id = %sid,
                alerts = report.hallucination_alerts,
                "PlasticityHallucinationAlert: generated claims frequently contradict recorded beliefs"
            );
        }

        // ── Phase 3.2: accumulate risk telemetry ──
        use std::sync::atomic::Ordering;
        self.runtime
            .plasticity_hallucination_alerts
            .fetch_add(report.hallucination_alerts as u64, Ordering::Relaxed);
        self.runtime
            .plasticity_contradictions_total
            .fetch_add(report.volatility_increases as u64, Ordering::Relaxed);
        self.runtime
            .plasticity_events_total
            .fetch_add(report.events_processed as u64, Ordering::Relaxed);

        report
    }

    /// Check if a claim text contradicts any stable recorded/retrieved belief.
    fn claim_contradicts_recorded(&self, claim: &str) -> bool {
        use crate::experience::{
            CONFIRMATION_TANIMOTO_THRESHOLD, CONTRADICTION_TANIMOTO_THRESHOLD,
        };

        let claim_sdr = self.sdr.text_to_sdr_lowered(claim, false);
        let lower = claim.to_lowercase();
        let has_negation = [
            "not",
            "no ",
            "never",
            "wrong",
            "incorrect",
            "false",
            "не є",
            "не буде",
            "хибно",
            "помилково",
        ]
        .iter()
        .any(|kw| lower.contains(kw));

        if !has_negation {
            return false;
        }

        let records = self.records.read();
        records.values().any(|rec| {
            if !matches!(rec.source_type.as_str(), "recorded" | "retrieved") {
                return false;
            }
            let rec_sdr = self.sdr.text_to_sdr_lowered(&rec.content, false);
            let sim = self.sdr.tanimoto_sparse(&claim_sdr, &rec_sdr);
            sim >= CONTRADICTION_TANIMOTO_THRESHOLD && sim < CONFIRMATION_TANIMOTO_THRESHOLD
        })
    }

    /// Write a plasticity audit entry to the audit log if available.
    fn write_plasticity_audit(&self, entry: crate::experience::PlasticityAuditEntry) {
        if let Some(ref log) = self.audit_log {
            let reason = format!(
                "plasticity:{}:{}",
                entry.event_kind, entry.source_prompt_hash
            );
            let _ = log.log_correction("experience", &entry.target_id, &entry.event_kind, &reason);
        }
    }

    // ── Ingest queue ─────────────────────────────────────────────────────────

    /// Enqueue experience captures for processing in the next maintenance cycle.
    ///
    /// This is non-blocking. Captures are drained during maintenance phase 3.6.
    /// Returns the number of captures accepted.
    pub fn ingest_experience_batch(
        &self,
        batch: Vec<crate::experience::ExperienceCapture>,
    ) -> Result<usize> {
        let count = batch.len();
        self.runtime.experience_queue.enqueue(batch);
        Ok(count)
    }

    /// Drain all pending experience captures (called by MaintenanceService phase 3.6).
    pub(crate) fn drain_experience_queue(&self) -> Vec<crate::experience::ExperienceCapture> {
        self.runtime.experience_queue.drain()
    }

    /// Number of captures currently waiting in the queue.
    pub fn experience_queue_len(&self) -> usize {
        self.runtime.experience_queue.len()
    }

    /// Compute the current plasticity risk assessment (Phase 3.2).
    ///
    /// Reads cumulative telemetry from atomic counters accumulated across all
    /// `capture_experience()` calls since the last `reset_plasticity_telemetry()`.
    ///
    /// The returned `PlasticityRiskAssessment` also drives automatic policy
    /// throttling: when `capture_experience()` is called in Limited/Full mode,
    /// the current risk is evaluated and the effective policy is tightened
    /// if risk is Restrict or Pause.
    pub fn get_plasticity_risk(&self) -> crate::experience::PlasticityRiskAssessment {
        use crate::experience::PlasticityRiskAssessment;
        use std::sync::atomic::Ordering;

        let hallucination_alerts = self
            .runtime
            .plasticity_hallucination_alerts
            .load(Ordering::Relaxed) as u32;
        let contradictions = self
            .runtime
            .plasticity_contradictions_total
            .load(Ordering::Relaxed) as usize;
        let events_total = self.runtime.plasticity_events_total.load(Ordering::Relaxed) as usize;

        // Count generated records in the store.
        let (total_records, generated_records) = {
            let records = self.records.read();
            let total = records.len();
            let generated = records
                .values()
                .filter(|r| r.source_type == "generated")
                .count();
            (total, generated)
        };

        PlasticityRiskAssessment::compute(
            total_records,
            generated_records,
            events_total,
            contradictions,
            hallucination_alerts,
        )
    }

    /// Reset all accumulated plasticity risk telemetry counters to zero.
    ///
    /// Call this after an operator review, or after purging generated records,
    /// to give the system a clean baseline.
    pub fn reset_plasticity_telemetry(&self) {
        use std::sync::atomic::Ordering;
        self.runtime
            .plasticity_hallucination_alerts
            .store(0, Ordering::Relaxed);
        self.runtime
            .plasticity_contradictions_total
            .store(0, Ordering::Relaxed);
        self.runtime
            .plasticity_events_total
            .store(0, Ordering::Relaxed);
    }

    // ── Bug 3 fix: Operator-supplied policy for PlasticityMode::Full ─────────

    /// Set the custom `PlasticityPolicy` used when the system operates in
    /// `PlasticityMode::Full`.
    ///
    /// In Full mode the operator takes responsibility for tuning the policy.
    /// If this method is never called the policy defaults to
    /// `PlasticityPolicy::default()` — conservative limits, no ceiling override.
    ///
    /// The policy is stored persistently in `AuraRuntimeState` and read by
    /// maintenance phase 3.6 on every cycle.
    pub fn set_plasticity_policy(&self, policy: crate::experience::PlasticityPolicy) {
        *self.runtime.custom_plasticity_policy.write() = policy;
    }

    /// Return a clone of the currently stored custom plasticity policy.
    ///
    /// This is the policy used by `PlasticityMode::Full` during maintenance
    /// phase 3.6.  Returns `PlasticityPolicy::default()` if no custom policy
    /// has been set yet.
    pub fn get_plasticity_policy(&self) -> crate::experience::PlasticityPolicy {
        self.runtime.custom_plasticity_policy.read().clone()
    }

    // ── Phase 4.2: Purge inference records ───────────────────────────────────

    /// Delete all records with `source_type == "generated"` that match the
    /// given filters.
    ///
    /// This is the operator rollback tool: if the model generated garbage,
    /// purge it and let the next maintenance cycle rebuild clean beliefs.
    ///
    /// - `before_timestamp`: only delete records created before this Unix
    ///   timestamp.  `None` means "all time" (delete all generated records).
    /// - `namespace`: only delete records belonging to this namespace.
    ///   `None` means all namespaces.
    ///
    /// After purging, call `reset_plasticity_telemetry()` to clear risk
    /// counters so the risk score reflects the fresh state.
    ///
    /// **Does not call `flush()` — the caller should call `run_maintenance()`
    /// or `flush()` explicitly to persist the deletion.**
    pub fn purge_inference_records(
        &self,
        before_timestamp: Option<u64>,
        namespace: Option<&str>,
    ) -> Result<crate::experience::PurgeReport> {
        let mut records = self.records.write();
        let examined = records.len();

        let mut to_remove: Vec<String> = Vec::new();

        for (id, rec) in records.iter() {
            if rec.source_type != "generated" {
                continue;
            }

            // Namespace filter
            if let Some(ns) = namespace {
                if rec.namespace != ns {
                    continue;
                }
            }

            // Timestamp filter
            if let Some(cutoff) = before_timestamp {
                // Record.created_at is stored as Unix seconds in metadata or as a field.
                // We check metadata["created_at"] as ISO-8601 string or use a fallback.
                let created = rec
                    .metadata
                    .get("created_at")
                    .or_else(|| rec.metadata.get("timestamp"))
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.timestamp() as u64)
                    .unwrap_or(0);

                if created >= cutoff {
                    continue; // keep records created at or after cutoff
                }
            }

            to_remove.push(id.clone());
        }

        let removed = to_remove.len();
        for id in &to_remove {
            records.remove(id);
        }
        drop(records);

        // Invalidate caches — beliefs/concepts/causal built on removed records
        // are stale and will be rebuilt on next maintenance.
        self.runtime.clear_recall_caches();

        tracing::info!(
            removed,
            namespace = namespace.unwrap_or("*"),
            before_timestamp,
            "purge_inference_records: removed generated records"
        );

        Ok(crate::experience::PurgeReport {
            examined,
            removed,
            namespace_filter: namespace.map(|s| s.to_string()),
            before_timestamp,
        })
    }

    // ── Phase 4.3: Namespace plasticity freeze ───────────────────────────────

    /// Prevent inference from creating new records in `namespace`.
    ///
    /// Existing records are not modified — only `capture_experience()` is
    /// blocked from injecting new generated records into this namespace.
    ///
    /// Typical use: freeze "medical", "legal", "identity" namespaces so that
    /// model inference cannot silently pollute high-trust knowledge areas.
    pub fn freeze_namespace_plasticity(&self, namespace: &str) -> Result<()> {
        self.runtime
            .frozen_plasticity_namespaces
            .write()
            .insert(namespace.to_string());
        tracing::info!(namespace, "freeze_namespace_plasticity: namespace frozen");
        Ok(())
    }

    /// Re-enable inference for a previously frozen namespace.
    pub fn unfreeze_namespace_plasticity(&self, namespace: &str) -> Result<()> {
        self.runtime
            .frozen_plasticity_namespaces
            .write()
            .remove(namespace);
        tracing::info!(
            namespace,
            "unfreeze_namespace_plasticity: namespace unfrozen"
        );
        Ok(())
    }

    /// Check whether a namespace is currently frozen from plasticity.
    pub fn is_namespace_plasticity_frozen(&self, namespace: &str) -> bool {
        self.runtime
            .frozen_plasticity_namespaces
            .read()
            .contains(namespace)
    }

    /// List all currently frozen namespaces.
    pub fn get_frozen_plasticity_namespaces(&self) -> Vec<String> {
        self.runtime
            .frozen_plasticity_namespaces
            .read()
            .iter()
            .cloned()
            .collect()
    }
}

fn preview_text(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

#[cfg(feature = "python")]
fn belief_to_py(py: Python<'_>, belief: &crate::belief::Belief) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("id", &belief.id)?;
    dict.set_item("key", &belief.key)?;
    dict.set_item("state", format!("{:?}", belief.state).to_lowercase())?;
    dict.set_item("winner_id", &belief.winner_id)?;
    dict.set_item("hypothesis_ids", &belief.hypothesis_ids)?;
    dict.set_item("score", belief.score)?;
    dict.set_item("confidence", belief.confidence)?;
    dict.set_item("support_mass", belief.support_mass)?;
    dict.set_item("conflict_mass", belief.conflict_mass)?;
    dict.set_item("stability", belief.stability)?;
    dict.set_item("volatility", belief.volatility)?;
    dict.set_item("last_updated", belief.last_updated)?;
    Ok(dict.unbind().into_any())
}

#[cfg(feature = "python")]
fn policy_hint_to_py(py: Python<'_>, hint: &crate::policy::PolicyHint) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("id", &hint.id)?;
    dict.set_item("key", &hint.key)?;
    dict.set_item("namespace", &hint.namespace)?;
    dict.set_item("domain", &hint.domain)?;
    dict.set_item(
        "action_kind",
        match hint.action_kind {
            crate::policy::PolicyActionKind::Prefer => "prefer",
            crate::policy::PolicyActionKind::Recommend => "recommend",
            crate::policy::PolicyActionKind::VerifyFirst => "verify_first",
            crate::policy::PolicyActionKind::Avoid => "avoid",
            crate::policy::PolicyActionKind::Warn => "warn",
        },
    )?;
    dict.set_item("recommendation", &hint.recommendation)?;
    dict.set_item(
        "state",
        match hint.state {
            crate::policy::PolicyState::Candidate => "candidate",
            crate::policy::PolicyState::Stable => "stable",
            crate::policy::PolicyState::Suppressed => "suppressed",
            crate::policy::PolicyState::Rejected => "rejected",
        },
    )?;
    dict.set_item("trigger_causal_ids", &hint.trigger_causal_ids)?;
    dict.set_item("trigger_concept_ids", &hint.trigger_concept_ids)?;
    dict.set_item("trigger_belief_ids", &hint.trigger_belief_ids)?;
    dict.set_item("supporting_record_ids", &hint.supporting_record_ids)?;
    dict.set_item("cause_record_ids", &hint.cause_record_ids)?;
    dict.set_item("confidence", hint.confidence)?;
    dict.set_item("utility_score", hint.utility_score)?;
    dict.set_item("risk_score", hint.risk_score)?;
    dict.set_item("policy_strength", hint.policy_strength)?;
    dict.set_item("last_updated", hint.last_updated)?;
    Ok(dict.unbind().into_any())
}

fn maintenance_trends_path(root: &Path) -> PathBuf {
    root.join(MAINTENANCE_TRENDS_FILE)
}

fn reflection_summaries_path(root: &Path) -> PathBuf {
    root.join(REFLECTION_SUMMARIES_FILE)
}

fn persistence_manifest_path(root: &Path) -> PathBuf {
    root.join(PERSISTENCE_MANIFEST_FILE)
}

fn startup_event(
    surface: &str,
    path: String,
    status: &str,
    detail: Option<String>,
    recovered: bool,
) -> StartupValidationEvent {
    StartupValidationEvent {
        surface: surface.to_string(),
        path,
        status: status.to_string(),
        detail,
        recovered,
    }
}

fn finalize_startup_validation_report(
    events: Vec<StartupValidationEvent>,
) -> StartupValidationReport {
    let mut report = StartupValidationReport {
        events,
        ..StartupValidationReport::default()
    };
    for event in &report.events {
        match event.status.as_str() {
            "loaded" => report.loaded_surfaces += 1,
            "missing_fallback" | "empty_fallback" => report.missing_fallbacks += 1,
            "load_error_fallback" => report.recovered_fallbacks += 1,
            "derived_skipped" => report.derived_skips += 1,
            _ => {}
        }
        if event.recovered {
            report.has_recovery_warnings = true;
        }
    }
    report
}

fn issue_severity(score: f32, high_threshold: f32, medium_threshold: f32) -> String {
    if score >= high_threshold {
        "high".into()
    } else if score >= medium_threshold {
        "medium".into()
    } else {
        "low".into()
    }
}

fn derive_maintenance_trend_direction(
    summary: &background_brain::MaintenanceTrendSummary,
) -> String {
    if summary.recent.len() < 2 {
        return "insufficient_data".into();
    }

    let first = &summary.recent[0];
    let last = summary.recent.last().expect("recent has at least 2 items");
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
        "worsening".into()
    } else if delta < -1.0 {
        "improving".into()
    } else {
        "stable".into()
    }
}

fn infer_namespace_from_correction_target(entry: &CorrectionLogEntry) -> Option<String> {
    entry
        .target_id
        .split(':')
        .next()
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn save_persistence_manifest(root: &Path, manifest: &PersistenceManifest) -> Result<()> {
    let path = persistence_manifest_path(root);
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn load_persistence_manifest_with_validation(
    root: &Path,
    events: &mut Vec<StartupValidationEvent>,
) -> PersistenceManifest {
    let path = persistence_manifest_path(root);
    let current = PersistenceManifest::current();

    let manifest = if !path.exists() {
        events.push(startup_event(
            "persistence_manifest",
            path.display().to_string(),
            "missing_fallback",
            Some("persistence manifest missing; created current manifest".into()),
            true,
        ));
        current.clone()
    } else {
        match std::fs::read_to_string(&path) {
            Ok(contents) if contents.trim().is_empty() => {
                events.push(startup_event(
                    "persistence_manifest",
                    path.display().to_string(),
                    "empty_fallback",
                    Some("persistence manifest was empty; created current manifest".into()),
                    true,
                ));
                current.clone()
            }
            Ok(contents) => match serde_json::from_str::<PersistenceManifest>(&contents) {
                Ok(manifest) => manifest,
                Err(err) => {
                    events.push(startup_event(
                        "persistence_manifest",
                        path.display().to_string(),
                        "load_error_fallback",
                        Some(err.to_string()),
                        true,
                    ));
                    current.clone()
                }
            },
            Err(err) => {
                events.push(startup_event(
                    "persistence_manifest",
                    path.display().to_string(),
                    "load_error_fallback",
                    Some(err.to_string()),
                    true,
                ));
                current.clone()
            }
        }
    };

    let mut normalized = manifest.clone();
    let mut mismatch_details = Vec::new();
    if normalized.schema_version != current.schema_version {
        mismatch_details.push(format!(
            "schema_version {} -> {}",
            normalized.schema_version, current.schema_version
        ));
        normalized.schema_version = current.schema_version;
    }
    for (surface, expected) in &current.surfaces {
        let actual = normalized
            .surfaces
            .get(surface)
            .copied()
            .unwrap_or_default();
        if actual != *expected {
            mismatch_details.push(format!("{surface} {actual} -> {expected}"));
        }
        normalized.surfaces.insert(surface.clone(), *expected);
    }

    if mismatch_details.is_empty() {
        events.push(startup_event(
            "persistence_manifest",
            path.display().to_string(),
            "loaded",
            Some("loaded current persistence manifest".into()),
            false,
        ));
    } else {
        events.push(startup_event(
            "persistence_manifest",
            path.display().to_string(),
            "version_mismatch",
            Some(format!(
                "normalized manifest to current versions: {}",
                mismatch_details.join(", ")
            )),
            true,
        ));
    }

    let _ = save_persistence_manifest(root, &normalized);
    normalized
}

fn load_belief_engine_with_validation(
    store: &BeliefStore,
    path: &Path,
    events: &mut Vec<StartupValidationEvent>,
) -> BeliefEngine {
    if !path.exists() {
        events.push(startup_event(
            "belief",
            path.display().to_string(),
            "missing_fallback",
            Some("belief store missing; started with empty belief engine".into()),
            true,
        ));
        return BeliefEngine::new();
    }
    if std::fs::metadata(path)
        .map(|meta| meta.len())
        .unwrap_or_default()
        == 0
    {
        events.push(startup_event(
            "belief",
            path.display().to_string(),
            "empty_fallback",
            Some("belief store file was empty; started with empty belief engine".into()),
            true,
        ));
        return BeliefEngine::new();
    }
    match store.load() {
        Ok(engine) => {
            events.push(startup_event(
                "belief",
                path.display().to_string(),
                "loaded",
                Some(format!("loaded {} beliefs", engine.beliefs.len())),
                false,
            ));
            engine
        }
        Err(err) => {
            events.push(startup_event(
                "belief",
                path.display().to_string(),
                "load_error_fallback",
                Some(err.to_string()),
                true,
            ));
            BeliefEngine::new()
        }
    }
}

fn load_concept_engine_with_validation(
    store: &ConceptStore,
    path: &Path,
    events: &mut Vec<StartupValidationEvent>,
) -> ConceptEngine {
    if !path.exists() {
        events.push(startup_event(
            "concept",
            path.display().to_string(),
            "missing_fallback",
            Some("concept store missing; started with empty concept engine".into()),
            false,
        ));
        return ConceptEngine::new();
    }
    if std::fs::metadata(path)
        .map(|meta| meta.len())
        .unwrap_or_default()
        == 0
    {
        events.push(startup_event(
            "concept",
            path.display().to_string(),
            "empty_fallback",
            Some("concept store file was empty; started with empty concept engine".into()),
            false,
        ));
        return ConceptEngine::new();
    }
    match store.load() {
        Ok(engine) => {
            events.push(startup_event(
                "concept",
                path.display().to_string(),
                "loaded",
                Some(format!("loaded {} concepts", engine.concepts.len())),
                false,
            ));
            engine
        }
        Err(err) => {
            events.push(startup_event(
                "concept",
                path.display().to_string(),
                "load_error_fallback",
                Some(err.to_string()),
                true,
            ));
            ConceptEngine::new()
        }
    }
}

fn load_causal_engine_with_validation(
    store: &CausalStore,
    path: &Path,
    events: &mut Vec<StartupValidationEvent>,
) -> CausalEngine {
    if !path.exists() {
        events.push(startup_event(
            "causal",
            path.display().to_string(),
            "missing_fallback",
            Some("causal store missing; started with empty causal engine".into()),
            true,
        ));
        return CausalEngine::new();
    }
    match store.load() {
        Ok(engine) => {
            events.push(startup_event(
                "causal",
                path.display().to_string(),
                "loaded",
                Some(format!("loaded {} causal patterns", engine.patterns.len())),
                false,
            ));
            engine
        }
        Err(err) => {
            events.push(startup_event(
                "causal",
                path.display().to_string(),
                "load_error_fallback",
                Some(err.to_string()),
                true,
            ));
            CausalEngine::new()
        }
    }
}

fn load_policy_engine_with_validation(
    store: &PolicyStore,
    path: &Path,
    events: &mut Vec<StartupValidationEvent>,
) -> PolicyEngine {
    if !path.exists() {
        events.push(startup_event(
            "policy",
            path.display().to_string(),
            "missing_fallback",
            Some("policy store missing; started with empty policy engine".into()),
            true,
        ));
        return PolicyEngine::new();
    }
    match store.load() {
        Ok(engine) => {
            events.push(startup_event(
                "policy",
                path.display().to_string(),
                "loaded",
                Some(format!("loaded {} policy hints", engine.hints.len())),
                false,
            ));
            engine
        }
        Err(err) => {
            events.push(startup_event(
                "policy",
                path.display().to_string(),
                "load_error_fallback",
                Some(err.to_string()),
                true,
            ));
            PolicyEngine::new()
        }
    }
}

fn load_maintenance_trends_with_validation(
    root: &Path,
    events: &mut Vec<StartupValidationEvent>,
) -> Vec<background_brain::MaintenanceTrendSnapshot> {
    let path = maintenance_trends_path(root);
    let Ok(contents) = std::fs::read_to_string(&path) else {
        events.push(startup_event(
            "maintenance_trends",
            path.display().to_string(),
            "missing_fallback",
            Some("maintenance trend history missing; started with empty trend history".into()),
            true,
        ));
        return Vec::new();
    };

    if contents.trim().is_empty() {
        events.push(startup_event(
            "maintenance_trends",
            path.display().to_string(),
            "empty_fallback",
            Some("maintenance trend history file was empty".into()),
            true,
        ));
        return Vec::new();
    }

    match serde_json::from_str::<Vec<background_brain::MaintenanceTrendSnapshot>>(&contents) {
        Ok(history) => {
            events.push(startup_event(
                "maintenance_trends",
                path.display().to_string(),
                "loaded",
                Some(format!(
                    "loaded {} maintenance trend snapshots",
                    history.len()
                )),
                false,
            ));
            history
        }
        Err(err) => {
            events.push(startup_event(
                "maintenance_trends",
                path.display().to_string(),
                "load_error_fallback",
                Some(err.to_string()),
                true,
            ));
            Vec::new()
        }
    }
}

fn load_reflection_summaries_with_validation(
    root: &Path,
    events: &mut Vec<StartupValidationEvent>,
) -> Vec<background_brain::ReflectionSummary> {
    let path = reflection_summaries_path(root);
    let Ok(contents) = std::fs::read_to_string(&path) else {
        events.push(startup_event(
            "reflection_summaries",
            path.display().to_string(),
            "missing_fallback",
            Some(
                "reflection summary history missing; started with empty reflection history".into(),
            ),
            true,
        ));
        return Vec::new();
    };

    if contents.trim().is_empty() {
        events.push(startup_event(
            "reflection_summaries",
            path.display().to_string(),
            "empty_fallback",
            Some("reflection summary history file was empty".into()),
            true,
        ));
        return Vec::new();
    }

    match serde_json::from_str::<Vec<background_brain::ReflectionSummary>>(&contents) {
        Ok(history) => {
            events.push(startup_event(
                "reflection_summaries",
                path.display().to_string(),
                "loaded",
                Some(format!("loaded {} reflection summaries", history.len())),
                false,
            ));
            history
        }
        Err(err) => {
            events.push(startup_event(
                "reflection_summaries",
                path.display().to_string(),
                "load_error_fallback",
                Some(err.to_string()),
                true,
            ));
            Vec::new()
        }
    }
}

fn save_maintenance_trends(
    root: &Path,
    trends: &[background_brain::MaintenanceTrendSnapshot],
) -> Result<()> {
    let path = maintenance_trends_path(root);
    let json = serde_json::to_string_pretty(trends)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn save_reflection_summaries(
    root: &Path,
    summaries: &[background_brain::ReflectionSummary],
) -> Result<()> {
    let path = reflection_summaries_path(root);
    let json = serde_json::to_string_pretty(summaries)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn build_recall_explanation_item(
    rank: usize,
    score: f32,
    rec: &Record,
    records: &HashMap<String, Record>,
    belief_eng: &BeliefEngine,
    concept_eng: &ConceptEngine,
    causal_eng: &CausalEngine,
    policy_eng: &PolicyEngine,
    correction_log: &[CorrectionLogEntry],
    reflection_summaries: &[background_brain::ReflectionSummary],
    trace: Option<&recall::RecallScoreTrace>,
) -> RecallExplanationItem {
    let belief = belief_eng
        .belief_for_record(&rec.id)
        .map(|belief| RecallBeliefExplanation {
            id: belief.id.clone(),
            state: format!("{:?}", belief.state).to_lowercase(),
            confidence: belief.confidence,
            support_mass: belief.support_mass,
            conflict_mass: belief.conflict_mass,
            stability: belief.stability,
            volatility: belief.volatility,
            has_unresolved_evidence: matches!(
                belief.state,
                crate::belief::BeliefState::Unresolved | crate::belief::BeliefState::Empty
            ) || belief.volatility >= 0.20
                || belief.conflict_mass > belief.support_mass,
        });
    let belief_id = belief.as_ref().map(|b| b.id.clone());
    let has_unresolved_evidence = belief
        .as_ref()
        .map(|belief| belief.has_unresolved_evidence)
        .unwrap_or(false);
    let contradiction_dependency = belief
        .as_ref()
        .map(|belief| belief.conflict_mass > 0.0 || belief.has_unresolved_evidence)
        .unwrap_or(false);

    let concepts = concept_eng
        .concepts
        .values()
        .filter(|concept| {
            concept.record_ids.contains(&rec.id)
                || belief_id
                    .as_ref()
                    .is_some_and(|bid| concept.belief_ids.contains(bid))
        })
        .map(|concept| RecallConceptExplanation {
            id: concept.id.clone(),
            key: concept.key.clone(),
            state: format!("{:?}", concept.state).to_lowercase(),
            confidence: concept.confidence,
        })
        .collect();

    let causal_patterns = causal_eng
        .patterns
        .values()
        .filter(|pattern| {
            pattern.cause_record_ids.contains(&rec.id)
                || pattern.effect_record_ids.contains(&rec.id)
                || belief_id.as_ref().is_some_and(|bid| {
                    pattern.cause_belief_ids.contains(bid)
                        || pattern.effect_belief_ids.contains(bid)
                })
        })
        .map(|pattern| RecallCausalExplanation {
            id: pattern.id.clone(),
            key: pattern.key.clone(),
            state: format!("{:?}", pattern.state).to_lowercase(),
            causal_strength: pattern.causal_strength,
            invalidation_reason: pattern.invalidation_reason.clone(),
            invalidated_at: pattern.invalidated_at,
            corrections: correction_log
                .iter()
                .filter(|entry| entry.matches_target("causal_pattern", &pattern.id))
                .cloned()
                .collect(),
        })
        .collect();

    let policy_hints: Vec<RecallPolicyExplanation> = policy_eng
        .hints
        .values()
        .filter(|hint| {
            hint.supporting_record_ids.contains(&rec.id)
                || belief_id
                    .as_ref()
                    .is_some_and(|bid| hint.trigger_belief_ids.contains(bid))
        })
        .map(|hint| RecallPolicyExplanation {
            id: hint.id.clone(),
            key: hint.key.clone(),
            state: format!("{:?}", hint.state).to_lowercase(),
            action_kind: format!("{:?}", hint.action_kind).to_lowercase(),
            policy_strength: hint.policy_strength,
        })
        .collect();
    let honesty_note = if has_unresolved_evidence {
        if !policy_hints.is_empty() {
            Some("This recommendation depends on unresolved evidence.".into())
        } else {
            Some("This memory is linked to unstable or conflicting evidence.".into())
        }
    } else {
        None
    };

    let because_record_id = rec.caused_by_id.clone();
    let because_preview = because_record_id
        .as_ref()
        .and_then(|parent_id| records.get(parent_id))
        .map(|parent| preview_text(&parent.content, 120));
    let salience_explanation = if rec.salience >= 0.70 {
        Some(match rec.metadata.get(RECORD_SALIENCE_REASON_KEY) {
            Some(reason) => format!("High-significance memory due to {}.", reason),
            None => "High-significance memory due to elevated salience.".into(),
        })
    } else if rec.salience > 0.0 {
        Some("This memory carries non-zero significance weighting.".into())
    } else {
        None
    };
    let reflection_references = reflection_summaries
        .iter()
        .flat_map(|summary| summary.findings.iter())
        .filter(|finding| {
            finding.related_ids.iter().any(|id| id == &rec.id) || finding.namespace == rec.namespace
        })
        .map(|finding| finding.title.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .take(4)
        .collect::<Vec<_>>();
    let answer_support = HonestAnswerSupport {
        significance_phrase: salience_explanation.clone(),
        uncertainty_phrase: honesty_note.clone(),
        contradiction_phrase: if contradiction_dependency {
            Some("This answer should acknowledge conflicting or unresolved evidence.".into())
        } else {
            None
        },
        reflection_phrase: if reflection_references.is_empty() {
            None
        } else {
            Some(format!(
                "Recent reflection findings touching this area: {}.",
                reflection_references.join(", ")
            ))
        },
        recommended_framing: if has_unresolved_evidence {
            "State the useful evidence, then explicitly note uncertainty or conflict.".into()
        } else if rec.salience >= 0.70 {
            "State the answer confidently and note that this memory is high-significance.".into()
        } else {
            "State the answer directly without anthropomorphic language.".into()
        },
    };

    let trace = trace.cloned().unwrap_or_default();
    let signal = |signal: Option<recall::SignalTrace>| {
        signal.map(|signal| RecallSignalScore {
            raw_score: signal.raw_score,
            rank: signal.rank,
            rrf_share: signal.rrf_share,
        })
    };

    RecallExplanationItem {
        rank,
        record_id: rec.id.clone(),
        score,
        namespace: rec.namespace.clone(),
        salience: rec.salience,
        salience_reason: rec.metadata.get(RECORD_SALIENCE_REASON_KEY).cloned(),
        salience_explanation,
        content_preview: preview_text(&rec.content, 160),
        because_record_id,
        because_preview,
        belief,
        has_unresolved_evidence,
        honesty_note,
        contradiction_dependency,
        reflection_references,
        answer_support,
        concepts,
        causal_patterns,
        policy_hints,
        trace: RecallTraceScore {
            sdr: signal(trace.sdr),
            ngram: signal(trace.ngram),
            tags: signal(trace.tags),
            embedding: signal(trace.embedding),
            rrf_score: trace.rrf_score,
            graph_score: trace.graph_score,
            causal_score: trace.causal_score,
            pre_trust_score: trace.pre_trust_score,
            trust_multiplier: trace.trust_multiplier,
            pre_rerank_score: trace.pre_rerank_score,
            rerank_delta: score - trace.pre_rerank_score,
            final_score: score,
        },
    }
}

fn build_provenance_chain(item: &RecallExplanationItem, build_latency_ms: f64) -> ProvenanceChain {
    let mut steps = Vec::new();
    steps.push(format!(
        "record {} in namespace {}",
        item.record_id, item.namespace
    ));

    if let Some(because_id) = &item.because_record_id {
        if let Some(preview) = &item.because_preview {
            steps.push(format!(
                "caused_by {} from \"{}\"",
                because_id,
                preview_text(preview, 80)
            ));
        } else {
            steps.push(format!("caused_by {}", because_id));
        }
    }

    if let Some(belief) = &item.belief {
        steps.push(format!(
            "belief {} is {} at confidence {:.2}",
            belief.id, belief.state, belief.confidence
        ));
        if belief.has_unresolved_evidence {
            steps.push(format!(
                "belief {} carries unresolved evidence (volatility {:.2}, conflict {:.2}, support {:.2})",
                belief.id, belief.volatility, belief.conflict_mass, belief.support_mass
            ));
        }
    }

    for concept in &item.concepts {
        steps.push(format!(
            "concept {} ({}) is {} at confidence {:.2}",
            concept.id, concept.key, concept.state, concept.confidence
        ));
    }

    for pattern in &item.causal_patterns {
        let mut step = format!(
            "causal pattern {} ({}) is {} with strength {:.2}",
            pattern.id, pattern.key, pattern.state, pattern.causal_strength
        );
        if let Some(reason) = &pattern.invalidation_reason {
            step.push_str(&format!("; invalidated because {}", reason));
        }
        if let Some(last) = pattern.corrections.last() {
            step.push_str(&format!(
                "; last correction {} at {}",
                last.operation, last.time_iso
            ));
        }
        steps.push(step);
    }

    for hint in &item.policy_hints {
        steps.push(format!(
            "policy hint {} ({}) is {} as {} with strength {:.2}",
            hint.id, hint.key, hint.state, hint.action_kind, hint.policy_strength
        ));
    }

    let mut narrative_parts = vec![format!(
        "Record {} surfaced in namespace {}",
        item.record_id, item.namespace
    )];
    if let Some(because_id) = &item.because_record_id {
        narrative_parts.push(format!("because it points to source record {}", because_id));
    }
    if let Some(belief) = &item.belief {
        narrative_parts.push(format!(
            "it maps to belief {} ({}, confidence {:.2})",
            belief.id, belief.state, belief.confidence
        ));
        if belief.has_unresolved_evidence {
            narrative_parts.push(format!(
                "that belief remains epistemically unstable (volatility {:.2}, conflict {:.2}, support {:.2})",
                belief.volatility, belief.conflict_mass, belief.support_mass
            ));
        }
    }
    if !item.concepts.is_empty() {
        let summary = item
            .concepts
            .iter()
            .map(|concept| concept.key.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        narrative_parts.push(format!("it participates in concepts [{}]", summary));
    }
    if !item.causal_patterns.is_empty() {
        let summary = item
            .causal_patterns
            .iter()
            .map(|pattern| {
                if let Some(reason) = &pattern.invalidation_reason {
                    format!("{}[invalidated:{}]", pattern.key, reason)
                } else {
                    pattern.key.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        narrative_parts.push(format!("it appears in causal patterns [{}]", summary));
    }
    if !item.policy_hints.is_empty() {
        let summary = item
            .policy_hints
            .iter()
            .map(|hint| format!("{}:{}", hint.action_kind, hint.key))
            .collect::<Vec<_>>()
            .join(", ");
        narrative_parts.push(format!("it supports policy hints [{}]", summary));
    }
    if let Some(salience) = &item.salience_explanation {
        steps.push(salience.clone());
        narrative_parts.push(salience.clone());
    }
    if !item.reflection_references.is_empty() {
        let summary = item.reflection_references.join(", ");
        steps.push(format!("related reflection findings [{}]", summary));
        narrative_parts.push(format!(
            "recent reflection findings reference it [{}]",
            summary
        ));
    }
    if let Some(note) = &item.honesty_note {
        steps.push(note.clone());
        narrative_parts.push(note.clone());
    }

    ProvenanceChain {
        record_id: item.record_id.clone(),
        namespace: item.namespace.clone(),
        content_preview: item.content_preview.clone(),
        build_latency_ms,
        because_record_id: item.because_record_id.clone(),
        because_preview: item.because_preview.clone(),
        belief: item.belief.clone(),
        concepts: item.concepts.clone(),
        causal_patterns: item.causal_patterns.clone(),
        policy_hints: item.policy_hints.clone(),
        steps,
        narrative: format!("{}.", narrative_parts.join("; ")),
    }
}

fn normalize_analytics_term(term: &str) -> String {
    relation::normalize_relation_text(term).trim().to_string()
}

fn collect_record_signature_terms(
    record_ids: &[String],
    records: &HashMap<String, Record>,
) -> Vec<String> {
    let mut terms = HashSet::new();
    for record_id in record_ids {
        let Some(record) = records.get(record_id) else {
            continue;
        };
        for tag in &record.tags {
            let normalized = normalize_analytics_term(tag);
            if !normalized.is_empty() {
                terms.insert(normalized);
            }
        }
        let normalized_content_type = normalize_analytics_term(&record.content_type);
        if !normalized_content_type.is_empty() {
            terms.insert(normalized_content_type);
        }
        let normalized_semantic_type = normalize_analytics_term(&record.semantic_type);
        if !normalized_semantic_type.is_empty() {
            terms.insert(normalized_semantic_type);
        }
    }
    let mut terms: Vec<String> = terms.into_iter().collect();
    terms.sort();
    terms.truncate(4);
    terms
}

fn canonical_causal_signature(
    pattern: &crate::causal::CausalPattern,
    records: &HashMap<String, Record>,
) -> String {
    let cause_terms = collect_record_signature_terms(&pattern.cause_record_ids, records);
    let effect_terms = collect_record_signature_terms(&pattern.effect_record_ids, records);
    format!("{}=>{}", cause_terms.join("+"), effect_terms.join("+"))
}

fn canonical_concept_signature(concept: &crate::concept::ConceptCandidate) -> String {
    let mut terms: Vec<String> = concept
        .core_terms
        .iter()
        .chain(concept.tags.iter())
        .map(|term| normalize_analytics_term(term))
        .filter(|term| !term.is_empty())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    terms.sort();
    terms.truncate(4);
    format!("{}:{}", concept.semantic_type, terms.join("+"))
}

fn jaccard_similarity(left: &HashSet<&str>, right: &HashSet<&str>) -> f32 {
    let union = left.union(right).count();
    if union == 0 {
        return 0.0;
    }
    left.intersection(right).count() as f32 / union as f32
}

pub(crate) fn apply_cross_namespace_dimension_flags(
    options: &mut CrossNamespaceDigestOptions,
    include_dimensions: Option<&[&str]>,
) {
    let Some(dimensions) = include_dimensions else {
        return;
    };

    options.include_concepts = false;
    options.include_tags = false;
    options.include_structural = false;
    options.include_causal = false;
    options.include_belief_states = false;
    options.include_corrections = false;

    for dimension in dimensions {
        match dimension.trim().to_ascii_lowercase().as_str() {
            "concepts" => options.include_concepts = true,
            "tags" => options.include_tags = true,
            "structural" => options.include_structural = true,
            "causal" => options.include_causal = true,
            "beliefs" | "belief_state" | "belief_states" => options.include_belief_states = true,
            "corrections" | "correction_density" => options.include_corrections = true,
            _ => {}
        }
    }
}

#[cfg(feature = "python")]
fn recall_explanation_item_to_py(
    py: Python<'_>,
    item: &RecallExplanationItem,
) -> PyResult<PyObject> {
    let signal_to_py = |signal: &RecallSignalScore| -> PyResult<PyObject> {
        let signal_dict = pyo3::types::PyDict::new_bound(py);
        signal_dict.set_item("raw_score", signal.raw_score)?;
        signal_dict.set_item("rank", signal.rank)?;
        signal_dict.set_item("rrf_share", signal.rrf_share)?;
        Ok(signal_dict.unbind().into_any())
    };

    let item_dict = pyo3::types::PyDict::new_bound(py);
    item_dict.set_item("rank", item.rank)?;
    item_dict.set_item("record_id", &item.record_id)?;
    item_dict.set_item("score", item.score)?;
    item_dict.set_item("namespace", &item.namespace)?;
    item_dict.set_item("salience", item.salience)?;
    item_dict.set_item("salience_reason", &item.salience_reason)?;
    item_dict.set_item("salience_explanation", &item.salience_explanation)?;
    item_dict.set_item("content_preview", &item.content_preview)?;
    item_dict.set_item("because_record_id", &item.because_record_id)?;
    item_dict.set_item("because_preview", &item.because_preview)?;
    item_dict.set_item("has_unresolved_evidence", item.has_unresolved_evidence)?;
    item_dict.set_item("honesty_note", &item.honesty_note)?;
    item_dict.set_item("contradiction_dependency", item.contradiction_dependency)?;
    item_dict.set_item("reflection_references", &item.reflection_references)?;
    let answer_support = pyo3::types::PyDict::new_bound(py);
    answer_support.set_item(
        "significance_phrase",
        &item.answer_support.significance_phrase,
    )?;
    answer_support.set_item(
        "uncertainty_phrase",
        &item.answer_support.uncertainty_phrase,
    )?;
    answer_support.set_item(
        "contradiction_phrase",
        &item.answer_support.contradiction_phrase,
    )?;
    answer_support.set_item("reflection_phrase", &item.answer_support.reflection_phrase)?;
    answer_support.set_item(
        "recommended_framing",
        &item.answer_support.recommended_framing,
    )?;
    item_dict.set_item("answer_support", answer_support)?;

    if let Some(belief) = &item.belief {
        let belief_dict = pyo3::types::PyDict::new_bound(py);
        belief_dict.set_item("id", &belief.id)?;
        belief_dict.set_item("state", &belief.state)?;
        belief_dict.set_item("confidence", belief.confidence)?;
        belief_dict.set_item("support_mass", belief.support_mass)?;
        belief_dict.set_item("conflict_mass", belief.conflict_mass)?;
        belief_dict.set_item("stability", belief.stability)?;
        belief_dict.set_item("volatility", belief.volatility)?;
        belief_dict.set_item("has_unresolved_evidence", belief.has_unresolved_evidence)?;
        item_dict.set_item("belief", belief_dict)?;
    } else {
        item_dict.set_item("belief", py.None())?;
    }

    let concepts = pyo3::types::PyList::empty_bound(py);
    for concept in &item.concepts {
        let concept_dict = pyo3::types::PyDict::new_bound(py);
        concept_dict.set_item("id", &concept.id)?;
        concept_dict.set_item("key", &concept.key)?;
        concept_dict.set_item("state", &concept.state)?;
        concept_dict.set_item("confidence", concept.confidence)?;
        concepts.append(concept_dict)?;
    }
    item_dict.set_item("concepts", concepts)?;

    let causal_patterns = pyo3::types::PyList::empty_bound(py);
    for pattern in &item.causal_patterns {
        let pattern_dict = pyo3::types::PyDict::new_bound(py);
        pattern_dict.set_item("id", &pattern.id)?;
        pattern_dict.set_item("key", &pattern.key)?;
        pattern_dict.set_item("state", &pattern.state)?;
        pattern_dict.set_item("causal_strength", pattern.causal_strength)?;
        match &pattern.invalidation_reason {
            Some(reason) => pattern_dict.set_item("invalidation_reason", reason)?,
            None => pattern_dict.set_item("invalidation_reason", py.None())?,
        }
        match pattern.invalidated_at {
            Some(ts) => pattern_dict.set_item("invalidated_at", ts)?,
            None => pattern_dict.set_item("invalidated_at", py.None())?,
        }
        let corrections = pyo3::types::PyList::empty_bound(py);
        for correction in &pattern.corrections {
            let correction_dict = pyo3::types::PyDict::new_bound(py);
            correction_dict.set_item("timestamp", correction.timestamp)?;
            correction_dict.set_item("time_iso", &correction.time_iso)?;
            correction_dict.set_item("target_kind", &correction.target_kind)?;
            correction_dict.set_item("target_id", &correction.target_id)?;
            correction_dict.set_item("operation", &correction.operation)?;
            correction_dict.set_item("reason", &correction.reason)?;
            correction_dict.set_item("session_id", &correction.session_id)?;
            corrections.append(correction_dict)?;
        }
        pattern_dict.set_item("corrections", corrections)?;
        causal_patterns.append(pattern_dict)?;
    }
    item_dict.set_item("causal_patterns", causal_patterns)?;

    let policy_hints = pyo3::types::PyList::empty_bound(py);
    for hint in &item.policy_hints {
        let hint_dict = pyo3::types::PyDict::new_bound(py);
        hint_dict.set_item("id", &hint.id)?;
        hint_dict.set_item("key", &hint.key)?;
        hint_dict.set_item("state", &hint.state)?;
        hint_dict.set_item("action_kind", &hint.action_kind)?;
        hint_dict.set_item("policy_strength", hint.policy_strength)?;
        policy_hints.append(hint_dict)?;
    }
    item_dict.set_item("policy_hints", policy_hints)?;

    let trace_dict = pyo3::types::PyDict::new_bound(py);
    match &item.trace.sdr {
        Some(signal) => trace_dict.set_item("sdr", signal_to_py(signal)?)?,
        None => trace_dict.set_item("sdr", py.None())?,
    }
    match &item.trace.ngram {
        Some(signal) => trace_dict.set_item("ngram", signal_to_py(signal)?)?,
        None => trace_dict.set_item("ngram", py.None())?,
    }
    match &item.trace.tags {
        Some(signal) => trace_dict.set_item("tags", signal_to_py(signal)?)?,
        None => trace_dict.set_item("tags", py.None())?,
    }
    match &item.trace.embedding {
        Some(signal) => trace_dict.set_item("embedding", signal_to_py(signal)?)?,
        None => trace_dict.set_item("embedding", py.None())?,
    }
    trace_dict.set_item("rrf_score", item.trace.rrf_score)?;
    trace_dict.set_item("graph_score", item.trace.graph_score)?;
    trace_dict.set_item("causal_score", item.trace.causal_score)?;
    trace_dict.set_item("pre_trust_score", item.trace.pre_trust_score)?;
    trace_dict.set_item("trust_multiplier", item.trace.trust_multiplier)?;
    trace_dict.set_item("pre_rerank_score", item.trace.pre_rerank_score)?;
    trace_dict.set_item("rerank_delta", item.trace.rerank_delta)?;
    trace_dict.set_item("final_score", item.trace.final_score)?;
    item_dict.set_item("trace", trace_dict)?;

    Ok(item_dict.unbind().into_any())
}

#[cfg(feature = "python")]
fn recall_explanation_to_py(py: Python<'_>, explanation: &RecallExplanation) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("query", &explanation.query)?;
    dict.set_item("top_k", explanation.top_k)?;
    dict.set_item("result_count", explanation.result_count)?;
    dict.set_item("latency_ms", explanation.latency_ms)?;
    dict.set_item("belief_rerank_mode", &explanation.belief_rerank_mode)?;
    dict.set_item("concept_surface_mode", &explanation.concept_surface_mode)?;
    dict.set_item("causal_rerank_mode", &explanation.causal_rerank_mode)?;
    dict.set_item("policy_rerank_mode", &explanation.policy_rerank_mode)?;

    let py_items = pyo3::types::PyList::empty_bound(py);
    for item in &explanation.items {
        py_items.append(recall_explanation_item_to_py(py, item)?)?;
    }
    dict.set_item("items", py_items)?;
    Ok(dict.unbind().into_any())
}

#[cfg(feature = "python")]
fn cross_namespace_digest_to_py(
    py: Python<'_>,
    digest: &CrossNamespaceDigest,
) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("namespace_count", digest.namespace_count)?;
    dict.set_item("latency_ms", digest.latency_ms)?;
    dict.set_item("compact_summary", digest.compact_summary)?;
    dict.set_item("included_dimensions", &digest.included_dimensions)?;

    let py_namespaces = pyo3::types::PyList::empty_bound(py);
    for namespace in &digest.namespaces {
        let ns_dict = pyo3::types::PyDict::new_bound(py);
        ns_dict.set_item("namespace", &namespace.namespace)?;
        ns_dict.set_item("record_count", namespace.record_count)?;
        ns_dict.set_item("concept_count", namespace.concept_count)?;
        ns_dict.set_item("stable_concept_count", namespace.stable_concept_count)?;
        ns_dict.set_item("concept_signatures", &namespace.concept_signatures)?;
        ns_dict.set_item("tags", &namespace.tags)?;
        ns_dict.set_item(
            "structural_relation_types",
            &namespace.structural_relation_types,
        )?;
        ns_dict.set_item("causal_signatures", &namespace.causal_signatures)?;
        match &namespace.belief_state_summary {
            Some(summary) => {
                let summary_dict = pyo3::types::PyDict::new_bound(py);
                summary_dict.set_item("resolved", summary.resolved)?;
                summary_dict.set_item("unresolved", summary.unresolved)?;
                summary_dict.set_item("singleton", summary.singleton)?;
                summary_dict.set_item("empty", summary.empty)?;
                summary_dict.set_item("high_volatility_count", summary.high_volatility_count)?;
                summary_dict.set_item("avg_volatility", summary.avg_volatility)?;
                ns_dict.set_item("belief_state_summary", summary_dict)?;
            }
            None => ns_dict.set_item("belief_state_summary", py.None())?,
        }
        ns_dict.set_item("correction_count", namespace.correction_count)?;
        ns_dict.set_item("correction_density", namespace.correction_density)?;

        let top_concepts = pyo3::types::PyList::empty_bound(py);
        for concept in &namespace.top_concepts {
            let concept_dict = pyo3::types::PyDict::new_bound(py);
            concept_dict.set_item("concept_id", &concept.concept_id)?;
            concept_dict.set_item("key", &concept.key)?;
            concept_dict.set_item("confidence", concept.confidence)?;
            concept_dict.set_item("state", &concept.state)?;
            concept_dict.set_item("record_count", concept.record_count)?;
            concept_dict.set_item("belief_count", concept.belief_count)?;
            top_concepts.append(concept_dict)?;
        }
        ns_dict.set_item("top_concepts", top_concepts)?;
        py_namespaces.append(ns_dict)?;
    }
    dict.set_item("namespaces", py_namespaces)?;

    let py_pairs = pyo3::types::PyList::empty_bound(py);
    for pair in &digest.pairs {
        let pair_dict = pyo3::types::PyDict::new_bound(py);
        pair_dict.set_item("namespace_a", &pair.namespace_a)?;
        pair_dict.set_item("namespace_b", &pair.namespace_b)?;
        pair_dict.set_item("shared_concept_signatures", &pair.shared_concept_signatures)?;
        pair_dict.set_item(
            "concept_signature_similarity",
            pair.concept_signature_similarity,
        )?;
        pair_dict.set_item("shared_tags", &pair.shared_tags)?;
        pair_dict.set_item("tag_jaccard", pair.tag_jaccard)?;
        pair_dict.set_item(
            "shared_structural_relation_types",
            &pair.shared_structural_relation_types,
        )?;
        pair_dict.set_item("structural_similarity", pair.structural_similarity)?;
        pair_dict.set_item("shared_causal_signatures", &pair.shared_causal_signatures)?;
        pair_dict.set_item(
            "causal_signature_similarity",
            pair.causal_signature_similarity,
        )?;
        py_pairs.append(pair_dict)?;
    }
    dict.set_item("pairs", py_pairs)?;
    Ok(dict.unbind().into_any())
}

#[cfg(feature = "python")]
fn provenance_chain_to_py(py: Python<'_>, chain: &ProvenanceChain) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("record_id", &chain.record_id)?;
    dict.set_item("namespace", &chain.namespace)?;
    dict.set_item("content_preview", &chain.content_preview)?;
    dict.set_item("build_latency_ms", chain.build_latency_ms)?;
    dict.set_item("because_record_id", &chain.because_record_id)?;
    dict.set_item("because_preview", &chain.because_preview)?;
    dict.set_item("narrative", &chain.narrative)?;

    let steps = pyo3::types::PyList::empty_bound(py);
    for step in &chain.steps {
        steps.append(step)?;
    }
    dict.set_item("steps", steps)?;

    let item = RecallExplanationItem {
        rank: 1,
        record_id: chain.record_id.clone(),
        score: 0.0,
        namespace: chain.namespace.clone(),
        salience: 0.0,
        salience_reason: None,
        salience_explanation: None,
        content_preview: chain.content_preview.clone(),
        because_record_id: chain.because_record_id.clone(),
        because_preview: chain.because_preview.clone(),
        belief: chain.belief.clone(),
        has_unresolved_evidence: chain
            .belief
            .as_ref()
            .map(|belief| belief.has_unresolved_evidence)
            .unwrap_or(false),
        honesty_note: chain.belief.as_ref().and_then(|belief| {
            if belief.has_unresolved_evidence {
                if chain.policy_hints.is_empty() {
                    Some("This memory is linked to unstable or conflicting evidence.".into())
                } else {
                    Some("This recommendation depends on unresolved evidence.".into())
                }
            } else {
                None
            }
        }),
        contradiction_dependency: chain
            .belief
            .as_ref()
            .map(|belief| belief.conflict_mass > 0.0 || belief.has_unresolved_evidence)
            .unwrap_or(false),
        reflection_references: Vec::new(),
        answer_support: HonestAnswerSupport {
            significance_phrase: None,
            uncertainty_phrase: chain.belief.as_ref().and_then(|belief| {
                if belief.has_unresolved_evidence {
                    if chain.policy_hints.is_empty() {
                        Some("This memory is linked to unstable or conflicting evidence.".into())
                    } else {
                        Some("This recommendation depends on unresolved evidence.".into())
                    }
                } else {
                    None
                }
            }),
            contradiction_phrase: chain.belief.as_ref().and_then(|belief| {
                if belief.conflict_mass > 0.0 || belief.has_unresolved_evidence {
                    Some(
                        "This answer should acknowledge conflicting or unresolved evidence.".into(),
                    )
                } else {
                    None
                }
            }),
            reflection_phrase: None,
            recommended_framing: if chain
                .belief
                .as_ref()
                .is_some_and(|belief| belief.has_unresolved_evidence)
            {
                "State the useful evidence, then explicitly note uncertainty or conflict.".into()
            } else {
                "State the answer directly without anthropomorphic language.".into()
            },
        },
        concepts: chain.concepts.clone(),
        causal_patterns: chain.causal_patterns.clone(),
        policy_hints: chain.policy_hints.clone(),
        trace: RecallTraceScore {
            sdr: None,
            ngram: None,
            tags: None,
            embedding: None,
            rrf_score: 0.0,
            graph_score: 0.0,
            causal_score: 0.0,
            pre_trust_score: 0.0,
            trust_multiplier: 0.0,
            pre_rerank_score: 0.0,
            rerank_delta: 0.0,
            final_score: 0.0,
        },
    };
    let item_py = recall_explanation_item_to_py(py, &item)?;
    dict.set_item("details", item_py)?;
    Ok(dict.unbind().into_any())
}

#[cfg(feature = "python")]
fn correction_log_entries_to_py(
    py: Python<'_>,
    entries: &[CorrectionLogEntry],
) -> PyResult<PyObject> {
    let results = pyo3::types::PyList::empty_bound(py);
    for entry in entries {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("timestamp", entry.timestamp)?;
        dict.set_item("time_iso", &entry.time_iso)?;
        dict.set_item("target_kind", &entry.target_kind)?;
        dict.set_item("target_id", &entry.target_id)?;
        dict.set_item("operation", &entry.operation)?;
        dict.set_item("reason", &entry.reason)?;
        dict.set_item("session_id", &entry.session_id)?;
        results.append(dict)?;
    }
    Ok(results.unbind().into_any())
}

#[cfg(feature = "python")]
fn correction_review_candidates_to_py(
    py: Python<'_>,
    entries: &[CorrectionReviewCandidate],
) -> PyResult<PyObject> {
    let results = pyo3::types::PyList::empty_bound(py);
    for entry in entries {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("timestamp", entry.timestamp)?;
        dict.set_item("time_iso", &entry.time_iso)?;
        dict.set_item("target_kind", &entry.target_kind)?;
        dict.set_item("target_id", &entry.target_id)?;
        dict.set_item("operation", &entry.operation)?;
        dict.set_item("reason", &entry.reason)?;
        dict.set_item("session_id", &entry.session_id)?;
        dict.set_item("namespace", &entry.namespace)?;
        dict.set_item("title", &entry.title)?;
        dict.set_item("repeat_count", entry.repeat_count)?;
        dict.set_item("dependent_causal_patterns", entry.dependent_causal_patterns)?;
        dict.set_item("dependent_policy_hints", entry.dependent_policy_hints)?;
        dict.set_item("downstream_impact", entry.downstream_impact)?;
        dict.set_item("priority_score", entry.priority_score)?;
        dict.set_item("severity", &entry.severity)?;
        results.append(dict)?;
    }
    Ok(results.unbind().into_any())
}

#[cfg(feature = "python")]
fn contradiction_review_candidates_to_py(
    py: Python<'_>,
    entries: &[ContradictionReviewCandidate],
) -> PyResult<PyObject> {
    let results = pyo3::types::PyList::empty_bound(py);
    for entry in entries {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("cluster_id", &entry.cluster_id)?;
        dict.set_item("namespace", &entry.namespace)?;
        dict.set_item("title", &entry.title)?;
        dict.set_item("belief_ids", &entry.belief_ids)?;
        dict.set_item("belief_keys", &entry.belief_keys)?;
        dict.set_item("record_ids", &entry.record_ids)?;
        dict.set_item("shared_tags", &entry.shared_tags)?;
        dict.set_item("unresolved_belief_count", entry.unresolved_belief_count)?;
        dict.set_item(
            "high_volatility_belief_count",
            entry.high_volatility_belief_count,
        )?;
        dict.set_item("dependent_causal_patterns", entry.dependent_causal_patterns)?;
        dict.set_item("dependent_policy_hints", entry.dependent_policy_hints)?;
        dict.set_item("downstream_impact", entry.downstream_impact)?;
        dict.set_item("total_conflict_mass", entry.total_conflict_mass)?;
        dict.set_item("avg_volatility", entry.avg_volatility)?;
        dict.set_item("avg_stability", entry.avg_stability)?;
        dict.set_item("priority_score", entry.priority_score)?;
        dict.set_item("severity", &entry.severity)?;
        results.append(dict)?;
    }
    Ok(results.unbind().into_any())
}

#[cfg(feature = "python")]
fn suggested_corrections_to_py(
    py: Python<'_>,
    entries: &[SuggestedCorrection],
) -> PyResult<PyObject> {
    let results = pyo3::types::PyList::empty_bound(py);
    for entry in entries {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("target_kind", &entry.target_kind)?;
        dict.set_item("target_id", &entry.target_id)?;
        dict.set_item("namespace", &entry.namespace)?;
        dict.set_item("reason_kind", &entry.reason_kind)?;
        dict.set_item("suggested_action", &entry.suggested_action)?;
        dict.set_item("reason_detail", &entry.reason_detail)?;
        dict.set_item("priority_score", entry.priority_score)?;
        dict.set_item("severity", &entry.severity)?;
        dict.set_item("supporting_record_id", &entry.supporting_record_id)?;
        match &entry.provenance {
            Some(chain) => dict.set_item("provenance", provenance_chain_to_py(py, chain)?)?,
            None => dict.set_item("provenance", py.None())?,
        }
        results.append(dict)?;
    }
    Ok(results.unbind().into_any())
}

#[cfg(feature = "python")]
fn suggested_corrections_report_to_py(
    py: Python<'_>,
    report: &SuggestedCorrectionsReport,
) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("scan_latency_ms", report.scan_latency_ms)?;
    dict.set_item("entries", suggested_corrections_to_py(py, &report.entries)?)?;
    Ok(dict.unbind().into_any())
}

#[cfg(feature = "python")]
fn namespace_governance_statuses_to_py(
    py: Python<'_>,
    entries: &[NamespaceGovernanceStatus],
) -> PyResult<PyObject> {
    let results = pyo3::types::PyList::empty_bound(py);
    for entry in entries {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("namespace", &entry.namespace)?;
        dict.set_item("record_count", entry.record_count)?;
        dict.set_item("belief_count", entry.belief_count)?;
        dict.set_item("correction_count", entry.correction_count)?;
        dict.set_item("correction_density", entry.correction_density)?;
        dict.set_item(
            "high_volatility_belief_count",
            entry.high_volatility_belief_count,
        )?;
        dict.set_item(
            "low_stability_belief_count",
            entry.low_stability_belief_count,
        )?;
        dict.set_item("instability_score", entry.instability_score)?;
        dict.set_item("instability_level", &entry.instability_level)?;
        dict.set_item(
            "policy_pressure_area_count",
            entry.policy_pressure_area_count,
        )?;
        dict.set_item(
            "suggested_correction_count",
            entry.suggested_correction_count,
        )?;
        dict.set_item("last_maintenance_cycle", &entry.last_maintenance_cycle)?;
        dict.set_item("latest_dominant_phase", &entry.latest_dominant_phase)?;
        results.append(dict)?;
    }
    Ok(results.unbind().into_any())
}

#[cfg(feature = "python")]
fn maintenance_trend_summary_to_py(
    py: Python<'_>,
    summary: &background_brain::MaintenanceTrendSummary,
) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("snapshot_count", summary.snapshot_count)?;
    dict.set_item("avg_belief_churn", summary.avg_belief_churn)?;
    dict.set_item(
        "avg_causal_rejection_rate",
        summary.avg_causal_rejection_rate,
    )?;
    dict.set_item(
        "avg_policy_suppression_rate",
        summary.avg_policy_suppression_rate,
    )?;
    dict.set_item("avg_cycle_time_ms", summary.avg_cycle_time_ms)?;
    dict.set_item("avg_correction_events", summary.avg_correction_events)?;
    dict.set_item(
        "total_corrections_in_window",
        summary.total_corrections_in_window,
    )?;
    dict.set_item("latest_dominant_phase", &summary.latest_dominant_phase)?;

    let recent = pyo3::types::PyList::empty_bound(py);
    for snapshot in &summary.recent {
        let snapshot_dict = pyo3::types::PyDict::new_bound(py);
        snapshot_dict.set_item("timestamp", &snapshot.timestamp)?;
        snapshot_dict.set_item("total_records", snapshot.total_records)?;
        snapshot_dict.set_item("records_archived", snapshot.records_archived)?;
        snapshot_dict.set_item("insights_found", snapshot.insights_found)?;
        snapshot_dict.set_item("volatile_records", snapshot.volatile_records)?;
        snapshot_dict.set_item("belief_churn", snapshot.belief_churn)?;
        snapshot_dict.set_item("causal_rejection_rate", snapshot.causal_rejection_rate)?;
        snapshot_dict.set_item("policy_suppression_rate", snapshot.policy_suppression_rate)?;
        snapshot_dict.set_item(
            "feedback_beliefs_touched",
            snapshot.feedback_beliefs_touched,
        )?;
        snapshot_dict.set_item(
            "feedback_net_confidence_delta",
            snapshot.feedback_net_confidence_delta,
        )?;
        snapshot_dict.set_item(
            "feedback_net_volatility_delta",
            snapshot.feedback_net_volatility_delta,
        )?;
        snapshot_dict.set_item("correction_events", snapshot.correction_events)?;
        snapshot_dict.set_item("cumulative_corrections", snapshot.cumulative_corrections)?;
        snapshot_dict.set_item("cycle_time_ms", snapshot.cycle_time_ms)?;
        snapshot_dict.set_item("dominant_phase", &snapshot.dominant_phase)?;
        recent.append(snapshot_dict)?;
    }
    dict.set_item("recent", recent)?;
    Ok(dict.unbind().into_any())
}

#[cfg(feature = "python")]
fn reflection_findings_to_py(
    py: Python<'_>,
    findings: &[background_brain::ReflectionFinding],
) -> PyResult<PyObject> {
    let results = pyo3::types::PyList::empty_bound(py);
    for finding in findings {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("kind", &finding.kind)?;
        dict.set_item("namespace", &finding.namespace)?;
        dict.set_item("title", &finding.title)?;
        dict.set_item("detail", &finding.detail)?;
        dict.set_item("related_ids", &finding.related_ids)?;
        dict.set_item("score", finding.score)?;
        dict.set_item("severity", &finding.severity)?;
        results.append(dict)?;
    }
    Ok(results.unbind().into_any())
}

#[cfg(feature = "python")]
fn reflection_digest_to_py(
    py: Python<'_>,
    digest: &background_brain::ReflectionDigest,
) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("summary_count", digest.summary_count)?;
    dict.set_item("total_findings", digest.total_findings)?;
    dict.set_item("high_severity_findings", digest.high_severity_findings)?;
    dict.set_item("latest_timestamp", &digest.latest_timestamp)?;
    dict.set_item("latest_dominant_phase", &digest.latest_dominant_phase)?;

    let kinds = pyo3::types::PyList::empty_bound(py);
    for kind in &digest.kinds {
        let item = pyo3::types::PyDict::new_bound(py);
        item.set_item("kind", &kind.kind)?;
        item.set_item("count", kind.count)?;
        item.set_item("high_severity_count", kind.high_severity_count)?;
        item.set_item("avg_score", kind.avg_score)?;
        kinds.append(item)?;
    }
    dict.set_item("kinds", kinds)?;
    dict.set_item("namespaces", &digest.namespaces)?;
    dict.set_item(
        "top_findings",
        reflection_findings_to_py(py, &digest.top_findings)?,
    )?;
    Ok(dict.unbind().into_any())
}

#[cfg(feature = "python")]
fn explainability_bundle_to_py(
    py: Python<'_>,
    bundle: &ExplainabilityBundle,
) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("record_id", &bundle.record_id)?;
    dict.set_item(
        "explanation",
        recall_explanation_item_to_py(py, &bundle.explanation)?,
    )?;
    dict.set_item(
        "provenance",
        provenance_chain_to_py(py, &bundle.provenance)?,
    )?;
    dict.set_item(
        "record_corrections",
        correction_log_entries_to_py(py, &bundle.record_corrections)?,
    )?;
    dict.set_item(
        "belief_corrections",
        correction_log_entries_to_py(py, &bundle.belief_corrections)?,
    )?;
    dict.set_item(
        "causal_corrections",
        correction_log_entries_to_py(py, &bundle.causal_corrections)?,
    )?;
    dict.set_item(
        "policy_corrections",
        correction_log_entries_to_py(py, &bundle.policy_corrections)?,
    )?;
    dict.set_item(
        "reflection_digest",
        reflection_digest_to_py(py, &bundle.reflection_digest)?,
    )?;
    dict.set_item(
        "related_reflection_findings",
        reflection_findings_to_py(py, &bundle.related_reflection_findings)?,
    )?;

    let instability = pyo3::types::PyDict::new_bound(py);
    instability.set_item("total_beliefs", bundle.belief_instability.total_beliefs)?;
    instability.set_item("resolved", bundle.belief_instability.resolved)?;
    instability.set_item("unresolved", bundle.belief_instability.unresolved)?;
    instability.set_item("singleton", bundle.belief_instability.singleton)?;
    instability.set_item("empty", bundle.belief_instability.empty)?;
    instability.set_item(
        "contradiction_cluster_count",
        bundle.belief_instability.contradiction_cluster_count,
    )?;
    instability.set_item(
        "high_volatility_count",
        bundle.belief_instability.high_volatility_count,
    )?;
    instability.set_item(
        "low_stability_count",
        bundle.belief_instability.low_stability_count,
    )?;
    instability.set_item("avg_volatility", bundle.belief_instability.avg_volatility)?;
    instability.set_item("avg_stability", bundle.belief_instability.avg_stability)?;
    let bands = pyo3::types::PyDict::new_bound(py);
    bands.set_item("low", bundle.belief_instability.volatility_bands.low)?;
    bands.set_item("medium", bundle.belief_instability.volatility_bands.medium)?;
    bands.set_item("high", bundle.belief_instability.volatility_bands.high)?;
    instability.set_item("volatility_bands", bands)?;
    dict.set_item("belief_instability", instability)?;

    dict.set_item(
        "maintenance_trends",
        maintenance_trend_summary_to_py(py, &bundle.maintenance_trends)?,
    )?;
    Ok(dict.unbind().into_any())
}

#[cfg(feature = "python")]
#[pymethods]
impl Aura {
    #[new]
    #[pyo3(signature = (path, password=None))]
    fn py_new(path: &str, password: Option<&str>) -> PyResult<Self> {
        Self::open_with_password(path, password)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[pyo3(name = "store", signature = (content, level=None, tags=None, pin=None, content_type=None, source_type=None, metadata=None, deduplicate=None, caused_by_id=None, channel=None, auto_promote=None, namespace=None, semantic_type=None))]
    fn py_store(
        &self,
        py: Python<'_>,
        content: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        pin: Option<bool>,
        content_type: Option<&str>,
        source_type: Option<&str>,
        metadata: Option<HashMap<String, String>>,
        deduplicate: Option<bool>,
        caused_by_id: Option<&str>,
        channel: Option<&str>,
        auto_promote: Option<bool>,
        namespace: Option<&str>,
        semantic_type: Option<&str>,
    ) -> PyResult<String> {
        let rec = py
            .allow_threads(|| {
                self.store_with_channel(
                    content,
                    level,
                    tags,
                    pin,
                    content_type,
                    source_type,
                    metadata,
                    deduplicate,
                    caused_by_id,
                    channel,
                    auto_promote,
                    namespace,
                    semantic_type,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "recall", signature = (query, token_budget=None, min_strength=None, expand_connections=None, session_id=None, namespace=None))]
    fn py_recall(
        &self,
        py: Python<'_>,
        query: &str,
        token_budget: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<String> {
        let ns_vec = extract_namespaces(namespace)?;
        let ns_refs: Option<Vec<&str>> = ns_vec
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        py.allow_threads(|| {
            self.recall(
                query,
                token_budget,
                min_strength,
                expand_connections,
                session_id,
                ns_slice,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "recall_structured", signature = (query, top_k=None, min_strength=None, expand_connections=None, session_id=None, namespace=None))]
    fn py_recall_structured(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<Vec<PyObject>> {
        let ns_vec = extract_namespaces(namespace)?;
        let ns_refs: Option<Vec<&str>> = ns_vec
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        let results = py
            .allow_threads(|| {
                self.recall_structured(
                    query,
                    top_k,
                    min_strength,
                    expand_connections,
                    session_id,
                    ns_slice,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("salience", rec.salience)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("source_type", &rec.source_type)?;
            // Include trust metadata
            if let Some(trust) = rec.metadata.get("trust_score") {
                dict.set_item("trust", trust)?;
            }
            if let Some(source) = rec.metadata.get("source") {
                dict.set_item("source", source)?;
            }
            if matches!(
                self.get_concept_surface_mode(),
                ConceptSurfaceMode::Inspect | ConceptSurfaceMode::Limited
            ) {
                let concepts = self.get_surfaced_concepts_for_record(&rec.id, Some(3));
                if !concepts.is_empty() {
                    let py_concepts = pyo3::types::PyList::empty_bound(py);
                    for concept in concepts {
                        let cdict = pyo3::types::PyDict::new_bound(py);
                        cdict.set_item("id", concept.id)?;
                        cdict.set_item("key", concept.key)?;
                        cdict.set_item("state", concept.state)?;
                        cdict.set_item("namespace", concept.namespace)?;
                        cdict.set_item("semantic_type", concept.semantic_type)?;
                        cdict.set_item("core_terms", concept.core_terms)?;
                        cdict.set_item("tags", concept.tags)?;
                        cdict.set_item("abstraction_score", concept.abstraction_score)?;
                        cdict.set_item("confidence", concept.confidence)?;
                        cdict.set_item("cluster_size", concept.cluster_size)?;
                        py_concepts.append(cdict)?;
                    }
                    dict.set_item("concepts", py_concepts)?;
                }
            }
            py_results.push(dict.unbind().into_any());
        }
        Ok(py_results)
    }

    /// Temporal recall: only consider records created at or before `timestamp` (Unix seconds).
    #[pyo3(name = "recall_at", signature = (query, timestamp, top_k=None, min_strength=None, expand_connections=None, session_id=None, namespace=None))]
    fn py_recall_at(
        &self,
        py: Python<'_>,
        query: &str,
        timestamp: f64,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<Vec<PyObject>> {
        let ns_vec = extract_namespaces(namespace)?;
        let ns_refs: Option<Vec<&str>> = ns_vec
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        let results = py
            .allow_threads(|| {
                self.recall_at(
                    query,
                    timestamp,
                    top_k,
                    min_strength,
                    expand_connections,
                    session_id,
                    ns_slice,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("created_at", rec.created_at)?;
            dict.set_item("source_type", &rec.source_type)?;
            if let Some(trust) = rec.metadata.get("trust_score") {
                dict.set_item("trust", trust)?;
            }
            if matches!(
                self.get_concept_surface_mode(),
                ConceptSurfaceMode::Inspect | ConceptSurfaceMode::Limited
            ) {
                let concepts = self.get_surfaced_concepts_for_record(&rec.id, Some(3));
                if !concepts.is_empty() {
                    let py_concepts = pyo3::types::PyList::empty_bound(py);
                    for concept in concepts {
                        let cdict = pyo3::types::PyDict::new_bound(py);
                        cdict.set_item("id", concept.id)?;
                        cdict.set_item("key", concept.key)?;
                        cdict.set_item("state", concept.state)?;
                        cdict.set_item("namespace", concept.namespace)?;
                        cdict.set_item("semantic_type", concept.semantic_type)?;
                        cdict.set_item("core_terms", concept.core_terms)?;
                        cdict.set_item("tags", concept.tags)?;
                        cdict.set_item("abstraction_score", concept.abstraction_score)?;
                        cdict.set_item("confidence", concept.confidence)?;
                        cdict.set_item("cluster_size", concept.cluster_size)?;
                        py_concepts.append(cdict)?;
                    }
                    dict.set_item("concepts", py_concepts)?;
                }
            }
            py_results.push(dict.unbind().into_any());
        }
        Ok(py_results)
    }

    /// Return access/strength timeline snapshot for a single record.
    #[pyo3(name = "history")]
    fn py_history(&self, py: Python<'_>, record_id: &str) -> PyResult<PyObject> {
        let info = py
            .allow_threads(|| self.history(record_id))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = pyo3::types::PyDict::new_bound(py);
        for (k, v) in &info {
            dict.set_item(k, v)?;
        }
        Ok(dict.unbind().into_any())
    }

    #[pyo3(name = "recall_full", signature = (query, top_k=None, include_failures=None, min_strength=None, expand_connections=None, session_id=None, namespace=None))]
    fn py_recall_full(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: Option<usize>,
        include_failures: Option<bool>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<Vec<PyObject>> {
        let ns_vec = extract_namespaces(namespace)?;
        let ns_refs: Option<Vec<&str>> = ns_vec
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        let results = py
            .allow_threads(|| {
                self.recall_full(
                    query,
                    top_k,
                    include_failures,
                    min_strength,
                    expand_connections,
                    session_id,
                    ns_slice,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("source_type", &rec.source_type)?;
            dict.set_item("metadata", &rec.metadata)?;
            if let Some(trust) = rec.metadata.get("trust_score") {
                dict.set_item("trust", trust)?;
            }
            if let Some(source) = rec.metadata.get("source") {
                dict.set_item("source", source)?;
            }
            if matches!(
                self.get_concept_surface_mode(),
                ConceptSurfaceMode::Inspect | ConceptSurfaceMode::Limited
            ) {
                let concepts = self.get_surfaced_concepts_for_record(&rec.id, Some(3));
                if !concepts.is_empty() {
                    let py_concepts = pyo3::types::PyList::empty_bound(py);
                    for concept in concepts {
                        let cdict = pyo3::types::PyDict::new_bound(py);
                        cdict.set_item("id", concept.id)?;
                        cdict.set_item("key", concept.key)?;
                        cdict.set_item("state", concept.state)?;
                        cdict.set_item("namespace", concept.namespace)?;
                        cdict.set_item("semantic_type", concept.semantic_type)?;
                        cdict.set_item("core_terms", concept.core_terms)?;
                        cdict.set_item("tags", concept.tags)?;
                        cdict.set_item("abstraction_score", concept.abstraction_score)?;
                        cdict.set_item("confidence", concept.confidence)?;
                        cdict.set_item("cluster_size", concept.cluster_size)?;
                        py_concepts.append(cdict)?;
                    }
                    dict.set_item("concepts", py_concepts)?;
                }
            }
            py_results.push(dict.unbind().into_any());
        }
        Ok(py_results)
    }

    #[pyo3(name = "search", signature = (query=None, level=None, tags=None, limit=None, content_type=None, source_type=None, namespace=None, semantic_type=None))]
    fn py_search(
        &self,
        py: Python<'_>,
        query: Option<&str>,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        limit: Option<usize>,
        content_type: Option<&str>,
        source_type: Option<&str>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
        semantic_type: Option<&str>,
    ) -> PyResult<Vec<Record>> {
        let ns_vec = extract_namespaces(namespace)?;
        let ns_refs: Option<Vec<&str>> = ns_vec
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        Ok(py.allow_threads(|| {
            self.search(
                query,
                level,
                tags,
                limit,
                content_type,
                source_type,
                ns_slice,
                semantic_type,
            )
        }))
    }

    #[pyo3(name = "get")]
    fn py_get(&self, record_id: &str) -> Option<Record> {
        self.get(record_id)
    }

    #[pyo3(name = "mark_record_salience", signature = (record_id, salience, reason=None))]
    fn py_mark_record_salience(
        &self,
        record_id: &str,
        salience: f32,
        reason: Option<&str>,
    ) -> PyResult<Option<Record>> {
        self.mark_record_salience(record_id, salience, reason)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "delete")]
    fn py_delete(&self, record_id: &str) -> PyResult<bool> {
        self.delete(record_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "explain_recall", signature = (query, top_k=None, min_strength=None, expand_connections=None, namespace=None))]
    fn py_explain_recall(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<PyObject> {
        let namespaces_owned = extract_namespaces(namespace)?;
        let namespaces_ref = namespaces_owned
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect::<Vec<_>>());
        let explanation = self.explain_recall(
            query,
            top_k,
            min_strength,
            expand_connections,
            namespaces_ref.as_deref(),
        );
        recall_explanation_to_py(py, &explanation)
    }

    #[pyo3(name = "explain_record")]
    fn py_explain_record(&self, py: Python<'_>, record_id: &str) -> PyResult<Option<PyObject>> {
        let item = self.explain_record(record_id);
        match item {
            Some(item) => recall_explanation_item_to_py(py, &item).map(Some),
            None => Ok(None),
        }
    }

    #[pyo3(name = "provenance_chain")]
    fn py_provenance_chain(&self, py: Python<'_>, record_id: &str) -> PyResult<Option<PyObject>> {
        let chain = self.provenance_chain(record_id);
        match chain {
            Some(chain) => provenance_chain_to_py(py, &chain).map(Some),
            None => Ok(None),
        }
    }

    #[pyo3(name = "explainability_bundle")]
    fn py_explainability_bundle(
        &self,
        py: Python<'_>,
        record_id: &str,
    ) -> PyResult<Option<PyObject>> {
        let bundle = self.explainability_bundle(record_id);
        match bundle {
            Some(bundle) => explainability_bundle_to_py(py, &bundle).map(Some),
            None => Ok(None),
        }
    }

    #[pyo3(name = "cross_namespace_digest", signature = (namespace=None, top_concepts_limit=None, min_record_count=None, pairwise_similarity_threshold=None, include_dimensions=None, compact_summary=None))]
    fn py_cross_namespace_digest(
        &self,
        py: Python<'_>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
        top_concepts_limit: Option<usize>,
        min_record_count: Option<usize>,
        pairwise_similarity_threshold: Option<f32>,
        include_dimensions: Option<Vec<String>>,
        compact_summary: Option<bool>,
    ) -> PyResult<PyObject> {
        let namespaces_owned = extract_namespaces(namespace)?;
        let namespaces_ref = namespaces_owned
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect::<Vec<_>>());
        let include_dimensions_ref = include_dimensions
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect::<Vec<_>>());
        let mut options = CrossNamespaceDigestOptions {
            min_record_count: min_record_count.unwrap_or(1),
            top_concepts_limit: top_concepts_limit.unwrap_or(5).clamp(1, 10),
            pairwise_similarity_threshold: pairwise_similarity_threshold
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            compact_summary: compact_summary.unwrap_or(false),
            ..CrossNamespaceDigestOptions::default()
        };
        apply_cross_namespace_dimension_flags(&mut options, include_dimensions_ref.as_deref());
        cross_namespace_digest_to_py(
            py,
            &self.cross_namespace_digest_with_options(namespaces_ref.as_deref(), options),
        )
    }

    #[pyo3(name = "update", signature = (record_id, content=None, level=None, tags=None, strength=None, metadata=None, source_type=None))]
    fn py_update(
        &self,
        record_id: &str,
        content: Option<&str>,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        strength: Option<f32>,
        metadata: Option<HashMap<String, String>>,
        source_type: Option<&str>,
    ) -> PyResult<Option<Record>> {
        self.update(
            record_id,
            content,
            level,
            tags,
            strength,
            metadata,
            source_type,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "connect", signature = (id_a, id_b, weight=None, relationship=None))]
    fn py_connect(
        &self,
        id_a: &str,
        id_b: &str,
        weight: Option<f32>,
        relationship: Option<&str>,
    ) -> PyResult<()> {
        self.connect(id_a, id_b, weight, relationship)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "decay")]
    fn py_decay(&self, py: Python<'_>) -> PyResult<(usize, usize)> {
        py.allow_threads(|| self.decay())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "consolidate")]
    fn py_consolidate(&self, py: Python<'_>) -> PyResult<HashMap<String, usize>> {
        py.allow_threads(|| self.consolidate())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "reflect")]
    fn py_reflect(&self, py: Python<'_>) -> PyResult<HashMap<String, usize>> {
        py.allow_threads(|| self.reflect())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "end_session")]
    fn py_end_session(&self, py: Python<'_>, session_id: &str) -> PyResult<HashMap<String, usize>> {
        py.allow_threads(|| self.end_session(session_id))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Run all insight detectors (phase 0 + 1 + 2) and return list of dicts.
    #[pyo3(name = "insights", signature = (phase=None))]
    fn py_insights(&self, py: Python<'_>, phase: Option<u8>) -> PyResult<Vec<PyObject>> {
        let records = self.records.read();
        let raw = match phase {
            Some(0) => crate::insights::detect_phase0(&records),
            Some(1) => crate::insights::detect_phase1(&records),
            Some(2) => crate::insights::detect_phase2(&records),
            None => crate::insights::detect_all(&records),
            Some(p) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid phase {}. Must be 0, 1, or 2.",
                    p
                )))
            }
        };

        let results: Vec<PyObject> = raw
            .into_iter()
            .map(|insight| {
                let dict = pyo3::types::PyDict::new_bound(py);
                let _ = dict.set_item("insight_type", &insight.insight_type);
                let _ = dict.set_item("severity", format!("{:?}", insight.severity));
                let _ = dict.set_item(
                    "phase",
                    match insight.phase {
                        crate::insights::Phase::RecordHealth => "record_health",
                        crate::insights::Phase::Relationships => "relationships",
                        crate::insights::Phase::CrossDomain => "cross_domain",
                    },
                );
                let _ = dict.set_item("record_ids", &insight.record_ids);
                let _ = dict.set_item("description", &insight.description);
                let evidence_dict = pyo3::types::PyDict::new_bound(py);
                for (k, v) in &insight.evidence {
                    let _ = evidence_dict.set_item(k, v);
                }
                let _ = dict.set_item("evidence", &evidence_dict);
                dict.unbind().into_any()
            })
            .collect();

        Ok(results)
    }

    #[pyo3(name = "stats")]
    fn py_stats(&self) -> HashMap<String, usize> {
        self.stats()
    }

    #[pyo3(name = "count", signature = (level=None))]
    fn py_count(&self, level: Option<Level>) -> usize {
        self.count(level)
    }

    // ── SDK Wrapper PyO3 Methods ──

    #[pyo3(name = "set_taxonomy")]
    fn py_set_taxonomy(&self, taxonomy: TagTaxonomy) {
        self.set_taxonomy(taxonomy);
    }

    #[pyo3(name = "get_taxonomy")]
    fn py_get_taxonomy(&self) -> TagTaxonomy {
        self.get_taxonomy()
    }

    #[pyo3(name = "set_trust_config")]
    fn py_set_trust_config(&self, config: TrustConfig) {
        self.set_trust_config(config);
    }

    #[pyo3(name = "configure_maintenance")]
    fn py_configure_maintenance(&self, config: MaintenanceConfig) {
        self.configure_maintenance(config);
    }

    #[pyo3(name = "run_maintenance")]
    fn py_run_maintenance(&self, py: Python<'_>) -> MaintenanceReport {
        py.allow_threads(|| self.run_maintenance())
    }

    #[pyo3(name = "get_maintenance_trend_history")]
    fn py_get_maintenance_trend_history(&self) -> Vec<background_brain::MaintenanceTrendSnapshot> {
        self.get_maintenance_trend_history()
    }

    #[pyo3(name = "get_maintenance_trend_summary")]
    fn py_get_maintenance_trend_summary(&self) -> background_brain::MaintenanceTrendSummary {
        self.get_maintenance_trend_summary()
    }

    #[pyo3(name = "get_reflection_summaries", signature = (limit=None))]
    fn py_get_reflection_summaries(
        &self,
        limit: Option<usize>,
    ) -> Vec<background_brain::ReflectionSummary> {
        self.get_reflection_summaries(limit)
    }

    #[pyo3(name = "get_latest_reflection_digest")]
    fn py_get_latest_reflection_digest(&self) -> Option<background_brain::ReflectionSummary> {
        self.get_latest_reflection_digest()
    }

    #[pyo3(name = "get_reflection_digest", signature = (limit=None))]
    fn py_get_reflection_digest(&self, limit: Option<usize>) -> background_brain::ReflectionDigest {
        self.get_reflection_digest(limit)
    }

    #[pyo3(name = "get_startup_validation_report")]
    fn py_get_startup_validation_report(&self) -> StartupValidationReport {
        self.get_startup_validation_report()
    }

    #[pyo3(name = "get_persistence_manifest")]
    fn py_get_persistence_manifest(&self) -> PersistenceManifest {
        self.get_persistence_manifest()
    }

    #[pyo3(name = "get_memory_health_digest", signature = (limit=None))]
    fn py_get_memory_health_digest(&self, limit: Option<usize>) -> MemoryHealthDigest {
        self.get_memory_health_digest(limit)
    }

    #[pyo3(name = "deprecate_belief")]
    fn py_deprecate_belief(&self, belief_id: &str) -> PyResult<bool> {
        self.deprecate_belief(belief_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "deprecate_belief_with_reason")]
    fn py_deprecate_belief_with_reason(&self, belief_id: &str, reason: &str) -> PyResult<bool> {
        self.deprecate_belief_with_reason(belief_id, reason)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "invalidate_causal_pattern")]
    fn py_invalidate_causal_pattern(&self, pattern_id: &str) -> PyResult<bool> {
        self.invalidate_causal_pattern(pattern_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "invalidate_causal_pattern_with_reason")]
    fn py_invalidate_causal_pattern_with_reason(
        &self,
        pattern_id: &str,
        reason: &str,
    ) -> PyResult<bool> {
        self.invalidate_causal_pattern_with_reason(pattern_id, reason)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "retract_causal_pattern")]
    fn py_retract_causal_pattern(&self, pattern_id: &str) -> PyResult<bool> {
        self.retract_causal_pattern(pattern_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "retract_causal_pattern_with_reason")]
    fn py_retract_causal_pattern_with_reason(
        &self,
        pattern_id: &str,
        reason: &str,
    ) -> PyResult<bool> {
        self.retract_causal_pattern_with_reason(pattern_id, reason)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "retract_policy_hint")]
    fn py_retract_policy_hint(&self, hint_id: &str) -> PyResult<bool> {
        self.retract_policy_hint(hint_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "retract_policy_hint_with_reason")]
    fn py_retract_policy_hint_with_reason(&self, hint_id: &str, reason: &str) -> PyResult<bool> {
        self.retract_policy_hint_with_reason(hint_id, reason)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "get_correction_log")]
    fn py_get_correction_log(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let entries = self.get_correction_log();
        let mut results = Vec::with_capacity(entries.len());
        for entry in entries {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("timestamp", entry.timestamp)?;
            dict.set_item("time_iso", entry.time_iso)?;
            dict.set_item("target_kind", entry.target_kind)?;
            dict.set_item("target_id", entry.target_id)?;
            dict.set_item("operation", entry.operation)?;
            dict.set_item("reason", entry.reason)?;
            dict.set_item("session_id", entry.session_id)?;
            results.push(dict.unbind().into_any());
        }
        Ok(results)
    }

    #[pyo3(name = "get_correction_log_for_target")]
    fn py_get_correction_log_for_target(
        &self,
        py: Python<'_>,
        target_kind: &str,
        target_id: &str,
    ) -> PyResult<Vec<PyObject>> {
        let entries = self.get_correction_log_for_target(target_kind, target_id);
        let mut results = Vec::with_capacity(entries.len());
        for entry in entries {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("timestamp", entry.timestamp)?;
            dict.set_item("time_iso", entry.time_iso)?;
            dict.set_item("target_kind", entry.target_kind)?;
            dict.set_item("target_id", entry.target_id)?;
            dict.set_item("operation", entry.operation)?;
            dict.set_item("reason", entry.reason)?;
            dict.set_item("session_id", entry.session_id)?;
            results.push(dict.unbind().into_any());
        }
        Ok(results)
    }

    #[pyo3(name = "get_correction_review_queue", signature = (limit=None))]
    fn py_get_correction_review_queue(
        &self,
        py: Python<'_>,
        limit: Option<usize>,
    ) -> PyResult<PyObject> {
        correction_review_candidates_to_py(py, &self.get_correction_review_queue(limit))
    }

    #[pyo3(name = "get_suggested_corrections", signature = (limit=None))]
    fn py_get_suggested_corrections(
        &self,
        py: Python<'_>,
        limit: Option<usize>,
    ) -> PyResult<PyObject> {
        suggested_corrections_to_py(py, &self.get_suggested_corrections(limit))
    }

    #[pyo3(name = "get_suggested_corrections_report", signature = (limit=None))]
    fn py_get_suggested_corrections_report(
        &self,
        py: Python<'_>,
        limit: Option<usize>,
    ) -> PyResult<PyObject> {
        suggested_corrections_report_to_py(py, &self.get_suggested_corrections_report(limit))
    }

    #[pyo3(name = "get_namespace_governance_status", signature = (namespace=None))]
    fn py_get_namespace_governance_status(
        &self,
        py: Python<'_>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<PyObject> {
        let namespaces_owned = extract_namespaces(namespace)?;
        let namespaces_ref = namespaces_owned
            .as_ref()
            .map(|items| items.iter().map(String::as_str).collect::<Vec<_>>());
        namespace_governance_statuses_to_py(
            py,
            &self.get_namespace_governance_status_filtered(namespaces_ref.as_deref()),
        )
    }

    #[pyo3(name = "get_high_volatility_beliefs", signature = (min_volatility=None, limit=None))]
    fn py_get_high_volatility_beliefs(
        &self,
        py: Python<'_>,
        min_volatility: Option<f32>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        self.get_high_volatility_beliefs(min_volatility, limit)
            .iter()
            .map(|belief| belief_to_py(py, belief))
            .collect()
    }

    #[pyo3(name = "get_high_salience_records", signature = (min_salience=None, limit=None))]
    fn py_get_high_salience_records(
        &self,
        min_salience: Option<f32>,
        limit: Option<usize>,
    ) -> Vec<Record> {
        self.get_high_salience_records(min_salience, limit)
    }

    #[pyo3(name = "get_salience_summary")]
    fn py_get_salience_summary(&self) -> SalienceSummary {
        self.get_salience_summary()
    }

    #[pyo3(name = "get_low_stability_beliefs", signature = (max_stability=None, limit=None))]
    fn py_get_low_stability_beliefs(
        &self,
        py: Python<'_>,
        max_stability: Option<f32>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        self.get_low_stability_beliefs(max_stability, limit)
            .iter()
            .map(|belief| belief_to_py(py, belief))
            .collect()
    }

    #[pyo3(name = "get_belief_instability_summary")]
    fn py_get_belief_instability_summary(
        &self,
    ) -> crate::epistemic_runtime::BeliefInstabilitySummary {
        self.get_belief_instability_summary()
    }

    #[pyo3(name = "get_contradiction_clusters", signature = (namespace=None, limit=None))]
    fn py_get_contradiction_clusters(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<crate::epistemic_runtime::ContradictionCluster> {
        self.get_contradiction_clusters(namespace, limit)
    }

    #[pyo3(name = "get_recently_corrected_beliefs", signature = (limit=None))]
    fn py_get_recently_corrected_beliefs(
        &self,
        py: Python<'_>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        self.get_recently_corrected_beliefs(limit)
            .iter()
            .map(|belief| belief_to_py(py, belief))
            .collect()
    }

    #[pyo3(name = "get_contradiction_review_queue", signature = (namespace=None, limit=None))]
    fn py_get_contradiction_review_queue(
        &self,
        py: Python<'_>,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> PyResult<PyObject> {
        contradiction_review_candidates_to_py(
            py,
            &self.get_contradiction_review_queue(namespace, limit),
        )
    }

    #[pyo3(name = "get_suppressed_policy_hints", signature = (namespace=None, limit=None))]
    fn py_get_suppressed_policy_hints(
        &self,
        py: Python<'_>,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        self.get_suppressed_policy_hints(namespace, limit)
            .iter()
            .map(|hint| policy_hint_to_py(py, hint))
            .collect()
    }

    #[pyo3(name = "get_rejected_policy_hints", signature = (namespace=None, limit=None))]
    fn py_get_rejected_policy_hints(
        &self,
        py: Python<'_>,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        self.get_rejected_policy_hints(namespace, limit)
            .iter()
            .map(|hint| policy_hint_to_py(py, hint))
            .collect()
    }

    #[pyo3(name = "get_policy_lifecycle_summary", signature = (action_limit=None, domain_limit=None))]
    fn py_get_policy_lifecycle_summary(
        &self,
        action_limit: Option<usize>,
        domain_limit: Option<usize>,
    ) -> crate::epistemic_runtime::PolicyLifecycleSummary {
        self.get_policy_lifecycle_summary(action_limit, domain_limit)
    }

    #[pyo3(name = "get_policy_pressure_report", signature = (namespace=None, limit=None))]
    fn py_get_policy_pressure_report(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<crate::epistemic_runtime::PolicyPressureArea> {
        self.get_policy_pressure_report(namespace, limit)
    }

    #[pyo3(name = "get_surfaced_concepts", signature = (limit=None))]
    fn py_get_surfaced_concepts(
        &self,
        limit: Option<usize>,
    ) -> Vec<crate::concept::SurfacedConcept> {
        self.get_surfaced_concepts(limit)
    }

    #[pyo3(name = "get_surfaced_concepts_for_namespace", signature = (namespace, limit=None))]
    fn py_get_surfaced_concepts_for_namespace(
        &self,
        namespace: &str,
        limit: Option<usize>,
    ) -> Vec<crate::concept::SurfacedConcept> {
        self.get_surfaced_concepts_for_namespace(namespace, limit)
    }

    #[pyo3(name = "get_structural_relations", signature = (limit=None))]
    fn py_get_structural_relations(&self, limit: Option<usize>) -> Vec<StructuralRelation> {
        self.get_structural_relations(limit)
    }

    #[pyo3(name = "get_relations", signature = (limit=None))]
    fn py_get_relations(&self, limit: Option<usize>) -> Vec<RelationEdge> {
        self.get_relations(limit)
    }

    #[pyo3(name = "get_structural_relations_for_record", signature = (record_id, limit=None))]
    fn py_get_structural_relations_for_record(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Vec<StructuralRelation> {
        self.get_structural_relations_for_record(record_id, limit)
    }

    #[pyo3(name = "get_relations_for_record", signature = (record_id, limit=None))]
    fn py_get_relations_for_record(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Vec<RelationEdge> {
        self.get_relations_for_record(record_id, limit)
    }

    #[pyo3(name = "get_relation_digest", signature = (record_id, limit=None))]
    fn py_get_relation_digest(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Option<RelationDigest> {
        self.get_relation_digest(record_id, limit)
    }

    #[pyo3(name = "get_entity_digest")]
    fn py_get_entity_digest(&self, entity_id: &str) -> Option<EntityDigest> {
        self.get_entity_digest(entity_id)
    }

    #[pyo3(name = "get_entity_digest_for_record")]
    fn py_get_entity_digest_for_record(&self, record_id: &str) -> Option<EntityDigest> {
        self.get_entity_digest_for_record(record_id)
    }

    #[pyo3(name = "get_entity_graph_digest", signature = (entity_id, limit=None))]
    fn py_get_entity_graph_digest(
        &self,
        entity_id: &str,
        limit: Option<usize>,
    ) -> Option<EntityGraphDigest> {
        self.get_entity_graph_digest(entity_id, limit)
    }

    #[pyo3(name = "get_entity_graph_digest_for_record", signature = (record_id, limit=None))]
    fn py_get_entity_graph_digest_for_record(
        &self,
        record_id: &str,
        limit: Option<usize>,
    ) -> Option<EntityGraphDigest> {
        self.get_entity_graph_digest_for_record(record_id, limit)
    }

    #[pyo3(name = "get_entity_graph_neighbors", signature = (entity_id, top_n=None, min_weight=None, relation_type_filter=None, edge_limit=None))]
    fn py_get_entity_graph_neighbors(
        &self,
        entity_id: &str,
        top_n: Option<usize>,
        min_weight: Option<f32>,
        relation_type_filter: Option<&str>,
        edge_limit: Option<usize>,
    ) -> Vec<EntityGraphDigest> {
        self.get_entity_graph_neighbors(
            entity_id,
            top_n,
            min_weight,
            relation_type_filter,
            edge_limit,
        )
    }

    #[pyo3(name = "get_entity_relations", signature = (entity_id, limit=None))]
    fn py_get_entity_relations(
        &self,
        entity_id: &str,
        limit: Option<usize>,
    ) -> Vec<EntityRelationEdge> {
        self.get_entity_relations(entity_id, limit)
    }

    #[pyo3(name = "get_family_graph", signature = (namespace=None))]
    fn py_get_family_graph(&self, namespace: Option<&str>) -> Option<FamilyGraphSnapshot> {
        self.get_family_graph(namespace)
    }

    #[pyo3(name = "get_family_graph_for_record")]
    fn py_get_family_graph_for_record(&self, record_id: &str) -> Option<FamilyGraphSnapshot> {
        self.get_family_graph_for_record(record_id)
    }

    #[pyo3(name = "get_person_digest")]
    fn py_get_person_digest(&self, record_id: &str) -> Option<PersonDigest> {
        self.get_person_digest(record_id)
    }

    #[pyo3(name = "get_project_graph")]
    fn py_get_project_graph(&self, project_id: &str) -> Option<ProjectGraphSnapshot> {
        self.get_project_graph(project_id)
    }

    #[pyo3(name = "get_project_graph_for_record")]
    fn py_get_project_graph_for_record(&self, record_id: &str) -> Option<ProjectGraphSnapshot> {
        self.get_project_graph_for_record(record_id)
    }

    #[pyo3(name = "get_project_status")]
    fn py_get_project_status(&self, project_id: &str) -> Option<ProjectStatusSnapshot> {
        self.get_project_status(project_id)
    }

    #[pyo3(name = "get_project_status_for_record")]
    fn py_get_project_status_for_record(&self, record_id: &str) -> Option<ProjectStatusSnapshot> {
        self.get_project_status_for_record(record_id)
    }

    #[pyo3(name = "get_project_timeline")]
    fn py_get_project_timeline(&self, project_id: &str) -> Option<ProjectTimelineSnapshot> {
        self.get_project_timeline(project_id)
    }

    #[pyo3(name = "get_project_timeline_for_record")]
    fn py_get_project_timeline_for_record(
        &self,
        record_id: &str,
    ) -> Option<ProjectTimelineSnapshot> {
        self.get_project_timeline_for_record(record_id)
    }

    #[pyo3(name = "get_project_digest")]
    fn py_get_project_digest(&self, project_id: &str) -> Option<ProjectDigest> {
        self.get_project_digest(project_id)
    }

    #[pyo3(name = "get_project_digest_for_record")]
    fn py_get_project_digest_for_record(&self, record_id: &str) -> Option<ProjectDigest> {
        self.get_project_digest_for_record(record_id)
    }

    #[pyo3(name = "recall_family_context", signature = (query, top_k=None, min_strength=None, expand_connections=None, session_id=None, namespace=None))]
    fn py_recall_family_context(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let results = py
            .allow_threads(|| {
                self.recall_family_context(
                    query,
                    top_k,
                    min_strength,
                    expand_connections,
                    session_id,
                    namespace,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("source_type", &rec.source_type)?;
            dict.set_item("metadata", &rec.metadata)?;
            py_results.push(dict.unbind().into_any());
        }

        Ok(py_results)
    }

    #[pyo3(name = "recall_person_context", signature = (record_id, query, top_k=None, min_strength=None, expand_connections=None, session_id=None))]
    fn py_recall_person_context(
        &self,
        py: Python<'_>,
        record_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let results = py
            .allow_threads(|| {
                self.recall_person_context(
                    record_id,
                    query,
                    top_k,
                    min_strength,
                    expand_connections,
                    session_id,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("source_type", &rec.source_type)?;
            dict.set_item("metadata", &rec.metadata)?;
            py_results.push(dict.unbind().into_any());
        }

        Ok(py_results)
    }

    #[pyo3(name = "recall_relation_context", signature = (record_id, query, top_k=None, min_strength=None, expand_connections=None, session_id=None, limit=None))]
    fn py_recall_relation_context(
        &self,
        py: Python<'_>,
        record_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        let results = py
            .allow_threads(|| {
                self.recall_relation_context(
                    record_id,
                    query,
                    top_k,
                    min_strength,
                    expand_connections,
                    session_id,
                    limit,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("source_type", &rec.source_type)?;
            dict.set_item("metadata", &rec.metadata)?;
            py_results.push(dict.unbind().into_any());
        }

        Ok(py_results)
    }

    #[pyo3(name = "recall_entity_context", signature = (entity_id, query, top_k=None, min_strength=None, expand_connections=None, session_id=None))]
    fn py_recall_entity_context(
        &self,
        py: Python<'_>,
        entity_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let results = py
            .allow_threads(|| {
                self.recall_entity_context(
                    entity_id,
                    query,
                    top_k,
                    min_strength,
                    expand_connections,
                    session_id,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("source_type", &rec.source_type)?;
            dict.set_item("metadata", &rec.metadata)?;
            py_results.push(dict.unbind().into_any());
        }

        Ok(py_results)
    }

    #[pyo3(name = "recall_entity_context_for_record", signature = (record_id, query, top_k=None, min_strength=None, expand_connections=None, session_id=None))]
    fn py_recall_entity_context_for_record(
        &self,
        py: Python<'_>,
        record_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let results = py
            .allow_threads(|| {
                self.recall_entity_context_for_record(
                    record_id,
                    query,
                    top_k,
                    min_strength,
                    expand_connections,
                    session_id,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("source_type", &rec.source_type)?;
            dict.set_item("metadata", &rec.metadata)?;
            py_results.push(dict.unbind().into_any());
        }

        Ok(py_results)
    }

    #[pyo3(name = "recall_entity_graph_context", signature = (entity_id, query, top_k=None, min_strength=None, expand_connections=None, session_id=None, limit=None))]
    fn py_recall_entity_graph_context(
        &self,
        py: Python<'_>,
        entity_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        let results = py
            .allow_threads(|| {
                self.recall_entity_graph_context(
                    entity_id,
                    query,
                    top_k,
                    min_strength,
                    expand_connections,
                    session_id,
                    limit,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("source_type", &rec.source_type)?;
            dict.set_item("metadata", &rec.metadata)?;
            py_results.push(dict.unbind().into_any());
        }

        Ok(py_results)
    }

    #[pyo3(name = "recall_project_context", signature = (project_id, query, top_k=None, min_strength=None, expand_connections=None, session_id=None))]
    fn py_recall_project_context(
        &self,
        py: Python<'_>,
        project_id: &str,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let results = py
            .allow_threads(|| {
                self.recall_project_context(
                    project_id,
                    query,
                    top_k,
                    min_strength,
                    expand_connections,
                    session_id,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut py_results = Vec::new();
        for (score, rec) in results {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", &rec.id)?;
            dict.set_item("content", &rec.content)?;
            dict.set_item("score", score)?;
            dict.set_item("level", rec.level.name())?;
            dict.set_item("strength", rec.strength)?;
            dict.set_item("tags", &rec.tags)?;
            dict.set_item("source_type", &rec.source_type)?;
            dict.set_item("metadata", &rec.metadata)?;
            py_results.push(dict.unbind().into_any());
        }
        Ok(py_results)
    }

    #[pyo3(name = "get_surfaced_policy_hints", signature = (limit=None))]
    fn py_get_surfaced_policy_hints(
        &self,
        limit: Option<usize>,
    ) -> Vec<crate::policy::SurfacedPolicyHint> {
        self.get_surfaced_policy_hints(limit)
    }

    #[pyo3(name = "get_surfaced_policy_hints_for_namespace", signature = (namespace, limit=None))]
    fn py_get_surfaced_policy_hints_for_namespace(
        &self,
        namespace: &str,
        limit: Option<usize>,
    ) -> Vec<crate::policy::SurfacedPolicyHint> {
        self.get_surfaced_policy_hints_for_namespace(namespace, limit)
    }

    /// Enable all four cognitive recall reranking signals (Python binding).
    ///
    /// Equivalent to calling `set_belief_rerank_mode("limited")`,
    /// `set_concept_surface_mode("limited")`, `set_causal_rerank_mode("limited")`,
    /// and `set_policy_rerank_mode("limited")` in one call.
    #[pyo3(name = "enable_full_cognitive_stack")]
    fn py_enable_full_cognitive_stack(&self) {
        self.enable_full_cognitive_stack();
    }

    /// Disable all four cognitive recall reranking signals (Python binding).
    ///
    /// Resets every phase to Off. Counterpart to `enable_full_cognitive_stack()`.
    #[pyo3(name = "disable_full_cognitive_stack")]
    fn py_disable_full_cognitive_stack(&self) {
        self.disable_full_cognitive_stack();
    }

    /// Set belief-aware recall reranking mode (Python binding).
    /// mode: "off" | "shadow" | "limited"
    #[pyo3(name = "set_belief_rerank_mode")]
    fn py_set_belief_rerank_mode(&self, mode: &str) {
        use crate::recall::BeliefRerankMode;
        let m = match mode {
            "limited" => BeliefRerankMode::Limited,
            "shadow" => BeliefRerankMode::Shadow,
            _ => BeliefRerankMode::Off,
        };
        self.set_belief_rerank_mode(m);
    }

    /// Set concept surface mode (Python binding).
    /// mode: "off" | "inspect" | "limited"
    #[pyo3(name = "set_concept_surface_mode")]
    fn py_set_concept_surface_mode(&self, mode: &str) {
        use crate::concept::ConceptSurfaceMode;
        let m = match mode {
            "limited" => ConceptSurfaceMode::Limited,
            "inspect" => ConceptSurfaceMode::Inspect,
            _ => ConceptSurfaceMode::Off,
        };
        self.set_concept_surface_mode(m);
    }

    /// Set causal pattern recall reranking mode (Python binding).
    /// mode: "off" | "limited"
    /// Set causal evidence gating mode (Python binding).
    /// mode: "strict" | "temporal_cluster_recovery" | "explicit_trusted"
    ///
    /// "explicit_trusted" — recommended when causality is user-declared via
    /// link_records(). A single explicit link suffices for the repeated-evidence
    /// gate; the effect-variants gate is bypassed for consistent-polarity outcomes.
    #[pyo3(name = "set_causal_evidence_mode")]
    fn py_set_causal_evidence_mode(&self, mode: &str) {
        let m = match mode {
            "explicit_trusted" => CausalEvidenceMode::ExplicitTrusted,
            "temporal_cluster_recovery" => CausalEvidenceMode::TemporalClusterRecovery,
            _ => CausalEvidenceMode::StrictRepeatedWindows,
        };
        self.set_causal_evidence_mode(m);
    }

    #[pyo3(name = "set_causal_rerank_mode")]
    fn py_set_causal_rerank_mode(&self, mode: &str) {
        use crate::causal::CausalRerankMode;
        let m = match mode {
            "limited" => CausalRerankMode::Limited,
            _ => CausalRerankMode::Off,
        };
        self.set_causal_rerank_mode(m);
    }

    /// Set policy hint recall reranking mode (Python binding).
    /// mode: "off" | "limited"
    #[pyo3(name = "set_policy_rerank_mode")]
    fn py_set_policy_rerank_mode(&self, mode: &str) {
        use crate::policy::PolicyRerankMode;
        let m = match mode {
            "limited" => PolicyRerankMode::Limited,
            _ => PolicyRerankMode::Off,
        };
        self.set_policy_rerank_mode(m);
    }

    #[pyo3(name = "start_background", signature = (interval_secs=None))]
    fn py_start_background(&self, interval_secs: Option<u64>) {
        self.start_background(interval_secs);
    }

    #[pyo3(name = "stop_background")]
    fn py_stop_background(&self) {
        self.stop_background();
    }

    #[pyo3(name = "is_background_running")]
    fn py_is_background_running(&self) -> bool {
        self.is_background_running()
    }

    #[pyo3(name = "start_research", signature = (topic, depth=None))]
    fn py_start_research(
        &self,
        py: Python<'_>,
        topic: &str,
        depth: Option<&str>,
    ) -> PyResult<PyObject> {
        let project = self.start_research(topic, depth);
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("id", &project.id)?;
        dict.set_item("topic", &project.topic)?;
        dict.set_item("depth", &project.depth)?;
        dict.set_item("queries", &project.queries)?;
        Ok(dict.unbind().into_any())
    }

    #[pyo3(name = "add_research_finding", signature = (project_id, query, result, url=None))]
    fn py_add_research_finding(
        &self,
        project_id: &str,
        query: &str,
        result: &str,
        url: Option<&str>,
    ) -> PyResult<()> {
        self.add_research_finding(project_id, query, result, url)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "complete_research", signature = (project_id, synthesis=None))]
    fn py_complete_research(
        &self,
        project_id: &str,
        synthesis: Option<String>,
    ) -> PyResult<String> {
        let rec = self
            .complete_research(project_id, synthesis)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "store_project_task", signature = (project_id, content, due_date=None, metadata=None))]
    fn py_store_project_task(
        &self,
        project_id: &str,
        content: &str,
        due_date: Option<&str>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<String> {
        let rec = self
            .store_project_task(project_id, content, due_date, metadata)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "store_project_todo", signature = (project_id, content, metadata=None))]
    fn py_store_project_todo(
        &self,
        project_id: &str,
        content: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<String> {
        let rec = self
            .store_project_todo(project_id, content, metadata)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "store_project_note", signature = (project_id, content, metadata=None))]
    fn py_store_project_note(
        &self,
        project_id: &str,
        content: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<String> {
        let rec = self
            .store_project_note(project_id, content, metadata)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "store_family_person", signature = (relation_type, content, metadata=None, namespace=None))]
    fn py_store_family_person(
        &self,
        relation_type: &str,
        content: &str,
        metadata: Option<HashMap<String, String>>,
        namespace: Option<&str>,
    ) -> PyResult<String> {
        let rec = self
            .store_family_person(relation_type, content, metadata, namespace)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "link_records", signature = (source_id, target_id, relation_type, weight=None))]
    fn py_link_records(
        &self,
        source_id: &str,
        target_id: &str,
        relation_type: &str,
        weight: Option<f32>,
    ) -> PyResult<RelationEdge> {
        self.link_records(source_id, target_id, relation_type, weight)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "link_entities", signature = (source_entity_id, target_entity_id, relation_type, weight=None))]
    fn py_link_entities(
        &self,
        source_entity_id: &str,
        target_entity_id: &str,
        relation_type: &str,
        weight: Option<f32>,
    ) -> PyResult<EntityRelationEdge> {
        self.link_entities(source_entity_id, target_entity_id, relation_type, weight)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "store_user_profile")]
    fn py_store_user_profile(&self, fields: HashMap<String, String>) -> PyResult<String> {
        let rec = self
            .store_user_profile(fields)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "get_user_profile")]
    fn py_get_user_profile(&self, py: Python<'_>) -> Option<PyObject> {
        self.get_user_profile().map(|fields| {
            let dict = pyo3::types::PyDict::new_bound(py);
            for (k, v) in &fields {
                let _ = dict.set_item(k, v);
            }
            dict.unbind().into_any()
        })
    }

    #[pyo3(name = "set_persona")]
    fn py_set_persona(&self, persona: AgentPersona) -> PyResult<String> {
        let rec = self
            .set_persona(persona)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "get_persona")]
    fn py_get_persona(&self) -> Option<AgentPersona> {
        self.get_persona()
    }

    #[pyo3(name = "store_image")]
    #[pyo3(signature = (path, description, level=None, tags=None, namespace=None))]
    fn py_store_image(
        &self,
        path: &str,
        description: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        namespace: Option<&str>,
    ) -> PyResult<String> {
        self.store_image(path, description, level, tags, namespace)
            .map(|r| r.id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "store_audio_transcript")]
    #[pyo3(signature = (transcript, source_path, level=None, tags=None, namespace=None))]
    fn py_store_audio_transcript(
        &self,
        transcript: &str,
        source_path: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        namespace: Option<&str>,
    ) -> PyResult<String> {
        self.store_audio_transcript(transcript, source_path, level, tags, namespace)
            .map(|r| r.id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    // ── Phase 6: Adaptive Recall ──

    #[pyo3(name = "feedback")]
    fn py_feedback(&self, record_id: &str, useful: bool) -> PyResult<bool> {
        self.feedback(record_id, useful)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "feedback_stats")]
    fn py_feedback_stats(&self, record_id: &str) -> Option<(u32, u32, i32)> {
        self.feedback_stats(record_id)
    }

    // ── Phase 6: Semantic Versioning ──

    #[pyo3(name = "supersede")]
    #[pyo3(signature = (old_id, new_content, level=None, tags=None, namespace=None))]
    fn py_supersede(
        &self,
        old_id: &str,
        new_content: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        namespace: Option<&str>,
    ) -> PyResult<String> {
        self.supersede(old_id, new_content, level, tags, namespace)
            .map(|r| r.id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "superseded_by")]
    fn py_superseded_by(&self, record_id: &str) -> Option<String> {
        self.superseded_by(record_id)
    }

    #[pyo3(name = "version_chain")]
    fn py_version_chain(&self, record_id: &str) -> Vec<Record> {
        self.version_chain(record_id)
    }

    // ── Phase 6: Snapshots & Rollback ──

    #[pyo3(name = "snapshot")]
    fn py_snapshot(&self, label: &str) -> PyResult<String> {
        self.snapshot(label)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "rollback")]
    fn py_rollback(&self, label: &str) -> PyResult<usize> {
        self.rollback(label)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "diff")]
    fn py_diff(&self, label_a: &str, label_b: &str) -> PyResult<HashMap<String, Vec<String>>> {
        self.diff(label_a, label_b)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "list_snapshots")]
    fn py_list_snapshots(&self) -> Vec<String> {
        self.list_snapshots()
    }

    // ── Phase 6: Agent-to-Agent Sharing ──

    #[pyo3(name = "export_context")]
    #[pyo3(signature = (query, top_k=None, namespace=None))]
    fn py_export_context(
        &self,
        query: &str,
        top_k: Option<usize>,
        namespace: Option<&str>,
    ) -> PyResult<String> {
        let ns_vec: Vec<&str>;
        let ns_slice = match namespace {
            Some(ns) => {
                ns_vec = vec![ns];
                Some(ns_vec.as_slice())
            }
            None => None,
        };
        self.export_context(query, top_k, ns_slice)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "import_context")]
    fn py_import_context(&self, fragment_json: &str) -> PyResult<usize> {
        self.import_context(fragment_json)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "get_credibility")]
    fn py_get_credibility(&self, url: &str) -> f32 {
        self.get_credibility(url)
    }

    #[pyo3(name = "set_credibility_override")]
    fn py_set_credibility_override(&self, domain: &str, score: f32) {
        self.set_credibility_override(domain, score);
    }

    #[pyo3(name = "record_tool_failure")]
    fn py_record_tool_failure(&self, tool_name: &str) {
        self.record_tool_failure(tool_name);
    }

    #[pyo3(name = "record_tool_success")]
    fn py_record_tool_success(&self, tool_name: &str) {
        self.record_tool_success(tool_name);
    }

    #[pyo3(name = "is_tool_available")]
    fn py_is_tool_available(&self, tool_name: &str) -> bool {
        self.is_tool_available(tool_name)
    }

    #[pyo3(name = "tool_health")]
    fn py_tool_health(&self) -> HashMap<String, String> {
        self.tool_health()
    }

    // ── Existing PyO3 Methods ──

    #[pyo3(name = "close")]
    fn py_close(&self) -> PyResult<()> {
        self.close()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &self,
        _exc_type: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
        _exc_val: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
        _exc_tb: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<bool> {
        self.close()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(false)
    }

    #[pyo3(name = "flush")]
    fn py_flush(&self) -> PyResult<()> {
        self.flush()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "export_json")]
    fn py_export_json(&self) -> PyResult<String> {
        self.export_json()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "import_json")]
    fn py_import_json(&self, json_str: &str) -> PyResult<usize> {
        self.import_json(json_str)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "is_encrypted")]
    fn py_is_encrypted(&self) -> bool {
        self.is_encrypted()
    }

    #[pyo3(name = "load_synonyms")]
    fn py_load_synonyms(&self, path: &str) -> PyResult<usize> {
        self.load_synonyms(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[pyo3(name = "has_synonyms")]
    fn py_has_synonyms(&self) -> bool {
        self.has_synonyms()
    }

    #[pyo3(name = "process", signature = (text, pin=None))]
    fn py_process(&self, text: &str, pin: Option<bool>) -> PyResult<String> {
        self.process(text, pin)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    // ── Embedding Support ──

    /// Set a Python callable as the embedding function.
    /// The function receives a string and returns a list of floats.
    /// When set, embeddings are computed on store and used as a 4th RRF signal.
    ///
    /// Example:
    ///   brain.set_embedding_fn(lambda text: model.encode(text).tolist())
    #[pyo3(name = "set_embedding_fn")]
    fn py_set_embedding_fn(&self, func: PyObject) {
        *self.embedding_fn.write() = Some(func);
    }

    /// Clear the embedding function.
    #[pyo3(name = "clear_embedding_fn")]
    fn py_clear_embedding_fn(&self) {
        *self.embedding_fn.write() = None;
    }

    /// Store an embedding vector for a specific record.
    #[pyo3(name = "store_embedding")]
    fn py_store_embedding(&self, record_id: &str, embedding: Vec<f32>) {
        self.store_embedding(record_id, embedding);
    }

    /// Check if embedding support is active.
    #[pyo3(name = "has_embeddings")]
    fn py_has_embeddings(&self) -> bool {
        self.has_embeddings()
    }

    // ── Two-Tier API PyO3 Bindings ──

    #[pyo3(name = "recall_cognitive", signature = (query=None, limit=None, namespace=None))]
    fn py_recall_cognitive(
        &self,
        py: Python<'_>,
        query: Option<&str>,
        limit: Option<usize>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<Vec<Record>> {
        let ns_vec = extract_namespaces(namespace)?;
        let ns_refs: Option<Vec<&str>> = ns_vec
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        Ok(py.allow_threads(|| self.recall_cognitive(query, limit, ns_slice)))
    }

    #[pyo3(name = "recall_core_tier", signature = (query=None, limit=None, namespace=None))]
    fn py_recall_core_tier(
        &self,
        py: Python<'_>,
        query: Option<&str>,
        limit: Option<usize>,
        namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<Vec<Record>> {
        let ns_vec = extract_namespaces(namespace)?;
        let ns_refs: Option<Vec<&str>> = ns_vec
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        Ok(py.allow_threads(|| self.recall_core_tier(query, limit, ns_slice)))
    }

    #[pyo3(name = "tier_stats")]
    fn py_tier_stats(&self) -> HashMap<String, usize> {
        self.tier_stats()
    }

    #[pyo3(name = "promotion_candidates", signature = (min_activations=None, min_strength=None))]
    fn py_promotion_candidates(
        &self,
        min_activations: Option<u32>,
        min_strength: Option<f32>,
    ) -> Vec<Record> {
        self.promotion_candidates(min_activations, min_strength)
    }

    #[pyo3(name = "promote_record")]
    fn py_promote_record(&self, record_id: &str) -> Option<Level> {
        self.promote_record(record_id)
    }

    // ── Namespace PyO3 Methods ──

    #[pyo3(name = "list_namespaces")]
    fn py_list_namespaces(&self) -> Vec<String> {
        self.list_namespaces()
    }

    #[pyo3(name = "move_record")]
    fn py_move_record(&self, record_id: &str, new_namespace: &str) -> Option<Record> {
        self.move_record(record_id, new_namespace)
    }

    #[pyo3(name = "namespace_stats")]
    fn py_namespace_stats(&self) -> HashMap<String, usize> {
        self.namespace_stats()
    }

    fn __repr__(&self) -> String {
        let records = self.records.read();
        format!(
            "Aura(path='{}', records={}, encrypted={})",
            self.config.path.display(),
            records.len(),
            self.is_encrypted()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_recall() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let rec = aura.store(
            "Teo loves Rust",
            Some(Level::Identity),
            Some(vec!["person".into()]),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;

        assert_eq!(rec.content, "Teo loves Rust");
        assert_eq!(rec.level, Level::Identity);

        let preamble = aura.recall("Who is Teo?", None, None, None, None, None)?;
        assert!(preamble.contains("Teo loves Rust"));

        aura.close()?;
        Ok(())
    }

    #[test]
    fn test_deduplication() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "The quick brown fox jumps over the lazy dog",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        aura.store(
            "The quick brown fox jumps over the lazy dog",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;

        // Should be deduplicated to 1 record
        assert_eq!(aura.count(None), 1);

        Ok(())
    }

    #[test]
    fn test_stats() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "test1",
            Some(Level::Working),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "test2",
            Some(Level::Identity),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        let stats = aura.stats();
        assert_eq!(stats["total_records"], 2);
        assert_eq!(stats["working"], 1);
        assert_eq!(stats["identity"], 1);

        Ok(())
    }

    #[test]
    fn test_auto_protect_on_store() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let rec = aura.store(
            "My phone is +380991234567",
            None,
            Some(vec!["personal".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        // Should have auto-added "contact" tag
        assert!(rec.tags.contains(&"contact".to_string()));
        Ok(())
    }

    #[test]
    fn test_provenance_stamping() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let rec = aura.store_with_channel(
            "User said hello",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("telegram"),
            None,
            None,
            None,
        )?;

        assert_eq!(rec.metadata.get("source").unwrap(), "user-telegram");
        assert_eq!(rec.metadata.get("verified").unwrap(), "true");
        Ok(())
    }

    #[test]
    fn test_recall_cache() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "Teo loves Rust programming",
            Some(Level::Identity),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        // First recall — cache miss
        let r1 = aura.recall("Who is Teo?", None, None, None, None, None)?;
        // Second recall — cache hit (same result)
        let r2 = aura.recall("Who is Teo?", None, None, None, None, None)?;
        assert_eq!(r1, r2);

        // Store invalidates cache
        aura.store(
            "Teo is 25 years old",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        Ok(())
    }

    #[test]
    fn test_recall_full() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        // Store a normal record
        aura.store(
            "Teo loves Rust programming",
            Some(Level::Domain),
            Some(vec!["fact".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        // Store a failure record
        aura.store(
            "Failed to register on site X: captcha required",
            Some(Level::Decisions),
            Some(vec!["outcome-failure".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        // recall_full should find records matching "Teo Rust"
        let results = aura.recall_full("Teo Rust", None, Some(true), None, None, None, None)?;
        assert!(
            !results.is_empty(),
            "recall_full should find at least one record"
        );

        // recall_full with include_failures=true should find failure records
        let results_fail = aura.recall_full(
            "register site captcha",
            None,
            Some(true),
            None,
            None,
            None,
            None,
        )?;
        let has_failure = results_fail
            .iter()
            .any(|(_, r)| r.tags.contains(&"outcome-failure".to_string()));
        assert!(
            has_failure,
            "recall_full with include_failures=true should find failure records"
        );

        // recall_full with include_failures=false should still work
        let results_no_fail =
            aura.recall_full("register site", None, Some(false), None, None, None, None)?;
        // Should not crash — just verify it returns
        drop(results_no_fail);

        Ok(())
    }

    #[test]
    fn test_taxonomy_config() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let mut taxonomy = TagTaxonomy::default();
        taxonomy.identity_tags.insert("medical-id".into());
        aura.set_taxonomy(taxonomy);

        let tax = aura.get_taxonomy();
        assert!(tax.identity_tags.contains("medical-id"));
        Ok(())
    }

    #[test]
    fn test_run_maintenance() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "test record",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        let report = aura.run_maintenance();
        assert!(report.total_records > 0);
        assert!(!report.timestamp.is_empty());
        assert!(report.hotspots.records_before_cycle > 0);
        assert_eq!(
            report.hotspots.belief_snapshot_records,
            report.hotspots.sdr_vectors_built
        );
        assert_eq!(
            report.hotspots.sdr_vectors_built,
            report.hotspots.sdr_vectors_computed + report.hotspots.sdr_vectors_reused
        );
        assert!(report.feedback.entries.len() >= report.feedback.beliefs_touched);
        if let Some(entry) = report.feedback.entries.first() {
            assert!(entry.volatility_after >= entry.volatility_before);
        }
        assert!(!report.hotspots.dominant_phase.is_empty());
        assert!(report.hotspots.dominant_phase_ms >= 0.0);
        assert!(report.hotspots.dominant_phase_share >= 0.0);
        assert_eq!(report.trend_summary.snapshot_count, 1);
        assert_eq!(report.trend_summary.recent.len(), 1);
        assert!(!report.reflection.digest.is_empty());
        assert!(report.reflection.report.jobs_run >= 1);
        assert_eq!(
            report.trend_summary.recent[0].total_records,
            report.total_records
        );
        assert_eq!(
            aura.get_maintenance_trend_summary().snapshot_count,
            report.trend_summary.snapshot_count
        );

        Ok(())
    }

    #[test]
    fn test_maintenance_trend_history_persists_across_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();

        {
            let aura = Aura::open(root)?;
            aura.store(
                "trend record one",
                None,
                None,
                None,
                None,
                None,
                None,
                Some(false),
                None,
                None,
                None,
            )?;

            let first = aura.run_maintenance();
            let second = aura.run_maintenance();
            assert_eq!(first.trend_summary.snapshot_count, 1);
            assert_eq!(second.trend_summary.snapshot_count, 2);
            assert_eq!(aura.get_maintenance_trend_history().len(), 2);
        }

        let reopened = Aura::open(root)?;
        let summary = reopened.get_maintenance_trend_summary();
        assert_eq!(summary.snapshot_count, 2);
        assert_eq!(summary.recent.len(), 2);
        assert!(!summary.latest_dominant_phase.is_empty());

        Ok(())
    }

    #[test]
    fn test_reflection_summaries_persist_across_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let due = (chrono::Utc::now() - chrono::Duration::days(2)).to_rfc3339();

        {
            let aura = Aura::open(root)?;
            aura.store(
                "Follow up with operator review",
                Some(Level::Working),
                Some(vec!["scheduled-task".into()]),
                None,
                None,
                None,
                Some(HashMap::from([
                    ("status".to_string(), "active".to_string()),
                    ("due_date".to_string(), due.clone()),
                ])),
                Some(false),
                None,
                None,
                None,
            )?;

            let report = aura.run_maintenance();
            assert!(report
                .reflection
                .findings
                .iter()
                .any(|finding| finding.kind == "repeated_blocker"));
            assert_eq!(aura.get_reflection_summaries(Some(8)).len(), 1);
        }

        let reopened = Aura::open(root)?;
        let latest = reopened
            .get_latest_reflection_digest()
            .expect("latest reflection digest should persist");
        assert!(latest
            .findings
            .iter()
            .any(|finding| finding.kind == "repeated_blocker"));
        assert_eq!(reopened.get_reflection_summaries(Some(8)).len(), 1);

        Ok(())
    }

    #[test]
    fn test_reflection_digest_aggregates_recent_summaries() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let due = (chrono::Utc::now() - chrono::Duration::days(2)).to_rfc3339();
        let aura = Aura::open(root)?;

        aura.store(
            "Resolve overdue deployment checklist",
            Some(Level::Working),
            Some(vec!["scheduled-task".into(), "deploy".into()]),
            None,
            None,
            None,
            Some(HashMap::from([
                ("status".to_string(), "active".to_string()),
                ("due_date".to_string(), due.clone()),
            ])),
            Some(false),
            None,
            Some("ops"),
            None,
        )?;
        let _ = aura.run_maintenance();

        aura.store(
            "Deploy risk remains unresolved",
            Some(Level::Working),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("ops"),
            Some("fact"),
        )?;
        aura.store(
            "Deploy path looks safe",
            Some(Level::Working),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("ops"),
            Some("fact"),
        )?;
        let _ = aura.run_maintenance();

        let digest = aura.get_reflection_digest(Some(8));
        assert_eq!(digest.summary_count, 2);
        assert!(digest.total_findings >= 2);
        assert!(digest.high_severity_findings >= 1);
        assert!(!digest.kinds.is_empty());
        assert!(digest
            .kinds
            .iter()
            .any(|kind| kind.kind == "repeated_blocker"));
        assert!(digest.namespaces.iter().any(|namespace| namespace == "ops"));
        assert!(!digest.top_findings.is_empty());

        let health = aura.get_memory_health_digest(Some(8));
        assert!(health.reflection_summary_count >= 2);
        assert!(health.reflection_high_severity_findings >= 1);

        Ok(())
    }

    #[test]
    fn test_run_maintenance_updates_epistemic_signals() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let id1 = aura
            .store(
                "Deploy to staging before production deploys",
                Some(Level::Domain),
                Some(vec!["deploy".into(), "safety".into()]),
                None,
                None,
                None,
                None,
                Some(false),
                None,
                None,
                Some("decision"),
            )?
            .id;
        let _id2 = aura
            .store(
                "Always use staging for safe production deploys",
                Some(Level::Domain),
                Some(vec!["deploy".into(), "safety".into()]),
                None,
                None,
                None,
                None,
                Some(false),
                None,
                None,
                Some("decision"),
            )?
            .id;
        let _id3 = aura
            .store(
                "Skip staging when shipping directly to production",
                Some(Level::Working),
                Some(vec!["deploy".into(), "safety".into()]),
                None,
                None,
                None,
                None,
                Some(false),
                None,
                None,
                Some("contradiction"),
            )?
            .id;

        let report = aura.run_maintenance();
        let rec = aura.get(&id1).unwrap();

        assert!(report.epistemic.updated_records > 0);
        assert!(report.epistemic.total_support_links > 0);
        assert!(report.epistemic.total_conflict_links > 0);
        assert!(rec.support_mass > 0);
        assert!(rec.conflict_mass > 0);
        Ok(())
    }

    #[test]
    fn test_user_profile() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let mut fields = HashMap::new();
        fields.insert("name".into(), "Teo".into());
        fields.insert("age".into(), "25".into());
        fields.insert("city".into(), "Kyiv".into());

        aura.store_user_profile(fields)?;

        let profile = aura.get_user_profile();
        assert!(profile.is_some());
        let profile = profile.unwrap();
        assert_eq!(profile.get("name").unwrap(), "Teo");

        Ok(())
    }

    #[test]
    fn test_deterministic_family_relation_links_to_user_profile() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let mut fields = HashMap::new();
        fields.insert("name".into(), "Teo".into());
        let profile = aura.store_user_profile(fields)?;

        let brother = aura.store(
            "My brother Andriy works as a doctor.",
            Some(Level::Identity),
            Some(vec!["family".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;

        let relations = aura.get_structural_relations_for_record(&brother.id, Some(8));
        assert_eq!(relations.len(), 1);
        let relation = &relations[0];
        assert_eq!(relation.source_record_id, profile.id);
        assert_eq!(relation.target_record_id, brother.id);
        assert_eq!(relation.relation_type, "family.brother");
        assert!((relation.weight - relation::STRUCTURAL_FAMILY_WEIGHT).abs() < 0.001);

        let refreshed_profile = aura.get(&profile.id).unwrap();
        assert_eq!(
            refreshed_profile.connection_type(&brother.id),
            Some("family.brother")
        );

        let refreshed_brother = aura.get(&brother.id).unwrap();
        assert_eq!(
            refreshed_brother.connection_type(&profile.id),
            Some("family.brother")
        );

        Ok(())
    }

    #[test]
    fn test_deterministic_family_relation_stays_in_namespace() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let profile_alpha = aura.store(
            "User Profile:\n  name: Alpha",
            Some(Level::Identity),
            Some(vec![identity::PROFILE_TAG.into()]),
            Some(true),
            None,
            None,
            None,
            Some(false),
            None,
            Some("alpha"),
            None,
        )?;
        let profile_beta = aura.store(
            "User Profile:\n  name: Beta",
            Some(Level::Identity),
            Some(vec![identity::PROFILE_TAG.into()]),
            Some(true),
            None,
            None,
            None,
            Some(false),
            None,
            Some("beta"),
            None,
        )?;

        let brother_alpha = aura.store(
            "My brother Mark lives in Warsaw.",
            Some(Level::Identity),
            Some(vec!["family".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            Some("alpha"),
            Some("fact"),
        )?;
        let sister_beta = aura.store(
            "My sister Anna studies chemistry.",
            Some(Level::Identity),
            Some(vec!["family".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            Some("beta"),
            Some("fact"),
        )?;

        let alpha_relations = aura.get_structural_relations_for_record(&brother_alpha.id, Some(8));
        assert_eq!(alpha_relations.len(), 1);
        assert_eq!(alpha_relations[0].source_record_id, profile_alpha.id);
        assert_eq!(alpha_relations[0].target_record_id, brother_alpha.id);

        let beta_relations = aura.get_structural_relations_for_record(&sister_beta.id, Some(8));
        assert_eq!(beta_relations.len(), 1);
        assert_eq!(beta_relations[0].source_record_id, profile_beta.id);
        assert_eq!(beta_relations[0].target_record_id, sister_beta.id);

        let all_relations = aura.get_structural_relations(Some(8));
        assert_eq!(all_relations.len(), 2);
        assert!(all_relations.iter().all(|rel| {
            (rel.namespace == "alpha" && rel.source_record_id == profile_alpha.id)
                || (rel.namespace == "beta" && rel.source_record_id == profile_beta.id)
        }));

        Ok(())
    }

    #[test]
    fn test_get_family_graph_and_recall_family_context() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let mut fields = HashMap::new();
        fields.insert("name".into(), "Teo".into());
        let profile = aura.store_user_profile(fields)?;

        let brother = aura.store(
            "My brother Andriy works as a doctor and likes cycling.",
            Some(Level::Identity),
            Some(vec!["family".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;
        let sister = aura.store(
            "My sister Anna studies chemistry and biology.",
            Some(Level::Identity),
            Some(vec!["family".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;
        aura.store(
            "Project release note about canary rollout.",
            Some(Level::Domain),
            Some(vec!["project-note".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;

        let graph = aura
            .get_family_graph(None)
            .expect("family graph should exist");
        assert_eq!(graph.profile_record_id, profile.id);
        assert_eq!(graph.relation_count, 2);
        assert_eq!(graph.relation_types.get("family.brother"), Some(&1));
        assert_eq!(graph.relation_types.get("family.sister"), Some(&1));
        assert!(graph
            .members
            .iter()
            .any(|member| member.record_id == brother.id));
        assert!(graph
            .members
            .iter()
            .any(|member| member.record_id == sister.id));

        let by_record = aura
            .get_family_graph_for_record(&brother.id)
            .expect("record-scoped family graph should exist");
        assert_eq!(by_record.profile_record_id, profile.id);
        assert_eq!(by_record.relation_count, graph.relation_count);

        let results = aura.recall_family_context(
            "doctor cycling",
            Some(5),
            Some(0.1),
            Some(true),
            None,
            None,
        )?;
        assert!(!results.is_empty());
        assert!(results.iter().all(|(_, rec)| {
            rec.id == profile.id || rec.id == brother.id || rec.id == sister.id
        }));
        assert!(results.iter().any(|(_, rec)| rec.id == brother.id));

        Ok(())
    }

    #[test]
    fn test_get_person_digest_and_recall_person_context() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let mut fields = HashMap::new();
        fields.insert("name".into(), "Teo".into());
        let profile = aura.store_user_profile(fields)?;

        let brother = aura.store(
            "My brother Andriy works as a doctor and likes cycling.",
            Some(Level::Identity),
            Some(vec!["family".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;
        let sister = aura.store(
            "My sister Anna studies chemistry and biology.",
            Some(Level::Identity),
            Some(vec!["family".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;

        let digest = aura
            .get_person_digest(&brother.id)
            .expect("person digest should exist");
        assert_eq!(digest.profile_record_id, profile.id);
        assert_eq!(digest.person_record_id, brother.id);
        assert_eq!(digest.relation_type, "family.brother");
        assert!(digest.person_tags.contains(&"family".to_string()));
        assert!(digest.person_content.contains("Andriy"));

        let results = aura.recall_person_context(
            &brother.id,
            "doctor cycling",
            Some(5),
            Some(0.1),
            Some(true),
            None,
        )?;
        assert!(!results.is_empty());
        assert!(results
            .iter()
            .all(|(_, rec)| rec.id == profile.id || rec.id == brother.id));
        assert!(results.iter().any(|(_, rec)| rec.id == brother.id));
        assert!(!results.iter().any(|(_, rec)| rec.id == sister.id));

        Ok(())
    }

    #[test]
    fn test_store_family_person_helper_writes_explicit_relation_contract() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let mut fields = HashMap::new();
        fields.insert("name".into(), "Teo".into());
        let profile = aura.store_user_profile(fields)?;

        let brother = aura.store_family_person(
            "family.brother",
            "Andriy works as a doctor and likes cycling.",
            Some(HashMap::from([("name".to_string(), "Andriy".to_string())])),
            None,
        )?;

        assert!(brother.tags.contains(&"family".to_string()));
        assert_eq!(
            brother.metadata.get("family_relation"),
            Some(&"family.brother".to_string())
        );
        assert_eq!(brother.metadata.get("profile_record_id"), Some(&profile.id));

        let digest = aura
            .get_person_digest(&brother.id)
            .expect("person digest should exist");
        assert_eq!(digest.relation_type, "family.brother");
        assert_eq!(digest.profile_record_id, profile.id);

        let relations = aura.get_structural_relations_for_record(&brother.id, Some(8));
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].source_record_id, profile.id);
        assert_eq!(relations[0].target_record_id, brother.id);
        assert_eq!(relations[0].relation_type, "family.brother");

        Ok(())
    }

    #[test]
    fn test_persona() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let persona = AgentPersona {
            name: "Remy".into(),
            role: "health assistant".into(),
            tone: "warm and caring".into(),
            traits: crate::identity::PersonaTraits {
                warmth: 0.9,
                humor: 0.7,
                ..Default::default()
            },
            ..Default::default()
        };

        aura.set_persona(persona)?;

        let loaded = aura.get_persona();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.name, "Remy");
        assert_eq!(loaded.traits.warmth, 0.9);

        Ok(())
    }

    #[test]
    fn test_research_lifecycle() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("GRPO for memory ranking", Some("standard"));
        assert_eq!(project.queries.len(), 4);

        aura.add_research_finding(
            &project.id,
            "GRPO paper",
            "Published by DeepSeek in 2024...",
            Some("https://arxiv.org/paper/123"),
        )?;

        let rec = aura.complete_research(
            &project.id,
            Some("GRPO is a group-relative optimization...".into()),
        )?;
        assert!(rec.content.contains("GRPO"));
        assert!(rec.tags.contains(&"research-report".to_string()));
        assert_eq!(rec.metadata.get("project_id"), Some(&project.id));

        let project_anchor = aura
            .search(
                None,
                Some(Level::Domain),
                Some(vec!["research-project".into()]),
                Some(10),
                None,
                None,
                None,
                None,
            )
            .into_iter()
            .find(|r| r.metadata.get("project_id") == Some(&project.id))
            .expect("research project anchor should exist");
        assert_eq!(
            project_anchor.metadata.get("project_topic"),
            Some(&project.topic)
        );

        let relations = aura.get_structural_relations_for_record(&rec.id, Some(8));
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].source_record_id, project_anchor.id);
        assert_eq!(relations[0].target_record_id, rec.id);
        assert_eq!(
            relations[0].relation_type,
            relation::PROJECT_MEMBERSHIP_RELATION
        );

        Ok(())
    }

    #[test]
    fn test_scheduled_task_links_to_project_by_explicit_project_id() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Memory safety rollout", Some("standard"));
        let project_anchor = aura.ensure_research_project_anchor(&project)?;

        let task = aura.store(
            "Prepare rollout checklist for tomorrow morning.",
            Some(Level::Working),
            Some(vec!["scheduled-task".into()]),
            Some(false),
            None,
            Some("recorded"),
            Some(HashMap::from([
                ("project_id".to_string(), project.id.clone()),
                ("due_date".to_string(), "tomorrow".to_string()),
            ])),
            Some(false),
            None,
            None,
            Some("decision"),
        )?;

        let relations = aura.get_structural_relations_for_record(&task.id, Some(8));
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].source_record_id, project_anchor.id);
        assert_eq!(relations[0].target_record_id, task.id);
        assert_eq!(
            relations[0].relation_type,
            relation::PROJECT_MEMBERSHIP_RELATION
        );
        assert!((relations[0].weight - relation::STRUCTURAL_PROJECT_WEIGHT).abs() < 0.001);

        let refreshed_project = aura.get(&project_anchor.id).unwrap();
        assert_eq!(
            refreshed_project.connection_type(&task.id),
            Some(relation::PROJECT_MEMBERSHIP_RELATION)
        );

        let refreshed_task = aura.get(&task.id).unwrap();
        assert_eq!(
            refreshed_task.connection_type(&project_anchor.id),
            Some(relation::PROJECT_MEMBERSHIP_RELATION)
        );

        Ok(())
    }

    #[test]
    fn test_link_records_creates_general_typed_relation() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project_note = aura.store(
            "Release note about the Aura rollout plan.",
            Some(Level::Domain),
            Some(vec!["project-note".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;
        let task = aura.store(
            "Prepare rollout checklist for next release.",
            Some(Level::Working),
            Some(vec!["task".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("decision"),
        )?;

        let edge = aura.link_records(&project_note.id, &task.id, "supports.task", Some(0.88))?;
        assert_eq!(edge.source_record_id, project_note.id);
        assert_eq!(edge.target_record_id, task.id);
        assert_eq!(edge.relation_type, "supports.task");
        assert!(!edge.structural);

        let note = aura.get(&project_note.id).unwrap();
        let linked_task = aura.get(&task.id).unwrap();
        assert_eq!(note.connection_type(&task.id), Some("supports.task"));
        assert_eq!(
            linked_task.connection_type(&project_note.id),
            Some("supports.task")
        );

        let relations = aura.get_relations_for_record(&project_note.id, Some(8));
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation_type, "supports.task");
        assert_eq!(relations[0].source_record_id, project_note.id);
        assert_eq!(relations[0].target_record_id, task.id);

        Ok(())
    }

    #[test]
    fn test_link_records_rejects_cross_namespace_links() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let alpha = aura.store(
            "Alpha record",
            Some(Level::Domain),
            None,
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            Some("alpha"),
            Some("fact"),
        )?;
        let beta = aura.store(
            "Beta record",
            Some(Level::Domain),
            None,
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            Some("beta"),
            Some("fact"),
        )?;

        let err = aura
            .link_records(&alpha.id, &beta.id, "related.topic", Some(0.7))
            .expect_err("cross-namespace links should fail");
        assert!(err
            .to_string()
            .contains("Cannot link records across namespaces"));

        Ok(())
    }

    #[test]
    fn test_get_relation_digest_and_recall_relation_context() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let note = aura.store(
            "Release note about the Aura rollout plan.",
            Some(Level::Domain),
            Some(vec!["project-note".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;
        let task = aura.store(
            "Prepare rollout checklist for next release.",
            Some(Level::Working),
            Some(vec!["task".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("decision"),
        )?;
        let unrelated = aura.store(
            "Unrelated medical note about hydration.",
            Some(Level::Domain),
            Some(vec!["health".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;

        aura.link_records(&note.id, &task.id, "supports.task", Some(0.88))?;

        let digest = aura
            .get_relation_digest(&note.id, Some(8))
            .expect("relation digest should exist");
        assert_eq!(digest.anchor_record_id, note.id);
        assert_eq!(digest.relation_count, 1);
        assert_eq!(digest.structural_relations, 0);
        assert_eq!(digest.non_structural_relations, 1);
        assert_eq!(digest.relation_types.get("supports.task"), Some(&1));
        assert_eq!(digest.linked_record_ids, vec![task.id.clone()]);

        let results = aura.recall_relation_context(
            &note.id,
            "rollout checklist",
            Some(5),
            Some(0.1),
            Some(true),
            None,
            Some(8),
        )?;
        assert!(!results.is_empty());
        assert!(results
            .iter()
            .all(|(_, rec)| rec.id == note.id || rec.id == task.id));
        assert!(results.iter().any(|(_, rec)| rec.id == task.id));
        assert!(!results.iter().any(|(_, rec)| rec.id == unrelated.id));

        Ok(())
    }

    #[test]
    fn test_get_entity_digest_and_recall_entity_context() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Entity anchor rollout", Some("quick"));
        let report = aura.complete_research(&project.id, Some("Entity summary".into()))?;
        let task = aura.store_project_task(
            &project.id,
            "Prepare entity checklist",
            Some("2026-03-14"),
            None,
        )?;
        let note = aura.store_project_note(&project.id, "Entity rollout note", None)?;
        let other = aura.store(
            "Unrelated health note about hydration.",
            Some(Level::Domain),
            Some(vec!["health".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;

        let entity_id = format!("project:{}", project.id);
        let digest = aura
            .get_entity_digest(&entity_id)
            .expect("entity digest should exist");
        assert_eq!(digest.entity_id, entity_id);
        assert_eq!(digest.record_ids.len(), 4);
        let project_graph = aura
            .get_project_graph(&project.id)
            .expect("project graph should exist");
        assert!(digest.record_ids.contains(&project_graph.project_record_id));
        assert!(digest.record_ids.contains(&report.id));
        assert!(digest.record_ids.contains(&task.id));
        assert!(digest.record_ids.contains(&note.id));
        assert_eq!(digest.tags.get("research-project"), Some(&1));
        assert_eq!(digest.tags.get("research-report"), Some(&1));
        assert_eq!(digest.tags.get("scheduled-task"), Some(&1));
        assert_eq!(digest.tags.get("project-note"), Some(&1));

        let by_record = aura
            .get_entity_digest_for_record(&task.id)
            .expect("record-scoped entity digest should exist");
        assert_eq!(by_record.entity_id, digest.entity_id);

        let results = aura.recall_entity_context(
            &digest.entity_id,
            "entity checklist",
            Some(5),
            Some(0.1),
            Some(true),
            None,
        )?;
        assert!(!results.is_empty());
        assert!(results
            .iter()
            .all(|(_, rec)| rec.metadata.get("entity_id") == Some(&digest.entity_id)));
        assert!(!results.iter().any(|(_, rec)| rec.id == other.id));

        let by_record_results = aura.recall_entity_context_for_record(
            &note.id,
            "rollout note",
            Some(5),
            Some(0.1),
            Some(true),
            None,
        )?;
        assert!(by_record_results.iter().any(|(_, rec)| rec.id == note.id));

        Ok(())
    }

    #[test]
    fn test_link_entities_and_recall_entity_graph_context() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Entity graph rollout", Some("quick"));
        let task = aura.store_project_task(
            &project.id,
            "Prepare entity graph checklist",
            Some("2026-03-14"),
            None,
        )?;
        let person = aura.store_family_person(
            "family.brother",
            "Andriy owns deployment checklist knowledge.",
            Some(HashMap::from([("name".to_string(), "Andriy".to_string())])),
            None,
        )?;
        let other = aura.store(
            "Unrelated finance note about invoices.",
            Some(Level::Domain),
            Some(vec!["finance".into()]),
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            None,
            Some("fact"),
        )?;

        let project_entity = format!("project:{}", project.id);
        let person_entity = aura
            .get(&person.id)
            .and_then(|rec| rec.metadata.get("entity_id").cloned())
            .expect("person entity_id should exist");

        let edge = aura.link_entities(
            &project_entity,
            &person_entity,
            "owner.collaborates",
            Some(0.91),
        )?;
        assert_eq!(edge.source_entity_id, project_entity);
        assert_eq!(edge.target_entity_id, person_entity);
        assert_eq!(edge.relation_type, "owner.collaborates");

        let entity_edges = aura.get_entity_relations(&edge.source_entity_id, Some(8));
        assert_eq!(entity_edges.len(), 1);
        assert_eq!(entity_edges[0].target_entity_id, edge.target_entity_id);

        let results = aura.recall_entity_graph_context(
            &edge.source_entity_id,
            "deployment checklist",
            Some(5),
            Some(0.1),
            Some(true),
            None,
            Some(8),
        )?;
        assert!(!results.is_empty());
        assert!(results.iter().any(|(_, rec)| rec.id == task.id));
        assert!(results.iter().any(|(_, rec)| rec.id == person.id));
        assert!(!results.iter().any(|(_, rec)| rec.id == other.id));

        Ok(())
    }

    #[test]
    fn test_link_records_promotes_strong_cross_entity_edge_to_anchors() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Anchor promotion rollout", Some("quick"));
        let task = aura.store_project_task(
            &project.id,
            "Prepare deployment checklist for the rollout.",
            Some("2026-03-14"),
            None,
        )?;
        let person = aura.store_family_person(
            "family.brother",
            "Andriy owns deployment checklist knowledge.",
            Some(HashMap::from([("name".to_string(), "Andriy".to_string())])),
            None,
        )?;

        aura.link_records(&task.id, &person.id, "supports.person", Some(0.91))?;

        let project_graph = aura
            .get_project_graph(&project.id)
            .expect("project graph should exist");
        let anchor_digest = aura
            .get_relation_digest(&project_graph.project_record_id, Some(8))
            .expect("anchor digest should exist");
        assert!(anchor_digest
            .edges
            .iter()
            .any(|edge| edge.target_record_id == person.id
                && edge.relation_type == "supports.person"
                && edge.weight >= 0.91));

        let project_entity = format!("project:{}", project.id);
        let entity_edges = aura.get_entity_relations(&project_entity, Some(8));
        assert_eq!(entity_edges.len(), 1);
        assert_eq!(entity_edges[0].relation_type, "supports.person");
        assert_eq!(
            entity_edges[0].source_record_id,
            project_graph.project_record_id
        );
        assert_eq!(entity_edges[0].target_record_id, person.id);

        Ok(())
    }

    #[test]
    fn test_get_entity_graph_digest_aggregates_direct_neighbors() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Entity graph digest rollout", Some("quick"));
        let project_entity = format!("project:{}", project.id);
        let task = aura.store_project_task(
            &project.id,
            "Prepare shared deployment checklist.",
            Some("2026-03-14"),
            None,
        )?;
        let person = aura.store_family_person(
            "family.brother",
            "Andriy owns deployment checklist knowledge.",
            Some(HashMap::from([("name".to_string(), "Andriy".to_string())])),
            None,
        )?;
        let teammate = aura.store(
            "Teammate note for rollout approvals.",
            Some(Level::Identity),
            Some(vec!["contact".into()]),
            Some(false),
            None,
            None,
            Some(HashMap::from([(
                "entity_id".to_string(),
                "person:teammate:olena".to_string(),
            )])),
            Some(false),
            None,
            None,
            Some("fact"),
        )?;

        aura.link_records(&task.id, &person.id, "supports.person", Some(0.91))?;
        aura.link_records(&task.id, &teammate.id, "works_with", Some(0.84))?;

        let digest = aura
            .get_entity_graph_digest(&project_entity, Some(8))
            .expect("entity graph digest should exist");
        assert_eq!(digest.entity.entity_id, project_entity);
        assert_eq!(digest.neighbor_count, 2);
        assert_eq!(digest.relation_types.get("supports.person"), Some(&1));
        assert_eq!(digest.relation_types.get("works_with"), Some(&1));
        assert_eq!(digest.edges.len(), 2);
        assert_eq!(digest.neighbors.len(), 2);
        assert_eq!(digest.neighbors[0].relation_count, 1);
        assert!(digest.neighbors[0].strongest_weight >= digest.neighbors[1].strongest_weight);
        assert!(digest.neighbors.iter().any(|neighbor| neighbor.entity_id
            == "person:family.brother:andriy"
            && neighbor.relation_types.get("supports.person") == Some(&1)));
        assert!(digest
            .neighbors
            .iter()
            .any(|neighbor| neighbor.entity_id == "person:teammate:olena"
                && neighbor.relation_types.get("works_with") == Some(&1)));

        let by_record = aura
            .get_entity_graph_digest_for_record(&task.id, Some(8))
            .expect("record-scoped entity graph digest should exist");
        assert_eq!(by_record.entity.entity_id, digest.entity.entity_id);
        assert_eq!(by_record.neighbor_count, digest.neighbor_count);

        Ok(())
    }

    // ── get_entity_graph_neighbors ──────────────────────────────────────────────

    /// Helper that builds the three-entity project graph used by neighbor tests.
    fn build_neighbor_test_graph() -> Result<(
        tempfile::TempDir,
        Aura,
        String, // project_entity
        String, // person entity_id
        String, // teammate entity_id
    )> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;
        let project = aura.start_research("Neighbor test project", Some("quick"));
        let project_entity = format!("project:{}", project.id);
        let task = aura.store_project_task(
            &project.id,
            "Deploy checklist for neighbor test.",
            Some("2026-03-14"),
            None,
        )?;
        let person = aura.store_family_person(
            "family.sibling",
            "Olena owns deployment knowledge.",
            Some(HashMap::from([("name".to_string(), "Olena".to_string())])),
            None,
        )?;
        let teammate = aura.store(
            "Teammate note for neighbor test approvals.",
            Some(Level::Identity),
            Some(vec!["contact".into()]),
            Some(false),
            None,
            None,
            Some(HashMap::from([(
                "entity_id".to_string(),
                "person:teammate:mykola".to_string(),
            )])),
            Some(false),
            None,
            None,
            Some("fact"),
        )?;
        aura.link_records(&task.id, &person.id, "supports.person", Some(0.91))?;
        aura.link_records(&task.id, &teammate.id, "works_with", Some(0.75))?;
        let person_entity = aura
            .get(&person.id)
            .and_then(|r| r.metadata.get("entity_id").cloned())
            .expect("person entity_id");
        Ok((
            dir,
            aura,
            project_entity,
            person_entity,
            "person:teammate:mykola".to_string(),
        ))
    }

    #[test]
    fn test_get_entity_graph_neighbors_returns_all_by_default() -> Result<()> {
        let (_dir, aura, project_entity, _, _) = build_neighbor_test_graph()?;
        let neighbors = aura.get_entity_graph_neighbors(&project_entity, None, None, None, None);
        assert_eq!(neighbors.len(), 2, "expected both neighbors without filter");
        Ok(())
    }

    #[test]
    fn test_get_entity_graph_neighbors_top_n_truncates() -> Result<()> {
        let (_dir, aura, project_entity, _, _) = build_neighbor_test_graph()?;
        let neighbors = aura.get_entity_graph_neighbors(&project_entity, Some(1), None, None, None);
        assert_eq!(neighbors.len(), 1);
        // strongest weight neighbor should be first (supports.person @ 0.91 > works_with @ 0.75)
        assert!(
            neighbors[0].entity.entity_id.contains("sibling")
                || neighbors[0].entity.entity_id.contains("Olena")
                || neighbors[0].entity.entity_id.contains("olena"),
            "first neighbor should be the highest-weight one, got {}",
            neighbors[0].entity.entity_id
        );
        Ok(())
    }

    #[test]
    fn test_get_entity_graph_neighbors_min_weight_filters() -> Result<()> {
        let (_dir, aura, project_entity, _, _) = build_neighbor_test_graph()?;
        // 0.91 passes, 0.75 is below 0.80 cutoff
        let neighbors =
            aura.get_entity_graph_neighbors(&project_entity, None, Some(0.80), None, None);
        assert_eq!(neighbors.len(), 1);
        assert!(
            neighbors[0].entity.entity_id.contains("sibling")
                || neighbors[0].entity.entity_id.contains("olena"),
            "only the supports.person neighbor should survive; got {}",
            neighbors[0].entity.entity_id
        );
        Ok(())
    }

    #[test]
    fn test_get_entity_graph_neighbors_relation_type_filter() -> Result<()> {
        let (_dir, aura, project_entity, _, _) = build_neighbor_test_graph()?;
        let neighbors =
            aura.get_entity_graph_neighbors(&project_entity, None, None, Some("works_with"), None);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].entity.entity_id, "person:teammate:mykola");
        Ok(())
    }

    #[test]
    fn test_get_entity_graph_neighbors_each_result_is_valid_digest() -> Result<()> {
        let (_dir, aura, project_entity, _, _) = build_neighbor_test_graph()?;
        let neighbors = aura.get_entity_graph_neighbors(&project_entity, None, None, None, None);
        for n in &neighbors {
            assert!(!n.entity.entity_id.is_empty());
            assert!(!n.anchor_record_id.is_empty());
            // each returned digest is valid entity from anchor entity's perspective
        }
        Ok(())
    }

    #[test]
    fn test_get_entity_graph_neighbors_unknown_entity_returns_empty() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;
        let neighbors =
            aura.get_entity_graph_neighbors("entity:does:not:exist", None, None, None, None);
        assert!(neighbors.is_empty());
        Ok(())
    }

    #[test]
    fn test_store_project_task_helper_writes_contract_metadata() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Contract enforcement", Some("quick"));
        let task = aura.store_project_task(
            &project.id,
            "Write rollout note for the team.",
            Some("2026-03-14"),
            Some(HashMap::from([("owner".to_string(), "ops".to_string())])),
        )?;

        assert!(task.tags.contains(&"scheduled-task".to_string()));
        assert_eq!(task.metadata.get("project_id"), Some(&project.id));
        assert_eq!(
            task.metadata.get("due_date"),
            Some(&"2026-03-14".to_string())
        );
        assert_eq!(task.metadata.get("owner"), Some(&"ops".to_string()));

        let relations = aura.get_structural_relations_for_record(&task.id, Some(8));
        assert_eq!(relations.len(), 1);
        assert_eq!(
            relations[0].relation_type,
            relation::PROJECT_MEMBERSHIP_RELATION
        );

        Ok(())
    }

    #[test]
    fn test_store_project_todo_helper_writes_contract_metadata() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Todo contract", Some("quick"));
        let todo = aura.store_project_todo(
            &project.id,
            "Document the rollout edge cases.",
            Some(HashMap::from([(
                "priority".to_string(),
                "high".to_string(),
            )])),
        )?;

        assert!(todo.tags.contains(&"todo-item".to_string()));
        assert_eq!(todo.metadata.get("project_id"), Some(&project.id));
        assert_eq!(todo.metadata.get("priority"), Some(&"high".to_string()));

        let relations = aura.get_structural_relations_for_record(&todo.id, Some(8));
        assert_eq!(relations.len(), 1);
        assert_eq!(
            relations[0].relation_type,
            relation::PROJECT_MEMBERSHIP_RELATION
        );

        Ok(())
    }

    #[test]
    fn test_store_project_note_helper_writes_contract_metadata() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Project note contract", Some("quick"));
        let note = aura.store_project_note(
            &project.id,
            "The rollout must stay inspect-only for this tenant.",
            Some(HashMap::from([(
                "kind".to_string(),
                "constraint".to_string(),
            )])),
        )?;

        assert!(note.tags.contains(&"project-note".to_string()));
        assert_eq!(note.metadata.get("project_id"), Some(&project.id));
        assert_eq!(note.metadata.get("kind"), Some(&"constraint".to_string()));

        let relations = aura.get_structural_relations_for_record(&note.id, Some(8));
        assert_eq!(relations.len(), 1);
        assert_eq!(
            relations[0].relation_type,
            relation::PROJECT_MEMBERSHIP_RELATION
        );

        Ok(())
    }

    #[test]
    fn test_get_project_graph_collects_project_members() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Project graph view", Some("quick"));
        let report = aura.complete_research(&project.id, Some("Graph summary".into()))?;
        let task = aura.store_project_task(
            &project.id,
            "Prepare rollout checklist",
            Some("2026-03-14"),
            None,
        )?;
        let todo = aura.store_project_todo(
            &project.id,
            "Document edge cases",
            Some(HashMap::from([(
                "priority".to_string(),
                "high".to_string(),
            )])),
        )?;
        let note = aura.store_project_note(
            &project.id,
            "Keep rollout inspect-only",
            Some(HashMap::from([(
                "kind".to_string(),
                "constraint".to_string(),
            )])),
        )?;

        let snapshot = aura
            .get_project_graph(&project.id)
            .expect("project graph should exist");
        assert_eq!(snapshot.project_id, project.id);
        assert_eq!(snapshot.project_topic, project.topic);
        assert_eq!(snapshot.relation_count, 4);
        assert_eq!(snapshot.member_record_ids.len(), 4);
        assert!(snapshot.member_record_ids.contains(&report.id));
        assert!(snapshot.member_record_ids.contains(&task.id));
        assert!(snapshot.member_record_ids.contains(&todo.id));
        assert!(snapshot.member_record_ids.contains(&note.id));
        assert_eq!(snapshot.member_tags.get("research-report"), Some(&1));
        assert_eq!(snapshot.member_tags.get("scheduled-task"), Some(&1));
        assert_eq!(snapshot.member_tags.get("todo-item"), Some(&1));
        assert_eq!(snapshot.member_tags.get("project-note"), Some(&1));

        let by_record = aura
            .get_project_graph_for_record(&task.id)
            .expect("record-scoped project graph should exist");
        assert_eq!(by_record.project_record_id, snapshot.project_record_id);
        assert_eq!(by_record.member_record_ids, snapshot.member_record_ids);

        Ok(())
    }

    #[test]
    fn test_recall_project_context_stays_bounded_to_project_records() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let alpha = aura.start_research("Alpha rollout", Some("quick"));
        aura.store_project_task(
            &alpha.id,
            "Prepare alpha checklist",
            Some("2026-03-14"),
            None,
        )?;
        let alpha_note = aura.store_project_note(
            &alpha.id,
            "Alpha rollback uses canary analysis and staged metrics.",
            None,
        )?;

        let beta = aura.start_research("Beta rollout", Some("quick"));
        let beta_note = aura.store_project_note(
            &beta.id,
            "Beta rollback uses manual review and freeze windows.",
            None,
        )?;

        let results = aura.recall_project_context(
            &alpha.id,
            "rollback canary metrics",
            Some(5),
            Some(0.0),
            Some(true),
            None,
        )?;

        assert!(!results.is_empty());
        assert!(results.iter().all(|(_, rec)| {
            rec.id != beta_note.id && rec.metadata.get("project_id") != Some(&beta.id)
        }));
        assert!(results.iter().any(|(_, rec)| rec.id == alpha_note.id));

        Ok(())
    }

    #[test]
    fn test_get_project_status_counts_project_members() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Status snapshot", Some("quick"));
        let report = aura.complete_research(&project.id, Some("Status summary".into()))?;
        aura.store_project_task(
            &project.id,
            "Prepare checklist",
            Some("2026-03-14"),
            Some(HashMap::from([("status".to_string(), "done".to_string())])),
        )?;
        let todo = aura.store_project_todo(
            &project.id,
            "Document edge cases",
            Some(HashMap::from([
                ("priority".to_string(), "high".to_string()),
                ("completed".to_string(), "false".to_string()),
            ])),
        )?;
        aura.store_project_note(
            &project.id,
            "Keep rollout inspect-only",
            Some(HashMap::from([(
                "kind".to_string(),
                "constraint".to_string(),
            )])),
        )?;

        let status = aura
            .get_project_status(&project.id)
            .expect("project status should exist");
        assert_eq!(status.project_id, project.id);
        assert_eq!(status.project_status, "completed");
        assert_eq!(status.total_members, 4);
        assert_eq!(status.reports, 1);
        assert_eq!(status.scheduled_tasks, 1);
        assert_eq!(status.completed_tasks, 1);
        assert_eq!(status.open_tasks, 0);
        assert_eq!(status.todos, 1);
        assert_eq!(status.open_todos, 1);
        assert_eq!(status.completed_todos, 0);
        assert_eq!(status.notes, 1);
        assert_eq!(status.due_tasks, 1);
        assert_eq!(status.high_priority_todos, 1);

        let by_record = aura
            .get_project_status_for_record(&todo.id)
            .expect("record-scoped status should exist");
        assert_eq!(by_record.project_record_id, status.project_record_id);
        assert_eq!(by_record.total_members, status.total_members);
        assert_eq!(report.metadata.get("project_id"), Some(&project.id));

        Ok(())
    }

    #[test]
    fn test_get_project_timeline_sorts_due_and_flags_overdue() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Timeline snapshot", Some("quick"));
        let overdue_task = aura.store_project_task(
            &project.id,
            "Fix expired checklist item",
            Some("2000-01-01"),
            None,
        )?;
        let upcoming_task = aura.store_project_task(
            &project.id,
            "Prepare today checklist",
            Some(&chrono::Utc::now().date_naive().to_string()),
            None,
        )?;
        let note = aura.store_project_note(&project.id, "Project context note", None)?;

        let timeline = aura
            .get_project_timeline(&project.id)
            .expect("project timeline should exist");
        assert_eq!(timeline.project_id, project.id);
        assert_eq!(timeline.total_entries, 3);
        assert_eq!(timeline.overdue_entries, 1);
        assert!(timeline.upcoming_entries >= 1);
        assert_eq!(timeline.entries[0].record_id, overdue_task.id);
        assert!(timeline.entries[0].overdue);
        assert_eq!(timeline.entries[1].record_id, upcoming_task.id);
        assert!(timeline
            .entries
            .iter()
            .any(|entry| entry.record_id == note.id));

        let by_record = aura
            .get_project_timeline_for_record(&note.id)
            .expect("record-scoped timeline should exist");
        assert_eq!(by_record.project_record_id, timeline.project_record_id);
        assert_eq!(by_record.total_entries, timeline.total_entries);

        Ok(())
    }

    #[test]
    fn test_get_project_digest_combines_graph_status_and_timeline() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let project = aura.start_research("Digest snapshot", Some("quick"));
        aura.complete_research(&project.id, Some("Digest summary".into()))?;
        let task = aura.store_project_task(
            &project.id,
            "Prepare digest checklist",
            Some("2026-03-14"),
            Some(HashMap::from([(
                "status".to_string(),
                "in_progress".to_string(),
            )])),
        )?;
        aura.store_project_todo(
            &project.id,
            "Document digest edge cases",
            Some(HashMap::from([(
                "priority".to_string(),
                "high".to_string(),
            )])),
        )?;
        aura.store_project_note(&project.id, "Digest note", None)?;

        let digest = aura
            .get_project_digest(&project.id)
            .expect("project digest should exist");
        assert_eq!(digest.graph.project_id, project.id);
        assert_eq!(digest.status.project_id, project.id);
        assert_eq!(digest.timeline.project_id, project.id);
        assert_eq!(digest.graph.relation_count, 4);
        assert_eq!(digest.status.total_members, 4);
        assert_eq!(digest.timeline.total_entries, 4);
        assert_eq!(digest.status.reports, 1);
        assert_eq!(digest.status.scheduled_tasks, 1);
        assert_eq!(digest.status.todos, 1);
        assert_eq!(digest.status.notes, 1);
        assert_eq!(
            digest.graph.project_record_id,
            digest.status.project_record_id
        );
        assert_eq!(
            digest.graph.project_record_id,
            digest.timeline.project_record_id
        );

        let by_record = aura
            .get_project_digest_for_record(&task.id)
            .expect("record-scoped digest should exist");
        assert_eq!(by_record.graph.project_id, digest.graph.project_id);
        assert_eq!(by_record.status.total_members, digest.status.total_members);
        assert_eq!(
            by_record.timeline.total_entries,
            digest.timeline.total_entries
        );

        Ok(())
    }

    #[test]
    fn test_circuit_breaker() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        assert!(aura.is_tool_available("web_search"));

        // 3 failures should trip the breaker (default threshold)
        aura.record_tool_failure("web_search");
        aura.record_tool_failure("web_search");
        aura.record_tool_failure("web_search");
        assert!(!aura.is_tool_available("web_search"));

        // Success on a different tool
        aura.record_tool_success("http_get");
        assert!(aura.is_tool_available("http_get"));

        Ok(())
    }

    #[test]
    fn test_credibility() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        assert!(aura.get_credibility("https://arxiv.org/paper/123") > 0.8);
        assert!(aura.get_credibility("https://reddit.com/r/test") < 0.5);

        aura.set_credibility_override("my-company.com", 0.95);
        assert_eq!(aura.get_credibility("https://my-company.com/docs"), 0.95);

        Ok(())
    }

    // ── Two-Tier API Tests ──

    #[test]
    fn test_recall_cognitive() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "session note about testing",
            Some(Level::Working),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "recent decision on architecture",
            Some(Level::Decisions),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "domain fact about Rust language",
            Some(Level::Domain),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "core identity preferences",
            Some(Level::Identity),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        // No query — returns all cognitive records
        let cognitive = aura.recall_cognitive(None, None, None);
        assert_eq!(cognitive.len(), 2);
        assert!(cognitive.iter().all(|r| r.level.is_cognitive()));

        // With query — RRF pipeline, filtered to cognitive tier only
        let filtered = aura.recall_cognitive(Some("session note about testing"), None, None);
        assert!(!filtered.is_empty());
        assert!(filtered.iter().all(|r| r.level.is_cognitive()));
        // Best match should be the session note
        assert_eq!(filtered[0].content, "session note about testing");

        Ok(())
    }

    #[test]
    fn test_recall_core_tier() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "session note about testing",
            Some(Level::Working),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "recent decision on architecture",
            Some(Level::Decisions),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "domain fact about Rust language",
            Some(Level::Domain),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "core identity preferences",
            Some(Level::Identity),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        // No query — returns all core records
        let core = aura.recall_core_tier(None, None, None);
        assert_eq!(core.len(), 2);
        assert!(core.iter().all(|r| r.level.is_core()));

        // With query — RRF pipeline, filtered to core tier only
        let filtered = aura.recall_core_tier(Some("domain fact about Rust language"), None, None);
        assert!(!filtered.is_empty());
        assert!(filtered.iter().all(|r| r.level.is_core()));
        // Best match should be the domain fact
        assert_eq!(filtered[0].content, "domain fact about Rust language");

        Ok(())
    }

    #[test]
    fn test_tier_stats() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "w1",
            Some(Level::Working),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "w2",
            Some(Level::Working),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "d1",
            Some(Level::Decisions),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "dom1",
            Some(Level::Domain),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "id1",
            Some(Level::Identity),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "id2",
            Some(Level::Identity),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        let ts = aura.tier_stats();
        assert_eq!(ts["cognitive_total"], 3);
        assert_eq!(ts["cognitive_working"], 2);
        assert_eq!(ts["cognitive_decisions"], 1);
        assert_eq!(ts["core_total"], 3);
        assert_eq!(ts["core_domain"], 1);
        assert_eq!(ts["core_identity"], 2);
        assert_eq!(ts["total"], 6);

        Ok(())
    }

    #[test]
    fn test_promotion_candidates() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        // Store a working-level record
        let rec = aura.store(
            "frequently recalled fact",
            Some(Level::Working),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        // Simulate frequent recalls to bump activation_count
        {
            let mut records = aura.records.write();
            if let Some(r) = records.get_mut(&rec.id) {
                r.activation_count = 10;
                r.strength = 0.9;
            }
        }

        let candidates = aura.promotion_candidates(None, None);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, rec.id);

        // No candidates with high threshold
        let none = aura.promotion_candidates(Some(20), None);
        assert_eq!(none.len(), 0);

        Ok(())
    }

    #[test]
    fn test_promote_record() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let rec = aura.store(
            "promotable",
            Some(Level::Working),
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        assert_eq!(rec.level, Level::Working);

        let new_level = aura.promote_record(&rec.id);
        assert_eq!(new_level, Some(Level::Decisions));

        let updated = aura.get(&rec.id).unwrap();
        assert_eq!(updated.level, Level::Decisions);

        // Promote again
        let new_level = aura.promote_record(&rec.id);
        assert_eq!(new_level, Some(Level::Domain));

        // Promote to Identity
        let new_level = aura.promote_record(&rec.id);
        assert_eq!(new_level, Some(Level::Identity));

        // Can't promote beyond Identity
        let new_level = aura.promote_record(&rec.id);
        assert_eq!(new_level, None);

        Ok(())
    }

    // ── Namespace Tests ──

    #[test]
    fn test_namespace_isolation_recall() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "User is 25 years old",
            Some(Level::Identity),
            Some(vec!["user".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store_with_channel(
            "Test case: user is 30 years old",
            Some(Level::Identity),
            Some(vec!["user".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            Some("test-data"),
            None,
        )?;

        let results =
            aura.recall_structured("user age", None, None, None, None, Some(&["default"]))?;
        assert!(results.iter().all(|(_, r)| r.namespace == "default"));
        assert!(results.iter().any(|(_, r)| r.content.contains("25")));
        assert!(!results.iter().any(|(_, r)| r.content.contains("30")));

        let results =
            aura.recall_structured("user age", None, None, None, None, Some(&["test-data"]))?;
        assert!(results.iter().all(|(_, r)| r.namespace == "test-data"));

        Ok(())
    }

    #[test]
    fn test_namespace_isolation_search() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "Real data about cats",
            None,
            Some(vec!["animal".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store_with_channel(
            "Test data about cats",
            None,
            Some(vec!["animal".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            Some("sandbox"),
            None,
        )?;

        let results = aura.search(
            None,
            None,
            Some(vec!["animal".into()]),
            None,
            None,
            None,
            Some(&["default"]),
            None,
        );
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Real"));

        let results = aura.search(
            None,
            None,
            Some(vec!["animal".into()]),
            None,
            None,
            None,
            Some(&["sandbox"]),
            None,
        );
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Test"));

        Ok(())
    }

    #[test]
    fn test_list_namespaces() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "A",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store_with_channel(
            "B",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            Some("project-x"),
            None,
        )?;

        let ns = aura.list_namespaces();
        assert!(ns.contains(&"default".to_string()));
        assert!(ns.contains(&"project-x".to_string()));

        Ok(())
    }

    #[test]
    fn test_move_record() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let rec = aura.store(
            "Moveable record content here",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        assert_eq!(rec.namespace, "default");

        let moved = aura.move_record(&rec.id, "archive").unwrap();
        assert_eq!(moved.namespace, "archive");

        let results = aura.search(
            Some("Moveable"),
            None,
            None,
            None,
            None,
            None,
            Some(&["default"]),
            None,
        );
        assert!(results.is_empty());

        let results = aura.search(
            Some("Moveable"),
            None,
            None,
            None,
            None,
            None,
            Some(&["archive"]),
            None,
        );
        assert_eq!(results.len(), 1);

        Ok(())
    }

    #[test]
    fn test_namespace_stats() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "A",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store(
            "B",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store_with_channel(
            "C",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            Some("sandbox"),
            None,
        )?;

        let stats = aura.namespace_stats();
        assert_eq!(*stats.get("default").unwrap_or(&0), 2);
        assert_eq!(*stats.get("sandbox").unwrap_or(&0), 1);

        Ok(())
    }

    #[test]
    fn test_dedup_within_namespace_only() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "The quick brown fox jumps over the lazy dog repeatedly",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        aura.store_with_channel(
            "The quick brown fox jumps over the lazy dog repeatedly",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("sandbox"),
            None,
        )?;

        let stats = aura.namespace_stats();
        assert_eq!(*stats.get("default").unwrap_or(&0), 1);
        assert_eq!(*stats.get("sandbox").unwrap_or(&0), 1);

        Ok(())
    }

    #[test]
    fn test_default_namespace_when_none() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let rec = aura.store(
            "No namespace specified here",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        assert_eq!(rec.namespace, "default");

        // Search without namespace (None defaults to "default")
        let results = aura.search(
            Some("No namespace"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(results.len(), 1);

        Ok(())
    }

    // ── Multi-namespace tests (v1.2.0) ──

    #[test]
    fn test_multi_namespace_recall() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "User health data about blood pressure monitoring",
            None,
            Some(vec!["health".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store_with_channel(
            "Test case health scenario about blood pressure",
            None,
            Some(vec!["health".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            Some("sandbox"),
            None,
        )?;
        aura.store_with_channel(
            "Project health metrics dashboard",
            None,
            Some(vec!["health".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            Some("project-x"),
            None,
        )?;

        // Single namespace — only default
        let results = aura.recall_structured(
            "health blood pressure",
            None,
            None,
            None,
            None,
            Some(&["default"]),
        )?;
        assert!(results.iter().all(|(_, r)| r.namespace == "default"));

        // Multi-namespace — default + sandbox
        let results = aura.recall_structured(
            "health blood pressure",
            None,
            None,
            None,
            None,
            Some(&["default", "sandbox"]),
        )?;
        let found_ns: std::collections::HashSet<String> =
            results.iter().map(|(_, r)| r.namespace.clone()).collect();
        assert!(found_ns.contains("default") || found_ns.contains("sandbox"));
        // project-x should NOT be in results
        assert!(!results.iter().any(|(_, r)| r.namespace == "project-x"));

        Ok(())
    }

    #[test]
    fn test_multi_namespace_search() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "Record A in default",
            None,
            Some(vec!["multi".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;
        aura.store_with_channel(
            "Record B in sandbox",
            None,
            Some(vec!["multi".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            Some("sandbox"),
            None,
        )?;
        aura.store_with_channel(
            "Record C in project",
            None,
            Some(vec!["multi".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
            Some("project-x"),
            None,
        )?;

        // Search across 2 namespaces
        let results = aura.search(
            None,
            None,
            Some(vec!["multi".into()]),
            None,
            None,
            None,
            Some(&["default", "sandbox"]),
            None,
        );
        assert_eq!(results.len(), 2);
        let found_ns: std::collections::HashSet<String> =
            results.iter().map(|r| r.namespace.clone()).collect();
        assert!(found_ns.contains("default"));
        assert!(found_ns.contains("sandbox"));
        assert!(!found_ns.contains("project-x"));

        // Search across all 3
        let results = aura.search(
            None,
            None,
            Some(vec!["multi".into()]),
            None,
            None,
            None,
            Some(&["default", "sandbox", "project-x"]),
            None,
        );
        assert_eq!(results.len(), 3);

        Ok(())
    }

    #[test]
    fn test_single_namespace_slice_backward_compat() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store(
            "Only in default ns content here",
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            None,
            None,
            None,
        )?;

        // Single-element slice should behave same as old Option<&str>
        let results = aura.search(
            Some("Only in default"),
            None,
            None,
            None,
            None,
            None,
            Some(&["default"]),
            None,
        );
        assert_eq!(results.len(), 1);

        // None should also work (defaults to ["default"])
        let results = aura.search(
            Some("Only in default"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(results.len(), 1);

        Ok(())
    }

    #[test]
    fn test_startup_persistence_semantics() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();

        {
            let aura = Aura::open(root)?;

            let belief = crate::belief::Belief::new("default:test:fact".to_string());
            {
                let mut engine = aura.belief_engine.write();
                engine
                    .key_index
                    .insert(belief.key.clone(), belief.id.clone());
                engine.beliefs.insert(belief.id.clone(), belief.clone());
            }
            {
                let engine = aura.belief_engine.read();
                aura.belief_store.save(&engine)?;
            }

            let concept = crate::concept::ConceptCandidate {
                id: Record::generate_id(),
                key: "default:fact:test:concept".to_string(),
                namespace: "default".to_string(),
                semantic_type: "fact".to_string(),
                belief_ids: vec![belief.id.clone()],
                record_ids: vec!["record-1".to_string()],
                core_terms: vec!["test".to_string()],
                shell_terms: vec![],
                tags: vec!["test".to_string()],
                support_mass: 1.0,
                confidence: 0.9,
                stability: 3.0,
                cohesion: 1.0,
                abstraction_score: 0.9,
                state: crate::concept::ConceptState::Stable,
                last_updated: 1.0,
            };
            {
                let mut engine = aura.concept_engine.write();
                engine
                    .key_index
                    .insert(concept.key.clone(), concept.id.clone());
                engine.concepts.insert(concept.id.clone(), concept);
            }
            {
                let engine = aura.concept_engine.read();
                aura.concept_store.save(&engine)?;
            }

            let pattern = crate::causal::CausalPattern {
                id: "ca-pattern-1".to_string(),
                key: "default:cause:effect:edge".to_string(),
                namespace: "default".to_string(),
                cause_belief_ids: vec![belief.id.clone()],
                effect_belief_ids: vec![],
                cause_record_ids: vec!["cause-record".to_string()],
                effect_record_ids: vec!["effect-record".to_string()],
                support_count: 2,
                explicit_support_count: 1,
                temporal_support_count: 1,
                unique_temporal_windows: 1,
                effect_record_signature_variants: 1,
                positive_effect_signals: 1,
                negative_effect_signals: 0,
                counterevidence: 0,
                explicit_support_total_for_cause: 1,
                explicit_effect_variants_for_cause: 1,
                transition_lift: 1.2,
                temporal_consistency: 1.0,
                outcome_stability: 1.0,
                causal_strength: 0.8,
                invalidation_reason: None,
                invalidated_at: None,
                state: crate::causal::CausalState::Stable,
                last_updated: 1.0,
            };
            {
                let mut engine = aura.causal_engine.write();
                engine
                    .key_index
                    .insert(pattern.key.clone(), pattern.id.clone());
                engine.patterns.insert(pattern.id.clone(), pattern.clone());
            }
            {
                let engine = aura.causal_engine.read();
                aura.causal_store.save(&engine)?;
            }

            let hint = crate::policy::PolicyHint {
                id: "policy-hint-1".to_string(),
                key: "default:deploy:verify".to_string(),
                namespace: "default".to_string(),
                domain: "deploy".to_string(),
                action_kind: crate::policy::PolicyActionKind::VerifyFirst,
                recommendation: "Verify deploy health before rollout".to_string(),
                trigger_causal_ids: vec![pattern.id],
                trigger_concept_ids: vec![],
                trigger_belief_ids: vec![belief.id],
                supporting_record_ids: vec![
                    "cause-record".to_string(),
                    "effect-record".to_string(),
                ],
                cause_record_ids: vec!["cause-record".to_string()],
                confidence: 0.8,
                utility_score: 0.7,
                risk_score: 0.2,
                policy_strength: 0.75,
                state: crate::policy::PolicyState::Stable,
                last_updated: 1.0,
            };
            {
                let mut engine = aura.policy_engine.write();
                engine.key_index.insert(hint.key.clone(), hint.id.clone());
                engine.hints.insert(hint.id.clone(), hint);
            }
            {
                let engine = aura.policy_engine.read();
                aura.policy_store.save(&engine)?;
            }
        }

        let reopened = Aura::open(root)?;

        assert_eq!(reopened.belief_engine.read().beliefs.len(), 1);
        assert_eq!(reopened.concept_engine.read().concepts.len(), 1);
        assert_eq!(reopened.causal_engine.read().patterns.len(), 1);
        assert_eq!(reopened.policy_engine.read().hints.len(), 1);

        Ok(())
    }

    #[test]
    fn test_startup_validation_report_captures_recovery_fallbacks() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path();

        std::fs::write(root.join("policies.cog"), b"{ not valid json")?;
        std::fs::write(root.join(MAINTENANCE_TRENDS_FILE), b"{ not valid json")?;
        std::fs::write(root.join(REFLECTION_SUMMARIES_FILE), b"{ not valid json")?;

        let aura = Aura::open(root.to_str().unwrap())?;
        let report = aura.get_startup_validation_report();

        assert!(report.has_recovery_warnings);
        assert!(report.recovered_fallbacks >= 2 || report.missing_fallbacks >= 2);
        assert!(report.events.iter().any(|event| {
            event.surface == "policy" && event.status == "load_error_fallback" && event.recovered
        }));
        assert!(report.events.iter().any(|event| {
            event.surface == "maintenance_trends"
                && event.status == "load_error_fallback"
                && event.recovered
        }));
        assert!(report.events.iter().any(|event| {
            event.surface == "reflection_summaries"
                && event.status == "load_error_fallback"
                && event.recovered
        }));
        assert!(report
            .events
            .iter()
            .any(|event| { event.surface == "concept" && event.status == "missing_fallback" }));
        assert!(aura.policy_engine.read().hints.is_empty());
        assert!(aura.get_maintenance_trend_history().is_empty());
        assert!(aura.get_reflection_summaries(Some(8)).is_empty());

        Ok(())
    }

    #[test]
    fn test_persistence_manifest_is_created_and_normalized_on_open() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path();

        let outdated_manifest = serde_json::json!({
            "schema_version": 0,
            "surfaces": {
                "belief": 0,
                "policy": 0
            }
        });
        std::fs::write(
            root.join(PERSISTENCE_MANIFEST_FILE),
            serde_json::to_vec_pretty(&outdated_manifest)?,
        )?;

        let aura = Aura::open(root.to_str().unwrap())?;
        let manifest = aura.get_persistence_manifest();
        let report = aura.get_startup_validation_report();

        assert_eq!(
            manifest.schema_version,
            PersistenceManifest::current().schema_version
        );
        assert_eq!(
            manifest.surfaces.get("belief").copied(),
            PersistenceManifest::current()
                .surfaces
                .get("belief")
                .copied()
        );
        assert_eq!(
            manifest.surfaces.get("maintenance_trends").copied(),
            PersistenceManifest::current()
                .surfaces
                .get("maintenance_trends")
                .copied()
        );
        assert_eq!(
            manifest.surfaces.get("reflection_summaries").copied(),
            PersistenceManifest::current()
                .surfaces
                .get("reflection_summaries")
                .copied()
        );
        assert!(report.events.iter().any(|event| {
            event.surface == "persistence_manifest" && event.status == "version_mismatch"
        }));

        let persisted: PersistenceManifest =
            serde_json::from_slice(&std::fs::read(root.join(PERSISTENCE_MANIFEST_FILE))?)?;
        assert_eq!(persisted.schema_version, manifest.schema_version);

        Ok(())
    }

    #[test]
    fn test_memory_health_digest_aggregates_operator_review_surfaces() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let record = aura.store(
            "Deploy rollback needs review",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;
        aura.mark_record_salience(&record.id, 0.88, Some("operator_priority"))?;

        let belief_id = "default:belief:health".to_string();
        {
            let mut engine = aura.belief_engine.write();
            let hypothesis = crate::belief::Hypothesis::from_records(&belief_id, &[&record]);
            let mut belief = crate::belief::Belief::new("default:ops:health".into());
            belief.id = belief_id.clone();
            belief.state = crate::belief::BeliefState::Resolved;
            belief.winner_id = Some(hypothesis.id.clone());
            belief.hypothesis_ids = vec![hypothesis.id.clone()];
            belief.score = hypothesis.score;
            belief.confidence = hypothesis.confidence;
            belief.volatility = 0.62;
            belief.stability = 0.40;
            engine
                .key_index
                .insert(belief.key.clone(), belief.id.clone());
            engine
                .record_index
                .insert(record.id.clone(), hypothesis.id.clone());
            engine.hypotheses.insert(hypothesis.id.clone(), hypothesis);
            engine.beliefs.insert(belief.id.clone(), belief);
        }
        {
            let engine = aura.belief_engine.read();
            aura.belief_store.save(&engine)?;
        }

        {
            let mut policy_engine = aura.policy_engine.write();
            for hint in [
                crate::policy::PolicyHint {
                    id: "policy-health-pressure".into(),
                    key: "default:deploy:avoid".into(),
                    namespace: "default".into(),
                    domain: "deploy".into(),
                    action_kind: crate::policy::PolicyActionKind::Avoid,
                    recommendation: "Avoid direct deploy".into(),
                    trigger_causal_ids: vec![],
                    trigger_concept_ids: vec![],
                    trigger_belief_ids: vec![belief_id.clone()],
                    supporting_record_ids: vec![record.id.clone()],
                    cause_record_ids: vec![record.id.clone()],
                    confidence: 0.8,
                    utility_score: 0.3,
                    risk_score: 0.9,
                    policy_strength: 0.88,
                    state: crate::policy::PolicyState::Stable,
                    last_updated: 1.0,
                },
                crate::policy::PolicyHint {
                    id: "policy-health-suppressed".into(),
                    key: "default:deploy:verify".into(),
                    namespace: "default".into(),
                    domain: "deploy".into(),
                    action_kind: crate::policy::PolicyActionKind::VerifyFirst,
                    recommendation: "Verify first".into(),
                    trigger_causal_ids: vec![],
                    trigger_concept_ids: vec![],
                    trigger_belief_ids: vec![belief_id.clone()],
                    supporting_record_ids: vec![record.id.clone()],
                    cause_record_ids: vec![record.id.clone()],
                    confidence: 0.6,
                    utility_score: 0.2,
                    risk_score: 0.4,
                    policy_strength: 0.66,
                    state: crate::policy::PolicyState::Suppressed,
                    last_updated: 1.0,
                },
            ] {
                policy_engine
                    .key_index
                    .insert(hint.key.clone(), hint.id.clone());
                policy_engine.hints.insert(hint.id.clone(), hint);
            }
        }

        {
            let mut history = aura.runtime.maintenance_trends.write();
            history.push(background_brain::MaintenanceTrendSnapshot {
                timestamp: "t1".into(),
                total_records: 1,
                records_archived: 0,
                insights_found: 0,
                volatile_records: 1,
                belief_churn: 0.1,
                causal_rejection_rate: 0.1,
                policy_suppression_rate: 0.1,
                feedback_beliefs_touched: 1,
                feedback_net_confidence_delta: 0.0,
                feedback_net_volatility_delta: 0.0,
                correction_events: 0,
                cumulative_corrections: 0,
                cycle_time_ms: 1.0,
                dominant_phase: "policy".into(),
            });
            history.push(background_brain::MaintenanceTrendSnapshot {
                timestamp: "t2".into(),
                total_records: 1,
                records_archived: 0,
                insights_found: 0,
                volatile_records: 2,
                belief_churn: 0.2,
                causal_rejection_rate: 0.2,
                policy_suppression_rate: 0.2,
                feedback_beliefs_touched: 1,
                feedback_net_confidence_delta: 0.0,
                feedback_net_volatility_delta: 0.0,
                correction_events: 1,
                cumulative_corrections: 1,
                cycle_time_ms: 1.0,
                dominant_phase: "policy".into(),
            });
        }
        {
            let mut reflections = aura.runtime.reflection_summaries.write();
            reflections.push(background_brain::ReflectionSummary {
                timestamp: "r1".into(),
                digest: "1 reflection finding(s): Deploy rollback blocker remains active".into(),
                dominant_phase: "policy".into(),
                report: background_brain::ReflectionJobReport {
                    jobs_run: 3,
                    blocker_findings: 1,
                    contradiction_findings: 0,
                    trend_findings: 0,
                    total_findings: 1,
                    capped: false,
                },
                findings: vec![background_brain::ReflectionFinding {
                    kind: "repeated_blocker".into(),
                    namespace: "default".into(),
                    title: "Deploy rollback blocker remains active".into(),
                    detail: "Task is overdue and still active.".into(),
                    related_ids: vec![record.id.clone()],
                    score: 1.2,
                    severity: "high".into(),
                }],
            });
        }

        assert!(aura.deprecate_belief_with_reason(&belief_id, "manual_review")?);

        let digest = aura.get_memory_health_digest(Some(10));
        assert_eq!(digest.total_records, 1);
        assert_eq!(digest.high_salience_record_count, 1);
        assert!(digest.avg_salience > 0.0);
        assert!(digest.max_salience >= 0.88);
        assert!(digest.high_volatility_belief_count >= 1);
        assert!(digest.recent_correction_count >= 1);
        assert!(digest.suppressed_policy_hint_count >= 1);
        assert!(digest.policy_pressure_area_count >= 1);
        assert_eq!(digest.latest_dominant_phase, "policy");
        assert_eq!(digest.maintenance_trend_direction, "worsening");
        assert!(!digest.top_issues.is_empty());
        assert!(digest
            .top_issues
            .iter()
            .any(|issue| issue.kind == "belief_instability"));
        assert!(
            digest
                .top_issues
                .iter()
                .any(|issue| issue.kind == "contradiction_cluster")
                || digest.contradiction_cluster_count == 0
        );
        assert!(digest
            .top_issues
            .iter()
            .any(|issue| issue.kind == "policy_pressure"));
        assert!(digest
            .top_issues
            .iter()
            .any(|issue| issue.kind == "high_salience_record"));
        assert!(digest
            .top_issues
            .iter()
            .any(|issue| issue.kind == "recent_correction"));
        assert!(digest.reflection_summary_count >= 1);
        assert!(digest
            .top_issues
            .iter()
            .any(|issue| issue.kind == "reflection_finding"));

        Ok(())
    }

    #[test]
    fn test_targeted_retraction_and_deprecation_persist() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();

        {
            let aura = Aura::open(root)?;

            let mut belief = crate::belief::Belief::new("default:test:fact".to_string());
            belief.id = "belief-phase4".to_string();
            belief.state = crate::belief::BeliefState::Resolved;
            belief.confidence = 0.84;
            belief.score = 0.84;
            {
                let mut engine = aura.belief_engine.write();
                engine
                    .key_index
                    .insert(belief.key.clone(), belief.id.clone());
                engine.beliefs.insert(belief.id.clone(), belief);
            }
            {
                let engine = aura.belief_engine.read();
                aura.belief_store.save(&engine)?;
            }

            let pattern = crate::causal::CausalPattern {
                id: "causal-phase4".to_string(),
                key: "default:cause:effect:phase4".to_string(),
                namespace: "default".to_string(),
                cause_belief_ids: vec!["belief-phase4".to_string()],
                effect_belief_ids: vec![],
                cause_record_ids: vec!["cause-record".to_string()],
                effect_record_ids: vec!["effect-record".to_string()],
                support_count: 2,
                explicit_support_count: 1,
                temporal_support_count: 1,
                unique_temporal_windows: 1,
                effect_record_signature_variants: 1,
                positive_effect_signals: 1,
                negative_effect_signals: 0,
                counterevidence: 0,
                explicit_support_total_for_cause: 1,
                explicit_effect_variants_for_cause: 1,
                transition_lift: 1.0,
                temporal_consistency: 1.0,
                outcome_stability: 1.0,
                causal_strength: 0.8,
                invalidation_reason: None,
                invalidated_at: None,
                state: crate::causal::CausalState::Stable,
                last_updated: 1.0,
            };
            {
                let mut engine = aura.causal_engine.write();
                engine
                    .key_index
                    .insert(pattern.key.clone(), pattern.id.clone());
                engine.patterns.insert(pattern.id.clone(), pattern);
            }
            {
                let engine = aura.causal_engine.read();
                aura.causal_store.save(&engine)?;
            }

            let hint = crate::policy::PolicyHint {
                id: "policy-phase4".to_string(),
                key: "default:verify:phase4".to_string(),
                namespace: "default".to_string(),
                domain: "deploy".to_string(),
                action_kind: crate::policy::PolicyActionKind::VerifyFirst,
                recommendation: "Verify rollout".to_string(),
                trigger_causal_ids: vec!["causal-phase4".to_string()],
                trigger_concept_ids: vec![],
                trigger_belief_ids: vec!["belief-phase4".to_string()],
                supporting_record_ids: vec![
                    "cause-record".to_string(),
                    "effect-record".to_string(),
                ],
                cause_record_ids: vec!["cause-record".to_string()],
                confidence: 0.75,
                utility_score: 0.6,
                risk_score: 0.2,
                policy_strength: 0.7,
                state: crate::policy::PolicyState::Stable,
                last_updated: 1.0,
            };
            {
                let mut engine = aura.policy_engine.write();
                engine.key_index.insert(hint.key.clone(), hint.id.clone());
                engine.hints.insert(hint.id.clone(), hint);
            }
            {
                let engine = aura.policy_engine.read();
                aura.policy_store.save(&engine)?;
            }

            assert!(aura.deprecate_belief_with_reason("belief-phase4", "contradicted_by_review")?);
            assert!(aura
                .invalidate_causal_pattern_with_reason("causal-phase4", "spurious_correlation")?);
            assert!(aura.retract_policy_hint_with_reason("policy-phase4", "superseded_runbook")?);

            let log = aura.get_correction_log();
            let causal_log = aura.get_correction_log_for_target("causal_pattern", "causal-phase4");
            assert_eq!(log.len(), 3);
            assert_eq!(causal_log.len(), 1);
            assert_eq!(causal_log[0].operation, "invalidate");
            assert_eq!(causal_log[0].reason, "spurious_correlation");
            assert!(log.iter().any(|entry| {
                entry.target_kind == "belief"
                    && entry.target_id == "belief-phase4"
                    && entry.reason == "contradicted_by_review"
            }));
            assert!(log.iter().any(|entry| {
                entry.target_kind == "causal_pattern"
                    && entry.target_id == "causal-phase4"
                    && entry.operation == "invalidate"
                    && entry.reason == "spurious_correlation"
            }));
            assert!(log.iter().any(|entry| {
                entry.target_kind == "policy_hint"
                    && entry.target_id == "policy-phase4"
                    && entry.reason == "superseded_runbook"
            }));
        }

        let reopened = Aura::open(root)?;
        let belief = reopened
            .belief_engine
            .read()
            .beliefs
            .get("belief-phase4")
            .cloned()
            .expect("belief should remain persisted");
        let pattern = reopened
            .causal_engine
            .read()
            .patterns
            .get("causal-phase4")
            .cloned()
            .expect("invalidated causal pattern should remain persisted");
        assert_eq!(belief.state, crate::belief::BeliefState::Unresolved);
        assert!(belief.confidence < 0.84);
        assert_eq!(pattern.state, crate::causal::CausalState::Invalidated);
        assert_eq!(
            pattern.invalidation_reason.as_deref(),
            Some("spurious_correlation")
        );
        assert!(pattern.invalidated_at.is_some());
        assert!(reopened.policy_engine.read().hints.is_empty());

        Ok(())
    }

    #[test]
    fn test_correction_review_queue_prioritizes_repeated_high_impact_targets() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let record = aura.store(
            "Review queue deploy regression record",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "incident".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;

        let belief = crate::belief::Belief {
            id: "belief-review".to_string(),
            key: "default:deploy:review".to_string(),
            hypothesis_ids: vec!["hyp-review".to_string()],
            winner_id: Some("hyp-review".to_string()),
            state: crate::belief::BeliefState::Resolved,
            score: 0.78,
            confidence: 0.78,
            support_mass: 1.0,
            conflict_mass: 0.0,
            stability: 3.0,
            volatility: 0.12,
            last_updated: 1.0,
        };
        {
            let mut engine = aura.belief_engine.write();
            engine
                .key_index
                .insert(belief.key.clone(), belief.id.clone());
            engine.beliefs.insert(belief.id.clone(), belief);
        }
        {
            let engine = aura.belief_engine.read();
            aura.belief_store.save(&engine)?;
        }

        let pattern = crate::causal::CausalPattern {
            id: "causal-review".to_string(),
            key: "default:belief-review->effect-review".to_string(),
            namespace: "default".to_string(),
            cause_belief_ids: vec!["belief-review".to_string()],
            effect_belief_ids: vec!["belief-effect".to_string()],
            cause_record_ids: vec![record.id.clone()],
            effect_record_ids: vec!["effect-review-record".to_string()],
            support_count: 2,
            explicit_support_count: 1,
            temporal_support_count: 1,
            unique_temporal_windows: 1,
            effect_record_signature_variants: 1,
            positive_effect_signals: 0,
            negative_effect_signals: 1,
            counterevidence: 0,
            explicit_support_total_for_cause: 1,
            explicit_effect_variants_for_cause: 1,
            transition_lift: 0.4,
            temporal_consistency: 0.7,
            outcome_stability: 0.65,
            causal_strength: 0.74,
            invalidation_reason: None,
            invalidated_at: None,
            state: crate::causal::CausalState::Stable,
            last_updated: 1.0,
        };
        {
            let mut engine = aura.causal_engine.write();
            engine
                .key_index
                .insert(pattern.key.clone(), pattern.id.clone());
            engine.patterns.insert(pattern.id.clone(), pattern);
        }
        {
            let engine = aura.causal_engine.read();
            aura.causal_store.save(&engine)?;
        }

        let hint_from_belief = crate::policy::PolicyHint {
            id: "policy-belief-review".to_string(),
            key: "default:verify:belief-review".to_string(),
            namespace: "default".to_string(),
            domain: "deploy".to_string(),
            action_kind: crate::policy::PolicyActionKind::VerifyFirst,
            recommendation: "Verify deploy health".to_string(),
            trigger_causal_ids: vec![],
            trigger_concept_ids: vec![],
            trigger_belief_ids: vec!["belief-review".to_string()],
            supporting_record_ids: vec![record.id.clone()],
            cause_record_ids: vec![record.id.clone()],
            confidence: 0.72,
            utility_score: 0.4,
            risk_score: 0.3,
            policy_strength: 0.71,
            state: crate::policy::PolicyState::Stable,
            last_updated: 1.0,
        };
        let hint_from_causal = crate::policy::PolicyHint {
            id: "policy-causal-review".to_string(),
            key: "default:avoid:causal-review".to_string(),
            namespace: "default".to_string(),
            domain: "deploy".to_string(),
            action_kind: crate::policy::PolicyActionKind::Avoid,
            recommendation: "Avoid rollout corridor".to_string(),
            trigger_causal_ids: vec!["causal-review".to_string()],
            trigger_concept_ids: vec![],
            trigger_belief_ids: vec![],
            supporting_record_ids: vec![record.id.clone()],
            cause_record_ids: vec![record.id.clone()],
            confidence: 0.74,
            utility_score: 0.5,
            risk_score: 0.4,
            policy_strength: 0.76,
            state: crate::policy::PolicyState::Stable,
            last_updated: 1.0,
        };
        {
            let mut engine = aura.policy_engine.write();
            engine
                .key_index
                .insert(hint_from_belief.key.clone(), hint_from_belief.id.clone());
            engine
                .key_index
                .insert(hint_from_causal.key.clone(), hint_from_causal.id.clone());
            engine
                .hints
                .insert(hint_from_belief.id.clone(), hint_from_belief);
            engine
                .hints
                .insert(hint_from_causal.id.clone(), hint_from_causal);
        }
        {
            let engine = aura.policy_engine.read();
            aura.policy_store.save(&engine)?;
        }

        if let Some(log) = aura.audit_log.as_ref() {
            let _ = log.log_correction("belief", "belief-review", "deprecate", "repeat_review");
            let _ = log.log_correction("belief", "belief-review", "deprecate", "repeat_review_2");
            let _ = log.log_correction(
                "causal_pattern",
                "causal-review",
                "invalidate",
                "spurious_correlation",
            );
        }

        let queue = aura.get_correction_review_queue(Some(5));
        assert_eq!(queue.len(), 3);
        assert_eq!(queue[0].target_kind, "belief");
        assert_eq!(queue[0].target_id, "belief-review");
        assert_eq!(queue[0].repeat_count, 2);
        assert_eq!(queue[0].dependent_causal_patterns, 1);
        assert_eq!(queue[0].dependent_policy_hints, 1);
        assert_eq!(queue[0].downstream_impact, 2);
        assert_eq!(queue[0].namespace, "default");
        assert!(queue[0].priority_score > queue[1].priority_score);
        assert!(queue.iter().any(|entry| {
            entry.target_kind == "causal_pattern"
                && entry.target_id == "causal-review"
                && entry.dependent_policy_hints == 1
        }));

        Ok(())
    }

    #[test]
    fn test_suggested_corrections_returns_bounded_advisory_candidates() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let record = aura.store(
            "Suggested correction deploy review record",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "review".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;

        let belief = crate::belief::Belief {
            id: "belief-suggest".to_string(),
            key: "default:deploy:suggest".to_string(),
            hypothesis_ids: vec!["hyp-suggest".to_string()],
            winner_id: Some("hyp-suggest".to_string()),
            state: crate::belief::BeliefState::Resolved,
            score: 0.72,
            confidence: 0.72,
            support_mass: 1.0,
            conflict_mass: 0.1,
            stability: 0.5,
            volatility: 0.34,
            last_updated: 1.0,
        };
        let hypothesis = crate::belief::Hypothesis {
            id: "hyp-suggest".to_string(),
            belief_id: "belief-suggest".to_string(),
            prototype_record_ids: vec![record.id.clone()],
            score: 0.72,
            confidence: 0.72,
            support_mass: 1.0,
            conflict_mass: 0.1,
            recency: 1.0,
            consistency: 1.0,
        };
        {
            let mut engine = aura.belief_engine.write();
            engine
                .key_index
                .insert(belief.key.clone(), belief.id.clone());
            engine
                .record_index
                .insert(record.id.clone(), hypothesis.id.clone());
            engine.hypotheses.insert(hypothesis.id.clone(), hypothesis);
            engine.beliefs.insert(belief.id.clone(), belief);
        }
        {
            let engine = aura.belief_engine.read();
            aura.belief_store.save(&engine)?;
        }

        let rejected_pattern = crate::causal::CausalPattern {
            id: "causal-suggest".to_string(),
            key: "default:belief-suggest->effect-suggest".to_string(),
            namespace: "default".to_string(),
            cause_belief_ids: vec!["belief-suggest".to_string()],
            effect_belief_ids: vec!["belief-effect".to_string()],
            cause_record_ids: vec![record.id.clone()],
            effect_record_ids: vec!["effect-suggest-record".to_string()],
            support_count: 1,
            explicit_support_count: 0,
            temporal_support_count: 1,
            unique_temporal_windows: 1,
            effect_record_signature_variants: 1,
            positive_effect_signals: 0,
            negative_effect_signals: 1,
            counterevidence: 2,
            explicit_support_total_for_cause: 0,
            explicit_effect_variants_for_cause: 0,
            transition_lift: 0.2,
            temporal_consistency: 0.4,
            outcome_stability: 0.3,
            causal_strength: 0.42,
            invalidation_reason: None,
            invalidated_at: None,
            state: crate::causal::CausalState::Rejected,
            last_updated: 1.0,
        };
        {
            let mut engine = aura.causal_engine.write();
            engine
                .key_index
                .insert(rejected_pattern.key.clone(), rejected_pattern.id.clone());
            engine
                .patterns
                .insert(rejected_pattern.id.clone(), rejected_pattern);
        }
        {
            let engine = aura.causal_engine.read();
            aura.causal_store.save(&engine)?;
        }

        let suppressed_hint = crate::policy::PolicyHint {
            id: "policy-suggest".to_string(),
            key: "default:avoid:suggest".to_string(),
            namespace: "default".to_string(),
            domain: "deploy".to_string(),
            action_kind: crate::policy::PolicyActionKind::Avoid,
            recommendation: "Avoid deploy path".to_string(),
            trigger_causal_ids: vec!["causal-suggest".to_string()],
            trigger_concept_ids: vec![],
            trigger_belief_ids: vec!["belief-suggest".to_string()],
            supporting_record_ids: vec![record.id.clone()],
            cause_record_ids: vec![record.id.clone()],
            confidence: 0.7,
            utility_score: 0.4,
            risk_score: 0.6,
            policy_strength: 0.77,
            state: crate::policy::PolicyState::Suppressed,
            last_updated: 1.0,
        };
        {
            let mut engine = aura.policy_engine.write();
            engine
                .key_index
                .insert(suppressed_hint.key.clone(), suppressed_hint.id.clone());
            engine
                .hints
                .insert(suppressed_hint.id.clone(), suppressed_hint);
        }
        {
            let engine = aura.policy_engine.read();
            aura.policy_store.save(&engine)?;
        }

        if let Some(log) = aura.audit_log.as_ref() {
            let _ = log.log_correction("policy_hint", "policy-suggest", "retract", "repeat_review");
            let _ = log.log_correction(
                "policy_hint",
                "policy-suggest",
                "retract",
                "repeat_review_2",
            );
        }

        let suggestions = aura.get_suggested_corrections(Some(10));
        assert!(suggestions.iter().any(|item| {
            item.target_kind == "belief"
                && item.target_id == "belief-suggest"
                && item.suggested_action == "Deprecate"
        }));
        assert!(suggestions.iter().any(|item| {
            item.target_kind == "causal_pattern"
                && item.target_id == "causal-suggest"
                && item.suggested_action == "Invalidate"
        }));
        assert!(suggestions.iter().any(|item| {
            item.target_kind == "policy_hint"
                && item.target_id == "policy-suggest"
                && item.suggested_action == "Retract"
        }));
        assert!(suggestions.iter().all(|item| item.provenance.is_some()));

        Ok(())
    }

    #[test]
    fn test_namespace_governance_status_is_read_only_and_filtered() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let alpha_record = aura.store(
            "Alpha namespace governance record",
            Some(Level::Domain),
            Some(vec!["alpha".into(), "ops".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("alpha"),
            Some("fact"),
        )?;
        let _beta_record = aura.store(
            "Beta namespace governance record",
            Some(Level::Domain),
            Some(vec!["beta".into(), "ops".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("beta"),
            Some("fact"),
        )?;

        let alpha_belief = crate::belief::Belief {
            id: "belief-alpha-gov".to_string(),
            key: "alpha:ops:governance".to_string(),
            hypothesis_ids: vec![],
            winner_id: None,
            state: crate::belief::BeliefState::Resolved,
            score: 0.7,
            confidence: 0.7,
            support_mass: 1.0,
            conflict_mass: 0.0,
            stability: 0.8,
            volatility: 0.3,
            last_updated: 1.0,
        };
        let beta_belief = crate::belief::Belief {
            id: "belief-beta-gov".to_string(),
            key: "beta:ops:governance".to_string(),
            hypothesis_ids: vec![],
            winner_id: None,
            state: crate::belief::BeliefState::Resolved,
            score: 0.7,
            confidence: 0.7,
            support_mass: 1.0,
            conflict_mass: 0.0,
            stability: 2.5,
            volatility: 0.05,
            last_updated: 1.0,
        };
        {
            let mut engine = aura.belief_engine.write();
            engine
                .key_index
                .insert(alpha_belief.key.clone(), alpha_belief.id.clone());
            engine
                .key_index
                .insert(beta_belief.key.clone(), beta_belief.id.clone());
            engine
                .beliefs
                .insert(alpha_belief.id.clone(), alpha_belief.clone());
            engine
                .beliefs
                .insert(beta_belief.id.clone(), beta_belief.clone());
        }
        {
            let engine = aura.belief_engine.read();
            aura.belief_store.save(&engine)?;
        }

        if let Some(log) = aura.audit_log.as_ref() {
            let _ = log.log_correction("record", &alpha_record.id, "deprecate", "alpha_review");
        }

        aura.runtime.maintenance_trends.write().push(
            crate::background_brain::MaintenanceTrendSnapshot {
                timestamp: "2026-03-21T10:15:00Z".to_string(),
                total_records: 2,
                records_archived: 0,
                insights_found: 0,
                volatile_records: 1,
                belief_churn: 0.1,
                causal_rejection_rate: 0.0,
                policy_suppression_rate: 0.0,
                feedback_beliefs_touched: 1,
                feedback_net_confidence_delta: 0.0,
                feedback_net_volatility_delta: 0.1,
                correction_events: 1,
                cumulative_corrections: 1,
                cycle_time_ms: 12.0,
                dominant_phase: "epistemic".to_string(),
            },
        );

        let statuses = aura.get_namespace_governance_status_filtered(Some(&["alpha"]));
        assert_eq!(statuses.len(), 1);
        let alpha = &statuses[0];
        assert_eq!(alpha.namespace, "alpha");
        assert_eq!(alpha.record_count, 1);
        assert_eq!(alpha.belief_count, 1);
        assert_eq!(alpha.correction_count, 1);
        assert_eq!(alpha.high_volatility_belief_count, 1);
        assert_eq!(alpha.low_stability_belief_count, 1);
        assert_eq!(
            alpha.last_maintenance_cycle.as_deref(),
            Some("2026-03-21T10:15:00Z")
        );
        assert_eq!(alpha.latest_dominant_phase, "epistemic");

        let all = aura.get_namespace_governance_status();
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|item| item.namespace == "beta"));

        Ok(())
    }

    #[test]
    fn test_explain_recall_returns_provenance_chain() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let cause = aura.store(
            "Canary deploy gate enabled before rollout",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "safety".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("default"),
            Some("decision"),
        )?;
        let effect = aura.store(
            "Deploy health improved after canary gate rollout",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "improvement".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            Some(&cause.id),
            Some("default"),
            Some("fact"),
        )?;
        aura.mark_record_salience(&effect.id, 0.82, Some("operator_priority"))?;

        let belief_id = "belief-explain".to_string();
        {
            let mut engine = aura.belief_engine.write();
            let hyp = crate::belief::Hypothesis::from_records(&belief_id, &[&effect]);
            let mut belief = crate::belief::Belief::new("default:deploy:observation".into());
            belief.id = belief_id.clone();
            belief.state = crate::belief::BeliefState::Singleton;
            belief.winner_id = Some(hyp.id.clone());
            belief.hypothesis_ids = vec![hyp.id.clone()];
            belief.score = hyp.score;
            belief.confidence = hyp.confidence;
            belief.support_mass = hyp.support_mass;
            belief.conflict_mass = hyp.conflict_mass + 2.0;
            belief.volatility = 0.32;
            belief.stability = 0.6;
            engine
                .key_index
                .insert(belief.key.clone(), belief.id.clone());
            engine
                .record_index
                .insert(effect.id.clone(), hyp.id.clone());
            engine.hypotheses.insert(hyp.id.clone(), hyp);
            engine.beliefs.insert(belief.id.clone(), belief);
        }

        {
            let mut engine = aura.concept_engine.write();
            let concept = crate::concept::ConceptCandidate {
                id: "concept-explain".into(),
                key: "default:deploy:concept".into(),
                namespace: "default".into(),
                semantic_type: "fact".into(),
                belief_ids: vec![belief_id.clone()],
                record_ids: vec![effect.id.clone()],
                core_terms: vec!["deploy".into(), "health".into()],
                shell_terms: vec!["canary".into()],
                tags: vec!["deploy".into()],
                support_mass: 2.0,
                confidence: 0.88,
                stability: 2.0,
                cohesion: 0.9,
                abstraction_score: 0.8,
                state: crate::concept::ConceptState::Stable,
                last_updated: 1.0,
            };
            engine
                .key_index
                .insert(concept.key.clone(), concept.id.clone());
            engine.concepts.insert(concept.id.clone(), concept);
        }

        {
            let mut engine = aura.causal_engine.write();
            let pattern = crate::causal::CausalPattern {
                id: "causal-explain".into(),
                key: "default:deploy:cause-effect".into(),
                namespace: "default".into(),
                cause_belief_ids: vec![belief_id.clone()],
                effect_belief_ids: vec![],
                cause_record_ids: vec![cause.id.clone()],
                effect_record_ids: vec![effect.id.clone()],
                support_count: 2,
                explicit_support_count: 1,
                temporal_support_count: 1,
                unique_temporal_windows: 1,
                effect_record_signature_variants: 1,
                positive_effect_signals: 2,
                negative_effect_signals: 0,
                counterevidence: 0,
                explicit_support_total_for_cause: 1,
                explicit_effect_variants_for_cause: 1,
                transition_lift: 1.1,
                temporal_consistency: 1.0,
                outcome_stability: 0.95,
                causal_strength: 0.82,
                invalidation_reason: None,
                invalidated_at: None,
                state: crate::causal::CausalState::Stable,
                last_updated: 1.0,
            };
            engine
                .key_index
                .insert(pattern.key.clone(), pattern.id.clone());
            engine.patterns.insert(pattern.id.clone(), pattern);
        }

        {
            let mut engine = aura.policy_engine.write();
            let hint = crate::policy::PolicyHint {
                id: "policy-explain".into(),
                key: "default:prefer:deploy".into(),
                namespace: "default".into(),
                domain: "deploy".into(),
                action_kind: crate::policy::PolicyActionKind::Prefer,
                recommendation: "Prefer canary gates before rollout".into(),
                trigger_causal_ids: vec!["causal-explain".into()],
                trigger_concept_ids: vec!["concept-explain".into()],
                trigger_belief_ids: vec![belief_id.clone()],
                supporting_record_ids: vec![cause.id.clone(), effect.id.clone()],
                cause_record_ids: vec![cause.id.clone()],
                confidence: 0.8,
                utility_score: 0.7,
                risk_score: 0.0,
                policy_strength: 0.78,
                state: crate::policy::PolicyState::Stable,
                last_updated: 1.0,
            };
            engine.key_index.insert(hint.key.clone(), hint.id.clone());
            engine.hints.insert(hint.id.clone(), hint);
        }
        {
            let mut reflections = aura.runtime.reflection_summaries.write();
            reflections.push(background_brain::ReflectionSummary {
                timestamp: "r1".into(),
                digest: "1 reflection finding(s): Deploy blocker remains under review".into(),
                dominant_phase: "belief".into(),
                report: background_brain::ReflectionJobReport {
                    jobs_run: 3,
                    blocker_findings: 0,
                    contradiction_findings: 1,
                    trend_findings: 0,
                    total_findings: 1,
                    capped: false,
                },
                findings: vec![background_brain::ReflectionFinding {
                    kind: "unresolved_contradiction".into(),
                    namespace: "default".into(),
                    title: "Deploy blocker remains under review".into(),
                    detail: "Recent deploy evidence remains conflicted.".into(),
                    related_ids: vec![effect.id.clone()],
                    score: 0.9,
                    severity: "high".into(),
                }],
            });
        }

        let explanation = aura.explain_recall(
            "deploy health canary",
            Some(5),
            Some(0.1),
            Some(true),
            Some(&["default"]),
        );

        let item = explanation
            .items
            .iter()
            .find(|item| item.record_id == effect.id)
            .expect("effect record should be explained");

        assert_eq!(explanation.query, "deploy health canary");
        assert_eq!(item.because_record_id.as_deref(), Some(cause.id.as_str()));
        assert_eq!(
            item.belief.as_ref().map(|b| b.id.as_str()),
            Some("belief-explain")
        );
        assert!(item.has_unresolved_evidence);
        assert_eq!(
            item.honesty_note.as_deref(),
            Some("This recommendation depends on unresolved evidence.")
        );
        assert!(item.contradiction_dependency);
        assert_eq!(
            item.salience_explanation.as_deref(),
            Some("High-significance memory due to operator_priority.")
        );
        assert!(item
            .reflection_references
            .iter()
            .any(|title| title.contains("Deploy blocker remains under review")));
        assert_eq!(
            item.answer_support.significance_phrase.as_deref(),
            Some("High-significance memory due to operator_priority.")
        );
        assert_eq!(
            item.answer_support.uncertainty_phrase.as_deref(),
            Some("This recommendation depends on unresolved evidence.")
        );
        assert_eq!(
            item.answer_support.contradiction_phrase.as_deref(),
            Some("This answer should acknowledge conflicting or unresolved evidence.")
        );
        assert!(item
            .answer_support
            .reflection_phrase
            .as_deref()
            .is_some_and(|value| value.contains("Deploy blocker remains under review")));
        assert_eq!(
            item.answer_support.recommended_framing,
            "State the useful evidence, then explicitly note uncertainty or conflict."
        );
        assert!(item
            .belief
            .as_ref()
            .is_some_and(|belief| belief.has_unresolved_evidence));
        assert!(item.trace.rrf_score > 0.0);
        assert!(item.trace.pre_trust_score > 0.0);
        assert!(item.trace.pre_rerank_score > 0.0);
        assert!(item.trace.final_score >= item.trace.pre_rerank_score);
        assert!(
            item.trace.sdr.is_some()
                || item.trace.ngram.is_some()
                || item.trace.tags.is_some()
                || item.trace.embedding.is_some()
        );
        assert!(item.concepts.iter().any(|c| c.id == "concept-explain"));
        assert!(item
            .causal_patterns
            .iter()
            .any(|c| c.id == "causal-explain"));
        assert!(item.policy_hints.iter().any(|h| h.id == "policy-explain"));

        Ok(())
    }

    #[test]
    fn test_explain_record_returns_direct_provenance() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let cause = aura.store(
            "Manual rollback gate enabled",
            Some(Level::Domain),
            Some(vec!["rollback".into(), "safety".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("default"),
            Some("decision"),
        )?;
        let effect = aura.store(
            "Rollback reliability improved after gate",
            Some(Level::Domain),
            Some(vec!["rollback".into(), "improvement".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            Some(&cause.id),
            Some("default"),
            Some("fact"),
        )?;

        let belief_id = "belief-record-explain".to_string();
        {
            let mut causal_engine = aura.causal_engine.write();
            let pattern = crate::causal::CausalPattern {
                id: "causal-record-explain".into(),
                key: "default:rollback:cause-effect".into(),
                namespace: "default".into(),
                cause_belief_ids: vec![belief_id.clone()],
                effect_belief_ids: vec![],
                cause_record_ids: vec![cause.id.clone()],
                effect_record_ids: vec![effect.id.clone()],
                support_count: 2,
                explicit_support_count: 1,
                temporal_support_count: 1,
                unique_temporal_windows: 1,
                effect_record_signature_variants: 1,
                positive_effect_signals: 1,
                negative_effect_signals: 0,
                counterevidence: 0,
                explicit_support_total_for_cause: 1,
                explicit_effect_variants_for_cause: 1,
                transition_lift: 1.0,
                temporal_consistency: 1.0,
                outcome_stability: 0.9,
                causal_strength: 0.77,
                invalidation_reason: None,
                invalidated_at: None,
                state: crate::causal::CausalState::Stable,
                last_updated: 1.0,
            };
            causal_engine
                .key_index
                .insert(pattern.key.clone(), pattern.id.clone());
            causal_engine.patterns.insert(pattern.id.clone(), pattern);
        }

        assert!(aura.invalidate_causal_pattern_with_reason(
            "causal-record-explain",
            "superseded_by_manual_review"
        )?);

        {
            let mut engine = aura.belief_engine.write();
            let hyp = crate::belief::Hypothesis::from_records(&belief_id, &[&effect]);
            let mut belief = crate::belief::Belief::new("default:rollback:fact".into());
            belief.id = belief_id.clone();
            belief.state = crate::belief::BeliefState::Singleton;
            belief.winner_id = Some(hyp.id.clone());
            belief.hypothesis_ids = vec![hyp.id.clone()];
            belief.score = hyp.score;
            belief.confidence = hyp.confidence;
            belief.support_mass = hyp.support_mass;
            belief.conflict_mass = hyp.conflict_mass;
            engine
                .record_index
                .insert(effect.id.clone(), hyp.id.clone());
            engine.hypotheses.insert(hyp.id.clone(), hyp);
            engine
                .key_index
                .insert(belief.key.clone(), belief.id.clone());
            engine.beliefs.insert(belief.id.clone(), belief);
        }

        let item = aura
            .explain_record(&effect.id)
            .expect("record explanation should exist");

        assert_eq!(item.record_id, effect.id);
        assert_eq!(item.because_record_id.as_deref(), Some(cause.id.as_str()));
        assert_eq!(
            item.belief.as_ref().map(|belief| belief.id.as_str()),
            Some("belief-record-explain")
        );
        assert_eq!(item.trace.final_score, item.score);
        assert_eq!(item.trace.pre_rerank_score, 0.0);
        assert_eq!(item.trace.rerank_delta, item.score);
        assert!(item.trace.sdr.is_none());
        assert!(item.trace.ngram.is_none());
        assert!(item.trace.tags.is_none());
        assert!(item.trace.embedding.is_none());
        let pattern = item
            .causal_patterns
            .iter()
            .find(|pattern| pattern.id == "causal-record-explain")
            .expect("causal pattern should be present in explanation");
        assert_eq!(pattern.state, "invalidated");
        assert_eq!(
            pattern.invalidation_reason.as_deref(),
            Some("superseded_by_manual_review")
        );
        assert!(pattern.invalidated_at.is_some());
        assert_eq!(pattern.corrections.len(), 1);
        assert_eq!(pattern.corrections[0].operation, "invalidate");

        Ok(())
    }

    #[test]
    fn test_provenance_chain_builds_deterministic_narrative() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let cause = aura.store(
            "Canary deploy gate enabled before rollout",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "safety".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("default"),
            Some("decision"),
        )?;
        let effect = aura.store(
            "Deploy health improved after canary gate rollout",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "improvement".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            Some(&cause.id),
            Some("default"),
            Some("fact"),
        )?;

        let belief_id = "belief-provenance".to_string();
        {
            let mut belief_engine = aura.belief_engine.write();
            let hyp = crate::belief::Hypothesis::from_records(&belief_id, &[&effect]);
            let mut belief = crate::belief::Belief::new("default:deploy:observation".into());
            belief.id = belief_id.clone();
            belief.state = crate::belief::BeliefState::Singleton;
            belief.winner_id = Some(hyp.id.clone());
            belief.hypothesis_ids = vec![hyp.id.clone()];
            belief.score = hyp.score;
            belief.confidence = hyp.confidence;
            belief.support_mass = hyp.support_mass;
            belief.conflict_mass = hyp.conflict_mass;
            belief_engine
                .key_index
                .insert(belief.key.clone(), belief.id.clone());
            belief_engine
                .record_index
                .insert(effect.id.clone(), hyp.id.clone());
            belief_engine.hypotheses.insert(hyp.id.clone(), hyp);
            belief_engine.beliefs.insert(belief.id.clone(), belief);
        }

        {
            let mut concept_engine = aura.concept_engine.write();
            let concept = crate::concept::ConceptCandidate {
                id: "concept-provenance".into(),
                key: "default:deploy:concept".into(),
                namespace: "default".into(),
                semantic_type: "fact".into(),
                belief_ids: vec![belief_id.clone()],
                record_ids: vec![effect.id.clone()],
                core_terms: vec!["deploy".into(), "health".into()],
                shell_terms: vec!["canary".into()],
                tags: vec!["deploy".into()],
                support_mass: 2.0,
                confidence: 0.88,
                stability: 2.0,
                cohesion: 0.9,
                abstraction_score: 0.8,
                state: crate::concept::ConceptState::Stable,
                last_updated: 1.0,
            };
            concept_engine
                .key_index
                .insert(concept.key.clone(), concept.id.clone());
            concept_engine.concepts.insert(concept.id.clone(), concept);
        }

        {
            let mut causal_engine = aura.causal_engine.write();
            let pattern = crate::causal::CausalPattern {
                id: "causal-provenance".into(),
                key: "default:deploy:cause-effect".into(),
                namespace: "default".into(),
                cause_belief_ids: vec![belief_id.clone()],
                effect_belief_ids: vec![],
                cause_record_ids: vec![cause.id.clone()],
                effect_record_ids: vec![effect.id.clone()],
                support_count: 2,
                explicit_support_count: 1,
                temporal_support_count: 1,
                unique_temporal_windows: 1,
                effect_record_signature_variants: 1,
                positive_effect_signals: 2,
                negative_effect_signals: 0,
                counterevidence: 0,
                explicit_support_total_for_cause: 1,
                explicit_effect_variants_for_cause: 1,
                transition_lift: 1.1,
                temporal_consistency: 1.0,
                outcome_stability: 0.95,
                causal_strength: 0.79,
                invalidation_reason: None,
                invalidated_at: None,
                state: crate::causal::CausalState::Stable,
                last_updated: 1.0,
            };
            causal_engine
                .key_index
                .insert(pattern.key.clone(), pattern.id.clone());
            causal_engine.patterns.insert(pattern.id.clone(), pattern);
        }

        {
            let mut policy_engine = aura.policy_engine.write();
            let hint = crate::policy::PolicyHint {
                id: "policy-provenance".into(),
                key: "default:deploy:policy".into(),
                namespace: "default".into(),
                domain: "deploy".into(),
                action_kind: crate::policy::PolicyActionKind::Prefer,
                recommendation: "Prefer canary gates before rollout".into(),
                trigger_causal_ids: vec!["causal-provenance".into()],
                trigger_concept_ids: vec!["concept-provenance".into()],
                trigger_belief_ids: vec![belief_id.clone()],
                supporting_record_ids: vec![cause.id.clone(), effect.id.clone()],
                cause_record_ids: vec![cause.id.clone()],
                confidence: 0.8,
                utility_score: 0.7,
                risk_score: 0.0,
                policy_strength: 0.78,
                state: crate::policy::PolicyState::Stable,
                last_updated: 1.0,
            };
            policy_engine
                .key_index
                .insert(hint.key.clone(), hint.id.clone());
            policy_engine.hints.insert(hint.id.clone(), hint);
        }

        let chain = aura
            .provenance_chain(&effect.id)
            .expect("provenance chain should exist");

        assert_eq!(chain.record_id, effect.id);
        assert_eq!(chain.because_record_id.as_deref(), Some(cause.id.as_str()));
        assert!(chain
            .steps
            .iter()
            .any(|step| step.contains("belief-provenance")));
        assert!(chain
            .steps
            .iter()
            .any(|step| step.contains("concept-provenance")));
        assert!(chain
            .steps
            .iter()
            .any(|step| step.contains("causal-provenance")));
        assert!(chain
            .steps
            .iter()
            .any(|step| step.contains("policy-provenance")));
        assert!(chain.narrative.contains("Record"));
        assert!(chain.narrative.contains("belief belief-provenance"));
        assert!(chain
            .narrative
            .contains("concepts [default:deploy:concept]"));
        assert!(chain
            .narrative
            .contains("causal patterns [default:deploy:cause-effect]"));
        assert!(chain
            .narrative
            .contains("policy hints [prefer:default:deploy:policy]"));

        Ok(())
    }

    #[test]
    fn test_explainability_bundle_combines_surfaces() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let cause = aura.store(
            "Canary gate enabled before deploy",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "canary".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("default"),
            Some("decision"),
        )?;
        let effect = aura.store(
            "Deploy reliability improved after canary gate",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "reliability".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            Some(&cause.id),
            Some("default"),
            Some("fact"),
        )?;
        let overdue = (chrono::Utc::now() - chrono::Duration::days(1)).to_rfc3339();
        aura.store(
            "Review deploy rollout blocker",
            Some(Level::Working),
            Some(vec!["scheduled-task".into(), "deploy".into()]),
            None,
            None,
            None,
            Some(HashMap::from([
                ("status".to_string(), "active".to_string()),
                ("due_date".to_string(), overdue),
            ])),
            Some(false),
            None,
            Some("default"),
            Some("decision"),
        )?;

        let belief_id = "belief-bundle".to_string();
        {
            let mut belief_engine = aura.belief_engine.write();
            let hyp = crate::belief::Hypothesis::from_records(&belief_id, &[&effect]);
            let mut belief = crate::belief::Belief::new("default:deploy:bundle".into());
            belief.id = belief_id.clone();
            belief.state = crate::belief::BeliefState::Singleton;
            belief.winner_id = Some(hyp.id.clone());
            belief.hypothesis_ids = vec![hyp.id.clone()];
            belief.score = hyp.score;
            belief.confidence = hyp.confidence;
            belief.support_mass = hyp.support_mass;
            belief.conflict_mass = hyp.conflict_mass;
            belief.volatility = 0.22;
            belief.stability = 0.8;
            belief_engine
                .record_index
                .insert(effect.id.clone(), hyp.id.clone());
            belief_engine.hypotheses.insert(hyp.id.clone(), hyp);
            belief_engine
                .key_index
                .insert(belief.key.clone(), belief.id.clone());
            belief_engine.beliefs.insert(belief.id.clone(), belief);
        }
        {
            let engine = aura.belief_engine.read();
            aura.belief_store.save(&engine)?;
        }

        {
            let mut causal_engine = aura.causal_engine.write();
            let pattern = crate::causal::CausalPattern {
                id: "causal-bundle".into(),
                key: "default:deploy:cause-effect-bundle".into(),
                namespace: "default".into(),
                cause_belief_ids: vec![belief_id.clone()],
                effect_belief_ids: vec![],
                cause_record_ids: vec![cause.id.clone()],
                effect_record_ids: vec![effect.id.clone()],
                support_count: 2,
                explicit_support_count: 1,
                temporal_support_count: 1,
                unique_temporal_windows: 1,
                effect_record_signature_variants: 1,
                positive_effect_signals: 1,
                negative_effect_signals: 0,
                counterevidence: 0,
                explicit_support_total_for_cause: 1,
                explicit_effect_variants_for_cause: 1,
                transition_lift: 1.0,
                temporal_consistency: 1.0,
                outcome_stability: 1.0,
                causal_strength: 0.81,
                invalidation_reason: None,
                invalidated_at: None,
                state: crate::causal::CausalState::Stable,
                last_updated: 1.0,
            };
            causal_engine
                .key_index
                .insert(pattern.key.clone(), pattern.id.clone());
            causal_engine.patterns.insert(pattern.id.clone(), pattern);
        }

        assert!(aura.deprecate_belief_with_reason("belief-bundle", "manual_bundle_review")?);
        assert!(aura.invalidate_causal_pattern_with_reason("causal-bundle", "bundle_correction")?);
        let _ = aura.run_maintenance();

        let bundle = aura
            .explainability_bundle(&effect.id)
            .expect("bundle should exist");

        assert_eq!(bundle.record_id, effect.id);
        assert_eq!(bundle.explanation.record_id, effect.id);
        assert_eq!(bundle.provenance.record_id, effect.id);
        assert!(bundle
            .causal_corrections
            .iter()
            .any(|entry| entry.target_id == "causal-bundle"));
        assert!(bundle.reflection_digest.summary_count >= 1);
        assert!(!bundle.related_reflection_findings.is_empty());
        assert!(bundle.maintenance_trends.snapshot_count >= 1);
        assert!(!bundle.provenance.narrative.is_empty());

        Ok(())
    }

    #[test]
    fn test_epistemic_runtime_matches_aura_inspection_surfaces() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;
        aura.set_concept_surface_mode(crate::concept::ConceptSurfaceMode::Inspect);

        let rec = aura.store(
            "Deploy health improved after canary gate rollout",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "improvement".into()]),
            None,
            None,
            None,
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;

        let belief_id = "belief-runtime".to_string();
        {
            let mut belief_engine = aura.belief_engine.write();
            let hyp = crate::belief::Hypothesis::from_records(&belief_id, &[&rec]);
            let mut belief = crate::belief::Belief::new("default:deploy:observation".into());
            belief.id = belief_id.clone();
            belief.state = crate::belief::BeliefState::Singleton;
            belief.winner_id = Some(hyp.id.clone());
            belief.hypothesis_ids = vec![hyp.id.clone()];
            belief.score = hyp.score;
            belief.confidence = hyp.confidence;
            belief.support_mass = hyp.support_mass;
            belief.conflict_mass = hyp.conflict_mass;
            belief_engine
                .key_index
                .insert(belief.key.clone(), belief.id.clone());
            belief_engine
                .record_index
                .insert(rec.id.clone(), hyp.id.clone());
            belief_engine.hypotheses.insert(hyp.id.clone(), hyp);
            belief_engine.beliefs.insert(belief.id.clone(), belief);
        }

        {
            let mut concept_engine = aura.concept_engine.write();
            let concept = crate::concept::ConceptCandidate {
                id: "concept-runtime".into(),
                key: "default:deploy:concept".into(),
                namespace: "default".into(),
                semantic_type: "fact".into(),
                belief_ids: vec![belief_id.clone()],
                record_ids: vec![rec.id.clone()],
                core_terms: vec!["deploy".into(), "health".into()],
                shell_terms: vec!["canary".into()],
                tags: vec!["deploy".into()],
                support_mass: 2.0,
                confidence: 0.88,
                stability: 2.0,
                cohesion: 0.9,
                abstraction_score: 0.8,
                state: crate::concept::ConceptState::Stable,
                last_updated: 1.0,
            };
            concept_engine
                .key_index
                .insert(concept.key.clone(), concept.id.clone());
            concept_engine.concepts.insert(concept.id.clone(), concept);
        }

        {
            let mut causal_engine = aura.causal_engine.write();
            let pattern = crate::causal::CausalPattern {
                id: "causal-runtime".into(),
                key: "default:deploy:cause-effect".into(),
                namespace: "default".into(),
                cause_belief_ids: vec![belief_id.clone()],
                effect_belief_ids: vec![],
                cause_record_ids: vec![rec.id.clone()],
                effect_record_ids: vec![rec.id.clone()],
                support_count: 1,
                explicit_support_count: 1,
                temporal_support_count: 0,
                unique_temporal_windows: 1,
                effect_record_signature_variants: 1,
                positive_effect_signals: 1,
                negative_effect_signals: 0,
                counterevidence: 0,
                explicit_support_total_for_cause: 1,
                explicit_effect_variants_for_cause: 1,
                transition_lift: 1.0,
                temporal_consistency: 1.0,
                outcome_stability: 1.0,
                causal_strength: 0.8,
                invalidation_reason: None,
                invalidated_at: None,
                state: crate::causal::CausalState::Stable,
                last_updated: 1.0,
            };
            causal_engine
                .key_index
                .insert(pattern.key.clone(), pattern.id.clone());
            causal_engine.patterns.insert(pattern.id.clone(), pattern);
        }

        {
            let mut policy_engine = aura.policy_engine.write();
            let hint = crate::policy::PolicyHint {
                id: "policy-runtime".into(),
                key: "default:deploy:policy".into(),
                namespace: "default".into(),
                domain: "deploy".into(),
                action_kind: crate::policy::PolicyActionKind::Prefer,
                recommendation: "Prefer canary gates before rollout".into(),
                trigger_causal_ids: vec!["causal-runtime".into()],
                trigger_concept_ids: vec!["concept-runtime".into()],
                trigger_belief_ids: vec![belief_id.clone()],
                supporting_record_ids: vec![rec.id.clone()],
                cause_record_ids: vec![rec.id.clone()],
                confidence: 0.8,
                utility_score: 0.7,
                risk_score: 0.0,
                policy_strength: 0.78,
                state: crate::policy::PolicyState::Stable,
                last_updated: 1.0,
            };
            policy_engine
                .key_index
                .insert(hint.key.clone(), hint.id.clone());
            policy_engine.hints.insert(hint.id.clone(), hint);
        }

        let runtime = aura.epistemic_runtime();
        assert_eq!(runtime.get_beliefs(Some("singleton")).len(), 1);
        assert_eq!(
            runtime
                .get_belief_for_record(&rec.id)
                .as_ref()
                .map(|belief| belief.id.as_str()),
            Some("belief-runtime")
        );
        assert_eq!(runtime.get_concepts(Some("stable")).len(), 1);
        assert_eq!(runtime.get_surfaced_concepts(Some(5)).len(), 1);
        assert_eq!(
            runtime
                .get_surfaced_concepts_for_namespace("default", Some(5))
                .len(),
            1
        );
        assert_eq!(
            runtime
                .get_surfaced_concepts_for_record(&rec.id, Some(5))
                .len(),
            1
        );
        assert_eq!(runtime.get_causal_patterns(Some("stable")).len(), 1);
        assert_eq!(runtime.get_policy_hints(Some("stable")).len(), 1);
        assert_eq!(runtime.get_surfaced_policy_hints(Some(5)).len(), 1);
        assert_eq!(
            runtime
                .get_surfaced_policy_hints_for_namespace("default", Some(5))
                .len(),
            1
        );
        let instability = runtime.get_belief_instability_summary();
        assert_eq!(instability.total_beliefs, 1);
        assert_eq!(instability.singleton, 1);
        assert_eq!(
            runtime
                .get_high_volatility_beliefs(Some(0.01), Some(5))
                .len(),
            0
        );
        assert_eq!(
            runtime.get_low_stability_beliefs(Some(2.0), Some(5)).len(),
            1
        );

        assert_eq!(aura.get_beliefs(Some("singleton")).len(), 1);
        assert_eq!(aura.get_concepts(Some("stable")).len(), 1);
        assert_eq!(aura.get_causal_patterns(Some("stable")).len(), 1);
        assert_eq!(aura.get_policy_hints(Some("stable")).len(), 1);

        Ok(())
    }

    #[test]
    fn test_policy_lifecycle_surfaces_and_pressure_report() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let hints = vec![
            crate::policy::PolicyHint {
                id: "policy-stable-avoid".into(),
                key: "alpha:deploy:avoid".into(),
                namespace: "alpha".into(),
                domain: "deploy".into(),
                action_kind: crate::policy::PolicyActionKind::Avoid,
                recommendation: "Avoid direct prod deploys".into(),
                trigger_causal_ids: vec!["causal-a".into()],
                trigger_concept_ids: Vec::new(),
                trigger_belief_ids: vec!["belief-a".into()],
                supporting_record_ids: vec!["record-a".into()],
                cause_record_ids: vec!["record-a".into()],
                confidence: 0.84,
                utility_score: 0.71,
                risk_score: 0.82,
                policy_strength: 0.88,
                state: crate::policy::PolicyState::Stable,
                last_updated: 1.0,
            },
            crate::policy::PolicyHint {
                id: "policy-candidate-prefer".into(),
                key: "alpha:deploy:prefer".into(),
                namespace: "alpha".into(),
                domain: "deploy".into(),
                action_kind: crate::policy::PolicyActionKind::Prefer,
                recommendation: "Prefer canary rollout".into(),
                trigger_causal_ids: vec!["causal-b".into()],
                trigger_concept_ids: Vec::new(),
                trigger_belief_ids: vec!["belief-b".into()],
                supporting_record_ids: vec!["record-b".into()],
                cause_record_ids: vec!["record-b".into()],
                confidence: 0.66,
                utility_score: 0.69,
                risk_score: 0.10,
                policy_strength: 0.73,
                state: crate::policy::PolicyState::Candidate,
                last_updated: 1.0,
            },
            crate::policy::PolicyHint {
                id: "policy-suppressed-verify".into(),
                key: "alpha:deploy:verify".into(),
                namespace: "alpha".into(),
                domain: "deploy".into(),
                action_kind: crate::policy::PolicyActionKind::VerifyFirst,
                recommendation: "Verify rollback plan first".into(),
                trigger_causal_ids: vec!["causal-c".into()],
                trigger_concept_ids: Vec::new(),
                trigger_belief_ids: vec!["belief-c".into()],
                supporting_record_ids: vec!["record-c".into()],
                cause_record_ids: vec!["record-c".into()],
                confidence: 0.61,
                utility_score: 0.52,
                risk_score: 0.41,
                policy_strength: 0.64,
                state: crate::policy::PolicyState::Suppressed,
                last_updated: 1.0,
            },
            crate::policy::PolicyHint {
                id: "policy-rejected-warn".into(),
                key: "beta:ops:warn".into(),
                namespace: "beta".into(),
                domain: "ops".into(),
                action_kind: crate::policy::PolicyActionKind::Warn,
                recommendation: "Warn on low-signal ops anomaly".into(),
                trigger_causal_ids: vec!["causal-d".into()],
                trigger_concept_ids: Vec::new(),
                trigger_belief_ids: vec!["belief-d".into()],
                supporting_record_ids: vec!["record-d".into()],
                cause_record_ids: vec!["record-d".into()],
                confidence: 0.43,
                utility_score: 0.30,
                risk_score: 0.21,
                policy_strength: 0.42,
                state: crate::policy::PolicyState::Rejected,
                last_updated: 1.0,
            },
        ];

        {
            let mut engine = aura.policy_engine.write();
            for hint in hints {
                engine.key_index.insert(hint.key.clone(), hint.id.clone());
                engine.hints.insert(hint.id.clone(), hint);
            }
        }

        let suppressed = aura.get_suppressed_policy_hints(Some("alpha"), Some(10));
        assert_eq!(suppressed.len(), 1);
        assert_eq!(suppressed[0].id, "policy-suppressed-verify");

        let rejected = aura.get_rejected_policy_hints(None, Some(10));
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].id, "policy-rejected-warn");

        let summary = aura.get_policy_lifecycle_summary(Some(10), Some(10));
        assert_eq!(summary.total_hints, 4);
        assert_eq!(summary.active_hints, 2);
        assert_eq!(summary.stable_hints, 1);
        assert_eq!(summary.candidate_hints, 1);
        assert_eq!(summary.suppressed_hints, 1);
        assert_eq!(summary.rejected_hints, 1);
        assert!(summary
            .action_summaries
            .iter()
            .any(|item| item.action_kind == "avoid" && item.stable_hints == 1));
        assert!(summary
            .domain_summaries
            .iter()
            .any(|item| item.namespace == "alpha"
                && item.domain == "deploy"
                && item.active_hints == 2
                && item.suppressed_hints == 1));

        let pressure = aura.get_policy_pressure_report(None, Some(10));
        assert!(!pressure.is_empty());
        assert_eq!(pressure[0].namespace, "alpha");
        assert_eq!(pressure[0].domain, "deploy");
        assert_eq!(pressure[0].strongest_hint_id, "policy-stable-avoid");
        assert_eq!(pressure[0].suppressed_hints, 1);

        Ok(())
    }

    #[test]
    fn test_cross_namespace_digest_is_read_only_and_bounded() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let alpha_cause = aura.store(
            "Alpha deploy rollback decision",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "rollback".into(), "ops".into()]),
            None,
            Some("decision"),
            None,
            None,
            Some(false),
            None,
            Some("alpha"),
            Some("fact"),
        )?;
        let alpha_effect = aura.store(
            "Alpha deploy stability improved",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "stable".into(), "ops".into()]),
            None,
            Some("report"),
            None,
            None,
            Some(false),
            Some(&alpha_cause.id),
            Some("alpha"),
            Some("fact"),
        )?;
        let beta_cause = aura.store(
            "Beta deploy rollback decision",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "rollback".into(), "ops".into()]),
            None,
            Some("decision"),
            None,
            None,
            Some(false),
            None,
            Some("beta"),
            Some("fact"),
        )?;
        let beta_effect = aura.store(
            "Beta deploy stability improved",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "stable".into(), "ops".into()]),
            None,
            Some("report"),
            None,
            None,
            Some(false),
            Some(&beta_cause.id),
            Some("beta"),
            Some("fact"),
        )?;

        aura.link_records(
            &alpha_cause.id,
            &alpha_effect.id,
            "belongs_to_project",
            Some(0.92),
        )?;
        aura.link_records(
            &beta_cause.id,
            &beta_effect.id,
            "belongs_to_project",
            Some(0.92),
        )?;

        {
            let mut concept_engine = aura.concept_engine.write();
            for (id, namespace, record_id) in [
                ("concept-alpha", "alpha", &alpha_effect.id),
                ("concept-beta", "beta", &beta_effect.id),
            ] {
                let concept = crate::concept::ConceptCandidate {
                    id: id.into(),
                    key: format!("{namespace}:deploy:stability"),
                    namespace: namespace.into(),
                    semantic_type: "event".into(),
                    belief_ids: vec![format!("belief-{namespace}")],
                    record_ids: vec![record_id.clone()],
                    core_terms: vec!["deploy".into(), "stable".into()],
                    shell_terms: vec!["rollback".into()],
                    tags: vec!["deploy".into(), "ops".into()],
                    support_mass: 1.0,
                    confidence: 0.88,
                    stability: 2.0,
                    cohesion: 0.9,
                    abstraction_score: 0.82,
                    state: crate::concept::ConceptState::Stable,
                    last_updated: 1.0,
                };
                concept_engine
                    .key_index
                    .insert(concept.key.clone(), concept.id.clone());
                concept_engine.concepts.insert(concept.id.clone(), concept);
            }
        }

        {
            let mut causal_engine = aura.causal_engine.write();
            for (id, namespace, cause_id, effect_id) in [
                ("causal-alpha", "alpha", &alpha_cause.id, &alpha_effect.id),
                ("causal-beta", "beta", &beta_cause.id, &beta_effect.id),
            ] {
                let pattern = crate::causal::CausalPattern {
                    id: id.into(),
                    key: format!("{namespace}:deploy:rollback=>stable"),
                    namespace: namespace.into(),
                    cause_belief_ids: Vec::new(),
                    effect_belief_ids: Vec::new(),
                    cause_record_ids: vec![cause_id.clone()],
                    effect_record_ids: vec![effect_id.clone()],
                    support_count: 2,
                    explicit_support_count: 1,
                    temporal_support_count: 1,
                    unique_temporal_windows: 1,
                    effect_record_signature_variants: 1,
                    positive_effect_signals: 1,
                    negative_effect_signals: 0,
                    counterevidence: 0,
                    explicit_support_total_for_cause: 1,
                    explicit_effect_variants_for_cause: 1,
                    transition_lift: 1.0,
                    temporal_consistency: 1.0,
                    outcome_stability: 1.0,
                    causal_strength: 0.81,
                    invalidation_reason: None,
                    invalidated_at: None,
                    state: crate::causal::CausalState::Stable,
                    last_updated: 1.0,
                };
                causal_engine
                    .key_index
                    .insert(pattern.key.clone(), pattern.id.clone());
                causal_engine.patterns.insert(pattern.id.clone(), pattern);
            }
        }

        let before_alpha = aura.get("alpha").is_none();
        let digest = aura.cross_namespace_digest();

        assert_eq!(digest.namespace_count, 2);
        assert_eq!(digest.namespaces.len(), 2);
        assert_eq!(digest.pairs.len(), 1);
        assert!(before_alpha);

        let alpha = digest
            .namespaces
            .iter()
            .find(|namespace| namespace.namespace == "alpha")
            .expect("alpha namespace digest");
        assert_eq!(alpha.record_count, 2);
        assert_eq!(alpha.stable_concept_count, 1);
        assert_eq!(alpha.top_concepts.len(), 1);
        assert_eq!(alpha.concept_signatures.len(), 1);
        assert!(alpha.tags.iter().any(|tag| tag == "deploy"));
        assert!(alpha
            .structural_relation_types
            .iter()
            .any(|relation_type| relation_type == "belongs_to_project"));

        let pair = &digest.pairs[0];
        assert_eq!(pair.namespace_a, "alpha");
        assert_eq!(pair.namespace_b, "beta");
        assert_eq!(pair.shared_concept_signatures.len(), 1);
        assert!(pair.concept_signature_similarity > 0.0);
        assert!(pair.shared_tags.iter().any(|tag| tag == "deploy"));
        assert!(pair.tag_jaccard > 0.0);
        assert!(pair
            .shared_structural_relation_types
            .iter()
            .any(|relation_type| relation_type == "belongs_to_project"));
        assert!(pair.structural_similarity > 0.0);
        assert_eq!(pair.shared_causal_signatures.len(), 1);
        assert!(pair.causal_signature_similarity > 0.0);

        Ok(())
    }

    #[test]
    fn test_belief_instability_surfaces_and_recent_corrections() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let mut stable_belief = crate::belief::Belief::new("default:ops:stable".into());
        stable_belief.id = "belief-stable".into();
        stable_belief.state = crate::belief::BeliefState::Resolved;
        stable_belief.confidence = 0.82;
        stable_belief.score = 0.82;
        stable_belief.stability = 3.5;
        stable_belief.volatility = 0.01;

        let mut volatile_belief = crate::belief::Belief::new("default:ops:volatile".into());
        volatile_belief.id = "belief-volatile".into();
        volatile_belief.state = crate::belief::BeliefState::Unresolved;
        volatile_belief.confidence = 0.41;
        volatile_belief.score = 0.41;
        volatile_belief.stability = 0.4;
        volatile_belief.volatility = 0.31;

        {
            let mut engine = aura.belief_engine.write();
            engine
                .key_index
                .insert(stable_belief.key.clone(), stable_belief.id.clone());
            engine
                .key_index
                .insert(volatile_belief.key.clone(), volatile_belief.id.clone());
            engine
                .beliefs
                .insert(stable_belief.id.clone(), stable_belief.clone());
            engine
                .beliefs
                .insert(volatile_belief.id.clone(), volatile_belief.clone());
        }
        {
            let engine = aura.belief_engine.read();
            aura.belief_store.save(&engine)?;
        }

        assert!(aura.deprecate_belief_with_reason("belief-volatile", "manual_review")?);

        let high_vol = aura.get_high_volatility_beliefs(Some(0.20), Some(10));
        assert_eq!(high_vol.len(), 1);
        assert_eq!(high_vol[0].id, "belief-volatile");

        let low_stability = aura.get_low_stability_beliefs(Some(1.0), Some(10));
        assert_eq!(low_stability.len(), 1);
        assert_eq!(low_stability[0].id, "belief-volatile");

        let summary = aura.get_belief_instability_summary();
        assert_eq!(summary.total_beliefs, 2);
        assert_eq!(summary.high_volatility_count, 1);
        assert_eq!(summary.low_stability_count, 1);
        assert_eq!(summary.volatility_bands.high, 1);

        let corrected = aura.get_recently_corrected_beliefs(Some(10));
        assert_eq!(corrected.len(), 1);
        assert_eq!(corrected[0].id, "belief-volatile");

        Ok(())
    }

    #[test]
    fn test_contradiction_clusters_are_deterministic_and_persist_across_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let rec_a = aura.store(
            "Deploy path A failed under smoke load",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            Some("recorded"),
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;
        let rec_b = aura.store(
            "Deploy path B succeeded under smoke load",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            Some("recorded"),
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;

        {
            let mut engine = aura.belief_engine.write();

            let hyp_a = crate::belief::Hypothesis {
                id: "hyp-cluster-a".into(),
                belief_id: "belief-cluster-a".into(),
                prototype_record_ids: vec![rec_a.id.clone()],
                score: 0.52,
                confidence: 0.72,
                support_mass: 1.0,
                conflict_mass: 0.8,
                recency: 1.0,
                consistency: 0.9,
            };
            let belief_a = crate::belief::Belief {
                id: "belief-cluster-a".into(),
                key: "default:deploy:cluster-a".into(),
                hypothesis_ids: vec![hyp_a.id.clone()],
                winner_id: None,
                state: crate::belief::BeliefState::Unresolved,
                score: 0.52,
                confidence: 0.72,
                support_mass: 1.0,
                conflict_mass: 0.8,
                stability: 0.6,
                volatility: 0.26,
                last_updated: 1.0,
            };

            let hyp_b = crate::belief::Hypothesis {
                id: "hyp-cluster-b".into(),
                belief_id: "belief-cluster-b".into(),
                prototype_record_ids: vec![rec_b.id.clone()],
                score: 0.49,
                confidence: 0.68,
                support_mass: 1.0,
                conflict_mass: 0.7,
                recency: 1.0,
                consistency: 0.9,
            };
            let belief_b = crate::belief::Belief {
                id: "belief-cluster-b".into(),
                key: "default:deploy:cluster-b".into(),
                hypothesis_ids: vec![hyp_b.id.clone()],
                winner_id: None,
                state: crate::belief::BeliefState::Unresolved,
                score: 0.49,
                confidence: 0.68,
                support_mass: 1.0,
                conflict_mass: 0.7,
                stability: 0.8,
                volatility: 0.22,
                last_updated: 1.0,
            };

            engine
                .key_index
                .insert(belief_a.key.clone(), belief_a.id.clone());
            engine
                .key_index
                .insert(belief_b.key.clone(), belief_b.id.clone());
            engine.hypotheses.insert(hyp_a.id.clone(), hyp_a);
            engine.hypotheses.insert(hyp_b.id.clone(), hyp_b);
            engine.beliefs.insert(belief_a.id.clone(), belief_a);
            engine.beliefs.insert(belief_b.id.clone(), belief_b);
        }
        {
            let engine = aura.belief_engine.read();
            aura.belief_store.save(&engine)?;
        }

        let first = aura.get_contradiction_clusters(None, Some(10));
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].belief_ids.len(), 2);
        assert_eq!(first[0].namespace, "default");
        assert!(first[0].shared_tags.iter().any(|tag| tag == "deploy"));

        drop(aura);
        let reopened = Aura::open(root)?;
        let second = reopened.get_contradiction_clusters(None, Some(10));
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].id, first[0].id);
        assert_eq!(second[0].belief_ids, first[0].belief_ids);
        assert_eq!(second[0].record_ids, first[0].record_ids);

        let summary = reopened.get_belief_instability_summary();
        assert_eq!(summary.contradiction_cluster_count, 1);

        Ok(())
    }

    #[test]
    fn test_contradiction_review_queue_prioritizes_downstream_impact() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let rec_a = aura.store(
            "Deploy path A failed under smoke load",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            Some("recorded"),
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;
        let rec_b = aura.store(
            "Deploy path B succeeded under smoke load",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            Some("recorded"),
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;
        let rec_c = aura.store(
            "Logging path mismatch under smoke load",
            Some(Level::Domain),
            Some(vec!["logging".into()]),
            None,
            Some("note"),
            Some("recorded"),
            None,
            Some(false),
            None,
            Some("default"),
            Some("fact"),
        )?;

        {
            let mut engine = aura.belief_engine.write();
            let make_hyp = |id: &str, belief_id: &str, record_id: &str| crate::belief::Hypothesis {
                id: id.into(),
                belief_id: belief_id.into(),
                prototype_record_ids: vec![record_id.into()],
                score: 0.5,
                confidence: 0.7,
                support_mass: 1.0,
                conflict_mass: 0.8,
                recency: 1.0,
                consistency: 0.9,
            };
            let make_belief =
                |id: &str, key: &str, hyp_id: &str, conflict_mass: f32, volatility: f32| {
                    crate::belief::Belief {
                        id: id.into(),
                        key: key.into(),
                        hypothesis_ids: vec![hyp_id.into()],
                        winner_id: None,
                        state: crate::belief::BeliefState::Unresolved,
                        score: 0.5,
                        confidence: 0.7,
                        support_mass: 1.0,
                        conflict_mass,
                        stability: 0.7,
                        volatility,
                        last_updated: 1.0,
                    }
                };

            let hyp_a = make_hyp("hyp-q-a", "belief-q-a", &rec_a.id);
            let hyp_b = make_hyp("hyp-q-b", "belief-q-b", &rec_b.id);
            let hyp_c = make_hyp("hyp-q-c", "belief-q-c", &rec_c.id);
            let belief_a =
                make_belief("belief-q-a", "default:deploy:queue-a", &hyp_a.id, 0.9, 0.28);
            let belief_b =
                make_belief("belief-q-b", "default:deploy:queue-b", &hyp_b.id, 0.8, 0.24);
            let belief_c = make_belief(
                "belief-q-c",
                "default:logging:queue-c",
                &hyp_c.id,
                0.2,
                0.12,
            );

            for belief in [&belief_a, &belief_b, &belief_c] {
                engine
                    .key_index
                    .insert(belief.key.clone(), belief.id.clone());
            }
            engine.hypotheses.insert(hyp_a.id.clone(), hyp_a);
            engine.hypotheses.insert(hyp_b.id.clone(), hyp_b);
            engine.hypotheses.insert(hyp_c.id.clone(), hyp_c);
            engine.beliefs.insert(belief_a.id.clone(), belief_a.clone());
            engine.beliefs.insert(belief_b.id.clone(), belief_b.clone());
            engine.beliefs.insert(belief_c.id.clone(), belief_c.clone());
        }
        {
            let mut causal_engine = aura.causal_engine.write();
            let pattern = crate::causal::CausalPattern {
                id: "causal-q-1".into(),
                key: "default:deploy:impact".into(),
                namespace: "default".into(),
                cause_belief_ids: vec!["belief-q-a".into()],
                effect_belief_ids: vec!["belief-q-b".into()],
                cause_record_ids: vec![rec_a.id.clone()],
                effect_record_ids: vec![rec_b.id.clone()],
                support_count: 2,
                explicit_support_count: 1,
                temporal_support_count: 1,
                unique_temporal_windows: 1,
                effect_record_signature_variants: 1,
                positive_effect_signals: 1,
                negative_effect_signals: 0,
                counterevidence: 0,
                explicit_support_total_for_cause: 1,
                explicit_effect_variants_for_cause: 1,
                transition_lift: 1.2,
                temporal_consistency: 1.0,
                causal_strength: 0.82,
                outcome_stability: 0.9,
                state: crate::causal::CausalState::Stable,
                last_updated: 1.0,
                invalidation_reason: None,
                invalidated_at: None,
            };
            causal_engine
                .key_index
                .insert(pattern.key.clone(), pattern.id.clone());
            causal_engine.patterns.insert(pattern.id.clone(), pattern);
        }
        {
            let mut policy_engine = aura.policy_engine.write();
            let hint = crate::policy::PolicyHint {
                id: "policy-q-1".into(),
                key: "default:deploy:verify".into(),
                namespace: "default".into(),
                domain: "deploy".into(),
                action_kind: crate::policy::PolicyActionKind::VerifyFirst,
                recommendation: "Verify deploy corridor".into(),
                trigger_causal_ids: vec![],
                trigger_concept_ids: vec![],
                trigger_belief_ids: vec!["belief-q-a".into(), "belief-q-b".into()],
                supporting_record_ids: vec![rec_a.id.clone(), rec_b.id.clone()],
                cause_record_ids: vec![rec_a.id.clone()],
                confidence: 0.8,
                utility_score: 0.2,
                risk_score: 0.7,
                policy_strength: 0.75,
                state: crate::policy::PolicyState::Stable,
                last_updated: 1.0,
            };
            policy_engine
                .key_index
                .insert(hint.key.clone(), hint.id.clone());
            policy_engine.hints.insert(hint.id.clone(), hint);
        }

        let queue = aura.get_contradiction_review_queue(None, Some(10));
        assert_eq!(queue.len(), 2);
        assert_eq!(queue[0].belief_ids.len(), 2);
        assert!(queue[0].downstream_impact > queue[1].downstream_impact);
        assert!(queue[0].priority_score > queue[1].priority_score);
        assert_eq!(queue[0].namespace, "default");

        Ok(())
    }

    #[test]
    fn test_record_salience_surfaces_and_persists() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let alpha = aura.store(
            "Mission-critical family contact",
            Some(Level::Identity),
            Some(vec!["family".into(), "critical".into()]),
            None,
            Some("note"),
            Some("recorded"),
            None,
            Some(false),
            None,
            Some("default"),
            Some("preference"),
        )?;
        let _beta = aura.store(
            "Routine grocery reminder",
            Some(Level::Working),
            Some(vec!["routine".into()]),
            None,
            Some("note"),
            Some("recorded"),
            None,
            Some(false),
            None,
            Some("default"),
            Some("preference"),
        )?;

        let marked = aura
            .mark_record_salience(&alpha.id, 0.85, Some("user_priority"))?
            .expect("record should exist");
        assert!((marked.salience - 0.85).abs() < 0.001);
        assert_eq!(
            marked.metadata.get(RECORD_SALIENCE_REASON_KEY),
            Some(&"user_priority".to_string())
        );

        let top = aura.get_high_salience_records(Some(0.50), Some(10));
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].id, alpha.id);

        let summary = aura.get_salience_summary();
        assert_eq!(summary.total_records, 2);
        assert_eq!(summary.high_salience_count, 1);
        assert_eq!(summary.bands.high, 1);
        assert!(summary.max_salience >= 0.85);

        let explained = aura
            .explain_record(&alpha.id)
            .expect("explanation should exist");
        assert!((explained.salience - 0.85).abs() < 0.001);
        assert_eq!(explained.salience_reason.as_deref(), Some("user_priority"));

        drop(aura);
        let reopened = Aura::open(root)?;
        let persisted = reopened.get(&alpha.id).expect("record should persist");
        assert!((persisted.salience - 0.85).abs() < 0.001);
        assert_eq!(
            persisted.metadata.get(RECORD_SALIENCE_REASON_KEY),
            Some(&"user_priority".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_cross_namespace_digest_filtered_limits_namespaces_and_concepts() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let alpha = aura.store(
            "Alpha deploy note",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            None,
            None,
            Some(false),
            None,
            Some("alpha"),
            Some("fact"),
        )?;
        let _beta = aura.store(
            "Beta deploy note",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            None,
            None,
            Some(false),
            None,
            Some("beta"),
            Some("fact"),
        )?;

        {
            let mut concept_engine = aura.concept_engine.write();
            for idx in 0..3 {
                let concept = crate::concept::ConceptCandidate {
                    id: format!("concept-alpha-{idx}"),
                    key: format!("alpha:deploy:{idx}"),
                    namespace: "alpha".into(),
                    semantic_type: "fact".into(),
                    belief_ids: vec![format!("belief-alpha-{idx}")],
                    record_ids: vec![alpha.id.clone()],
                    core_terms: vec!["deploy".into()],
                    shell_terms: vec!["ops".into()],
                    tags: vec!["deploy".into()],
                    support_mass: 1.0,
                    confidence: 0.90 - idx as f32 * 0.05,
                    stability: 1.0,
                    cohesion: 1.0,
                    abstraction_score: 0.8,
                    state: crate::concept::ConceptState::Stable,
                    last_updated: 1.0,
                };
                concept_engine
                    .key_index
                    .insert(concept.key.clone(), concept.id.clone());
                concept_engine.concepts.insert(concept.id.clone(), concept);
            }
        }

        let digest = aura.cross_namespace_digest_filtered(Some(&["alpha"]), Some(2));
        assert_eq!(digest.namespace_count, 1);
        assert_eq!(digest.namespaces.len(), 1);
        assert_eq!(digest.pairs.len(), 0);
        assert_eq!(digest.namespaces[0].namespace, "alpha");
        assert_eq!(digest.namespaces[0].top_concepts.len(), 2);

        Ok(())
    }

    #[test]
    fn test_cross_namespace_digest_v2_filters_and_compact_summary() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().to_str().unwrap();
        let aura = Aura::open(root)?;

        let _alpha = aura.store(
            "Alpha deploy improvement note",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            None,
            None,
            Some(false),
            None,
            Some("alpha"),
            Some("fact"),
        )?;
        let _beta = aura.store(
            "Beta deploy improvement note",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            None,
            None,
            Some(false),
            None,
            Some("beta"),
            Some("fact"),
        )?;
        let beta_second = aura.store(
            "Beta deploy summary note",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            None,
            None,
            Some(false),
            None,
            Some("beta"),
            Some("fact"),
        )?;
        let alpha_second = aura.store(
            "Alpha deploy summary note",
            Some(Level::Domain),
            Some(vec!["deploy".into(), "ops".into()]),
            None,
            Some("note"),
            None,
            None,
            Some(false),
            None,
            Some("alpha"),
            Some("fact"),
        )?;
        let _gamma = aura.store(
            "Gamma singleton note",
            Some(Level::Domain),
            Some(vec!["misc".into()]),
            None,
            Some("note"),
            None,
            None,
            Some(false),
            None,
            Some("gamma"),
            Some("fact"),
        )?;

        {
            let mut concept_engine = aura.concept_engine.write();
            for namespace in ["alpha", "beta"] {
                let concept = crate::concept::ConceptCandidate {
                    id: format!("concept-{namespace}"),
                    key: format!("{namespace}:deploy:stable"),
                    namespace: namespace.into(),
                    semantic_type: "fact".into(),
                    belief_ids: vec![format!("belief-{namespace}")],
                    record_ids: vec![if namespace == "alpha" {
                        alpha_second.id.clone()
                    } else {
                        beta_second.id.clone()
                    }],
                    core_terms: vec!["deploy".into()],
                    shell_terms: vec!["ops".into()],
                    tags: vec!["deploy".into()],
                    support_mass: 1.0,
                    confidence: 0.9,
                    stability: 1.0,
                    cohesion: 1.0,
                    abstraction_score: 0.8,
                    state: crate::concept::ConceptState::Stable,
                    last_updated: 1.0,
                };
                concept_engine
                    .key_index
                    .insert(concept.key.clone(), concept.id.clone());
                concept_engine.concepts.insert(concept.id.clone(), concept);
            }
        }

        {
            let mut belief_engine = aura.belief_engine.write();
            let mut belief = crate::belief::Belief::new("alpha:deploy:belief".into());
            belief.id = "alpha-belief".into();
            belief.state = crate::belief::BeliefState::Unresolved;
            belief.volatility = 0.24;
            belief.stability = 0.6;
            belief_engine
                .key_index
                .insert(belief.key.clone(), belief.id.clone());
            belief_engine.beliefs.insert(belief.id.clone(), belief);
        }
        {
            let engine = aura.belief_engine.read();
            aura.belief_store.save(&engine)?;
        }

        assert!(aura.deprecate_belief_with_reason("alpha-belief", "digest_density")?);

        let options = CrossNamespaceDigestOptions {
            min_record_count: 2,
            top_concepts_limit: 5,
            pairwise_similarity_threshold: 0.1,
            compact_summary: true,
            include_concepts: true,
            include_tags: true,
            include_structural: true,
            include_causal: true,
            include_belief_states: true,
            include_corrections: true,
        };
        let digest = aura.cross_namespace_digest_with_options(None, options);

        assert_eq!(digest.namespace_count, 2);
        assert!(digest.compact_summary);
        assert!(digest
            .included_dimensions
            .iter()
            .any(|dimension| dimension == "belief_states"));
        assert_eq!(digest.namespaces.len(), 2);
        assert!(digest
            .namespaces
            .iter()
            .all(|namespace| namespace.namespace != "gamma"));
        assert!(digest
            .namespaces
            .iter()
            .all(|namespace| namespace.top_concepts.is_empty()));
        let alpha_ns = digest
            .namespaces
            .iter()
            .find(|namespace| namespace.namespace == "alpha")
            .expect("alpha namespace");
        assert!(alpha_ns.belief_state_summary.is_some());
        assert!(alpha_ns.correction_count.unwrap_or_default() >= 1);
        assert!(alpha_ns.correction_density.unwrap_or_default() >= 0.0);
        // compact_summary: true suppresses all signatures → similarity=0 → no pairs formed
        assert_eq!(digest.pairs.len(), 0);

        Ok(())
    }
}
