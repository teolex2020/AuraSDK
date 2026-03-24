use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use parking_lot::RwLock;

use crate::background_brain::{BackgroundBrain, MaintenanceConfig};
use crate::cache::{RecallCache, StructuredRecallCache};
use crate::causal::CausalRerankMode;
use crate::circuit_breaker::CircuitBreaker;
use crate::concept::ConceptSurfaceMode;
use crate::experience::{ExperienceQueue, PlasticityMode, PlasticityPolicy};
use crate::maintenance_service::ConceptSurfaceCounters;
use crate::policy::PolicyRerankMode;
use crate::recall::BeliefRerankMode;
use crate::trust::{TagTaxonomy, TrustConfig};

pub(crate) struct AuraConfigState {
    pub(crate) taxonomy: RwLock<TagTaxonomy>,
    pub(crate) trust_config: RwLock<TrustConfig>,
    pub(crate) maintenance_config: RwLock<MaintenanceConfig>,
    pub(crate) circuit_breaker: CircuitBreaker,
    pub(crate) path: PathBuf,
}

impl AuraConfigState {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self {
            taxonomy: RwLock::new(TagTaxonomy::default()),
            trust_config: RwLock::new(TrustConfig::default()),
            maintenance_config: RwLock::new(MaintenanceConfig::default()),
            circuit_breaker: CircuitBreaker::default(),
            path,
        }
    }
}

pub(crate) struct AuraRuntimeState {
    pub(crate) recall_cache: RecallCache,
    pub(crate) structured_recall_cache: StructuredRecallCache,
    pub(crate) sdr_lookup_cache: RwLock<HashMap<String, Vec<u16>>>,
    pub(crate) background: RwLock<Option<BackgroundBrain>>,
    pub(crate) belief_rerank_mode: AtomicU8,
    pub(crate) concept_surface_mode: AtomicU8,
    pub(crate) causal_rerank_mode: AtomicU8,
    pub(crate) policy_rerank_mode: AtomicU8,
    pub(crate) concept_surface_global_calls: AtomicU64,
    pub(crate) concept_surface_namespace_calls: AtomicU64,
    pub(crate) concept_surface_record_calls: AtomicU64,
    pub(crate) concept_surface_results_returned: AtomicU64,
    pub(crate) concept_surface_record_results_returned: AtomicU64,
    pub(crate) maintenance_trends: RwLock<Vec<crate::background_brain::MaintenanceTrendSnapshot>>,
    pub(crate) reflection_summaries: RwLock<Vec<crate::background_brain::ReflectionSummary>>,
    pub(crate) persistence_manifest: RwLock<crate::persistence_contract::PersistenceManifest>,
    pub(crate) startup_validation: RwLock<crate::startup_validation::StartupValidationReport>,

    // ── v5: Autonomous Cognitive Plasticity ──
    /// Atomic plasticity mode. Default: Off — system never changes silently.
    pub(crate) plasticity_mode: AtomicU8,
    /// Pending experience captures awaiting the next maintenance cycle (phase 3.6).
    pub(crate) experience_queue: ExperienceQueue,

    // ── Phase 3.2: Plasticity risk telemetry (cumulative across calls) ──
    /// Total hallucination alerts since last reset.
    pub(crate) plasticity_hallucination_alerts: AtomicU64,
    /// Total contradiction events seen across all capture_experience() calls.
    pub(crate) plasticity_contradictions_total: AtomicU64,
    /// Total events processed across all capture_experience() calls.
    pub(crate) plasticity_events_total: AtomicU64,

    // ── Phase 4.3: Namespace plasticity freeze ──
    /// Namespaces where inference is not allowed to create new records.
    /// Checked inside capture_experience() before apply_experience().
    pub(crate) frozen_plasticity_namespaces: RwLock<HashSet<String>>,

    // ── Bug 3 fix: Full mode operator-supplied policy ──
    /// Custom policy used when PlasticityMode::Full.
    /// Set via set_plasticity_policy(), defaults to PlasticityPolicy::default().
    pub(crate) custom_plasticity_policy: RwLock<PlasticityPolicy>,
}

impl AuraRuntimeState {
    pub(crate) fn new() -> Self {
        Self {
            recall_cache: RecallCache::default(),
            structured_recall_cache: StructuredRecallCache::default(),
            sdr_lookup_cache: RwLock::new(HashMap::new()),
            background: RwLock::new(None),
            belief_rerank_mode: AtomicU8::new(BeliefRerankMode::Limited as u8),
            concept_surface_mode: AtomicU8::new(ConceptSurfaceMode::Inspect as u8),
            causal_rerank_mode: AtomicU8::new(CausalRerankMode::Limited as u8),
            policy_rerank_mode: AtomicU8::new(PolicyRerankMode::Limited as u8),
            concept_surface_global_calls: AtomicU64::new(0),
            concept_surface_namespace_calls: AtomicU64::new(0),
            concept_surface_record_calls: AtomicU64::new(0),
            concept_surface_results_returned: AtomicU64::new(0),
            concept_surface_record_results_returned: AtomicU64::new(0),
            maintenance_trends: RwLock::new(Vec::new()),
            reflection_summaries: RwLock::new(Vec::new()),
            persistence_manifest: RwLock::new(
                crate::persistence_contract::PersistenceManifest::current(),
            ),
            startup_validation: RwLock::new(Default::default()),
            plasticity_mode: AtomicU8::new(PlasticityMode::Off as u8),
            experience_queue: ExperienceQueue::new(),
            plasticity_hallucination_alerts: AtomicU64::new(0),
            plasticity_contradictions_total: AtomicU64::new(0),
            plasticity_events_total: AtomicU64::new(0),
            frozen_plasticity_namespaces: RwLock::new(HashSet::new()),
            custom_plasticity_policy: RwLock::new(PlasticityPolicy::default()),
        }
    }

    pub(crate) fn clear_recall_caches(&self) {
        self.recall_cache.clear();
        self.structured_recall_cache.clear();
    }

    pub(crate) fn concept_surface_mode(&self) -> ConceptSurfaceMode {
        match self.concept_surface_mode.load(Ordering::Relaxed) {
            1 => ConceptSurfaceMode::Inspect,
            2 => ConceptSurfaceMode::Limited,
            _ => ConceptSurfaceMode::Off,
        }
    }

    pub(crate) fn plasticity_mode(&self) -> PlasticityMode {
        PlasticityMode::from_u8(self.plasticity_mode.load(Ordering::Relaxed))
    }

    pub(crate) fn concept_surface_counters(&self) -> ConceptSurfaceCounters<'_> {
        ConceptSurfaceCounters {
            global_calls: &self.concept_surface_global_calls,
            namespace_calls: &self.concept_surface_namespace_calls,
            record_calls: &self.concept_surface_record_calls,
            concepts_returned: &self.concept_surface_results_returned,
            record_annotations_returned: &self.concept_surface_record_results_returned,
        }
    }
}
