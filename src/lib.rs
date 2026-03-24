//! Aura — Unified Cognitive Memory SDK
//!
//! A high-performance cognitive memory system for AI agents.
//! Combines SDR-based retrieval with hierarchical decay and RRF Fusion recall.
//!
//! # Features
//! - `full` - Complete feature set (default)
//! - `python` - Python bindings via PyO3
//! - `server` - HTTP/REST API server
//! - `encryption` - ChaCha20-Poly1305 at-rest encryption
//! - `sync` - P2P synchronization with CRDT merge
//! - `embedded` - Minimal footprint for IoT/Edge
//! - `lite` - Reduced SDR resolution (16k bits vs 256k)

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::new_without_default)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_guards)]

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ── Storage abstraction (WASM-portable) ──
pub mod backend;

// ── FROM aura-memory (already Rust) ──
mod anchors;
pub mod backup;
pub mod canonical;
mod cortex;
pub mod federated;
pub mod gates;
pub mod index;
pub mod learner;
pub mod neuromorphic;
pub mod rbac;
mod salience;
pub mod sdr;
pub mod semantic;
pub mod storage;
mod types;
pub mod versioning;

// ── v5: Autonomous Cognitive Plasticity ──
pub mod experience;

#[cfg(feature = "encryption")]
pub mod crypto;

#[cfg(not(feature = "encryption"))]
pub mod crypto {
    //! Crypto stub for builds without encryption
    use anyhow::{anyhow, Result};

    #[derive(Clone)]
    pub struct EncryptionKey;

    impl EncryptionKey {
        pub fn generate() -> Self {
            Self
        }
        pub fn from_password(_: &str, _: &[u8; 16]) -> Result<Self> {
            Err(anyhow!(
                "Encryption not enabled - rebuild with 'encryption' feature"
            ))
        }
        pub fn as_bytes(&self) -> &[u8; 32] {
            &[0u8; 32]
        }
        pub fn save_to_file(&self, _: &std::path::Path, _: &str) -> Result<()> {
            Err(anyhow!("Encryption not enabled"))
        }
        pub fn load_from_file(_: &std::path::Path, _: &str) -> Result<Self> {
            Err(anyhow!("Encryption not enabled"))
        }
    }

    pub fn encrypt_data(_: &[u8], _: &EncryptionKey) -> Result<Vec<u8>> {
        Err(anyhow!("Encryption not enabled"))
    }
    pub fn decrypt_data(_: &[u8], _: &EncryptionKey) -> Result<Vec<u8>> {
        Err(anyhow!("Encryption not enabled"))
    }
}

pub mod audit;
mod aura_state;
pub mod tenant;

pub mod sync {
    //! Sync stub for builds without sync feature
    use anyhow::{anyhow, Result};

    #[derive(Clone, Debug, Default)]
    pub struct SyncConfig;

    pub struct SyncManager;

    impl SyncManager {
        pub async fn new(_: &str, _: SyncConfig) -> Result<Self> {
            Err(anyhow!("Sync not enabled - rebuild with 'sync' feature"))
        }
    }
}

#[cfg(feature = "server")]
pub mod server;

#[cfg(not(feature = "server"))]
pub mod server {
    //! Server stub for builds without server feature
    use anyhow::{anyhow, Result};
    pub fn start_server(_port: u16, _path: &str) -> Result<()> {
        Err(anyhow!(
            "Server not enabled - rebuild with 'server' feature"
        ))
    }
}

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "telemetry")]
pub mod telemetry;

#[cfg(not(feature = "telemetry"))]
pub mod telemetry {
    //! Telemetry stub for builds without telemetry feature
    use anyhow::{anyhow, Result};

    pub struct TracerProvider;

    pub fn init_telemetry() -> Result<TracerProvider> {
        Err(anyhow!(
            "Telemetry not enabled - rebuild with 'telemetry' feature"
        ))
    }

    pub fn shutdown_telemetry(_: TracerProvider) {}
}

pub mod license;

// ── FROM aura-cognitive (rewritten to Rust) ──
pub mod cognitive_store;
pub mod consolidation;
pub mod graph;
pub mod insights;
pub mod levels;
pub mod ngram;
pub mod recall;
mod recall_service;
pub mod record;
pub mod retention;
pub mod scheduler;
pub mod semantic_learner;
pub mod synonym;

// ── SDK Wrapper (from brain_tools.py — generic parts) ──
pub mod cache;
pub mod circuit_breaker;
pub mod credibility;
pub mod guards;
pub mod identity;
pub mod relation;
pub mod research;
pub mod trust;

// ── Optional Embedding Support ──
pub mod embedding;

// ── Living Memory (Background Brain) ──
pub mod api_groups;
pub mod background_brain;
mod maintenance_service;

// ── Epistemic Belief Layer ──
pub mod belief;

// ── Concept Discovery Layer ──
pub mod concept;

// ── Causal Pattern Discovery Layer ──
pub mod causal;

// ── Policy Hint Layer ──
pub mod epistemic_runtime;
pub mod persistence_contract;
pub mod policy;
pub mod startup_validation;

// ── Main orchestrator (merges both) ──
pub mod aura;

// ── C FFI bindings ──
#[cfg(feature = "ffi")]
pub mod ffi;

// ── Legacy aura-memory API ──
mod memory;
pub use memory::AuraMemory;

// ── Unified API ──
pub use aura::Aura;
pub use levels::Level;
pub use record::Record;

// Enforce license at module load
#[ctor::ctor]
fn init_license_check() {
    license::enforce_license();
}

// ============= PYTHON BINDINGS =============
#[cfg(feature = "python")]
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main API
    m.add_class::<aura::Aura>()?;
    m.add_class::<levels::Level>()?;
    m.add_class::<record::Record>()?;

    // Tag & Trust Configuration
    m.add_class::<trust::TagTaxonomy>()?;
    m.add_class::<trust::TrustConfig>()?;

    // Living Memory (Background Maintenance)
    m.add_class::<background_brain::MaintenanceConfig>()?;
    m.add_class::<background_brain::MaintenanceReport>()?;
    m.add_class::<background_brain::ArchivalRule>()?;
    m.add_class::<background_brain::DecayReport>()?;
    m.add_class::<background_brain::ReflectReport>()?;
    m.add_class::<background_brain::ConsolidationReport>()?;
    m.add_class::<background_brain::EpistemicPhaseReport>()?;
    m.add_class::<background_brain::BeliefPhaseReport>()?;
    m.add_class::<background_brain::ConceptPhaseReport>()?;
    m.add_class::<background_brain::CausalPhaseReport>()?;
    m.add_class::<background_brain::PolicyPhaseReport>()?;
    m.add_class::<background_brain::MaintenanceTrendSnapshot>()?;
    m.add_class::<background_brain::MaintenanceTrendSummary>()?;
    m.add_class::<background_brain::ReflectionFinding>()?;
    m.add_class::<background_brain::ReflectionJobReport>()?;
    m.add_class::<background_brain::ReflectionKindSummary>()?;
    m.add_class::<background_brain::ReflectionDigest>()?;
    m.add_class::<background_brain::ReflectionSummary>()?;
    m.add_class::<background_brain::PhaseTimings>()?;
    m.add_class::<background_brain::LayerStability>()?;
    m.add_class::<persistence_contract::PersistenceManifest>()?;
    m.add_class::<startup_validation::StartupValidationEvent>()?;
    m.add_class::<startup_validation::StartupValidationReport>()?;
    m.add_class::<aura::OperatorReviewIssue>()?;
    m.add_class::<aura::MemoryHealthDigest>()?;
    m.add_class::<aura::SalienceBands>()?;
    m.add_class::<aura::SalienceSummary>()?;
    m.add_class::<epistemic_runtime::BeliefVolatilityBands>()?;
    m.add_class::<epistemic_runtime::BeliefInstabilitySummary>()?;
    m.add_class::<epistemic_runtime::ContradictionCluster>()?;
    m.add_class::<epistemic_runtime::PolicyActionSummary>()?;
    m.add_class::<epistemic_runtime::PolicyDomainSummary>()?;
    m.add_class::<epistemic_runtime::PolicyPressureArea>()?;
    m.add_class::<epistemic_runtime::PolicyLifecycleSummary>()?;
    m.add_class::<concept::SurfacedConcept>()?;
    m.add_class::<policy::SurfacedPolicyHint>()?;
    m.add_class::<relation::StructuralRelation>()?;
    m.add_class::<relation::RelationEdge>()?;
    m.add_class::<relation::RelationDigest>()?;
    m.add_class::<relation::EntityDigest>()?;
    m.add_class::<relation::EntityRelationEdge>()?;
    m.add_class::<relation::EntityGraphNeighbor>()?;
    m.add_class::<relation::EntityGraphDigest>()?;
    m.add_class::<relation::FamilyRelationMember>()?;
    m.add_class::<relation::FamilyGraphSnapshot>()?;
    m.add_class::<relation::PersonDigest>()?;
    m.add_class::<relation::ProjectGraphSnapshot>()?;
    m.add_class::<relation::ProjectStatusSnapshot>()?;
    m.add_class::<relation::ProjectTimelineEntry>()?;
    m.add_class::<relation::ProjectTimelineSnapshot>()?;
    m.add_class::<relation::ProjectDigest>()?;

    // Identity
    m.add_class::<identity::AgentPersona>()?;
    m.add_class::<identity::PersonaTraits>()?;

    // Circuit Breaker
    m.add_class::<circuit_breaker::CircuitBreakerConfig>()?;

    Ok(())
}
