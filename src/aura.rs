//! Aura — Unified Cognitive Memory orchestrator.
//!
//! This is the SINGLE entry point. Replaces both `AuraMemory` (Rust)
//! and `CognitiveMemory` (Python).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use anyhow::Result;
use tracing::instrument;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::levels::Level;
use crate::record::Record;
use crate::sdr::SDRInterpreter;
use crate::storage::AuraStorage;
use crate::index::InvertedIndex;
use crate::cortex::ActiveCortex;
use crate::cognitive_store::CognitiveStore;
use crate::ngram::NGramIndex;
use crate::synonym::SynonymRing;
use crate::graph::SessionTracker;
use crate::canonical::CanonicalProjector;
use crate::crypto::EncryptionKey;
use crate::audit::AuditLog;
use crate::recall;
use crate::consolidation;
use crate::insights;
use crate::semantic_learner::SemanticLearnerEngine;
use crate::embedding::EmbeddingStore;

// SDK Wrapper modules
use crate::trust::{self, TagTaxonomy, TrustConfig};
use crate::guards;
use crate::cache::{RecallCache, StructuredRecallCache};
use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use crate::research::{ResearchEngine, ResearchProject};
use crate::identity::{self, AgentPersona};
use crate::background_brain::{self, BackgroundBrain, MaintenanceConfig, MaintenanceReport};
use crate::belief::{BeliefEngine, BeliefStore, CoarseKeyMode, SdrLookup};
use crate::concept::{ConceptEngine, ConceptSeedMode, ConceptSimilarityMode, ConceptStore};
use crate::causal::{CausalEngine, CausalStore};
use crate::policy::{PolicyEngine, PolicyStore};
use crate::storage::StoredRecord;

/// Maximum content size (100KB).
const MAX_CONTENT_SIZE: usize = 100 * 1024;
/// Maximum tags per record.
const MAX_TAGS: usize = 50;
/// Surprise threshold — below this similarity, info is considered novel.
const SURPRISE_THRESHOLD: f32 = 0.2;

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
    taxonomy: RwLock<TagTaxonomy>,
    trust_config: RwLock<TrustConfig>,
    recall_cache: RecallCache,
    structured_recall_cache: StructuredRecallCache,
    circuit_breaker: CircuitBreaker,
    research_engine: ResearchEngine,
    maintenance_config: RwLock<MaintenanceConfig>,
    background: RwLock<Option<BackgroundBrain>>,

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

    // ── Belief Reranking (Phase 4 — tri-state: Off/Shadow/Limited) ──
    belief_rerank_mode: std::sync::atomic::AtomicU8,

    // ── Config ──
    #[allow(dead_code)]
    path: PathBuf,
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
        let mut loaded_records = cognitive_store.load_all()?;

        // Fix legacy records: confidence defaults to 0.90 on deserialization,
        // but should match the stored source_type for non-"recorded" records.
        let mut migrated_ids: Vec<String> = Vec::new();
        for rec in loaded_records.values_mut() {
            let expected = Record::default_confidence_for_source(&rec.source_type);
            // Only fix if confidence is at the default 0.90 and source_type disagrees
            if (rec.confidence - 0.90).abs() < 0.001
                && (expected - 0.90).abs() > 0.001
            {
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
            tracing::info!(count = migrated_ids.len(), "Migrated legacy record confidence values");
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
        let belief_engine = belief_store.load().unwrap_or_default();

        // Concept discovery layer
        // Concepts are derived state — always start empty, rebuild after first
        // maintenance. The concepts.cog file is only a cache for inspection.
        let concept_store = ConceptStore::new(&path_buf);
        let concept_engine = ConceptEngine::new();

        // Causal pattern discovery layer
        // Same pattern as concepts: derived state, always start empty.
        let causal_store = CausalStore::new(&path_buf);
        let causal_engine = CausalEngine::new();

        // Policy hint layer
        // Same pattern: derived state, always start empty.
        let policy_store = PolicyStore::new(&path_buf);
        let policy_engine = PolicyEngine::new();

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
            // SDK Wrapper defaults
            taxonomy: RwLock::new(TagTaxonomy::default()),
            trust_config: RwLock::new(TrustConfig::default()),
            recall_cache: RecallCache::default(),
            structured_recall_cache: StructuredRecallCache::default(),
            circuit_breaker: CircuitBreaker::default(),
            research_engine: ResearchEngine::new(),
            maintenance_config: RwLock::new(MaintenanceConfig::default()),
            background: RwLock::new(None),
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
            belief_rerank_mode: std::sync::atomic::AtomicU8::new(recall::BeliefRerankMode::Off as u8),
            path: path_buf,
        })
    }

    // ── Core Operations ──

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
        self.store_with_channel(content, level, tags, pin, content_type, source_type, metadata, deduplicate, caused_by_id, None, None, namespace, semantic_type)
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
        crate::record::Record::validate_source_type(source_type)
            .map_err(|e| anyhow::anyhow!(e))?;
        let deduplicate = deduplicate.unwrap_or(true);
        let semantic_type = semantic_type.unwrap_or(crate::record::DEFAULT_SEMANTIC_TYPE);
        crate::record::Record::validate_semantic_type(semantic_type)
            .map_err(|e| anyhow::anyhow!(e))?;

        // ── Namespace resolution & validation ──
        let ns = namespace.unwrap_or(crate::record::DEFAULT_NAMESPACE);
        crate::record::Record::validate_namespace(ns)
            .map_err(|e| anyhow::anyhow!(e))?;

        // ── Guard: Auto-protect tags (detect sensitive content) ──
        guards::auto_protect_tags(content, &mut tags);

        // ── Guard: Sensitive tag check ──
        let taxonomy = self.taxonomy.read();
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
                            self.recall_cache.clear();
        self.structured_recall_cache.clear();
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
            let trust_config = self.trust_config.read();
            trust::stamp_provenance(&mut rec.metadata, channel, &rec.tags, &taxonomy, &trust_config);
        }

        // ── Guard: Apply guard result metadata ──
        for (key, value) in &guard_result.extra_metadata {
            rec.metadata.entry(key.clone()).or_insert_with(|| value.clone());
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
                tag_idx.entry(tag.clone()).or_default().insert(rec.id.clone());
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

        // Invalidate recall cache on write
        self.recall_cache.clear();
        self.structured_recall_cache.clear();

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
        // Build cache key that includes namespaces (sorted for determinism)
        let default_ns = [crate::record::DEFAULT_NAMESPACE];
        let ns_list = namespaces.unwrap_or(&default_ns);
        let mut sorted_ns: Vec<&str> = ns_list.to_vec();
        sorted_ns.sort_unstable();
        let cache_key = format!("{}|ns={:?}", query, sorted_ns);

        // Check recall cache
        if let Some(cached) = self.recall_cache.get(&cache_key) {
            return Ok(cached);
        }

        let scored = self.recall_core(
            query,
            20,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            session_id,
            namespaces,
        )?;

        let records = self.records.read();
        let preamble = recall::format_preamble(
            &scored,
            token_budget.unwrap_or(2048),
            &records,
        );

        // Cache the result
        self.recall_cache.put(&cache_key, preamble.clone());

        Ok(preamble)
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

        // Check structured recall cache
        if let Some(cached) = self.structured_recall_cache.get(query, top, min_str, namespaces) {
            return Ok(cached);
        }

        let scored = self.recall_core(
            query,
            top,
            min_str,
            expand_connections.unwrap_or(true),
            session_id,
            namespaces,
        )?;

        // Cache the result
        self.structured_recall_cache.put(query, top, min_str, namespaces, scored.clone());

        Ok(scored)
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
        let trust_config = self.trust_config.read();

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
            &time_records,
            embedding_ranked,
            Some(&trust_config),
            namespaces,
        );

        drop(records);
        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        // Activate recalled records
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
        info.insert("last_activated".into(), format!("{:.3}", rec.last_activated));
        info.insert("age_days".into(), format!("{:.2}", (now - rec.created_at) / 86400.0));
        info.insert("days_since_activation".into(), format!("{:.2}", (now - rec.last_activated) / 86400.0));
        info.insert("namespace".into(), rec.namespace.clone());
        info.insert("source_type".into(), rec.source_type.clone());
        info.insert("tags".into(), rec.tags.join(", "));

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
                    if query_words
                        .iter()
                        .any(|w| content_lower.contains(w))
                    {
                        scored.push((0.8, rec.clone()));
                    }
                }
            }
        }

        // Re-sort by score desc, truncate
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
            stability: if rec.level == Level::Identity { 100.0 } else { 1.0 },
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
        let trust_config = self.trust_config.read();

        recall::recall_pipeline(
            query,
            top_k,
            min_strength,
            expand_connections,
            &self.sdr,
            &self.index,
            &self.storage,
            &ngram,
            &tag_idx,
            &aura_idx,
            &records,
            embedding_ranked,
            Some(&trust_config),
            namespaces,
        )
    }

    /// Post-recall side effects: activate records and log audit.
    fn recall_finalize(
        &self,
        scored: &[(f32, Record)],
        query: &str,
        session_id: Option<&str>,
    ) {
        {
            let mut records = self.records.write();
            let mut tracker = self.session_tracker.write();
            recall::activate_and_strengthen(scored, &mut records, &mut tracker, session_id);
        }
        if let Some(ref log) = self.audit_log {
            let _ = log.log_retrieve(query, scored.len());
        }
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
        let mut scored = self.recall_raw(query, top_k, min_strength, expand_connections, namespaces);

        // Phase 4: belief-aware reranking (only in Limited mode)
        let rerank_mode = recall::BeliefRerankMode::from_u8(
            self.belief_rerank_mode.load(std::sync::atomic::Ordering::Relaxed)
        );
        if rerank_mode == recall::BeliefRerankMode::Limited {
            let belief_eng = self.belief_engine.read();
            let _report = recall::apply_belief_rerank(&mut scored, &belief_eng, top_k);
        }

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
            query, top, min_str, expand_connections.unwrap_or(true), namespaces,
        );

        // Shadow scoring on raw baseline
        let belief_eng = self.belief_engine.read();
        let shadow_report = recall::compute_shadow_belief_scores(&scored, &belief_eng, top);

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
            query, top, min_str, expand_connections.unwrap_or(true), namespaces,
        );

        let belief_eng = self.belief_engine.read();
        let report = recall::apply_belief_rerank(&mut scored, &belief_eng, top);

        self.recall_finalize(&scored, query, session_id);
        Ok((scored, report))
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

        results.sort_by(|a, b| b.importance().partial_cmp(&a.importance()).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Get a single record by ID.
    pub fn get(&self, record_id: &str) -> Option<Record> {
        self.records.read().get(record_id).cloned()
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
            crate::record::Record::validate_source_type(st)
                .map_err(|e| anyhow::anyhow!(e))?;
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

        self.cognitive_store.append_update(rec)?;

        // Invalidate recall cache on write
        self.recall_cache.clear();
        self.structured_recall_cache.clear();

        Ok(Some(rec.clone()))
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

        // Invalidate recall cache on write
        self.recall_cache.clear();
        self.structured_recall_cache.clear();

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
                    "Cannot connect records across namespaces ('{}' vs '{}')", a, b
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

        // Semantic-aware promotion: decisions and preferences have lower thresholds
        let semantic_promotable: Vec<String> = records
            .values()
            .filter(|r| {
                let is_valuable_type = matches!(r.semantic_type.as_str(), "decision" | "preference");
                is_valuable_type
                    && r.activation_count >= 3  // Lower than standard 5
                    && r.strength >= 0.5        // Lower than standard 0.7
                    && r.level < Level::Identity
            })
            .map(|r| r.id.clone())
            .collect();

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

        // Archive dead records (contradictions get a reprieve — higher archival threshold)
        let dead: Vec<String> = records
            .values()
            .filter(|r| {
                if r.semantic_type == "contradiction" {
                    r.strength < 0.02 // Much harder to archive
                } else {
                    !r.is_alive() // Standard: strength < 0.05
                }
            })
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
            records.values().filter(|r| r.level == Level::Working).count(),
        );
        stats.insert(
            "decisions".into(),
            records.values().filter(|r| r.level == Level::Decisions).count(),
        );
        stats.insert(
            "domain".into(),
            records.values().filter(|r| r.level == Level::Domain).count(),
        );
        stats.insert(
            "identity".into(),
            records.values().filter(|r| r.level == Level::Identity).count(),
        );
        stats.insert(
            "total_connections".into(),
            records.values().map(|r| r.connections.len()).sum(),
        );
        stats.insert(
            "total_tags".into(),
            self.tag_index.read().len(),
        );

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

        let working = records.values().filter(|r| r.level == Level::Working).count();
        let decisions = records.values().filter(|r| r.level == Level::Decisions).count();
        let domain = records.values().filter(|r| r.level == Level::Domain).count();
        let identity = records.values().filter(|r| r.level == Level::Identity).count();

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
                r.level.is_cognitive()
                    && r.activation_count >= min_act
                    && r.strength >= min_str
            })
            .cloned()
            .collect();

        candidates.sort_by(|a, b| {
            b.activation_count
                .cmp(&a.activation_count)
                .then_with(|| {
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
                self.recall_cache.clear();
        self.structured_recall_cache.clear();
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
        let trust_config = self.trust_config.read();

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
        *self.taxonomy.write() = taxonomy;
    }

    /// Get current tag taxonomy.
    pub fn get_taxonomy(&self) -> TagTaxonomy {
        self.taxonomy.read().clone()
    }

    /// Set trust configuration.
    pub fn set_trust_config(&self, config: TrustConfig) {
        *self.trust_config.write() = config;
    }

    /// Get current trust configuration.
    pub fn get_trust_config(&self) -> TrustConfig {
        self.trust_config.read().clone()
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
        *self.maintenance_config.write() = config;
    }

    /// Get current maintenance configuration.
    pub fn get_maintenance_config(&self) -> MaintenanceConfig {
        self.maintenance_config.read().clone()
    }

    /// Run a single maintenance cycle across all maintenance phases.
    #[instrument(skip(self))]
    pub fn run_maintenance(&self) -> MaintenanceReport {
        use std::time::Instant;

        let cycle_start = Instant::now();
        let mut timings = background_brain::PhaseTimings::default();

        let config = self.maintenance_config.read().clone();
        let taxonomy = self.taxonomy.read().clone();

        let mut records = self.records.write();
        let total_records = records.len();

        // Get cycle count from background brain or use 0
        let cycle = {
            let bg = self.background.read();
            bg.as_ref().map_or(0, |b| b.cycles())
        };

        // Phase 0: Level fix (every Nth cycle)
        let t0 = Instant::now();
        if cycle % config.level_fix_interval == 0 {
            background_brain::fix_memory_levels(&mut records, &taxonomy);
        }
        timings.level_fix_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 1: Decay
        let t1 = Instant::now();
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

            // Decay connections
            for rec in records.values_mut() {
                let weak: Vec<String> = rec.connections.iter()
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
                let _ = self.cognitive_store.append_delete(id);
            }

            background_brain::DecayReport { decayed, archived }
        } else {
            background_brain::DecayReport::default()
        };
        timings.decay_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Guarded reflect
        let t2 = Instant::now();
        let reflect = if config.reflect_enabled {
            background_brain::guarded_reflect(&mut records, &taxonomy)
        } else {
            background_brain::ReflectReport::default()
        };
        timings.reflect_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // Phase 2.5: Epistemic update
        let t25 = Instant::now();
        let epistemic = background_brain::update_epistemic_state(&mut records);
        timings.epistemic_ms = t25.elapsed().as_secs_f64() * 1000.0;

        // Phase 3: Insights
        let t3 = Instant::now();
        let insights_found = if config.insights_enabled {
            let found = insights::detect_all(&records);
            found.len()
        } else {
            0
        };
        timings.insights_ms = t3.elapsed().as_secs_f64() * 1000.0;

        // Phase 3.5: Belief update (read-only — builds beliefs, does not affect recall)
        // Take a read-only snapshot of record refs to avoid holding write lock during
        // belief computation and disk persistence.
        let belief_snapshot: HashMap<String, Record> = records.iter()
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
        let t_sdr = Instant::now();
        let sdr_lookup: SdrLookup = belief_snapshot.iter()
            .map(|(rid, rec)| {
                let sdr_vec = self.sdr.text_to_sdr(&rec.content, false);
                (rid.clone(), sdr_vec)
            })
            .collect();
        timings.sdr_build_ms = t_sdr.elapsed().as_secs_f64() * 1000.0;

        let t35 = Instant::now();
        let belief_phase_report = {
            let mut engine = self.belief_engine.write();
            let br = engine.update_with_sdr(&belief_snapshot, &sdr_lookup);
            // Persist belief state (best-effort)
            let _ = self.belief_store.save(&engine);
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
        timings.belief_ms = t35.elapsed().as_secs_f64() * 1000.0;

        // Phase 3.7: Concept discovery (read-only — finds stable abstractions over beliefs)
        let t37 = Instant::now();
        let concept_phase_report = {
            let engine = self.belief_engine.read();
            let mut concept_eng = self.concept_engine.write();
            let cr = concept_eng.discover(&engine, &belief_snapshot, &sdr_lookup);
            // Persist concept state (best-effort)
            let _ = self.concept_store.save(&concept_eng);
            background_brain::ConceptPhaseReport {
                seeds_found: cr.seeds_found,
                candidates_found: cr.candidates_found,
                stable_count: cr.stable_count,
                rejected_count: cr.rejected_count,
                avg_abstraction_score: cr.avg_abstraction_score,
                centroids_built: cr.centroids_built,
                partitions_with_multiple_seeds: cr.partitions_with_multiple_seeds,
                pairwise_comparisons: cr.pairwise_comparisons,
                pairwise_above_threshold: cr.pairwise_above_threshold,
                tanimoto_min: cr.tanimoto_min,
                tanimoto_max: cr.tanimoto_max,
                tanimoto_avg: cr.tanimoto_avg,
                avg_centroid_size: cr.avg_centroid_size,
            }
        };
        timings.concept_ms = t37.elapsed().as_secs_f64() * 1000.0;

        // Phase 3.8: Causal pattern discovery (read-only — finds candidate causal relations)
        let t38 = Instant::now();
        let causal_phase_report = {
            let engine = self.belief_engine.read();
            let mut causal_eng = self.causal_engine.write();
            let cr = causal_eng.discover(&engine, &belief_snapshot, &sdr_lookup);
            // Persist causal state (best-effort)
            let _ = self.causal_store.save(&causal_eng);
            background_brain::CausalPhaseReport {
                edges_found: cr.edges_found,
                candidates_found: cr.candidates_found,
                stable_count: cr.stable_count,
                rejected_count: cr.rejected_count,
                avg_causal_strength: cr.avg_causal_strength,
            }
        };
        timings.causal_ms = t38.elapsed().as_secs_f64() * 1000.0;

        // Phase 3.9: Policy hint discovery (read-only — advisory hints from causal patterns)
        let t39 = Instant::now();
        let policy_phase_report = {
            let causal_eng = self.causal_engine.read();
            let concept_eng = self.concept_engine.read();
            let belief_eng = self.belief_engine.read();
            let mut policy_eng = self.policy_engine.write();
            let pr = policy_eng.discover(&causal_eng, &concept_eng, &belief_eng, &belief_snapshot);
            // Persist policy state (best-effort)
            let _ = self.policy_store.save(&policy_eng);
            background_brain::PolicyPhaseReport {
                seeds_found: pr.seeds_found,
                hints_found: pr.hints_found,
                stable_hints: pr.stable_hints,
                suppressed_hints: pr.suppressed_hints,
                rejected_hints: pr.rejected_hints,
                avg_policy_strength: pr.avg_policy_strength,
            }
        };
        timings.policy_ms = t39.elapsed().as_secs_f64() * 1000.0;

        // ── Compute cross-cycle identity stability ──
        let stability = {
            let belief_eng = self.belief_engine.read();
            let concept_eng = self.concept_engine.read();
            let causal_eng = self.causal_engine.read();
            let policy_eng = self.policy_engine.read();

            // Current semantic keys (stable identities, not random IDs).
            // Each engine has a key_index: semantic_key → entity_id.
            // Using semantic keys ensures churn measures real identity change,
            // not just ID regeneration on full rebuild.
            let cur_belief: HashSet<String> = belief_eng.key_index.keys().cloned().collect();
            let cur_concept: HashSet<String> = concept_eng.key_index.keys().cloned().collect();
            let cur_causal: HashSet<String> = causal_eng.key_index.keys().cloned().collect();
            let cur_policy: HashSet<String> = policy_eng.key_index.keys().cloned().collect();

            // Previous keys
            let prev_b = self.prev_belief_keys.read();
            let prev_c = self.prev_concept_keys.read();
            let prev_ca = self.prev_causal_keys.read();
            let prev_p = self.prev_policy_keys.read();

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

            // Update previous keys for next cycle
            *self.prev_belief_keys.write() = cur_belief;
            *self.prev_concept_keys.write() = cur_concept;
            *self.prev_causal_keys.write() = cur_causal;
            *self.prev_policy_keys.write() = cur_policy;

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
        };

        // Phase 4: Consolidation (fast pass only — no LLM)
        let t4 = Instant::now();
        let consolidation_report = if config.consolidation_enabled {
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

            background_brain::ConsolidationReport {
                native_merged: result.merged,
                clusters_found: 0,
                meta_created: 0,
            }
        } else {
            background_brain::ConsolidationReport::default()
        };
        timings.consolidation_ms = t4.elapsed().as_secs_f64() * 1000.0;

        // Re-acquire records for remaining phases
        let mut records = self.records.write();

        // Phase 5: Cross-connections
        let t5 = Instant::now();
        let cross_connections = if config.synthesis_enabled {
            let discoveries = background_brain::discover_cross_connections(&records, 3);
            let count = discoveries.len();
            if let Some(ref bg) = *self.background.read() {
                *bg.last_cross_connections.write() = discoveries;
            }
            count
        } else {
            0
        };
        timings.cross_connections_ms = t5.elapsed().as_secs_f64() * 1000.0;

        // Phase 6: Scheduled tasks + Phase 7: Archival
        let t67 = Instant::now();
        let task_reminders = background_brain::check_scheduled_tasks(
            &records,
            &config.task_tag,
        );

        let records_archived = if config.archival_enabled {
            background_brain::archive_old_records(&mut records, &config, &taxonomy)
        } else {
            0
        };
        timings.tasks_archival_ms = t67.elapsed().as_secs_f64() * 1000.0;

        // Persist changes
        drop(records);
        let _ = self.flush();

        // Invalidate cache after maintenance
        self.recall_cache.clear();
        self.structured_recall_cache.clear();

        timings.total_ms = cycle_start.elapsed().as_secs_f64() * 1000.0;

        MaintenanceReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            decay,
            reflect,
            epistemic,
            insights_found,
            belief: belief_phase_report,
            concept: concept_phase_report,
            causal: causal_phase_report,
            policy: policy_phase_report,
            consolidation: consolidation_report,
            cross_connections,
            task_reminders,
            records_archived,
            total_records,
            timings,
            stability,
        }
    }

    /// Start background maintenance loop (daemon thread).
    pub fn start_background(&self, interval_secs: Option<u64>) {
        let _interval = interval_secs.unwrap_or(120);
        let mut bg = self.background.write();
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
        let mut bg = self.background.write();
        if let Some(ref mut brain) = *bg {
            brain.stop();
        }
        *bg = None;
    }

    /// Check if background maintenance is running.
    pub fn is_background_running(&self) -> bool {
        let bg = self.background.read();
        bg.as_ref().is_some_and(|b| b.is_running())
    }

    // ── Inspection Helpers (observability) ──

    /// Return a snapshot of all current beliefs (cloned).
    /// Optional filter by state: "resolved", "unresolved", "singleton", "empty".
    pub fn get_beliefs(&self, state_filter: Option<&str>) -> Vec<crate::belief::Belief> {
        let engine = self.belief_engine.read();
        engine.beliefs.values()
            .filter(|b| {
                match state_filter {
                    Some("resolved") => b.state == crate::belief::BeliefState::Resolved,
                    Some("unresolved") => b.state == crate::belief::BeliefState::Unresolved,
                    Some("singleton") => b.state == crate::belief::BeliefState::Singleton,
                    Some("empty") => b.state == crate::belief::BeliefState::Empty,
                    _ => true,
                }
            })
            .cloned()
            .collect()
    }

    /// Return a snapshot of all current concepts (cloned).
    /// Optional filter by state: "stable", "candidate", "rejected".
    pub fn get_concepts(&self, state_filter: Option<&str>) -> Vec<crate::concept::ConceptCandidate> {
        let engine = self.concept_engine.read();
        engine.concepts.values()
            .filter(|c| {
                match state_filter {
                    Some("stable") => c.state == crate::concept::ConceptState::Stable,
                    Some("candidate") => c.state == crate::concept::ConceptState::Candidate,
                    Some("rejected") => c.state == crate::concept::ConceptState::Rejected,
                    _ => true,
                }
            })
            .cloned()
            .collect()
    }

    /// Return surfaced concepts for external inspection.
    /// Returns bounded, sorted, provenance-checked concepts suitable for public consumption.
    /// This is inspection-only — surfaced concepts do not affect recall, compression, or behavior.
    pub fn get_surfaced_concepts(&self, limit: Option<usize>) -> Vec<crate::concept::SurfacedConcept> {
        let engine = self.concept_engine.read();
        crate::concept::surface_concepts(&engine, limit)
    }

    /// Return surfaced concepts for a specific namespace.
    pub fn get_surfaced_concepts_for_namespace(
        &self,
        namespace: &str,
        limit: Option<usize>,
    ) -> Vec<crate::concept::SurfacedConcept> {
        let engine = self.concept_engine.read();
        crate::concept::surface_concepts_filtered(&engine, limit, Some(namespace))
    }

    /// Return a snapshot of all current causal patterns (cloned).
    /// Optional filter by state: "stable", "candidate", "rejected".
    pub fn get_causal_patterns(&self, state_filter: Option<&str>) -> Vec<crate::causal::CausalPattern> {
        let engine = self.causal_engine.read();
        engine.patterns.values()
            .filter(|p| {
                match state_filter {
                    Some("stable") => p.state == crate::causal::CausalState::Stable,
                    Some("candidate") => p.state == crate::causal::CausalState::Candidate,
                    Some("rejected") => p.state == crate::causal::CausalState::Rejected,
                    _ => true,
                }
            })
            .cloned()
            .collect()
    }

    /// Return a snapshot of all current policy hints (cloned).
    /// Optional filter by state: "stable", "candidate", "suppressed", "rejected".
    pub fn get_policy_hints(&self, state_filter: Option<&str>) -> Vec<crate::policy::PolicyHint> {
        let engine = self.policy_engine.read();
        engine.hints.values()
            .filter(|h| {
                match state_filter {
                    Some("stable") => h.state == crate::policy::PolicyState::Stable,
                    Some("candidate") => h.state == crate::policy::PolicyState::Candidate,
                    Some("suppressed") => h.state == crate::policy::PolicyState::Suppressed,
                    Some("rejected") => h.state == crate::policy::PolicyState::Rejected,
                    _ => true,
                }
            })
            .cloned()
            .collect()
    }

    // ── Surfaced Policy Output ──

    /// Return filtered, sorted, bounded advisory policy hints suitable for
    /// external consumption. Inspection-only — does not affect recall or behavior.
    ///
    /// Only surfaces Stable hints and strong Candidates (policy_strength >= 0.70,
    /// confidence >= 0.55) with complete provenance. Bounded to 10 global,
    /// 3 per domain.
    pub fn get_surfaced_policy_hints(&self, limit: Option<usize>) -> Vec<crate::policy::SurfacedPolicyHint> {
        let engine = self.policy_engine.read();
        crate::policy::surface_policy_hints(&engine, limit)
    }

    /// Return surfaced hints for a specific namespace.
    pub fn get_surfaced_policy_hints_for_namespace(
        &self,
        namespace: &str,
        limit: Option<usize>,
    ) -> Vec<crate::policy::SurfacedPolicyHint> {
        let engine = self.policy_engine.read();
        crate::policy::surface_policy_hints_filtered(&engine, limit, Some(namespace))
    }

    // ── Belief Reranking Config (Phase 4) ──

    /// Set belief reranking mode.
    ///
    /// - `Off`: no belief influence on ranking (default)
    /// - `Shadow`: compute shadow scores for logging, do not alter ranking
    /// - `Limited`: apply bounded reranking (±5% score cap, ±2 position cap)
    pub fn set_belief_rerank_mode(&self, mode: recall::BeliefRerankMode) {
        self.belief_rerank_mode.store(mode as u8, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current belief reranking mode.
    pub fn get_belief_rerank_mode(&self) -> recall::BeliefRerankMode {
        recall::BeliefRerankMode::from_u8(
            self.belief_rerank_mode.load(std::sync::atomic::Ordering::Relaxed)
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
        self.research_engine.add_finding(project_id, query, result, url)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Complete a research project and store as a record.
    pub fn complete_research(
        &self,
        project_id: &str,
        synthesis: Option<String>,
    ) -> Result<Record> {
        let project = self.research_engine.complete_research(project_id, synthesis)
            .map_err(|e| anyhow::anyhow!(e))?;

        // Build content from project
        let content = if let Some(ref syn) = project.synthesis {
            format!("Research: {}\n\n{}", project.topic, syn)
        } else {
            let findings: Vec<String> = project.findings.iter()
                .map(|f| format!("- {}: {}", f.query, f.result))
                .collect();
            format!("Research: {}\n\nFindings:\n{}", project.topic, findings.join("\n"))
        };

        // Store as a record
        let rec = self.store(
            &content,
            Some(Level::Domain),
            Some(vec!["research-report".into()]),
            None,
            None,
            Some("retrieved"),
            None,
            Some(false), // Don't dedup research
            None,
            None,
            None,
        )?;

        Ok(rec)
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
            return self.get(&existing_rec.id)
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
            return self.get(&existing_rec.id)
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
            r.metadata.get("persona_json")
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
        self.circuit_breaker.record_failure(tool_name);
    }

    /// Record a tool success.
    pub fn record_tool_success(&self, tool_name: &str) {
        self.circuit_breaker.record_success(tool_name);
    }

    /// Check if a tool is available (circuit closed).
    pub fn is_tool_available(&self, tool_name: &str) -> bool {
        self.circuit_breaker.is_available(tool_name)
    }

    /// Get health report for all tracked tools.
    pub fn tool_health(&self) -> HashMap<String, String> {
        self.circuit_breaker.health_report()
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
            let prev: u32 = rec.metadata.get(pos_key)
                .and_then(|v| v.parse().ok()).unwrap_or(0);
            rec.metadata.insert(pos_key.into(), (prev + 1).to_string());
            // Positive: boost strength (like an activation, but weaker)
            rec.strength = (rec.strength + 0.1).min(1.0);
        } else {
            let prev: u32 = rec.metadata.get(neg_key)
                .and_then(|v| v.parse().ok()).unwrap_or(0);
            rec.metadata.insert(neg_key.into(), (prev + 1).to_string());
            // Negative: weaken strength
            rec.strength = (rec.strength - 0.15).max(0.0);
        }

        // Persist change
        self.cognitive_store.append_update(rec)?;
        self.recall_cache.clear();
        self.structured_recall_cache.clear();

        Ok(true)
    }

    /// Get feedback stats for a record.
    ///
    /// Returns (positive_count, negative_count, net_score).
    pub fn feedback_stats(&self, record_id: &str) -> Option<(u32, u32, i32)> {
        let records = self.records.read();
        let rec = records.get(record_id)?;

        let pos: u32 = rec.metadata.get("feedback_positive")
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        let neg: u32 = rec.metadata.get("feedback_negative")
            .and_then(|v| v.parse().ok()).unwrap_or(0);

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
            let old_rec = records.get_mut(old_id)
                .ok_or_else(|| anyhow::anyhow!("Record '{}' not found", old_id))?;

            // Mark as superseded
            old_rec.metadata.insert("superseded_by".into(), "pending".into());
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
                old_rec.metadata.insert("superseded_by".into(), new_rec.id.clone());
                let _ = self.cognitive_store.append_update(old_rec);
            }
        }

        self.recall_cache.clear();
        self.structured_recall_cache.clear();
        Ok(new_rec)
    }

    /// Check if a record has been superseded.
    ///
    /// Returns `Some(new_record_id)` if superseded, `None` otherwise.
    pub fn superseded_by(&self, record_id: &str) -> Option<String> {
        let records = self.records.read();
        records.get(record_id)
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
        if !label.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            return Err(anyhow::anyhow!("Label must contain only alphanumeric, dash, or underscore characters"));
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
                tag_idx.entry(tag.clone()).or_default().insert(rec.id.clone());
            }
            if let Some(ref aid) = rec.aura_id {
                aura_idx.insert(aid.clone(), rec.id.clone());
            }
            records.insert(rec.id.clone(), rec);
        }

        self.recall_cache.clear();
        self.structured_recall_cache.clear();
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
        let modified: Vec<String> = keys_a.intersection(&keys_b)
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
        let dir = self.path.parent().unwrap_or(Path::new("."));
        let prefix = format!("{}_snapshot_",
            self.path.file_stem()
                .unwrap_or_default()
                .to_string_lossy());

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
        let stem = self.path.file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let dir = self.path.parent().unwrap_or(Path::new("."));
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
        let results = self.recall_structured(
            query,
            top_k,
            None,
            None,
            None,
            namespaces,
        )?;

        // Build portable fragment with provenance
        let fragment: Vec<serde_json::Value> = results.iter().map(|(score, rec)| {
            let mut meta = rec.metadata.clone();
            meta.insert("shared_score".into(), format!("{:.4}", score));
            meta.insert("shared_from".into(),
                self.path.to_string_lossy().to_string());

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
        }).collect();

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

        let format = envelope.get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if format != "aura_context" {
            return Err(anyhow::anyhow!("Unknown format: '{}'. Expected 'aura_context'", format));
        }

        let records_val = envelope.get("records")
            .ok_or_else(|| anyhow::anyhow!("Missing 'records' field"))?;
        let records_arr = records_val.as_array()
            .ok_or_else(|| anyhow::anyhow!("'records' must be an array"))?;

        let mut imported = 0;
        for item in records_arr {
            let content = item.get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("Record missing 'content'"))?;

            let level_str = item.get("level")
                .and_then(|v| v.as_str())
                .unwrap_or("Working");
            let level = match level_str.to_uppercase().as_str() {
                "WORKING" => Level::Working,
                "DECISIONS" => Level::Decisions,
                "DOMAIN" => Level::Domain,
                "IDENTITY" => Level::Identity,
                _ => Level::Working,
            };

            let mut tags: Vec<String> = item.get("tags")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            if !tags.contains(&"shared".to_string()) {
                tags.push("shared".into());
            }

            let mut metadata: HashMap<String, String> = item.get("metadata")
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

        self.recall_cache.clear();
        self.structured_recall_cache.clear();
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
                tag_idx.entry(tag.clone()).or_default().insert(rec.id.clone());
            }
            self.cognitive_store.append_store(&rec)?;
            records.insert(rec.id.clone(), rec);
        }

        // Invalidate recall cache
        self.recall_cache.clear();
        self.structured_recall_cache.clear();

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
        Ok(format!("Stored record {} (level={})", result.id, result.level))
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
    pub fn list_memories(&self, offset: usize, limit: usize, filter_dna: Option<&str>) -> (Vec<StoredRecord>, usize) {
        let cache = self.storage.header_cache.read();

        let mut entries: Vec<_> = cache.values()
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
        Ok(self.storage.get_prediction(current_id).map(|next| StoredRecord {
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
            let similarity = self.sdr.tanimoto_sparse(&predicted.sdr_indices, &actual_sdr);
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
        let mut ns_set: std::collections::HashSet<String> = records.values().map(|r| r.namespace.clone()).collect();
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
        let cross_ns_ids: Vec<String> = outgoing_keys.into_iter()
            .filter(|cid| {
                records.get(cid.as_str())
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
        let peers_to_clean: Vec<String> = records.iter()
            .filter(|(id, r)| *id != record_id && r.namespace == old_namespace && r.connections.contains_key(record_id))
            .map(|(id, _)| id.clone())
            .collect();
        for pid in &peers_to_clean {
            if let Some(peer) = records.get_mut(pid.as_str()) {
                peer.connections.remove(record_id);
            }
        }

        let _ = self.cognitive_store.append_update(records.get(record_id).unwrap());
        self.recall_cache.clear();
        self.structured_recall_cache.clear();
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
fn extract_namespaces(ns: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>) -> PyResult<Option<Vec<String>>> {
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
                "namespace must be a str, list[str], or None"
            ))
        }
    }
}

impl Drop for Aura {
    fn drop(&mut self) {
        let _ = self.close();
    }
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
        let rec = py.allow_threads(|| {
            self.store_with_channel(content, level, tags, pin, content_type, source_type, metadata, deduplicate, caused_by_id, channel, auto_promote, namespace, semantic_type)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
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
        let ns_refs: Option<Vec<&str>> = ns_vec.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        py.allow_threads(|| {
            self.recall(query, token_budget, min_strength, expand_connections, session_id, ns_slice)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
        let ns_refs: Option<Vec<&str>> = ns_vec.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        let results = py.allow_threads(|| {
            self.recall_structured(query, top_k, min_strength, expand_connections, session_id, ns_slice)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

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
            // Include trust metadata
            if let Some(trust) = rec.metadata.get("trust_score") {
                dict.set_item("trust", trust)?;
            }
            if let Some(source) = rec.metadata.get("source") {
                dict.set_item("source", source)?;
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
        let ns_refs: Option<Vec<&str>> = ns_vec.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        let results = py.allow_threads(|| {
            self.recall_at(query, timestamp, top_k, min_strength, expand_connections, session_id, ns_slice)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

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
            py_results.push(dict.unbind().into_any());
        }
        Ok(py_results)
    }

    /// Return access/strength timeline snapshot for a single record.
    #[pyo3(name = "history")]
    fn py_history(&self, py: Python<'_>, record_id: &str) -> PyResult<PyObject> {
        let info = py.allow_threads(|| {
            self.history(record_id)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

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
        let ns_refs: Option<Vec<&str>> = ns_vec.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        let results = py.allow_threads(|| {
            self.recall_full(query, top_k, include_failures, min_strength, expand_connections, session_id, ns_slice)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

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
        let ns_refs: Option<Vec<&str>> = ns_vec.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        Ok(py.allow_threads(|| self.search(query, level, tags, limit, content_type, source_type, ns_slice, semantic_type)))
    }

    #[pyo3(name = "get")]
    fn py_get(&self, record_id: &str) -> Option<Record> {
        self.get(record_id)
    }

    #[pyo3(name = "delete")]
    fn py_delete(&self, record_id: &str) -> PyResult<bool> {
        self.delete(record_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
        self.update(record_id, content, level, tags, strength, metadata, source_type)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "connect", signature = (id_a, id_b, weight=None, relationship=None))]
    fn py_connect(&self, id_a: &str, id_b: &str, weight: Option<f32>, relationship: Option<&str>) -> PyResult<()> {
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
            Some(p) => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid phase {}. Must be 0, 1, or 2.", p),
            )),
        };

        let results: Vec<PyObject> = raw
            .into_iter()
            .map(|insight| {
                let dict = pyo3::types::PyDict::new_bound(py);
                let _ = dict.set_item("insight_type", &insight.insight_type);
                let _ = dict.set_item("severity", format!("{:?}", insight.severity));
                let _ = dict.set_item("phase", match insight.phase {
                    crate::insights::Phase::RecordHealth => "record_health",
                    crate::insights::Phase::Relationships => "relationships",
                    crate::insights::Phase::CrossDomain => "cross_domain",
                });
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

    #[pyo3(name = "get_surfaced_concepts", signature = (limit=None))]
    fn py_get_surfaced_concepts(&self, limit: Option<usize>) -> Vec<crate::concept::SurfacedConcept> {
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

    #[pyo3(name = "get_surfaced_policy_hints", signature = (limit=None))]
    fn py_get_surfaced_policy_hints(&self, limit: Option<usize>) -> Vec<crate::policy::SurfacedPolicyHint> {
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
    fn py_start_research(&self, py: Python<'_>, topic: &str, depth: Option<&str>) -> PyResult<PyObject> {
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
    fn py_complete_research(&self, project_id: &str, synthesis: Option<String>) -> PyResult<String> {
        let rec = self.complete_research(project_id, synthesis)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "store_user_profile")]
    fn py_store_user_profile(&self, fields: HashMap<String, String>) -> PyResult<String> {
        let rec = self.store_user_profile(fields)
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
        let rec = self.set_persona(persona)
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
            Some(ns) => { ns_vec = vec![ns]; Some(ns_vec.as_slice()) }
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
    fn py_recall_cognitive(&self, py: Python<'_>, query: Option<&str>, limit: Option<usize>, namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>) -> PyResult<Vec<Record>> {
        let ns_vec = extract_namespaces(namespace)?;
        let ns_refs: Option<Vec<&str>> = ns_vec.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
        let ns_slice: Option<&[&str]> = ns_refs.as_deref();
        Ok(py.allow_threads(|| self.recall_cognitive(query, limit, ns_slice)))
    }

    #[pyo3(name = "recall_core_tier", signature = (query=None, limit=None, namespace=None))]
    fn py_recall_core_tier(&self, py: Python<'_>, query: Option<&str>, limit: Option<usize>, namespace: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>) -> PyResult<Vec<Record>> {
        let ns_vec = extract_namespaces(namespace)?;
        let ns_refs: Option<Vec<&str>> = ns_vec.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
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
            self.path.display(),
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
            None, None, None, None, None, None, None, None,
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

        aura.store("The quick brown fox jumps over the lazy dog", None, None, None, None, None, None, None, None, None, None)?;
        aura.store("The quick brown fox jumps over the lazy dog", None, None, None, None, None, None, None, None, None, None)?;

        // Should be deduplicated to 1 record
        assert_eq!(aura.count(None), 1);

        Ok(())
    }

    #[test]
    fn test_stats() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("test1", Some(Level::Working), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("test2", Some(Level::Identity), None, None, None, None, None, Some(false), None, None, None)?;

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
            None, None, None, None, Some(false), None, None, None,
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
            None, None, None, None, Some(false), None,
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

        aura.store("Teo loves Rust programming", Some(Level::Identity), None, None, None, None, None, Some(false), None, None, None)?;

        // First recall — cache miss
        let r1 = aura.recall("Who is Teo?", None, None, None, None, None)?;
        // Second recall — cache hit (same result)
        let r2 = aura.recall("Who is Teo?", None, None, None, None, None)?;
        assert_eq!(r1, r2);

        // Store invalidates cache
        aura.store("Teo is 25 years old", None, None, None, None, None, None, Some(false), None, None, None)?;

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
            None, None, None, None, Some(false), None, None, None,
        )?;

        // Store a failure record
        aura.store(
            "Failed to register on site X: captcha required",
            Some(Level::Decisions),
            Some(vec!["outcome-failure".into()]),
            None, None, None, None, Some(false), None, None, None,
        )?;

        // recall_full should find records matching "Teo Rust"
        let results = aura.recall_full("Teo Rust", None, Some(true), None, None, None, None)?;
        assert!(!results.is_empty(), "recall_full should find at least one record");

        // recall_full with include_failures=true should find failure records
        let results_fail = aura.recall_full("register site captcha", None, Some(true), None, None, None, None)?;
        let has_failure = results_fail.iter().any(|(_, r)| r.tags.contains(&"outcome-failure".to_string()));
        assert!(has_failure, "recall_full with include_failures=true should find failure records");

        // recall_full with include_failures=false should still work
        let results_no_fail = aura.recall_full("register site", None, Some(false), None, None, None, None)?;
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

        aura.store("test record", None, None, None, None, None, None, Some(false), None, None, None)?;

        let report = aura.run_maintenance();
        assert!(report.total_records > 0);
        assert!(!report.timestamp.is_empty());

        Ok(())
    }

    #[test]
    fn test_run_maintenance_updates_epistemic_signals() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let id1 = aura.store(
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
        )?.id;
        let _id2 = aura.store(
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
        )?.id;
        let _id3 = aura.store(
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
        )?.id;

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

        let rec = aura.complete_research(&project.id, Some("GRPO is a group-relative optimization...".into()))?;
        assert!(rec.content.contains("GRPO"));
        assert!(rec.tags.contains(&"research-report".to_string()));

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

        aura.store("session note about testing", Some(Level::Working), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("recent decision on architecture", Some(Level::Decisions), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("domain fact about Rust language", Some(Level::Domain), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("core identity preferences", Some(Level::Identity), None, None, None, None, None, Some(false), None, None, None)?;

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

        aura.store("session note about testing", Some(Level::Working), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("recent decision on architecture", Some(Level::Decisions), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("domain fact about Rust language", Some(Level::Domain), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("core identity preferences", Some(Level::Identity), None, None, None, None, None, Some(false), None, None, None)?;

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

        aura.store("w1", Some(Level::Working), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("w2", Some(Level::Working), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("d1", Some(Level::Decisions), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("dom1", Some(Level::Domain), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("id1", Some(Level::Identity), None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("id2", Some(Level::Identity), None, None, None, None, None, Some(false), None, None, None)?;

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
        let rec = aura.store("frequently recalled fact", Some(Level::Working), None, None, None, None, None, Some(false), None, None, None)?;

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

        let rec = aura.store("promotable", Some(Level::Working), None, None, None, None, None, Some(false), None, None, None)?;
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

        aura.store("User is 25 years old", Some(Level::Identity), Some(vec!["user".into()]), None, None, None, None, Some(false), None, None, None)?;
        aura.store_with_channel("Test case: user is 30 years old", Some(Level::Identity), Some(vec!["user".into()]), None, None, None, None, Some(false), None, None, None, Some("test-data"), None)?;

        let results = aura.recall_structured("user age", None, None, None, None, Some(&["default"]))?;
        assert!(results.iter().all(|(_, r)| r.namespace == "default"));
        assert!(results.iter().any(|(_, r)| r.content.contains("25")));
        assert!(!results.iter().any(|(_, r)| r.content.contains("30")));

        let results = aura.recall_structured("user age", None, None, None, None, Some(&["test-data"]))?;
        assert!(results.iter().all(|(_, r)| r.namespace == "test-data"));

        Ok(())
    }

    #[test]
    fn test_namespace_isolation_search() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("Real data about cats", None, Some(vec!["animal".into()]), None, None, None, None, Some(false), None, None, None)?;
        aura.store_with_channel("Test data about cats", None, Some(vec!["animal".into()]), None, None, None, None, Some(false), None, None, None, Some("sandbox"), None)?;

        let results = aura.search(None, None, Some(vec!["animal".into()]), None, None, None, Some(&["default"]), None);
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Real"));

        let results = aura.search(None, None, Some(vec!["animal".into()]), None, None, None, Some(&["sandbox"]), None);
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Test"));

        Ok(())
    }

    #[test]
    fn test_list_namespaces() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("A", None, None, None, None, None, None, Some(false), None, None, None)?;
        aura.store_with_channel("B", None, None, None, None, None, None, Some(false), None, None, None, Some("project-x"), None)?;

        let ns = aura.list_namespaces();
        assert!(ns.contains(&"default".to_string()));
        assert!(ns.contains(&"project-x".to_string()));

        Ok(())
    }

    #[test]
    fn test_move_record() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let rec = aura.store("Moveable record content here", None, None, None, None, None, None, Some(false), None, None, None)?;
        assert_eq!(rec.namespace, "default");

        let moved = aura.move_record(&rec.id, "archive").unwrap();
        assert_eq!(moved.namespace, "archive");

        let results = aura.search(Some("Moveable"), None, None, None, None, None, Some(&["default"]), None);
        assert!(results.is_empty());

        let results = aura.search(Some("Moveable"), None, None, None, None, None, Some(&["archive"]), None);
        assert_eq!(results.len(), 1);

        Ok(())
    }

    #[test]
    fn test_namespace_stats() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("A", None, None, None, None, None, None, Some(false), None, None, None)?;
        aura.store("B", None, None, None, None, None, None, Some(false), None, None, None)?;
        aura.store_with_channel("C", None, None, None, None, None, None, Some(false), None, None, None, Some("sandbox"), None)?;

        let stats = aura.namespace_stats();
        assert_eq!(*stats.get("default").unwrap_or(&0), 2);
        assert_eq!(*stats.get("sandbox").unwrap_or(&0), 1);

        Ok(())
    }

    #[test]
    fn test_dedup_within_namespace_only() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("The quick brown fox jumps over the lazy dog repeatedly", None, None, None, None, None, None, None, None, None, None)?;
        aura.store_with_channel("The quick brown fox jumps over the lazy dog repeatedly", None, None, None, None, None, None, None, None, None, None, Some("sandbox"), None)?;

        let stats = aura.namespace_stats();
        assert_eq!(*stats.get("default").unwrap_or(&0), 1);
        assert_eq!(*stats.get("sandbox").unwrap_or(&0), 1);

        Ok(())
    }

    #[test]
    fn test_default_namespace_when_none() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        let rec = aura.store("No namespace specified here", None, None, None, None, None, None, Some(false), None, None, None)?;
        assert_eq!(rec.namespace, "default");

        // Search without namespace (None defaults to "default")
        let results = aura.search(Some("No namespace"), None, None, None, None, None, None, None);
        assert_eq!(results.len(), 1);

        Ok(())
    }

    // ── Multi-namespace tests (v1.2.0) ──

    #[test]
    fn test_multi_namespace_recall() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("User health data about blood pressure monitoring", None, Some(vec!["health".into()]), None, None, None, None, Some(false), None, None, None)?;
        aura.store_with_channel("Test case health scenario about blood pressure", None, Some(vec!["health".into()]), None, None, None, None, Some(false), None, None, None, Some("sandbox"), None)?;
        aura.store_with_channel("Project health metrics dashboard", None, Some(vec!["health".into()]), None, None, None, None, Some(false), None, None, None, Some("project-x"), None)?;

        // Single namespace — only default
        let results = aura.recall_structured("health blood pressure", None, None, None, None, Some(&["default"]))?;
        assert!(results.iter().all(|(_, r)| r.namespace == "default"));

        // Multi-namespace — default + sandbox
        let results = aura.recall_structured("health blood pressure", None, None, None, None, Some(&["default", "sandbox"]))?;
        let found_ns: std::collections::HashSet<String> = results.iter().map(|(_, r)| r.namespace.clone()).collect();
        assert!(found_ns.contains("default") || found_ns.contains("sandbox"));
        // project-x should NOT be in results
        assert!(!results.iter().any(|(_, r)| r.namespace == "project-x"));

        Ok(())
    }

    #[test]
    fn test_multi_namespace_search() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("Record A in default", None, Some(vec!["multi".into()]), None, None, None, None, Some(false), None, None, None)?;
        aura.store_with_channel("Record B in sandbox", None, Some(vec!["multi".into()]), None, None, None, None, Some(false), None, None, None, Some("sandbox"), None)?;
        aura.store_with_channel("Record C in project", None, Some(vec!["multi".into()]), None, None, None, None, Some(false), None, None, None, Some("project-x"), None)?;

        // Search across 2 namespaces
        let results = aura.search(None, None, Some(vec!["multi".into()]), None, None, None, Some(&["default", "sandbox"]), None);
        assert_eq!(results.len(), 2);
        let found_ns: std::collections::HashSet<String> = results.iter().map(|r| r.namespace.clone()).collect();
        assert!(found_ns.contains("default"));
        assert!(found_ns.contains("sandbox"));
        assert!(!found_ns.contains("project-x"));

        // Search across all 3
        let results = aura.search(None, None, Some(vec!["multi".into()]), None, None, None, Some(&["default", "sandbox", "project-x"]), None);
        assert_eq!(results.len(), 3);

        Ok(())
    }

    #[test]
    fn test_single_namespace_slice_backward_compat() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("Only in default ns content here", None, None, None, None, None, None, Some(false), None, None, None)?;

        // Single-element slice should behave same as old Option<&str>
        let results = aura.search(Some("Only in default"), None, None, None, None, None, Some(&["default"]), None);
        assert_eq!(results.len(), 1);

        // None should also work (defaults to ["default"])
        let results = aura.search(Some("Only in default"), None, None, None, None, None, None, None);
        assert_eq!(results.len(), 1);

        Ok(())
    }
}
