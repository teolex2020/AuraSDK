//! Aura — Unified Cognitive Memory orchestrator.
//!
//! This is the SINGLE entry point. Replaces both `AuraMemory` (Rust)
//! and `CognitiveMemory` (Python).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use anyhow::Result;

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
use crate::cache::RecallCache;
use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use crate::research::{ResearchEngine, ResearchProject};
use crate::identity::{self, AgentPersona};
use crate::background_brain::{self, BackgroundBrain, MaintenanceConfig, MaintenanceReport};

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
    circuit_breaker: CircuitBreaker,
    research_engine: ResearchEngine,
    maintenance_config: RwLock<MaintenanceConfig>,
    background: RwLock<Option<BackgroundBrain>>,

    // ── Optional Embedding Support ──
    embedding_store: EmbeddingStore,
    #[cfg(feature = "python")]
    embedding_fn: RwLock<Option<PyObject>>,

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
        let loaded_records = cognitive_store.load_all()?;

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
            circuit_breaker: CircuitBreaker::default(),
            research_engine: ResearchEngine::new(),
            maintenance_config: RwLock::new(MaintenanceConfig::default()),
            background: RwLock::new(None),
            // Optional embedding support
            embedding_store: EmbeddingStore::new(),
            #[cfg(feature = "python")]
            embedding_fn: RwLock::new(None),
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
        metadata: Option<HashMap<String, String>>,
        deduplicate: Option<bool>,
        caused_by_id: Option<&str>,
    ) -> Result<Record> {
        self.store_with_channel(content, level, tags, pin, content_type, metadata, deduplicate, caused_by_id, None, None)
    }

    /// Store with explicit channel for provenance stamping.
    /// `auto_promote`: if Some(false), disables surprise-based level promotion.
    pub fn store_with_channel(
        &self,
        content: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        pin: Option<bool>,
        content_type: Option<&str>,
        metadata: Option<HashMap<String, String>>,
        deduplicate: Option<bool>,
        caused_by_id: Option<&str>,
        channel: Option<&str>,
        auto_promote: Option<bool>,
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
        let deduplicate = deduplicate.unwrap_or(true);

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
                    // Strong match — activate existing record instead
                    let mut records = self.records.write();
                    if let Some(existing) = records.get_mut(existing_id) {
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
                        return Ok(existing.clone());
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
        if let Some(meta) = metadata {
            rec.metadata = meta;
        }
        if let Some(parent_id) = caused_by_id {
            rec.caused_by_id = Some(parent_id.to_string());
        }

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
    ) -> Result<String> {
        // Check recall cache
        if let Some(cached) = self.recall_cache.get(query) {
            return Ok(cached);
        }

        let scored = self.recall_core(
            query,
            20,
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            session_id,
        )?;

        let records = self.records.read();
        let preamble = recall::format_preamble(
            &scored,
            token_budget.unwrap_or(2048),
            &records,
        );

        // Cache the result
        self.recall_cache.put(query, preamble.clone());

        Ok(preamble)
    }

    /// Recall structured (raw results with trust scoring).
    pub fn recall_structured(
        &self,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> Result<Vec<(f32, Record)>> {
        let scored = self.recall_core(
            query,
            top_k.unwrap_or(20),
            min_strength.unwrap_or(0.1),
            expand_connections.unwrap_or(true),
            session_id,
        )?;
        // Trust-aware recency scoring is now applied inside recall_pipeline

        Ok(scored)
    }

    /// Core recall pipeline.
    fn recall_core(
        &self,
        query: &str,
        top_k: usize,
        min_strength: f32,
        expand_connections: bool,
        session_id: Option<&str>,
    ) -> Result<Vec<(f32, Record)>> {
        let records = self.records.read();
        let ngram = self.ngram_index.read();
        let tag_idx = self.tag_index.read();
        let aura_idx = self.aura_index.read();

        // Optional 4th signal: embedding similarity
        let embedding_ranked = self.collect_embedding_signal(query, top_k);
        let trust_config = self.trust_config.read();

        let scored = recall::recall_pipeline(
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
        );

        // Drop read locks before taking write locks
        drop(records);
        drop(ngram);
        drop(tag_idx);
        drop(aura_idx);
        drop(trust_config);

        // Activate and strengthen
        {
            let mut records = self.records.write();
            let mut tracker = self.session_tracker.write();
            recall::activate_and_strengthen(&scored, &mut records, &mut tracker, session_id);
        }

        // Audit
        if let Some(ref log) = self.audit_log {
            let _ = log.log_retrieve(query, scored.len());
        }

        Ok(scored)
    }

    /// Search with filters.
    pub fn search(
        &self,
        query: Option<&str>,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        limit: Option<usize>,
        content_type: Option<&str>,
    ) -> Vec<Record> {
        let records = self.records.read();
        let limit = limit.unwrap_or(20);

        let mut results: Vec<Record> = records
            .values()
            .filter(|r| {
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
    ) -> Result<Option<Record>> {
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

        self.cognitive_store.append_update(rec)?;

        // Invalidate recall cache on write
        self.recall_cache.clear();

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

        if let Some(a) = records.get_mut(id_a) {
            if let Some(rel) = relationship {
                a.add_typed_connection(id_b, weight, rel);
            } else {
                a.add_connection(id_b, weight);
            }
        } else {
            return Err(anyhow::anyhow!("Record {} not found", id_a));
        }

        if let Some(b) = records.get_mut(id_b) {
            if let Some(rel) = relationship {
                b.add_typed_connection(id_a, weight, rel);
            } else {
                b.add_connection(id_a, weight);
            }
        } else {
            return Err(anyhow::anyhow!("Record {} not found", id_b));
        }

        Ok(())
    }

    // ── Maintenance Operations ──

    /// Apply decay to all records.
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

        // Archive dead records
        let dead: Vec<String> = records
            .values()
            .filter(|r| !r.is_alive())
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

    /// Run a single maintenance cycle (all 8 phases).
    pub fn run_maintenance(&self) -> MaintenanceReport {
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
        if cycle % config.level_fix_interval == 0 {
            background_brain::fix_memory_levels(&mut records, &taxonomy);
        }

        // Phase 1: Decay
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

        // Phase 2: Guarded reflect
        let reflect = if config.reflect_enabled {
            background_brain::guarded_reflect(&mut records, &taxonomy)
        } else {
            background_brain::ReflectReport::default()
        };

        // Phase 3: Insights
        let insights_found = if config.insights_enabled {
            let found = insights::detect_all(&records);
            found.len()
        } else {
            0
        };

        // Phase 4: Consolidation (fast pass only — no LLM)
        let consolidation_report = if config.consolidation_enabled {
            drop(records);
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
            // Need to drop records lock if not used
            drop(records);
            background_brain::ConsolidationReport::default()
        };

        // Re-acquire records for remaining phases
        let mut records = self.records.write();

        // Phase 5: Cross-connections
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

        // Phase 6: Scheduled tasks
        let task_reminders = background_brain::check_scheduled_tasks(
            &records,
            &config.task_tag,
        );

        // Phase 7: Archival
        let records_archived = if config.archival_enabled {
            background_brain::archive_old_records(&mut records, &config, &taxonomy)
        } else {
            0
        };

        // Persist changes
        drop(records);
        let _ = self.flush();

        // Invalidate cache after maintenance
        self.recall_cache.clear();

        MaintenanceReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            decay,
            reflect,
            insights_found,
            consolidation: consolidation_report,
            cross_connections,
            task_reminders,
            records_archived,
            total_records,
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
            None,
            Some(false), // Don't dedup research
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
            Some(fields),
            Some(false), // Don't dedup
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
        );

        if let Some(existing_rec) = existing.first() {
            self.update(
                &existing_rec.id,
                Some(&content),
                None,
                None,
                None,
                Some(metadata),
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
            Some(metadata),
            Some(false),
            None,
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
        );
        results.first().and_then(|r| {
            r.metadata.get("persona_json")
                .and_then(|json| serde_json::from_str(json).ok())
        })
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
        self.storage.flush()?;
        let _ = self.index.save();
        Ok(())
    }

    /// Flush pending writes.
    pub fn flush(&self) -> Result<()> {
        self.cognitive_store.flush()?;
        self.storage.flush()?;
        Ok(())
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
        )?;
        Ok(format!("Stored record {} (level={})", result.id, result.level))
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
}

// ── PyO3 Bindings ──

#[cfg(feature = "python")]
#[pymethods]
impl Aura {
    #[new]
    #[pyo3(signature = (path, password=None))]
    fn py_new(path: &str, password: Option<&str>) -> PyResult<Self> {
        Self::open_with_password(path, password)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[pyo3(name = "store", signature = (content, level=None, tags=None, pin=None, content_type=None, metadata=None, deduplicate=None, caused_by_id=None, channel=None, auto_promote=None))]
    fn py_store(
        &self,
        content: &str,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        pin: Option<bool>,
        content_type: Option<&str>,
        metadata: Option<HashMap<String, String>>,
        deduplicate: Option<bool>,
        caused_by_id: Option<&str>,
        channel: Option<&str>,
        auto_promote: Option<bool>,
    ) -> PyResult<String> {
        let rec = self.store_with_channel(content, level, tags, pin, content_type, metadata, deduplicate, caused_by_id, channel, auto_promote)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(rec.id.clone())
    }

    #[pyo3(name = "recall", signature = (query, token_budget=None, min_strength=None, expand_connections=None, session_id=None))]
    fn py_recall(
        &self,
        query: &str,
        token_budget: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> PyResult<String> {
        self.recall(query, token_budget, min_strength, expand_connections, session_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "recall_structured", signature = (query, top_k=None, min_strength=None, expand_connections=None, session_id=None))]
    fn py_recall_structured(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: Option<usize>,
        min_strength: Option<f32>,
        expand_connections: Option<bool>,
        session_id: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let results = self
            .recall_structured(query, top_k, min_strength, expand_connections, session_id)
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

    #[pyo3(name = "search", signature = (query=None, level=None, tags=None, limit=None, content_type=None))]
    fn py_search(
        &self,
        query: Option<&str>,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        limit: Option<usize>,
        content_type: Option<&str>,
    ) -> Vec<Record> {
        self.search(query, level, tags, limit, content_type)
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

    #[pyo3(name = "update", signature = (record_id, content=None, level=None, tags=None, strength=None, metadata=None))]
    fn py_update(
        &self,
        record_id: &str,
        content: Option<&str>,
        level: Option<Level>,
        tags: Option<Vec<String>>,
        strength: Option<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<Option<Record>> {
        self.update(record_id, content, level, tags, strength, metadata)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "connect", signature = (id_a, id_b, weight=None, relationship=None))]
    fn py_connect(&self, id_a: &str, id_b: &str, weight: Option<f32>, relationship: Option<&str>) -> PyResult<()> {
        self.connect(id_a, id_b, weight, relationship)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "decay")]
    fn py_decay(&self) -> PyResult<(usize, usize)> {
        self.decay()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "consolidate")]
    fn py_consolidate(&self) -> PyResult<HashMap<String, usize>> {
        self.consolidate()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "reflect")]
    fn py_reflect(&self) -> PyResult<HashMap<String, usize>> {
        self.reflect()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "end_session")]
    fn py_end_session(&self, session_id: &str) -> PyResult<HashMap<String, usize>> {
        self.end_session(session_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
    fn py_run_maintenance(&self) -> MaintenanceReport {
        self.run_maintenance()
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
            None, None, None, None, None,
        )?;

        assert_eq!(rec.content, "Teo loves Rust");
        assert_eq!(rec.level, Level::Identity);

        let preamble = aura.recall("Who is Teo?", None, None, None, None)?;
        assert!(preamble.contains("Teo loves Rust"));

        aura.close()?;
        Ok(())
    }

    #[test]
    fn test_deduplication() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("The quick brown fox jumps over the lazy dog", None, None, None, None, None, None, None)?;
        aura.store("The quick brown fox jumps over the lazy dog", None, None, None, None, None, None, None)?;

        // Should be deduplicated to 1 record
        assert_eq!(aura.count(None), 1);

        Ok(())
    }

    #[test]
    fn test_stats() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let aura = Aura::open(dir.path().to_str().unwrap())?;

        aura.store("test1", Some(Level::Working), None, None, None, None, Some(false), None)?;
        aura.store("test2", Some(Level::Identity), None, None, None, None, Some(false), None)?;

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
            None, None, None, Some(false), None,
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
            None, None, None, Some(false), None,
            Some("telegram"),
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

        aura.store("Teo loves Rust programming", Some(Level::Identity), None, None, None, None, Some(false), None)?;

        // First recall — cache miss
        let r1 = aura.recall("Who is Teo?", None, None, None, None)?;
        // Second recall — cache hit (same result)
        let r2 = aura.recall("Who is Teo?", None, None, None, None)?;
        assert_eq!(r1, r2);

        // Store invalidates cache
        aura.store("Teo is 25 years old", None, None, None, None, None, Some(false), None)?;

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

        aura.store("test record", None, None, None, None, None, Some(false), None)?;

        let report = aura.run_maintenance();
        assert!(report.total_records > 0);
        assert!(!report.timestamp.is_empty());

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
}
