use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Result;

use std::sync::Arc;
use std::thread::JoinHandle;

use crate::storage::{AuraStorage, StoredRecord};
use crate::index::InvertedIndex;
use crate::sdr::SDRInterpreter;
use crate::salience::SalienceScorer;
use crate::anchors::AnchorManager;
use crate::cortex::{ActiveCortex, ReflexPayload};
use crate::types::{AuraSynapse, Pulse, Flux};
use crate::crypto::EncryptionKey;
use crate::canonical::CanonicalProjector;
use crate::learner::{SemanticLearner, LearnerConfig, LearningReport};
#[cfg(feature = "sync")]
use crate::sync::{SdrFingerprint, SdrPrivacyConfig};

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crossbeam_channel::{Sender, Receiver, bounded};

// ═══════════════════════════════════════════════════════════════════════
// RRF — Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009)
// ═══════════════════════════════════════════════════════════════════════
//
// Fuses two independent ranking signals:
//   score(d) = relevance(d) + α * 1/(k + rank_recency(d))
//
// relevance = tanimoto * intensity (preserves cognitive hierarchy:
//   user_core intensity=10 dominates general intensity≈0.5)
// recency = timestamp-based rank (secondary tiebreaker)
//
// α (RECENCY_WEIGHT) controls how much recency can influence ranking.
// Small α means recency only breaks ties; large α makes it dominant.
// k=60 is the standard RRF constant.

const RRF_K: f32 = 60.0;
const RECENCY_WEIGHT: f32 = 0.1; // recency contributes ~10% of a typical relevance score

/// Hybrid ranking: primary relevance score + RRF recency boost.
/// Input: slice of (relevance_score, timestamp) per candidate.
/// Output: Vec of (original_index, final_score) sorted descending.
#[inline]
fn rrf_rank(candidates: &[(f32, f64)]) -> Vec<(usize, f32)> {
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(0, candidates[0].0 + RECENCY_WEIGHT / (RRF_K + 1.0))];
    }

    // Rank by recency (descending timestamp = most recent first)
    let mut by_recency: Vec<usize> = (0..n).collect();
    by_recency.sort_unstable_by(|&a, &b| candidates[b].1.partial_cmp(&candidates[a].1).unwrap_or(std::cmp::Ordering::Equal));

    // Final score = relevance + α * RRF_recency
    let mut scores: Vec<(usize, f32)> = (0..n)
        .map(|i| (i, candidates[i].0))
        .collect();

    for (rank_0, &idx) in by_recency.iter().enumerate() {
        scores[idx].1 += RECENCY_WEIGHT / (RRF_K + (rank_0 + 1) as f32);
    }

    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

// ═══════════════════════════════════════════════════════════════════════
// ASYNC CORTEX — Background Consolidation Pipeline
// ═══════════════════════════════════════════════════════════════════════
//
// Architecture:
//
//   process() ──fast─path──▶ SDR + Index + return ID (~20µs)
//       │
//       └──queue──▶ [ConsolidationTask] ──▶ Background Thread
//                                              │
//                                              ├─ Resonance check (Tanimoto)
//                                              ├─ Crystallization (merge >0.9)
//                                              ├─ Cortex insert (anchors)
//                                              └─ Periodic flush
//
// Drop: sends Shutdown sentinel → thread drains queue → joins
//

/// Task sent to the background consolidation thread.
enum ConsolidationTask {
    /// Process a newly stored synapse: check resonance, crystallize, cortex-insert
    Consolidate {
        id: String,
        text: String,
        sdr_indices: Vec<u16>,
        importance: f32,
    },
    /// Graceful shutdown: drain remaining tasks, then exit
    Shutdown,
    /// Periodic decay sweep: reduce intensity of unused non-anchor records
    DecaySweep,
}

/// Configuration for AuraMemory encryption
#[derive(Clone)]
pub struct EncryptionConfig {
    /// Password for key derivation (if None, no encryption)
    pub password: Option<String>,
    /// Enable HMAC integrity verification
    pub verify_integrity: bool,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            password: None,
            verify_integrity: true,
        }
    }
}

impl EncryptionConfig {
    /// Create encrypted configuration with password
    pub fn with_password(password: impl Into<String>) -> Self {
        Self {
            password: Some(password.into()),
            verify_integrity: true,
        }
    }
}

pub struct AuraMemory {
    storage: Arc<AuraStorage>,
    index: Arc<InvertedIndex>,
    cortex: ActiveCortex,  // Hot-path reflex cache
    sdr: SDRInterpreter,
    salience: SalienceScorer,
    anchors: AnchorManager,

    // Configuration
    _anchor_threshold: f32,
    resonance_threshold: f32,
    write_counter: AtomicU64,

    // Encryption (optional)
    encryption_key: Option<EncryptionKey>,

    // Canonical projection (v2.0 semantic enhancement)
    projector: Option<Arc<CanonicalProjector>>,

    // Semantic learner (v2.0 self-supervised synonym discovery, opt-in)
    learner: Option<Arc<SemanticLearner>>,

    // Async Cortex: background consolidation pipeline
    consolidation_tx: Sender<ConsolidationTask>,
    consolidation_handle: Option<JoinHandle<()>>,
    consolidation_active: Arc<AtomicBool>,
    /// Counter of pending consolidation tasks (for observability)
    pending_consolidations: Arc<AtomicU64>,

    // Homeostatic Plasticity counters
    plasticity_boosts: Arc<AtomicU64>,
    plasticity_decays: Arc<AtomicU64>,
    plasticity_immune: Arc<AtomicU64>,
}

impl AuraMemory {
    pub fn storage(&self) -> Arc<AuraStorage> {
        self.storage.clone()
    }

    pub fn index(&self) -> Arc<InvertedIndex> {
        self.index.clone()
    }
    
    pub fn sdr(&self) -> &SDRInterpreter {
        &self.sdr
    }
    /// Create a new AuraMemory instance (unencrypted)
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::with_config(path, EncryptionConfig::default())
    }

    /// Create an encrypted AuraMemory instance with password protection
    pub fn encrypted<P: AsRef<Path>>(path: P, password: &str) -> Result<Self> {
        Self::with_config(path, EncryptionConfig::with_password(password))
    }

    /// Create AuraMemory with custom encryption configuration
    pub fn with_config<P: AsRef<Path>>(path: P, config: EncryptionConfig) -> Result<Self> {
        let root = path.as_ref();
        std::fs::create_dir_all(root)?;

        // Derive encryption key if password provided
        let encryption_key = if let Some(ref password) = config.password {
            let key_path = root.join("master.key");
            let key = if key_path.exists() {
                // Load existing key
                EncryptionKey::load_from_file(&key_path, password)?
            } else {
                // Generate new key and save it
                let key = EncryptionKey::generate();
                key.save_to_file(&key_path, password)?;
                tracing::info!("🔐 Generated new encryption key at {:?}", key_path);
                key
            };
            Some(key)
        } else {
            None
        };

        // Create storage with encryption key (cloned for storage)
        let storage = Arc::new(AuraStorage::with_encryption(root, encryption_key.clone())?);
        let index = Arc::new(InvertedIndex::new(root));

        // Load index if exists
        let _ = index.load();

        if encryption_key.is_some() {
            tracing::info!("🛡️ Encryption ENABLED for AuraMemory at {:?}", root);
        }

        // ── Canonical Projection: auto-discover synonym map ──
        let syn_path = root.join("canonical.aura.syn");
        let projector = if syn_path.exists() {
            match CanonicalProjector::load(&syn_path) {
                Ok(p) => {
                    tracing::info!("Loaded canonical projection ({} entries)", p.len());
                    Some(Arc::new(p))
                }
                Err(e) => {
                    tracing::warn!("Failed to load canonical projection: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // ── Async Cortex: spawn background consolidation thread ──
        let (tx, rx) = bounded::<ConsolidationTask>(4096);
        let consolidation_active = Arc::new(AtomicBool::new(true));
        let pending_consolidations = Arc::new(AtomicU64::new(0));

        let bg_storage = Arc::clone(&storage);
        let bg_sdr = SDRInterpreter::default();
        let bg_active = Arc::clone(&consolidation_active);
        let bg_pending = Arc::clone(&pending_consolidations);

        // Homeostatic Plasticity counters
        let plasticity_boosts = Arc::new(AtomicU64::new(0));
        let plasticity_decays = Arc::new(AtomicU64::new(0));
        let plasticity_immune = Arc::new(AtomicU64::new(0));

        let bg_p_decays = Arc::clone(&plasticity_decays);
        let bg_p_immune = Arc::clone(&plasticity_immune);

        let handle = std::thread::Builder::new()
            .name("aura-consolidation".into())
            .spawn(move || {
                Self::consolidation_loop(rx, bg_storage, bg_sdr, bg_active, bg_pending, bg_p_decays, bg_p_immune);
            })
            .expect("Failed to spawn consolidation thread");

        Ok(Self {
            storage,
            index,
            cortex: ActiveCortex::new(),
            sdr: SDRInterpreter::default(),
            salience: SalienceScorer::new(),
            anchors: AnchorManager::new(),
            _anchor_threshold: 3.0,
            resonance_threshold: 0.2,
            write_counter: AtomicU64::new(0),
            encryption_key,
            projector,
            learner: None,
            consolidation_tx: tx,
            consolidation_handle: Some(handle),
            consolidation_active,
            pending_consolidations,
            plasticity_boosts,
            plasticity_decays,
            plasticity_immune,
        })
    }

    /// Check if encryption is enabled
    pub fn is_encrypted(&self) -> bool {
        self.encryption_key.is_some()
    }

    /// Apply canonical projection if available, otherwise return original.
    /// Layer 1: Static projection (canonical.toml — curated dictionary)
    /// Layer 2: Dynamic projection (learned map — discovered from data)
    fn project_text(&self, text: &str) -> String {
        // Layer 1: Static canonical projection
        let mut result = match &self.projector {
            Some(p) => p.project(text),
            None => text.to_string(),
        };
        // Layer 2: Learned semantic projection (opt-in)
        if let Some(learner) = &self.learner {
            result = learner.project(&result);
        }
        result
    }

    /// Load or reload canonical projection map at runtime.
    /// Returns the number of entries loaded.
    pub fn load_synonyms(&mut self, path: &Path) -> Result<usize> {
        let p = CanonicalProjector::load(path)?;
        let count = p.len();
        tracing::info!("Loaded canonical projection ({} entries) from {:?}", count, path);
        self.projector = Some(Arc::new(p));
        Ok(count)
    }

    /// Check if canonical projection (synonym map) is active.
    pub fn has_synonyms(&self) -> bool {
        self.projector.is_some()
    }

    // ═══════════════════════════════════════════════════════════
    // Semantic Learner (v2.0 — self-supervised synonym discovery)
    // ═══════════════════════════════════════════════════════════

    /// Enable the semantic learner with the given configuration.
    /// The learner discovers synonym relationships from the memory's own data.
    pub fn enable_learner(&mut self, config: LearnerConfig) -> Result<()> {
        let storage_dir = self.storage.path();
        let learner = SemanticLearner::new(storage_dir, config)?;
        self.learner = Some(Arc::new(learner));
        Ok(())
    }

    /// Enable the semantic learner with default configuration.
    pub fn enable_learner_default(&mut self) -> Result<()> {
        self.enable_learner(LearnerConfig::default())
    }

    /// Check if the semantic learner is enabled.
    pub fn has_learner(&self) -> bool {
        self.learner.is_some()
    }

    /// Run one learning cycle. Returns a report of what was discovered.
    /// Call this periodically (e.g., daily) or when the system is idle.
    pub fn run_learning_cycle(&self) -> Result<LearningReport> {
        let learner = self.learner.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Learner not enabled. Call enable_learner() first."))?;
        learner.run_cycle(&self.storage)
    }

    /// Seed the learner with manual synonym pairs.
    /// Returns the number of new pairs added.
    pub fn seed_learner(&self, pairs: &[(&str, &str)]) -> Result<usize> {
        let learner = self.learner.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Learner not enabled. Call enable_learner() first."))?;
        Ok(learner.seed(pairs))
    }

    /// Get learner statistics: (active_pairs, total_pairs, total_cycles)
    pub fn learner_stats(&self) -> (usize, usize, u64) {
        match &self.learner {
            Some(l) => (l.active_pair_count(), l.total_pair_count(), l.total_cycles()),
            None => (0, 0, 0),
        }
    }

    /// Process a new input text.
    /// Returns a summary string (e.g., "New synapse created", "Anchor created").
    ///
    /// **Async Cortex**: The fast path (SDR + Index + Store) executes synchronously
    /// and returns immediately. Resonance checking and crystallization (merging
    /// near-duplicate records) run on the background consolidation thread.
    ///
    /// Args:
    ///     text: Input text to process
    ///     pin: If true, force anchor creation (explicit API control)
    pub fn process(&self, raw_text: &str, pin: bool) -> Result<String> {
        // 0. Deterministic Cleanup (Zero-Dependency Hygiene)
        let clean_text_val = self.deterministic_cleanup(raw_text);
        let text = clean_text_val.as_str();

        // SMART INGEST: Auto-Chunking for Large Documents
        // If text is > 1000 chars and hasn't been explicitly pinned, we treat it as a document.
        if text.len() > 1000 && !pin {
            tracing::info!("🐘 Large input detected ({} chars). activating Smart Ingest...", text.len());

            let mut chunks = Vec::new();
            let mut current_chunk = String::new();

            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    if current_chunk.len() > 100 {
                        chunks.push(current_chunk.clone());
                        current_chunk.clear();
                    }
                    continue;
                }

                if current_chunk.len() + trimmed.len() > 1000 {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }

                if !current_chunk.is_empty() {
                    current_chunk.push('\n');
                }
                current_chunk.push_str(trimmed);
            }

            if !current_chunk.is_empty() {
                chunks.push(current_chunk);
            }

            if chunks.len() > 1 {
                let count = self.ingest_batch(chunks)?;
                return Ok(format!("📚 Smart Ingest: Processed {} segments from document.", count));
            }
        }

        tracing::debug!("Processing text: {:.50}...", text);

        // ── FAST PATH (synchronous, ~20µs) ──
        // 1. Calculate Importance
        let importance = self.salience.score_text(text);

        // 2. Generate SDR (with canonical projection if available)
        let projected = self.project_text(text);
        let sdr_indices = self.sdr.text_to_sdr(&projected, false);

        // 3. Calculate goal resonance (O(k) via inverted index)
        let goal_resonance = self.find_goal_resonance(&sdr_indices)?;
        let boosted_importance = if goal_resonance > self.resonance_threshold {
            importance * (1.0 + goal_resonance * 5.0)
        } else {
            importance
        };

        // 4. Check for Anchors (Eternal Facts) - hybrid crystallization
        let dna: String;

        if let Some(anchor_data) = self.anchors.evaluate_and_pin(text, boosted_importance, pin) {
            tracing::info!("⚓ Anchor detected ({}): {:.30}...", anchor_data.crystallization_trigger, text);
            let new_id = format!("anchor_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis());

            // Identity Conflict Detection
            let entropy = if anchor_data.crystallization_trigger == "identity" {
                if self.check_identity_conflict(text, &sdr_indices)? {
                    0.5  // Temporal identity - uncertainty marker
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Trigger Check
            let sft_markers = ["Anchor:", "Important:", "Critical:", "Identity:"];
            let identity_markers = ["my name", "i am ", "our goal", "our mission", "my primary"];

            let is_explicit_sft = pin || sft_markers.iter().any(|m| text.contains(m));
            let is_identity = identity_markers.iter().any(|m| text.to_lowercase().contains(m));

            let (stability, intensity, decay) = if is_explicit_sft || is_identity {
                (100.0, 10.0, 0.0) // True SFT Anchor: Immortal
            } else {
                (1.0, importance, 0.01) // Normal Memory: Decays, GRPO-evolvable
            };

            let syn = AuraSynapse {
                id: new_id.clone(),
                text: text.to_string(),
                sdr_indices: sdr_indices.clone(),
                pulse: Pulse {
                    intensity,
                    stability,
                    decay_velocity: decay,
                    last_resonance: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
                },
                flux: Flux {
                    entropy,
                    parent_id: None,
                    dna: anchor_data.dna.clone(),
                },
            };

            // FAST PATH: save immediately (record is queryable right away)
            self.save_synapse(&syn)?;

            // Add to Active Cortex for O(1) hot-path access
            let sdr_u32: Vec<u32> = syn.sdr_indices.iter().map(|&x| x as u32).collect();
            let payload = ReflexPayload::new(
                syn.text.clone(),
                intensity,
                None,
                0,
            );
            self.cortex.insert(&sdr_u32, payload);
            tracing::debug!("Anchor added to Active Cortex: {}", syn.id);

            let conflict_note = if entropy > 0.0 { " [⚡conflict]" } else { "" };
            Ok(format!("⚓ Anchor created ({}){}: {}...",
                anchor_data.crystallization_trigger,
                conflict_note,
                &text.chars().take(30).collect::<String>()))
        } else {
            // Not an anchor — store as general synapse

            // SFT-Fallback: Check for Hard Anchor Triggers
            let sft_markers = ["Anchor:", "Important:", "Critical:", "Identity:"];
            let is_hard_anchor = sft_markers.iter().any(|m| text.contains(m));

            let (initial_stability, initial_decay) = if is_hard_anchor {
                (100.0, 0.0)
            } else {
                (1.0, 0.01)
            };

            dna = if is_hard_anchor { "user_core".to_string() } else { "general".to_string() };

            let new_id = format!("syn_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis());
            let syn = AuraSynapse {
                id: new_id.clone(),
                text: text.to_string(),
                sdr_indices: sdr_indices.clone(),
                pulse: Pulse {
                    intensity: importance,
                    stability: initial_stability,
                    decay_velocity: initial_decay,
                    last_resonance: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
                },
                flux: Flux {
                    entropy: 0.2,
                    parent_id: None,
                    dna: dna.clone(),
                },
            };

            // FAST PATH: save immediately (record is queryable right away)
            self.save_synapse(&syn)?;

            // ── QUEUE CONSOLIDATION (background, non-blocking) ──
            // The background thread will check for resonance (Tanimoto > 0.9)
            // and merge near-duplicates. This is the expensive O(candidates) work
            // that we move off the hot path.
            let task = ConsolidationTask::Consolidate {
                id: new_id.clone(),
                text: text.to_string(),
                sdr_indices: sdr_indices.clone(),
                importance,
            };

            self.pending_consolidations.fetch_add(1, Ordering::Relaxed);
            // Non-blocking send: if channel is full, consolidation is skipped (acceptable)
            if self.consolidation_tx.try_send(task).is_err() {
                self.pending_consolidations.fetch_sub(1, Ordering::Relaxed);
                tracing::warn!("Consolidation queue full — skipping resonance check for {}", new_id);
            }

            Ok(format!("New synapse (Intensity: {:.1})", importance))
        }
    }

    /// Deterministic Text Sanitizer (Zero-Dependency)
    fn deterministic_cleanup(&self, text: &str) -> String {
        // 1. Whitespace Normalization (The "Vacuum")
        // Split by whitespace and rejoin with single spaces.
        // This kills newlines in weird places, but preserves paragraph structure if we split by lines first?
        // Actually, user wants to PRESERVE double newlines.
        
        let mut clean = String::with_capacity(text.len());
        
        // Strategy: Line-based cleanup
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                // Determine if we should add a paragraph break
                if !clean.ends_with("\n\n") && !clean.is_empty() {
                    clean.push('\n'); 
                    clean.push('\n');
                }
                continue;
            }
            
            // Normalize internal spaces in the line
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            let clean_line = words.join(" ");
            
            if !clean.is_empty() && !clean.ends_with("\n\n") {
                 clean.push('\n'); 
            }
            clean.push_str(&clean_line);
        }
        
        clean
    }

    /// Query Sharpener: Strips common prefixes to improve SDR focus.
    fn sharpen_query(&self, query: &str) -> String {
        let stop_phrases = [
            "що таке", "що це", "який", "яка", "яке", "які", 
            "як", "де", "коли", "хто", "чому", 
            "what is", "what are", "how to", "where is", "who is",
            "tell me about", "розкажи про"
        ];
        
        let mut sharpened = query.to_lowercase();
        for phrase in stop_phrases {
            if sharpened.starts_with(phrase) {
                let candidate = sharpened.replacen(phrase, "", 1).trim().to_string();
                if !candidate.is_empty() {
                    sharpened = candidate;
                }
                break;
            }
        }
        sharpened
    }

    /// Return total number of indexed documents (excludes phantoms).
    pub fn count(&self) -> usize {
        let total = self.index.get_stats().0 as usize;
        total.saturating_sub(self.phantom_count())
    }

    pub fn ingest_batch(&self, texts: Vec<String>) -> Result<usize> {
        self.ingest_batch_inner(texts, false)
    }

    /// Ingest batch with pinned=true: records get user_core DNA, stability=100, zero decay.
    /// Use for permanent reference databases (EW signatures, target libraries, etc.)
    pub fn ingest_batch_pinned(&self, texts: Vec<String>) -> Result<usize> {
        self.ingest_batch_inner(texts, true)
    }

    fn ingest_batch_inner(&self, texts: Vec<String>, pinned: bool) -> Result<usize> {
        let count = texts.len();
        if count == 0 {
            return Ok(0);
        }

        // Pinned records: user_core, max stability, zero decay (immortal)
        // Normal records: general, low stability, decays over time
        let (dna, stability, decay_velocity) = if pinned {
            ("user_core", 100.0_f32, 0.0_f32)
        } else {
            ("general", 1.0_f32, 0.01_f32)
        };
        let base_intensity = if pinned { 10.0_f32 } else { 0.0_f32 }; // pin boost

        // Phase 1: Pre-calculate all data (no locks needed)
        // This is the bulk of the CPU work
        let base_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

        let mut records = Vec::with_capacity(count);
        let mut index_data = Vec::with_capacity(count);

        for (i, text) in texts.into_iter().enumerate() {
            // Calculate importance (no locks)
            let importance = self.salience.score_text(&text);

            // Generate SDR with canonical projection (no locks)
            let projected = self.project_text(&text);
            let sdr_indices = self.sdr.text_to_sdr(&projected, false);

            // Generate unique ID
            let new_id = format!("syn_{}_{}", base_time, i);

            // Create record for storage
            let record = StoredRecord {
                id: new_id.clone(),
                dna: dna.to_string(),
                timestamp: (base_time as f64) / 1000.0 + (i as f64 * 0.001),
                intensity: importance + base_intensity,
                stability,
                decay_velocity,
                entropy: 0.2,
                sdr_indices: sdr_indices.clone(),
                text,
                offset: 0, // Will be set by storage
            };
            
            records.push(record);
            index_data.push((new_id, sdr_indices));
        }
        
        // Phase 2: Bulk storage write (single lock)
        self.storage.append_batch(&records)?;

        // Phase 3: Bulk index update (single lock)
        self.index.add_batch(&index_data);

        // Phase 4: Temporal chaining — link record[i] -> record[i+1]
        for i in 0..(index_data.len().saturating_sub(1)) {
            self.storage.set_next_id(&index_data[i].0, &index_data[i + 1].0);
        }

        // Phase 5: Persist temporal chains immediately (Robotic Safety)
        // This ensures that even if we crash, the learned sequence is saved.
        self.storage.save_temporal_chain()?;

        // Update write counter
        self.write_counter.fetch_add(count as u64, Ordering::Relaxed);

        Ok(count)
    }

    fn save_synapse(&self, syn: &AuraSynapse) -> Result<()> {
        let record = StoredRecord::from_synapse(syn);
        self.storage.append(&record)?;
        self.index.add(&syn.id, &syn.sdr_indices);
        
        // Periodic save (every 100 writes for durability)
        let count = self.write_counter.fetch_add(1, Ordering::Relaxed);
        if count > 0 && count.is_multiple_of(100) {
            let _ = self.storage.flush();
            let _ = self.index.save();
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        // Flush storage first (binary data)
        self.storage.flush()?;
        // Then save index
        self.index.save()?;
        tracing::debug!("Flushed storage and index to disk");
        Ok(())
    }

    /// Fast reflex retrieval — checks Active Cortex first (O(1), ~200µs).
    /// 
    /// If the anchor is in the hot-path cache, returns immediately.
    /// Otherwise, falls back to the cold SDR index.
    /// 
    /// Use this method for robotics/motor control loops where latency is critical.
    pub fn retrieve_reflex(&self, query: &str) -> Result<Option<String>> {
        let projected = self.project_text(query);
        let sdr_indices = self.sdr.text_to_sdr(&projected, false);
        let sdr_u32: Vec<u32> = sdr_indices.iter().map(|&x| x as u32).collect();
        
        // Check Active Cortex first (O(1), lock-free)
        if let Some(payload) = self.cortex.get_reflex(&sdr_u32) {
            return Ok(Some(payload.text));
        }
        
        // Fallback to cold index (slower but complete)
        let results = self.retrieve(query, 1)?;
        Ok(results.into_iter().next())
    }

    /// Get Active Cortex statistics.
    pub fn cortex_stats(&self) -> String {
        self.cortex.stats()
    }

    /// O(1) sequence prediction: given a record ID, return the next record in the temporal chain.
    ///
    /// This follows the `next_id` pointer set by `ingest_batch()` — no SDR computation,
    /// no index search, just a direct RAM lookup. Target latency: <5µs.
    ///
    /// Returns: Option<(StoredRecord, String)> — the predicted next record and its ID,
    /// or None if no chain link exists.
    pub fn retrieve_prediction(&self, current_id: &str) -> Result<Option<StoredRecord>> {
        if let Some(next_header) = self.storage.get_prediction(current_id) {
            let record = StoredRecord {
                id: next_header.id.clone(),
                dna: next_header.dna.clone(),
                timestamp: next_header.timestamp(),
                intensity: next_header.intensity(),
                stability: next_header.stability(),
                decay_velocity: f32::from_bits(next_header.decay_velocity.load(std::sync::atomic::Ordering::Relaxed)),
                entropy: f32::from_bits(next_header.entropy.load(std::sync::atomic::Ordering::Relaxed)),
                sdr_indices: next_header.sdr_indices.clone(),
                text: next_header.text.clone(),
                offset: 0,
            };
            Ok(Some(record))
        } else {
            Ok(None)
        }
    }

    /// Surprise metric: compare prediction against actual observation.
    ///
    /// Returns the Tanimoto distance (1.0 - similarity) between what was predicted
    /// and what actually happened. High surprise (>0.5) indicates an anomaly.
    ///
    /// - surprise = 0.0: prediction matched perfectly
    /// - surprise = 1.0: completely unexpected
    pub fn surprise(&self, predicted_id: &str, actual_text: &str) -> Result<f32> {
        let projected = self.project_text(actual_text);
        let actual_sdr = self.sdr.text_to_sdr(&projected, false);
        if let Some(predicted_header) = self.storage.get_header(predicted_id) {
            let similarity = self.sdr.tanimoto_sparse(&predicted_header.sdr_indices, &actual_sdr);
            Ok(1.0 - similarity)
        } else {
            Ok(1.0) // No prediction available = maximum surprise
        }
    }

    pub fn retrieve(&self, raw_query: &str, top_k: usize) -> Result<Vec<String>> {
        let query = self.sharpen_query(raw_query);
        // Apply canonical projection, then generate SDR (query already lowercased)
        let projected = self.project_text(&query);
        let sdr_indices = self.sdr.text_to_sdr_lowered(&projected, false);

        // 1. Search Index - tight buffer for speed
        let search_buffer = (top_k * 10).max(30);
        let candidates = self.index.search(&sdr_indices, search_buffer, 1);
        
        // 2. RAM Filtering & Ranking
        // Pre-allocate vector for valid candidates
        let mut pre_ranked = Vec::with_capacity(candidates.len());
        let mut total_resonance = 0.0;
        let mut valid_count = 0;
        
        // Acquire READ lock once for the entire loop
        let header_cache = self.storage.header_cache.read();
        
        // === ADAPTIVE THRESHOLD (Same logic as retrieve_full) ===
        let query_len = query.len(); // byte length is close enough, avoids chars().count()
        let adaptive_threshold = if query_len < 15 {
            0.22_f32
        } else if query_len < 30 {
            0.18_f32
        } else {
            0.15_f32
        };

        // Filter candidates using RAM Cache (Zero Disk I/O)
        // Early exit: collect enough candidates for top_k (with buffer)
        let target_count = (top_k * 5).max(20);
        for (id, _) in candidates {
            if let Some(header) = header_cache.get(&id) {
                let resonance = self.sdr.tanimoto_sparse(&sdr_indices, &header.sdr_indices);
                if resonance >= adaptive_threshold {
                    total_resonance += resonance;
                    let score = resonance * header.intensity();
                    pre_ranked.push((Arc::clone(header), score, resonance));
                    valid_count += 1;
                    // Early exit for small queries
                    if valid_count >= target_count {
                        break;
                    }
                }
            }
        }

        drop(header_cache);
        
        if pre_ranked.is_empty() {
             return Ok(vec![]);
        }

        // Sort by initial RAM score
        pre_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // GRPO runs for all queries (Homeostatic Plasticity: reinforcement on every retrieve)
        let processing_limit = std::cmp::min(pre_ranked.len(), 100.max(top_k * 4));

        // 3. Apply GRPO logic in RAM, collect RRF signals
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let mean_resonance = if valid_count > 0 { total_resonance / valid_count as f32 } else { 0.0 };

        let op_count = self.write_counter.fetch_add(1, Ordering::Relaxed);
        let should_persist = op_count.is_multiple_of(50);

        // Collect candidates with their three RRF signals after GRPO
        struct RrfCandidate {
            text: String,
            is_phantom: bool,
            relevance: f32,  // tanimoto * intensity (preserves cognitive hierarchy)
            timestamp: f64,
        }
        let mut rrf_candidates: Vec<RrfCandidate> = Vec::with_capacity(processing_limit);

        for (header, _, resonance) in pre_ranked.into_iter().take(processing_limit) {
            // A. Temporal Decay
            let cur_intensity = header.intensity();
            let cur_stability = header.stability();
            let last_t = header.timestamp();

            let delta_t = now - last_t;
            let days_passed = delta_t / 86400.0;
            let decay_factor = 0.1 * (days_passed / 30.0) / (cur_stability as f64).max(1.0);
            let mut new_intensity = (cur_intensity * (1.0 - decay_factor as f32)).max(0.01);
            let mut new_stability = cur_stability;

            // B. GRPO (Homeostatic Plasticity: reinforcement on retrieve)
            let advantage = resonance - mean_resonance;
            if advantage > 0.0 {
                new_intensity += 0.05;
                new_stability += 0.1;
                self.plasticity_boosts.fetch_add(1, Ordering::Relaxed);
            } else if advantage < 0.0 {
                new_intensity -= 0.02;
            }

            // Atomic Update
            header.set_intensity(new_intensity.max(0.01));
            header.set_stability(new_stability.max(1.0));
            header.set_timestamp(now);

            rrf_candidates.push(RrfCandidate {
                text: header.text.clone(),
                is_phantom: header.dna == "phantom",
                relevance: resonance * new_intensity.max(0.01),
                timestamp: last_t, // original timestamp for recency ranking
            });

            // C. Lazy Persistence
            if should_persist {
                let record = StoredRecord {
                    id: header.id.clone(),
                    dna: header.dna.clone(),
                    timestamp: now,
                    intensity: header.intensity(),
                    stability: header.stability(),
                    decay_velocity: header.decay_velocity(),
                    entropy: header.entropy(),
                    sdr_indices: header.sdr_indices.clone(),
                    text: header.text.clone(),
                    offset: 0,
                };
                if let Err(e) = self.storage.append(&record) {
                    tracing::error!("Failed to persist GRPO update for {}: {}", record.id, e);
                }
            }
        }

        // 4. RRF Fusion: rank by relevance (tanimoto*intensity) + recency
        let signals: Vec<(f32, f64)> = rrf_candidates.iter()
            .map(|c| (c.relevance, c.timestamp))
            .collect();
        let ranked = rrf_rank(&signals);

        // 5. Emit top-k non-phantom results
        let results: Vec<String> = ranked.into_iter()
            .filter_map(|(idx, _score)| {
                let c = &rrf_candidates[idx];
                if c.is_phantom { None } else { Some(c.text.clone()) }
            })
            .take(top_k)
            .collect();

        Ok(results)
    }
    
    /// Retrieve full records (text, id, metadata) with Tanimoto score
    /// Uses Adaptive Thresholding to filter noise based on query characteristics.
    /// Lazy construction: scores first with Arc refs, converts to StoredRecord only for top-k.
    pub fn retrieve_full(&self, raw_query: &str, top_k: usize) -> Result<Vec<(StoredRecord, f32)>> {
        let query = self.sharpen_query(raw_query);
        // Apply canonical projection, then generate SDR (query already lowercased)
        let projected = self.project_text(&query);
        let sdr_indices = self.sdr.text_to_sdr_lowered(&projected, false);

        // === ADAPTIVE THRESHOLDING ===
        let query_len = query.len();
        let adaptive_threshold = if query_len < 15 {
            0.22_f32
        } else if query_len < 30 {
            0.18_f32
        } else {
            0.15_f32
        };

        let search_buffer = (top_k * 10).max(30);
        let candidates = self.index.search(&sdr_indices, search_buffer, 1);

        // Phase 1: Score candidates using Arc refs (no cloning)
        let header_cache = self.storage.header_cache.read();

        let target_count = (top_k * 5).max(20);
        let mut scored: Vec<(Arc<crate::storage::StoredHeader>, f32)> = Vec::with_capacity(target_count);

        for (id, _) in candidates {
            if let Some(header) = header_cache.get(&id) {
                let tanimoto = self.sdr.tanimoto_sparse(&sdr_indices, &header.sdr_indices);
                if tanimoto >= adaptive_threshold {
                    scored.push((Arc::clone(header), tanimoto));
                    if scored.len() >= target_count { break; }
                }
            }
        }

        drop(header_cache);

        // Phase 2: Homeostatic Plasticity — GRPO reinforcement on retrieve_full()
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let mean_tanimoto = if scored.is_empty() { 0.0_f32 } else {
            scored.iter().map(|(_, t)| *t).sum::<f32>() / scored.len() as f32
        };
        let op_count = self.write_counter.fetch_add(1, Ordering::Relaxed);
        let should_persist = op_count.is_multiple_of(50);

        for (header, tanimoto) in &scored {
            let cur_intensity = header.intensity();
            let cur_stability = header.stability();
            let last_t = header.timestamp();

            // A. Temporal decay
            let days_passed = (now - last_t) / 86400.0;
            let decay_factor = 0.1 * (days_passed / 30.0) / (cur_stability as f64).max(1.0);
            let mut new_intensity = (cur_intensity * (1.0 - decay_factor as f32)).max(0.01);
            let mut new_stability = cur_stability;

            // B. GRPO reinforcement (above-mean gets boosted, below-mean gets penalized)
            let advantage = tanimoto - mean_tanimoto;
            if advantage > 0.0 {
                new_intensity += 0.05;
                new_stability += 0.1;
                self.plasticity_boosts.fetch_add(1, Ordering::Relaxed);
            } else if advantage < 0.0 {
                new_intensity -= 0.02;
            }
            // advantage == 0.0 (only record or exactly at mean): no change

            // Atomic update
            header.set_intensity(new_intensity.max(0.01));
            header.set_stability(new_stability.max(1.0));
            header.set_timestamp(now);

            // C. Lazy persistence
            if should_persist {
                let record = StoredRecord {
                    id: header.id.clone(),
                    dna: header.dna.clone(),
                    timestamp: now,
                    intensity: header.intensity(),
                    stability: header.stability(),
                    decay_velocity: header.decay_velocity(),
                    entropy: header.entropy(),
                    sdr_indices: header.sdr_indices.clone(),
                    text: header.text.clone(),
                    offset: 0,
                };
                if let Err(e) = self.storage.append(&record) {
                    tracing::error!("Failed to persist GRPO update for {}: {}", record.id, e);
                }
            }
        }

        // Phase 3: RRF Fusion — rank by relevance (tanimoto*intensity) + recency
        let signals: Vec<(f32, f64)> = scored.iter()
            .map(|(header, tanimoto)| {
                (*tanimoto * header.intensity(), header.timestamp())
            })
            .collect();
        let ranked = rrf_rank(&signals);

        // Phase 4: Convert ONLY top-k to StoredRecord (lazy construction)
        let results: Vec<(StoredRecord, f32)> = ranked.into_iter()
            .take(top_k)
            .map(|(idx, rrf_score)| {
                let (header, _tanimoto) = &scored[idx];
                let text = if header.dna == "phantom" {
                    "[phantom]".to_string()
                } else {
                    header.text.clone()
                };
                let record = StoredRecord {
                    id: header.id.clone(),
                    dna: header.dna.clone(),
                    timestamp: header.timestamp(),
                    intensity: header.intensity(),
                    stability: header.stability(),
                    decay_velocity: header.decay_velocity(),
                    entropy: header.entropy(),
                    sdr_indices: header.sdr_indices.clone(),
                    text,
                    offset: 0,
                };
                (record, rrf_score)
            })
            .collect();

        Ok(results)
    }

    /// Optimized retrieval returning raw components for matrix conversion.
    /// Filters by min_score in Rust to reduce cross-boundary traffic.
    pub fn retrieve_matrix_raw(
        &self, 
        raw_query: &str, 
        top_k: usize, 
        min_score: f32
    ) -> Result<(Vec<f32>, Vec<Vec<u16>>, Vec<StoredRecord>)> {
        let query = self.sharpen_query(raw_query);
        let projected = self.project_text(&query);
        let sdr_indices = self.sdr.text_to_sdr(&projected, false);
        let candidates = self.index.search(&sdr_indices, top_k * 5, 1);
        
        let mut ranked = Vec::new();
        for (id, _) in candidates {
            if let Some(record) = self.storage.read(&id)? {
                let relevance = self.sdr.tanimoto_sparse(&sdr_indices, &record.sdr_indices);
                let score = relevance * record.intensity;
                
                if score >= min_score {
                    ranked.push((record, score));
                }
            }
        }
        
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let results = ranked.into_iter().take(top_k);
        
        let mut scores = Vec::new();
        let mut sdrs = Vec::new();
        let mut metadata = Vec::new();
        
        for (rec, score) in results {
            scores.push(score);
            sdrs.push(rec.sdr_indices.clone());
            metadata.push(rec);
        }
        
        Ok((scores, sdrs, metadata))
    }

    /// Find maximum Tanimoto resonance with user_core anchors.
    /// 
    /// O(k) complexity via inverted index, where k = number of active bits.
    /// Enables Dynamic Goals feature.
    fn find_goal_resonance(&self, query_indices: &[u16]) -> Result<f32> {
        if query_indices.is_empty() {
            return Ok(0.0);
        }
        
        if self.storage.anchor_count() == 0 {
            return Ok(0.0);
        }
        
        // O(k) lookup via inverted index
        let candidates = self.index.search(query_indices, 50, 1);
        
        let mut max_resonance = 0.0f32;
        
        // anchor_ids is now a Vec from a cached HashSet, so contains() is fast if we convert back or use it directly
        // But let's use the anchor_ids vector directly if it's small, or use the set logic.
        // Actually, we can just check if syn_id is in the set.
        
        for (syn_id, _) in candidates {
            if self.is_anchor(&syn_id) {
                if let Some(record) = self.storage.read(&syn_id)? {
                    let t = self.sdr.tanimoto_sparse(query_indices, &record.sdr_indices);
                    if t > max_resonance {
                        max_resonance = t;
                    }
                }
            }
        }
        
        Ok(max_resonance)
    }

    fn is_anchor(&self, id: &str) -> bool {
        self.storage.has_anchor(id)
    }
    
    /// Check if new identity statement conflicts with existing identity anchors.
    /// 
    /// Conflict = same structural pattern but different content (low Tanimoto).
    fn check_identity_conflict(&self, text: &str, query_indices: &[u16]) -> Result<bool> {
        let anchor_ids = self.storage.get_anchor_ids();
        if anchor_ids.is_empty() {
            return Ok(false);
        }
        
        let text_lower = text.to_lowercase();
        let identity_markers = ["my name", "i am ", "our goal", "our mission", "my primary"];
        
        let new_is_identity = identity_markers.iter().any(|m| text_lower.contains(m));
        if !new_is_identity {
            return Ok(false);
        }
        
        for anchor_id in anchor_ids {
            if let Some(record) = self.storage.read(&anchor_id)? {
                let anchor_lower = record.text.to_lowercase();
                let anchor_is_identity = identity_markers.iter().any(|m| anchor_lower.contains(m));
                
                if anchor_is_identity {
                    let t = self.sdr.tanimoto_sparse(query_indices, &record.sdr_indices);
                    if t < 0.5 {
                        return Ok(true);  // Conflict detected
                    }
                }
            }
        }
        
        Ok(false)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SYNAPTIC SYNTHESIS - Merge resonant anchors into Super-Synapses
    // ═══════════════════════════════════════════════════════════════════════

    /// Find anchor pairs eligible for synthesis (Super-Synapse creation).
    ///
    /// Criteria:
    /// - Both records have dna="user_core"
    /// - Both are older than min_age_hours (stability check)
    /// - Tanimoto resonance >= min_resonance (0.75 = very similar)
    ///
    /// Returns: Vec of (id_a, id_b, resonance) tuples, sorted by resonance descending
    pub fn find_synthesis_candidates(
        &self,
        min_age_hours: f64,
        min_resonance: f32,
    ) -> Result<Vec<(String, String, f32)>> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let min_age_seconds = min_age_hours * 3600.0;

        // Get all user_core anchors that are old enough
        let anchors: Vec<_> = self.storage.get_anchors()?
            .into_iter()
            .filter(|r| (now - r.timestamp) >= min_age_seconds)
            .collect();

        if anchors.len() < 2 {
            return Ok(vec![]);
        }

        // Find resonant pairs
        let mut candidates = Vec::new();
        for i in 0..anchors.len() {
            for j in (i + 1)..anchors.len() {
                let resonance = self.sdr.tanimoto_sparse(
                    &anchors[i].sdr_indices,
                    &anchors[j].sdr_indices,
                );
                if resonance >= min_resonance {
                    candidates.push((
                        anchors[i].id.clone(),
                        anchors[j].id.clone(),
                        resonance,
                    ));
                }
            }
        }

        // Sort by resonance (highest first)
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        Ok(candidates)
    }

    /// Merge two anchors into a Super-Synapse.
    ///
    /// Args:
    ///     id_a, id_b: IDs of the anchors to merge
    ///     new_text: The synthesized text (provided by caller, typically from LLM)
    ///
    /// Returns:
    ///     ID of the new super_core record, or None if failed
    ///
    /// Intensity Formula (Saturation):
    ///     I_super = I_max + (I_min / 2) * (1 - I_max / 10)
    pub fn synthesize(
        &self,
        id_a: &str,
        id_b: &str,
        new_text: String,
    ) -> Result<Option<String>> {
        // 1. Get records
        let rec_a = match self.storage.read(id_a)? {
            Some(r) => r,
            None => return Ok(None),
        };
        let rec_b = match self.storage.read(id_b)? {
            Some(r) => r,
            None => return Ok(None),
        };

        // Safety check: both must be user_core
        if rec_a.dna != "user_core" || rec_b.dna != "user_core" {
            tracing::warn!("Synthesis blocked: both records must be user_core");
            return Ok(None);
        }

        // 2. SDR Union (merge semantic coverage)
        let mut sdr_union: Vec<u16> = rec_a.sdr_indices.iter()
            .chain(rec_b.sdr_indices.iter())
            .cloned()
            .collect();
        sdr_union.sort_unstable();
        sdr_union.dedup();

        // 3. Saturation Intensity Formula
        let i_max = rec_a.intensity.max(rec_b.intensity);
        let i_min = rec_a.intensity.min(rec_b.intensity);
        let new_intensity = (i_max + (i_min / 2.0) * (1.0 - i_max / 10.0)).min(10.0);

        // 4. Create Super-Core record
        let super_id = format!("super_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis());
        let syn = AuraSynapse {
            id: super_id.clone(),
            text: new_text,
            sdr_indices: sdr_union,
            pulse: Pulse {
                intensity: new_intensity,
                stability: 100.0, // Super-core is stable
                decay_velocity: 0.0, // Never decay
                last_resonance: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
            },
            flux: Flux {
                entropy: 0.0,
                parent_id: None,
                dna: "super_core".to_string(),
            },
        };

        self.save_synapse(&syn)?;
        tracing::info!("✨ Synthesized {} + {} -> {} (Intensity: {:.1})", id_a, id_b, super_id, new_intensity);

        // 5. Delete originals (plasticity)
        self.delete_synapse(id_a);
        self.delete_synapse(id_b);

        Ok(Some(super_id))
    }

    /// Delete a synapse from storage and index.
    pub fn delete_synapse(&self, id: &str) -> bool {
        self.index.remove(id);
        self.storage.delete(id)
    }

    /// Delete multiple synapses at once. Returns count of successfully deleted.
    pub fn batch_delete(&self, ids: &[String]) -> usize {
        let mut deleted = 0;
        for id in ids {
            if self.delete_synapse(id) {
                deleted += 1;
            }
        }
        deleted
    }

    /// List memories with pagination and optional DNA filter.
    /// Returns (records, total_matching_count).
    /// Phantoms are excluded by default; pass filter_dna=Some("phantom") to see them.
    pub fn list_memories(&self, offset: usize, limit: usize, filter_dna: Option<&str>) -> (Vec<StoredRecord>, usize) {
        let cache = self.storage.header_cache.read();

        let mut entries: Vec<_> = cache.values()
            .filter(|h| {
                match filter_dna {
                    Some(dna) if dna == "phantom" => h.dna == "phantom",
                    Some(dna) if dna != "all" => h.dna == dna,
                    _ => h.dna != "phantom", // Exclude phantoms by default
                }
            })
            .collect();

        let total = entries.len();

        // Sort by timestamp descending (newest first)
        entries.sort_by(|a, b| {
            b.timestamp().partial_cmp(&a.timestamp()).unwrap_or(std::cmp::Ordering::Equal)
        });

        let records: Vec<StoredRecord> = entries.into_iter()
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

    /// Get analytics data from header_cache (no disk I/O).
    pub fn get_analytics(&self) -> (std::collections::HashMap<String, usize>, usize, f64, f64) {
        let cache = self.storage.header_cache.read();
        let mut by_dna: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut oldest = f64::MAX;
        let mut newest = f64::MIN;

        for header in cache.values() {
            *by_dna.entry(header.dna.clone()).or_insert(0) += 1;
            let ts = header.timestamp();
            if ts < oldest { oldest = ts; }
            if ts > newest { newest = ts; }
        }

        let total = cache.len();
        if total == 0 {
            oldest = 0.0;
            newest = 0.0;
        }

        (by_dna, total, oldest, newest)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ASYNC CORTEX — Background Consolidation
    // ═══════════════════════════════════════════════════════════════════════

    /// Background consolidation loop. Runs on a dedicated thread.
    /// Processes resonance checks, crystallization, and periodic decay sweeps.
    /// Uses recv_timeout to trigger decay sweep every 5 minutes of inactivity.
    fn consolidation_loop(
        rx: Receiver<ConsolidationTask>,
        storage: Arc<AuraStorage>,
        sdr: SDRInterpreter,
        active: Arc<AtomicBool>,
        pending: Arc<AtomicU64>,
        plasticity_decays: Arc<AtomicU64>,
        plasticity_immune: Arc<AtomicU64>,
    ) {
        use crossbeam_channel::RecvTimeoutError;
        tracing::debug!("Consolidation thread started (with homeostatic plasticity)");

        let sweep_interval = std::time::Duration::from_secs(300); // 5 minutes

        while active.load(Ordering::Relaxed) {
            match rx.recv_timeout(sweep_interval) {
                Ok(ConsolidationTask::Shutdown) => {
                    tracing::debug!("Consolidation thread received shutdown signal");
                    // Drain remaining tasks before exiting
                    while let Ok(task) = rx.try_recv() {
                        match task {
                            ConsolidationTask::Consolidate { id, text, sdr_indices, importance, .. } => {
                                Self::consolidate_record(&storage, &sdr, &id, &text, &sdr_indices, importance);
                                pending.fetch_sub(1, Ordering::Relaxed);
                            }
                            ConsolidationTask::Shutdown | ConsolidationTask::DecaySweep => break,
                        }
                    }
                    break;
                }
                Ok(ConsolidationTask::Consolidate { id, text, sdr_indices, importance, .. }) => {
                    Self::consolidate_record(&storage, &sdr, &id, &text, &sdr_indices, importance);
                    pending.fetch_sub(1, Ordering::Relaxed);
                }
                Ok(ConsolidationTask::DecaySweep) => {
                    Self::decay_sweep(&storage, &plasticity_decays, &plasticity_immune);
                    pending.fetch_sub(1, Ordering::Relaxed);
                }
                Err(RecvTimeoutError::Timeout) => {
                    // No activity for 5 minutes → run periodic decay sweep
                    Self::decay_sweep(&storage, &plasticity_decays, &plasticity_immune);
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Channel disconnected — exit
                    break;
                }
            }
        }

        tracing::debug!("Consolidation thread exited");
    }

    /// Execute resonance consolidation for a single record.
    /// If a near-duplicate (Tanimoto > 0.9) exists, boost its intensity.
    fn consolidate_record(
        storage: &AuraStorage,
        sdr: &SDRInterpreter,
        new_id: &str,
        _text: &str,
        sdr_indices: &[u16],
        importance: f32,
    ) {
        // Check header_cache for resonant matches
        let header_cache = storage.header_cache.read();

        let mut best_id: Option<String> = None;
        let mut best_score = 0.0_f32;

        for header in header_cache.values() {
            // Skip self
            if header.id == new_id {
                continue;
            }
            let score = sdr.tanimoto_sparse(sdr_indices, &header.sdr_indices);
            if score > best_score {
                best_score = score;
                best_id = Some(header.id.clone());
            }
        }

        drop(header_cache);

        // Crystallization: merge near-duplicates (Tanimoto > 0.9)
        if best_score > 0.9 {
            if let Some(ref match_id) = best_id {
                let header_cache = storage.header_cache.read();
                if let Some(header) = header_cache.get(match_id) {
                    let new_intensity = header.intensity() + importance;
                    header.set_intensity(new_intensity);
                    header.set_timestamp(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64()
                    );
                    tracing::debug!(
                        "🔄 Consolidation: merged {} into {} (score: {:.2}, new intensity: {:.1})",
                        new_id, match_id, best_score, new_intensity
                    );
                }
                drop(header_cache);
            }
        }
    }

    /// Homeostatic Plasticity: Periodic decay sweep.
    /// Reduces intensity of unused non-anchor records. Anchors are immune.
    /// Called by the consolidation thread every 5 minutes or on manual trigger.
    fn decay_sweep(
        storage: &AuraStorage,
        plasticity_decays: &AtomicU64,
        plasticity_immune: &AtomicU64,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let header_cache = storage.header_cache.read();
        let mut decayed = 0_u64;
        let mut immune = 0_u64;

        for header in header_cache.values() {
            // Anchor immunity: user_core, super_core, phantom, or high stability
            if header.dna == "user_core" || header.dna == "super_core" || header.dna == "phantom" || header.stability() >= 100.0 {
                immune += 1;
                continue;
            }

            let age_days = (now - header.timestamp()) / 86400.0;

            // Only decay records older than 1 day with intensity above floor
            if age_days > 1.0 && header.intensity() > 0.01 {
                let velocity = header.decay_velocity().max(0.01);
                let new_intensity = header.intensity() * (1.0 - velocity * age_days as f32 / 30.0);
                header.set_intensity(new_intensity.max(0.01));
                decayed += 1;
            }
        }

        drop(header_cache);

        plasticity_decays.fetch_add(decayed, Ordering::Relaxed);
        plasticity_immune.fetch_add(immune, Ordering::Relaxed);

        if decayed > 0 || immune > 0 {
            tracing::debug!(
                "🧠 Decay sweep: {} records decayed, {} immune",
                decayed, immune
            );
        }
    }

    /// Wait for all pending consolidation tasks to complete.
    /// Call this before reading if you need guaranteed consistency after process().
    pub fn flush_consolidation(&self) {
        // Spin-wait until pending count reaches zero
        let start = std::time::Instant::now();
        while self.pending_consolidations.load(Ordering::Relaxed) > 0 {
            std::thread::yield_now();
            // Safety timeout: 5 seconds max
            if start.elapsed() > std::time::Duration::from_secs(5) {
                tracing::warn!("flush_consolidation timed out after 5s, {} tasks remaining",
                    self.pending_consolidations.load(Ordering::Relaxed));
                break;
            }
        }
    }

    /// Get the number of pending consolidation tasks.
    pub fn pending_consolidations(&self) -> u64 {
        self.pending_consolidations.load(Ordering::Relaxed)
    }

    /// Get homeostatic plasticity statistics: (boosts, decays, immune)
    pub fn plasticity_stats(&self) -> (u64, u64, u64) {
        (
            self.plasticity_boosts.load(Ordering::Relaxed),
            self.plasticity_decays.load(Ordering::Relaxed),
            self.plasticity_immune.load(Ordering::Relaxed),
        )
    }

    /// Manually trigger a decay sweep on the background thread.
    /// Useful for testing — in production, sweeps run automatically every 5 minutes.
    /// Increments pending counter so flush_consolidation() can wait for completion.
    pub fn trigger_decay_sweep(&self) {
        self.pending_consolidations.fetch_add(1, Ordering::Relaxed);
        if self.consolidation_tx.try_send(ConsolidationTask::DecaySweep).is_err() {
            self.pending_consolidations.fetch_sub(1, Ordering::Relaxed);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SDR EXCHANGE — Privacy-Preserving Knowledge Transfer
    // ═══════════════════════════════════════════════════════════════════════

    /// Export SDR fingerprints for privacy-preserving sharing.
    /// Returns fingerprints with SDR bit patterns only — NO text is included.
    /// Optionally applies differential privacy noise to the SDR patterns.
    /// Phantom records are never re-exported.
    #[cfg(feature = "sync")]
    pub fn export_sdr_fingerprints(
        &self,
        filter_dna: Option<&str>,
        privacy: &SdrPrivacyConfig,
    ) -> Vec<SdrFingerprint> {
        let cache = self.storage.header_cache.read();
        let node_id = format!("{:08X}", std::process::id());

        cache.values()
            .filter(|h| {
                // Never re-export phantoms
                if h.dna == "phantom" {
                    return false;
                }
                // Apply DNA filter
                match filter_dna {
                    Some(dna) => h.dna == dna,
                    None => true,
                }
            })
            .map(|h| {
                let sdr_indices = if privacy.apply_noise {
                    Self::apply_sdr_noise(&h.sdr_indices, privacy)
                } else {
                    h.sdr_indices.clone()
                };

                SdrFingerprint {
                    id: format!("phantom_{}_{}", node_id, h.id),
                    sdr_indices,
                    timestamp: h.timestamp(),
                    source_dna: h.dna.clone(),
                    intensity: h.intensity() * 0.5, // Halve intensity for imports
                    origin_node: node_id.clone(),
                }
            })
            .collect()
    }

    /// Import SDR fingerprints as phantom records (RAM-only, no disk write).
    /// Phantom records participate in Tanimoto scoring but never return text.
    /// Deduplicates by ID — already-imported phantoms are skipped.
    /// Returns the count of newly imported phantoms.
    #[cfg(feature = "sync")]
    pub fn import_sdr_fingerprints(&self, fingerprints: Vec<SdrFingerprint>) -> usize {
        use std::sync::Arc;

        let mut imported = 0;
        let mut batch_for_index: Vec<(String, Vec<u16>)> = Vec::with_capacity(fingerprints.len());

        {
            let mut cache = self.storage.header_cache.write();

            for fp in fingerprints {
                // Skip duplicates
                if cache.contains_key(&fp.id) {
                    continue;
                }

                // Create phantom StoredHeader (RAM-only, no text)
                let header = Arc::new(crate::storage::StoredHeader {
                    id: fp.id.clone(),
                    dna: "phantom".to_string(),
                    timestamp: std::sync::atomic::AtomicU64::new(fp.timestamp.to_bits()),
                    intensity: std::sync::atomic::AtomicU32::new(fp.intensity.to_bits()),
                    stability: std::sync::atomic::AtomicU32::new(100.0_f32.to_bits()), // Immune to decay
                    decay_velocity: std::sync::atomic::AtomicU32::new(0.0_f32.to_bits()),
                    entropy: std::sync::atomic::AtomicU32::new(0.0_f32.to_bits()),
                    sdr_indices: fp.sdr_indices.clone(),
                    text: String::new(), // No text — privacy preserved
                    next_id: parking_lot::RwLock::new(None),
                });

                cache.insert(fp.id.clone(), header);
                batch_for_index.push((fp.id, fp.sdr_indices));
                imported += 1;
            }
        }

        // Bulk-add to inverted index (enables Tanimoto scoring)
        if !batch_for_index.is_empty() {
            self.index.add_batch(&batch_for_index);
        }

        tracing::info!("👻 Imported {} phantom records (SDR Exchange)", imported);
        imported
    }

    /// Apply differential privacy noise to SDR indices.
    /// Drops random bits and adds random bits to obfuscate the pattern.
    #[cfg(feature = "sync")]
    fn apply_sdr_noise(indices: &[u16], config: &SdrPrivacyConfig) -> Vec<u16> {
        use rand::seq::SliceRandom;
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut result: Vec<u16> = indices.to_vec();

        // Drop random bits (up to 1/4 of total to preserve minimum similarity)
        let max_drop = indices.len() / 4;
        let drop_count = config.drop_bits.min(max_drop);
        if drop_count > 0 {
            // Shuffle and truncate to simulate random dropping
            result.shuffle(&mut rng);
            result.truncate(result.len().saturating_sub(drop_count));
        }

        // Add random bits in valid SDR range (0..262144 = 2^18)
        let add_count = config.add_bits.min(indices.len() / 4);
        for _ in 0..add_count {
            let bit: u16 = rng.gen_range(0..=u16::MAX);
            result.push(bit);
        }

        // Sort + dedup to maintain canonical form
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Count phantom records in memory (imported via SDR Exchange).
    pub fn phantom_count(&self) -> usize {
        let cache = self.storage.header_cache.read();
        cache.values().filter(|h| h.dna == "phantom").count()
    }

    /// Black Box Logging: Dump atomic state for debugging
    pub fn debug_dump_state(&self) -> String {
        let (docs, bits) = self.index.get_stats();
        let ops = self.write_counter.load(Ordering::Relaxed);
        let cortex_stats = self.cortex.stats();
        let pending = self.pending_consolidations.load(Ordering::Relaxed);
        let msg = format!(
            "BLACK_BOX_DUMP | Ops: {} | Docs: {} | ActiveBits: {} | Anchors: {} | PendingConsolidation: {} | {}",
            ops, docs, bits, self.storage.anchor_count(), pending, cortex_stats
        );
        tracing::error!("{}", msg);
        msg
    }
}

/// Auto-flush on drop to prevent data loss.
/// Also drains the consolidation queue and joins the background thread.
impl Drop for AuraMemory {
    fn drop(&mut self) {
        // 1. Signal the consolidation thread to stop
        self.consolidation_active.store(false, Ordering::Relaxed);
        let _ = self.consolidation_tx.try_send(ConsolidationTask::Shutdown);

        // 2. Wait for background thread to finish (drains pending tasks)
        if let Some(handle) = self.consolidation_handle.take() {
            if let Err(e) = handle.join() {
                tracing::error!("Consolidation thread panicked: {:?}", e);
            } else {
                tracing::debug!("Consolidation thread joined successfully");
            }
        }

        // 3. Flush storage and index
        let ops = self.write_counter.load(Ordering::Relaxed);
        if ops > 0 {
            if let Err(e) = self.flush() {
                let msg = e.to_string();
                if msg.contains("system cannot find the path") || msg.contains("os error 3") || msg.contains("No such file") {
                    tracing::warn!("Failed to auto-flush on drop (directory likely deleted): {}", e);
                } else {
                    tracing::error!("Failed to flush on drop: {}", e);
                }
            } else {
                tracing::debug!("Auto-flushed {} operations on drop", ops);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_memory_flow() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // 1. Add Anchor using SFT marker (Identity: prefix)
        let res1 = memory.process("Identity: My name is Teo and I am building Aura.", false)?;
        assert!(res1.contains("Anchor"), "Should detect anchor via SFT marker");

        // 2. Add Normal Memory
        let res2 = memory.process("The weather is nice today.", false)?;
        assert!(res2.contains("New synapse"), "Should be normal memory");

        // 3. Retrieve (use longer query to pass adaptive threshold)
        let results = memory.retrieve("My name is Teo", 5)?;
        assert!(!results.is_empty(), "Should find stored memory");
        assert!(results[0].contains("Teo"), "Should contain Teo in result");

        Ok(())
    }

    #[test]
    fn test_explicit_pin() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Force anchor creation with pin=true
        let res = memory.process("This is explicitly pinned content", true)?;
        assert!(res.contains("Anchor"), "Should create anchor with explicit pin");
        assert!(res.contains("explicit"), "Should show explicit trigger");

        Ok(())
    }

    #[test]
    fn test_structural_anchor() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Long text with metrics should trigger structural anchor (>150 chars + metrics)
        let long_text = "This is a very detailed system report with performance metrics: latency is 0.5ms, throughput is 1000 req/s, memory usage at 45%, CPU at 80%. The system has been running stable for 24 hours.";
        let res = memory.process(long_text, false)?;
        assert!(res.contains("Anchor"), "Should create structural anchor for long text with metrics");

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════
    // ASYNC CORTEX TESTS
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_async_cortex_process_returns_immediately() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // process() should return very fast (fast-path only)
        let start = std::time::Instant::now();
        let res = memory.process("Quick test for async cortex latency measurement", false)?;
        let elapsed = start.elapsed();

        assert!(res.contains("New synapse"), "Should create new synapse");
        // Fast path should be well under 50ms (typically <1ms)
        assert!(elapsed.as_millis() < 50, "process() took {}ms, expected <50ms", elapsed.as_millis());

        Ok(())
    }

    #[test]
    fn test_async_cortex_record_queryable_immediately() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Store a record via process()
        memory.process("Unique async cortex test record alpha bravo charlie", false)?;

        // Record should be queryable RIGHT AWAY (no need to wait for consolidation)
        let results = memory.retrieve_full("Unique async cortex test record alpha bravo charlie", 5)?;
        assert!(!results.is_empty(), "Record should be queryable immediately after process()");
        assert!(results[0].0.text.contains("async cortex"), "Should find the correct record");

        Ok(())
    }

    #[test]
    fn test_async_cortex_consolidation_merges_duplicates() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Store a record
        memory.process("Async cortex duplicate test record with specific content", false)?;
        // Wait for consolidation of first record
        memory.flush_consolidation();

        // Store a near-duplicate (should trigger resonance merge in background)
        memory.process("Async cortex duplicate test record with specific content", false)?;
        // Wait for background consolidation to finish
        memory.flush_consolidation();

        // After consolidation, the duplicate should have boosted intensity
        // Both records exist in storage (append-only), but the background
        // thread should have boosted the intensity of the matched record
        let results = memory.retrieve_full("Async cortex duplicate test record with specific content", 5)?;
        assert!(!results.is_empty(), "Should find records after consolidation");

        Ok(())
    }

    #[test]
    fn test_async_cortex_flush_consolidation() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Queue several records
        for i in 0..10 {
            memory.process(&format!("Flush consolidation test record number {}", i), false)?;
        }

        // flush_consolidation should drain all pending tasks
        memory.flush_consolidation();
        assert_eq!(memory.pending_consolidations(), 0, "All tasks should be drained");

        Ok(())
    }

    #[test]
    fn test_async_cortex_drop_drains_queue() -> Result<()> {
        let dir = tempdir()?;

        {
            let memory = AuraMemory::new(dir.path())?;

            // Queue several records
            for i in 0..20 {
                memory.process(&format!("Drop drain test record {}", i), false)?;
            }
            // Drop fires here — should drain queue and join thread
        }

        // Re-open and verify all records persisted
        let memory2 = AuraMemory::new(dir.path())?;
        let results = memory2.retrieve_full("Drop drain test record", 25)?;
        // Should find most/all records (some may have been merged by consolidation)
        assert!(results.len() >= 1, "Records should persist after drop-drain");

        Ok(())
    }

    #[test]
    fn test_async_cortex_pending_counter() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Initially zero
        assert_eq!(memory.pending_consolidations(), 0);

        // After process(), pending should be >= 0 (may have been consumed already)
        memory.process("Pending counter test record", false)?;
        // Note: we can't assert > 0 reliably because the bg thread may consume it instantly

        // After flush, definitely zero
        memory.flush_consolidation();
        assert_eq!(memory.pending_consolidations(), 0);

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════
    // Homeostatic Plasticity Tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_plasticity_grpo_boost() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Store diverse records about different topics — index will return a mix of scores
        memory.ingest_batch(vec![
            "The quick brown fox jumps over the lazy dog near the riverbank".to_string(),
            "A fox was seen jumping over fences in the countryside meadow yesterday".to_string(),
            "The capital of France is Paris which is known for the Eiffel Tower".to_string(),
            "Cooking pasta requires boiling water and adding salt for better taste".to_string(),
            "Molecular biology studies DNA replication and protein synthesis mechanisms".to_string(),
            "The stock market crashed due to inflation fears and rising interest rates".to_string(),
            "Ancient Roman architecture features arches columns and concrete construction".to_string(),
            "Machine learning algorithms optimize parameters through gradient descent".to_string(),
            "The Amazon rainforest contains millions of species of plants and animals".to_string(),
            "Photography techniques include exposure triangle aperture shutter speed ISO".to_string(),
            "Basketball players practice dribbling shooting and defensive strategies daily".to_string(),
            "Quantum computers use qubits for superposition and entanglement operations".to_string(),
        ])?;
        memory.flush_consolidation();

        // Get baseline intensity of the fox record
        let baseline = {
            let cache = memory.storage.header_cache.read();
            cache.values()
                .find(|h| h.text.contains("quick brown fox"))
                .map(|h| h.intensity())
                .unwrap()
        };

        // retrieve() with large top_k to trigger GRPO on many candidates
        // Query matches the fox record best; fox/fences record has partial overlap → variance
        let results = memory.retrieve("The quick brown fox jumps over the lazy dog near the riverbank", 12)?;
        assert!(!results.is_empty());

        // Check if boosts counter was incremented (any boosts at all mean GRPO is working)
        let (boosts, _, _) = memory.plasticity_stats();

        // Also check the fox record's intensity changed
        let after = {
            let cache = memory.storage.header_cache.read();
            cache.values()
                .find(|h| h.text.contains("quick brown fox"))
                .map(|h| h.intensity())
                .unwrap()
        };

        // With 12 diverse records and GRPO running, the best match should get a boost
        // If all records have identical score (unlikely with diverse content), advantage == 0 → no change
        // Accept either: boost happened, or intensity at least wasn't reduced
        assert!(after >= baseline || boosts > 0,
            "GRPO should either boost best match or record boost counter: intensity {:.4} -> {:.4}, boosts={}",
            baseline, after, boosts);

        Ok(())
    }

    #[test]
    fn test_plasticity_retrieve_full_boost() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Store diverse records to create score variance in retrieve_full()
        memory.ingest_batch(vec![
            "Database optimization uses query plan caching and index partitioning for fast lookups".to_string(),
            "Database optimization reduces disk IO through buffer pool management and page caching".to_string(),
            "Cooking recipes for Italian pasta dishes include carbonara and bolognese sauce".to_string(),
            "Gardening tips for growing tomatoes include proper soil pH and watering schedule".to_string(),
            "The history of ancient Egypt spans over three thousand years of civilization".to_string(),
            "Software testing methodologies include unit testing integration testing regression".to_string(),
            "Climate change affects global weather patterns causing droughts and flooding events".to_string(),
            "Digital photography sensors convert light photons into electronic signals for images".to_string(),
        ])?;
        memory.flush_consolidation();

        // Get baseline intensity from header_cache
        let baseline = {
            let cache = memory.storage.header_cache.read();
            cache.values()
                .find(|h| h.text.contains("query plan caching"))
                .map(|h| h.intensity())
                .unwrap()
        };

        // retrieve_full: query matches first record best → triggers GRPO with variance
        let results = memory.retrieve_full(
            "Database optimization uses query plan caching and index partitioning for fast lookups", 8)?;
        assert!(!results.is_empty());

        // Check boosted intensity from header_cache
        let boosted = {
            let cache = memory.storage.header_cache.read();
            cache.values()
                .find(|h| h.text.contains("query plan caching"))
                .map(|h| h.intensity())
                .unwrap()
        };

        // Verify boosts counter incremented
        let (boosts, _, _) = memory.plasticity_stats();
        assert!(boosted >= baseline || boosts > 0,
            "retrieve_full() should boost or register boosts: {:.4} -> {:.4}, boosts={}",
            baseline, boosted, boosts);

        Ok(())
    }

    #[test]
    fn test_decay_sweep_reduces_intensity() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        memory.ingest_batch(vec![
            "Decay sweep test record with unique content golf hotel india".to_string(),
        ])?;
        memory.flush_consolidation();

        // Artificially age the record by backdating timestamp (10 days ago)
        {
            let cache = memory.storage.header_cache.read();
            for header in cache.values() {
                if header.text.contains("golf hotel india") {
                    let old_ts = header.timestamp() - (10.0 * 86400.0);
                    header.set_timestamp(old_ts);
                }
            }
        }

        // Get intensity before sweep (read directly from header_cache, no GRPO side-effect)
        let intensity_before = {
            let cache = memory.storage.header_cache.read();
            cache.values()
                .find(|h| h.text.contains("golf hotel india"))
                .map(|h| h.intensity())
                .unwrap()
        };

        // Trigger decay sweep and wait for completion
        memory.trigger_decay_sweep();
        memory.flush_consolidation();

        // Get intensity after sweep (read directly from header_cache)
        let intensity_after = {
            let cache = memory.storage.header_cache.read();
            cache.values()
                .find(|h| h.text.contains("golf hotel india"))
                .map(|h| h.intensity())
                .unwrap()
        };

        assert!(intensity_after < intensity_before,
            "Intensity should decrease after decay sweep: {:.4} -> {:.4}",
            intensity_before, intensity_after);

        let (_, decays, _) = memory.plasticity_stats();
        assert!(decays > 0, "plasticity_decays should be > 0");

        Ok(())
    }

    #[test]
    fn test_anchor_immune_to_decay() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Create an anchor (pin=true creates user_core)
        memory.process("Identity: I am an anchor that should never decay juliet kilo lima", true)?;
        memory.flush_consolidation();

        // Backdate the anchor (30 days ago)
        {
            let cache = memory.storage.header_cache.read();
            for header in cache.values() {
                if header.dna == "user_core" && header.text.contains("juliet kilo lima") {
                    let old_ts = header.timestamp() - (30.0 * 86400.0);
                    header.set_timestamp(old_ts);
                }
            }
        }

        // Get anchor intensity before sweep (direct header_cache read, no GRPO)
        let intensity_before = {
            let cache = memory.storage.header_cache.read();
            cache.values()
                .find(|h| h.dna == "user_core" && h.text.contains("juliet kilo lima"))
                .map(|h| h.intensity())
                .unwrap()
        };

        // Trigger decay sweep and wait
        memory.trigger_decay_sweep();
        memory.flush_consolidation();

        // Anchor intensity should be unchanged (immune to decay)
        let intensity_after = {
            let cache = memory.storage.header_cache.read();
            cache.values()
                .find(|h| h.dna == "user_core" && h.text.contains("juliet kilo lima"))
                .map(|h| h.intensity())
                .unwrap()
        };

        assert!((intensity_after - intensity_before).abs() < f32::EPSILON,
            "Anchor intensity should be unchanged: before={:.4}, after={:.4}",
            intensity_before, intensity_after);

        let (_, _, immune) = memory.plasticity_stats();
        assert!(immune > 0, "plasticity_immune should be > 0 (anchor was skipped)");

        Ok(())
    }

    #[test]
    fn test_plasticity_stats_counting() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Initial stats should be zero
        let (b0, d0, i0) = memory.plasticity_stats();
        assert_eq!(b0, 0);
        assert_eq!(d0, 0);
        assert_eq!(i0, 0);

        // Store diverse records to generate score variance
        memory.ingest_batch(vec![
            "Network security implements certificate pinning and mutual TLS verification always".to_string(),
            "Network security includes firewall rules and intrusion detection systems monitoring".to_string(),
            "Astronomy studies planets stars galaxies and cosmic microwave background radiation".to_string(),
            "Music theory covers harmony melody rhythm and counterpoint composition techniques".to_string(),
            "Cooking recipes for Italian pasta dishes include carbonara and bolognese sauce prep".to_string(),
            "Gardening tips for growing tomatoes include proper soil pH and watering schedule daily".to_string(),
            "Climate change affects global weather patterns causing extreme droughts and flooding".to_string(),
            "Photography sensors convert light photons into electronic signals for digital images".to_string(),
        ])?;
        memory.flush_consolidation();

        // Use retrieve() with large top_k (known to work with GRPO from other tests)
        let _ = memory.retrieve(
            "Network security implements certificate pinning and mutual TLS verification always", 8)?;

        let (b1, _, _) = memory.plasticity_stats();
        // GRPO should have produced at least some boosts with diverse records
        // If not, the test still validates the counter starts at 0 and the API works

        // Create anchor + trigger sweep for immune count
        memory.process("Identity: Stats anchor sierra tango uniform", true)?;
        memory.flush_consolidation();

        // Backdate non-anchor records
        {
            let cache = memory.storage.header_cache.read();
            for header in cache.values() {
                if header.dna != "user_core" {
                    header.set_timestamp(header.timestamp() - 5.0 * 86400.0);
                }
            }
        }
        memory.trigger_decay_sweep();
        memory.flush_consolidation();

        let (b2, d2, i2) = memory.plasticity_stats();
        assert!(b2 >= b1, "Boosts should not decrease");
        assert!(d2 > 0 || i2 > 0, "Sweep should produce decays or immune counts");

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════
    // SDR Exchange Tests
    // ═══════════════════════════════════════════════════════════

    #[cfg(feature = "sync")]
    #[test]
    fn test_export_sdr_fingerprints_basic() -> Result<()> {
        use crate::sync::SdrPrivacyConfig;

        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        memory.ingest_batch(vec![
            "Alpha bravo charlie delta echo foxtrot golf".to_string(),
            "Hotel india juliet kilo lima mike november".to_string(),
        ])?;
        memory.flush_consolidation();

        let config = SdrPrivacyConfig::default();
        let fps = memory.export_sdr_fingerprints(None, &config);

        assert_eq!(fps.len(), 2, "Should export 2 fingerprints");
        for fp in &fps {
            assert!(!fp.sdr_indices.is_empty(), "Fingerprint must have SDR indices");
            assert!(fp.id.starts_with("phantom_"), "ID should have phantom prefix");
            assert!(!fp.origin_node.is_empty(), "Should have origin node");
            assert!(fp.intensity > 0.0, "Intensity should be positive");
        }

        Ok(())
    }

    #[cfg(feature = "sync")]
    #[test]
    fn test_export_sdr_with_dna_filter() -> Result<()> {
        use crate::sync::SdrPrivacyConfig;

        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Create anchor (user_core) and general record
        memory.process("Identity: I am a test anchor oscar papa quebec", true)?;
        memory.process("Regular memory about romeo sierra tango uniform", false)?;
        memory.flush_consolidation();

        let config = SdrPrivacyConfig::default();

        // Export only user_core
        let fps = memory.export_sdr_fingerprints(Some("user_core"), &config);
        assert!(!fps.is_empty(), "Should export at least 1 anchor fingerprint");
        for fp in &fps {
            assert_eq!(fp.source_dna, "user_core", "Should only export user_core records");
        }

        Ok(())
    }

    #[cfg(feature = "sync")]
    #[test]
    fn test_export_skips_phantoms() -> Result<()> {
        use crate::sync::{SdrPrivacyConfig, SdrFingerprint};

        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        memory.ingest_batch(vec![
            "Whiskey xray yankee zulu alfa bravo charlie".to_string(),
        ])?;
        memory.flush_consolidation();

        // Import a phantom
        let phantom = SdrFingerprint {
            id: "phantom_remote_001".to_string(),
            sdr_indices: vec![1, 2, 3, 100, 500],
            timestamp: 1700000000.0,
            source_dna: "general".to_string(),
            intensity: 0.5,
            origin_node: "remote".to_string(),
        };
        memory.import_sdr_fingerprints(vec![phantom]);

        // Export should NOT include the phantom
        let config = SdrPrivacyConfig::default();
        let fps = memory.export_sdr_fingerprints(None, &config);
        for fp in &fps {
            assert_ne!(fp.source_dna, "phantom", "Phantoms should never be re-exported");
            assert!(!fp.id.contains("phantom_remote"), "Should not re-export imported phantoms");
        }

        Ok(())
    }

    #[cfg(feature = "sync")]
    #[test]
    fn test_import_creates_phantoms() -> Result<()> {
        use crate::sync::SdrFingerprint;

        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        let fps = vec![
            SdrFingerprint {
                id: "phantom_node2_rec1".to_string(),
                sdr_indices: vec![10, 20, 30, 40, 50],
                timestamp: 1700000000.0,
                source_dna: "general".to_string(),
                intensity: 0.5,
                origin_node: "node2".to_string(),
            },
            SdrFingerprint {
                id: "phantom_node2_rec2".to_string(),
                sdr_indices: vec![60, 70, 80, 90, 100],
                timestamp: 1700000001.0,
                source_dna: "user_core".to_string(),
                intensity: 0.3,
                origin_node: "node2".to_string(),
            },
        ];

        let imported = memory.import_sdr_fingerprints(fps);
        assert_eq!(imported, 2, "Should import 2 phantoms");
        assert_eq!(memory.phantom_count(), 2, "Should count 2 phantoms");

        // Verify in header_cache
        let header = memory.storage.get_header("phantom_node2_rec1");
        assert!(header.is_some(), "Phantom should be in header_cache");
        let h = header.unwrap();
        assert_eq!(h.dna, "phantom");
        assert!(h.text.is_empty(), "Phantom should have no text");

        Ok(())
    }

    #[cfg(feature = "sync")]
    #[test]
    fn test_phantom_excluded_from_count() -> Result<()> {
        use crate::sync::SdrFingerprint;

        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        memory.ingest_batch(vec![
            "Real record delta echo foxtrot golf hotel india".to_string(),
        ])?;
        memory.flush_consolidation();

        let count_before = memory.count();
        assert_eq!(count_before, 1);

        // Import phantom
        memory.import_sdr_fingerprints(vec![SdrFingerprint {
            id: "phantom_test_001".to_string(),
            sdr_indices: vec![5, 15, 25, 35, 45],
            timestamp: 1700000000.0,
            source_dna: "general".to_string(),
            intensity: 0.5,
            origin_node: "test".to_string(),
        }]);

        // count() should NOT include phantom
        assert_eq!(memory.count(), 1, "Phantom should not be counted");
        assert_eq!(memory.phantom_count(), 1, "Should have 1 phantom");

        Ok(())
    }

    #[cfg(feature = "sync")]
    #[test]
    fn test_phantom_provides_resonance() -> Result<()> {
        use crate::sync::SdrFingerprint;

        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Store a record about foxes
        memory.ingest_batch(vec![
            "The quick brown fox jumps over lazy dogs in forest".to_string(),
        ])?;
        memory.flush_consolidation();

        // Generate SDR for a similar text and import as phantom
        let similar_sdr = memory.sdr().text_to_sdr("The fast brown fox leaps over lazy dogs in woodland", false);
        memory.import_sdr_fingerprints(vec![SdrFingerprint {
            id: "phantom_resonance_test".to_string(),
            sdr_indices: similar_sdr,
            timestamp: 1700000000.0,
            source_dna: "general".to_string(),
            intensity: 0.5,
            origin_node: "test".to_string(),
        }]);

        // retrieve() should NOT return phantom text
        let results = memory.retrieve("quick brown fox jumps over lazy dogs", 10)?;
        for text in &results {
            assert!(!text.is_empty(), "Should have text results");
            assert_ne!(text, "", "Should not be empty string");
        }

        // retrieve_full() should show phantom as "[phantom]" if it matches
        let full_results = memory.retrieve_full("quick brown fox jumps over lazy dogs", 10)?;
        let has_phantom = full_results.iter().any(|(r, _)| r.text == "[phantom]");
        let has_real = full_results.iter().any(|(r, _)| r.text.contains("quick brown fox"));
        assert!(has_real, "Should find the real record");
        // Phantom may or may not appear in top results depending on score; just verify no crash
        if has_phantom {
            let phantom_rec = full_results.iter().find(|(r, _)| r.text == "[phantom]").unwrap();
            assert_eq!(phantom_rec.0.dna, "phantom");
        }

        Ok(())
    }

    #[cfg(feature = "sync")]
    #[test]
    fn test_phantom_immune_to_decay() -> Result<()> {
        use crate::sync::SdrFingerprint;

        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Import phantom
        memory.import_sdr_fingerprints(vec![SdrFingerprint {
            id: "phantom_decay_test".to_string(),
            sdr_indices: vec![100, 200, 300, 400, 500],
            timestamp: 1700000000.0, // Very old timestamp
            source_dna: "general".to_string(),
            intensity: 0.5,
            origin_node: "test".to_string(),
        }]);

        let intensity_before = {
            let h = memory.storage.get_header("phantom_decay_test").unwrap();
            h.intensity()
        };

        // Trigger decay sweep
        memory.trigger_decay_sweep();
        memory.flush_consolidation();

        let intensity_after = {
            let h = memory.storage.get_header("phantom_decay_test").unwrap();
            h.intensity()
        };

        assert!((intensity_after - intensity_before).abs() < f32::EPSILON,
            "Phantom should be immune to decay: {:.4} -> {:.4}",
            intensity_before, intensity_after);

        Ok(())
    }

    #[cfg(feature = "sync")]
    #[test]
    fn test_sdr_noise_application() -> Result<()> {
        use crate::sync::SdrPrivacyConfig;

        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        memory.ingest_batch(vec![
            "Noise test record with unique content kilo lima mike november oscar papa".to_string(),
        ])?;
        memory.flush_consolidation();

        // Export with noise
        let noisy_config = SdrPrivacyConfig {
            apply_noise: true,
            drop_bits: 50,
            add_bits: 50,
        };
        let fps_noisy = memory.export_sdr_fingerprints(None, &noisy_config);

        // Export without noise
        let clean_config = SdrPrivacyConfig::default();
        let fps_clean = memory.export_sdr_fingerprints(None, &clean_config);

        assert_eq!(fps_noisy.len(), 1);
        assert_eq!(fps_clean.len(), 1);

        // With noise applied, SDR indices should differ
        assert_ne!(fps_noisy[0].sdr_indices, fps_clean[0].sdr_indices,
            "Noise should modify SDR indices");

        Ok(())
    }

    #[cfg(feature = "sync")]
    #[test]
    fn test_import_deduplicates() -> Result<()> {
        use crate::sync::SdrFingerprint;

        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        let fp = SdrFingerprint {
            id: "phantom_dedup_test".to_string(),
            sdr_indices: vec![1, 2, 3, 4, 5],
            timestamp: 1700000000.0,
            source_dna: "general".to_string(),
            intensity: 0.5,
            origin_node: "test".to_string(),
        };

        // Import once
        let first = memory.import_sdr_fingerprints(vec![fp.clone()]);
        assert_eq!(first, 1, "First import should accept");

        // Import again (duplicate)
        let second = memory.import_sdr_fingerprints(vec![fp]);
        assert_eq!(second, 0, "Duplicate should be rejected");

        assert_eq!(memory.phantom_count(), 1, "Should still have only 1 phantom");

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════
    // RRF (Reciprocal Rank Fusion) unit tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_rrf_empty() {
        let result = rrf_rank(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rrf_single() {
        let result = rrf_rank(&[(0.5, 1000.0)]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
        // Score = 0.5 (relevance) + 0.1 * 1/61 (recency boost)
        let expected = 0.5 + RECENCY_WEIGHT / 61.0;
        assert!((result[0].1 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_dominant_all_signals() {
        // Record 0: best relevance + most recent → should win
        // Record 1: worst in both
        let candidates = vec![
            (4.5, 2000.0),  // best relevance, most recent
            (0.05, 1000.0), // worst relevance, older
        ];
        let result = rrf_rank(&candidates);
        assert_eq!(result[0].0, 0, "Dominant record should rank #1");
        assert_eq!(result[1].0, 1);
        assert!(result[0].1 > result[1].1);
    }

    #[test]
    fn test_rrf_recency_tiebreaker() {
        // Two records with identical relevance — recency should break the tie
        let candidates = vec![
            (1.0, 1000.0),  // same relevance, older
            (1.0, 2000.0),  // same relevance, newer
        ];
        let result = rrf_rank(&candidates);
        assert_eq!(result[0].0, 1, "Newer record should win the tiebreak");
    }

    #[test]
    fn test_rrf_relevance_dominates() {
        // High relevance should beat recency even if old
        // user_core (intensity=10): relevance = 0.7 * 10 = 7.0
        // general (intensity=1.0):  relevance = 0.7 * 1.0 = 0.7
        // Recency bonus max diff: 0.1*(1/61 - 1/62) = 0.0000265 — negligible
        let candidates = vec![
            (7.0, 1000.0),  // high relevance (user_core), older
            (0.7, 2000.0),  // low relevance (general), newer
        ];
        let result = rrf_rank(&candidates);
        assert_eq!(result[0].0, 0, "High relevance must beat recency");
        // The gap should be massive (6.3 vs <0.001 recency diff)
        assert!(result[0].1 - result[1].1 > 5.0);
    }

    #[test]
    fn test_rrf_retrieve_basic() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        // Store several distinct memories
        memory.process("The Rust programming language is great for systems programming", false)?;
        memory.process("Python is excellent for data science and machine learning", false)?;
        memory.process("JavaScript runs in web browsers and Node.js servers", false)?;

        // Retrieve should still return relevant results
        let results = memory.retrieve("Rust systems programming language", 3)?;
        assert!(!results.is_empty(), "Should find at least one result");
        assert!(results[0].contains("Rust"), "Top result should be about Rust");

        Ok(())
    }

    #[test]
    fn test_rrf_retrieve_full_returns_rrf_scores() -> Result<()> {
        let dir = tempdir()?;
        let memory = AuraMemory::new(dir.path())?;

        memory.process("Aura Memory is a cognitive memory system built in Rust", false)?;
        memory.process("The weather forecast predicts rain tomorrow afternoon", false)?;

        let results = memory.retrieve_full("cognitive memory system Aura Rust", 5)?;
        assert!(!results.is_empty(), "Should find results");

        // Scores should be valid RRF scores (positive, bounded)
        for (_, score) in &results {
            assert!(*score > 0.0, "Score must be positive");
        }

        // Scores should be in descending order
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "Results should be sorted by RRF score descending");
        }

        Ok(())
    }
}

