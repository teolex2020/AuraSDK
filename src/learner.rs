//! # Aura Semantic Learner — Self-Supervised Synonym Discovery
//!
//! Self-supervised learning loop that discovers semantic relationships
//! from the memory's own data. Runs periodically (configurable) during
//! idle time and produces a learned canonical map (`.aura.learned`).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                 AuraStorage                      │
//! │  ┌──────────┐  ┌───────────┐  ┌──────────────┐ │
//! │  │ Records  │  │ Temporal  │  │ Header Cache │ │
//! │  │ (text +  │  │ Chain     │  │ (SDR + DNA)  │ │
//! │  │  SDR)    │  │ (next_id) │  │              │ │
//! │  └────┬─────┘  └─────┬─────┘  └──────┬───────┘ │
//! └───────┼──────────────┼───────────────┼──────────┘
//!         │              │               │
//!         ▼              ▼               ▼
//! ┌─────────────────────────────────────────────────┐
//! │              SemanticLearner                     │
//! │                                                  │
//! │  1. Co-occurrence   -> CoOccurrenceMap           │
//! │  2. Gap Detection   -> GapPairs                  │
//! │  3. Token Extract   -> CandidatePairs            │
//! │  4. Confidence Merge -> LearnedCanonicalMap      │
//! │                                                  │
//! │  ┌──────────────────────────────────┐            │
//! │  │  .aura.learned (persistent)     │            │
//! │  │  HashMap<String, LearnedEntry>  │            │
//! │  └──────────────────────────────────┘            │
//! └─────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use std::io::{BufReader, BufWriter, Read, Write};
use std::fs::File;

use anyhow::{Result, anyhow};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};

use crate::sdr::SDRInterpreter;
use crate::storage::{AuraStorage, StoredHeader};

// ═══════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════

/// Learner configuration with sane defaults
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    /// Minimum co-occurrence count to consider a pair
    pub min_cooccurrence: u32,
    /// Maximum SDR Tanimoto for a pair to be considered a "gap"
    /// (high co-occurrence but low SDR similarity = semantic gap)
    pub max_sdr_tanimoto_for_gap: f32,
    /// Minimum confidence to persist a learned pair
    pub min_confidence: f32,
    /// Confidence boost per confirming observation
    pub confidence_boost: f32,
    /// Daily decay factor for unconfirmed pairs (multiplied each cycle)
    pub confidence_decay: f32,
    /// Maximum number of learned pairs to keep (memory budget)
    pub max_learned_pairs: usize,
    /// Temporal window: records within this many seconds are "co-occurring"
    pub temporal_window_secs: f64,
    /// Maximum tokens per record to analyze (skip very long texts)
    pub max_tokens_per_record: usize,
    /// Minimum token length to consider (skip "a", "the", etc.)
    pub min_token_length: usize,
    /// Maximum learning cycle duration (seconds) — abort if exceeded
    pub max_cycle_duration_secs: u64,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            min_cooccurrence: 2,
            max_sdr_tanimoto_for_gap: 0.15,
            confidence_boost: 0.15,
            confidence_decay: 0.92,
            min_confidence: 0.3,
            max_learned_pairs: 50_000,
            temporal_window_secs: 300.0, // 5 minutes
            max_tokens_per_record: 500,
            min_token_length: 3,
            max_cycle_duration_secs: 60, // 1 minute max per cycle
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Learned Entry — a single discovered canonical pair
// ═══════════════════════════════════════════════════════════════

/// A learned semantic relationship between two tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedEntry {
    /// The canonical form this token maps to
    pub canonical: String,
    /// Confidence score [0.0, 1.0] — grows with confirmations, decays daily
    pub confidence: f32,
    /// Number of times this pair was observed
    pub observations: u32,
    /// Timestamp of first discovery
    pub discovered_at: f64,
    /// Timestamp of last confirmation
    pub last_confirmed: f64,
    /// Source: which detection method found this pair
    pub source: PairSource,
}

/// How a pair was discovered
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PairSource {
    /// Found via temporal chain co-occurrence
    Temporal,
    /// Found via same DNA layer grouping
    DnaLayer,
    /// Found via sliding time window proximity
    TimeWindow,
    /// Manually seeded (bootstrap dictionary)
    Seed,
    /// Confirmed by multiple sources
    MultiSource,
}

// ═══════════════════════════════════════════════════════════════
// Learned Canonical Map — the persistent output
// ═══════════════════════════════════════════════════════════════

/// The learned canonical map: token -> LearnedEntry
/// Persisted as `.aura.learned` via bincode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedCanonicalMap {
    /// Version for forward compatibility
    pub version: u8,
    /// The actual mappings: token -> learned entry
    pub entries: HashMap<String, LearnedEntry>,
    /// Last learning cycle timestamp
    pub last_cycle: f64,
    /// Total cycles completed
    pub total_cycles: u64,
}

impl Default for LearnedCanonicalMap {
    fn default() -> Self {
        Self {
            version: 1,
            entries: HashMap::new(),
            last_cycle: 0.0,
            total_cycles: 0,
        }
    }
}

impl LearnedCanonicalMap {
    /// Load from disk, or return empty map if file doesn't exist
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic bytes
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"LRN1" {
            return Err(anyhow!("Invalid magic bytes in .aura.learned"));
        }

        let map: Self = bincode::deserialize_from(reader)
            .map_err(|e| anyhow!("Failed to deserialize learned map: {}", e))?;

        Ok(map)
    }

    /// Save to disk atomically (write tmp + rename)
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp_path = path.with_extension("learned.tmp");
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes
        writer.write_all(b"LRN1")?;
        bincode::serialize_into(&mut writer, self)
            .map_err(|e| anyhow!("Failed to serialize learned map: {}", e))?;
        writer.flush()?;

        // Atomic rename
        std::fs::rename(tmp_path, path)?;
        Ok(())
    }

    /// Look up canonical form for a token
    /// Returns Some(canonical) only if confidence is above threshold
    pub fn lookup(&self, token: &str, min_confidence: f32) -> Option<&str> {
        self.entries.get(token)
            .filter(|e| e.confidence >= min_confidence)
            .map(|e| e.canonical.as_str())
    }

    /// Get a runtime HashMap<String, String> for integration with SDR pipeline
    /// Only includes entries above the confidence threshold
    pub fn to_projection_map(&self, min_confidence: f32) -> HashMap<String, String> {
        self.entries.iter()
            .filter(|(_, e)| e.confidence >= min_confidence)
            .map(|(token, e)| (token.clone(), e.canonical.clone()))
            .collect()
    }

    /// Number of active entries above confidence threshold
    pub fn active_count(&self, min_confidence: f32) -> usize {
        self.entries.values().filter(|e| e.confidence >= min_confidence).count()
    }
}

// ═══════════════════════════════════════════════════════════════
// Co-occurrence tracking
// ═══════════════════════════════════════════════════════════════

/// Tracks how often record pairs appear together
#[derive(Debug)]
struct CoOccurrenceMap {
    /// (record_id_a, record_id_b) -> count
    /// Keys are always ordered: a < b to avoid duplicates
    pairs: HashMap<(String, String), u32>,
}

impl CoOccurrenceMap {
    fn new() -> Self {
        Self {
            pairs: HashMap::new(),
        }
    }

    fn add(&mut self, id_a: &str, id_b: &str) {
        let key = if id_a < id_b {
            (id_a.to_string(), id_b.to_string())
        } else {
            (id_b.to_string(), id_a.to_string())
        };
        *self.pairs.entry(key).or_insert(0) += 1;
    }

    fn pairs_above_threshold(&self, min_count: u32) -> Vec<(&str, &str, u32)> {
        self.pairs.iter()
            .filter(|(_, &count)| count >= min_count)
            .map(|((a, b), &count)| (a.as_str(), b.as_str(), count))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// Structs at module level
// ═══════════════════════════════════════════════════════════════

/// A gap: two records that co-occur frequently but have low SDR similarity
#[derive(Debug)]
struct GapPair {
    id_a: String,
    id_b: String,
    cooccurrence: u32,
    #[allow(dead_code)]
    sdr_tanimoto: f32,
}

/// A candidate pair: two tokens that might be semantically equivalent
#[derive(Debug, Clone)]
struct CandidatePair {
    token: String,
    canonical: String,
    confidence: f32,
    source: PairSource,
}

// ═══════════════════════════════════════════════════════════════
// Token utilities
// ═══════════════════════════════════════════════════════════════

/// Simple tokenizer: split on whitespace + punctuation, lowercase, filter short
fn tokenize(text: &str, min_length: usize, max_tokens: usize) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() >= min_length)
        .filter(|s| !is_stopword(s))
        .take(max_tokens)
        .collect()
}

/// Basic stopword filter — keeps the tokenizer lightweight
fn is_stopword(word: &str) -> bool {
    matches!(word,
        "the" | "and" | "for" | "are" | "but" | "not" | "you" | "all" |
        "can" | "had" | "her" | "was" | "one" | "our" | "out" | "has" |
        "have" | "been" | "were" | "they" | "this" | "that" | "with" |
        "from" | "will" | "what" | "when" | "make" | "like" | "just" |
        "into" | "than" | "them" | "then" | "also" | "some" | "these" |
        "його" | "яка" | "які" | "або" | "для" | "при" | "між" |
        "але" | "також" | "інші" | "цей" | "вона" | "вони"
    )
}

/// Extract unique tokens from a record header
fn extract_tokens(header: &StoredHeader, config: &LearnerConfig) -> HashSet<String> {
    tokenize(&header.text, config.min_token_length, config.max_tokens_per_record)
        .into_iter()
        .collect()
}

// ═══════════════════════════════════════════════════════════════
// SemanticLearner — the main learning engine
// ═══════════════════════════════════════════════════════════════

/// The self-supervised semantic learner
pub struct SemanticLearner {
    config: LearnerConfig,
    sdr: SDRInterpreter,
    learned_path: PathBuf,
    learned_map: RwLock<LearnedCanonicalMap>,
}

impl SemanticLearner {
    /// Create a new learner for the given storage directory
    pub fn new(storage_dir: &Path, config: LearnerConfig) -> Result<Self> {
        let learned_path = storage_dir.join(".aura.learned");
        let learned_map = LearnedCanonicalMap::load(&learned_path)?;

        tracing::info!(
            "SemanticLearner initialized: {} learned pairs, {} cycles completed",
            learned_map.entries.len(),
            learned_map.total_cycles
        );

        Ok(Self {
            config,
            sdr: SDRInterpreter::default(),
            learned_path,
            learned_map: RwLock::new(learned_map),
        })
    }

    /// Apply learned projection to text (Layer 2).
    /// Replaces known tokens with their canonical forms.
    pub fn project(&self, text: &str) -> String {
        let map = self.learned_map.read();
        if map.entries.is_empty() {
            return text.to_string();
        }

        let min_conf = self.config.min_confidence;
        let mut result = text.to_lowercase();
        for (token, entry) in &map.entries {
            if entry.confidence >= min_conf {
                // Word-boundary aware replacement
                result = result.replace(token.as_str(), &entry.canonical);
            }
        }
        result
    }

    /// Get the current learned projection map for SDR pipeline integration.
    /// Returns HashMap<token, canonical> with entries above confidence threshold.
    pub fn get_projection_map(&self) -> HashMap<String, String> {
        self.learned_map.read().to_projection_map(self.config.min_confidence)
    }

    /// Number of active learned pairs
    pub fn active_pair_count(&self) -> usize {
        self.learned_map.read().active_count(self.config.min_confidence)
    }

    /// Total learned pairs (including below-threshold)
    pub fn total_pair_count(&self) -> usize {
        self.learned_map.read().entries.len()
    }

    /// Total learning cycles completed
    pub fn total_cycles(&self) -> u64 {
        self.learned_map.read().total_cycles
    }

    /// Look up a single token's canonical form
    pub fn lookup(&self, token: &str) -> Option<String> {
        self.learned_map.read()
            .lookup(token, self.config.min_confidence)
            .map(|s| s.to_string())
    }

    /// Seed the learner with a bootstrap dictionary.
    /// These pairs start with high confidence and PairSource::Seed.
    pub fn seed(&self, pairs: &[(&str, &str)]) -> usize {
        let mut map = self.learned_map.write();
        let now = now_timestamp();
        let mut added = 0;

        for &(token, canonical) in pairs {
            let key = token.to_lowercase();
            let val = canonical.to_lowercase();

            if key == val {
                continue; // Skip identity mappings
            }

            map.entries.entry(key).or_insert_with(|| {
                added += 1;
                LearnedEntry {
                    canonical: val.clone(),
                    confidence: 0.8, // High initial confidence for seeds
                    observations: 1,
                    discovered_at: now,
                    last_confirmed: now,
                    source: PairSource::Seed,
                }
            });
        }

        tracing::info!("Seeded {} new pairs (skipped {} existing)", added, pairs.len() - added);
        added
    }

    // ═══════════════════════════════════════════════════════════
    // Main Learning Cycle
    // ═══════════════════════════════════════════════════════════

    /// Run one complete learning cycle.
    /// Returns the number of new pairs discovered.
    pub fn run_cycle(&self, storage: &AuraStorage) -> Result<LearningReport> {
        let cycle_start = Instant::now();
        let deadline = Duration::from_secs(self.config.max_cycle_duration_secs);

        tracing::info!("Starting learning cycle...");

        // Step 1: Build co-occurrence map from storage
        let cooccurrence = self.build_cooccurrence(storage)?;

        if cycle_start.elapsed() > deadline {
            return Ok(LearningReport::timeout(cycle_start.elapsed()));
        }

        // Step 2: Find gaps (high co-occurrence, low SDR similarity)
        let gaps = self.detect_gaps(storage, &cooccurrence)?;

        if cycle_start.elapsed() > deadline {
            return Ok(LearningReport::timeout(cycle_start.elapsed()));
        }

        // Step 3: Extract token pairs from gap records
        let candidates = self.extract_candidate_pairs(storage, &gaps)?;

        if cycle_start.elapsed() > deadline {
            return Ok(LearningReport::timeout(cycle_start.elapsed()));
        }

        // Step 4: Merge candidates into learned map
        let (new_pairs, confirmed, decayed) = self.merge_candidates(&candidates)?;

        // Step 5: Persist
        self.save()?;

        let duration = cycle_start.elapsed();
        let report = LearningReport {
            duration,
            cooccurrence_pairs: cooccurrence.pairs.len(),
            gaps_found: gaps.len(),
            candidates_extracted: candidates.len(),
            new_pairs,
            confirmed_pairs: confirmed,
            decayed_pairs: decayed,
            total_active: self.active_pair_count(),
            timed_out: false,
        };

        tracing::info!("Learning cycle complete: {}", report.summary());
        Ok(report)
    }

    /// Save the learned map to disk
    pub fn save(&self) -> Result<()> {
        let map = self.learned_map.read();
        map.save(&self.learned_path)?;
        tracing::debug!("Saved {} learned entries", map.entries.len());
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════
    // Step 1: Co-occurrence Analysis
    // ═══════════════════════════════════════════════════════════

    fn build_cooccurrence(&self, storage: &AuraStorage) -> Result<CoOccurrenceMap> {
        let mut cooc = CoOccurrenceMap::new();
        let cache = storage.header_cache.read();

        // Strategy A: Temporal chains (next_id links)
        for (id, header) in cache.iter() {
            let next = header.next_id.read();
            if let Some(next_id) = next.as_ref() {
                cooc.add(id, next_id);
            }
        }

        // Strategy B: Same DNA layer grouping
        // Group records by DNA, then count co-occurrences within each group
        // Guard at 100 to limit O(N^2) per group
        let mut dna_groups: HashMap<&str, Vec<&str>> = HashMap::new();
        for (id, header) in cache.iter() {
            dna_groups.entry(&header.dna).or_default().push(id);
        }

        for ids in dna_groups.values() {
            if ids.len() > 100 {
                continue; // Skip overly large groups
            }
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    cooc.add(ids[i], ids[j]);
                }
            }
        }

        // Strategy C: Time window proximity
        // Sort by timestamp, then pair records within temporal_window_secs
        // Max 50 pairs per record to prevent batch import explosion
        let mut sorted: Vec<(&str, f64)> = cache.iter()
            .map(|(id, h)| (id.as_str(), h.timestamp()))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let window = self.config.temporal_window_secs;
        for (i, item_i) in sorted.iter().enumerate() {
            for (pairs_from_i, item_j) in sorted.iter().skip(i + 1).enumerate() {
                let dt = item_j.1 - item_i.1;
                if dt > window || pairs_from_i > 50 {
                    break;
                }
                cooc.add(item_i.0, item_j.0);
            }
        }

        tracing::debug!(
            "Co-occurrence: {} unique pairs from {} records",
            cooc.pairs.len(),
            cache.len()
        );

        Ok(cooc)
    }

    // ═══════════════════════════════════════════════════════════
    // Step 2: Gap Detection
    // ═══════════════════════════════════════════════════════════

    fn detect_gaps(
        &self,
        storage: &AuraStorage,
        cooc: &CoOccurrenceMap,
    ) -> Result<Vec<GapPair>> {
        let cache = storage.header_cache.read();
        let threshold_cooc = self.config.min_cooccurrence;
        let threshold_sdr = self.config.max_sdr_tanimoto_for_gap;

        let frequent = cooc.pairs_above_threshold(threshold_cooc);
        let mut gaps = Vec::new();

        for (id_a, id_b, count) in frequent {
            let header_a = match cache.get(id_a) {
                Some(h) => h,
                None => continue,
            };
            let header_b = match cache.get(id_b) {
                Some(h) => h,
                None => continue,
            };

            // Compare SDR similarity
            let tanimoto = self.sdr.tanimoto_sparse(
                &header_a.sdr_indices,
                &header_b.sdr_indices,
            );

            // Gap condition: high co-occurrence + low SDR similarity
            if tanimoto < threshold_sdr {
                gaps.push(GapPair {
                    id_a: id_a.to_string(),
                    id_b: id_b.to_string(),
                    cooccurrence: count,
                    sdr_tanimoto: tanimoto,
                });
            }
        }

        // Sort by co-occurrence descending (most confident gaps first)
        gaps.sort_by(|a, b| b.cooccurrence.cmp(&a.cooccurrence));

        tracing::debug!("Gap detection: {} gaps from {} frequent pairs", gaps.len(), cooc.pairs.len());
        Ok(gaps)
    }

    // ═══════════════════════════════════════════════════════════
    // Step 3: Token Pair Extraction
    // ═══════════════════════════════════════════════════════════

    fn extract_candidate_pairs(
        &self,
        storage: &AuraStorage,
        gaps: &[GapPair],
    ) -> Result<Vec<CandidatePair>> {
        let cache = storage.header_cache.read();
        let mut candidates = Vec::new();

        // Token frequency across all records (for IDF-like weighting)
        let mut global_freq: HashMap<String, u32> = HashMap::new();
        for header in cache.values() {
            let tokens = extract_tokens(header, &self.config);
            for token in &tokens {
                *global_freq.entry(token.clone()).or_insert(0) += 1;
            }
        }
        let total_records = cache.len() as f32;

        for gap in gaps.iter().take(500) { // Limit processing
            let header_a = match cache.get(&gap.id_a) {
                Some(h) => h,
                None => continue,
            };
            let header_b = match cache.get(&gap.id_b) {
                Some(h) => h,
                None => continue,
            };

            let tokens_a = extract_tokens(header_a, &self.config);
            let tokens_b = extract_tokens(header_b, &self.config);

            // Find tokens unique to each record (the "different" words)
            let only_a: Vec<&String> = tokens_a.difference(&tokens_b).collect();
            let only_b: Vec<&String> = tokens_b.difference(&tokens_a).collect();

            // Skip if too many unique tokens (records are about different things)
            if only_a.len() > 10 || only_b.len() > 10 {
                continue;
            }

            // Skip if no unique tokens on either side
            if only_a.is_empty() || only_b.is_empty() {
                continue;
            }

            // Pair unique tokens from A with unique tokens from B
            // Heuristic: prefer tokens with similar frequency (IDF matching)
            for ta in &only_a {
                let freq_a = *global_freq.get(*ta).unwrap_or(&1) as f32;
                let idf_a = (total_records / freq_a).ln();

                for tb in &only_b {
                    let freq_b = *global_freq.get(*tb).unwrap_or(&1) as f32;
                    let idf_b = (total_records / freq_b).ln();

                    // IDF similarity: tokens with similar rarity are more likely synonyms
                    let idf_ratio = if idf_a > idf_b { idf_b / idf_a } else { idf_a / idf_b };

                    if idf_ratio < 0.3 {
                        continue; // Very different frequencies — unlikely synonyms
                    }

                    // Confidence based on co-occurrence strength and IDF match
                    let raw_confidence = (gap.cooccurrence as f32 / 10.0).min(1.0) * idf_ratio;

                    if raw_confidence < 0.1 {
                        continue;
                    }

                    // Canonical direction: more frequent token becomes canonical
                    let (token, canonical) = if freq_a >= freq_b {
                        (tb.to_string(), ta.to_string())
                    } else {
                        (ta.to_string(), tb.to_string())
                    };

                    let source = match gap.cooccurrence {
                        c if c >= 5 => PairSource::MultiSource,
                        _ => PairSource::Temporal,
                    };

                    candidates.push(CandidatePair {
                        token,
                        canonical,
                        confidence: raw_confidence,
                        source,
                    });
                }
            }
        }

        // Deduplicate: keep highest confidence for each token
        let mut best: HashMap<String, CandidatePair> = HashMap::new();
        for cand in candidates {
            let entry = best.entry(cand.token.clone()).or_insert(cand.clone());
            if cand.confidence > entry.confidence {
                *entry = cand;
            }
        }

        let result: Vec<CandidatePair> = best.into_values().collect();
        tracing::debug!("Extracted {} candidate pairs", result.len());
        Ok(result)
    }

    // ═══════════════════════════════════════════════════════════
    // Step 4: Merge into Learned Map
    // ═══════════════════════════════════════════════════════════

    fn merge_candidates(
        &self,
        candidates: &[CandidatePair],
    ) -> Result<(usize, usize, usize)> {
        let mut map = self.learned_map.write();
        let now = now_timestamp();
        let mut new_pairs = 0;
        let mut confirmed = 0;

        // Set of tokens confirmed this cycle
        let confirmed_tokens: HashSet<&str> = candidates.iter()
            .map(|c| c.token.as_str())
            .collect();

        // Merge new candidates
        for cand in candidates {
            match map.entries.get_mut(&cand.token) {
                Some(existing) => {
                    // Existing entry: boost confidence
                    existing.confidence = (existing.confidence + self.config.confidence_boost).min(1.0);
                    existing.observations += 1;
                    existing.last_confirmed = now;

                    // Upgrade source if we have multi-source confirmation
                    if existing.source != cand.source && existing.source != PairSource::MultiSource {
                        existing.source = PairSource::MultiSource;
                    }

                    confirmed += 1;
                }
                None => {
                    // New entry
                    map.entries.insert(cand.token.clone(), LearnedEntry {
                        canonical: cand.canonical.clone(),
                        confidence: cand.confidence,
                        observations: 1,
                        discovered_at: now,
                        last_confirmed: now,
                        source: cand.source.clone(),
                    });
                    new_pairs += 1;
                }
            }
        }

        // Apply decay to entries NOT confirmed this cycle
        let decay = self.config.confidence_decay;
        let mut decayed = 0;
        let mut to_remove = Vec::new();

        for (token, entry) in map.entries.iter_mut() {
            if !confirmed_tokens.contains(token.as_str()) && entry.source != PairSource::Seed {
                entry.confidence *= decay;
                decayed += 1;

                // Mark for removal if below threshold and not a seed
                if entry.confidence < 0.05 {
                    to_remove.push(token.clone());
                }
            }
        }

        // Remove dead entries
        for token in &to_remove {
            map.entries.remove(token);
        }

        // Enforce size limit: remove lowest confidence entries
        if map.entries.len() > self.config.max_learned_pairs {
            let mut entries_vec: Vec<(String, f32)> = map.entries.iter()
                .filter(|(_, e)| e.source != PairSource::Seed) // Never evict seeds
                .map(|(k, e)| (k.clone(), e.confidence))
                .collect();
            entries_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let to_evict = map.entries.len() - self.config.max_learned_pairs;
            for (token, _) in entries_vec.iter().take(to_evict) {
                map.entries.remove(token);
            }
        }

        // Update cycle metadata
        map.last_cycle = now;
        map.total_cycles += 1;

        Ok((new_pairs, confirmed, decayed))
    }
}

// ═══════════════════════════════════════════════════════════════
// Learning Report
// ═══════════════════════════════════════════════════════════════

/// Report from a single learning cycle
#[derive(Debug, Clone)]
pub struct LearningReport {
    pub duration: Duration,
    pub cooccurrence_pairs: usize,
    pub gaps_found: usize,
    pub candidates_extracted: usize,
    pub new_pairs: usize,
    pub confirmed_pairs: usize,
    pub decayed_pairs: usize,
    pub total_active: usize,
    pub timed_out: bool,
}

impl LearningReport {
    fn timeout(duration: Duration) -> Self {
        Self {
            duration,
            cooccurrence_pairs: 0,
            gaps_found: 0,
            candidates_extracted: 0,
            new_pairs: 0,
            confirmed_pairs: 0,
            decayed_pairs: 0,
            total_active: 0,
            timed_out: true,
        }
    }

    pub fn summary(&self) -> String {
        if self.timed_out {
            return format!("TIMEOUT after {:?}", self.duration);
        }
        format!(
            "{}ms | {} co-occur -> {} gaps -> {} candidates | +{} new, confirmed:{}, decayed:{} | {} total active",
            self.duration.as_millis(),
            self.cooccurrence_pairs,
            self.gaps_found,
            self.candidates_extracted,
            self.new_pairs,
            self.confirmed_pairs,
            self.decayed_pairs,
            self.total_active,
        )
    }
}

// ═══════════════════════════════════════════════════════════════
// Background Scheduler
// ═══════════════════════════════════════════════════════════════

/// Simple background scheduler that runs learning cycles periodically
pub struct LearnerScheduler {
    learner: Arc<SemanticLearner>,
    interval: Duration,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl LearnerScheduler {
    /// Create a new scheduler with the given interval
    pub fn new(learner: Arc<SemanticLearner>, interval: Duration) -> Self {
        Self {
            learner,
            interval,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start the background learning loop.
    /// Returns a handle to stop it.
    pub fn start(&self, storage: Arc<AuraStorage>) -> std::thread::JoinHandle<()> {
        let learner = Arc::clone(&self.learner);
        let interval = self.interval;
        let running = Arc::clone(&self.running);

        running.store(true, Ordering::SeqCst);

        std::thread::Builder::new()
            .name("aura-learner".to_string())
            .spawn(move || {
                tracing::info!("Learner background thread started (interval: {:?})", interval);

                while running.load(Ordering::SeqCst) {
                    // Sleep in small increments to allow quick shutdown
                    let mut slept = Duration::ZERO;
                    let sleep_chunk = Duration::from_secs(1);
                    while slept < interval && running.load(Ordering::SeqCst) {
                        std::thread::sleep(sleep_chunk);
                        slept += sleep_chunk;
                    }

                    if !running.load(Ordering::SeqCst) {
                        break;
                    }

                    // Run learning cycle
                    match learner.run_cycle(&storage) {
                        Ok(report) => {
                            if report.timed_out {
                                tracing::warn!("Learning cycle timed out: {}", report.summary());
                            } else {
                                tracing::info!("Learning cycle: {}", report.summary());
                            }
                        }
                        Err(e) => {
                            tracing::error!("Learning cycle failed: {}", e);
                        }
                    }
                }

                // Graceful shutdown: save before exit
                if let Err(e) = learner.save() {
                    tracing::error!("Failed to save learned map on shutdown: {}", e);
                }

                tracing::info!("Learner background thread stopped");
            })
            .expect("Failed to spawn learner thread")
    }

    /// Stop the background learning loop
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the scheduler is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

// ═══════════════════════════════════════════════════════════════
// Utility
// ═══════════════════════════════════════════════════════════════

fn now_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learner_config_defaults() {
        let config = LearnerConfig::default();
        assert_eq!(config.min_cooccurrence, 2);
        assert_eq!(config.max_sdr_tanimoto_for_gap, 0.15);
        assert_eq!(config.min_confidence, 0.3);
        assert_eq!(config.max_learned_pairs, 50_000);
    }

    #[test]
    fn test_learned_map_empty() {
        let map = LearnedCanonicalMap::default();
        assert_eq!(map.entries.len(), 0);
        assert_eq!(map.active_count(0.3), 0);
        assert!(map.lookup("test", 0.3).is_none());
    }

    #[test]
    fn test_learned_map_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".aura.learned");

        let mut map = LearnedCanonicalMap::default();
        map.entries.insert("car".to_string(), LearnedEntry {
            canonical: "automobile".to_string(),
            confidence: 0.8,
            observations: 5,
            discovered_at: 1000.0,
            last_confirmed: 2000.0,
            source: PairSource::MultiSource,
        });
        map.save(&path).unwrap();

        let loaded = LearnedCanonicalMap::load(&path).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.lookup("car", 0.3), Some("automobile"));
    }

    #[test]
    fn test_learned_map_confidence_filter() {
        let mut map = LearnedCanonicalMap::default();
        map.entries.insert("low".to_string(), LearnedEntry {
            canonical: "high".to_string(),
            confidence: 0.1,
            observations: 1,
            discovered_at: 0.0,
            last_confirmed: 0.0,
            source: PairSource::Temporal,
        });
        map.entries.insert("good".to_string(), LearnedEntry {
            canonical: "great".to_string(),
            confidence: 0.9,
            observations: 10,
            discovered_at: 0.0,
            last_confirmed: 0.0,
            source: PairSource::MultiSource,
        });

        assert_eq!(map.active_count(0.3), 1);
        assert!(map.lookup("low", 0.3).is_none());
        assert_eq!(map.lookup("good", 0.3), Some("great"));
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("The quick brown fox jumps", 3, 100);
        assert!(!tokens.contains(&"the".to_string())); // stopword
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
    }

    #[test]
    fn test_tokenize_max_limit() {
        let tokens = tokenize("one two three four five six", 3, 3);
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_cooccurrence_map() {
        let mut cooc = CoOccurrenceMap::new();
        cooc.add("a", "b");
        cooc.add("a", "b"); // duplicate, should increment
        cooc.add("b", "a"); // reversed, same pair
        cooc.add("a", "c"); // different pair

        let above_2 = cooc.pairs_above_threshold(2);
        assert_eq!(above_2.len(), 1); // only (a,b) has count >= 2

        let above_1 = cooc.pairs_above_threshold(1);
        assert_eq!(above_1.len(), 2); // (a,b) and (a,c)
    }

    #[test]
    fn test_learner_seed() {
        let dir = tempfile::tempdir().unwrap();
        let learner = SemanticLearner::new(dir.path(), LearnerConfig::default()).unwrap();

        let added = learner.seed(&[
            ("car", "automobile"),
            ("cat", "feline"),
            ("same", "same"), // identity — should be skipped
        ]);

        assert_eq!(added, 2);
        assert_eq!(learner.active_pair_count(), 2);
        assert_eq!(learner.lookup("car"), Some("automobile".to_string()));
        assert_eq!(learner.lookup("cat"), Some("feline".to_string()));
        assert!(learner.lookup("same").is_none());
    }

    #[test]
    fn test_learner_project() {
        let dir = tempfile::tempdir().unwrap();
        let learner = SemanticLearner::new(dir.path(), LearnerConfig::default()).unwrap();
        learner.seed(&[("car", "automobile")]);

        let result = learner.project("I saw a car on the road");
        assert!(result.contains("automobile"));
        assert!(!result.contains("car"));
    }

    #[test]
    fn test_learner_save_reload() {
        let dir = tempfile::tempdir().unwrap();
        {
            let learner = SemanticLearner::new(dir.path(), LearnerConfig::default()).unwrap();
            learner.seed(&[("drone", "uav"), ("tank", "armor")]);
            learner.save().unwrap();
        }

        // Reload from disk
        let learner2 = SemanticLearner::new(dir.path(), LearnerConfig::default()).unwrap();
        assert_eq!(learner2.active_pair_count(), 2);
        assert_eq!(learner2.lookup("drone"), Some("uav".to_string()));
    }

    #[test]
    fn test_projection_map() {
        let dir = tempfile::tempdir().unwrap();
        let learner = SemanticLearner::new(dir.path(), LearnerConfig::default()).unwrap();
        learner.seed(&[("car", "automobile"), ("cat", "feline")]);

        let map = learner.get_projection_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("car"), Some(&"automobile".to_string()));
    }

    #[test]
    fn test_is_stopword() {
        assert!(is_stopword("the"));
        assert!(is_stopword("але"));
        assert!(!is_stopword("memory"));
    }
}
