//! Epistemic Belief Layer — aggregates records into competing hypotheses.
//!
//! This module implements the second tier of the cognitive hierarchy:
//!   Record → **Belief** → Concept → Causal Pattern → Policy
//!
//! A Belief groups records that address the same underlying claim into
//! competing Hypotheses, scores them, and determines a winner or marks
//! the belief as unresolved.
//!
//! Currently operates in **read-only mode**: it builds and updates beliefs
//! during maintenance but does NOT influence recall ranking or policy.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::record::Record;
use crate::sdr::SDRInterpreter;

/// SDR lookup table: record_id → sparse SDR indices.
/// Passed into the belief engine so it can do content-aware grouping
/// without directly depending on AuraStorage.
pub type SdrLookup = HashMap<String, Vec<u16>>;

// ── Constants ──

/// Conflict penalty weight in hypothesis scoring.
const LAMBDA: f32 = 0.35;

/// Recency time constant (days). Controls how fast recency decays.
const TAU_DAYS: f64 = 14.0;

/// Belief revision threshold — opposing must exceed current by this factor.
const REVISION_THRESHOLD: f32 = 1.15;

/// Uncertainty band — if top two scores are within this range, belief is unresolved.
const UNCERTAINTY_BAND: f32 = 0.10;

/// Maximum SDR Tanimoto distance to **split** records within a coarse
/// tag-group into separate beliefs. Records with Tanimoto ≥ this
/// threshold are considered to address the same claim.
///
/// Empirically calibrated against n-gram SDR:
///   - near-identical text: ~0.85
///   - same-topic paraphrase: 0.30–0.66
///   - different topic: 0.05–0.10
///
/// A threshold of 0.15 safely separates genuinely different topics
/// while keeping paraphrases together.
const CLAIM_SIMILARITY_THRESHOLD: f32 = 0.15;
/// Minimum Tanimoto overlap between tag SDR fingerprints in `SdrTagPool`.
/// Slightly lower than content threshold because tag strings are much shorter.
const TAG_FINGERPRINT_SIMILARITY_THRESHOLD: f32 = 0.08;

/// Maximum records per hypothesis before pruning weakest.
/// Reserved for large-scale belief groups.
#[allow(dead_code)]
const MAX_RECORDS_PER_HYPOTHESIS: usize = 50;

// ── Coarse Key Mode ──

/// Controls how the coarse belief key is constructed before SDR subclustering.
///
/// The default (`Standard`) uses `namespace:sorted_tags(top3):semantic_type`.
/// Alternative modes reduce tag granularity to increase belief density
/// per partition, allowing SDR subclustering to do more of the work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoarseKeyMode {
    /// `namespace:sorted_tags(top3):semantic_type` — original behavior.
    Standard,
    /// `namespace:top_1_tag:semantic_type` — only the first sorted tag.
    TopOneTag,
    /// `namespace:semantic_type` — no tags; SDR does all fine grouping.
    SemanticOnly,
    /// Variant A: `namespace:dominant_tag_family:semantic_type`.
    /// Tag family = alphabetically first tag. Reduces fragmentation while
    /// keeping some topic boundary signal.
    TagFamily,
    /// Variant A1: `TagFamily` with a guarded softer SDR threshold.
    /// Keeps the same coarse corridor as `TagFamily`, but only lowers the SDR
    /// split threshold for non-generic families if the strict baseline pass
    /// would otherwise collapse the whole corridor into singleton clusters.
    TagFamilyAdaptive,
    /// Variant A2: `TagFamily` with guarded coarse fallback.
    /// Runs normal SDR subclustering first, but if an entire coarse corridor
    /// collapses into singleton subclusters, retains the original coarse group
    /// as one belief candidate instead of dropping belief creation to zero.
    TagFamilyBackoff,
    /// Variant A3: `TagFamilyBackoff` but with a richer family fingerprint.
    /// Uses the first two sorted tags instead of only the first tag for the
    /// coarse corridor, reducing same-family over-merges on dense synthetic
    /// corpora while keeping the same guarded coarse fallback semantics.
    TagFamilyPairBackoff,
    /// Variant A4: `TagFamilyBackoff` with dense-corridor local refinement.
    /// Starts from the normal dominant-family corridor, but if a corridor is
    /// dense enough and contains stable secondary tags, records are regrouped
    /// into bounded corridor-local keys of `family + stable_secondary_tag`.
    /// This is narrower than `TagFamilyPairBackoff` because it only applies
    /// inside dense same-family corridors and only uses corridor-stable tags.
    TagFamilyDenseBackoff,
    /// Variant B: `namespace:semantic_type` corridor with tag-guarded SDR subclustering.
    /// Broad pool like SemanticOnly, but sdr_subcluster requires shared_tags >= 1
    /// before merging, preventing cross-topic false merges.
    DualKey,
    /// Variant C: `namespace:semantic_type` corridor with neighborhood formation.
    /// Uses tag overlap + SDR similarity jointly: records must share >= 1 tag AND
    /// have Tanimoto >= threshold to merge. Like DualKey but with relaxed threshold (0.08).
    NeighborhoodPool,
    /// Variant D: `namespace:normalized_tags(top3):semantic_type`.
    /// Uses a tiny deterministic bridge table to collapse a few safe near-synonym
    /// tag families before SDR subclustering.
    BridgeKey,
    /// Variant E: `namespace:semantic_type` corridor with SDR tag fingerprint guard.
    /// Records may only merge if their tag-SDR fingerprints overlap above a
    /// threshold before normal content SDR subclustering is allowed.
    SdrTagPool,
}

impl Default for CoarseKeyMode {
    fn default() -> Self {
        CoarseKeyMode::Standard
    }
}

// ── Belief State ──

/// Resolution state of a belief.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefState {
    /// One hypothesis clearly dominates.
    Resolved,
    /// Top hypotheses are too close to call.
    Unresolved,
    /// Only one hypothesis exists (no competition).
    Singleton,
    /// No hypotheses yet.
    Empty,
}

// ── Hypothesis ──

/// A single hypothesis within a belief — one possible "truth" for a claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    /// Unique hypothesis ID.
    pub id: String,
    /// Parent belief ID.
    pub belief_id: String,
    /// Record IDs that support this hypothesis (prototype records).
    pub prototype_record_ids: Vec<String>,
    /// Composite score (higher = stronger hypothesis).
    pub score: f32,
    /// Weighted average confidence of supporting records.
    pub confidence: f32,
    /// Aggregated support mass.
    pub support_mass: f32,
    /// Aggregated conflict mass.
    pub conflict_mass: f32,
    /// Recency factor — exponential decay from most recent record.
    pub recency: f32,
    /// Internal consistency — 1/(1 + variance of record confidences).
    pub consistency: f32,
}

impl Hypothesis {
    /// Compute a deterministic hypothesis ID from belief_id + sorted record IDs.
    ///
    /// Uses xxh3 (stable across Rust versions and platforms) so that the same
    /// set of records always produces the same hypothesis ID — critical for
    /// stability tracking across maintenance cycles and toolchain upgrades.
    fn deterministic_id(belief_id: &str, records: &[&Record]) -> String {
        let mut ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
        ids.sort();
        // Build a canonical byte string: "belief_id\0id1\0id2\0..."
        let mut buf = belief_id.to_string();
        for id in &ids {
            buf.push('\0');
            buf.push_str(id);
        }
        let hash = xxhash_rust::xxh3::xxh3_64(buf.as_bytes());
        format!("{:012x}", hash)
    }

    /// Create a new hypothesis from a set of records.
    pub fn from_records(belief_id: &str, records: &[&Record]) -> Self {
        let id = Self::deterministic_id(belief_id, records);
        let now = now_secs();

        let mut confidence_sum = 0.0_f32;
        let mut support_sum = 0.0_f32;
        let mut conflict_sum = 0.0_f32;
        let mut most_recent = 0.0_f64;
        let mut confidences = Vec::with_capacity(records.len());

        for rec in records {
            confidence_sum += rec.confidence;
            support_sum += rec.support_mass as f32;
            conflict_sum += rec.conflict_mass as f32;
            if rec.last_activated > most_recent {
                most_recent = rec.last_activated;
            }
            confidences.push(rec.confidence);
        }

        let n = records.len().max(1) as f32;
        let confidence = confidence_sum / n;
        let support_mass = support_sum;
        let conflict_mass = conflict_sum;

        // Recency = exp(-age_days / tau)
        let age_days = ((now - most_recent) / 86400.0).max(0.0);
        let recency = (-age_days / TAU_DAYS).exp() as f32;

        // Consistency = 1 / (1 + variance(confidences))
        let consistency = if confidences.len() > 1 {
            let mean = confidence;
            let variance: f32 = confidences.iter().map(|c| (c - mean).powi(2)).sum::<f32>()
                / (confidences.len() - 1) as f32;
            1.0 / (1.0 + variance)
        } else {
            1.0
        };

        let mut h = Self {
            id,
            belief_id: belief_id.to_string(),
            prototype_record_ids: records.iter().map(|r| r.id.clone()).collect(),
            score: 0.0,
            confidence,
            support_mass,
            conflict_mass,
            recency,
            consistency,
        };
        h.score = h.compute_score();
        h
    }

    /// Compute the final hypothesis score.
    ///
    /// `final_score = (base + support_ln) * confidence * recency * consistency - λ * conflict_penalty`
    ///
    /// The base term (1.0) ensures that even a hypothesis with zero support_mass
    /// (fresh records, no neighbors yet) gets a non-zero score from its confidence
    /// and recency alone.
    pub fn compute_score(&self) -> f32 {
        let support_score = 1.0 + (1.0 + self.support_mass).ln();
        let conflict_penalty = (1.0 + self.conflict_mass).ln();
        let belief_score = support_score * self.confidence * self.recency * self.consistency;
        (belief_score - LAMBDA * conflict_penalty).max(0.0)
    }

    /// Refresh score from current field values.
    pub fn refresh_score(&mut self) {
        self.score = self.compute_score();
    }
}

// ── Belief ──

/// An aggregated epistemic position on a claim.
///
/// Groups multiple hypotheses that compete to explain the same underlying
/// statement. The winner is the hypothesis with the highest score, unless
/// the top two are within the uncertainty band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    /// Unique belief ID.
    pub id: String,
    /// Canonical key — identifies the claim this belief addresses.
    /// Composed from: namespace + dominant tags + semantic cluster fingerprint.
    pub key: String,
    /// Hypothesis IDs belonging to this belief.
    pub hypothesis_ids: Vec<String>,
    /// Current winning hypothesis ID (if resolved).
    pub winner_id: Option<String>,
    /// Current belief state.
    pub state: BeliefState,
    /// Best hypothesis score.
    pub score: f32,
    /// Weighted confidence across hypotheses.
    pub confidence: f32,
    /// Total support mass across all hypotheses.
    pub support_mass: f32,
    /// Total conflict mass across all hypotheses.
    pub conflict_mass: f32,
    /// Stability — how many cycles the winner has remained the same.
    pub stability: f32,
    /// Unix timestamp of last update.
    pub last_updated: f64,
}

impl Belief {
    /// Create a new empty belief for a given claim key.
    pub fn new(key: String) -> Self {
        Self {
            id: Record::generate_id(),
            key,
            hypothesis_ids: Vec::new(),
            winner_id: None,
            state: BeliefState::Empty,
            score: 0.0,
            confidence: 0.0,
            support_mass: 0.0,
            conflict_mass: 0.0,
            stability: 0.0,
            last_updated: now_secs(),
        }
    }

    /// Resolve winner from a set of hypotheses.
    ///
    /// Returns the previous winner ID (for stability tracking).
    pub fn resolve(&mut self, hypotheses: &[&Hypothesis]) -> Option<String> {
        let prev_winner = self.winner_id.clone();

        if hypotheses.is_empty() {
            self.state = BeliefState::Empty;
            self.winner_id = None;
            self.score = 0.0;
            return prev_winner;
        }

        if hypotheses.len() == 1 {
            let h = hypotheses[0];
            self.state = BeliefState::Singleton;
            self.winner_id = Some(h.id.clone());
            self.score = h.score;
            self.confidence = h.confidence;
            self.support_mass = h.support_mass;
            self.conflict_mass = h.conflict_mass;
            self.last_updated = now_secs();
            // Track stability
            if prev_winner.as_ref() == Some(&h.id) {
                self.stability += 1.0;
            } else {
                self.stability = 1.0;
            }
            return prev_winner;
        }

        // Find top-2 by score
        let mut sorted: Vec<&Hypothesis> = hypotheses.to_vec();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let top1 = sorted[0];
        let top2 = sorted[1];

        // Revision check
        let eps = 1e-6_f32;
        let ratio = top1.score / (top2.score + eps);

        if ratio < REVISION_THRESHOLD && (top1.score - top2.score).abs() < UNCERTAINTY_BAND {
            // Too close to call
            self.state = BeliefState::Unresolved;
            self.winner_id = None;
            self.stability = 0.0;
        } else {
            self.state = BeliefState::Resolved;
            self.winner_id = Some(top1.id.clone());
            if prev_winner.as_ref() == Some(&top1.id) {
                self.stability += 1.0;
            } else {
                self.stability = 1.0;
            }
        }

        // Aggregate stats from winner (or top1 if unresolved)
        self.score = top1.score;
        self.confidence = top1.confidence;
        self.support_mass = hypotheses.iter().map(|h| h.support_mass).sum();
        self.conflict_mass = hypotheses.iter().map(|h| h.conflict_mass).sum();
        self.last_updated = now_secs();

        prev_winner
    }
}

// ── Belief Engine ──

/// The belief engine — maintains the full belief state.
///
/// Currently read-only: builds beliefs from records during maintenance
/// but does not modify recall or agent behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefEngine {
    /// All beliefs, keyed by belief ID.
    pub beliefs: HashMap<String, Belief>,
    /// All hypotheses, keyed by hypothesis ID.
    pub hypotheses: HashMap<String, Hypothesis>,
    /// Index: claim key → belief ID.
    pub key_index: HashMap<String, String>,
    /// Index: record ID → hypothesis ID (which hypothesis owns this record).
    pub record_index: HashMap<String, String>,
    /// Coarse key construction mode (default: Standard).
    #[serde(default)]
    pub coarse_key_mode: CoarseKeyMode,
    /// Override for the SDR subclustering threshold (default: CLAIM_SIMILARITY_THRESHOLD = 0.15).
    /// Lower values allow more records to cluster together within a coarse group.
    #[serde(default)]
    pub claim_similarity_override: Option<f32>,
}

impl BeliefEngine {
    pub fn new() -> Self {
        Self {
            beliefs: HashMap::new(),
            hypotheses: HashMap::new(),
            key_index: HashMap::new(),
            record_index: HashMap::new(),
            coarse_key_mode: CoarseKeyMode::default(),
            claim_similarity_override: None,
        }
    }

    /// Create a new engine with the specified coarse key mode.
    pub fn with_coarse_key_mode(mode: CoarseKeyMode) -> Self {
        let mut engine = Self::new();
        engine.coarse_key_mode = mode;
        engine
    }

    /// Effective SDR subclustering threshold.
    fn effective_similarity_threshold(&self) -> f32 {
        self.claim_similarity_override
            .unwrap_or(CLAIM_SIMILARITY_THRESHOLD)
    }

    /// Build a coarse claim key from a record (tag-group level).
    ///
    /// Key = namespace + sorted dominant tags (top 3) + semantic_type.
    /// This produces the **coarse group**; within each group, SDR similarity
    /// further splits records into sub-clusters (fine grouping).
    pub fn claim_key(record: &Record) -> String {
        Self::claim_key_with_mode(record, CoarseKeyMode::Standard)
    }

    /// Build a coarse claim key using the specified mode.
    pub fn claim_key_with_mode(record: &Record, mode: CoarseKeyMode) -> String {
        match mode {
            CoarseKeyMode::Standard => {
                let mut tags: Vec<&str> = record.tags.iter().map(|s| s.as_str()).collect();
                tags.sort();
                tags.truncate(3);
                format!(
                    "{}:{}:{}",
                    record.namespace,
                    tags.join(","),
                    record.semantic_type
                )
            }
            CoarseKeyMode::TopOneTag => {
                let mut tags: Vec<&str> = record.tags.iter().map(|s| s.as_str()).collect();
                tags.sort();
                tags.truncate(1);
                format!(
                    "{}:{}:{}",
                    record.namespace,
                    tags.join(","),
                    record.semantic_type
                )
            }
            CoarseKeyMode::SemanticOnly => {
                format!("{}:{}", record.namespace, record.semantic_type)
            }
            CoarseKeyMode::TagFamily
            | CoarseKeyMode::TagFamilyAdaptive
            | CoarseKeyMode::TagFamilyBackoff
            | CoarseKeyMode::TagFamilyDenseBackoff => {
                // Dominant tag family = alphabetically first tag
                let mut tags: Vec<&str> = record.tags.iter().map(|s| s.as_str()).collect();
                tags.sort();
                let family = tags.first().copied().unwrap_or("");
                format!("{}:{}:{}", record.namespace, family, record.semantic_type)
            }
            CoarseKeyMode::TagFamilyPairBackoff => {
                let mut tags: Vec<&str> = record.tags.iter().map(|s| s.as_str()).collect();
                tags.sort();
                tags.truncate(2);
                format!(
                    "{}:{}:{}",
                    record.namespace,
                    tags.join(","),
                    record.semantic_type
                )
            }
            CoarseKeyMode::DualKey | CoarseKeyMode::NeighborhoodPool => {
                // Broad corridor: namespace + semantic_type only
                // Fine grouping handled by tag-guarded sdr_subcluster
                format!("{}:{}", record.namespace, record.semantic_type)
            }
            CoarseKeyMode::SdrTagPool => {
                // Broad corridor: namespace + semantic_type only
                // Fine grouping handled by tag-SDR guard + content SDR subclustering
                format!("{}:{}", record.namespace, record.semantic_type)
            }
            CoarseKeyMode::BridgeKey => {
                let tags = Self::normalized_bridge_tags(record);
                format!(
                    "{}:{}:{}",
                    record.namespace,
                    tags.join(","),
                    record.semantic_type
                )
            }
        }
    }

    /// Minimal deterministic tag bridge table for safe densification experiments.
    fn normalize_bridge_tag(tag: &str) -> String {
        let lower = tag.trim().to_ascii_lowercase();
        match lower.as_str() {
            "ui" | "frontend" => "frontend".to_string(),
            "auth" | "authentication" => "authentication".to_string(),
            "deploy" | "release" => "release".to_string(),
            _ => lower,
        }
    }

    fn normalized_bridge_tags(record: &Record) -> Vec<String> {
        let mut tags: Vec<String> = record
            .tags
            .iter()
            .map(|t| Self::normalize_bridge_tag(t))
            .collect();
        tags.sort();
        tags.dedup();
        tags.truncate(3);
        tags
    }

    fn canonical_tag_text(record: &Record) -> String {
        let mut tags: Vec<String> = record
            .tags
            .iter()
            .map(|tag| tag.trim().to_ascii_lowercase())
            .filter(|tag| !tag.is_empty())
            .collect();
        tags.sort();
        tags.dedup();
        tags.join(" ")
    }

    fn parse_tag_family_from_key(key: &str) -> &str {
        let mut parts = key.split(':');
        let _ns = parts.next();
        parts.next().unwrap_or("")
    }

    fn is_generic_tag_family(family: &str) -> bool {
        matches!(family, "alerts")
    }

    fn dense_corridor_stable_tags(records: &[&Record], family: &str) -> Vec<String> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for rec in records {
            let mut tags: Vec<String> = rec
                .tags
                .iter()
                .map(|t| t.trim().to_ascii_lowercase())
                .filter(|t| !t.is_empty())
                .collect();
            tags.sort();
            tags.dedup();
            for tag in tags {
                if Self::is_generic_tag_family(&tag) {
                    continue;
                }
                *counts.entry(tag).or_default() += 1;
            }
        }

        let mut stable: Vec<String> = counts
            .into_iter()
            .filter_map(|(tag, count)| {
                if tag == family {
                    Some(tag)
                } else if count >= 4 {
                    Some(tag)
                } else {
                    None
                }
            })
            .collect();
        stable.sort();
        if !stable.iter().any(|tag| tag == family) {
            stable.insert(0, family.to_string());
        }
        stable
    }

    fn dense_backoff_group_key(coarse_key: &str, records: &[&Record], record: &Record) -> String {
        let family = Self::parse_tag_family_from_key(coarse_key);
        let stable_tags = Self::dense_corridor_stable_tags(records, family);
        if records.len() < 4 || stable_tags.len() < 2 {
            return coarse_key.to_string();
        }

        let mut record_tags: Vec<String> = record
            .tags
            .iter()
            .map(|t| t.trim().to_ascii_lowercase())
            .filter(|t| !t.is_empty())
            .collect();
        record_tags.sort();
        record_tags.dedup();

        let mut picked: Vec<String> = stable_tags
            .into_iter()
            .filter(|tag| record_tags.iter().any(|t| t == tag))
            .collect();
        picked.sort();
        picked.dedup();
        picked.truncate(3);

        if picked.len() >= 2 {
            format!(
                "{}:{}:{}",
                record.namespace,
                picked.join(","),
                record.semantic_type
            )
        } else {
            coarse_key.to_string()
        }
    }

    /// Tanimoto coefficient for two sorted sparse SDR vectors.
    /// Duplicated from SDRInterpreter to avoid coupling belief.rs to sdr.rs.
    fn tanimoto(a: &[u16], b: &[u16]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let mut intersection = 0usize;
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                i += 1;
            } else if a[i] > b[j] {
                j += 1;
            } else {
                intersection += 1;
                i += 1;
                j += 1;
            }
        }
        let union = a.len() + b.len() - intersection;
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Split a coarse tag-group into SDR sub-clusters.
    ///
    /// Uses single-linkage clustering: two records belong to the same
    /// cluster if their Tanimoto similarity ≥ `CLAIM_SIMILARITY_THRESHOLD`.
    /// Records without SDR data form singleton clusters.
    ///
    /// Returns a list of clusters, each cluster being a Vec of record refs.
    fn sdr_subcluster<'a>(
        records: &[&'a Record],
        sdr_lookup: &SdrLookup,
        threshold: f32,
    ) -> Vec<Vec<&'a Record>> {
        let n = records.len();
        if n <= 1 {
            return vec![records.to_vec()];
        }

        // Union-Find for clustering
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path compression
                x = parent[x];
            }
            x
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        // Pairwise Tanimoto comparison
        for i in 0..n {
            let sdr_i = sdr_lookup.get(&records[i].id);
            if sdr_i.is_none() {
                continue;
            }
            let sdr_i = sdr_i.unwrap();

            for j in (i + 1)..n {
                let sdr_j = sdr_lookup.get(&records[j].id);
                if sdr_j.is_none() {
                    continue;
                }
                let sdr_j = sdr_j.unwrap();

                if Self::tanimoto(sdr_i, sdr_j) >= threshold {
                    union(&mut parent, i, j);
                }
            }
        }

        // Collect clusters
        let mut clusters: HashMap<usize, Vec<&'a Record>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(records[i]);
        }
        clusters.into_values().collect()
    }

    /// Tag-guarded SDR subclustering for DualKey/NeighborhoodPool modes.
    ///
    /// Like `sdr_subcluster`, but additionally requires that two records share
    /// at least 1 tag before they can be merged. This prevents cross-topic
    /// false merges when using a broad coarse key (namespace:semantic_type).
    fn sdr_subcluster_tag_guarded<'a>(
        records: &[&'a Record],
        sdr_lookup: &SdrLookup,
        threshold: f32,
    ) -> Vec<Vec<&'a Record>> {
        let n = records.len();
        if n <= 1 {
            return vec![records.to_vec()];
        }

        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        for i in 0..n {
            let sdr_i = sdr_lookup.get(&records[i].id);
            if sdr_i.is_none() {
                continue;
            }
            let sdr_i = sdr_i.unwrap();

            let tags_i: std::collections::HashSet<&str> =
                records[i].tags.iter().map(|s| s.as_str()).collect();

            for j in (i + 1)..n {
                // Tag barrier: require shared tags >= 1
                let shared = records[j].tags.iter().any(|t| tags_i.contains(t.as_str()));
                if !shared && !tags_i.is_empty() && !records[j].tags.is_empty() {
                    continue;
                }

                let sdr_j = sdr_lookup.get(&records[j].id);
                if sdr_j.is_none() {
                    continue;
                }
                let sdr_j = sdr_j.unwrap();

                if Self::tanimoto(sdr_i, sdr_j) >= threshold {
                    union(&mut parent, i, j);
                }
            }
        }

        let mut clusters: HashMap<usize, Vec<&'a Record>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(records[i]);
        }
        clusters.into_values().collect()
    }

    /// Bridge-tag-guarded SDR subclustering for BridgeKey mode.
    ///
    /// Records may merge only if they share at least one normalized bridge tag
    /// family and pass the SDR threshold. This keeps corridor widening local and
    /// deterministic.
    fn sdr_subcluster_bridge_guarded<'a>(
        records: &[&'a Record],
        sdr_lookup: &SdrLookup,
        threshold: f32,
    ) -> Vec<Vec<&'a Record>> {
        let n = records.len();
        if n <= 1 {
            return vec![records.to_vec()];
        }

        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        let bridge_sets: Vec<HashSet<String>> = records
            .iter()
            .map(|rec| Self::normalized_bridge_tags(rec).into_iter().collect())
            .collect();

        for i in 0..n {
            let sdr_i = sdr_lookup.get(&records[i].id);
            if sdr_i.is_none() {
                continue;
            }
            let sdr_i = sdr_i.unwrap();

            for j in (i + 1)..n {
                let shared_bridge = bridge_sets[i]
                    .iter()
                    .any(|tag| bridge_sets[j].contains(tag));
                if !shared_bridge {
                    continue;
                }

                let sdr_j = sdr_lookup.get(&records[j].id);
                if sdr_j.is_none() {
                    continue;
                }
                let sdr_j = sdr_j.unwrap();

                if Self::tanimoto(sdr_i, sdr_j) >= threshold {
                    union(&mut parent, i, j);
                }
            }
        }

        let mut clusters: HashMap<usize, Vec<&'a Record>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(records[i]);
        }
        clusters.into_values().collect()
    }

    /// SDR-tag-guarded subclustering for `SdrTagPool`.
    ///
    /// Uses two deterministic guards:
    /// 1. tag fingerprint overlap must exceed the tag threshold
    /// 2. content SDR overlap must exceed the normal claim threshold
    fn sdr_subcluster_tag_sdr_guarded<'a>(
        records: &[&'a Record],
        sdr_lookup: &SdrLookup,
        threshold: f32,
    ) -> Vec<Vec<&'a Record>> {
        let n = records.len();
        if n <= 1 {
            return vec![records.to_vec()];
        }

        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        let tag_sdr = SDRInterpreter::default();
        let tag_fingerprints: Vec<Vec<u16>> = records
            .iter()
            .map(|rec| {
                let tag_text = Self::canonical_tag_text(rec);
                if tag_text.is_empty() {
                    Vec::new()
                } else {
                    tag_sdr.text_to_sdr_lowered(&tag_text, false)
                }
            })
            .collect();

        for i in 0..n {
            let sdr_i = sdr_lookup.get(&records[i].id);
            if sdr_i.is_none() {
                continue;
            }
            let sdr_i = sdr_i.unwrap();

            for j in (i + 1)..n {
                if tag_fingerprints[i].is_empty() || tag_fingerprints[j].is_empty() {
                    continue;
                }

                let tag_similarity = Self::tanimoto(&tag_fingerprints[i], &tag_fingerprints[j]);
                if tag_similarity < TAG_FINGERPRINT_SIMILARITY_THRESHOLD {
                    continue;
                }

                let sdr_j = sdr_lookup.get(&records[j].id);
                if sdr_j.is_none() {
                    continue;
                }
                let sdr_j = sdr_j.unwrap();

                if Self::tanimoto(sdr_i, sdr_j) >= threshold {
                    union(&mut parent, i, j);
                }
            }
        }

        let mut clusters: HashMap<usize, Vec<&'a Record>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(records[i]);
        }
        clusters.into_values().collect()
    }

    /// Run a full belief update cycle over all records (without SDR data).
    ///
    /// Fallback path: groups by coarse claim key only.
    /// Prefer `update_with_sdr()` for accurate claim grouping.
    pub fn update(&mut self, records: &HashMap<String, Record>) -> BeliefReport {
        self.update_with_sdr(records, &HashMap::new())
    }

    /// Run a full belief update cycle with SDR-backed claim grouping.
    ///
    /// This is the primary entry point called from maintenance Phase 3.5.
    /// It:
    /// 1. Groups records by coarse claim key (namespace + tags + semantic_type)
    /// 2. Within each coarse group, splits into sub-clusters by SDR Tanimoto ≥ 0.60
    /// 3. For each sub-cluster, builds/updates hypotheses and resolves beliefs
    ///
    /// Returns a `BeliefReport` with stats.
    pub fn update_with_sdr(
        &mut self,
        records: &HashMap<String, Record>,
        sdr_lookup: &SdrLookup,
    ) -> BeliefReport {
        let mut report = BeliefReport::default();
        let has_sdr = !sdr_lookup.is_empty();

        // Step 1: Coarse grouping by tag key
        let mut coarse_groups: HashMap<String, Vec<&Record>> = HashMap::new();
        for rec in records.values() {
            if !rec.is_alive() {
                continue;
            }
            // Skip records with trivial content
            if rec.content.len() < 10 {
                continue;
            }
            let key = Self::claim_key_with_mode(rec, self.coarse_key_mode);
            // Skip empty keys (no tags) — only applies to tag-based modes
            let uses_tags = matches!(
                self.coarse_key_mode,
                CoarseKeyMode::Standard
                    | CoarseKeyMode::TopOneTag
                    | CoarseKeyMode::TagFamily
                    | CoarseKeyMode::TagFamilyAdaptive
                    | CoarseKeyMode::TagFamilyBackoff
                    | CoarseKeyMode::TagFamilyDenseBackoff
                    | CoarseKeyMode::TagFamilyPairBackoff
                    | CoarseKeyMode::BridgeKey
            );
            if uses_tags && key.contains("::") {
                continue;
            }
            coarse_groups.entry(key).or_default().push(rec);
        }

        if self.coarse_key_mode == CoarseKeyMode::TagFamilyDenseBackoff {
            let mut refined_groups: HashMap<String, Vec<&Record>> = HashMap::new();
            for (coarse_key, group_records) in coarse_groups {
                let family = Self::parse_tag_family_from_key(&coarse_key);
                if group_records.len() >= 4
                    && !Self::is_generic_tag_family(family)
                    && Self::dense_corridor_stable_tags(&group_records, family).len() >= 2
                {
                    for rec in &group_records {
                        let refined_key =
                            Self::dense_backoff_group_key(&coarse_key, &group_records, rec);
                        refined_groups.entry(refined_key).or_default().push(rec);
                    }
                } else {
                    refined_groups.insert(coarse_key, group_records);
                }
            }
            coarse_groups = refined_groups;
        }

        // Step 2: Fine grouping — split each coarse group by SDR similarity
        let mut groups: HashMap<String, Vec<&Record>> = HashMap::new();
        for (coarse_key, group_records) in &coarse_groups {
            if !has_sdr || group_records.len() < 2 {
                // No SDR data or single record → keep as-is
                groups.insert(coarse_key.clone(), group_records.clone());
                continue;
            }

            // Choose subclustering strategy based on mode
            let threshold = match self.coarse_key_mode {
                // NeighborhoodPool uses relaxed threshold (0.08) unless overridden
                CoarseKeyMode::NeighborhoodPool => self.claim_similarity_override.unwrap_or(0.08),
                // TagFamilyAdaptive starts with the strict baseline threshold.
                // A second, softer pass may be allowed below only if the strict
                // pass would leave the entire corridor fragmented into singletons.
                CoarseKeyMode::TagFamilyAdaptive => self
                    .claim_similarity_override
                    .unwrap_or(CLAIM_SIMILARITY_THRESHOLD),
                // DualKey uses lowered threshold (0.10) unless overridden
                CoarseKeyMode::DualKey => self.claim_similarity_override.unwrap_or(0.10),
                // SdrTagPool uses a broad corridor, so keep a slightly stricter
                // content threshold than Neighborhood while letting tag-SDR
                // decide whether two records are even comparable.
                CoarseKeyMode::SdrTagPool => self.claim_similarity_override.unwrap_or(0.10),
                _ => self.effective_similarity_threshold(),
            };

            let use_tag_guard = matches!(
                self.coarse_key_mode,
                CoarseKeyMode::DualKey | CoarseKeyMode::NeighborhoodPool
            );
            let use_bridge_guard = self.coarse_key_mode == CoarseKeyMode::BridgeKey;
            let use_tag_sdr_guard = self.coarse_key_mode == CoarseKeyMode::SdrTagPool;

            let subclusters = if use_tag_guard {
                Self::sdr_subcluster_tag_guarded(group_records, sdr_lookup, threshold)
            } else if use_bridge_guard {
                Self::sdr_subcluster_bridge_guarded(group_records, sdr_lookup, threshold)
            } else if use_tag_sdr_guard {
                Self::sdr_subcluster_tag_sdr_guarded(group_records, sdr_lookup, threshold)
            } else {
                Self::sdr_subcluster(group_records, sdr_lookup, threshold)
            };

            let subclusters = if self.coarse_key_mode == CoarseKeyMode::TagFamilyAdaptive {
                let family = Self::parse_tag_family_from_key(coarse_key);
                let strict_all_singletons = subclusters.iter().all(|cluster| cluster.len() < 2);
                if group_records.len() >= 2
                    && strict_all_singletons
                    && !Self::is_generic_tag_family(family)
                {
                    let adaptive_threshold = self.claim_similarity_override.unwrap_or(0.10);
                    Self::sdr_subcluster(group_records, sdr_lookup, adaptive_threshold)
                } else {
                    subclusters
                }
            } else {
                subclusters
            };

            let use_tagfamily_backoff = matches!(
                self.coarse_key_mode,
                CoarseKeyMode::TagFamilyBackoff
                    | CoarseKeyMode::TagFamilyPairBackoff
                    | CoarseKeyMode::TagFamilyDenseBackoff
            );
            if use_tagfamily_backoff
                && group_records.len() >= 2
                && !Self::is_generic_tag_family(Self::parse_tag_family_from_key(coarse_key))
                && subclusters.iter().all(|cluster| cluster.len() < 2)
            {
                groups.insert(coarse_key.clone(), group_records.clone());
                continue;
            }

            if subclusters.len() == 1 {
                // All records are similar enough — single group
                groups.insert(coarse_key.clone(), group_records.clone());
            } else {
                // Multiple sub-clusters — give each a unique key
                for (idx, cluster) in subclusters.iter().enumerate() {
                    let sub_key = format!("{}#{}", coarse_key, idx);
                    groups.insert(sub_key, cluster.clone());
                }
            }
        }

        // Step 2: For each group, build/update hypotheses and belief
        for (key, group_records) in &groups {
            if group_records.len() < 2 {
                // Not enough records for a meaningful belief
                continue;
            }

            // Get or create belief
            let belief_id = self.key_index.get(key).cloned().unwrap_or_else(|| {
                let b = Belief::new(key.clone());
                let bid = b.id.clone();
                self.key_index.insert(key.clone(), bid.clone());
                self.beliefs.insert(bid.clone(), b);
                report.beliefs_created += 1;
                bid
            });

            // Cluster records into hypotheses.
            // Simple strategy: split by whether records have contradictions.
            let (supporting, opposing) = Self::split_by_contradiction(group_records);

            let mut hyp_refs = Vec::new();

            // Build supporting hypothesis
            if !supporting.is_empty() {
                let h = Hypothesis::from_records(&belief_id, &supporting);
                for rid in &h.prototype_record_ids {
                    self.record_index.insert(rid.clone(), h.id.clone());
                }
                let hid = h.id.clone();
                self.hypotheses.insert(hid.clone(), h);
                hyp_refs.push(hid);
            }

            // Build opposing hypothesis (if any contradictions)
            if !opposing.is_empty() {
                let h = Hypothesis::from_records(&belief_id, &opposing);
                for rid in &h.prototype_record_ids {
                    self.record_index.insert(rid.clone(), h.id.clone());
                }
                let hid = h.id.clone();
                self.hypotheses.insert(hid.clone(), h);
                hyp_refs.push(hid);
            }

            // Resolve belief
            if let Some(belief) = self.beliefs.get_mut(&belief_id) {
                // Clean up old hypothesis IDs that are NOT reused this cycle
                let new_set: std::collections::HashSet<&String> = hyp_refs.iter().collect();
                for old_hid in &belief.hypothesis_ids {
                    if !new_set.contains(old_hid) {
                        self.hypotheses.remove(old_hid);
                    }
                }
                belief.hypothesis_ids = hyp_refs.clone();

                let hyps: Vec<&Hypothesis> = hyp_refs
                    .iter()
                    .filter_map(|hid| self.hypotheses.get(hid))
                    .collect();

                let prev_winner = belief.resolve(&hyps);

                // Track revisions
                if let Some(ref prev) = prev_winner {
                    if belief.winner_id.as_ref() != Some(prev) {
                        report.revisions += 1;
                    }
                }

                match belief.state {
                    BeliefState::Resolved | BeliefState::Singleton => report.resolved += 1,
                    BeliefState::Unresolved => report.unresolved += 1,
                    BeliefState::Empty => {}
                }
            }
        }

        // Prune beliefs for groups that no longer exist
        let active_keys: std::collections::HashSet<&String> = groups.keys().collect();
        let stale_keys: Vec<String> = self
            .key_index
            .keys()
            .filter(|k| !active_keys.contains(k))
            .cloned()
            .collect();
        for key in stale_keys {
            if let Some(bid) = self.key_index.remove(&key) {
                if let Some(belief) = self.beliefs.remove(&bid) {
                    for hid in &belief.hypothesis_ids {
                        self.hypotheses.remove(hid);
                    }
                    report.beliefs_pruned += 1;
                }
            }
        }

        // Prune stale record_index entries — only keep records that belong
        // to a live hypothesis.
        let live_record_ids: std::collections::HashSet<String> = self
            .hypotheses
            .values()
            .flat_map(|h| h.prototype_record_ids.iter().cloned())
            .collect();
        self.record_index
            .retain(|rid, _| live_record_ids.contains(rid));

        report.total_beliefs = self.beliefs.len();
        report.total_hypotheses = self.hypotheses.len();
        report.churn_rate = report.revisions as f32 / report.total_beliefs.max(1) as f32;
        report
    }

    /// Split records into (supporting, opposing) based on contradiction markers.
    ///
    /// Records with semantic_type="contradiction" or conflict_mass > support_mass
    /// go into the opposing group.
    fn split_by_contradiction<'a>(records: &[&'a Record]) -> (Vec<&'a Record>, Vec<&'a Record>) {
        let mut supporting = Vec::new();
        let mut opposing = Vec::new();

        for rec in records {
            if rec.semantic_type == "contradiction" || rec.conflict_mass > rec.support_mass {
                opposing.push(*rec);
            } else {
                supporting.push(*rec);
            }
        }

        (supporting, opposing)
    }

    /// Get belief for a record (if it participates in one).
    pub fn belief_for_record(&self, record_id: &str) -> Option<&Belief> {
        let hid = self.record_index.get(record_id)?;
        let hyp = self.hypotheses.get(hid)?;
        self.beliefs.get(&hyp.belief_id)
    }

    /// Get all unresolved beliefs.
    pub fn unresolved_beliefs(&self) -> Vec<&Belief> {
        self.beliefs
            .values()
            .filter(|b| b.state == BeliefState::Unresolved)
            .collect()
    }

    /// Get summary statistics.
    pub fn stats(&self) -> BeliefStats {
        let resolved = self
            .beliefs
            .values()
            .filter(|b| b.state == BeliefState::Resolved)
            .count();
        let unresolved = self
            .beliefs
            .values()
            .filter(|b| b.state == BeliefState::Unresolved)
            .count();
        let singleton = self
            .beliefs
            .values()
            .filter(|b| b.state == BeliefState::Singleton)
            .count();
        let avg_stability = if self.beliefs.is_empty() {
            0.0
        } else {
            self.beliefs.values().map(|b| b.stability).sum::<f32>() / self.beliefs.len() as f32
        };

        BeliefStats {
            total_beliefs: self.beliefs.len(),
            total_hypotheses: self.hypotheses.len(),
            resolved,
            unresolved,
            singleton,
            avg_stability,
        }
    }
}

impl Default for BeliefEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Reports ──

/// Report from a single belief update cycle.
#[derive(Debug, Clone, Default)]
pub struct BeliefReport {
    pub beliefs_created: usize,
    pub beliefs_pruned: usize,
    pub revisions: usize,
    pub resolved: usize,
    pub unresolved: usize,
    pub total_beliefs: usize,
    pub total_hypotheses: usize,
    /// Churn rate = revisions / max(total_beliefs, 1).
    /// A churn rate > 0.10 on stable data indicates belief layer instability.
    pub churn_rate: f32,
}

/// Summary statistics for the belief engine.
#[derive(Debug, Clone)]
pub struct BeliefStats {
    pub total_beliefs: usize,
    pub total_hypotheses: usize,
    pub resolved: usize,
    pub unresolved: usize,
    pub singleton: usize,
    pub avg_stability: f32,
}

// ── Belief Store (persistence) ──

/// Append-only persistence for the belief engine state.
///
/// Stores as a single JSON snapshot in `beliefs.cog`.
/// This is intentionally simple — beliefs are rebuilt each cycle
/// but persisted for stability tracking and cross-session continuity.
pub struct BeliefStore {
    path: std::path::PathBuf,
}

impl BeliefStore {
    /// Open or create a belief store at the given directory.
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
        let path = path.as_ref().to_path_buf();
        Self { path }
    }

    /// Load the belief engine state from disk.
    pub fn load(&self) -> anyhow::Result<BeliefEngine> {
        let file_path = self.path.join("beliefs.cog");
        if !file_path.exists() {
            return Ok(BeliefEngine::new());
        }
        let data = std::fs::read(&file_path)?;
        if data.is_empty() {
            return Ok(BeliefEngine::new());
        }
        let engine: BeliefEngine = serde_json::from_slice(&data)?;
        Ok(engine)
    }

    /// Save the belief engine state to disk.
    pub fn save(&self, engine: &BeliefEngine) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.path)?;
        let file_path = self.path.join("beliefs.cog");
        let data = serde_json::to_vec(engine)?;
        std::fs::write(&file_path, &data)?;
        Ok(())
    }
}

// ── Helpers ──

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::Level;

    fn make_record(content: &str, tags: &[&str], semantic_type: &str) -> Record {
        let mut rec = Record::new(content.to_string(), Level::Domain);
        rec.tags = tags.iter().map(|s| s.to_string()).collect();
        rec.semantic_type = semantic_type.to_string();
        rec.confidence = Record::default_confidence_for_source("recorded");
        rec
    }

    #[test]
    fn test_hypothesis_scoring() {
        let r1 = make_record(
            "user prefers dark mode in editor",
            &["ui", "preferences"],
            "preference",
        );
        let r2 = make_record(
            "user always uses dark theme for coding",
            &["ui", "preferences"],
            "preference",
        );

        let h = Hypothesis::from_records("b1", &[&r1, &r2]);
        assert!(h.score > 0.0);
        assert!((h.confidence - 0.9).abs() < 0.01);
        assert_eq!(h.prototype_record_ids.len(), 2);
    }

    #[test]
    fn test_belief_resolve_singleton() {
        let r1 = make_record(
            "user uses vim keybindings always",
            &["editor", "preferences"],
            "preference",
        );
        let h = Hypothesis::from_records("b1", &[&r1]);

        let mut belief = Belief::new("default:editor,preferences:preference".into());
        belief.resolve(&[&h]);

        assert_eq!(belief.state, BeliefState::Singleton);
        assert_eq!(belief.winner_id, Some(h.id.clone()));
    }

    #[test]
    fn test_belief_resolve_competing() {
        // Strong supporting hypothesis
        let mut r1 = make_record(
            "user prefers dark mode absolutely",
            &["ui", "theme"],
            "preference",
        );
        r1.support_mass = 10;
        r1.confidence = 0.95;

        // Weak opposing hypothesis
        let mut r2 = make_record(
            "user sometimes uses light mode",
            &["ui", "theme"],
            "contradiction",
        );
        r2.support_mass = 1;
        r2.confidence = 0.50;

        let h1 = Hypothesis::from_records("b1", &[&r1]);
        let h2 = Hypothesis::from_records("b1", &[&r2]);

        let mut belief = Belief::new("default:theme,ui:preference".into());
        belief.resolve(&[&h1, &h2]);

        assert_eq!(belief.state, BeliefState::Resolved);
        assert_eq!(belief.winner_id, Some(h1.id.clone()));
    }

    #[test]
    fn test_belief_engine_update() {
        let mut engine = BeliefEngine::new();

        let mut records = HashMap::new();
        let r1 = make_record(
            "deploy staging first before production",
            &["deploy", "safety"],
            "decision",
        );
        let r2 = make_record(
            "deploy staging first before production always",
            &["deploy", "safety"],
            "decision",
        );
        let mut r3 = make_record(
            "skip deploy staging first before production",
            &["deploy", "safety"],
            "contradiction",
        );
        r3.conflict_mass = 2;

        records.insert(r1.id.clone(), r1);
        records.insert(r2.id.clone(), r2);
        records.insert(r3.id.clone(), r3);

        let report = engine.update(&records);

        assert!(report.total_beliefs > 0);
        assert!(report.total_hypotheses > 0);
    }

    #[test]
    fn test_belief_engine_stability_tracking() {
        let mut engine = BeliefEngine::new();

        let mut records = HashMap::new();
        let r1 = make_record(
            "Rust is the primary language for backend",
            &["tech", "language"],
            "fact",
        );
        let r2 = make_record(
            "Rust is the primary language for backend services",
            &["tech", "language"],
            "fact",
        );
        records.insert(r1.id.clone(), r1);
        records.insert(r2.id.clone(), r2);

        // Run twice — stability should increase
        engine.update(&records);
        let report = engine.update(&records);

        // Check that beliefs exist and have been tracked
        assert!(report.total_beliefs > 0 || report.beliefs_pruned > 0 || report.resolved > 0);
    }

    #[test]
    fn test_claim_key_generation() {
        let rec = make_record(
            "test content for claim key",
            &["deploy", "safety", "production"],
            "decision",
        );
        let key = BeliefEngine::claim_key(&rec);
        assert!(key.starts_with("default:"));
        assert!(key.contains("deploy"));
        assert!(key.ends_with(":decision")); // format: namespace:tags:type
    }

    #[test]
    fn test_bridge_key_normalizes_safe_tag_families() {
        let rec_a = make_record(
            "project ui work item for dashboard",
            &["project", "ui"],
            "fact",
        );
        let rec_b = make_record(
            "project frontend work item for dashboard",
            &["project", "frontend"],
            "fact",
        );

        let key_a = BeliefEngine::claim_key_with_mode(&rec_a, CoarseKeyMode::BridgeKey);
        let key_b = BeliefEngine::claim_key_with_mode(&rec_b, CoarseKeyMode::BridgeKey);

        assert_eq!(
            key_a, key_b,
            "BridgeKey should normalize ui/frontend into the same corridor"
        );
    }

    #[test]
    fn test_bridge_key_keeps_risky_tags_separate() {
        let rec_a = make_record(
            "gRPC architecture for service mesh traffic",
            &["architecture", "api"],
            "decision",
        );
        let rec_b = make_record(
            "API rate limit configuration for public endpoints",
            &["api", "config"],
            "decision",
        );

        let key_a = BeliefEngine::claim_key_with_mode(&rec_a, CoarseKeyMode::BridgeKey);
        let key_b = BeliefEngine::claim_key_with_mode(&rec_b, CoarseKeyMode::BridgeKey);

        assert_ne!(
            key_a, key_b,
            "BridgeKey must not collapse broad topic tags like api/architecture"
        );
    }

    #[test]
    fn test_bridge_guard_merges_synonym_tag_pairs_with_sdr_support() {
        let r1 = make_record(
            "project ui dashboard polish in current release",
            &["project", "ui"],
            "fact",
        );
        let r2 = make_record(
            "project frontend dashboard polish in current release",
            &["project", "frontend"],
            "fact",
        );

        let sdr1: Vec<u16> = (0..100).collect();
        let sdr2: Vec<u16> = (0..82).chain(100..118).collect();
        let mut lookup = HashMap::new();
        lookup.insert(r1.id.clone(), sdr1);
        lookup.insert(r2.id.clone(), sdr2);

        let clusters = BeliefEngine::sdr_subcluster_bridge_guarded(
            &[&r1, &r2],
            &lookup,
            CLAIM_SIMILARITY_THRESHOLD,
        );
        assert_eq!(
            clusters.len(),
            1,
            "BridgeKey should merge safe bridge-tag pairs when SDR also agrees"
        );
    }

    #[test]
    fn test_bridge_guard_blocks_records_without_bridge_overlap() {
        let r1 = make_record(
            "release pipeline policy for production rollout",
            &["deploy", "pipeline"],
            "decision",
        );
        let r2 = make_record(
            "database backup retention policy for replicas",
            &["database", "backup"],
            "decision",
        );

        let sdr1: Vec<u16> = (0..100).collect();
        let sdr2: Vec<u16> = (0..85).chain(100..115).collect();
        let mut lookup = HashMap::new();
        lookup.insert(r1.id.clone(), sdr1);
        lookup.insert(r2.id.clone(), sdr2);

        let clusters = BeliefEngine::sdr_subcluster_bridge_guarded(
            &[&r1, &r2],
            &lookup,
            CLAIM_SIMILARITY_THRESHOLD,
        );
        assert_eq!(
            clusters.len(),
            2,
            "BridgeKey must not merge records without shared normalized bridge tags"
        );
    }

    #[test]
    fn test_sdr_tag_pool_uses_namespace_semantic_corridor() {
        let rec = make_record(
            "api authentication decision for admin access",
            &["api", "auth"],
            "decision",
        );
        let key = BeliefEngine::claim_key_with_mode(&rec, CoarseKeyMode::SdrTagPool);
        assert_eq!(key, "default:decision");
    }

    #[test]
    fn test_canonical_tag_text_is_sorted_and_deduped() {
        let rec = make_record(
            "duplicate tag text ordering",
            &["Auth", "api", "auth", "API"],
            "fact",
        );
        let text = BeliefEngine::canonical_tag_text(&rec);
        assert_eq!(text, "api auth");
    }

    #[test]
    fn test_sdr_tag_pool_guard_merges_overlapping_tag_fingerprints() {
        let r1 = make_record(
            "api authentication policy for internal admin routes",
            &["api", "auth"],
            "decision",
        );
        let r2 = make_record(
            "api security policy for internal admin routes",
            &["api", "security"],
            "decision",
        );

        let sdr1: Vec<u16> = (0..100).collect();
        let sdr2: Vec<u16> = (0..85).chain(100..115).collect();
        let mut lookup = HashMap::new();
        lookup.insert(r1.id.clone(), sdr1);
        lookup.insert(r2.id.clone(), sdr2);

        let clusters = BeliefEngine::sdr_subcluster_tag_sdr_guarded(&[&r1, &r2], &lookup, 0.10);
        assert_eq!(
            clusters.len(),
            1,
            "SdrTagPool should merge when tag fingerprints overlap and content SDR agrees"
        );
    }

    #[test]
    fn test_sdr_tag_pool_guard_blocks_disjoint_tag_fingerprints() {
        let r1 = make_record(
            "deployment canary validation for release pipeline",
            &["deploy", "canary"],
            "decision",
        );
        let r2 = make_record(
            "serif reading theme preference for documentation",
            &["ui", "reading"],
            "decision",
        );

        let sdr1: Vec<u16> = (0..100).collect();
        let sdr2: Vec<u16> = (0..85).chain(100..115).collect();
        let mut lookup = HashMap::new();
        lookup.insert(r1.id.clone(), sdr1);
        lookup.insert(r2.id.clone(), sdr2);

        let clusters = BeliefEngine::sdr_subcluster_tag_sdr_guarded(&[&r1, &r2], &lookup, 0.10);
        assert_eq!(
            clusters.len(),
            2,
            "SdrTagPool must block records whose tag fingerprints are disjoint"
        );
    }

    #[test]
    fn test_belief_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = BeliefStore::new(dir.path());

        let mut engine = BeliefEngine::new();
        let mut belief = Belief::new("test-key".into());
        belief.state = BeliefState::Resolved;
        belief.stability = 5.0;
        let bid = belief.id.clone();
        engine.beliefs.insert(bid.clone(), belief);
        engine.key_index.insert("test-key".into(), bid);

        store.save(&engine).unwrap();
        let loaded = store.load().unwrap();

        assert_eq!(loaded.beliefs.len(), 1);
        assert_eq!(loaded.beliefs.values().next().unwrap().stability, 5.0);
    }

    #[test]
    fn test_empty_belief_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = BeliefStore::new(dir.path());

        let engine = store.load().unwrap();
        assert!(engine.beliefs.is_empty());
    }

    #[test]
    fn test_unresolved_beliefs() {
        let mut engine = BeliefEngine::new();

        let mut b1 = Belief::new("key1".into());
        b1.state = BeliefState::Unresolved;
        let mut b2 = Belief::new("key2".into());
        b2.state = BeliefState::Resolved;

        engine.beliefs.insert(b1.id.clone(), b1);
        engine.beliefs.insert(b2.id.clone(), b2);

        let unresolved = engine.unresolved_beliefs();
        assert_eq!(unresolved.len(), 1);
    }

    #[test]
    fn test_split_by_contradiction() {
        let r1 = make_record(
            "normal fact about deployment safety patterns",
            &["deploy"],
            "fact",
        );
        let mut r2 = make_record(
            "contradicting previous deployment claim here",
            &["deploy"],
            "contradiction",
        );
        r2.conflict_mass = 3;

        let (supporting, opposing) = BeliefEngine::split_by_contradiction(&[&r1, &r2]);
        assert_eq!(supporting.len(), 1);
        assert_eq!(opposing.len(), 1);
    }

    // ── Replay stability: running the same data twice should produce zero revisions ──

    #[test]
    fn test_replay_stability_no_churn() {
        let mut engine = BeliefEngine::new();

        let mut records = HashMap::new();
        let r1 = make_record(
            "deploy to staging before production release",
            &["deploy", "safety"],
            "decision",
        );
        let r2 = make_record(
            "deploy to staging before production release always",
            &["deploy", "safety"],
            "decision",
        );

        let key1 = BeliefEngine::claim_key(&r1);
        let key2 = BeliefEngine::claim_key(&r2);
        assert_eq!(
            key1, key2,
            "test records must share the same claim key: key1={}, key2={}",
            key1, key2
        );

        records.insert(r1.id.clone(), r1);
        records.insert(r2.id.clone(), r2);

        // First cycle — beliefs are created, revisions expected
        let report1 = engine.update(&records);
        assert!(
            report1.total_beliefs > 0,
            "first cycle should create beliefs (key={})",
            key1
        );

        // Second cycle — same data, no changes → zero revisions
        let report2 = engine.update(&records);
        assert_eq!(
            report2.revisions, 0,
            "replaying same data should produce zero revisions"
        );
        assert!(
            report2.churn_rate < 0.01,
            "churn rate should be near-zero on stable data"
        );
    }

    // ── Edge case 1: context-dependent preferences should not falsely conflict ──

    #[test]
    fn test_context_dependent_preferences_no_false_conflict() {
        // "dark mode in editor" and "light mode in documentation" share the "ui" tag
        // but have different content contexts. They should form separate beliefs,
        // NOT a single belief with competing hypotheses.
        let mut engine = BeliefEngine::new();

        let mut records = HashMap::new();
        let r1 = make_record(
            "user prefers dark mode in code editor environment",
            &["ui", "preferences"],
            "preference",
        );
        let r2 = make_record(
            "user prefers light mode for reading documentation",
            &["ui", "preferences"],
            "preference",
        );
        records.insert(r1.id.clone(), r1.clone());
        records.insert(r2.id.clone(), r2.clone());

        engine.update(&records);

        let key1 = BeliefEngine::claim_key(&r1);
        let key2 = BeliefEngine::claim_key(&r2);

        // With content bucketing, different contexts should ideally land in
        // different buckets. If they do collide (possible with 3-bit resolution),
        // they should at least not be marked as conflicting (no contradiction type).
        // Check: no belief has state == Unresolved (no false conflict detected)
        let unresolved = engine.unresolved_beliefs();
        assert_eq!(
            unresolved.len(),
            0,
            "context-dependent preferences should not produce unresolved conflicts \
             (key1={}, key2={})",
            key1,
            key2
        );
    }

    // ── Edge case 2: shared tags alone should not count as support ──

    #[test]
    fn test_shared_tags_without_true_support() {
        // Two records share the same tag "deploy" but are about completely
        // different aspects. In the epistemic layer they should NOT
        // accumulate support_mass just from shared tags if their content
        // and semantic types differ enough.
        use crate::background_brain::update_epistemic_state;

        let mut records = HashMap::new();
        let mut r1 = make_record(
            "deploy to staging before production release always",
            &["deploy"],
            "decision",
        );
        r1.support_mass = 0;
        r1.conflict_mass = 0;
        let mut r2 = make_record(
            "deploy container images with proper tagging scheme",
            &["deploy"],
            "observation",
        );
        r2.support_mass = 0;
        r2.conflict_mass = 0;

        records.insert(r1.id.clone(), r1);
        records.insert(r2.id.clone(), r2);

        let report = update_epistemic_state(&mut records);

        // Different semantic_type + only 1 shared tag → should NOT count as confirming.
        // The rule is: (shared_semantic && shared_tags > 0) || reinforcing_relation || shared_tags >= 2
        // Here: shared_semantic is false (decision vs observation), shared_tags == 1 (< 2),
        // no connection → confirming should be 0.
        assert_eq!(report.total_support_links, 0,
            "records with different semantic_type and only 1 shared tag should not generate support");
    }

    // ── Edge case 3: contradiction record should not pull entire cluster into conflict ──

    #[test]
    fn test_contradiction_scoping() {
        // One contradiction record tagged ["deploy", "safety"] should conflict
        // only with records that share tags or connections, NOT with an unrelated
        // record tagged ["logging"].
        use crate::background_brain::update_epistemic_state;

        let mut records = HashMap::new();
        let mut r1 = make_record(
            "deploy staging first for safety always",
            &["deploy", "safety"],
            "decision",
        );
        r1.support_mass = 0;
        r1.conflict_mass = 0;
        let mut r2 = make_record(
            "this contradicts the deploy safety policy",
            &["deploy", "safety"],
            "contradiction",
        );
        r2.support_mass = 0;
        r2.conflict_mass = 0;
        let mut r3 = make_record(
            "use structured logging for all services",
            &["logging"],
            "decision",
        );
        r3.support_mass = 0;
        r3.conflict_mass = 0;

        records.insert(r1.id.clone(), r1.clone());
        records.insert(r2.id.clone(), r2.clone());
        records.insert(r3.id.clone(), r3.clone());

        let _report = update_epistemic_state(&mut records);

        // r1 should have conflict (shares tags with contradiction r2)
        let r1_after = records.get(&r1.id).unwrap();
        assert!(
            r1_after.conflict_mass > 0,
            "r1 should gain conflict_mass from contradicting r2"
        );

        // r3 should have NO conflict (no shared tags with r2, no connection)
        let r3_after = records.get(&r3.id).unwrap();
        assert_eq!(
            r3_after.conflict_mass, 0,
            "r3 (logging) should not be pulled into conflict by unrelated contradiction r2"
        );
    }

    // ── Edge case 4: Unicode/non-English records should still group stably ──

    #[test]
    fn test_unicode_non_english_belief_grouping_boundary() {
        let mut engine = BeliefEngine::new();

        let mut records = HashMap::new();
        let r1 = make_record(
            "користувач віддає перевагу темній темі в редакторі коду",
            &["ui", "preferences"],
            "preference",
        );
        let r2 = make_record(
            "у редакторі коду користувач обирає темну тему для роботи",
            &["ui", "preferences"],
            "preference",
        );

        let key1 = BeliefEngine::claim_key(&r1);
        let key2 = BeliefEngine::claim_key(&r2);

        records.insert(r1.id.clone(), r1);
        records.insert(r2.id.clone(), r2);

        let report = engine.update(&records);

        // Diagnostic boundary: Unicode paraphrases may still split buckets with the
        // current coarse fingerprint. If they do, the belief layer must fail safe:
        // no false unresolved conflict should be created from a mere bucket split.
        assert_eq!(report.unresolved, 0,
            "aligned non-English paraphrases should not create a false unresolved belief: key1={}, key2={}",
            key1, key2);
        assert!(key1 == key2 || report.total_beliefs == 0,
            "if Unicode paraphrases split into different buckets, the engine should avoid fabricating a belief");
    }

    // ── Edge case 5: synonym-heavy paraphrases show current coarse-bucket limits ──

    #[test]
    fn test_synonym_heavy_paraphrases_bucket_boundary() {
        let mut engine = BeliefEngine::new();

        let mut records = HashMap::new();
        let r1 = make_record(
            "deploy to staging before production release always",
            &["deploy", "safety"],
            "decision",
        );
        let r2 = make_record(
            "ship to preprod prior to live rollout every time",
            &["deploy", "safety"],
            "decision",
        );

        let key1 = BeliefEngine::claim_key(&r1);
        let key2 = BeliefEngine::claim_key(&r2);

        records.insert(r1.id.clone(), r1);
        records.insert(r2.id.clone(), r2);

        let report = engine.update(&records);

        // This regression test is intentionally diagnostic rather than prescriptive:
        // synonym-heavy paraphrases may or may not collide with the current coarse bucket.
        // The important invariant is that they must not create a false conflict state.
        assert_eq!(report.unresolved, 0,
            "synonym-heavy paraphrases should not create false unresolved beliefs: key1={}, key2={}",
            key1, key2);
        assert!(
            report.total_beliefs <= 1 || key1 != key2,
            "if both records stay separate, the claim keys should explain the split"
        );
    }

    // ── SDR sub-clustering tests ──

    #[test]
    fn test_sdr_subcluster_similar_records_merge() {
        // Two records with SDR overlap above threshold (0.15) should cluster together
        let r1 = make_record(
            "deploy staging before production release",
            &["deploy"],
            "decision",
        );
        let r2 = make_record(
            "deploy staging before production release always",
            &["deploy"],
            "decision",
        );

        // Simulate moderate overlap: Tanimoto = 80/120 ≈ 0.67 (well above 0.15)
        let sdr1: Vec<u16> = (0..100).collect();
        let sdr2: Vec<u16> = (0..80).chain(100..120).collect();
        let mut lookup = HashMap::new();
        lookup.insert(r1.id.clone(), sdr1);
        lookup.insert(r2.id.clone(), sdr2);

        let clusters =
            BeliefEngine::sdr_subcluster(&[&r1, &r2], &lookup, CLAIM_SIMILARITY_THRESHOLD);
        assert_eq!(
            clusters.len(),
            1,
            "similar SDR records should merge into 1 cluster"
        );
    }

    #[test]
    fn test_sdr_subcluster_different_records_split() {
        // Two records with SDR overlap below threshold (< 0.15) should stay separate
        let r1 = make_record(
            "deploy staging before production release",
            &["deploy", "safety"],
            "decision",
        );
        let r2 = make_record(
            "database connection pool configuration tuning",
            &["deploy", "safety"],
            "decision",
        );

        // Simulate low overlap: only 10% shared
        let sdr1: Vec<u16> = (0..100).collect();
        let sdr2: Vec<u16> = (0..10).chain(200..290).collect(); // 10 shared / (100+90) = 10/190 ≈ 0.05
        let mut lookup = HashMap::new();
        lookup.insert(r1.id.clone(), sdr1);
        lookup.insert(r2.id.clone(), sdr2);

        let clusters =
            BeliefEngine::sdr_subcluster(&[&r1, &r2], &lookup, CLAIM_SIMILARITY_THRESHOLD);
        assert_eq!(
            clusters.len(),
            2,
            "different SDR records should split into 2 clusters"
        );
    }

    #[test]
    fn test_sdr_grouping_creates_separate_beliefs() {
        // Full engine test: two records with same tags but different SDR
        // should form separate beliefs when SDR lookup is provided.
        let mut engine = BeliefEngine::new();

        let mut records = HashMap::new();
        let r1 = make_record(
            "deploy staging before production release always",
            &["ops", "process"],
            "decision",
        );
        let r2 = make_record(
            "deploy staging before production release every time",
            &["ops", "process"],
            "decision",
        );
        let r3 = make_record(
            "database connection pool sizing and timeout config",
            &["ops", "process"],
            "decision",
        );
        let r4 = make_record(
            "database connection pool sizing and timeout tuning",
            &["ops", "process"],
            "decision",
        );

        // r1 & r2 are similar (deploy topic), r3 & r4 are similar (database topic)
        // But r1 & r3 are dissimilar
        let sdr_deploy: Vec<u16> = (0..100).collect();
        let sdr_deploy2: Vec<u16> = (0..85).chain(100..115).collect(); // Tanimoto ≈ 0.74
        let sdr_db: Vec<u16> = (500..600).collect();
        let sdr_db2: Vec<u16> = (500..580).chain(600..620).collect(); // Tanimoto ≈ 0.67

        let mut lookup = HashMap::new();
        lookup.insert(r1.id.clone(), sdr_deploy);
        lookup.insert(r2.id.clone(), sdr_deploy2);
        lookup.insert(r3.id.clone(), sdr_db);
        lookup.insert(r4.id.clone(), sdr_db2);

        records.insert(r1.id.clone(), r1);
        records.insert(r2.id.clone(), r2);
        records.insert(r3.id.clone(), r3);
        records.insert(r4.id.clone(), r4);

        // Without SDR — all 4 records have same coarse key → 1 belief
        let report_no_sdr = engine.update(&records);
        let beliefs_without_sdr = report_no_sdr.total_beliefs;

        // With SDR — should split into 2 beliefs (deploy cluster + db cluster)
        let mut engine2 = BeliefEngine::new();
        let report_with_sdr = engine2.update_with_sdr(&records, &lookup);

        assert!(
            report_with_sdr.total_beliefs >= 2,
            "SDR grouping should create ≥2 beliefs (deploy + db), got {} (without SDR: {})",
            report_with_sdr.total_beliefs,
            beliefs_without_sdr
        );
    }

    #[test]
    fn test_sdr_grouping_fallback_without_sdr_data() {
        // Records without SDR data should still group by coarse key
        let mut engine = BeliefEngine::new();

        let mut records = HashMap::new();
        let r1 = make_record(
            "deploy staging before production release always",
            &["deploy", "safety"],
            "decision",
        );
        let r2 = make_record(
            "deploy staging before production release every time",
            &["deploy", "safety"],
            "decision",
        );
        records.insert(r1.id.clone(), r1);
        records.insert(r2.id.clone(), r2);

        // Empty SDR lookup — should still form beliefs via coarse grouping
        let report = engine.update_with_sdr(&records, &HashMap::new());
        assert!(
            report.total_beliefs > 0,
            "empty SDR lookup should fall back to coarse grouping"
        );
    }

    #[test]
    fn test_tanimoto_correctness() {
        // Verify our tanimoto implementation
        let a: Vec<u16> = vec![1, 2, 3, 4, 5];
        let b: Vec<u16> = vec![3, 4, 5, 6, 7];
        // intersection = {3,4,5} = 3, union = {1,2,3,4,5,6,7} = 7
        let sim = BeliefEngine::tanimoto(&a, &b);
        assert!(
            (sim - 3.0 / 7.0).abs() < 0.001,
            "expected 3/7 ≈ 0.4286, got {}",
            sim
        );

        // Identical
        let sim_same = BeliefEngine::tanimoto(&a, &a);
        assert!(
            (sim_same - 1.0).abs() < 0.001,
            "identical vectors should have similarity 1.0"
        );

        // Disjoint
        let c: Vec<u16> = vec![10, 20, 30];
        let sim_disjoint = BeliefEngine::tanimoto(&a, &c);
        assert!(
            (sim_disjoint).abs() < 0.001,
            "disjoint vectors should have similarity 0.0"
        );
    }
}
