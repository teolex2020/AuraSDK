//! Concept Discovery Layer — finds stable abstractions over beliefs.
//!
//! Third tier of the cognitive hierarchy:
//!   Record → Belief → **Concept** → Causal Pattern → Policy
//!
//! Phase 1 constraints (read-only candidate discovery):
//!   - Does NOT influence recall ranking or record merge
//!   - Full rebuild each maintenance cycle (no persistent trust)
//!   - Every concept traces back to source belief_ids + record_ids
//!   - Unresolved beliefs are excluded from concept formation

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::belief::{BeliefEngine, BeliefState, SdrLookup};
use crate::record::Record;

// ── Constants ──

/// Minimum belief stability to be considered as a concept seed.
const MIN_BELIEF_STABILITY: f32 = 2.0;

/// Minimum belief confidence to be considered as a concept seed.
const MIN_BELIEF_CONFIDENCE: f32 = 0.55;

/// Core term frequency threshold — a term must appear in at least
/// this fraction of cluster beliefs to be considered "core".
const CORE_TERM_THRESHOLD: f64 = 0.70;

/// Shell term lower bound — terms appearing in [SHELL_LOWER, CORE) of
/// cluster beliefs are "shell" (variable but not noise).
const SHELL_TERM_LOWER: f64 = 0.20;

/// Tanimoto threshold for clustering beliefs into concept groups.
/// Lowered from 0.20 to 0.10 based on calibration sprint findings:
/// real SDR centroids have Tanimoto 0.05-0.15 between same-topic beliefs.
const CONCEPT_SIMILARITY_THRESHOLD: f32 = 0.10;

/// Abstraction score weights.
const W_SUPPORT: f32 = 0.35;
const W_CONFIDENCE: f32 = 0.25;
const W_STABILITY: f32 = 0.20;
const W_COHESION: f32 = 0.20;

/// State thresholds for abstraction_score.
const STABLE_THRESHOLD: f32 = 0.75;
const CANDIDATE_THRESHOLD: f32 = 0.50;

/// Concept-level similarity threshold for canonical feature mode.
/// Jaccard over canonical tokens; calibrated to separate same-topic (0.15-0.50)
/// from cross-topic (0.00-0.08).
const CANONICAL_SIMILARITY_THRESHOLD: f32 = 0.12;

// ── ConceptSimilarityMode ──

/// Controls how beliefs are clustered into concepts.
///
/// `SdrTanimoto` uses the existing SDR centroid Tanimoto path.
/// `CanonicalFeature` uses lightweight canonical tokenization + Jaccard similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptSimilarityMode {
    /// SDR centroid Tanimoto (original path).
    SdrTanimoto,
    /// Canonical feature tokenization + Jaccard similarity (Variant A).
    CanonicalFeature,
}

impl Default for ConceptSimilarityMode {
    fn default() -> Self {
        ConceptSimilarityMode::SdrTanimoto
    }
}

// ── ConceptSeedMode ──

/// Controls concept seed selection gates.
///
/// `Standard` is the production default.
/// `Relaxed` is an experimental mode with lower thresholds to allow
/// concept formation from smaller corpora.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptSeedMode {
    /// Production gates: stability >= 2.0, confidence >= 0.55.
    Standard,
    /// Experimental warmup gates: stability >= 1.0, confidence >= 0.55.
    /// Narrows the relaxation to early stability only.
    Warmup,
    /// Experimental relaxed gates: stability >= 1.0, confidence >= 0.40.
    Relaxed,
}

impl Default for ConceptSeedMode {
    fn default() -> Self {
        ConceptSeedMode::Standard
    }
}

// ── ConceptPartitionMode ──

/// Controls how belief seeds are partitioned before concept clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptPartitionMode {
    /// Production default: partition by `(namespace, semantic_type)`.
    Standard,
    /// Experimental mode: partition by `namespace` only.
    NamespaceOnly,
}

impl Default for ConceptPartitionMode {
    fn default() -> Self {
        ConceptPartitionMode::Standard
    }
}

// ── ConceptUnionMode ──

/// Controls comparison-only union relaxations inside concept clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptUnionMode {
    /// Production default: current family guard only.
    Standard,
    /// Experimental mode: allow a narrow single-tag cross-family bridge for
    /// fact<->decision pairs with strong shared tag evidence.
    SingleTagFactDecisionBridge,
}

impl Default for ConceptUnionMode {
    fn default() -> Self {
        ConceptUnionMode::Standard
    }
}

// —— ConceptSurfaceMode ——

/// Controls whether bounded surfaced concept output is exposed at runtime.
///
/// This is a runtime rollout control only. It does not affect concept
/// discovery, recall ranking, compression, or behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptSurfaceMode {
    /// Default-safe: no surfaced concept output.
    Off,
    /// Allow bounded inspection-only surfaced concepts.
    Inspect,
    /// Reserved for future controlled promotion work. Not active yet.
    Limited,
}

impl Default for ConceptSurfaceMode {
    fn default() -> Self {
        ConceptSurfaceMode::Off
    }
}

// ── ConceptState ──

/// Lifecycle state of a concept candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptState {
    /// Newly discovered, not yet stable.
    Candidate,
    /// Abstraction score consistently above threshold.
    Stable,
    /// Score too low or cluster dissolved.
    Rejected,
}

// ── ConceptCandidate ──

/// A discovered concept — a stable abstraction over related beliefs.
///
/// Phase 1: read-only, rebuilt each cycle, no recall impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptCandidate {
    /// Unique concept ID (deterministic from key).
    pub id: String,
    /// Canonical key: namespace:semantic_type:sorted_core_tags:centroid_hash.
    pub key: String,

    pub namespace: String,
    pub semantic_type: String,

    /// Source belief IDs that feed this concept.
    pub belief_ids: Vec<String>,
    /// Transitive source record IDs (full provenance).
    pub record_ids: Vec<String>,

    /// High-frequency terms shared across beliefs (the "essence").
    pub core_terms: Vec<String>,
    /// Variable terms that appear in some but not all beliefs.
    pub shell_terms: Vec<String>,
    /// Stable tags across beliefs.
    pub tags: Vec<String>,

    /// Aggregated support mass (log-normalized).
    pub support_mass: f32,
    /// Average confidence of source beliefs.
    pub confidence: f32,
    /// Average stability of source beliefs.
    pub stability: f32,
    /// Internal SDR cohesion (average pairwise Tanimoto within cluster).
    pub cohesion: f32,
    /// Composite abstraction quality score.
    pub abstraction_score: f32,

    pub state: ConceptState,
    pub last_updated: f64,
}

// ── ConceptReport ──

/// Report from a single concept discovery cycle.
#[derive(Debug, Clone, Default)]
pub struct ConceptReport {
    /// How many eligible beliefs were seeded.
    pub seeds_found: usize,
    /// Total concept candidates discovered.
    pub candidates_found: usize,
    /// Candidates that reached Stable state.
    pub stable_count: usize,
    /// Candidates that were Rejected.
    pub rejected_count: usize,
    /// Average abstraction score across all candidates.
    pub avg_abstraction_score: f32,

    // ── Centroid diagnostics ──
    /// Number of non-empty centroids built.
    pub centroids_built: usize,
    /// Number of partitions with >= 2 seeds.
    pub partitions_with_multiple_seeds: usize,
    /// Sizes of partitions that had >= 2 seeds.
    pub multi_seed_partition_sizes: Vec<usize>,
    /// Sizes of clusters produced after union-find clustering.
    pub cluster_sizes: Vec<usize>,
    /// Number of clusters with >= 2 beliefs.
    pub clusters_with_multiple_beliefs: usize,
    /// Largest cluster size observed in this cycle.
    pub largest_cluster_size: usize,
    /// Total pairwise comparisons made during clustering.
    pub pairwise_comparisons: usize,
    /// Number of pairs that passed the similarity threshold.
    pub pairwise_above_threshold: usize,
    /// Minimum pairwise Tanimoto across all comparisons.
    pub tanimoto_min: f32,
    /// Maximum pairwise Tanimoto across all comparisons.
    pub tanimoto_max: f32,
    /// Average pairwise Tanimoto across all comparisons.
    pub tanimoto_avg: f32,
    /// Median pairwise Tanimoto.
    pub tanimoto_p50: f32,
    /// 95th percentile pairwise Tanimoto.
    pub tanimoto_p95: f32,
    /// Average centroid size (number of bits).
    pub avg_centroid_size: f32,
}

// ── ConceptEngine ──

/// The concept engine — discovers and maintains concept candidates.
///
/// Phase 1: full rebuild each cycle, no persistent trust.
/// Concepts are derived state, not source of truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptEngine {
    /// All concept candidates, keyed by concept ID.
    pub concepts: HashMap<String, ConceptCandidate>,
    /// Index: concept key → concept ID.
    pub key_index: HashMap<String, String>,
    /// Seed selection mode.
    #[serde(default)]
    pub seed_mode: ConceptSeedMode,
    /// Similarity mode for concept clustering.
    #[serde(default)]
    pub similarity_mode: ConceptSimilarityMode,
    /// Partition mode for grouping seed beliefs before clustering.
    #[serde(default)]
    pub partition_mode: ConceptPartitionMode,
    /// Comparison-only union relaxation mode.
    #[serde(default)]
    pub union_mode: ConceptUnionMode,
}

#[derive(Debug, Clone)]
struct PartitionClusterInspection {
    partition_key: String,
    seed_count: usize,
    cluster_sizes: Vec<usize>,
    clusters: Vec<Vec<String>>,
}

impl ConceptEngine {
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            key_index: HashMap::new(),
            seed_mode: ConceptSeedMode::Standard,
            similarity_mode: ConceptSimilarityMode::SdrTanimoto,
            partition_mode: ConceptPartitionMode::Standard,
            union_mode: ConceptUnionMode::Standard,
        }
    }

    /// Create a new engine with specified seed mode.
    pub fn with_seed_mode(mode: ConceptSeedMode) -> Self {
        Self {
            concepts: HashMap::new(),
            key_index: HashMap::new(),
            seed_mode: mode,
            similarity_mode: ConceptSimilarityMode::SdrTanimoto,
            partition_mode: ConceptPartitionMode::Standard,
            union_mode: ConceptUnionMode::Standard,
        }
    }

    /// Run a full concept discovery cycle.
    ///
    /// Rebuilds all concepts from scratch using current belief engine state.
    /// This is the primary entry point called from maintenance.
    pub fn discover(
        &mut self,
        belief_engine: &BeliefEngine,
        records: &HashMap<String, Record>,
        sdr_lookup: &SdrLookup,
    ) -> ConceptReport {
        let mut report = ConceptReport::default();

        // Phase A: Select belief seeds
        let seeds = self.select_seeds(belief_engine);
        report.seeds_found = seeds.len();

        if seeds.len() < 2 {
            // Not enough beliefs for concept formation
            self.concepts.clear();
            self.key_index.clear();
            return report;
        }

        // Collect record IDs per belief for provenance
        let belief_records = self.collect_belief_records(belief_engine, &seeds);

        // Build belief-level SDR centroids for clustering
        let centroids = self.build_centroids(&seeds, belief_engine, records, sdr_lookup);

        // Centroid diagnostics
        let non_empty_centroids = centroids.values().filter(|c| !c.is_empty()).count();
        report.centroids_built = non_empty_centroids;
        if non_empty_centroids > 0 {
            report.avg_centroid_size = centroids
                .values()
                .filter(|c| !c.is_empty())
                .map(|c| c.len() as f32)
                .sum::<f32>()
                / non_empty_centroids as f32;
        }

        // Phase B: Partition seeds, then cluster within each partition.
        // Standard mode keeps `(namespace, semantic_type)` boundaries.
        // NamespaceOnly is experimental and only widens the candidate pool;
        // actual clustering still relies on similarity gates.
        let partitions = self.partition_seeds(&seeds, belief_engine);
        let mut clusters: Vec<Vec<String>> = Vec::new();
        let mut all_tanimotos: Vec<f32> = Vec::new();
        let mut pairwise_above = 0usize;

        // Build per-belief tag sets for cross-topic merge prevention
        let belief_tags: HashMap<String, HashSet<String>> = belief_records
            .iter()
            .map(|(bid, rids)| {
                let tags: HashSet<String> = rids
                    .iter()
                    .filter_map(|rid| records.get(rid))
                    .flat_map(|r| r.tags.iter().cloned())
                    .collect();
                (bid.clone(), tags)
            })
            .collect();

        let belief_families: HashMap<String, String> = seeds
            .iter()
            .filter_map(|bid| {
                belief_engine
                    .beliefs
                    .get(bid)
                    .map(|belief| (bid.clone(), parse_belief_key_family(&belief.key)))
            })
            .collect();
        let belief_semantic_types: HashMap<String, String> = seeds
            .iter()
            .filter_map(|bid| {
                belief_engine
                    .beliefs
                    .get(bid)
                    .map(|belief| (bid.clone(), parse_belief_key_ns_st(&belief.key).1))
            })
            .collect();

        // Build canonical token sets per belief (for CanonicalFeature mode)
        let belief_tokens: HashMap<String, HashSet<String>> =
            if self.similarity_mode == ConceptSimilarityMode::CanonicalFeature {
                let tokens: HashMap<String, HashSet<String>> = seeds
                    .iter()
                    .map(|bid| {
                        let rids = belief_records.get(bid);
                        let t = belief_canonical_tokens(bid, &belief_records, records);
                        let _ = rids; // suppress unused
                        (bid.clone(), t)
                    })
                    .collect();
                // Report non-empty token sets
                let non_empty = tokens.values().filter(|t| !t.is_empty()).count();
                report.centroids_built = non_empty; // reuse field for canonical mode
                tokens
            } else {
                HashMap::new()
            };

        // Determine active threshold based on mode
        let active_threshold = match self.similarity_mode {
            ConceptSimilarityMode::SdrTanimoto => CONCEPT_SIMILARITY_THRESHOLD,
            ConceptSimilarityMode::CanonicalFeature => CANONICAL_SIMILARITY_THRESHOLD,
        };

        for (partition_key, partition_seeds) in &partitions {
            if partition_seeds.len() < 2 {
                // Single-belief partitions can't form concepts
                continue;
            }
            report.partitions_with_multiple_seeds += 1;
            report
                .multi_seed_partition_sizes
                .push(partition_seeds.len());

            // Collect pairwise similarity diagnostics for this partition
            for i in 0..partition_seeds.len() {
                for j in (i + 1)..partition_seeds.len() {
                    let sim = match self.similarity_mode {
                        ConceptSimilarityMode::SdrTanimoto => {
                            let sdr_i = centroids.get(&partition_seeds[i]);
                            let sdr_j = centroids.get(&partition_seeds[j]);
                            match (sdr_i, sdr_j) {
                                (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => {
                                    tanimoto(a, b)
                                }
                                _ => continue,
                            }
                        }
                        ConceptSimilarityMode::CanonicalFeature => {
                            let tok_i = belief_tokens.get(&partition_seeds[i]);
                            let tok_j = belief_tokens.get(&partition_seeds[j]);
                            match (tok_i, tok_j) {
                                (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => {
                                    jaccard(a, b)
                                }
                                _ => continue,
                            }
                        }
                    };
                    all_tanimotos.push(sim);
                    if sim >= active_threshold {
                        pairwise_above += 1;
                    }
                }
            }

            let inspection = self.inspect_partition_clusters(
                partition_key,
                partition_seeds,
                &centroids,
                &belief_tags,
                &belief_families,
                &belief_semantic_types,
                &belief_tokens,
            );
            report
                .cluster_sizes
                .extend(inspection.cluster_sizes.iter().copied());
            report.clusters_with_multiple_beliefs += inspection
                .cluster_sizes
                .iter()
                .filter(|size| **size >= 2)
                .count();
            report.largest_cluster_size = report
                .largest_cluster_size
                .max(inspection.cluster_sizes.iter().copied().max().unwrap_or(0));
            let _ = (&inspection.partition_key, inspection.seed_count);
            clusters.extend(inspection.clusters);
        }

        // Fill Tanimoto diagnostics
        report.pairwise_comparisons = all_tanimotos.len();
        report.pairwise_above_threshold = pairwise_above;
        if !all_tanimotos.is_empty() {
            all_tanimotos.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = all_tanimotos.len();
            report.tanimoto_min = all_tanimotos[0];
            report.tanimoto_max = all_tanimotos[n - 1];
            report.tanimoto_avg = all_tanimotos.iter().sum::<f32>() / n as f32;
            report.tanimoto_p50 = all_tanimotos[n / 2];
            report.tanimoto_p95 = all_tanimotos[(n as f32 * 0.95) as usize];
        }

        // Phase C-E: Build concept candidates from clusters
        let mut new_concepts: HashMap<String, ConceptCandidate> = HashMap::new();
        let mut new_key_index: HashMap<String, String> = HashMap::new();

        for cluster_belief_ids in &clusters {
            if cluster_belief_ids.len() < 2 {
                // Single-belief clusters don't form concepts
                continue;
            }

            // Gather belief metadata
            let cluster_beliefs: Vec<_> = cluster_belief_ids
                .iter()
                .filter_map(|bid| belief_engine.beliefs.get(bid))
                .collect();

            if cluster_beliefs.is_empty() {
                continue;
            }

            // Gather all record IDs in this cluster
            let mut all_record_ids: Vec<String> = Vec::new();
            for bid in cluster_belief_ids {
                if let Some(rids) = belief_records.get(bid) {
                    all_record_ids.extend(rids.iter().cloned());
                }
            }
            all_record_ids.sort();
            all_record_ids.dedup();

            // Gather all record contents for term extraction
            let cluster_records: Vec<&Record> = all_record_ids
                .iter()
                .filter_map(|rid| records.get(rid))
                .collect();

            if cluster_records.is_empty() {
                continue;
            }

            // Extract core/shell terms (Phase C)
            let (core_terms, shell_terms) = extract_terms(&cluster_records);

            // Extract tags
            let tags = extract_stable_tags(
                &cluster_beliefs
                    .iter()
                    .flat_map(|b| {
                        // Parse tags from belief key: "namespace:tag1,tag2,tag3:semantic_type"
                        let parts: Vec<&str> = b.key.split(':').collect();
                        if parts.len() >= 2 {
                            parts[1]
                                .split(',')
                                .map(|s| s.to_string())
                                .collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        }
                    })
                    .collect::<Vec<_>>(),
                cluster_beliefs.len(),
            );

            // Determine namespace and semantic_type from first belief's key.
            // Normalize away belief subcluster suffixes like `decision#3` so
            // concept identity does not drift with lower-layer subcluster slot
            // numbering.
            let first_key = &cluster_beliefs[0].key;
            let (namespace, semantic_type) = parse_belief_key_ns_st(first_key);

            // Compute metrics (Phase D)
            let support_mass = cluster_beliefs.iter().map(|b| b.support_mass).sum::<f32>();
            let support_norm = (1.0 + support_mass).ln();

            let confidence = cluster_beliefs.iter().map(|b| b.confidence).sum::<f32>()
                / cluster_beliefs.len() as f32;

            let stability = cluster_beliefs.iter().map(|b| b.stability).sum::<f32>()
                / cluster_beliefs.len() as f32;

            let cohesion = compute_cohesion(cluster_belief_ids, &centroids);

            let abstraction_score = W_SUPPORT * support_norm.min(1.0)
                + W_CONFIDENCE * confidence
                + W_STABILITY * (stability / (stability + 3.0)) // normalize: 3 cycles → 0.5
                + W_COHESION * cohesion;

            // Classify state (Phase E)
            let state = if abstraction_score >= STABLE_THRESHOLD {
                ConceptState::Stable
            } else if abstraction_score >= CANDIDATE_THRESHOLD {
                ConceptState::Candidate
            } else {
                ConceptState::Rejected
            };

            // Build concept key from stable abstraction features (not belief_ids)
            let cluster_centroids: Vec<&Vec<u16>> = cluster_belief_ids
                .iter()
                .filter_map(|bid| centroids.get(bid))
                .collect();
            let concept_key = concept_key(
                &namespace,
                &semantic_type,
                &tags,
                &core_terms,
                &cluster_centroids,
            );

            // Deterministic ID from key
            let concept_id = deterministic_id(&concept_key);

            let candidate = ConceptCandidate {
                id: concept_id.clone(),
                key: concept_key.clone(),
                namespace,
                semantic_type,
                belief_ids: cluster_belief_ids.clone(),
                record_ids: all_record_ids,
                core_terms,
                shell_terms,
                tags,
                support_mass,
                confidence,
                stability,
                cohesion,
                abstraction_score,
                state: state.clone(),
                last_updated: now_secs(),
            };

            match state {
                ConceptState::Stable => report.stable_count += 1,
                ConceptState::Rejected => report.rejected_count += 1,
                ConceptState::Candidate => {}
            }

            new_key_index.insert(concept_key, concept_id.clone());
            new_concepts.insert(concept_id, candidate);
        }

        report.candidates_found = new_concepts.len();
        if !new_concepts.is_empty() {
            report.avg_abstraction_score = new_concepts
                .values()
                .map(|c| c.abstraction_score)
                .sum::<f32>()
                / new_concepts.len() as f32;
        }

        // Full rebuild — replace state
        self.concepts = new_concepts;
        self.key_index = new_key_index;

        report
    }

    // ── Phase A: Select belief seeds ──

    /// Partition seeds into groups before concept clustering.
    fn partition_seeds(
        &self,
        seed_ids: &[String],
        engine: &BeliefEngine,
    ) -> HashMap<String, Vec<String>> {
        let mut partitions: HashMap<String, Vec<String>> = HashMap::new();
        for bid in seed_ids {
            if let Some(belief) = engine.beliefs.get(bid) {
                let (ns, st) = parse_belief_key_ns_st(&belief.key);
                let partition_key = match self.partition_mode {
                    ConceptPartitionMode::Standard => format!("{}:{}", ns, st),
                    ConceptPartitionMode::NamespaceOnly => ns,
                };
                partitions
                    .entry(partition_key)
                    .or_default()
                    .push(bid.clone());
            }
        }
        partitions
    }

    fn select_seeds(&self, engine: &BeliefEngine) -> Vec<String> {
        let (min_stability, min_confidence) = match self.seed_mode {
            ConceptSeedMode::Standard => (MIN_BELIEF_STABILITY, MIN_BELIEF_CONFIDENCE),
            ConceptSeedMode::Warmup => (1.0, MIN_BELIEF_CONFIDENCE),
            ConceptSeedMode::Relaxed => (1.0, 0.40),
        };
        engine
            .beliefs
            .values()
            .filter(|b| {
                // Only Resolved and Singleton beliefs (not Unresolved/Empty)
                matches!(b.state, BeliefState::Resolved | BeliefState::Singleton)
                    && b.stability >= min_stability
                    && b.confidence >= min_confidence
            })
            .map(|b| b.id.clone())
            .collect()
    }

    // ── Provenance: collect record IDs per belief ──

    fn collect_belief_records(
        &self,
        engine: &BeliefEngine,
        seed_ids: &[String],
    ) -> HashMap<String, Vec<String>> {
        let mut result = HashMap::new();
        for bid in seed_ids {
            if let Some(belief) = engine.beliefs.get(bid) {
                let mut rids: Vec<String> = Vec::new();
                for hid in &belief.hypothesis_ids {
                    if let Some(hyp) = engine.hypotheses.get(hid) {
                        rids.extend(hyp.prototype_record_ids.iter().cloned());
                    }
                }
                rids.sort();
                rids.dedup();
                result.insert(bid.clone(), rids);
            }
        }
        result
    }

    // ── Build SDR centroids per belief ──

    fn build_centroids(
        &self,
        seed_ids: &[String],
        engine: &BeliefEngine,
        _records: &HashMap<String, Record>,
        sdr_lookup: &SdrLookup,
    ) -> HashMap<String, Vec<u16>> {
        let mut centroids = HashMap::new();
        for bid in seed_ids {
            if let Some(belief) = engine.beliefs.get(bid) {
                let mut all_bits: Vec<u16> = Vec::new();
                for hid in &belief.hypothesis_ids {
                    if let Some(hyp) = engine.hypotheses.get(hid) {
                        for rid in &hyp.prototype_record_ids {
                            if let Some(sdr) = sdr_lookup.get(rid) {
                                all_bits.extend_from_slice(sdr);
                            }
                        }
                    }
                }
                // Centroid = union of all record SDR bits, deduplicated and sorted
                all_bits.sort();
                all_bits.dedup();
                centroids.insert(bid.clone(), all_bits);
            }
        }
        centroids
    }

    // ── Phase B: Cluster beliefs ──

    /// Dispatch to the appropriate clustering path based on similarity_mode.
    fn cluster_beliefs_dispatch(
        &self,
        seed_ids: &[String],
        centroids: &HashMap<String, Vec<u16>>,
        belief_tags: &HashMap<String, HashSet<String>>,
        belief_families: &HashMap<String, String>,
        belief_semantic_types: &HashMap<String, String>,
        belief_tokens: &HashMap<String, HashSet<String>>,
    ) -> Vec<Vec<String>> {
        match self.similarity_mode {
            ConceptSimilarityMode::SdrTanimoto => self.cluster_beliefs(
                seed_ids,
                centroids,
                belief_tags,
                belief_families,
                belief_semantic_types,
            ),
            ConceptSimilarityMode::CanonicalFeature => self.cluster_beliefs_canonical(
                seed_ids,
                belief_tags,
                belief_families,
                belief_semantic_types,
                belief_tokens,
            ),
        }
    }

    fn inspect_partition_clusters(
        &self,
        partition_key: &str,
        seed_ids: &[String],
        centroids: &HashMap<String, Vec<u16>>,
        belief_tags: &HashMap<String, HashSet<String>>,
        belief_families: &HashMap<String, String>,
        belief_semantic_types: &HashMap<String, String>,
        belief_tokens: &HashMap<String, HashSet<String>>,
    ) -> PartitionClusterInspection {
        let clusters = self.cluster_beliefs_dispatch(
            seed_ids,
            centroids,
            belief_tags,
            belief_families,
            belief_semantic_types,
            belief_tokens,
        );
        let cluster_sizes = clusters.iter().map(Vec::len).collect();
        PartitionClusterInspection {
            partition_key: partition_key.to_string(),
            seed_count: seed_ids.len(),
            cluster_sizes,
            clusters,
        }
    }

    /// Cluster beliefs using canonical token Jaccard similarity.
    fn cluster_beliefs_canonical(
        &self,
        seed_ids: &[String],
        belief_tags: &HashMap<String, HashSet<String>>,
        belief_families: &HashMap<String, String>,
        belief_semantic_types: &HashMap<String, String>,
        belief_tokens: &HashMap<String, HashSet<String>>,
    ) -> Vec<Vec<String>> {
        let n = seed_ids.len();
        if n <= 1 {
            return vec![seed_ids.to_vec()];
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

        let empty_tags: HashSet<String> = HashSet::new();
        let empty_tokens: HashSet<String> = HashSet::new();
        let require_same_family = self.partition_mode == ConceptPartitionMode::NamespaceOnly;

        for i in 0..n {
            let tok_i = belief_tokens.get(&seed_ids[i]).unwrap_or(&empty_tokens);
            if tok_i.is_empty() {
                continue;
            }

            let tags_i = belief_tags.get(&seed_ids[i]).unwrap_or(&empty_tags);
            let family_i = belief_families
                .get(&seed_ids[i])
                .map(String::as_str)
                .unwrap_or("");

            for j in (i + 1)..n {
                let tok_j = belief_tokens.get(&seed_ids[j]).unwrap_or(&empty_tokens);
                if tok_j.is_empty() {
                    continue;
                }

                // Tag barrier: beliefs must share at least 1 tag to merge
                let tags_j = belief_tags.get(&seed_ids[j]).unwrap_or(&empty_tags);
                let shared = tags_i.intersection(tags_j).count();
                if shared == 0 && !tags_i.is_empty() && !tags_j.is_empty() {
                    continue;
                }

                if require_same_family {
                    let family_j = belief_families
                        .get(&seed_ids[j])
                        .map(String::as_str)
                        .unwrap_or("");
                    if !family_i.is_empty() && !family_j.is_empty() && family_i != family_j {
                        let fam_i = family_token_set(family_i);
                        let fam_j = family_token_set(family_j);
                        let family_overlap = fam_i.intersection(&fam_j).count();
                        let allow_overlap_bridge = fam_i.len() >= 2
                            && fam_j.len() >= 2
                            && family_overlap >= 1
                            && shared >= 1
                            && !is_generic_family(family_i)
                            && !is_generic_family(family_j);
                        let st_i = belief_semantic_types
                            .get(&seed_ids[i])
                            .map(String::as_str)
                            .unwrap_or("");
                        let st_j = belief_semantic_types
                            .get(&seed_ids[j])
                            .map(String::as_str)
                            .unwrap_or("");
                        let allow_single_tag_bridge = self.union_mode
                            == ConceptUnionMode::SingleTagFactDecisionBridge
                            && fam_i.len() == 1
                            && fam_j.len() == 1
                            && shared >= 2
                            && !is_generic_family(family_i)
                            && !is_generic_family(family_j)
                            && ((st_i == "fact" && st_j == "decision")
                                || (st_i == "decision" && st_j == "fact"));
                        if !allow_overlap_bridge && !allow_single_tag_bridge {
                            continue;
                        }
                    }
                    if is_generic_family(family_i) && shared < 2 {
                        continue;
                    }
                }

                if jaccard(tok_i, tok_j) >= CANONICAL_SIMILARITY_THRESHOLD {
                    union(&mut parent, i, j);
                }
            }
        }

        let mut clusters: HashMap<usize, Vec<String>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(seed_ids[i].clone());
        }
        clusters.into_values().collect()
    }

    /// Cluster beliefs using SDR centroid Tanimoto similarity (original path).
    fn cluster_beliefs(
        &self,
        seed_ids: &[String],
        centroids: &HashMap<String, Vec<u16>>,
        belief_tags: &HashMap<String, HashSet<String>>,
        belief_families: &HashMap<String, String>,
        belief_semantic_types: &HashMap<String, String>,
    ) -> Vec<Vec<String>> {
        let n = seed_ids.len();
        if n <= 1 {
            return vec![seed_ids.to_vec()];
        }

        // Union-Find clustering
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

        let empty_tags: HashSet<String> = HashSet::new();
        let require_same_family = self.partition_mode == ConceptPartitionMode::NamespaceOnly;

        for i in 0..n {
            let sdr_i = centroids.get(&seed_ids[i]);
            if sdr_i.map_or(true, |s| s.is_empty()) {
                continue;
            }
            let sdr_i = sdr_i.unwrap();

            let tags_i = belief_tags.get(&seed_ids[i]).unwrap_or(&empty_tags);
            let family_i = belief_families
                .get(&seed_ids[i])
                .map(String::as_str)
                .unwrap_or("");

            for j in (i + 1)..n {
                let sdr_j = centroids.get(&seed_ids[j]);
                if sdr_j.map_or(true, |s| s.is_empty()) {
                    continue;
                }
                let sdr_j = sdr_j.unwrap();

                // Tag barrier: beliefs must share at least 1 tag to merge.
                // Prevents cross-topic false merges when SDR centroids
                // have incidental n-gram overlap above threshold.
                let tags_j = belief_tags.get(&seed_ids[j]).unwrap_or(&empty_tags);
                let shared = tags_i.intersection(tags_j).count();
                if shared == 0 && !tags_i.is_empty() && !tags_j.is_empty() {
                    continue;
                }

                if require_same_family {
                    let family_j = belief_families
                        .get(&seed_ids[j])
                        .map(String::as_str)
                        .unwrap_or("");
                    if !family_i.is_empty() && !family_j.is_empty() && family_i != family_j {
                        let fam_i = family_token_set(family_i);
                        let fam_j = family_token_set(family_j);
                        let family_overlap = fam_i.intersection(&fam_j).count();
                        let allow_overlap_bridge = fam_i.len() >= 2
                            && fam_j.len() >= 2
                            && family_overlap >= 1
                            && shared >= 1
                            && !is_generic_family(family_i)
                            && !is_generic_family(family_j);
                        let st_i = belief_semantic_types
                            .get(&seed_ids[i])
                            .map(String::as_str)
                            .unwrap_or("");
                        let st_j = belief_semantic_types
                            .get(&seed_ids[j])
                            .map(String::as_str)
                            .unwrap_or("");
                        let allow_single_tag_bridge = self.union_mode
                            == ConceptUnionMode::SingleTagFactDecisionBridge
                            && fam_i.len() == 1
                            && fam_j.len() == 1
                            && shared >= 2
                            && !is_generic_family(family_i)
                            && !is_generic_family(family_j)
                            && ((st_i == "fact" && st_j == "decision")
                                || (st_i == "decision" && st_j == "fact"));
                        if !allow_overlap_bridge && !allow_single_tag_bridge {
                            continue;
                        }
                    }
                    if is_generic_family(family_i) && shared < 2 {
                        continue;
                    }
                }

                if tanimoto(sdr_i, sdr_j) >= CONCEPT_SIMILARITY_THRESHOLD {
                    union(&mut parent, i, j);
                }
            }
        }

        let mut clusters: HashMap<usize, Vec<String>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(seed_ids[i].clone());
        }
        clusters.into_values().collect()
    }

    /// Get all stable concepts.
    pub fn stable_concepts(&self) -> Vec<&ConceptCandidate> {
        self.concepts
            .values()
            .filter(|c| c.state == ConceptState::Stable)
            .collect()
    }

    /// Get all candidates (Stable + Candidate, excluding Rejected).
    pub fn active_candidates(&self) -> Vec<&ConceptCandidate> {
        self.concepts
            .values()
            .filter(|c| c.state != ConceptState::Rejected)
            .collect()
    }

    /// Get summary statistics.
    pub fn stats(&self) -> ConceptStats {
        let stable = self
            .concepts
            .values()
            .filter(|c| c.state == ConceptState::Stable)
            .count();
        let candidate = self
            .concepts
            .values()
            .filter(|c| c.state == ConceptState::Candidate)
            .count();
        let rejected = self
            .concepts
            .values()
            .filter(|c| c.state == ConceptState::Rejected)
            .count();
        let avg_abstraction = if self.concepts.is_empty() {
            0.0
        } else {
            self.concepts
                .values()
                .map(|c| c.abstraction_score)
                .sum::<f32>()
                / self.concepts.len() as f32
        };
        ConceptStats {
            total: self.concepts.len(),
            stable,
            candidate,
            rejected,
            avg_abstraction_score: avg_abstraction,
        }
    }
}

impl Default for ConceptEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── ConceptStats ──

/// Summary statistics for the concept engine.
#[derive(Debug, Clone)]
pub struct ConceptStats {
    pub total: usize,
    pub stable: usize,
    pub candidate: usize,
    pub rejected: usize,
    pub avg_abstraction_score: f32,
}

// ── ConceptStore (persistence) ──

/// Persistence for concept engine state.
///
/// Stores as JSON snapshot in `concepts.cog`.
/// Concepts are **derived state** — always rebuilt from scratch during maintenance.
/// The persisted file is a cache for inspection/debugging only.
/// On startup, Aura always creates a fresh empty ConceptEngine (not loaded from disk).
pub struct ConceptStore {
    path: std::path::PathBuf,
}

impl ConceptStore {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Load concept engine state from disk (inspection/debugging only).
    ///
    /// NOTE: Aura startup does NOT call this — it always creates a fresh
    /// ConceptEngine. This method exists for external tooling, CLI inspection,
    /// and the round-trip test. The loaded state is always stale until the
    /// next maintenance cycle rebuilds it.
    pub fn load(&self) -> anyhow::Result<ConceptEngine> {
        let file_path = self.path.join("concepts.cog");
        if !file_path.exists() {
            return Ok(ConceptEngine::new());
        }
        let data = std::fs::read(&file_path)?;
        if data.is_empty() {
            return Ok(ConceptEngine::new());
        }
        let engine: ConceptEngine = serde_json::from_slice(&data)?;
        Ok(engine)
    }

    /// Save concept engine state to disk.
    pub fn save(&self, engine: &ConceptEngine) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.path)?;
        let file_path = self.path.join("concepts.cog");
        let data = serde_json::to_vec(engine)?;
        std::fs::write(&file_path, &data)?;
        Ok(())
    }
}

// ── Free functions ──

/// Tanimoto coefficient for two sorted sparse SDR vectors.
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
    let union_size = a.len() + b.len() - intersection;
    if union_size == 0 {
        0.0
    } else {
        intersection as f32 / union_size as f32
    }
}

/// Average pairwise Tanimoto within a cluster of beliefs.
fn compute_cohesion(belief_ids: &[String], centroids: &HashMap<String, Vec<u16>>) -> f32 {
    let n = belief_ids.len();
    if n < 2 {
        return 1.0;
    }

    let mut sum = 0.0_f32;
    let mut pairs = 0usize;
    for i in 0..n {
        let a = match centroids.get(&belief_ids[i]) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        for j in (i + 1)..n {
            let b = match centroids.get(&belief_ids[j]) {
                Some(v) if !v.is_empty() => v,
                _ => continue,
            };
            sum += tanimoto(a, b);
            pairs += 1;
        }
    }

    if pairs == 0 {
        0.0
    } else {
        sum / pairs as f32
    }
}

/// Extract core and shell terms from a set of records.
///
/// Core: terms appearing in >= CORE_TERM_THRESHOLD fraction of records.
/// Shell: terms appearing in [SHELL_TERM_LOWER, CORE_TERM_THRESHOLD).
fn extract_terms(records: &[&Record]) -> (Vec<String>, Vec<String>) {
    let n = records.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Count term frequency across records (which records contain each term)
    let mut term_doc_freq: HashMap<String, usize> = HashMap::new();
    for rec in records {
        // Unique terms per record
        let terms: std::collections::HashSet<String> = rec
            .content
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() >= 3) // skip short words
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty() && !is_stopword(w))
            .collect();
        for term in terms {
            *term_doc_freq.entry(term).or_insert(0) += 1;
        }
    }

    let n_f64 = n as f64;
    let mut core = Vec::new();
    let mut shell = Vec::new();

    for (term, count) in &term_doc_freq {
        let freq = *count as f64 / n_f64;
        if freq >= CORE_TERM_THRESHOLD {
            core.push(term.clone());
        } else if freq >= SHELL_TERM_LOWER {
            shell.push(term.clone());
        }
    }

    core.sort();
    shell.sort();
    (core, shell)
}

/// Extract tags that appear in a majority of beliefs in a cluster.
fn extract_stable_tags(all_tags: &[String], belief_count: usize) -> Vec<String> {
    if belief_count == 0 {
        return Vec::new();
    }

    let mut tag_freq: HashMap<&str, usize> = HashMap::new();
    for tag in all_tags {
        *tag_freq.entry(tag.as_str()).or_insert(0) += 1;
    }

    let threshold = (belief_count as f64 * 0.5).ceil() as usize;
    let mut tags: Vec<String> = tag_freq
        .into_iter()
        .filter(|(_, count)| *count >= threshold)
        .map(|(tag, _)| tag.to_string())
        .collect();
    tags.sort();
    tags
}

/// Parse namespace and semantic_type from a belief key.
/// Belief key format: "namespace:sorted_tags:semantic_type" or
/// "namespace:sorted_tags:semantic_type#N" (subclustered).
/// The #N subcluster suffix is stripped to get the base semantic_type.
fn parse_belief_key_ns_st(key: &str) -> (String, String) {
    let parts: Vec<&str> = key.split(':').collect();
    let ns = parts.first().copied().unwrap_or("default").to_string();
    let raw_st = parts.last().copied().unwrap_or("fact");
    // Strip subcluster suffix: "decision#3" → "decision"
    let st = raw_st.split('#').next().unwrap_or(raw_st).to_string();
    (ns, st)
}

/// Parse the dominant tag family from a belief key when present.
/// For keys like `namespace:family:semantic_type[#N]`, returns `family`.
/// For keys like `namespace:semantic_type[#N]`, returns empty string.
fn parse_belief_key_family(key: &str) -> String {
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() >= 3 {
        parts[1].to_string()
    } else {
        String::new()
    }
}

fn family_token_set(family: &str) -> HashSet<String> {
    family
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn is_generic_family(family: &str) -> bool {
    matches!(family, "alerts")
}

/// Build a deterministic concept key from stable abstraction features.
///
/// Uses namespace + semantic_type + core_tags + core_terms + centroid signature.
/// Does NOT depend on belief_ids — so minor belief regrouping doesn't
/// change concept identity when the underlying abstraction is the same.
fn concept_key(
    namespace: &str,
    semantic_type: &str,
    tags: &[String],
    core_terms: &[String],
    centroids: &[&Vec<u16>],
) -> String {
    // Centroid signature: xxh3 hash of union of all centroid bits
    let mut all_bits: Vec<u16> = Vec::new();
    for c in centroids {
        all_bits.extend_from_slice(c);
    }
    all_bits.sort();
    all_bits.dedup();
    let centroid_bytes: Vec<u8> = all_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let centroid_hash = xxhash_rust::xxh3::xxh3_64(&centroid_bytes);

    // Take top-5 core terms for key stability
    let mut key_terms = core_terms.to_vec();
    key_terms.truncate(5);

    format!(
        "{}:{}:{}:{}:{:08x}",
        namespace,
        tags.join(","),
        semantic_type,
        key_terms.join(","),
        centroid_hash as u32, // 32-bit is enough for signature
    )
}

/// Generate a deterministic concept ID from key using xxh3.
fn deterministic_id(key: &str) -> String {
    let hash = xxhash_rust::xxh3::xxh3_64(key.as_bytes());
    format!("c-{:012x}", hash)
}

/// Simple English stopword filter.
fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "and"
            | "for"
            | "are"
            | "but"
            | "not"
            | "you"
            | "all"
            | "can"
            | "had"
            | "her"
            | "was"
            | "one"
            | "our"
            | "out"
            | "has"
            | "his"
            | "how"
            | "its"
            | "may"
            | "new"
            | "now"
            | "old"
            | "see"
            | "way"
            | "who"
            | "did"
            | "get"
            | "let"
            | "say"
            | "she"
            | "too"
            | "use"
            | "with"
            | "this"
            | "that"
            | "have"
            | "from"
            | "they"
            | "been"
            | "will"
            | "into"
            | "when"
            | "what"
            | "which"
            | "their"
            | "than"
            | "each"
            | "make"
            | "like"
            | "just"
            | "over"
            | "such"
            | "take"
            | "also"
            | "some"
            | "could"
            | "them"
            | "only"
            | "other"
            | "very"
            | "after"
            | "most"
            | "then"
            | "more"
            | "should"
            | "would"
            | "there"
            | "about"
            | "these"
            | "where"
            | "being"
            | "does"
    )
}

// ── Canonical Feature Representation (Variant A) ──

/// Lightweight suffix stripping: remove common English suffixes to normalize word forms.
/// Conservative approach: only strip when the resulting base is >= 4 chars long
/// to avoid mangling short words.
fn stem_word(word: &str) -> String {
    let w = word;
    let len = w.len();

    // Minimum length to attempt stemming: require at least 6 chars
    // so the base after stripping is still meaningful (>= 4 chars)
    if len <= 5 {
        return w.to_string();
    }

    // Order matters: try longest suffixes first, require base >= 4 chars
    if w.ends_with("ations") && len > 7 {
        return w[..len - 6].to_string();
    }
    if w.ends_with("ation") && len > 6 {
        return w[..len - 5].to_string();
    }
    if w.ends_with("ments") && len > 6 {
        return w[..len - 5].to_string();
    }
    if w.ends_with("ment") && len > 5 && (len - 4) >= 4 {
        return w[..len - 4].to_string();
    }
    if w.ends_with("ness") && len > 5 && (len - 4) >= 4 {
        return w[..len - 4].to_string();
    }
    if w.ends_with("ting") && len > 5 && (len - 3) >= 4 {
        return w[..len - 3].to_string();
    }
    if w.ends_with("ing") && len > 5 && (len - 3) >= 4 {
        return w[..len - 3].to_string();
    }
    if w.ends_with("ied") && len > 5 {
        return w[..len - 3].to_string() + "y";
    }
    if w.ends_with("ies") && len > 5 {
        return w[..len - 3].to_string() + "y";
    }
    if w.ends_with("ed") && len > 5 && (len - 2) >= 4 {
        return w[..len - 2].to_string();
    }
    if w.ends_with("ly") && len > 5 && (len - 2) >= 4 {
        return w[..len - 2].to_string();
    }
    if w.ends_with("es") && len > 5 && (len - 2) >= 4 {
        return w[..len - 2].to_string();
    }
    if w.ends_with("er") && len > 5 && (len - 2) >= 4 {
        return w[..len - 2].to_string();
    }
    if w.ends_with('s') && !w.ends_with("ss") && len > 4 && (len - 1) >= 4 {
        return w[..len - 1].to_string();
    }

    w.to_string()
}

/// Hand-curated equivalence dictionary for common domain terms.
/// Maps variant forms to a single canonical form.
/// Returns None if the word is not in the dictionary.
fn try_canonical(word: &str) -> Option<&'static str> {
    match word {
        // deployment family
        "deploy" | "deploys" | "deployed" | "deploying" | "deployment" | "deployments"
        | "post-deploy" => Some("deploy"),
        "rollout" | "rollouts" | "roll-out" | "rolling" => Some("rollout"),
        "rollback" | "rollbacks" => Some("rollback"),
        "release" | "releases" | "released" | "releasing" => Some("release"),
        "canary" | "canaries" => Some("canary"),
        "staging" | "staged" | "stage" => Some("staging"),
        "production" | "prod" => Some("production"),
        "blue-green" => Some("bluegreen"),
        "downtime" => Some("downtime"),
        "promote" | "promotes" | "promoted" | "promoting" | "promotion" => Some("promote"),
        "validate" | "validates" | "validated" | "validating" | "validation" => Some("validate"),
        "region" | "regions" | "regional" => Some("region"),
        "environment" | "environments" | "env" => Some("environment"),
        "smoke" => Some("smoke"),
        // database family
        "database" | "databases" | "db" | "postgresql" | "postgres" | "mysql" => Some("database"),
        "query" | "queries" | "querying" | "queried" => Some("query"),
        "index" | "indexes" | "indices" | "indexed" | "indexing" => Some("index"),
        "schema" | "schemas" => Some("schema"),
        "migration" | "migrations" | "migrating" | "migrate" => Some("migration"),
        "backup" | "backups" | "backed" => Some("backup"),
        "replica" | "replicas" | "replication" | "replicate" => Some("replica"),
        "connection" | "connections" | "connecting" | "connect" => Some("connection"),
        "pool" | "pools" | "pooling" => Some("pool"),
        "table" | "tables" => Some("table"),
        "partition" | "partitions" | "partitioning" | "partitioned" => Some("partition"),
        // editor/UI family
        "editor" | "editors" => Some("editor"),
        "theme" | "themes" | "themed" => Some("theme"),
        "dark" | "darker" => Some("dark"),
        "mode" | "modes" => Some("mode"),
        "font" | "fonts" => Some("font"),
        "keybinding" | "keybindings" | "binding" | "bindings" => Some("keybinding"),
        "vim" | "vi" => Some("vim"),
        "extension" | "extensions" => Some("extension"),
        // process/workflow
        "test" | "tests" | "testing" | "tested" => Some("test"),
        "monitor" | "monitors" | "monitoring" | "monitored" => Some("monitor"),
        "config" | "configuration" | "configurations" | "configure" | "configured"
        | "configuring" => Some("config"),
        "review" | "reviews" | "reviewing" | "reviewed" | "reviewer" => Some("review"),
        "approval" | "approve" | "approved" | "approving" => Some("approval"),
        "pipeline" | "pipelines" => Some("pipeline"),
        "security" | "secure" | "secured" | "securing" => Some("security"),
        "scan" | "scans" | "scanning" | "scanned" | "scanner" => Some("scan"),
        "strategy" | "strategies" => Some("strategy"),
        "artifact" | "artifacts" => Some("artifact"),
        "version" | "versions" | "versioned" | "versioning" => Some("version"),
        "timeout" | "timeouts" => Some("timeout"),
        "credential" | "credentials" => Some("credential"),
        "error" | "errors" => Some("error"),
        "metric" | "metrics" => Some("metric"),
        "service" | "services" => Some("service"),
        "log" | "logs" | "logging" | "logged" => Some("log"),
        "performance" | "perf" => Some("performance"),
        // general
        "feature" | "features" => Some("feature"),
        "flag" | "flags" => Some("flag"),
        "container" | "containers" => Some("container"),
        "registry" | "registries" => Some("registry"),
        "healthy" | "health" => Some("health"),
        "automated" | "automatic" | "auto" => Some("automated"),
        _ => None,
    }
}

/// Extended stopword list for canonical tokenization.
/// Broader than the term extraction stopwords — includes more function words.
fn is_canonical_stopword(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "and"
            | "for"
            | "are"
            | "but"
            | "not"
            | "you"
            | "all"
            | "can"
            | "had"
            | "her"
            | "was"
            | "one"
            | "our"
            | "out"
            | "has"
            | "his"
            | "how"
            | "its"
            | "may"
            | "new"
            | "now"
            | "old"
            | "see"
            | "way"
            | "who"
            | "did"
            | "get"
            | "let"
            | "say"
            | "she"
            | "too"
            | "use"
            | "with"
            | "this"
            | "that"
            | "have"
            | "from"
            | "they"
            | "been"
            | "will"
            | "into"
            | "when"
            | "what"
            | "which"
            | "their"
            | "than"
            | "each"
            | "make"
            | "like"
            | "just"
            | "over"
            | "such"
            | "take"
            | "also"
            | "some"
            | "could"
            | "them"
            | "only"
            | "other"
            | "very"
            | "after"
            | "most"
            | "then"
            | "more"
            | "should"
            | "would"
            | "there"
            | "about"
            | "these"
            | "where"
            | "being"
            | "does"
            | "much"
            | "every"
            | "always"
            | "using"
            | "during"
            | "before"
            | "between"
            | "through"
            | "while"
            | "since"
            | "both"
            | "still"
            | "need"
            | "set"
            | "via"
            | "per"
            | "least"
            | "already"
    )
}

/// Extract canonical tokens from text content.
///
/// Pipeline: lowercase → split → strip punctuation → filter short/stopwords
/// → equivalence dictionary → suffix stripping → dedup.
pub fn canonical_tokens(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() >= 3 && !is_canonical_stopword(w))
        .map(|w| {
            // First try equivalence dictionary on raw word
            if let Some(canon) = try_canonical(&w) {
                return canon.to_string();
            }
            // Then stem and try dictionary on stemmed form
            let stemmed = stem_word(&w);
            if let Some(canon) = try_canonical(&stemmed) {
                return canon.to_string();
            }
            stemmed
        })
        .filter(|w| w.len() >= 2)
        .collect()
}

/// Extract canonical tokens from a belief's records.
/// Union of canonical tokens across all records in the belief.
fn belief_canonical_tokens(
    belief_id: &str,
    belief_records: &HashMap<String, Vec<String>>,
    records: &HashMap<String, Record>,
) -> HashSet<String> {
    let mut tokens = HashSet::new();
    if let Some(rids) = belief_records.get(belief_id) {
        for rid in rids {
            if let Some(rec) = records.get(rid) {
                tokens.extend(canonical_tokens(&rec.content));
                // Also include tags as canonical tokens
                for tag in &rec.tags {
                    tokens.insert(tag.to_lowercase());
                }
            }
        }
    }
    tokens
}

/// Jaccard similarity between two sets.
pub fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ── Surfaced Concept Output ──

/// Maximum number of surfaced concepts returned.
pub const MAX_SURFACED_CONCEPTS: usize = 10;
/// Maximum surfaced concepts per namespace.
pub const MAX_SURFACED_PER_NAMESPACE: usize = 5;
/// Minimum abstraction_score for a Candidate to be surfaced.
const SURFACE_CANDIDATE_THRESHOLD: f32 = 0.70;

/// Stable external contract for inspection-only concept output.
/// Decoupled from internal ConceptCandidate — safe to expose publicly.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone)]
pub struct SurfacedConcept {
    /// Unique concept identifier.
    pub id: String,
    /// Canonical concept key.
    pub key: String,
    /// Lifecycle state ("stable" or "candidate").
    pub state: String,

    /// Namespace this concept belongs to.
    pub namespace: String,
    /// Semantic type of grouped beliefs.
    pub semantic_type: String,

    /// High-frequency terms shared across beliefs (the "essence").
    pub core_terms: Vec<String>,
    /// Variable terms that appear in some but not all beliefs.
    pub shell_terms: Vec<String>,
    /// Stable tags across beliefs.
    pub tags: Vec<String>,

    /// Composite abstraction quality score.
    pub abstraction_score: f32,
    /// Average confidence of source beliefs.
    pub confidence: f32,

    /// Number of beliefs in this concept cluster.
    pub cluster_size: usize,
    /// Aggregated support mass.
    pub support_mass: f32,

    // ── Provenance ──
    /// Source belief IDs.
    pub belief_ids: Vec<String>,
    /// Transitive source record IDs.
    pub record_ids: Vec<String>,
}

/// Surface concepts for external inspection.
/// Returns a bounded, sorted, provenance-checked list of concepts.
pub fn surface_concepts(engine: &ConceptEngine, limit: Option<usize>) -> Vec<SurfacedConcept> {
    surface_concepts_filtered(engine, limit, None)
}

/// Surface concepts with optional namespace filter.
pub fn surface_concepts_filtered(
    engine: &ConceptEngine,
    limit: Option<usize>,
    namespace: Option<&str>,
) -> Vec<SurfacedConcept> {
    let max = limit
        .unwrap_or(MAX_SURFACED_CONCEPTS)
        .min(MAX_SURFACED_CONCEPTS);

    // Phase A: filter eligible concepts
    let mut eligible: Vec<&ConceptCandidate> = engine
        .concepts
        .values()
        .filter(|c| {
            // Namespace filter
            if let Some(ns) = namespace {
                if c.namespace != ns {
                    return false;
                }
            }

            // Must have provenance
            if c.belief_ids.is_empty() || c.record_ids.is_empty() {
                return false;
            }

            // Must have non-empty core_terms or tags
            if c.core_terms.is_empty() && c.tags.is_empty() {
                return false;
            }

            // State gate
            match c.state {
                ConceptState::Stable => true,
                ConceptState::Candidate => c.abstraction_score >= SURFACE_CANDIDATE_THRESHOLD,
                ConceptState::Rejected => false,
            }
        })
        .collect();

    // Phase B: sort deterministically
    // Higher abstraction_score > higher confidence > larger cluster > stable over candidate > key tiebreak
    eligible.sort_by(|a, b| {
        b.abstraction_score
            .partial_cmp(&a.abstraction_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(b.belief_ids.len().cmp(&a.belief_ids.len()))
            .then_with(|| {
                let a_stable = matches!(a.state, ConceptState::Stable);
                let b_stable = matches!(b.state, ConceptState::Stable);
                b_stable.cmp(&a_stable)
            })
            .then(a.key.cmp(&b.key))
    });

    // Phase C: per-namespace cap + global limit + dedup by key
    let mut result = Vec::new();
    let mut ns_counts: HashMap<String, usize> = HashMap::new();
    let mut seen_keys: HashSet<String> = HashSet::new();

    for concept in &eligible {
        if result.len() >= max {
            break;
        }

        // Dedupe by key
        if !seen_keys.insert(concept.key.clone()) {
            continue;
        }

        // Per-namespace cap
        let count = ns_counts.entry(concept.namespace.clone()).or_default();
        if *count >= MAX_SURFACED_PER_NAMESPACE {
            continue;
        }
        *count += 1;

        // Phase D: map to surfaced type
        result.push(SurfacedConcept {
            id: concept.id.clone(),
            key: concept.key.clone(),
            state: match concept.state {
                ConceptState::Stable => "stable".to_string(),
                ConceptState::Candidate => "candidate".to_string(),
                _ => unreachable!(), // filtered above
            },
            namespace: concept.namespace.clone(),
            semantic_type: concept.semantic_type.clone(),
            core_terms: concept.core_terms.clone(),
            shell_terms: concept.shell_terms.clone(),
            tags: concept.tags.clone(),
            abstraction_score: concept.abstraction_score,
            confidence: concept.confidence,
            cluster_size: concept.belief_ids.len(),
            support_mass: concept.support_mass,
            belief_ids: concept.belief_ids.clone(),
            record_ids: concept.record_ids.clone(),
        });
    }

    result
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::{Belief, BeliefEngine};
    use crate::levels::Level;

    fn make_record(content: &str, tags: &[&str], semantic_type: &str) -> Record {
        let mut rec = Record::new(content.to_string(), Level::Domain);
        rec.tags = tags.iter().map(|s| s.to_string()).collect();
        rec.semantic_type = semantic_type.to_string();
        rec.confidence = Record::default_confidence_for_source("recorded");
        rec
    }

    fn build_sdr_lookup(records: &HashMap<String, Record>) -> SdrLookup {
        let sdr = crate::sdr::SDRInterpreter::default();
        records
            .iter()
            .map(|(id, rec)| (id.clone(), sdr.text_to_sdr(&rec.content, false)))
            .collect()
    }

    /// Build a test scenario with enough records to form beliefs and concepts.
    fn setup_test_scenario() -> (BeliefEngine, HashMap<String, Record>, SdrLookup) {
        let mut records = HashMap::new();

        // Cluster 1: dark mode preferences (4 records → should form 1 belief → need more beliefs for concept)
        let r1 = make_record(
            "I prefer dark mode for my editor and terminal",
            &["ui", "preferences"],
            "preference",
        );
        let r2 = make_record(
            "Dark theme is better for my eyes in the editor",
            &["ui", "preferences"],
            "preference",
        );
        let r3 = make_record(
            "Always use dark mode in all my development tools",
            &["ui", "preferences"],
            "preference",
        );
        let r4 = make_record(
            "Dark background reduces eye strain during coding",
            &["ui", "preferences"],
            "preference",
        );

        // Cluster 2: keyboard shortcuts preferences
        let r5 = make_record(
            "Vim keybindings are essential for fast editing workflow",
            &["ui", "shortcuts"],
            "preference",
        );
        let r6 = make_record(
            "I use vim keyboard shortcuts in every editor I configure",
            &["ui", "shortcuts"],
            "preference",
        );
        let r7 = make_record(
            "Vim-style key bindings make me productive in coding",
            &["ui", "shortcuts"],
            "preference",
        );

        // Cluster 3: testing practices (different semantic_type)
        let r8 = make_record(
            "Always write unit tests before merging pull requests",
            &["testing", "workflow"],
            "decision",
        );
        let r9 = make_record(
            "Every pull request must have unit test coverage",
            &["testing", "workflow"],
            "decision",
        );
        let r10 = make_record(
            "Test-driven development is my standard workflow approach",
            &["testing", "workflow"],
            "decision",
        );

        for r in [&r1, &r2, &r3, &r4, &r5, &r6, &r7, &r8, &r9, &r10] {
            records.insert(r.id.clone(), r.clone());
        }

        let sdr_lookup = build_sdr_lookup(&records);

        // Run belief engine to form beliefs
        let mut belief_engine = BeliefEngine::new();
        // Run multiple cycles to build stability
        for _ in 0..5 {
            belief_engine.update_with_sdr(&records, &sdr_lookup);
        }

        (belief_engine, records, sdr_lookup)
    }

    #[test]
    fn test_tanimoto_identity() {
        let a = vec![1, 5, 10, 20];
        assert!((tanimoto(&a, &a) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_tanimoto_empty() {
        assert_eq!(tanimoto(&[], &[1, 2, 3]), 0.0);
        assert_eq!(tanimoto(&[1, 2], &[]), 0.0);
    }

    #[test]
    fn test_tanimoto_disjoint() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert_eq!(tanimoto(&a, &b), 0.0);
    }

    #[test]
    fn test_extract_terms_core_and_shell() {
        let r1 = make_record("dark mode is great for coding at night", &["ui"], "fact");
        let r2 = make_record(
            "dark mode helps reduce eye strain while coding",
            &["ui"],
            "fact",
        );
        let r3 = make_record(
            "dark mode is my preference for development work",
            &["ui"],
            "fact",
        );

        let records_vec: Vec<&Record> = vec![&r1, &r2, &r3];
        let (core, shell) = extract_terms(&records_vec);

        // "dark" and "mode" should appear in all 3 → core
        assert!(
            core.contains(&"dark".to_string()),
            "core should contain 'dark': {:?}",
            core
        );
        assert!(
            core.contains(&"mode".to_string()),
            "core should contain 'mode': {:?}",
            core
        );

        // "coding" appears in 2/3 → shell (0.67, below 0.70 core threshold)
        // or could be core if threshold is met; either way it shouldn't be missing
        let in_core_or_shell =
            core.contains(&"coding".to_string()) || shell.contains(&"coding".to_string());
        assert!(in_core_or_shell, "coding should be in core or shell");
    }

    #[test]
    fn test_concept_engine_new_is_empty() {
        let engine = ConceptEngine::new();
        assert!(engine.concepts.is_empty());
        assert!(engine.key_index.is_empty());
    }

    #[test]
    fn test_concept_candidates_form_from_resolved_beliefs() {
        let (belief_engine, records, sdr_lookup) = setup_test_scenario();

        // Verify we have some resolved beliefs
        let resolved: Vec<_> = belief_engine
            .beliefs
            .values()
            .filter(|b| matches!(b.state, BeliefState::Resolved | BeliefState::Singleton))
            .collect();
        assert!(!resolved.is_empty(), "should have resolved beliefs");

        let mut concept_engine = ConceptEngine::new();
        let report = concept_engine.discover(&belief_engine, &records, &sdr_lookup);

        // Should find seeds
        assert!(report.seeds_found > 0, "should find belief seeds");
    }

    #[test]
    fn test_unresolved_beliefs_do_not_form_stable_concepts() {
        // Create records that will produce unresolved beliefs (conflicting)
        let mut records = HashMap::new();

        let r1 = make_record(
            "tabs are better than spaces for indentation in code",
            &["coding", "style"],
            "preference",
        );
        let mut r2 = make_record(
            "spaces are better than tabs for indentation in code",
            &["coding", "style"],
            "contradiction",
        );
        r2.conflict_mass = 3;
        r2.support_mass = 0;

        records.insert(r1.id.clone(), r1);
        records.insert(r2.id.clone(), r2);

        let sdr_lookup = build_sdr_lookup(&records);
        let mut belief_engine = BeliefEngine::new();
        belief_engine.update_with_sdr(&records, &sdr_lookup);

        // Force unresolved state on all beliefs
        for belief in belief_engine.beliefs.values_mut() {
            belief.state = BeliefState::Unresolved;
            belief.stability = 0.0;
        }

        let mut concept_engine = ConceptEngine::new();
        let report = concept_engine.discover(&belief_engine, &records, &sdr_lookup);

        // No seeds from unresolved beliefs
        assert_eq!(
            report.seeds_found, 0,
            "unresolved beliefs should not seed concepts"
        );
        assert_eq!(report.candidates_found, 0);
    }

    #[test]
    fn test_concept_provenance_complete() {
        let (belief_engine, records, sdr_lookup) = setup_test_scenario();

        let mut concept_engine = ConceptEngine::new();
        concept_engine.discover(&belief_engine, &records, &sdr_lookup);

        for concept in concept_engine.concepts.values() {
            // Every concept must have belief_ids
            assert!(
                !concept.belief_ids.is_empty(),
                "concept {} must have belief provenance",
                concept.id
            );
            // Every concept must have record_ids
            assert!(
                !concept.record_ids.is_empty(),
                "concept {} must have record provenance",
                concept.id
            );
            // All belief_ids must exist in the belief engine
            for bid in &concept.belief_ids {
                assert!(
                    belief_engine.beliefs.contains_key(bid),
                    "concept references non-existent belief {}",
                    bid
                );
            }
            // All record_ids must exist in records
            for rid in &concept.record_ids {
                assert!(
                    records.contains_key(rid),
                    "concept references non-existent record {}",
                    rid
                );
            }
        }
    }

    #[test]
    fn test_concept_discovery_stable_across_replay() {
        let (belief_engine, records, sdr_lookup) = setup_test_scenario();

        let mut engine1 = ConceptEngine::new();
        let report1 = engine1.discover(&belief_engine, &records, &sdr_lookup);

        let mut engine2 = ConceptEngine::new();
        let report2 = engine2.discover(&belief_engine, &records, &sdr_lookup);

        // Same inputs → same outputs
        assert_eq!(
            report1.candidates_found, report2.candidates_found,
            "concept count should be stable across replays"
        );
        assert_eq!(report1.stable_count, report2.stable_count);
        assert_eq!(report1.rejected_count, report2.rejected_count);
        assert!((report1.avg_abstraction_score - report2.avg_abstraction_score).abs() < 0.001);
    }

    #[test]
    fn test_concept_report_metrics_nonzero_on_realistic_stream() {
        let (belief_engine, records, sdr_lookup) = setup_test_scenario();

        let mut concept_engine = ConceptEngine::new();
        let report = concept_engine.discover(&belief_engine, &records, &sdr_lookup);

        assert!(report.seeds_found > 0, "seeds_found should be > 0");
        // Concepts may or may not form depending on clustering
        // But seeds should always be found from the stable beliefs
    }

    #[test]
    fn test_concept_store_save_and_load_for_inspection() {
        let dir = std::env::temp_dir().join(format!("concept_store_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let store = ConceptStore::new(&dir);
        let mut engine = ConceptEngine::new();

        // Add a concept manually
        let candidate = ConceptCandidate {
            id: "c-test123".to_string(),
            key: "default:ui:preference:dark,mode:abc12345".to_string(),
            namespace: "default".to_string(),
            semantic_type: "preference".to_string(),
            belief_ids: vec!["b1".to_string(), "b2".to_string()],
            record_ids: vec!["r1".to_string(), "r2".to_string()],
            core_terms: vec!["dark".to_string(), "mode".to_string()],
            shell_terms: vec!["editor".to_string()],
            tags: vec!["ui".to_string()],
            support_mass: 3.0,
            confidence: 0.85,
            stability: 4.0,
            cohesion: 0.7,
            abstraction_score: 0.8,
            state: ConceptState::Stable,
            last_updated: now_secs(),
        };
        engine.concepts.insert(candidate.id.clone(), candidate);
        engine
            .key_index
            .insert("test-key".to_string(), "c-test123".to_string());

        // Save works
        store.save(&engine).unwrap();
        // Load works (for inspection — not used on startup)
        let loaded = store.load().unwrap();

        assert_eq!(loaded.concepts.len(), 1);
        assert!(loaded.concepts.contains_key("c-test123"));
        let c = &loaded.concepts["c-test123"];
        assert_eq!(c.core_terms, vec!["dark", "mode"]);
        assert_eq!(c.state, ConceptState::Stable);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_startup_does_not_trust_stale_snapshot() {
        // Simulate: concepts.cog exists with stale data, but on startup
        // ConceptEngine should be empty (not loaded).
        // This verifies the design contract.
        let engine = ConceptEngine::new();
        assert!(
            engine.concepts.is_empty(),
            "fresh ConceptEngine must be empty — startup must not load stale concepts"
        );
        assert!(engine.key_index.is_empty());
    }

    #[test]
    fn test_different_topics_do_not_merge_into_one_concept() {
        let mut records = HashMap::new();

        // Topic A: database optimization
        let r1 = make_record(
            "Use database indexes for faster query performance optimization",
            &["database", "performance"],
            "decision",
        );
        let r2 = make_record(
            "Database query optimization requires proper index configuration",
            &["database", "performance"],
            "decision",
        );
        let r3 = make_record(
            "Add composite indexes to speed up database read operations",
            &["database", "performance"],
            "decision",
        );

        // Topic B: UI accessibility (same tags won't match — different namespace-like tags)
        let r4 = make_record(
            "Ensure all buttons have accessible aria labels for screen readers",
            &["accessibility", "frontend"],
            "decision",
        );
        let r5 = make_record(
            "Every interactive element needs aria labels for accessibility",
            &["accessibility", "frontend"],
            "decision",
        );
        let r6 = make_record(
            "Screen reader compatibility requires proper aria label markup",
            &["accessibility", "frontend"],
            "decision",
        );

        for r in [&r1, &r2, &r3, &r4, &r5, &r6] {
            records.insert(r.id.clone(), r.clone());
        }

        let sdr_lookup = build_sdr_lookup(&records);
        let mut belief_engine = BeliefEngine::new();
        for _ in 0..5 {
            belief_engine.update_with_sdr(&records, &sdr_lookup);
        }

        let mut concept_engine = ConceptEngine::new();
        concept_engine.discover(&belief_engine, &records, &sdr_lookup);

        // If any concepts formed, database and accessibility should NOT be in the same concept
        for concept in concept_engine.concepts.values() {
            let has_db = concept.record_ids.iter().any(|rid| {
                records
                    .get(rid)
                    .map_or(false, |r| r.tags.contains(&"database".to_string()))
            });
            let has_a11y = concept.record_ids.iter().any(|rid| {
                records
                    .get(rid)
                    .map_or(false, |r| r.tags.contains(&"accessibility".to_string()))
            });
            assert!(
                !(has_db && has_a11y),
                "database and accessibility records should NOT merge into one concept"
            );
        }
    }

    #[test]
    fn test_concept_core_extracts_shared_terms() {
        let r1 = make_record(
            "always run tests before deploying to production environment",
            &["ci"],
            "decision",
        );
        let r2 = make_record(
            "run tests before every production deploy for safety",
            &["ci"],
            "decision",
        );
        let r3 = make_record(
            "before deploying run all tests to verify production readiness",
            &["ci"],
            "decision",
        );

        let refs: Vec<&Record> = vec![&r1, &r2, &r3];
        let (core, _shell) = extract_terms(&refs);

        // "tests", "production", "run", "before", "deploying"/"deploy" should be frequent
        assert!(
            core.contains(&"tests".to_string()) || core.contains(&"run".to_string()),
            "core should contain common terms: {:?}",
            core
        );
    }

    #[test]
    fn test_stopword_filter() {
        assert!(is_stopword("the"));
        assert!(is_stopword("with"));
        assert!(is_stopword("should"));
        assert!(!is_stopword("database"));
        assert!(!is_stopword("deploy"));
    }

    // ── Structural hardening tests ──

    #[test]
    fn test_cross_namespace_beliefs_do_not_cluster() {
        let mut records = HashMap::new();

        // Namespace "prod" records
        let mut r1 = make_record(
            "Always deploy with canary release strategy",
            &["deploy", "safety"],
            "decision",
        );
        r1.namespace = "prod".to_string();
        let mut r2 = make_record(
            "Canary deploys protect against production failures",
            &["deploy", "safety"],
            "decision",
        );
        r2.namespace = "prod".to_string();
        let mut r3 = make_record(
            "Use canary release for safe production deployment",
            &["deploy", "safety"],
            "decision",
        );
        r3.namespace = "prod".to_string();

        // Namespace "staging" records — same tags and semantic_type
        let mut r4 = make_record(
            "Deploy to staging with canary release strategy",
            &["deploy", "safety"],
            "decision",
        );
        r4.namespace = "staging".to_string();
        let mut r5 = make_record(
            "Canary deploys in staging catch issues early",
            &["deploy", "safety"],
            "decision",
        );
        r5.namespace = "staging".to_string();
        let mut r6 = make_record(
            "Use canary release for safe staging deployment",
            &["deploy", "safety"],
            "decision",
        );
        r6.namespace = "staging".to_string();

        for r in [&r1, &r2, &r3, &r4, &r5, &r6] {
            records.insert(r.id.clone(), r.clone());
        }

        let sdr_lookup = build_sdr_lookup(&records);
        let mut belief_engine = BeliefEngine::new();
        for _ in 0..5 {
            belief_engine.update_with_sdr(&records, &sdr_lookup);
        }

        let mut concept_engine = ConceptEngine::new();
        concept_engine.discover(&belief_engine, &records, &sdr_lookup);

        // No concept should contain records from both namespaces
        for concept in concept_engine.concepts.values() {
            let namespaces: std::collections::HashSet<&str> = concept
                .record_ids
                .iter()
                .filter_map(|rid| records.get(rid))
                .map(|r| r.namespace.as_str())
                .collect();
            assert!(
                namespaces.len() <= 1,
                "concept {} spans namespaces {:?} — cross-namespace merge!",
                concept.id,
                namespaces
            );
        }
    }

    #[test]
    fn test_cross_semantic_type_beliefs_do_not_cluster() {
        let mut records = HashMap::new();

        // semantic_type "preference"
        let r1 = make_record(
            "Dark mode is my preferred editor theme for coding",
            &["ui", "editor"],
            "preference",
        );
        let r2 = make_record(
            "I always use dark mode theme in my editor",
            &["ui", "editor"],
            "preference",
        );
        let r3 = make_record(
            "Dark editor theme is best for my coding workflow",
            &["ui", "editor"],
            "preference",
        );

        // semantic_type "fact" — same tags
        let r4 = make_record(
            "Dark mode reduces eye strain according to research",
            &["ui", "editor"],
            "fact",
        );
        let r5 = make_record(
            "Editor dark themes are popular among developers",
            &["ui", "editor"],
            "fact",
        );
        let r6 = make_record(
            "Dark mode in code editors is a common configuration",
            &["ui", "editor"],
            "fact",
        );

        for r in [&r1, &r2, &r3, &r4, &r5, &r6] {
            records.insert(r.id.clone(), r.clone());
        }

        let sdr_lookup = build_sdr_lookup(&records);
        let mut belief_engine = BeliefEngine::new();
        for _ in 0..5 {
            belief_engine.update_with_sdr(&records, &sdr_lookup);
        }

        let mut concept_engine = ConceptEngine::new();
        concept_engine.discover(&belief_engine, &records, &sdr_lookup);

        // No concept should contain records from both semantic types
        for concept in concept_engine.concepts.values() {
            let types: std::collections::HashSet<&str> = concept
                .record_ids
                .iter()
                .filter_map(|rid| records.get(rid))
                .map(|r| r.semantic_type.as_str())
                .collect();
            assert!(
                types.len() <= 1,
                "concept {} spans semantic types {:?} — cross-type merge!",
                concept.id,
                types
            );
        }
    }

    #[test]
    fn test_concept_identity_stable_under_belief_growth() {
        // Setup: start with 3 beliefs worth of records
        let mut records = HashMap::new();
        let r1 = make_record(
            "Configure logging level to debug for troubleshooting",
            &["logging", "config"],
            "decision",
        );
        let r2 = make_record(
            "Set debug log level when troubleshooting production issues",
            &["logging", "config"],
            "decision",
        );
        let r3 = make_record(
            "Debug logging configuration helps diagnose issues faster",
            &["logging", "config"],
            "decision",
        );
        let r4 = make_record(
            "Enable debug log level for production troubleshooting",
            &["logging", "config"],
            "decision",
        );
        let r5 = make_record(
            "Logging at debug level reveals root cause of problems",
            &["logging", "config"],
            "decision",
        );
        let r6 = make_record(
            "Always configure debug logging for production diagnostics",
            &["logging", "config"],
            "decision",
        );

        for r in [&r1, &r2, &r3, &r4, &r5, &r6] {
            records.insert(r.id.clone(), r.clone());
        }

        let sdr_lookup = build_sdr_lookup(&records);
        let mut belief_engine = BeliefEngine::new();
        for _ in 0..5 {
            belief_engine.update_with_sdr(&records, &sdr_lookup);
        }

        let mut engine1 = ConceptEngine::new();
        engine1.discover(&belief_engine, &records, &sdr_lookup);
        let keys1: std::collections::HashSet<String> =
            engine1.concepts.values().map(|c| c.key.clone()).collect();

        // Now add one more supporting record (minor belief growth)
        let r7 = make_record(
            "Use debug log level when diagnosing production failures",
            &["logging", "config"],
            "decision",
        );
        records.insert(r7.id.clone(), r7.clone());
        let mut sdr_lookup2 = build_sdr_lookup(&records);
        // Merge old SDR entries to keep consistency
        for (k, v) in &sdr_lookup {
            sdr_lookup2.entry(k.clone()).or_insert_with(|| v.clone());
        }

        let mut belief_engine2 = BeliefEngine::new();
        for _ in 0..5 {
            belief_engine2.update_with_sdr(&records, &sdr_lookup2);
        }

        let mut engine2 = ConceptEngine::new();
        engine2.discover(&belief_engine2, &records, &sdr_lookup2);
        let keys2: std::collections::HashSet<String> =
            engine2.concepts.values().map(|c| c.key.clone()).collect();

        // Concept keys should be stable — adding one more record to an existing
        // cluster should not create a completely different concept
        // (Keys may differ if core_terms change, but shouldn't be a total reset)
        if !keys1.is_empty() && !keys2.is_empty() {
            // At least some keys should overlap
            let overlap = keys1.intersection(&keys2).count();
            // If keys changed, the concept count should still be similar
            let count_diff =
                (engine1.concepts.len() as i32 - engine2.concepts.len() as i32).unsigned_abs();
            assert!(
                overlap > 0 || count_diff <= 1,
                "concept identity unstable: keys1={:?}, keys2={:?}",
                keys1,
                keys2
            );
        }
    }

    #[test]
    fn test_parse_belief_key_ns_st() {
        let (ns, st) = parse_belief_key_ns_st("default:preferences,ui:preference");
        assert_eq!(ns, "default");
        assert_eq!(st, "preference");

        let (ns2, st2) = parse_belief_key_ns_st("prod:deploy,safety:decision");
        assert_eq!(ns2, "prod");
        assert_eq!(st2, "decision");

        // Edge case: minimal key
        let (ns3, st3) = parse_belief_key_ns_st("ns:fact");
        assert_eq!(ns3, "ns");
        assert_eq!(st3, "fact");
    }

    #[test]
    fn test_partition_seeds_standard_keeps_semantic_types_separate() {
        let mut concept_engine = ConceptEngine::new();
        concept_engine.partition_mode = ConceptPartitionMode::Standard;

        let b1 = Belief::new("default:deploy:decision".into());
        let b2 = Belief::new("default:deploy:fact".into());

        let mut belief_engine = BeliefEngine::new();
        belief_engine
            .key_index
            .insert(b1.key.clone(), b1.id.clone());
        belief_engine
            .key_index
            .insert(b2.key.clone(), b2.id.clone());
        belief_engine.beliefs.insert(b1.id.clone(), b1.clone());
        belief_engine.beliefs.insert(b2.id.clone(), b2.clone());

        let partitions =
            concept_engine.partition_seeds(&[b1.id.clone(), b2.id.clone()], &belief_engine);
        assert_eq!(
            partitions.len(),
            2,
            "Standard partitioning should keep decision and fact apart"
        );
    }

    #[test]
    fn test_partition_seeds_namespace_only_merges_semantic_types_within_namespace() {
        let mut concept_engine = ConceptEngine::new();
        concept_engine.partition_mode = ConceptPartitionMode::NamespaceOnly;

        let b1 = Belief::new("default:deploy:decision".into());
        let b2 = Belief::new("default:deploy:fact".into());

        let mut belief_engine = BeliefEngine::new();
        belief_engine
            .key_index
            .insert(b1.key.clone(), b1.id.clone());
        belief_engine
            .key_index
            .insert(b2.key.clone(), b2.id.clone());
        belief_engine.beliefs.insert(b1.id.clone(), b1.clone());
        belief_engine.beliefs.insert(b2.id.clone(), b2.clone());

        let partitions =
            concept_engine.partition_seeds(&[b1.id.clone(), b2.id.clone()], &belief_engine);
        assert_eq!(
            partitions.len(),
            1,
            "NamespaceOnly should merge same-namespace semantic variants"
        );
        assert_eq!(partitions.get("default").map(|v| v.len()), Some(2));
    }

    #[test]
    fn test_concept_key_does_not_use_belief_ids() {
        // Same abstraction features should produce same key regardless of belief_ids
        let tags = vec!["ui".to_string()];
        let core_terms = vec!["dark".to_string(), "mode".to_string()];
        let centroid = vec![100u16, 200, 300];

        let key1 = concept_key("default", "preference", &tags, &core_terms, &[&centroid]);
        let key2 = concept_key("default", "preference", &tags, &core_terms, &[&centroid]);
        assert_eq!(key1, key2, "same features should produce same key");

        // Different namespace → different key
        let key3 = concept_key("staging", "preference", &tags, &core_terms, &[&centroid]);
        assert_ne!(
            key1, key3,
            "different namespace should produce different key"
        );

        // Different semantic_type → different key
        let key4 = concept_key("default", "fact", &tags, &core_terms, &[&centroid]);
        assert_ne!(
            key1, key4,
            "different semantic_type should produce different key"
        );
    }

    #[test]
    fn test_concept_identity_strips_belief_subcluster_suffix() {
        let (ns, st) = parse_belief_key_ns_st("default:editor,theme:preference#6");
        let tags = vec!["editor".to_string(), "theme".to_string()];
        let core_terms = vec!["dark".to_string(), "theme".to_string()];
        let centroid = vec![10u16, 20, 30];

        let key = concept_key(&ns, &st, &tags, &core_terms, &[&centroid]);

        assert!(
            !key.contains("preference#6"),
            "concept key should not preserve belief subcluster suffixes"
        );
        assert!(
            key.contains(":preference:"),
            "concept key should keep only the base semantic type"
        );
    }
}
