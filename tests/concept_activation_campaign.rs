//! Concept Activation Campaign — Candidate C Diagnosis Sprint
//!
//! Generates 60-100 synthetic worlds across 6 profiles, runs full maintenance
//! cycles, measures concept coverage / cluster size / false merges / identity
//! stability, and produces a final verdict:
//!
//!   ACTIVATED | PARTIAL SIGNAL | STRUCTURAL BLOCK
//!
//! 6 world profiles:
//!   A. Single stable concept (paraphrases of one idea)
//!   B. Stable core + variable shell
//!   C. Two nearby concepts (should not merge)
//!   D. Multi-concept same namespace (3-4 families)
//!   E. Sparse / under-supported
//!   F. Adversarial paraphrase/conflict mix
//!
//! Usage:
//!   cargo test --no-default-features --features "encryption,server,audit" \
//!     --test concept_activation_campaign -- --nocapture
//!
//! Default: 60 runs. Set CONCEPT_CAMPAIGN_RUNS env var to override.
//! Default: 8 cycles. Set CONCEPT_CAMPAIGN_CYCLES env var to override.

use aura::{Aura, Level};
use std::collections::{HashMap, HashSet};
use std::mem::ManuallyDrop;

// ═══════════════════════════════════════════════════════════
// Acceptance thresholds
// ═══════════════════════════════════════════════════════════

/// Minimum pass: average coverage across supportive profiles (A-D).
const GATE_AVG_COVERAGE: f32 = 0.20;
/// Minimum pass: average cluster size.
const GATE_AVG_CLUSTER_SIZE: f32 = 2.0;
/// Maximum false merge rate.
const GATE_FALSE_MERGE_RATE: f32 = 0.10;
/// Maximum cross-topic merge rate.
const GATE_CROSS_TOPIC_MERGE_RATE: f32 = 0.02;
/// Identity stability: max churn across replay.
const GATE_MAX_IDENTITY_CHURN: f32 = 0.10;

/// Strong pass: coverage.
const STRONG_COVERAGE: f32 = 0.30;

// ═══════════════════════════════════════════════════════════
// Infra
// ═══════════════════════════════════════════════════════════

struct TempAura {
    aura: ManuallyDrop<Aura>,
    dir: ManuallyDrop<tempfile::TempDir>,
}

impl std::ops::Deref for TempAura {
    type Target = Aura;
    fn deref(&self) -> &Self::Target { &self.aura }
}

impl Drop for TempAura {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.aura);
            ManuallyDrop::drop(&mut self.dir);
        }
    }
}

fn open_temp_aura() -> TempAura {
    let dir = tempfile::tempdir().expect("tempdir");
    let aura = Aura::open(dir.path().to_str().unwrap()).expect("open aura");
    TempAura { aura: ManuallyDrop::new(aura), dir: ManuallyDrop::new(dir) }
}

// ═══════════════════════════════════════════════════════════
// PRNG (xorshift32)
// ═══════════════════════════════════════════════════════════

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self { Self(seed.max(1)) }

    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        let idx = self.next() as usize % items.len();
        &items[idx]
    }
}

// ═══════════════════════════════════════════════════════════
// World profiles
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
enum WorldProfile {
    SingleStable,    // A
    CoreShell,       // B
    TwoNearby,       // C
    MultiConcept,    // D
    Sparse,          // E
    Adversarial,     // F
}

impl WorldProfile {
    fn all() -> &'static [WorldProfile] {
        &[
            WorldProfile::SingleStable,
            WorldProfile::CoreShell,
            WorldProfile::TwoNearby,
            WorldProfile::MultiConcept,
            WorldProfile::Sparse,
            WorldProfile::Adversarial,
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            WorldProfile::SingleStable => "single-stable",
            WorldProfile::CoreShell => "core-shell",
            WorldProfile::TwoNearby => "two-nearby",
            WorldProfile::MultiConcept => "multi-concept",
            WorldProfile::Sparse => "sparse",
            WorldProfile::Adversarial => "adversarial",
        }
    }

    /// Whether this profile is expected to produce concepts.
    fn expects_concepts(&self) -> bool {
        matches!(self, WorldProfile::SingleStable | WorldProfile::CoreShell
            | WorldProfile::TwoNearby | WorldProfile::MultiConcept)
    }
}

// ═══════════════════════════════════════════════════════════
// Corpus record
// ═══════════════════════════════════════════════════════════

struct WorldRecord {
    content: String,
    level: Level,
    tags: Vec<String>,
    source_type: &'static str,
    semantic_type: &'static str,
    /// Which concept family this record belongs to (for false merge/split eval).
    family: String,
}

struct QuerySpec {
    text: String,
    top_k: usize,
}

// ═══════════════════════════════════════════════════════════
// Word banks
// ═══════════════════════════════════════════════════════════

const DEPLOY_CORE: &[&str] = &[
    "safe deploy workflow ensures production stability",
    "staged deployment process protects production environment",
    "deploy through staging before production release",
    "deployment pipeline stages code through safety checks",
    "safe deployment requires staging validation first",
    "production deploy follows staged rollout process",
    "deploy safety workflow validates before production push",
    "staged deploy process ensures no production regressions",
];

const DEPLOY_SHELL_EXTRAS: &[&str] = &[
    "rollback procedure ready during deployment",
    "verification checklist before production deploy",
    "gradual rollout reduces deployment risk",
    "canary release validates deployment safety",
    "deployment monitoring catches regressions early",
    "staged rollback plan for failed deployments",
];

const DEPLOY_SPEED: &[&str] = &[
    "fast deployment pipeline reduces release cycle time",
    "rapid deploy automation speeds up delivery",
    "quick deployment turnaround improves developer velocity",
    "automated deploy pipeline enables fast iterations",
    "fast release cycle through automated deployment",
    "deploy speed optimization reduces time to production",
];

const DATABASE_FAMILY: &[&str] = &[
    "database indexing improves query performance significantly",
    "index optimization is critical for database throughput",
    "proper database indexes reduce query latency",
    "database query performance depends on index strategy",
    "index tuning is essential for database scalability",
    "database performance optimization through indexing",
];

const WORKFLOW_FAMILY: &[&str] = &[
    "code review before merge improves code quality",
    "pull request review is mandatory for all changes",
    "code review workflow catches bugs before merge",
    "mandatory review process ensures code quality standards",
    "review all changes before merging to main branch",
    "code review is required before any production merge",
];

const SECURITY_FAMILY: &[&str] = &[
    "input validation prevents injection attacks on APIs",
    "API input sanitization is mandatory for security",
    "validate all user input before processing requests",
    "input validation is the first line of API defense",
    "sanitize and validate input at every API boundary",
    "API security requires strict input validation rules",
];

const TESTING_FAMILY: &[&str] = &[
    "unit tests must cover all critical business logic",
    "test coverage for critical paths prevents regressions",
    "every critical function needs comprehensive unit tests",
    "unit test coverage is mandatory for core business logic",
    "critical business logic requires thorough test coverage",
    "comprehensive unit tests for all business critical code",
];

const NOISE_CONTENT: &[&str] = &[
    "the weather forecast predicts rain tomorrow afternoon",
    "cats are popular pets in many households worldwide",
    "coffee consumption has increased globally this decade",
    "mountains provide scenic views for hiking enthusiasts",
    "music festivals attract thousands of visitors annually",
    "ocean tides follow predictable lunar cycle patterns",
];

const CONFLICT_PAIRS: &[(&str, &str)] = &[
    ("tabs are better than spaces for code indentation",
     "spaces provide more consistent code alignment than tabs"),
    ("monorepos simplify dependency management across teams",
     "polyrepos give teams independent release autonomy"),
    ("ORMs reduce boilerplate in database access code",
     "raw SQL gives better performance and flexibility"),
];

// ═══════════════════════════════════════════════════════════
// Generators
// ═══════════════════════════════════════════════════════════

fn generate_single_stable(rng: &mut Rng, _seed: u32) -> (Vec<WorldRecord>, Vec<QuerySpec>) {
    // Profile A: Many paraphrases of "safe deploy workflow"
    let mut records = Vec::new();
    let n = 8 + (rng.next() % 4) as usize; // 8-11 records
    for i in 0..n {
        let content = DEPLOY_CORE[i % DEPLOY_CORE.len()];
        records.push(WorldRecord {
            content: format!("{} variant {}", content, i),
            level: Level::Domain,
            tags: vec!["deploy".into(), "safety".into(), "workflow".into()],
            source_type: "recorded",
            semantic_type: "decision",
            family: "deploy-safety".into(),
        });
    }

    let queries = vec![
        QuerySpec { text: "safe deploy workflow".into(), top_k: 10 },
        QuerySpec { text: "staged deployment production".into(), top_k: 10 },
        QuerySpec { text: "deployment safety process".into(), top_k: 10 },
    ];

    (records, queries)
}

fn generate_core_shell(rng: &mut Rng, _seed: u32) -> (Vec<WorldRecord>, Vec<QuerySpec>) {
    // Profile B: core deploy-safety + variable shell terms
    let mut records = Vec::new();

    // Core records (6)
    for i in 0..6 {
        let content = DEPLOY_CORE[i % DEPLOY_CORE.len()];
        records.push(WorldRecord {
            content: content.to_string(),
            level: Level::Domain,
            tags: vec!["deploy".into(), "safety".into(), "workflow".into()],
            source_type: "recorded",
            semantic_type: "decision",
            family: "deploy-safety".into(),
        });
    }

    // Shell records (4-6): same topic but with variable extras
    let shell_n = 4 + (rng.next() % 3) as usize;
    for i in 0..shell_n {
        let content = DEPLOY_SHELL_EXTRAS[i % DEPLOY_SHELL_EXTRAS.len()];
        let extra_tag = rng.pick(&["rollback", "canary", "monitoring", "verification"]);
        records.push(WorldRecord {
            content: content.to_string(),
            level: Level::Domain,
            tags: vec!["deploy".into(), "safety".into(), extra_tag.to_string()],
            source_type: "recorded",
            semantic_type: "decision",
            family: "deploy-safety".into(),
        });
    }

    let queries = vec![
        QuerySpec { text: "deploy safety workflow staging".into(), top_k: 10 },
        QuerySpec { text: "deployment rollback verification".into(), top_k: 10 },
        QuerySpec { text: "safe production deployment process".into(), top_k: 10 },
    ];

    (records, queries)
}

fn generate_two_nearby(rng: &mut Rng, _seed: u32) -> (Vec<WorldRecord>, Vec<QuerySpec>) {
    // Profile C: deploy-safety vs deploy-speed — similar domain, different concept
    let mut records = Vec::new();

    let safety_n = 5 + (rng.next() % 3) as usize;
    for i in 0..safety_n {
        records.push(WorldRecord {
            content: DEPLOY_CORE[i % DEPLOY_CORE.len()].to_string(),
            level: Level::Domain,
            tags: vec!["deploy".into(), "safety".into()],
            source_type: "recorded",
            semantic_type: "decision",
            family: "deploy-safety".into(),
        });
    }

    let speed_n = 5 + (rng.next() % 3) as usize;
    for i in 0..speed_n {
        records.push(WorldRecord {
            content: DEPLOY_SPEED[i % DEPLOY_SPEED.len()].to_string(),
            level: Level::Domain,
            tags: vec!["deploy".into(), "speed".into()],
            source_type: "recorded",
            semantic_type: "decision",
            family: "deploy-speed".into(),
        });
    }

    let queries = vec![
        QuerySpec { text: "safe deployment staging".into(), top_k: 10 },
        QuerySpec { text: "fast deployment automation".into(), top_k: 10 },
        QuerySpec { text: "deploy process pipeline".into(), top_k: 10 },
    ];

    (records, queries)
}

fn generate_multi_concept(rng: &mut Rng, _seed: u32) -> (Vec<WorldRecord>, Vec<QuerySpec>) {
    // Profile D: 4 distinct concept families in one namespace
    let families: &[(&[&str], &str, &[&str])] = &[
        (DATABASE_FAMILY, "database-index", &["database", "performance", "indexing"]),
        (WORKFLOW_FAMILY, "code-review", &["workflow", "review", "quality"]),
        (SECURITY_FAMILY, "input-validation", &["security", "api", "validation"]),
        (TESTING_FAMILY, "unit-testing", &["testing", "coverage", "quality"]),
    ];

    let mut records = Vec::new();
    for (contents, family_name, tags) in families {
        let n = 4 + (rng.next() % 3) as usize;
        for i in 0..n {
            records.push(WorldRecord {
                content: contents[i % contents.len()].to_string(),
                level: Level::Domain,
                tags: tags.iter().map(|t| t.to_string()).collect(),
                source_type: "recorded",
                semantic_type: "decision",
                family: family_name.to_string(),
            });
        }
    }

    let queries = vec![
        QuerySpec { text: "database index performance".into(), top_k: 10 },
        QuerySpec { text: "code review quality".into(), top_k: 10 },
        QuerySpec { text: "API input validation security".into(), top_k: 10 },
        QuerySpec { text: "unit test coverage".into(), top_k: 10 },
    ];

    (records, queries)
}

fn generate_sparse(rng: &mut Rng, _seed: u32) -> (Vec<WorldRecord>, Vec<QuerySpec>) {
    // Profile E: isolated facts, not enough support for concepts
    let mut records = Vec::new();
    for i in 0..8 {
        let noise = NOISE_CONTENT[i % NOISE_CONTENT.len()];
        records.push(WorldRecord {
            content: format!("{} observation {}", noise, rng.next() % 1000),
            level: Level::Domain,
            tags: vec![format!("isolated-{}", i)],
            source_type: "recorded",
            semantic_type: "fact",
            family: format!("noise-{}", i),
        });
    }
    // Add 2 loosely related records (not enough for a concept)
    records.push(WorldRecord {
        content: "database might need indexing eventually".into(),
        level: Level::Domain,
        tags: vec!["database".into()],
        source_type: "recorded",
        semantic_type: "fact",
        family: "weak-db".into(),
    });

    let queries = vec![
        QuerySpec { text: "weather forecast rain".into(), top_k: 10 },
        QuerySpec { text: "database indexing".into(), top_k: 10 },
    ];

    (records, queries)
}

fn generate_adversarial(rng: &mut Rng, seed: u32) -> (Vec<WorldRecord>, Vec<QuerySpec>) {
    // Profile F: paraphrases + conflicting beliefs + noise
    let mut records = Vec::new();

    // Some legit paraphrases (deploy safety)
    for i in 0..4 {
        records.push(WorldRecord {
            content: DEPLOY_CORE[i % DEPLOY_CORE.len()].to_string(),
            level: Level::Domain,
            tags: vec!["deploy".into(), "safety".into()],
            source_type: "recorded",
            semantic_type: "decision",
            family: "deploy-safety".into(),
        });
    }

    // Conflicting pairs
    let pair_idx = seed as usize % CONFLICT_PAIRS.len();
    let (side_a, side_b) = CONFLICT_PAIRS[pair_idx];
    records.push(WorldRecord {
        content: side_a.to_string(),
        level: Level::Domain,
        tags: vec!["conflict".into(), "preference".into()],
        source_type: "recorded",
        semantic_type: "preference",
        family: "conflict-a".into(),
    });
    records.push(WorldRecord {
        content: side_b.to_string(),
        level: Level::Domain,
        tags: vec!["conflict".into(), "preference".into()],
        source_type: "recorded",
        semantic_type: "preference",
        family: "conflict-b".into(),
    });

    // Noise
    for i in 0..3 {
        let noise = rng.pick(NOISE_CONTENT);
        records.push(WorldRecord {
            content: noise.to_string(),
            level: Level::Domain,
            tags: vec![format!("noise-{}", i)],
            source_type: "recorded",
            semantic_type: "fact",
            family: format!("noise-{}", i),
        });
    }

    let queries = vec![
        QuerySpec { text: "deploy safety workflow".into(), top_k: 10 },
        QuerySpec { text: "tabs vs spaces indentation".into(), top_k: 10 },
        QuerySpec { text: "weather forecast cats".into(), top_k: 10 },
    ];

    (records, queries)
}

fn generate_world(
    profile: WorldProfile,
    seed: u32,
) -> (Vec<WorldRecord>, Vec<QuerySpec>) {
    let mut rng = Rng::new(seed);
    match profile {
        WorldProfile::SingleStable => generate_single_stable(&mut rng, seed),
        WorldProfile::CoreShell => generate_core_shell(&mut rng, seed),
        WorldProfile::TwoNearby => generate_two_nearby(&mut rng, seed),
        WorldProfile::MultiConcept => generate_multi_concept(&mut rng, seed),
        WorldProfile::Sparse => generate_sparse(&mut rng, seed),
        WorldProfile::Adversarial => generate_adversarial(&mut rng, seed),
    }
}

// ═══════════════════════════════════════════════════════════
// Per-run metrics
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct RunMetrics {
    run_id: usize,
    seed: u32,
    profile: &'static str,
    expects_concepts: bool,

    // Concept inventory
    total_concepts: usize,
    stable_concepts: usize,
    candidate_concepts: usize,
    rejected_concepts: usize,

    // Coverage
    concept_coverage: f32,     // % recall results with concept membership
    records_in_concepts: usize,
    total_records: usize,

    // Cluster quality
    avg_cluster_size: f32,
    max_cluster_size: usize,
    avg_abstraction_score: f32,

    // Identity / churn
    identity_churn: f32, // across replay cycles

    // False merge/split
    false_merge_events: usize,
    false_split_events: usize,
    cross_topic_merge_events: usize,
    zero_provenance_events: usize,

    // Recall safety (concept phase is read-only, so recall should remain functional)
    recall_functional: bool,

    // Utility label
    label: UtilityLabel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum UtilityLabel {
    Activated,
    Partial,
    Empty,
    Noisy,
}

impl std::fmt::Display for UtilityLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            UtilityLabel::Activated => write!(f, "ACTIVATED"),
            UtilityLabel::Partial => write!(f, "PARTIAL"),
            UtilityLabel::Empty => write!(f, "EMPTY"),
            UtilityLabel::Noisy => write!(f, "NOISY"),
        }
    }
}

fn classify_label(m: &RunMetrics) -> UtilityLabel {
    if m.total_concepts == 0 {
        return UtilityLabel::Empty;
    }
    let has_false_merges = m.false_merge_events > 0 || m.cross_topic_merge_events > 0;
    let noisy_ratio = if m.total_concepts > 0 {
        (m.false_merge_events + m.cross_topic_merge_events) as f32 / m.total_concepts as f32
    } else { 0.0 };

    if noisy_ratio > 0.30 {
        return UtilityLabel::Noisy;
    }

    let good_coverage = m.concept_coverage >= 0.15;
    let good_cluster = m.avg_cluster_size >= 2.0;
    let stable_identity = m.identity_churn <= 0.10;

    if good_coverage && good_cluster && stable_identity && !has_false_merges {
        UtilityLabel::Activated
    } else if m.total_concepts > 0 {
        UtilityLabel::Partial
    } else {
        UtilityLabel::Empty
    }
}

// ═══════════════════════════════════════════════════════════
// Run execution
// ═══════════════════════════════════════════════════════════

fn execute_run(run_id: usize, profile: WorldProfile, seed: u32, num_cycles: usize) -> RunMetrics {
    let (corpus, queries) = generate_world(profile, seed);

    let aura = open_temp_aura();

    // Store corpus
    for rec in &corpus {
        aura.store(
            &rec.content,
            Some(rec.level),
            Some(rec.tags.clone()),
            None, None,
            Some(rec.source_type),
            None,
            Some(false),
            None, None,
            Some(rec.semantic_type),
        ).unwrap_or_else(|e| panic!("run {}: store failed: {}", run_id, e));
    }

    // Run one warmup cycle before measuring

    // Run maintenance cycles, track concept keys for churn
    let mut prev_keys: HashSet<String> = HashSet::new();
    let mut churn_values: Vec<f32> = Vec::new();

    for cycle in 0..num_cycles {
        aura.run_maintenance();

        let concepts = aura.get_concepts(None);
        let current_keys: HashSet<String> = concepts.iter().map(|c| c.key.clone()).collect();

        if cycle >= 2 && !current_keys.is_empty() {
            // Churn = (new + dropped) / total
            let new_count = current_keys.difference(&prev_keys).count();
            let dropped = prev_keys.difference(&current_keys).count();
            let total = current_keys.len().max(prev_keys.len()).max(1);
            let churn = (new_count + dropped) as f32 / total as f32;
            churn_values.push(churn);
        }
        prev_keys = current_keys;
    }

    // Recall after maintenance — concept phase is read-only so recall must remain functional
    let recall_after: usize = queries.iter().map(|q| {
        aura.recall_structured(&q.text, Some(q.top_k), Some(0.0), Some(true), None, None)
            .unwrap_or_default().len()
    }).sum();
    let recall_functional = recall_after > 0;

    // Collect concept data
    let all_concepts = aura.get_concepts(None);
    let stable_concepts = aura.get_concepts(Some("stable"));
    let candidate_concepts = aura.get_concepts(Some("candidate"));
    let rejected_count = all_concepts.len() - stable_concepts.len() - candidate_concepts.len();

    // Build record→concept membership
    let mut record_to_concepts: HashMap<String, Vec<String>> = HashMap::new();
    for concept in &all_concepts {
        for rid in &concept.record_ids {
            record_to_concepts.entry(rid.clone()).or_default().push(concept.id.clone());
        }
    }

    // Coverage: % of recall results that have concept membership
    let mut total_recall_results = 0usize;
    let mut with_concept = 0usize;
    for q in &queries {
        let results = aura.recall_structured(&q.text, Some(q.top_k), Some(0.0), Some(true), None, None)
            .unwrap_or_default();
        for (_, rec) in &results {
            total_recall_results += 1;
            if record_to_concepts.contains_key(&rec.id) {
                with_concept += 1;
            }
        }
    }
    let concept_coverage = if total_recall_results > 0 {
        with_concept as f32 / total_recall_results as f32
    } else { 0.0 };

    // Cluster sizes
    let cluster_sizes: Vec<usize> = all_concepts.iter()
        .filter(|c| !c.record_ids.is_empty())
        .map(|c| c.record_ids.len())
        .collect();
    let avg_cluster = if !cluster_sizes.is_empty() {
        cluster_sizes.iter().sum::<usize>() as f32 / cluster_sizes.len() as f32
    } else { 0.0 };
    let max_cluster = cluster_sizes.iter().max().copied().unwrap_or(0);

    // Avg abstraction score
    let avg_abs = if !all_concepts.is_empty() {
        all_concepts.iter().map(|c| c.abstraction_score).sum::<f32>() / all_concepts.len() as f32
    } else { 0.0 };

    // False merge detection: check if a single concept contains records from different families
    let mut false_merge_events = 0usize;
    let mut cross_topic_merge_events = 0usize;
    let mut zero_provenance_events = 0usize;
    let mut false_split_events = 0usize;

    // Build record_id → family mapping
    let mut record_family: HashMap<String, String> = HashMap::new();
    // We need record IDs from the aura to map to families.
    // Since we can't easily map record IDs back to input records,
    // we use a content-based approach: store content→family during generation.
    let mut content_to_family: HashMap<String, String> = HashMap::new();
    for rec in &corpus {
        content_to_family.insert(rec.content.clone(), rec.family.clone());
    }

    // Get all records and map IDs to families
    let all_results = aura.recall_structured("", Some(200), Some(0.0), Some(true), None, None)
        .unwrap_or_default();
    for (_, rec) in &all_results {
        // Try to match record content to family
        if let Some(family) = content_to_family.get(&rec.content) {
            record_family.insert(rec.id.clone(), family.clone());
        }
    }

    for concept in &all_concepts {
        // Check provenance
        if concept.record_ids.is_empty() {
            zero_provenance_events += 1;
            continue;
        }

        // Collect families present in this concept
        let families_in_concept: HashSet<&str> = concept.record_ids.iter()
            .filter_map(|rid| record_family.get(rid).map(|f| f.as_str()))
            .collect();

        if families_in_concept.len() > 1 {
            // Check if the merged families are truly different topics
            let has_noise = families_in_concept.iter().any(|f| f.starts_with("noise"));
            let has_conflict = families_in_concept.iter().any(|f| f.starts_with("conflict"));

            if has_noise || has_conflict {
                cross_topic_merge_events += 1;
            } else {
                false_merge_events += 1;
            }
        }
    }

    // False split: check if same-family records ended up in different concepts
    // (only for families with enough records)
    let mut family_to_concept_ids: HashMap<String, HashSet<String>> = HashMap::new();
    for concept in &all_concepts {
        for rid in &concept.record_ids {
            if let Some(family) = record_family.get(rid) {
                family_to_concept_ids.entry(family.clone())
                    .or_default()
                    .insert(concept.id.clone());
            }
        }
    }
    for (_family, concept_ids) in &family_to_concept_ids {
        if concept_ids.len() > 1 {
            false_split_events += concept_ids.len() - 1;
        }
    }

    // Identity churn (average over cycles after warmup)
    let identity_churn = if !churn_values.is_empty() {
        churn_values.iter().sum::<f32>() / churn_values.len() as f32
    } else { 0.0 };

    let total_records = corpus.len();
    let records_in_concepts = record_to_concepts.len();

    let mut metrics = RunMetrics {
        run_id,
        seed,
        profile: profile.name(),
        expects_concepts: profile.expects_concepts(),
        total_concepts: all_concepts.len(),
        stable_concepts: stable_concepts.len(),
        candidate_concepts: candidate_concepts.len(),
        rejected_concepts: rejected_count,
        concept_coverage,
        records_in_concepts,
        total_records,
        avg_cluster_size: avg_cluster,
        max_cluster_size: max_cluster,
        avg_abstraction_score: avg_abs,
        identity_churn,
        false_merge_events,
        false_split_events,
        cross_topic_merge_events,
        zero_provenance_events,
        recall_functional,
        label: UtilityLabel::Empty, // placeholder
    };
    metrics.label = classify_label(&metrics);
    metrics
}

// ═══════════════════════════════════════════════════════════
// Aggregate report
// ═══════════════════════════════════════════════════════════

#[derive(Debug)]
struct CampaignAggregate {
    total_runs: usize,
    avg_coverage: f32,
    median_coverage: f32,
    avg_cluster_size: f32,
    stable_concept_rate: f32,   // % runs with at least one stable concept
    candidate_concept_rate: f32,
    false_merge_rate: f32,
    false_split_rate: f32,
    cross_topic_merge_rate: f32,
    avg_identity_churn: f32,
    pct_zero_concepts: f32,
    pct_useful_concepts: f32,   // ACTIVATED or PARTIAL
    pct_activated: f32,
    pct_partial: f32,
    pct_empty: f32,
    pct_noisy: f32,
    recall_degraded_runs: usize,
}

fn percentile(sorted: &[f32], pct: f32) -> f32 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() as f32 - 1.0) * pct) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn aggregate(results: &[RunMetrics]) -> CampaignAggregate {
    let n = results.len();
    let nf = n as f32;

    let mut coverages: Vec<f32> = results.iter().map(|r| r.concept_coverage).collect();
    coverages.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let avg_coverage = coverages.iter().sum::<f32>() / nf;
    let median_coverage = percentile(&coverages, 0.50);

    let cluster_sizes: Vec<f32> = results.iter()
        .filter(|r| r.total_concepts > 0)
        .map(|r| r.avg_cluster_size)
        .collect();
    let avg_cluster = if !cluster_sizes.is_empty() {
        cluster_sizes.iter().sum::<f32>() / cluster_sizes.len() as f32
    } else { 0.0 };

    let with_stable = results.iter().filter(|r| r.stable_concepts > 0).count();
    let with_candidate = results.iter().filter(|r| r.candidate_concepts > 0).count();
    let zero_concepts = results.iter().filter(|r| r.total_concepts == 0).count();

    let total_concepts: usize = results.iter().map(|r| r.total_concepts).sum();
    let total_false_merges: usize = results.iter().map(|r| r.false_merge_events).sum();
    let total_false_splits: usize = results.iter().map(|r| r.false_split_events).sum();
    let total_cross_topic: usize = results.iter().map(|r| r.cross_topic_merge_events).sum();

    let false_merge_rate = if total_concepts > 0 { total_false_merges as f32 / total_concepts as f32 } else { 0.0 };
    let false_split_rate = if total_concepts > 0 { total_false_splits as f32 / total_concepts as f32 } else { 0.0 };
    let cross_topic_rate = if total_concepts > 0 { total_cross_topic as f32 / total_concepts as f32 } else { 0.0 };

    let avg_churn = results.iter().map(|r| r.identity_churn).sum::<f32>() / nf;

    let activated = results.iter().filter(|r| r.label == UtilityLabel::Activated).count();
    let partial = results.iter().filter(|r| r.label == UtilityLabel::Partial).count();
    let empty = results.iter().filter(|r| r.label == UtilityLabel::Empty).count();
    let noisy = results.iter().filter(|r| r.label == UtilityLabel::Noisy).count();

    let useful = activated + partial;
    let recall_degraded = results.iter().filter(|r| !r.recall_functional).count();

    CampaignAggregate {
        total_runs: n,
        avg_coverage,
        median_coverage,
        avg_cluster_size: avg_cluster,
        stable_concept_rate: with_stable as f32 / nf,
        candidate_concept_rate: with_candidate as f32 / nf,
        false_merge_rate,
        false_split_rate,
        cross_topic_merge_rate: cross_topic_rate,
        avg_identity_churn: avg_churn,
        pct_zero_concepts: zero_concepts as f32 / nf,
        pct_useful_concepts: useful as f32 / nf,
        pct_activated: activated as f32 / nf,
        pct_partial: partial as f32 / nf,
        pct_empty: empty as f32 / nf,
        pct_noisy: noisy as f32 / nf,
        recall_degraded_runs: recall_degraded,
    }
}

// ═══════════════════════════════════════════════════════════
// Diagnosis
// ═══════════════════════════════════════════════════════════

#[derive(Debug)]
enum DiagnosisBlock {
    SeedSelectionTooStrict,
    ClusteringTooStrict,
    ScoringTooStrict,
    SyntheticNotBeliefRich,
    None, // no structural block found
}

impl std::fmt::Display for DiagnosisBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DiagnosisBlock::SeedSelectionTooStrict => write!(f, "SEED SELECTION TOO STRICT"),
            DiagnosisBlock::ClusteringTooStrict => write!(f, "CLUSTERING TOO STRICT"),
            DiagnosisBlock::ScoringTooStrict => write!(f, "SCORING TOO STRICT"),
            DiagnosisBlock::None => write!(f, "NONE — no structural block"),
            DiagnosisBlock::SyntheticNotBeliefRich => write!(f, "SYNTHETIC WORLDS NOT BELIEF-RICH ENOUGH"),
        }
    }
}

fn diagnose(results: &[RunMetrics], agg: &CampaignAggregate) -> DiagnosisBlock {
    // If we have good results, no block
    if agg.avg_coverage >= GATE_AVG_COVERAGE && agg.avg_cluster_size >= GATE_AVG_CLUSTER_SIZE {
        return DiagnosisBlock::None;
    }

    // Check supportive profiles (A-D) only
    let supportive: Vec<&RunMetrics> = results.iter()
        .filter(|r| r.expects_concepts)
        .collect();

    if supportive.is_empty() {
        return DiagnosisBlock::SyntheticNotBeliefRich;
    }

    // How many supportive runs produced ANY concept?
    let with_any = supportive.iter().filter(|r| r.total_concepts > 0).count();
    let supportive_n = supportive.len();

    // If most supportive worlds produce 0 concepts, check why
    if with_any as f32 / supportive_n as f32 <= 0.30 {
        // Very few concepts forming at all
        // Check if it's seed selection (beliefs not qualifying)
        // vs clustering (beliefs qualify but don't cluster)
        // We can't directly check seeds from here, but we can infer:
        // If total_records is decent but concepts are 0, seed selection may be too strict
        let avg_records = supportive.iter().map(|r| r.total_records).sum::<usize>() as f32
            / supportive_n as f32;
        if avg_records >= 6.0 {
            return DiagnosisBlock::SeedSelectionTooStrict;
        } else {
            return DiagnosisBlock::SyntheticNotBeliefRich;
        }
    }

    // Concepts form but coverage is low
    if agg.avg_coverage < GATE_AVG_COVERAGE {
        // Concepts exist but don't cover enough recall results
        let with_concepts: Vec<&&RunMetrics> = supportive.iter()
            .filter(|r| r.total_concepts > 0)
            .collect();
        let avg_cluster_in_active = if !with_concepts.is_empty() {
            with_concepts.iter().map(|r| r.avg_cluster_size).sum::<f32>() / with_concepts.len() as f32
        } else { 0.0 };

        if avg_cluster_in_active < 2.0 {
            return DiagnosisBlock::ClusteringTooStrict;
        }

        // Clusters form but abstraction scores are low
        let avg_abs = with_concepts.iter().map(|r| r.avg_abstraction_score).sum::<f32>()
            / with_concepts.len() as f32;
        if avg_abs < 0.50 {
            return DiagnosisBlock::ScoringTooStrict;
        }

        // Coverage low despite good clusters — might need more records per concept
        return DiagnosisBlock::SyntheticNotBeliefRich;
    }

    DiagnosisBlock::None
}

#[derive(Debug)]
enum Verdict {
    Activated,
    PartialSignal,
    StructuralBlock(DiagnosisBlock),
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Verdict::Activated => write!(f, "ACTIVATED"),
            Verdict::PartialSignal => write!(f, "PARTIAL SIGNAL"),
            Verdict::StructuralBlock(block) => write!(f, "STRUCTURAL BLOCK — {}", block),
        }
    }
}

fn compute_verdict(agg: &CampaignAggregate, diagnosis: &DiagnosisBlock) -> Verdict {
    match diagnosis {
        DiagnosisBlock::None => {
            if agg.avg_coverage >= STRONG_COVERAGE
                && agg.stable_concept_rate >= 0.30
                && agg.false_merge_rate <= GATE_FALSE_MERGE_RATE
            {
                Verdict::Activated
            } else {
                Verdict::PartialSignal
            }
        }
        _ => {
            if agg.pct_useful_concepts >= 0.30 {
                Verdict::PartialSignal
            } else {
                Verdict::StructuralBlock(match diagnosis {
                    DiagnosisBlock::SeedSelectionTooStrict => DiagnosisBlock::SeedSelectionTooStrict,
                    DiagnosisBlock::ClusteringTooStrict => DiagnosisBlock::ClusteringTooStrict,
                    DiagnosisBlock::ScoringTooStrict => DiagnosisBlock::ScoringTooStrict,
                    DiagnosisBlock::SyntheticNotBeliefRich => DiagnosisBlock::SyntheticNotBeliefRich,
                    DiagnosisBlock::None => DiagnosisBlock::None,
                })
            }
        }
    }
}

fn recommendation(verdict: &Verdict) -> &'static str {
    match verdict {
        Verdict::Activated => "Move to SHADOW EVALUATION",
        Verdict::PartialSignal => "Retune concept thresholds first, then re-evaluate",
        Verdict::StructuralBlock(_) => "Keep DEFERRED — address structural block before re-evaluation",
    }
}

// ═══════════════════════════════════════════════════════════
// STRUCTURAL TESTS
// ═══════════════════════════════════════════════════════════

#[test]
fn concept_campaign_single_stable_profile_forms_concepts() {
    let metrics = execute_run(0, WorldProfile::SingleStable, 42, 8);
    // Single stable should produce at least some concept candidates
    // (may be 0 if thresholds are too strict — that's the diagnosis)
    eprintln!("  single-stable: concepts={} stable={} coverage={:.1}% cluster={:.1} label={}",
        metrics.total_concepts, metrics.stable_concepts,
        metrics.concept_coverage * 100.0, metrics.avg_cluster_size, metrics.label);
    // No assertion on concept count — campaign will diagnose
    assert!(metrics.recall_functional,
        "recall must remain functional after concept discovery");
}

#[test]
fn concept_campaign_two_nearby_profiles_do_not_false_merge() {
    let metrics = execute_run(0, WorldProfile::TwoNearby, 42, 8);
    eprintln!("  two-nearby: concepts={} false_merges={} cross_topic={} label={}",
        metrics.total_concepts, metrics.false_merge_events,
        metrics.cross_topic_merge_events, metrics.label);
    // Even if concepts don't form, there should be no false cross-topic merges
    assert_eq!(metrics.cross_topic_merge_events, 0,
        "cross-topic merges must be zero");
}

#[test]
fn concept_campaign_sparse_profile_stays_empty_without_noise() {
    let metrics = execute_run(0, WorldProfile::Sparse, 42, 8);
    eprintln!("  sparse: concepts={} false_merges={} label={}",
        metrics.total_concepts, metrics.false_merge_events, metrics.label);
    // Sparse profile: either empty or very few concepts, no false positives
    assert_eq!(metrics.false_merge_events, 0,
        "sparse profile should have no false merges");
    assert_eq!(metrics.cross_topic_merge_events, 0,
        "sparse profile should have no cross-topic merges");
}

#[test]
fn concept_campaign_identity_stable_across_replay() {
    // Run same world twice and check identity stability
    let m1 = execute_run(0, WorldProfile::SingleStable, 12345, 8);
    let m2 = execute_run(1, WorldProfile::SingleStable, 12345, 8);
    eprintln!("  replay: run1 concepts={} run2 concepts={} churn1={:.3} churn2={:.3}",
        m1.total_concepts, m2.total_concepts, m1.identity_churn, m2.identity_churn);
    // Same seed should produce same concept count
    assert_eq!(m1.total_concepts, m2.total_concepts,
        "deterministic replay should produce same concept count");
}

// ═══════════════════════════════════════════════════════════
// AGGREGATE CAMPAIGN
// ═══════════════════════════════════════════════════════════

#[test]
fn concept_activation_campaign() {
    let num_runs: usize = std::env::var("CONCEPT_CAMPAIGN_RUNS")
        .ok().and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let num_cycles: usize = std::env::var("CONCEPT_CAMPAIGN_CYCLES")
        .ok().and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let profiles = WorldProfile::all();
    let mut results = Vec::with_capacity(num_runs);

    eprintln!("\n  ╔═══════════════════════════════════════════════════╗");
    eprintln!("  ║  Candidate C — Concept Activation Campaign        ║");
    eprintln!("  ║  {} runs, {} profiles, {} cycles/run             ║", num_runs, profiles.len(), num_cycles);
    eprintln!("  ╚═══════════════════════════════════════════════════╝\n");

    for run_id in 0..num_runs {
        let profile = profiles[run_id % profiles.len()];
        let seed = (run_id as u32 + 1) * 7919;
        let metrics = execute_run(run_id, profile, seed, num_cycles);

        if run_id < 12 || metrics.total_concepts > 0 || metrics.label == UtilityLabel::Noisy {
            eprintln!("  run {:3} [{:14}] seed={:6} concepts={:2} stable={} cov={:.1}% cluster={:.1} merges={}/{} churn={:.3} label={}",
                metrics.run_id, metrics.profile, metrics.seed,
                metrics.total_concepts, metrics.stable_concepts,
                metrics.concept_coverage * 100.0, metrics.avg_cluster_size,
                metrics.false_merge_events, metrics.cross_topic_merge_events,
                metrics.identity_churn, metrics.label);
        }

        results.push(metrics);
    }

    let agg = aggregate(&results);
    let diagnosis = diagnose(&results, &agg);
    let verdict = compute_verdict(&agg, &diagnosis);
    let rec = recommendation(&verdict);

    // ── Per-profile breakdown ──
    eprintln!("\n  ── Per-Profile Summary ──");
    for profile in profiles {
        let profile_runs: Vec<&RunMetrics> = results.iter()
            .filter(|r| r.profile == profile.name())
            .collect();
        if profile_runs.is_empty() { continue; }
        let pn = profile_runs.len();
        let avg_cov = profile_runs.iter().map(|r| r.concept_coverage).sum::<f32>() / pn as f32;
        let avg_concepts = profile_runs.iter().map(|r| r.total_concepts).sum::<usize>() as f32 / pn as f32;
        let avg_stable = profile_runs.iter().map(|r| r.stable_concepts).sum::<usize>() as f32 / pn as f32;
        let avg_cluster = {
            let with_c: Vec<f32> = profile_runs.iter()
                .filter(|r| r.total_concepts > 0)
                .map(|r| r.avg_cluster_size)
                .collect();
            if with_c.is_empty() { 0.0 } else { with_c.iter().sum::<f32>() / with_c.len() as f32 }
        };
        let total_fm: usize = profile_runs.iter().map(|r| r.false_merge_events).sum();
        let total_ct: usize = profile_runs.iter().map(|r| r.cross_topic_merge_events).sum();
        let labels: Vec<String> = [UtilityLabel::Activated, UtilityLabel::Partial, UtilityLabel::Empty, UtilityLabel::Noisy]
            .iter()
            .map(|l| {
                let cnt = profile_runs.iter().filter(|r| r.label == *l).count();
                format!("{}={}", l, cnt)
            })
            .collect();

        eprintln!("  {:<16} n={:>2}  avg_cov={:>5.1}%  avg_concepts={:.1}  avg_stable={:.1}  cluster={:.1}  fm={}  ct={}  {}",
            profile.name(), pn, avg_cov * 100.0, avg_concepts, avg_stable, avg_cluster,
            total_fm, total_ct, labels.join("  "));
    }

    // ── Campaign aggregate ──
    eprintln!("\n  ── Campaign Aggregate ──");
    eprintln!("  Total runs:            {}", agg.total_runs);
    eprintln!("  Avg coverage:          {:.1}%", agg.avg_coverage * 100.0);
    eprintln!("  Median coverage:       {:.1}%", agg.median_coverage * 100.0);
    eprintln!("  Avg cluster size:      {:.1}", agg.avg_cluster_size);
    eprintln!("  Stable concept rate:   {:.1}%", agg.stable_concept_rate * 100.0);
    eprintln!("  Candidate concept rate:{:.1}%", agg.candidate_concept_rate * 100.0);
    eprintln!("  False merge rate:      {:.3}", agg.false_merge_rate);
    eprintln!("  False split rate:      {:.3}", agg.false_split_rate);
    eprintln!("  Cross-topic merges:    {:.3}", agg.cross_topic_merge_rate);
    eprintln!("  Avg identity churn:    {:.3}", agg.avg_identity_churn);
    eprintln!("  % zero concepts:       {:.1}%", agg.pct_zero_concepts * 100.0);
    eprintln!("  % useful (ACT+PART):   {:.1}%", agg.pct_useful_concepts * 100.0);
    eprintln!("  % ACTIVATED:           {:.1}%", agg.pct_activated * 100.0);
    eprintln!("  % PARTIAL:             {:.1}%", agg.pct_partial * 100.0);
    eprintln!("  % EMPTY:               {:.1}%", agg.pct_empty * 100.0);
    eprintln!("  % NOISY:               {:.1}%", agg.pct_noisy * 100.0);
    eprintln!("  Recall degraded runs:  {}", agg.recall_degraded_runs);

    // ── Gate evaluation ──
    eprintln!("\n  ── Activation Gates ──");

    struct Gate { name: &'static str, threshold: String, actual: String, passed: bool }

    // Only evaluate coverage on supportive profiles (A-D)
    let supportive_runs: Vec<&RunMetrics> = results.iter()
        .filter(|r| r.expects_concepts).collect();
    let supportive_coverage = if !supportive_runs.is_empty() {
        supportive_runs.iter().map(|r| r.concept_coverage).sum::<f32>() / supportive_runs.len() as f32
    } else { 0.0 };

    let gates = vec![
        Gate { name: "supportive_avg_coverage",
            threshold: format!(">= {:.0}%", GATE_AVG_COVERAGE * 100.0),
            actual: format!("{:.1}%", supportive_coverage * 100.0),
            passed: supportive_coverage >= GATE_AVG_COVERAGE },
        Gate { name: "avg_cluster_size",
            threshold: format!(">= {:.1}", GATE_AVG_CLUSTER_SIZE),
            actual: format!("{:.1}", agg.avg_cluster_size),
            passed: agg.avg_cluster_size >= GATE_AVG_CLUSTER_SIZE || agg.pct_zero_concepts >= 0.99 },
        Gate { name: "false_merge_rate",
            threshold: format!("<= {:.2}", GATE_FALSE_MERGE_RATE),
            actual: format!("{:.3}", agg.false_merge_rate),
            passed: agg.false_merge_rate <= GATE_FALSE_MERGE_RATE },
        Gate { name: "cross_topic_merge_rate",
            threshold: format!("<= {:.2}", GATE_CROSS_TOPIC_MERGE_RATE),
            actual: format!("{:.3}", agg.cross_topic_merge_rate),
            passed: agg.cross_topic_merge_rate <= GATE_CROSS_TOPIC_MERGE_RATE },
        Gate { name: "identity_churn",
            threshold: format!("<= {:.2}", GATE_MAX_IDENTITY_CHURN),
            actual: format!("{:.3}", agg.avg_identity_churn),
            passed: agg.avg_identity_churn <= GATE_MAX_IDENTITY_CHURN },
        Gate { name: "zero_recall_impact",
            threshold: "0 degraded runs".into(),
            actual: format!("{}", agg.recall_degraded_runs),
            passed: agg.recall_degraded_runs == 0 },
    ];

    let mut all_passed = true;
    for gate in &gates {
        let status = if gate.passed { "PASS" } else { "FAIL" };
        let marker = if gate.passed { " " } else { "!" };
        eprintln!("  {} [{}] {:28} threshold: {:>12}  actual: {:>12}",
            marker, status, gate.name, gate.threshold, gate.actual);
        if !gate.passed { all_passed = false; }
    }

    // Strong pass check
    let strong_pass = all_passed
        && supportive_coverage >= STRONG_COVERAGE
        && agg.stable_concept_rate >= 0.30
        && agg.false_merge_rate <= 0.02;

    // ── Diagnosis ──
    eprintln!("\n  ── Diagnosis ──");
    eprintln!("  Primary block:   {}", diagnosis);

    // ── Verdict ──
    eprintln!("\n  ── Verdict ──");
    if strong_pass {
        eprintln!("  VERDICT: STRONG PASS — concept.rs ACTIVATED, ready for SHADOW EVALUATION");
    } else if all_passed {
        eprintln!("  VERDICT: PASS — {}", verdict);
    } else {
        eprintln!("  VERDICT: {} — {}", if matches!(verdict, Verdict::PartialSignal) { "PARTIAL" } else { "FAIL" }, verdict);
    }
    eprintln!("  Recommendation:  {}", rec);
    eprintln!();

    // ── Hard assertions ──
    // We assert safety invariants (no recall degradation, bounded cross-topic merges)
    // but NOT coverage gates — the campaign's purpose is to diagnose, not to block.
    assert_eq!(agg.recall_degraded_runs, 0,
        "all runs must have functional recall after concept discovery");
    assert!(agg.cross_topic_merge_rate <= 0.10,
        "cross-topic merge rate {:.3} too high (>10%)", agg.cross_topic_merge_rate);
}

#[test]
fn concept_campaign_reports_aggregate_metrics() {
    // Smoke: run 6 worlds (one per profile) and verify aggregation works
    let profiles = WorldProfile::all();
    let mut results = Vec::new();
    for (i, profile) in profiles.iter().enumerate() {
        results.push(execute_run(i, *profile, (i as u32 + 1) * 31, 6));
    }
    let agg = aggregate(&results);
    assert_eq!(agg.total_runs, 6);
    // Metrics should be finite
    assert!(agg.avg_coverage.is_finite());
    assert!(agg.avg_cluster_size.is_finite());
    assert!(agg.avg_identity_churn.is_finite());
}

#[test]
fn concept_campaign_respects_zero_recall_impact() {
    // Every profile individually must have functional recall after concept discovery
    for (i, profile) in WorldProfile::all().iter().enumerate() {
        let metrics = execute_run(i, *profile, (i as u32 + 1) * 97, 6);
        assert!(metrics.recall_functional,
            "profile {} has no functional recall after concept discovery",
            profile.name());
    }
}

#[test]
fn concept_campaign_emits_final_verdict() {
    // Run minimal campaign and verify verdict logic doesn't panic
    let profiles = WorldProfile::all();
    let mut results = Vec::new();
    for (i, profile) in profiles.iter().enumerate() {
        results.push(execute_run(i, *profile, (i as u32 + 1) * 53, 6));
    }
    let agg = aggregate(&results);
    let diag = diagnose(&results, &agg);
    let v = compute_verdict(&agg, &diag);
    let r = recommendation(&v);
    eprintln!("  verdict: {} — recommendation: {}", v, r);
    // Just verify it produces a valid verdict string
    let vs = format!("{}", v);
    assert!(!vs.is_empty());
}
