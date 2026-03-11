//! Synthetic Validation Campaign for Candidate B (Belief Rerank)
//!
//! Generates many varied synthetic corpora + query packs, runs baseline vs limited
//! rerank on each, aggregates metrics across the entire campaign, and evaluates
//! pass/fail gates.
//!
//! 6 scenario classes:
//!   A. Stable domains (factual, low conflict)
//!   B. Belief-heavy preference worlds (strong opinions)
//!   C. Conflicting worlds (tabs-vs-spaces style disputes)
//!   D. Mixed-topic worlds (multi-domain corpus)
//!   E. Sparse-belief worlds (few clusters, many isolated facts)
//!   F. No-match / nonsense worlds (queries outside corpus)
//!
//! Usage:
//!   cargo test --no-default-features --features "encryption,server,audit" \
//!     --test belief_rerank_campaign -- --nocapture
//!
//! Default: 100 runs. Set CAMPAIGN_RUNS env var to override.

use aura::{Aura, Level};
use std::collections::HashSet;
use std::mem::ManuallyDrop;

// ═══════════════════════════════════════════════════════════
// Campaign gate thresholds
// ═══════════════════════════════════════════════════════════

/// Campaign passes if avg worse across all runs <= this.
const GATE_AVG_WORSE: f32 = 0.01;
/// Campaign passes if % of runs with any worse <= this.
const GATE_RUNS_WITH_WORSE: f32 = 0.05;
/// Campaign passes if avg overlap across all runs >= this.
const GATE_AVG_OVERLAP: f32 = 0.90;
/// Campaign passes if p95 overlap >= this.
const GATE_P95_OVERLAP: f32 = 0.80;
/// Campaign passes if avg contradiction worsened <= this.
const GATE_CONTRADICTION_WORSENED: f32 = 0.01;
/// Campaign passes if max observed shift <= this.
const GATE_MAX_SHIFT: usize = 2;
/// Campaign passes if p95 latency delta <= this (us).
const GATE_P95_LATENCY_US: f64 = 500.0;

// Per-run alert thresholds (same as monitor)
const ALERT_MAX_WORSE_PCT: f32 = 0.05;
const ALERT_MIN_AVG_OVERLAP: f32 = 0.70;
const ALERT_MAX_POS_SHIFT: usize = 2;
const ALERT_MAX_AVG_LATENCY_US: f64 = 2000.0;
const ALERT_MAX_CONTRADICTION_WORSENED_PCT: f32 = 0.05;

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
// Scenario profiles
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
enum ScenarioProfile {
    Stable,
    BeliefHeavy,
    Conflicting,
    Mixed,
    Sparse,
    NoMatch,
}

impl ScenarioProfile {
    fn all() -> &'static [ScenarioProfile] {
        &[
            ScenarioProfile::Stable,
            ScenarioProfile::BeliefHeavy,
            ScenarioProfile::Conflicting,
            ScenarioProfile::Mixed,
            ScenarioProfile::Sparse,
            ScenarioProfile::NoMatch,
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            ScenarioProfile::Stable => "stable",
            ScenarioProfile::BeliefHeavy => "belief-heavy",
            ScenarioProfile::Conflicting => "conflicting",
            ScenarioProfile::Mixed => "mixed",
            ScenarioProfile::Sparse => "sparse",
            ScenarioProfile::NoMatch => "no-match",
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Deterministic seed-based content generation
// ═══════════════════════════════════════════════════════════

/// Simple deterministic PRNG (xorshift32) for reproducible variation.
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

    fn pick_n<'a, T>(&mut self, items: &'a [T], n: usize) -> Vec<&'a T> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            result.push(self.pick(items));
        }
        result
    }
}

// ── Word banks for structured variation ──

const SUBJECTS: &[&str] = &[
    "Rust", "Python", "TypeScript", "Go", "Java", "C++", "Kotlin", "Swift",
    "Elixir", "Haskell", "Scala", "Ruby", "Dart", "Zig", "Nim",
];

const BACKEND_ADJECTIVES: &[&str] = &[
    "fast", "reliable", "scalable", "robust", "efficient", "safe",
    "performant", "stable", "modern", "production-ready",
];

const DOMAINS: &[&str] = &[
    "backend", "frontend", "devops", "database", "security", "testing",
    "architecture", "monitoring", "networking", "caching",
];

const DEPLOY_METHODS: &[&str] = &[
    "canary", "blue-green", "rolling", "feature-flag", "shadow",
    "A/B", "progressive", "immutable",
];

const DB_TYPES: &[&str] = &[
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "DynamoDB",
    "Cassandra", "CockroachDB", "TimescaleDB", "ClickHouse",
];

const TOOLS: &[&str] = &[
    "Docker", "Kubernetes", "Terraform", "Ansible", "Prometheus",
    "Grafana", "Nginx", "Envoy", "Jenkins", "GitHub Actions",
];

const SECURITY_TOPICS: &[&str] = &[
    "authentication", "authorization", "encryption", "TLS", "rate limiting",
    "input validation", "secrets management", "RBAC", "audit logging", "CORS",
];

const WORKFLOW_PREFS: &[&str] = &[
    "dark mode", "light mode", "split screen", "terminal-first", "IDE-heavy",
    "pair programming", "code review", "TDD", "REPL-driven", "documentation-first",
];

const CONFLICT_PAIRS: &[(&str, &str)] = &[
    ("tabs are better than spaces for indentation", "spaces are the standard for code indentation"),
    ("ORMs simplify database access and reduce boilerplate", "raw SQL is faster and more flexible than ORMs"),
    ("monorepos simplify cross-team dependency management", "polyrepos give teams independent release cycles"),
    ("GraphQL eliminates over-fetching in API design", "REST APIs are simpler and more widely understood"),
    ("microservices enable independent scaling", "monoliths are simpler and easier to debug"),
    ("serverless reduces operational overhead", "containers give more control than serverless"),
    ("NoSQL databases handle unstructured data better", "relational databases ensure data integrity"),
    ("dynamic typing enables faster prototyping", "static typing catches errors at compile time"),
    ("functional programming reduces side effects", "OOP models real-world domains naturally"),
    ("agile methodology adapts to changing requirements", "waterfall ensures thorough upfront planning"),
];

const NONSENSE_QUERIES: &[&str] = &[
    "quantum entanglement cooking recipes",
    "medieval pottery glazing techniques",
    "alpine flower taxonomy classification",
    "underwater basket weaving methods",
    "ancient Egyptian hieroglyph parsing",
    "deep sea bioluminescence patterns",
    "baroque music theory counterpoint",
    "origami crane folding instructions",
    "volcanic soil mineral composition",
    "arctic fox migration patterns winter",
];

// ═══════════════════════════════════════════════════════════
// Corpus + query generators per profile
// ═══════════════════════════════════════════════════════════

struct CorpusRecord {
    content: String,
    level: Level,
    tags: Vec<String>,
    source_type: &'static str,
    semantic_type: &'static str,
}

struct QuerySpec {
    text: String,
    top_k: usize,
    category: String,
}

fn generate_stable(rng: &mut Rng, seed: u32) -> (Vec<CorpusRecord>, Vec<QuerySpec>) {
    let lang = SUBJECTS[seed as usize % SUBJECTS.len()];
    let domain = DOMAINS[seed as usize % DOMAINS.len()];
    let adj1 = rng.pick(BACKEND_ADJECTIVES);
    let adj2 = rng.pick(BACKEND_ADJECTIVES);
    let adj3 = rng.pick(BACKEND_ADJECTIVES);
    let tool = rng.pick(TOOLS);

    let records = vec![
        CorpusRecord {
            content: format!("{} is a {} language for {} development", lang, adj1, domain),
            level: Level::Domain, tags: vec![lang.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} provides {} guarantees in production {}", lang, adj2, domain),
            level: Level::Domain, tags: vec![lang.to_lowercase(), domain.to_string(), "production".into()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} ecosystem includes {} for {} workloads", lang, tool, domain),
            level: Level::Domain, tags: vec![lang.to_lowercase(), tool.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} is {} and {} for typical {} tasks", lang, adj1, adj3, domain),
            level: Level::Domain, tags: vec![lang.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} compile times are acceptable for {} projects", lang, domain),
            level: Level::Domain, tags: vec![lang.to_lowercase(), domain.to_string(), "performance".into()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} error handling is explicit and {} in {}", lang, adj2, domain),
            level: Level::Domain, tags: vec![lang.to_lowercase(), domain.to_string(), "errors".into()],
            source_type: "recorded", semantic_type: "fact",
        },
    ];

    let queries = vec![
        QuerySpec { text: format!("{} {} development", lang, domain), top_k: 10, category: "stable".into() },
        QuerySpec { text: format!("{} {} production", lang, adj1), top_k: 10, category: "stable".into() },
    ];

    (records, queries)
}

fn generate_belief_heavy(rng: &mut Rng, seed: u32) -> (Vec<CorpusRecord>, Vec<QuerySpec>) {
    let lang1 = SUBJECTS[seed as usize % SUBJECTS.len()];
    let lang2 = SUBJECTS[(seed as usize + 3) % SUBJECTS.len()];
    let domain = DOMAINS[(seed as usize + 1) % DOMAINS.len()];
    let adj = rng.pick(BACKEND_ADJECTIVES);
    let deploy = rng.pick(DEPLOY_METHODS);
    let db = rng.pick(DB_TYPES);

    let records = vec![
        CorpusRecord {
            content: format!("I prefer {} over {} for {} services", lang1, lang2, domain),
            level: Level::Domain, tags: vec![lang1.to_lowercase(), domain.to_string(), "preference".into()],
            source_type: "recorded", semantic_type: "preference",
        },
        CorpusRecord {
            content: format!("{} is the best choice for {} work", lang1, domain),
            level: Level::Domain, tags: vec![lang1.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "preference",
        },
        CorpusRecord {
            content: format!("Always use {} deployment for {} services", deploy, domain),
            level: Level::Domain, tags: vec!["deploy".into(), deploy.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "decision",
        },
        CorpusRecord {
            content: format!("{} is my go-to {} for production data", db, adj),
            level: Level::Domain, tags: vec!["database".into(), db.to_lowercase()],
            source_type: "recorded", semantic_type: "preference",
        },
        CorpusRecord {
            content: format!("{} handles {} better than alternatives", lang1, domain),
            level: Level::Domain, tags: vec![lang1.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} deployment reduces risk in {} pipelines", deploy, domain),
            level: Level::Domain, tags: vec!["deploy".into(), deploy.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "fact",
        },
    ];

    let queries = vec![
        QuerySpec { text: format!("best language for {} services", domain), top_k: 10, category: "belief-heavy".into() },
        QuerySpec { text: format!("{} deployment strategy", deploy), top_k: 10, category: "belief-heavy".into() },
    ];

    (records, queries)
}

fn generate_conflicting(rng: &mut Rng, seed: u32) -> (Vec<CorpusRecord>, Vec<QuerySpec>) {
    let pair_idx = seed as usize % CONFLICT_PAIRS.len();
    let (side_a, side_b) = CONFLICT_PAIRS[pair_idx];
    let extra_pair = CONFLICT_PAIRS[(pair_idx + 1) % CONFLICT_PAIRS.len()];
    let adj = rng.pick(BACKEND_ADJECTIVES);

    let records = vec![
        CorpusRecord {
            content: side_a.to_string(),
            level: Level::Domain, tags: vec!["conflict".into(), "preference".into()],
            source_type: "recorded", semantic_type: "preference",
        },
        CorpusRecord {
            content: side_b.to_string(),
            level: Level::Domain, tags: vec!["conflict".into(), "preference".into()],
            source_type: "recorded", semantic_type: "preference",
        },
        CorpusRecord {
            content: format!("Both sides of the debate have {} arguments", adj),
            level: Level::Domain, tags: vec!["conflict".into()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: extra_pair.0.to_string(),
            level: Level::Domain, tags: vec!["conflict".into(), "preference".into()],
            source_type: "recorded", semantic_type: "preference",
        },
        CorpusRecord {
            content: extra_pair.1.to_string(),
            level: Level::Domain, tags: vec!["conflict".into(), "preference".into()],
            source_type: "recorded", semantic_type: "preference",
        },
    ];

    // Extract a keyword from side_a for the query
    let keyword = side_a.split_whitespace().take(4).collect::<Vec<_>>().join(" ");

    let queries = vec![
        QuerySpec { text: keyword, top_k: 10, category: "conflicting".into() },
        QuerySpec { text: extra_pair.0.split_whitespace().take(4).collect::<Vec<_>>().join(" "), top_k: 10, category: "conflicting".into() },
    ];

    (records, queries)
}

fn generate_mixed(rng: &mut Rng, seed: u32) -> (Vec<CorpusRecord>, Vec<QuerySpec>) {
    let lang = rng.pick(SUBJECTS);
    let deploy = rng.pick(DEPLOY_METHODS);
    let db = rng.pick(DB_TYPES);
    let sec = rng.pick(SECURITY_TOPICS);
    let tool = rng.pick(TOOLS);
    let adj = rng.pick(BACKEND_ADJECTIVES);
    let domain1 = DOMAINS[seed as usize % DOMAINS.len()];
    let domain2 = DOMAINS[(seed as usize + 2) % DOMAINS.len()];

    let records = vec![
        CorpusRecord {
            content: format!("{} is {} for {} projects", lang, adj, domain1),
            level: Level::Domain, tags: vec![lang.to_lowercase(), domain1.to_string()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} deployment ensures safety in production", deploy),
            level: Level::Domain, tags: vec!["deploy".into(), deploy.to_lowercase()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} handles {} workloads efficiently", db, domain2),
            level: Level::Domain, tags: vec!["database".into(), db.to_lowercase()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} is critical for {} applications", sec, domain1),
            level: Level::Domain, tags: vec!["security".into(), sec.to_string()],
            source_type: "recorded", semantic_type: "decision",
        },
        CorpusRecord {
            content: format!("{} simplifies {} infrastructure management", tool, domain2),
            level: Level::Domain, tags: vec!["infra".into(), tool.to_lowercase()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("Production {} requires {} and {}", domain1, sec, deploy),
            level: Level::Domain, tags: vec![domain1.to_string(), "production".into()],
            source_type: "recorded", semantic_type: "fact",
        },
    ];

    let queries = vec![
        QuerySpec { text: format!("{} {} production", lang, domain1), top_k: 10, category: "mixed".into() },
        QuerySpec { text: format!("{} {} safety", deploy, domain2), top_k: 10, category: "mixed".into() },
    ];

    (records, queries)
}

fn generate_sparse(rng: &mut Rng, _seed: u32) -> (Vec<CorpusRecord>, Vec<QuerySpec>) {
    // Many isolated facts, few overlapping tags
    let mut records = Vec::new();
    for i in 0..8 {
        let subj = rng.pick(SUBJECTS);
        let adj = rng.pick(BACKEND_ADJECTIVES);
        let domain = rng.pick(DOMAINS);
        records.push(CorpusRecord {
            content: format!("{} is {} for isolated {} task number {}", subj, adj, domain, i),
            level: Level::Domain,
            tags: vec![format!("isolated-{}", i)], // unique tag per record = sparse clusters
            source_type: "recorded", semantic_type: "fact",
        });
    }

    let queries = vec![
        QuerySpec { text: format!("{} isolated task", rng.pick(SUBJECTS)), top_k: 10, category: "sparse".into() },
        QuerySpec { text: "isolated task number".to_string(), top_k: 5, category: "sparse".into() },
    ];

    (records, queries)
}

fn generate_no_match(rng: &mut Rng, seed: u32) -> (Vec<CorpusRecord>, Vec<QuerySpec>) {
    // Normal corpus but queries are completely outside
    let lang = rng.pick(SUBJECTS);
    let adj = rng.pick(BACKEND_ADJECTIVES);
    let domain = rng.pick(DOMAINS);

    let records = vec![
        CorpusRecord {
            content: format!("{} is {} for {} work", lang, adj, domain),
            level: Level::Domain, tags: vec![lang.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} ecosystem is mature for production {}", lang, domain),
            level: Level::Domain, tags: vec![lang.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "fact",
        },
        CorpusRecord {
            content: format!("{} tooling integrates well with {} pipelines", lang, domain),
            level: Level::Domain, tags: vec![lang.to_lowercase(), domain.to_string()],
            source_type: "recorded", semantic_type: "fact",
        },
    ];

    let q1 = NONSENSE_QUERIES[seed as usize % NONSENSE_QUERIES.len()];
    let q2 = NONSENSE_QUERIES[(seed as usize + 1) % NONSENSE_QUERIES.len()];

    let queries = vec![
        QuerySpec { text: q1.to_string(), top_k: 10, category: "no-match".into() },
        QuerySpec { text: q2.to_string(), top_k: 5, category: "no-match".into() },
    ];

    (records, queries)
}

fn generate_corpus_and_queries(
    profile: ScenarioProfile,
    seed: u32,
) -> (Vec<CorpusRecord>, Vec<QuerySpec>) {
    let mut rng = Rng::new(seed);
    match profile {
        ScenarioProfile::Stable => generate_stable(&mut rng, seed),
        ScenarioProfile::BeliefHeavy => generate_belief_heavy(&mut rng, seed),
        ScenarioProfile::Conflicting => generate_conflicting(&mut rng, seed),
        ScenarioProfile::Mixed => generate_mixed(&mut rng, seed),
        ScenarioProfile::Sparse => generate_sparse(&mut rng, seed),
        ScenarioProfile::NoMatch => generate_no_match(&mut rng, seed),
    }
}

// ═══════════════════════════════════════════════════════════
// Per-query metrics (same as monitor)
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
enum QualityLabel { Better, Same, Worse, Unclear }

impl std::fmt::Display for QualityLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            QualityLabel::Better => write!(f, "BETTER"),
            QualityLabel::Same => write!(f, "SAME"),
            QualityLabel::Worse => write!(f, "WORSE"),
            QualityLabel::Unclear => write!(f, "UNCLEAR"),
        }
    }
}

fn compute_quality_label(
    overlap: f32, was_applied: bool, records_moved: usize,
    contradiction_delta: i32, belief_coverage: f32,
) -> QualityLabel {
    if !was_applied || records_moved == 0 { return QualityLabel::Same; }
    if contradiction_delta > 0 { return QualityLabel::Worse; }
    if overlap < 0.5 { return QualityLabel::Worse; }
    if contradiction_delta < 0 { return QualityLabel::Better; }
    if belief_coverage >= 0.10 && overlap >= 0.80 { return QualityLabel::Better; }
    if belief_coverage < 0.05 { return QualityLabel::Unclear; }
    QualityLabel::Unclear
}

// ═══════════════════════════════════════════════════════════
// Per-run metrics
// ═══════════════════════════════════════════════════════════

#[derive(Debug)]
struct RunResult {
    run_id: usize,
    seed: u32,
    profile: &'static str,
    total_queries: usize,
    reranked: usize,
    better: usize,
    same: usize,
    worse: usize,
    unclear: usize,
    avg_overlap: f32,
    avg_coverage: f32,
    avg_latency_delta_us: f64,
    max_latency_delta_us: i64,
    contradiction_worsened: usize,
    max_up_shift: usize,
    max_down_shift: usize,
    alert_passed: bool,
}

fn execute_run(run_id: usize, profile: ScenarioProfile, seed: u32) -> RunResult {
    let (corpus, queries) = generate_corpus_and_queries(profile, seed);

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

    // Run maintenance
    for _ in 0..8 { aura.run_maintenance(); }

    // Collect per-query metrics
    let mut total = 0usize;
    let mut reranked = 0usize;
    let mut better = 0usize;
    let mut same = 0usize;
    let mut worse = 0usize;
    let mut unclear = 0usize;
    let mut overlaps = Vec::new();
    let mut coverages = Vec::new();
    let mut latency_deltas = Vec::new();
    let mut contradiction_worsened = 0usize;
    let mut max_up = 0usize;
    let mut max_down = 0usize;

    for q in &queries {
        total += 1;

        // Baseline via shadow API
        let baseline_start = std::time::Instant::now();
        let (baseline, shadow_report) = aura
            .recall_structured_with_shadow(
                &q.text, Some(q.top_k), Some(0.0), Some(true), None, None,
            )
            .expect("shadow recall failed");
        let baseline_us = baseline_start.elapsed().as_micros() as u64;

        // Limited via rerank report API
        let limited_start = std::time::Instant::now();
        let (limited, rerank_report) = aura
            .recall_structured_with_rerank_report(
                &q.text, Some(q.top_k), Some(0.0), Some(true), None, None,
            )
            .expect("rerank report failed");
        let limited_us = limited_start.elapsed().as_micros() as u64;

        if rerank_report.was_applied { reranked += 1; }

        // Overlap
        let baseline_ids: Vec<String> = baseline.iter().map(|(_, r)| r.id.clone()).collect();
        let limited_ids: Vec<String> = limited.iter().map(|(_, r)| r.id.clone()).collect();
        let ek = baseline_ids.len().min(q.top_k);
        let b_set: HashSet<&str> = baseline_ids.iter().take(ek).map(|s| s.as_str()).collect();
        let l_set: HashSet<&str> = limited_ids.iter().take(ek).map(|s| s.as_str()).collect();
        let overlap = if ek > 0 {
            b_set.intersection(&l_set).count() as f32 / ek as f32
        } else { 1.0 };
        overlaps.push(overlap);

        // Contradiction
        let unresolved: HashSet<&str> = shadow_report.scores.iter()
            .filter(|s| s.belief_state.as_deref() == Some("unresolved"))
            .map(|s| s.record_id.as_str())
            .collect();
        let bu = baseline_ids.iter().take(ek).filter(|id| unresolved.contains(id.as_str())).count();
        let lu = limited_ids.iter().take(ek).filter(|id| unresolved.contains(id.as_str())).count();
        let cd = lu as i32 - bu as i32;
        if cd > 0 { contradiction_worsened += 1; }

        // Coverage + latency
        coverages.push(rerank_report.belief_coverage);
        latency_deltas.push(limited_us as i64 - baseline_us as i64);

        // Shifts
        if rerank_report.max_up_shift > max_up { max_up = rerank_report.max_up_shift; }
        if rerank_report.max_down_shift > max_down { max_down = rerank_report.max_down_shift; }

        // Label
        let label = compute_quality_label(
            overlap, rerank_report.was_applied, rerank_report.records_moved,
            cd, shadow_report.belief_coverage,
        );
        match label {
            QualityLabel::Better => better += 1,
            QualityLabel::Same => same += 1,
            QualityLabel::Worse => worse += 1,
            QualityLabel::Unclear => unclear += 1,
        }
    }

    let n = total.max(1) as f32;
    let avg_overlap = overlaps.iter().sum::<f32>() / n;
    let avg_coverage = coverages.iter().sum::<f32>() / n;
    let avg_latency = latency_deltas.iter().sum::<i64>() as f64 / n as f64;
    let max_latency = latency_deltas.iter().copied().max().unwrap_or(0);

    // Per-run alert check
    let worse_pct = worse as f32 / n;
    let contradiction_pct = contradiction_worsened as f32 / n;
    let alert_passed = worse_pct <= ALERT_MAX_WORSE_PCT
        && avg_overlap >= ALERT_MIN_AVG_OVERLAP
        && max_up <= ALERT_MAX_POS_SHIFT
        && max_down <= ALERT_MAX_POS_SHIFT
        && avg_latency <= ALERT_MAX_AVG_LATENCY_US
        && contradiction_pct <= ALERT_MAX_CONTRADICTION_WORSENED_PCT;

    RunResult {
        run_id, seed, profile: profile.name(),
        total_queries: total, reranked, better, same, worse, unclear,
        avg_overlap, avg_coverage, avg_latency_delta_us: avg_latency,
        max_latency_delta_us: max_latency,
        contradiction_worsened, max_up_shift: max_up, max_down_shift: max_down,
        alert_passed,
    }
}

// ═══════════════════════════════════════════════════════════
// Campaign aggregation
// ═══════════════════════════════════════════════════════════

struct CampaignReport {
    total_runs: usize,
    total_queries: usize,
    avg_better_pct: f32,
    avg_same_pct: f32,
    avg_worse_pct: f32,
    runs_with_worse: usize,
    avg_overlap: f32,
    p50_overlap: f32,
    p95_overlap: f32,
    avg_coverage: f32,
    p50_coverage: f32,
    p95_latency_us: f64,
    avg_latency_us: f64,
    total_contradiction_worsened: usize,
    max_shift_up: usize,
    max_shift_down: usize,
    runs_with_alert_failure: usize,
    worst_run: Option<usize>,
}

fn percentile(sorted: &[f32], pct: f32) -> f32 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() as f32 - 1.0) * pct) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn percentile_f64(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() as f64 - 1.0) * pct) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn aggregate_campaign(results: &[RunResult]) -> CampaignReport {
    let n = results.len();
    let total_queries: usize = results.iter().map(|r| r.total_queries).sum();

    let total_better: usize = results.iter().map(|r| r.better).sum();
    let total_same: usize = results.iter().map(|r| r.same).sum();
    let total_worse: usize = results.iter().map(|r| r.worse).sum();
    let tq = total_queries.max(1) as f32;

    let runs_with_worse = results.iter().filter(|r| r.worse > 0).count();

    let mut overlaps: Vec<f32> = results.iter().map(|r| r.avg_overlap).collect();
    overlaps.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut coverages: Vec<f32> = results.iter().map(|r| r.avg_coverage).collect();
    coverages.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut latencies: Vec<f64> = results.iter().map(|r| r.avg_latency_delta_us).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total_contradiction: usize = results.iter().map(|r| r.contradiction_worsened).sum();
    let max_up = results.iter().map(|r| r.max_up_shift).max().unwrap_or(0);
    let max_down = results.iter().map(|r| r.max_down_shift).max().unwrap_or(0);
    let alert_failures = results.iter().filter(|r| !r.alert_passed).count();

    // Worst run = highest worse count
    let worst = results.iter()
        .filter(|r| r.worse > 0)
        .max_by_key(|r| r.worse)
        .map(|r| r.run_id);

    CampaignReport {
        total_runs: n,
        total_queries,
        avg_better_pct: total_better as f32 / tq,
        avg_same_pct: total_same as f32 / tq,
        avg_worse_pct: total_worse as f32 / tq,
        runs_with_worse,
        avg_overlap: overlaps.iter().sum::<f32>() / n as f32,
        p50_overlap: percentile(&overlaps, 0.50),
        p95_overlap: percentile(&overlaps, 0.05), // p95 = 5th percentile of sorted overlaps (worst 5%)
        avg_coverage: coverages.iter().sum::<f32>() / n as f32,
        p50_coverage: percentile(&coverages, 0.50),
        p95_latency_us: percentile_f64(&latencies, 0.95),
        avg_latency_us: latencies.iter().sum::<f64>() / n as f64,
        total_contradiction_worsened: total_contradiction,
        max_shift_up: max_up,
        max_shift_down: max_down,
        runs_with_alert_failure: alert_failures,
        worst_run: worst,
    }
}

// ═══════════════════════════════════════════════════════════
// CAMPAIGN TEST
// ═══════════════════════════════════════════════════════════

#[test]
fn belief_rerank_synthetic_campaign() {
    let num_runs: usize = std::env::var("CAMPAIGN_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let profiles = ScenarioProfile::all();
    let mut results = Vec::with_capacity(num_runs);

    eprintln!("\n  ╔═══════════════════════════════════════════════════╗");
    eprintln!("  ║  Candidate B — Synthetic Validation Campaign      ║");
    eprintln!("  ║  {} runs, {} scenario classes                      ║", num_runs, profiles.len());
    eprintln!("  ╚═══════════════════════════════════════════════════╝\n");

    for run_id in 0..num_runs {
        let profile = profiles[run_id % profiles.len()];
        let seed = (run_id as u32 + 1) * 7919; // prime-based deterministic seed
        let result = execute_run(run_id, profile, seed);

        if run_id < 10 || result.worse > 0 || !result.alert_passed {
            eprintln!("  run {:3} [{:12}] seed={:6} q={:2} reranked={} better={} worse={} overlap={:.3} coverage={:.3} alert={}",
                result.run_id, result.profile, result.seed,
                result.total_queries, result.reranked, result.better, result.worse,
                result.avg_overlap, result.avg_coverage,
                if result.alert_passed { "PASS" } else { "FAIL" });
        }

        results.push(result);
    }

    let report = aggregate_campaign(&results);

    // ── Per-profile breakdown ──
    eprintln!("\n  ── Per-Profile Summary ──");
    for profile in profiles {
        let profile_runs: Vec<&RunResult> = results.iter()
            .filter(|r| r.profile == profile.name())
            .collect();
        if profile_runs.is_empty() { continue; }
        let pn = profile_runs.len();
        let total_better: usize = profile_runs.iter().map(|r| r.better).sum();
        let total_worse: usize = profile_runs.iter().map(|r| r.worse).sum();
        let total_q: usize = profile_runs.iter().map(|r| r.total_queries).sum();
        let avg_overlap = profile_runs.iter().map(|r| r.avg_overlap).sum::<f32>() / pn as f32;
        let avg_coverage = profile_runs.iter().map(|r| r.avg_coverage).sum::<f32>() / pn as f32;
        eprintln!("  {:14} runs={:3} queries={:4} better={:3} worse={:3} avg_overlap={:.3} avg_coverage={:.3}",
            profile.name(), pn, total_q, total_better, total_worse, avg_overlap, avg_coverage);
    }

    // ── Campaign aggregate ──
    eprintln!("\n  ── Campaign Aggregate ──");
    eprintln!("  Total runs:          {}", report.total_runs);
    eprintln!("  Total queries:       {}", report.total_queries);
    eprintln!("  Avg BETTER:          {:.1}%", report.avg_better_pct * 100.0);
    eprintln!("  Avg SAME:            {:.1}%", report.avg_same_pct * 100.0);
    eprintln!("  Avg WORSE:           {:.1}%", report.avg_worse_pct * 100.0);
    eprintln!("  Runs with WORSE:     {}/{} ({:.1}%)", report.runs_with_worse, report.total_runs,
        report.runs_with_worse as f32 / report.total_runs as f32 * 100.0);
    eprintln!("  Avg overlap:         {:.3}", report.avg_overlap);
    eprintln!("  p50 overlap:         {:.3}", report.p50_overlap);
    eprintln!("  p95 overlap:         {:.3}", report.p95_overlap);
    eprintln!("  Avg coverage:        {:.3}", report.avg_coverage);
    eprintln!("  p50 coverage:        {:.3}", report.p50_coverage);
    eprintln!("  Avg latency delta:   {:.0}us", report.avg_latency_us);
    eprintln!("  p95 latency delta:   {:.0}us", report.p95_latency_us);
    eprintln!("  Contradiction total: {}", report.total_contradiction_worsened);
    eprintln!("  Max shift:           up {} / down {}", report.max_shift_up, report.max_shift_down);
    eprintln!("  Alert failures:      {}/{}", report.runs_with_alert_failure, report.total_runs);
    if let Some(worst) = report.worst_run {
        let wr = &results[worst];
        eprintln!("  Worst run:           #{} [{}] worse={} overlap={:.3}", worst, wr.profile, wr.worse, wr.avg_overlap);
    }

    // ── Gate evaluation ──
    eprintln!("\n  ── Campaign Gates ──");

    struct Gate { name: &'static str, threshold: String, actual: String, passed: bool }
    let gates = vec![
        Gate { name: "avg_worse", threshold: format!("<= {:.0}%", GATE_AVG_WORSE * 100.0),
            actual: format!("{:.1}%", report.avg_worse_pct * 100.0),
            passed: report.avg_worse_pct <= GATE_AVG_WORSE },
        Gate { name: "runs_with_worse", threshold: format!("<= {:.0}%", GATE_RUNS_WITH_WORSE * 100.0),
            actual: format!("{:.1}%", report.runs_with_worse as f32 / report.total_runs as f32 * 100.0),
            passed: (report.runs_with_worse as f32 / report.total_runs as f32) <= GATE_RUNS_WITH_WORSE },
        Gate { name: "avg_overlap", threshold: format!(">= {:.2}", GATE_AVG_OVERLAP),
            actual: format!("{:.3}", report.avg_overlap),
            passed: report.avg_overlap >= GATE_AVG_OVERLAP },
        Gate { name: "p95_overlap", threshold: format!(">= {:.2}", GATE_P95_OVERLAP),
            actual: format!("{:.3}", report.p95_overlap),
            passed: report.p95_overlap >= GATE_P95_OVERLAP },
        Gate { name: "contradiction", threshold: format!("<= {:.0}%", GATE_CONTRADICTION_WORSENED * 100.0),
            actual: format!("{:.1}%", report.total_contradiction_worsened as f32 / report.total_queries as f32 * 100.0),
            passed: (report.total_contradiction_worsened as f32 / report.total_queries as f32) <= GATE_CONTRADICTION_WORSENED },
        Gate { name: "max_shift", threshold: format!("<= {}", GATE_MAX_SHIFT),
            actual: format!("{}/{}", report.max_shift_up, report.max_shift_down),
            passed: report.max_shift_up <= GATE_MAX_SHIFT && report.max_shift_down <= GATE_MAX_SHIFT },
        Gate { name: "p95_latency", threshold: format!("<= {:.0}us", GATE_P95_LATENCY_US),
            actual: format!("{:.0}us", report.p95_latency_us),
            passed: report.p95_latency_us <= GATE_P95_LATENCY_US },
        Gate { name: "alert_failures", threshold: "0".to_string(),
            actual: format!("{}", report.runs_with_alert_failure),
            passed: report.runs_with_alert_failure == 0 },
    ];

    let mut all_passed = true;
    for gate in &gates {
        let status = if gate.passed { "PASS" } else { "FAIL" };
        let marker = if gate.passed { " " } else { "!" };
        eprintln!("  {} [{}] {:20} threshold: {:>10}  actual: {:>10}",
            marker, status, gate.name, gate.threshold, gate.actual);
        if !gate.passed { all_passed = false; }
    }

    // Strong pass check
    let strong_pass = report.avg_worse_pct == 0.0
        && report.total_contradiction_worsened == 0
        && report.avg_overlap >= 0.93
        && report.runs_with_alert_failure == 0;

    eprintln!();
    if strong_pass {
        eprintln!("  VERDICT: STRONG PASS — B fully hardened");
    } else if all_passed {
        eprintln!("  VERDICT: PASS — B validated, minor signals present");
    } else {
        eprintln!("  VERDICT: FAIL — investigate before declaring hardened");
    }
    eprintln!();

    // ── Hard assertions ──
    assert!(report.avg_worse_pct <= GATE_AVG_WORSE,
        "avg worse {:.1}% > gate {:.0}%", report.avg_worse_pct * 100.0, GATE_AVG_WORSE * 100.0);
    assert!((report.runs_with_worse as f32 / report.total_runs as f32) <= GATE_RUNS_WITH_WORSE,
        "runs with worse {:.1}% > gate {:.0}%",
        report.runs_with_worse as f32 / report.total_runs as f32 * 100.0, GATE_RUNS_WITH_WORSE * 100.0);
    assert!(report.avg_overlap >= GATE_AVG_OVERLAP,
        "avg overlap {:.3} < gate {:.2}", report.avg_overlap, GATE_AVG_OVERLAP);
    assert!(report.p95_overlap >= GATE_P95_OVERLAP,
        "p95 overlap {:.3} < gate {:.2}", report.p95_overlap, GATE_P95_OVERLAP);
    assert!(report.max_shift_up <= GATE_MAX_SHIFT,
        "max up shift {} > gate {}", report.max_shift_up, GATE_MAX_SHIFT);
    assert!(report.max_shift_down <= GATE_MAX_SHIFT,
        "max down shift {} > gate {}", report.max_shift_down, GATE_MAX_SHIFT);
    assert!(report.runs_with_alert_failure == 0,
        "alert failures: {}", report.runs_with_alert_failure);
}
